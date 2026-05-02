mod backends;

use crate::chunking::{ChunkingAlgorithm, SlidingWindowChunker};
use chrono::Duration;
use indexmap::IndexMap;
use rand::prelude::*;
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;

use crate::config::{
    ChunkingStrategy, NegativeStrategy, SamplerConfig, Selector, TextRecipe, TripletRecipe,
};
use crate::constants::sampler::AUTO_INJECTED_LONG_SECTION_CHUNK_PAIR_RECIPE_NAME;
use crate::constants::sampler::{
    ANCHOR_POSITIVE_SWAP_MASK, EPOCH_SEED_OFFSET, EXHAUSTION_RETRY_LIMIT, NEG_REASON_WRONG_ARTICLE,
    NEG_REASON_WRONG_DATE, NEG_REASON_WRONG_QA, PREFETCHER_SOURCE_ID, PREFETCHER_STOPPED_REASON,
    RECIPE_LABEL_TEXT, RECIPE_LABEL_TRIPLETS, RECIPE_ORDER_MAX_WEIGHT_MULTIPLIER,
    ROLE_LABEL_ANCHOR, ROLE_LABEL_CONTEXT, SAME_SELECTOR_PAIR_RETRY_LIMIT,
};
use crate::data::{
    ChunkView, DataRecord, PairLabel, RecordChunk, RecordSection, SampleBatch, SamplePair,
    SampleTriplet, SectionRole, TextBatch, TextSample, TripletBatch,
};
use crate::epoch::EpochTracker;
use crate::errors::SamplerError;
use crate::hash::{derive_epoch_seed, stable_hash_str};
use crate::ingestion::IngestionManager;
use crate::metadata::{META_FIELD_DATE, MetadataKey};
use crate::metrics::{chunk_proximity_score, window_index_proximity};
use crate::source::DataSource;
use crate::splits::{
    EpochStateStore, PersistedSamplerState, SamplerStateStore, SplitLabel, SplitStore,
};
use crate::tokenizer::{Tokenizer, WhitespaceTokenizer};
use crate::types::{RecipeKey, RecordId, SourceId};
use crate::utils::platform_newline;

// AUTO-RECIPE HANDLING OVERVIEW (end-to-end):
// Stage A: Source-level injection eligibility ("should this source even get the recipe?")
//   - `sync_records_from_cache` -> `record_has_long_anchor_or_context_section` marks
//     `sources_with_long_sections`.
//   - `resolve_source_triplet_plan` -> `should_auto_inject_chunk_pair_recipe` appends
//     `source_chunk_pair_recipe` for eligible sources.
// Stage B: Record-level execution eligibility ("can this specific record run it now?")
//   - `make_auto_chunk_pair_triplet_with_anchor` checks
//     `record_has_at_least_two_window_chunks_for_selector` before sampling.
// Stage C: Actual chunk window layout ("how are windows formed?")
//   - `materialize_chunks` is the source of truth for token window construction.

#[derive(Debug, Clone)]
/// Small deterministic RNG used for reproducible sampler behavior.
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn from_state(state: u64) -> Self {
        Self { state }
    }

    fn state(&self) -> u64 {
        self.state
    }

    fn next_u64_internal(&mut self) -> u64 {
        let mut z = self.state.wrapping_add(0x9E3779B97F4A7C15);
        self.state = z;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

impl rand::RngCore for DeterministicRng {
    fn next_u32(&mut self) -> u32 {
        self.next_u64_internal() as u32
    }

    fn next_u64(&mut self) -> u64 {
        self.next_u64_internal()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut offset = 0;
        while offset < dest.len() {
            let value = self.next_u64_internal();
            let bytes = value.to_le_bytes();
            let remaining = dest.len() - offset;
            let copy_len = remaining.min(bytes.len());
            dest[offset..offset + copy_len].copy_from_slice(&bytes[..copy_len]);
            offset += copy_len;
        }
    }
}

/// Public helper so callers (and examples) can compute the same chunk weighting the sampler uses.
///
/// Definitions:
/// - `trust` (range 0.0-1.0): comes from `RecordChunk.quality.trust`; scales the contribution of a chunk.
/// - `base`: window chunks use proximity-to-head (`1 / (index + 1)`); summary chunks use their configured `summary_fallback_weight`.
/// - `chunk_weight_floor`: minimum weight applied after scaling.
///
/// Formula: `max(chunk_weight_floor, base * trust)`.
pub fn chunk_weight(strategy: &ChunkingStrategy, chunk: &RecordChunk) -> f32 {
    let floor = strategy.chunk_weight_floor;
    let trust = chunk.quality.trust.clamp(0.0, 1.0);
    let base = match &chunk.view {
        ChunkView::Window { index, .. } => window_index_proximity(*index),
        ChunkView::SummaryFallback { weight, .. } => *weight,
    };
    (base * trust).max(floor)
}

/// Public sampling interface for pair, triplet, and text batch generation.
pub trait Sampler {
    /// Returns a batch of pairs. Consumes the shared epoch cursor for anchor selection.
    fn next_pair_batch(&self, split: SplitLabel) -> Result<SampleBatch, SamplerError> {
        self.next_pair_batch_with_weights(split, &HashMap::new())
    }
    /// Returns a batch of text samples. Consumes the shared epoch cursor for anchor selection.
    fn next_text_batch(&self, split: SplitLabel) -> Result<TextBatch, SamplerError> {
        self.next_text_batch_with_weights(split, &HashMap::new())
    }
    /// Returns a batch of triplets. Consumes the shared epoch cursor for anchor selection.
    fn next_triplet_batch(&self, split: SplitLabel) -> Result<TripletBatch, SamplerError> {
        self.next_triplet_batch_with_weights(split, &HashMap::new())
    }
    /// Returns a batch of pairs with per-call source weights.
    fn next_pair_batch_with_weights(
        &self,
        split: SplitLabel,
        weights: &HashMap<SourceId, f32>,
    ) -> Result<SampleBatch, SamplerError>;
    /// Returns a batch of text samples with per-call source weights.
    fn next_text_batch_with_weights(
        &self,
        split: SplitLabel,
        weights: &HashMap<SourceId, f32>,
    ) -> Result<TextBatch, SamplerError>;
    /// Returns a batch of triplets with per-call source weights.
    fn next_triplet_batch_with_weights(
        &self,
        split: SplitLabel,
        weights: &HashMap<SourceId, f32>,
    ) -> Result<TripletBatch, SamplerError>;
}

/// Background prefetcher that fills a bounded queue with sample batches.
pub struct BatchPrefetcher<T> {
    receiver: Option<mpsc::Receiver<Result<T, SamplerError>>>,
    handle: Option<thread::JoinHandle<()>>,
    stats: Arc<PrefetcherStats>,
}

#[derive(Default)]
/// Prefetcher runtime counters.
struct PrefetcherStats {
    queued: AtomicUsize,
    produced: AtomicUsize,
    errors: AtomicUsize,
}

impl<T: Send + 'static> BatchPrefetcher<T> {
    fn new<F>(capacity: usize, mut producer: F) -> Self
    where
        F: FnMut() -> Result<T, SamplerError> + Send + 'static,
    {
        let (sender, receiver) = mpsc::sync_channel(capacity.max(1));
        let stats = Arc::new(PrefetcherStats::default());
        let stats_thread = Arc::clone(&stats);
        let handle = thread::spawn(move || {
            loop {
                let result = producer();
                if result.is_err() {
                    stats_thread.errors.fetch_add(1, Ordering::Relaxed);
                }
                if sender.send(result).is_err() {
                    return;
                }
                stats_thread.queued.fetch_add(1, Ordering::Relaxed);
                stats_thread.produced.fetch_add(1, Ordering::Relaxed);
            }
        });
        Self {
            receiver: Some(receiver),
            handle: Some(handle),
            stats,
        }
    }

    /// Block until the next prefetched batch is available.
    pub fn next(&self) -> Result<T, SamplerError> {
        let receiver = self
            .receiver
            .as_ref()
            .ok_or_else(|| SamplerError::SourceUnavailable {
                source_id: PREFETCHER_SOURCE_ID.into(),
                reason: PREFETCHER_STOPPED_REASON.into(),
            })?;
        let result = receiver.recv().unwrap_or_else(|_| {
            Err(SamplerError::SourceUnavailable {
                source_id: PREFETCHER_SOURCE_ID.into(),
                reason: PREFETCHER_STOPPED_REASON.into(),
            })
        });
        self.stats
            .queued
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                Some(value.saturating_sub(1))
            })
            .ok();
        result
    }

    /// Number of prefetched batches currently queued.
    pub fn queue_len(&self) -> usize {
        self.stats.queued.load(Ordering::Relaxed)
    }

    /// Total number of batches produced by the background worker.
    pub fn produced_count(&self) -> usize {
        self.stats.produced.load(Ordering::Relaxed)
    }

    /// Total number of errors produced by the background worker.
    pub fn error_count(&self) -> usize {
        self.stats.errors.load(Ordering::Relaxed)
    }
}

impl<T> Drop for BatchPrefetcher<T> {
    fn drop(&mut self) {
        self.receiver.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Sampler that draws anchors from a single shared epoch cursor and then
/// selects chunks from those records. Ingestion happens on demand when sampling.
pub struct TripletSampler<S: SplitStore + EpochStateStore + SamplerStateStore + 'static> {
    inner: Mutex<TripletSamplerInner<S>>,
}

/// Internal sampler state implementation guarded by `TripletSampler`.
struct TripletSamplerInner<S: SplitStore + EpochStateStore + SamplerStateStore + 'static> {
    /// Immutable sampler configuration (seed, batch size, recipes, splits, etc.).
    config: SamplerConfig,
    /// Active chunking implementation used to materialize section chunks.
    chunker: Arc<dyn ChunkingAlgorithm>,
    /// Split store backing train/val/test assignments and persisted sampler state.
    split_store: Arc<S>,
    /// On-demand ingestion manager that fills the batch-sized buffer.
    ingestion: IngestionManager,
    /// Current in-memory record pool keyed by record id.
    records: IndexMap<RecordId, Arc<DataRecord>>,
    /// Deterministic RNG for per-batch shuffles and sampling.
    rng: DeterministicRng,
    /// Config-level triplet recipes used when sources do not supply their own.
    triplet_recipes: Vec<TripletRecipe>,
    /// Config-level text recipes used when sources do not supply their own.
    text_recipes: Vec<TextRecipe>,
    /// Per-source triplet recipes keyed by source id.
    source_triplet_recipes: HashMap<SourceId, Vec<TripletRecipe>>,
    /// Sources that currently contain at least one section larger than the chunk window.
    sources_with_long_sections: HashSet<SourceId>,
    /// Per-source text recipes keyed by source id.
    source_text_recipes: HashMap<SourceId, Vec<TextRecipe>>,
    /// True if triplet recipes came from config (no source defaults).
    using_config_triplet_recipes: bool,
    /// True if text recipes came from config (no derivation).
    using_config_text_recipes: bool,
    /// Last observed ingestion counter for cache updates.
    last_observed_ingest: u64,
    /// Epoch tracker for split-aware deterministic sampling.
    epoch_tracker: EpochTracker,
    /// Per-record, per-section chunk cursor to rotate through chunk windows.
    chunk_cursors: HashMap<(RecordId, usize), usize>,
    /// Per-record, per-role cursor to rotate through role-specific sections.
    role_cursors: HashMap<(RecordId, String), usize>,
    /// Pluggable negative-selection backend (uniform-random or BM25).
    negative_backend: Box<dyn backends::NegativeBackend>,
    /// Chunk id to record id lookup (used by epoch tracker).
    chunk_index: HashMap<RecordId, RecordId>,
    /// Round-robin order of source ids (deterministic).
    source_order: Vec<SourceId>,
    /// Current index into `source_order` for shuffled-cycle sampling.
    source_cycle_idx: usize,
    /// True once persisted source state is loaded.
    source_state_loaded: bool,
    /// True once persisted ingestion cursors are loaded.
    ingestion_cursors_loaded: bool,
    /// Marks whether source state needs persistence.
    source_state_dirty: bool,
    /// Record indices per source for round-robin within a source.
    source_record_indices: HashMap<SourceId, Vec<usize>>,
    /// Per-source cursor into `source_record_indices`.
    source_record_cursors: HashMap<SourceId, usize>,
    /// Round-robin index for triplet recipe cycling.
    triplet_recipe_rr_idx: usize,
    /// Round-robin index for text recipe cycling.
    text_recipe_rr_idx: usize,
    /// Epoch counter for per-source deterministic shuffling (seed ^ epoch).
    source_epoch: u64,
    /// Tracks whether each source has wrapped its cursor in the current epoch.
    source_wrapped: HashMap<SourceId, bool>,
}

impl<S: SplitStore + EpochStateStore + SamplerStateStore + 'static> TripletSamplerInner<S> {
    fn new(config: SamplerConfig, split_store: Arc<S>) -> Self {
        Self::new_with_chunker(config, split_store, Arc::new(SlidingWindowChunker))
    }

    fn new_with_chunker(
        config: SamplerConfig,
        split_store: Arc<S>,
        chunker: Arc<dyn ChunkingAlgorithm>,
    ) -> Self {
        let buffer_size = config.ingestion_max_records.max(config.batch_size).max(2);
        let using_config_triplet_recipes = !config.recipes.is_empty();
        let using_config_text_recipes = !config.text_recipes.is_empty();
        let triplet_recipes = if using_config_triplet_recipes {
            config.recipes.clone()
        } else {
            Vec::new()
        };
        let text_recipes = if using_config_text_recipes {
            config.text_recipes.clone()
        } else if !triplet_recipes.is_empty() {
            Self::build_derived_text_recipes(&triplet_recipes)
        } else {
            Vec::new()
        };
        let ingestion = IngestionManager::new(buffer_size, config.clone());
        let epoch_backend = Some(Arc::clone(&split_store) as Arc<dyn EpochStateStore>);
        let epoch_tracker = EpochTracker::new(
            true,
            epoch_backend,
            derive_epoch_seed(config.seed, EPOCH_SEED_OFFSET),
        );
        let mut sampler = Self {
            rng: DeterministicRng::new(config.seed),
            config,
            chunker,
            split_store,
            ingestion,
            records: IndexMap::new(),
            triplet_recipes,
            text_recipes,
            source_triplet_recipes: HashMap::new(),
            sources_with_long_sections: HashSet::new(),
            source_text_recipes: HashMap::new(),
            using_config_triplet_recipes,
            using_config_text_recipes,
            last_observed_ingest: 0,
            epoch_tracker,
            chunk_cursors: HashMap::new(),
            role_cursors: HashMap::new(),
            negative_backend: {
                #[cfg(feature = "bm25-mining")]
                {
                    Box::new(backends::Bm25Backend::new())
                }
                #[cfg(not(feature = "bm25-mining"))]
                {
                    Box::new(backends::DefaultBackend)
                }
            },
            chunk_index: HashMap::new(),
            source_order: Vec::new(),
            source_cycle_idx: 0,
            source_state_loaded: false,
            ingestion_cursors_loaded: false,
            source_state_dirty: false,
            source_record_indices: HashMap::new(),
            source_record_cursors: HashMap::new(),
            triplet_recipe_rr_idx: 0,
            text_recipe_rr_idx: 0,
            source_epoch: 0,
            source_wrapped: HashMap::new(),
        };
        if !sampler.using_config_text_recipes {
            sampler.rebuild_derived_text_recipes();
        }
        sampler
    }

    fn text_recipes(&self) -> &[TextRecipe] {
        &self.text_recipes
    }

    /// Current epoch-adjusted seed: mixes `source_epoch` into `config.seed` so every epoch
    /// produces a distinct permutation across all seed-dependent operations.
    fn epoch_seed(&self) -> u64 {
        derive_epoch_seed(self.config.seed, self.source_epoch)
    }

    fn register_source(&mut self, source: Box<dyn DataSource + 'static>) {
        let source_id = source.id().to_string();
        if !self.using_config_triplet_recipes {
            let triplets = source.default_triplet_recipes();
            if !triplets.is_empty() {
                self.source_triplet_recipes
                    .insert(source_id.clone(), triplets.clone());
                if !self.using_config_text_recipes {
                    let derived = Self::build_derived_text_recipes(&triplets);
                    self.source_text_recipes
                        .insert(source_id.clone(), derived.clone());
                    self.extend_text_recipes_unique(&derived);
                }
            }
        }
        self.ingestion.register_source(source);
    }

    fn set_epoch(&mut self, epoch: u64) -> Result<(), SamplerError> {
        self.epoch_tracker.ensure_loaded()?;
        self.epoch_tracker.force_epoch(epoch);
        self.source_epoch = epoch;
        self.ingestion.set_source_epoch(epoch);
        self.ingestion.reset_stream_cursors();
        self.source_record_cursors.clear();
        self.source_cycle_idx = 0;
        for source in &self.source_order {
            self.source_wrapped.insert(source.clone(), false);
        }
        self.rebuild_source_index()?;
        self.source_state_dirty = self.source_order.len() > 1;
        Ok(())
    }

    fn next_chunk_from_pool(
        &mut self,
        record_id: &str,
        section_idx: usize,
        pool: Vec<RecordChunk>,
    ) -> Option<RecordChunk> {
        if pool.is_empty() {
            return None;
        }
        let key = (record_id.to_string(), section_idx);
        if !self.chunk_cursors.contains_key(&key) {
            // First touch should **not** always start at window 0.
            // We derive a deterministic per-(record,section) starting offset so:
            // - repeated runs with the same seed/epoch are reproducible,
            // - first sampled window is spread across records/sections,
            // - subsequent calls still rotate cyclically through the pool.
            let cursor_key = format!("{}::{}", record_id, section_idx);
            let start = (stable_hash_str(self.epoch_seed(), &cursor_key) as usize) % pool.len();
            self.chunk_cursors.insert(key.clone(), start);
        }
        let cursor = self.chunk_cursors.entry(key).or_insert(0);
        if *cursor >= pool.len() {
            *cursor = 0;
        }
        let chunk = pool.get(*cursor).cloned();
        *cursor = (*cursor + 1) % pool.len();
        chunk
    }

    fn prune_cursor_state(&mut self) {
        if self.chunk_cursors.is_empty()
            && self.role_cursors.is_empty()
            && self.negative_backend.cursors_empty()
        {
            return;
        }
        let valid_ids: HashSet<RecordId> = self.records.keys().cloned().collect();
        self.chunk_cursors
            .retain(|(record_id, _), _| valid_ids.contains(record_id));
        self.role_cursors
            .retain(|(record_id, _), _| valid_ids.contains(record_id));
        self.negative_backend.prune_cursors(&valid_ids);
    }

    fn rebuild_chunk_index(&mut self) {
        self.chunk_index.clear();
        for record in self.records.values() {
            self.chunk_index
                .insert(record.id.clone(), record.id.clone());
        }
    }

    fn rebuild_source_index(&mut self) -> Result<(), SamplerError> {
        self.source_record_indices.clear();
        let mut label_cache: HashMap<RecordId, SplitLabel> = HashMap::new();
        let allowed = self.allowed_target_splits();
        let allowed_set: HashSet<SplitLabel> = allowed.into_iter().collect();
        for (idx, record) in self.records.values().enumerate() {
            let label = if let Some(label) = label_cache.get(&record.id) {
                *label
            } else {
                let label = match self.split_store.label_for(&record.id) {
                    Some(label) => label,
                    None => self.split_store.ensure(record.id.clone())?,
                };
                label_cache.insert(record.id.clone(), label);
                label
            };
            if !allowed_set.contains(&label) {
                continue;
            }
            self.source_record_indices
                .entry(record.source.clone())
                .or_default()
                .push(idx);
        }

        let shuffle_seed = self.epoch_seed();
        for indices in self.source_record_indices.values_mut() {
            indices.sort_by_key(|idx| {
                self.records
                    .get_index(*idx)
                    .map(|(_, record)| stable_hash_str(shuffle_seed, &record.id))
                    .unwrap_or(0)
            });
        }

        self.source_order = self.source_record_indices.keys().cloned().collect();
        self.source_order.sort();
        self.refresh_source_wrapped();

        self.source_record_cursors
            .retain(|source, _| self.source_record_indices.contains_key(source));
        if self.source_state_loaded {
            if self.source_order.is_empty() {
                self.source_cycle_idx = 0;
            }
            self.source_state_dirty = self.source_order.len() > 1;
        }
        Ok(())
    }

    fn refresh_source_wrapped(&mut self) {
        self.source_wrapped.clear();
        for source in &self.source_order {
            let len = self
                .source_record_indices
                .get(source)
                .map(|items| items.len())
                .unwrap_or(0);
            if len == 0 {
                self.source_wrapped.insert(source.clone(), false);
                continue;
            }
            let cursor = self.source_record_cursors.get(source).copied().unwrap_or(0);
            let wrapped = cursor > 0 && cursor % len == 0;
            self.source_wrapped.insert(source.clone(), wrapped);
        }
    }

    fn shuffled_source_cycle(&self, cycle: u64) -> Vec<SourceId> {
        let mut sources = self.source_order.clone();
        let seed = self.epoch_seed() ^ cycle;
        sources.sort_by_key(|source| stable_hash_str(seed, source));
        sources
    }

    fn ensure_source_state(&mut self) -> Result<(), SamplerError> {
        if self.source_state_loaded {
            return Ok(());
        }
        let persisted = self.split_store.load_sampler_state()?;
        self.source_cycle_idx = persisted
            .as_ref()
            .map(|state| state.source_cycle_idx as usize)
            .unwrap_or(0);
        if let Some(state) = persisted {
            for (source, cursor) in state.source_record_cursors {
                if self.source_record_indices.contains_key(&source) {
                    self.source_record_cursors.insert(source, cursor as usize);
                }
            }
            self.source_epoch = state.source_epoch;
            self.ingestion.set_source_epoch(state.source_epoch);
            self.rng = DeterministicRng::from_state(state.rng_state);
            self.triplet_recipe_rr_idx = state.triplet_recipe_rr_idx as usize;
            self.text_recipe_rr_idx = state.text_recipe_rr_idx as usize;
        }
        self.refresh_source_wrapped();
        self.source_state_loaded = true;
        self.source_state_dirty = true;
        Ok(())
    }

    fn persist_source_state(&mut self, save_to: Option<&Path>) -> Result<(), SamplerError> {
        if !self.source_state_loaded {
            return Ok(());
        }
        let state = PersistedSamplerState {
            source_cycle_idx: self.source_cycle_idx as u64,
            source_record_cursors: self
                .source_record_cursors
                .iter()
                .map(|(source, cursor)| (source.clone(), *cursor as u64))
                .collect(),
            source_epoch: self.source_epoch,
            rng_state: self.rng.state(),
            triplet_recipe_rr_idx: self.triplet_recipe_rr_idx as u64,
            text_recipe_rr_idx: self.text_recipe_rr_idx as u64,
            source_stream_cursors: self.ingestion.snapshot_cursors(),
        };
        self.split_store.save_sampler_state(&state, save_to)?;
        self.source_state_dirty = false;
        Ok(())
    }

    fn rebuild_derived_text_recipes(&mut self) {
        if self.using_config_text_recipes {
            return;
        }
        if self.triplet_recipes.is_empty() {
            self.text_recipes.clear();
        } else {
            self.text_recipes = Self::build_derived_text_recipes(&self.triplet_recipes);
        }
    }

    fn extend_text_recipes_unique(&mut self, recipes: &[TextRecipe]) {
        for recipe in recipes {
            if self
                .text_recipes
                .iter()
                .any(|existing| existing.name == recipe.name)
            {
                continue;
            }
            self.text_recipes.push(recipe.clone());
        }
    }

    fn configured_triplet_recipes_for_source<'a>(&'a self, source: &str) -> &'a [TripletRecipe] {
        if self.using_config_triplet_recipes {
            return &self.triplet_recipes;
        }
        self.source_triplet_recipes
            .get(source)
            .map(|recipes| recipes.as_slice())
            .unwrap_or(&[])
    }

    /// Returns true when `recipes` already contains the long-section auto recipe.
    fn contains_auto_chunk_pair_recipe(recipes: &[TripletRecipe]) -> bool {
        recipes
            .iter()
            .any(|recipe| recipe.name.as_ref() == AUTO_INJECTED_LONG_SECTION_CHUNK_PAIR_RECIPE_NAME)
    }

    fn source_supports_chunk_pair_recipe(&self, source: &str) -> bool {
        if self.config.chunking.max_window_tokens == 0 {
            return false;
        }
        self.sources_with_long_sections.contains(source)
    }

    /// Stage A (source-level): decide whether to append auto recipe for `source`.
    ///
    /// Decision criteria:
    /// - source must be eligible (`source_supports_chunk_pair_recipe`), and
    /// - configured pool must not already include the auto recipe name.
    fn should_auto_inject_chunk_pair_recipe(
        &self,
        source: &str,
        recipes: &[TripletRecipe],
    ) -> bool {
        self.source_supports_chunk_pair_recipe(source)
            && !Self::contains_auto_chunk_pair_recipe(recipes)
    }

    /// Build the auto-injected long-section recipe.
    ///
    /// Semantics:
    /// - Anchor selector: `Context`
    /// - Positive selector: `Context`
    /// - Negative selector: `Context` on a different record (`WrongArticle`)
    ///
    /// Anchor and positive are selected by two independent `select_chunk` calls
    /// from the same context chunk candidate pool for the chosen record.
    /// They are not concatenated and one is not derived from the other's text.
    fn source_chunk_pair_recipe() -> TripletRecipe {
        TripletRecipe {
            name: Cow::Borrowed(AUTO_INJECTED_LONG_SECTION_CHUNK_PAIR_RECIPE_NAME),
            anchor: Selector::Role(SectionRole::Context),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
            allow_same_anchor_positive: false,
        }
    }

    /// Resolve the source triplet plan.
    ///
    /// This is Stage A (source-level injection), not record-level execution.
    ///
    /// Algorithm for the auto-triplet injection path:
    /// 1) Start from the configured source recipe pool (`configured_triplet_recipes_for_source`).
    /// 2) Check auto-injection eligibility (`should_auto_inject_chunk_pair_recipe`), which is true only when:
    ///    - the source has long sections discovered during ingest sync, and
    ///    - the pool does not already include the auto recipe name.
    /// 3) If eligible, append `source_chunk_pair_recipe`.
    /// 4) Return both the effective recipe pool and whether step (3) happened.
    fn resolve_source_triplet_plan(&self, source: &str) -> (Vec<TripletRecipe>, bool) {
        let mut recipes = self.configured_triplet_recipes_for_source(source).to_vec();
        let mut auto_injected = false;
        if self.should_auto_inject_chunk_pair_recipe(source, &recipes) {
            recipes.push(Self::source_chunk_pair_recipe());
            auto_injected = true;
        }
        (recipes, auto_injected)
    }

    #[cfg(test)]
    fn triplet_recipes_for_source(&self, source: &str) -> Vec<TripletRecipe> {
        self.resolve_source_triplet_plan(source).0
    }

    fn triplet_recipe_count_for_source(&self, source: &str) -> usize {
        let (recipes, _auto_injected) = self.resolve_source_triplet_plan(source);
        recipes.len()
    }

    fn text_recipes_for_source<'a>(&'a self, source: &str) -> &'a [TextRecipe] {
        if self.using_config_text_recipes || self.using_config_triplet_recipes {
            return &self.text_recipes;
        }
        self.source_text_recipes
            .get(source)
            .map(|recipes| recipes.as_slice())
            .unwrap_or(&[])
    }

    /// Build a weighted, shuffled selection order from a slice of recipe weights.
    ///
    /// Each recipe with `weight > 0.0` receives a number of slots proportional to its weight
    /// relative to the smallest positive weight, capped at
    /// [`RECIPE_ORDER_MAX_WEIGHT_MULTIPLIER`].  The slot list is then shuffled so that
    /// deterministic round-robin cycling naturally draws recipes at the requested frequency:
    ///
    /// * `weight = [1.0, 1.0, 1.0]` → one slot each — identical to uniform round-robin.
    /// * `weight = [3.0, 1.0]`       → three slots for recipe 0, one for recipe 1.
    /// * `weight = [1.0, 0.0]`       → recipe 1 is excluded entirely (zero slots).
    fn recipe_order_weighted_shuffled(
        &mut self,
        weights: &[f32],
        rng: &mut DeterministicRng,
    ) -> Vec<usize> {
        weighted_recipe_order(weights, rng)
    }

    /// Weighted shuffle rotated by `rr_idx` for deterministic round-robin cycling.
    ///
    /// Calls [`Self::recipe_order_weighted_shuffled`] and rotates the result by
    /// `rr_idx % order.len()` so successive batches start from different recipes,
    /// preserving both the proportional frequency and the cross-batch cycling.
    fn recipe_order_weighted_cycled(
        &mut self,
        weights: &[f32],
        rr_idx: usize,
        rng: &mut DeterministicRng,
    ) -> Vec<usize> {
        let base = self.recipe_order_weighted_shuffled(weights, rng);
        if base.is_empty() {
            return base;
        }
        let start = rr_idx % base.len();
        let mut order = Vec::with_capacity(base.len());
        order.extend_from_slice(&base[start..]);
        order.extend_from_slice(&base[..start]);
        order
    }

    /// Weighted shuffled selection order for text recipes — same algorithm as
    /// [`Self::recipe_order_weighted_shuffled`], kept separate so the text and triplet
    /// round-robin counters can advance independently.
    fn text_recipe_order_weighted_shuffled(
        &mut self,
        weights: &[f32],
        rng: &mut DeterministicRng,
    ) -> Vec<usize> {
        weighted_recipe_order(weights, rng)
    }

    fn text_recipe_order_weighted_cycled(
        &mut self,
        weights: &[f32],
        rr_idx: usize,
        rng: &mut DeterministicRng,
    ) -> Vec<usize> {
        let base = self.text_recipe_order_weighted_shuffled(weights, rng);
        if base.is_empty() {
            return base;
        }
        let start = rr_idx % base.len();
        let mut order = Vec::with_capacity(base.len());
        order.extend_from_slice(&base[start..]);
        order.extend_from_slice(&base[..start]);
        order
    }

    fn allowed_target_splits(&self) -> Vec<SplitLabel> {
        self.config.allowed_splits.clone()
    }

    fn ensure_split_allowed(&self, split: SplitLabel) -> Result<(), SamplerError> {
        let allowed = self.allowed_target_splits();
        if allowed.contains(&split) {
            return Ok(());
        }
        Err(SamplerError::Configuration(format!(
            "requested split {:?} is not in allowed_splits {:?}",
            split, allowed
        )))
    }

    fn ensure_split_has_records(&mut self, target_split: SplitLabel) -> Result<(), SamplerError> {
        let records_by_split = self.records_by_split()?;
        if records_by_split
            .get(&target_split)
            .map(|records| !records.is_empty())
            .unwrap_or(false)
        {
            return Ok(());
        }
        Err(SamplerError::Exhausted(
            "no records available for target split".into(),
        ))
    }

    fn records_by_split(
        &self,
    ) -> Result<HashMap<SplitLabel, Vec<(RecordId, SourceId)>>, SamplerError> {
        let mut map: HashMap<SplitLabel, Vec<(RecordId, SourceId)>> = HashMap::new();
        let mut label_cache: HashMap<RecordId, SplitLabel> = HashMap::new();
        for (chunk_id, record_id) in &self.chunk_index {
            let Some(record) = self.records.get(record_id) else {
                continue;
            };
            let label = if let Some(label) = label_cache.get(record_id) {
                *label
            } else {
                let label = match self.split_store.label_for(record_id) {
                    Some(label) => label,
                    None => self.split_store.ensure(record_id.clone())?,
                };
                label_cache.insert(record_id.clone(), label);
                label
            };
            map.entry(label)
                .or_default()
                .push((chunk_id.clone(), record.source.clone()));
        }
        Ok(map)
    }

    fn choose_anchor_record(
        &mut self,
        source: Option<&str>,
        split: SplitLabel,
    ) -> Option<Arc<DataRecord>> {
        if let Some(source) = source {
            let indices = self.source_record_indices.get(source)?;
            if indices.is_empty() {
                return None;
            }
            let mut cursor = *self.source_record_cursors.get(source).unwrap_or(&0);
            let cycle = cursor / indices.len();
            let offset_seed = self.epoch_seed() ^ (cycle as u64);
            let offset = (stable_hash_str(offset_seed, source) as usize) % indices.len();
            let mut wrapped = false;
            let mut selected: Option<Arc<DataRecord>> = None;
            for _ in 0..indices.len() {
                let pos = (cursor % indices.len()).saturating_add(offset) % indices.len();
                let idx = indices[pos];
                cursor = cursor.saturating_add(1);
                if cursor.is_multiple_of(indices.len()) {
                    wrapped = true;
                }
                if let Some((_, record)) = self.records.get_index(idx) {
                    if self.split_store.label_for(&record.id) != Some(split) {
                        continue;
                    }
                    selected = Some(Arc::clone(record));
                    break;
                }
            }
            self.source_record_cursors
                .insert(source.to_string(), cursor);
            if wrapped {
                self.mark_source_wrapped(source);
            }
            return selected;
        }
        while let Some(chunk_id) = self.epoch_tracker.next_record(split) {
            if let Some(record_id) = self.chunk_index.get(&chunk_id)
                && let Some(record) = self.records.get(record_id)
            {
                return Some(Arc::clone(record));
            }
        }
        None
    }

    fn save_sampler_state(&mut self, save_to: Option<&Path>) -> Result<(), SamplerError> {
        if self.epoch_tracker.is_enabled() {
            self.epoch_tracker.persist()?;
        }
        self.persist_source_state(save_to)?;
        Ok(())
    }

    fn mark_source_wrapped(&mut self, source: &str) {
        self.source_wrapped.insert(source.to_string(), true);
        if self.source_order.is_empty() {
            return;
        }
        let all_wrapped = self
            .source_order
            .iter()
            .all(|name| self.source_wrapped.get(name).copied().unwrap_or(false));
        if all_wrapped {
            self.advance_source_epoch();
        }
    }

    fn advance_source_epoch(&mut self) {
        self.source_epoch = self.source_epoch.saturating_add(1);
        self.ingestion.set_source_epoch(self.source_epoch);
        self.source_record_cursors.clear();
        self.source_cycle_idx = 0;
        for source in &self.source_order {
            self.source_wrapped.insert(source.clone(), false);
        }
        let _ = self.rebuild_source_index();
        self.source_state_dirty = self.source_order.len() > 1;
    }

    fn select_temporal_neighbor(
        &'_ self,
        record: &DataRecord,
        offset_days: i32,
    ) -> Option<Arc<DataRecord>> {
        let target = record.created_at + Duration::days(offset_days.into());
        let key = record.taxonomy.first().cloned();
        let record_split = self.split_store.label_for(&record.id)?;
        self.records
            .values()
            .filter(|candidate| {
                candidate.id != record.id
                    && self
                        .split_store
                        .label_for(&candidate.id)
                        .map(|label| label == record_split)
                        .unwrap_or(false)
                    && (candidate.source == record.source
                        || key
                            .as_ref()
                            .zip(candidate.taxonomy.first())
                            .map(|(a, b)| a == b)
                            .unwrap_or(false))
            })
            .min_by_key(|candidate| (candidate.created_at - target).num_seconds().abs())
            .cloned()
    }

    fn select_negative_record(
        &self,
        anchor_record: &DataRecord,
        strategy: &NegativeStrategy,
        anchor_query_text: Option<&str>,
        rng: &mut dyn rand::RngCore,
    ) -> Option<(Arc<DataRecord>, bool)> {
        let anchor_split = self.split_store.label_for(&anchor_record.id)?;

        let in_anchor_split = |candidate: &DataRecord| {
            self.split_store
                .label_for(&candidate.id)
                .map(|label| label == anchor_split)
                .unwrap_or(false)
        };

        match strategy {
            NegativeStrategy::WrongArticle => {
                let anchor_date =
                    taxonomy_value(anchor_record, META_FIELD_DATE).map(|d| d.to_string());
                let mut same_date: Vec<Arc<DataRecord>> = self
                    .records
                    .values()
                    .filter(|candidate| {
                        candidate.source == anchor_record.source
                            && candidate.id != anchor_record.id
                            && in_anchor_split(candidate)
                    })
                    .filter(|candidate| {
                        anchor_date
                            .as_deref()
                            .zip(taxonomy_value(candidate, META_FIELD_DATE))
                            .map(|(a, b)| a == b)
                            .unwrap_or(false)
                    })
                    .cloned()
                    .collect();
                if same_date.is_empty() {
                    same_date = self
                        .records
                        .values()
                        .filter(|candidate| {
                            candidate.source == anchor_record.source
                                && candidate.id != anchor_record.id
                                && in_anchor_split(candidate)
                        })
                        .cloned()
                        .collect();
                }
                if !same_date.is_empty() {
                    return self.negative_backend.choose_negative(
                        anchor_record,
                        anchor_split,
                        same_date,
                        false,
                        anchor_query_text,
                        rng,
                    );
                }
                let pool = self
                    .records
                    .values()
                    .filter(|candidate| {
                        candidate.id != anchor_record.id && in_anchor_split(candidate)
                    })
                    .cloned()
                    .collect::<Vec<_>>();
                self.negative_backend.choose_negative(
                    anchor_record,
                    anchor_split,
                    pool,
                    true,
                    anchor_query_text,
                    rng,
                )
            }
            NegativeStrategy::WrongPublicationDate => {
                let anchor_date =
                    taxonomy_value(anchor_record, META_FIELD_DATE).map(|d| d.to_string());
                let pool: Vec<Arc<DataRecord>> = self
                    .records
                    .values()
                    .filter(|candidate| {
                        candidate.source == anchor_record.source
                            && candidate.id != anchor_record.id
                            && in_anchor_split(candidate)
                    })
                    .filter(|candidate| {
                        match (
                            anchor_date.as_deref(),
                            taxonomy_value(candidate, META_FIELD_DATE),
                        ) {
                            (Some(anchor), Some(candidate_date)) => anchor != candidate_date,
                            (Some(_), None) => true,
                            (None, Some(_)) => true,
                            (None, None) => false,
                        }
                    })
                    .cloned()
                    .collect();
                if pool.is_empty() {
                    // Fallback to any other record in the same split so split boundaries
                    // remain strictly isolated.
                    let fallback_pool = self
                        .records
                        .values()
                        .filter(|candidate| {
                            candidate.id != anchor_record.id && in_anchor_split(candidate)
                        })
                        .cloned()
                        .collect::<Vec<_>>();

                    return self.negative_backend.choose_negative(
                        anchor_record,
                        anchor_split,
                        fallback_pool,
                        true,
                        anchor_query_text,
                        rng,
                    );
                }

                self.negative_backend.choose_negative(
                    anchor_record,
                    anchor_split,
                    pool,
                    false,
                    anchor_query_text,
                    rng,
                )
            }
            NegativeStrategy::QuestionAnswerMismatch => {
                let pool: Vec<Arc<DataRecord>> = self
                    .records
                    .values()
                    .filter(|candidate| {
                        candidate.source == anchor_record.source
                            && candidate.id != anchor_record.id
                            && in_anchor_split(candidate)
                    })
                    .cloned()
                    .collect();
                if pool.is_empty() {
                    // Fallback to any other record in the same split so split boundaries
                    // remain strictly isolated.
                    let fallback_pool = self
                        .records
                        .values()
                        .filter(|candidate| {
                            candidate.id != anchor_record.id && in_anchor_split(candidate)
                        })
                        .cloned()
                        .collect::<Vec<_>>();

                    return self.negative_backend.choose_negative(
                        anchor_record,
                        anchor_split,
                        fallback_pool,
                        true,
                        anchor_query_text,
                        rng,
                    );
                }

                self.negative_backend.choose_negative(
                    anchor_record,
                    anchor_split,
                    pool,
                    false,
                    anchor_query_text,
                    rng,
                )
            }
        }
    }

    /// True when the recipe is the special auto-injected long-section chunk-pair recipe.
    fn is_auto_chunk_pair_recipe(recipe: &TripletRecipe) -> bool {
        recipe.name.as_ref() == AUTO_INJECTED_LONG_SECTION_CHUNK_PAIR_RECIPE_NAME
    }

    /// Select anchor/positive chunk pair from selectors for a single record.
    ///
    /// When selectors are the same, retries until a valid pair is found or retry limit is hit.
    fn select_anchor_positive_pair(
        &mut self,
        record: &DataRecord,
        anchor_selector: &Selector,
        positive_selector: &Selector,
        enforce_window_pair: bool,
    ) -> Option<(RecordChunk, RecordChunk)> {
        let mut anchor_chunk = self.select_chunk(record, anchor_selector)?;
        let mut positive_chunk = self.select_chunk(record, positive_selector)?;
        if anchor_selector == positive_selector {
            let mut retries = 0usize;
            while !same_selector_pair_is_valid(&anchor_chunk, &positive_chunk, enforce_window_pair)
                && retries < SAME_SELECTOR_PAIR_RETRY_LIMIT
            {
                let Some(redraw_anchor) = self.select_chunk(record, anchor_selector) else {
                    break;
                };
                let Some(redraw_positive) = self.select_chunk(record, positive_selector) else {
                    break;
                };
                anchor_chunk = redraw_anchor;
                positive_chunk = redraw_positive;
                retries += 1;
            }
            if !same_selector_pair_is_valid(&anchor_chunk, &positive_chunk, enforce_window_pair) {
                return None;
            }
        }
        Some((anchor_chunk, positive_chunk))
    }

    /// Select two distinct *window* chunks for the auto-injected recipe using recipe selectors.
    ///
    /// This method intentionally uses the recipe's selectors (no hardcoded selector values).
    /// For the current auto recipe those selectors are both `Role(Context)`, but the behavior
    /// remains tied to recipe configuration, not this helper body.
    ///
    /// NOTE: this does *not* decide whether the auto recipe is enabled for a source.
    /// Source-level enablement happens in `resolve_source_triplet_plan` via
    /// `should_auto_inject_chunk_pair_recipe`.
    fn select_distinct_window_pair_for_auto_recipe(
        &mut self,
        recipe: &TripletRecipe,
        record: &DataRecord,
    ) -> Option<(RecordChunk, RecordChunk)> {
        if recipe.anchor != recipe.positive_selector {
            return None;
        }
        self.select_anchor_positive_pair(record, &recipe.anchor, &recipe.positive_selector, true)
    }

    /// Stage B (record-level): guard for the auto-injected recipe.
    ///
    /// Requires at least two `ChunkView::Window` candidates across sections addressed by
    /// `selector`. This makes the auto recipe eligibility check explicit at execution time.
    fn record_has_at_least_two_window_chunks_for_selector(
        &self,
        record: &DataRecord,
        selector: &Selector,
    ) -> bool {
        let section_indices: Vec<usize> = match selector {
            Selector::Role(role) => record
                .sections
                .iter()
                .enumerate()
                .filter(|(_, section)| roles_match(role, &section.role))
                .map(|(idx, _)| idx)
                .collect(),
            Selector::Paragraph(idx) => {
                if *idx < record.sections.len() {
                    vec![*idx]
                } else {
                    Vec::new()
                }
            }
            Selector::Random => (0..record.sections.len()).collect(),
            Selector::TemporalOffset(_) => return false,
        };

        let mut window_count = 0usize;
        for section_idx in section_indices {
            let Some(section) = record.sections.get(section_idx) else {
                continue;
            };
            let chunks = self.materialize_chunks(record, section_idx, section);
            window_count += chunks
                .iter()
                .filter(|chunk| matches!(chunk.view, ChunkView::Window { .. }))
                .count();
            if window_count >= 2 {
                return true;
            }
        }
        false
    }

    /// Shared selector-pair stage for both standard and auto-injected triplet recipes.
    fn build_triplet_with_selector_pair_policy(
        &mut self,
        recipe: &TripletRecipe,
        record: &DataRecord,
        enforce_window_pair: bool,
        rng: &mut DeterministicRng,
    ) -> Option<SampleTriplet> {
        let (mut anchor_chunk, mut positive_chunk) = self.select_anchor_positive_pair(
            record,
            &recipe.anchor,
            &recipe.positive_selector,
            enforce_window_pair,
        )?;
        let anchor_raw_text = anchor_chunk.text.clone();
        self.decorate_chunk(record, &mut anchor_chunk, rng);
        self.decorate_chunk(record, &mut positive_chunk, rng);
        // Snapshot the raw anchor text for BM25 querying before decoration
        // added the metadata prefix — prefix tokens are absent from the index.
        self.finalize_triplet_with_negative(
            recipe,
            record,
            anchor_chunk,
            positive_chunk,
            &anchor_raw_text,
            rng,
        )
    }

    /// Execute the special auto-injected long-section chunk-pair recipe.
    ///
    /// Algorithm:
    /// 0) Stage B check: current record must expose at least two window chunks for selector.
    /// 1) Draw anchor and positive from the same record using recipe selectors.
    /// 2) Require the two chunks to be distinct.
    /// 3) Additionally require both chunks to be `ChunkView::Window` variants.
    /// 4) Apply metadata decoration to both chunks.
    /// 5) Draw the negative according to recipe negative strategy.
    fn make_auto_chunk_pair_triplet_with_anchor(
        &mut self,
        recipe: &TripletRecipe,
        record: &DataRecord,
        rng: &mut DeterministicRng,
    ) -> Option<SampleTriplet> {
        if !self.record_has_at_least_two_window_chunks_for_selector(record, &recipe.anchor) {
            return None;
        }
        let (mut anchor_chunk, mut positive_chunk) =
            self.select_distinct_window_pair_for_auto_recipe(recipe, record)?;
        let anchor_raw_text = anchor_chunk.text.clone();
        self.decorate_chunk(record, &mut anchor_chunk, rng);
        self.decorate_chunk(record, &mut positive_chunk, rng);
        self.finalize_triplet_with_negative(
            recipe,
            record,
            anchor_chunk,
            positive_chunk,
            &anchor_raw_text,
            rng,
        )
    }

    fn make_standard_triplet_with_anchor(
        &mut self,
        recipe: &TripletRecipe,
        record: &DataRecord,
        rng: &mut DeterministicRng,
    ) -> Option<SampleTriplet> {
        self.build_triplet_with_selector_pair_policy(recipe, record, false, rng)
    }

    /// Finalize a triplet by selecting a negative and applying a deterministic 50 % coin-flip
    /// that swaps anchor and positive.
    ///
    /// # Anchor/positive swap
    ///
    /// Before constructing the [`SampleTriplet`], the sampler tests the least-significant bit of
    /// the next RNG word. When the bit is zero (≈ 50 % of the time) the anchor and positive
    /// slots are exchanged so that what was originally selected as the positive becomes the
    /// anchor, and vice-versa.
    ///
    /// **Why this matters:** contrastive objectives such as InfoNCE treat the two non-negative
    /// slots asymmetrically only in so far as downstream code does. If the model can learn a
    /// positional shortcut — for example "the anchor is always the shorter text" — it can
    /// achieve low loss without learning the intended similarity structure. The swap eliminates
    /// that opportunity by ensuring both orderings appear at equal frequency, forcing the model
    /// to treat the two positive-pair slots symmetrically.
    ///
    /// The negative chunk is unaffected by the swap.
    fn finalize_triplet_with_negative(
        &mut self,
        recipe: &TripletRecipe,
        record: &DataRecord,
        anchor_chunk: RecordChunk,
        positive_chunk: RecordChunk,
        anchor_raw_text: &str,
        rng: &mut DeterministicRng,
    ) -> Option<SampleTriplet> {
        let (negative_record, fallback_used) = self.select_negative_record(
            record,
            &recipe.negative_strategy,
            Some(anchor_raw_text),
            rng,
        )?;
        let mut negative_chunk = self.select_chunk(&negative_record, &recipe.negative_selector)?;
        self.decorate_chunk(&negative_record, &mut negative_chunk, rng);

        // 50 % coin-flip: swap anchor and positive to prevent positional shortcuts.
        let (anchor_chunk, positive_chunk) = if rng.next_u64() & ANCHOR_POSITIVE_SWAP_MASK == 0 {
            (positive_chunk, anchor_chunk)
        } else {
            (anchor_chunk, positive_chunk)
        };

        // Reject if any two slots share identical rendered text.  This catches sources
        // that produce multiple records with the same string content: a negative whose
        // text matches the positive (or the anchor) would produce a trivially-invalid
        // training example.  String equality short-circuits on the first differing byte,
        // so the guard is effectively free when texts differ (the common case).
        //
        // When `allow_same_anchor_positive` is set the anchor==positive check is skipped
        // to support SimCSE-style training where both slots carry identical text and the
        // model's dropout layers produce the required embedding variation at train time.
        // The negative must still differ from both anchor and positive.
        if (!recipe.allow_same_anchor_positive && anchor_chunk.text == positive_chunk.text)
            || negative_chunk.text == positive_chunk.text
            || negative_chunk.text == anchor_chunk.text
        {
            return None;
        }

        let chunk_weight =
            self.triplet_chunk_weight(recipe, &anchor_chunk, &positive_chunk, &negative_chunk);
        let weight = recipe.weight * chunk_weight;
        let recipe_name = if fallback_used {
            format!("{}_fallback_same_split", recipe.name)
        } else {
            recipe.name.to_string()
        };
        Some(SampleTriplet {
            recipe: recipe_name,
            anchor: anchor_chunk,
            positive: positive_chunk,
            negative: negative_chunk,
            weight,
            instruction: recipe.instruction.as_ref().map(|s| s.to_string()),
        })
    }

    fn make_triplet_with_anchor(
        &mut self,
        recipe: &TripletRecipe,
        record: &DataRecord,
        rng: &mut DeterministicRng,
    ) -> Option<SampleTriplet> {
        if Self::is_auto_chunk_pair_recipe(recipe) {
            return self.make_auto_chunk_pair_triplet_with_anchor(recipe, record, rng);
        }
        self.make_standard_triplet_with_anchor(recipe, record, rng)
    }

    fn make_text_sample_for_split(
        &mut self,
        recipe: &TextRecipe,
        source: Option<&str>,
        split: SplitLabel,
        rng: &mut DeterministicRng,
    ) -> Option<TextSample> {
        let record = self.choose_anchor_record(source, split)?;
        let mut chunk = self.select_chunk(&record, &recipe.selector)?;
        self.decorate_chunk(&record, &mut chunk, rng);
        let weight = recipe.weight * self.chunk_weight(&chunk);
        Some(TextSample {
            recipe: recipe.name.to_string(),
            chunk,
            weight,
            instruction: recipe.instruction.as_ref().map(|s| s.to_string()),
        })
    }

    fn chunk_weight(&self, chunk: &RecordChunk) -> f32 {
        chunk_weight(&self.config.chunking, chunk)
    }

    fn triplet_chunk_weight(
        &self,
        recipe: &TripletRecipe,
        anchor: &RecordChunk,
        positive: &RecordChunk,
        negative: &RecordChunk,
    ) -> f32 {
        let floor = self.config.chunking.chunk_weight_floor;
        let negative_weight = negative.quality.trust.clamp(0.0, 1.0).max(floor);
        if Self::is_auto_chunk_pair_recipe(recipe) {
            // For the auto long-section recipe, use one coherence signal (proximity)
            // for the anchor/positive pair.
            let pair_trust = ((anchor.quality.trust.clamp(0.0, 1.0)
                + positive.quality.trust.clamp(0.0, 1.0))
                / 2.0)
                .clamp(0.0, 1.0);
            let pair_weight = (chunk_proximity_score(anchor, positive) * pair_trust).max(floor);
            // Keep negative weighting simple in this recipe: trust + floor only.
            return (pair_weight + pair_weight + negative_weight) / 3.0;
        }
        // Non-auto recipes also apply anchor-positive proximity, while keeping
        // negative weighting trust-only.
        let pair_proximity = chunk_proximity_score(anchor, positive);
        let anchor_weight = (self.chunk_weight(anchor) * pair_proximity).max(floor);
        let positive_weight = (self.chunk_weight(positive) * pair_proximity).max(floor);
        (anchor_weight + positive_weight + negative_weight) / 3.0
    }

    fn decorate_chunk(
        &mut self,
        record: &DataRecord,
        chunk: &mut RecordChunk,
        rng: &mut DeterministicRng,
    ) {
        chunk.kvp_meta = record
            .meta_prefix
            .as_ref()
            .map(|s| s.all_metadata())
            .unwrap_or_default();
        if let Some(spec) = record.meta_prefix.as_ref()
            && let Some(prefix) = spec.sample(rng)
        {
            let body_tokens: Vec<&str> = WhitespaceTokenizer.tokenize(&chunk.text);
            let prefix_tokens: Vec<&str> = WhitespaceTokenizer.tokenize(&prefix);
            let total_tokens = prefix_tokens.len() + body_tokens.len();
            let max_window = self.config.chunking.max_window_tokens;
            if max_window > 0 && total_tokens > max_window {
                if prefix_tokens.len() >= max_window {
                    chunk.text = prefix_tokens
                        .into_iter()
                        .take(max_window)
                        .collect::<Vec<_>>()
                        .join(" ");
                    chunk.tokens_estimate = max_window;
                } else {
                    let remaining = max_window - prefix_tokens.len();
                    let trimmed_body: Vec<&str> = body_tokens.into_iter().take(remaining).collect();
                    chunk.text =
                        format!("{}{}{}", prefix, platform_newline(), trimmed_body.join(" "));
                    chunk.tokens_estimate = max_window;
                }
            } else {
                chunk.text = format!("{}{}{}", prefix, platform_newline(), chunk.text);
                chunk.tokens_estimate = total_tokens;
            }
        }
    }

    // ── parallel-batch helpers (&self, rng-based, no cursor maps) ────────────

    /// Select a chunk using rng instead of cursor maps (for parallel execution).
    fn select_chunk_parallel(
        &self,
        record: &DataRecord,
        selector: &Selector,
        rng: &mut DeterministicRng,
    ) -> Option<RecordChunk> {
        match selector {
            Selector::Role(role) => self.select_role_parallel(record, role, rng),
            Selector::Paragraph(idx) => record.sections.get(*idx).and_then(|section| {
                let pool = self.materialize_chunks(record, *idx, section);
                if pool.is_empty() {
                    return None;
                }
                let i = rng.random_range(0..pool.len());
                pool.into_iter().nth(i)
            }),
            Selector::TemporalOffset(offset) => self
                .select_temporal_neighbor(record, *offset)
                .and_then(|neighbor| {
                    self.select_role_parallel(&neighbor, &SectionRole::Context, rng)
                }),
            Selector::Random => {
                if record.sections.is_empty() {
                    return None;
                }
                let idx = rng.random_range(0..record.sections.len());
                record.sections.get(idx).and_then(|section| {
                    let pool = self.materialize_chunks(record, idx, section);
                    if pool.is_empty() {
                        return None;
                    }
                    let i = rng.random_range(0..pool.len());
                    pool.into_iter().nth(i)
                })
            }
        }
    }

    /// Select a chunk by role using rng (parallel path — no role_cursors written).
    fn select_role_parallel(
        &self,
        record: &DataRecord,
        role: &SectionRole,
        rng: &mut DeterministicRng,
    ) -> Option<RecordChunk> {
        let indices: Vec<usize> = record
            .sections
            .iter()
            .enumerate()
            .filter(|(_, s)| roles_match(role, &s.role))
            .map(|(i, _)| i)
            .collect();
        if indices.is_empty() {
            return None;
        }
        let start = rng.random_range(0..indices.len());
        for offset in 0..indices.len() {
            let section_idx = indices[(start + offset) % indices.len()];
            let section = &record.sections[section_idx];
            let pool = self.materialize_chunks(record, section_idx, section);
            if !pool.is_empty() {
                let i = rng.random_range(0..pool.len());
                return pool.into_iter().nth(i);
            }
        }
        None
    }

    /// Decorate a chunk using a provided rng (parallel-safe, &self).
    fn decorate_chunk_parallel(
        &self,
        record: &DataRecord,
        chunk: &mut RecordChunk,
        rng: &mut DeterministicRng,
    ) {
        chunk.kvp_meta = record
            .meta_prefix
            .as_ref()
            .map(|s| s.all_metadata())
            .unwrap_or_default();
        if let Some(spec) = record.meta_prefix.as_ref()
            && let Some(prefix) = spec.sample(rng)
        {
            let body_tokens: Vec<&str> = WhitespaceTokenizer.tokenize(&chunk.text);
            let prefix_tokens: Vec<&str> = WhitespaceTokenizer.tokenize(&prefix);
            let total_tokens = prefix_tokens.len() + body_tokens.len();
            let max_window = self.config.chunking.max_window_tokens;
            if max_window > 0 && total_tokens > max_window {
                if prefix_tokens.len() >= max_window {
                    chunk.text = prefix_tokens
                        .into_iter()
                        .take(max_window)
                        .collect::<Vec<_>>()
                        .join(" ");
                    chunk.tokens_estimate = max_window;
                } else {
                    let remaining = max_window - prefix_tokens.len();
                    let trimmed_body: Vec<&str> = body_tokens.into_iter().take(remaining).collect();
                    chunk.text =
                        format!("{}{}{}", prefix, platform_newline(), trimmed_body.join(" "));
                    chunk.tokens_estimate = max_window;
                }
            } else {
                chunk.text = format!("{}{}{}", prefix, platform_newline(), chunk.text);
                chunk.tokens_estimate = total_tokens;
            }
        }
    }

    /// Select an anchor/positive pair for the parallel path (rng-based).
    fn select_anchor_positive_parallel(
        &self,
        record: &DataRecord,
        anchor_selector: &Selector,
        positive_selector: &Selector,
        enforce_window_pair: bool,
        rng: &mut DeterministicRng,
    ) -> Option<(RecordChunk, RecordChunk)> {
        let anchor_chunk = self.select_chunk_parallel(record, anchor_selector, rng)?;
        let mut positive_chunk = self.select_chunk_parallel(record, positive_selector, rng)?;
        if anchor_selector == positive_selector {
            let mut retries = 0usize;
            while !same_selector_pair_is_valid(&anchor_chunk, &positive_chunk, enforce_window_pair)
                && retries < SAME_SELECTOR_PAIR_RETRY_LIMIT
            {
                positive_chunk = self.select_chunk_parallel(record, positive_selector, rng)?;
                retries += 1;
            }
            if !same_selector_pair_is_valid(&anchor_chunk, &positive_chunk, enforce_window_pair) {
                return None;
            }
        }
        // No swap here — finalize_triplet_parallel applies the coin-flip swap.
        Some((anchor_chunk, positive_chunk))
    }

    /// Select (anchor_chunk, positive_chunk, anchor_raw_text) for a recipe — parallel-safe.
    /// Does NOT select a negative; that happens sequentially in Phase 3.
    fn select_anchor_positive_for_recipe(
        &self,
        recipe: &TripletRecipe,
        anchor_record: &DataRecord,
        rng: &mut DeterministicRng,
    ) -> Option<(RecordChunk, RecordChunk, String)> {
        if Self::is_auto_chunk_pair_recipe(recipe) {
            if !self
                .record_has_at_least_two_window_chunks_for_selector(anchor_record, &recipe.anchor)
            {
                return None;
            }
            let mut anchor_chunk =
                self.select_chunk_parallel(anchor_record, &recipe.anchor, rng)?;
            let mut positive_chunk =
                self.select_chunk_parallel(anchor_record, &recipe.anchor, rng)?;
            let mut tries = 0usize;
            while !same_selector_pair_is_valid(&anchor_chunk, &positive_chunk, true) {
                tries += 1;
                if tries >= SAME_SELECTOR_PAIR_RETRY_LIMIT {
                    return None;
                }
                anchor_chunk = self.select_chunk_parallel(anchor_record, &recipe.anchor, rng)?;
                positive_chunk = self.select_chunk_parallel(anchor_record, &recipe.anchor, rng)?;
            }
            let anchor_raw_text = anchor_chunk.text.clone();
            self.decorate_chunk_parallel(anchor_record, &mut anchor_chunk, rng);
            self.decorate_chunk_parallel(anchor_record, &mut positive_chunk, rng);
            return Some((anchor_chunk, positive_chunk, anchor_raw_text));
        }
        let (mut anchor_chunk, mut positive_chunk) = self.select_anchor_positive_parallel(
            anchor_record,
            &recipe.anchor,
            &recipe.positive_selector,
            false,
            rng,
        )?;
        let anchor_raw_text = anchor_chunk.text.clone();
        self.decorate_chunk_parallel(anchor_record, &mut anchor_chunk, rng);
        self.decorate_chunk_parallel(anchor_record, &mut positive_chunk, rng);
        Some((anchor_chunk, positive_chunk, anchor_raw_text))
    }

    fn select_chunk(&mut self, record: &DataRecord, selector: &Selector) -> Option<RecordChunk> {
        match selector {
            Selector::Role(role) => self.select_by_role(record, role),
            Selector::Paragraph(idx) => record.sections.get(*idx).and_then(|section| {
                let pool = self.materialize_chunks(record, *idx, section);
                self.next_chunk_from_pool(&record.id, *idx, pool)
            }),
            Selector::TemporalOffset(offset) => self
                .select_temporal_neighbor(record, *offset)
                .and_then(|neighbor| self.select_by_role(&neighbor, &SectionRole::Context)),
            Selector::Random => {
                if record.sections.is_empty() {
                    return None;
                }
                let idx = self.rng.random_range(0..record.sections.len());
                record.sections.get(idx).and_then(|section| {
                    let pool = self.materialize_chunks(record, idx, section);
                    self.next_chunk_from_pool(&record.id, idx, pool)
                })
            }
        }
    }

    fn select_by_role(&mut self, record: &DataRecord, role: &SectionRole) -> Option<RecordChunk> {
        let indices: Vec<usize> = record
            .sections
            .iter()
            .enumerate()
            .filter(|(_, section)| roles_match(role, &section.role))
            .map(|(idx, _)| idx)
            .collect();
        if indices.is_empty() {
            return None;
        }
        let key = role_cursor_key(&record.id, role);
        let start_offset = self
            .role_cursors
            .get(&key)
            .and_then(|last_idx| indices.iter().position(|idx| idx == last_idx))
            .map(|pos| (pos + 1) % indices.len())
            .unwrap_or_else(|| {
                // On first use for this (record,role), choose a deterministic hashed
                // section offset instead of always starting at the first matching section.
                // This avoids systematic head-bias while preserving reproducibility.
                let seed_key = format!("{}::{}", key.0, key.1);
                (stable_hash_str(self.epoch_seed(), &seed_key) as usize) % indices.len()
            });
        for offset in 0..indices.len() {
            let section_idx = indices[(start_offset + offset) % indices.len()];
            let section = &record.sections[section_idx];
            let pool = self.materialize_chunks(record, section_idx, section);
            if let Some(chunk) = self.next_chunk_from_pool(&record.id, section_idx, pool) {
                self.role_cursors.insert(key.clone(), section_idx);
                return Some(chunk);
            }
        }
        None
    }

    /// Materialize chunk windows for one section according to `ChunkingStrategy`.
    ///
    /// Window layout algorithm:
    /// 1) Tokenize section text with `split_whitespace`.
    /// 2) Set `span = min(max_window_tokens, total_tokens)`.
    /// 3) If `span == total_tokens`, emit one full-section `ChunkView::Window`.
    /// 4) Otherwise, for each configured overlap:
    ///    - `stride = max(1, span - overlap)`
    ///    - emit windows `[start, min(start + span, total_tokens))` while advancing by `stride`.
    /// 5) Optionally append one `SummaryFallback` chunk when section is longer than `max_window_tokens`.
    ///
    /// This is the source of truth for how chunk windows are "laid" in a record.
    fn materialize_chunks(
        &self,
        record: &DataRecord,
        section_idx: usize,
        section: &RecordSection,
    ) -> Vec<RecordChunk> {
        self.chunker
            .materialize(&self.config.chunking, record, section_idx, section)
    }

    fn build_derived_text_recipes(recipes: &[TripletRecipe]) -> Vec<TextRecipe> {
        let mut derived = Vec::new();
        for recipe in recipes {
            let base = recipe.name.as_ref();
            derived.push(TextRecipe {
                name: Cow::Owned(format!("{base}_anchor")),
                selector: recipe.anchor.clone(),
                weight: recipe.weight,
                instruction: None,
            });
            derived.push(TextRecipe {
                name: Cow::Owned(format!("{base}_positive")),
                selector: recipe.positive_selector.clone(),
                weight: recipe.weight,
                instruction: None,
            });
            derived.push(TextRecipe {
                name: Cow::Owned(format!("{base}_negative")),
                selector: recipe.negative_selector.clone(),
                weight: recipe.weight,
                instruction: None,
            });
        }
        derived
    }

    /// Stage A helper: true when record contains an Anchor/Context section whose token
    /// count exceeds `chunking.max_window_tokens`.
    fn record_has_long_anchor_or_context_section(&self, record: &DataRecord) -> bool {
        let window = self.config.chunking.max_window_tokens;
        if window == 0 {
            return false;
        }
        record.sections.iter().any(|section| {
            matches!(section.role, SectionRole::Anchor | SectionRole::Context)
                && WhitespaceTokenizer.token_count(&section.text) > window
        })
    }

    fn sync_records_from_cache(&mut self) -> Result<(), SamplerError> {
        let mut snapshot = self.ingestion.all_records_snapshot();
        snapshot.sort_by(|a, b| a.id.cmp(&b.id));
        self.records.clear();
        self.sources_with_long_sections.clear();
        // Cursor state must never outlive a record snapshot boundary.
        self.negative_backend.on_sync_start();
        for record in snapshot {
            if self.split_store.label_for(&record.id).is_none() {
                self.split_store.ensure(record.id.clone())?;
            }
            if self.record_has_long_anchor_or_context_section(&record) {
                // Mark source-level eligibility for auto-injected chunk-pair recipe.
                self.sources_with_long_sections
                    .insert(record.source.clone());
            }
            self.records.insert(record.id.clone(), Arc::new(record));
        }
        self.prune_cursor_state();
        self.rebuild_chunk_index();
        self.rebuild_source_index()?;
        Ok(())
    }

    fn ingest_internal_for_split(&mut self, target_split: SplitLabel) -> Result<(), SamplerError> {
        if !self.ingestion.has_sources() {
            return Ok(());
        }
        if !self.ingestion_cursors_loaded {
            if let Some(state) = self.split_store.load_sampler_state()? {
                self.ingestion.load_cursors(&state.source_stream_cursors);
                self.ingestion.set_source_epoch(state.source_epoch);
            }
            self.ingestion_cursors_loaded = true;
        }
        if self.ingestion.all_caches_empty() {
            self.ingestion.refresh_all();
        } else {
            self.ingestion.advance(self.config.batch_size);
        }
        let observed = self.ingestion.total_ingest_count();
        if observed == self.last_observed_ingest && !self.records.is_empty() {
            return Ok(());
        }
        self.last_observed_ingest = observed;
        self.sync_records_from_cache()?;
        let max_window_tokens = self.config.chunking.max_window_tokens;
        self.negative_backend.on_records_refreshed(
            &self.records,
            max_window_tokens,
            &|id| self.split_store.label_for(id),
            self.ingestion.last_refreshed_sources(),
        );
        // Epoch tracking and source-state management must run every batch regardless
        // of whether the record pool changed — reconcile advances the sampling cursor.
        self.epoch_tracker.ensure_loaded()?;
        let records_by_split = self.records_by_split()?;
        self.epoch_tracker
            .reconcile(target_split, &records_by_split);
        self.ensure_source_state()?;
        Ok(())
    }

    #[cfg(test)]
    fn ingest_internal(&mut self, split: SplitLabel) -> Result<(), SamplerError> {
        self.ingest_internal_for_split(split)
    }

    fn ingest_internal_with_weights_for_split(
        &mut self,
        target_split: SplitLabel,
        weights: &HashMap<SourceId, f32>,
    ) -> Result<(), SamplerError> {
        if !self.ingestion.has_sources() {
            return Ok(());
        }
        if !self.ingestion_cursors_loaded {
            if let Some(state) = self.split_store.load_sampler_state()? {
                self.ingestion.load_cursors(&state.source_stream_cursors);
                self.ingestion.set_source_epoch(state.source_epoch);
            }
            self.ingestion_cursors_loaded = true;
        }
        if self.ingestion.all_caches_empty() {
            self.ingestion.refresh_all_with_weights(weights)?;
        } else {
            self.ingestion
                .advance_with_weights(self.config.batch_size, weights)?;
        }
        let observed = self.ingestion.total_ingest_count();
        if observed == self.last_observed_ingest && !self.records.is_empty() {
            return Ok(());
        }
        self.last_observed_ingest = observed;
        self.sync_records_from_cache()?;
        let max_window_tokens = self.config.chunking.max_window_tokens;
        self.negative_backend.on_records_refreshed(
            &self.records,
            max_window_tokens,
            &|id| self.split_store.label_for(id),
            self.ingestion.last_refreshed_sources(),
        );
        self.epoch_tracker.ensure_loaded()?;
        let records_by_split = self.records_by_split()?;
        self.epoch_tracker
            .reconcile(target_split, &records_by_split);
        self.ensure_source_state()?;
        Ok(())
    }

    /// Helper that centralises the weight-fallback logic.
    ///
    /// If `weights` is `Some` and non-uniform or if `weights` is `None`,
    /// delegates to the appropriate ingest method.
    fn ingest_with_weights_fallback(
        &mut self,
        target_split: SplitLabel,
        weights: Option<&HashMap<SourceId, f32>>,
    ) -> Result<(), SamplerError> {
        match weights {
            Some(weights)
                if !weights.is_empty()
                    && !weights
                        .values()
                        .all(|&w| w == *weights.values().next().unwrap()) =>
            {
                self.ingest_internal_with_weights_for_split(target_split, weights)?
            }
            _ => self.ingest_internal_for_split(target_split)?,
        }
        Ok(())
    }

    fn force_ingest_refresh_with_weights_for_split(
        &mut self,
        target_split: SplitLabel,
        weights: &HashMap<SourceId, f32>,
    ) -> Result<(), SamplerError> {
        if !self.ingestion.has_sources() {
            return Ok(());
        }
        if !self.ingestion_cursors_loaded {
            if let Some(state) = self.split_store.load_sampler_state()? {
                self.ingestion.load_cursors(&state.source_stream_cursors);
                self.ingestion.set_source_epoch(state.source_epoch);
            }
            self.ingestion_cursors_loaded = true;
        }
        self.ingestion.force_refresh_all_with_weights(weights)?;
        self.last_observed_ingest = self.ingestion.total_ingest_count();
        self.sync_records_from_cache()?;
        let max_window_tokens = self.config.chunking.max_window_tokens;
        self.negative_backend.on_records_refreshed(
            &self.records,
            max_window_tokens,
            &|id| self.split_store.label_for(id),
            self.ingestion.last_refreshed_sources(),
        );
        self.epoch_tracker.ensure_loaded()?;
        let records_by_split = self.records_by_split()?;
        self.epoch_tracker
            .reconcile(target_split, &records_by_split);
        self.ensure_source_state()?;
        Ok(())
    }

    /// Select one triplet candidate for a specific source and split.
    ///
    /// This helper is the source-level recipe execution path used by both pair and
    /// triplet batch builders. It always starts from `triplet_recipes_for_source`,
    /// which already includes any auto-injected long-section recipe when eligible.
    ///
    /// Returns:
    /// - candidate triplet (with recipe) if one was sampled,
    /// - number of recipe attempts consumed (for round-robin bookkeeping).
    fn sample_source_triplet_candidate(
        &mut self,
        source: &str,
        target_split: SplitLabel,
        recipe_orders: &mut HashMap<RecipeKey, Vec<usize>>,
        recipe_positions: &mut HashMap<RecipeKey, usize>,
        rng: &mut DeterministicRng,
    ) -> (Option<(TripletRecipe, SampleTriplet)>, usize) {
        // Stage A (source-level injection): resolve effective recipe pool,
        // including auto-injected long-section recipe when source is eligible.
        let (recipes, _auto_injected) = self.resolve_source_triplet_plan(source);
        if recipes.is_empty() {
            return (None, 0);
        }
        if !recipe_orders.contains_key(source) {
            let recipe_weights: Vec<f32> = recipes.iter().map(|r| r.weight).collect();
            let order =
                self.recipe_order_weighted_cycled(&recipe_weights, self.triplet_recipe_rr_idx, rng);
            recipe_orders.insert(source.to_string(), order);
        }
        let order = recipe_orders
            .get(source)
            .expect("recipe order missing for source");
        let pos = recipe_positions.entry(source.to_string()).or_insert(0);
        let Some(anchor) = self.choose_anchor_record(Some(source), target_split) else {
            return (None, 0);
        };

        let mut attempts = 0usize;
        for offset in 0..order.len() {
            let idx = order[(*pos + offset) % order.len()];
            attempts = attempts.saturating_add(1);
            let recipe = recipes[idx].clone();
            // Stage B/C happen inside `make_triplet_with_anchor`:
            // - Stage B: record-level auto-recipe eligibility gate,
            // - Stage C: chunk window materialization/selection.
            if let Some(sample) = self.make_triplet_with_anchor(&recipe, &anchor, rng) {
                *pos = (*pos + offset + 1) % order.len();
                return (Some((recipe, sample)), attempts);
            }
        }

        (None, attempts)
    }

    fn next_pair_batch_inner_with_weights(
        &mut self,
        target_split: SplitLabel,
        weights: Option<&HashMap<SourceId, f32>>,
    ) -> Result<SampleBatch, SamplerError> {
        if let Some(weights) = weights {
            if weights.is_empty()
                || weights
                    .values()
                    .all(|&w| w == *weights.values().next().unwrap())
            {
                self.ingest_internal_for_split(target_split)?;
            } else {
                self.ingest_internal_with_weights_for_split(target_split, weights)?;
            }
        } else {
            self.ingest_internal_for_split(target_split)?;
        }
        self.ensure_split_has_records(target_split)?;
        let sources = self.source_order.clone();
        if sources.is_empty() {
            if self.triplet_recipes.is_empty() {
                return Err(SamplerError::Configuration(
                    "no triplet recipes available".into(),
                ));
            }
            let mut rng = std::mem::replace(&mut self.rng, DeterministicRng::new(0));
            let recipe_weights: Vec<f32> = self.triplet_recipes.iter().map(|r| r.weight).collect();
            let recipe_order = self.recipe_order_weighted_cycled(
                &recipe_weights,
                self.triplet_recipe_rr_idx,
                &mut rng,
            );
            let mut pairs = Vec::new();
            let mut seen = HashSet::new();
            let mut last_recipe_name = None;
            let mut recipe_pos = 0usize;
            let mut recipe_steps = 0usize;
            let attempts = self.config.batch_size * 4 * recipe_order.len().max(1);
            for _ in 0..attempts {
                if pairs.len() >= self.config.batch_size {
                    break;
                }
                let Some(anchor) = self.choose_anchor_record(None, target_split) else {
                    break;
                };
                let mut triplet = None;
                for offset in 0..recipe_order.len() {
                    let idx = recipe_order[(recipe_pos + offset) % recipe_order.len()];
                    recipe_steps = recipe_steps.saturating_add(1);
                    let recipe = self.triplet_recipes[idx].clone();
                    last_recipe_name = Some(recipe.name.clone());
                    if let Some(sample) = self.make_triplet_with_anchor(&recipe, &anchor, &mut rng)
                    {
                        triplet = Some((recipe, sample));
                        recipe_pos = (recipe_pos + offset + 1) % recipe_order.len();
                        break;
                    }
                }
                if let Some((recipe, triplet)) = triplet {
                    let key = (
                        triplet.anchor.record_id.clone(),
                        triplet.positive.record_id.clone(),
                        triplet.negative.record_id.clone(),
                    );
                    if seen.insert(key) {
                        let SampleTriplet {
                            recipe: triplet_recipe_name,
                            anchor,
                            positive,
                            negative,
                            weight,
                            instruction,
                        } = triplet;
                        if pairs.len() < self.config.batch_size {
                            pairs.push(SamplePair {
                                recipe: triplet_recipe_name.clone(),
                                anchor: anchor.clone(),
                                positive: positive.clone(),
                                weight,
                                instruction: instruction.clone(),
                                label: PairLabel::Positive,
                                reason: None,
                            });
                        }
                        if pairs.len() < self.config.batch_size {
                            pairs.push(SamplePair {
                                recipe: triplet_recipe_name,
                                anchor,
                                positive: negative,
                                weight,
                                instruction,
                                label: PairLabel::Negative,
                                reason: Some(
                                    strategy_reason(&recipe.negative_strategy).to_string(),
                                ),
                            });
                        }
                    }
                }
            }
            if recipe_steps > 0 {
                self.triplet_recipe_rr_idx =
                    self.triplet_recipe_rr_idx.saturating_add(recipe_steps);
            }
            self.rng = rng;
            pad_with_reuse(&mut pairs, self.config.batch_size);
            if pairs.len() == self.config.batch_size {
                return Ok(SampleBatch { pairs });
            }
            return Err(SamplerError::Exhausted(
                last_recipe_name
                    .unwrap_or(Cow::Borrowed(RECIPE_LABEL_TRIPLETS))
                    .to_string(),
            ));
        }

        let mut pairs = Vec::new();
        let mut seen = HashSet::new();
        let mut source_steps = 0usize;
        let mut cycle = (self.source_cycle_idx / sources.len()) as u64;
        let mut source_idx = self.source_cycle_idx % sources.len();
        let mut cycle_sources = self.shuffled_source_cycle(cycle);
        let mut recipe_orders: HashMap<RecipeKey, Vec<usize>> = HashMap::new();
        let mut recipe_positions: HashMap<RecipeKey, usize> = HashMap::new();
        let mut recipe_steps = 0usize;
        let max_recipe_len = sources
            .iter()
            .map(|source| self.triplet_recipe_count_for_source(source))
            .max()
            .unwrap_or(1)
            .max(1);
        let attempts = self.config.batch_size * 4 * sources.len() * max_recipe_len;
        let mut rng = std::mem::replace(&mut self.rng, DeterministicRng::new(0));
        for _ in 0..attempts {
            if pairs.len() >= self.config.batch_size {
                break;
            }
            let source = cycle_sources[source_idx].as_str();
            let (triplet, attempts_used) = self.sample_source_triplet_candidate(
                source,
                target_split,
                &mut recipe_orders,
                &mut recipe_positions,
                &mut rng,
            );
            recipe_steps = recipe_steps.saturating_add(attempts_used);
            if let Some((recipe, triplet)) = triplet {
                let key = (
                    triplet.anchor.record_id.clone(),
                    triplet.positive.record_id.clone(),
                    triplet.negative.record_id.clone(),
                );
                if seen.insert(key) {
                    let SampleTriplet {
                        recipe: triplet_recipe_name,
                        anchor,
                        positive,
                        negative,
                        weight,
                        instruction,
                    } = triplet;
                    if pairs.len() < self.config.batch_size {
                        pairs.push(SamplePair {
                            recipe: triplet_recipe_name.clone(),
                            anchor: anchor.clone(),
                            positive: positive.clone(),
                            weight,
                            instruction: instruction.clone(),
                            label: PairLabel::Positive,
                            reason: None,
                        });
                    }
                    if pairs.len() < self.config.batch_size {
                        pairs.push(SamplePair {
                            recipe: triplet_recipe_name,
                            anchor,
                            positive: negative,
                            weight,
                            instruction,
                            label: PairLabel::Negative,
                            reason: Some(strategy_reason(&recipe.negative_strategy).to_string()),
                        });
                    }
                }
            }
            source_idx += 1;
            source_steps += 1;
            if source_idx >= cycle_sources.len() {
                source_idx = 0;
                cycle = cycle.saturating_add(1);
                cycle_sources = self.shuffled_source_cycle(cycle);
            }
        }
        if recipe_steps > 0 {
            self.triplet_recipe_rr_idx = self.triplet_recipe_rr_idx.saturating_add(recipe_steps);
        }
        self.rng = rng;
        pad_with_reuse(&mut pairs, self.config.batch_size);
        if pairs.len() == self.config.batch_size {
            self.source_cycle_idx = self.source_cycle_idx.saturating_add(source_steps);
            self.source_state_dirty = sources.len() > 1;
            return Ok(SampleBatch { pairs });
        }
        Err(SamplerError::Exhausted(RECIPE_LABEL_TRIPLETS.into()))
    }

    fn next_text_batch_inner_with_weights(
        &mut self,
        target_split: SplitLabel,
        weights: Option<&HashMap<SourceId, f32>>,
    ) -> Result<TextBatch, SamplerError> {
        self.ingest_with_weights_fallback(target_split, weights)?;
        self.ensure_split_has_records(target_split)?;
        let sources = self.source_order.clone();
        if sources.is_empty() {
            if self.text_recipes.is_empty() {
                return Err(SamplerError::Configuration(
                    "no text recipes configured".into(),
                ));
            }
            let mut rng = std::mem::replace(&mut self.rng, DeterministicRng::new(0));
            let recipe_weights: Vec<f32> = self.text_recipes.iter().map(|r| r.weight).collect();
            let recipe_order = self.text_recipe_order_weighted_cycled(
                &recipe_weights,
                self.text_recipe_rr_idx,
                &mut rng,
            );
            let mut samples = Vec::new();
            let mut seen = HashSet::new();
            let mut last_recipe_name = None;
            let mut recipe_pos = 0usize;
            let mut recipe_steps = 0usize;
            let attempts = self.config.batch_size * 4 * recipe_order.len().max(1);
            for _ in 0..attempts {
                if samples.len() >= self.config.batch_size {
                    break;
                }
                let recipe_idx = recipe_order[recipe_pos];
                recipe_pos = (recipe_pos + 1) % recipe_order.len();
                recipe_steps = recipe_steps.saturating_add(1);
                let recipe = self.text_recipes[recipe_idx].clone();
                last_recipe_name = Some(recipe.name.clone());
                if let Some(sample) =
                    self.make_text_sample_for_split(&recipe, None, target_split, &mut rng)
                {
                    let key = chunk_key(&sample.chunk);
                    if seen.insert(key) {
                        samples.push(sample);
                    }
                }
            }
            if recipe_steps > 0 {
                self.text_recipe_rr_idx = self.text_recipe_rr_idx.saturating_add(recipe_steps);
            }
            self.rng = rng;
            pad_with_reuse(&mut samples, self.config.batch_size);
            if samples.len() == self.config.batch_size {
                return Ok(TextBatch { samples });
            }
            return Err(SamplerError::Exhausted(
                last_recipe_name
                    .unwrap_or(Cow::Borrowed(RECIPE_LABEL_TEXT))
                    .to_string(),
            ));
        }

        let mut samples = Vec::new();
        let mut seen = HashSet::new();
        let mut source_steps = 0usize;
        let mut cycle = (self.source_cycle_idx / sources.len()) as u64;
        let mut idx = self.source_cycle_idx % sources.len();
        let mut cycle_sources = self.shuffled_source_cycle(cycle);
        let mut recipe_orders: HashMap<RecipeKey, Vec<usize>> = HashMap::new();
        let mut recipe_positions: HashMap<RecipeKey, usize> = HashMap::new();
        let mut recipe_steps = 0usize;
        let max_recipe_len = sources
            .iter()
            .map(|source| self.text_recipes_for_source(source).len())
            .max()
            .unwrap_or(1)
            .max(1);
        let attempts = self.config.batch_size * 4 * sources.len() * max_recipe_len;
        let mut rng = std::mem::replace(&mut self.rng, DeterministicRng::new(0));
        for _ in 0..attempts {
            if samples.len() >= self.config.batch_size {
                break;
            }
            let source = cycle_sources[idx].as_str();
            let recipes = self.text_recipes_for_source(source).to_vec();
            if recipes.is_empty() {
                idx += 1;
                source_steps += 1;
                if idx >= cycle_sources.len() {
                    idx = 0;
                    cycle = cycle.saturating_add(1);
                    cycle_sources = self.shuffled_source_cycle(cycle);
                }
                continue;
            }
            if !recipe_orders.contains_key(source) {
                let recipe_weights: Vec<f32> = recipes.iter().map(|r| r.weight).collect();
                let order = self.text_recipe_order_weighted_cycled(
                    &recipe_weights,
                    self.text_recipe_rr_idx,
                    &mut rng,
                );
                recipe_orders.insert(source.to_string(), order);
            }
            let order = recipe_orders
                .get(source)
                .expect("recipe order missing for source");
            let pos = recipe_positions.entry(source.to_string()).or_insert(0);
            let mut sample: Option<(TextRecipe, TextSample)> = None;
            for offset in 0..order.len() {
                let recipe_idx = order[(*pos + offset) % order.len()];
                let recipe = recipes[recipe_idx].clone();
                if let Some(item) =
                    self.make_text_sample_for_split(&recipe, Some(source), target_split, &mut rng)
                {
                    recipe_steps = recipe_steps.saturating_add(offset + 1);
                    *pos = (*pos + offset + 1) % order.len();
                    sample = Some((recipe, item));
                    break;
                }
            }
            if sample.is_none() {
                recipe_steps = recipe_steps.saturating_add(order.len());
            }
            if let Some((_recipe, sample)) = sample {
                let key = chunk_key(&sample.chunk);
                if seen.insert(key) {
                    samples.push(sample);
                }
            }
            idx += 1;
            source_steps += 1;
            if idx >= cycle_sources.len() {
                idx = 0;
                cycle = cycle.saturating_add(1);
                cycle_sources = self.shuffled_source_cycle(cycle);
            }
        }
        if samples.len() != self.config.batch_size {
            pad_with_reuse(&mut samples, self.config.batch_size);
        }
        if samples.len() != self.config.batch_size {
            self.rng = rng;
            return Err(SamplerError::Exhausted(RECIPE_LABEL_TEXT.into()));
        }
        self.rng = rng;
        self.source_cycle_idx = self.source_cycle_idx.saturating_add(source_steps);
        self.source_state_dirty = sources.len() > 1;
        if recipe_steps > 0 {
            self.text_recipe_rr_idx = self.text_recipe_rr_idx.saturating_add(recipe_steps);
        }
        Ok(TextBatch { samples })
    }

    fn next_triplet_batch_inner_with_weights(
        &mut self,
        target_split: SplitLabel,
        weights: Option<&HashMap<SourceId, f32>>,
    ) -> Result<TripletBatch, SamplerError> {
        self.ingest_with_weights_fallback(target_split, weights)?;
        self.ensure_split_has_records(target_split)?;
        let sources = self.source_order.clone();
        if sources.is_empty() {
            if self.triplet_recipes.is_empty() {
                return Err(SamplerError::Configuration(
                    "no triplet recipes configured".into(),
                ));
            }
            let mut rng = std::mem::replace(&mut self.rng, DeterministicRng::new(0));
            let recipe_weights: Vec<f32> = self.triplet_recipes.iter().map(|r| r.weight).collect();
            let recipe_order = self.recipe_order_weighted_cycled(
                &recipe_weights,
                self.triplet_recipe_rr_idx,
                &mut rng,
            );
            let mut triplets = Vec::new();
            let mut seen = HashSet::new();
            let mut last_recipe_name = None;
            let mut recipe_pos = 0usize;
            let mut recipe_steps = 0usize;
            let attempts = self.config.batch_size * 4 * recipe_order.len().max(1);
            for _ in 0..attempts {
                if triplets.len() >= self.config.batch_size {
                    break;
                }
                let Some(anchor) = self.choose_anchor_record(None, target_split) else {
                    break;
                };
                let mut triplet = None;
                for offset in 0..recipe_order.len() {
                    let idx = recipe_order[(recipe_pos + offset) % recipe_order.len()];
                    recipe_steps = recipe_steps.saturating_add(1);
                    let recipe = self.triplet_recipes[idx].clone();
                    last_recipe_name = Some(recipe.name.clone());
                    if let Some(sample) = self.make_triplet_with_anchor(&recipe, &anchor, &mut rng)
                    {
                        triplet = Some(sample);
                        recipe_pos = (recipe_pos + offset + 1) % recipe_order.len();
                        break;
                    }
                }
                if let Some(triplet) = triplet {
                    let key = (
                        triplet.anchor.record_id.clone(),
                        triplet.positive.record_id.clone(),
                        triplet.negative.record_id.clone(),
                    );
                    if seen.insert(key) {
                        triplets.push(triplet);
                    }
                }
            }
            if recipe_steps > 0 {
                self.triplet_recipe_rr_idx =
                    self.triplet_recipe_rr_idx.saturating_add(recipe_steps);
            }
            self.rng = rng;
            pad_with_reuse(&mut triplets, self.config.batch_size);
            if triplets.len() == self.config.batch_size {
                return Ok(TripletBatch { triplets });
            }
            return Err(SamplerError::Exhausted(
                last_recipe_name
                    .unwrap_or(Cow::Borrowed(RECIPE_LABEL_TRIPLETS))
                    .to_string(),
            ));
        }

        // ── Declarations for multi-source parallel path ────────────────────────
        let mut triplets: Vec<SampleTriplet> = Vec::new();
        let mut seen: HashSet<(RecordId, RecordId, RecordId)> = HashSet::new();
        let mut source_steps = 0usize;
        let mut cycle = (self.source_cycle_idx / sources.len()) as u64;
        let mut source_idx = self.source_cycle_idx % sources.len();
        let mut cycle_sources = self.shuffled_source_cycle(cycle);
        let mut recipe_steps = 0usize;
        let max_recipe_len = sources
            .iter()
            .map(|source| self.triplet_recipe_count_for_source(source.as_str()))
            .max()
            .unwrap_or(1)
            .max(1);
        let mut rng = std::mem::replace(&mut self.rng, DeterministicRng::new(0));

        // ── Phase 1: sequential anchor pre-assignment ──────────────────────────
        // Advance cursors and pick anchors deterministically. Capture a fork seed
        // for each slot so that Phase 2 can reconstruct independent rngs in
        // deterministic order.
        struct SlotPlan {
            anchor: Arc<DataRecord>,
            recipes: Vec<TripletRecipe>,
            fork_seed: u64,
        }
        let target_slots = self.config.batch_size * 4 * max_recipe_len;
        let mut slot_plans: Vec<SlotPlan> = Vec::with_capacity(target_slots);

        for _ in 0..target_slots {
            if slot_plans.len() >= target_slots {
                break;
            }
            let source = cycle_sources[source_idx].as_str();
            let (recipes, _) = self.resolve_source_triplet_plan(source);
            if !recipes.is_empty() {
                let fork_seed = rng.next_u64();
                if let Some(anchor) = self.choose_anchor_record(Some(source), target_split) {
                    slot_plans.push(SlotPlan {
                        anchor,
                        recipes,
                        fork_seed,
                    });
                    recipe_steps = recipe_steps.saturating_add(1);
                }
            }
            source_idx += 1;
            source_steps += 1;
            if source_idx >= cycle_sources.len() {
                source_idx = 0;
                cycle = cycle.saturating_add(1);
                cycle_sources = self.shuffled_source_cycle(cycle);
            }
        }
        self.rng = rng;

        // ── Phase 2: parallel anchor+positive selection ────────────────────────
        // Only chunk selection runs in parallel; negative selection and the
        // coin-flip swap remain sequential in Phase 3 so that the BM25 cursor
        // rotation (shared RwLock state) is deterministic across runs.
        struct SlotCandidate {
            recipe: TripletRecipe,
            anchor: Arc<DataRecord>,
            anchor_chunk: RecordChunk,
            positive_chunk: RecordChunk,
            anchor_raw_text: String,
        }
        let mut raw_candidates: Vec<(usize, Option<SlotCandidate>)> = slot_plans
            .par_iter()
            .enumerate()
            .map(|(slot_idx, plan)| {
                let mut fork_rng = DeterministicRng::new(plan.fork_seed);
                // Use weighted shuffle so recipes with higher weight are tried more
                // often and zero-weight recipes are excluded entirely.
                let weights: Vec<f32> = plan.recipes.iter().map(|r| r.weight).collect();
                let order = weighted_recipe_order(&weights, &mut fork_rng);
                let mut candidate = None;
                for &idx in &order {
                    let recipe = &plan.recipes[idx];
                    if let Some((ac, pc, raw)) =
                        self.select_anchor_positive_for_recipe(recipe, &plan.anchor, &mut fork_rng)
                    {
                        candidate = Some(SlotCandidate {
                            recipe: recipe.clone(),
                            anchor: Arc::clone(&plan.anchor),
                            anchor_chunk: ac,
                            positive_chunk: pc,
                            anchor_raw_text: raw,
                        });
                        break;
                    }
                }
                (slot_idx, candidate)
            })
            .collect();

        // ── Phase 3: sequential finalization (negative selection + coin-flip) ───
        // Sort by slot_idx to restore deterministic ordering, then finalize each
        // candidate sequentially so that BM25 cursor rotation advances in a fixed
        // order regardless of rayon's thread scheduling.
        raw_candidates.sort_unstable_by_key(|(i, _)| *i);
        let mut rng = std::mem::replace(&mut self.rng, DeterministicRng::new(0));
        for (_, candidate) in raw_candidates {
            if triplets.len() >= self.config.batch_size {
                break;
            }
            let Some(sc) = candidate else { continue };
            let (negative_record, fallback_used) = match self.select_negative_record(
                &sc.anchor,
                &sc.recipe.negative_strategy,
                Some(&sc.anchor_raw_text),
                &mut rng,
            ) {
                Some(pair) => pair,
                None => continue,
            };
            let mut negative_chunk = match self.select_chunk_parallel(
                &negative_record,
                &sc.recipe.negative_selector,
                &mut rng,
            ) {
                Some(c) => c,
                None => continue,
            };
            self.decorate_chunk_parallel(&negative_record, &mut negative_chunk, &mut rng);

            // 50 % coin-flip swap using the pre-sampled per-slot seed + main rng.
            let (anchor_chunk, positive_chunk) = if rng.next_u64() & ANCHOR_POSITIVE_SWAP_MASK == 0
            {
                (sc.positive_chunk, sc.anchor_chunk)
            } else {
                (sc.anchor_chunk, sc.positive_chunk)
            };

            if (!sc.recipe.allow_same_anchor_positive && anchor_chunk.text == positive_chunk.text)
                || negative_chunk.text == positive_chunk.text
                || negative_chunk.text == anchor_chunk.text
            {
                continue;
            }

            let chunk_weight = self.triplet_chunk_weight(
                &sc.recipe,
                &anchor_chunk,
                &positive_chunk,
                &negative_chunk,
            );
            let weight = sc.recipe.weight * chunk_weight;
            let recipe_name = if fallback_used {
                format!("{}_fallback_same_split", sc.recipe.name)
            } else {
                sc.recipe.name.to_string()
            };
            let triplet = SampleTriplet {
                recipe: recipe_name,
                anchor: anchor_chunk,
                positive: positive_chunk,
                negative: negative_chunk,
                weight,
                instruction: sc.recipe.instruction.as_ref().map(|s| s.to_string()),
            };
            let key = (
                triplet.anchor.record_id.clone(),
                triplet.positive.record_id.clone(),
                triplet.negative.record_id.clone(),
            );
            if seen.insert(key) && triplets.len() < self.config.batch_size {
                triplets.push(triplet);
            }
        }
        self.rng = rng;

        if recipe_steps > 0 {
            self.triplet_recipe_rr_idx = self.triplet_recipe_rr_idx.saturating_add(recipe_steps);
        }
        pad_with_reuse(&mut triplets, self.config.batch_size);
        if triplets.len() == self.config.batch_size {
            self.source_cycle_idx = self.source_cycle_idx.saturating_add(source_steps);
            self.source_state_dirty = sources.len() > 1;
            let batch = TripletBatch { triplets };
            return Ok(batch);
        }
        Err(SamplerError::Exhausted(RECIPE_LABEL_TRIPLETS.into()))
    }

    // ── test-only helpers ─────────────────────────────────────────────────────

    /// Return a mutable reference to the concrete [`backends::Bm25Backend`], panicking
    /// when the feature is disabled (which won't happen — call sites are
    /// equally gated on `#[cfg(feature = "bm25-mining")]`).
    #[cfg(test)]
    #[cfg(feature = "bm25-mining")]
    fn bm25_backend_mut(&mut self) -> &mut backends::Bm25Backend {
        self.negative_backend
            .as_any_mut()
            .downcast_mut::<backends::Bm25Backend>()
            .expect("bm25_backend_mut: negative_backend is Bm25Backend when bm25-mining feature is active")
    }

    /// Test shim: `recipe_order_weighted_shuffled` using the inner rng.
    #[cfg(test)]
    fn recipe_order_weighted_shuffled_seeded(&mut self, weights: &[f32]) -> Vec<usize> {
        let mut rng = std::mem::replace(&mut self.rng, DeterministicRng::new(0));
        let result = self.recipe_order_weighted_shuffled(weights, &mut rng);
        self.rng = rng;
        result
    }

    /// Test shim: `recipe_order_weighted_cycled` using the inner rng.
    #[cfg(test)]
    fn recipe_order_weighted_cycled_seeded(
        &mut self,
        weights: &[f32],
        rr_idx: usize,
    ) -> Vec<usize> {
        let mut rng = std::mem::replace(&mut self.rng, DeterministicRng::new(0));
        let result = self.recipe_order_weighted_cycled(weights, rr_idx, &mut rng);
        self.rng = rng;
        result
    }

    /// Test shim: `text_recipe_order_weighted_shuffled` using the inner rng.
    #[cfg(test)]
    fn text_recipe_order_weighted_shuffled_seeded(&mut self, weights: &[f32]) -> Vec<usize> {
        let mut rng = std::mem::replace(&mut self.rng, DeterministicRng::new(0));
        let result = self.text_recipe_order_weighted_shuffled(weights, &mut rng);
        self.rng = rng;
        result
    }

    /// Test shim: `text_recipe_order_weighted_cycled` using the inner rng.
    #[cfg(test)]
    fn text_recipe_order_weighted_cycled_seeded(
        &mut self,
        weights: &[f32],
        rr_idx: usize,
    ) -> Vec<usize> {
        let mut rng = std::mem::replace(&mut self.rng, DeterministicRng::new(0));
        let result = self.text_recipe_order_weighted_cycled(weights, rr_idx, &mut rng);
        self.rng = rng;
        result
    }

    /// Test shim: `select_negative_record` using the inner rng.
    #[cfg(test)]
    fn select_negative_record_seeded(
        &mut self,
        anchor_record: &DataRecord,
        strategy: &NegativeStrategy,
        anchor_query_text: Option<&str>,
    ) -> Option<(Arc<DataRecord>, bool)> {
        let mut rng = std::mem::replace(&mut self.rng, DeterministicRng::new(0));
        let result =
            self.select_negative_record(anchor_record, strategy, anchor_query_text, &mut rng);
        self.rng = rng;
        result
    }

    /// Test shim: `make_triplet_with_anchor` using the inner rng.
    #[cfg(test)]
    fn make_triplet_with_anchor_seeded(
        &mut self,
        recipe: &TripletRecipe,
        anchor: &DataRecord,
    ) -> Option<SampleTriplet> {
        let mut rng = std::mem::replace(&mut self.rng, DeterministicRng::new(0));
        let result = self.make_triplet_with_anchor(recipe, anchor, &mut rng);
        self.rng = rng;
        result
    }

    /// Test shim: `decorate_chunk` using the inner rng.
    #[cfg(test)]
    fn decorate_chunk_seeded(&mut self, record: &DataRecord, chunk: &mut RecordChunk) {
        let mut rng = std::mem::replace(&mut self.rng, DeterministicRng::new(0));
        self.decorate_chunk(record, chunk, &mut rng);
        self.rng = rng;
    }

    // ── extended-metrics helpers ───────────────────────────────────────────────

    /// Return cumulative `(fallback_count, selection_count)` from the BM25
    /// backend since it was created.  Only available when both `bm25-mining`
    /// and `extended-metrics` features are active.
    #[cfg(all(feature = "bm25-mining", feature = "extended-metrics"))]
    fn bm25_fallback_stats(&self) -> (u64, u64) {
        self.negative_backend.bm25_fallback_stats()
    }

    /// Look up `anchor`'s split label and return BM25-ranked candidate IDs.
    ///
    /// Mirrors the old `bm25_ranked_candidates` method that tests called on
    /// `TripletSamplerInner` before BM25 state moved into `Bm25Backend`.
    #[cfg(test)]
    #[cfg(feature = "bm25-mining")]
    fn bm25_ranked_candidates(&mut self, anchor: &crate::data::DataRecord) -> Vec<RecordId> {
        let split = self
            .split_store
            .label_for(&anchor.id)
            .unwrap_or(SplitLabel::Train);
        self.negative_backend
            .as_any_mut()
            .downcast_mut::<backends::Bm25Backend>()
            .expect("bm25_ranked_candidates: Bm25Backend")
            .ranked_candidates_pub(anchor, split)
    }
}

/// Build a shuffled selection order from a slice of recipe weights.
///
/// Each recipe with `weight > 0.0` receives a number of slots proportional to its weight
/// relative to the smallest positive weight, capped at
/// [`RECIPE_ORDER_MAX_WEIGHT_MULTIPLIER`].  The resulting expanded slot list is then
/// shuffled, giving each recipe approximately the requested frequency in round-robin
/// cycling.
///
/// This is a free function (no `self`) so it can be called from rayon parallel closures as
/// well as from within the `TripletSamplerInner` instance methods.
fn weighted_recipe_order(weights: &[f32], rng: &mut DeterministicRng) -> Vec<usize> {
    let nonzero: Vec<(usize, f32)> = weights
        .iter()
        .enumerate()
        .filter(|(_, w)| **w > 0.0)
        .map(|(i, &w)| (i, w))
        .collect();
    if nonzero.is_empty() {
        return Vec::new();
    }
    let w_min = nonzero
        .iter()
        .map(|(_, w)| *w)
        .fold(f32::INFINITY, f32::min);
    let mut order: Vec<usize> = Vec::new();
    for (recipe_idx, w) in &nonzero {
        let tickets = ((w / w_min).round() as usize).clamp(1, RECIPE_ORDER_MAX_WEIGHT_MULTIPLIER);
        for _ in 0..tickets {
            order.push(*recipe_idx);
        }
    }
    order.shuffle(rng);
    order
}

fn same_selector_pair_is_valid(
    anchor_chunk: &RecordChunk,
    positive_chunk: &RecordChunk,
    enforce_window_pair: bool,
) -> bool {
    if chunk_key(anchor_chunk) == chunk_key(positive_chunk) {
        return false;
    }
    if !enforce_window_pair {
        return true;
    }
    matches!(
        (&anchor_chunk.view, &positive_chunk.view),
        (ChunkView::Window { .. }, ChunkView::Window { .. })
    )
}

impl<S: SplitStore + EpochStateStore + SamplerStateStore + 'static> TripletSampler<S> {
    /// Create a sampler from config and a split-state backend.
    pub fn new(config: SamplerConfig, split_store: Arc<S>) -> Self {
        let inner = TripletSamplerInner::new(config, split_store);
        Self {
            inner: Mutex::new(inner),
        }
    }

    /// Create a sampler from config with a custom chunking implementation.
    pub fn new_with_chunker(
        config: SamplerConfig,
        split_store: Arc<S>,
        chunker: Arc<dyn ChunkingAlgorithm>,
    ) -> Self {
        let inner = TripletSamplerInner::new_with_chunker(config, split_store, chunker);
        Self {
            inner: Mutex::new(inner),
        }
    }

    /// Return an unweighted pair batch for `split`.
    pub fn next_pair_batch_for_split(
        &self,
        split: SplitLabel,
    ) -> Result<SampleBatch, SamplerError> {
        self.next_pair_batch_with_weights_for_split(split, &HashMap::new())
    }

    /// Return an unweighted text batch for `split`.
    pub fn next_text_batch_for_split(&self, split: SplitLabel) -> Result<TextBatch, SamplerError> {
        self.next_text_batch_with_weights_for_split(split, &HashMap::new())
    }

    /// Return an unweighted triplet batch for `split`.
    pub fn next_triplet_batch_for_split(
        &self,
        split: SplitLabel,
    ) -> Result<TripletBatch, SamplerError> {
        self.next_triplet_batch_with_weights_for_split(split, &HashMap::new())
    }

    /// Return a weighted pair batch for `split` using per-source weights.
    pub fn next_pair_batch_with_weights_for_split(
        &self,
        split: SplitLabel,
        weights: &HashMap<SourceId, f32>,
    ) -> Result<SampleBatch, SamplerError> {
        let mut inner = self.inner.lock().unwrap();
        inner.ensure_split_allowed(split)?;
        for attempt in 0..=EXHAUSTION_RETRY_LIMIT {
            match inner.next_pair_batch_inner_with_weights(split, Some(weights)) {
                Ok(batch) => return Ok(batch),
                Err(SamplerError::Exhausted(_)) => {
                    if attempt < EXHAUSTION_RETRY_LIMIT {
                        inner.force_ingest_refresh_with_weights_for_split(split, weights)?;
                    }
                }
                Err(err) => return Err(err),
            }
        }
        Err(SamplerError::Exhausted(RECIPE_LABEL_TRIPLETS.into()))
    }

    /// Return a weighted text batch for `split` using per-source weights.
    pub fn next_text_batch_with_weights_for_split(
        &self,
        split: SplitLabel,
        weights: &HashMap<SourceId, f32>,
    ) -> Result<TextBatch, SamplerError> {
        let mut inner = self.inner.lock().unwrap();
        inner.ensure_split_allowed(split)?;
        for attempt in 0..=EXHAUSTION_RETRY_LIMIT {
            match inner.next_text_batch_inner_with_weights(split, Some(weights)) {
                Ok(batch) => return Ok(batch),
                Err(SamplerError::Exhausted(_)) => {
                    if attempt < EXHAUSTION_RETRY_LIMIT {
                        inner.force_ingest_refresh_with_weights_for_split(split, weights)?;
                    }
                }
                Err(err) => return Err(err),
            }
        }
        Err(SamplerError::Exhausted(RECIPE_LABEL_TEXT.into()))
    }

    /// Return a weighted triplet batch for `split` using per-source weights.
    pub fn next_triplet_batch_with_weights_for_split(
        &self,
        split: SplitLabel,
        weights: &HashMap<SourceId, f32>,
    ) -> Result<TripletBatch, SamplerError> {
        let mut inner = self.inner.lock().unwrap();
        inner.ensure_split_allowed(split)?;
        for attempt in 0..=EXHAUSTION_RETRY_LIMIT {
            match inner.next_triplet_batch_inner_with_weights(split, Some(weights)) {
                Ok(batch) => return Ok(batch),
                Err(SamplerError::Exhausted(_)) => {
                    if attempt < EXHAUSTION_RETRY_LIMIT {
                        inner.force_ingest_refresh_with_weights_for_split(split, weights)?;
                    }
                }
                Err(err) => return Err(err),
            }
        }
        Err(SamplerError::Exhausted(RECIPE_LABEL_TRIPLETS.into()))
    }

    /// Spawn a background prefetcher for triplet batches.
    pub fn prefetch_triplet_batches(
        self: Arc<Self>,
        split: SplitLabel,
        capacity: usize,
    ) -> BatchPrefetcher<TripletBatch> {
        BatchPrefetcher::new(capacity, move || self.next_triplet_batch_for_split(split))
    }

    /// Spawn a background prefetcher for weighted triplet batches.
    pub fn prefetch_triplet_batches_with_weights(
        self: Arc<Self>,
        split: SplitLabel,
        capacity: usize,
        weights: HashMap<SourceId, f32>,
    ) -> BatchPrefetcher<TripletBatch> {
        BatchPrefetcher::new(capacity, move || {
            self.next_triplet_batch_with_weights_for_split(split, &weights)
        })
    }

    /// Spawn a background prefetcher for pair batches.
    pub fn prefetch_pair_batches(
        self: Arc<Self>,
        split: SplitLabel,
        capacity: usize,
    ) -> BatchPrefetcher<SampleBatch> {
        BatchPrefetcher::new(capacity, move || self.next_pair_batch_for_split(split))
    }

    /// Spawn a background prefetcher for weighted pair batches.
    pub fn prefetch_pair_batches_with_weights(
        self: Arc<Self>,
        split: SplitLabel,
        capacity: usize,
        weights: HashMap<SourceId, f32>,
    ) -> BatchPrefetcher<SampleBatch> {
        BatchPrefetcher::new(capacity, move || {
            self.next_pair_batch_with_weights_for_split(split, &weights)
        })
    }

    /// Spawn a background prefetcher for text batches.
    pub fn prefetch_text_batches(
        self: Arc<Self>,
        split: SplitLabel,
        capacity: usize,
    ) -> BatchPrefetcher<TextBatch> {
        BatchPrefetcher::new(capacity, move || self.next_text_batch_for_split(split))
    }

    /// Spawn a background prefetcher for weighted text batches.
    pub fn prefetch_text_batches_with_weights(
        self: Arc<Self>,
        split: SplitLabel,
        capacity: usize,
        weights: HashMap<SourceId, f32>,
    ) -> BatchPrefetcher<TextBatch> {
        BatchPrefetcher::new(capacity, move || {
            self.next_text_batch_with_weights_for_split(split, &weights)
        })
    }

    /// Return the currently active text recipes.
    pub fn text_recipes(&self) -> Vec<TextRecipe> {
        let inner = self.inner.lock().unwrap();
        inner.text_recipes().to_vec()
    }

    /// Register a data source for ingestion and sampling.
    pub fn register_source(&self, source: Box<dyn DataSource + 'static>) {
        let mut inner = self.inner.lock().unwrap();
        inner.register_source(source);
    }

    /// Force sampler epoch to `epoch` (advanced deterministic replay control).
    pub fn set_epoch(&self, epoch: u64) -> Result<(), SamplerError> {
        let mut inner = self.inner.lock().unwrap();
        inner.set_epoch(epoch)
    }

    /// Persist sampler and split runtime state for restart-resume.
    ///
    /// When `save_to` is `Some(path)`, current persisted runtime state is also
    /// mirrored to `path` when supported by the split-store backend.
    pub fn save_sampler_state(&self, save_to: Option<&Path>) -> Result<(), SamplerError> {
        let mut inner = self.inner.lock().unwrap();
        inner.save_sampler_state(save_to)
    }

    /// Return cumulative `(fallback_count, selection_count)` from the BM25
    /// backend since it was created.  Only available when both `bm25-mining`
    /// and `extended-metrics` features are active.
    ///
    /// `selection_count` counts calls where the pool was non-empty.
    /// `fallback_count` counts the subset where BM25 produced no candidates
    /// intersecting the pool and random selection was used instead.
    #[cfg(all(feature = "bm25-mining", feature = "extended-metrics"))]
    pub fn bm25_fallback_stats(&self) -> (u64, u64) {
        let inner = self.inner.lock().unwrap();
        inner.bm25_fallback_stats()
    }
}

impl<S: SplitStore + EpochStateStore + SamplerStateStore + 'static> Sampler for TripletSampler<S> {
    fn next_pair_batch(&self, split: SplitLabel) -> Result<SampleBatch, SamplerError> {
        self.next_pair_batch_for_split(split)
    }

    fn next_pair_batch_with_weights(
        &self,
        split: SplitLabel,
        weights: &HashMap<SourceId, f32>,
    ) -> Result<SampleBatch, SamplerError> {
        self.next_pair_batch_with_weights_for_split(split, weights)
    }

    fn next_text_batch(&self, split: SplitLabel) -> Result<TextBatch, SamplerError> {
        self.next_text_batch_for_split(split)
    }

    fn next_text_batch_with_weights(
        &self,
        split: SplitLabel,
        weights: &HashMap<SourceId, f32>,
    ) -> Result<TextBatch, SamplerError> {
        self.next_text_batch_with_weights_for_split(split, weights)
    }

    fn next_triplet_batch(&self, split: SplitLabel) -> Result<TripletBatch, SamplerError> {
        self.next_triplet_batch_for_split(split)
    }

    fn next_triplet_batch_with_weights(
        &self,
        split: SplitLabel,
        weights: &HashMap<SourceId, f32>,
    ) -> Result<TripletBatch, SamplerError> {
        self.next_triplet_batch_with_weights_for_split(split, weights)
    }
}

fn roles_match(target: &SectionRole, candidate: &SectionRole) -> bool {
    target == candidate
}

fn role_cursor_key(record_id: &RecordId, role: &SectionRole) -> (RecordId, String) {
    (record_id.clone(), role_label(role))
}

fn role_label(role: &SectionRole) -> String {
    match role {
        SectionRole::Anchor => ROLE_LABEL_ANCHOR.into(),
        SectionRole::Context => ROLE_LABEL_CONTEXT.into(),
    }
}

fn taxonomy_value(record: &DataRecord, field: MetadataKey) -> Option<&str> {
    record.taxonomy.iter().find_map(|entry| field.strip(entry))
}

fn strategy_reason(strategy: &NegativeStrategy) -> &'static str {
    match strategy {
        NegativeStrategy::WrongPublicationDate => NEG_REASON_WRONG_DATE,
        NegativeStrategy::WrongArticle => NEG_REASON_WRONG_ARTICLE,
        NegativeStrategy::QuestionAnswerMismatch => NEG_REASON_WRONG_QA,
    }
}

fn chunk_key(chunk: &RecordChunk) -> String {
    match &chunk.view {
        ChunkView::Window { index, .. } => {
            format!("{}|{}|w|{}", chunk.record_id, chunk.section_idx, index)
        }
        ChunkView::SummaryFallback { strategy, .. } => {
            format!("{}|{}|s|{}", chunk.record_id, chunk.section_idx, strategy)
        }
    }
}

fn pad_with_reuse<T: Clone>(items: &mut Vec<T>, target: usize) {
    if items.is_empty() || items.len() >= target {
        return;
    }
    let seed = items.clone();
    let base_len = seed.len();
    for idx in 0..(target - items.len()) {
        items.push(seed[idx % base_len].clone());
    }
}

#[cfg(test)]
mod tests;
