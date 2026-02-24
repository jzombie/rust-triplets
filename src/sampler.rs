use chrono::Duration;
use indexmap::IndexMap;
use rand::prelude::*;
use rand::seq::IndexedRandom;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;

use crate::config::{
    ChunkingStrategy, NegativeStrategy, SamplerConfig, Selector, TextRecipe, TripletRecipe,
};
use crate::constants::sampler::{
    EPOCH_SEED_OFFSET, EXHAUSTION_RETRY_LIMIT, NEG_REASON_WRONG_ARTICLE, NEG_REASON_WRONG_DATE,
    NEG_REASON_WRONG_QA, PREFETCHER_SOURCE_ID, PREFETCHER_STOPPED_REASON, RECIPE_LABEL_TEXT,
    RECIPE_LABEL_TRIPLETS, ROLE_LABEL_ANCHOR, ROLE_LABEL_CONTEXT,
};
use crate::data::{
    ChunkView, DataRecord, PairLabel, RecordChunk, RecordSection, SampleBatch, SamplePair,
    SampleTriplet, SectionRole, TextBatch, TextSample, TripletBatch,
};
use crate::epoch::EpochTracker;
use crate::errors::SamplerError;
use crate::hash::stable_hash_str;
use crate::ingestion::IngestionManager;
use crate::metadata::{META_FIELD_DATE, MetadataKey};
use crate::source::DataSource;
use crate::splits::{
    EpochStateStore, PersistedSamplerState, SamplerStateStore, SplitLabel, SplitStore,
};
use crate::types::{RecipeKey, RecordId, SourceId};

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
/// - `trust` (range 0.0–1.0): comes from `RecordChunk.quality.trust`; scales the contribution of a chunk.
/// - `start_ratio` (range 0.0–1.0): fraction of the section where the window starts; 0.0 is the first token.
/// - `base`: window chunks use `1.0 - start_ratio`; summary chunks use their configured `summary_fallback_weight`.
/// - `chunk_weight_floor`: minimum weight applied after scaling.
///
/// Formula: `max(chunk_weight_floor, base * trust)`.
pub fn chunk_weight(strategy: &ChunkingStrategy, chunk: &RecordChunk) -> f32 {
    let floor = strategy.chunk_weight_floor;
    let trust = chunk.quality.trust.clamp(0.0, 1.0);
    let base = match &chunk.view {
        ChunkView::Window { start_ratio, .. } => 1.0 - start_ratio,
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
pub struct PairSampler<S: SplitStore + EpochStateStore + SamplerStateStore + 'static> {
    inner: Mutex<PairSamplerInner<S>>,
}

/// Internal sampler state implementation guarded by `PairSampler`.
struct PairSamplerInner<S: SplitStore + EpochStateStore + SamplerStateStore + 'static> {
    /// Immutable sampler configuration (seed, batch size, recipes, splits, etc.).
    config: SamplerConfig,
    /// Split store backing train/val/test assignments and persisted sampler state.
    split_store: Arc<S>,
    /// On-demand ingestion manager that fills the batch-sized buffer.
    ingestion: IngestionManager,
    /// Current in-memory record pool keyed by record id.
    records: IndexMap<RecordId, DataRecord>,
    /// Deterministic RNG for per-batch shuffles and sampling.
    rng: DeterministicRng,
    /// Config-level triplet recipes used when sources do not supply their own.
    triplet_recipes: Vec<TripletRecipe>,
    /// Config-level text recipes used when sources do not supply their own.
    text_recipes: Vec<TextRecipe>,
    /// Per-source triplet recipes keyed by source id.
    source_triplet_recipes: HashMap<SourceId, Vec<TripletRecipe>>,
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

impl<S: SplitStore + EpochStateStore + SamplerStateStore + 'static> PairSamplerInner<S> {
    fn new(config: SamplerConfig, split_store: Arc<S>) -> Self {
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
        let epoch_backend = Some(Arc::clone(&split_store) as Arc<dyn EpochStateStore>);
        let epoch_tracker = EpochTracker::new(true, epoch_backend, config.seed ^ EPOCH_SEED_OFFSET);
        let mut sampler = Self {
            rng: DeterministicRng::new(config.seed),
            config,
            split_store,
            ingestion: IngestionManager::new(buffer_size),
            records: IndexMap::new(),
            triplet_recipes,
            text_recipes,
            source_triplet_recipes: HashMap::new(),
            source_text_recipes: HashMap::new(),
            using_config_triplet_recipes,
            using_config_text_recipes,
            last_observed_ingest: 0,
            epoch_tracker,
            chunk_cursors: HashMap::new(),
            role_cursors: HashMap::new(),
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

    fn register_source(&mut self, source: Box<dyn DataSource + 'static>) {
        source.configure_sampler(&self.config);
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
            let start = (stable_hash_str(self.config.seed ^ self.source_epoch, &cursor_key)
                as usize)
                % pool.len();
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
        if self.chunk_cursors.is_empty() && self.role_cursors.is_empty() {
            return;
        }
        let valid_ids: HashSet<RecordId> = self.records.keys().cloned().collect();
        self.chunk_cursors
            .retain(|(record_id, _), _| valid_ids.contains(record_id));
        self.role_cursors
            .retain(|(record_id, _), _| valid_ids.contains(record_id));
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

        let shuffle_seed = self.config.seed ^ self.source_epoch;
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
        let seed = self.config.seed ^ self.source_epoch ^ cycle;
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
            self.rng = DeterministicRng::from_state(state.rng_state);
            self.triplet_recipe_rr_idx = state.triplet_recipe_rr_idx as usize;
            self.text_recipe_rr_idx = state.text_recipe_rr_idx as usize;
        }
        self.refresh_source_wrapped();
        self.source_state_loaded = true;
        self.source_state_dirty = true;
        Ok(())
    }

    fn persist_source_state(&mut self) -> Result<(), SamplerError> {
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
        self.split_store.store_sampler_state(&state)?;
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

    fn triplet_recipes_for_source<'a>(&'a self, source: &str) -> &'a [TripletRecipe] {
        if self.using_config_triplet_recipes {
            return &self.triplet_recipes;
        }
        self.source_triplet_recipes
            .get(source)
            .map(|recipes| recipes.as_slice())
            .unwrap_or(&[])
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

    fn recipe_order_shuffled(&mut self, count: usize) -> Vec<usize> {
        if count == 0 {
            return Vec::new();
        }
        let mut order: Vec<usize> = (0..count).collect();
        order.shuffle(&mut self.rng);
        order
    }

    fn recipe_order_cycled(&mut self, count: usize, rr_idx: usize) -> Vec<usize> {
        let base = self.recipe_order_shuffled(count);
        if base.is_empty() {
            return base;
        }
        let start = rr_idx % base.len();
        let mut order = Vec::with_capacity(base.len());
        order.extend_from_slice(&base[start..]);
        order.extend_from_slice(&base[..start]);
        order
    }

    fn text_recipe_order_shuffled(&mut self, count: usize) -> Vec<usize> {
        if count == 0 {
            return Vec::new();
        }
        let mut order: Vec<usize> = (0..count).collect();
        order.shuffle(&mut self.rng);
        order
    }

    fn text_recipe_order_cycled(&mut self, count: usize, rr_idx: usize) -> Vec<usize> {
        let base = self.text_recipe_order_shuffled(count);
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
    ) -> Option<DataRecord> {
        if let Some(source) = source {
            let indices = self.source_record_indices.get(source)?;
            if indices.is_empty() {
                return None;
            }
            let mut cursor = *self.source_record_cursors.get(source).unwrap_or(&0);
            let cycle = cursor / indices.len();
            let offset_seed = self.config.seed ^ self.source_epoch ^ (cycle as u64);
            let offset = (stable_hash_str(offset_seed, source) as usize) % indices.len();
            let mut wrapped = false;
            let mut selected: Option<DataRecord> = None;
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
                    selected = Some(record.clone());
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
                return Some(record.clone());
            }
        }
        None
    }

    fn persist_state(&mut self) -> Result<(), SamplerError> {
        if self.epoch_tracker.is_enabled() {
            self.epoch_tracker.persist()?;
        }
        self.persist_source_state()?;
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
    ) -> Option<DataRecord> {
        let target = record.created_at + Duration::days(offset_days.into());
        let key = record.taxonomy.first().cloned();
        self.records
            .values()
            .filter(|candidate| {
                candidate.id != record.id
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
        &mut self,
        anchor_record: &DataRecord,
        strategy: &NegativeStrategy,
    ) -> Option<(DataRecord, bool)> {
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
                let mut same_date: Vec<DataRecord> = self
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
                    return same_date
                        .choose(&mut self.rng)
                        .cloned()
                        .map(|record| (record, false));
                }
                // Fallback to any other record in the same split so split boundaries
                // remain strictly isolated.
                self.records
                    .values()
                    .filter(|candidate| {
                        candidate.id != anchor_record.id && in_anchor_split(candidate)
                    })
                    .cloned()
                    .collect::<Vec<_>>()
                    .choose(&mut self.rng)
                    .cloned()
                    .map(|record| (record, true))
            }
            NegativeStrategy::WrongPublicationDate => {
                let anchor_date =
                    taxonomy_value(anchor_record, META_FIELD_DATE).map(|d| d.to_string());
                let pool: Vec<DataRecord> = self
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
                    return self
                        .records
                        .values()
                        .filter(|candidate| {
                            candidate.id != anchor_record.id && in_anchor_split(candidate)
                        })
                        .cloned()
                        .collect::<Vec<_>>()
                        .choose(&mut self.rng)
                        .cloned()
                        .map(|record| (record, true));
                }
                pool.choose(&mut self.rng)
                    .cloned()
                    .map(|record| (record, false))
            }
            NegativeStrategy::QuestionAnswerMismatch => {
                let pool: Vec<DataRecord> = self
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
                    return self
                        .records
                        .values()
                        .filter(|candidate| {
                            candidate.id != anchor_record.id && in_anchor_split(candidate)
                        })
                        .cloned()
                        .collect::<Vec<_>>()
                        .choose(&mut self.rng)
                        .cloned()
                        .map(|record| (record, true));
                }
                pool.choose(&mut self.rng)
                    .cloned()
                    .map(|record| (record, false))
            }
        }
    }

    fn make_triplet_with_anchor(
        &mut self,
        recipe: &TripletRecipe,
        record: &DataRecord,
    ) -> Option<SampleTriplet> {
        let mut anchor_chunk = self.select_chunk(record, &recipe.anchor)?;
        self.decorate_chunk(record, &mut anchor_chunk);
        // TODO: When anchor and positive selectors overlap, consider re-drawing positives
        // to avoid identical chunks when multiple windows are available.
        let mut positive_chunk = self.select_chunk(record, &recipe.positive_selector)?;
        self.decorate_chunk(record, &mut positive_chunk);
        let (negative_record, fallback_used) =
            self.select_negative_record(record, &recipe.negative_strategy)?;
        let mut negative_chunk = self.select_chunk(&negative_record, &recipe.negative_selector)?;
        self.decorate_chunk(&negative_record, &mut negative_chunk);
        let weight = recipe.weight
            * self.triplet_chunk_weight(&anchor_chunk, &positive_chunk, &negative_chunk);
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

    fn make_text_sample_for_split(
        &mut self,
        recipe: &TextRecipe,
        source: Option<&str>,
        split: SplitLabel,
    ) -> Option<TextSample> {
        let record = self.choose_anchor_record(source, split)?;
        let mut chunk = self.select_chunk(&record, &recipe.selector)?;
        self.decorate_chunk(&record, &mut chunk);
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
        anchor: &RecordChunk,
        positive: &RecordChunk,
        negative: &RecordChunk,
    ) -> f32 {
        (self.chunk_weight(anchor) + self.chunk_weight(positive) + self.chunk_weight(negative))
            / 3.0
    }

    fn decorate_chunk(&mut self, record: &DataRecord, chunk: &mut RecordChunk) {
        if let Some(spec) = record.meta_prefix.as_ref()
            && let Some(prefix) = spec.sample(&mut self.rng)
        {
            chunk.text = format!("{}\n{}", prefix, chunk.text);
        }
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
                (stable_hash_str(self.config.seed ^ self.source_epoch, &seed_key) as usize)
                    % indices.len()
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

    fn materialize_chunks(
        &self,
        record: &DataRecord,
        section_idx: usize,
        section: &RecordSection,
    ) -> Vec<RecordChunk> {
        let strategy = &self.config.chunking;
        let text = section.text.as_str();
        let tokens: Vec<&str> = text.split_whitespace().collect();
        if tokens.is_empty() {
            return Vec::new();
        }
        let mut chunks = Vec::new();
        let total_tokens = tokens.len();
        let span = strategy.max_window_tokens.min(total_tokens);
        if span == tokens.len() {
            let text = text.to_string();
            chunks.push(RecordChunk {
                record_id: record.id.clone(),
                section_idx,
                view: ChunkView::Window {
                    index: 0,
                    overlap: 0,
                    span,
                    start_ratio: 0.0,
                },
                text,
                tokens_estimate: span,
                quality: record.quality,
            });
            return chunks;
        }
        for overlap in &strategy.overlap_tokens {
            let stride = span.saturating_sub(*overlap).max(1);
            let mut start = 0;
            let mut index = 0;
            while start < tokens.len() {
                let end = (start + span).min(tokens.len());
                let window = tokens[start..end].join(" ");
                chunks.push(RecordChunk {
                    record_id: record.id.clone(),
                    section_idx,
                    view: ChunkView::Window {
                        index,
                        overlap: *overlap,
                        span,
                        start_ratio: start as f32 / total_tokens as f32,
                    },
                    text: window,
                    tokens_estimate: end - start,
                    quality: record.quality,
                });
                if end == tokens.len() {
                    break;
                }
                start += stride;
                index += 1;
            }
        }
        if tokens.len() > strategy.max_window_tokens && strategy.summary_fallback_tokens > 0 {
            let fallback_cap = strategy
                .summary_fallback_tokens
                .min(strategy.max_window_tokens)
                .max(1);
            let fallback_len = tokens.len().min(fallback_cap);
            let summary_tokens = tokens
                .iter()
                .take(fallback_len)
                .copied()
                .collect::<Vec<_>>()
                .join(" ");
            chunks.push(RecordChunk {
                record_id: record.id.clone(),
                section_idx,
                view: ChunkView::SummaryFallback {
                    strategy: "head".into(),
                    weight: strategy.summary_fallback_weight,
                },
                text: summary_tokens,
                tokens_estimate: fallback_len,
                quality: record.quality,
            });
        }
        chunks
    }

    fn build_derived_text_recipes(recipes: &[TripletRecipe]) -> Vec<TextRecipe> {
        let mut derived = Vec::new();
        for recipe in recipes {
            let base = recipe.name.as_ref();
            derived.push(TextRecipe {
                name: Cow::Owned(format!("{base}_anchor")),
                selector: recipe.anchor.clone(),
                weight: recipe.weight.max(0.0001),
                instruction: None,
            });
            derived.push(TextRecipe {
                name: Cow::Owned(format!("{base}_positive")),
                selector: recipe.positive_selector.clone(),
                weight: recipe.weight.max(0.0001),
                instruction: None,
            });
            derived.push(TextRecipe {
                name: Cow::Owned(format!("{base}_negative")),
                selector: recipe.negative_selector.clone(),
                weight: recipe.weight.max(0.0001),
                instruction: None,
            });
        }
        derived
    }

    fn sync_records_from_cache(&mut self) -> Result<(), SamplerError> {
        let mut snapshot = self.ingestion.cache().snapshot();
        snapshot.sort_by(|a, b| a.id.cmp(&b.id));
        self.records.clear();
        for record in snapshot {
            if self.split_store.label_for(&record.id).is_none() {
                self.split_store.ensure(record.id.clone())?;
            }
            self.records.insert(record.id.clone(), record);
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
            }
            self.ingestion_cursors_loaded = true;
        }
        if self.ingestion.cache().is_empty() {
            self.ingestion.refresh_all();
        } else {
            self.ingestion.advance(self.config.batch_size);
        }
        let observed = self.ingestion.cache().ingest_count();
        if observed == self.last_observed_ingest && !self.records.is_empty() {
            return Ok(());
        }
        self.last_observed_ingest = observed;
        self.sync_records_from_cache()?;
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
            }
            self.ingestion_cursors_loaded = true;
        }
        if self.ingestion.cache().is_empty() {
            self.ingestion.refresh_all_with_weights(weights);
        } else {
            self.ingestion
                .advance_with_weights(self.config.batch_size, weights);
        }
        let observed = self.ingestion.cache().ingest_count();
        if observed == self.last_observed_ingest && !self.records.is_empty() {
            return Ok(());
        }
        self.last_observed_ingest = observed;
        self.sync_records_from_cache()?;
        self.epoch_tracker.ensure_loaded()?;
        let records_by_split = self.records_by_split()?;
        self.epoch_tracker
            .reconcile(target_split, &records_by_split);
        self.ensure_source_state()?;
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
            }
            self.ingestion_cursors_loaded = true;
        }
        self.ingestion.force_refresh_all_with_weights(weights);
        self.last_observed_ingest = self.ingestion.cache().ingest_count();
        self.sync_records_from_cache()?;
        self.epoch_tracker.ensure_loaded()?;
        let records_by_split = self.records_by_split()?;
        self.epoch_tracker
            .reconcile(target_split, &records_by_split);
        self.ensure_source_state()?;
        Ok(())
    }

    fn next_pair_batch_inner_with_weights(
        &mut self,
        target_split: SplitLabel,
        weights: Option<&HashMap<SourceId, f32>>,
    ) -> Result<SampleBatch, SamplerError> {
        if let Some(weights) = weights {
            self.ingest_internal_with_weights_for_split(target_split, weights)?;
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
            let recipe_order =
                self.recipe_order_cycled(self.triplet_recipes.len(), self.triplet_recipe_rr_idx);
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
                    if let Some(sample) = self.make_triplet_with_anchor(&recipe, &anchor) {
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
            .map(|source| self.triplet_recipes_for_source(source).len())
            .max()
            .unwrap_or(1)
            .max(1);
        let attempts = self.config.batch_size * 4 * sources.len() * max_recipe_len;
        for _ in 0..attempts {
            if pairs.len() >= self.config.batch_size {
                break;
            }
            let source = cycle_sources[source_idx].as_str();
            let recipes = self.triplet_recipes_for_source(source).to_vec();
            if recipes.is_empty() {
                source_idx += 1;
                source_steps += 1;
                if source_idx >= cycle_sources.len() {
                    source_idx = 0;
                    cycle = cycle.saturating_add(1);
                    cycle_sources = self.shuffled_source_cycle(cycle);
                }
                continue;
            }
            if !recipe_orders.contains_key(source) {
                let order = self.recipe_order_cycled(recipes.len(), self.triplet_recipe_rr_idx);
                recipe_orders.insert(source.to_string(), order);
            }
            let order = recipe_orders
                .get(source)
                .expect("recipe order missing for source");
            let pos = recipe_positions.entry(source.to_string()).or_insert(0);
            let Some(anchor) = self.choose_anchor_record(Some(source), target_split) else {
                source_idx += 1;
                source_steps += 1;
                if source_idx >= cycle_sources.len() {
                    source_idx = 0;
                    cycle = cycle.saturating_add(1);
                    cycle_sources = self.shuffled_source_cycle(cycle);
                }
                continue;
            };
            let mut triplet = None;
            for offset in 0..order.len() {
                let idx = order[(*pos + offset) % order.len()];
                recipe_steps = recipe_steps.saturating_add(1);
                let recipe = recipes[idx].clone();
                if let Some(sample) = self.make_triplet_with_anchor(&recipe, &anchor) {
                    *pos = (*pos + offset + 1) % order.len();
                    triplet = Some((recipe, sample));
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
        if let Some(weights) = weights {
            self.ingest_internal_with_weights_for_split(target_split, weights)?;
        } else {
            self.ingest_internal_for_split(target_split)?;
        }
        self.ensure_split_has_records(target_split)?;
        let sources = self.source_order.clone();
        if sources.is_empty() {
            if self.text_recipes.is_empty() {
                return Err(SamplerError::Configuration(
                    "no text recipes configured".into(),
                ));
            }
            let recipe_order =
                self.text_recipe_order_cycled(self.text_recipes.len(), self.text_recipe_rr_idx);
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
                if let Some(sample) = self.make_text_sample_for_split(&recipe, None, target_split) {
                    let key = chunk_key(&sample.chunk);
                    if seen.insert(key) {
                        samples.push(sample);
                    }
                }
            }
            if recipe_steps > 0 {
                self.text_recipe_rr_idx = self.text_recipe_rr_idx.saturating_add(recipe_steps);
            }
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
                let order = self.text_recipe_order_cycled(recipes.len(), self.text_recipe_rr_idx);
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
                    self.make_text_sample_for_split(&recipe, Some(source), target_split)
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
            return Err(SamplerError::Exhausted(RECIPE_LABEL_TEXT.into()));
        }
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
        if let Some(weights) = weights {
            self.ingest_internal_with_weights_for_split(target_split, weights)?;
        } else {
            self.ingest_internal_for_split(target_split)?;
        }
        self.ensure_split_has_records(target_split)?;
        let sources = self.source_order.clone();
        if sources.is_empty() {
            if self.triplet_recipes.is_empty() {
                return Err(SamplerError::Configuration(
                    "no triplet recipes configured".into(),
                ));
            }
            let recipe_order =
                self.recipe_order_cycled(self.triplet_recipes.len(), self.triplet_recipe_rr_idx);
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
                    if let Some(sample) = self.make_triplet_with_anchor(&recipe, &anchor) {
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
            if triplets.len() == self.config.batch_size {
                return Ok(TripletBatch { triplets });
            }
            return Err(SamplerError::Exhausted(
                last_recipe_name
                    .unwrap_or(Cow::Borrowed(RECIPE_LABEL_TRIPLETS))
                    .to_string(),
            ));
        }

        let mut triplets = Vec::new();
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
            .map(|source| self.triplet_recipes_for_source(source).len())
            .max()
            .unwrap_or(1)
            .max(1);
        let attempts = self.config.batch_size * 4 * sources.len() * max_recipe_len;
        for _ in 0..attempts {
            if triplets.len() >= self.config.batch_size {
                break;
            }
            let source = cycle_sources[source_idx].as_str();
            let recipes = self.triplet_recipes_for_source(source).to_vec();
            if recipes.is_empty() {
                source_idx += 1;
                source_steps += 1;
                if source_idx >= cycle_sources.len() {
                    source_idx = 0;
                    cycle = cycle.saturating_add(1);
                    cycle_sources = self.shuffled_source_cycle(cycle);
                }
                continue;
            }
            if !recipe_orders.contains_key(source) {
                let order = self.recipe_order_cycled(recipes.len(), self.triplet_recipe_rr_idx);
                recipe_orders.insert(source.to_string(), order);
            }
            let order = recipe_orders
                .get(source)
                .expect("recipe order missing for source");
            let pos = recipe_positions.entry(source.to_string()).or_insert(0);
            let Some(anchor) = self.choose_anchor_record(Some(source), target_split) else {
                source_idx += 1;
                source_steps += 1;
                if source_idx >= cycle_sources.len() {
                    source_idx = 0;
                    cycle = cycle.saturating_add(1);
                    cycle_sources = self.shuffled_source_cycle(cycle);
                }
                continue;
            };
            let mut triplet: Option<(TripletRecipe, SampleTriplet)> = None;
            for offset in 0..order.len() {
                let idx = order[(*pos + offset) % order.len()];
                let recipe = recipes[idx].clone();
                recipe_steps = recipe_steps.saturating_add(1);
                if let Some(sample) = self.make_triplet_with_anchor(&recipe, &anchor) {
                    *pos = (*pos + offset + 1) % order.len();
                    triplet = Some((recipe, sample));
                    break;
                }
            }
            if let Some((_recipe, triplet)) = triplet {
                let key = (
                    triplet.anchor.record_id.clone(),
                    triplet.positive.record_id.clone(),
                    triplet.negative.record_id.clone(),
                );
                if seen.insert(key) {
                    triplets.push(triplet);
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
        if triplets.len() == self.config.batch_size {
            self.source_cycle_idx = self.source_cycle_idx.saturating_add(source_steps);
            self.source_state_dirty = sources.len() > 1;
            let batch = TripletBatch { triplets };
            return Ok(batch);
        }
        Err(SamplerError::Exhausted(RECIPE_LABEL_TRIPLETS.into()))
    }
}

impl<S: SplitStore + EpochStateStore + SamplerStateStore + 'static> PairSampler<S> {
    /// Create a sampler from config and a split-state backend.
    pub fn new(config: SamplerConfig, split_store: Arc<S>) -> Self {
        let inner = PairSamplerInner::new(config, split_store);
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
                Err(SamplerError::Exhausted(_)) if attempt < EXHAUSTION_RETRY_LIMIT => {
                    inner.force_ingest_refresh_with_weights_for_split(split, weights)?;
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
                Err(SamplerError::Exhausted(_)) if attempt < EXHAUSTION_RETRY_LIMIT => {
                    inner.force_ingest_refresh_with_weights_for_split(split, weights)?;
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
                Err(SamplerError::Exhausted(_)) if attempt < EXHAUSTION_RETRY_LIMIT => {
                    inner.force_ingest_refresh_with_weights_for_split(split, weights)?;
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
    pub fn persist_state(&self) -> Result<(), SamplerError> {
        let mut inner = self.inner.lock().unwrap();
        inner.persist_state()
    }
}

impl<S: SplitStore + EpochStateStore + SamplerStateStore + 'static> Sampler for PairSampler<S> {
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

#[cfg(test)]
mod tests {
    fn base_config() -> super::SamplerConfig {
        super::SamplerConfig::default()
    }

    use super::*;
    use crate::config::{ChunkingStrategy, NegativeStrategy, Selector, TextRecipe, TripletRecipe};
    use crate::constants::sampler_tests::{
        FNV1A64_OFFSET, FNV1A64_PRIME, FULL_SEQUENCE_LEN, PAIR_BATCH_SEQUENCE_HASH,
        PREFETCH_PAIR_BATCH_SEQUENCE_HASH, PREFETCH_TEXT_BATCH_SEQUENCE_HASH,
        PREFETCH_TRIPLET_BATCH_SEQUENCE_HASH, PRIMARY_SOURCE_ID, SECONDARY_SOURCE_ID,
        TEXT_BATCH_SEQUENCE_HASH, TRIPLET_BATCH_SEQUENCE_HASH,
    };
    use crate::data::{ChunkView, QualityScore, RecordChunk, RecordSection};
    use crate::kvp::{KvpField, KvpPrefixSampler};
    use crate::metadata::META_FIELD_DATE;
    use crate::source::{DataSource, InMemorySource, SourceCursor, SourceSnapshot};
    use crate::splits::{
        DeterministicSplitStore, FileSplitStore, SplitLabel, SplitRatios, SplitStore,
    };
    use chrono::Utc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use std::time::Duration as StdDuration;
    use tempfile::tempdir;

    /// `DataSource` wrapper that exposes custom default recipes in tests.
    struct RecipeSource {
        inner: InMemorySource,
        triplet_recipes: Vec<TripletRecipe>,
    }

    #[test]
    fn role_helpers_and_taxonomy_value_cover_branches() {
        assert!(roles_match(&SectionRole::Anchor, &SectionRole::Anchor));
        assert!(!roles_match(&SectionRole::Anchor, &SectionRole::Context));

        let key = role_cursor_key(&"rec-1".to_string(), &SectionRole::Anchor);
        assert_eq!(key.0, "rec-1");
        assert_eq!(key.1, role_label(&SectionRole::Anchor));
        assert_ne!(
            role_label(&SectionRole::Anchor),
            role_label(&SectionRole::Context)
        );

        let mut record = sample_record();
        record.taxonomy = vec!["source_a".into(), META_FIELD_DATE.encode("2026-02-23")];
        assert_eq!(taxonomy_value(&record, META_FIELD_DATE), Some("2026-02-23"));

        record.taxonomy = vec!["source_a".into(), "other=value".into()];
        assert_eq!(taxonomy_value(&record, META_FIELD_DATE), None);
    }

    #[test]
    fn strategy_reason_and_chunk_key_cover_all_variants() {
        let reason_a = strategy_reason(&NegativeStrategy::WrongPublicationDate);
        let reason_b = strategy_reason(&NegativeStrategy::WrongArticle);
        let reason_c = strategy_reason(&NegativeStrategy::QuestionAnswerMismatch);
        assert!(!reason_a.is_empty());
        assert!(!reason_b.is_empty());
        assert!(!reason_c.is_empty());
        assert_ne!(reason_a, reason_b);
        assert_ne!(reason_b, reason_c);

        let base = RecordChunk {
            record_id: "r1".into(),
            section_idx: 0,
            view: ChunkView::Window {
                index: 2,
                overlap: 0,
                span: 8,
                start_ratio: 0.25,
            },
            text: "window".into(),
            tokens_estimate: 8,
            quality: QualityScore { trust: 1.0 },
        };
        let key_window = chunk_key(&base);
        assert!(key_window.contains("|w|2"));

        let summary = RecordChunk {
            view: ChunkView::SummaryFallback {
                strategy: "summary".into(),
                weight: 0.8,
            },
            ..base
        };
        let key_summary = chunk_key(&summary);
        assert!(key_summary.contains("|s|summary"));
    }

    #[test]
    fn deterministic_rng_state_roundtrip_and_fill_bytes_are_stable() {
        let mut rng_a = DeterministicRng::new(123);
        let first = rng_a.next_u64();
        let saved = rng_a.state();

        let mut rng_b = DeterministicRng::from_state(saved);
        assert_eq!(rng_a.next_u64(), rng_b.next_u64());
        assert_ne!(first, 0);

        let mut bytes_a = [0u8; 13];
        let mut bytes_b = [0u8; 13];
        let mut rng_c = DeterministicRng::new(999);
        let mut rng_d = DeterministicRng::new(999);
        rng_c.fill_bytes(&mut bytes_a);
        rng_d.fill_bytes(&mut bytes_b);
        assert_eq!(bytes_a, bytes_b);
        assert!(bytes_a.iter().any(|b| *b != 0));

        let mut rng_e = DeterministicRng::new(999);
        let mut rng_f = DeterministicRng::new(999);
        assert_eq!(rng_e.next_u32() as u64, (rng_f.next_u64() as u32) as u64);
    }

    #[test]
    fn prefetcher_tracks_errors() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_ref = Arc::clone(&calls);
        let prefetcher = BatchPrefetcher::new(2, move || {
            let attempt = calls_ref.fetch_add(1, Ordering::Relaxed);
            if attempt == 0 {
                Err(SamplerError::SourceUnavailable {
                    source_id: PREFETCHER_SOURCE_ID.into(),
                    reason: "forced error".into(),
                })
            } else {
                Ok(TripletBatch {
                    triplets: Vec::new(),
                })
            }
        });

        let start = std::time::Instant::now();
        while prefetcher.produced_count() < 2 && start.elapsed() < StdDuration::from_millis(200) {
            std::thread::sleep(StdDuration::from_millis(5));
        }

        let _ = prefetcher.next();
        let _ = prefetcher.next();

        assert!(prefetcher.error_count() >= 1);
        assert!(prefetcher.produced_count() >= 2);
    }

    impl RecipeSource {
        fn new(records: Vec<DataRecord>, recipes: Vec<TripletRecipe>) -> Self {
            Self {
                inner: InMemorySource::new("recipe_source", records),
                triplet_recipes: recipes,
            }
        }
    }

    impl DataSource for RecipeSource {
        fn id(&self) -> &str {
            <InMemorySource as DataSource>::id(&self.inner)
        }

        fn refresh(
            &self,
            cursor: Option<&SourceCursor>,
            limit: Option<usize>,
        ) -> Result<SourceSnapshot, SamplerError> {
            <InMemorySource as DataSource>::refresh(&self.inner, cursor, limit)
        }

        fn reported_record_count(&self) -> Result<u128, SamplerError> {
            <InMemorySource as DataSource>::reported_record_count(&self.inner)
        }

        fn configure_sampler(&self, config: &SamplerConfig) {
            <InMemorySource as DataSource>::configure_sampler(&self.inner, config)
        }

        fn default_triplet_recipes(&self) -> Vec<TripletRecipe> {
            self.triplet_recipes.clone()
        }
    }

    #[derive(Clone)]
    /// Test source that counts refresh calls.
    struct CountingSource {
        id: SourceId,
        records: Vec<DataRecord>,
        refresh_calls: Arc<AtomicUsize>,
    }

    impl CountingSource {
        fn new(id: &str, records: Vec<DataRecord>, refresh_calls: Arc<AtomicUsize>) -> Self {
            Self {
                id: id.to_string(),
                records,
                refresh_calls,
            }
        }
    }

    impl DataSource for CountingSource {
        fn id(&self) -> &str {
            &self.id
        }

        fn refresh(
            &self,
            _cursor: Option<&SourceCursor>,
            limit: Option<usize>,
        ) -> Result<SourceSnapshot, SamplerError> {
            self.refresh_calls.fetch_add(1, Ordering::Relaxed);
            let mut records = self.records.clone();
            if let Some(max) = limit {
                records.truncate(max);
            }
            Ok(SourceSnapshot {
                records,
                cursor: SourceCursor {
                    last_seen: Utc::now(),
                    revision: 0,
                },
            })
        }

        fn reported_record_count(&self) -> Result<u128, SamplerError> {
            Ok(self.records.len() as u128)
        }

        fn configure_sampler(&self, _config: &SamplerConfig) {}
    }

    /// Test source that always returns a refresh error.
    struct FailingSource {
        id: SourceId,
    }

    impl FailingSource {
        fn new(id: &str) -> Self {
            Self { id: id.to_string() }
        }
    }

    #[derive(Clone)]
    /// Test source that fails once then succeeds.
    struct FlakySource {
        id: SourceId,
        records: Vec<DataRecord>,
        refresh_calls: Arc<AtomicUsize>,
    }

    impl FlakySource {
        fn new(id: &str, records: Vec<DataRecord>, refresh_calls: Arc<AtomicUsize>) -> Self {
            Self {
                id: id.to_string(),
                records,
                refresh_calls,
            }
        }
    }

    impl DataSource for FlakySource {
        fn id(&self) -> &str {
            &self.id
        }

        fn refresh(
            &self,
            _cursor: Option<&SourceCursor>,
            limit: Option<usize>,
        ) -> Result<SourceSnapshot, SamplerError> {
            let call = self.refresh_calls.fetch_add(1, Ordering::Relaxed);
            if call == 0 {
                return Err(SamplerError::SourceUnavailable {
                    source_id: self.id.clone(),
                    reason: "first refresh intentionally fails".into(),
                });
            }

            let mut records = self.records.clone();
            if let Some(max) = limit {
                records.truncate(max);
            }
            Ok(SourceSnapshot {
                records,
                cursor: SourceCursor {
                    last_seen: Utc::now(),
                    revision: call as u64,
                },
            })
        }

        fn reported_record_count(&self) -> Result<u128, SamplerError> {
            Ok(self.records.len() as u128)
        }

        fn configure_sampler(&self, _config: &SamplerConfig) {}
    }

    impl DataSource for FailingSource {
        fn id(&self) -> &str {
            &self.id
        }

        fn refresh(
            &self,
            _cursor: Option<&SourceCursor>,
            _limit: Option<usize>,
        ) -> Result<SourceSnapshot, SamplerError> {
            Err(SamplerError::SourceUnavailable {
                source_id: self.id.clone(),
                reason: "forced failure".into(),
            })
        }

        fn reported_record_count(&self) -> Result<u128, SamplerError> {
            Err(SamplerError::SourceUnavailable {
                source_id: self.id.clone(),
                reason: "forced failure".into(),
            })
        }

        fn configure_sampler(&self, _config: &SamplerConfig) {}
    }

    fn sample_record() -> DataRecord {
        let now = Utc::now();
        DataRecord {
            id: "record_1".into(),
            source: "unit".into(),
            created_at: now,
            updated_at: now,
            quality: QualityScore { trust: 0.9 },
            taxonomy: vec!["SampleCorp".into()],
            sections: vec![
                RecordSection {
                    role: SectionRole::Anchor,
                    heading: Some("Title".into()),
                    text: "Sample title".into(),
                    sentences: vec!["Sample title".into()],
                },
                RecordSection {
                    role: SectionRole::Context,
                    heading: Some("Intro".into()),
                    text: "This is the introduction paragraph with enough words for sampling."
                        .into(),
                    sentences: vec![
                        "This is the introduction paragraph with enough words for sampling.".into(),
                    ],
                },
            ],
            meta_prefix: None,
        }
    }

    fn record_with_offset(
        id: &str,
        base: chrono::DateTime<Utc>,
        offset_seconds: i64,
    ) -> DataRecord {
        let timestamp = base + Duration::seconds(offset_seconds);
        let mut record = sample_record();
        record.id = id.into();
        record.created_at = timestamp;
        record.updated_at = timestamp;
        record
    }

    fn trader_record(id: &str, date: &str, title: &str, body: &str) -> DataRecord {
        let now = Utc::now();
        DataRecord {
            id: id.into(),
            source: PRIMARY_SOURCE_ID.into(),
            created_at: now,
            updated_at: now,
            quality: QualityScore { trust: 0.9 },
            taxonomy: vec![PRIMARY_SOURCE_ID.into(), META_FIELD_DATE.encode(date)],
            sections: vec![
                RecordSection {
                    role: SectionRole::Anchor,
                    heading: Some("Title".into()),
                    text: title.into(),
                    sentences: vec![title.into()],
                },
                RecordSection {
                    role: SectionRole::Context,
                    heading: Some("Summary".into()),
                    text: body.into(),
                    sentences: vec![body.into()],
                },
            ],
            meta_prefix: None,
        }
    }

    fn extract_date_prefix(chunk_text: &str) -> Option<String> {
        let first_line = chunk_text.lines().next()?;
        let prefix = first_line.strip_prefix("meta: ")?;
        for part in prefix.split(" | ") {
            if let Some(date) = part.strip_prefix("date=") {
                return Some(date.to_string());
            }
        }
        None
    }

    fn extract_meta_prefix(chunk_text: &str) -> Option<String> {
        let first_line = chunk_text.lines().next()?;
        if first_line.starts_with("meta: ") {
            Some(first_line.to_string())
        } else {
            None
        }
    }

    fn split_meta_parts(meta_prefix: &str) -> Vec<String> {
        let body = meta_prefix.strip_prefix("meta: ").unwrap_or(meta_prefix);
        body.split(" | ").map(|part| part.to_string()).collect()
    }

    #[test]
    fn exhaustion_retry_limit_returns_exhausted() {
        let split = SplitRatios {
            train: 1.0,
            validation: 0.0,
            test: 0.0,
        };
        let store = Arc::new(DeterministicSplitStore::new(split, 101).unwrap());
        let mut config = base_config();
        config.seed = 202;
        config.batch_size = 1;
        config.ingestion_max_records = 2;
        config.allowed_splits = vec![SplitLabel::Train];
        config.split = split;
        config.recipes = vec![TripletRecipe {
            name: "exhaust_triplet".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
        }];
        config.text_recipes = Vec::new();

        let records = vec![sample_record()];
        let refresh_calls = Arc::new(AtomicUsize::new(0));
        let source = CountingSource::new("unit", records, Arc::clone(&refresh_calls));
        let sampler = PairSampler::new(config, store);
        sampler.register_source(Box::new(source));

        let result = sampler.next_triplet_batch(SplitLabel::Train);
        assert!(matches!(result, Err(SamplerError::Exhausted(_))));
        assert_eq!(
            refresh_calls.load(Ordering::Relaxed),
            EXHAUSTION_RETRY_LIMIT * 2 + 1
        );
    }

    #[test]
    fn single_source_failure_does_not_fail_batch_when_other_source_has_data() {
        let split = SplitRatios {
            train: 1.0,
            validation: 0.0,
            test: 0.0,
        };
        let store = Arc::new(DeterministicSplitStore::new(split, 909).unwrap());

        let mut config = base_config();
        config.seed = 1337;
        config.batch_size = 1;
        config.ingestion_max_records = 8;
        config.allowed_splits = vec![SplitLabel::Train];
        config.split = split;
        config.recipes = vec![TripletRecipe {
            name: "resilient_triplet".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
        }];
        config.text_recipes = Vec::new();

        let healthy_records = vec![
            trader_record("healthy_a", "2025-01-01", "A", "Body A"),
            trader_record("healthy_b", "2025-01-02", "B", "Body B"),
            trader_record("healthy_c", "2025-01-03", "C", "Body C"),
        ];

        let sampler = PairSampler::new(config, store);
        sampler.register_source(Box::new(FailingSource::new("failing_source")));
        sampler.register_source(Box::new(InMemorySource::new(
            "healthy_source",
            healthy_records,
        )));

        let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        assert_eq!(batch.triplets.len(), 1);
        assert!(batch.triplets[0].anchor.record_id.starts_with("healthy_"));
        assert!(batch.triplets[0].positive.record_id.starts_with("healthy_"));
        assert!(batch.triplets[0].negative.record_id.starts_with("healthy_"));
    }

    #[test]
    fn failed_source_is_retried_on_next_batch_call() {
        let split = SplitRatios {
            train: 1.0,
            validation: 0.0,
            test: 0.0,
        };
        let store = Arc::new(DeterministicSplitStore::new(split, 404).unwrap());

        let mut config = base_config();
        config.seed = 505;
        config.batch_size = 1;
        config.ingestion_max_records = 8;
        config.allowed_splits = vec![SplitLabel::Train];
        config.split = split;
        config.recipes = vec![TripletRecipe {
            name: "retry_triplet".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
        }];
        config.text_recipes = Vec::new();

        let flaky_calls = Arc::new(AtomicUsize::new(0));
        let flaky_records = vec![
            trader_record("flaky_a", "2025-02-01", "Flaky A", "Flaky body A"),
            trader_record("flaky_b", "2025-02-02", "Flaky B", "Flaky body B"),
        ];
        let healthy_records = vec![
            trader_record("steady_a", "2025-03-01", "Steady A", "Steady body A"),
            trader_record("steady_b", "2025-03-02", "Steady B", "Steady body B"),
            trader_record("steady_c", "2025-03-03", "Steady C", "Steady body C"),
        ];

        let sampler = PairSampler::new(config, store);
        sampler.register_source(Box::new(FlakySource::new(
            "flaky_source",
            flaky_records,
            Arc::clone(&flaky_calls),
        )));
        sampler.register_source(Box::new(InMemorySource::new(
            "steady_source",
            healthy_records,
        )));

        sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        assert_eq!(flaky_calls.load(Ordering::Relaxed), 1);

        sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        assert!(flaky_calls.load(Ordering::Relaxed) >= 2);
    }

    fn qa_pair_record(id: &str, question: &str, answer: &str) -> DataRecord {
        let now = Utc::now();
        DataRecord {
            id: id.into(),
            source: SECONDARY_SOURCE_ID.into(),
            created_at: now,
            updated_at: now,
            quality: QualityScore { trust: 0.9 },
            taxonomy: vec![SECONDARY_SOURCE_ID.into(), "factual".into()],
            sections: vec![
                RecordSection {
                    role: SectionRole::Anchor,
                    heading: Some("Question".into()),
                    text: question.into(),
                    sentences: vec![question.into()],
                },
                RecordSection {
                    role: SectionRole::Context,
                    heading: Some("Answer".into()),
                    text: answer.into(),
                    sentences: vec![answer.into()],
                },
            ],
            meta_prefix: None,
        }
    }

    /// Test source that returns custom records plus default recipes.
    struct RecipeDecoratedSource {
        records: Vec<DataRecord>,
        recipes: Vec<TripletRecipe>,
    }

    impl RecipeDecoratedSource {
        fn new(records: Vec<DataRecord>, recipes: Vec<TripletRecipe>) -> Self {
            Self { records, recipes }
        }
    }

    impl DataSource for RecipeDecoratedSource {
        fn id(&self) -> &str {
            "recipe_decorated_source"
        }

        fn refresh(
            &self,
            cursor: Option<&SourceCursor>,
            limit: Option<usize>,
        ) -> Result<SourceSnapshot, crate::errors::SamplerError> {
            let mut records = self.records.clone();
            if let Some(cap) = limit {
                records.truncate(cap);
            }
            Ok(SourceSnapshot {
                records,
                cursor: SourceCursor {
                    last_seen: Utc::now(),
                    revision: cursor.map(|c| c.revision + 1).unwrap_or_default(),
                },
            })
        }

        fn reported_record_count(&self) -> Result<u128, crate::errors::SamplerError> {
            Ok(self.records.len() as u128)
        }

        fn configure_sampler(&self, _config: &SamplerConfig) {}

        fn default_triplet_recipes(&self) -> Vec<TripletRecipe> {
            self.recipes.clone()
        }
    }

    #[test]
    fn chunk_view_carries_start_ratio() {
        let split = SplitRatios::default();
        let mut config = base_config();
        config.chunking = ChunkingStrategy {
            max_window_tokens: 4,
            overlap_tokens: vec![0],
            summary_fallback_weight: 0.0,
            summary_fallback_tokens: 2,
            chunk_weight_floor: 0.0,
        };
        let store = Arc::new(DeterministicSplitStore::new(split, 3).unwrap());
        let sampler = PairSampler::new(config, store);

        let section_text = "one two three four five six seven eight nine ten";
        let record = DataRecord {
            id: "ratio_record".into(),
            source: "unit".into(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            quality: QualityScore { trust: 1.0 },
            taxonomy: vec![],
            sections: vec![RecordSection {
                role: SectionRole::Context,
                heading: None,
                text: section_text.into(),
                sentences: vec![section_text.into()],
            }],
            meta_prefix: None,
        };

        let section = &record.sections[0];
        let chunks = sampler
            .inner
            .lock()
            .unwrap()
            .materialize_chunks(&record, 0, section);
        let ratios: Vec<f32> = chunks
            .iter()
            .filter_map(|chunk| match chunk.view {
                ChunkView::Window { start_ratio, .. } => Some(start_ratio),
                _ => None,
            })
            .collect();
        assert!(ratios.len() >= 3);
        assert!((ratios[0] - 0.0).abs() < 1e-6);
        assert!((ratios[1] - 0.4).abs() < 1e-6);
        assert!((ratios[2] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn chunk_windows_follow_stride_for_large_sections() {
        let split = SplitRatios::default();
        let mut config = base_config();
        config.chunking = ChunkingStrategy {
            max_window_tokens: 5,
            overlap_tokens: vec![1],
            summary_fallback_weight: 0.0,
            summary_fallback_tokens: 0,
            chunk_weight_floor: 0.0,
        };
        let store = Arc::new(DeterministicSplitStore::new(split, 11).unwrap());
        let sampler = PairSampler::new(config, store);

        let block = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu";
        let record = DataRecord {
            id: "stride_record".into(),
            source: "unit".into(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            quality: QualityScore { trust: 1.0 },
            taxonomy: vec![],
            sections: vec![RecordSection {
                role: SectionRole::Context,
                heading: None,
                text: block.into(),
                sentences: vec![
                    "alpha beta gamma delta.".into(),
                    "epsilon zeta eta theta.".into(),
                    "iota kappa lambda mu.".into(),
                ],
            }],
            meta_prefix: None,
        };

        let section = &record.sections[0];
        let chunks = sampler
            .inner
            .lock()
            .unwrap()
            .materialize_chunks(&record, 0, section);

        let texts: Vec<String> = chunks
            .iter()
            .filter_map(|chunk| match chunk.view {
                ChunkView::Window { .. } => Some(chunk.text.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            texts,
            vec![
                "alpha beta gamma delta epsilon".to_string(),
                "epsilon zeta eta theta iota".to_string(),
                "iota kappa lambda mu".to_string(),
            ]
        );

        let estimates: Vec<usize> = chunks
            .iter()
            .filter_map(|chunk| match chunk.view {
                ChunkView::Window { .. } => Some(chunk.tokens_estimate),
                _ => None,
            })
            .collect();
        assert_eq!(estimates, vec![5, 5, 4]);
    }

    #[test]
    fn chunk_weight_applies_linear_offset_and_floor() {
        let split = SplitRatios::default();
        let mut config = base_config();
        config.chunking.chunk_weight_floor = 0.25;
        let store = Arc::new(DeterministicSplitStore::new(split, 5).unwrap());
        let sampler = PairSampler::new(config, store);

        let base_chunk = RecordChunk {
            record_id: "unit".into(),
            section_idx: 0,
            view: ChunkView::Window {
                index: 0,
                overlap: 0,
                span: 10,
                start_ratio: 0.75,
            },
            text: "dummy".into(),
            tokens_estimate: 10,
            quality: QualityScore::default(),
        };
        assert_eq!(
            sampler.inner.lock().unwrap().chunk_weight(&base_chunk),
            0.25
        );

        let mut early_chunk = base_chunk.clone();
        early_chunk.view = ChunkView::Window {
            index: 0,
            overlap: 0,
            span: 10,
            start_ratio: 0.1,
        };
        assert_eq!(
            sampler.inner.lock().unwrap().chunk_weight(&early_chunk),
            0.9
        );
    }

    #[test]
    fn summary_fallback_weight_is_clamped() {
        let split = SplitRatios::default();
        let mut config = base_config();
        config.chunking.chunk_weight_floor = 0.5;
        let store = Arc::new(DeterministicSplitStore::new(split, 6).unwrap());
        let sampler = PairSampler::new(config, store);

        let summary_chunk = RecordChunk {
            record_id: "unit".into(),
            section_idx: 0,
            view: ChunkView::SummaryFallback {
                strategy: "head".into(),
                weight: 0.4,
            },
            text: "summary".into(),
            tokens_estimate: 10,
            quality: QualityScore::default(),
        };
        assert_eq!(
            sampler.inner.lock().unwrap().chunk_weight(&summary_chunk),
            0.5
        );
    }

    #[test]
    fn chunk_weight_applies_trust_scaling() {
        let split = SplitRatios::default();
        let mut config = base_config();
        config.chunking.chunk_weight_floor = 0.0;
        let store = Arc::new(DeterministicSplitStore::new(split, 10).unwrap());
        let sampler = PairSampler::new(config, store);

        let trusted_chunk = RecordChunk {
            record_id: "unit".into(),
            section_idx: 0,
            view: ChunkView::Window {
                index: 0,
                overlap: 0,
                span: 10,
                start_ratio: 0.2,
            },
            text: "dummy".into(),
            tokens_estimate: 10,
            quality: QualityScore { trust: 0.5 },
        };

        let weight = sampler.inner.lock().unwrap().chunk_weight(&trusted_chunk);
        assert!((weight - 0.4).abs() < f32::EPSILON);
    }

    #[test]
    fn triplet_weight_averages_chunk_weights() {
        let split = SplitRatios::default();
        let mut config = base_config();
        config.chunking.chunk_weight_floor = 0.0;
        let store = Arc::new(DeterministicSplitStore::new(split, 7).unwrap());
        let sampler = PairSampler::new(config, store);

        let anchor = RecordChunk {
            record_id: "a".into(),
            section_idx: 0,
            view: ChunkView::Window {
                index: 0,
                overlap: 0,
                span: 10,
                start_ratio: 0.0,
            },
            text: "a".into(),
            tokens_estimate: 10,
            quality: QualityScore::default(),
        };
        let positive = RecordChunk {
            record_id: "b".into(),
            section_idx: 0,
            view: ChunkView::Window {
                index: 0,
                overlap: 0,
                span: 10,
                start_ratio: 0.5,
            },
            text: "b".into(),
            tokens_estimate: 10,
            quality: QualityScore::default(),
        };
        let negative = RecordChunk {
            record_id: "c".into(),
            section_idx: 0,
            view: ChunkView::Window {
                index: 0,
                overlap: 0,
                span: 10,
                start_ratio: 1.0,
            },
            text: "c".into(),
            tokens_estimate: 10,
            quality: QualityScore::default(),
        };

        let avg = sampler
            .inner
            .lock()
            .unwrap()
            .triplet_chunk_weight(&anchor, &positive, &negative);
        assert!((avg - (1.0 + 0.5 + 0.0) / 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn end_to_end_text_weighting_uses_chunk_offsets() {
        let split = SplitRatios {
            train: 1.0,
            validation: 0.0,
            test: 0.0,
        };
        let mut config = base_config();
        config.seed = 9;
        config.batch_size = 1;
        config.allowed_splits = vec![SplitLabel::Train];
        config.text_recipes = vec![TextRecipe {
            name: "weighted".into(),
            selector: Selector::Role(SectionRole::Context),
            weight: 2.0,
            instruction: None,
        }];
        config.chunking = ChunkingStrategy {
            max_window_tokens: 2,
            overlap_tokens: vec![0],
            summary_fallback_weight: 0.0,
            summary_fallback_tokens: 0,
            chunk_weight_floor: 0.0,
        };

        let store = Arc::new(DeterministicSplitStore::new(split, 9).unwrap());
        let sampler = PairSampler::new(config, store);
        let record = DataRecord {
            id: "weighted_record".into(),
            source: "unit".into(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            quality: QualityScore::default(),
            taxonomy: vec![],
            sections: vec![RecordSection {
                role: SectionRole::Context,
                heading: None,
                text: "one two three four".into(),
                sentences: vec!["one two three four".into()],
            }],
            meta_prefix: None,
        };
        sampler.register_source(Box::new(InMemorySource::new("unit", vec![record])));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();

        let first = sampler.next_text_batch(SplitLabel::Train).unwrap();
        let second = sampler.next_text_batch(SplitLabel::Train).unwrap();

        assert!((first.samples[0].weight - 2.0).abs() < f32::EPSILON);
        assert!((second.samples[0].weight - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn end_to_end_text_weighting_respects_splits() {
        let split = SplitRatios {
            train: 0.34,
            validation: 0.33,
            test: 0.33,
        };
        let store = Arc::new(DeterministicSplitStore::new(split, 21).unwrap());

        let find_id = |label: SplitLabel, prefix: &str| -> String {
            for i in 0..2000 {
                let id = format!("{prefix}_{i}");
                if store.ensure(id.clone()).unwrap() == label {
                    return id;
                }
            }
            panic!("unable to find id for {:?}", label);
        };

        let train_id = find_id(SplitLabel::Train, "train_weighted");
        let val_id = find_id(SplitLabel::Validation, "val_weighted");
        let test_id = find_id(SplitLabel::Test, "test_weighted");

        let mut config = base_config();
        config.seed = 21;
        config.batch_size = 1;
        config.allowed_splits = vec![SplitLabel::Train];
        config.text_recipes = vec![TextRecipe {
            name: "weighted".into(),
            selector: Selector::Role(SectionRole::Context),
            weight: 3.0,
            instruction: None,
        }];
        config.chunking = ChunkingStrategy {
            max_window_tokens: 2,
            overlap_tokens: vec![0],
            summary_fallback_weight: 0.0,
            summary_fallback_tokens: 0,
            chunk_weight_floor: 0.0,
        };
        let chunking = config.chunking.clone();

        let sampler = PairSampler::new(config, store);
        let mut train_record =
            trader_record(&train_id, "2025-01-01", "Train Title", "one two three four");
        let mut val_record =
            trader_record(&val_id, "2025-01-02", "Val Title", "alpha beta gamma delta");
        let mut test_record =
            trader_record(&test_id, "2025-01-03", "Test Title", "foo bar baz qux");
        train_record.source = "split_weighted".into();
        val_record.source = "split_weighted".into();
        test_record.source = "split_weighted".into();

        sampler.register_source(Box::new(InMemorySource::new(
            "split_weighted",
            vec![train_record, val_record, test_record],
        )));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();

        let mut labels = std::collections::HashSet::new();
        let mut checked = 0;
        for _ in 0..20 {
            let batch = sampler.next_text_batch(SplitLabel::Train).unwrap();
            let sample = &batch.samples[0];
            let label = sampler
                .inner
                .lock()
                .unwrap()
                .split_store
                .label_for(&sample.chunk.record_id)
                .unwrap();
            labels.insert(label);
            assert_eq!(label, SplitLabel::Train, "text sample leaked across splits");
            let expected = 3.0 * chunk_weight(&chunking, &sample.chunk);
            assert!((sample.weight - expected).abs() < f32::EPSILON);
            checked += 1;
            if labels.len() == 1 {
                break;
            }
        }
        assert_eq!(labels.len(), 1, "all samples must stay in target split");
        assert!(checked > 0);
    }

    /// Helper bundle for split-order determinism tests.
    struct SplitOrderFixture {
        sampler: Arc<PairSampler<DeterministicSplitStore>>,
    }

    fn build_split_order_sampler(seed: u64, batch_size: usize) -> SplitOrderFixture {
        let split = SplitRatios {
            train: 0.34,
            validation: 0.33,
            test: 0.33,
        };
        let store = Arc::new(DeterministicSplitStore::new(split, seed).unwrap());

        let mut config = base_config();
        config.seed = seed;
        config.batch_size = batch_size;
        config.ingestion_max_records = 16;
        config.allowed_splits = vec![SplitLabel::Train];
        config.text_recipes = vec![TextRecipe {
            name: "split_text".into(),
            selector: Selector::Role(SectionRole::Context),
            weight: 1.0,
            instruction: None,
        }];
        config.recipes = vec![TripletRecipe {
            name: "split_triplet".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
        }];

        let sampler = Arc::new(PairSampler::new(config, Arc::clone(&store)));

        let make_records = |source: &str| {
            let mut records = Vec::new();
            for idx in 0..15 {
                let record_id = format!("{source}::record_{idx:02}");
                let title = format!("{source} title {idx}");
                let body = format!("{source} body {idx}");
                records.push(trader_record(&record_id, "2025-01-01", &title, &body));
            }
            records
        };

        sampler.register_source(Box::new(InMemorySource::new(
            "source_a",
            make_records("source_a"),
        )));
        sampler.register_source(Box::new(InMemorySource::new(
            "source_b",
            make_records("source_b"),
        )));
        sampler.register_source(Box::new(InMemorySource::new(
            "source_c",
            make_records("source_c"),
        )));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();

        SplitOrderFixture { sampler }
    }

    fn fnv1a_64(input: &str) -> u64 {
        let mut hash = FNV1A64_OFFSET;
        for byte in input.as_bytes() {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(FNV1A64_PRIME);
        }
        hash
    }

    fn fmt_weight(weight: f32) -> String {
        format!("{:.6}", weight)
    }

    fn text_snapshot_hash(batches: &[TextBatch]) -> u64 {
        let parts: Vec<crate::types::HashPart> = batches
            .iter()
            .map(|batch| {
                let sample = &batch.samples[0];
                format!(
                    "text|{}|{}|{}|{}",
                    sample.recipe,
                    sample.chunk.record_id,
                    chunk_key(&sample.chunk),
                    fmt_weight(sample.weight)
                )
            })
            .collect();
        fnv1a_64(&parts.join(";"))
    }

    fn triplet_snapshot_hash(batches: &[TripletBatch]) -> u64 {
        let parts: Vec<crate::types::HashPart> = batches
            .iter()
            .map(|batch| {
                let triplet = &batch.triplets[0];
                format!(
                    "triplet|{}|{}|{}|{}|{}|{}|{}|{}",
                    triplet.recipe,
                    triplet.anchor.record_id,
                    chunk_key(&triplet.anchor),
                    triplet.positive.record_id,
                    chunk_key(&triplet.positive),
                    triplet.negative.record_id,
                    chunk_key(&triplet.negative),
                    fmt_weight(triplet.weight)
                )
            })
            .collect();
        fnv1a_64(&parts.join(";"))
    }

    fn label_str(label: &PairLabel) -> &'static str {
        match label {
            PairLabel::Positive => "positive",
            PairLabel::Negative => "negative",
        }
    }

    fn pair_snapshot_hash(batches: &[SampleBatch]) -> u64 {
        let mut parts = Vec::new();
        for batch in batches {
            for pair in &batch.pairs {
                let reason = pair.reason.as_deref().unwrap_or("");
                parts.push(format!(
                    "pair|{}|{}|{}|{}|{}|{}|{}|{}",
                    pair.recipe,
                    label_str(&pair.label),
                    pair.anchor.record_id,
                    chunk_key(&pair.anchor),
                    pair.positive.record_id,
                    chunk_key(&pair.positive),
                    fmt_weight(pair.weight),
                    reason
                ));
            }
        }
        fnv1a_64(&parts.join(";"))
    }

    #[test]
    fn split_order_is_train_val_test_for_text_batches() {
        let fixture = build_split_order_sampler(31, 1);
        let mut record_ids = Vec::new();
        for _ in 0..9 {
            let batch = fixture.sampler.next_text_batch(SplitLabel::Train).unwrap();
            record_ids.push(batch.samples[0].chunk.record_id.clone());
        }
        assert_eq!(
            record_ids,
            vec![
                "source_b::record_03".to_string(),
                "source_c::record_02".to_string(),
                "source_c::record_03".to_string(),
                "source_b::record_03".to_string(),
                "source_b::record_03".to_string(),
                "source_c::record_03".to_string(),
                "source_c::record_02".to_string(),
                "source_b::record_03".to_string(),
                "source_c::record_03".to_string()
            ]
        );
    }

    #[test]
    fn split_order_is_train_val_test_for_triplet_batches() {
        let fixture = build_split_order_sampler(32, 1);
        let mut record_ids = Vec::new();
        for _ in 0..9 {
            let batch = fixture
                .sampler
                .next_triplet_batch(SplitLabel::Train)
                .unwrap();
            record_ids.push(batch.triplets[0].anchor.record_id.clone());
        }
        assert_eq!(
            record_ids,
            vec![
                "source_c::record_02".to_string(),
                "source_b::record_04".to_string(),
                "source_a::record_07".to_string(),
                "source_b::record_04".to_string(),
                "source_c::record_04".to_string(),
                "source_a::record_02".to_string(),
                "source_b::record_04".to_string(),
                "source_c::record_02".to_string(),
                "source_a::record_04".to_string()
            ]
        );
    }

    #[test]
    fn split_order_is_train_val_test_for_pair_batches() {
        let fixture = build_split_order_sampler(33, 2);
        let mut record_ids = Vec::new();
        for _ in 0..9 {
            let batch = fixture.sampler.next_pair_batch(SplitLabel::Train).unwrap();
            record_ids.push(batch.pairs[0].anchor.record_id.clone());
        }
        assert_eq!(
            record_ids,
            vec![
                "source_b::record_04".to_string(),
                "source_c::record_02".to_string(),
                "source_a::record_05".to_string(),
                "source_c::record_04".to_string(),
                "source_b::record_04".to_string(),
                "source_a::record_07".to_string(),
                "source_b::record_08".to_string(),
                "source_a::record_06".to_string(),
                "source_b::record_08".to_string()
            ]
        );
    }

    #[test]
    fn prefetch_text_batches_preserve_split_order() {
        let fixture = build_split_order_sampler(41, 1);
        let prefetcher = Arc::clone(&fixture.sampler).prefetch_text_batches(SplitLabel::Train, 1);
        let mut record_ids = Vec::new();
        for _ in 0..9 {
            let batch = prefetcher.next().unwrap();
            record_ids.push(batch.samples[0].chunk.record_id.clone());
        }
        drop(prefetcher);
        assert_eq!(
            record_ids,
            vec![
                "source_c::record_03".to_string(),
                "source_a::record_04".to_string(),
                "source_a::record_07".to_string(),
                "source_c::record_02".to_string(),
                "source_c::record_04".to_string(),
                "source_a::record_08".to_string(),
                "source_a::record_07".to_string(),
                "source_c::record_02".to_string(),
                "source_c::record_03".to_string()
            ]
        );
    }

    #[test]
    fn prefetch_triplet_batches_preserve_split_order() {
        let fixture = build_split_order_sampler(42, 1);
        let prefetcher =
            Arc::clone(&fixture.sampler).prefetch_triplet_batches(SplitLabel::Train, 1);
        let mut record_ids = Vec::new();
        for _ in 0..9 {
            let batch = prefetcher.next().unwrap();
            record_ids.push(batch.triplets[0].anchor.record_id.clone());
        }
        drop(prefetcher);
        assert_eq!(
            record_ids,
            vec![
                "source_b::record_01".to_string(),
                "source_a::record_04".to_string(),
                "source_c::record_02".to_string(),
                "source_c::record_03".to_string(),
                "source_a::record_04".to_string(),
                "source_a::record_11".to_string(),
                "source_c::record_02".to_string(),
                "source_c::record_03".to_string(),
                "source_a::record_04".to_string()
            ]
        );
    }

    #[test]
    fn prefetch_pair_batches_preserve_split_order() {
        let fixture = build_split_order_sampler(43, 2);
        let prefetcher = Arc::clone(&fixture.sampler).prefetch_pair_batches(SplitLabel::Train, 1);
        let mut record_ids = Vec::new();
        for _ in 0..9 {
            let batch = prefetcher.next().unwrap();
            record_ids.push(batch.pairs[0].anchor.record_id.clone());
        }
        drop(prefetcher);
        assert_eq!(
            record_ids,
            vec![
                "source_c::record_02".to_string(),
                "source_b::record_06".to_string(),
                "source_a::record_02".to_string(),
                "source_b::record_06".to_string(),
                "source_b::record_06".to_string(),
                "source_a::record_08".to_string(),
                "source_a::record_06".to_string(),
                "source_a::record_08".to_string(),
                "source_b::record_13".to_string()
            ]
        );
    }

    #[test]
    fn prefetch_triplet_batches_with_weights_match_direct() {
        let fixture_prefetch = build_split_order_sampler(101, 1);
        let fixture_direct = build_split_order_sampler(101, 1);
        let mut weights = HashMap::new();
        weights.insert("source_a".to_string(), 1.0);
        weights.insert("source_b".to_string(), 2.0);
        weights.insert("source_c".to_string(), 0.5);

        let prefetcher = Arc::clone(&fixture_prefetch.sampler)
            .prefetch_triplet_batches_with_weights(SplitLabel::Train, 1, weights.clone());
        let mut prefetch_ids = Vec::new();
        for _ in 0..5 {
            let batch = prefetcher.next().unwrap();
            prefetch_ids.push(batch.triplets[0].anchor.record_id.clone());
        }
        drop(prefetcher);

        let mut direct_ids = Vec::new();
        for _ in 0..5 {
            let batch = fixture_direct
                .sampler
                .next_triplet_batch_with_weights(SplitLabel::Train, &weights)
                .unwrap();
            direct_ids.push(batch.triplets[0].anchor.record_id.clone());
        }

        assert_eq!(prefetch_ids, direct_ids);
    }

    #[test]
    fn prefetch_pair_batches_with_weights_match_direct() {
        let fixture_prefetch = build_split_order_sampler(102, 2);
        let fixture_direct = build_split_order_sampler(102, 2);
        let mut weights = HashMap::new();
        weights.insert("source_a".to_string(), 1.0);
        weights.insert("source_b".to_string(), 2.0);
        weights.insert("source_c".to_string(), 0.5);

        let prefetcher = Arc::clone(&fixture_prefetch.sampler).prefetch_pair_batches_with_weights(
            SplitLabel::Train,
            1,
            weights.clone(),
        );
        let mut prefetch_ids = Vec::new();
        for _ in 0..5 {
            let batch = prefetcher.next().unwrap();
            prefetch_ids.push(batch.pairs[0].anchor.record_id.clone());
        }
        drop(prefetcher);

        let mut direct_ids = Vec::new();
        for _ in 0..5 {
            let batch = fixture_direct
                .sampler
                .next_pair_batch_with_weights(SplitLabel::Train, &weights)
                .unwrap();
            direct_ids.push(batch.pairs[0].anchor.record_id.clone());
        }

        assert_eq!(prefetch_ids, direct_ids);
    }

    #[test]
    fn prefetch_text_batches_with_weights_match_direct() {
        let fixture_prefetch = build_split_order_sampler(103, 1);
        let fixture_direct = build_split_order_sampler(103, 1);
        let mut weights = HashMap::new();
        weights.insert("source_a".to_string(), 1.0);
        weights.insert("source_b".to_string(), 2.0);
        weights.insert("source_c".to_string(), 0.5);

        let prefetcher = Arc::clone(&fixture_prefetch.sampler).prefetch_text_batches_with_weights(
            SplitLabel::Train,
            1,
            weights.clone(),
        );
        let mut prefetch_ids = Vec::new();
        for _ in 0..5 {
            let batch = prefetcher.next().unwrap();
            prefetch_ids.push(batch.samples[0].chunk.record_id.clone());
        }
        drop(prefetcher);

        let mut direct_ids = Vec::new();
        for _ in 0..5 {
            let batch = fixture_direct
                .sampler
                .next_text_batch_with_weights(SplitLabel::Train, &weights)
                .unwrap();
            direct_ids.push(batch.samples[0].chunk.record_id.clone());
        }

        assert_eq!(prefetch_ids, direct_ids);
    }

    #[test]
    fn split_order_differs_with_seed() {
        let a = build_split_order_sampler(71, 1);
        let b = build_split_order_sampler(72, 1);
        let mut a_batches = Vec::new();
        let mut b_batches = Vec::new();
        for _ in 0..3 {
            a_batches.push(a.sampler.next_text_batch(SplitLabel::Train).unwrap());
            b_batches.push(b.sampler.next_text_batch(SplitLabel::Train).unwrap());
        }
        let a_hash = text_snapshot_hash(&a_batches);
        let b_hash = text_snapshot_hash(&b_batches);
        assert_ne!(a_hash, b_hash);
    }

    #[test]
    fn full_sequence_hashes_match_for_text_batches() {
        let fixture = build_split_order_sampler(81, 1);
        let mut record_ids = Vec::new();
        let mut batches = Vec::new();
        for _ in 0..FULL_SEQUENCE_LEN {
            batches.push(fixture.sampler.next_text_batch(SplitLabel::Train).unwrap());
            let sample = &batches.last().unwrap().samples[0];
            record_ids.push(sample.chunk.record_id.clone());
        }
        assert_eq!(text_snapshot_hash(&batches), TEXT_BATCH_SEQUENCE_HASH);
    }

    #[test]
    fn full_sequence_hashes_match_for_triplet_batches() {
        let fixture = build_split_order_sampler(82, 1);
        let mut batches = Vec::new();
        for _ in 0..FULL_SEQUENCE_LEN {
            batches.push(
                fixture
                    .sampler
                    .next_triplet_batch(SplitLabel::Train)
                    .unwrap(),
            );
        }
        assert_eq!(triplet_snapshot_hash(&batches), TRIPLET_BATCH_SEQUENCE_HASH);
    }

    #[test]
    fn full_sequence_hashes_match_for_pair_batches() {
        let fixture = build_split_order_sampler(83, 2);
        let mut batches = Vec::new();
        for _ in 0..FULL_SEQUENCE_LEN {
            batches.push(fixture.sampler.next_pair_batch(SplitLabel::Train).unwrap());
        }
        assert_eq!(pair_snapshot_hash(&batches), PAIR_BATCH_SEQUENCE_HASH);
    }

    #[test]
    fn full_sequence_hashes_match_for_prefetch_text_batches() {
        let fixture = build_split_order_sampler(91, 1);
        let prefetcher = Arc::clone(&fixture.sampler).prefetch_text_batches(SplitLabel::Train, 1);
        let mut batches = Vec::new();
        for _ in 0..FULL_SEQUENCE_LEN {
            batches.push(prefetcher.next().unwrap());
        }
        drop(prefetcher);
        assert_eq!(
            text_snapshot_hash(&batches),
            PREFETCH_TEXT_BATCH_SEQUENCE_HASH
        );
    }

    #[test]
    fn full_sequence_hashes_match_for_prefetch_triplet_batches() {
        let fixture = build_split_order_sampler(92, 1);
        let prefetcher =
            Arc::clone(&fixture.sampler).prefetch_triplet_batches(SplitLabel::Train, 1);
        let mut batches = Vec::new();
        for _ in 0..FULL_SEQUENCE_LEN {
            batches.push(prefetcher.next().unwrap());
        }
        drop(prefetcher);
        assert_eq!(
            triplet_snapshot_hash(&batches),
            PREFETCH_TRIPLET_BATCH_SEQUENCE_HASH
        );
    }

    #[test]
    fn full_sequence_hashes_match_for_prefetch_pair_batches() {
        let fixture = build_split_order_sampler(93, 2);
        let prefetcher = Arc::clone(&fixture.sampler).prefetch_pair_batches(SplitLabel::Train, 1);
        let mut batches = Vec::new();
        for _ in 0..FULL_SEQUENCE_LEN {
            batches.push(prefetcher.next().unwrap());
        }
        drop(prefetcher);
        assert_eq!(
            pair_snapshot_hash(&batches),
            PREFETCH_PAIR_BATCH_SEQUENCE_HASH
        );
    }

    #[test]
    fn generates_pairs_from_single_source() {
        let split = SplitRatios::default();
        let config = SamplerConfig {
            seed: 1,
            batch_size: 4,
            chunking: ChunkingStrategy::default(),
            recipes: vec![TripletRecipe {
                name: "title_summary_triplet".into(),
                anchor: Selector::Role(SectionRole::Anchor),
                positive_selector: Selector::Role(SectionRole::Context),
                negative_selector: Selector::Role(SectionRole::Context),
                negative_strategy: NegativeStrategy::WrongArticle,
                weight: 1.0,
                instruction: None,
            }],
            text_recipes: vec![TextRecipe {
                name: "teacher_chunk".into(),
                selector: Selector::Role(SectionRole::Context),
                weight: 1.0,
                instruction: None,
            }],
            split,
            ..SamplerConfig::default()
        };
        let store = Arc::new(DeterministicSplitStore::new(split, 7).unwrap());
        let records = vec![
            trader_record(
                "source_a::2025/01-01/article_a.txt",
                "2025-01-01",
                "Alpha",
                "Body alpha",
            ),
            trader_record(
                "source_a::2025/01-02/article_b.txt",
                "2025-01-02",
                "Beta",
                "Body beta",
            ),
        ];
        let sampler = PairSampler::new(config, store);
        sampler.register_source(Box::new(InMemorySource::new("unit", records)));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();
        let batch = sampler.next_pair_batch(SplitLabel::Train).unwrap();
        assert!(!batch.is_empty());
        assert_eq!(batch.pairs.len(), 4);
    }

    #[test]
    fn produces_text_samples() {
        let split = SplitRatios::default();
        let config = SamplerConfig {
            seed: 2,
            batch_size: 1,
            chunking: ChunkingStrategy::default(),
            recipes: vec![],
            text_recipes: vec![TextRecipe {
                name: "teacher_chunk".into(),
                selector: Selector::Role(SectionRole::Context),
                weight: 1.0,
                instruction: None,
            }],
            split,
            ..SamplerConfig::default()
        };
        let store = Arc::new(DeterministicSplitStore::new(split, 11).unwrap());
        let sampler = PairSampler::new(config, store);
        sampler.register_source(Box::new(InMemorySource::new("unit", vec![sample_record()])));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();
        let batch = sampler.next_text_batch(SplitLabel::Train).unwrap();
        assert!(!batch.is_empty());
        assert_eq!(batch.samples.len(), 1);
    }

    #[test]
    fn cycles_through_section_windows_before_repeating() {
        let split = SplitRatios::default();
        let mut config = base_config();
        config.seed = 5;
        config.batch_size = 1;
        config.chunking = ChunkingStrategy {
            max_window_tokens: 2,
            overlap_tokens: vec![0],
            summary_fallback_weight: 0.0,
            summary_fallback_tokens: 0,
            chunk_weight_floor: 0.0,
        };
        config.text_recipes = vec![TextRecipe {
            name: "evidence_chunks".into(),
            selector: Selector::Role(SectionRole::Context),
            weight: 1.0,
            instruction: None,
        }];
        let store = Arc::new(DeterministicSplitStore::new(split, 13).unwrap());
        let record = DataRecord {
            id: "window_record".into(),
            source: "unit".into(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            quality: QualityScore::default(),
            taxonomy: vec![],
            sections: vec![RecordSection {
                role: SectionRole::Context,
                heading: None,
                text: "one two three four".into(),
                sentences: vec!["one two three four".into()],
            }],
            meta_prefix: None,
        };
        let sampler = PairSampler::new(config, store);
        sampler.register_source(Box::new(InMemorySource::new("unit", vec![record])));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();

        let mut outputs = Vec::new();
        for _ in 0..3 {
            let batch = sampler.next_text_batch(SplitLabel::Train).unwrap();
            outputs.push(batch.samples[0].chunk.text.clone());
        }

        assert_eq!(outputs[0], "one two");
        assert_eq!(outputs[1], "three four");
        assert_eq!(outputs[2], "one two");
    }

    #[test]
    fn first_chunk_offset_is_deterministic_and_nonzero_when_hash_demands_it() {
        let split = SplitRatios::default();
        let key = "window_record::0";
        let pool_len = 3usize;
        // In single-source mode, the first anchor selection wraps immediately and
        // advances source_epoch to 1 before chunk selection runs.
        let epoch_seed_mask = 1u64;
        let mut seed = 1u64;
        while (stable_hash_str(seed ^ epoch_seed_mask, key) as usize).is_multiple_of(pool_len) {
            seed = seed.saturating_add(1);
        }

        let build_sampler = || {
            let mut config = base_config();
            config.seed = seed;
            config.batch_size = 1;
            config.chunking = ChunkingStrategy {
                max_window_tokens: 2,
                overlap_tokens: vec![0],
                summary_fallback_weight: 0.0,
                summary_fallback_tokens: 0,
                chunk_weight_floor: 0.0,
            };
            config.text_recipes = vec![TextRecipe {
                name: "context_chunks".into(),
                selector: Selector::Role(SectionRole::Context),
                weight: 1.0,
                instruction: None,
            }];

            let store = Arc::new(DeterministicSplitStore::new(split, 13).unwrap());
            let record = DataRecord {
                id: "window_record".into(),
                source: "unit".into(),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                quality: QualityScore::default(),
                taxonomy: vec![],
                sections: vec![RecordSection {
                    role: SectionRole::Context,
                    heading: None,
                    text: "one two three four five six".into(),
                    sentences: vec!["one two three four five six".into()],
                }],
                meta_prefix: None,
            };

            let sampler = PairSampler::new(config, store);
            sampler.register_source(Box::new(InMemorySource::new("unit", vec![record])));
            sampler
                .inner
                .lock()
                .unwrap()
                .ingest_internal(SplitLabel::Train)
                .unwrap();
            sampler
        };

        let expected_start = (stable_hash_str(seed ^ epoch_seed_mask, key) as usize) % pool_len;
        assert_ne!(expected_start, 0);
        let expected = ["one two", "three four", "five six"][expected_start];

        let sampler_a = build_sampler();
        let first_a = sampler_a
            .next_text_batch(SplitLabel::Train)
            .unwrap()
            .samples[0]
            .chunk
            .text
            .clone();
        assert_eq!(first_a, expected);

        let sampler_b = build_sampler();
        let first_b = sampler_b
            .next_text_batch(SplitLabel::Train)
            .unwrap()
            .samples[0]
            .chunk
            .text
            .clone();
        assert_eq!(first_b, expected);
        assert_eq!(first_a, first_b);
    }

    #[test]
    fn first_role_section_offset_is_deterministic_and_nonzero_when_hash_demands_it() {
        let split = SplitRatios::default();
        let key = "role_offset_record::context";
        let section_count = 3usize;
        // In single-source mode, the first anchor selection wraps immediately and
        // advances source_epoch to 1 before role section selection runs.
        let epoch_seed_mask = 1u64;
        let mut seed = 1u64;
        while (stable_hash_str(seed ^ epoch_seed_mask, key) as usize).is_multiple_of(section_count)
        {
            seed = seed.saturating_add(1);
        }

        let build_sampler = || {
            let mut config = base_config();
            config.seed = seed;
            config.batch_size = 1;
            config.chunking = ChunkingStrategy {
                max_window_tokens: 8,
                overlap_tokens: vec![0],
                summary_fallback_weight: 0.0,
                summary_fallback_tokens: 0,
                chunk_weight_floor: 0.0,
            };
            config.text_recipes = vec![TextRecipe {
                name: "context_role".into(),
                selector: Selector::Role(SectionRole::Context),
                weight: 1.0,
                instruction: None,
            }];

            let store = Arc::new(DeterministicSplitStore::new(split, 19).unwrap());
            let record = DataRecord {
                id: "role_offset_record".into(),
                source: "unit".into(),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                quality: QualityScore::default(),
                taxonomy: vec![],
                sections: vec![
                    RecordSection {
                        role: SectionRole::Context,
                        heading: Some("A".into()),
                        text: "alpha".into(),
                        sentences: vec!["alpha".into()],
                    },
                    RecordSection {
                        role: SectionRole::Context,
                        heading: Some("B".into()),
                        text: "beta".into(),
                        sentences: vec!["beta".into()],
                    },
                    RecordSection {
                        role: SectionRole::Context,
                        heading: Some("C".into()),
                        text: "gamma".into(),
                        sentences: vec!["gamma".into()],
                    },
                ],
                meta_prefix: None,
            };

            let sampler = PairSampler::new(config, store);
            sampler.register_source(Box::new(InMemorySource::new("unit", vec![record])));
            sampler
                .inner
                .lock()
                .unwrap()
                .ingest_internal(SplitLabel::Train)
                .unwrap();
            sampler
        };

        let expected_start =
            (stable_hash_str(seed ^ epoch_seed_mask, key) as usize) % section_count;
        assert_ne!(expected_start, 0);
        let expected = ["alpha", "beta", "gamma"][expected_start];

        let sampler_a = build_sampler();
        let first_a = sampler_a
            .next_text_batch(SplitLabel::Train)
            .unwrap()
            .samples[0]
            .chunk
            .text
            .clone();
        assert_eq!(first_a, expected);

        let sampler_b = build_sampler();
        let first_b = sampler_b
            .next_text_batch(SplitLabel::Train)
            .unwrap()
            .samples[0]
            .chunk
            .text
            .clone();
        assert_eq!(first_b, expected);
        assert_eq!(first_a, first_b);
    }

    #[test]
    fn reentry_same_epoch_restarts_from_same_chunk_offset() {
        let split = SplitRatios::default();
        let store = Arc::new(DeterministicSplitStore::new(split, 23).unwrap());
        let mut config = base_config();
        config.seed = 101;
        let mut inner = PairSamplerInner::new(config, store);

        let mk_chunk = |index: usize, text: &str| RecordChunk {
            record_id: "reentry_record".into(),
            section_idx: 0,
            view: ChunkView::Window {
                index,
                overlap: 0,
                span: 2,
                start_ratio: index as f32 / 3.0,
            },
            text: text.to_string(),
            tokens_estimate: 2,
            quality: QualityScore::default(),
        };
        let pool = vec![mk_chunk(0, "zero"), mk_chunk(1, "one"), mk_chunk(2, "two")];

        let first = inner
            .next_chunk_from_pool("reentry_record", 0, pool.clone())
            .unwrap();

        // Simulate record dropping out of the in-memory window.
        inner
            .chunk_cursors
            .remove(&("reentry_record".to_string(), 0));

        let restarted = inner
            .next_chunk_from_pool("reentry_record", 0, pool)
            .unwrap();

        assert_eq!(restarted.text, first.text);
    }

    #[test]
    fn reentry_after_epoch_change_can_restart_from_different_chunk_offset() {
        let split = SplitRatios::default();
        let store = Arc::new(DeterministicSplitStore::new(split, 29).unwrap());
        let key = "reentry_record::0";
        let pool_len = 3usize;
        let mut seed = 1u64;
        while (stable_hash_str(seed, key) as usize) % pool_len
            == (stable_hash_str(seed ^ 1, key) as usize) % pool_len
        {
            seed = seed.saturating_add(1);
        }

        let mut config = base_config();
        config.seed = seed;
        let mut inner = PairSamplerInner::new(config, store);

        let mk_chunk = |index: usize, text: &str| RecordChunk {
            record_id: "reentry_record".into(),
            section_idx: 0,
            view: ChunkView::Window {
                index,
                overlap: 0,
                span: 2,
                start_ratio: index as f32 / 3.0,
            },
            text: text.to_string(),
            tokens_estimate: 2,
            quality: QualityScore::default(),
        };
        let pool = vec![mk_chunk(0, "zero"), mk_chunk(1, "one"), mk_chunk(2, "two")];

        let first_epoch0 = inner
            .next_chunk_from_pool("reentry_record", 0, pool.clone())
            .unwrap();

        // Simulate record eviction + later re-entry after source epoch advanced.
        inner
            .chunk_cursors
            .remove(&("reentry_record".to_string(), 0));
        inner.source_epoch = inner.source_epoch.saturating_add(1);

        let first_epoch1 = inner
            .next_chunk_from_pool("reentry_record", 0, pool)
            .unwrap();

        assert_ne!(first_epoch1.text, first_epoch0.text);
    }

    #[test]
    fn kvp_date_formats_can_differ_within_same_triplet_across_all_splits() {
        let split = SplitRatios {
            train: 0.34,
            validation: 0.33,
            test: 0.33,
        };
        let mut config = base_config();
        config.seed = 777;
        config.batch_size = 1;
        config.allowed_splits = vec![SplitLabel::Train];
        config.recipes = vec![TripletRecipe {
            name: "kvp_date_triplet".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
        }];
        config.text_recipes = Vec::new();

        let store = Arc::new(DeterministicSplitStore::new(split, 73).unwrap());

        let find_id = |label: SplitLabel, prefix: &str| -> String {
            for i in 0..5000 {
                let id = format!("{prefix}_{i}");
                if store.ensure(id.clone()).unwrap() == label {
                    return id;
                }
            }
            panic!("unable to find id for {:?}", label);
        };

        let ids = vec![
            find_id(SplitLabel::Train, "kvp_date_train_a"),
            find_id(SplitLabel::Train, "kvp_date_train_b"),
            find_id(SplitLabel::Validation, "kvp_date_val_a"),
            find_id(SplitLabel::Validation, "kvp_date_val_b"),
            find_id(SplitLabel::Test, "kvp_date_test_a"),
            find_id(SplitLabel::Test, "kvp_date_test_b"),
        ];

        let sampler = PairSampler::new(config, Arc::clone(&store));

        let records: Vec<DataRecord> = ids
            .into_iter()
            .enumerate()
            .map(|(idx, id)| {
                let mut record = trader_record(&id, "2025-05-01", &format!("T{idx}"), "Body");
                let mut prefix = KvpPrefixSampler::new(1.0);
                if idx % 2 == 0 {
                    prefix.add_variant([("date", "2025-05-01")]);
                    prefix.add_variant([("date", "May 1, 2025")]);
                } else {
                    prefix.add_variant([("date", "05/01/2025")]);
                    prefix.add_variant([("date", "2025-05-01")]);
                }
                record.meta_prefix = Some(prefix);
                record
            })
            .collect();

        sampler.register_source(Box::new(InMemorySource::new("tt", records)));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();

        let mut seen_splits = std::collections::HashSet::new();
        let mut saw_mixed_date_formats = false;
        for _ in 0..180 {
            let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
            let triplet = &batch.triplets[0];

            seen_splits.insert(store.label_for(&triplet.anchor.record_id).unwrap());
            seen_splits.insert(store.label_for(&triplet.positive.record_id).unwrap());
            seen_splits.insert(store.label_for(&triplet.negative.record_id).unwrap());

            let dates = [
                extract_date_prefix(&triplet.anchor.text),
                extract_date_prefix(&triplet.positive.text),
                extract_date_prefix(&triplet.negative.text),
            ];
            if dates.iter().all(Option::is_some) {
                let mut uniq = std::collections::HashSet::new();
                for date in dates.into_iter().flatten() {
                    uniq.insert(date);
                }
                if uniq.len() >= 2 {
                    saw_mixed_date_formats = true;
                }
            }

            if saw_mixed_date_formats && seen_splits.len() == 1 {
                break;
            }
        }

        assert_eq!(
            seen_splits.len(),
            1,
            "expected sampling to stay in the target split"
        );
        assert!(
            saw_mixed_date_formats,
            "expected at least one triplet with multiple date formats across anchor/positive/negative"
        );
    }

    #[test]
    fn kvp_date_formats_can_differ_between_anchor_and_positive_across_all_splits() {
        let split = SplitRatios {
            train: 0.34,
            validation: 0.33,
            test: 0.33,
        };
        let store = Arc::new(DeterministicSplitStore::new(split, 83).unwrap());

        let find_id = |label: SplitLabel, prefix: &str| -> String {
            for i in 0..5000 {
                let id = format!("{prefix}_{i}");
                if store.ensure(id.clone()).unwrap() == label {
                    return id;
                }
            }
            panic!("unable to find id for {:?}", label);
        };

        let ids = vec![
            find_id(SplitLabel::Train, "kvp_anchor_pos_train_a"),
            find_id(SplitLabel::Train, "kvp_anchor_pos_train_b"),
            find_id(SplitLabel::Validation, "kvp_anchor_pos_val_a"),
            find_id(SplitLabel::Validation, "kvp_anchor_pos_val_b"),
            find_id(SplitLabel::Test, "kvp_anchor_pos_test_a"),
            find_id(SplitLabel::Test, "kvp_anchor_pos_test_b"),
        ];

        let mut config = base_config();
        config.seed = 919;
        config.batch_size = 1;
        config.allowed_splits = vec![SplitLabel::Train];
        config.recipes = vec![TripletRecipe {
            name: "kvp_date_anchor_positive_all_splits".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
        }];
        config.text_recipes = Vec::new();

        let sampler = PairSampler::new(config, Arc::clone(&store));

        let records: Vec<DataRecord> = ids
            .into_iter()
            .map(|id| {
                let mut record = trader_record(&id, "2025-01-31", "T", "B");
                let mut prefix = KvpPrefixSampler::new(1.0);
                prefix.add_variant([("date", "2025-01-31")]);
                prefix.add_variant([("date", "Jan 31, 2025")]);
                prefix.add_variant([("date", "01/31/2025")]);
                record.meta_prefix = Some(prefix);
                record
            })
            .collect();

        sampler.register_source(Box::new(InMemorySource::new("tt", records)));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();

        let mut seen_splits = std::collections::HashSet::new();
        let mut saw_anchor_positive_diff = false;
        for _ in 0..180 {
            let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
            let triplet = &batch.triplets[0];

            seen_splits.insert(store.label_for(&triplet.anchor.record_id).unwrap());
            seen_splits.insert(store.label_for(&triplet.positive.record_id).unwrap());
            seen_splits.insert(store.label_for(&triplet.negative.record_id).unwrap());

            let anchor_date = extract_date_prefix(&triplet.anchor.text);
            let positive_date = extract_date_prefix(&triplet.positive.text);
            if let (Some(a), Some(p)) = (anchor_date, positive_date)
                && a != p
            {
                saw_anchor_positive_diff = true;
            }

            if saw_anchor_positive_diff && seen_splits.len() == 1 {
                break;
            }
        }

        assert_eq!(
            seen_splits.len(),
            1,
            "expected sampling to stay in the target split"
        );
        assert!(
            saw_anchor_positive_diff,
            "expected at least one anchor/positive pair with different date formats"
        );
    }

    #[test]
    fn kvp_prefix_signatures_are_not_constant_across_triplets_with_all_splits() {
        let split = SplitRatios {
            train: 0.34,
            validation: 0.33,
            test: 0.33,
        };
        let mut config = base_config();
        config.seed = 12345;
        config.batch_size = 1;
        config.allowed_splits = vec![SplitLabel::Train];
        config.recipes = vec![TripletRecipe {
            name: "kvp_prefix_diversity_triplet".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
        }];
        config.text_recipes = Vec::new();

        let store = Arc::new(DeterministicSplitStore::new(split, 97).unwrap());

        let find_id = |label: SplitLabel, prefix: &str| -> String {
            for i in 0..5000 {
                let id = format!("{prefix}_{i}");
                if store.ensure(id.clone()).unwrap() == label {
                    return id;
                }
            }
            panic!("unable to find id for {:?}", label);
        };

        let ids = vec![
            find_id(SplitLabel::Train, "kvp_sign_train_a"),
            find_id(SplitLabel::Train, "kvp_sign_train_b"),
            find_id(SplitLabel::Validation, "kvp_sign_val_a"),
            find_id(SplitLabel::Validation, "kvp_sign_val_b"),
            find_id(SplitLabel::Test, "kvp_sign_test_a"),
            find_id(SplitLabel::Test, "kvp_sign_test_b"),
        ];

        let sampler = PairSampler::new(config, Arc::clone(&store));

        let records: Vec<DataRecord> = ids
            .into_iter()
            .enumerate()
            .map(|(idx, id)| {
                let mut record = trader_record(&id, "2025-06-01", &format!("R{idx}"), "Body");
                let mut prefix = KvpPrefixSampler::new(1.0);
                if idx % 2 == 0 {
                    prefix.add_variant([("date", "2025-06-01"), ("source", "tt")]);
                    prefix.add_variant([("date", "Jun 1, 2025"), ("source", "trader")]);
                    prefix.add_variant([("date", "06/01/2025"), ("source", "times")]);
                } else {
                    prefix.add_variant([("date", "2025-06-01"), ("source", "tt")]);
                    prefix.add_variant([("date", "June 1 2025"), ("source", "trader")]);
                    prefix.add_variant([("date", "01-06-2025"), ("source", "times")]);
                }
                record.meta_prefix = Some(prefix);
                record
            })
            .collect();

        sampler.register_source(Box::new(InMemorySource::new("tt", records)));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();

        let mut seen_splits = std::collections::HashSet::new();
        let mut signatures = std::collections::HashSet::new();
        for _ in 0..180 {
            let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
            let triplet = &batch.triplets[0];

            seen_splits.insert(store.label_for(&triplet.anchor.record_id).unwrap());
            seen_splits.insert(store.label_for(&triplet.positive.record_id).unwrap());
            seen_splits.insert(store.label_for(&triplet.negative.record_id).unwrap());

            let anchor = extract_meta_prefix(&triplet.anchor.text);
            let positive = extract_meta_prefix(&triplet.positive.text);
            let negative = extract_meta_prefix(&triplet.negative.text);
            if let (Some(a), Some(p), Some(n)) = (anchor, positive, negative) {
                signatures.insert(format!("{a} || {p} || {n}"));
            }

            if seen_splits.len() == 1 && signatures.len() >= 2 {
                break;
            }
        }

        assert_eq!(
            seen_splits.len(),
            1,
            "expected sampling to stay in the target split"
        );
        assert!(
            signatures.len() >= 2,
            "expected at least two distinct triplet KVP signatures across samples"
        );
    }

    #[test]
    fn triplets_cover_kvp_behaviors_across_all_splits() {
        // Same KVP guarantees as the train-only test, but with split cycling enabled
        // across Train/Validation/Test and explicit verification that all splits
        // are observed while sampling.
        let split = SplitRatios {
            train: 0.34,
            validation: 0.33,
            test: 0.33,
        };
        let store = Arc::new(DeterministicSplitStore::new(split, 211).unwrap());

        let find_id = |label: SplitLabel, prefix: &str| -> String {
            for i in 0..5000 {
                let id = format!("{prefix}_{i}");
                if store.ensure(id.clone()).unwrap() == label {
                    return id;
                }
            }
            panic!("unable to find id for {:?}", label);
        };

        let ids = vec![
            find_id(SplitLabel::Train, "kvp_split_train_a"),
            find_id(SplitLabel::Train, "kvp_split_train_b"),
            find_id(SplitLabel::Validation, "kvp_split_val_a"),
            find_id(SplitLabel::Validation, "kvp_split_val_b"),
            find_id(SplitLabel::Test, "kvp_split_test_a"),
            find_id(SplitLabel::Test, "kvp_split_test_b"),
        ];

        let mut config = base_config();
        config.seed = 515151;
        config.batch_size = 1;
        config.allowed_splits = vec![SplitLabel::Train];
        config.recipes = vec![TripletRecipe {
            name: "kvp_behavior_triplet_all_splits".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
        }];
        config.text_recipes = Vec::new();

        let sampler = PairSampler::new(config, Arc::clone(&store));

        let build_prefix = || {
            let mut prefix = KvpPrefixSampler::new(1.0);
            prefix.add_variant_fields([
                KvpField::many("date", ["2025-08-01", "Aug 1, 2025", "08/01/2025"]),
                KvpField::many("source", ["source_a", "source_primary"]),
                KvpField::one("ticker", "TT").with_presence(0.5),
                KvpField::one("quarter", "Q3").with_presence(0.5),
            ]);
            prefix
        };

        let records: Vec<DataRecord> = ids
            .into_iter()
            .map(|id| {
                let mut record = trader_record(&id, "2025-08-01", "Split Title", "Split Body");
                record.source = "source_a".into();
                record.meta_prefix = Some(build_prefix());
                record
            })
            .collect();

        sampler.register_source(Box::new(InMemorySource::new("tt", records)));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();

        let mut seen_splits = std::collections::HashSet::new();
        let mut saw_triplet_component_divergence = false;
        let mut saw_ticker_present = false;
        let mut saw_ticker_absent = false;
        let mut orderings_by_signature: std::collections::HashMap<
            String,
            std::collections::HashSet<String>,
        > = std::collections::HashMap::new();

        for _ in 0..180 {
            let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
            let triplet = &batch.triplets[0];

            let anchor_split = store.label_for(&triplet.anchor.record_id).unwrap();
            let positive_split = store.label_for(&triplet.positive.record_id).unwrap();
            let negative_split = store.label_for(&triplet.negative.record_id).unwrap();
            seen_splits.insert(anchor_split);
            seen_splits.insert(positive_split);
            seen_splits.insert(negative_split);

            let prefixes = [
                extract_meta_prefix(&triplet.anchor.text),
                extract_meta_prefix(&triplet.positive.text),
                extract_meta_prefix(&triplet.negative.text),
            ];

            if let (Some(a_pref), Some(p_pref), Some(n_pref)) = (
                prefixes[0].as_ref(),
                prefixes[1].as_ref(),
                prefixes[2].as_ref(),
            ) && (a_pref != p_pref || p_pref != n_pref)
            {
                saw_triplet_component_divergence = true;
            }

            for pref in prefixes.into_iter().flatten() {
                let parts = split_meta_parts(&pref);
                let has_ticker = parts.iter().any(|part| part.starts_with("ticker="));
                if has_ticker {
                    saw_ticker_present = true;
                } else {
                    saw_ticker_absent = true;
                }

                let ordered = parts.join(" | ");
                let mut normalized = parts;
                normalized.sort();
                let signature = normalized.join(" | ");
                orderings_by_signature
                    .entry(signature)
                    .or_default()
                    .insert(ordered);
            }

            if seen_splits.len() == 1
                && saw_triplet_component_divergence
                && saw_ticker_present
                && saw_ticker_absent
                && orderings_by_signature
                    .values()
                    .any(|ordered_forms| ordered_forms.len() >= 2)
            {
                break;
            }
        }

        let saw_order_permutation = orderings_by_signature
            .values()
            .any(|ordered_forms| ordered_forms.len() >= 2);

        assert_eq!(
            seen_splits.len(),
            1,
            "expected sampling to stay in the target split"
        );
        assert!(
            saw_triplet_component_divergence,
            "expected anchor/positive/negative KVP prefixes to diverge in at least one triplet"
        );
        assert!(
            saw_ticker_present && saw_ticker_absent,
            "expected optional field to be present on some samples and absent on others"
        );
        assert!(
            saw_order_permutation,
            "expected at least one identical KVP field-set signature to appear in multiple key orders"
        );
    }

    #[test]
    fn role_reentry_same_epoch_restarts_from_same_section_offset() {
        let split = SplitRatios::default();
        let store = Arc::new(DeterministicSplitStore::new(split, 31).unwrap());
        let mut config = base_config();
        config.seed = 131;
        config.chunking = ChunkingStrategy {
            max_window_tokens: 64,
            overlap_tokens: vec![0],
            summary_fallback_weight: 0.0,
            summary_fallback_tokens: 0,
            chunk_weight_floor: 0.0,
        };
        let mut inner = PairSamplerInner::new(config, store);

        let record = DataRecord {
            id: "role_reentry_record".into(),
            source: "unit".into(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            quality: QualityScore::default(),
            taxonomy: vec![],
            sections: vec![
                RecordSection {
                    role: SectionRole::Context,
                    heading: Some("A".into()),
                    text: "alpha".into(),
                    sentences: vec!["alpha".into()],
                },
                RecordSection {
                    role: SectionRole::Context,
                    heading: Some("B".into()),
                    text: "beta".into(),
                    sentences: vec!["beta".into()],
                },
                RecordSection {
                    role: SectionRole::Context,
                    heading: Some("C".into()),
                    text: "gamma".into(),
                    sentences: vec!["gamma".into()],
                },
            ],
            meta_prefix: None,
        };

        let first = inner
            .select_by_role(&record, &SectionRole::Context)
            .expect("first role chunk");

        // Simulate record dropping out and coming back in the same epoch.
        inner
            .role_cursors
            .remove(&(record.id.clone(), role_label(&SectionRole::Context)));
        inner
            .chunk_cursors
            .retain(|(record_id, _), _| record_id != &record.id);

        let restarted = inner
            .select_by_role(&record, &SectionRole::Context)
            .expect("restarted role chunk");

        assert_eq!(restarted.text, first.text);
    }

    #[test]
    fn role_reentry_after_epoch_change_can_restart_from_different_section_offset() {
        let split = SplitRatios::default();
        let store = Arc::new(DeterministicSplitStore::new(split, 37).unwrap());
        let role_key = "role_reentry_record::context";
        let section_count = 3usize;
        let mut seed = 1u64;
        while (stable_hash_str(seed, role_key) as usize) % section_count
            == (stable_hash_str(seed ^ 1, role_key) as usize) % section_count
        {
            seed = seed.saturating_add(1);
        }

        let mut config = base_config();
        config.seed = seed;
        config.chunking = ChunkingStrategy {
            max_window_tokens: 64,
            overlap_tokens: vec![0],
            summary_fallback_weight: 0.0,
            summary_fallback_tokens: 0,
            chunk_weight_floor: 0.0,
        };
        let mut inner = PairSamplerInner::new(config, store);

        let record = DataRecord {
            id: "role_reentry_record".into(),
            source: "unit".into(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            quality: QualityScore::default(),
            taxonomy: vec![],
            sections: vec![
                RecordSection {
                    role: SectionRole::Context,
                    heading: Some("A".into()),
                    text: "alpha".into(),
                    sentences: vec!["alpha".into()],
                },
                RecordSection {
                    role: SectionRole::Context,
                    heading: Some("B".into()),
                    text: "beta".into(),
                    sentences: vec!["beta".into()],
                },
                RecordSection {
                    role: SectionRole::Context,
                    heading: Some("C".into()),
                    text: "gamma".into(),
                    sentences: vec!["gamma".into()],
                },
            ],
            meta_prefix: None,
        };

        let first_epoch0 = inner
            .select_by_role(&record, &SectionRole::Context)
            .expect("first role chunk epoch0");

        // Simulate record eviction + re-entry after source epoch advances.
        inner
            .role_cursors
            .remove(&(record.id.clone(), role_label(&SectionRole::Context)));
        inner
            .chunk_cursors
            .retain(|(record_id, _), _| record_id != &record.id);
        inner.source_epoch = inner.source_epoch.saturating_add(1);

        let first_epoch1 = inner
            .select_by_role(&record, &SectionRole::Context)
            .expect("first role chunk epoch1");

        assert_ne!(first_epoch1.text, first_epoch0.text);
    }

    #[test]
    fn derives_text_recipes_from_triplets() {
        let split = SplitRatios::default();
        let config = SamplerConfig {
            seed: 3,
            batch_size: 1,
            chunking: ChunkingStrategy::default(),
            recipes: vec![TripletRecipe {
                name: "title_to_intro".into(),
                anchor: Selector::Role(SectionRole::Anchor),
                positive_selector: Selector::Role(SectionRole::Context),
                negative_selector: Selector::Role(SectionRole::Context),
                negative_strategy: NegativeStrategy::WrongPublicationDate,
                weight: 1.0,
                instruction: None,
            }],
            text_recipes: Vec::new(),
            split,
            ..SamplerConfig::default()
        };
        let store = Arc::new(DeterministicSplitStore::new(split, 17).unwrap());
        let sampler = PairSampler::new(config, store);
        sampler.register_source(Box::new(InMemorySource::new("unit", vec![sample_record()])));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();
        let batch = sampler.next_text_batch(SplitLabel::Train).unwrap();
        assert!(!batch.is_empty());
        assert_eq!(batch.samples.len(), 1);
        assert!(batch.samples[0].recipe.starts_with("title_to_intro_"));
    }

    #[test]
    fn source_triplets_drive_text_sampling() {
        let split = SplitRatios::default();
        let mut config = base_config();
        config.seed = 7;
        config.batch_size = 1;
        config.recipes.clear();
        config.text_recipes.clear();

        let store = Arc::new(DeterministicSplitStore::new(split, 41).unwrap());
        let records = vec![
            trader_record(
                "source_a::2025/01-01/article_a.txt",
                "2025-01-01",
                "Alpha",
                "Body alpha",
            ),
            trader_record(
                "source_a::2025/01-02/article_b.txt",
                "2025-01-02",
                "Beta",
                "Body beta",
            ),
        ];
        let recipes = vec![TripletRecipe {
            name: Cow::Borrowed("source_auto"),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
        }];
        let decorated = RecipeDecoratedSource::new(records, recipes);
        let sampler = PairSampler::new(config, store);
        sampler.register_source(Box::new(decorated));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();

        let batch = sampler.next_text_batch(SplitLabel::Train).unwrap();
        assert!(batch.samples[0].recipe.starts_with("source_auto_"));
        assert_eq!(batch.samples.len(), 1);
    }

    #[test]
    fn source_defined_recipes_fill_config_gap() {
        let split = SplitRatios::default();
        let config = SamplerConfig {
            seed: 41,
            batch_size: 2,
            chunking: ChunkingStrategy::default(),
            recipes: vec![],
            text_recipes: vec![],
            split,
            ..SamplerConfig::default()
        };
        let store = Arc::new(DeterministicSplitStore::new(split, 19).unwrap());
        let recipes = vec![TripletRecipe {
            name: "inline_title_summary".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
        }];
        let records = vec![
            trader_record(
                "source_a::2025/01-01/article_a.txt",
                "2025-01-01",
                "Alpha",
                "Body alpha",
            ),
            trader_record(
                "source_a::2025/01-02/article_b.txt",
                "2025-01-02",
                "Beta",
                "Body beta",
            ),
        ];
        for record in &records {
            store.upsert(record.id.clone(), SplitLabel::Train).unwrap();
        }
        let sampler = PairSampler::new(config, store);
        sampler.register_source(Box::new(RecipeSource::new(records, recipes.clone())));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();
        let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        assert_eq!(batch.triplets[0].recipe, recipes[0].name.as_ref());
        assert!(!batch.triplets.is_empty());
    }

    #[test]
    fn source_recipes_drive_text_sampling() {
        let split = SplitRatios::default();
        let config = SamplerConfig {
            seed: 43,
            batch_size: 1,
            chunking: ChunkingStrategy::default(),
            recipes: vec![],
            text_recipes: vec![],
            split,
            ..SamplerConfig::default()
        };
        let store = Arc::new(DeterministicSplitStore::new(split, 29).unwrap());
        let recipes = vec![TripletRecipe {
            name: "inline_title_summary".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
        }];
        let records = vec![
            trader_record(
                "source_a::2025/01-01/article_a.txt",
                "2025-01-01",
                "Alpha",
                "Body alpha",
            ),
            trader_record(
                "source_a::2025/01-02/article_b.txt",
                "2025-01-02",
                "Beta",
                "Body beta",
            ),
        ];
        let sampler = PairSampler::new(config, store);
        sampler.register_source(Box::new(RecipeSource::new(records, recipes)));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();
        let batch = sampler.next_text_batch(SplitLabel::Train).unwrap();
        assert_eq!(batch.samples.len(), 1);
        assert!(batch.samples[0].recipe.starts_with("inline_title_summary_"));
    }

    #[test]
    fn source_a_negative_pairs_follow_strategy() {
        let split = SplitRatios::default();
        let config = SamplerConfig {
            seed: 4,
            batch_size: 2,
            chunking: ChunkingStrategy::default(),
            recipes: vec![TripletRecipe {
                name: "tt_wrong_article".into(),
                anchor: Selector::Role(SectionRole::Anchor),
                positive_selector: Selector::Role(SectionRole::Context),
                negative_selector: Selector::Role(SectionRole::Context),
                negative_strategy: NegativeStrategy::WrongArticle,
                weight: 1.0,
                instruction: None,
            }],
            text_recipes: Vec::new(),
            split,
            ..SamplerConfig::default()
        };
        let store = Arc::new(DeterministicSplitStore::new(split, 23).unwrap());
        let records = vec![
            trader_record(
                "source_a::2025/01-01/article_a.txt",
                "2025-01-01",
                "Alpha",
                "Body alpha",
            ),
            trader_record(
                "source_a::2025/01-01/article_b.txt",
                "2025-01-01",
                "Beta",
                "Body beta",
            ),
        ];
        let sampler = PairSampler::new(config, store);
        sampler.register_source(Box::new(InMemorySource::new("tt", records)));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();
        let batch = sampler.next_pair_batch(SplitLabel::Train).unwrap();
        assert!(!batch.pairs.is_empty());
        let negative = batch
            .pairs
            .iter()
            .find(|pair| pair.label == PairLabel::Negative)
            .expect("expected a negative pair");
        assert_eq!(negative.reason.as_deref(), Some("wrong_article"));
        assert_ne!(negative.anchor.record_id, negative.positive.record_id);
    }

    #[test]
    fn qa_negative_pairs_mismatch() {
        let split = SplitRatios::default();
        let config = SamplerConfig {
            seed: 5,
            batch_size: 2,
            chunking: ChunkingStrategy::default(),
            recipes: vec![TripletRecipe {
                name: "qa_wrong_match".into(),
                anchor: Selector::Role(SectionRole::Anchor),
                positive_selector: Selector::Role(SectionRole::Context),
                negative_selector: Selector::Role(SectionRole::Context),
                negative_strategy: NegativeStrategy::QuestionAnswerMismatch,
                weight: 1.0,
                instruction: None,
            }],
            text_recipes: Vec::new(),
            split,
            ..SamplerConfig::default()
        };
        let store = Arc::new(DeterministicSplitStore::new(split, 31).unwrap());
        let records = vec![
            qa_pair_record(
                "source_b::factual/alpha.txt",
                "What is alpha?",
                "Alpha is excess return.",
            ),
            qa_pair_record(
                "source_b::factual/beta.txt",
                "What is beta?",
                "Beta tracks market sensitivity.",
            ),
        ];
        let sampler = PairSampler::new(config, store);
        sampler.register_source(Box::new(InMemorySource::new("qa", records)));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();
        let batch = sampler.next_pair_batch(SplitLabel::Train).unwrap();
        assert!(!batch.pairs.is_empty());
        let negative = batch
            .pairs
            .iter()
            .find(|pair| pair.label == PairLabel::Negative)
            .expect("expected a negative pair");
        assert_eq!(negative.reason.as_deref(), Some("wrong_qa_pairing"));
        assert_ne!(negative.anchor.record_id, negative.positive.record_id);
    }

    #[test]
    fn wrong_article_falls_back_within_same_split() {
        let split = SplitRatios {
            train: 0.34,
            validation: 0.33,
            test: 0.33,
        };
        let config = SamplerConfig {
            seed: 9,
            batch_size: 1,
            chunking: ChunkingStrategy::default(),
            recipes: Vec::new(),
            text_recipes: Vec::new(),
            split,
            ..SamplerConfig::default()
        };
        let store = Arc::new(DeterministicSplitStore::new(split, 47).unwrap());

        let find_id = |label: SplitLabel, prefix: &str| -> String {
            for i in 0..5000 {
                let id = format!("{prefix}_{i}");
                if store.ensure(id.clone()).unwrap() == label {
                    return id;
                }
            }
            panic!("unable to find id for {:?}", label);
        };

        let anchor_ids = vec![
            find_id(SplitLabel::Train, "wa_anchor_train"),
            find_id(SplitLabel::Validation, "wa_anchor_val"),
            find_id(SplitLabel::Test, "wa_anchor_test"),
        ];
        let other_ids = [
            find_id(SplitLabel::Train, "wa_other_train"),
            find_id(SplitLabel::Validation, "wa_other_val"),
            find_id(SplitLabel::Test, "wa_other_test"),
        ];

        let anchor_records: Vec<DataRecord> = anchor_ids
            .iter()
            .enumerate()
            .map(|(i, id)| trader_record(id, "2025-01-01", &format!("Anchor {i}"), "Body alpha"))
            .collect();
        let other_records: Vec<DataRecord> = other_ids
            .iter()
            .enumerate()
            .map(|(i, id)| trader_record(id, "2025-01-02", &format!("Other {i}"), "Body beta"))
            .collect();

        let sampler = PairSampler::new(config, store);
        sampler.register_source(Box::new(InMemorySource::new("tt", anchor_records)));
        sampler.register_source(Box::new(InMemorySource::new("other", other_records)));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();

        let mut inner = sampler.inner.lock().unwrap();
        let mut seen_splits = std::collections::HashSet::new();
        for anchor_id in anchor_ids {
            let anchor = inner.records.get(&anchor_id).cloned().expect("anchor");
            let (negative, _fallback) = inner
                .select_negative_record(&anchor, &NegativeStrategy::WrongArticle)
                .expect("negative");
            assert_ne!(negative.id, anchor.id);
            let anchor_label = inner.split_store.label_for(&anchor.id).unwrap();
            let negative_label = inner.split_store.label_for(&negative.id).unwrap();
            seen_splits.insert(anchor_label);
            assert_eq!(negative_label, anchor_label);
        }
        assert_eq!(seen_splits.len(), 3);
    }

    #[test]
    fn wrong_publication_date_falls_back_within_same_split() {
        let split = SplitRatios {
            train: 0.34,
            validation: 0.33,
            test: 0.33,
        };
        let config = SamplerConfig {
            seed: 7,
            batch_size: 1,
            chunking: ChunkingStrategy::default(),
            recipes: Vec::new(),
            text_recipes: Vec::new(),
            split,
            ..SamplerConfig::default()
        };
        let store = Arc::new(DeterministicSplitStore::new(split, 37).unwrap());

        let find_id = |label: SplitLabel, prefix: &str| -> String {
            for i in 0..5000 {
                let id = format!("{prefix}_{i}");
                if store.ensure(id.clone()).unwrap() == label {
                    return id;
                }
            }
            panic!("unable to find id for {:?}", label);
        };

        let anchor_ids = vec![
            find_id(SplitLabel::Train, "wpd_anchor_train"),
            find_id(SplitLabel::Validation, "wpd_anchor_val"),
            find_id(SplitLabel::Test, "wpd_anchor_test"),
        ];
        let other_ids = [
            find_id(SplitLabel::Train, "wpd_other_train"),
            find_id(SplitLabel::Validation, "wpd_other_val"),
            find_id(SplitLabel::Test, "wpd_other_test"),
        ];

        let anchor_records: Vec<DataRecord> = anchor_ids
            .iter()
            .enumerate()
            .map(|(i, id)| trader_record(id, "2025-01-01", &format!("Anchor {i}"), "Body"))
            .collect();
        let other_records: Vec<DataRecord> = other_ids
            .iter()
            .enumerate()
            .map(|(i, id)| trader_record(id, "2025-01-01", &format!("Other {i}"), "Body"))
            .collect();

        let sampler = PairSampler::new(config, store);
        sampler.register_source(Box::new(InMemorySource::new("tt", anchor_records)));
        sampler.register_source(Box::new(InMemorySource::new("other", other_records)));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();

        let mut inner = sampler.inner.lock().unwrap();
        let mut seen_splits = std::collections::HashSet::new();
        for anchor_id in anchor_ids {
            let anchor = inner.records.get(&anchor_id).cloned().expect("anchor");
            let (negative, _fallback) = inner
                .select_negative_record(&anchor, &NegativeStrategy::WrongPublicationDate)
                .expect("negative");
            assert_ne!(negative.id, anchor.id);
            let anchor_label = inner.split_store.label_for(&anchor.id).unwrap();
            let negative_label = inner.split_store.label_for(&negative.id).unwrap();
            seen_splits.insert(anchor_label);
            assert_eq!(negative_label, anchor_label);
        }
        assert_eq!(seen_splits.len(), 3);
    }

    #[test]
    fn qa_mismatch_falls_back_within_same_split() {
        let split = SplitRatios {
            train: 0.34,
            validation: 0.33,
            test: 0.33,
        };
        let config = SamplerConfig {
            seed: 11,
            batch_size: 1,
            chunking: ChunkingStrategy::default(),
            recipes: Vec::new(),
            text_recipes: Vec::new(),
            split,
            ..SamplerConfig::default()
        };
        let store = Arc::new(DeterministicSplitStore::new(split, 53).unwrap());

        let find_id = |label: SplitLabel, prefix: &str| -> String {
            for i in 0..5000 {
                let id = format!("{prefix}_{i}");
                if store.ensure(id.clone()).unwrap() == label {
                    return id;
                }
            }
            panic!("unable to find id for {:?}", label);
        };

        let anchor_ids = vec![
            find_id(SplitLabel::Train, "qam_anchor_train"),
            find_id(SplitLabel::Validation, "qam_anchor_val"),
            find_id(SplitLabel::Test, "qam_anchor_test"),
        ];
        let other_ids = [
            find_id(SplitLabel::Train, "qam_other_train"),
            find_id(SplitLabel::Validation, "qam_other_val"),
            find_id(SplitLabel::Test, "qam_other_test"),
        ];

        let qa_records: Vec<DataRecord> = anchor_ids
            .iter()
            .enumerate()
            .map(|(i, id)| {
                qa_pair_record(
                    id,
                    &format!("What is item {i}?"),
                    &format!("Item {i} answer."),
                )
            })
            .collect();
        let other_records: Vec<DataRecord> = other_ids
            .iter()
            .enumerate()
            .map(|(i, id)| trader_record(id, "2025-01-02", &format!("Beta {i}"), "Body beta"))
            .collect();

        let sampler = PairSampler::new(config, store);
        sampler.register_source(Box::new(InMemorySource::new("qa", qa_records)));
        sampler.register_source(Box::new(InMemorySource::new("other", other_records)));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();

        let mut inner = sampler.inner.lock().unwrap();
        let mut seen_splits = std::collections::HashSet::new();
        for anchor_id in anchor_ids {
            let anchor = inner.records.get(&anchor_id).cloned().expect("anchor");
            let (negative, _fallback) = inner
                .select_negative_record(&anchor, &NegativeStrategy::QuestionAnswerMismatch)
                .expect("negative");
            assert_ne!(negative.id, anchor.id);
            let anchor_label = inner.split_store.label_for(&anchor.id).unwrap();
            let negative_label = inner.split_store.label_for(&negative.id).unwrap();
            seen_splits.insert(anchor_label);
            assert_eq!(negative_label, anchor_label);
        }
        assert_eq!(seen_splits.len(), 3);
    }

    #[test]
    fn negative_selection_never_falls_back_across_splits() {
        let split = SplitRatios {
            train: 0.34,
            validation: 0.33,
            test: 0.33,
        };
        let store = Arc::new(DeterministicSplitStore::new(split, 17).unwrap());

        let find_id = |label: SplitLabel, prefix: &str| -> String {
            for i in 0..2000 {
                let id = format!("{prefix}_{i}");
                if store.ensure(id.clone()).unwrap() == label {
                    return id;
                }
            }
            panic!("unable to find id for {:?}", label);
        };

        let train_id = find_id(SplitLabel::Train, "neg_train");
        let val_id = find_id(SplitLabel::Validation, "neg_val");
        let test_id = find_id(SplitLabel::Test, "neg_test");

        let config = SamplerConfig {
            seed: 21,
            batch_size: 1,
            chunking: ChunkingStrategy::default(),
            recipes: Vec::new(),
            text_recipes: Vec::new(),
            split,
            ..SamplerConfig::default()
        };

        let anchor = trader_record(&train_id, "2025-01-01", "Anchor", "Body A");
        let other_val = trader_record(&val_id, "2025-01-02", "Other Val", "Body B");
        let other_test = trader_record(&test_id, "2025-01-03", "Other Test", "Body C");
        let sampler = PairSampler::new(config, store);
        sampler.register_source(Box::new(InMemorySource::new("a", vec![anchor.clone()])));
        sampler.register_source(Box::new(InMemorySource::new(
            "b",
            vec![other_val, other_test],
        )));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();

        let mut inner = sampler.inner.lock().unwrap();
        let selected = inner.select_negative_record(&anchor, &NegativeStrategy::WrongArticle);
        assert!(
            selected.is_none(),
            "cross-split fallback must be disallowed when same-split candidates are unavailable"
        );
    }

    #[test]
    fn fallback_triplet_negative_never_matches_anchor() {
        let split = SplitRatios {
            train: 0.34,
            validation: 0.33,
            test: 0.33,
        };
        let store = Arc::new(DeterministicSplitStore::new(split, 59).unwrap());

        let find_id = |label: SplitLabel, prefix: &str| -> String {
            for i in 0..5000 {
                let id = format!("{prefix}_{i}");
                if store.ensure(id.clone()).unwrap() == label {
                    return id;
                }
            }
            panic!("unable to find id for {:?}", label);
        };

        let records = vec![
            trader_record(
                &find_id(SplitLabel::Train, "fallback_train_a"),
                "2025-01-01",
                "Train A",
                "Body",
            ),
            trader_record(
                &find_id(SplitLabel::Train, "fallback_train_b"),
                "2025-01-01",
                "Train B",
                "Body",
            ),
            trader_record(
                &find_id(SplitLabel::Validation, "fallback_val_a"),
                "2025-01-01",
                "Val A",
                "Body",
            ),
            trader_record(
                &find_id(SplitLabel::Validation, "fallback_val_b"),
                "2025-01-01",
                "Val B",
                "Body",
            ),
            trader_record(
                &find_id(SplitLabel::Test, "fallback_test_a"),
                "2025-01-01",
                "Test A",
                "Body",
            ),
            trader_record(
                &find_id(SplitLabel::Test, "fallback_test_b"),
                "2025-01-01",
                "Test B",
                "Body",
            ),
        ];

        let mut config = SamplerConfig {
            seed: 13,
            batch_size: 1,
            chunking: ChunkingStrategy::default(),
            recipes: vec![TripletRecipe {
                name: "wrong_date".into(),
                anchor: Selector::Role(SectionRole::Anchor),
                positive_selector: Selector::Role(SectionRole::Context),
                negative_selector: Selector::Role(SectionRole::Context),
                negative_strategy: NegativeStrategy::WrongPublicationDate,
                weight: 1.0,
                instruction: None,
            }],
            text_recipes: Vec::new(),
            split,
            ..SamplerConfig::default()
        };
        config.allowed_splits = vec![SplitLabel::Train];

        let sampler = PairSampler::new(config, store);
        sampler.register_source(Box::new(InMemorySource::new("tt", records)));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();

        let mut seen_splits = std::collections::HashSet::new();
        let mut saw_fallback = false;
        for _ in 0..120 {
            let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
            let triplet = &batch.triplets[0];
            let anchor_label = sampler
                .inner
                .lock()
                .unwrap()
                .split_store
                .label_for(&triplet.anchor.record_id)
                .unwrap();
            let negative_label = sampler
                .inner
                .lock()
                .unwrap()
                .split_store
                .label_for(&triplet.negative.record_id)
                .unwrap();

            seen_splits.insert(anchor_label);
            assert_eq!(anchor_label, negative_label);
            assert_ne!(triplet.anchor.record_id, triplet.negative.record_id);
            assert_ne!(triplet.positive.record_id, triplet.negative.record_id);
            if triplet.recipe.ends_with("_fallback_same_split") {
                saw_fallback = true;
            }
            if seen_splits.len() == 3 && saw_fallback {
                break;
            }
        }
        assert_eq!(seen_splits.len(), 1);
        assert!(saw_fallback, "expected fallback_same_split to occur");
    }

    #[test]
    fn triplets_never_cross_split_boundaries() {
        let split = SplitRatios {
            train: 0.34,
            validation: 0.33,
            test: 0.33,
        };
        let store = Arc::new(DeterministicSplitStore::new(split, 123).unwrap());

        let find_id = |label: SplitLabel, prefix: &str| -> String {
            for i in 0..5000 {
                let id = format!("{prefix}_{i}");
                if store.ensure(id.clone()).unwrap() == label {
                    return id;
                }
            }
            panic!("unable to find id for {:?}", label);
        };

        let records = vec![
            trader_record(
                &find_id(SplitLabel::Train, "triplet_split_train_a"),
                "2025-01-01",
                "Train A",
                "Body",
            ),
            trader_record(
                &find_id(SplitLabel::Train, "triplet_split_train_b"),
                "2025-01-02",
                "Train B",
                "Body",
            ),
            trader_record(
                &find_id(SplitLabel::Validation, "triplet_split_val_a"),
                "2025-01-03",
                "Val A",
                "Body",
            ),
            trader_record(
                &find_id(SplitLabel::Validation, "triplet_split_val_b"),
                "2025-01-04",
                "Val B",
                "Body",
            ),
            trader_record(
                &find_id(SplitLabel::Test, "triplet_split_test_a"),
                "2025-01-05",
                "Test A",
                "Body",
            ),
            trader_record(
                &find_id(SplitLabel::Test, "triplet_split_test_b"),
                "2025-01-06",
                "Test B",
                "Body",
            ),
        ];

        let mut config = base_config();
        config.seed = 777;
        config.batch_size = 1;
        config.allowed_splits = vec![SplitLabel::Train];
        config.recipes = vec![TripletRecipe {
            name: "split_isolation_triplet".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
        }];
        config.text_recipes = Vec::new();

        let sampler = PairSampler::new(config, Arc::clone(&store));
        sampler.register_source(Box::new(InMemorySource::new("split_iso", records)));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();

        for _ in 0..40 {
            let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
            for triplet in batch.triplets {
                let anchor = store.label_for(&triplet.anchor.record_id).unwrap();
                let positive = store.label_for(&triplet.positive.record_id).unwrap();
                let negative = store.label_for(&triplet.negative.record_id).unwrap();
                assert_eq!(anchor, positive, "anchor and positive must share split");
                assert_eq!(anchor, negative, "negative must stay in anchor split");
            }
        }
    }

    #[test]
    fn split_specific_batch_apis_return_exact_size_and_requested_split_only() {
        let split = SplitRatios {
            train: 0.34,
            validation: 0.33,
            test: 0.33,
        };
        let store = Arc::new(DeterministicSplitStore::new(split, 333).unwrap());

        let find_id = |label: SplitLabel, prefix: &str| -> String {
            for i in 0..10000 {
                let id = format!("{prefix}_{i}");
                if store.ensure(id.clone()).unwrap() == label {
                    return id;
                }
            }
            panic!("unable to find id for {:?}", label);
        };

        let mut records = Vec::new();
        for split_label in [SplitLabel::Train, SplitLabel::Validation, SplitLabel::Test] {
            for idx in 0..6 {
                let id = find_id(split_label, &format!("split_api_{split_label:?}_{idx}"));
                records.push(trader_record(
                    &id,
                    "2025-01-01",
                    &format!("{split_label:?} {idx}"),
                    "body",
                ));
            }
        }

        let mut config = base_config();
        config.seed = 444;
        config.batch_size = 2;
        config.allowed_splits = vec![SplitLabel::Train, SplitLabel::Validation, SplitLabel::Test];
        config.recipes = vec![TripletRecipe {
            name: "split_api_triplet".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
        }];
        config.text_recipes = vec![TextRecipe {
            name: "split_api_text".into(),
            selector: Selector::Role(SectionRole::Context),
            weight: 1.0,
            instruction: None,
        }];

        let sampler = PairSampler::new(config, Arc::clone(&store));
        sampler.register_source(Box::new(InMemorySource::new("split_api", records)));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();

        for requested_split in [SplitLabel::Train, SplitLabel::Validation, SplitLabel::Test] {
            let pair_batch = sampler.next_pair_batch_for_split(requested_split).unwrap();
            assert_eq!(pair_batch.pairs.len(), 2);
            for pair in &pair_batch.pairs {
                assert_eq!(
                    store.label_for(&pair.anchor.record_id).unwrap(),
                    requested_split
                );
                assert_eq!(
                    store.label_for(&pair.positive.record_id).unwrap(),
                    requested_split
                );
            }

            let text_batch = sampler.next_text_batch_for_split(requested_split).unwrap();
            assert_eq!(text_batch.samples.len(), 2);
            for sample in &text_batch.samples {
                assert_eq!(
                    store.label_for(&sample.chunk.record_id).unwrap(),
                    requested_split
                );
            }

            let triplet_batch = sampler
                .next_triplet_batch_for_split(requested_split)
                .unwrap();
            assert_eq!(triplet_batch.triplets.len(), 2);
            for triplet in &triplet_batch.triplets {
                assert_eq!(
                    store.label_for(&triplet.anchor.record_id).unwrap(),
                    requested_split
                );
                assert_eq!(
                    store.label_for(&triplet.positive.record_id).unwrap(),
                    requested_split
                );
                assert_eq!(
                    store.label_for(&triplet.negative.record_id).unwrap(),
                    requested_split
                );
            }
        }
    }

    #[test]
    fn split_specific_triplet_api_keeps_anchor_positive_negative_in_same_split() {
        let split = SplitRatios {
            train: 0.34,
            validation: 0.33,
            test: 0.33,
        };
        let store = Arc::new(DeterministicSplitStore::new(split, 445).unwrap());

        let find_id = |label: SplitLabel, prefix: &str| -> String {
            for i in 0..10000 {
                let id = format!("{prefix}_{i}");
                if store.ensure(id.clone()).unwrap() == label {
                    return id;
                }
            }
            panic!("unable to find id for {:?}", label);
        };

        let mut records = Vec::new();
        for split_label in [SplitLabel::Train, SplitLabel::Validation, SplitLabel::Test] {
            for idx in 0..8 {
                let id = find_id(
                    split_label,
                    &format!("split_triplet_iso_{split_label:?}_{idx}"),
                );
                records.push(trader_record(
                    &id,
                    "2025-01-01",
                    &format!("{split_label:?} {idx}"),
                    "body",
                ));
            }
        }

        let mut config = base_config();
        config.seed = 446;
        config.batch_size = 3;
        config.allowed_splits = vec![SplitLabel::Train, SplitLabel::Validation, SplitLabel::Test];
        config.recipes = vec![TripletRecipe {
            name: "split_triplet_only".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
        }];
        config.text_recipes = Vec::new();

        let sampler = PairSampler::new(config, Arc::clone(&store));
        sampler.register_source(Box::new(InMemorySource::new("split_triplet_iso", records)));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();

        for requested_split in [SplitLabel::Train, SplitLabel::Validation, SplitLabel::Test] {
            let batch = sampler
                .next_triplet_batch_for_split(requested_split)
                .unwrap();
            assert_eq!(batch.triplets.len(), 3);
            for triplet in &batch.triplets {
                let anchor = store.label_for(&triplet.anchor.record_id).unwrap();
                let positive = store.label_for(&triplet.positive.record_id).unwrap();
                let negative = store.label_for(&triplet.negative.record_id).unwrap();
                assert_eq!(anchor, requested_split);
                assert_eq!(positive, requested_split);
                assert_eq!(negative, requested_split);
                assert_eq!(anchor, positive);
                assert_eq!(anchor, negative);
            }
        }
    }

    #[test]
    fn split_specific_batch_apis_reject_disallowed_splits() {
        let mut config = base_config();
        config.allowed_splits = vec![SplitLabel::Train];
        let split = config.split;
        let store = Arc::new(DeterministicSplitStore::new(split, 999).unwrap());
        let sampler = PairSampler::new(config, store);

        let pair_err = sampler
            .next_pair_batch_for_split(SplitLabel::Validation)
            .unwrap_err();
        assert!(matches!(
            pair_err,
            SamplerError::Configuration(ref msg) if msg.contains("not in allowed_splits")
        ));

        let text_err = sampler
            .next_text_batch_for_split(SplitLabel::Validation)
            .unwrap_err();
        assert!(matches!(
            text_err,
            SamplerError::Configuration(ref msg) if msg.contains("not in allowed_splits")
        ));

        let triplet_err = sampler
            .next_triplet_batch_for_split(SplitLabel::Validation)
            .unwrap_err();
        assert!(matches!(
            triplet_err,
            SamplerError::Configuration(ref msg) if msg.contains("not in allowed_splits")
        ));
    }

    #[test]
    fn triplet_sampling_produces_anchor_positive_and_negative() {
        let split = SplitRatios {
            train: 1.0,
            validation: 0.0,
            test: 0.0,
        };
        let config = SamplerConfig {
            seed: 6,
            batch_size: 1,
            chunking: ChunkingStrategy::default(),
            recipes: vec![TripletRecipe {
                name: "tt_triplet".into(),
                anchor: Selector::Role(SectionRole::Anchor),
                positive_selector: Selector::Role(SectionRole::Context),
                negative_selector: Selector::Role(SectionRole::Context),
                negative_strategy: NegativeStrategy::WrongArticle,
                weight: 1.0,
                instruction: None,
            }],
            text_recipes: Vec::new(),
            split,
            ..SamplerConfig::default()
        };
        let store = Arc::new(DeterministicSplitStore::new(split, 43).unwrap());
        let records = vec![
            trader_record(
                "source_a::2025/01-01/article_a.txt",
                "2025-01-01",
                "Alpha",
                "Body alpha",
            ),
            trader_record(
                "source_a::2025/01-02/article_b.txt",
                "2025-01-02",
                "Beta",
                "Body beta",
            ),
        ];
        let sampler = PairSampler::new(config, store);
        sampler.register_source(Box::new(InMemorySource::new("tt", records)));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();
        let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        assert_eq!(batch.triplets.len(), 1);
        let triplet = &batch.triplets[0];
        assert_ne!(triplet.anchor.record_id, triplet.negative.record_id);
        assert_eq!(triplet.anchor.record_id, triplet.positive.record_id);
        assert!(triplet.instruction.is_none());
    }

    #[test]
    fn refresh_limit_caps_records_per_source() {
        let split = SplitRatios {
            train: 1.0,
            validation: 0.0,
            test: 0.0,
        };
        let mut config = base_config();
        config.split = split;
        config.batch_size = 3;
        config.ingestion_max_records = 3;
        let store = Arc::new(DeterministicSplitStore::new(split, 37).unwrap());
        let base = Utc::now() - Duration::seconds(60);
        let records: Vec<DataRecord> = (0..10)
            .map(|idx| record_with_offset(&format!("record_{idx}"), base, idx as i64))
            .collect();
        let sampler = PairSampler::new(config, store);
        sampler.register_source(Box::new(InMemorySource::new("unit", records)));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();
        assert_eq!(sampler.inner.lock().unwrap().records.len(), 3);
    }

    #[test]
    fn triplet_sampling_cycles_recipes_over_time() {
        let split = SplitRatios::default();
        let mut config = base_config();
        config.batch_size = 1;
        config.allowed_splits = vec![SplitLabel::Train];
        config.split = split;
        config.recipes = vec![
            TripletRecipe {
                name: "recipe_a".into(),
                anchor: Selector::Role(SectionRole::Anchor),
                positive_selector: Selector::Role(SectionRole::Context),
                negative_selector: Selector::Role(SectionRole::Context),
                negative_strategy: NegativeStrategy::WrongArticle,
                weight: 1.0,
                instruction: None,
            },
            TripletRecipe {
                name: "recipe_b".into(),
                anchor: Selector::Role(SectionRole::Anchor),
                positive_selector: Selector::Role(SectionRole::Context),
                negative_selector: Selector::Role(SectionRole::Context),
                negative_strategy: NegativeStrategy::WrongArticle,
                weight: 1.0,
                instruction: None,
            },
        ];
        config.text_recipes = Vec::new();
        let store = Arc::new(DeterministicSplitStore::new(split, 11).unwrap());
        let sampler = PairSampler::new(config, store);
        let mut rec_a = sample_record();
        rec_a.id = "record_a".into();
        let mut rec_b = sample_record();
        rec_b.id = "record_b".into();
        let mut rec_c = sample_record();
        rec_c.id = "record_c".into();
        sampler.register_source(Box::new(InMemorySource::new(
            "unit",
            vec![rec_a, rec_b, rec_c],
        )));

        let mut seen = std::collections::HashSet::new();
        for _ in 0..10 {
            let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
            seen.insert(batch.triplets[0].recipe.clone());
            if seen.len() == 2 {
                break;
            }
        }
        assert!(seen.contains("recipe_a"));
        assert!(seen.contains("recipe_b"));
    }

    #[test]
    fn triplet_batch_dedupes_identical_triplets() {
        let split = SplitRatios {
            train: 1.0,
            validation: 0.0,
            test: 0.0,
        };
        let mut config = base_config();
        config.batch_size = 1;
        config.allowed_splits = vec![SplitLabel::Train];
        config.split = split;
        config.recipes = vec![TripletRecipe {
            name: "dedupe_triplet".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
        }];

        let store = Arc::new(DeterministicSplitStore::new(split, 77).unwrap());
        let sampler = PairSampler::new(config, store);

        let records = vec![
            trader_record(
                "source_a::2025/01-01/dedupe_a.txt",
                "2025-01-01",
                "Dedupe A",
                "Body A",
            ),
            trader_record(
                "source_a::2025/01-02/dedupe_b.txt",
                "2025-01-02",
                "Dedupe B",
                "Body B",
            ),
        ];
        sampler.register_source(Box::new(InMemorySource::new("tt", records)));

        let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        let mut seen = std::collections::HashSet::new();
        for triplet in &batch.triplets {
            let key = (
                triplet.anchor.record_id.clone(),
                triplet.positive.record_id.clone(),
                triplet.negative.record_id.clone(),
            );
            assert!(seen.insert(key), "triplet should be unique within batch");
        }
    }

    #[test]
    fn text_batch_dedupes_identical_chunks() {
        let split = SplitRatios {
            train: 1.0,
            validation: 0.0,
            test: 0.0,
        };
        let mut config = base_config();
        config.batch_size = 1;
        config.allowed_splits = vec![SplitLabel::Train];
        config.split = split;
        config.text_recipes = vec![TextRecipe {
            name: "context_only".into(),
            selector: Selector::Role(SectionRole::Context),
            weight: 1.0,
            instruction: None,
        }];

        let store = Arc::new(DeterministicSplitStore::new(split, 91).unwrap());
        let sampler = PairSampler::new(config, store);

        let records = vec![
            trader_record(
                "source_a::2025/01-01/dedupe_a.txt",
                "2025-01-01",
                "Dedupe A",
                "Body A",
            ),
            trader_record(
                "source_a::2025/01-02/dedupe_b.txt",
                "2025-01-02",
                "Dedupe B",
                "Body B",
            ),
        ];
        sampler.register_source(Box::new(InMemorySource::new("tt", records)));

        let batch = sampler.next_text_batch(SplitLabel::Train).unwrap();
        let mut seen = std::collections::HashSet::new();
        for sample in &batch.samples {
            let key = chunk_key(&sample.chunk);
            assert!(
                seen.insert(key),
                "text sample should be unique within batch"
            );
        }
    }

    #[test]
    fn text_sampling_cycles_recipes_over_time() {
        let split = SplitRatios::default();
        let mut config = base_config();
        config.batch_size = 1;
        config.allowed_splits = vec![SplitLabel::Train];
        config.split = split;
        config.recipes = Vec::new();
        config.text_recipes = vec![
            TextRecipe {
                name: "text_a".into(),
                selector: Selector::Role(SectionRole::Anchor),
                weight: 1.0,
                instruction: None,
            },
            TextRecipe {
                name: "text_b".into(),
                selector: Selector::Role(SectionRole::Context),
                weight: 1.0,
                instruction: None,
            },
        ];
        let store = Arc::new(DeterministicSplitStore::new(split, 11).unwrap());
        let sampler = PairSampler::new(config, store);
        let mut rec_a = sample_record();
        rec_a.id = "record_a".into();
        let mut rec_b = sample_record();
        rec_b.id = "record_b".into();
        sampler.register_source(Box::new(InMemorySource::new("unit", vec![rec_a, rec_b])));

        let mut seen = std::collections::HashSet::new();
        for _ in 0..10 {
            let batch = sampler.next_text_batch(SplitLabel::Train).unwrap();
            seen.insert(batch.samples[0].recipe.clone());
            if seen.len() == 2 {
                break;
            }
        }
        assert!(seen.contains("text_a"));
        assert!(seen.contains("text_b"));
    }

    #[test]
    fn epoch_sampling_visits_each_record_before_repeat() {
        let split = SplitRatios {
            train: 1.0,
            validation: 0.0,
            test: 0.0,
        };
        let mut config = SamplerConfig {
            seed: 101,
            batch_size: 1,
            chunking: ChunkingStrategy::default(),
            recipes: vec![TripletRecipe {
                name: "epoch_triplet".into(),
                anchor: Selector::Role(SectionRole::Anchor),
                positive_selector: Selector::Role(SectionRole::Context),
                negative_selector: Selector::Role(SectionRole::Context),
                negative_strategy: NegativeStrategy::WrongArticle,
                weight: 1.0,
                instruction: None,
            }],
            text_recipes: Vec::new(),
            split,
            ..SamplerConfig::default()
        };
        config.allowed_splits = vec![SplitLabel::Train];
        let store = Arc::new(DeterministicSplitStore::new(split, 59).unwrap());
        let records = vec![
            trader_record(
                "source_a::2025/01-01/epoch_a.txt",
                "2025-01-01",
                "Epoch Alpha",
                "Body alpha",
            ),
            trader_record(
                "source_a::2025/01-02/epoch_b.txt",
                "2025-01-02",
                "Epoch Beta",
                "Body beta",
            ),
            trader_record(
                "source_a::2025/01-03/epoch_c.txt",
                "2025-01-03",
                "Epoch Gamma",
                "Body gamma",
            ),
        ];
        let sampler = PairSampler::new(config, store);
        sampler.register_source(Box::new(InMemorySource::new("tt", records)));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();
        let mut anchors = Vec::new();
        for _ in 0..10 {
            let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
            anchors.extend(batch.triplets.iter().map(|t| t.anchor.record_id.clone()));
        }
        let mut dedup = anchors.clone();
        dedup.sort();
        dedup.dedup();
        assert_eq!(dedup.len(), 3, "all records should appear over time");
    }

    #[test]
    fn epoch_sampling_persists_between_runs() {
        let split = SplitRatios {
            train: 1.0,
            validation: 0.0,
            test: 0.0,
        };
        let temp = tempdir().unwrap();
        let store_path = temp.path().join("epoch_store");
        let build_config = || {
            let mut cfg = SamplerConfig {
                seed: 202,
                batch_size: 3,
                chunking: ChunkingStrategy::default(),
                recipes: vec![TripletRecipe {
                    name: "persist_triplet".into(),
                    anchor: Selector::Role(SectionRole::Anchor),
                    positive_selector: Selector::Role(SectionRole::Context),
                    negative_selector: Selector::Role(SectionRole::Context),
                    negative_strategy: NegativeStrategy::WrongArticle,
                    weight: 1.0,
                    instruction: None,
                }],
                text_recipes: Vec::new(),
                split,
                ..SamplerConfig::default()
            };
            cfg.allowed_splits = vec![SplitLabel::Train];
            cfg
        };
        let dataset = vec![
            trader_record(
                "source_a::2025/02-01/persist_a.txt",
                "2025-02-01",
                "Persist A",
                "Body a",
            ),
            trader_record(
                "source_a::2025/02-02/persist_b.txt",
                "2025-02-02",
                "Persist B",
                "Body b",
            ),
            trader_record(
                "source_a::2025/02-03/persist_c.txt",
                "2025-02-03",
                "Persist C",
                "Body c",
            ),
        ];

        let first_anchor = {
            let store = Arc::new(FileSplitStore::open(&store_path, split, 73).unwrap());
            let sampler = PairSampler::new(build_config(), store);
            sampler.register_source(Box::new(InMemorySource::new("tt", dataset.clone())));
            sampler
                .inner
                .lock()
                .unwrap()
                .ingest_internal(SplitLabel::Train)
                .unwrap();
            let anchor = sampler
                .next_triplet_batch(SplitLabel::Train)
                .unwrap()
                .triplets[0]
                .anchor
                .record_id
                .clone();
            sampler.persist_state().unwrap();
            anchor
        };

        let store = Arc::new(FileSplitStore::open(&store_path, split, 73).unwrap());
        let sampler = PairSampler::new(build_config(), store);
        sampler.register_source(Box::new(InMemorySource::new("tt", dataset.clone())));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();
        let mut anchors = Vec::new();
        for _ in 0..5 {
            let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
            anchors.extend(batch.triplets.iter().map(|t| t.anchor.record_id.clone()));
        }
        sampler.persist_state().unwrap();
        assert!(
            anchors.contains(&first_anchor),
            "previously consumed records may reappear with streaming paging"
        );
    }

    #[test]
    fn epoch_sampling_handles_new_records_after_restart() {
        let split = SplitRatios {
            train: 1.0,
            validation: 0.0,
            test: 0.0,
        };
        let temp = tempdir().unwrap();
        let store_path = temp.path().join("epoch_store_new_records");
        let mut base_config = SamplerConfig {
            seed: 404,
            batch_size: 1,
            chunking: ChunkingStrategy::default(),
            recipes: vec![TripletRecipe {
                name: "persist_triplet_new".into(),
                anchor: Selector::Role(SectionRole::Anchor),
                positive_selector: Selector::Role(SectionRole::Context),
                negative_selector: Selector::Role(SectionRole::Context),
                negative_strategy: NegativeStrategy::WrongArticle,
                weight: 1.0,
                instruction: None,
            }],
            text_recipes: Vec::new(),
            split,
            ..SamplerConfig::default()
        };
        base_config.allowed_splits = vec![SplitLabel::Train];

        let initial_records = vec![
            trader_record(
                "source_a::2025/03-01/restart_a.txt",
                "2025-03-01",
                "Restart Alpha",
                "Body alpha",
            ),
            trader_record(
                "source_a::2025/03-02/restart_b.txt",
                "2025-03-02",
                "Restart Beta",
                "Body beta",
            ),
        ];

        // Prime the store and consume one record.
        let _first_anchor = {
            let store = Arc::new(FileSplitStore::open(&store_path, split, 111).unwrap());
            let sampler = PairSampler::new(base_config.clone(), store);
            sampler.register_source(Box::new(InMemorySource::new("tt", initial_records.clone())));
            sampler
                .inner
                .lock()
                .unwrap()
                .ingest_internal(SplitLabel::Train)
                .unwrap();
            let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
            let anchor = batch.triplets[0].anchor.record_id.clone();
            sampler.persist_state().unwrap();
            anchor
        };

        // Restart with an extra record added.
        let mut expanded_records = initial_records.clone();
        expanded_records.push(trader_record(
            "source_a::2025/03-03/restart_c.txt",
            "2025-03-03",
            "Restart Gamma",
            "Body gamma",
        ));

        let store = Arc::new(FileSplitStore::open(&store_path, split, 111).unwrap());
        let sampler = PairSampler::new(base_config, store);
        sampler.register_source(Box::new(InMemorySource::new(
            "tt",
            expanded_records.clone(),
        )));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();

        let mut seen = std::collections::HashSet::new();
        let max_draws = expanded_records.len() * 3;
        for _ in 0..max_draws {
            if let Ok(batch) = sampler.next_triplet_batch(SplitLabel::Train) {
                for triplet in batch.triplets {
                    seen.insert(triplet.anchor.record_id);
                }
            }
        }
        assert!(seen.contains("source_a::2025/03-03/restart_c.txt"));
    }

    #[test]
    fn oversampling_advances_cursors_on_large_records() {
        // Setup: Force all records into TRAIN split to avoid split-cycling confusion.
        let split = SplitRatios {
            train: 1.0,
            validation: 0.0,
            test: 0.0,
        };
        let mut config = base_config();
        config.batch_size = 3;
        config.text_recipes = vec![TextRecipe {
            name: "context".into(),
            selector: Selector::Role(SectionRole::Context),
            weight: 1.0,
            instruction: None,
        }];
        config.chunking = ChunkingStrategy {
            max_window_tokens: 1,
            overlap_tokens: vec![0],
            summary_fallback_weight: 0.0,
            summary_fallback_tokens: 0,
            chunk_weight_floor: 0.0,
        };

        let store = Arc::new(DeterministicSplitStore::new(split, 123).unwrap());
        let sampler = PairSampler::new(config, store);

        // Record 1: Small Source, Huge Content
        // "One Two Three" -> With max_window_tokens=1 -> Chunks: ["One", "Two", "Three"]
        let multi_chunk_record = DataRecord {
            id: "long_record".into(),
            source: "small".into(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            quality: QualityScore::default(),
            taxonomy: vec![],
            sections: vec![RecordSection {
                role: SectionRole::Context,
                heading: None,
                text: "One Two Three".into(),
                sentences: vec!["One Two Three".into()],
            }],
            meta_prefix: None,
        };

        // Records 2, 3, 4: Large Source, Small Content
        let mut large_source_records = Vec::new();
        for char in ['A', 'B', 'C'] {
            large_source_records.push(DataRecord {
                id: format!("short_{}", char),
                source: "large".into(),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                quality: QualityScore::default(),
                taxonomy: vec![],
                sections: vec![RecordSection {
                    role: SectionRole::Context,
                    heading: None,
                    text: char.to_string(),
                    sentences: vec![char.to_string()],
                }],
                meta_prefix: None,
            });
        }

        sampler.register_source(Box::new(InMemorySource::new(
            "small",
            vec![multi_chunk_record],
        )));
        sampler.register_source(Box::new(InMemorySource::new("large", large_source_records)));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();

        // We expect 6 samples total (3 from Small, 3 from Large)
        // The "Small" samples should progress through the content.
        let mut small_samples = Vec::new();
        let mut large_samples = Vec::new();

        for _ in 0..12 {
            let batch = sampler.next_text_batch(SplitLabel::Train).unwrap();
            for sample in batch.samples {
                let text = sample.chunk.text;
                if sample.chunk.record_id == "long_record" {
                    small_samples.push(text);
                } else {
                    large_samples.push(text);
                }
            }
        }

        assert!(
            small_samples.len() >= 3,
            "Should sample small source multiple times"
        );
        assert!(
            large_samples.len() >= 3,
            "Should sample large source multiple times"
        );

        small_samples.sort();
        assert!(small_samples.contains(&"One".to_string()));
        assert!(small_samples.contains(&"Two".to_string()));
        assert!(small_samples.contains(&"Three".to_string()));

        // Verify coverage of large source
        large_samples.sort();
        large_samples.dedup();
        assert!(
            large_samples.len() >= 2,
            "large source should contribute multiple distinct samples"
        );
    }

    #[test]
    fn text_sampling_balances_sources_without_epoch_tracker() {
        let split = SplitRatios {
            train: 1.0,
            validation: 0.0,
            test: 0.0,
        };
        let mut config = base_config();
        config.batch_size = 2;
        config.split = split;
        config.allowed_splits = vec![SplitLabel::Train];
        config.text_recipes = vec![TextRecipe {
            name: "anchors".into(),
            selector: Selector::Role(SectionRole::Anchor),
            weight: 1.0,
            instruction: None,
        }];

        let store = Arc::new(DeterministicSplitStore::new(split, 73).unwrap());
        let sampler = PairSampler::new(config, store);

        let mut factual = sample_record();
        factual.id = "factual_record".into();
        factual.source = "qa_factual".into();

        let mut opinion = sample_record();
        opinion.id = "opinionated_record".into();
        opinion.source = "qa_opinionated".into();

        sampler.register_source(Box::new(InMemorySource::new(
            "qa_factual_source",
            vec![factual.clone()],
        )));
        sampler.register_source(Box::new(InMemorySource::new(
            "qa_opinion_source",
            vec![opinion.clone()],
        )));

        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();

        let batch = sampler.next_text_batch(SplitLabel::Train).unwrap();
        assert_eq!(batch.samples.len(), 2);
        let mut ids: Vec<_> = batch
            .samples
            .iter()
            .map(|sample| sample.chunk.record_id.as_str())
            .collect();
        ids.sort();
        assert_eq!(ids, vec!["factual_record", "opinionated_record"]);
    }

    #[test]
    fn chunk_sampling_respects_split_boundaries() {
        let split = SplitRatios {
            train: 0.5,
            validation: 0.5,
            test: 0.0,
        };
        let store = Arc::new(DeterministicSplitStore::new(split, 88).unwrap());

        let find_id = |label: SplitLabel, prefix: &str| -> String {
            for i in 0..2000 {
                let id = format!("{prefix}_{i}");
                if store.ensure(id.clone()).unwrap() == label {
                    return id;
                }
            }
            panic!("unable to find id for {:?}", label);
        };

        let train_id = find_id(SplitLabel::Train, "train_candidate");
        let val_id = find_id(SplitLabel::Validation, "val_candidate");

        let mut config = base_config();
        config.batch_size = 1;
        config.allowed_splits = vec![SplitLabel::Train];
        config.text_recipes = vec![TextRecipe {
            name: "context".into(),
            selector: Selector::Role(SectionRole::Context),
            weight: 1.0,
            instruction: None,
        }];
        config.chunking = ChunkingStrategy {
            max_window_tokens: 1,
            overlap_tokens: vec![0],
            summary_fallback_weight: 0.0,
            summary_fallback_tokens: 0,
            chunk_weight_floor: 0.0,
        };

        let sampler = PairSampler::new(config, store);
        let mut train_record = trader_record(&train_id, "2025-01-01", "Train Title", "One Two");
        let mut val_record = trader_record(&val_id, "2025-01-02", "Val Title", "Alpha Beta");
        train_record.source = "split_test".into();
        val_record.source = "split_test".into();

        sampler.register_source(Box::new(InMemorySource::new(
            "split_test",
            vec![train_record, val_record],
        )));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();

        for _ in 0..4 {
            let batch = sampler.next_text_batch(SplitLabel::Train).unwrap();
            let sample = &batch.samples[0];
            let label = sampler
                .inner
                .lock()
                .unwrap()
                .split_store
                .label_for(&sample.chunk.record_id)
                .unwrap();
            assert_eq!(label, SplitLabel::Train);
        }
    }

    #[test]
    fn sampler_allows_concurrent_batch_requests() {
        let split = SplitRatios {
            train: 1.0,
            validation: 0.0,
            test: 0.0,
        };
        let store = Arc::new(DeterministicSplitStore::new(split, 9).unwrap());
        let mut config = base_config();
        config.seed = 7;
        config.batch_size = 1;
        config.ingestion_max_records = 8;
        config.allowed_splits = vec![SplitLabel::Train];
        config.split = split;
        config.text_recipes = vec![TextRecipe {
            name: "concurrent_text".into(),
            selector: Selector::Role(SectionRole::Context),
            weight: 1.0,
            instruction: None,
        }];

        let records = vec![
            sample_record(),
            sample_record(),
            sample_record(),
            sample_record(),
        ];
        let sampler = Arc::new(PairSampler::new(config, store));
        sampler.register_source(Box::new(InMemorySource::new("unit", records)));

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let sampler = Arc::clone(&sampler);
                thread::spawn(move || sampler.next_text_batch(SplitLabel::Train))
            })
            .collect();

        for handle in handles {
            let batch = handle.join().unwrap().unwrap();
            assert_eq!(batch.samples.len(), 1);
        }
    }
}
