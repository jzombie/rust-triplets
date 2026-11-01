//! Generic local parquet/row-store data source.
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use chrono::Utc;
use tracing::info;
use triplets_core::SamplerError;
use triplets_core::config::{NegativeStrategy, SamplerConfig, Selector, TripletRecipe};
use triplets_core::data::SectionRole;
use triplets_core::source::{DataSource, SourceCursor, SourceSnapshot};

use crate::config::ParquetRowsConfig;
use crate::constants::{
    HF_RECIPE_ANCHOR_ANCHOR_WRONG_ARTICLE, HF_RECIPE_ANCHOR_CONTEXT_WRONG_ARTICLE,
    HF_RECIPE_DICT_ANCHOR_POSITIVE_SAME_RECORD, HF_RECIPE_TEXT_SIMCSE_WRONG_ARTICLE,
};
use crate::row_view::{EligibleIndexCache, ParquetCache, RowCache};
use crate::shard_index::build_shard_index;
use crate::types::{LocalShardState, ShardIndex};
use crate::{rows, shard_indexer};

/// Bulk-oriented row source backed by local shard files.
///
/// Reads parquet row groups (or transcoded `.simdr` row stores) through a
/// deterministic permuted pager.  Remote-backed wrappers add their own
/// download/expansion layer on top and merge new shards via
/// [`ParquetRowSource::ingest_shard`].
///
/// ## Determinism guarantees
///
/// * **Row selection within `refresh`** — deterministic for a fixed local
///   snapshot: the permutation is seeded by `(source_id,
///   materialized_rows, sampler_seed)`.  Growing the snapshot (via
///   [`ParquetRowSource::ingest_shard`]) changes `materialized_rows` and
///   therefore later pages — the same caveat as any lazily-expanded source.
pub struct ParquetRowSource {
    pub(crate) config: ParquetRowsConfig,
    pub(crate) sampler_config: Arc<Mutex<Option<SamplerConfig>>>,
    /// Local row frontier. Public so wrappers and tests can inspect it;
    /// prefer [`ParquetRowSource::ingest_shard`] for mutations.
    pub local: Arc<Mutex<LocalShardState>>,
    /// Decoded-row cache. Public for white-box inspection by wrappers/tests.
    pub cache: Arc<Mutex<RowCache>>,
    pub(crate) parquet_cache: Arc<Mutex<ParquetCache>>,
    pub(crate) eligible_index: Arc<Mutex<EligibleIndexCache>>,
}

impl Clone for ParquetRowSource {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            sampler_config: Arc::clone(&self.sampler_config),
            local: Arc::clone(&self.local),
            cache: Arc::clone(&self.cache),
            parquet_cache: Arc::clone(&self.parquet_cache),
            eligible_index: Arc::clone(&self.eligible_index),
        }
    }
}

impl ParquetRowSource {
    /// Build a new source by indexing local shard files.
    pub fn new(config: ParquetRowsConfig) -> Result<Self, SamplerError> {
        if !config.has_explicit_mapping() {
            return Err(SamplerError::Configuration(
                "parquet row source requires explicit field mapping (anchor/positive/context/text_columns)"
                    .to_string(),
            ));
        }

        fs::create_dir_all(&config.snapshot_dir).map_err(|err| {
            SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!(
                    "failed creating snapshot_dir {}: {err}",
                    config.snapshot_dir.display()
                ),
            }
        })?;

        info!(
            "[triplets:parquet] {} indexing local shards in {}",
            config.source_id,
            config.snapshot_dir.display()
        );
        let (shards, discovered) = build_shard_index(&config).unwrap_or_default();

        let source = Self {
            config,
            sampler_config: Arc::new(Mutex::new(None)),
            local: Arc::new(Mutex::new(LocalShardState {
                materialized_rows: discovered,
                shards,
            })),
            cache: Arc::new(Mutex::new(RowCache::default())),
            parquet_cache: Arc::new(Mutex::new(ParquetCache::default())),
            eligible_index: Arc::new(Mutex::new(EligibleIndexCache::default())),
        };

        // Post-initialization: transcode any transient files to `.simdr`
        // stores with pre-corrected global offsets.
        {
            let mut local = source
                .local
                .lock()
                .map_err(|_| SamplerError::SourceUnavailable {
                    source_id: source.config.source_id.clone(),
                    reason: "parquet row source state lock poisoned".to_string(),
                })?;
            let mut running_total = 0usize;
            let mut transcoded_shards = Vec::with_capacity(local.shards.len());
            for shard in local.shards.drain(..) {
                let mut shard = shard;
                shard.global_start = running_total;
                if let Some(transcoded) = rows::transcode_transient_shard_to_store(&source, &shard)?
                {
                    running_total = running_total.saturating_add(transcoded.row_count);
                    transcoded_shards.push(transcoded);
                }
            }
            local.shards = transcoded_shards;
            local.materialized_rows = running_total;
        }

        info!(
            "[triplets:parquet] {} source ready (rows={}, shards={})",
            source.config.source_id,
            source
                .local
                .lock()
                .map(|local| local.materialized_rows)
                .unwrap_or(0),
            source
                .local
                .lock()
                .map(|local| local.shards.len())
                .unwrap_or(0),
        );

        Ok(source)
    }

    /// Build an empty source without touching the filesystem.
    ///
    /// Used by tests and by wrappers that index shards themselves: callers
    /// register shards via [`ParquetRowSource::ingest_shard`] or the index
    /// helpers in [`crate::shard_index`].
    pub fn new_empty(config: ParquetRowsConfig) -> Self {
        Self {
            config,
            sampler_config: Arc::new(Mutex::new(None)),
            local: Arc::new(Mutex::new(LocalShardState::default())),
            cache: Arc::new(Mutex::new(RowCache::default())),
            parquet_cache: Arc::new(Mutex::new(ParquetCache::default())),
            eligible_index: Arc::new(Mutex::new(EligibleIndexCache::default())),
        }
    }

    /// Borrow this source's configuration.
    pub fn config(&self) -> &ParquetRowsConfig {
        &self.config
    }

    /// Return the stable source id.
    pub fn source_id(&self) -> &str {
        &self.config.source_id
    }

    /// Return the current materialized row count.
    pub fn materialized_row_count(&self) -> usize {
        self.local
            .lock()
            .map(|local| local.materialized_rows)
            .unwrap_or(0)
    }

    /// Return the current indexed shard count.
    pub fn shard_count(&self) -> usize {
        self.local
            .lock()
            .map(|local| local.shards.len())
            .unwrap_or(0)
    }

    /// Snapshot the indexed shards (for logging and usage accounting).
    pub fn shards_snapshot(&self) -> Vec<ShardIndex> {
        self.local
            .lock()
            .map(|local| local.shards.clone())
            .unwrap_or_default()
    }

    /// Return on-disk size for a shard path, or 0 if metadata lookup fails.
    pub fn shard_size_bytes(path: &Path) -> u64 {
        fs::metadata(path).map(|meta| meta.len()).unwrap_or(0)
    }

    /// Sum on-disk bytes of indexed shards under `root`.
    pub fn usage_bytes_under(&self, root: &Path) -> u64 {
        self.local
            .lock()
            .map(|local| {
                local
                    .shards
                    .iter()
                    .filter(|shard| shard.path.starts_with(root))
                    .map(|shard| Self::shard_size_bytes(&shard.path))
                    .sum::<u64>()
            })
            .unwrap_or(0)
    }

    /// Ingest a freshly materialized shard into the local index.
    ///
    /// Hot-reload hook for remote-backed wrappers: assigns
    /// `global_start = materialized_rows`, bumps the row count, appends to
    /// the shard list, and invalidates the eligible-row index — all under
    /// the local-frontier lock, so concurrent readers (which snapshot the
    /// shard list under the same lock) see a consistent mapping and two
    /// concurrent ingests can never collide on `global_start`.
    pub fn ingest_shard(&self, mut shard: ShardIndex) -> Result<(), SamplerError> {
        let mut local = self
            .local
            .lock()
            .map_err(|_| SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: "parquet row source state lock poisoned".to_string(),
            })?;
        shard.global_start = local.materialized_rows;
        local.materialized_rows = local.materialized_rows.saturating_add(shard.row_count);
        local.shards.push(shard);
        drop(local);
        shard_indexer::invalidate_eligible_index(self);
        Ok(())
    }

    /// Apply the disk-cap eviction policy and sync in-memory state.
    ///
    /// Eviction applies to `managed_subdir` under `snapshot_dir` when both
    /// it and `local_disk_cap_bytes` are configured; otherwise this is a
    /// no-op returning `false` (user-owned files are never deleted).
    /// Returns `true` when any shard was evicted, in which case row caches
    /// and the eligible-row index are cleared.
    pub fn enforce_disk_cap(&self) -> Result<bool, SamplerError> {
        let evicted_any = {
            let mut local = self
                .local
                .lock()
                .map_err(|_| SamplerError::SourceUnavailable {
                    source_id: self.config.source_id.clone(),
                    reason: "parquet row source state lock poisoned".to_string(),
                })?;
            shard_indexer::enforce_disk_cap_locked(self, &mut local)?
        };
        if evicted_any {
            if let Ok(mut cache) = self.cache.lock() {
                cache.rows.clear();
                cache.order.clear();
            }
            if let Ok(mut parquet_cache) = self.parquet_cache.lock() {
                parquet_cache.readers.clear();
                parquet_cache.row_groups.clear();
                parquet_cache.row_group_order.clear();
            }
            shard_indexer::invalidate_eligible_index(self);
            shard_indexer::prune_store_cache_to_shards(self, &self.shards_snapshot());
        }
        Ok(evicted_any)
    }

    /// Active sampler config snapshot, if one has been set.
    pub fn sampler_config_snapshot(&self) -> Option<SamplerConfig> {
        self.sampler_config.lock().ok()?.clone()
    }

    /// Record the active sampler config for seed/headroom derivation.
    pub fn set_active_sampler_config(&self, config: &SamplerConfig) {
        if let Ok(mut slot) = self.sampler_config.lock() {
            *slot = Some(config.clone());
        }
    }

    /// Compute the effective internal row read target from refresh `limit`.
    pub fn effective_refresh_batch_target(&self, limit: usize) -> usize {
        let multiplier = self.config.refresh_batch_multiplier.max(1);
        limit.saturating_mul(multiplier)
    }

    /// Active ingestion pool size: sampler `per_source_max_pool` when a
    /// sampler config is set, else the row-cache capacity.
    pub fn active_pool_size(&self) -> usize {
        self.sampler_config
            .lock()
            .ok()
            .and_then(|config| config.as_ref().map(|value| value.per_source_max_pool))
            .unwrap_or(self.config.cache_capacity)
            .max(1)
    }

    /// Seed for deterministic refresh paging.
    pub fn configured_sampler_seed(&self) -> Result<u64, SamplerError> {
        self.sampler_config
            .lock()
            .map_err(|_| SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: "parquet row source sampler-config lock poisoned".to_string(),
            })?
            .as_ref()
            .map(|config| config.seed)
            .ok_or_else(|| SamplerError::SourceInconsistent {
                source_id: self.config.source_id.clone(),
                details: "parquet row source sampler configuration not provided".to_string(),
            })
    }

    /// Paging seed derived from `(source_id, total, sampler_seed)`.
    pub fn paging_seed(&self, total: usize) -> Result<u64, SamplerError> {
        let sampler_seed = self.configured_sampler_seed()?;
        Ok(triplets_core::source::IndexablePager::seed_for_sampler(
            &self.config.source_id,
            total,
            sampler_seed,
        ))
    }

    /// Return the current index-domain upper bound for refresh paging.
    pub fn len_hint(&self) -> Option<usize> {
        self.local.lock().ok().map(|local| local.materialized_rows)
    }

    /// Normalize shard extensions (trim dots/whitespace, lowercase).
    pub fn normalized_shard_extensions(extensions: &[String]) -> Vec<String> {
        extensions
            .iter()
            .map(|value| value.trim().trim_start_matches('.').to_ascii_lowercase())
            .collect::<Vec<_>>()
    }

    /// Read records for the requested global row indices.
    ///
    /// Indices at or beyond the materialized frontier resolve to `None`
    /// (skipped); unresolvable cached indices are an error.
    pub fn read_row_batch(
        &self,
        indices: &[usize],
        out: &mut Vec<triplets_core::data::DataRecord>,
        limit: Option<usize>,
    ) -> Result<(), SamplerError> {
        rows::read_row_batch(self, indices, out, limit)
    }
}

impl DataSource for ParquetRowSource {
    /// Return stable source id.
    fn id(&self) -> &str {
        &self.config.source_id
    }

    /// Refresh source records for the requested cursor and row limit.
    ///
    /// Pages only over materialized local rows with a seed-derived
    /// permutation.  An empty snapshot yields an empty snapshot result.
    fn refresh(
        &self,
        config: &SamplerConfig,
        cursor: Option<&SourceCursor>,
        limit: Option<usize>,
    ) -> Result<SourceSnapshot, SamplerError> {
        self.set_active_sampler_config(config);
        let total = self.materialized_row_count();

        if total == 0 {
            return Ok(SourceSnapshot {
                records: Vec::new(),
                cursor: SourceCursor {
                    last_seen: Utc::now(),
                    revision: 0,
                },
            });
        }

        let max = limit.unwrap_or(total);
        let mut start = cursor.map(|state| state.revision as usize).unwrap_or(0);
        if start >= total {
            start = 0;
        }

        let source_id = self.config.source_id.clone();
        let seed = self.paging_seed(total)?;
        let mut permutation =
            triplets_core::source::IndexPermutation::new(total, seed, start as u64);

        let mut records = Vec::new();
        let read_batch_target = self.effective_refresh_batch_target(max);
        let mut pending_indices = Vec::with_capacity(read_batch_target);
        let should_report = total >= 10_000 || max >= 1_024;
        let report_every = Duration::from_millis(750);
        let refresh_start = Instant::now();
        let mut last_report = refresh_start;
        let mut attempts = 0usize;

        if should_report {
            info!(
                "[triplets:source] refresh start source='{}' total={} target={}",
                source_id, total, max
            );
        }

        while attempts < total && records.len() < max {
            pending_indices.clear();
            let remaining_attempts = total.saturating_sub(attempts);
            let to_collect = read_batch_target.min(remaining_attempts);
            for _ in 0..to_collect {
                if records.len() + pending_indices.len() >= max {
                    break;
                }
                pending_indices.push(permutation.next());
                attempts += 1;
            }

            if pending_indices.is_empty() {
                break;
            }

            if should_report {
                info!(
                    "[triplets:source] refresh batch source='{}' batch_size={} attempted={} fetched={} elapsed={:.1}s",
                    source_id,
                    pending_indices.len(),
                    attempts,
                    records.len(),
                    refresh_start.elapsed().as_secs_f64()
                );
            }

            rows::read_row_batch(self, &pending_indices, &mut records, Some(max))?;

            if should_report && last_report.elapsed() >= report_every {
                info!(
                    "[triplets:source] refresh progress source='{}' attempted={}/{} fetched={}/{} elapsed={:.1}s",
                    source_id,
                    attempts,
                    total,
                    records.len(),
                    max,
                    refresh_start.elapsed().as_secs_f64()
                );
                last_report = Instant::now();
            }
        }

        if should_report {
            info!(
                "[triplets:source] refresh done source='{}' attempted={} fetched={}/{} elapsed={:.2}s",
                source_id,
                attempts,
                records.len(),
                max,
                refresh_start.elapsed().as_secs_f64()
            );
        }

        // Use the seed-derived permutation cursor as the returned revision
        // so the next refresh continues from where this one left off.
        let next_start = permutation.cursor();
        let last_seen = records
            .iter()
            .map(|record| record.updated_at)
            .max()
            .unwrap_or_else(Utc::now);

        Ok(SourceSnapshot {
            records,
            cursor: SourceCursor {
                last_seen,
                revision: next_start as u64,
            },
        })
    }

    /// Return exact reported record count from current len hint.
    fn reported_record_count(&self, config: &SamplerConfig) -> Result<u128, SamplerError> {
        self.set_active_sampler_config(config);
        self.len_hint()
            .map(|count| count as u128)
            .ok_or_else(|| SamplerError::SourceInconsistent {
                source_id: self.config.source_id.clone(),
                details: "parquet row source did not provide len_hint".to_string(),
            })
    }

    /// Return mixed default triplet recipes for row sources.
    fn default_triplet_recipes(&self) -> Vec<TripletRecipe> {
        // Dict-dataset mode: negative_columns is set, meaning negatives come
        // from the same row rather than different records.
        if !self.config.negative_columns.is_empty() {
            return vec![TripletRecipe {
                name: HF_RECIPE_DICT_ANCHOR_POSITIVE_SAME_RECORD.into(),
                anchor: Selector::Role(SectionRole::Anchor),
                positive_selector: Selector::Role(SectionRole::Context),
                negative_selector: Selector::Role(SectionRole::Context),
                negative_strategy: NegativeStrategy::SameRecord,
                weight: 1.0,
                instruction: None,
                allow_same_anchor_positive: false,
                retrieval_sampling: None,
            }];
        }

        // Text-columns mode: anchor_columns is empty and only a plain text column is
        // mapped.  In this configuration every record's Anchor and Context sections
        // carry identical text (the single text field is duplicated by row_to_record).
        // Emit a single SimCSE-style recipe that deliberately allows same-text pairs:
        // the negative comes from a different record, and the model's dropout layers
        // provide the necessary embedding variation between the two identical slots.
        if self.config.anchor_columns.is_empty() {
            return vec![TripletRecipe {
                name: HF_RECIPE_TEXT_SIMCSE_WRONG_ARTICLE.into(),
                anchor: Selector::Role(SectionRole::Anchor),
                positive_selector: Selector::Role(SectionRole::Context),
                negative_selector: Selector::Role(SectionRole::Context),
                negative_strategy: NegativeStrategy::WrongArticle,
                weight: 1.0,
                instruction: None,
                allow_same_anchor_positive: true,
                retrieval_sampling: None,
            }];
        }

        vec![
            // Majority lane remains context negatives for broad coverage and
            // stable optimization across varied schemas.
            TripletRecipe {
                name: HF_RECIPE_ANCHOR_CONTEXT_WRONG_ARTICLE.into(),
                anchor: Selector::Role(SectionRole::Anchor),
                positive_selector: Selector::Role(SectionRole::Context),
                negative_selector: Selector::Role(SectionRole::Context),
                negative_strategy: NegativeStrategy::WrongArticle,
                weight: 0.75,
                instruction: None,
                allow_same_anchor_positive: false,
                retrieval_sampling: None,
            },
            // Medium-hard lane adds anchor-as-negative pressure to improve
            // discrimination between title-like anchor fields.
            TripletRecipe {
                name: HF_RECIPE_ANCHOR_ANCHOR_WRONG_ARTICLE.into(),
                anchor: Selector::Role(SectionRole::Anchor),
                positive_selector: Selector::Role(SectionRole::Context),
                negative_selector: Selector::Role(SectionRole::Anchor),
                negative_strategy: NegativeStrategy::WrongArticle,
                weight: 0.25,
                instruction: None,
                allow_same_anchor_positive: false,
                retrieval_sampling: None,
            },
        ]
    }
}
