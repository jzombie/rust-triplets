//! BM25-based hard negative mining for [`TripletSamplerInner`].
//!
//! This module lives at `src/sampler/bm25.rs` and is compiled only when the
//! `bm25-mining` feature is enabled (`mod bm25` in `mod.rs` carries the gate).
//!
//! Adding a new retrieval backend follows the same pattern: create
//! `sampler/my_backend.rs`, gate the `mod` declaration in `mod.rs` with
//! `#[cfg(feature = "my-backend")]`, and implement `choose_negative_from_pool`
//! there.

use std::collections::HashMap;

use bm25::{Document, Language, SearchEngineBuilder};
use rand::Rng;

use crate::constants::sampler::BM25_HARD_NEGATIVE_ROTATION_TOP_K;
use crate::data::DataRecord;
use crate::splits::{EpochStateStore, SamplerStateStore, SplitLabel, SplitStore};
use crate::types::RecordId;

use super::record_bm25_text;
use super::{Bm25GlobalIndex, Bm25RecordMeta, TripletSamplerInner};

impl<S: SplitStore + EpochStateStore + SamplerStateStore + 'static> TripletSamplerInner<S> {
    /// BM25 flavor of pool-based negative selection.
    ///
    /// Called by `select_negative_record` once the strategy pool is built.
    /// Replaces the uniform-random selection used without `bm25-mining`.
    pub(super) fn choose_negative_from_pool(
        &mut self,
        anchor_record: &DataRecord,
        anchor_split: SplitLabel,
        pool: Vec<DataRecord>,
        fallback_used: bool,
    ) -> Option<(DataRecord, bool)> {
        self.select_bm25_hard_negative_record(anchor_record, anchor_split, &pool, fallback_used)
    }

    fn select_bm25_hard_negative_record(
        &mut self,
        anchor_record: &DataRecord,
        anchor_split: SplitLabel,
        pool: &[DataRecord],
        fallback_used: bool,
    ) -> Option<(DataRecord, bool)> {
        if pool.is_empty() {
            return None;
        }

        // BM25 top-K rotation (deterministic, per anchor+split).
        //
        // `pool` is already strategy-filtered by the caller (`select_negative_record`):
        // same-source, same-split, and any strategy-specific date predicates have all
        // been applied before arriving here.  This function's only job is to re-rank
        // the pool by BM25 lexical score and rotate through the top-K candidates.
        //
        // Algorithm:
        // 1) Get globally BM25-ranked candidate IDs for the anchor (same-split only,
        //    cached per anchor).  Intersect with `pool` IDs to restrict to the
        //    already-filtered candidate set — no predicates need to be re-checked here.
        // 2) Compute `top_k = min(configured_top_k, ranked_pool.len())`.
        // 3) Read per-(anchor_id, split) cursor, defaulting to 0.
        // 4) Return `ranked_pool[cursor]`, then advance cursor modulo `top_k`.
        //
        // Cursors are cleared whenever the index is rebuilt so a refreshed corpus
        // restarts rotation from rank-1 for each anchor.
        let pool_by_id: HashMap<&str, &DataRecord> =
            pool.iter().map(|r| (r.id.as_str(), r)).collect();

        let candidate_ids = self.bm25_ranked_candidates(anchor_record);
        let ranked_pool: Vec<DataRecord> = candidate_ids
            .iter()
            .filter_map(|id| pool_by_id.get(id.as_str()).copied().cloned())
            .collect();

        if !ranked_pool.is_empty() {
            // Rotate only within the highest lexical matches, bounded by top_k.
            let top_k = ranked_pool
                .len()
                .min(BM25_HARD_NEGATIVE_ROTATION_TOP_K.max(1));
            let cursor_key = (anchor_record.id.clone(), anchor_split);
            let cursor = self.bm25_negative_cursors.entry(cursor_key).or_insert(0);
            if *cursor >= top_k {
                *cursor = 0;
            }
            let selected = ranked_pool.get(*cursor).cloned();
            *cursor = (*cursor + 1) % top_k;
            return selected.map(|record| (record, fallback_used));
        }

        // BM25 yielded nothing in the strategy pool — fall back to deterministic
        // random sampling within `pool`.
        let mut fallback = pool.to_vec();
        fallback.sort_by(|a, b| a.id.cmp(&b.id));
        if fallback.is_empty() {
            return None;
        }
        let idx = self.rng.random_range(0..fallback.len());
        fallback
            .get(idx)
            .cloned()
            .map(|record| (record, fallback_used))
    }

    /// Return BM25-ranked candidate record IDs for `anchor_record`, restricted
    /// to records assigned to the same split as the anchor.
    ///
    /// Results are cached per anchor ID for the lifetime of the current corpus
    /// snapshot.  The cache is cleared by `rebuild_bm25_hard_negative_index`
    /// whenever the index is rebuilt.
    pub(super) fn bm25_ranked_candidates(&mut self, anchor_record: &DataRecord) -> Vec<RecordId> {
        if let Some(cached) = self.bm25_hard_negatives.get(&anchor_record.id) {
            return cached.clone();
        }

        let Some(index) = self.bm25_global_index.as_ref() else {
            self.bm25_hard_negatives
                .insert(anchor_record.id.clone(), Vec::new());
            return Vec::new();
        };

        // Split membership is determined once here so candidates are restricted
        // to the same split as the anchor, matching the split-isolation invariant
        // upheld everywhere else in the sampler.
        let anchor_split = self.split_store.label_for(&anchor_record.id);

        let bm25_query_text =
            record_bm25_text(anchor_record, self.config.chunking.max_window_tokens);
        let max_results = index.meta.len();
        let mut results = index.search_engine.search(&bm25_query_text, max_results);
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.document.id.cmp(&b.document.id))
        });

        let mut ranked = Vec::new();
        for result in results {
            let candidate_idx = result.document.id;
            let Some(m) = index.meta.get(candidate_idx) else {
                continue;
            };
            if m.record_id == anchor_record.id {
                continue;
            }
            // Only keep candidates assigned to the same split.
            if m.split != anchor_split {
                continue;
            }
            ranked.push(m.record_id.clone());
        }

        self.bm25_hard_negatives
            .insert(anchor_record.id.clone(), ranked.clone());
        ranked
    }

    /// Build a single global BM25 search engine over every record that exists
    /// in all per-source buffers, carrying per-document metadata for post-rank
    /// filtering.
    ///
    /// This replaces the old per-`(SourceId, SplitLabel)` index map.  The old
    /// design partitioned the shared LRU cache by source, which meant each
    /// source's BM25 candidate pool was at most `ingestion_max_records / N_sources`.
    /// By indexing the full per-source buffers instead, each source's BM25 pool
    /// is `ingestion_max_records` records regardless of how many sources are active.
    ///
    /// Strategy constraints (same source, same split, date match) are applied as
    /// post-rank predicates inside `bm25_ranked_candidates` rather than as
    /// pre-partitioning keys, so one query against the global engine serves all
    /// strategies without re-building separate engines.
    ///
    /// The index is rebuilt from scratch on every source-refresh cycle and is a
    /// no-op otherwise.
    pub(super) fn rebuild_bm25_hard_negative_index(&mut self) {
        self.bm25_hard_negatives.clear();
        self.bm25_global_index = None;
        self.bm25_negative_cursors.clear();

        // Build the BM25 index directly from `self.records`, which is populated
        // from all per-source caches via `sync_records_from_cache`.  Every record
        // in the index is therefore reachable from `self.records` at selection
        // time — no overflow copy is needed.
        let total = self.records.len();
        if total < 2 {
            return;
        }

        let max_tokens = self.config.chunking.max_window_tokens;
        let mut meta: Vec<Bm25RecordMeta> = Vec::with_capacity(total);
        let mut docs: Vec<Document<usize>> = Vec::with_capacity(total);

        for (idx, record) in self.records.values().enumerate() {
            let split = self.split_store.label_for(&record.id);
            meta.push(Bm25RecordMeta {
                record_id: record.id.clone(),
                split,
            });
            docs.push(Document {
                id: idx,
                contents: record_bm25_text(record, max_tokens),
            });
        }

        let search_engine =
            SearchEngineBuilder::<usize>::with_documents(Language::English, docs).build();
        self.bm25_global_index = Some(Bm25GlobalIndex {
            meta,
            search_engine,
        });
    }

    /// Trigger a BM25 index rebuild only when at least one source actually
    /// refreshed during the current ingestion cycle.
    pub(super) fn rebuild_bm25_index_after_refresh_if_needed(&mut self) {
        if !self.ingestion.last_refreshed_sources().is_empty() {
            self.rebuild_bm25_hard_negative_index();
        }
    }
}
