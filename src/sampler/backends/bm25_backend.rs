//! BM25-based hard-negative mining backend.
//!
//! Compiled only when the `bm25-mining` feature is enabled (`pub(super) mod
//! bm25_backend;` in `backends/mod.rs` is cfg-gated accordingly).
//!
//! All BM25 state that previously lived as scattered `#[cfg]`-gated fields
//! inside `TripletSamplerInner` is encapsulated in [`Bm25Backend`], which
//! implements [`super::NegativeBackend`].  The sampler core in `mod.rs` holds
//! only a `Box<dyn NegativeBackend>` and knows nothing about BM25 internals.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use bm25::{Document, Language, SearchEngine, SearchEngineBuilder};
use indexmap::IndexMap;

use crate::constants::sampler::BM25_HARD_NEGATIVE_ROTATION_TOP_K;
use crate::constants::sampler::BM25_QUERY_TOKEN_LIMIT;
use crate::constants::sampler::BM25_SEARCH_TOP_K;
use crate::data::DataRecord;
use crate::splits::SplitLabel;
use crate::types::{RecordId, SourceId};
use crate::utils::tokenize;

use super::super::platform_newline;
use super::NegativeBackend;

// ── internal structures ───────────────────────────────────────────────────────

/// Metadata stored alongside each indexed document in the global BM25 index.
struct Bm25RecordMeta {
    record_id: RecordId,
    /// Cached split label (`None` when the record has not yet been assigned).
    split: Option<SplitLabel>,
}

/// Per-source BM25 search engine, containing only records from one source.
///
/// One `PerSourceBm25Index` is maintained per active source so that a refresh
/// of source A rebuilds only A's index, leaving all other sources' indexes
/// and their associated hard-negative caches intact.
struct PerSourceBm25Index {
    /// Metadata for each document at its sequential position within this
    /// source's search engine.
    meta: Vec<Bm25RecordMeta>,
    search_engine: SearchEngine<usize>,
}

// ── public backend struct ─────────────────────────────────────────────────────

/// BM25-backed hard-negative selection backend.
///
/// Owns the global BM25 index, the per-anchor candidate cache, and the
/// top-K rotation cursors.  The sampler core holds this as
/// `Box<dyn NegativeBackend>` and interacts only through that trait.
pub struct Bm25Backend {
    /// BM25-ranked candidate IDs keyed by anchor record ID.
    /// Written once per anchor (full-article query), then read-only until records refresh.
    hard_negatives: RwLock<HashMap<RecordId, Vec<RecordId>>>,
    /// Per-source BM25 search engines, keyed by source identifier.
    /// Rebuilt on refresh; read-only during sampling.
    source_indexes: HashMap<SourceId, PerSourceBm25Index>,
    /// Per-`(anchor_id, split)` cursor for deterministic top-K rotation.
    negative_cursors: RwLock<HashMap<(RecordId, SplitLabel), usize>>,
    /// Token limit used when building BM25 document text; mirrors
    /// `config.chunking.max_window_tokens` and is refreshed on every
    /// `on_records_refreshed` call.
    max_window_tokens: usize,
    /// Total calls to `select_hard_negative` (non-empty pool).
    #[cfg(feature = "extended-metrics")]
    bm25_selection_count: std::sync::atomic::AtomicU64,
    /// Calls where BM25 yielded no candidates intersecting the pool and
    /// the random fallback path was taken.
    #[cfg(feature = "extended-metrics")]
    bm25_fallback_count: std::sync::atomic::AtomicU64,
}

impl Bm25Backend {
    pub fn new() -> Self {
        Self {
            hard_negatives: RwLock::new(HashMap::new()),
            source_indexes: HashMap::new(),
            negative_cursors: RwLock::new(HashMap::new()),
            max_window_tokens: 0,
            #[cfg(feature = "extended-metrics")]
            bm25_selection_count: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "extended-metrics")]
            bm25_fallback_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    // ── private helpers ───────────────────────────────────────────────────────

    /// Core selection logic: intersect globally BM25-ranked candidates with
    /// the caller-supplied strategy pool, then rotate through the top-K.
    fn select_hard_negative(
        &self,
        anchor: &DataRecord,
        anchor_split: SplitLabel,
        pool: &[Arc<DataRecord>],
        fallback_used: bool,
        anchor_query_text: Option<&str>,
        rng: &mut dyn rand::RngCore,
    ) -> Option<(Arc<DataRecord>, bool)> {
        if pool.is_empty() {
            return None;
        }

        #[cfg(feature = "extended-metrics")]
        {
            self.bm25_selection_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // BM25 top-K rotation.
        //
        // `pool` is already strategy-filtered by `select_negative_record`:
        // same-source, same-split, and any strategy-specific predicates have
        // been applied before arriving here.  This function re-ranks the pool
        // by BM25 lexical score and rotates through the top-K candidates.
        //
        // 1) Fetch globally BM25-ranked candidate IDs (same-split, cached per
        //    anchor).  Intersect with `pool` IDs to restrict to the
        //    pre-filtered set — no predicates need to be re-checked here.
        // 2) Compute top_k = min(configured_top_k, ranked_pool.len()).
        // 3) Read per-(anchor_id, split) cursor, defaulting to 0.
        // 4) Return ranked_pool[cursor], then advance cursor mod top_k.
        //
        // Cursors are cleared in on_sync_start() so a refreshed corpus
        // restarts rotation from rank-1 for each anchor.
        let pool_by_id: HashMap<&str, &Arc<DataRecord>> =
            pool.iter().map(|r| (r.id.as_str(), r)).collect();

        let candidate_ids = self.ranked_candidates(anchor, anchor_split, anchor_query_text);
        let ranked_pool: Vec<Arc<DataRecord>> = candidate_ids
            .iter()
            .filter_map(|id| pool_by_id.get(id.as_str()).copied().cloned())
            .collect();

        if !ranked_pool.is_empty() {
            let top_k = ranked_pool
                .len()
                .min(BM25_HARD_NEGATIVE_ROTATION_TOP_K.max(1));
            let cursor_key = (anchor.id.clone(), anchor_split);
            let mut cursors = self.negative_cursors.write().unwrap();
            let cursor = cursors.entry(cursor_key).or_insert(0);
            if *cursor >= top_k {
                *cursor = 0;
            }
            let selected = ranked_pool.get(*cursor).cloned();
            *cursor = (*cursor + 1) % top_k;
            drop(cursors);
            return selected.map(|record| (record, fallback_used));
        }

        // BM25 yielded nothing in the pool — fall back to deterministic random
        // sampling within `pool`.
        #[cfg(feature = "extended-metrics")]
        {
            self.bm25_fallback_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let mut fallback = pool.to_vec();
        fallback.sort_by(|a, b| a.id.cmp(&b.id));
        if fallback.is_empty() {
            return None;
        }
        let idx = {
            use rand::Rng as _;
            rng.random_range(0..fallback.len())
        };
        fallback.get(idx).cloned().map(|r| (r, fallback_used))
    }

    /// Return BM25-ranked candidate record IDs for `anchor`, restricted to
    /// records assigned to `anchor_split`.
    ///
    /// When `anchor_query_text` is `Some`, it is used as the BM25 query directly
    /// (the rendered text of the anchor's already-selected chunk window).  Results
    /// are **not** cached in this case because different windows of the same record
    /// produce different queries.
    ///
    /// When `anchor_query_text` is `None`, the query is derived from the full
    /// article text via `record_bm25_text`, and the result is cached per anchor ID.
    /// The cache is invalidated in `on_records_refreshed` when the anchor's own
    /// source refreshes, and stale entries are removed by `prune_cursors`.
    fn ranked_candidates(
        &self,
        anchor: &DataRecord,
        anchor_split: SplitLabel,
        anchor_query_text: Option<&str>,
    ) -> Vec<RecordId> {
        // When using full-article text, serve from cache if available.
        if anchor_query_text.is_none()
            && let Some(cached) = self.hard_negatives.read().unwrap().get(&anchor.id).cloned()
        {
            return cached;
        }

        let Some(index) = self.source_indexes.get(anchor.source.as_str()) else {
            if anchor_query_text.is_none() {
                self.hard_negatives
                    .write()
                    .unwrap()
                    .insert(anchor.id.clone(), Vec::new());
            }
            return Vec::new();
        };

        let owned_text: String;
        let query_limited: String;
        let bm25_query_text: &str = if let Some(text) = anchor_query_text {
            // Truncate window text to BM25_QUERY_TOKEN_LIMIT tokens before
            // querying.  BM25 search cost is O(unique query tokens): a full
            // 1 024-token window yields ~400–600 unique terms after stop-word
            // removal, making each search ~170 ms.  Capping at 64 tokens
            // reduces that to ~10 ms with no loss in hard-negative quality —
            // the leading tokens of a financial window are the most specific.
            let tokens: Vec<&str> = tokenize(text);
            if tokens.len() <= BM25_QUERY_TOKEN_LIMIT {
                text
            } else {
                query_limited = tokens[..BM25_QUERY_TOKEN_LIMIT].join(" ");
                &query_limited
            }
        } else {
            owned_text = record_bm25_text(anchor, self.max_window_tokens);
            &owned_text
        };

        // Search only the anchor's own source index.  The negative pool passed
        // to `choose_negative` is always scoped to `candidate.source ==
        // anchor.source` (with a cross-source fallback handled separately), so
        // results from other sources would be discarded during pool intersection
        // anyway.  Restricting to the anchor's source index avoids O(N_sources)
        // full scans per anchor.
        let results = index
            .search_engine
            .search(bm25_query_text, BM25_SEARCH_TOP_K);
        let mut all_scored: Vec<(f32, RecordId)> = results
            .into_iter()
            .filter_map(|result| {
                let m = index.meta.get(result.document.id)?;
                if m.record_id == anchor.id {
                    return None;
                }
                if m.split != Some(anchor_split) {
                    return None;
                }
                Some((result.score, m.record_id.clone()))
            })
            .collect();

        all_scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });

        let ranked: Vec<RecordId> = all_scored.into_iter().map(|(_, id)| id).collect();
        // Only cache full-article results; chunk-window results are not cached
        // because different windows of the same record produce different rankings.
        if anchor_query_text.is_none() {
            self.hard_negatives
                .write()
                .unwrap()
                .insert(anchor.id.clone(), ranked.clone());
        }
        ranked
    }

    /// Build a fresh BM25 index for `source_id` from `source_records`, which
    /// must all belong to that source.
    ///
    /// If fewer than two records are provided the source's index is removed so
    /// that searches skip it.  All other sources' indexes are left untouched.
    fn rebuild_source_index(
        &mut self,
        source_id: &SourceId,
        source_records: &[&DataRecord],
        split_fn: &dyn Fn(&RecordId) -> Option<SplitLabel>,
    ) {
        if source_records.len() < 2 {
            self.source_indexes.remove(source_id);
            return;
        }

        let mut meta: Vec<Bm25RecordMeta> = Vec::with_capacity(source_records.len());
        let mut docs: Vec<Document<usize>> = Vec::with_capacity(source_records.len());

        for (idx, record) in source_records.iter().enumerate() {
            let split = split_fn(&record.id);
            meta.push(Bm25RecordMeta {
                record_id: record.id.clone(),
                split,
            });
            docs.push(Document {
                id: idx,
                contents: record_bm25_text(record, self.max_window_tokens),
            });
        }

        let search_engine =
            SearchEngineBuilder::<usize>::with_documents(Language::English, docs).build();
        self.source_indexes.insert(
            source_id.clone(),
            PerSourceBm25Index {
                meta,
                search_engine,
            },
        );
    }
}

// ── NegativeBackend impl ──────────────────────────────────────────────────────

impl NegativeBackend for Bm25Backend {
    fn choose_negative(
        &self,
        anchor: &DataRecord,
        anchor_split: SplitLabel,
        pool: Vec<Arc<DataRecord>>,
        fallback_used: bool,
        anchor_query_text: Option<&str>,
        rng: &mut dyn rand::RngCore,
    ) -> Option<(Arc<DataRecord>, bool)> {
        self.select_hard_negative(
            anchor,
            anchor_split,
            &pool,
            fallback_used,
            anchor_query_text,
            rng,
        )
    }

    fn on_sync_start(&mut self) {
        // Strict mode: per-anchor cursor state must never outlive a corpus
        // snapshot boundary.  Clear it before the new snapshot is loaded.
        self.negative_cursors.write().unwrap().clear();
    }

    fn on_records_refreshed(
        &mut self,
        records: &IndexMap<RecordId, Arc<DataRecord>>,
        max_window_tokens: usize,
        split_fn: &dyn Fn(&RecordId) -> Option<SplitLabel>,
        refreshed_source_ids: &[SourceId],
    ) {
        if refreshed_source_ids.is_empty() {
            return;
        }
        self.max_window_tokens = max_window_tokens;

        // Invalidate cached hard-negative lists only for anchors whose source
        // was refreshed — those entries may miss newly-arrived candidates or
        // reference candidates that are no longer ranked the same way.  Anchors
        // from unchanged sources keep their cached lists intact.
        let refreshed_set: HashSet<&str> =
            refreshed_source_ids.iter().map(|s| s.as_str()).collect();
        self.hard_negatives.write().unwrap().retain(|anchor_id, _| {
            records
                .get(anchor_id)
                .map(|r| !refreshed_set.contains(r.source.as_str()))
                .unwrap_or(false)
        });

        // Group current records by source, then rebuild only each refreshed
        // source's index from its current record slice.
        let mut records_by_source: HashMap<&str, Vec<&DataRecord>> = HashMap::new();
        for r in records.values() {
            records_by_source
                .entry(r.source.as_str())
                .or_default()
                .push(r.as_ref());
        }
        for source_id in refreshed_source_ids {
            let source_records = records_by_source
                .get(source_id.as_str())
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            self.rebuild_source_index(source_id, source_records, split_fn);
        }

        // Remove indexes for sources that no longer have any records in the
        // pool (e.g., a source whose stream was exhausted and evicted).
        let active_sources: HashSet<&str> = records.values().map(|r| r.source.as_str()).collect();
        self.source_indexes
            .retain(|source_id, _| active_sources.contains(source_id.as_str()));
    }

    fn prune_cursors(&mut self, valid_ids: &HashSet<RecordId>) {
        self.negative_cursors
            .write()
            .unwrap()
            .retain(|(record_id, _), _| valid_ids.contains(record_id));
        // Also remove hard-negative cache entries for anchors that are no
        // longer in the record pool.
        self.hard_negatives
            .write()
            .unwrap()
            .retain(|anchor_id, _| valid_ids.contains(anchor_id));
    }

    fn cursors_empty(&self) -> bool {
        self.negative_cursors.read().unwrap().is_empty()
    }

    #[cfg(all(feature = "bm25-mining", feature = "extended-metrics"))]
    fn bm25_fallback_stats(&self) -> (u64, u64) {
        (
            self.bm25_fallback_count
                .load(std::sync::atomic::Ordering::Relaxed),
            self.bm25_selection_count
                .load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

// ── test helpers ──────────────────────────────────────────────────────────────

impl Bm25Backend {
    /// Expose `ranked_candidates` for test code in `sampler::tests`.
    #[cfg(test)]
    pub(in crate::sampler) fn ranked_candidates_pub(
        &self,
        anchor: &DataRecord,
        anchor_split: SplitLabel,
    ) -> Vec<RecordId> {
        self.ranked_candidates(anchor, anchor_split, None)
    }

    /// Return a clone of the hard-negative candidate list for `anchor_id`, or
    /// `None` when no cache entry exists.
    #[cfg(test)]
    pub(in crate::sampler) fn hard_negatives_get(
        &self,
        anchor_id: &RecordId,
    ) -> Option<Vec<RecordId>> {
        self.hard_negatives.read().unwrap().get(anchor_id).cloned()
    }

    /// Return the record IDs of all documents across all per-source indexes.
    ///
    /// Sources are visited in sorted order; within each source records appear
    /// in their per-source index order.  For single-source tests this matches
    /// the previous global-index ordering exactly.  Returns `None` when no
    /// indexes have been built yet.
    #[cfg(test)]
    pub(in crate::sampler) fn index_meta_record_ids(&self) -> Option<Vec<RecordId>> {
        if self.source_indexes.is_empty() {
            return None;
        }
        let mut source_keys: Vec<&SourceId> = self.source_indexes.keys().collect();
        source_keys.sort();
        let mut all_ids: Vec<RecordId> = Vec::new();
        for source_id in source_keys {
            if let Some(idx) = self.source_indexes.get(source_id) {
                all_ids.extend(idx.meta.iter().map(|m| m.record_id.clone()));
            }
        }
        Some(all_ids)
    }

    /// Return the number of active negative-cursor entries.
    #[cfg(test)]
    pub(in crate::sampler) fn negative_cursors_len(&self) -> usize {
        self.negative_cursors.read().unwrap().len()
    }

    /// Return `true` when no negative-cursor entries are active.
    #[cfg(test)]
    pub(in crate::sampler) fn negative_cursors_is_empty(&self) -> bool {
        self.negative_cursors.read().unwrap().is_empty()
    }

    /// Insert a synthetic cursor entry.  Used by tests that need to pre-seed
    /// cursor state before calling `prune_cursor_state` or
    /// `sync_records_from_cache`.
    #[cfg(test)]
    pub(in crate::sampler) fn negative_cursors_insert(
        &self,
        key: (RecordId, SplitLabel),
        value: usize,
    ) {
        self.negative_cursors.write().unwrap().insert(key, value);
    }
}

// ── shared text helper ────────────────────────────────────────────────────────

/// Concatenate heading and body text from all sections of `record` for use as
/// a BM25 document or query string.  Truncates to `max_tokens` whitespace-
/// delimited tokens when `max_tokens > 0`.
///
/// Visible to `sampler::tests` so BM25 unit tests can reproduce the exact
/// text the backend feeds to the search engine.
pub(in crate::sampler) fn record_bm25_text(record: &DataRecord, max_tokens: usize) -> String {
    let mut out = String::new();
    for section in &record.sections {
        if let Some(heading) = &section.heading
            && !heading.trim().is_empty()
        {
            out.push_str(heading);
            out.push_str(platform_newline());
        }
        if !section.text.trim().is_empty() {
            out.push_str(&section.text);
            out.push_str(platform_newline());
        }
    }
    if out.trim().is_empty() {
        return record.id.clone();
    }
    if max_tokens == 0 {
        return out;
    }
    let tokens: Vec<&str> = tokenize(&out);
    if tokens.len() <= max_tokens {
        return out;
    }
    tokens
        .into_iter()
        .take(max_tokens)
        .collect::<Vec<_>>()
        .join(" ")
}
