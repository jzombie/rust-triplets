//! BM25-based hard-negative mining backend.
//!
//! Compiled only when the `bm25-mining` feature is enabled (`pub(super) mod
//! bm25_backend;` in `backends/mod.rs` is cfg-gated accordingly).
//!
//! Implements [`super::NegativeBackend`] with rank-windowed hardness sampling
//! (`RetrievalSamplingConfig { min_rank, max_rank, search_top_k }`):
//! uniform sample from `ranked[start..end]`, skipping rank-0 false positives.
//! Ranked keys are cached per `(anchor, search_top_k)` for full-article
//! queries; window queries bypass the cache. Stateless — no rotation cursors.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use bm25::{Document, Language, SearchEngine, SearchEngineBuilder};
use indexmap::IndexMap;

use crate::config::RetrievalSamplingConfig;
use crate::constants::sampler::BM25_QUERY_TOKEN_LIMIT;
use crate::data::DataRecord;
use crate::splits::SplitLabel;
use crate::tokenizer::{Tokenizer, WhitespaceTokenizer};
use crate::types::{RecordId, RecordKey, SourceId};
use crate::utils::platform_newline;

use super::NegativeBackend;

// ── internal structures ───────────────────────────────────────────────────────

/// Metadata stored alongside each indexed document in the global BM25 index.
struct Bm25RecordMeta {
    source: SourceId,
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
/// Owns the per-source BM25 indexes and the per-anchor candidate cache.
/// Sampling is stateless rank-window uniform: no rotation cursors.
pub struct Bm25Backend {
    /// BM25-ranked candidate keys keyed by (anchor record key, search_top_k).
    /// Written once per anchor+depth (full-article query), then read-only until records refresh.
    hard_negatives: RwLock<HashMap<(RecordKey, usize), Vec<RecordKey>>>,
    /// Per-source BM25 search engines, keyed by source identifier.
    /// Rebuilt on refresh; read-only during sampling.
    source_indexes: HashMap<SourceId, PerSourceBm25Index>,
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
            max_window_tokens: 0,
            #[cfg(feature = "extended-metrics")]
            bm25_selection_count: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "extended-metrics")]
            bm25_fallback_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    // ── private helpers ───────────────────────────────────────────────────────

    /// Core selection logic: intersect globally BM25-ranked candidates with
    /// the caller-supplied strategy pool, then sample from the rank window.
    #[allow(clippy::too_many_arguments)]
    fn select_hard_negative(
        &self,
        anchor: &DataRecord,
        anchor_split: SplitLabel,
        pool: &[Arc<DataRecord>],
        fallback_used: bool,
        anchor_query_text: Option<&str>,
        hardness: &RetrievalSamplingConfig,
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

        // `pool` is already strategy-filtered by `select_negative_record`.
        // 1) Fetch BM25-ranked candidate keys (same-split, cached per
        //    anchor+search_top_k). Intersect with `pool` keys.
        // 2) If empty → fallback random below.
        // 3) Else sample uniformly from rank window [start..end).
        let pool_by_key: HashMap<RecordKey, &Arc<DataRecord>> =
            pool.iter().map(|r| (RecordKey::from(r), r)).collect();

        let candidate_keys = self.ranked_candidates(
            anchor,
            anchor_split,
            anchor_query_text,
            hardness.search_top_k,
        );
        let ranked_pool: Vec<Arc<DataRecord>> = candidate_keys
            .iter()
            .filter_map(|key| pool_by_key.get(key).copied().cloned())
            .collect();

        if !ranked_pool.is_empty() {
            let len = ranked_pool.len();
            let start = hardness.min_rank.min(len.saturating_sub(1));
            let end = hardness.max_rank.clamp(start + 1, len);
            let idx = {
                use rand::Rng as _;
                rng.random_range(start..end)
            };
            return ranked_pool.get(idx).cloned().map(|r| (r, fallback_used));
        }

        // BM25 yielded nothing in the pool — fall back to deterministic random
        // sampling within `pool`.
        #[cfg(feature = "extended-metrics")]
        {
            self.bm25_fallback_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let mut fallback = pool.to_vec();
        fallback.sort_by(|a, b| RecordKey::from(a).cmp(&RecordKey::from(b)));
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
        search_top_k: usize,
    ) -> Vec<RecordKey> {
        let anchor_key = RecordKey::from(anchor);
        let depth = search_top_k.max(1);
        let cache_key = (anchor_key.clone(), depth);
        // When using full-article text, serve from cache if available.
        if anchor_query_text.is_none()
            && let Some(cached) = self.hard_negatives.read().unwrap().get(&cache_key).cloned()
        {
            return cached;
        }

        let Some(index) = self.source_indexes.get(anchor.source.as_str()) else {
            if anchor_query_text.is_none() {
                self.hard_negatives
                    .write()
                    .unwrap()
                    .insert(cache_key, Vec::new());
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
            let tokens: Vec<&str> = WhitespaceTokenizer.tokenize(text);
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
        let results = index.search_engine.search(bm25_query_text, depth);
        let mut all_scored: Vec<(f32, RecordKey)> = results
            .into_iter()
            .filter_map(|result| {
                let m = index.meta.get(result.document.id)?;
                let key = RecordKey::new(m.source.clone(), m.record_id.clone());
                if key == anchor_key {
                    return None;
                }
                if m.split != Some(anchor_split) {
                    return None;
                }
                Some((result.score, key))
            })
            .collect();

        all_scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });

        let ranked: Vec<RecordKey> = all_scored.into_iter().map(|(_, key)| key).collect();
        // Only cache full-article results; chunk-window results are not cached
        // because different windows of the same record produce different rankings.
        if anchor_query_text.is_none() {
            self.hard_negatives
                .write()
                .unwrap()
                .insert(cache_key, ranked.clone());
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
        split_fn: &dyn Fn(&RecordKey) -> Option<SplitLabel>,
    ) {
        if source_records.len() < 2 {
            self.source_indexes.remove(source_id);
            return;
        }

        let mut meta: Vec<Bm25RecordMeta> = Vec::with_capacity(source_records.len());
        let mut docs: Vec<Document<usize>> = Vec::with_capacity(source_records.len());

        for (idx, record) in source_records.iter().enumerate() {
            let split = split_fn(&RecordKey::from(*record));
            meta.push(Bm25RecordMeta {
                source: record.source.clone(),
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
    #[allow(clippy::too_many_arguments)]
    fn choose_negative(
        &self,
        anchor: &DataRecord,
        anchor_split: SplitLabel,
        pool: Vec<Arc<DataRecord>>,
        fallback_used: bool,
        anchor_query_text: Option<&str>,
        hardness: &RetrievalSamplingConfig,
        rng: &mut dyn rand::RngCore,
    ) -> Option<(Arc<DataRecord>, bool)> {
        self.select_hard_negative(
            anchor,
            anchor_split,
            &pool,
            fallback_used,
            anchor_query_text,
            hardness,
            rng,
        )
    }

    fn on_sync_start(&mut self) {
        // Rank-window sampling is stateless; per-anchor candidate cache
        // is invalidated in on_records_refreshed, not here.
    }

    fn on_records_refreshed(
        &mut self,
        records: &IndexMap<RecordKey, Arc<DataRecord>>,
        max_window_tokens: usize,
        split_fn: &dyn Fn(&RecordKey) -> Option<SplitLabel>,
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
        self.hard_negatives
            .write()
            .unwrap()
            .retain(|(anchor_key, _depth), _| {
                records
                    .get(anchor_key)
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

    fn prune_cursors(&mut self, valid_ids: &HashSet<RecordKey>) {
        self.hard_negatives
            .write()
            .unwrap()
            .retain(|(anchor_key, _), _| valid_ids.contains(anchor_key));
    }

    fn cursors_empty(&self) -> bool {
        self.hard_negatives.read().unwrap().is_empty()
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
    ) -> Vec<RecordKey> {
        let depth = crate::config::RetrievalSamplingConfig::default().search_top_k;
        self.ranked_candidates(anchor, anchor_split, None, depth)
    }

    /// Return a clone of the hard-negative candidate list for `anchor`, or
    /// `None` when no cache entry exists.
    ///
    /// Prefers the default `search_top_k` entry; falls back to any depth for
    /// the same anchor to keep legacy single-depth tests stable.
    #[cfg(test)]
    pub(in crate::sampler) fn hard_negatives_get(
        &self,
        anchor: &DataRecord,
    ) -> Option<Vec<RecordKey>> {
        let key = RecordKey::from(anchor);
        let guard = self.hard_negatives.read().unwrap();
        let default_depth = crate::config::RetrievalSamplingConfig::default().search_top_k;
        if let Some(v) = guard.get(&(key.clone(), default_depth)) {
            return Some(v.clone());
        }
        guard
            .iter()
            .find_map(|((k, _), v)| (k == &key).then(|| v.clone()))
    }

    /// Return the record keys of all documents across all per-source indexes.
    ///
    /// Sources are visited in sorted order; within each source records appear
    /// in their per-source index order.  For single-source tests this matches
    /// the previous global-index ordering exactly.  Returns `None` when no
    /// indexes have been built yet.
    #[cfg(test)]
    pub(in crate::sampler) fn index_meta_record_ids(&self) -> Option<Vec<RecordKey>> {
        if self.source_indexes.is_empty() {
            return None;
        }
        let mut source_keys: Vec<&SourceId> = self.source_indexes.keys().collect();
        source_keys.sort();
        let mut all_keys: Vec<RecordKey> = Vec::new();
        for source_id in source_keys {
            if let Some(idx) = self.source_indexes.get(source_id) {
                all_keys.extend(
                    idx.meta
                        .iter()
                        .map(|m| RecordKey::new(m.source.clone(), m.record_id.clone())),
                );
            }
        }
        Some(all_keys)
    }

    /// Stateless backend: no rotation cursors exist. Kept for
    /// `prune_cursor_state` fast-path compatibility in tests.
    #[cfg(test)]
    pub(in crate::sampler) fn negative_cursors_len(&self) -> usize {
        0
    }

    /// Always true: rank-window sampling holds no cursor state.
    #[cfg(test)]
    pub(in crate::sampler) fn negative_cursors_is_empty(&self) -> bool {
        true
    }

    /// No-op shim: cursor state was removed with the enum flattening.
    #[cfg(test)]
    #[allow(dead_code)]
    pub(in crate::sampler) fn negative_cursors_insert(
        &self,
        _key: (RecordKey, SplitLabel),
        _value: usize,
    ) {
    }

    /// Return cached search depths for anchor (test-only).
    #[cfg(test)]
    pub(in crate::sampler) fn hard_negatives_keys_for_test(
        &self,
        anchor: &RecordKey,
    ) -> Vec<usize> {
        self.hard_negatives
            .read()
            .unwrap()
            .keys()
            .filter_map(|(k, d)| (k == anchor).then_some(*d))
            .collect()
    }

    /// No-op shim for removed cursor key.
    #[cfg(test)]
    #[allow(dead_code)]
    pub(in crate::sampler) fn negative_cursors_insert_full(
        &self,
        _key: (RecordKey, SplitLabel, usize, usize),
        _value: usize,
    ) {
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
    let tokens: Vec<&str> = WhitespaceTokenizer.tokenize(&out);
    if tokens.len() <= max_tokens {
        return out;
    }
    tokens
        .into_iter()
        .take(max_tokens)
        .collect::<Vec<_>>()
        .join(" ")
}
