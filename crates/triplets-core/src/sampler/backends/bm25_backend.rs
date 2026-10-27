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

use crate::constants::sampler::BM25_QUERY_TOKEN_LIMIT;
use crate::data::DataRecord;
use crate::splits::SplitLabel;
use crate::tokenizer::{Tokenizer, WhitespaceTokenizer};
use crate::types::{RecordId, RecordKey, SourceId};
use crate::utils::platform_newline;

use super::{NegativeBackend, NegativeRequest};

// ── internal structures ───────────────────────────────────────────────────────

/// Metadata stored alongside each indexed document in the global BM25 index.
struct Bm25RecordMeta {
    source: SourceId,
    record_id: RecordId,
    /// Cached split label (`None` when the record has not yet been assigned).
    split: Option<SplitLabel>,
}

/// Cached margin-aware ranking for one anchor+depth: the anchor self-score
/// reference (`None` when the anchor fell outside its own top-K) plus the
/// scored candidates in rank order (anchor key itself excluded).
type CachedScoredRanking = (Option<f32>, Vec<(RecordKey, f32)>);

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
    /// Scored BM25 candidates plus the anchor self-score, keyed by
    /// (anchor record key, search_top_k). The anchor key itself stays
    /// excluded from the ranked vec (self-selection must remain
    /// impossible); its own score is stored alongside as the margin
    /// reference (`None` when the anchor fell outside its own top-K).
    /// Written once per anchor+depth (full-article query), then read-only
    /// until records refresh.
    hard_negatives: RwLock<HashMap<(RecordKey, usize), CachedScoredRanking>>,
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
    /// Margin-filter evaluations where the reference resolved (margin
    /// configured and anchor self-score present).
    #[cfg(feature = "extended-metrics")]
    bm25_margin_evaluated: std::sync::atomic::AtomicU64,
    /// Margin-filter evaluations where no candidate cleared the margin and
    /// the uniform-random pool fallback was taken.
    #[cfg(feature = "extended-metrics")]
    bm25_margin_exhausted: std::sync::atomic::AtomicU64,
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
            #[cfg(feature = "extended-metrics")]
            bm25_margin_evaluated: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "extended-metrics")]
            bm25_margin_exhausted: std::sync::atomic::AtomicU64::new(0),
        }
    }

    // ── private helpers ───────────────────────────────────────────────────────

    /// Core selection logic: intersect globally BM25-ranked candidates with
    /// the caller-supplied strategy pool, then sample from the rank window.
    fn select_hard_negative(
        &self,
        request: NegativeRequest<'_>,
        rng: &mut dyn rand::RngCore,
    ) -> Option<(Arc<DataRecord>, bool)> {
        let NegativeRequest {
            anchor,
            anchor_split,
            pool,
            fallback_used,
            anchor_query_text,
            hardness,
        } = request;
        let pool: &[Arc<DataRecord>] = &pool;
        if pool.is_empty() {
            return None;
        }

        #[cfg(feature = "extended-metrics")]
        {
            self.bm25_selection_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // `pool` is already strategy-filtered by `select_negative_record`.
        // 1) Fetch scored BM25 candidates (self-score + ranked keys with
        //    scores, same-split, cached per anchor+search_top_k for
        //    full-article queries). Intersect with `pool` keys.
        // 2) If empty → fallback random below.
        // 3) If a margin is configured and the self-score resolved, filter
        //    to candidates below `self_score - margin`; exhaustion falls
        //    back to uniform random (NOT Rank-1). Unresolvable reference
        //    bypasses the margin entirely (neither margin counter moves).
        // 4) Else sample uniformly from rank window [start..end).
        let pool_by_key: HashMap<RecordKey, &Arc<DataRecord>> =
            pool.iter().map(|r| (RecordKey::from(r), r)).collect();

        let (self_score, scored_candidates) = self.ranked_candidates_scored(
            anchor,
            anchor_split,
            anchor_query_text,
            hardness.search_top_k,
        );
        // Preserve BM25 rank order (descending score, key tiebreak) through
        // the pool intersection.
        let mut ranked_pool_scored: Vec<(Arc<DataRecord>, f32)> = scored_candidates
            .iter()
            .filter_map(|(key, score)| pool_by_key.get(key).copied().cloned().map(|r| (r, *score)))
            .collect();

        if ranked_pool_scored.is_empty() {
            // Index miss: BM25 yielded nothing intersecting the pool.
            // Unchanged semantics — counted separately from margin misses
            // even when a margin is configured.
            #[cfg(feature = "extended-metrics")]
            {
                self.bm25_fallback_count
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            return Self::random_pool_pick(pool, fallback_used, rng);
        }

        if let Some(margin) = hardness.negative_margin {
            match self_score {
                // Reference unresolvable: bypass the margin for this draw.
                // Neither margin counter moves; visible as
                // `selections − evaluated` drift.
                None => {}
                Some(reference) => {
                    #[cfg(feature = "extended-metrics")]
                    {
                        self.bm25_margin_evaluated
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    let threshold = reference - margin;
                    ranked_pool_scored.retain(|(_, score)| *score < threshold);
                    if ranked_pool_scored.is_empty() {
                        // Margin exhaustion: candidates existed but none
                        // cleared the margin — uniform random pool pick
                        // (NOT Rank-1), flagged as a fallback draw.
                        #[cfg(feature = "extended-metrics")]
                        {
                            self.bm25_margin_exhausted
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        return Self::random_pool_pick(pool, true, rng);
                    }
                }
            }
        }

        let len = ranked_pool_scored.len();
        let start = hardness.min_rank.min(len.saturating_sub(1));
        let end = hardness.max_rank.clamp(start + 1, len);
        let idx = {
            use rand::Rng as _;
            rng.random_range(start..end)
        };
        ranked_pool_scored
            .get(idx)
            .map(|(r, _)| (r.clone(), fallback_used))
    }

    /// Deterministic uniform-random pick from `pool` (sorted by key first so
    /// the draw is stable given the RNG state).
    fn random_pool_pick(
        pool: &[Arc<DataRecord>],
        fallback_used: bool,
        rng: &mut dyn rand::RngCore,
    ) -> Option<(Arc<DataRecord>, bool)> {
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

    /// Return the anchor self-score plus scored BM25 candidates for `anchor`,
    /// restricted to records assigned to `anchor_split`.
    ///
    /// The self-score is the anchor record's *own* score against the query,
    /// read *before* the self-exclusion filter. It is `None` when the anchor
    /// falls outside its own top-K (extreme truncation, duplicate
    /// saturation) or when no index exists for the anchor's source.
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
    fn ranked_candidates_scored(
        &self,
        anchor: &DataRecord,
        anchor_split: SplitLabel,
        anchor_query_text: Option<&str>,
        search_top_k: usize,
    ) -> CachedScoredRanking {
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
                let entry = (None, Vec::new());
                self.hard_negatives
                    .write()
                    .unwrap()
                    .insert(cache_key, entry.clone());
                return entry;
            }
            return (None, Vec::new());
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
        // Capture the anchor's own score BEFORE the self-exclusion filter —
        // it is the margin reference. The anchor record is expected in its
        // own top-K (the query derives from its own text) but not
        // guaranteed; absence yields `None` (unresolvable reference).
        let mut self_score: Option<f32> = None;
        let mut all_scored: Vec<(f32, RecordKey)> = Vec::new();
        for result in results {
            let Some(m) = index.meta.get(result.document.id) else {
                continue;
            };
            let key = RecordKey::new(m.source.clone(), m.record_id.clone());
            if key == anchor_key {
                self_score = Some(result.score);
                continue;
            }
            if m.split != Some(anchor_split) {
                continue;
            }
            all_scored.push((result.score, key));
        }

        all_scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });

        let ranked: Vec<(RecordKey, f32)> = all_scored
            .into_iter()
            .map(|(score, key)| (key, score))
            .collect();
        let entry = (self_score, ranked);
        // Only cache full-article results; chunk-window results are not cached
        // because different windows of the same record produce different rankings.
        if anchor_query_text.is_none() {
            self.hard_negatives
                .write()
                .unwrap()
                .insert(cache_key, entry.clone());
        }
        entry
    }

    /// Legacy key-only view of [`Self::ranked_candidates_scored`], kept for
    /// existing callers/tests that only need the ranking order.
    #[cfg(test)]
    fn ranked_candidates(
        &self,
        anchor: &DataRecord,
        anchor_split: SplitLabel,
        anchor_query_text: Option<&str>,
        search_top_k: usize,
    ) -> Vec<RecordKey> {
        self.ranked_candidates_scored(anchor, anchor_split, anchor_query_text, search_top_k)
            .1
            .into_iter()
            .map(|(key, _)| key)
            .collect()
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
    fn choose_negative(
        &self,
        request: NegativeRequest<'_>,
        rng: &mut dyn rand::RngCore,
    ) -> Option<(Arc<DataRecord>, bool)> {
        self.select_hard_negative(request, rng)
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

    #[cfg(all(feature = "bm25-mining", feature = "extended-metrics"))]
    fn bm25_margin_stats(&self) -> (u64, u64) {
        (
            self.bm25_margin_exhausted
                .load(std::sync::atomic::Ordering::Relaxed),
            self.bm25_margin_evaluated
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
        let to_keys = |entry: &CachedScoredRanking| {
            entry.1.iter().map(|(k, _)| k.clone()).collect::<Vec<_>>()
        };
        let default_depth = crate::config::RetrievalSamplingConfig::default().search_top_k;
        if let Some(v) = guard.get(&(key.clone(), default_depth)) {
            return Some(to_keys(v));
        }
        guard
            .iter()
            .find_map(|((k, _), v)| (k == &key).then(|| to_keys(v)))
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
