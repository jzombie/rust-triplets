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

use bm25::{Document, Language, SearchEngine, SearchEngineBuilder};
use indexmap::IndexMap;

use crate::constants::sampler::BM25_HARD_NEGATIVE_ROTATION_TOP_K;
use crate::data::DataRecord;
use crate::splits::SplitLabel;
use crate::types::RecordId;

use super::super::platform_newline;
use super::NegativeBackend;

// ── internal structures ───────────────────────────────────────────────────────

/// Metadata stored alongside each indexed document in the global BM25 index.
struct Bm25RecordMeta {
    record_id: RecordId,
    /// Cached split label (`None` when the record has not yet been assigned).
    split: Option<SplitLabel>,
}

/// Single global BM25 search engine built over all in-memory records.
///
/// Replaces the old per-`(SourceId, SplitLabel)` index map so BM25 ranking
/// sees the full `ingestion_max_records` pool regardless of how many sources
/// are active.
struct Bm25GlobalIndex {
    /// Metadata for the document at each sequential index position.
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
    /// Cached BM25-ranked candidate IDs keyed by anchor record ID.
    hard_negatives: HashMap<RecordId, Vec<RecordId>>,
    /// Global BM25 search engine built from the current record snapshot.
    global_index: Option<Bm25GlobalIndex>,
    /// Per-`(anchor_id, split)` cursor for deterministic top-K rotation.
    negative_cursors: HashMap<(RecordId, SplitLabel), usize>,
    /// Token limit used when building BM25 document text; mirrors
    /// `config.chunking.max_window_tokens` and is refreshed on every
    /// `on_records_refreshed` call.
    max_window_tokens: usize,
}

impl Bm25Backend {
    pub fn new() -> Self {
        Self {
            hard_negatives: HashMap::new(),
            global_index: None,
            negative_cursors: HashMap::new(),
            max_window_tokens: 0,
        }
    }

    // ── private helpers ───────────────────────────────────────────────────────

    /// Core selection logic: intersect globally BM25-ranked candidates with
    /// the caller-supplied strategy pool, then rotate through the top-K.
    fn select_hard_negative(
        &mut self,
        anchor: &DataRecord,
        anchor_split: SplitLabel,
        pool: &[DataRecord],
        fallback_used: bool,
        rng: &mut dyn rand::RngCore,
    ) -> Option<(DataRecord, bool)> {
        if pool.is_empty() {
            return None;
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
        let pool_by_id: HashMap<&str, &DataRecord> =
            pool.iter().map(|r| (r.id.as_str(), r)).collect();

        let candidate_ids = self.ranked_candidates(anchor, anchor_split);
        let ranked_pool: Vec<DataRecord> = candidate_ids
            .iter()
            .filter_map(|id| pool_by_id.get(id.as_str()).copied().cloned())
            .collect();

        if !ranked_pool.is_empty() {
            let top_k = ranked_pool
                .len()
                .min(BM25_HARD_NEGATIVE_ROTATION_TOP_K.max(1));
            let cursor_key = (anchor.id.clone(), anchor_split);
            let cursor = self.negative_cursors.entry(cursor_key).or_insert(0);
            if *cursor >= top_k {
                *cursor = 0;
            }
            let selected = ranked_pool.get(*cursor).cloned();
            *cursor = (*cursor + 1) % top_k;
            return selected.map(|record| (record, fallback_used));
        }

        // BM25 yielded nothing in the pool — fall back to deterministic random
        // sampling within `pool`.
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
    /// Results are cached per anchor ID for the lifetime of the current corpus
    /// snapshot.  The cache is cleared by `on_sync_start` before each new
    /// snapshot is loaded.
    fn ranked_candidates(
        &mut self,
        anchor: &DataRecord,
        anchor_split: SplitLabel,
    ) -> Vec<RecordId> {
        if let Some(cached) = self.hard_negatives.get(&anchor.id) {
            return cached.clone();
        }

        let Some(index) = self.global_index.as_ref() else {
            self.hard_negatives.insert(anchor.id.clone(), Vec::new());
            return Vec::new();
        };

        let bm25_query_text = record_bm25_text(anchor, self.max_window_tokens);
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
            if m.record_id == anchor.id {
                continue;
            }
            // Restrict to the same split as the anchor.
            if m.split != Some(anchor_split) {
                continue;
            }
            ranked.push(m.record_id.clone());
        }

        self.hard_negatives
            .insert(anchor.id.clone(), ranked.clone());
        ranked
    }

    /// Build a fresh global BM25 index from `records`.
    ///
    /// Called by `on_records_refreshed` when at least one source advanced.
    /// Clears all cached state first so stale entries from the previous
    /// snapshot are never returned.
    fn rebuild_index(
        &mut self,
        records: &IndexMap<RecordId, DataRecord>,
        max_window_tokens: usize,
        split_fn: &dyn Fn(&RecordId) -> Option<SplitLabel>,
    ) {
        self.hard_negatives.clear();
        self.global_index = None;
        self.negative_cursors.clear();
        self.max_window_tokens = max_window_tokens;

        let total = records.len();
        if total < 2 {
            return;
        }

        let mut meta: Vec<Bm25RecordMeta> = Vec::with_capacity(total);
        let mut docs: Vec<Document<usize>> = Vec::with_capacity(total);

        for (idx, record) in records.values().enumerate() {
            let split = split_fn(&record.id);
            meta.push(Bm25RecordMeta {
                record_id: record.id.clone(),
                split,
            });
            docs.push(Document {
                id: idx,
                contents: record_bm25_text(record, max_window_tokens),
            });
        }

        let search_engine =
            SearchEngineBuilder::<usize>::with_documents(Language::English, docs).build();
        self.global_index = Some(Bm25GlobalIndex {
            meta,
            search_engine,
        });
    }
}

// ── NegativeBackend impl ──────────────────────────────────────────────────────

impl NegativeBackend for Bm25Backend {
    fn choose_negative(
        &mut self,
        anchor: &DataRecord,
        anchor_split: SplitLabel,
        pool: Vec<DataRecord>,
        fallback_used: bool,
        rng: &mut dyn rand::RngCore,
    ) -> Option<(DataRecord, bool)> {
        self.select_hard_negative(anchor, anchor_split, &pool, fallback_used, rng)
    }

    fn on_sync_start(&mut self) {
        // Strict mode: per-anchor cursor state must never outlive a corpus
        // snapshot boundary.  Clear it before the new snapshot is loaded.
        self.negative_cursors.clear();
    }

    fn on_records_refreshed(
        &mut self,
        records: &IndexMap<RecordId, DataRecord>,
        max_window_tokens: usize,
        split_fn: &dyn Fn(&RecordId) -> Option<SplitLabel>,
        sources_refreshed: bool,
    ) {
        if sources_refreshed {
            self.rebuild_index(records, max_window_tokens, split_fn);
        }
    }

    fn prune_cursors(&mut self, valid_ids: &HashSet<RecordId>) {
        self.negative_cursors
            .retain(|(record_id, _), _| valid_ids.contains(record_id));
    }

    fn cursors_empty(&self) -> bool {
        self.negative_cursors.is_empty()
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
        &mut self,
        anchor: &DataRecord,
        anchor_split: SplitLabel,
    ) -> Vec<RecordId> {
        self.ranked_candidates(anchor, anchor_split)
    }

    /// Return a clone of the hard-negative candidate list for `anchor_id`, or
    /// `None` when no cache entry exists.
    #[cfg(test)]
    pub(in crate::sampler) fn hard_negatives_get(
        &self,
        anchor_id: &RecordId,
    ) -> Option<Vec<RecordId>> {
        self.hard_negatives.get(anchor_id).cloned()
    }

    /// Return the record IDs of all documents in the global index, in index
    /// order.  Returns `None` when no index has been built yet.
    #[cfg(test)]
    pub(in crate::sampler) fn index_meta_record_ids(&self) -> Option<Vec<RecordId>> {
        self.global_index
            .as_ref()
            .map(|gi| gi.meta.iter().map(|m| m.record_id.clone()).collect())
    }

    /// Return the number of active negative-cursor entries.
    #[cfg(test)]
    pub(in crate::sampler) fn negative_cursors_len(&self) -> usize {
        self.negative_cursors.len()
    }

    /// Return `true` when no negative-cursor entries are active.
    #[cfg(test)]
    pub(in crate::sampler) fn negative_cursors_is_empty(&self) -> bool {
        self.negative_cursors.is_empty()
    }

    /// Insert a synthetic cursor entry.  Used by tests that need to pre-seed
    /// cursor state before calling `prune_cursor_state` or
    /// `sync_records_from_cache`.
    #[cfg(test)]
    pub(in crate::sampler) fn negative_cursors_insert(
        &mut self,
        key: (RecordId, SplitLabel),
        value: usize,
    ) {
        self.negative_cursors.insert(key, value);
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
    let tokens: Vec<&str> = out.split_whitespace().collect();
    if tokens.len() <= max_tokens {
        return out;
    }
    tokens
        .into_iter()
        .take(max_tokens)
        .collect::<Vec<_>>()
        .join(" ")
}
