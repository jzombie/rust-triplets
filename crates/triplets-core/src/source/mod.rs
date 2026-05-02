//! Data source interfaces and paging helpers.
//!
//! Ownership model:
//! - `DataSource` is the sampler-facing interface that produces batches.
//! - `IndexableSource` exposes stable, index-based access into a corpus.
//! - `IndexablePager` owns the deterministic pseudo-random paging logic and
//!   can page any indexable source without retaining per-record state.

use chrono::{DateTime, Utc};
use std::hash::Hash;
use std::time::Instant;

use crate::config::{SamplerConfig, TripletRecipe};
use crate::data::DataRecord;
use crate::errors::SamplerError;
use crate::hash::stable_hash_with;
use crate::types::SourceId;

/// Source implementation modules.
pub mod backends;
/// Utility helpers used by source implementations.
pub mod indexing;
pub use backends::csv_source::{CsvSource, CsvSourceConfig};
pub use backends::file_source::{
    FileSource, FileSourceConfig, SectionBuilder, TaxonomyBuilder, anchor_context_sections,
    taxonomy_from_path,
};

pub use backends::in_memory_source::InMemorySource;

/// Source-owned incremental refresh position.
///
/// The sampler stores and returns this value between refresh calls.
/// `revision` is opaque to the sampler and interpreted only by the source.
#[derive(Clone, Debug)]
pub struct SourceCursor {
    /// Most recent observation timestamp produced by the source.
    pub last_seen: DateTime<Utc>,
    /// Opaque paging position token used to continue incremental refresh.
    pub revision: u64,
}

/// Result of a single source refresh call.
///
/// Pass the returned `cursor` back into the next refresh to continue paging.
#[derive(Clone, Debug)]
pub struct SourceSnapshot {
    /// Records returned by the refresh operation.
    pub records: Vec<DataRecord>,
    /// Next cursor to pass into a future refresh call.
    pub cursor: SourceCursor,
}

/// Sampler-facing data source interface.
///
/// Implementations may be streaming or index-backed. For a fixed dataset state
/// and cursor, refresh output should be deterministic.
pub trait DataSource: Send + Sync {
    /// Stable source identifier used in records, metrics, and persistence state.
    fn id(&self) -> &str;
    /// Fetch up to `limit` records starting from `cursor` state.
    ///
    /// Return the next cursor position in `SourceSnapshot.cursor`.
    fn refresh(
        &self,
        config: &SamplerConfig,
        cursor: Option<&SourceCursor>,
        limit: Option<usize>,
    ) -> Result<SourceSnapshot, SamplerError>;

    /// Exact metadata record count reported by the source.
    ///
    /// This is intended for estimators that must avoid iterating records.
    /// Implementations should return `Ok(count)` only when the count is
    /// exact for the source scope. Return `Err` when exact counting is not
    /// possible or the source is unavailable.
    ///
    /// Keep this consistent with `refresh` by using the same backend scope,
    /// filtering, and logical corpus definition.
    fn reported_record_count(&self, config: &SamplerConfig) -> Result<u128, SamplerError>;

    /// Optional source-provided default triplet recipes.
    ///
    /// Used when sampler config does not provide explicit recipes.
    fn default_triplet_recipes(&self) -> Vec<TripletRecipe> {
        Vec::new()
    }
}

/// Index-addressable source interface used by deterministic pagers.
///
/// `len_hint` must be stable within an epoch, and `record_at` must return the
/// record corresponding to the same index across runs.
///
/// Dense indexing is strongly recommended: implement indices as `0..len_hint`
/// with minimal gaps. Sparse indexes (returning `None` for many positions)
/// still work but waste paging capacity and reduce batch fill rates.
pub trait IndexableSource: Send + Sync {
    /// Stable source identifier.
    fn id(&self) -> &str;
    /// Current index domain size, typically `Some(total_records)`.
    fn len_hint(&self) -> Option<usize>;
    /// Return the record at index `idx`, or `None` for sparse/missing positions.
    fn record_at(&self, idx: usize) -> Result<Option<DataRecord>, SamplerError>;
}

/// Deterministic pager for `IndexableSource`.
///
/// Encapsulates shuffle seed and cursor math so callers can reuse a stable
/// paging algorithm without implementing permutation logic themselves.
pub struct IndexablePager {
    source_id: SourceId,
}

impl IndexablePager {
    /// Create a new deterministic pager for `source_id`.
    pub fn new(source_id: impl Into<SourceId>) -> Self {
        Self {
            source_id: source_id.into(),
        }
    }

    /// Page records from an `IndexableSource` using the provided cursor.
    pub fn refresh(
        &self,
        source: &dyn IndexableSource,
        cursor: Option<&SourceCursor>,
        limit: Option<usize>,
    ) -> Result<SourceSnapshot, SamplerError> {
        let total = source
            .len_hint()
            .ok_or_else(|| SamplerError::SourceInconsistent {
                source_id: source.id().to_string(),
                details: "indexable source did not provide len_hint".into(),
            })?;
        self.refresh_with(total, cursor, limit, |idx| source.record_at(idx))
    }

    /// Page records using a custom index fetcher.
    ///
    /// Useful when records are indexable but not exposed through `IndexableSource`
    /// (for example, temporary index stores or precomputed path lists).
    ///
    /// The fetcher is called concurrently using rayon. It must be `Fn + Send + Sync`
    /// (not merely `FnMut`). All callers that pass a closure over a shared
    /// `&IndexableSource` satisfy this because `record_at` takes `&self`.
    pub fn refresh_with<F>(
        &self,
        total: usize,
        cursor: Option<&SourceCursor>,
        limit: Option<usize>,
        fetch: F,
    ) -> Result<SourceSnapshot, SamplerError>
    where
        F: Fn(usize) -> Result<Option<DataRecord>, SamplerError> + Send + Sync,
    {
        if total == 0 {
            return Ok(SourceSnapshot {
                records: Vec::new(),
                cursor: SourceCursor {
                    last_seen: Utc::now(),
                    revision: 0,
                },
            });
        }
        let mut start = cursor.map(|cursor| cursor.revision as usize).unwrap_or(0);
        if start >= total {
            start = 0;
        }
        let max = limit.unwrap_or(total);
        let seed = Self::seed_for(&self.source_id, total);

        // Pre-generate the full permuted index sequence with per-position cursor
        // values. Pure integer arithmetic — negligible cost vs. record fetch.
        let mut permutation = IndexPermutation::new(total, seed, start as u64);
        let seq: Vec<(usize, usize)> = (0..total)
            .map(|_| {
                let idx = permutation.next();
                (idx, permutation.cursor())
            })
            .collect();

        let should_report = total >= 10_000 || max >= 1_024;
        let refresh_start = Instant::now();
        if should_report {
            eprintln!(
                "[triplets:source] refresh start source='{}' source_records={} ingestion_limit={}",
                self.source_id, total, max
            );
        }

        use rayon::prelude::*;
        // Only process the first `max` entries in parallel. Since almost
        // all index positions return Some, this fills the quota in a single
        // parallel pass. Any residual shortage (from None returns) is
        // handled by a short sequential sweep of the remaining entries.
        let par_end = max.min(total);
        let results: Vec<Result<Option<DataRecord>, SamplerError>> = seq[..par_end]
            .par_iter()
            .map(|&(idx, _)| fetch(idx))
            .collect();
        let mut records = Vec::with_capacity(max.min(total));
        let mut final_cursor = start;
        for (result, &(_, cursor_after)) in results.into_iter().zip(seq[..par_end].iter()) {
            if records.len() >= max {
                break;
            }
            if let Some(r) = result? {
                records.push(r)
            }
            final_cursor = cursor_after;
        }
        // Sequential fallback for any shortage caused by None returns.
        for &(idx, cursor_after) in &seq[par_end..] {
            if records.len() >= max {
                break;
            }
            if let Some(r) = fetch(idx)? {
                records.push(r);
            }
            final_cursor = cursor_after;
        }

        if should_report {
            eprintln!(
                "[triplets:source] refresh done source='{}' source_records={} ingested={} elapsed={:.2}s",
                self.source_id,
                total,
                records.len(),
                refresh_start.elapsed().as_secs_f64()
            );
        }
        let last_seen = records
            .iter()
            .map(|record| record.updated_at)
            .max()
            .unwrap_or_else(Utc::now);
        Ok(SourceSnapshot {
            records,
            cursor: SourceCursor {
                last_seen,
                revision: final_cursor as u64,
            },
        })
    }

    /// Build a deterministic seed for a source and total size.
    pub(crate) fn seed_for(source_id: &SourceId, total: usize) -> u64 {
        Self::stable_index_shuffle_key(source_id, 0)
            ^ Self::stable_index_shuffle_key(source_id, total)
    }

    /// Build a deterministic seed for a source/total pair with explicit sampler seed.
    pub fn seed_for_sampler(source_id: &SourceId, total: usize, sampler_seed: u64) -> u64 {
        Self::seed_for(source_id, total)
            ^ stable_hash_with(|hasher| {
                "triplets_sampler_seed".hash(hasher);
                source_id.hash(hasher);
                total.hash(hasher);
                sampler_seed.hash(hasher);
            })
    }

    fn stable_index_shuffle_key(source_id: &SourceId, idx: usize) -> u64 {
        stable_hash_with(|hasher| {
            source_id.hash(hasher);
            idx.hash(hasher);
        })
    }
}

/// DataSource adapter that pages an `IndexableSource` via `IndexablePager`.
pub struct IndexableAdapter<T: IndexableSource> {
    inner: T,
}

impl<T: IndexableSource> IndexableAdapter<T> {
    /// Wrap an `IndexableSource` so it can be registered as a `DataSource`.
    pub fn new(inner: T) -> Self {
        Self { inner }
    }
}

impl<T: IndexableSource> DataSource for IndexableAdapter<T> {
    fn id(&self) -> &str {
        self.inner.id()
    }

    fn refresh(
        &self,
        _config: &SamplerConfig,
        cursor: Option<&SourceCursor>,
        limit: Option<usize>,
    ) -> Result<SourceSnapshot, SamplerError> {
        let pager = IndexablePager::new(self.inner.id());
        pager.refresh(&self.inner, cursor, limit)
    }

    fn reported_record_count(&self, _config: &SamplerConfig) -> Result<u128, SamplerError> {
        self.inner
            .len_hint()
            .map(|value| value as u128)
            .ok_or_else(|| SamplerError::SourceInconsistent {
                source_id: self.inner.id().to_string(),
                details: "indexable source did not provide len_hint".into(),
            })
    }
}

/// Internal permutation used by `IndexablePager`.
pub struct IndexPermutation {
    total: u64,
    domain_bits: u32,
    domain_size: u64,
    seed: u64,
    counter: u64,
}

impl IndexPermutation {
    /// Creates a new deterministic permutation over `[0, total)`.
    pub fn new(total: usize, seed: u64, counter: u64) -> Self {
        let total_u64 = total as u64;
        let domain_bits = (64 - (total_u64 - 1).leading_zeros()).max(1);
        let domain_size = 1u64 << domain_bits;
        Self {
            total: total_u64,
            domain_bits,
            domain_size,
            seed,
            counter,
        }
    }

    /// Returns the next permuted index, staying within `[0, total)`.
    ///
    /// Each call advances the internal counter and returns a deterministic
    /// pseudo-random index that is guaranteed to be less than `total`.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> usize {
        loop {
            let v =
                Self::permute_bits(self.counter % self.domain_size, self.domain_bits, self.seed);
            self.counter = self.counter.wrapping_add(1);
            if v < self.total {
                return v as usize;
            }
        }
    }

    /// Current consumption cursor position.
    pub fn cursor(&self) -> usize {
        (self.counter as usize) % (self.total as usize)
    }
    fn permute_bits(value: u64, bits: u32, seed: u64) -> u64 {
        if bits == 0 {
            return 0;
        }
        let mask = if bits == 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        let mut a = (seed | 1) & mask;
        if a == 0 {
            a = 1;
        }
        let b = (seed >> 1) & mask;
        a.wrapping_mul(value).wrapping_add(b) & mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{QualityScore, RecordSection, SectionRole};
    use crate::types::RecordId;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::thread;
    use std::time::Duration as StdDuration;

    /// Minimal `IndexableSource` test fixture.
    struct IndexableStub {
        id: SourceId,
        count: usize,
    }

    struct NoLenHintStub {
        id: SourceId,
    }

    impl IndexableStub {
        fn new(id: &str, count: usize) -> Self {
            Self {
                id: id.to_string(),
                count,
            }
        }
    }

    impl NoLenHintStub {
        fn new(id: &str) -> Self {
            Self { id: id.to_string() }
        }
    }

    impl IndexableSource for IndexableStub {
        fn id(&self) -> &str {
            &self.id
        }

        fn len_hint(&self) -> Option<usize> {
            Some(self.count)
        }

        fn record_at(&self, idx: usize) -> Result<Option<DataRecord>, SamplerError> {
            if idx >= self.count {
                return Ok(None);
            }
            let now = Utc::now();
            Ok(Some(DataRecord {
                id: format!("record_{idx}"),
                source: self.id.clone(),
                created_at: now,
                updated_at: now,
                quality: QualityScore { trust: 1.0 },
                taxonomy: Vec::new(),
                sections: vec![RecordSection {
                    role: SectionRole::Anchor,
                    heading: None,
                    text: "stub".into(),
                    sentences: vec!["stub".into()],
                }],
                meta_prefix: None,
            }))
        }
    }

    impl IndexableSource for NoLenHintStub {
        fn id(&self) -> &str {
            &self.id
        }

        fn len_hint(&self) -> Option<usize> {
            None
        }

        fn record_at(&self, _idx: usize) -> Result<Option<DataRecord>, SamplerError> {
            Ok(None)
        }
    }

    #[test]
    fn indexable_adapter_pages_in_stable_order() {
        let adapter = IndexableAdapter::new(IndexableStub::new("stub", 6));
        let config = SamplerConfig::default();
        let full = adapter.refresh(&config, None, None).unwrap();
        let full_ids: Vec<RecordId> = full.records.into_iter().map(|r| r.id).collect();

        let mut cursor = None;
        let mut paged = Vec::new();
        for _ in 0..3 {
            let snapshot = adapter.refresh(&config, cursor.as_ref(), Some(2)).unwrap();
            cursor = Some(snapshot.cursor);
            paged.extend(snapshot.records.into_iter().map(|r| r.id));
        }
        assert_eq!(paged, full_ids);
    }

    #[test]
    fn indexable_paging_spans_multiple_regimes() {
        // Use a source id whose permutation step is not 1 or -1 mod 2^k,
        // otherwise the sequence would be a simple rotation/reversal.
        let total = 256usize;
        let mask = (1u64 << (64 - (total as u64 - 1).leading_zeros())) - 1;
        let source_id = (0..512)
            .map(|idx| format!("regime_test_{idx}"))
            .find(|id| {
                let seed = IndexablePager::seed_for(id, total);
                let a = (seed | 1) & mask;
                a != 1 && a != mask
            })
            .unwrap();

        // Pull a single page and ensure the indices are spread across the space,
        // which indicates the permutation isn't stuck in a narrow regime.
        let adapter = IndexableAdapter::new(IndexableStub::new(&source_id, total));
        let snapshot = adapter
            .refresh(&SamplerConfig::default(), None, Some(64))
            .unwrap();
        let indices: Vec<usize> = snapshot
            .records
            .into_iter()
            .map(|r| {
                r.id.strip_prefix("record_")
                    .unwrap()
                    .parse::<usize>()
                    .unwrap()
            })
            .collect();
        let min_idx = *indices.iter().min().unwrap();
        let max_idx = *indices.iter().max().unwrap();
        assert!(
            max_idx - min_idx >= total / 2,
            "expected spread across the index space, got min={min_idx} max={max_idx}"
        );
    }

    #[test]
    fn indexable_pager_errors_when_len_hint_missing() {
        let pager = IndexablePager::new("no_len_hint");
        let source = NoLenHintStub::new("no_len_hint");
        let result = pager.refresh(&source, None, Some(3));
        assert!(result.is_err());
    }

    #[test]
    fn indexable_adapter_reported_count_errors_when_len_hint_missing() {
        let adapter = IndexableAdapter::new(NoLenHintStub::new("no_len_hint"));
        let result = adapter.reported_record_count(&SamplerConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn indexable_pager_refresh_with_zero_total_returns_empty_snapshot() {
        let pager = IndexablePager::new("empty");
        let snapshot = pager
            .refresh_with(0, None, Some(4), |_idx| Ok(None))
            .unwrap();
        assert!(snapshot.records.is_empty());
        assert_eq!(snapshot.cursor.revision, 0);
    }

    #[test]
    fn index_permutation_permute_bits_handles_zero_bits_and_zero_seed_path() {
        assert_eq!(IndexPermutation::permute_bits(123, 0, 99), 0);

        let bits = 1;
        let value = 1;
        let out = IndexPermutation::permute_bits(value, bits, 0);
        assert!(out <= 1);
    }

    #[test]
    fn index_permutation_next_stays_within_total_and_cursor_advances() {
        let mut perm = IndexPermutation::new(3, 7, 0);
        let mut seen = Vec::new();
        for _ in 0..8 {
            seen.push(perm.next());
        }
        assert!(seen.iter().all(|idx| *idx < 3));
        assert!(perm.cursor() < 3);
    }

    #[test]
    fn indexable_pager_large_refresh_triggers_reporting_branch_and_wraps_cursor() {
        let pager = IndexablePager::new("reporting");
        let cursor = SourceCursor {
            last_seen: Utc::now(),
            revision: 20_000,
        };
        let snapshot = pager
            .refresh_with(10_000, Some(&cursor), Some(4), |idx| {
                Ok(Some(DataRecord {
                    id: format!("record_{idx}"),
                    source: "reporting".to_string(),
                    created_at: Utc::now(),
                    updated_at: Utc::now(),
                    quality: QualityScore { trust: 1.0 },
                    taxonomy: Vec::new(),
                    sections: vec![RecordSection {
                        role: SectionRole::Anchor,
                        heading: None,
                        text: "t".to_string(),
                        sentences: vec!["t".to_string()],
                    }],
                    meta_prefix: None,
                }))
            })
            .unwrap();

        assert_eq!(snapshot.records.len(), 4);
        assert!(snapshot.cursor.revision < 10_000);
    }

    #[test]
    fn indexable_pager_reporting_branch_emits_progress_when_refresh_is_slow() {
        let pager = IndexablePager::new("slow_reporting");
        let slept = AtomicBool::new(false);
        let snapshot = pager
            .refresh_with(2_000, None, Some(1_024), |_idx| {
                if !slept.swap(true, Ordering::Relaxed) {
                    thread::sleep(StdDuration::from_millis(800));
                }
                Ok(None)
            })
            .unwrap();

        assert!(snapshot.records.is_empty());
        assert!(snapshot.cursor.revision < 2_000);
    }

    #[test]
    fn source_ids_and_reported_counts_are_exposed() {
        let adapter = IndexableAdapter::new(IndexableStub::new("stub_id", 3));
        assert_eq!(adapter.id(), "stub_id");
        assert_eq!(
            adapter
                .reported_record_count(&SamplerConfig::default())
                .unwrap(),
            3
        );
    }

    #[test]
    fn indexable_pager_sequential_fallback_fills_quota_when_parallel_pass_yields_none() {
        // Exercise the seq[par_end..] fallback loop: parallel pass entries all
        // return None, so the sequential sweep has to supply the records.
        // total=8, limit=4 -> par_end=4. First 4 calls (parallel) get None;
        // next 4 calls (sequential fallback) get Some, filling the quota.
        let pager = IndexablePager::new("fallback_fill");
        let call_count = AtomicUsize::new(0);
        let par_end = 4usize;
        let snapshot = pager
            .refresh_with(8, None, Some(par_end), |idx| {
                let n = call_count.fetch_add(1, Ordering::Relaxed);
                if n < par_end {
                    Ok(None)
                } else {
                    Ok(Some(DataRecord {
                        id: format!("r_{idx}"),
                        source: "fallback_fill".to_string(),
                        created_at: Utc::now(),
                        updated_at: Utc::now(),
                        quality: QualityScore { trust: 1.0 },
                        taxonomy: Vec::new(),
                        sections: vec![RecordSection {
                            role: SectionRole::Anchor,
                            heading: None,
                            text: "t".to_string(),
                            sentences: vec!["t".to_string()],
                        }],
                        meta_prefix: None,
                    }))
                }
            })
            .unwrap();
        assert_eq!(snapshot.records.len(), par_end);
    }

    #[test]
    fn indexable_pager_refresh_with_propagates_fetch_error() {
        let pager = IndexablePager::new("err");
        let err = pager
            .refresh_with(8, None, Some(2), |_idx| {
                Err(SamplerError::SourceUnavailable {
                    source_id: "err".to_string(),
                    reason: "fetch failed".to_string(),
                })
            })
            .unwrap_err();
        assert!(matches!(
            err,
            SamplerError::SourceUnavailable { ref reason, .. } if reason.contains("fetch failed")
        ));
    }

    #[test]
    fn seed_for_sampler_depends_on_sampler_seed() {
        let source_id = "seeded".to_string();
        let base = IndexablePager::seed_for(&source_id, 17);
        let with_a = IndexablePager::seed_for_sampler(&source_id, 17, 1);
        let with_b = IndexablePager::seed_for_sampler(&source_id, 17, 2);
        assert_ne!(with_a, with_b);
        assert_ne!(with_a, base);
    }
}
