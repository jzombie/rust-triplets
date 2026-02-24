//! Data source interfaces and paging helpers.
//!
//! Ownership model:
//! - `DataSource` is the sampler-facing interface that produces batches.
//! - `IndexableSource` exposes stable, index-based access into a corpus.
//! - `IndexablePager` owns the deterministic pseudo-random paging logic and
//!   can page any indexable source without retaining per-record state.

use chrono::{DateTime, Utc};
use std::hash::Hash;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::config::{SamplerConfig, TripletRecipe};
use crate::data::DataRecord;
use crate::errors::SamplerError;
use crate::hash::stable_hash_with;
use crate::types::SourceId;

/// Utility helpers used by source implementations.
pub mod utilities;
/// Source implementation modules.
pub mod sources;
#[cfg(feature = "huggingface")]
pub use sources::huggingface::{HuggingFaceRowSource, HuggingFaceRowsConfig};

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
    fn reported_record_count(&self) -> Result<u128, SamplerError>;

    /// Provide the active sampler configuration to this source.
    ///
    /// Called when the source is registered with a sampler. Sources can use
    /// this to align internal heuristics with runtime sampler settings.
    fn configure_sampler(&self, _config: &SamplerConfig) {}

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
    pub fn refresh_with(
        &self,
        total: usize,
        cursor: Option<&SourceCursor>,
        limit: Option<usize>,
        mut fetch: impl FnMut(usize) -> Result<Option<DataRecord>, SamplerError>,
    ) -> Result<SourceSnapshot, SamplerError> {
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
        let mut records = Vec::new();
        let seed = Self::seed_for(&self.source_id, total);
        let mut permutation = IndexPermutation::new(total, seed, start as u64);
        let report_every = Duration::from_millis(750);
        let refresh_start = Instant::now();
        let mut last_report = refresh_start;
        let mut attempts = 0usize;
        let should_report = total >= 10_000 || max >= 1_024;
        if should_report {
            eprintln!(
                "[triplets:source] refresh start source='{}' total={} target={}",
                self.source_id, total, max
            );
        }
        for _ in 0..total {
            attempts += 1;
            if records.len() >= max {
                break;
            }
            let idx = permutation.next();
            if let Some(record) = fetch(idx)? {
                records.push(record);
            }
            if should_report && last_report.elapsed() >= report_every {
                eprintln!(
                    "[triplets:source] refresh progress source='{}' attempted={}/{} fetched={}/{} elapsed={:.1}s",
                    self.source_id,
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
            eprintln!(
                "[triplets:source] refresh done source='{}' attempted={} fetched={} elapsed={:.2}s",
                self.source_id,
                attempts,
                records.len(),
                refresh_start.elapsed().as_secs_f64()
            );
        }
        let last_seen = records
            .iter()
            .map(|record| record.updated_at)
            .max()
            .unwrap_or_else(Utc::now);
        let next_start = permutation.cursor();
        Ok(SourceSnapshot {
            records,
            cursor: SourceCursor {
                last_seen,
                revision: next_start as u64,
            },
        })
    }

    /// Build a deterministic seed for a source and total size.
    pub(crate) fn seed_for(source_id: &SourceId, total: usize) -> u64 {
        Self::stable_index_shuffle_key(source_id, 0)
            ^ Self::stable_index_shuffle_key(source_id, total)
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
        cursor: Option<&SourceCursor>,
        limit: Option<usize>,
    ) -> Result<SourceSnapshot, SamplerError> {
        let pager = IndexablePager::new(self.inner.id());
        pager.refresh(&self.inner, cursor, limit)
    }

    fn reported_record_count(&self) -> Result<u128, SamplerError> {
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
pub(crate) struct IndexPermutation {
    total: u64,
    domain_bits: u32,
    domain_size: u64,
    seed: u64,
    counter: u64,
}

impl IndexPermutation {
    fn new(total: usize, seed: u64, counter: u64) -> Self {
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

    fn next(&mut self) -> usize {
        loop {
            let v =
                Self::permute_bits(self.counter % self.domain_size, self.domain_bits, self.seed);
            self.counter = self.counter.wrapping_add(1);
            if v < self.total {
                return v as usize;
            }
        }
    }

    fn cursor(&self) -> usize {
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

/// In-memory data source for tests and small datasets.
pub struct InMemorySource {
    id: SourceId,
    records: Arc<Vec<DataRecord>>,
}

impl InMemorySource {
    /// Create an in-memory source from prebuilt records.
    pub fn new(id: impl Into<SourceId>, records: Vec<DataRecord>) -> Self {
        Self {
            id: id.into(),
            records: Arc::new(records),
        }
    }
}

impl DataSource for InMemorySource {
    fn id(&self) -> &str {
        &self.id
    }

    fn refresh(
        &self,
        cursor: Option<&SourceCursor>,
        limit: Option<usize>,
    ) -> Result<SourceSnapshot, SamplerError> {
        let records = &*self.records;
        let total = records.len();
        let mut start = cursor.map(|cursor| cursor.revision as usize).unwrap_or(0);
        if total > 0 && start >= total {
            start = 0;
        }
        let max = limit.unwrap_or(total);
        let mut filtered = Vec::new();
        for idx in 0..total {
            if filtered.len() >= max {
                break;
            }
            let pos = (start + idx) % total;
            filtered.push(records[pos].clone());
        }
        let last_seen = filtered
            .iter()
            .map(|record| record.updated_at)
            .max()
            .unwrap_or_else(Utc::now);
        let next_start = if total == 0 {
            0
        } else {
            (start + filtered.len()) % total
        };
        Ok(SourceSnapshot {
            records: filtered,
            cursor: SourceCursor {
                last_seen,
                revision: next_start as u64,
            },
        })
    }

    fn reported_record_count(&self) -> Result<u128, SamplerError> {
        Ok(self.records.len() as u128)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{QualityScore, RecordSection, SectionRole};
    use crate::types::RecordId;

    /// Minimal `IndexableSource` test fixture.
    struct IndexableStub {
        id: SourceId,
        count: usize,
    }

    impl IndexableStub {
        fn new(id: &str, count: usize) -> Self {
            Self {
                id: id.to_string(),
                count,
            }
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

    #[test]
    fn indexable_adapter_pages_in_stable_order() {
        let adapter = IndexableAdapter::new(IndexableStub::new("stub", 6));
        let full = adapter.refresh(None, None).unwrap();
        let full_ids: Vec<RecordId> = full.records.into_iter().map(|r| r.id).collect();

        let mut cursor = None;
        let mut paged = Vec::new();
        for _ in 0..3 {
            let snapshot = adapter.refresh(cursor.as_ref(), Some(2)).unwrap();
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
        let snapshot = adapter.refresh(None, Some(64)).unwrap();
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
}
