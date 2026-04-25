use chrono::Utc;

use crate::config::SamplerConfig;
use crate::data::DataRecord;
use crate::errors::SamplerError;
use crate::source::{DataSource, IndexableSource, SourceCursor, SourceSnapshot};
use crate::types::SourceId;

/// An in-memory data source backed by a `Vec<DataRecord>`.
///
/// Useful for tests, documentation examples, and small corpora that are
/// constructed entirely at runtime without a file or network backend.
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use chrono::Utc;
/// use triplets::{DataRecord, DeterministicSplitStore, InMemorySource, SamplerConfig, SplitRatios, TripletSampler};
/// use triplets::data::{RecordSection, SectionRole};
///
/// let mut source = InMemorySource::new("my_source");
/// source.add_record(DataRecord {
///     id: "rec-0".into(),
///     source: "my_source".into(),
///     created_at: Utc::now(),
///     updated_at: Utc::now(),
///     quality: Default::default(),
///     taxonomy: vec![],
///     sections: vec![RecordSection {
///         role: SectionRole::Context,
///         heading: None,
///         text: "The quick brown fox.".into(),
///         sentences: vec![],
///     }],
///     meta_prefix: None,
/// });
///
/// // InMemorySource implements DataSource directly — no adapter needed.
/// let ratios = SplitRatios { train: 0.8, validation: 0.1, test: 0.1 };
/// let store = Arc::new(DeterministicSplitStore::new(ratios, 42).unwrap());
/// let sampler = TripletSampler::new(SamplerConfig::default(), store);
/// sampler.register_source(Box::new(source));
/// ```
pub struct InMemorySource {
    id: SourceId,
    records: Vec<DataRecord>,
}

impl InMemorySource {
    /// Create a new empty in-memory source with the given stable identifier.
    pub fn new(id: impl Into<SourceId>) -> Self {
        Self {
            id: id.into(),
            records: Vec::new(),
        }
    }

    /// Append a single record to the source.
    pub fn add_record(&mut self, record: DataRecord) {
        self.records.push(record);
    }

    /// Append multiple records to the source.
    pub fn add_records(&mut self, records: impl IntoIterator<Item = DataRecord>) {
        self.records.extend(records);
    }

    /// Create an in-memory source pre-populated with the given records.
    pub fn from_records(id: impl Into<SourceId>, records: Vec<DataRecord>) -> Self {
        Self {
            id: id.into(),
            records,
        }
    }
}

impl IndexableSource for InMemorySource {
    fn id(&self) -> &str {
        &self.id
    }

    fn len_hint(&self) -> Option<usize> {
        Some(self.records.len())
    }

    fn record_at(&self, idx: usize) -> Result<Option<DataRecord>, SamplerError> {
        Ok(self.records.get(idx).cloned())
    }
}

impl DataSource for InMemorySource {
    fn id(&self) -> &str {
        IndexableSource::id(self)
    }

    fn refresh(
        &self,
        _config: &SamplerConfig,
        cursor: Option<&SourceCursor>,
        limit: Option<usize>,
    ) -> Result<SourceSnapshot, SamplerError> {
        let records = &self.records;
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

    fn reported_record_count(&self, _config: &SamplerConfig) -> Result<u128, SamplerError> {
        Ok(self.records.len() as u128)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{QualityScore, RecordSection, SectionRole};
    use chrono::{Duration, Utc};

    fn make_record(id: &str, ts: chrono::DateTime<Utc>) -> DataRecord {
        DataRecord {
            id: id.to_string(),
            source: "mem".to_string(),
            created_at: ts,
            updated_at: ts,
            quality: QualityScore { trust: 1.0 },
            taxonomy: Vec::new(),
            sections: vec![RecordSection {
                role: SectionRole::Anchor,
                heading: None,
                text: id.to_string(),
                sentences: vec![id.to_string()],
            }],
            meta_prefix: None,
        }
    }

    #[test]
    fn in_memory_source_refresh_wraps_cursor_and_uses_latest_timestamp() {
        let now = Utc::now();
        let older = now - Duration::seconds(5);
        let newer = now + Duration::seconds(5);

        let mut source = InMemorySource::new("mem");
        source.add_records([make_record("a", older), make_record("b", newer)]);

        let cursor = SourceCursor {
            last_seen: now,
            revision: 7,
        };

        let snapshot = source
            .refresh(&SamplerConfig::default(), Some(&cursor), Some(1))
            .unwrap();
        assert_eq!(snapshot.records.len(), 1);
        assert_eq!(snapshot.records[0].id, "a");
        assert_eq!(snapshot.cursor.revision, 1);
        assert_eq!(snapshot.cursor.last_seen, older);
    }

    #[test]
    fn source_id_and_reported_count_are_exposed() {
        let memory = InMemorySource::new("mem_id");
        assert_eq!(DataSource::id(&memory), "mem_id");
        assert_eq!(
            memory
                .reported_record_count(&SamplerConfig::default())
                .unwrap(),
            0
        );
    }

    #[test]
    fn add_record_and_add_records_increase_len() {
        let mut source = InMemorySource::new("s");
        assert_eq!(source.len_hint(), Some(0));

        let now = Utc::now();
        source.add_record(make_record("r0", now));
        assert_eq!(source.len_hint(), Some(1));

        source.add_records([make_record("r1", now), make_record("r2", now)]);
        assert_eq!(source.len_hint(), Some(3));
    }

    #[test]
    fn record_at_returns_correct_record_and_none_out_of_bounds() {
        let now = Utc::now();
        let mut source = InMemorySource::new("s");
        source.add_record(make_record("only", now));

        let found = source.record_at(0).unwrap();
        assert_eq!(found.unwrap().id, "only");

        let oob = source.record_at(1).unwrap();
        assert!(oob.is_none());
    }
}
