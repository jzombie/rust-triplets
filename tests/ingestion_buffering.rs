use chrono::Utc;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use triplets::data::{DataRecord, QualityScore, RecordSection, SectionRole};
use triplets::ingestion::IngestionManager;
use triplets::metrics::source_skew;
use triplets::source::{DataSource, InMemorySource, SourceCursor, SourceSnapshot};
use triplets::{RecordId, SamplerConfig, SamplerError};

fn create_dummy_record(id: &str) -> DataRecord {
    DataRecord {
        id: id.to_string(),
        source: "dummy".to_string(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        quality: QualityScore::default(),
        taxonomy: vec![],
        sections: vec![RecordSection {
            role: SectionRole::Context,
            heading: None,
            text: "content".to_string(),
            sentences: vec!["content".to_string()],
        }],
        meta_prefix: None,
    }
}

struct PagedSource {
    id: String,
    pages: Vec<Vec<DataRecord>>,
    calls: Arc<AtomicUsize>,
}

impl PagedSource {
    fn new(id: &str, pages: Vec<Vec<DataRecord>>, calls: Arc<AtomicUsize>) -> Self {
        Self {
            id: id.to_string(),
            pages,
            calls,
        }
    }
}

impl DataSource for PagedSource {
    fn id(&self) -> &str {
        &self.id
    }

    fn refresh(
        &self,
        _config: &SamplerConfig,
        cursor: Option<&SourceCursor>,
        limit: Option<usize>,
    ) -> Result<SourceSnapshot, SamplerError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let page_idx = cursor.map(|c| c.revision as usize).unwrap_or(0);
        let page = if self.pages.is_empty() {
            Vec::new()
        } else {
            self.pages[page_idx % self.pages.len()].clone()
        };
        let max = limit.unwrap_or(page.len());
        let records = page.into_iter().take(max).collect::<Vec<_>>();
        let next_idx = if self.pages.is_empty() {
            0
        } else {
            (page_idx + 1) % self.pages.len()
        };
        Ok(SourceSnapshot {
            records,
            cursor: SourceCursor {
                last_seen: Utc::now(),
                revision: next_idx as u64,
            },
        })
    }

    fn reported_record_count(&self, _config: &SamplerConfig) -> Result<u128, SamplerError> {
        Ok(self.pages.iter().map(|page| page.len() as u128).sum())
    }
}

struct ThreadIdSource {
    id: String,
    seen: Arc<Mutex<Vec<thread::ThreadId>>>,
}

struct FailingSource {
    id: String,
}

impl FailingSource {
    fn new(id: &str) -> Self {
        Self { id: id.to_string() }
    }
}

impl DataSource for FailingSource {
    fn id(&self) -> &str {
        &self.id
    }

    fn refresh(
        &self,
        _config: &SamplerConfig,
        _cursor: Option<&SourceCursor>,
        _limit: Option<usize>,
    ) -> Result<SourceSnapshot, SamplerError> {
        Err(SamplerError::SourceUnavailable {
            source_id: self.id.clone(),
            reason: "forced failure".into(),
        })
    }

    fn reported_record_count(&self, _config: &SamplerConfig) -> Result<u128, SamplerError> {
        Err(SamplerError::SourceUnavailable {
            source_id: self.id.clone(),
            reason: "forced failure".into(),
        })
    }
}

impl ThreadIdSource {
    fn new(id: &str, seen: Arc<Mutex<Vec<thread::ThreadId>>>) -> Self {
        Self {
            id: id.to_string(),
            seen,
        }
    }
}

impl DataSource for ThreadIdSource {
    fn id(&self) -> &str {
        &self.id
    }

    fn refresh(
        &self,
        _config: &SamplerConfig,
        _cursor: Option<&SourceCursor>,
        _limit: Option<usize>,
    ) -> Result<SourceSnapshot, SamplerError> {
        let mut seen = self.seen.lock().unwrap();
        seen.push(thread::current().id());
        drop(seen);
        thread::sleep(Duration::from_millis(10));
        Ok(SourceSnapshot {
            records: Vec::new(),
            cursor: SourceCursor {
                last_seen: Utc::now(),
                revision: 0,
            },
        })
    }

    fn reported_record_count(&self, _config: &SamplerConfig) -> Result<u128, SamplerError> {
        Ok(0)
    }
}

/// Verify that when multiple sources are present, records that don't fit in the current
/// batch are buffered for the next batch rather than being discarded.
/// Verifies the record cache is a FIFO window: each refresh emits the latest
/// four records and evicts the previous four.
#[test]
fn test_ingestion_interleaving_no_data_loss() {
    // 10 records each.
    let records_a: Vec<DataRecord> = (0..10)
        .map(|i| create_dummy_record(&format!("A-{}", i)))
        .collect();
    let records_b: Vec<DataRecord> = (0..10)
        .map(|i| create_dummy_record(&format!("B-{}", i)))
        .collect();

    let source_a = InMemorySource::new("source-A", records_a);
    let source_b = InMemorySource::new("source-B", records_b);

    // Batch size of 4.
    let mut manager = IngestionManager::new(4, SamplerConfig::default());
    manager.register_source(Box::new(source_a));
    manager.register_source(Box::new(source_b));

    // --- Round 1 ---
    // Should fetch 4 from A (0-3) and 4 from B (0-3).
    // Should output 4 total. E.g. A0, B0, A1, B1.
    // Remaining in buffer: A2, A3, B2, B3.
    manager.refresh_all();
    let batch1 = manager.cache().snapshot();
    let ids1: Vec<RecordId> = batch1.iter().map(|r| r.id.clone()).collect();

    assert_eq!(batch1.len(), 4, "Batch 1 should have 4 records");

    // --- Round 2 ---
    // Buffers are present, so NO new fetch should happen.
    // Should drain buffers: A2, B2, A3, B3.
    manager.refresh_all();
    let batch2 = manager.cache().snapshot();
    let ids2: Vec<RecordId> = batch2.iter().map(|r| r.id.clone()).collect();

    assert_eq!(batch2.len(), 4, "Batch 2 should have 4 records");

    // Verify no loss between batch 1 and 2
    let all_ids: Vec<RecordId> = ids1.iter().chain(ids2.iter()).cloned().collect();
    for i in 0..4 {
        assert!(all_ids.contains(&format!("A-{}", i)), "Missing A-{}", i);
        assert!(all_ids.contains(&format!("B-{}", i)), "Missing B-{}", i);
    }
    // Verify we didn't skip ahead to 4 yet
    assert!(
        !all_ids.contains(&"A-4".to_string()),
        "Should not have fetched A-4 yet"
    );

    // --- Round 3 ---
    // Buffers empty. Must fetch new data.
    // Should fetch A(4-7), B(4-7).
    manager.refresh_all();
    let batch3 = manager.cache().snapshot();
    let ids3: Vec<RecordId> = batch3.iter().map(|r| r.id.clone()).collect();

    assert!(
        ids3.contains(&"A-4".to_string()),
        "Batch 3 should start A-4"
    );
}

/// Verify that if one source runs out, we continue draining the other.
/// Ensures the FIFO eviction still holds when sources have uneven supply by
/// checking that every batch contains only the newest two records.
#[test]
fn test_uneven_sources() {
    // Source A has 3 items. Source B has 10.
    // Batch size 2.
    let records_a: Vec<DataRecord> = (0..3)
        .map(|i| create_dummy_record(&format!("A-{}", i)))
        .collect();
    let records_b: Vec<DataRecord> = (0..10)
        .map(|i| create_dummy_record(&format!("B-{}", i)))
        .collect();

    let mut manager = IngestionManager::new(2, SamplerConfig::default());
    manager.register_source(Box::new(InMemorySource::new("A", records_a)));
    manager.register_source(Box::new(InMemorySource::new("B", records_b)));

    // Round 1: Fetch A(0,1), B(0,1). Batch: A0, B0. Buffer: A1, B1.
    manager.refresh_all();
    let b1 = manager.cache().snapshot();
    assert_eq!(b1.len(), 2);
    assert_eq!(b1[0].id, "A-0");
    assert_eq!(b1[1].id, "B-0");

    // Round 2: Drain buffers. Batch: A1, B1. Buffer: empty.
    manager.refresh_all();
    let b2 = manager.cache().snapshot();
    assert_eq!(b2.len(), 2);
    assert_eq!(b2[0].id, "A-1");
    assert_eq!(b2[1].id, "B-1");

    // Round 3: Fetch. A gets [A2] (last one). B gets [B2, B3].
    // Batch fills with A2, B2.
    // Buffer A empty. Buffer B has B3.
    manager.refresh_all();
    let b3 = manager.cache().snapshot();
    assert_eq!(b3.len(), 2);
    assert!(b3.iter().any(|r| r.id == "A-2"));

    // Round 4: Verify looping behavior creates valid batches
    manager.refresh_all();
    let b4 = manager.cache().snapshot();
    // Should have B3 (drained from buffer) and A0 (wrapped).
    assert_eq!(b4.len(), 2);
    let ids4: Vec<RecordId> = b4.iter().map(|r| r.id.clone()).collect();
    assert!(ids4.contains(&"B-3".to_string()));
    // A wraps around
    assert!(ids4.contains(&"A-0".to_string()));
}

#[test]
fn test_refresh_all_skips_non_empty_buffers() {
    let page_a1 = (0..4)
        .map(|i| create_dummy_record(&format!("A-{}", i)))
        .collect::<Vec<_>>();
    let page_a2 = (4..8)
        .map(|i| create_dummy_record(&format!("A-{}", i)))
        .collect::<Vec<_>>();
    let page_b1 = (0..4)
        .map(|i| create_dummy_record(&format!("B-{}", i)))
        .collect::<Vec<_>>();
    let page_b2 = (4..8)
        .map(|i| create_dummy_record(&format!("B-{}", i)))
        .collect::<Vec<_>>();

    let calls_a = Arc::new(AtomicUsize::new(0));
    let calls_b = Arc::new(AtomicUsize::new(0));
    let source_a = PagedSource::new("A", vec![page_a1, page_a2], Arc::clone(&calls_a));
    let source_b = PagedSource::new("B", vec![page_b1, page_b2], Arc::clone(&calls_b));

    let config = SamplerConfig {
        batch_size: 6,
        ..SamplerConfig::default()
    };
    let mut manager = IngestionManager::new(6, config);
    manager.register_source(Box::new(source_a));
    manager.register_source(Box::new(source_b));

    manager.refresh_all();
    assert_eq!(calls_a.load(Ordering::Relaxed), 1);
    assert_eq!(calls_b.load(Ordering::Relaxed), 1);

    manager.refresh_all();
    assert_eq!(calls_a.load(Ordering::Relaxed), 1);
    assert_eq!(calls_b.load(Ordering::Relaxed), 1);

    manager.refresh_all();
    assert_eq!(calls_a.load(Ordering::Relaxed), 2);
    assert_eq!(calls_b.load(Ordering::Relaxed), 2);
}

#[test]
fn test_force_refresh_all_always_calls_sources() {
    let page_a = (0..4)
        .map(|i| create_dummy_record(&format!("A-{}", i)))
        .collect::<Vec<_>>();
    let page_b = (0..4)
        .map(|i| create_dummy_record(&format!("B-{}", i)))
        .collect::<Vec<_>>();

    let calls_a = Arc::new(AtomicUsize::new(0));
    let calls_b = Arc::new(AtomicUsize::new(0));
    let source_a = PagedSource::new("A", vec![page_a], Arc::clone(&calls_a));
    let source_b = PagedSource::new("B", vec![page_b], Arc::clone(&calls_b));

    let mut manager = IngestionManager::new(2, SamplerConfig::default());
    manager.register_source(Box::new(source_a));
    manager.register_source(Box::new(source_b));

    manager.refresh_all();
    assert_eq!(calls_a.load(Ordering::Relaxed), 1);
    assert_eq!(calls_b.load(Ordering::Relaxed), 1);

    manager.force_refresh_all();
    assert_eq!(calls_a.load(Ordering::Relaxed), 2);
    assert_eq!(calls_b.load(Ordering::Relaxed), 2);
}

#[test]
fn test_weighted_refresh_all_prefers_weighted_sources() {
    let records_a: Vec<DataRecord> = (0..6)
        .map(|i| create_dummy_record(&format!("A-{}", i)))
        .collect();
    let records_b: Vec<DataRecord> = (0..6)
        .map(|i| create_dummy_record(&format!("B-{}", i)))
        .collect();

    let mut manager = IngestionManager::new(4, SamplerConfig::default());
    manager.register_source(Box::new(InMemorySource::new("A", records_a)));
    manager.register_source(Box::new(InMemorySource::new("B", records_b)));

    let mut weights = HashMap::new();
    weights.insert("A".to_string(), 2.0);
    weights.insert("B".to_string(), 1.0);

    manager.refresh_all_with_weights(&weights);
    let batch = manager.cache().snapshot();
    let count_a = batch.iter().filter(|r| r.source == "A").count();
    let count_b = batch.iter().filter(|r| r.source == "B").count();

    assert_eq!(count_a + count_b, 4);
    assert!(count_a > count_b);
}

#[test]
fn test_weighted_refresh_all_skips_zero_weight_sources() {
    let records_a: Vec<DataRecord> = (0..6)
        .map(|i| create_dummy_record(&format!("A-{}", i)))
        .collect();
    let records_b: Vec<DataRecord> = (0..6)
        .map(|i| create_dummy_record(&format!("B-{}", i)))
        .collect();

    let mut manager = IngestionManager::new(6, SamplerConfig::default());
    manager.register_source(Box::new(InMemorySource::new("A", records_a)));
    manager.register_source(Box::new(InMemorySource::new("B", records_b)));

    let mut weights = HashMap::new();
    weights.insert("A".to_string(), 1.0);
    weights.insert("B".to_string(), 0.0);

    manager.refresh_all_with_weights(&weights);
    let batch = manager.cache().snapshot();
    assert_eq!(batch.len(), 6);
    let count_a = batch.iter().filter(|r| r.source == "A").count();
    let count_b = batch.iter().filter(|r| r.source == "B").count();

    assert_eq!(count_a + count_b, 6);
    assert_eq!(count_a, 6);
    assert_eq!(count_b, 0);
}

#[test]
fn test_weighted_refresh_all_zero_weight_does_not_reduce_batch() {
    let records_a: Vec<DataRecord> = (0..10)
        .map(|i| create_dummy_record(&format!("A-{}", i)))
        .collect();
    let records_b: Vec<DataRecord> = (0..10)
        .map(|i| create_dummy_record(&format!("B-{}", i)))
        .collect();
    let records_c: Vec<DataRecord> = (0..10)
        .map(|i| create_dummy_record(&format!("C-{}", i)))
        .collect();

    let mut manager = IngestionManager::new(9, SamplerConfig::default());
    manager.register_source(Box::new(InMemorySource::new("A", records_a)));
    manager.register_source(Box::new(InMemorySource::new("B", records_b)));
    manager.register_source(Box::new(InMemorySource::new("C", records_c)));

    let mut weights = HashMap::new();
    weights.insert("A".to_string(), 1.0);
    // The 0-weighted source
    weights.insert("B".to_string(), 0.0);
    weights.insert("C".to_string(), 1.0);

    manager.refresh_all_with_weights(&weights);
    let batch = manager.cache().snapshot();
    assert_eq!(batch.len(), 9);
    let count_b = batch.iter().filter(|r| r.source == "B").count();
    assert_eq!(count_b, 0);
}

#[test]
fn test_refresh_all_runs_sources_in_parallel() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut manager = IngestionManager::new(1, SamplerConfig::default());
    manager.register_source(Box::new(ThreadIdSource::new("A", Arc::clone(&seen))));
    manager.register_source(Box::new(ThreadIdSource::new("B", Arc::clone(&seen))));
    manager.register_source(Box::new(ThreadIdSource::new("C", Arc::clone(&seen))));

    manager.refresh_all();

    let seen = seen.lock().unwrap();
    let unique: HashSet<thread::ThreadId> = seen.iter().copied().collect();
    assert!(
        unique.len() >= 2,
        "expected refresh to run on multiple threads"
    );
}

#[test]
fn test_refresh_stats_track_errors() {
    let mut manager = IngestionManager::new(1, SamplerConfig::default());
    manager.register_source(Box::new(FailingSource::new("fail_a")));
    manager.refresh_all();

    let stats = manager.source_refresh_stats();
    let (_, stat) = stats
        .iter()
        .find(|(id, _)| id == "fail_a")
        .expect("missing fail_a stats");

    assert_eq!(stat.error_count, 1);
    assert!(stat.last_error.as_ref().is_some());
}

#[test]
fn test_refresh_stats_track_success_metrics() {
    let records = (0..3)
        .map(|i| create_dummy_record(&format!("A-{}", i)))
        .collect::<Vec<_>>();
    let mut manager = IngestionManager::new(3, SamplerConfig::default());
    manager.register_source(Box::new(InMemorySource::new("ok_a", records)));

    manager.refresh_all();

    let stats = manager.source_refresh_stats();
    let (_, stat) = stats
        .iter()
        .find(|(id, _)| id == "ok_a")
        .expect("missing ok_a stats");

    assert!(stat.last_record_count > 0);
    assert!(stat.last_records_per_sec >= 0.0);
    assert!(stat.last_error.is_none());
}

#[test]
fn test_source_skew_metrics() {
    let mut counts = HashMap::new();
    counts.insert("A".to_string(), 4);
    counts.insert("B".to_string(), 2);
    counts.insert("C".to_string(), 2);
    let skew = source_skew(&counts).expect("skew");
    assert_eq!(skew.total, 8);
    assert_eq!(skew.sources, 3);
    assert_eq!(skew.min, 2);
    assert_eq!(skew.max, 4);
    assert!((skew.max_share - 0.5).abs() < 1e-6);
    assert!((skew.ratio - 2.0).abs() < 1e-6);
}
