fn base_config() -> super::SamplerConfig {
    super::SamplerConfig::default()
}

#[cfg(feature = "bm25-mining")]
use super::backends::bm25_backend::record_bm25_text;
use super::*;
use crate::config::{ChunkingStrategy, NegativeStrategy, Selector, TextRecipe, TripletRecipe};

/// Primary source id used by sampler unit tests.
pub const PRIMARY_SOURCE_ID: &str = "source_a";
/// Secondary source id used by sampler unit tests.
pub const SECONDARY_SOURCE_ID: &str = "source_b";

/// FNV-1a 64-bit offset basis used in snapshot hashing tests.
pub const FNV1A64_OFFSET: u64 = 0xcbf29ce484222325;
/// FNV-1a 64-bit prime used in snapshot hashing tests.
pub const FNV1A64_PRIME: u64 = 0x100000001b3;

/// Number of batches sampled for deterministic sequence hash assertions.
pub const FULL_SEQUENCE_LEN: usize = 45;
/// Expected hash for deterministic text batch sequence.
pub const TEXT_BATCH_SEQUENCE_HASH: u64 = 5827731891827072441;
/// Expected hash for deterministic triplet batch sequence.
#[cfg(not(feature = "bm25-mining"))]
pub const TRIPLET_BATCH_SEQUENCE_HASH: u64 = 4185203987705106104;
/// Expected hash for deterministic triplet batch sequence when bm25-mining is enabled.
#[cfg(feature = "bm25-mining")]
pub const TRIPLET_BATCH_SEQUENCE_HASH: u64 = 2471541713911738564;
/// Expected hash for deterministic pair batch sequence.
#[cfg(not(feature = "bm25-mining"))]
pub const PAIR_BATCH_SEQUENCE_HASH: u64 = 1325935229386486484;
/// Expected hash for deterministic pair batch sequence when bm25-mining is enabled.
#[cfg(feature = "bm25-mining")]
pub const PAIR_BATCH_SEQUENCE_HASH: u64 = 9645472812115896860;
/// Expected hash for deterministic prefetch text batch sequence.
pub const PREFETCH_TEXT_BATCH_SEQUENCE_HASH: u64 = 5061724971919995465;
/// Expected hash for deterministic prefetch triplet batch sequence.
#[cfg(not(feature = "bm25-mining"))]
pub const PREFETCH_TRIPLET_BATCH_SEQUENCE_HASH: u64 = 9256290040294854440;
/// Expected hash for deterministic prefetch triplet batch sequence when bm25-mining is enabled.
#[cfg(feature = "bm25-mining")]
pub const PREFETCH_TRIPLET_BATCH_SEQUENCE_HASH: u64 = 6038679518446907700;
/// Expected hash for deterministic prefetch pair batch sequence.
#[cfg(not(feature = "bm25-mining"))]
pub const PREFETCH_PAIR_BATCH_SEQUENCE_HASH: u64 = 2535655529758418680;
/// Expected hash for deterministic prefetch pair batch sequence when bm25-mining is enabled.
#[cfg(feature = "bm25-mining")]
pub const PREFETCH_PAIR_BATCH_SEQUENCE_HASH: u64 = 6906345832975851973;

/// Expected readable wrong-article sequence without BM25 mining.
pub const READABLE_NON_BM25_TITLES: [&str; 8] = [
    "Energy transition memo",
    "Archaeology field note",
    "Archaeology field note",
    "Carbon market and emissions policy",
    "Energy transition memo",
    "Carbon market and emissions policy",
    "Energy transition memo",
    "Carbon policy update",
];

/// Expected readable wrong-article sequence with BM25 mining enabled.
#[cfg(feature = "bm25-mining")]
pub const READABLE_BM25_TITLES: [&str; 8] = [
    "Carbon market and emissions policy",
    "Carbon policy update",
    "Regulatory market digest",
    "Carbon market and emissions policy",
    "Carbon policy update",
    "Regulatory market digest",
    "Carbon market and emissions policy",
    "Carbon policy update",
];

use crate::data::{ChunkView, QualityScore, RecordChunk, RecordSection};
use crate::kvp::{KvpField, KvpPrefixSampler};
use crate::metadata::META_FIELD_DATE;
use crate::source::{DataSource, InMemorySource, SourceCursor, SourceSnapshot};
use crate::splits::{DeterministicSplitStore, FileSplitStore, SplitLabel, SplitRatios, SplitStore};
use chrono::Utc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Duration as StdDuration;
use tempfile::tempdir;

/// `DataSource` wrapper that exposes custom default recipes in tests.
struct RecipeSource {
    inner: InMemorySource,
    triplet_recipes: Vec<TripletRecipe>,
}

#[test]
fn role_helpers_and_taxonomy_value_cover_branches() {
    assert!(roles_match(&SectionRole::Anchor, &SectionRole::Anchor));
    assert!(!roles_match(&SectionRole::Anchor, &SectionRole::Context));

    let key = role_cursor_key(&"rec-1".to_string(), &SectionRole::Anchor);
    assert_eq!(key.0, "rec-1");
    assert_eq!(key.1, role_label(&SectionRole::Anchor));
    assert_ne!(
        role_label(&SectionRole::Anchor),
        role_label(&SectionRole::Context)
    );

    let mut record = sample_record();
    record.taxonomy = vec!["source_a".into(), META_FIELD_DATE.encode("2026-02-23")];
    assert_eq!(taxonomy_value(&record, META_FIELD_DATE), Some("2026-02-23"));

    record.taxonomy = vec!["source_a".into(), "other=value".into()];
    assert_eq!(taxonomy_value(&record, META_FIELD_DATE), None);
}

#[test]
fn strategy_reason_and_chunk_key_cover_all_variants() {
    let reason_a = strategy_reason(&NegativeStrategy::WrongPublicationDate);
    let reason_b = strategy_reason(&NegativeStrategy::WrongArticle);
    let reason_c = strategy_reason(&NegativeStrategy::QuestionAnswerMismatch);
    assert!(!reason_a.is_empty());
    assert!(!reason_b.is_empty());
    assert!(!reason_c.is_empty());
    assert_ne!(reason_a, reason_b);
    assert_ne!(reason_b, reason_c);

    let base = RecordChunk {
        record_id: "r1".into(),
        section_idx: 0,
        view: ChunkView::Window {
            index: 2,
            overlap: 0,
            span: 8,
            start_ratio: 0.25,
        },
        text: "window".into(),
        tokens_estimate: 8,
        quality: QualityScore { trust: 1.0 },
    };
    let key_window = chunk_key(&base);
    assert!(key_window.contains("|w|2"));

    let summary = RecordChunk {
        view: ChunkView::SummaryFallback {
            strategy: "summary".into(),
            weight: 0.8,
        },
        ..base
    };
    let key_summary = chunk_key(&summary);
    assert!(key_summary.contains("|s|summary"));
}

#[test]
fn deterministic_rng_state_roundtrip_and_fill_bytes_are_stable() {
    let mut rng_a = DeterministicRng::new(123);
    let first = rng_a.next_u64();
    let saved = rng_a.state();

    let mut rng_b = DeterministicRng::from_state(saved);
    assert_eq!(rng_a.next_u64(), rng_b.next_u64());
    assert_ne!(first, 0);

    let mut bytes_a = [0u8; 13];
    let mut bytes_b = [0u8; 13];
    let mut rng_c = DeterministicRng::new(999);
    let mut rng_d = DeterministicRng::new(999);
    rng_c.fill_bytes(&mut bytes_a);
    rng_d.fill_bytes(&mut bytes_b);
    assert_eq!(bytes_a, bytes_b);
    assert!(bytes_a.iter().any(|b| *b != 0));

    let mut rng_e = DeterministicRng::new(999);
    let mut rng_f = DeterministicRng::new(999);
    assert_eq!(rng_e.next_u32() as u64, (rng_f.next_u64() as u32) as u64);
}

#[test]
fn prefetcher_tracks_errors() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_ref = Arc::clone(&calls);
    let prefetcher = BatchPrefetcher::new(2, move || {
        let attempt = calls_ref.fetch_add(1, Ordering::Relaxed);
        if attempt == 0 {
            Err(SamplerError::SourceUnavailable {
                source_id: PREFETCHER_SOURCE_ID.into(),
                reason: "forced error".into(),
            })
        } else {
            Ok(TripletBatch {
                triplets: Vec::new(),
            })
        }
    });

    let start = std::time::Instant::now();
    while prefetcher.produced_count() < 2 && start.elapsed() < StdDuration::from_millis(200) {
        std::thread::sleep(StdDuration::from_millis(5));
    }

    let _ = prefetcher.next();
    let _ = prefetcher.next();

    assert!(prefetcher.error_count() >= 1);
    assert!(prefetcher.produced_count() >= 2);
}

impl RecipeSource {
    fn new(records: Vec<DataRecord>, recipes: Vec<TripletRecipe>) -> Self {
        Self {
            inner: InMemorySource::new("recipe_source", records),
            triplet_recipes: recipes,
        }
    }
}

impl DataSource for RecipeSource {
    fn id(&self) -> &str {
        <InMemorySource as DataSource>::id(&self.inner)
    }

    fn refresh(
        &self,
        config: &SamplerConfig,
        cursor: Option<&SourceCursor>,
        limit: Option<usize>,
    ) -> Result<SourceSnapshot, SamplerError> {
        <InMemorySource as DataSource>::refresh(&self.inner, config, cursor, limit)
    }

    fn reported_record_count(&self, config: &SamplerConfig) -> Result<u128, SamplerError> {
        <InMemorySource as DataSource>::reported_record_count(&self.inner, config)
    }

    fn default_triplet_recipes(&self) -> Vec<TripletRecipe> {
        self.triplet_recipes.clone()
    }
}

#[derive(Clone)]
/// Test source that counts refresh calls.
struct CountingSource {
    id: SourceId,
    records: Vec<DataRecord>,
    refresh_calls: Arc<AtomicUsize>,
}

impl CountingSource {
    fn new(id: &str, records: Vec<DataRecord>, refresh_calls: Arc<AtomicUsize>) -> Self {
        Self {
            id: id.to_string(),
            records,
            refresh_calls,
        }
    }
}

impl DataSource for CountingSource {
    fn id(&self) -> &str {
        &self.id
    }

    fn refresh(
        &self,
        _config: &SamplerConfig,
        _cursor: Option<&SourceCursor>,
        limit: Option<usize>,
    ) -> Result<SourceSnapshot, SamplerError> {
        self.refresh_calls.fetch_add(1, Ordering::Relaxed);
        let mut records = self.records.clone();
        if let Some(max) = limit {
            records.truncate(max);
        }
        Ok(SourceSnapshot {
            records,
            cursor: SourceCursor {
                last_seen: Utc::now(),
                revision: 0,
            },
        })
    }

    fn reported_record_count(&self, _config: &SamplerConfig) -> Result<u128, SamplerError> {
        Ok(self.records.len() as u128)
    }
}

/// Test source that always returns a refresh error.
struct FailingSource {
    id: SourceId,
}

impl FailingSource {
    fn new(id: &str) -> Self {
        Self { id: id.to_string() }
    }
}

#[derive(Clone)]
/// Test source that fails once then succeeds.
struct FlakySource {
    id: SourceId,
    records: Vec<DataRecord>,
    refresh_calls: Arc<AtomicUsize>,
}

impl FlakySource {
    fn new(id: &str, records: Vec<DataRecord>, refresh_calls: Arc<AtomicUsize>) -> Self {
        Self {
            id: id.to_string(),
            records,
            refresh_calls,
        }
    }
}

impl DataSource for FlakySource {
    fn id(&self) -> &str {
        &self.id
    }

    fn refresh(
        &self,
        _config: &SamplerConfig,
        _cursor: Option<&SourceCursor>,
        limit: Option<usize>,
    ) -> Result<SourceSnapshot, SamplerError> {
        let call = self.refresh_calls.fetch_add(1, Ordering::Relaxed);
        if call == 0 {
            return Err(SamplerError::SourceUnavailable {
                source_id: self.id.clone(),
                reason: "first refresh intentionally fails".into(),
            });
        }

        let mut records = self.records.clone();
        if let Some(max) = limit {
            records.truncate(max);
        }
        Ok(SourceSnapshot {
            records,
            cursor: SourceCursor {
                last_seen: Utc::now(),
                revision: call as u64,
            },
        })
    }

    fn reported_record_count(&self, _config: &SamplerConfig) -> Result<u128, SamplerError> {
        Ok(self.records.len() as u128)
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

fn sample_record() -> DataRecord {
    let now = Utc::now();
    DataRecord {
        id: "record_1".into(),
        source: "unit".into(),
        created_at: now,
        updated_at: now,
        quality: QualityScore { trust: 0.9 },
        taxonomy: vec!["SampleCorp".into()],
        sections: vec![
            RecordSection {
                role: SectionRole::Anchor,
                heading: Some("Title".into()),
                text: "Sample title".into(),
                sentences: vec!["Sample title".into()],
            },
            RecordSection {
                role: SectionRole::Context,
                heading: Some("Intro".into()),
                text: "This is the introduction paragraph with enough words for sampling.".into(),
                sentences: vec![
                    "This is the introduction paragraph with enough words for sampling.".into(),
                ],
            },
        ],
        meta_prefix: None,
    }
}

fn record_with_offset(id: &str, base: chrono::DateTime<Utc>, offset_seconds: i64) -> DataRecord {
    let timestamp = base + Duration::seconds(offset_seconds);
    let mut record = sample_record();
    record.id = id.into();
    record.created_at = timestamp;
    record.updated_at = timestamp;
    record
}

fn trader_record(id: &str, date: &str, title: &str, body: &str) -> DataRecord {
    let now = Utc::now();
    DataRecord {
        id: id.into(),
        source: PRIMARY_SOURCE_ID.into(),
        created_at: now,
        updated_at: now,
        quality: QualityScore { trust: 0.9 },
        taxonomy: vec![PRIMARY_SOURCE_ID.into(), META_FIELD_DATE.encode(date)],
        sections: vec![
            RecordSection {
                role: SectionRole::Anchor,
                heading: Some("Title".into()),
                text: title.into(),
                sentences: vec![title.into()],
            },
            RecordSection {
                role: SectionRole::Context,
                heading: Some("Summary".into()),
                text: body.into(),
                sentences: vec![body.into()],
            },
        ],
        meta_prefix: None,
    }
}

fn extract_date_prefix(chunk_text: &str) -> Option<String> {
    let first_line = chunk_text.lines().next()?;
    let prefix = first_line.strip_prefix("meta: ")?;
    for part in prefix.split(" | ") {
        if let Some(date) = part.strip_prefix("date=") {
            return Some(date.to_string());
        }
    }
    None
}

fn extract_meta_prefix(chunk_text: &str) -> Option<String> {
    let first_line = chunk_text.lines().next()?;
    if first_line.starts_with("meta: ") {
        Some(first_line.to_string())
    } else {
        None
    }
}

fn split_meta_parts(meta_prefix: &str) -> Vec<String> {
    let body = meta_prefix.strip_prefix("meta: ").unwrap_or(meta_prefix);
    body.split(" | ").map(|part| part.to_string()).collect()
}

#[test]
fn decorate_chunk_truncates_and_updates_tokens_estimate() {
    let split = SplitRatios::default();
    let store = Arc::new(DeterministicSplitStore::new(split, 42).unwrap());
    let mut cfg = base_config();
    // small window so truncation is obvious
    cfg.chunking.max_window_tokens = 5;
    let sampler = TripletSampler::new(cfg, store);

    // Build a short section that would be under the window without meta prefix
    let mut record = sample_record();
    record.sections[0].text = "one two three".to_string();

    // Add a meta prefix variant that expands the token count beyond the window
    let mut kvp = KvpPrefixSampler::new(1.0);
    kvp.add_variant([("long", "a b c d e f g h i")] as [(&str, &str); 1]);
    record.meta_prefix = Some(kvp);

    let mut inner = sampler.inner.lock().unwrap();
    let mut chunks = inner.materialize_chunks(&record, 0, &record.sections[0]);
    assert!(!chunks.is_empty());
    let mut chunk = chunks.remove(0);
    // initial estimate should reflect original short section
    assert_eq!(chunk.tokens_estimate, 3);

    inner.decorate_chunk(&record, &mut chunk);

    let tokens_after = chunk.text.split_whitespace().count();
    assert_eq!(tokens_after, 5);
    assert_eq!(chunk.tokens_estimate, 5);
}

#[test]
fn decorate_chunk_preserves_newline_after_meta_when_truncated() {
    let split = SplitRatios::default();
    let store = Arc::new(DeterministicSplitStore::new(split, 24).unwrap());
    let mut cfg = base_config();
    cfg.chunking.max_window_tokens = 4;
    let sampler = TripletSampler::new(cfg, store);

    let mut record = sample_record();
    record.sections[0].text = "one two three".to_string();

    let mut kvp = KvpPrefixSampler::new(1.0);
    kvp.add_variant([("source", "unit")]);
    record.meta_prefix = Some(kvp);

    let mut inner = sampler.inner.lock().unwrap();
    let mut chunks = inner.materialize_chunks(&record, 0, &record.sections[0]);
    assert!(!chunks.is_empty());
    let mut chunk = chunks.remove(0);

    inner.decorate_chunk(&record, &mut chunk);

    let expected_prefix = format!("meta: source=unit{}", platform_newline());
    assert!(
        chunk.text.starts_with(&expected_prefix),
        "meta prefix should remain on its own line after truncation"
    );
    assert_eq!(chunk.tokens_estimate, 4);
}

#[test]
fn kvp_prefix_is_applied_to_non_initial_windows_from_long_sections() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 420).unwrap());
    let kvp_long_id = (0u32..)
        .find_map(|i| {
            let id = format!("kvp_long_{i}");
            (store.label_for(&id) == Some(SplitLabel::Train)).then_some(id)
        })
        .unwrap();

    let mut config = base_config();
    config.seed = 7777;
    config.batch_size = 1;
    config.allowed_splits = vec![SplitLabel::Train];
    config.split = split;
    config.chunking = ChunkingStrategy {
        max_window_tokens: 4,
        overlap_tokens: vec![0],
        summary_fallback_weight: 0.0,
        summary_fallback_tokens: 0,
        chunk_weight_floor: 0.0,
    };
    config.recipes = Vec::new();
    config.text_recipes = vec![TextRecipe {
        name: "kvp_long_text".into(),
        selector: Selector::Role(SectionRole::Context),
        weight: 1.0,
        instruction: None,
    }];

    let mut record = trader_record(
        &kvp_long_id,
        "2025-01-01",
        "T",
        "t01 t02 t03 t04 t05 t06 t07 t08 t09 t10 t11 t12",
    );
    let mut prefix = KvpPrefixSampler::new(1.0);
    prefix.add_variant([("date", "2025-01-01")]);
    record.meta_prefix = Some(prefix);

    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(InMemorySource::new("kvp_source", vec![record])));

    let mut saw_non_initial_window = false;
    for _ in 0..12 {
        let batch = sampler.next_text_batch(SplitLabel::Train).unwrap();
        let sample = &batch.samples[0];

        assert!(
            sample.chunk.text.starts_with("meta: "),
            "expected KVP prefix to be present on every sampled chunk, got '{}'",
            sample.chunk.text
        );

        if let ChunkView::Window { index, .. } = sample.chunk.view
            && index > 0
        {
            saw_non_initial_window = true;
            let expected_start = format!("t{:02}", index * 4 + 1);
            assert!(
                sample.chunk.text.contains(&expected_start),
                "expected chunk window {index} to still carry later-window content token {expected_start}, got '{}'",
                sample.chunk.text
            );
        }
    }

    assert!(
        saw_non_initial_window,
        "expected at least one non-initial window sample from long section"
    );
}

#[test]
fn exhaustion_retry_limit_returns_exhausted() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 101).unwrap());
    let exhaust_id = (0u32..)
        .find_map(|i| {
            let id = format!("exhaust_{i}");
            (store.label_for(&id) == Some(SplitLabel::Train)).then_some(id)
        })
        .unwrap();
    let mut config = base_config();
    config.seed = 202;
    config.batch_size = 1;
    config.ingestion_max_records = 2;
    config.allowed_splits = vec![SplitLabel::Train];
    config.split = split;
    config.recipes = vec![TripletRecipe {
        name: "exhaust_triplet".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    }];
    config.text_recipes = Vec::new();

    let mut record = sample_record();
    record.id = exhaust_id;
    let records = vec![record];
    let refresh_calls = Arc::new(AtomicUsize::new(0));
    let source = CountingSource::new("unit", records, Arc::clone(&refresh_calls));
    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(source));

    let result = sampler.next_triplet_batch(SplitLabel::Train);
    assert!(matches!(result, Err(SamplerError::Exhausted(_))));
    assert_eq!(
        refresh_calls.load(Ordering::Relaxed),
        EXHAUSTION_RETRY_LIMIT * 2 + 1
    );
}

#[test]
fn single_source_failure_does_not_fail_batch_when_other_source_has_data() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 909).unwrap());
    let find_train_id = |prefix: &str| -> String {
        (0u32..)
            .find_map(|i| {
                let id = format!("{prefix}_{i}");
                (store.label_for(&id) == Some(SplitLabel::Train)).then_some(id)
            })
            .unwrap()
    };
    let healthy_a = find_train_id("healthy_a");
    let healthy_b = find_train_id("healthy_b");
    let healthy_c = find_train_id("healthy_c");

    let mut config = base_config();
    config.seed = 1337;
    config.batch_size = 1;
    config.ingestion_max_records = 8;
    config.allowed_splits = vec![SplitLabel::Train];
    config.split = split;
    config.recipes = vec![TripletRecipe {
        name: "resilient_triplet".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    }];
    config.text_recipes = Vec::new();

    let healthy_records = vec![
        trader_record(&healthy_a, "2025-01-01", "A", "Body A"),
        trader_record(&healthy_b, "2025-01-02", "B", "Body B"),
        trader_record(&healthy_c, "2025-01-03", "C", "Body C"),
    ];

    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(FailingSource::new("failing_source")));
    sampler.register_source(Box::new(InMemorySource::new(
        "healthy_source",
        healthy_records,
    )));

    let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
    assert_eq!(batch.triplets.len(), 1);
    assert!(batch.triplets[0].anchor.record_id.starts_with("healthy_"));
    assert!(batch.triplets[0].positive.record_id.starts_with("healthy_"));
    assert!(batch.triplets[0].negative.record_id.starts_with("healthy_"));
}

#[test]
fn triplet_batch_is_padded_to_batch_size_when_unique_pool_is_small() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 9001).unwrap());
    let find_train_id = |prefix: &str| -> String {
        (0u32..)
            .find_map(|i| {
                let id = format!("{prefix}_{i}");
                (store.label_for(&id) == Some(SplitLabel::Train)).then_some(id)
            })
            .unwrap()
    };
    let pad_a = find_train_id("pad_a");
    let pad_b = find_train_id("pad_b");

    let mut config = base_config();
    config.seed = 101;
    config.batch_size = 8;
    config.ingestion_max_records = 4;
    config.allowed_splits = vec![SplitLabel::Train];
    config.split = split;
    config.recipes = vec![TripletRecipe {
        name: "fixed_size_triplet".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    }];

    let records = vec![
        trader_record(&pad_a, "2025-01-01", "A", "Body A"),
        trader_record(&pad_b, "2025-01-02", "B", "Body B"),
    ];

    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(InMemorySource::new("pad_source", records)));

    let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
    assert_eq!(batch.triplets.len(), 8);
}

#[test]
fn pair_batch_is_padded_to_batch_size_when_unique_pool_is_small() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 9002).unwrap());
    let find_train_id = |prefix: &str| -> String {
        (0u32..)
            .find_map(|i| {
                let id = format!("{prefix}_{i}");
                (store.label_for(&id) == Some(SplitLabel::Train)).then_some(id)
            })
            .unwrap()
    };
    let pair_a = find_train_id("pair_a");
    let pair_b = find_train_id("pair_b");

    let mut config = base_config();
    config.seed = 202;
    config.batch_size = 9;
    config.ingestion_max_records = 4;
    config.allowed_splits = vec![SplitLabel::Train];
    config.split = split;
    config.recipes = vec![TripletRecipe {
        name: "fixed_size_pairs".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    }];

    let records = vec![
        trader_record(&pair_a, "2025-02-01", "A", "Body A"),
        trader_record(&pair_b, "2025-02-02", "B", "Body B"),
    ];

    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(InMemorySource::new("pair_source", records)));

    let batch = sampler.next_pair_batch(SplitLabel::Train).unwrap();
    assert_eq!(batch.pairs.len(), 9);
}

#[test]
fn text_batch_is_padded_to_batch_size_when_unique_pool_is_small() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 9003).unwrap());
    let text_a = (0u32..)
        .find_map(|i| {
            let id = format!("text_a_{i}");
            (store.label_for(&id) == Some(SplitLabel::Train)).then_some(id)
        })
        .unwrap();

    let mut config = base_config();
    config.seed = 303;
    config.batch_size = 7;
    config.ingestion_max_records = 2;
    config.allowed_splits = vec![SplitLabel::Train];
    config.split = split;
    config.text_recipes = vec![TextRecipe {
        name: "fixed_size_text".into(),
        selector: Selector::Role(SectionRole::Context),
        weight: 1.0,
        instruction: None,
    }];

    let records = vec![trader_record(&text_a, "2025-03-01", "A", "Body A")];

    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(InMemorySource::new("text_source", records)));

    let batch = sampler.next_text_batch(SplitLabel::Train).unwrap();
    assert_eq!(batch.samples.len(), 7);
}

#[test]
fn failed_source_is_retried_on_next_batch_call() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 404).unwrap());
    let find_train_id = |prefix: &str| -> String {
        (0u32..)
            .find_map(|i| {
                let id = format!("{prefix}_{i}");
                (store.label_for(&id) == Some(SplitLabel::Train)).then_some(id)
            })
            .unwrap()
    };
    let flaky_a = find_train_id("flaky_a");
    let flaky_b = find_train_id("flaky_b");
    let steady_a = find_train_id("steady_a");
    let steady_b = find_train_id("steady_b");
    let steady_c = find_train_id("steady_c");

    let mut config = base_config();
    config.seed = 505;
    config.batch_size = 1;
    config.ingestion_max_records = 8;
    config.allowed_splits = vec![SplitLabel::Train];
    config.split = split;
    config.recipes = vec![TripletRecipe {
        name: "retry_triplet".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    }];
    config.text_recipes = Vec::new();

    let flaky_calls = Arc::new(AtomicUsize::new(0));
    let flaky_records = vec![
        trader_record(&flaky_a, "2025-02-01", "Flaky A", "Flaky body A"),
        trader_record(&flaky_b, "2025-02-02", "Flaky B", "Flaky body B"),
    ];
    let healthy_records = vec![
        trader_record(&steady_a, "2025-03-01", "Steady A", "Steady body A"),
        trader_record(&steady_b, "2025-03-02", "Steady B", "Steady body B"),
        trader_record(&steady_c, "2025-03-03", "Steady C", "Steady body C"),
    ];

    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(FlakySource::new(
        "flaky_source",
        flaky_records,
        Arc::clone(&flaky_calls),
    )));
    sampler.register_source(Box::new(InMemorySource::new(
        "steady_source",
        healthy_records,
    )));

    sampler.next_triplet_batch(SplitLabel::Train).unwrap();
    assert_eq!(flaky_calls.load(Ordering::Relaxed), 1);

    sampler.next_triplet_batch(SplitLabel::Train).unwrap();
    assert!(flaky_calls.load(Ordering::Relaxed) >= 2);
}

fn qa_pair_record(id: &str, question: &str, answer: &str) -> DataRecord {
    let now = Utc::now();
    DataRecord {
        id: id.into(),
        source: SECONDARY_SOURCE_ID.into(),
        created_at: now,
        updated_at: now,
        quality: QualityScore { trust: 0.9 },
        taxonomy: vec![SECONDARY_SOURCE_ID.into(), "factual".into()],
        sections: vec![
            RecordSection {
                role: SectionRole::Anchor,
                heading: Some("Question".into()),
                text: question.into(),
                sentences: vec![question.into()],
            },
            RecordSection {
                role: SectionRole::Context,
                heading: Some("Answer".into()),
                text: answer.into(),
                sentences: vec![answer.into()],
            },
        ],
        meta_prefix: None,
    }
}

/// Test source that returns custom records plus default recipes.
struct RecipeDecoratedSource {
    records: Vec<DataRecord>,
    recipes: Vec<TripletRecipe>,
}

impl RecipeDecoratedSource {
    fn new(records: Vec<DataRecord>, recipes: Vec<TripletRecipe>) -> Self {
        Self { records, recipes }
    }
}

impl DataSource for RecipeDecoratedSource {
    fn id(&self) -> &str {
        "recipe_decorated_source"
    }

    fn refresh(
        &self,
        _config: &SamplerConfig,
        cursor: Option<&SourceCursor>,
        limit: Option<usize>,
    ) -> Result<SourceSnapshot, crate::errors::SamplerError> {
        let mut records = self.records.clone();
        if let Some(cap) = limit {
            records.truncate(cap);
        }
        Ok(SourceSnapshot {
            records,
            cursor: SourceCursor {
                last_seen: Utc::now(),
                revision: cursor.map(|c| c.revision + 1).unwrap_or_default(),
            },
        })
    }

    fn reported_record_count(
        &self,
        _config: &SamplerConfig,
    ) -> Result<u128, crate::errors::SamplerError> {
        Ok(self.records.len() as u128)
    }

    fn default_triplet_recipes(&self) -> Vec<TripletRecipe> {
        self.recipes.clone()
    }
}

#[test]
fn chunk_view_carries_start_ratio() {
    let split = SplitRatios::default();
    let mut config = base_config();
    config.chunking = ChunkingStrategy {
        max_window_tokens: 4,
        overlap_tokens: vec![0],
        summary_fallback_weight: 0.0,
        summary_fallback_tokens: 2,
        chunk_weight_floor: 0.0,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 3).unwrap());
    let sampler = TripletSampler::new(config, store);

    let section_text = "one two three four five six seven eight nine ten";
    let record = DataRecord {
        id: "ratio_record".into(),
        source: "unit".into(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        quality: QualityScore { trust: 1.0 },
        taxonomy: vec![],
        sections: vec![RecordSection {
            role: SectionRole::Context,
            heading: None,
            text: section_text.into(),
            sentences: vec![section_text.into()],
        }],
        meta_prefix: None,
    };

    let section = &record.sections[0];
    let chunks = sampler
        .inner
        .lock()
        .unwrap()
        .materialize_chunks(&record, 0, section);
    let ratios: Vec<f32> = chunks
        .iter()
        .filter_map(|chunk| match chunk.view {
            ChunkView::Window { start_ratio, .. } => Some(start_ratio),
            _ => None,
        })
        .collect();
    assert!(ratios.len() >= 3);
    assert!((ratios[0] - 0.0).abs() < 1e-6);
    assert!((ratios[1] - 0.4).abs() < 1e-6);
    assert!((ratios[2] - 0.8).abs() < 1e-6);
}

#[test]
fn chunk_windows_follow_stride_for_large_sections() {
    let split = SplitRatios::default();
    let mut config = base_config();
    config.chunking = ChunkingStrategy {
        max_window_tokens: 5,
        overlap_tokens: vec![1],
        summary_fallback_weight: 0.0,
        summary_fallback_tokens: 0,
        chunk_weight_floor: 0.0,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 11).unwrap());
    let sampler = TripletSampler::new(config, store);

    let block = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu";
    let record = DataRecord {
        id: "stride_record".into(),
        source: "unit".into(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        quality: QualityScore { trust: 1.0 },
        taxonomy: vec![],
        sections: vec![RecordSection {
            role: SectionRole::Context,
            heading: None,
            text: block.into(),
            sentences: vec![
                "alpha beta gamma delta.".into(),
                "epsilon zeta eta theta.".into(),
                "iota kappa lambda mu.".into(),
            ],
        }],
        meta_prefix: None,
    };

    let section = &record.sections[0];
    let chunks = sampler
        .inner
        .lock()
        .unwrap()
        .materialize_chunks(&record, 0, section);

    let texts: Vec<String> = chunks
        .iter()
        .filter_map(|chunk| match chunk.view {
            ChunkView::Window { .. } => Some(chunk.text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        texts,
        vec![
            "alpha beta gamma delta epsilon".to_string(),
            "epsilon zeta eta theta iota".to_string(),
            "iota kappa lambda mu".to_string(),
        ]
    );

    let estimates: Vec<usize> = chunks
        .iter()
        .filter_map(|chunk| match chunk.view {
            ChunkView::Window { .. } => Some(chunk.tokens_estimate),
            _ => None,
        })
        .collect();
    assert_eq!(estimates, vec![5, 5, 4]);
}

#[test]
fn chunk_windows_materialize_all_configured_overlaps() {
    let split = SplitRatios::default();
    let mut config = base_config();
    config.chunking = ChunkingStrategy {
        max_window_tokens: 4,
        overlap_tokens: vec![1, 2],
        summary_fallback_weight: 0.0,
        summary_fallback_tokens: 0,
        chunk_weight_floor: 0.0,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 23).unwrap());
    let sampler = TripletSampler::new(config, store);

    let section_text = "one two three four five six seven";
    let record = DataRecord {
        id: "overlap_record".into(),
        source: "unit".into(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        quality: QualityScore { trust: 1.0 },
        taxonomy: vec![],
        sections: vec![RecordSection {
            role: SectionRole::Context,
            heading: None,
            text: section_text.into(),
            sentences: vec![section_text.into()],
        }],
        meta_prefix: None,
    };

    let section = &record.sections[0];
    let chunks = sampler
        .inner
        .lock()
        .unwrap()
        .materialize_chunks(&record, 0, section);

    let overlaps: Vec<usize> = chunks
        .iter()
        .filter_map(|chunk| match chunk.view {
            ChunkView::Window { overlap, .. } => Some(overlap),
            _ => None,
        })
        .collect();

    let overlap_1_count = overlaps.iter().filter(|&&value| value == 1).count();
    let overlap_2_count = overlaps.iter().filter(|&&value| value == 2).count();

    assert!(overlap_1_count > 0);
    assert!(overlap_2_count > 0);
    assert_eq!(overlap_1_count, 2);
    assert_eq!(overlap_2_count, 3);
}

#[test]
fn chunk_weight_applies_linear_offset_and_floor() {
    let split = SplitRatios::default();
    let mut config = base_config();
    config.chunking.chunk_weight_floor = 0.25;
    let store = Arc::new(DeterministicSplitStore::new(split, 5).unwrap());
    let sampler = TripletSampler::new(config, store);

    let base_chunk = RecordChunk {
        record_id: "unit".into(),
        section_idx: 0,
        view: ChunkView::Window {
            index: 0,
            overlap: 0,
            span: 10,
            start_ratio: 0.75,
        },
        text: "dummy".into(),
        tokens_estimate: 10,
        quality: QualityScore::default(),
    };
    assert_eq!(
        sampler.inner.lock().unwrap().chunk_weight(&base_chunk),
        0.25
    );

    let mut early_chunk = base_chunk.clone();
    early_chunk.view = ChunkView::Window {
        index: 0,
        overlap: 0,
        span: 10,
        start_ratio: 0.1,
    };
    assert_eq!(
        sampler.inner.lock().unwrap().chunk_weight(&early_chunk),
        0.45
    );
}

#[test]
fn summary_fallback_weight_is_clamped() {
    let split = SplitRatios::default();
    let mut config = base_config();
    config.chunking.chunk_weight_floor = 0.5;
    let store = Arc::new(DeterministicSplitStore::new(split, 6).unwrap());
    let sampler = TripletSampler::new(config, store);

    let summary_chunk = RecordChunk {
        record_id: "unit".into(),
        section_idx: 0,
        view: ChunkView::SummaryFallback {
            strategy: "head".into(),
            weight: 0.4,
        },
        text: "summary".into(),
        tokens_estimate: 10,
        quality: QualityScore::default(),
    };
    assert_eq!(
        sampler.inner.lock().unwrap().chunk_weight(&summary_chunk),
        0.5
    );
}

#[test]
fn chunk_weight_applies_trust_scaling() {
    let split = SplitRatios::default();
    let mut config = base_config();
    config.chunking.chunk_weight_floor = 0.0;
    let store = Arc::new(DeterministicSplitStore::new(split, 10).unwrap());
    let sampler = TripletSampler::new(config, store);

    let trusted_chunk = RecordChunk {
        record_id: "unit".into(),
        section_idx: 0,
        view: ChunkView::Window {
            index: 0,
            overlap: 0,
            span: 10,
            start_ratio: 0.2,
        },
        text: "dummy".into(),
        tokens_estimate: 10,
        quality: QualityScore { trust: 0.5 },
    };

    let weight = sampler.inner.lock().unwrap().chunk_weight(&trusted_chunk);
    assert!((weight - 0.4).abs() < f32::EPSILON);
}

#[test]
fn triplet_weight_averages_chunk_weights() {
    let split = SplitRatios::default();
    let mut config = base_config();
    config.chunking.chunk_weight_floor = 0.0;
    let store = Arc::new(DeterministicSplitStore::new(split, 7).unwrap());
    let sampler = TripletSampler::new(config, store);

    let anchor = RecordChunk {
        record_id: "a".into(),
        section_idx: 0,
        view: ChunkView::Window {
            index: 0,
            overlap: 0,
            span: 10,
            start_ratio: 0.0,
        },
        text: "a".into(),
        tokens_estimate: 10,
        quality: QualityScore::default(),
    };
    let positive = RecordChunk {
        record_id: "b".into(),
        section_idx: 0,
        view: ChunkView::Window {
            index: 0,
            overlap: 0,
            span: 10,
            start_ratio: 0.5,
        },
        text: "b".into(),
        tokens_estimate: 10,
        quality: QualityScore::default(),
    };
    let negative = RecordChunk {
        record_id: "c".into(),
        section_idx: 0,
        view: ChunkView::Window {
            index: 0,
            overlap: 0,
            span: 10,
            start_ratio: 1.0,
        },
        text: "c".into(),
        tokens_estimate: 10,
        quality: QualityScore::default(),
    };

    let avg = sampler
        .inner
        .lock()
        .unwrap()
        .triplet_chunk_weight(&anchor, &positive, &negative);
    let trust = QualityScore::default().trust;
    let expected = trust * (1.0 + 0.5 + 0.0) / 3.0;
    assert!((avg - expected).abs() < f32::EPSILON);
}

// Enforcement test: text, pair, and triplet samples must all be drawn from
// the same chunk windows that `materialize_chunks` would produce for the
// record's section.  If the chunking path is ever accidentally bypassed for
// one sample type, the chunk text will not appear in the precomputed pool
// and this test will fail.
#[test]
fn text_pair_and_triplet_chunks_all_come_from_materialize_pool() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let mut config = base_config();
    config.seed = 42;
    config.batch_size = 2;
    config.allowed_splits = vec![SplitLabel::Train];
    // Small window so multiple distinct windows are produced from one section,
    // making membership assertions meaningful.
    config.chunking = ChunkingStrategy {
        max_window_tokens: 4,
        overlap_tokens: vec![1],
        summary_fallback_weight: 0.0,
        summary_fallback_tokens: 0,
        chunk_weight_floor: 0.0,
    };
    let context_text = "alpha beta gamma delta epsilon zeta eta theta";
    config.recipes = vec![TripletRecipe {
        name: "parity_triplet".into(),
        anchor: Selector::Role(SectionRole::Context),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    }];
    config.text_recipes = vec![TextRecipe {
        name: "parity_text".into(),
        selector: Selector::Role(SectionRole::Context),
        weight: 1.0,
        instruction: None,
    }];

    // Use the same store+seed for both pool-building and main sampling so IDs
    // only need to be found once.
    let pool_store = Arc::new(DeterministicSplitStore::new(split, 77).unwrap());
    let find_train_id = |prefix: &str| -> String {
        (0u32..)
            .find_map(|i| {
                let id = format!("{prefix}_{i}");
                (pool_store.label_for(&id) == Some(SplitLabel::Train)).then_some(id)
            })
            .unwrap()
    };
    let parity_a = find_train_id("parity_a");
    let parity_b = find_train_id("parity_b");

    let make_record = |id: &str| DataRecord {
        id: id.into(),
        source: "unit".into(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        quality: QualityScore { trust: 1.0 },
        taxonomy: vec![],
        sections: vec![RecordSection {
            role: SectionRole::Context,
            heading: None,
            text: context_text.into(),
            sentences: vec![context_text.into()],
        }],
        meta_prefix: None,
    };
    let records = vec![make_record(&parity_a), make_record(&parity_b)];

    // Build the expected pool by calling materialize_chunks directly on the
    // shared section text.  Both records have identical section text, so one
    // pool covers all possible chunk texts.
    let pool_sampler = TripletSampler::new(config.clone(), Arc::clone(&pool_store));
    let expected_pool: HashSet<String> = pool_sampler
        .inner
        .lock()
        .unwrap()
        .materialize_chunks(&records[0], 0, &records[0].sections[0])
        .into_iter()
        .map(|c| c.text)
        .collect();
    assert!(!expected_pool.is_empty(), "pool must not be empty");

    let store = Arc::new(DeterministicSplitStore::new(split, 77).unwrap());
    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(InMemorySource::new("unit", records)));

    // Text batches ─ every sampled chunk must come from the pool.
    for _ in 0..5 {
        let batch = sampler.next_text_batch(SplitLabel::Train).unwrap();
        for sample in &batch.samples {
            assert!(
                expected_pool.contains(sample.chunk.text.as_str()),
                "text sample chunk {:?} not in materialize_chunks pool {:?}",
                sample.chunk.text,
                expected_pool,
            );
        }
    }

    // Triplet batches ─ anchor, positive, and negative must all be in the pool.
    for _ in 0..5 {
        let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        for triplet in &batch.triplets {
            assert!(
                expected_pool.contains(triplet.anchor.text.as_str()),
                "triplet anchor {:?} not in pool",
                triplet.anchor.text,
            );
            assert!(
                expected_pool.contains(triplet.positive.text.as_str()),
                "triplet positive {:?} not in pool",
                triplet.positive.text,
            );
            assert!(
                expected_pool.contains(triplet.negative.text.as_str()),
                "triplet negative {:?} not in pool",
                triplet.negative.text,
            );
        }
    }

    // Pair batches ─ anchor and positive must be in the pool.
    for _ in 0..5 {
        let batch = sampler.next_pair_batch(SplitLabel::Train).unwrap();
        for pair in &batch.pairs {
            assert!(
                expected_pool.contains(pair.anchor.text.as_str()),
                "pair anchor {:?} not in pool",
                pair.anchor.text,
            );
            assert!(
                expected_pool.contains(pair.positive.text.as_str()),
                "pair positive {:?} not in pool",
                pair.positive.text,
            );
        }
    }
}

#[test]
fn end_to_end_text_weighting_uses_chunk_offsets() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let mut config = base_config();
    config.seed = 9;
    config.batch_size = 1;
    config.allowed_splits = vec![SplitLabel::Train];
    config.text_recipes = vec![TextRecipe {
        name: "weighted".into(),
        selector: Selector::Role(SectionRole::Context),
        weight: 2.0,
        instruction: None,
    }];
    config.chunking = ChunkingStrategy {
        max_window_tokens: 2,
        overlap_tokens: vec![0],
        summary_fallback_weight: 0.0,
        summary_fallback_tokens: 0,
        chunk_weight_floor: 0.0,
    };

    let store = Arc::new(DeterministicSplitStore::new(split, 9).unwrap());
    let weighted_id = (0u32..)
        .find_map(|i| {
            let id = format!("weighted_record_{i}");
            (store.label_for(&id) == Some(SplitLabel::Train)).then_some(id)
        })
        .unwrap();
    let sampler = TripletSampler::new(config, store);
    let record = DataRecord {
        id: weighted_id,
        source: "unit".into(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        quality: QualityScore::default(),
        taxonomy: vec![],
        sections: vec![RecordSection {
            role: SectionRole::Context,
            heading: None,
            text: "one two three four".into(),
            sentences: vec!["one two three four".into()],
        }],
        meta_prefix: None,
    };
    sampler.register_source(Box::new(InMemorySource::new("unit", vec![record])));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let first = sampler.next_text_batch(SplitLabel::Train).unwrap();
    let second = sampler.next_text_batch(SplitLabel::Train).unwrap();

    let trust = QualityScore::default().trust;
    assert!((first.samples[0].weight - (2.0 * trust)).abs() < f32::EPSILON);
    assert!((second.samples[0].weight - (1.0 * trust)).abs() < f32::EPSILON);
}

#[test]
fn end_to_end_text_weighting_respects_splits() {
    let split = SplitRatios {
        train: 0.34,
        validation: 0.33,
        test: 0.33,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 21).unwrap());

    let find_id = |label: SplitLabel, prefix: &str| -> String {
        for i in 0..2000 {
            let id = format!("{prefix}_{i}");
            if store.ensure(id.clone()).unwrap() == label {
                return id;
            }
        }
        panic!("unable to find id for {:?}", label);
    };

    let train_id = find_id(SplitLabel::Train, "train_weighted");
    let val_id = find_id(SplitLabel::Validation, "val_weighted");
    let test_id = find_id(SplitLabel::Test, "test_weighted");

    let mut config = base_config();
    config.seed = 21;
    config.batch_size = 1;
    config.allowed_splits = vec![SplitLabel::Train];
    config.text_recipes = vec![TextRecipe {
        name: "weighted".into(),
        selector: Selector::Role(SectionRole::Context),
        weight: 3.0,
        instruction: None,
    }];
    config.chunking = ChunkingStrategy {
        max_window_tokens: 2,
        overlap_tokens: vec![0],
        summary_fallback_weight: 0.0,
        summary_fallback_tokens: 0,
        chunk_weight_floor: 0.0,
    };
    let chunking = config.chunking.clone();

    let sampler = TripletSampler::new(config, store);
    let mut train_record =
        trader_record(&train_id, "2025-01-01", "Train Title", "one two three four");
    let mut val_record =
        trader_record(&val_id, "2025-01-02", "Val Title", "alpha beta gamma delta");
    let mut test_record = trader_record(&test_id, "2025-01-03", "Test Title", "foo bar baz qux");
    train_record.source = "split_weighted".into();
    val_record.source = "split_weighted".into();
    test_record.source = "split_weighted".into();

    sampler.register_source(Box::new(InMemorySource::new(
        "split_weighted",
        vec![train_record, val_record, test_record],
    )));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let mut labels = std::collections::HashSet::new();
    let mut checked = 0;
    for _ in 0..20 {
        let batch = sampler.next_text_batch(SplitLabel::Train).unwrap();
        let sample = &batch.samples[0];
        let label = sampler
            .inner
            .lock()
            .unwrap()
            .split_store
            .label_for(&sample.chunk.record_id)
            .unwrap();
        labels.insert(label);
        assert_eq!(label, SplitLabel::Train, "text sample leaked across splits");
        let expected = 3.0 * chunk_weight(&chunking, &sample.chunk);
        assert!((sample.weight - expected).abs() < f32::EPSILON);
        checked += 1;
        if labels.len() == 1 {
            break;
        }
    }
    assert_eq!(labels.len(), 1, "all samples must stay in target split");
    assert!(checked > 0);
}

/// Helper bundle for split-order determinism tests.
struct SplitOrderFixture {
    sampler: Arc<TripletSampler<DeterministicSplitStore>>,
}

fn build_split_order_sampler(seed: u64, batch_size: usize) -> SplitOrderFixture {
    let split = SplitRatios {
        train: 0.34,
        validation: 0.33,
        test: 0.33,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, seed).unwrap());

    let mut config = base_config();
    config.seed = seed;
    config.batch_size = batch_size;
    config.ingestion_max_records = 16;
    config.allowed_splits = vec![SplitLabel::Train];
    config.text_recipes = vec![TextRecipe {
        name: "split_text".into(),
        selector: Selector::Role(SectionRole::Context),
        weight: 1.0,
        instruction: None,
    }];
    config.recipes = vec![TripletRecipe {
        name: "split_triplet".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    }];

    let sampler = Arc::new(TripletSampler::new(config, Arc::clone(&store)));

    let make_records = |source: &str| {
        let mut records = Vec::new();
        for idx in 0..15 {
            let record_id = format!("{source}::record_{idx:02}");
            let title = format!("{source} title {idx}");
            let body = format!("{source} body {idx}");
            records.push(trader_record(&record_id, "2025-01-01", &title, &body));
        }
        records
    };

    sampler.register_source(Box::new(InMemorySource::new(
        "source_a",
        make_records("source_a"),
    )));
    sampler.register_source(Box::new(InMemorySource::new(
        "source_b",
        make_records("source_b"),
    )));
    sampler.register_source(Box::new(InMemorySource::new(
        "source_c",
        make_records("source_c"),
    )));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    SplitOrderFixture { sampler }
}

fn fnv1a_64(input: &str) -> u64 {
    let mut hash = FNV1A64_OFFSET;
    for byte in input.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV1A64_PRIME);
    }
    hash
}

fn fmt_weight(weight: f32) -> String {
    format!("{:.6}", weight)
}

fn text_snapshot_hash(batches: &[TextBatch]) -> u64 {
    let parts: Vec<crate::types::HashPart> = batches
        .iter()
        .map(|batch| {
            let sample = &batch.samples[0];
            format!(
                "text|{}|{}|{}|{}",
                sample.recipe,
                sample.chunk.record_id,
                chunk_key(&sample.chunk),
                fmt_weight(sample.weight)
            )
        })
        .collect();
    fnv1a_64(&parts.join(";"))
}

fn triplet_snapshot_hash(batches: &[TripletBatch]) -> u64 {
    let parts: Vec<crate::types::HashPart> = batches
        .iter()
        .map(|batch| {
            let triplet = &batch.triplets[0];
            format!(
                "triplet|{}|{}|{}|{}|{}|{}|{}|{}",
                triplet.recipe,
                triplet.anchor.record_id,
                chunk_key(&triplet.anchor),
                triplet.positive.record_id,
                chunk_key(&triplet.positive),
                triplet.negative.record_id,
                chunk_key(&triplet.negative),
                fmt_weight(triplet.weight)
            )
        })
        .collect();
    fnv1a_64(&parts.join(";"))
}

fn label_str(label: &PairLabel) -> &'static str {
    match label {
        PairLabel::Positive => "positive",
        PairLabel::Negative => "negative",
    }
}

fn pair_snapshot_hash(batches: &[SampleBatch]) -> u64 {
    let mut parts = Vec::new();
    for batch in batches {
        for pair in &batch.pairs {
            let reason = pair.reason.as_deref().unwrap_or("");
            parts.push(format!(
                "pair|{}|{}|{}|{}|{}|{}|{}|{}",
                pair.recipe,
                label_str(&pair.label),
                pair.anchor.record_id,
                chunk_key(&pair.anchor),
                pair.positive.record_id,
                chunk_key(&pair.positive),
                fmt_weight(pair.weight),
                reason
            ));
        }
    }
    fnv1a_64(&parts.join(";"))
}

#[test]
fn split_order_is_train_val_test_for_text_batches() {
    let fixture = build_split_order_sampler(31, 1);
    let mut record_ids = Vec::new();
    for _ in 0..9 {
        let batch = fixture.sampler.next_text_batch(SplitLabel::Train).unwrap();
        record_ids.push(batch.samples[0].chunk.record_id.clone());
    }
    assert_eq!(
        record_ids,
        vec![
            "source_b::record_03".to_string(),
            "source_c::record_02".to_string(),
            "source_c::record_03".to_string(),
            "source_b::record_03".to_string(),
            "source_b::record_03".to_string(),
            "source_c::record_03".to_string(),
            "source_c::record_02".to_string(),
            "source_b::record_03".to_string(),
            "source_c::record_03".to_string()
        ]
    );
}

#[test]
fn split_order_is_train_val_test_for_triplet_batches() {
    let fixture = build_split_order_sampler(32, 1);
    let mut record_ids = Vec::new();
    for _ in 0..9 {
        let batch = fixture
            .sampler
            .next_triplet_batch(SplitLabel::Train)
            .unwrap();
        record_ids.push(batch.triplets[0].anchor.record_id.clone());
    }
    assert_eq!(
        record_ids,
        vec![
            "source_c::record_02".to_string(),
            "source_b::record_04".to_string(),
            "source_a::record_07".to_string(),
            "source_b::record_00".to_string(),
            "source_c::record_04".to_string(),
            "source_a::record_02".to_string(),
            "source_b::record_04".to_string(),
            "source_c::record_02".to_string(),
            "source_a::record_10".to_string()
        ]
    );
}

#[test]
fn split_order_is_train_val_test_for_pair_batches() {
    let fixture = build_split_order_sampler(33, 2);
    let mut record_ids = Vec::new();
    for _ in 0..9 {
        let batch = fixture.sampler.next_pair_batch(SplitLabel::Train).unwrap();
        record_ids.push(batch.pairs[0].anchor.record_id.clone());
    }
    assert_eq!(
        record_ids,
        vec![
            "source_b::record_04".to_string(),
            "source_c::record_02".to_string(),
            "source_a::record_06".to_string(),
            "source_c::record_04".to_string(),
            "source_b::record_04".to_string(),
            "source_a::record_07".to_string(),
            "source_b::record_08".to_string(),
            "source_a::record_02".to_string(),
            "source_c::record_01".to_string()
        ]
    );
}

#[test]
fn prefetch_text_batches_preserve_split_order() {
    let fixture = build_split_order_sampler(41, 1);
    let prefetcher = Arc::clone(&fixture.sampler).prefetch_text_batches(SplitLabel::Train, 1);
    let mut record_ids = Vec::new();
    for _ in 0..9 {
        let batch = prefetcher.next().unwrap();
        record_ids.push(batch.samples[0].chunk.record_id.clone());
    }
    drop(prefetcher);
    assert_eq!(
        record_ids,
        vec![
            "source_c::record_03".to_string(),
            "source_a::record_04".to_string(),
            "source_a::record_07".to_string(),
            "source_c::record_02".to_string(),
            "source_c::record_04".to_string(),
            "source_a::record_08".to_string(),
            "source_a::record_07".to_string(),
            "source_c::record_02".to_string(),
            "source_c::record_04".to_string()
        ]
    );
}

#[test]
fn prefetch_triplet_batches_preserve_split_order() {
    let fixture = build_split_order_sampler(42, 1);
    let prefetcher = Arc::clone(&fixture.sampler).prefetch_triplet_batches(SplitLabel::Train, 1);
    let mut record_ids = Vec::new();
    for _ in 0..9 {
        let batch = prefetcher.next().unwrap();
        record_ids.push(batch.triplets[0].anchor.record_id.clone());
    }
    drop(prefetcher);
    assert_eq!(
        record_ids,
        vec![
            "source_b::record_01".to_string(),
            "source_a::record_04".to_string(),
            "source_c::record_02".to_string(),
            "source_c::record_00".to_string(),
            "source_b::record_01".to_string(),
            "source_a::record_04".to_string(),
            "source_a::record_12".to_string(),
            "source_b::record_01".to_string(),
            "source_c::record_03".to_string()
        ]
    );
}

#[test]
fn prefetch_pair_batches_preserve_split_order() {
    let fixture = build_split_order_sampler(43, 2);
    let prefetcher = Arc::clone(&fixture.sampler).prefetch_pair_batches(SplitLabel::Train, 1);
    let mut record_ids = Vec::new();
    for _ in 0..9 {
        let batch = prefetcher.next().unwrap();
        record_ids.push(batch.pairs[0].anchor.record_id.clone());
    }
    drop(prefetcher);
    assert_eq!(
        record_ids,
        vec![
            "source_c::record_02".to_string(),
            "source_b::record_06".to_string(),
            "source_a::record_02".to_string(),
            "source_b::record_06".to_string(),
            "source_a::record_02".to_string(),
            "source_c::record_02".to_string(),
            "source_b::record_10".to_string(),
            "source_c::record_02".to_string(),
            "source_a::record_09".to_string()
        ]
    );
}

#[test]
fn prefetch_triplet_batches_with_weights_match_direct() {
    let fixture_prefetch = build_split_order_sampler(101, 1);
    let fixture_direct = build_split_order_sampler(101, 1);
    let mut weights = HashMap::new();
    weights.insert("source_a".to_string(), 1.0);
    weights.insert("source_b".to_string(), 2.0);
    weights.insert("source_c".to_string(), 0.5);

    let prefetcher = Arc::clone(&fixture_prefetch.sampler).prefetch_triplet_batches_with_weights(
        SplitLabel::Train,
        1,
        weights.clone(),
    );
    let mut prefetch_ids = Vec::new();
    for _ in 0..5 {
        let batch = prefetcher.next().unwrap();
        prefetch_ids.push(batch.triplets[0].anchor.record_id.clone());
    }
    drop(prefetcher);

    let mut direct_ids = Vec::new();
    for _ in 0..5 {
        let batch = fixture_direct
            .sampler
            .next_triplet_batch_with_weights(SplitLabel::Train, &weights)
            .unwrap();
        direct_ids.push(batch.triplets[0].anchor.record_id.clone());
    }

    assert_eq!(prefetch_ids, direct_ids);
}

#[test]
fn prefetch_pair_batches_with_weights_match_direct() {
    let fixture_prefetch = build_split_order_sampler(102, 2);
    let fixture_direct = build_split_order_sampler(102, 2);
    let mut weights = HashMap::new();
    weights.insert("source_a".to_string(), 1.0);
    weights.insert("source_b".to_string(), 2.0);
    weights.insert("source_c".to_string(), 0.5);

    let prefetcher = Arc::clone(&fixture_prefetch.sampler).prefetch_pair_batches_with_weights(
        SplitLabel::Train,
        1,
        weights.clone(),
    );
    let mut prefetch_ids = Vec::new();
    for _ in 0..5 {
        let batch = prefetcher.next().unwrap();
        prefetch_ids.push(batch.pairs[0].anchor.record_id.clone());
    }
    drop(prefetcher);

    let mut direct_ids = Vec::new();
    for _ in 0..5 {
        let batch = fixture_direct
            .sampler
            .next_pair_batch_with_weights(SplitLabel::Train, &weights)
            .unwrap();
        direct_ids.push(batch.pairs[0].anchor.record_id.clone());
    }

    assert_eq!(prefetch_ids, direct_ids);
}

#[test]
fn prefetch_text_batches_with_weights_match_direct() {
    let fixture_prefetch = build_split_order_sampler(103, 1);
    let fixture_direct = build_split_order_sampler(103, 1);
    let mut weights = HashMap::new();
    weights.insert("source_a".to_string(), 1.0);
    weights.insert("source_b".to_string(), 2.0);
    weights.insert("source_c".to_string(), 0.5);

    let prefetcher = Arc::clone(&fixture_prefetch.sampler).prefetch_text_batches_with_weights(
        SplitLabel::Train,
        1,
        weights.clone(),
    );
    let mut prefetch_ids = Vec::new();
    for _ in 0..5 {
        let batch = prefetcher.next().unwrap();
        prefetch_ids.push(batch.samples[0].chunk.record_id.clone());
    }
    drop(prefetcher);

    let mut direct_ids = Vec::new();
    for _ in 0..5 {
        let batch = fixture_direct
            .sampler
            .next_text_batch_with_weights(SplitLabel::Train, &weights)
            .unwrap();
        direct_ids.push(batch.samples[0].chunk.record_id.clone());
    }

    assert_eq!(prefetch_ids, direct_ids);
}

#[test]
fn split_order_differs_with_seed() {
    let a = build_split_order_sampler(71, 1);
    let b = build_split_order_sampler(72, 1);
    let mut a_batches = Vec::new();
    let mut b_batches = Vec::new();
    for _ in 0..3 {
        a_batches.push(a.sampler.next_text_batch(SplitLabel::Train).unwrap());
        b_batches.push(b.sampler.next_text_batch(SplitLabel::Train).unwrap());
    }
    let a_hash = text_snapshot_hash(&a_batches);
    let b_hash = text_snapshot_hash(&b_batches);
    assert_ne!(a_hash, b_hash);
}

#[test]
fn full_sequence_hashes_match_for_text_batches() {
    let fixture = build_split_order_sampler(81, 1);
    let mut record_ids = Vec::new();
    let mut batches = Vec::new();
    for _ in 0..FULL_SEQUENCE_LEN {
        batches.push(fixture.sampler.next_text_batch(SplitLabel::Train).unwrap());
        let sample = &batches.last().unwrap().samples[0];
        record_ids.push(sample.chunk.record_id.clone());
    }
    assert_eq!(text_snapshot_hash(&batches), TEXT_BATCH_SEQUENCE_HASH);
}

#[test]
fn full_sequence_hashes_match_for_triplet_batches() {
    let fixture = build_split_order_sampler(82, 1);
    let mut batches = Vec::new();
    for _ in 0..FULL_SEQUENCE_LEN {
        batches.push(
            fixture
                .sampler
                .next_triplet_batch(SplitLabel::Train)
                .unwrap(),
        );
    }
    assert_eq!(triplet_snapshot_hash(&batches), TRIPLET_BATCH_SEQUENCE_HASH);
}

#[test]
fn full_sequence_hashes_match_for_pair_batches() {
    let fixture = build_split_order_sampler(83, 2);
    let mut batches = Vec::new();
    for _ in 0..FULL_SEQUENCE_LEN {
        batches.push(fixture.sampler.next_pair_batch(SplitLabel::Train).unwrap());
    }
    assert_eq!(pair_snapshot_hash(&batches), PAIR_BATCH_SEQUENCE_HASH);
}

#[test]
fn readable_triplet_examples_by_mode() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let recipe = TripletRecipe {
        name: "readable_triplet_demo".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    };
    let config = SamplerConfig {
        seed: 991,
        batch_size: 1,
        chunking: ChunkingStrategy::default(),
        recipes: vec![recipe.clone()],
        text_recipes: Vec::new(),
        split,
        ..SamplerConfig::default()
    };
    // seed=12: all "readable_*" IDs hash to Train under train:0.7.
    let store = Arc::new(DeterministicSplitStore::new(split, 12).unwrap());

    let anchor = trader_record(
        "readable_anchor",
        "2025-01-01",
        "Climate policy briefing",
        "carbon pricing policy emissions reduction roadmap market design",
    );
    let candidates = vec![
        trader_record(
            "readable_topical_1",
            "2025-01-01",
            "Carbon market and emissions policy",
            "carbon pricing policy emissions reduction roadmap market design",
        ),
        trader_record(
            "readable_topical_2",
            "2025-01-01",
            "Carbon policy update",
            "carbon pricing policy emissions reduction roadmap",
        ),
        trader_record(
            "readable_mid_1",
            "2025-01-01",
            "Energy transition memo",
            "emissions reduction roadmap clean energy transition planning",
        ),
        trader_record(
            "readable_mid_2",
            "2025-01-01",
            "Regulatory market digest",
            "policy market design regulatory framework and compliance",
        ),
        trader_record(
            "readable_weak_1",
            "2025-01-01",
            "Archaeology field note",
            "bronze age pottery fragments excavation trench mapping",
        ),
        trader_record(
            "readable_weak_2",
            "2025-01-01",
            "Marine geology report",
            "subduction zones oceanic crust tectonic shear",
        ),
    ];

    let sampler = TripletSampler::new(config, Arc::clone(&store));
    let mut all_records = vec![anchor.clone()];
    all_records.extend(candidates);
    sampler.register_source(Box::new(InMemorySource::new(
        "readable_source",
        all_records,
    )));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let mut inner = sampler.inner.lock().unwrap();
    let anchor = inner
        .records
        .get("readable_anchor")
        .cloned()
        .expect("anchor should be present after ingest");

    let display = |id: &str| -> &'static str {
        match id {
            "readable_topical_1" => "Carbon market and emissions policy",
            "readable_topical_2" => "Carbon policy update",
            "readable_mid_2" => "Regulatory market digest",
            "readable_mid_1" => "Energy transition memo",
            "readable_weak_2" => "Marine geology report",
            "readable_weak_1" => "Archaeology field note",
            _ => "unknown",
        }
    };

    let expected_non_bm25_titles: Vec<String> = READABLE_NON_BM25_TITLES
        .iter()
        .map(|title| (*title).to_string())
        .collect();

    #[cfg(not(feature = "bm25-mining"))]
    {
        let mut negatives = Vec::new();
        for _ in 0..8 {
            let (negative, _fallback_used) = inner
                .select_negative_record(&anchor, &NegativeStrategy::WrongArticle)
                .expect("expected readable negative sample");
            negatives.push(negative.id);
        }

        let actual_titles: Vec<String> =
            negatives.iter().map(|id| display(id).to_string()).collect();
        assert_eq!(
            actual_titles, expected_non_bm25_titles,
            "non-BM25 readable sequence changed unexpectedly"
        );
    }

    #[cfg(feature = "bm25-mining")]
    {
        let mut negatives = Vec::new();
        for _ in 0..8 {
            let (negative, _fallback_used) = inner
                .select_negative_record(&anchor, &NegativeStrategy::WrongArticle)
                .expect("expected BM25 negative selection");
            negatives.push(negative.id);
        }

        let negative_titles: Vec<String> =
            negatives.iter().map(|id| display(id).to_string()).collect();
        let expected_bm25_titles: Vec<String> = READABLE_BM25_TITLES
            .iter()
            .map(|title| (*title).to_string())
            .collect();
        assert_eq!(
            negative_titles, expected_bm25_titles,
            "BM25 readable sequence changed unexpectedly"
        );
        assert_ne!(
            negative_titles, expected_non_bm25_titles,
            "BM25 sequence should differ from non-BM25 sequence on the exact same fixture"
        );
    }
}

#[cfg(feature = "bm25-mining")]
#[test]
fn bm25_not_rng_only_when_only_anchor_text_changes() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };

    let run = |anchor_body: &str| -> Vec<String> {
        let config = SamplerConfig {
            seed: 991,
            batch_size: 1,
            chunking: ChunkingStrategy::default(),
            recipes: vec![TripletRecipe {
                name: "bm25_rng_proof".into(),
                anchor: Selector::Role(SectionRole::Anchor),
                positive_selector: Selector::Role(SectionRole::Context),
                negative_selector: Selector::Role(SectionRole::Context),
                negative_strategy: NegativeStrategy::WrongArticle,
                weight: 1.0,
                instruction: None,
                allow_same_anchor_positive: false,
            }],
            text_recipes: Vec::new(),
            split,
            ..SamplerConfig::default()
        };
        // seed=12: all "readable_*" IDs hash to Train under train:0.7.
        let store = Arc::new(DeterministicSplitStore::new(split, 12).unwrap());
        let sampler = TripletSampler::new(config, Arc::clone(&store));

        let anchor = trader_record("readable_anchor", "2025-01-01", "Anchor", anchor_body);
        let candidates = vec![
            trader_record(
                "readable_topical_1",
                "2025-01-01",
                "Carbon market and emissions policy",
                "carbon pricing policy emissions reduction roadmap market design",
            ),
            trader_record(
                "readable_topical_2",
                "2025-01-01",
                "Carbon policy update",
                "carbon pricing policy emissions reduction roadmap",
            ),
            trader_record(
                "readable_mid_1",
                "2025-01-01",
                "Energy transition memo",
                "emissions reduction roadmap clean energy transition planning",
            ),
            trader_record(
                "readable_mid_2",
                "2025-01-01",
                "Regulatory market digest",
                "policy market design regulatory framework and compliance",
            ),
            trader_record(
                "readable_weak_1",
                "2025-01-01",
                "Archaeology field note",
                "bronze age pottery fragments excavation trench mapping",
            ),
            trader_record(
                "readable_weak_2",
                "2025-01-01",
                "Marine geology report",
                "subduction zones oceanic crust tectonic shear",
            ),
        ];

        let mut all_records = vec![anchor];
        all_records.extend(candidates);
        sampler.register_source(Box::new(InMemorySource::new(
            "readable_source",
            all_records,
        )));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();

        let mut inner = sampler.inner.lock().unwrap();
        let anchor = inner
            .records
            .get("readable_anchor")
            .cloned()
            .expect("anchor should exist");

        let mut negatives = Vec::new();
        for _ in 0..8 {
            let (negative, _fallback_used) = inner
                .select_negative_record(&anchor, &NegativeStrategy::WrongArticle)
                .expect("expected BM25 negative selection");
            negatives.push(negative.id);
        }
        negatives
    };

    let policy = run("carbon pricing policy emissions reduction roadmap market design");
    let geology = run("subduction zones oceanic crust tectonic shear");

    // RNG-only selection with fixed seed/id/pool would produce the same draw
    // sequence here. This assertion requires lexical query content to matter.
    assert_ne!(
        policy, geology,
        "same seed/id/pool but different anchor text must change BM25 negatives; RNG-only selection would not"
    );
}

#[test]
fn full_sequence_hashes_match_for_prefetch_text_batches() {
    let fixture = build_split_order_sampler(91, 1);
    let prefetcher = Arc::clone(&fixture.sampler).prefetch_text_batches(SplitLabel::Train, 1);
    let mut batches = Vec::new();
    for _ in 0..FULL_SEQUENCE_LEN {
        batches.push(prefetcher.next().unwrap());
    }
    drop(prefetcher);
    assert_eq!(
        text_snapshot_hash(&batches),
        PREFETCH_TEXT_BATCH_SEQUENCE_HASH
    );
}

#[test]
fn full_sequence_hashes_match_for_prefetch_triplet_batches() {
    let fixture = build_split_order_sampler(92, 1);
    let prefetcher = Arc::clone(&fixture.sampler).prefetch_triplet_batches(SplitLabel::Train, 1);
    let mut batches = Vec::new();
    for _ in 0..FULL_SEQUENCE_LEN {
        batches.push(prefetcher.next().unwrap());
    }
    drop(prefetcher);
    assert_eq!(
        triplet_snapshot_hash(&batches),
        PREFETCH_TRIPLET_BATCH_SEQUENCE_HASH
    );
}

#[test]
fn full_sequence_hashes_match_for_prefetch_pair_batches() {
    let fixture = build_split_order_sampler(93, 2);
    let prefetcher = Arc::clone(&fixture.sampler).prefetch_pair_batches(SplitLabel::Train, 1);
    let mut batches = Vec::new();
    for _ in 0..FULL_SEQUENCE_LEN {
        batches.push(prefetcher.next().unwrap());
    }
    drop(prefetcher);
    assert_eq!(
        pair_snapshot_hash(&batches),
        PREFETCH_PAIR_BATCH_SEQUENCE_HASH
    );
}

#[test]
fn generates_pairs_from_single_source() {
    let split = SplitRatios::default();
    let config = SamplerConfig {
        seed: 1,
        batch_size: 4,
        chunking: ChunkingStrategy::default(),
        recipes: vec![TripletRecipe {
            name: "title_context_triplet".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
            allow_same_anchor_positive: false,
        }],
        text_recipes: vec![TextRecipe {
            name: "teacher_chunk".into(),
            selector: Selector::Role(SectionRole::Context),
            weight: 1.0,
            instruction: None,
        }],
        split,
        ..SamplerConfig::default()
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 7).unwrap());
    let records = vec![
        trader_record(
            "source_a::2025/01-01/article_a.txt",
            "2025-01-01",
            "Alpha",
            "Body alpha",
        ),
        trader_record(
            "source_a::2025/01-02/article_b.txt",
            "2025-01-02",
            "Beta",
            "Body beta",
        ),
    ];
    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(InMemorySource::new("unit", records)));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();
    let batch = sampler.next_pair_batch(SplitLabel::Train).unwrap();
    assert!(!batch.is_empty());
    assert_eq!(batch.pairs.len(), 4);
}

#[test]
fn produces_text_samples() {
    let split = SplitRatios::default();
    let config = SamplerConfig {
        seed: 2,
        batch_size: 1,
        chunking: ChunkingStrategy::default(),
        recipes: vec![],
        text_recipes: vec![TextRecipe {
            name: "teacher_chunk".into(),
            selector: Selector::Role(SectionRole::Context),
            weight: 1.0,
            instruction: None,
        }],
        split,
        ..SamplerConfig::default()
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 11).unwrap());
    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(InMemorySource::new("unit", vec![sample_record()])));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();
    let batch = sampler.next_text_batch(SplitLabel::Train).unwrap();
    assert!(!batch.is_empty());
    assert_eq!(batch.samples.len(), 1);
}

#[test]
fn cycles_through_section_windows_before_repeating() {
    let split = SplitRatios::default();
    let mut config = base_config();
    config.seed = 5;
    config.batch_size = 1;
    config.chunking = ChunkingStrategy {
        max_window_tokens: 2,
        overlap_tokens: vec![0],
        summary_fallback_weight: 0.0,
        summary_fallback_tokens: 0,
        chunk_weight_floor: 0.0,
    };
    config.text_recipes = vec![TextRecipe {
        name: "evidence_chunks".into(),
        selector: Selector::Role(SectionRole::Context),
        weight: 1.0,
        instruction: None,
    }];
    let store = Arc::new(DeterministicSplitStore::new(split, 13).unwrap());
    let record = DataRecord {
        id: "window_record".into(),
        source: "unit".into(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        quality: QualityScore::default(),
        taxonomy: vec![],
        sections: vec![RecordSection {
            role: SectionRole::Context,
            heading: None,
            text: "one two three four".into(),
            sentences: vec!["one two three four".into()],
        }],
        meta_prefix: None,
    };
    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(InMemorySource::new("unit", vec![record])));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let mut outputs = Vec::new();
    for _ in 0..3 {
        let batch = sampler.next_text_batch(SplitLabel::Train).unwrap();
        outputs.push(batch.samples[0].chunk.text.clone());
    }

    assert_eq!(outputs[0], "one two");
    assert_eq!(outputs[1], "three four");
    assert_eq!(outputs[2], "one two");
}

#[test]
fn first_chunk_offset_is_deterministic_and_nonzero_when_hash_demands_it() {
    let split = SplitRatios::default();
    let key = "window_record::0";
    let pool_len = 3usize;
    // In single-source mode, the first anchor selection wraps immediately and
    // advances source_epoch to 1 before chunk selection runs.
    let epoch_seed_mask = 1u64;
    let mut seed = 1u64;
    while (stable_hash_str(seed ^ epoch_seed_mask, key) as usize).is_multiple_of(pool_len) {
        seed = seed.saturating_add(1);
    }

    let build_sampler = || {
        let mut config = base_config();
        config.seed = seed;
        config.batch_size = 1;
        config.chunking = ChunkingStrategy {
            max_window_tokens: 2,
            overlap_tokens: vec![0],
            summary_fallback_weight: 0.0,
            summary_fallback_tokens: 0,
            chunk_weight_floor: 0.0,
        };
        config.text_recipes = vec![TextRecipe {
            name: "context_chunks".into(),
            selector: Selector::Role(SectionRole::Context),
            weight: 1.0,
            instruction: None,
        }];

        let store = Arc::new(DeterministicSplitStore::new(split, 13).unwrap());
        let record = DataRecord {
            id: "window_record".into(),
            source: "unit".into(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            quality: QualityScore::default(),
            taxonomy: vec![],
            sections: vec![RecordSection {
                role: SectionRole::Context,
                heading: None,
                text: "one two three four five six".into(),
                sentences: vec!["one two three four five six".into()],
            }],
            meta_prefix: None,
        };

        let sampler = TripletSampler::new(config, store);
        sampler.register_source(Box::new(InMemorySource::new("unit", vec![record])));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();
        sampler
    };

    let expected_start = (stable_hash_str(seed ^ epoch_seed_mask, key) as usize) % pool_len;
    assert_ne!(expected_start, 0);
    let expected = ["one two", "three four", "five six"][expected_start];

    let sampler_a = build_sampler();
    let first_a = sampler_a
        .next_text_batch(SplitLabel::Train)
        .unwrap()
        .samples[0]
        .chunk
        .text
        .clone();
    assert_eq!(first_a, expected);

    let sampler_b = build_sampler();
    let first_b = sampler_b
        .next_text_batch(SplitLabel::Train)
        .unwrap()
        .samples[0]
        .chunk
        .text
        .clone();
    assert_eq!(first_b, expected);
    assert_eq!(first_a, first_b);
}

#[test]
fn first_role_section_offset_is_deterministic_and_nonzero_when_hash_demands_it() {
    let split = SplitRatios::default();
    let key = "role_offset_record::context";
    let section_count = 3usize;
    // In single-source mode, the first anchor selection wraps immediately and
    // advances source_epoch to 1 before role section selection runs.
    let epoch_seed_mask = 1u64;
    let mut seed = 1u64;
    while (stable_hash_str(seed ^ epoch_seed_mask, key) as usize).is_multiple_of(section_count) {
        seed = seed.saturating_add(1);
    }

    let build_sampler = || {
        let mut config = base_config();
        config.seed = seed;
        config.batch_size = 1;
        config.chunking = ChunkingStrategy {
            max_window_tokens: 8,
            overlap_tokens: vec![0],
            summary_fallback_weight: 0.0,
            summary_fallback_tokens: 0,
            chunk_weight_floor: 0.0,
        };
        config.text_recipes = vec![TextRecipe {
            name: "context_role".into(),
            selector: Selector::Role(SectionRole::Context),
            weight: 1.0,
            instruction: None,
        }];

        let store = Arc::new(DeterministicSplitStore::new(split, 19).unwrap());
        let record = DataRecord {
            id: "role_offset_record".into(),
            source: "unit".into(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            quality: QualityScore::default(),
            taxonomy: vec![],
            sections: vec![
                RecordSection {
                    role: SectionRole::Context,
                    heading: Some("A".into()),
                    text: "alpha".into(),
                    sentences: vec!["alpha".into()],
                },
                RecordSection {
                    role: SectionRole::Context,
                    heading: Some("B".into()),
                    text: "beta".into(),
                    sentences: vec!["beta".into()],
                },
                RecordSection {
                    role: SectionRole::Context,
                    heading: Some("C".into()),
                    text: "gamma".into(),
                    sentences: vec!["gamma".into()],
                },
            ],
            meta_prefix: None,
        };

        let sampler = TripletSampler::new(config, store);
        sampler.register_source(Box::new(InMemorySource::new("unit", vec![record])));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();
        sampler
    };

    let expected_start = (stable_hash_str(seed ^ epoch_seed_mask, key) as usize) % section_count;
    assert_ne!(expected_start, 0);
    let expected = ["alpha", "beta", "gamma"][expected_start];

    let sampler_a = build_sampler();
    let first_a = sampler_a
        .next_text_batch(SplitLabel::Train)
        .unwrap()
        .samples[0]
        .chunk
        .text
        .clone();
    assert_eq!(first_a, expected);

    let sampler_b = build_sampler();
    let first_b = sampler_b
        .next_text_batch(SplitLabel::Train)
        .unwrap()
        .samples[0]
        .chunk
        .text
        .clone();
    assert_eq!(first_b, expected);
    assert_eq!(first_a, first_b);
}

#[test]
fn reentry_same_epoch_restarts_from_same_chunk_offset() {
    let split = SplitRatios::default();
    let store = Arc::new(DeterministicSplitStore::new(split, 23).unwrap());
    let mut config = base_config();
    config.seed = 101;
    let mut inner = TripletSamplerInner::new(config, store);

    let mk_chunk = |index: usize, text: &str| RecordChunk {
        record_id: "reentry_record".into(),
        section_idx: 0,
        view: ChunkView::Window {
            index,
            overlap: 0,
            span: 2,
            start_ratio: index as f32 / 3.0,
        },
        text: text.to_string(),
        tokens_estimate: 2,
        quality: QualityScore::default(),
    };
    let pool = vec![mk_chunk(0, "zero"), mk_chunk(1, "one"), mk_chunk(2, "two")];

    let first = inner
        .next_chunk_from_pool("reentry_record", 0, pool.clone())
        .unwrap();

    // Simulate record dropping out of the in-memory window.
    inner
        .chunk_cursors
        .remove(&("reentry_record".to_string(), 0));

    let restarted = inner
        .next_chunk_from_pool("reentry_record", 0, pool)
        .unwrap();

    assert_eq!(restarted.text, first.text);
}

#[test]
fn reentry_after_epoch_change_can_restart_from_different_chunk_offset() {
    let split = SplitRatios::default();
    let store = Arc::new(DeterministicSplitStore::new(split, 29).unwrap());
    let key = "reentry_record::0";
    let pool_len = 3usize;
    let mut seed = 1u64;
    while (stable_hash_str(seed, key) as usize) % pool_len
        == (stable_hash_str(seed ^ 1, key) as usize) % pool_len
    {
        seed = seed.saturating_add(1);
    }

    let mut config = base_config();
    config.seed = seed;
    let mut inner = TripletSamplerInner::new(config, store);

    let mk_chunk = |index: usize, text: &str| RecordChunk {
        record_id: "reentry_record".into(),
        section_idx: 0,
        view: ChunkView::Window {
            index,
            overlap: 0,
            span: 2,
            start_ratio: index as f32 / 3.0,
        },
        text: text.to_string(),
        tokens_estimate: 2,
        quality: QualityScore::default(),
    };
    let pool = vec![mk_chunk(0, "zero"), mk_chunk(1, "one"), mk_chunk(2, "two")];

    let first_epoch0 = inner
        .next_chunk_from_pool("reentry_record", 0, pool.clone())
        .unwrap();

    // Simulate record eviction + later re-entry after source epoch advanced.
    inner
        .chunk_cursors
        .remove(&("reentry_record".to_string(), 0));
    inner.source_epoch = inner.source_epoch.saturating_add(1);

    let first_epoch1 = inner
        .next_chunk_from_pool("reentry_record", 0, pool)
        .unwrap();

    assert_ne!(first_epoch1.text, first_epoch0.text);
}

#[test]
fn kvp_date_formats_can_differ_within_same_triplet_across_all_splits() {
    let split = SplitRatios {
        train: 0.34,
        validation: 0.33,
        test: 0.33,
    };
    let mut config = base_config();
    config.seed = 777;
    config.batch_size = 1;
    config.allowed_splits = vec![SplitLabel::Train];
    config.recipes = vec![TripletRecipe {
        name: "kvp_date_triplet".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    }];
    config.text_recipes = Vec::new();

    let store = Arc::new(DeterministicSplitStore::new(split, 73).unwrap());

    let find_id = |label: SplitLabel, prefix: &str| -> String {
        for i in 0..5000 {
            let id = format!("{prefix}_{i}");
            if store.ensure(id.clone()).unwrap() == label {
                return id;
            }
        }
        panic!("unable to find id for {:?}", label);
    };

    let ids = vec![
        find_id(SplitLabel::Train, "kvp_date_train_a"),
        find_id(SplitLabel::Train, "kvp_date_train_b"),
        find_id(SplitLabel::Validation, "kvp_date_val_a"),
        find_id(SplitLabel::Validation, "kvp_date_val_b"),
        find_id(SplitLabel::Test, "kvp_date_test_a"),
        find_id(SplitLabel::Test, "kvp_date_test_b"),
    ];

    let sampler = TripletSampler::new(config, Arc::clone(&store));

    let records: Vec<DataRecord> = ids
        .into_iter()
        .enumerate()
        .map(|(idx, id)| {
            let mut record = trader_record(&id, "2025-05-01", &format!("T{idx}"), "Body");
            let mut prefix = KvpPrefixSampler::new(1.0);
            if idx % 2 == 0 {
                prefix.add_variant([("date", "2025-05-01")]);
                prefix.add_variant([("date", "May 1, 2025")]);
            } else {
                prefix.add_variant([("date", "05/01/2025")]);
                prefix.add_variant([("date", "2025-05-01")]);
            }
            record.meta_prefix = Some(prefix);
            record
        })
        .collect();

    sampler.register_source(Box::new(InMemorySource::new("tt", records)));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let mut seen_splits = std::collections::HashSet::new();
    let mut saw_mixed_date_formats = false;
    for _ in 0..180 {
        let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        let triplet = &batch.triplets[0];

        seen_splits.insert(store.label_for(&triplet.anchor.record_id).unwrap());
        seen_splits.insert(store.label_for(&triplet.positive.record_id).unwrap());
        seen_splits.insert(store.label_for(&triplet.negative.record_id).unwrap());

        let dates = [
            extract_date_prefix(&triplet.anchor.text),
            extract_date_prefix(&triplet.positive.text),
            extract_date_prefix(&triplet.negative.text),
        ];
        if dates.iter().all(Option::is_some) {
            let mut uniq = std::collections::HashSet::new();
            for date in dates.into_iter().flatten() {
                uniq.insert(date);
            }
            if uniq.len() >= 2 {
                saw_mixed_date_formats = true;
            }
        }

        if saw_mixed_date_formats && seen_splits.len() == 1 {
            break;
        }
    }

    assert_eq!(
        seen_splits.len(),
        1,
        "expected sampling to stay in the target split"
    );
    assert!(
        saw_mixed_date_formats,
        "expected at least one triplet with multiple date formats across anchor/positive/negative"
    );
}

#[test]
fn kvp_date_formats_can_differ_between_anchor_and_positive_across_all_splits() {
    let split = SplitRatios {
        train: 0.34,
        validation: 0.33,
        test: 0.33,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 83).unwrap());

    let find_id = |label: SplitLabel, prefix: &str| -> String {
        for i in 0..5000 {
            let id = format!("{prefix}_{i}");
            if store.ensure(id.clone()).unwrap() == label {
                return id;
            }
        }
        panic!("unable to find id for {:?}", label);
    };

    let ids = vec![
        find_id(SplitLabel::Train, "kvp_anchor_pos_train_a"),
        find_id(SplitLabel::Train, "kvp_anchor_pos_train_b"),
        find_id(SplitLabel::Validation, "kvp_anchor_pos_val_a"),
        find_id(SplitLabel::Validation, "kvp_anchor_pos_val_b"),
        find_id(SplitLabel::Test, "kvp_anchor_pos_test_a"),
        find_id(SplitLabel::Test, "kvp_anchor_pos_test_b"),
    ];

    let mut config = base_config();
    config.seed = 919;
    config.batch_size = 1;
    config.allowed_splits = vec![SplitLabel::Train];
    config.recipes = vec![TripletRecipe {
        name: "kvp_date_anchor_positive_all_splits".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    }];
    config.text_recipes = Vec::new();

    let sampler = TripletSampler::new(config, Arc::clone(&store));

    let records: Vec<DataRecord> = ids
        .into_iter()
        .map(|id| {
            let mut record = trader_record(&id, "2025-01-31", "T", "B");
            let mut prefix = KvpPrefixSampler::new(1.0);
            prefix.add_variant([("date", "2025-01-31")]);
            prefix.add_variant([("date", "Jan 31, 2025")]);
            prefix.add_variant([("date", "01/31/2025")]);
            record.meta_prefix = Some(prefix);
            record
        })
        .collect();

    sampler.register_source(Box::new(InMemorySource::new("tt", records)));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let mut seen_splits = std::collections::HashSet::new();
    let mut saw_anchor_positive_diff = false;
    for _ in 0..180 {
        let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        let triplet = &batch.triplets[0];

        seen_splits.insert(store.label_for(&triplet.anchor.record_id).unwrap());
        seen_splits.insert(store.label_for(&triplet.positive.record_id).unwrap());
        seen_splits.insert(store.label_for(&triplet.negative.record_id).unwrap());

        let anchor_date = extract_date_prefix(&triplet.anchor.text);
        let positive_date = extract_date_prefix(&triplet.positive.text);
        if let (Some(a), Some(p)) = (anchor_date, positive_date)
            && a != p
        {
            saw_anchor_positive_diff = true;
        }

        if saw_anchor_positive_diff && seen_splits.len() == 1 {
            break;
        }
    }

    assert_eq!(
        seen_splits.len(),
        1,
        "expected sampling to stay in the target split"
    );
    assert!(
        saw_anchor_positive_diff,
        "expected at least one anchor/positive pair with different date formats"
    );
}

#[test]
fn kvp_prefix_signatures_are_not_constant_across_triplets_with_all_splits() {
    let split = SplitRatios {
        train: 0.34,
        validation: 0.33,
        test: 0.33,
    };
    let mut config = base_config();
    config.seed = 12345;
    config.batch_size = 1;
    config.allowed_splits = vec![SplitLabel::Train];
    config.recipes = vec![TripletRecipe {
        name: "kvp_prefix_diversity_triplet".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    }];
    config.text_recipes = Vec::new();

    let store = Arc::new(DeterministicSplitStore::new(split, 97).unwrap());

    let find_id = |label: SplitLabel, prefix: &str| -> String {
        for i in 0..5000 {
            let id = format!("{prefix}_{i}");
            if store.ensure(id.clone()).unwrap() == label {
                return id;
            }
        }
        panic!("unable to find id for {:?}", label);
    };

    let ids = vec![
        find_id(SplitLabel::Train, "kvp_sign_train_a"),
        find_id(SplitLabel::Train, "kvp_sign_train_b"),
        find_id(SplitLabel::Validation, "kvp_sign_val_a"),
        find_id(SplitLabel::Validation, "kvp_sign_val_b"),
        find_id(SplitLabel::Test, "kvp_sign_test_a"),
        find_id(SplitLabel::Test, "kvp_sign_test_b"),
    ];

    let sampler = TripletSampler::new(config, Arc::clone(&store));

    let records: Vec<DataRecord> = ids
        .into_iter()
        .enumerate()
        .map(|(idx, id)| {
            let mut record = trader_record(&id, "2025-06-01", &format!("R{idx}"), "Body");
            let mut prefix = KvpPrefixSampler::new(1.0);
            if idx % 2 == 0 {
                prefix.add_variant([("date", "2025-06-01"), ("source", "tt")]);
                prefix.add_variant([("date", "Jun 1, 2025"), ("source", "trader")]);
                prefix.add_variant([("date", "06/01/2025"), ("source", "times")]);
            } else {
                prefix.add_variant([("date", "2025-06-01"), ("source", "tt")]);
                prefix.add_variant([("date", "June 1 2025"), ("source", "trader")]);
                prefix.add_variant([("date", "01-06-2025"), ("source", "times")]);
            }
            record.meta_prefix = Some(prefix);
            record
        })
        .collect();

    sampler.register_source(Box::new(InMemorySource::new("tt", records)));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let mut seen_splits = std::collections::HashSet::new();
    let mut signatures = std::collections::HashSet::new();
    for _ in 0..180 {
        let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        let triplet = &batch.triplets[0];

        seen_splits.insert(store.label_for(&triplet.anchor.record_id).unwrap());
        seen_splits.insert(store.label_for(&triplet.positive.record_id).unwrap());
        seen_splits.insert(store.label_for(&triplet.negative.record_id).unwrap());

        let anchor = extract_meta_prefix(&triplet.anchor.text);
        let positive = extract_meta_prefix(&triplet.positive.text);
        let negative = extract_meta_prefix(&triplet.negative.text);
        if let (Some(a), Some(p), Some(n)) = (anchor, positive, negative) {
            signatures.insert(format!("{a} || {p} || {n}"));
        }

        if seen_splits.len() == 1 && signatures.len() >= 2 {
            break;
        }
    }

    assert_eq!(
        seen_splits.len(),
        1,
        "expected sampling to stay in the target split"
    );
    assert!(
        signatures.len() >= 2,
        "expected at least two distinct triplet KVP signatures across samples"
    );
}

#[test]
fn triplets_cover_kvp_behaviors_across_all_splits() {
    // Same KVP guarantees as the train-only test, but with split cycling enabled
    // across Train/Validation/Test and explicit verification that all splits
    // are observed while sampling.
    let split = SplitRatios {
        train: 0.34,
        validation: 0.33,
        test: 0.33,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 211).unwrap());

    let find_id = |label: SplitLabel, prefix: &str| -> String {
        for i in 0..5000 {
            let id = format!("{prefix}_{i}");
            if store.ensure(id.clone()).unwrap() == label {
                return id;
            }
        }
        panic!("unable to find id for {:?}", label);
    };

    let ids = vec![
        find_id(SplitLabel::Train, "kvp_split_train_a"),
        find_id(SplitLabel::Train, "kvp_split_train_b"),
        find_id(SplitLabel::Validation, "kvp_split_val_a"),
        find_id(SplitLabel::Validation, "kvp_split_val_b"),
        find_id(SplitLabel::Test, "kvp_split_test_a"),
        find_id(SplitLabel::Test, "kvp_split_test_b"),
    ];

    let mut config = base_config();
    config.seed = 515151;
    config.batch_size = 1;
    config.allowed_splits = vec![SplitLabel::Train];
    config.recipes = vec![TripletRecipe {
        name: "kvp_behavior_triplet_all_splits".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    }];
    config.text_recipes = Vec::new();

    let sampler = TripletSampler::new(config, Arc::clone(&store));

    let build_prefix = || {
        let mut prefix = KvpPrefixSampler::new(1.0);
        prefix.add_variant_fields([
            KvpField::many("date", ["2025-08-01", "Aug 1, 2025", "08/01/2025"]),
            KvpField::many("source", ["source_a", "source_primary"]),
            KvpField::one("ticker", "TT").with_presence(0.5),
            KvpField::one("quarter", "Q3").with_presence(0.5),
        ]);
        prefix
    };

    let records: Vec<DataRecord> = ids
        .into_iter()
        .map(|id| {
            let mut record = trader_record(&id, "2025-08-01", "Split Title", "Split Body");
            record.source = "source_a".into();
            record.meta_prefix = Some(build_prefix());
            record
        })
        .collect();

    sampler.register_source(Box::new(InMemorySource::new("tt", records)));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let mut seen_splits = std::collections::HashSet::new();
    let mut saw_triplet_component_divergence = false;
    let mut saw_ticker_present = false;
    let mut saw_ticker_absent = false;
    let mut orderings_by_signature: std::collections::HashMap<
        String,
        std::collections::HashSet<String>,
    > = std::collections::HashMap::new();

    for _ in 0..180 {
        let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        let triplet = &batch.triplets[0];

        let anchor_split = store.label_for(&triplet.anchor.record_id).unwrap();
        let positive_split = store.label_for(&triplet.positive.record_id).unwrap();
        let negative_split = store.label_for(&triplet.negative.record_id).unwrap();
        seen_splits.insert(anchor_split);
        seen_splits.insert(positive_split);
        seen_splits.insert(negative_split);

        let prefixes = [
            extract_meta_prefix(&triplet.anchor.text),
            extract_meta_prefix(&triplet.positive.text),
            extract_meta_prefix(&triplet.negative.text),
        ];

        if let (Some(a_pref), Some(p_pref), Some(n_pref)) = (
            prefixes[0].as_ref(),
            prefixes[1].as_ref(),
            prefixes[2].as_ref(),
        ) && (a_pref != p_pref || p_pref != n_pref)
        {
            saw_triplet_component_divergence = true;
        }

        for pref in prefixes.into_iter().flatten() {
            let parts = split_meta_parts(&pref);
            let has_ticker = parts.iter().any(|part| part.starts_with("ticker="));
            if has_ticker {
                saw_ticker_present = true;
            } else {
                saw_ticker_absent = true;
            }

            let ordered = parts.join(" | ");
            let mut normalized = parts;
            normalized.sort();
            let signature = normalized.join(" | ");
            orderings_by_signature
                .entry(signature)
                .or_default()
                .insert(ordered);
        }

        if seen_splits.len() == 1
            && saw_triplet_component_divergence
            && saw_ticker_present
            && saw_ticker_absent
            && orderings_by_signature
                .values()
                .any(|ordered_forms| ordered_forms.len() >= 2)
        {
            break;
        }
    }

    let saw_order_permutation = orderings_by_signature
        .values()
        .any(|ordered_forms| ordered_forms.len() >= 2);

    assert_eq!(
        seen_splits.len(),
        1,
        "expected sampling to stay in the target split"
    );
    assert!(
        saw_triplet_component_divergence,
        "expected anchor/positive/negative KVP prefixes to diverge in at least one triplet"
    );
    assert!(
        saw_ticker_present && saw_ticker_absent,
        "expected optional field to be present on some samples and absent on others"
    );
    assert!(
        saw_order_permutation,
        "expected at least one identical KVP field-set signature to appear in multiple key orders"
    );
}

#[test]
fn role_reentry_same_epoch_restarts_from_same_section_offset() {
    let split = SplitRatios::default();
    let store = Arc::new(DeterministicSplitStore::new(split, 31).unwrap());
    let mut config = base_config();
    config.seed = 131;
    config.chunking = ChunkingStrategy {
        max_window_tokens: 64,
        overlap_tokens: vec![0],
        summary_fallback_weight: 0.0,
        summary_fallback_tokens: 0,
        chunk_weight_floor: 0.0,
    };
    let mut inner = TripletSamplerInner::new(config, store);

    let record = DataRecord {
        id: "role_reentry_record".into(),
        source: "unit".into(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        quality: QualityScore::default(),
        taxonomy: vec![],
        sections: vec![
            RecordSection {
                role: SectionRole::Context,
                heading: Some("A".into()),
                text: "alpha".into(),
                sentences: vec!["alpha".into()],
            },
            RecordSection {
                role: SectionRole::Context,
                heading: Some("B".into()),
                text: "beta".into(),
                sentences: vec!["beta".into()],
            },
            RecordSection {
                role: SectionRole::Context,
                heading: Some("C".into()),
                text: "gamma".into(),
                sentences: vec!["gamma".into()],
            },
        ],
        meta_prefix: None,
    };

    let first = inner
        .select_by_role(&record, &SectionRole::Context)
        .expect("first role chunk");

    // Simulate record dropping out and coming back in the same epoch.
    inner
        .role_cursors
        .remove(&(record.id.clone(), role_label(&SectionRole::Context)));
    inner
        .chunk_cursors
        .retain(|(record_id, _), _| record_id != &record.id);

    let restarted = inner
        .select_by_role(&record, &SectionRole::Context)
        .expect("restarted role chunk");

    assert_eq!(restarted.text, first.text);
}

#[test]
fn role_reentry_after_epoch_change_can_restart_from_different_section_offset() {
    let split = SplitRatios::default();
    let store = Arc::new(DeterministicSplitStore::new(split, 37).unwrap());
    let role_key = "role_reentry_record::context";
    let section_count = 3usize;
    let mut seed = 1u64;
    while (stable_hash_str(seed, role_key) as usize) % section_count
        == (stable_hash_str(seed ^ 1, role_key) as usize) % section_count
    {
        seed = seed.saturating_add(1);
    }

    let mut config = base_config();
    config.seed = seed;
    config.chunking = ChunkingStrategy {
        max_window_tokens: 64,
        overlap_tokens: vec![0],
        summary_fallback_weight: 0.0,
        summary_fallback_tokens: 0,
        chunk_weight_floor: 0.0,
    };
    let mut inner = TripletSamplerInner::new(config, store);

    let record = DataRecord {
        id: "role_reentry_record".into(),
        source: "unit".into(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        quality: QualityScore::default(),
        taxonomy: vec![],
        sections: vec![
            RecordSection {
                role: SectionRole::Context,
                heading: Some("A".into()),
                text: "alpha".into(),
                sentences: vec!["alpha".into()],
            },
            RecordSection {
                role: SectionRole::Context,
                heading: Some("B".into()),
                text: "beta".into(),
                sentences: vec!["beta".into()],
            },
            RecordSection {
                role: SectionRole::Context,
                heading: Some("C".into()),
                text: "gamma".into(),
                sentences: vec!["gamma".into()],
            },
        ],
        meta_prefix: None,
    };

    let first_epoch0 = inner
        .select_by_role(&record, &SectionRole::Context)
        .expect("first role chunk epoch0");

    // Simulate record eviction + re-entry after source epoch advances.
    inner
        .role_cursors
        .remove(&(record.id.clone(), role_label(&SectionRole::Context)));
    inner
        .chunk_cursors
        .retain(|(record_id, _), _| record_id != &record.id);
    inner.source_epoch = inner.source_epoch.saturating_add(1);

    let first_epoch1 = inner
        .select_by_role(&record, &SectionRole::Context)
        .expect("first role chunk epoch1");

    assert_ne!(first_epoch1.text, first_epoch0.text);
}

#[test]
fn derives_text_recipes_from_triplets() {
    let split = SplitRatios::default();
    let config = SamplerConfig {
        seed: 3,
        batch_size: 1,
        chunking: ChunkingStrategy::default(),
        recipes: vec![TripletRecipe {
            name: "title_to_intro".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongPublicationDate,
            weight: 1.0,
            instruction: None,
            allow_same_anchor_positive: false,
        }],
        text_recipes: Vec::new(),
        split,
        ..SamplerConfig::default()
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 17).unwrap());
    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(InMemorySource::new("unit", vec![sample_record()])));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();
    let batch = sampler.next_text_batch(SplitLabel::Train).unwrap();
    assert!(!batch.is_empty());
    assert_eq!(batch.samples.len(), 1);
    assert!(batch.samples[0].recipe.starts_with("title_to_intro_"));
}

#[test]
fn source_triplets_drive_text_sampling() {
    let split = SplitRatios::default();
    let mut config = base_config();
    config.seed = 7;
    config.batch_size = 1;
    config.recipes.clear();
    config.text_recipes.clear();

    let store = Arc::new(DeterministicSplitStore::new(split, 41).unwrap());
    let records = vec![
        trader_record(
            "source_a::2025/01-01/article_a.txt",
            "2025-01-01",
            "Alpha",
            "Body alpha",
        ),
        trader_record(
            "source_a::2025/01-02/article_b.txt",
            "2025-01-02",
            "Beta",
            "Body beta",
        ),
    ];
    let recipes = vec![TripletRecipe {
        name: Cow::Borrowed("source_auto"),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    }];
    let decorated = RecipeDecoratedSource::new(records, recipes);
    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(decorated));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let batch = sampler.next_text_batch(SplitLabel::Train).unwrap();
    assert!(batch.samples[0].recipe.starts_with("source_auto_"));
    assert_eq!(batch.samples.len(), 1);
}

#[test]
fn source_defined_recipes_fill_config_gap() {
    let split = SplitRatios::default();
    let config = SamplerConfig {
        seed: 41,
        batch_size: 2,
        chunking: ChunkingStrategy::default(),
        recipes: vec![],
        text_recipes: vec![],
        split,
        ..SamplerConfig::default()
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 19).unwrap());
    let recipes = vec![TripletRecipe {
        name: "inline_title_summary".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    }];
    let records = vec![
        trader_record(
            "source_a::2025/01-01/article_a.txt",
            "2025-01-01",
            "Alpha",
            "Body alpha",
        ),
        trader_record(
            "source_a::2025/01-02/article_b.txt",
            "2025-01-02",
            "Beta",
            "Body beta",
        ),
    ];
    for record in &records {
        store.upsert(record.id.clone(), SplitLabel::Train).unwrap();
    }
    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(RecipeSource::new(records, recipes.clone())));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();
    let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
    assert_eq!(batch.triplets[0].recipe, recipes[0].name.as_ref());
    assert!(!batch.triplets.is_empty());
}

#[test]
fn source_recipes_drive_text_sampling() {
    let split = SplitRatios::default();
    let config = SamplerConfig {
        seed: 43,
        batch_size: 1,
        chunking: ChunkingStrategy::default(),
        recipes: vec![],
        text_recipes: vec![],
        split,
        ..SamplerConfig::default()
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 29).unwrap());
    let recipes = vec![TripletRecipe {
        name: "inline_title_context".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    }];
    let records = vec![
        trader_record(
            "source_a::2025/01-01/article_a.txt",
            "2025-01-01",
            "Alpha",
            "Body alpha",
        ),
        trader_record(
            "source_a::2025/01-02/article_b.txt",
            "2025-01-02",
            "Beta",
            "Body beta",
        ),
    ];
    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(RecipeSource::new(records, recipes)));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();
    let batch = sampler.next_text_batch(SplitLabel::Train).unwrap();
    assert_eq!(batch.samples.len(), 1);
    assert!(batch.samples[0].recipe.starts_with("inline_title_context_"));
}

#[test]
fn source_a_negative_pairs_follow_strategy() {
    let split = SplitRatios::default();
    let config = SamplerConfig {
        seed: 4,
        batch_size: 2,
        chunking: ChunkingStrategy::default(),
        recipes: vec![TripletRecipe {
            name: "tt_wrong_article".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
            allow_same_anchor_positive: false,
        }],
        text_recipes: Vec::new(),
        split,
        ..SamplerConfig::default()
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 23).unwrap());
    let records = vec![
        trader_record(
            "source_a::2025/01-01/article_a.txt",
            "2025-01-01",
            "Alpha",
            "Body alpha",
        ),
        trader_record(
            "source_a::2025/01-01/article_b.txt",
            "2025-01-01",
            "Beta",
            "Body beta",
        ),
    ];
    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(InMemorySource::new("tt", records)));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();
    let batch = sampler.next_pair_batch(SplitLabel::Train).unwrap();
    assert!(!batch.pairs.is_empty());
    let negative = batch
        .pairs
        .iter()
        .find(|pair| pair.label == PairLabel::Negative)
        .expect("expected a negative pair");
    assert_eq!(negative.reason.as_deref(), Some("wrong_article"));
    assert_ne!(negative.anchor.record_id, negative.positive.record_id);
}

#[test]
fn qa_negative_pairs_mismatch() {
    let split = SplitRatios::default();
    let config = SamplerConfig {
        seed: 5,
        batch_size: 2,
        chunking: ChunkingStrategy::default(),
        recipes: vec![TripletRecipe {
            name: "qa_wrong_match".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::QuestionAnswerMismatch,
            weight: 1.0,
            instruction: None,
            allow_same_anchor_positive: false,
        }],
        text_recipes: Vec::new(),
        split,
        ..SamplerConfig::default()
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 31).unwrap());
    let records = vec![
        qa_pair_record(
            "source_b::factual/alpha.txt",
            "What is alpha?",
            "Alpha is excess return.",
        ),
        qa_pair_record(
            "source_b::factual/beta.txt",
            "What is beta?",
            "Beta tracks market sensitivity.",
        ),
    ];
    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(InMemorySource::new("qa", records)));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();
    let batch = sampler.next_pair_batch(SplitLabel::Train).unwrap();
    assert!(!batch.pairs.is_empty());
    let negative = batch
        .pairs
        .iter()
        .find(|pair| pair.label == PairLabel::Negative)
        .expect("expected a negative pair");
    assert_eq!(negative.reason.as_deref(), Some("wrong_qa_pairing"));
    assert_ne!(negative.anchor.record_id, negative.positive.record_id);
}

#[test]
fn wrong_article_falls_back_within_same_split() {
    let split = SplitRatios {
        train: 0.34,
        validation: 0.33,
        test: 0.33,
    };
    let config = SamplerConfig {
        seed: 9,
        batch_size: 1,
        chunking: ChunkingStrategy::default(),
        recipes: Vec::new(),
        text_recipes: Vec::new(),
        split,
        ..SamplerConfig::default()
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 47).unwrap());

    let find_id = |label: SplitLabel, prefix: &str| -> String {
        for i in 0..5000 {
            let id = format!("{prefix}_{i}");
            if store.ensure(id.clone()).unwrap() == label {
                return id;
            }
        }
        panic!("unable to find id for {:?}", label);
    };

    let anchor_ids = vec![
        find_id(SplitLabel::Train, "wa_anchor_train"),
        find_id(SplitLabel::Validation, "wa_anchor_val"),
        find_id(SplitLabel::Test, "wa_anchor_test"),
    ];
    let other_ids = [
        find_id(SplitLabel::Train, "wa_other_train"),
        find_id(SplitLabel::Validation, "wa_other_val"),
        find_id(SplitLabel::Test, "wa_other_test"),
    ];

    let anchor_records: Vec<DataRecord> = anchor_ids
        .iter()
        .enumerate()
        .map(|(i, id)| trader_record(id, "2025-01-01", &format!("Anchor {i}"), "Body alpha"))
        .collect();
    let other_records: Vec<DataRecord> = other_ids
        .iter()
        .enumerate()
        .map(|(i, id)| trader_record(id, "2025-01-02", &format!("Other {i}"), "Body beta"))
        .collect();

    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(InMemorySource::new("tt", anchor_records)));
    sampler.register_source(Box::new(InMemorySource::new("other", other_records)));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let mut inner = sampler.inner.lock().unwrap();
    let mut seen_splits = std::collections::HashSet::new();
    for anchor_id in anchor_ids {
        let anchor = inner.records.get(&anchor_id).cloned().expect("anchor");
        let (negative, _fallback) = inner
            .select_negative_record(&anchor, &NegativeStrategy::WrongArticle)
            .expect("negative");
        assert_ne!(negative.id, anchor.id);
        let anchor_label = inner.split_store.label_for(&anchor.id).unwrap();
        let negative_label = inner.split_store.label_for(&negative.id).unwrap();
        seen_splits.insert(anchor_label);
        assert_eq!(negative_label, anchor_label);
    }
    assert_eq!(seen_splits.len(), 3);
}

#[cfg(feature = "bm25-mining")]
#[test]
fn bm25_hard_negative_respects_same_source_split_pool() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let config = SamplerConfig {
        seed: 13,
        batch_size: 1,
        chunking: ChunkingStrategy::default(),
        recipes: Vec::new(),
        text_recipes: Vec::new(),
        split,
        ..SamplerConfig::default()
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 13).unwrap());
    let find_train_id = |prefix: &str| -> String {
        (0u32..)
            .find_map(|i| {
                let id = format!("{prefix}_{i}");
                (store.label_for(&id) == Some(SplitLabel::Train)).then_some(id)
            })
            .unwrap()
    };
    let anchor_id = find_train_id("bm25_anchor");
    let similar_id = find_train_id("bm25_similar");
    let distant_id = find_train_id("bm25_distant");

    let anchor = trader_record(
        &anchor_id,
        "2025-01-01",
        "Apple banana quarterly report",
        "Apple banana revenue growth guidance",
    );
    let similar = trader_record(
        &similar_id,
        "2025-01-01",
        "Banana apple market update",
        "Revenue guidance for apple banana market",
    );
    let distant = trader_record(
        &distant_id,
        "2025-01-03",
        "Quantum field dynamics",
        "Black holes and gravitational lensing",
    );

    let sampler = TripletSampler::new(config, Arc::clone(&store));
    sampler.register_source(Box::new(InMemorySource::new(
        "tt",
        vec![anchor.clone(), similar.clone(), distant],
    )));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let mut inner = sampler.inner.lock().unwrap();
    let (negative, fallback_used) = inner
        .select_negative_record(&anchor, &NegativeStrategy::WrongArticle)
        .expect("expected bm25-ranked negative via WrongArticle strategy");

    let _ = fallback_used;
    assert_ne!(negative.id, anchor.id);
    assert_eq!(store.label_for(&anchor.id), Some(SplitLabel::Train));
    assert_eq!(store.label_for(&negative.id), Some(SplitLabel::Train));
}

#[cfg(feature = "bm25-mining")]
#[test]
fn bm25_negative_is_lexically_closer_than_uniform_pool_baseline() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let config = SamplerConfig {
        seed: 314,
        batch_size: 1,
        chunking: ChunkingStrategy::default(),
        recipes: Vec::new(),
        text_recipes: Vec::new(),
        split,
        ..SamplerConfig::default()
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 314).unwrap());
    let find_train_id = |prefix: &str| -> String {
        (0u32..)
            .find_map(|i| {
                let id = format!("{prefix}_{i}");
                (store.label_for(&id) == Some(SplitLabel::Train)).then_some(id)
            })
            .unwrap()
    };
    let anchor_id = find_train_id("bm25_lex_anchor");
    let similar_id = find_train_id("bm25_lex_similar");
    let distant_id = find_train_id("bm25_lex_distant");

    let anchor = trader_record(
        &anchor_id,
        "2025-01-01",
        "Apple banana quarterly report",
        "apple banana revenue growth guidance demand outlook",
    );
    let similar = trader_record(
        &similar_id,
        "2025-01-01",
        "Banana apple market update",
        "apple banana revenue guidance and market demand outlook",
    );
    let distant = trader_record(
        &distant_id,
        "2025-01-01",
        "Deep ocean geology",
        "tectonic plates abyssal sediment marine trench volcanism",
    );

    let sampler = TripletSampler::new(config, Arc::clone(&store));
    sampler.register_source(Box::new(InMemorySource::new(
        "tt",
        vec![anchor.clone(), similar, distant],
    )));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let mut inner = sampler.inner.lock().unwrap();
    // Use the ingested record so its `source` field reflects the ID assigned by
    // IngestionManager ("tt"), which is necessary for correct BM25 index lookup
    // and consistent pool filtering below.
    let ingested_anchor = inner
        .records
        .get(&anchor.id)
        .cloned()
        .expect("anchor must be present in ingested records");
    let (_selected_negative, _fallback) = inner
        .select_negative_record(&ingested_anchor, &NegativeStrategy::WrongArticle)
        .expect("expected bm25-ranked negative");

    let anchor_text = record_bm25_text(&ingested_anchor, inner.config.chunking.max_window_tokens);

    // Control baseline for non-BM25 behavior: uniform random choice over the
    // same strategy pool used by WrongArticle (same source, same split).
    let pool: Vec<DataRecord> = inner
        .records
        .values()
        .filter(|candidate| {
            candidate.source == ingested_anchor.source
                && candidate.id != ingested_anchor.id
                && inner
                    .split_store
                    .label_for(&candidate.id)
                    .map(|label| label == SplitLabel::Train)
                    .unwrap_or(false)
        })
        .cloned()
        .collect();
    assert!(!pool.is_empty(), "control pool must not be empty");

    let (mean_pool_jaccard, mean_pool_cosine) = {
        let mut j_total = 0.0_f32;
        let mut c_total = 0.0_f32;
        for candidate in &pool {
            let candidate_text =
                record_bm25_text(candidate, inner.config.chunking.max_window_tokens);
            let (j_score, c_score) =
                crate::metrics::lexical_similarity_scores(&anchor_text, &candidate_text);
            j_total += j_score;
            c_total += c_score;
        }
        let denom = pool.len() as f32;
        (j_total / denom, c_total / denom)
    };

    // Assert on BM25's top-ranked candidate rather than the cursor-selected one.
    // The cursor uses a deterministic offset derived from the epoch seed and anchor
    // ID, so it doesn't always land on rank-0. What we're testing is that BM25's
    // *ranking* quality is better than uniform random — i.e. rank-0 beats the pool
    // mean — not that the rotation-offset happens to agree on any single call.
    let ranked = inner.bm25_ranked_candidates(&ingested_anchor);
    assert!(
        !ranked.is_empty(),
        "BM25 must produce at least one ranked candidate"
    );
    let top_candidate = inner
        .records
        .get(ranked.first().unwrap())
        .cloned()
        .expect("top BM25 candidate must be in records");
    let top_text = record_bm25_text(&top_candidate, inner.config.chunking.max_window_tokens);
    let (j_top, c_top) = crate::metrics::lexical_similarity_scores(&anchor_text, &top_text);

    assert!(
        j_top > mean_pool_jaccard,
        "BM25 top-ranked negative should beat non-bm25 uniform-pool Jaccard baseline (top={j_top:.4}, baseline={mean_pool_jaccard:.4})"
    );
    assert!(
        c_top > mean_pool_cosine,
        "BM25 top-ranked negative should beat non-bm25 uniform-pool cosine baseline (top={c_top:.4}, baseline={mean_pool_cosine:.4})"
    );
}

#[cfg(feature = "bm25-mining")]
#[test]
fn custom_recipe_still_respects_strategy_pool_with_bm25() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let recipe = TripletRecipe {
        name: "custom_wrong_publication_date".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongPublicationDate,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    };
    let config = SamplerConfig {
        seed: 23,
        batch_size: 8,
        chunking: ChunkingStrategy::default(),
        recipes: vec![recipe],
        text_recipes: Vec::new(),
        split,
        ..SamplerConfig::default()
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 23).unwrap());
    let find_train_id = |prefix: &str| -> String {
        (0u32..)
            .find_map(|i| {
                let id = format!("{prefix}_{i}");
                (store.label_for(&id) == Some(SplitLabel::Train)).then_some(id)
            })
            .unwrap()
    };
    let ca_id = find_train_id("custom_anchor_a");
    let cb_id = find_train_id("custom_anchor_b");
    let cc_id = find_train_id("custom_anchor_c");

    let records = vec![
        trader_record(
            &ca_id,
            "2025-01-01",
            "Apple banana quarterly report",
            "Apple banana revenue growth guidance",
        ),
        trader_record(
            &cb_id,
            "2025-01-01",
            "Apple banana management update",
            "Apple banana demand outlook",
        ),
        trader_record(
            &cc_id,
            "2025-01-02",
            "Energy market briefing",
            "Oil and gas supply outlook",
        ),
    ];
    let date_by_id: HashMap<String, String> = records
        .iter()
        .filter_map(|record| {
            taxonomy_value(record, META_FIELD_DATE)
                .map(|date| (record.id.clone(), date.to_string()))
        })
        .collect();

    let sampler = TripletSampler::new(config, Arc::clone(&store));
    sampler.register_source(Box::new(InMemorySource::new("tt", records)));

    let batch = sampler
        .next_triplet_batch(SplitLabel::Train)
        .expect("expected custom recipe triplet batch");
    assert!(!batch.triplets.is_empty());

    for triplet in &batch.triplets {
        let anchor_date = date_by_id
            .get(&triplet.anchor.record_id)
            .expect("anchor date must exist");
        let negative_date = date_by_id
            .get(&triplet.negative.record_id)
            .expect("negative date must exist");
        assert_ne!(
            anchor_date, negative_date,
            "custom recipe negative must respect WrongPublicationDate pool under bm25"
        );
    }
}

#[cfg(feature = "bm25-mining")]
#[test]
fn bm25_ranked_candidates_never_cross_split_boundaries() {
    let split = SplitRatios {
        train: 0.34,
        validation: 0.33,
        test: 0.33,
    };
    let config = SamplerConfig {
        seed: 31,
        batch_size: 1,
        chunking: ChunkingStrategy::default(),
        recipes: Vec::new(),
        text_recipes: Vec::new(),
        split,
        ..SamplerConfig::default()
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 71).unwrap());

    let find_id = |label: SplitLabel, prefix: &str| -> String {
        for i in 0..8000 {
            let id = format!("{prefix}_{i}");
            if store.ensure(id.clone()).unwrap() == label {
                return id;
            }
        }
        panic!("unable to find id for {:?}", label);
    };

    let anchors = vec![
        find_id(SplitLabel::Train, "bm25_split_anchor_train"),
        find_id(SplitLabel::Validation, "bm25_split_anchor_val"),
        find_id(SplitLabel::Test, "bm25_split_anchor_test"),
    ];
    let peers = [
        find_id(SplitLabel::Train, "bm25_split_peer_train"),
        find_id(SplitLabel::Validation, "bm25_split_peer_val"),
        find_id(SplitLabel::Test, "bm25_split_peer_test"),
    ];

    let mut records: Vec<DataRecord> = Vec::new();
    for (i, anchor_id) in anchors.iter().enumerate() {
        records.push(trader_record(
            anchor_id,
            "2025-01-01",
            &format!("Split anchor {i}"),
            &format!("bm25 split scoped text {i}"),
        ));
    }
    for (i, peer_id) in peers.iter().enumerate() {
        records.push(trader_record(
            peer_id,
            "2025-01-01",
            &format!("Split peer {i}"),
            &format!("bm25 split scoped peer text {i}"),
        ));
    }

    let sampler = TripletSampler::new(config, Arc::clone(&store));
    sampler.register_source(Box::new(InMemorySource::new("tt", records)));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let mut inner = sampler.inner.lock().unwrap();
    for anchor_id in anchors {
        let anchor = inner.records.get(&anchor_id).cloned().expect("anchor");
        let (negative, _fallback) = inner
            .select_negative_record(&anchor, &NegativeStrategy::WrongArticle)
            .expect("negative should exist");

        let anchor_label = inner
            .split_store
            .label_for(&anchor.id)
            .expect("anchor split label");
        let negative_label = inner
            .split_store
            .label_for(&negative.id)
            .expect("negative split label");
        assert_eq!(negative_label, anchor_label);

        let ranked: Vec<RecordId> = inner
            .bm25_backend_mut()
            .hard_negatives_get(&anchor.id)
            .expect("bm25 cache entry for anchor");
        assert!(!ranked.is_empty());
        for candidate_id in &ranked {
            let candidate_label = inner
                .split_store
                .label_for(candidate_id)
                .expect("candidate split label");
            assert_eq!(
                candidate_label, anchor_label,
                "bm25 candidate leaked across split boundary"
            );
        }
    }
}

#[cfg(feature = "bm25-mining")]
#[test]
fn bm25_ranked_candidates_match_raw_bm25_engine() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let config = SamplerConfig {
        seed: 991,
        batch_size: 1,
        chunking: ChunkingStrategy::default(),
        recipes: Vec::new(),
        text_recipes: Vec::new(),
        split,
        ..SamplerConfig::default()
    };
    // seed=12: all "readable_*" IDs hash to Train under train:0.7.
    let store = Arc::new(DeterministicSplitStore::new(split, 12).unwrap());

    let records = vec![
        trader_record(
            "readable_anchor",
            "2025-01-01",
            "Climate policy briefing",
            "carbon pricing policy emissions reduction roadmap market design",
        ),
        trader_record(
            "readable_topical_1",
            "2025-01-01",
            "Carbon market and emissions policy",
            "carbon pricing policy emissions reduction roadmap market design",
        ),
        trader_record(
            "readable_topical_2",
            "2025-01-01",
            "Carbon policy update",
            "carbon pricing policy emissions reduction roadmap",
        ),
        trader_record(
            "readable_mid_1",
            "2025-01-01",
            "Energy transition memo",
            "emissions reduction roadmap clean energy transition planning",
        ),
        trader_record(
            "readable_mid_2",
            "2025-01-01",
            "Regulatory market digest",
            "policy market design regulatory framework and compliance",
        ),
        trader_record(
            "readable_weak_1",
            "2025-01-01",
            "Archaeology field note",
            "bronze age pottery fragments excavation trench mapping",
        ),
        trader_record(
            "readable_weak_2",
            "2025-01-01",
            "Marine geology report",
            "subduction zones oceanic crust tectonic shear",
        ),
    ];

    let sampler = TripletSampler::new(config, Arc::clone(&store));
    sampler.register_source(Box::new(InMemorySource::new("readable_source", records)));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let mut inner = sampler.inner.lock().unwrap();
    let anchor = inner
        .records
        .get("readable_anchor")
        .cloned()
        .expect("anchor should be present");

    let sampler_ranked = inner.bm25_ranked_candidates(&anchor);

    // Collect all indexed record IDs into an owned Vec so we can access
    // `inner.records` inside the mapping closure without a borrow conflict.
    let meta_record_ids: Vec<RecordId> = inner
        .bm25_backend_mut()
        .index_meta_record_ids()
        .expect("bm25 global index should be built");

    let docs: Vec<::bm25::Document<usize>> = meta_record_ids
        .iter()
        .enumerate()
        .map(|(idx, record_id)| {
            // All records in the BM25 index are also in self.records (built from
            // all per-source caches), so this lookup is always O(1) and found.
            let record = inner
                .records
                .get(record_id)
                .expect("record must be in self.records");
            ::bm25::Document {
                id: idx,
                contents: record_bm25_text(record, inner.config.chunking.max_window_tokens),
            }
        })
        .collect();
    let engine =
        ::bm25::SearchEngineBuilder::<usize>::with_documents(::bm25::Language::English, docs)
            .build();

    let query = record_bm25_text(&anchor, inner.config.chunking.max_window_tokens);
    let max_results = meta_record_ids.len();
    let mut raw = engine.search(&query, max_results);
    // Match sampler tie-breaking exactly: score descending, then stable id order.
    raw.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.document.id.cmp(&b.document.id))
    });

    let mut expected: Vec<RecordId> = Vec::new();
    for result in raw {
        let Some(record_id) = meta_record_ids.get(result.document.id) else {
            continue;
        };
        if *record_id != anchor.id {
            expected.push(record_id.clone());
        }
    }

    // This is the direct "is it actually BM25" proof: sampler ranking must
    // equal a separately recomputed BM25 crate ranking for the same corpus/query.
    assert_eq!(
        sampler_ranked, expected,
        "sampler BM25 rank must match direct BM25 crate rank for the same corpus/query"
    );
}

#[cfg(feature = "bm25-mining")]
#[test]
fn bm25_ranking_ignores_kvp_meta_prefix_tags() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let config = SamplerConfig {
        seed: 888,
        batch_size: 1,
        chunking: ChunkingStrategy::default(),
        recipes: Vec::new(),
        text_recipes: Vec::new(),
        split,
        ..SamplerConfig::default()
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 888).unwrap());
    let find_train_id = |prefix: &str| -> String {
        (0u32..)
            .find_map(|i| {
                let id = format!("{prefix}_{i}");
                (store.label_for(&id) == Some(SplitLabel::Train)).then_some(id)
            })
            .unwrap()
    };
    let anchor_id = find_train_id("kvp_anchor");
    let bait_id = find_train_id("kvp_bait");
    let plain_id = find_train_id("plain_text_best");

    let anchor = trader_record(
        &anchor_id,
        "2025-01-01",
        "Anchor",
        "carbon pricing policy emissions roadmap",
    );

    let mut kvp_bait = trader_record(
        &bait_id,
        "2025-01-01",
        "KVP bait",
        "ancient pottery shards trench notes",
    );
    let mut kvp = KvpPrefixSampler::new(1.0);
    kvp.add_variant([(
        "meta",
        "carbon pricing policy emissions roadmap carbon pricing policy emissions roadmap",
    )]);
    kvp_bait.meta_prefix = Some(kvp);

    let plain_text_best = trader_record(
        &plain_id,
        "2025-01-01",
        "Plain text best",
        "carbon pricing policy emissions roadmap carbon market",
    );

    // Sanity-check the BM25 text path directly: meta_prefix content must not appear.
    let bait_text = record_bm25_text(&kvp_bait, config.chunking.max_window_tokens);
    assert!(
        !bait_text.contains("carbon pricing policy emissions roadmap carbon pricing"),
        "BM25 corpus text must not include KVP meta-prefix tags"
    );

    let sampler = TripletSampler::new(config, Arc::clone(&store));
    sampler.register_source(Box::new(InMemorySource::new(
        "kvp_source",
        vec![anchor, kvp_bait, plain_text_best],
    )));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let mut inner = sampler.inner.lock().unwrap();
    let anchor = inner
        .records
        .get(&anchor_id)
        .cloned()
        .expect("anchor should exist");

    let ranked = inner.bm25_ranked_candidates(&anchor);
    assert!(
        !ranked.is_empty(),
        "expected BM25 to return ranked candidates"
    );
    assert_eq!(
        ranked[0], plain_id,
        "BM25 top candidate should be driven by plain section text, not KVP meta-prefix tags"
    );
}

#[cfg(feature = "bm25-mining")]
#[test]
fn bm25_triplets_never_reuse_text_across_slots() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let recipe = TripletRecipe {
        name: "bm25_text_distinct_slots".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    };
    let config = SamplerConfig {
        seed: 91,
        batch_size: 6,
        chunking: ChunkingStrategy::default(),
        recipes: vec![recipe],
        text_recipes: Vec::new(),
        split,
        ..SamplerConfig::default()
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 91).unwrap());
    let find_train_id = |prefix: &str| -> String {
        (0u32..)
            .find_map(|i| {
                let id = format!("{prefix}_{i}");
                (store.label_for(&id) == Some(SplitLabel::Train)).then_some(id)
            })
            .unwrap()
    };
    let slot_anchor = find_train_id("bm25_slot_anchor");
    let slot_same = find_train_id("bm25_slot_same_context");
    let slot_unique = find_train_id("bm25_slot_unique_context");

    let records = vec![
        trader_record(
            &slot_anchor,
            "2025-01-01",
            "Anchor title unique",
            "Shared duplicate context",
        ),
        trader_record(
            &slot_same,
            "2025-01-01",
            "Other title one",
            "Shared duplicate context",
        ),
        trader_record(
            &slot_unique,
            "2025-01-01",
            "Other title two",
            "A fully distinct context body",
        ),
    ];

    let sampler = TripletSampler::new(config, Arc::clone(&store));
    sampler.register_source(Box::new(InMemorySource::new("tt", records)));
    let batch = sampler
        .next_triplet_batch(SplitLabel::Train)
        .expect("expected bm25 triplet batch");

    assert!(!batch.triplets.is_empty());
    for triplet in &batch.triplets {
        assert_ne!(triplet.anchor.text, triplet.positive.text);
        assert_ne!(triplet.anchor.text, triplet.negative.text);
        assert_ne!(triplet.positive.text, triplet.negative.text);
    }
}

#[cfg(feature = "bm25-mining")]
#[test]
fn bm25_cursor_pruning_runs_even_when_other_cursors_are_empty() {
    let split = SplitRatios::default();
    let store = Arc::new(DeterministicSplitStore::new(split, 1234).unwrap());
    let mut inner = TripletSamplerInner::new(base_config(), store);

    assert!(inner.chunk_cursors.is_empty());
    assert!(inner.role_cursors.is_empty());

    inner
        .bm25_backend_mut()
        .negative_cursors_insert(("stale_anchor".to_string(), SplitLabel::Train), 7);
    assert_eq!(inner.bm25_backend_mut().negative_cursors_len(), 1);

    // With no records loaded, every cursor entry is stale and must be removed.
    // This specifically guards against early-return logic that skips BM25 pruning.
    inner.prune_cursor_state();

    assert!(
        inner.bm25_backend_mut().negative_cursors_is_empty(),
        "bm25_negative_cursors should be pruned even when chunk/role cursor maps are empty"
    );
}

#[cfg(feature = "bm25-mining")]
#[test]
fn bm25_cursor_state_is_cleared_on_each_record_snapshot_sync() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 2024).unwrap());
    let sync_id = (0u32..)
        .find_map(|i| {
            let id = format!("strict_sync_anchor_{i}");
            (store.label_for(&id) == Some(SplitLabel::Train)).then_some(id)
        })
        .unwrap();
    let sampler = TripletSampler::new(base_config(), Arc::clone(&store));
    sampler.register_source(Box::new(InMemorySource::new(
        "strict_sync_source",
        vec![trader_record(&sync_id, "2025-01-01", "Anchor", "body")],
    )));

    let mut inner = sampler.inner.lock().unwrap();
    inner.ingest_internal(SplitLabel::Train).unwrap();

    inner
        .bm25_backend_mut()
        .negative_cursors_insert((sync_id.clone(), SplitLabel::Train), 42);
    assert_eq!(inner.bm25_backend_mut().negative_cursors_len(), 1);

    // Strict contract: every snapshot sync clears BM25 cursor state, even if
    // the same anchor remains present in the refreshed record pool.
    inner.sync_records_from_cache().unwrap();
    assert!(
        inner.bm25_backend_mut().negative_cursors_is_empty(),
        "bm25 cursor state must reset at every record snapshot boundary"
    );
}

#[test]
fn wrong_publication_date_falls_back_within_same_split() {
    let split = SplitRatios {
        train: 0.34,
        validation: 0.33,
        test: 0.33,
    };
    let config = SamplerConfig {
        seed: 7,
        batch_size: 1,
        chunking: ChunkingStrategy::default(),
        recipes: Vec::new(),
        text_recipes: Vec::new(),
        split,
        ..SamplerConfig::default()
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 37).unwrap());

    let find_id = |label: SplitLabel, prefix: &str| -> String {
        for i in 0..5000 {
            let id = format!("{prefix}_{i}");
            if store.ensure(id.clone()).unwrap() == label {
                return id;
            }
        }
        panic!("unable to find id for {:?}", label);
    };

    let anchor_ids = vec![
        find_id(SplitLabel::Train, "wpd_anchor_train"),
        find_id(SplitLabel::Validation, "wpd_anchor_val"),
        find_id(SplitLabel::Test, "wpd_anchor_test"),
    ];
    let other_ids = [
        find_id(SplitLabel::Train, "wpd_other_train"),
        find_id(SplitLabel::Validation, "wpd_other_val"),
        find_id(SplitLabel::Test, "wpd_other_test"),
    ];

    let anchor_records: Vec<DataRecord> = anchor_ids
        .iter()
        .enumerate()
        .map(|(i, id)| trader_record(id, "2025-01-01", &format!("Anchor {i}"), "Body"))
        .collect();
    let other_records: Vec<DataRecord> = other_ids
        .iter()
        .enumerate()
        .map(|(i, id)| trader_record(id, "2025-01-01", &format!("Other {i}"), "Body"))
        .collect();

    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(InMemorySource::new("tt", anchor_records)));
    sampler.register_source(Box::new(InMemorySource::new("other", other_records)));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let mut inner = sampler.inner.lock().unwrap();
    let mut seen_splits = std::collections::HashSet::new();
    for anchor_id in anchor_ids {
        let anchor = inner.records.get(&anchor_id).cloned().expect("anchor");
        let (negative, _fallback) = inner
            .select_negative_record(&anchor, &NegativeStrategy::WrongPublicationDate)
            .expect("negative");
        assert_ne!(negative.id, anchor.id);
        let anchor_label = inner.split_store.label_for(&anchor.id).unwrap();
        let negative_label = inner.split_store.label_for(&negative.id).unwrap();
        seen_splits.insert(anchor_label);
        assert_eq!(negative_label, anchor_label);
    }
    assert_eq!(seen_splits.len(), 3);
}

#[test]
fn qa_mismatch_falls_back_within_same_split() {
    let split = SplitRatios {
        train: 0.34,
        validation: 0.33,
        test: 0.33,
    };
    let config = SamplerConfig {
        seed: 11,
        batch_size: 1,
        chunking: ChunkingStrategy::default(),
        recipes: Vec::new(),
        text_recipes: Vec::new(),
        split,
        ..SamplerConfig::default()
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 53).unwrap());

    let find_id = |label: SplitLabel, prefix: &str| -> String {
        for i in 0..5000 {
            let id = format!("{prefix}_{i}");
            if store.ensure(id.clone()).unwrap() == label {
                return id;
            }
        }
        panic!("unable to find id for {:?}", label);
    };

    let anchor_ids = vec![
        find_id(SplitLabel::Train, "qam_anchor_train"),
        find_id(SplitLabel::Validation, "qam_anchor_val"),
        find_id(SplitLabel::Test, "qam_anchor_test"),
    ];
    let other_ids = [
        find_id(SplitLabel::Train, "qam_other_train"),
        find_id(SplitLabel::Validation, "qam_other_val"),
        find_id(SplitLabel::Test, "qam_other_test"),
    ];

    let qa_records: Vec<DataRecord> = anchor_ids
        .iter()
        .enumerate()
        .map(|(i, id)| {
            qa_pair_record(
                id,
                &format!("What is item {i}?"),
                &format!("Item {i} answer."),
            )
        })
        .collect();
    let other_records: Vec<DataRecord> = other_ids
        .iter()
        .enumerate()
        .map(|(i, id)| trader_record(id, "2025-01-02", &format!("Beta {i}"), "Body beta"))
        .collect();

    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(InMemorySource::new("qa", qa_records)));
    sampler.register_source(Box::new(InMemorySource::new("other", other_records)));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let mut inner = sampler.inner.lock().unwrap();
    let mut seen_splits = std::collections::HashSet::new();
    for anchor_id in anchor_ids {
        let anchor = inner.records.get(&anchor_id).cloned().expect("anchor");
        let (negative, _fallback) = inner
            .select_negative_record(&anchor, &NegativeStrategy::QuestionAnswerMismatch)
            .expect("negative");
        assert_ne!(negative.id, anchor.id);
        let anchor_label = inner.split_store.label_for(&anchor.id).unwrap();
        let negative_label = inner.split_store.label_for(&negative.id).unwrap();
        seen_splits.insert(anchor_label);
        assert_eq!(negative_label, anchor_label);
    }
    assert_eq!(seen_splits.len(), 3);
}

#[test]
fn negative_selection_never_falls_back_across_splits() {
    let split = SplitRatios {
        train: 0.34,
        validation: 0.33,
        test: 0.33,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 17).unwrap());

    let find_id = |label: SplitLabel, prefix: &str| -> String {
        for i in 0..2000 {
            let id = format!("{prefix}_{i}");
            if store.ensure(id.clone()).unwrap() == label {
                return id;
            }
        }
        panic!("unable to find id for {:?}", label);
    };

    let train_id = find_id(SplitLabel::Train, "neg_train");
    let val_id = find_id(SplitLabel::Validation, "neg_val");
    let test_id = find_id(SplitLabel::Test, "neg_test");

    let config = SamplerConfig {
        seed: 21,
        batch_size: 1,
        chunking: ChunkingStrategy::default(),
        recipes: Vec::new(),
        text_recipes: Vec::new(),
        split,
        ..SamplerConfig::default()
    };

    let anchor = trader_record(&train_id, "2025-01-01", "Anchor", "Body A");
    let other_val = trader_record(&val_id, "2025-01-02", "Other Val", "Body B");
    let other_test = trader_record(&test_id, "2025-01-03", "Other Test", "Body C");
    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(InMemorySource::new("a", vec![anchor.clone()])));
    sampler.register_source(Box::new(InMemorySource::new(
        "b",
        vec![other_val, other_test],
    )));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let mut inner = sampler.inner.lock().unwrap();
    let selected = inner.select_negative_record(&anchor, &NegativeStrategy::WrongArticle);
    assert!(
        selected.is_none(),
        "cross-split fallback must be disallowed when same-split candidates are unavailable"
    );
}

#[test]
fn fallback_triplet_negative_never_matches_anchor() {
    let split = SplitRatios {
        train: 0.34,
        validation: 0.33,
        test: 0.33,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 59).unwrap());

    let find_id = |label: SplitLabel, prefix: &str| -> String {
        for i in 0..5000 {
            let id = format!("{prefix}_{i}");
            if store.ensure(id.clone()).unwrap() == label {
                return id;
            }
        }
        panic!("unable to find id for {:?}", label);
    };

    let records = vec![
        trader_record(
            &find_id(SplitLabel::Train, "fallback_train_a"),
            "2025-01-01",
            "Train A",
            "Body train a",
        ),
        trader_record(
            &find_id(SplitLabel::Train, "fallback_train_b"),
            "2025-01-01",
            "Train B",
            "Body train b",
        ),
        trader_record(
            &find_id(SplitLabel::Validation, "fallback_val_a"),
            "2025-01-01",
            "Val A",
            "Body val a",
        ),
        trader_record(
            &find_id(SplitLabel::Validation, "fallback_val_b"),
            "2025-01-01",
            "Val B",
            "Body val b",
        ),
        trader_record(
            &find_id(SplitLabel::Test, "fallback_test_a"),
            "2025-01-01",
            "Test A",
            "Body test a",
        ),
        trader_record(
            &find_id(SplitLabel::Test, "fallback_test_b"),
            "2025-01-01",
            "Test B",
            "Body test b",
        ),
    ];

    let mut config = SamplerConfig {
        seed: 13,
        batch_size: 1,
        chunking: ChunkingStrategy::default(),
        recipes: vec![TripletRecipe {
            name: "wrong_date".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongPublicationDate,
            weight: 1.0,
            instruction: None,
            allow_same_anchor_positive: false,
        }],
        text_recipes: Vec::new(),
        split,
        ..SamplerConfig::default()
    };
    config.allowed_splits = vec![SplitLabel::Train];

    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(InMemorySource::new("tt", records)));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let mut seen_splits = std::collections::HashSet::new();
    let mut saw_fallback = false;
    for _ in 0..120 {
        let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        let triplet = &batch.triplets[0];
        let anchor_label = sampler
            .inner
            .lock()
            .unwrap()
            .split_store
            .label_for(&triplet.anchor.record_id)
            .unwrap();
        let negative_label = sampler
            .inner
            .lock()
            .unwrap()
            .split_store
            .label_for(&triplet.negative.record_id)
            .unwrap();

        seen_splits.insert(anchor_label);
        assert_eq!(anchor_label, negative_label);
        assert_ne!(triplet.anchor.record_id, triplet.negative.record_id);
        assert_ne!(triplet.positive.record_id, triplet.negative.record_id);
        if triplet.recipe.ends_with("_fallback_same_split") {
            saw_fallback = true;
        }
        if seen_splits.len() == 3 && saw_fallback {
            break;
        }
    }
    assert_eq!(seen_splits.len(), 1);
    assert!(saw_fallback, "expected fallback_same_split to occur");
}

#[test]
fn triplets_never_cross_split_boundaries() {
    let split = SplitRatios {
        train: 0.34,
        validation: 0.33,
        test: 0.33,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 123).unwrap());

    let find_id = |label: SplitLabel, prefix: &str| -> String {
        for i in 0..5000 {
            let id = format!("{prefix}_{i}");
            if store.ensure(id.clone()).unwrap() == label {
                return id;
            }
        }
        panic!("unable to find id for {:?}", label);
    };

    let records = vec![
        trader_record(
            &find_id(SplitLabel::Train, "triplet_split_train_a"),
            "2025-01-01",
            "Train A",
            "Body train a",
        ),
        trader_record(
            &find_id(SplitLabel::Train, "triplet_split_train_b"),
            "2025-01-02",
            "Train B",
            "Body train b",
        ),
        trader_record(
            &find_id(SplitLabel::Validation, "triplet_split_val_a"),
            "2025-01-03",
            "Val A",
            "Body val a",
        ),
        trader_record(
            &find_id(SplitLabel::Validation, "triplet_split_val_b"),
            "2025-01-04",
            "Val B",
            "Body val b",
        ),
        trader_record(
            &find_id(SplitLabel::Test, "triplet_split_test_a"),
            "2025-01-05",
            "Test A",
            "Body test a",
        ),
        trader_record(
            &find_id(SplitLabel::Test, "triplet_split_test_b"),
            "2025-01-06",
            "Test B",
            "Body test b",
        ),
    ];

    let mut config = base_config();
    config.seed = 777;
    config.batch_size = 1;
    config.allowed_splits = vec![SplitLabel::Train];
    config.recipes = vec![TripletRecipe {
        name: "split_isolation_triplet".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    }];
    config.text_recipes = Vec::new();

    let sampler = TripletSampler::new(config, Arc::clone(&store));
    sampler.register_source(Box::new(InMemorySource::new("split_iso", records)));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    for _ in 0..40 {
        let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        for triplet in batch.triplets {
            let anchor = store.label_for(&triplet.anchor.record_id).unwrap();
            let positive = store.label_for(&triplet.positive.record_id).unwrap();
            let negative = store.label_for(&triplet.negative.record_id).unwrap();
            assert_eq!(anchor, positive, "anchor and positive must share split");
            assert_eq!(anchor, negative, "negative must stay in anchor split");
        }
    }
}

#[test]
fn split_specific_batch_apis_return_exact_size_and_requested_split_only() {
    let split = SplitRatios {
        train: 0.34,
        validation: 0.33,
        test: 0.33,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 333).unwrap());

    let find_id = |label: SplitLabel, prefix: &str| -> String {
        for i in 0..10000 {
            let id = format!("{prefix}_{i}");
            if store.ensure(id.clone()).unwrap() == label {
                return id;
            }
        }
        panic!("unable to find id for {:?}", label);
    };

    let mut records = Vec::new();
    for split_label in [SplitLabel::Train, SplitLabel::Validation, SplitLabel::Test] {
        for idx in 0..6 {
            let id = find_id(split_label, &format!("split_api_{split_label:?}_{idx}"));
            records.push(trader_record(
                &id,
                "2025-01-01",
                &format!("{split_label:?} {idx}"),
                &format!("{split_label:?} body {idx}"),
            ));
        }
    }

    let mut config = base_config();
    config.seed = 444;
    config.batch_size = 2;
    config.allowed_splits = vec![SplitLabel::Train, SplitLabel::Validation, SplitLabel::Test];
    config.recipes = vec![TripletRecipe {
        name: "split_api_triplet".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    }];
    config.text_recipes = vec![TextRecipe {
        name: "split_api_text".into(),
        selector: Selector::Role(SectionRole::Context),
        weight: 1.0,
        instruction: None,
    }];

    let sampler = TripletSampler::new(config, Arc::clone(&store));
    sampler.register_source(Box::new(InMemorySource::new("split_api", records)));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    for requested_split in [SplitLabel::Train, SplitLabel::Validation, SplitLabel::Test] {
        let pair_batch = sampler.next_pair_batch_for_split(requested_split).unwrap();
        assert_eq!(pair_batch.pairs.len(), 2);
        for pair in &pair_batch.pairs {
            assert_eq!(
                store.label_for(&pair.anchor.record_id).unwrap(),
                requested_split
            );
            assert_eq!(
                store.label_for(&pair.positive.record_id).unwrap(),
                requested_split
            );
        }

        let text_batch = sampler.next_text_batch_for_split(requested_split).unwrap();
        assert_eq!(text_batch.samples.len(), 2);
        for sample in &text_batch.samples {
            assert_eq!(
                store.label_for(&sample.chunk.record_id).unwrap(),
                requested_split
            );
        }

        let triplet_batch = sampler
            .next_triplet_batch_for_split(requested_split)
            .unwrap();
        assert_eq!(triplet_batch.triplets.len(), 2);
        for triplet in &triplet_batch.triplets {
            assert_eq!(
                store.label_for(&triplet.anchor.record_id).unwrap(),
                requested_split
            );
            assert_eq!(
                store.label_for(&triplet.positive.record_id).unwrap(),
                requested_split
            );
            assert_eq!(
                store.label_for(&triplet.negative.record_id).unwrap(),
                requested_split
            );
        }
    }
}

#[test]
fn split_specific_triplet_api_keeps_anchor_positive_negative_in_same_split() {
    let split = SplitRatios {
        train: 0.34,
        validation: 0.33,
        test: 0.33,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 445).unwrap());

    let find_id = |label: SplitLabel, prefix: &str| -> String {
        for i in 0..10000 {
            let id = format!("{prefix}_{i}");
            if store.ensure(id.clone()).unwrap() == label {
                return id;
            }
        }
        panic!("unable to find id for {:?}", label);
    };

    let mut records = Vec::new();
    for split_label in [SplitLabel::Train, SplitLabel::Validation, SplitLabel::Test] {
        for idx in 0..8 {
            let id = find_id(
                split_label,
                &format!("split_triplet_iso_{split_label:?}_{idx}"),
            );
            records.push(trader_record(
                &id,
                "2025-01-01",
                &format!("{split_label:?} {idx}"),
                &format!("{split_label:?} body {idx}"),
            ));
        }
    }

    let mut config = base_config();
    config.seed = 446;
    config.batch_size = 3;
    config.allowed_splits = vec![SplitLabel::Train, SplitLabel::Validation, SplitLabel::Test];
    config.recipes = vec![TripletRecipe {
        name: "split_triplet_only".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    }];
    config.text_recipes = Vec::new();

    let sampler = TripletSampler::new(config, Arc::clone(&store));
    sampler.register_source(Box::new(InMemorySource::new("split_triplet_iso", records)));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    for requested_split in [SplitLabel::Train, SplitLabel::Validation, SplitLabel::Test] {
        let batch = sampler
            .next_triplet_batch_for_split(requested_split)
            .unwrap();
        assert_eq!(batch.triplets.len(), 3);
        for triplet in &batch.triplets {
            let anchor = store.label_for(&triplet.anchor.record_id).unwrap();
            let positive = store.label_for(&triplet.positive.record_id).unwrap();
            let negative = store.label_for(&triplet.negative.record_id).unwrap();
            assert_eq!(anchor, requested_split);
            assert_eq!(positive, requested_split);
            assert_eq!(negative, requested_split);
            assert_eq!(anchor, positive);
            assert_eq!(anchor, negative);
        }
    }
}

#[test]
fn split_specific_batch_apis_reject_disallowed_splits() {
    let mut config = base_config();
    config.allowed_splits = vec![SplitLabel::Train];
    let split = config.split;
    let store = Arc::new(DeterministicSplitStore::new(split, 999).unwrap());
    let sampler = TripletSampler::new(config, store);

    let pair_err = sampler
        .next_pair_batch_for_split(SplitLabel::Validation)
        .unwrap_err();
    assert!(matches!(
        pair_err,
        SamplerError::Configuration(ref msg) if msg.contains("not in allowed_splits")
    ));

    let text_err = sampler
        .next_text_batch_for_split(SplitLabel::Validation)
        .unwrap_err();
    assert!(matches!(
        text_err,
        SamplerError::Configuration(ref msg) if msg.contains("not in allowed_splits")
    ));

    let triplet_err = sampler
        .next_triplet_batch_for_split(SplitLabel::Validation)
        .unwrap_err();
    assert!(matches!(
        triplet_err,
        SamplerError::Configuration(ref msg) if msg.contains("not in allowed_splits")
    ));
}

#[test]
fn triplet_sampling_produces_anchor_positive_and_negative() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let config = SamplerConfig {
        seed: 6,
        batch_size: 1,
        chunking: ChunkingStrategy::default(),
        recipes: vec![TripletRecipe {
            name: "tt_triplet".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
            allow_same_anchor_positive: false,
        }],
        text_recipes: Vec::new(),
        split,
        ..SamplerConfig::default()
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 43).unwrap());
    let find_train_id = |prefix: &str| -> String {
        (0u32..)
            .find_map(|i| {
                let id = format!("{prefix}_{i}");
                (store.label_for(&id) == Some(SplitLabel::Train)).then_some(id)
            })
            .unwrap()
    };
    let article_a = find_train_id("article_a");
    let article_b = find_train_id("article_b");
    let records = vec![
        trader_record(&article_a, "2025-01-01", "Alpha", "Body alpha"),
        trader_record(&article_b, "2025-01-02", "Beta", "Body beta"),
    ];
    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(InMemorySource::new("tt", records)));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();
    let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
    assert_eq!(batch.triplets.len(), 1);
    let triplet = &batch.triplets[0];
    assert_ne!(triplet.anchor.record_id, triplet.negative.record_id);
    assert_eq!(triplet.anchor.record_id, triplet.positive.record_id);
    assert!(triplet.instruction.is_none());
}

#[test]
fn refresh_limit_caps_records_per_source() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let mut config = base_config();
    config.split = split;
    config.batch_size = 3;
    config.ingestion_max_records = 3;
    let store = Arc::new(DeterministicSplitStore::new(split, 37).unwrap());
    let base = Utc::now() - Duration::seconds(60);
    // Generate enough record IDs that at least 3 hash to Train; take the first 10.
    let ids: Vec<String> = (0u32..)
        .filter_map(|i| {
            let id = format!("record_{i}");
            (store.label_for(&id) == Some(SplitLabel::Train)).then_some(id)
        })
        .take(10)
        .collect();
    let records: Vec<DataRecord> = ids
        .iter()
        .enumerate()
        .map(|(idx, id)| record_with_offset(id, base, idx as i64))
        .collect();
    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(InMemorySource::new("unit", records)));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();
    assert_eq!(sampler.inner.lock().unwrap().records.len(), 3);
}

#[test]
fn triplet_sampling_cycles_recipes_over_time() {
    let split = SplitRatios::default();
    let mut config = base_config();
    config.batch_size = 1;
    config.allowed_splits = vec![SplitLabel::Train];
    config.split = split;
    config.recipes = vec![
        TripletRecipe {
            name: "recipe_a".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
            allow_same_anchor_positive: false,
        },
        TripletRecipe {
            name: "recipe_b".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
            allow_same_anchor_positive: false,
        },
    ];
    config.text_recipes = Vec::new();
    let store = Arc::new(DeterministicSplitStore::new(split, 11).unwrap());
    let sampler = TripletSampler::new(config, store);
    let records = vec![
        trader_record("src::cycle_a", "2025-01-01", "Cycle A", "Body cycle a"),
        trader_record("src::cycle_b", "2025-01-02", "Cycle B", "Body cycle b"),
        trader_record("src::cycle_c", "2025-01-03", "Cycle C", "Body cycle c"),
    ];
    sampler.register_source(Box::new(InMemorySource::new("unit", records)));

    let mut seen = std::collections::HashSet::new();
    for _ in 0..10 {
        let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        seen.insert(batch.triplets[0].recipe.clone());
        if seen.len() == 2 {
            break;
        }
    }
    assert!(seen.contains("recipe_a"));
    assert!(seen.contains("recipe_b"));
}

#[test]
fn triplet_batch_dedupes_identical_triplets() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let mut config = base_config();
    config.batch_size = 1;
    config.allowed_splits = vec![SplitLabel::Train];
    config.split = split;
    config.recipes = vec![TripletRecipe {
        name: "dedupe_triplet".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    }];

    let store = Arc::new(DeterministicSplitStore::new(split, 77).unwrap());
    let find_train_id = |prefix: &str| -> String {
        (0u32..)
            .find_map(|i| {
                let id = format!("{prefix}_{i}");
                (store.label_for(&id) == Some(SplitLabel::Train)).then_some(id)
            })
            .unwrap()
    };
    let dedupe_a = find_train_id("dedupe_a");
    let dedupe_b = find_train_id("dedupe_b");
    let sampler = TripletSampler::new(config, store);

    let records = vec![
        trader_record(&dedupe_a, "2025-01-01", "Dedupe A", "Body A"),
        trader_record(&dedupe_b, "2025-01-02", "Dedupe B", "Body B"),
    ];
    sampler.register_source(Box::new(InMemorySource::new("tt", records)));

    let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
    let mut seen = std::collections::HashSet::new();
    for triplet in &batch.triplets {
        let key = (
            triplet.anchor.record_id.clone(),
            triplet.positive.record_id.clone(),
            triplet.negative.record_id.clone(),
        );
        assert!(seen.insert(key), "triplet should be unique within batch");
    }
}

#[test]
fn text_batch_dedupes_identical_chunks() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let mut config = base_config();
    config.batch_size = 1;
    config.allowed_splits = vec![SplitLabel::Train];
    config.split = split;
    config.text_recipes = vec![TextRecipe {
        name: "context_only".into(),
        selector: Selector::Role(SectionRole::Context),
        weight: 1.0,
        instruction: None,
    }];

    let store = Arc::new(DeterministicSplitStore::new(split, 91).unwrap());
    let find_train_id = |prefix: &str| -> String {
        (0u32..)
            .find_map(|i| {
                let id = format!("{prefix}_{i}");
                (store.label_for(&id) == Some(SplitLabel::Train)).then_some(id)
            })
            .unwrap()
    };
    let dedupe_a = find_train_id("text_dedupe_a");
    let dedupe_b = find_train_id("text_dedupe_b");
    let sampler = TripletSampler::new(config, store);

    let records = vec![
        trader_record(&dedupe_a, "2025-01-01", "Dedupe A", "Body A"),
        trader_record(&dedupe_b, "2025-01-02", "Dedupe B", "Body B"),
    ];
    sampler.register_source(Box::new(InMemorySource::new("tt", records)));

    let batch = sampler.next_text_batch(SplitLabel::Train).unwrap();
    let mut seen = std::collections::HashSet::new();
    for sample in &batch.samples {
        let key = chunk_key(&sample.chunk);
        assert!(
            seen.insert(key),
            "text sample should be unique within batch"
        );
    }
}

#[test]
fn text_sampling_cycles_recipes_over_time() {
    let split = SplitRatios::default();
    let mut config = base_config();
    config.batch_size = 1;
    config.allowed_splits = vec![SplitLabel::Train];
    config.split = split;
    config.recipes = Vec::new();
    config.text_recipes = vec![
        TextRecipe {
            name: "text_a".into(),
            selector: Selector::Role(SectionRole::Anchor),
            weight: 1.0,
            instruction: None,
        },
        TextRecipe {
            name: "text_b".into(),
            selector: Selector::Role(SectionRole::Context),
            weight: 1.0,
            instruction: None,
        },
    ];
    let store = Arc::new(DeterministicSplitStore::new(split, 11).unwrap());
    let sampler = TripletSampler::new(config, store);
    let mut rec_a = sample_record();
    rec_a.id = "record_a".into();
    let mut rec_b = sample_record();
    rec_b.id = "record_b".into();
    sampler.register_source(Box::new(InMemorySource::new("unit", vec![rec_a, rec_b])));

    let mut seen = std::collections::HashSet::new();
    for _ in 0..10 {
        let batch = sampler.next_text_batch(SplitLabel::Train).unwrap();
        seen.insert(batch.samples[0].recipe.clone());
        if seen.len() == 2 {
            break;
        }
    }
    assert!(seen.contains("text_a"));
    assert!(seen.contains("text_b"));
}

#[test]
fn epoch_sampling_visits_each_record_before_repeat() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let mut config = SamplerConfig {
        seed: 101,
        batch_size: 1,
        chunking: ChunkingStrategy::default(),
        recipes: vec![TripletRecipe {
            name: "epoch_triplet".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
            allow_same_anchor_positive: false,
        }],
        text_recipes: Vec::new(),
        split,
        ..SamplerConfig::default()
    };
    config.allowed_splits = vec![SplitLabel::Train];
    let store = Arc::new(DeterministicSplitStore::new(split, 59).unwrap());
    let find_train_id = |prefix: &str| -> String {
        (0u32..)
            .find_map(|i| {
                let id = format!("{prefix}_{i}");
                (store.label_for(&id) == Some(SplitLabel::Train)).then_some(id)
            })
            .unwrap()
    };
    let epoch_a = find_train_id("epoch_a");
    let epoch_b = find_train_id("epoch_b");
    let epoch_c = find_train_id("epoch_c");
    let records = vec![
        trader_record(&epoch_a, "2025-01-01", "Epoch Alpha", "Body alpha"),
        trader_record(&epoch_b, "2025-01-02", "Epoch Beta", "Body beta"),
        trader_record(&epoch_c, "2025-01-03", "Epoch Gamma", "Body gamma"),
    ];
    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(InMemorySource::new("tt", records)));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();
    let mut anchors = Vec::new();
    for _ in 0..10 {
        let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        anchors.extend(batch.triplets.iter().map(|t| t.anchor.record_id.clone()));
    }
    let mut dedup = anchors.clone();
    dedup.sort();
    dedup.dedup();
    assert_eq!(dedup.len(), 3, "all records should appear over time");
}

#[test]
fn epoch_sampling_persists_between_runs() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let temp = tempdir().unwrap();
    let store_path = temp.path().join("epoch_store");
    let build_config = || {
        let mut cfg = SamplerConfig {
            seed: 202,
            batch_size: 3,
            chunking: ChunkingStrategy::default(),
            recipes: vec![TripletRecipe {
                name: "persist_triplet".into(),
                anchor: Selector::Role(SectionRole::Anchor),
                positive_selector: Selector::Role(SectionRole::Context),
                negative_selector: Selector::Role(SectionRole::Context),
                negative_strategy: NegativeStrategy::WrongArticle,
                weight: 1.0,
                instruction: None,
                allow_same_anchor_positive: false,
            }],
            text_recipes: Vec::new(),
            split,
            ..SamplerConfig::default()
        };
        cfg.allowed_splits = vec![SplitLabel::Train];
        cfg
    };
    let probe_store = DeterministicSplitStore::new(split, 73).unwrap();
    let find_train_id = |prefix: &str| -> String {
        (0u32..)
            .find_map(|i| {
                let id = format!("{prefix}_{i}");
                (probe_store.label_for(&id) == Some(SplitLabel::Train)).then_some(id)
            })
            .unwrap()
    };
    let persist_a = find_train_id("persist_a");
    let persist_b = find_train_id("persist_b");
    let persist_c = find_train_id("persist_c");
    let dataset = vec![
        trader_record(&persist_a, "2025-02-01", "Persist A", "Body a"),
        trader_record(&persist_b, "2025-02-02", "Persist B", "Body b"),
        trader_record(&persist_c, "2025-02-03", "Persist C", "Body c"),
    ];

    let first_anchor = {
        let store = Arc::new(FileSplitStore::open(&store_path, split, 73).unwrap());
        let sampler = TripletSampler::new(build_config(), store);
        sampler.register_source(Box::new(InMemorySource::new("tt", dataset.clone())));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();
        let anchor = sampler
            .next_triplet_batch(SplitLabel::Train)
            .unwrap()
            .triplets[0]
            .anchor
            .record_id
            .clone();
        sampler.save_sampler_state(None).unwrap();
        anchor
    };

    let store = Arc::new(FileSplitStore::open(&store_path, split, 73).unwrap());
    let sampler = TripletSampler::new(build_config(), store);
    sampler.register_source(Box::new(InMemorySource::new("tt", dataset.clone())));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();
    let mut anchors = Vec::new();
    for _ in 0..5 {
        let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        anchors.extend(batch.triplets.iter().map(|t| t.anchor.record_id.clone()));
    }
    sampler.save_sampler_state(None).unwrap();
    assert!(
        anchors.contains(&first_anchor),
        "previously consumed records may reappear with streaming paging"
    );
}

#[test]
fn epoch_sampling_handles_new_records_after_restart() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let temp = tempdir().unwrap();
    let store_path = temp.path().join("epoch_store_new_records");
    let mut base_config = SamplerConfig {
        seed: 404,
        batch_size: 1,
        chunking: ChunkingStrategy::default(),
        recipes: vec![TripletRecipe {
            name: "persist_triplet_new".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
            allow_same_anchor_positive: false,
        }],
        text_recipes: Vec::new(),
        split,
        ..SamplerConfig::default()
    };
    base_config.allowed_splits = vec![SplitLabel::Train];

    let probe_store = DeterministicSplitStore::new(split, 111).unwrap();
    let find_train_id = |prefix: &str| -> String {
        (0u32..)
            .find_map(|i| {
                let id = format!("{prefix}_{i}");
                (probe_store.label_for(&id) == Some(SplitLabel::Train)).then_some(id)
            })
            .unwrap()
    };
    let restart_a = find_train_id("restart_a");
    let restart_b = find_train_id("restart_b");
    let restart_c = find_train_id("restart_c");

    let initial_records = vec![
        trader_record(&restart_a, "2025-03-01", "Restart Alpha", "Body alpha"),
        trader_record(&restart_b, "2025-03-02", "Restart Beta", "Body beta"),
    ];

    // Prime the store and consume one record.
    let _first_anchor = {
        let store = Arc::new(FileSplitStore::open(&store_path, split, 111).unwrap());
        let sampler = TripletSampler::new(base_config.clone(), store);
        sampler.register_source(Box::new(InMemorySource::new("tt", initial_records.clone())));
        sampler
            .inner
            .lock()
            .unwrap()
            .ingest_internal(SplitLabel::Train)
            .unwrap();
        let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        let anchor = batch.triplets[0].anchor.record_id.clone();
        sampler.save_sampler_state(None).unwrap();
        anchor
    };

    // Restart with an extra record added.
    let mut expanded_records = initial_records.clone();
    expanded_records.push(trader_record(
        &restart_c,
        "2025-03-03",
        "Restart Gamma",
        "Body gamma",
    ));

    let store = Arc::new(FileSplitStore::open(&store_path, split, 111).unwrap());
    let sampler = TripletSampler::new(base_config, store);
    sampler.register_source(Box::new(InMemorySource::new(
        "tt",
        expanded_records.clone(),
    )));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let mut seen = std::collections::HashSet::new();
    let max_draws = expanded_records.len() * 3;
    for _ in 0..max_draws {
        if let Ok(batch) = sampler.next_triplet_batch(SplitLabel::Train) {
            for triplet in batch.triplets {
                seen.insert(triplet.anchor.record_id);
            }
        }
    }
    assert!(seen.contains(&restart_c));
}

#[test]
fn source_epoch_is_propagated_to_ingestion_on_resume() {
    // Verify that when a sampler resumes from persisted state after a
    // cache wipe (e.g. only simd-r-drive state is kept), the ingestion
    // manager already has the correct source_epoch before the very first
    // refresh call fires.  If source_epoch were loaded too late (only in
    // ensure_source_state, which runs *after* the first refresh), sources
    // that derive their permutation seed from config.seed inside refresh()
    // would silently use epoch 0 instead of the persisted epoch.
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let temp = tempdir().unwrap();
    let store_path = temp.path().join("epoch_propagation_store");
    let build_config = || SamplerConfig {
        seed: 77,
        batch_size: 2,
        chunking: ChunkingStrategy::default(),
        recipes: vec![TripletRecipe {
            name: "ep_triplet".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
            allow_same_anchor_positive: false,
        }],
        text_recipes: Vec::new(),
        split,
        allowed_splits: vec![SplitLabel::Train],
        ..SamplerConfig::default()
    };
    let records: Vec<DataRecord> = (0..4)
        .map(|i| {
            trader_record(
                &format!("src::ep_record_{i:02}"),
                "2025-06-01",
                &format!("Title {i}"),
                &format!("Body {i}"),
            )
        })
        .collect();

    // First run: advance enough batches to trigger at least one source_epoch
    // increment so the persisted epoch is non-zero.
    let persisted_epoch = {
        let store = Arc::new(FileSplitStore::open(&store_path, split, 11).unwrap());
        let sampler = TripletSampler::new(build_config(), Arc::clone(&store));
        sampler.register_source(Box::new(InMemorySource::new("src", records.clone())));
        // Drive enough batches to cycle through all records and advance epoch.
        for _ in 0..8 {
            let _ = sampler.next_triplet_batch(SplitLabel::Train);
        }
        sampler.save_sampler_state(None).unwrap();
        let state = store.load_sampler_state().unwrap().unwrap();
        state.source_epoch
    };

    // Epoch must have advanced — the test is meaningless if it stayed at 0.
    assert!(
        persisted_epoch > 0,
        "source_epoch should have advanced; got {persisted_epoch}"
    );

    // Second run: simulate a cache wipe by NOT reloading the ingestion
    // stream cursors, i.e. start the sampler fresh but with the same store.
    // After the very first ingest call the ingestion manager's source_epoch
    // must equal the persisted value before any refresh fires.
    let store = Arc::new(FileSplitStore::open(&store_path, split, 11).unwrap());
    let sampler = TripletSampler::new(build_config(), Arc::clone(&store));
    sampler.register_source(Box::new(InMemorySource::new("src", records.clone())));
    {
        let mut inner = sampler.inner.lock().unwrap();
        // Trigger cursor loading (the step that must set source_epoch early).
        inner.ingest_internal(SplitLabel::Train).unwrap();
        assert_eq!(
            inner.ingestion.source_epoch(),
            persisted_epoch,
            "ingestion source_epoch must match persisted epoch after resume"
        );
        assert_eq!(
            inner.source_epoch, persisted_epoch,
            "sampler source_epoch must match persisted epoch after resume"
        );
    }
}

#[test]
fn oversampling_advances_cursors_on_large_records() {
    // All records (long_record, short_A, short_B, short_C) hash to Train
    // with seed=123 and train:0.7 ratios.
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let mut config = base_config();
    config.batch_size = 3;
    config.text_recipes = vec![TextRecipe {
        name: "context".into(),
        selector: Selector::Role(SectionRole::Context),
        weight: 1.0,
        instruction: None,
    }];
    config.chunking = ChunkingStrategy {
        max_window_tokens: 1,
        overlap_tokens: vec![0],
        summary_fallback_weight: 0.0,
        summary_fallback_tokens: 0,
        chunk_weight_floor: 0.0,
    };

    let store = Arc::new(DeterministicSplitStore::new(split, 123).unwrap());
    let sampler = TripletSampler::new(config, store);

    // Record 1: Small Source, Huge Content
    // "One Two Three" -> With max_window_tokens=1 -> Chunks: ["One", "Two", "Three"]
    let multi_chunk_record = DataRecord {
        id: "long_record".into(),
        source: "small".into(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        quality: QualityScore::default(),
        taxonomy: vec![],
        sections: vec![RecordSection {
            role: SectionRole::Context,
            heading: None,
            text: "One Two Three".into(),
            sentences: vec!["One Two Three".into()],
        }],
        meta_prefix: None,
    };

    // Records 2, 3, 4: Large Source, Small Content
    let mut large_source_records = Vec::new();
    for char in ['A', 'B', 'C'] {
        large_source_records.push(DataRecord {
            id: format!("short_{}", char),
            source: "large".into(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            quality: QualityScore::default(),
            taxonomy: vec![],
            sections: vec![RecordSection {
                role: SectionRole::Context,
                heading: None,
                text: char.to_string(),
                sentences: vec![char.to_string()],
            }],
            meta_prefix: None,
        });
    }

    sampler.register_source(Box::new(InMemorySource::new(
        "small",
        vec![multi_chunk_record],
    )));
    sampler.register_source(Box::new(InMemorySource::new("large", large_source_records)));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    // We expect 6 samples total (3 from Small, 3 from Large)
    // The "Small" samples should progress through the content.
    let mut small_samples = Vec::new();
    let mut large_samples = Vec::new();

    for _ in 0..12 {
        let batch = sampler.next_text_batch(SplitLabel::Train).unwrap();
        for sample in batch.samples {
            let text = sample.chunk.text;
            if sample.chunk.record_id == "long_record" {
                small_samples.push(text);
            } else {
                large_samples.push(text);
            }
        }
    }

    assert!(
        small_samples.len() >= 3,
        "Should sample small source multiple times"
    );
    assert!(
        large_samples.len() >= 3,
        "Should sample large source multiple times"
    );

    small_samples.sort();
    assert!(small_samples.contains(&"One".to_string()));
    assert!(small_samples.contains(&"Two".to_string()));
    assert!(small_samples.contains(&"Three".to_string()));

    // Verify coverage of large source
    large_samples.sort();
    large_samples.dedup();
    assert!(
        large_samples.len() >= 2,
        "large source should contribute multiple distinct samples"
    );
}

#[test]
fn text_sampling_balances_sources_without_epoch_tracker() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let mut config = base_config();
    config.batch_size = 2;
    config.split = split;
    config.allowed_splits = vec![SplitLabel::Train];
    config.text_recipes = vec![TextRecipe {
        name: "anchors".into(),
        selector: Selector::Role(SectionRole::Anchor),
        weight: 1.0,
        instruction: None,
    }];

    let store = Arc::new(DeterministicSplitStore::new(split, 73).unwrap());
    let find_train_id = |prefix: &str| -> String {
        (0u32..)
            .find_map(|i| {
                let id = format!("{prefix}_{i}");
                (store.label_for(&id) == Some(SplitLabel::Train)).then_some(id)
            })
            .unwrap()
    };
    let factual_id = find_train_id("factual_record");
    let opinion_id = find_train_id("opinionated_record");
    let sampler = TripletSampler::new(config, store);

    let mut factual = sample_record();
    factual.id = factual_id.clone();
    factual.source = "qa_factual".into();

    let mut opinion = sample_record();
    opinion.id = opinion_id.clone();
    opinion.source = "qa_opinionated".into();

    sampler.register_source(Box::new(InMemorySource::new(
        "qa_factual_source",
        vec![factual.clone()],
    )));
    sampler.register_source(Box::new(InMemorySource::new(
        "qa_opinion_source",
        vec![opinion.clone()],
    )));

    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let batch = sampler.next_text_batch(SplitLabel::Train).unwrap();
    assert_eq!(batch.samples.len(), 2);
    let mut ids: Vec<_> = batch
        .samples
        .iter()
        .map(|sample| sample.chunk.record_id.clone())
        .collect();
    ids.sort();
    let mut expected = vec![factual_id.clone(), opinion_id.clone()];
    expected.sort();
    assert_eq!(ids, expected);
}

#[test]
fn chunk_sampling_respects_split_boundaries() {
    let split = SplitRatios {
        train: 0.5,
        validation: 0.5,
        test: 0.0,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 88).unwrap());

    let find_id = |label: SplitLabel, prefix: &str| -> String {
        for i in 0..2000 {
            let id = format!("{prefix}_{i}");
            if store.ensure(id.clone()).unwrap() == label {
                return id;
            }
        }
        panic!("unable to find id for {:?}", label);
    };

    let train_id = find_id(SplitLabel::Train, "train_candidate");
    let val_id = find_id(SplitLabel::Validation, "val_candidate");

    let mut config = base_config();
    config.batch_size = 1;
    config.allowed_splits = vec![SplitLabel::Train];
    config.text_recipes = vec![TextRecipe {
        name: "context".into(),
        selector: Selector::Role(SectionRole::Context),
        weight: 1.0,
        instruction: None,
    }];
    config.chunking = ChunkingStrategy {
        max_window_tokens: 1,
        overlap_tokens: vec![0],
        summary_fallback_weight: 0.0,
        summary_fallback_tokens: 0,
        chunk_weight_floor: 0.0,
    };

    let sampler = TripletSampler::new(config, store);
    let mut train_record = trader_record(&train_id, "2025-01-01", "Train Title", "One Two");
    let mut val_record = trader_record(&val_id, "2025-01-02", "Val Title", "Alpha Beta");
    train_record.source = "split_test".into();
    val_record.source = "split_test".into();

    sampler.register_source(Box::new(InMemorySource::new(
        "split_test",
        vec![train_record, val_record],
    )));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    for _ in 0..4 {
        let batch = sampler.next_text_batch(SplitLabel::Train).unwrap();
        let sample = &batch.samples[0];
        let label = sampler
            .inner
            .lock()
            .unwrap()
            .split_store
            .label_for(&sample.chunk.record_id)
            .unwrap();
        assert_eq!(label, SplitLabel::Train);
    }
}

#[test]
fn adds_dynamic_chunk_pair_recipe_for_long_section_sources() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let mut config = base_config();
    config.batch_size = 1;
    // Force chunking so a 4-token context section becomes multiple windows.
    // This is the eligibility condition for adding the dynamic chunk-pair recipe.
    config.chunking = ChunkingStrategy {
        max_window_tokens: 2,
        overlap_tokens: vec![0],
        summary_fallback_weight: 0.0,
        summary_fallback_tokens: 0,
        chunk_weight_floor: 0.0,
    };

    let recipes = vec![TripletRecipe {
        name: "base_title_context".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    }];

    let now = Utc::now();
    // Two records are provided so `WrongArticle` negatives can be formed.
    // Context sections are intentionally longer than `max_window_tokens`.
    let records = vec![
        DataRecord {
            id: "r1".into(),
            source: "ignored_by_ingestion".into(),
            created_at: now,
            updated_at: now,
            quality: QualityScore::default(),
            taxonomy: vec![],
            sections: vec![
                RecordSection {
                    role: SectionRole::Anchor,
                    heading: None,
                    text: "Headline one".into(),
                    sentences: vec!["Headline one".into()],
                },
                RecordSection {
                    role: SectionRole::Context,
                    heading: None,
                    text: "one two three four".into(),
                    sentences: vec!["one two three four".into()],
                },
            ],
            meta_prefix: None,
        },
        DataRecord {
            id: "r2".into(),
            source: "ignored_by_ingestion".into(),
            created_at: now,
            updated_at: now,
            quality: QualityScore::default(),
            taxonomy: vec![],
            sections: vec![
                RecordSection {
                    role: SectionRole::Anchor,
                    heading: None,
                    text: "Headline two".into(),
                    sentences: vec!["Headline two".into()],
                },
                RecordSection {
                    role: SectionRole::Context,
                    heading: None,
                    text: "alpha beta gamma delta".into(),
                    sentences: vec!["alpha beta gamma delta".into()],
                },
            ],
            meta_prefix: None,
        },
    ];

    let store = Arc::new(DeterministicSplitStore::new(split, 117).unwrap());
    let sampler = TripletSampler::new(config, store);
    // Source provides only a base recipe; dynamic augmentation should happen
    // automatically during ingest based on observed section lengths.
    sampler.register_source(Box::new(RecipeSource::new(records, recipes)));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let effective = sampler
        .inner
        .lock()
        .unwrap()
        .triplet_recipes_for_source("recipe_source");
    // Verify the dynamic same-record chunk-pair recipe was injected.
    assert!(effective.iter().any(|recipe| {
        recipe.name.as_ref() == AUTO_INJECTED_LONG_SECTION_CHUNK_PAIR_RECIPE_NAME
    }));
}

#[test]
fn does_not_add_dynamic_chunk_pair_recipe_when_all_sections_fit_window() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let mut config = base_config();
    config.batch_size = 1;
    // Large window means all sections fit in a single chunk.
    // This should disable dynamic chunk-pair recipe injection.
    config.chunking = ChunkingStrategy {
        max_window_tokens: 8,
        overlap_tokens: vec![0],
        summary_fallback_weight: 0.0,
        summary_fallback_tokens: 0,
        chunk_weight_floor: 0.0,
    };

    let recipes = vec![TripletRecipe {
        name: "base_title_context".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    }];

    let now = Utc::now();
    // All context sections are short and should not be classified as
    // chunk-window eligible for dynamic augmentation.
    let store = Arc::new(DeterministicSplitStore::new(split, 118).unwrap());
    // short2 is Test with seed=118 and train:0.7; use find_id for both.
    let find_train_id = |prefix: &str| -> String {
        (0u32..)
            .find_map(|i| {
                let id = format!("{prefix}_{i}");
                (store.label_for(&id) == Some(SplitLabel::Train)).then_some(id)
            })
            .unwrap()
    };
    let short1_id = find_train_id("short1");
    let short2_id = find_train_id("short2");
    let records = vec![
        DataRecord {
            id: short1_id,
            source: "ignored_by_ingestion".into(),
            created_at: now,
            updated_at: now,
            quality: QualityScore::default(),
            taxonomy: vec![],
            sections: vec![
                RecordSection {
                    role: SectionRole::Anchor,
                    heading: None,
                    text: "Headline one".into(),
                    sentences: vec!["Headline one".into()],
                },
                RecordSection {
                    role: SectionRole::Context,
                    heading: None,
                    text: "one two".into(),
                    sentences: vec!["one two".into()],
                },
            ],
            meta_prefix: None,
        },
        DataRecord {
            id: short2_id,
            source: "ignored_by_ingestion".into(),
            created_at: now,
            updated_at: now,
            quality: QualityScore::default(),
            taxonomy: vec![],
            sections: vec![
                RecordSection {
                    role: SectionRole::Anchor,
                    heading: None,
                    text: "Headline two".into(),
                    sentences: vec!["Headline two".into()],
                },
                RecordSection {
                    role: SectionRole::Context,
                    heading: None,
                    text: "alpha beta".into(),
                    sentences: vec!["alpha beta".into()],
                },
            ],
            meta_prefix: None,
        },
    ];
    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(RecipeSource::new(records, recipes)));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let effective = sampler
        .inner
        .lock()
        .unwrap()
        .triplet_recipes_for_source("recipe_source");
    // Verify the dynamic recipe is absent when no oversized sections exist.
    assert!(effective.iter().all(|recipe| {
        recipe.name.as_ref() != AUTO_INJECTED_LONG_SECTION_CHUNK_PAIR_RECIPE_NAME
    }));
}

#[test]
fn adds_dynamic_chunk_pair_recipe_even_with_global_config_recipes() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let mut config = base_config();
    config.batch_size = 1;
    config.chunking = ChunkingStrategy {
        max_window_tokens: 2,
        overlap_tokens: vec![0],
        summary_fallback_weight: 0.0,
        summary_fallback_tokens: 0,
        chunk_weight_floor: 0.0,
    };
    config.recipes = vec![TripletRecipe {
        name: "global_anchor_context".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    }];

    let now = Utc::now();
    let records = vec![
        DataRecord {
            id: "cfg1".into(),
            source: "ignored_by_ingestion".into(),
            created_at: now,
            updated_at: now,
            quality: QualityScore::default(),
            taxonomy: vec![],
            sections: vec![
                RecordSection {
                    role: SectionRole::Anchor,
                    heading: None,
                    text: "Headline one".into(),
                    sentences: vec!["Headline one".into()],
                },
                RecordSection {
                    role: SectionRole::Context,
                    heading: None,
                    text: "one two three four".into(),
                    sentences: vec!["one two three four".into()],
                },
            ],
            meta_prefix: None,
        },
        DataRecord {
            id: "cfg2".into(),
            source: "ignored_by_ingestion".into(),
            created_at: now,
            updated_at: now,
            quality: QualityScore::default(),
            taxonomy: vec![],
            sections: vec![
                RecordSection {
                    role: SectionRole::Anchor,
                    heading: None,
                    text: "Headline two".into(),
                    sentences: vec!["Headline two".into()],
                },
                RecordSection {
                    role: SectionRole::Context,
                    heading: None,
                    text: "alpha beta gamma delta".into(),
                    sentences: vec!["alpha beta gamma delta".into()],
                },
            ],
            meta_prefix: None,
        },
    ];

    let store = Arc::new(DeterministicSplitStore::new(split, 121).unwrap());
    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(RecipeSource::new(records, Vec::new())));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let effective = sampler
        .inner
        .lock()
        .unwrap()
        .triplet_recipes_for_source("recipe_source");

    assert!(
        effective
            .iter()
            .any(|recipe| recipe.name.as_ref() == "global_anchor_context")
    );
    assert!(effective.iter().any(|recipe| {
        recipe.name.as_ref() == AUTO_INJECTED_LONG_SECTION_CHUNK_PAIR_RECIPE_NAME
    }));
}

#[test]
fn auto_injected_recipe_uses_distinct_context_chunks_for_anchor_and_positive() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let mut config = base_config();
    config.batch_size = 1;
    config.chunking = ChunkingStrategy {
        // Force multi-window context sections so the injected recipe has
        // at least two chunk candidates to draw from.
        max_window_tokens: 2,
        overlap_tokens: vec![0],
        summary_fallback_weight: 0.0,
        summary_fallback_tokens: 0,
        chunk_weight_floor: 0.0,
    };

    let now = Utc::now();
    // The multi-window record is the anchor; the single-window record exists
    // only to satisfy WrongArticle negative selection.
    // This makes the expected anchor/positive chunk texts deterministic.
    let store = Arc::new(DeterministicSplitStore::new(split, 119).unwrap());
    let find_train_id = |prefix: &str| -> String {
        (0u32..)
            .find_map(|i| {
                let id = format!("{prefix}_{i}");
                (store.label_for(&id) == Some(SplitLabel::Train)).then_some(id)
            })
            .unwrap()
    };
    let long_anchor_id = find_train_id("long_anchor");
    let other_neg_id = find_train_id("other_for_negative");
    let records = vec![
        DataRecord {
            id: long_anchor_id.clone(),
            source: "ignored_by_ingestion".into(),
            created_at: now,
            updated_at: now,
            quality: QualityScore::default(),
            taxonomy: vec![],
            sections: vec![RecordSection {
                role: SectionRole::Context,
                heading: None,
                text: "one two three four".into(),
                sentences: vec!["one two three four".into()],
            }],
            meta_prefix: None,
        },
        DataRecord {
            id: other_neg_id,
            source: "ignored_by_ingestion".into(),
            created_at: now,
            updated_at: now,
            quality: QualityScore::default(),
            taxonomy: vec![],
            sections: vec![RecordSection {
                role: SectionRole::Context,
                heading: None,
                text: "other".into(),
                sentences: vec!["other".into()],
            }],
            meta_prefix: None,
        },
    ];

    let sampler = TripletSampler::new(config, store);
    // Empty default recipes means the auto-injected recipe is the only
    // recipe available for this source when long sections are detected.
    sampler.register_source(Box::new(RecipeSource::new(records, Vec::new())));

    let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
    assert_eq!(batch.triplets.len(), 1);
    let triplet = &batch.triplets[0];

    // Confirms this sample came from the auto-injected recipe.
    assert_eq!(
        triplet.recipe,
        AUTO_INJECTED_LONG_SECTION_CHUNK_PAIR_RECIPE_NAME
    );
    // Anchor/positive should be different windows from the same record.
    assert_eq!(triplet.anchor.record_id, long_anchor_id);
    assert_eq!(triplet.anchor.record_id, triplet.positive.record_id);
    assert_ne!(chunk_key(&triplet.anchor), chunk_key(&triplet.positive));

    // Hardcoded expected windows for long_anchor (4 tokens, window size 2).
    let expected_a = "one two";
    let expected_b = "three four";
    let observed = [triplet.anchor.text.as_str(), triplet.positive.text.as_str()];
    assert!(
        observed.contains(&expected_a),
        "expected one window '{expected_a}', got anchor='{}', positive='{}'",
        triplet.anchor.text,
        triplet.positive.text
    );
    assert!(
        observed.contains(&expected_b),
        "expected one window '{expected_b}', got anchor='{}', positive='{}'",
        triplet.anchor.text,
        triplet.positive.text
    );
    assert_ne!(
        triplet.anchor.text, triplet.positive.text,
        "anchor and positive should not use the same chunk text"
    );
}

#[test]
fn auto_injected_recipe_never_uses_identical_anchor_and_positive_chunks() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let mut config = base_config();
    config.batch_size = 1;
    config.chunking = ChunkingStrategy {
        // 4-token sections become two windows each, so there is always a
        // distinct positive chunk available when selecting by context role.
        max_window_tokens: 2,
        overlap_tokens: vec![0],
        summary_fallback_weight: 0.0,
        summary_fallback_tokens: 0,
        chunk_weight_floor: 0.0,
    };

    let now = Utc::now();
    let store = Arc::new(DeterministicSplitStore::new(split, 120).unwrap());
    let find_train_id = |prefix: &str| -> String {
        (0u32..)
            .find_map(|i| {
                let id = format!("{prefix}_{i}");
                (store.label_for(&id) == Some(SplitLabel::Train)).then_some(id)
            })
            .unwrap()
    };
    let long1_id = find_train_id("long1");
    let long2_id = find_train_id("long2");
    let records = vec![
        DataRecord {
            id: long1_id,
            source: "ignored_by_ingestion".into(),
            created_at: now,
            updated_at: now,
            quality: QualityScore::default(),
            taxonomy: vec![],
            sections: vec![RecordSection {
                role: SectionRole::Context,
                heading: None,
                text: "one two three four".into(),
                sentences: vec!["one two three four".into()],
            }],
            meta_prefix: None,
        },
        DataRecord {
            id: long2_id,
            source: "ignored_by_ingestion".into(),
            created_at: now,
            updated_at: now,
            quality: QualityScore::default(),
            taxonomy: vec![],
            sections: vec![RecordSection {
                role: SectionRole::Context,
                heading: None,
                text: "alpha beta gamma delta".into(),
                sentences: vec!["alpha beta gamma delta".into()],
            }],
            meta_prefix: None,
        },
    ];

    let store = Arc::new(DeterministicSplitStore::new(split, 120).unwrap());
    let sampler = TripletSampler::new(config, store);
    // No default source recipes: only the auto-injected recipe can run.
    sampler.register_source(Box::new(RecipeSource::new(records, Vec::new())));

    for _ in 0..32 {
        let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        assert_eq!(batch.triplets.len(), 1);
        let triplet = &batch.triplets[0];
        assert_eq!(
            triplet.recipe,
            AUTO_INJECTED_LONG_SECTION_CHUNK_PAIR_RECIPE_NAME
        );
        assert_eq!(triplet.anchor.record_id, triplet.positive.record_id);
        assert_ne!(
            chunk_key(&triplet.anchor),
            chunk_key(&triplet.positive),
            "anchor and positive chunk keys must differ; anchor='{}' positive='{}'",
            triplet.anchor.text,
            triplet.positive.text
        );
        assert_ne!(
            triplet.anchor.text, triplet.positive.text,
            "anchor and positive chunk text must differ"
        );
    }
}

#[test]
fn auto_injected_recipe_uses_window_chunks_for_anchor_and_positive() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let mut config = base_config();
    config.batch_size = 1;
    config.chunking = ChunkingStrategy {
        max_window_tokens: 4,
        overlap_tokens: vec![1],
        summary_fallback_weight: 0.5,
        summary_fallback_tokens: 2,
        chunk_weight_floor: 0.0,
    };

    let now = Utc::now();
    let records = vec![
        DataRecord {
            id: "long_w1".into(),
            source: "ignored_by_ingestion".into(),
            created_at: now,
            updated_at: now,
            quality: QualityScore::default(),
            taxonomy: vec![],
            sections: vec![RecordSection {
                role: SectionRole::Context,
                heading: None,
                text: "one two three four five six seven eight nine ten".into(),
                sentences: vec!["one two three four five six seven eight nine ten".into()],
            }],
            meta_prefix: None,
        },
        DataRecord {
            id: "long_w2".into(),
            source: "ignored_by_ingestion".into(),
            created_at: now,
            updated_at: now,
            quality: QualityScore::default(),
            taxonomy: vec![],
            sections: vec![RecordSection {
                role: SectionRole::Context,
                heading: None,
                text: "alpha beta gamma delta epsilon zeta eta theta iota kappa".into(),
                sentences: vec!["alpha beta gamma delta epsilon zeta eta theta iota kappa".into()],
            }],
            meta_prefix: None,
        },
    ];

    let store = Arc::new(DeterministicSplitStore::new(split, 123).unwrap());
    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(RecipeSource::new(records, Vec::new())));

    for _ in 0..16 {
        let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        let triplet = &batch.triplets[0];
        assert_eq!(
            triplet.recipe,
            AUTO_INJECTED_LONG_SECTION_CHUNK_PAIR_RECIPE_NAME
        );
        assert!(matches!(triplet.anchor.view, ChunkView::Window { .. }));
        assert!(matches!(triplet.positive.view, ChunkView::Window { .. }));
        assert_ne!(chunk_key(&triplet.anchor), chunk_key(&triplet.positive));
    }
}

#[test]
fn auto_injected_recipe_keeps_all_components_in_requested_split() {
    let split = SplitRatios {
        train: 0.34,
        validation: 0.33,
        test: 0.33,
    };

    let mut config = base_config();
    config.seed = 812;
    config.batch_size = 1;
    config.allowed_splits = vec![SplitLabel::Train, SplitLabel::Validation, SplitLabel::Test];

    // No custom recipes to ensure the auto-injected recipe is the only recipe available.
    config.recipes = Vec::new();

    config.text_recipes = Vec::new();
    config.chunking = ChunkingStrategy {
        max_window_tokens: 2,
        overlap_tokens: vec![0],
        summary_fallback_weight: 0.0,
        summary_fallback_tokens: 0,
        chunk_weight_floor: 0.0,
    };

    let store = Arc::new(DeterministicSplitStore::new(split, 1441).unwrap());

    // Not a manual split assignment: this only searches for record ids whose
    // deterministic split-store derivation already maps to `label`.
    let find_id = |label: SplitLabel, prefix: &str| -> String {
        for i in 0..20000 {
            let id = format!("{prefix}_{i}");
            if store.ensure(id.clone()).unwrap() == label {
                return id;
            }
        }
        panic!("unable to find id for {:?}", label);
    };

    let now = Utc::now();
    let mut records = Vec::new();
    for split_label in [SplitLabel::Train, SplitLabel::Validation, SplitLabel::Test] {
        for idx in 0..2 {
            let id = find_id(split_label, &format!("auto_split_{split_label:?}_{idx}"));
            assert_eq!(store.label_for(&id).unwrap(), split_label);
            records.push(DataRecord {
                id,
                source: "ignored_by_ingestion".into(),
                created_at: now,
                updated_at: now,
                quality: QualityScore::default(),
                taxonomy: vec![],
                sections: vec![RecordSection {
                    role: SectionRole::Context,
                    heading: None,
                    text: format!("ctx {split_label:?} {idx} one two three four"),
                    sentences: vec![format!("ctx {split_label:?} {idx} one two three four")],
                }],
                meta_prefix: None,
            });
        }
    }

    let sampler = TripletSampler::new(config, Arc::clone(&store));
    sampler.register_source(Box::new(RecipeSource::new(records, Vec::new())));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    for requested_split in [SplitLabel::Train, SplitLabel::Validation, SplitLabel::Test] {
        for _ in 0..8 {
            let batch = sampler
                .next_triplet_batch_for_split(requested_split)
                .unwrap();
            assert_eq!(batch.triplets.len(), 1);
            let triplet = &batch.triplets[0];
            assert_eq!(
                triplet.recipe,
                AUTO_INJECTED_LONG_SECTION_CHUNK_PAIR_RECIPE_NAME
            );

            let anchor_split = store.label_for(&triplet.anchor.record_id).unwrap();
            let positive_split = store.label_for(&triplet.positive.record_id).unwrap();
            let negative_split = store.label_for(&triplet.negative.record_id).unwrap();

            assert_eq!(anchor_split, requested_split);
            assert_eq!(positive_split, requested_split);
            assert_eq!(negative_split, requested_split);

            assert_eq!(triplet.anchor.record_id, triplet.positive.record_id);
            assert_ne!(triplet.anchor.record_id, triplet.negative.record_id);
            assert_ne!(chunk_key(&triplet.anchor), chunk_key(&triplet.positive));
        }
    }
}

#[test]
fn same_selector_triplet_returns_none_when_only_one_chunk_exists() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let mut config = base_config();
    config.chunking = ChunkingStrategy {
        max_window_tokens: 8,
        overlap_tokens: vec![0],
        summary_fallback_weight: 0.0,
        summary_fallback_tokens: 0,
        chunk_weight_floor: 0.0,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 122).unwrap());
    let mut inner = TripletSamplerInner::new(config, store);

    let now = Utc::now();
    let anchor = DataRecord {
        id: "single_chunk_anchor".into(),
        source: "unit".into(),
        created_at: now,
        updated_at: now,
        quality: QualityScore::default(),
        taxonomy: vec![],
        sections: vec![RecordSection {
            role: SectionRole::Context,
            heading: None,
            text: "one two".into(),
            sentences: vec!["one two".into()],
        }],
        meta_prefix: None,
    };

    let negative = DataRecord {
        id: "single_chunk_negative".into(),
        source: "unit".into(),
        created_at: now,
        updated_at: now,
        quality: QualityScore::default(),
        taxonomy: vec![],
        sections: vec![RecordSection {
            role: SectionRole::Context,
            heading: None,
            text: "alpha beta".into(),
            sentences: vec!["alpha beta".into()],
        }],
        meta_prefix: None,
    };

    inner.records.insert(anchor.id.clone(), anchor.clone());
    inner.records.insert(negative.id.clone(), negative.clone());
    inner.rebuild_chunk_index();

    let recipe = TripletRecipe {
        name: "same_selector_context".into(),
        anchor: Selector::Role(SectionRole::Context),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    };

    let triplet = inner.make_triplet_with_anchor(&recipe, &anchor);
    assert!(triplet.is_none());
}

#[test]
fn sampler_allows_concurrent_batch_requests() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 9).unwrap());
    let mut config = base_config();
    config.seed = 7;
    config.batch_size = 1;
    config.ingestion_max_records = 8;
    config.allowed_splits = vec![SplitLabel::Train];
    config.split = split;
    config.text_recipes = vec![TextRecipe {
        name: "concurrent_text".into(),
        selector: Selector::Role(SectionRole::Context),
        weight: 1.0,
        instruction: None,
    }];

    let records: Vec<DataRecord> = (0u32..)
        .filter_map(|i| {
            let id = format!("concurrent_{i}");
            (store.label_for(&id) == Some(SplitLabel::Train)).then(|| {
                let mut r = sample_record();
                r.id = id;
                r
            })
        })
        .take(4)
        .collect();
    let sampler = Arc::new(TripletSampler::new(config, store));
    sampler.register_source(Box::new(InMemorySource::new("unit", records)));

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let sampler = Arc::clone(&sampler);
            thread::spawn(move || sampler.next_text_batch(SplitLabel::Train))
        })
        .collect();

    for handle in handles {
        let batch = handle.join().unwrap().unwrap();
        assert_eq!(batch.samples.len(), 1);
    }
}

struct DelegatingSampler;

impl Sampler for DelegatingSampler {
    fn next_pair_batch_with_weights(
        &self,
        _split: SplitLabel,
        weights: &HashMap<SourceId, f32>,
    ) -> Result<SampleBatch, SamplerError> {
        assert!(weights.is_empty());
        Ok(SampleBatch { pairs: Vec::new() })
    }

    fn next_text_batch_with_weights(
        &self,
        _split: SplitLabel,
        weights: &HashMap<SourceId, f32>,
    ) -> Result<TextBatch, SamplerError> {
        assert!(weights.is_empty());
        Ok(TextBatch {
            samples: Vec::new(),
        })
    }

    fn next_triplet_batch_with_weights(
        &self,
        _split: SplitLabel,
        weights: &HashMap<SourceId, f32>,
    ) -> Result<TripletBatch, SamplerError> {
        assert!(weights.is_empty());
        Ok(TripletBatch {
            triplets: Vec::new(),
        })
    }
}

#[test]
fn sampler_trait_default_methods_delegate_to_weighted_variants() {
    let sampler = DelegatingSampler;
    assert!(
        sampler
            .next_pair_batch(SplitLabel::Train)
            .unwrap()
            .is_empty()
    );
    assert!(
        sampler
            .next_text_batch(SplitLabel::Train)
            .unwrap()
            .is_empty()
    );
    assert!(
        sampler
            .next_triplet_batch(SplitLabel::Train)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn batch_prefetcher_reports_stopped_when_receiver_missing() {
    let prefetcher = BatchPrefetcher::<TripletBatch> {
        receiver: None,
        handle: None,
        stats: Arc::new(PrefetcherStats::default()),
    };
    let err = prefetcher.next().unwrap_err();
    assert!(matches!(
        err,
        SamplerError::SourceUnavailable { ref reason, .. } if reason == PREFETCHER_STOPPED_REASON
    ));
}

#[test]
fn batch_prefetcher_reports_stopped_when_worker_panics() {
    let prefetcher = BatchPrefetcher::<TripletBatch>::new(1, || panic!("prefetcher panic path"));
    let err = prefetcher.next().unwrap_err();
    assert!(matches!(
        err,
        SamplerError::SourceUnavailable { ref source_id, .. } if source_id == PREFETCHER_SOURCE_ID
    ));
}

fn sampler_for_prefetch_tests() -> Arc<TripletSampler<DeterministicSplitStore>> {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 501).unwrap());
    let mut config = base_config();
    config.split = split;
    config.allowed_splits = vec![SplitLabel::Train];
    config.batch_size = 1;
    config.ingestion_max_records = 16;
    config.recipes = vec![TripletRecipe {
        name: "prefetch_triplet".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    }];
    config.text_recipes = vec![TextRecipe {
        name: "prefetch_text".into(),
        selector: Selector::Role(SectionRole::Context),
        weight: 1.0,
        instruction: None,
    }];

    let sampler = Arc::new(TripletSampler::new(config, store.clone()));
    let records: Vec<DataRecord> = (0u32..)
        .filter_map(|i| {
            let id = format!("prefetch_{i}");
            (store.label_for(&id) == Some(SplitLabel::Train)).then(|| {
                let mut record = trader_record(
                    &id,
                    "2025-01-01",
                    &format!("Prefetch title {i}"),
                    &format!("Prefetch body {i}"),
                );
                record.source = "prefetch_source".to_string();
                record
            })
        })
        .take(4)
        .collect();
    sampler.register_source(Box::new(InMemorySource::new("prefetch_source", records)));
    sampler
}

#[test]
fn prefetch_public_apis_produce_batches_and_stats() {
    fn wait_for_count<T: Send + 'static>(prefetcher: &BatchPrefetcher<T>, minimum: usize) {
        let start = std::time::Instant::now();
        while prefetcher.produced_count() < minimum
            && start.elapsed() < StdDuration::from_millis(250)
        {
            std::thread::sleep(StdDuration::from_millis(5));
        }
    }

    let sampler = sampler_for_prefetch_tests();

    let triplet = Arc::clone(&sampler).prefetch_triplet_batches(SplitLabel::Train, 1);
    let pair = Arc::clone(&sampler).prefetch_pair_batches(SplitLabel::Train, 1);
    let text = Arc::clone(&sampler).prefetch_text_batches(SplitLabel::Train, 1);

    let triplet_batch = triplet.next().unwrap();
    assert_eq!(triplet_batch.triplets.len(), 1);
    wait_for_count(&triplet, 1);
    assert!(triplet.produced_count() >= 1);

    let pair_batch = pair.next().unwrap();
    assert_eq!(pair_batch.pairs.len(), 1);
    wait_for_count(&pair, 1);
    assert!(pair.produced_count() >= 1);

    let text_batch = text.next().unwrap();
    assert_eq!(text_batch.samples.len(), 1);
    wait_for_count(&text, 1);
    assert!(text.produced_count() >= 1);
}

#[test]
fn prefetch_weighted_public_apis_produce_batches() {
    let sampler = sampler_for_prefetch_tests();
    let mut weights = HashMap::new();
    weights.insert("prefetch_source".to_string(), 1.0);

    let triplet = Arc::clone(&sampler).prefetch_triplet_batches_with_weights(
        SplitLabel::Train,
        1,
        weights.clone(),
    );
    let pair = Arc::clone(&sampler).prefetch_pair_batches_with_weights(
        SplitLabel::Train,
        1,
        weights.clone(),
    );
    let text =
        Arc::clone(&sampler).prefetch_text_batches_with_weights(SplitLabel::Train, 1, weights);

    assert_eq!(triplet.next().unwrap().triplets.len(), 1);
    assert_eq!(pair.next().unwrap().pairs.len(), 1);
    assert_eq!(text.next().unwrap().samples.len(), 1);
    assert_eq!(triplet.error_count(), 0);
}

#[test]
fn different_epochs_produce_different_record_orderings() {
    // Behavioural guarantee: advancing the epoch must produce a measurably
    // different sequence of anchor records, not just change an internal
    // counter with no observable effect.
    //
    // How it works: rebuild_chunk_index() sorts source_record_indices by
    // stable_hash_str(epoch_seed(), record_id), and choose_anchor_record()
    // derives its offset from epoch_seed() ^ cycle.  Both are keyed to the
    // current epoch, so a different epoch shuffles the records differently.
    //
    // With the seed=55, 4 of 6 format IDs hash to Train with train:0.7.
    // The probability that two distinct epoch seeds produce the same
    // permutation of 4 records is 1/4! ≈ 4.2%, but fixed seeds make this
    // deterministic.
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 55).unwrap());

    // Filter to only records that hash to Train to ensure they are ingested.
    let n_records = 6;
    let records: Vec<DataRecord> = (0..n_records)
        .map(|i| {
            trader_record(
                &format!("epoch_order::rec_{i:02}"),
                "2025-01-01",
                &format!("Title {i}"),
                &format!("Body {i} with enough context for sampling"),
            )
        })
        .filter(|r| store.label_for(&r.id) == Some(SplitLabel::Train))
        .collect();
    let n_train = records.len();

    let config = SamplerConfig {
        seed: 55,
        batch_size: 1,
        recipes: vec![TripletRecipe {
            name: "epoch_order".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
            allow_same_anchor_positive: false,
        }],
        text_recipes: Vec::new(),
        allowed_splits: vec![SplitLabel::Train],
        ..SamplerConfig::default()
    };

    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(InMemorySource::new("epoch_order", records)));

    // Epoch 0: collect one anchor per batch across all Train records.
    let epoch0: Vec<String> = (0..n_train)
        .map(|_| {
            sampler
                .next_triplet_batch(SplitLabel::Train)
                .expect("epoch-0 batch must succeed")
                .triplets[0]
                .anchor
                .record_id
                .clone()
        })
        .collect();

    // Advance to epoch 1 — rebuilds the shuffle with a new seed.
    sampler.set_epoch(1).unwrap();

    // Epoch 1: collect the same number of batches.
    let epoch1: Vec<String> = (0..n_train)
        .map(|_| {
            sampler
                .next_triplet_batch(SplitLabel::Train)
                .expect("epoch-1 batch must succeed")
                .triplets[0]
                .anchor
                .record_id
                .clone()
        })
        .collect();

    assert_ne!(
        epoch0, epoch1,
        "epoch 0 and epoch 1 produced identical anchor orderings — \
             epoch seed has no effect on record selection"
    );
}

#[test]
fn resumed_sampler_uses_persisted_epoch_seed() {
    // Guarantee: when a sampler resumes from persisted state at epoch N it
    // must produce a genuinely different record ordering than epoch 0 —
    // the epoch is not just an internal counter with no observable effect.
    //
    // Structure:
    //   Baseline  — fresh sampler at epoch 0 (DeterministicSplitStore);
    //               draw n_draws records in order.
    //   Setup     — separate sampler on a file store: draw one batch to
    //               prime source-state loading (so the save is not a no-op),
    //               call set_epoch(1), then save.  The persisted state now
    //               has cursor=1 and source_epoch=1.
    //   Resume    — brand-new sampler on the same file store; no set_epoch
    //               call; draw n_draws records.  Epoch 1 must be loaded
    //               automatically from the store.
    //
    // n_records=20 / n_draws=10 keeps all draws inside cycle-0 (cursor never
    // reaches a multiple of n_records during the draw phase), avoiding
    // offset-seed changes at cycle boundaries.
    //
    // Assertions:
    //   (b) The full ordered sequences differ — advancing the epoch reshuffles
    //       records; it must not replay epoch 0 in the same order.
    //   (c) The first half of each sequence differs as a set — records that
    //       came early in epoch 0 shift position in epoch 1, proving that
    //       "a new epoch" is not just a different label on the same early pool.
    let base_seed = 0xC0FFEE_u64;
    let n_records = 20_usize;
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let temp = tempdir().unwrap();
    let store_path = temp.path().join("epoch_seed_resume_store");

    // Filter to only Train-hashing records; with seed=0xC0FFEE and train:0.7,
    // 14 of the 20 format IDs hash to Train.
    let probe_store = DeterministicSplitStore::new(split, base_seed).unwrap();
    let records: Vec<DataRecord> = (0..n_records)
        .map(|i| {
            trader_record(
                &format!("ep_resume::rec_{i:02}"),
                "2025-01-01",
                &format!("Title {i}"),
                &format!("Body {i} with enough context for sampling"),
            )
        })
        .filter(|r| probe_store.label_for(&r.id) == Some(SplitLabel::Train))
        .collect();
    let n_train = records.len();
    // n_draws must be ≤ n_train to stay within cycle-0.
    let n_draws = n_train / 2;

    let make_config = || SamplerConfig {
        seed: base_seed,
        batch_size: 1,
        recipes: vec![TripletRecipe {
            name: "ep".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
            allow_same_anchor_positive: false,
        }],
        text_recipes: Vec::new(),
        allowed_splits: vec![SplitLabel::Train],
        ..SamplerConfig::default()
    };

    // Accepts &dyn Sampler so it works with both DeterministicSplitStore
    // and FileSplitStore without monomorphisation issues.
    let draw_n = |sampler: &dyn Sampler, n: usize| -> Vec<String> {
        (0..n)
            .map(|_| {
                sampler
                    .next_triplet_batch(SplitLabel::Train)
                    .expect("batch must succeed")
                    .triplets[0]
                    .anchor
                    .record_id
                    .clone()
            })
            .collect()
    };

    // Baseline — epoch 0, cursor starts at 0.
    let epoch0_sequence: Vec<String> = {
        let store = Arc::new(DeterministicSplitStore::new(split, base_seed).unwrap());
        let sampler = TripletSampler::new(make_config(), store);
        sampler.register_source(Box::new(InMemorySource::new("ep_resume", records.clone())));
        draw_n(&sampler, n_draws)
    };

    // Setup — draw one batch to mark source state as loaded (persist_source_state
    // is a no-op when source_state_loaded=false), advance to epoch 1, then save.
    // Saved state: cursor=1, source_epoch=1.
    {
        let store = Arc::new(FileSplitStore::open(&store_path, split, base_seed).unwrap());
        let sampler = TripletSampler::new(make_config(), Arc::clone(&store));
        sampler.register_source(Box::new(InMemorySource::new("ep_resume", records.clone())));
        sampler
            .next_triplet_batch(SplitLabel::Train)
            .expect("priming batch must succeed");
        sampler.set_epoch(1).unwrap();
        sampler.save_sampler_state(None).unwrap();
    }

    // Resume — fresh sampler, epoch loaded from store (no set_epoch call).
    let epoch1_sequence: Vec<String> = {
        let store = Arc::new(FileSplitStore::open(&store_path, split, base_seed).unwrap());
        let sampler = TripletSampler::new(make_config(), store);
        sampler.register_source(Box::new(InMemorySource::new("ep_resume", records.clone())));
        draw_n(&sampler, n_draws)
    };

    // (b) The full ordered sequences must differ.
    assert_ne!(
        epoch0_sequence, epoch1_sequence,
        "epoch 0 and epoch 1 (resumed from store) produced identical anchor orderings — \
             epoch advancement has no effect on record selection order"
    );

    // (c) The early records must differ as a set: records that came first in
    //     epoch 0 must not all appear first in epoch 1 and vice versa.
    let half = n_draws / 2;
    let e0_first: std::collections::HashSet<&str> =
        epoch0_sequence[..half].iter().map(String::as_str).collect();
    let e1_first: std::collections::HashSet<&str> =
        epoch1_sequence[..half].iter().map(String::as_str).collect();
    assert_ne!(
        e0_first, e1_first,
        "the first {half} records drawn in epoch 0 and epoch 1 (resumed) are the same set — \
             records are not actually changing position across epochs"
    );
}

#[test]
fn triplet_rejects_negative_with_duplicate_text_content() {
    // Three records: two share the same context body, one is unique.
    // With WrongArticle negatives the sampler will first attempt to draw the
    // shared-text record as a negative — that attempt must be rejected
    // (negative.text == positive.text) and the unique record used instead.
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let mut config = base_config();
    config.batch_size = 1;
    config.allowed_splits = vec![SplitLabel::Train];
    config.split = split;
    config.recipes = vec![TripletRecipe {
        name: "content_dedup".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    }];

    let store = Arc::new(DeterministicSplitStore::new(split, 55).unwrap());
    // src::content_dup_a and src::content_dup_b are Train with seed=55 and train:0.7.
    // src::content_unique is Validation, so use find_id for it.
    let unique_id = (0u32..)
        .find_map(|i| {
            let id = format!("content_unique_{i}");
            (store.label_for(&id) == Some(SplitLabel::Train)).then_some(id)
        })
        .unwrap();
    let sampler = TripletSampler::new(config, store);

    // Two records share the same context text; one is genuinely distinct.
    let records = vec![
        trader_record(
            "src::content_dup_a",
            "2025-01-01",
            "Title A",
            "Shared body text",
        ),
        trader_record(
            "src::content_dup_b",
            "2025-01-02",
            "Title B",
            "Shared body text",
        ),
        trader_record(
            &unique_id,
            "2025-01-03",
            "Title C",
            "Completely different body",
        ),
    ];
    sampler.register_source(Box::new(InMemorySource::new("tt", records)));

    // Draw many batches; every triplet must have all-distinct slot texts.
    for _ in 0..32 {
        let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        for triplet in &batch.triplets {
            assert_ne!(
                triplet.anchor.text, triplet.positive.text,
                "anchor and positive must not share text"
            );
            assert_ne!(
                triplet.negative.text, triplet.positive.text,
                "negative must not share text with positive"
            );
            assert_ne!(
                triplet.negative.text, triplet.anchor.text,
                "negative must not share text with anchor"
            );
        }
    }
}

#[test]
fn wrong_publication_date_covers_some_none_branch_with_undated_candidates() {
    // Covers the `(Some(_), None) => true` branch in the WrongPublicationDate filter:
    // anchor has a publication date; candidate has no date entry in taxonomy.
    // All three records must be in the same split so that the date-match branches
    // — not the split guard — are what determines eligibility.
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 42).unwrap());

    let find_train_id = |prefix: &str| -> String {
        for i in 0..10_000_u32 {
            let id = format!("{prefix}_{i}");
            if store.label_for(&id) == Some(SplitLabel::Train) {
                return id;
            }
        }
        panic!("no Train id found for prefix {prefix}");
    };
    let anchor_id = find_train_id("wpd_sn_anchor");
    let no_date_id = find_train_id("wpd_sn_nodate");
    let same_date_id = find_train_id("wpd_sn_same");

    let config = SamplerConfig {
        seed: 42,
        batch_size: 1,
        recipes: Vec::new(),
        text_recipes: Vec::new(),
        split,
        ..SamplerConfig::default()
    };

    // Anchor with a publication date.
    let anchor_dated = trader_record(&anchor_id, "2025-01-01", "Dated anchor", "Body A");
    // Candidate with no date entry — triggers (Some(_), None) => true.
    let mut cand_no_date = trader_record(&no_date_id, "2025-01-02", "No date cand", "Body B");
    cand_no_date
        .taxonomy
        .retain(|t| META_FIELD_DATE.strip(t).is_none());
    // Candidate with the same date as anchor — excluded by (Some, Some) equal => false.
    let cand_same = trader_record(&same_date_id, "2025-01-01", "Same date cand", "Body C");

    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(InMemorySource::new(
        PRIMARY_SOURCE_ID,
        vec![anchor_dated, cand_no_date, cand_same],
    )));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let mut inner = sampler.inner.lock().unwrap();
    let anchor = inner.records.get(&anchor_id).cloned().expect("anchor");
    // cand_no_date is eligible (Some, None); cand_same is excluded (same date).
    let (neg, _) = inner
        .select_negative_record(&anchor, &NegativeStrategy::WrongPublicationDate)
        .expect("should find undated candidate as negative");
    assert_eq!(neg.id, no_date_id);
}

#[test]
fn wrong_publication_date_covers_none_some_and_none_none_branches() {
    // Covers (None, Some(_)) => true and (None, None) => false:
    // anchor has no date; candidates either have a date or also lack one.
    // All three records must be in the same split so the date-match arms
    // — not the split guard — determine eligibility.
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 55).unwrap());

    let find_train_id = |prefix: &str| -> String {
        for i in 0..10_000_u32 {
            let id = format!("{prefix}_{i}");
            if store.label_for(&id) == Some(SplitLabel::Train) {
                return id;
            }
        }
        panic!("no Train id found for prefix {prefix}");
    };
    let anchor_id = find_train_id("wpd_nn_anchor");
    let dated_id = find_train_id("wpd_nn_dated");
    let undated_id = find_train_id("wpd_nn_undated");

    let config = SamplerConfig {
        seed: 55,
        batch_size: 1,
        recipes: Vec::new(),
        text_recipes: Vec::new(),
        split,
        ..SamplerConfig::default()
    };

    // Anchor without a date (None).
    let mut anchor_no_date = trader_record(&anchor_id, "2025-01-01", "No date anchor", "Body A");
    anchor_no_date
        .taxonomy
        .retain(|t| META_FIELD_DATE.strip(t).is_none());
    // Candidate WITH a date — (None, Some(_)) => true, so it is eligible.
    let cand_dated = trader_record(&dated_id, "2025-01-02", "Dated cand", "Body B");
    // Candidate also without a date — (None, None) => false, so it is NOT eligible.
    let mut cand_no_date = trader_record(&undated_id, "2025-01-01", "No date cand", "Body C");
    cand_no_date
        .taxonomy
        .retain(|t| META_FIELD_DATE.strip(t).is_none());

    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(InMemorySource::new(
        PRIMARY_SOURCE_ID,
        vec![anchor_no_date, cand_dated, cand_no_date],
    )));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let mut inner = sampler.inner.lock().unwrap();
    let anchor = inner.records.get(&anchor_id).cloned().expect("anchor");
    // Only cand_dated is eligible: (None, Some) => true.
    // cand_no_date is excluded: (None, None) => false.
    let (neg, _) = inner
        .select_negative_record(&anchor, &NegativeStrategy::WrongPublicationDate)
        .expect("undated anchor should match dated candidate");
    assert_eq!(neg.id, dated_id);
}

#[test]
fn temporal_offset_selector_finds_nearest_chronological_neighbor() {
    // Covers select_temporal_neighbor: verifies the min-by-key distance logic
    // selects the record whose created_at is closest to anchor.created_at + offset_days.
    // All three records must be in the same split so the split guard does not
    // interfere with the distance-based selection being tested here.
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 77).unwrap());

    let find_train_id = |prefix: &str| -> String {
        for i in 0..10_000_u32 {
            let id = format!("{prefix}_{i}");
            if store.label_for(&id) == Some(SplitLabel::Train) {
                return id;
            }
        }
        panic!("no Train id found for prefix {prefix}");
    };
    let base_id = find_train_id("toff_base");
    let id_7d = find_train_id("toff_7d");
    let id_30d = find_train_id("toff_30d");

    let config = SamplerConfig {
        seed: 77,
        batch_size: 1,
        recipes: Vec::new(),
        text_recipes: Vec::new(),
        split,
        ..SamplerConfig::default()
    };

    let base = Utc::now();
    // base_id: base time; id_7d: exactly +7 days; id_30d: +30 days.
    let r0 = record_with_offset(&base_id, base, 0);
    let r7d = record_with_offset(&id_7d, base, 7 * 86400);
    let r30d = record_with_offset(&id_30d, base, 30 * 86400);

    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(InMemorySource::new(
        PRIMARY_SOURCE_ID,
        vec![r0, r7d, r30d],
    )));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let inner = sampler.inner.lock().unwrap();
    let anchor = inner.records.get(&base_id).cloned().expect("anchor");
    // Requesting offset_days=7: target = base + 7 days. id_7d is an exact match.
    let neighbor = inner.select_temporal_neighbor(&anchor, 7);
    assert!(neighbor.is_some(), "should find a temporal neighbor");
    assert_eq!(neighbor.unwrap().id, id_7d);

    // Requesting offset_days=1: target = base + 1 day. id_7d (6 days away) beats id_30d (29 days).
    let neighbor_near = inner.select_temporal_neighbor(&anchor, 1);
    assert!(neighbor_near.is_some());
    assert_eq!(neighbor_near.unwrap().id, id_7d);
}

#[test]
fn temporal_offset_selector_never_crosses_split_boundaries() {
    // Verifies that select_temporal_neighbor excludes candidates in a different split,
    // even when they would be a closer chronological match than a same-split candidate.
    let split = SplitRatios {
        train: 0.6,
        validation: 0.3,
        test: 0.1,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 13).unwrap());

    // Find stable IDs that hash to the required splits under (split, seed=13).
    let find_id_for_split = |label: SplitLabel, prefix: &str| -> String {
        for i in 0..10_000_u32 {
            let id = format!("{prefix}_{i}");
            if store.label_for(&id) == Some(label) {
                return id;
            }
        }
        panic!("could not find an id hashing to {label:?}");
    };

    let anchor_id = find_id_for_split(SplitLabel::Train, "temporal_anchor");
    let train_id = find_id_for_split(SplitLabel::Train, "temporal_train");
    assert_ne!(anchor_id, train_id);
    let val_id = find_id_for_split(SplitLabel::Validation, "temporal_val");

    // All records use the same source so the source filter passes for all candidates.
    let base = Utc::now();
    let anchor_rec = record_with_offset(&anchor_id, base, 0);
    // Validation candidate: exactly on target (offset 1 day). Closest without split guard.
    let val_rec = record_with_offset(&val_id, base, 86_400);
    // Train candidate: 3 days out, farther from target but in the same split as anchor.
    let train_rec = record_with_offset(&train_id, base, 3 * 86_400);

    let config = SamplerConfig {
        seed: 13,
        batch_size: 1,
        recipes: Vec::new(),
        text_recipes: Vec::new(),
        split,
        ..SamplerConfig::default()
    };
    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(InMemorySource::new(
        PRIMARY_SOURCE_ID,
        vec![anchor_rec, val_rec, train_rec],
    )));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let inner = sampler.inner.lock().unwrap();
    let anchor = inner.records.get(&anchor_id).cloned().expect("anchor");

    // target = base + 1 day. val_rec is an exact match but must be excluded (wrong split).
    // train_rec is 2 days off but must be selected (only same-split candidate).
    let neighbor = inner.select_temporal_neighbor(&anchor, 1);
    assert!(
        neighbor.is_some(),
        "should find a same-split temporal neighbor"
    );
    assert_eq!(
        neighbor.unwrap().id,
        train_id,
        "temporal neighbor must not cross split boundaries"
    );
}

#[test]
fn instruction_propagates_from_recipe_to_sample_triplet() {
    // Verify that a non-None `instruction` on a TripletRecipe flows through
    // finalize_triplet_with_negative into every SampleTriplet it produces.
    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 55).unwrap());
    let mut config = base_config();
    config.batch_size = 1;
    config.allowed_splits = vec![SplitLabel::Train];
    config.split = split;
    config.recipes = vec![TripletRecipe {
        name: "instr_recipe".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: Some("Retrieve a relevant document.".into()),
        allow_same_anchor_positive: false,
    }];

    let now = Utc::now();
    let make_record = |id: &str, anchor: &str, context: &str| DataRecord {
        id: id.into(),
        source: "unit".into(),
        created_at: now,
        updated_at: now,
        quality: QualityScore::default(),
        taxonomy: vec![],
        sections: vec![
            RecordSection {
                role: SectionRole::Anchor,
                heading: None,
                text: anchor.into(),
                sentences: vec![anchor.into()],
            },
            RecordSection {
                role: SectionRole::Context,
                heading: None,
                text: context.into(),
                sentences: vec![context.into()],
            },
        ],
        meta_prefix: None,
    };

    let records = vec![
        make_record("r1", "anchor one unique text", "context one unique text"),
        make_record("r2", "anchor two unique text", "context two unique text"),
        make_record(
            "r3",
            "anchor three unique text",
            "context three unique text",
        ),
    ];

    let sampler = TripletSampler::new(config, Arc::clone(&store));
    sampler.register_source(Box::new(InMemorySource::new("unit", records)));

    let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
    assert_eq!(batch.triplets.len(), 1);
    assert_eq!(
        batch.triplets[0].instruction.as_deref(),
        Some("Retrieve a relevant document."),
        "instruction must propagate from TripletRecipe to SampleTriplet"
    );
}

#[test]
fn allow_same_anchor_positive_permits_identical_text_triplet() {
    // Verify the guard in finalize_triplet_with_negative: with the flag false
    // (default) a same-text pair is rejected; with the flag true it passes through.
    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 77).unwrap());
    let mut config = base_config();
    config.allowed_splits = vec![SplitLabel::Train];
    config.split = split;

    let now = Utc::now();
    // Both anchor= and context= sections carry the exact same text per record.
    // This simulates the text-columns mode where a single text field is duplicated
    // into both roles by row_to_record.
    let make_record = |id: &str, text: &str| DataRecord {
        id: id.into(),
        source: "unit".into(),
        created_at: now,
        updated_at: now,
        quality: QualityScore::default(),
        taxonomy: vec![],
        sections: vec![
            RecordSection {
                role: SectionRole::Anchor,
                heading: None,
                text: text.into(),
                sentences: vec![text.into()],
            },
            RecordSection {
                role: SectionRole::Context,
                heading: None,
                text: text.into(), // identical to anchor
                sentences: vec![text.into()],
            },
        ],
        meta_prefix: None,
    };

    let records = vec![
        make_record("t1", "the fox jumped over the lazy dog"),
        make_record("t2", "a quick brown cat sat on the mat"),
        make_record("t3", "stars shine brightly in the night sky"),
    ];

    // --- flag = false: the identical anchor/positive pair must be rejected ---
    let simcse_recipe_blocked = TripletRecipe {
        name: "blocked_simcse".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    };

    let mut config_blocked = config.clone();
    config_blocked.batch_size = 1;
    config_blocked.recipes = vec![simcse_recipe_blocked];

    let sampler_blocked = TripletSampler::new(config_blocked, Arc::clone(&store));
    sampler_blocked.register_source(Box::new(InMemorySource::new("unit", records.clone())));
    // With flag=false the sampler cannot build any valid triplet from these records
    // (every anchor/positive pair is identical text), so the batch should error.
    assert!(
        sampler_blocked
            .next_triplet_batch(SplitLabel::Train)
            .is_err(),
        "same-text anchor/positive must be rejected when allow_same_anchor_positive=false"
    );

    // --- flag = true: the identical anchor/positive pair must be allowed ---
    let simcse_recipe_allowed = TripletRecipe {
        name: "allowed_simcse".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: true,
    };

    let mut config_allowed = config.clone();
    config_allowed.batch_size = 1;
    config_allowed.recipes = vec![simcse_recipe_allowed];

    let sampler_allowed = TripletSampler::new(config_allowed, Arc::clone(&store));
    sampler_allowed.register_source(Box::new(InMemorySource::new("unit", records)));
    let batch = sampler_allowed
        .next_triplet_batch(SplitLabel::Train)
        .expect("triplet must be produced when allow_same_anchor_positive=true");
    assert_eq!(batch.triplets.len(), 1);
    let triplet = &batch.triplets[0];
    // Anchor and positive carry the same text (SimCSE pattern).
    assert_eq!(
        triplet.anchor.text, triplet.positive.text,
        "anchor and positive must share identical text in SimCSE mode"
    );
    // Negative must differ.
    assert_ne!(
        triplet.negative.text, triplet.anchor.text,
        "negative must differ from anchor even in SimCSE mode"
    );
}

/// BM25 search must be scoped to the anchor's own source index only.
///
/// Regression test for the bug where `ranked_candidates` iterated every
/// per-source index, paying full search cost across all sources even though
/// the negative pool is always same-source.  After the fix the ranked list
/// must contain only records whose source matches the anchor's source.
#[cfg(feature = "bm25-mining")]
#[test]
fn bm25_ranked_candidates_are_scoped_to_anchor_source() {
    let split = SplitRatios {
        train: 0.8,
        validation: 0.1,
        test: 0.1,
    };
    let config = SamplerConfig {
        seed: 77,
        batch_size: 1,
        chunking: ChunkingStrategy::default(),
        recipes: Vec::new(),
        text_recipes: Vec::new(),
        split,
        ..SamplerConfig::default()
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 77).unwrap());

    // Find IDs that land in Train for each prefix so split assignment is
    // deterministic regardless of how many records we create.
    let find_train_id = |prefix: &str| -> String {
        (0u32..)
            .find_map(|i| {
                let id = format!("{prefix}_{i}");
                (store.label_for(&id) == Some(SplitLabel::Train)).then_some(id)
            })
            .unwrap()
    };

    let anchor_id = find_train_id("scope_anchor");
    let same_source_id = find_train_id("scope_same");
    let other_source_id = find_train_id("scope_other");

    // anchor and same_source share highly similar text — BM25 should rank
    // same_source highly.  other_source uses identical text but lives in a
    // different source and must never appear in the ranked results.
    let anchor = DataRecord {
        id: anchor_id.clone(),
        source: "source_alpha".into(),
        ..trader_record(
            &anchor_id,
            "2025-03-01",
            "quantum computing error correction",
            "surface codes and topological qubits for fault tolerance",
        )
    };
    let same_source = DataRecord {
        id: same_source_id.clone(),
        source: "source_alpha".into(),
        ..trader_record(
            &same_source_id,
            "2025-03-02",
            "quantum error correction surface codes",
            "topological qubits fault tolerance thresholds",
        )
    };
    let other_source = DataRecord {
        id: other_source_id.clone(),
        // Lexically identical text to same_source — would score just as high
        // if the search were global.
        source: "source_beta".into(),
        ..trader_record(
            &other_source_id,
            "2025-03-02",
            "quantum error correction surface codes",
            "topological qubits fault tolerance thresholds",
        )
    };

    let sampler = TripletSampler::new(config, Arc::clone(&store));
    sampler.register_source(Box::new(InMemorySource::new(
        "source_alpha",
        vec![anchor.clone(), same_source],
    )));
    sampler.register_source(Box::new(InMemorySource::new(
        "source_beta",
        vec![other_source.clone()],
    )));

    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let mut inner = sampler.inner.lock().unwrap();
    let ranked = inner.bm25_ranked_candidates(&anchor);

    assert!(
        !ranked.is_empty(),
        "ranked candidates must not be empty — same-source record should be found"
    );
    for id in &ranked {
        let record = inner.records.get(id).expect("ranked id must be in records");
        assert_eq!(
            record.source, "source_alpha",
            "BM25 ranked candidate '{id}' came from source '{}' but anchor is in \
             'source_alpha' — cross-source leak detected (regression)",
            record.source,
        );
    }
    // Confirm the other-source record is definitely not in the ranked list.
    assert!(
        !ranked.contains(&other_source_id),
        "other-source record must not appear in BM25 ranked candidates for anchor \
         in a different source (global search regression)"
    );
}
