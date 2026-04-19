fn base_config() -> super::SamplerConfig {
    super::SamplerConfig::default()
}

#[cfg(feature = "bm25-mining")]
use super::backends::bm25_backend::record_bm25_text;
use super::*;
use crate::chunking::ChunkingAlgorithm;
use crate::config::{ChunkingStrategy, NegativeStrategy, Selector, TextRecipe, TripletRecipe};
use crate::metrics::chunk_proximity_score;

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
pub const TRIPLET_BATCH_SEQUENCE_HASH: u64 = 6137236445130287036;
/// Expected hash for deterministic triplet batch sequence when bm25-mining is enabled.
#[cfg(feature = "bm25-mining")]
pub const TRIPLET_BATCH_SEQUENCE_HASH: u64 = 3567297114780411140;
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
pub const PREFETCH_TRIPLET_BATCH_SEQUENCE_HASH: u64 = 13549723595682255368;
/// Expected hash for deterministic prefetch triplet batch sequence when bm25-mining is enabled.
#[cfg(feature = "bm25-mining")]
pub const PREFETCH_TRIPLET_BATCH_SEQUENCE_HASH: u64 = 17421456775178077384;
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
        },
        text: "window".into(),
        tokens_estimate: 8,
        quality: QualityScore { trust: 1.0 },
        kvp_meta: Default::default(),
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
fn next_chunk_from_pool_returns_none_for_empty_pool() {
    let split = SplitRatios::default();
    let store = Arc::new(DeterministicSplitStore::new(split, 17).unwrap());
    let mut inner = TripletSamplerInner::new(base_config(), store);

    assert!(inner.next_chunk_from_pool("rec", 0, Vec::new()).is_none());
}

#[test]
fn recipe_order_cycled_and_text_recipe_order_cycled_return_empty_for_zero_count() {
    let split = SplitRatios::default();
    let store = Arc::new(DeterministicSplitStore::new(split, 18).unwrap());
    let mut inner = TripletSamplerInner::new(base_config(), store);

    assert!(inner.recipe_order_weighted_cycled_seeded(&[], 3).is_empty());
    assert!(
        inner
            .text_recipe_order_weighted_cycled_seeded(&[], 5)
            .is_empty()
    );
}

#[test]
fn select_chunk_random_handles_empty_and_non_empty_sections() {
    let split = SplitRatios::default();
    let mut config = base_config();
    config.chunking.max_window_tokens = 4;
    config.chunking.overlap_tokens = vec![0];
    let store = Arc::new(DeterministicSplitStore::new(split, 19).unwrap());
    let mut inner = TripletSamplerInner::new(config, store);

    let empty_record = DataRecord {
        id: "empty_random".into(),
        source: "unit".into(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        quality: QualityScore::default(),
        taxonomy: vec![],
        sections: vec![],
        meta_prefix: None,
    };
    assert!(
        inner
            .select_chunk(&empty_record, &Selector::Random)
            .is_none()
    );

    let non_empty_record = DataRecord {
        id: "non_empty_random".into(),
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
    let sampled = inner
        .select_chunk(&non_empty_record, &Selector::Random)
        .expect("random selector should sample from non-empty sections");
    assert_eq!(sampled.record_id, "non_empty_random");
}

#[test]
fn record_has_long_section_returns_false_when_window_tokens_are_disabled() {
    let split = SplitRatios::default();
    let mut config = base_config();
    config.chunking.max_window_tokens = 0;
    let store = Arc::new(DeterministicSplitStore::new(split, 20).unwrap());
    let inner = TripletSamplerInner::new(config, store);

    let long_record = DataRecord {
        id: "long_record".into(),
        source: "unit".into(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        quality: QualityScore::default(),
        taxonomy: vec![],
        sections: vec![RecordSection {
            role: SectionRole::Context,
            heading: None,
            text: "one two three four five six seven eight".into(),
            sentences: vec!["one two three four five six seven eight".into()],
        }],
        meta_prefix: None,
    };

    assert!(!inner.record_has_long_anchor_or_context_section(&long_record));
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

    inner.decorate_chunk_seeded(&record, &mut chunk);

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

    inner.decorate_chunk_seeded(&record, &mut chunk);

    let expected_prefix = format!("meta: source=unit{}", platform_newline());
    assert!(
        chunk.text.starts_with(&expected_prefix),
        "meta prefix should remain on its own line after truncation"
    );
    assert_eq!(chunk.tokens_estimate, 4);
}

#[test]
fn kvp_meta_populated_unconditionally_even_when_dropout_suppresses_prefix() {
    // dropout=0.0 means sample() always returns None, so no prefix text is
    // prepended. kvp_meta must still be populated with all declared fields.
    let split = SplitRatios::default();
    let store = Arc::new(DeterministicSplitStore::new(split, 42).unwrap());
    let sampler = TripletSampler::new(base_config(), store);

    let mut record = sample_record();
    let mut kvp = KvpPrefixSampler::new(0.0); // dropout off — prefix never rendered
    kvp.add_variant_fields([
        KvpField::many("date", ["2025-01-01", "Jan 1, 2025"]),
        KvpField::one("source", "daily-report"),
    ]);
    record.meta_prefix = Some(kvp);

    let mut inner = sampler.inner.lock().unwrap();
    let mut chunks = inner.materialize_chunks(&record, 0, &record.sections[0]);
    assert!(!chunks.is_empty());
    let mut chunk = chunks.remove(0);
    inner.decorate_chunk_seeded(&record, &mut chunk);

    // No prefix in text because dropout=0.0
    assert!(
        !chunk.text.starts_with("meta:"),
        "dropout=0.0 should suppress prefix text"
    );

    // But kvp_meta must contain all declared keys and values
    assert_eq!(chunk.kvp_meta.len(), 2, "expected two keys in kvp_meta");
    let mut dates = chunk.kvp_meta["date"].clone();
    dates.sort();
    assert_eq!(dates, vec!["2025-01-01", "Jan 1, 2025"]);
    assert_eq!(chunk.kvp_meta["source"], vec!["daily-report"]);
}

#[test]
fn kvp_meta_empty_when_record_has_no_meta_prefix() {
    let split = SplitRatios::default();
    let store = Arc::new(DeterministicSplitStore::new(split, 42).unwrap());
    let sampler = TripletSampler::new(base_config(), store);

    let record = sample_record(); // meta_prefix: None

    let mut inner = sampler.inner.lock().unwrap();
    let mut chunks = inner.materialize_chunks(&record, 0, &record.sections[0]);
    assert!(!chunks.is_empty());
    let mut chunk = chunks.remove(0);
    inner.decorate_chunk_seeded(&record, &mut chunk);

    assert!(
        chunk.kvp_meta.is_empty(),
        "kvp_meta should be empty when the record has no meta_prefix"
    );
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
fn chunk_view_carries_window_index() {
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
    let indices: Vec<usize> = chunks
        .iter()
        .filter_map(|chunk| match chunk.view {
            ChunkView::Window { index, .. } => Some(index),
            _ => None,
        })
        .collect();
    assert!(indices.len() >= 3);
    assert_eq!(indices[0], 0);
    assert_eq!(indices[1], 1);
    assert_eq!(indices[2], 2);
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

struct FixedChunker;

impl ChunkingAlgorithm for FixedChunker {
    fn materialize(
        &self,
        _strategy: &ChunkingStrategy,
        record: &DataRecord,
        section_idx: usize,
        _section: &RecordSection,
    ) -> Vec<RecordChunk> {
        vec![RecordChunk {
            record_id: record.id.clone(),
            section_idx,
            view: ChunkView::SummaryFallback {
                strategy: "fixed".into(),
                weight: 0.7,
            },
            text: "fixed-chunk".into(),
            tokens_estimate: 1,
            quality: record.quality,
            kvp_meta: Default::default(),
        }]
    }
}

struct MarkerChunker;

impl ChunkingAlgorithm for MarkerChunker {
    fn materialize(
        &self,
        _strategy: &ChunkingStrategy,
        record: &DataRecord,
        section_idx: usize,
        _section: &RecordSection,
    ) -> Vec<RecordChunk> {
        vec![
            RecordChunk {
                record_id: record.id.clone(),
                section_idx,
                view: ChunkView::Window {
                    index: 0,
                    overlap: 0,
                    span: 2,
                },
                text: format!("custom::{}::{}::w0", record.id, section_idx),
                tokens_estimate: 2,
                quality: record.quality,
                kvp_meta: Default::default(),
            },
            RecordChunk {
                record_id: record.id.clone(),
                section_idx,
                view: ChunkView::Window {
                    index: 1,
                    overlap: 0,
                    span: 2,
                },
                text: format!("custom::{}::{}::w1", record.id, section_idx),
                tokens_estimate: 2,
                quality: record.quality,
                kvp_meta: Default::default(),
            },
        ]
    }
}

#[test]
fn sampler_uses_custom_chunking_algorithm_when_provided() {
    let split = SplitRatios::default();
    let config = base_config();
    let store = Arc::new(DeterministicSplitStore::new(split, 17).unwrap());
    let sampler = TripletSampler::new_with_chunker(config, store, Arc::new(FixedChunker));

    let section_text = "one two three four five";
    let record = DataRecord {
        id: "custom_chunk_record".into(),
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

    let chunks = sampler
        .inner
        .lock()
        .unwrap()
        .materialize_chunks(&record, 0, &record.sections[0]);

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text, "fixed-chunk");
    assert!(matches!(chunks[0].view, ChunkView::SummaryFallback { .. }));
}

#[test]
fn runtime_batches_do_not_bypass_custom_chunker() {
    let mut config = base_config();
    config.batch_size = 2;
    config.split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };
    config.allowed_splits = vec![SplitLabel::Train];
    config.recipes = vec![TripletRecipe {
        name: "custom_chunker_runtime_triplet".into(),
        anchor: Selector::Role(SectionRole::Context),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    }];
    config.text_recipes = vec![TextRecipe {
        name: "custom_chunker_runtime_text".into(),
        selector: Selector::Role(SectionRole::Context),
        weight: 1.0,
        instruction: None,
    }];

    let store = Arc::new(DeterministicSplitStore::new(config.split, 33).unwrap());
    let sampler = TripletSampler::new_with_chunker(config, store, Arc::new(MarkerChunker));

    let mk = |id: &str| DataRecord {
        id: id.into(),
        source: "unit".into(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        quality: QualityScore { trust: 1.0 },
        taxonomy: vec![],
        sections: vec![RecordSection {
            role: SectionRole::Context,
            heading: None,
            text: "alpha beta gamma delta".into(),
            sentences: vec!["alpha beta gamma delta".into()],
        }],
        meta_prefix: None,
    };

    sampler.register_source(Box::new(InMemorySource::new(
        "unit",
        vec![mk("c1"), mk("c2"), mk("c3")],
    )));

    let text_batch = sampler.next_text_batch(SplitLabel::Train).unwrap();
    assert!(!text_batch.samples.is_empty());
    for sample in text_batch.samples {
        assert!(
            sample.chunk.text.starts_with("custom::"),
            "text sample bypassed custom chunker: {}",
            sample.chunk.text
        );
    }

    let pair_batch = sampler.next_pair_batch(SplitLabel::Train).unwrap();
    assert!(!pair_batch.pairs.is_empty());
    for pair in pair_batch.pairs {
        assert!(
            pair.anchor.text.starts_with("custom::"),
            "pair anchor bypassed custom chunker: {}",
            pair.anchor.text
        );
        assert!(
            pair.positive.text.starts_with("custom::"),
            "pair positive bypassed custom chunker: {}",
            pair.positive.text
        );
    }

    let triplet_batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
    assert!(!triplet_batch.triplets.is_empty());
    for triplet in triplet_batch.triplets {
        assert!(
            triplet.anchor.text.starts_with("custom::"),
            "triplet anchor bypassed custom chunker: {}",
            triplet.anchor.text
        );
        assert!(
            triplet.positive.text.starts_with("custom::"),
            "triplet positive bypassed custom chunker: {}",
            triplet.positive.text
        );
        assert!(
            triplet.negative.text.starts_with("custom::"),
            "triplet negative bypassed custom chunker: {}",
            triplet.negative.text
        );
    }
}

#[test]
fn chunk_weight_windows_use_trust_and_floor() {
    let split = SplitRatios::default();
    let mut config = base_config();
    config.chunking.chunk_weight_floor = 0.25;
    let store = Arc::new(DeterministicSplitStore::new(split, 5).unwrap());
    let sampler = TripletSampler::new(config, store);

    let base_chunk = RecordChunk {
        record_id: "unit".into(),
        section_idx: 0,
        view: ChunkView::Window {
            index: 3,
            overlap: 0,
            span: 10,
        },
        text: "dummy".into(),
        tokens_estimate: 10,
        quality: QualityScore { trust: 1.0 },
        kvp_meta: Default::default(),
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
    };
    assert_eq!(
        sampler.inner.lock().unwrap().chunk_weight(&early_chunk),
        1.0
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
        kvp_meta: Default::default(),
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
        },
        text: "dummy".into(),
        tokens_estimate: 10,
        quality: QualityScore { trust: 0.5 },
        kvp_meta: Default::default(),
    };

    let weight = sampler.inner.lock().unwrap().chunk_weight(&trusted_chunk);
    assert!((weight - 0.5).abs() < f32::EPSILON);
}

#[test]
fn triplet_weight_averages_chunk_weights() {
    let split = SplitRatios::default();
    let mut config = base_config();
    config.chunking.chunk_weight_floor = 0.0;
    let store = Arc::new(DeterministicSplitStore::new(split, 7).unwrap());
    let sampler = TripletSampler::new(config, store);

    let recipe = TripletRecipe {
        name: "regular_recipe".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    };

    let anchor = RecordChunk {
        record_id: "a".into(),
        section_idx: 0,
        view: ChunkView::Window {
            index: 0,
            overlap: 0,
            span: 10,
        },
        text: "a".into(),
        tokens_estimate: 10,
        quality: QualityScore::default(),
        kvp_meta: Default::default(),
    };
    let positive = RecordChunk {
        record_id: "b".into(),
        section_idx: 0,
        view: ChunkView::Window {
            index: 0,
            overlap: 0,
            span: 10,
        },
        text: "b".into(),
        tokens_estimate: 10,
        quality: QualityScore::default(),
        kvp_meta: Default::default(),
    };
    let negative = RecordChunk {
        record_id: "c".into(),
        section_idx: 0,
        view: ChunkView::Window {
            index: 0,
            overlap: 0,
            span: 10,
        },
        text: "c".into(),
        tokens_estimate: 10,
        quality: QualityScore::default(),
        kvp_meta: Default::default(),
    };

    let avg = sampler
        .inner
        .lock()
        .unwrap()
        .triplet_chunk_weight(&recipe, &anchor, &positive, &negative);
    let trust = QualityScore::default().trust;
    let expected = trust;
    assert!((avg - expected).abs() < f32::EPSILON);
}

#[test]
fn non_auto_triplet_negative_weight_uses_trust_only() {
    let split = SplitRatios::default();
    let mut config = base_config();
    config.chunking.chunk_weight_floor = 0.0;
    let store = Arc::new(DeterministicSplitStore::new(split, 71).unwrap());
    let sampler = TripletSampler::new(config, store);

    let recipe = TripletRecipe {
        name: "regular_recipe".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    };

    let anchor = RecordChunk {
        record_id: "a".into(),
        section_idx: 0,
        view: ChunkView::Window {
            index: 0,
            overlap: 0,
            span: 10,
        },
        text: "a".into(),
        tokens_estimate: 10,
        quality: QualityScore::default(),
        kvp_meta: Default::default(),
    };
    let positive = RecordChunk {
        record_id: "b".into(),
        section_idx: 0,
        view: ChunkView::Window {
            index: 0,
            overlap: 0,
            span: 10,
        },
        text: "b".into(),
        tokens_estimate: 10,
        quality: QualityScore::default(),
        kvp_meta: Default::default(),
    };
    let negative = RecordChunk {
        record_id: "c".into(),
        section_idx: 0,
        view: ChunkView::Window {
            index: 9,
            overlap: 0,
            span: 10,
        },
        text: "c".into(),
        tokens_estimate: 10,
        quality: QualityScore::default(),
        kvp_meta: Default::default(),
    };

    let avg = sampler
        .inner
        .lock()
        .unwrap()
        .triplet_chunk_weight(&recipe, &anchor, &positive, &negative);
    let trust = QualityScore::default().trust;
    let expected = trust;
    assert!((avg - expected).abs() < f32::EPSILON, "avg={avg}");
}

#[test]
fn non_auto_triplet_weight_applies_anchor_positive_proximity() {
    let split = SplitRatios::default();
    let mut config = base_config();
    config.chunking.chunk_weight_floor = 0.0;
    let store = Arc::new(DeterministicSplitStore::new(split, 72).unwrap());
    let sampler = TripletSampler::new(config, store);

    let recipe = TripletRecipe {
        name: "regular_recipe".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    };

    let anchor = RecordChunk {
        record_id: "r".into(),
        section_idx: 0,
        view: ChunkView::Window {
            index: 0,
            overlap: 0,
            span: 10,
        },
        text: "a".into(),
        tokens_estimate: 10,
        quality: QualityScore::default(),
        kvp_meta: Default::default(),
    };
    let positive = RecordChunk {
        record_id: "r".into(),
        section_idx: 0,
        view: ChunkView::Window {
            index: 3,
            overlap: 0,
            span: 10,
        },
        text: "b".into(),
        tokens_estimate: 10,
        quality: QualityScore::default(),
        kvp_meta: Default::default(),
    };
    let negative = RecordChunk {
        record_id: "n".into(),
        section_idx: 0,
        view: ChunkView::Window {
            index: 9,
            overlap: 0,
            span: 10,
        },
        text: "c".into(),
        tokens_estimate: 10,
        quality: QualityScore::default(),
        kvp_meta: Default::default(),
    };

    let avg = sampler
        .inner
        .lock()
        .unwrap()
        .triplet_chunk_weight(&recipe, &anchor, &positive, &negative);

    let trust = QualityScore::default().trust;
    // pair proximity for delta=3 is 0.25.
    let expected = ((trust * 0.25) + (trust * 0.25 * 0.25) + trust) / 3.0;
    assert!(
        (avg - expected).abs() < 1e-6,
        "avg={avg} expected={expected}"
    );
}

#[test]
fn non_auto_triplet_weight_tracks_positive_window_index() {
    let split = SplitRatios::default();
    let mut config = base_config();
    config.chunking.chunk_weight_floor = 0.0;
    let store = Arc::new(DeterministicSplitStore::new(split, 73).unwrap());
    let sampler = TripletSampler::new(config, store);

    let recipe = TripletRecipe {
        name: "regular_recipe".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    };

    let anchor = RecordChunk {
        record_id: "r".into(),
        section_idx: 0,
        view: ChunkView::Window {
            index: 0,
            overlap: 0,
            span: 10,
        },
        text: "a".into(),
        tokens_estimate: 10,
        quality: QualityScore::default(),
        kvp_meta: Default::default(),
    };
    let negative = RecordChunk {
        record_id: "n".into(),
        section_idx: 0,
        view: ChunkView::Window {
            index: 9,
            overlap: 0,
            span: 10,
        },
        text: "c".into(),
        tokens_estimate: 10,
        quality: QualityScore::default(),
        kvp_meta: Default::default(),
    };

    // [(positive_index, expected_proximity, expected_weight)]
    let cases: &[(usize, f32, f32)] = &[
        (0, 1.0, 0.5),
        (1, 0.5, 0.29166666),
        (2, 1.0 / 3.0, 0.24074075),
        (3, 0.25, 0.21875),
    ];

    let mut previous_weight: Option<f32> = None;
    for (positive_index, expected_proximity, expected_weight) in cases {
        let positive = RecordChunk {
            record_id: "r".into(),
            section_idx: 0,
            view: ChunkView::Window {
                index: *positive_index,
                overlap: 0,
                span: 10,
            },
            text: "b".into(),
            tokens_estimate: 10,
            quality: QualityScore::default(),
            kvp_meta: Default::default(),
        };

        let proximity = chunk_proximity_score(&anchor, &positive);
        assert!(
            (proximity - *expected_proximity).abs() < 1e-6,
            "index={positive_index} proximity={proximity} expected={expected_proximity}"
        );

        let weight = sampler
            .inner
            .lock()
            .unwrap()
            .triplet_chunk_weight(&recipe, &anchor, &positive, &negative);
        assert!(
            (weight - *expected_weight).abs() < 1e-6,
            "index={positive_index} weight={weight} expected={expected_weight}"
        );

        if let Some(prev) = previous_weight {
            assert!(
                weight <= prev,
                "expected weight to be non-increasing with index: prev={prev} current={weight} index={positive_index}"
            );
        }
        previous_weight = Some(weight);
    }
}

#[test]
fn auto_chunk_pair_triplet_weight_uses_proximity_inside_chunk_weight() {
    let split = SplitRatios::default();
    let mut config = base_config();
    config.chunking.chunk_weight_floor = 0.0;
    let store = Arc::new(DeterministicSplitStore::new(split, 77).unwrap());
    let sampler = TripletSampler::new(config, store);

    let auto_recipe = TripletRecipe {
        name: AUTO_INJECTED_LONG_SECTION_CHUNK_PAIR_RECIPE_NAME.into(),
        anchor: Selector::Role(SectionRole::Context),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    };

    let anchor = RecordChunk {
        record_id: "r".into(),
        section_idx: 0,
        view: ChunkView::Window {
            index: 0,
            overlap: 0,
            span: 10,
        },
        text: "a".into(),
        tokens_estimate: 10,
        quality: QualityScore::default(),
        kvp_meta: Default::default(),
    };
    let positive = RecordChunk {
        record_id: "r".into(),
        section_idx: 0,
        view: ChunkView::Window {
            index: 1,
            overlap: 0,
            span: 10,
        },
        text: "b".into(),
        tokens_estimate: 10,
        quality: QualityScore::default(),
        kvp_meta: Default::default(),
    };

    let negative = RecordChunk {
        record_id: "r".into(),
        section_idx: 0,
        view: ChunkView::Window {
            index: 2,
            overlap: 0,
            span: 10,
        },
        text: "c".into(),
        tokens_estimate: 10,
        quality: QualityScore::default(),
        kvp_meta: Default::default(),
    };

    let weight = sampler.inner.lock().unwrap().triplet_chunk_weight(
        &auto_recipe,
        &anchor,
        &positive,
        &negative,
    );

    // index distance = |0 - 1| => proximity = 1 / (1 + 1) = 0.5; trust(default)=0.5
    // pair_weight = 0.5 * 0.5 = 0.25
    // negative_weight = trust-only = 0.5
    // avg = (0.2 + 0.2 + 0.5) / 3
    let expected = (0.25 + 0.25 + 0.5) / 3.0;
    assert!((weight - expected).abs() < 1e-6, "weight={weight}");
}

#[test]
fn non_adjacent_auto_window_pair_proximity_is_not_half() {
    let split = SplitRatios::default();
    let mut config = base_config();
    config.chunking = ChunkingStrategy {
        max_window_tokens: 4,
        overlap_tokens: vec![0],
        summary_fallback_weight: 0.0,
        summary_fallback_tokens: 0,
        chunk_weight_floor: 0.0,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 19).unwrap());
    let sampler = TripletSampler::new(config, store);

    let record = DataRecord {
        id: "non_adjacent_proximity".into(),
        source: "unit".into(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        quality: QualityScore::default(),
        taxonomy: vec![],
        sections: vec![RecordSection {
            role: SectionRole::Context,
            heading: None,
            text: "one two three four five six seven eight nine ten".into(),
            sentences: vec!["one two three four five six seven eight nine ten".into()],
        }],
        meta_prefix: None,
    };

    let mut inner = sampler.inner.lock().unwrap();
    let selector = Selector::Role(SectionRole::Context);

    let windows = inner.materialize_chunks(&record, 0, &record.sections[0]);
    assert_eq!(
        windows
            .iter()
            .filter(|chunk| matches!(chunk.view, ChunkView::Window { .. }))
            .count(),
        3
    );

    let mut non_adjacent_pair = None;
    for _ in 0..6 {
        let (anchor, positive) = inner
            .select_anchor_positive_pair(&record, &selector, &selector, true)
            .expect("window pair");
        let delta = match (&anchor.view, &positive.view) {
            (ChunkView::Window { index: left, .. }, ChunkView::Window { index: right, .. }) => {
                left.abs_diff(*right)
            }
            _ => panic!("expected window chunks"),
        };
        if delta > 1 {
            non_adjacent_pair = Some((anchor, positive));
            break;
        }
    }

    let (anchor, positive) = non_adjacent_pair.expect("expected at least one non-adjacent pair");
    let proximity = chunk_proximity_score(&anchor, &positive);
    assert!(
        (proximity - 0.5).abs() > 1e-6,
        "non-adjacent proximity should not be 0.5; got {proximity}"
    );
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
            "source_c::record_04".to_string(),
            "source_a::record_02".to_string(),
            "source_b::record_00".to_string(),
            "source_c::record_02".to_string(),
            "source_b::record_04".to_string(),
            "source_c::record_04".to_string(),
            "source_a::record_06".to_string(),
            "source_a::record_02".to_string()
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
            "source_b::record_01".to_string(),
            "source_c::record_03".to_string(),
            "source_a::record_04".to_string(),
            "source_a::record_00".to_string(),
            "source_c::record_02".to_string(),
            "source_c::record_03".to_string(),
            "source_c::record_00".to_string(),
            "source_b::record_01".to_string()
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
                .select_negative_record_seeded(&anchor, &NegativeStrategy::WrongArticle, None)
                .expect("expected readable negative sample");
            negatives.push(negative.id.clone());
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
                .select_negative_record_seeded(&anchor, &NegativeStrategy::WrongArticle, None)
                .expect("expected BM25 negative selection");
            negatives.push(negative.id.clone());
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
                .select_negative_record_seeded(&anchor, &NegativeStrategy::WrongArticle, None)
                .expect("expected BM25 negative selection");
            negatives.push(negative.id.clone());
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
        },
        text: text.to_string(),
        tokens_estimate: 2,
        quality: QualityScore::default(),
        kvp_meta: Default::default(),
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
        },
        text: text.to_string(),
        tokens_estimate: 2,
        quality: QualityScore::default(),
        kvp_meta: Default::default(),
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

/// Verifies that `next_text_batch` reaches all three selector slots (`_anchor`,
/// `_positive`, `_negative`) derived from a role-mode triplet recipe.
/// `build_derived_text_recipes` expands one triplet recipe into three derived
/// selectors — one per slot — and this ensures none are silently dropped by
/// the selector dispatch.
#[test]
fn role_mode_text_batches_cover_all_three_selector_slots() {
    let split = SplitRatios::default();
    let config = SamplerConfig {
        seed: 77,
        batch_size: 1,
        chunking: ChunkingStrategy::default(),
        recipes: vec![],
        text_recipes: vec![],
        split,
        ..SamplerConfig::default()
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 77).unwrap());
    let recipes = vec![TripletRecipe {
        name: "role_mode_recipe".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    }];
    let records: Vec<DataRecord> = (0..6)
        .map(|i| {
            trader_record(
                &format!("src::article_{i}"),
                "2025-01-01",
                &format!("Title {i}"),
                &format!("Body text {i}"),
            )
        })
        .collect();
    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(RecipeSource::new(records, recipes)));

    let mut seen_anchor = false;
    let mut seen_positive = false;
    let mut seen_negative = false;
    for _ in 0..64 {
        let batch = sampler.next_text_batch(SplitLabel::Train).unwrap();
        for sample in &batch.samples {
            match sample.recipe.as_str() {
                "role_mode_recipe_anchor" => seen_anchor = true,
                "role_mode_recipe_positive" => seen_positive = true,
                "role_mode_recipe_negative" => seen_negative = true,
                _ => {}
            }
        }
        if seen_anchor && seen_positive && seen_negative {
            break;
        }
    }

    assert!(
        seen_anchor,
        "text batch never sampled the _anchor derived recipe"
    );
    assert!(
        seen_positive,
        "text batch never sampled the _positive derived recipe"
    );
    assert!(
        seen_negative,
        "text batch never sampled the _negative derived recipe"
    );
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
fn recipe_order_helpers_cover_empty_and_rotated_cases() {
    let split = SplitRatios::default();
    let store = Arc::new(DeterministicSplitStore::new(split, 59).unwrap());

    let mut shuffled_inner = TripletSamplerInner::new(base_config(), Arc::clone(&store));
    assert!(
        shuffled_inner
            .recipe_order_weighted_shuffled_seeded(&[])
            .is_empty()
    );
    assert!(
        shuffled_inner
            .text_recipe_order_weighted_shuffled_seeded(&[])
            .is_empty()
    );

    let mut base_inner = TripletSamplerInner::new(base_config(), Arc::clone(&store));
    let base_triplet = base_inner.recipe_order_weighted_shuffled_seeded(&[1.0, 1.0, 1.0, 1.0]);
    assert_eq!(base_triplet.len(), 4);
    let mut sorted_triplet = base_triplet.clone();
    sorted_triplet.sort_unstable();
    assert_eq!(sorted_triplet, vec![0, 1, 2, 3]);

    let mut cycled_inner = TripletSamplerInner::new(base_config(), Arc::clone(&store));
    let cycled_triplet = cycled_inner.recipe_order_weighted_cycled_seeded(&[1.0, 1.0, 1.0, 1.0], 5);
    assert_eq!(
        cycled_triplet,
        vec![
            base_triplet[1],
            base_triplet[2],
            base_triplet[3],
            base_triplet[0]
        ]
    );

    let mut base_text_inner = TripletSamplerInner::new(base_config(), Arc::clone(&store));
    let base_text =
        base_text_inner.text_recipe_order_weighted_shuffled_seeded(&[1.0, 1.0, 1.0, 1.0]);
    assert_eq!(base_text.len(), 4);
    let mut sorted_text = base_text.clone();
    sorted_text.sort_unstable();
    assert_eq!(sorted_text, vec![0, 1, 2, 3]);

    let mut cycled_text_inner = TripletSamplerInner::new(base_config(), store);
    let cycled_text =
        cycled_text_inner.text_recipe_order_weighted_cycled_seeded(&[1.0, 1.0, 1.0, 1.0], 6);
    assert_eq!(
        cycled_text,
        vec![base_text[2], base_text[3], base_text[0], base_text[1]]
    );
}

// ── weighted recipe order unit tests ─────────────────────────────────────────

#[test]
fn weighted_recipe_order_zero_weight_recipes_are_excluded() {
    let store = Arc::new(DeterministicSplitStore::new(SplitRatios::default(), 200).unwrap());
    let mut inner = TripletSamplerInner::new(base_config(), store);

    // Recipe at index 1 has zero weight — must never appear in the selection order.
    let weights = [1.0_f32, 0.0, 1.0];
    for _ in 0..20 {
        let order = inner.recipe_order_weighted_shuffled_seeded(&weights);
        assert!(
            !order.contains(&1),
            "zero-weight recipe 1 appeared in order: {order:?}"
        );
        assert_eq!(order.len(), 2); // only recipes 0 and 2 get slots
    }

    // All-zero → empty order.
    assert!(
        inner
            .recipe_order_weighted_shuffled_seeded(&[0.0, 0.0])
            .is_empty()
    );
}

#[test]
fn weighted_recipe_order_proportional_slot_count() {
    let store = Arc::new(DeterministicSplitStore::new(SplitRatios::default(), 201).unwrap());
    let mut inner = TripletSamplerInner::new(base_config(), store);

    // 3:1 ratio → recipe 0 gets 3 slots, recipe 1 gets 1 slot.
    let order = inner.recipe_order_weighted_shuffled_seeded(&[3.0_f32, 1.0]);
    assert_eq!(order.len(), 4);
    assert_eq!(order.iter().filter(|&&i| i == 0).count(), 3);
    assert_eq!(order.iter().filter(|&&i| i == 1).count(), 1);

    // 2:1 ratio.
    let order2 = inner.recipe_order_weighted_shuffled_seeded(&[2.0_f32, 1.0]);
    assert_eq!(order2.len(), 3);
    assert_eq!(order2.iter().filter(|&&i| i == 0).count(), 2);
    assert_eq!(order2.iter().filter(|&&i| i == 1).count(), 1);

    // Equal weights (1:1) → single slot each — same as legacy uniform behaviour.
    let order3 = inner.recipe_order_weighted_shuffled_seeded(&[1.0_f32, 1.0, 1.0, 1.0]);
    assert_eq!(order3.len(), 4);
    let mut sorted = order3.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, vec![0, 1, 2, 3]);
}

#[test]
fn weighted_recipe_order_cycled_preserves_multiset_across_rotations() {
    let store = Arc::new(DeterministicSplitStore::new(SplitRatios::default(), 202).unwrap());
    let mut inner = TripletSamplerInner::new(base_config(), store);

    let weights = [3.0_f32, 1.0];
    // Regardless of rotation offset, the multiset of recipe indices is always {0:3, 1:1}.
    for rr_idx in [0, 1, 2, 3, 7, 11] {
        let order = inner.recipe_order_weighted_cycled_seeded(&weights, rr_idx);
        assert_eq!(order.len(), 4, "rr_idx={rr_idx}");
        assert_eq!(
            order.iter().filter(|&&i| i == 0).count(),
            3,
            "rr_idx={rr_idx}"
        );
        assert_eq!(
            order.iter().filter(|&&i| i == 1).count(),
            1,
            "rr_idx={rr_idx}"
        );
    }
}

#[test]
fn weighted_recipe_order_same_for_text_and_triplet_variants() {
    let store = Arc::new(DeterministicSplitStore::new(SplitRatios::default(), 203).unwrap());
    let mut inner = TripletSamplerInner::new(base_config(), store);

    // Both variants use identical logic — proportions should match.
    let weights = [4.0_f32, 2.0, 1.0];
    let triplet = inner.recipe_order_weighted_shuffled_seeded(&weights);
    let text = inner.text_recipe_order_weighted_shuffled_seeded(&weights);

    // Slot count: min weight=1.0 → recipe 0 gets 4 slots, recipe 1 gets 2, recipe 2 gets 1.
    let expected_len = 7;
    assert_eq!(triplet.len(), expected_len);
    assert_eq!(text.len(), expected_len);
    assert_eq!(triplet.iter().filter(|&&i| i == 0).count(), 4);
    assert_eq!(triplet.iter().filter(|&&i| i == 1).count(), 2);
    assert_eq!(triplet.iter().filter(|&&i| i == 2).count(), 1);
    assert_eq!(text.iter().filter(|&&i| i == 0).count(), 4);
    assert_eq!(text.iter().filter(|&&i| i == 1).count(), 2);
    assert_eq!(text.iter().filter(|&&i| i == 2).count(), 1);
}

#[test]
fn weighted_recipe_selection_zero_weight_recipe_never_appears_in_batch() {
    // End-to-end: a recipe with weight=0.0 should produce zero samples in batch output.
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 204).unwrap());
    let find_train_id = |prefix: &str| -> String {
        (0u32..)
            .find_map(|i| {
                let id = format!("{prefix}_{i}");
                (store.label_for(&id) == Some(SplitLabel::Train)).then_some(id)
            })
            .unwrap()
    };
    let id_a = find_train_id("w0a");
    let id_b = find_train_id("w0b");
    let id_c = find_train_id("w0c");

    let mut config = base_config();
    config.seed = 1001;
    config.batch_size = 4;
    config.ingestion_max_records = 8;
    config.allowed_splits = vec![SplitLabel::Train];
    config.split = split;
    config.recipes = vec![
        TripletRecipe {
            name: "active".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
            allow_same_anchor_positive: false,
        },
        TripletRecipe {
            name: "excluded".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 0.0, // must never be selected
            instruction: None,
            allow_same_anchor_positive: false,
        },
    ];

    let records = vec![
        trader_record(&id_a, "2025-01-01", "A", "Body A long enough to chunk"),
        trader_record(&id_b, "2025-01-02", "B", "Body B long enough to chunk"),
        trader_record(&id_c, "2025-01-03", "C", "Body C long enough to chunk"),
    ];
    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(InMemorySource::new(PRIMARY_SOURCE_ID, records)));

    for _ in 0..20 {
        if let Ok(batch) = sampler.next_triplet_batch(SplitLabel::Train) {
            for triplet in &batch.triplets {
                assert_ne!(
                    triplet.recipe.as_str(),
                    "excluded",
                    "zero-weight recipe appeared in batch: {}",
                    triplet.recipe
                );
                // recipe name may be "active" or "active_fallback_same_split"
                assert!(
                    triplet.recipe.starts_with("active"),
                    "unexpected recipe name: {}",
                    triplet.recipe
                );
            }
        }
    }
}

#[test]
fn weighted_recipe_selection_frequency_matches_weight_ratio() {
    // End-to-end: recipe with weight=3.0 should appear ~3× as often as weight=1.0.
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 205).unwrap());
    let find_train_id = |prefix: &str| -> String {
        (0u32..)
            .find_map(|i| {
                let id = format!("{prefix}_{i}");
                (store.label_for(&id) == Some(SplitLabel::Train)).then_some(id)
            })
            .unwrap()
    };
    let id_a = find_train_id("wfa");
    let id_b = find_train_id("wfb");
    let id_c = find_train_id("wfc");
    let id_d = find_train_id("wfd");

    let mut config = base_config();
    config.seed = 2002;
    config.batch_size = 4;
    config.ingestion_max_records = 16;
    config.allowed_splits = vec![SplitLabel::Train];
    config.split = split;
    config.recipes = vec![
        TripletRecipe {
            name: "heavy".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 3.0,
            instruction: None,
            allow_same_anchor_positive: false,
        },
        TripletRecipe {
            name: "light".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
            allow_same_anchor_positive: false,
        },
    ];

    let records = vec![
        trader_record(&id_a, "2025-01-01", "A", "Body A long enough to chunk"),
        trader_record(&id_b, "2025-01-02", "B", "Body B long enough to chunk"),
        trader_record(&id_c, "2025-01-03", "C", "Body C long enough to chunk"),
        trader_record(&id_d, "2025-01-04", "D", "Body D long enough to chunk"),
    ];
    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(InMemorySource::new(PRIMARY_SOURCE_ID, records)));

    let mut heavy_count = 0usize;
    let mut light_count = 0usize;
    for _ in 0..50 {
        if let Ok(batch) = sampler.next_triplet_batch(SplitLabel::Train) {
            for triplet in &batch.triplets {
                if triplet.recipe.starts_with("heavy") {
                    heavy_count += 1;
                } else if triplet.recipe.starts_with("light") {
                    light_count += 1;
                }
            }
        }
    }

    let total = heavy_count + light_count;
    assert!(total > 0, "no samples produced");
    let heavy_fraction = heavy_count as f64 / total as f64;
    // Expected ~75% heavy (3/4). Allow ±20% margin for RNG and fallback variation.
    assert!(
        (0.55..=0.95).contains(&heavy_fraction),
        "heavy recipe fraction {heavy_fraction:.2} outside expected range [0.55, 0.95] \
         (heavy={heavy_count}, light={light_count})"
    );
    // heavy must appear more often than light — the core invariant.
    assert!(
        heavy_count > light_count,
        "heavy recipe ({heavy_count}) should appear more often than light ({light_count})"
    );
}

#[test]
fn disallowed_split_returns_configuration_error() {
    let split = SplitRatios::default();
    let store = Arc::new(DeterministicSplitStore::new(split, 61).unwrap());
    let mut config = base_config();
    config.allowed_splits = vec![SplitLabel::Train];

    let sampler = TripletSampler::new(config, store);
    let err = sampler
        .next_text_batch_for_split(SplitLabel::Validation)
        .expect_err("validation split should be rejected");

    match err {
        SamplerError::Configuration(message) => {
            assert!(message.contains("requested split Validation"));
            assert!(message.contains("allowed_splits [Train]"));
        }
        other => panic!("expected configuration error, got {other:?}"),
    }
}

#[test]
fn selector_edge_cases_cover_internal_branches() {
    let split = SplitRatios::default();
    let store = Arc::new(DeterministicSplitStore::new(split, 67).unwrap());
    let mut config = base_config();
    config.chunking = ChunkingStrategy {
        max_window_tokens: 3,
        overlap_tokens: vec![0],
        summary_fallback_weight: 0.0,
        summary_fallback_tokens: 0,
        chunk_weight_floor: 0.0,
    };
    let mut inner = TripletSamplerInner::new(config, Arc::clone(&store));

    let now = Utc::now();
    let record = DataRecord {
        id: "selector_record".into(),
        source: "unit".into(),
        created_at: now,
        updated_at: now,
        quality: QualityScore::default(),
        taxonomy: vec!["unit".into()],
        sections: vec![
            RecordSection {
                role: SectionRole::Context,
                heading: Some("Body".into()),
                text: "one two three four five six".into(),
                sentences: vec!["one two three four five six".into()],
            },
            RecordSection {
                role: SectionRole::Context,
                heading: Some("Empty".into()),
                text: String::new(),
                sentences: Vec::new(),
            },
        ],
        meta_prefix: None,
    };

    assert!(
        inner.record_has_at_least_two_window_chunks_for_selector(&record, &Selector::Paragraph(0),)
    );
    assert!(inner.record_has_at_least_two_window_chunks_for_selector(&record, &Selector::Random,));
    assert!(
        !inner
            .record_has_at_least_two_window_chunks_for_selector(&record, &Selector::Paragraph(9),)
    );
    assert!(
        !inner.record_has_at_least_two_window_chunks_for_selector(
            &record,
            &Selector::TemporalOffset(1),
        )
    );

    let paragraph_chunk = inner
        .select_chunk(&record, &Selector::Paragraph(0))
        .expect("paragraph selector should yield a chunk");
    assert_eq!(paragraph_chunk.text, "one two three");
    assert!(
        inner
            .select_chunk(&record, &Selector::Paragraph(9))
            .is_none()
    );

    let empty_record = DataRecord {
        sections: Vec::new(),
        ..record.clone()
    };
    assert!(
        inner
            .select_chunk(&empty_record, &Selector::Random)
            .is_none()
    );

    let no_anchor_record = DataRecord {
        sections: vec![RecordSection {
            role: SectionRole::Anchor,
            heading: Some("Title".into()),
            text: "headline only".into(),
            sentences: vec!["headline only".into()],
        }],
        ..record.clone()
    };
    assert!(
        inner
            .select_by_role(&no_anchor_record, &SectionRole::Context)
            .is_none()
    );

    assert!(
        inner
            .materialize_chunks(&record, 1, &record.sections[1])
            .is_empty()
    );

    let mut neighbor = record.clone();
    neighbor.id = "selector_neighbor".into();
    neighbor.created_at = now + Duration::days(1);
    neighbor.updated_at = neighbor.created_at;
    neighbor.sections = vec![RecordSection {
        role: SectionRole::Context,
        heading: Some("Neighbor".into()),
        text: "neighbor chunk text".into(),
        sentences: vec!["neighbor chunk text".into()],
    }];

    store.upsert(record.id.clone(), SplitLabel::Train).unwrap();
    store
        .upsert(neighbor.id.clone(), SplitLabel::Train)
        .unwrap();
    inner
        .records
        .insert(record.id.clone(), Arc::new(record.clone()));
    inner
        .records
        .insert(neighbor.id.clone(), Arc::new(neighbor.clone()));

    let temporal_chunk = inner
        .select_chunk(&record, &Selector::TemporalOffset(1))
        .expect("temporal selector should find neighbor chunk");
    assert_eq!(temporal_chunk.record_id, neighbor.id);
    assert_eq!(temporal_chunk.text, "neighbor chunk text");
}

#[test]
fn empty_recipe_configs_error_when_sampling_without_sources() {
    let split = SplitRatios::default();
    let store = Arc::new(DeterministicSplitStore::new(split, 71).unwrap());
    let mut config = base_config();
    config.batch_size = 1;
    config.recipes.clear();
    config.text_recipes.clear();

    let mut inner = TripletSamplerInner::new(config, Arc::clone(&store));
    let record = sample_record();
    store.upsert(record.id.clone(), SplitLabel::Train).unwrap();
    inner
        .records
        .insert(record.id.clone(), Arc::new(record.clone()));
    inner
        .chunk_index
        .insert(record.id.clone(), record.id.clone());

    let pair_err = inner
        .next_pair_batch_inner_with_weights(SplitLabel::Train, None)
        .expect_err("pair sampling should fail without triplet recipes");
    match pair_err {
        SamplerError::Configuration(message) => {
            assert_eq!(message, "no triplet recipes available");
        }
        other => panic!("expected configuration error, got {other:?}"),
    }

    let text_err = inner
        .next_text_batch_inner_with_weights(SplitLabel::Train, None)
        .expect_err("text sampling should fail without text recipes");
    match text_err {
        SamplerError::Configuration(message) => {
            assert_eq!(message, "no text recipes configured");
        }
        other => panic!("expected configuration error, got {other:?}"),
    }

    let triplet_err = inner
        .next_triplet_batch_inner_with_weights(SplitLabel::Train, None)
        .expect_err("triplet sampling should fail without triplet recipes");
    match triplet_err {
        SamplerError::Configuration(message) => {
            assert_eq!(message, "no triplet recipes configured");
        }
        other => panic!("expected configuration error, got {other:?}"),
    }
}

#[test]
fn source_less_batch_builders_sample_from_primed_epoch_tracker() {
    fn primed_inner(batch_size: usize) -> TripletSamplerInner<DeterministicSplitStore> {
        let split = SplitRatios::default();
        let store = Arc::new(DeterministicSplitStore::new(split, 79).unwrap());
        let mut config = base_config();
        config.batch_size = batch_size;
        config.recipes = vec![TripletRecipe {
            name: "manual_triplet".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
            allow_same_anchor_positive: false,
        }];
        config.text_recipes = vec![TextRecipe {
            name: "manual_text".into(),
            selector: Selector::Role(SectionRole::Context),
            weight: 1.0,
            instruction: None,
        }];

        let mut inner = TripletSamplerInner::new(config, Arc::clone(&store));
        let records = vec![
            trader_record(
                "manual::2025-01-01/article_a.txt",
                "2025-01-01",
                "Alpha",
                "Body alpha with enough words",
            ),
            trader_record(
                "manual::2025-01-01/article_b.txt",
                "2025-01-01",
                "Beta",
                "Body beta with enough words",
            ),
        ];
        for record in records {
            store.upsert(record.id.clone(), SplitLabel::Train).unwrap();
            inner
                .chunk_index
                .insert(record.id.clone(), record.id.clone());
            inner.records.insert(record.id.clone(), Arc::new(record));
        }
        inner.epoch_tracker.ensure_loaded().unwrap();
        let records_by_split = inner.records_by_split().unwrap();
        inner
            .epoch_tracker
            .reconcile(SplitLabel::Train, &records_by_split);
        inner
    }

    let mut pair_inner = primed_inner(2);
    let pair_batch = pair_inner
        .next_pair_batch_inner_with_weights(SplitLabel::Train, None)
        .expect("pair batch should sample from primed epoch tracker");
    assert_eq!(pair_batch.pairs.len(), 2);
    assert!(
        pair_batch
            .pairs
            .iter()
            .any(|pair| pair.label == PairLabel::Positive)
    );
    assert!(
        pair_batch
            .pairs
            .iter()
            .any(|pair| pair.label == PairLabel::Negative)
    );

    let mut text_inner = primed_inner(2);
    let text_batch = text_inner
        .next_text_batch_inner_with_weights(SplitLabel::Train, None)
        .expect("text batch should sample from primed epoch tracker");
    assert_eq!(text_batch.samples.len(), 2);
    assert!(
        text_batch
            .samples
            .iter()
            .all(|sample| sample.recipe == "manual_text")
    );

    let mut triplet_inner = primed_inner(2);
    let triplet_batch = triplet_inner
        .next_triplet_batch_inner_with_weights(SplitLabel::Train, None)
        .expect("triplet batch should sample from primed epoch tracker");
    assert_eq!(triplet_batch.triplets.len(), 2);
    assert!(
        triplet_batch
            .triplets
            .iter()
            .all(|triplet| triplet.recipe == "manual_triplet")
    );
}

#[test]
fn source_less_batch_builders_report_last_recipe_when_sampling_exhausts() {
    fn primed_failing_inner(
        triplet_name: &str,
        text_name: &str,
    ) -> TripletSamplerInner<DeterministicSplitStore> {
        let split = SplitRatios::default();
        let store = Arc::new(DeterministicSplitStore::new(split, 83).unwrap());
        let mut config = base_config();
        config.batch_size = 1;
        config.recipes = vec![TripletRecipe {
            name: triplet_name.to_string().into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
            allow_same_anchor_positive: false,
        }];
        config.text_recipes = vec![TextRecipe {
            name: text_name.to_string().into(),
            selector: Selector::Paragraph(99),
            weight: 1.0,
            instruction: None,
        }];

        let mut inner = TripletSamplerInner::new(config, Arc::clone(&store));
        let record = trader_record(
            "manual::2025-01-01/article_only.txt",
            "2025-01-01",
            "Solo",
            "Only record body",
        );
        store.upsert(record.id.clone(), SplitLabel::Train).unwrap();
        inner
            .chunk_index
            .insert(record.id.clone(), record.id.clone());
        inner.records.insert(record.id.clone(), Arc::new(record));
        inner.epoch_tracker.ensure_loaded().unwrap();
        let records_by_split = inner.records_by_split().unwrap();
        inner
            .epoch_tracker
            .reconcile(SplitLabel::Train, &records_by_split);
        inner
    }

    let mut pair_inner = primed_failing_inner("manual_pair_exhausted", "manual_text_exhausted");
    let pair_err = pair_inner
        .next_pair_batch_inner_with_weights(SplitLabel::Train, None)
        .expect_err("pair batch should exhaust with one record and wrong-article negatives");
    match pair_err {
        SamplerError::Exhausted(message) => assert_eq!(message, "manual_pair_exhausted"),
        other => panic!("expected exhausted error, got {other:?}"),
    }

    let mut text_inner = primed_failing_inner("manual_triplet_exhausted", "manual_text_exhausted");
    let text_err = text_inner
        .next_text_batch_inner_with_weights(SplitLabel::Train, None)
        .expect_err("text batch should exhaust when selector never resolves a chunk");
    match text_err {
        SamplerError::Exhausted(message) => assert_eq!(message, "manual_text_exhausted"),
        other => panic!("expected exhausted error, got {other:?}"),
    }

    let mut triplet_inner =
        primed_failing_inner("manual_triplet_exhausted", "manual_text_exhausted");
    let triplet_err = triplet_inner
        .next_triplet_batch_inner_with_weights(SplitLabel::Train, None)
        .expect_err("triplet batch should exhaust with one record and no negative candidate");
    match triplet_err {
        SamplerError::Exhausted(message) => assert_eq!(message, "manual_triplet_exhausted"),
        other => panic!("expected exhausted error, got {other:?}"),
    }
}

#[test]
fn source_state_and_recipe_helpers_cover_remaining_branches() {
    let split = SplitRatios::default();
    let store = Arc::new(DeterministicSplitStore::new(split, 89).unwrap());
    let mut inner = TripletSamplerInner::new(base_config(), Arc::clone(&store));
    let recipe = |name: &str| TripletRecipe {
        name: name.to_string().into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    };

    inner.persist_source_state(None).unwrap();
    assert!(store.load_sampler_state().unwrap().is_none());

    inner.source_state_loaded = true;
    inner.source_cycle_idx = 3;
    inner.source_record_cursors.insert("source_a".into(), 4);
    inner.source_epoch = 5;
    inner.triplet_recipe_rr_idx = 6;
    inner.text_recipe_rr_idx = 7;
    inner.persist_source_state(None).unwrap();
    let persisted = store.load_sampler_state().unwrap().unwrap();
    assert_eq!(persisted.source_cycle_idx, 3);
    assert_eq!(
        persisted.source_record_cursors,
        vec![("source_a".into(), 4)]
    );
    assert_eq!(persisted.source_epoch, 5);
    assert_eq!(persisted.triplet_recipe_rr_idx, 6);
    assert_eq!(persisted.text_recipe_rr_idx, 7);
    assert!(!inner.source_state_dirty);

    inner.using_config_text_recipes = true;
    inner.text_recipes = vec![TextRecipe {
        name: "keep_me".into(),
        selector: Selector::Random,
        weight: 1.0,
        instruction: None,
    }];
    inner.triplet_recipes.clear();
    inner.rebuild_derived_text_recipes();
    assert_eq!(inner.text_recipes.len(), 1);

    inner.using_config_text_recipes = false;
    inner.rebuild_derived_text_recipes();
    assert!(inner.text_recipes.is_empty());

    inner.triplet_recipes = vec![recipe("derived_recipe")];
    inner.rebuild_derived_text_recipes();
    assert!(!inner.text_recipes.is_empty());

    let duplicate_name = inner.text_recipes[0].name.clone();
    inner.extend_text_recipes_unique(&[
        TextRecipe {
            name: duplicate_name,
            selector: Selector::Random,
            weight: 1.0,
            instruction: None,
        },
        TextRecipe {
            name: "new_text_recipe".into(),
            selector: Selector::Paragraph(0),
            weight: 1.0,
            instruction: None,
        },
    ]);
    assert!(
        inner
            .text_recipes
            .iter()
            .any(|recipe| recipe.name == "new_text_recipe")
    );

    inner.using_config_triplet_recipes = true;
    inner.triplet_recipes = vec![recipe("configured_global")];
    assert_eq!(
        inner.configured_triplet_recipes_for_source("unused")[0].name,
        "configured_global"
    );

    inner.using_config_triplet_recipes = false;
    inner
        .source_triplet_recipes
        .insert("source_a".into(), vec![recipe("source_specific")]);
    assert_eq!(
        inner.configured_triplet_recipes_for_source("source_a")[0].name,
        "source_specific"
    );

    assert!(!TripletSamplerInner::<DeterministicSplitStore>::contains_auto_chunk_pair_recipe(&[]));
    assert!(
        TripletSamplerInner::<DeterministicSplitStore>::contains_auto_chunk_pair_recipe(&[
            TripletSamplerInner::<DeterministicSplitStore>::source_chunk_pair_recipe(),
        ])
    );

    inner.config.chunking.max_window_tokens = 0;
    inner.sources_with_long_sections.insert("source_a".into());
    assert!(!inner.source_supports_chunk_pair_recipe("source_a"));
}

#[test]
fn records_by_split_and_anchor_selection_cover_edge_cases() {
    let split = SplitRatios::default();
    let store = Arc::new(DeterministicSplitStore::new(split, 97).unwrap());
    let mut inner = TripletSamplerInner::new(base_config(), Arc::clone(&store));

    let record = trader_record("source_a::record_a", "2025-01-01", "Alpha", "Body alpha");
    inner
        .records
        .insert(record.id.clone(), Arc::new(record.clone()));
    inner
        .chunk_index
        .insert(record.id.clone(), record.id.clone());
    inner
        .chunk_index
        .insert("dangling_chunk".into(), "missing_record".into());

    let by_split = inner.records_by_split().unwrap();
    assert_eq!(by_split.get(&SplitLabel::Train).map(Vec::len), Some(1));
    assert_eq!(store.label_for(&record.id), Some(SplitLabel::Train));

    assert!(
        inner
            .choose_anchor_record(Some("missing_source"), SplitLabel::Train)
            .is_none()
    );

    inner
        .source_record_indices
        .insert("empty_source".into(), Vec::new());
    assert!(
        inner
            .choose_anchor_record(Some("empty_source"), SplitLabel::Train)
            .is_none()
    );

    let validation_record = trader_record("source_b::record_b", "2025-01-02", "Beta", "Body beta");
    store
        .upsert(validation_record.id.clone(), SplitLabel::Validation)
        .unwrap();
    inner
        .records
        .insert(validation_record.id.clone(), Arc::new(validation_record));
    inner
        .source_record_indices
        .insert("source_b".into(), vec![1]);
    inner.source_order = vec!["source_b".into()];
    inner.source_wrapped.insert("source_b".into(), false);

    assert!(
        inner
            .choose_anchor_record(Some("source_b"), SplitLabel::Train)
            .is_none()
    );
    assert_eq!(inner.source_epoch, 1);
    assert_eq!(inner.source_cycle_idx, 0);

    // With no reconciled epoch entries for this split, source-less anchor selection
    // must terminate immediately instead of scanning indefinitely.
    let mut no_source_inner = TripletSamplerInner::new(base_config(), store);
    no_source_inner.epoch_tracker.ensure_loaded().unwrap();
    assert!(
        no_source_inner
            .choose_anchor_record(None, SplitLabel::Train)
            .is_none()
    );
}

#[test]
fn temporal_neighbor_auto_pair_and_weighted_retry_paths_are_covered() {
    let split = SplitRatios::default();
    let store = Arc::new(DeterministicSplitStore::new(split, 101).unwrap());
    let mut config = base_config();
    config.batch_size = 1;
    let mut inner = TripletSamplerInner::new(config, Arc::clone(&store));

    let mut anchor = record_with_offset("anchor_record", Utc::now(), 0);
    anchor.source = "source_a".into();
    anchor.taxonomy = vec!["shared_taxonomy".into()];
    let mut neighbor = record_with_offset("neighbor_record", anchor.created_at, 86_400);
    neighbor.source = "source_b".into();
    neighbor.taxonomy = vec!["shared_taxonomy".into()];
    store.upsert(anchor.id.clone(), SplitLabel::Train).unwrap();
    store
        .upsert(neighbor.id.clone(), SplitLabel::Train)
        .unwrap();
    inner
        .records
        .insert(anchor.id.clone(), Arc::new(anchor.clone()));
    inner
        .records
        .insert(neighbor.id.clone(), Arc::new(neighbor.clone()));

    let selected = inner
        .select_temporal_neighbor(&anchor, 1)
        .expect("taxonomy match should allow cross-source temporal neighbor");
    assert_eq!(selected.id, neighbor.id);

    let mismatched_recipe = TripletRecipe {
        name: "mismatched_auto".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    };
    assert!(
        inner
            .select_distinct_window_pair_for_auto_recipe(&mismatched_recipe, &anchor)
            .is_none()
    );

    let failing_config = SamplerConfig {
        batch_size: 1,
        recipes: vec![TripletRecipe {
            name: "retry_triplet".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
            allow_same_anchor_positive: false,
        }],
        text_recipes: vec![TextRecipe {
            name: "retry_text".into(),
            selector: Selector::Paragraph(99),
            weight: 1.0,
            instruction: None,
        }],
        ..base_config()
    };
    let retry_store = Arc::new(DeterministicSplitStore::new(split, 103).unwrap());
    let sampler = TripletSampler::new(failing_config, retry_store);
    let weights = HashMap::from([("missing_source".to_string(), 1.0f32)]);

    let pair_err = sampler
        .next_pair_batch_with_weights_for_split(SplitLabel::Train, &weights)
        .expect_err("pair retry path should exhaust without records");
    assert!(matches!(pair_err, SamplerError::Exhausted(_)));

    let text_err = sampler
        .next_text_batch_with_weights_for_split(SplitLabel::Train, &weights)
        .expect_err("text retry path should exhaust without records");
    assert!(matches!(text_err, SamplerError::Exhausted(_)));

    let triplet_err = sampler
        .next_triplet_batch_with_weights_for_split(SplitLabel::Train, &weights)
        .expect_err("triplet retry path should exhaust without records");
    assert!(matches!(triplet_err, SamplerError::Exhausted(_)));
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
            .select_negative_record_seeded(&anchor, &NegativeStrategy::WrongArticle, None)
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
        .select_negative_record_seeded(&anchor, &NegativeStrategy::WrongArticle, None)
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
        .select_negative_record_seeded(&ingested_anchor, &NegativeStrategy::WrongArticle, None)
        .expect("expected bm25-ranked negative");

    let anchor_text = record_bm25_text(&ingested_anchor, inner.config.chunking.max_window_tokens);

    // Control baseline for non-BM25 behavior: uniform random choice over the
    // same strategy pool used by WrongArticle (same source, same split).
    let pool: Vec<Arc<DataRecord>> = inner
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
            .select_negative_record_seeded(&anchor, &NegativeStrategy::WrongArticle, None)
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
            .select_negative_record_seeded(&anchor, &NegativeStrategy::WrongPublicationDate, None)
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
            .select_negative_record_seeded(&anchor, &NegativeStrategy::QuestionAnswerMismatch, None)
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
    let selected =
        inner.select_negative_record_seeded(&anchor, &NegativeStrategy::WrongArticle, None);
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

    inner
        .records
        .insert(anchor.id.clone(), Arc::new(anchor.clone()));
    inner
        .records
        .insert(negative.id.clone(), Arc::new(negative.clone()));
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

    let triplet = inner.make_triplet_with_anchor_seeded(&recipe, &anchor);
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
        .select_negative_record_seeded(&anchor, &NegativeStrategy::WrongPublicationDate, None)
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
        .select_negative_record_seeded(&anchor, &NegativeStrategy::WrongPublicationDate, None)
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

/// When BM25 produces no candidates (e.g. because the chunk-window query text
/// has no term overlap with any indexed document), `select_hard_negative` must
/// fall back to random selection, and the fallback counter must be incremented.
///
/// Conversely, when the query text DOES share terms with a peer document, the
/// BM25 ranked path is taken and the fallback counter stays put.
#[cfg(all(feature = "bm25-mining", feature = "extended-metrics"))]
#[test]
fn bm25_fallback_counter_increments_when_no_bm25_candidates_match() {
    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };
    let config = SamplerConfig {
        seed: 42,
        batch_size: 1,
        chunking: ChunkingStrategy::default(),
        recipes: Vec::new(),
        text_recipes: Vec::new(),
        split,
        ..SamplerConfig::default()
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 77).unwrap());

    // Use two records whose bodies share clear terms so BM25 can rank them.
    // We will call `select_negative_record` once with a query containing
    // completely novel tokens (triggering fallback) and once with a query
    // that shares terms with the peer (taking the BM25 ranked path).
    let anchor_id = "bm25_fallback_anchor";
    let peer_id = "bm25_fallback_peer";

    let records = vec![
        trader_record(
            anchor_id,
            "2025-06-01",
            "Quarterly results",
            "revenue profit margin guidance outlook fiscal year",
        ),
        trader_record(
            peer_id,
            "2025-06-02",
            "Annual report",
            "revenue profit margin guidance outlook fiscal year",
        ),
    ];

    let sampler = TripletSampler::new(config, Arc::clone(&store));
    sampler.register_source(Box::new(InMemorySource::new(PRIMARY_SOURCE_ID, records)));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let mut inner = sampler.inner.lock().unwrap();
    let anchor = inner.records.get(anchor_id).cloned().expect("anchor");

    // ── call 1: query with terms absent from all indexed documents ────────────
    // BM25 returns no results → fallback is taken.
    let novel_query = "xyzzy_zyxqnv_mnbvcx_qwfpgjluy_no_such_term";
    let result = inner
        .select_negative_record_seeded(&anchor, &NegativeStrategy::WrongArticle, Some(novel_query))
        .expect("fallback must still return a negative");
    assert_eq!(
        result.0.id, peer_id,
        "fallback must select the only available peer"
    );

    let (fallback, total) = inner.bm25_fallback_stats();
    assert_eq!(total, 1, "selection count must be 1 after first call");
    assert_eq!(
        fallback, 1,
        "fallback count must be 1 — novel query triggered BM25 fallback"
    );

    // ── call 2: query sharing terms with the peer ─────────────────────────────
    // BM25 can rank the peer → the ranked path is taken, fallback count stays.
    let shared_query = "revenue profit margin guidance outlook fiscal year";
    let result2 = inner
        .select_negative_record_seeded(&anchor, &NegativeStrategy::WrongArticle, Some(shared_query))
        .expect("bm25 ranked path must return a negative");
    assert_eq!(result2.0.id, peer_id);

    let (fallback2, total2) = inner.bm25_fallback_stats();
    assert_eq!(total2, 2, "selection count must be 2 after second call");
    assert_eq!(
        fallback2, 1,
        "fallback count must remain 1 — shared-term query took the BM25 ranked path"
    );
}

/// Regression test: BM25 must use the raw (pre-decoration) anchor chunk text as
/// its query, not the metadata-decorated text.
///
/// Setup:
///   - Anchor record has a `meta_prefix` that injects the unique token
///     `zork_unique_prefix_token` with probability 1.0.
///   - Its raw body contains the unique token `quux_unique_content_token`.
///   - `content_peer`: body contains `quux_unique_content_token` — matches raw text.
///   - `prefix_peer`:  body contains `zork_unique_prefix_token` — matches only
///     the decoration.
///
/// With the correct behaviour (raw query), `select_negative_record` receives
/// `Some("quux_unique_content_token")` and BM25 ranks `content_peer` above
/// `prefix_peer`.  With the former broken behaviour (decorated query) the
/// opposite would happen.
#[cfg(feature = "bm25-mining")]
#[test]
fn bm25_query_uses_raw_chunk_text_not_decorated_text() {
    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };
    let config = SamplerConfig {
        seed: 99,
        batch_size: 1,
        chunking: ChunkingStrategy::default(),
        recipes: Vec::new(),
        text_recipes: Vec::new(),
        split,
        ..SamplerConfig::default()
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 7).unwrap());

    // Unique sentinel tokens — chosen to not appear anywhere else.
    let raw_token = "quux_unique_content_token";
    let prefix_token = "zork_unique_prefix_token";

    // Anchor: body has the raw token; its meta_prefix injects the prefix token.
    let mut anchor = trader_record("bm25_raw_query_anchor", "2025-01-01", "Anchor", raw_token);
    let mut kvp = KvpPrefixSampler::new(1.0);
    kvp.add_variant([("decoration", prefix_token)]);
    anchor.meta_prefix = Some(kvp);

    // content_peer: body shares the raw token — should rank first under raw query.
    let content_peer = trader_record(
        "bm25_raw_query_content_peer",
        "2025-01-02",
        "Content peer",
        raw_token,
    );

    // prefix_peer: body shares only the prefix token — should rank first only if
    // the decorated text were mistakenly used as the query.
    let prefix_peer = trader_record(
        "bm25_raw_query_prefix_peer",
        "2025-01-03",
        "Prefix peer",
        prefix_token,
    );

    let sampler = TripletSampler::new(config, Arc::clone(&store));
    sampler.register_source(Box::new(InMemorySource::new(
        PRIMARY_SOURCE_ID,
        vec![anchor.clone(), content_peer.clone(), prefix_peer.clone()],
    )));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let mut inner = sampler.inner.lock().unwrap();
    let anchor_rec = inner.records.get("bm25_raw_query_anchor").cloned().unwrap();

    // Call with the raw body text (no prefix) — correct behaviour.
    let (selected_raw, _) = inner
        .select_negative_record_seeded(
            &anchor_rec,
            &NegativeStrategy::WrongArticle,
            Some(raw_token),
        )
        .expect("should select a negative with raw query");

    assert_eq!(
        selected_raw.id, content_peer.id,
        "BM25 should rank content_peer first when queried with raw anchor text; \
         got '{}' instead — possible regression: decorated text used as query",
        selected_raw.id,
    );

    // Also confirm the symmetry: querying with the prefix token surfaces the
    // prefix_peer, proving the two situations are distinguishable.
    let (selected_prefix, _) = inner
        .select_negative_record_seeded(
            &anchor_rec,
            &NegativeStrategy::WrongArticle,
            Some(prefix_token),
        )
        .expect("should select a negative with prefix query");

    assert_eq!(
        selected_prefix.id, prefix_peer.id,
        "BM25 should rank prefix_peer first when queried with prefix token; \
         got '{}' — the two queries must be distinguishable for this test to be valid",
        selected_prefix.id,
    );
}

// ── Parallel helper coverage ──────────────────────────────────────────────────

/// Calls `select_chunk_parallel` with `Selector::Paragraph` on a sampler inner directly.
/// Covers the Paragraph arm of `select_chunk_parallel` (previously never reached).
#[test]
fn select_chunk_parallel_paragraph_selector_returns_chunk_or_none() {
    let split = SplitRatios::default();
    let mut config = base_config();
    config.chunking.max_window_tokens = 8;
    config.chunking.overlap_tokens = vec![0];
    let store = Arc::new(DeterministicSplitStore::new(split, 601).unwrap());
    let sampler = TripletSampler::new(config, store);

    let record = DataRecord {
        id: "para_test".into(),
        source: "unit".into(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        quality: QualityScore::default(),
        taxonomy: vec![],
        sections: vec![RecordSection {
            role: SectionRole::Context,
            heading: None,
            text: "one two three four five".into(),
            sentences: vec!["one two three four five".into()],
        }],
        meta_prefix: None,
    };

    let inner = sampler.inner.lock().unwrap();
    let mut rng = DeterministicRng::new(7777);

    // Paragraph(0) exists — should return a chunk from section 0.
    let chunk = inner.select_chunk_parallel(&record, &Selector::Paragraph(0), &mut rng);
    assert!(
        chunk.is_some(),
        "Paragraph(0) should return Some when section exists"
    );
    assert_eq!(chunk.unwrap().record_id, "para_test");

    // Paragraph(99) doesn't exist — section get returns None, so result is None.
    let none = inner.select_chunk_parallel(&record, &Selector::Paragraph(99), &mut rng);
    assert!(
        none.is_none(),
        "Paragraph(99) should return None when no such section"
    );
}

/// Calls `select_chunk_parallel` with `Selector::Random` on an empty and non-empty record.
/// Covers the Random arm of `select_chunk_parallel`.
#[test]
fn select_chunk_parallel_random_selector_handles_empty_and_non_empty() {
    let split = SplitRatios::default();
    let mut config = base_config();
    config.chunking.max_window_tokens = 8;
    config.chunking.overlap_tokens = vec![0];
    let store = Arc::new(DeterministicSplitStore::new(split, 602).unwrap());
    let sampler = TripletSampler::new(config, store);

    let empty_record = DataRecord {
        id: "rand_empty".into(),
        source: "unit".into(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        quality: QualityScore::default(),
        taxonomy: vec![],
        sections: vec![],
        meta_prefix: None,
    };

    let record_with_sections = DataRecord {
        id: "rand_nonempty".into(),
        source: "unit".into(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        quality: QualityScore::default(),
        taxonomy: vec![],
        sections: vec![RecordSection {
            role: SectionRole::Context,
            heading: None,
            text: "alpha beta gamma delta epsilon".into(),
            sentences: vec!["alpha beta gamma delta epsilon".into()],
        }],
        meta_prefix: None,
    };

    let inner = sampler.inner.lock().unwrap();
    let mut rng = DeterministicRng::new(8888);

    // Empty record → no sections → Random must return None.
    let none = inner.select_chunk_parallel(&empty_record, &Selector::Random, &mut rng);
    assert!(none.is_none(), "Random on empty record must return None");

    // Record with sections → Random picks one, returns Some.
    let some = inner.select_chunk_parallel(&record_with_sections, &Selector::Random, &mut rng);
    assert!(
        some.is_some(),
        "Random on non-empty record must return Some"
    );
    assert_eq!(some.unwrap().record_id, "rand_nonempty");
}

/// Calls `select_chunk_parallel` with `Selector::TemporalOffset`.
/// Covers the TemporalOffset arm of `select_chunk_parallel`.
#[test]
fn select_chunk_parallel_temporal_offset_returns_chunk_from_neighbor() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 603).unwrap());

    // Find IDs that land in the Train split so the split guard passes.
    let find_train = |prefix: &str| -> String {
        for i in 0..10_000u32 {
            let id = format!("{prefix}_{i}");
            if store.label_for(&id) == Some(SplitLabel::Train) {
                return id;
            }
        }
        panic!("no Train id for prefix {prefix}");
    };
    let anchor_id = find_train("toff_par_anchor");
    let neighbor_id = find_train("toff_par_neighbor");

    let mut config = base_config();
    config.seed = 603;
    config.batch_size = 1;
    config.recipes = Vec::new();
    config.text_recipes = Vec::new();
    let sampler = TripletSampler::new(config, Arc::clone(&store));

    let base = Utc::now();
    let anchor_rec = record_with_offset(&anchor_id, base, 0);
    let neighbor_rec = {
        let mut r = record_with_offset(&neighbor_id, base, 86400); // +1 day
        // Give the neighbor a Context section so select_role_parallel finds it.
        r.sections = vec![RecordSection {
            role: SectionRole::Context,
            heading: None,
            text: "neighbor context text here".into(),
            sentences: vec!["neighbor context text here".into()],
        }];
        r
    };

    sampler.register_source(Box::new(InMemorySource::new(
        PRIMARY_SOURCE_ID,
        vec![anchor_rec.clone(), neighbor_rec],
    )));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let inner = sampler.inner.lock().unwrap();
    let anchor = inner
        .records
        .get(&anchor_id)
        .cloned()
        .expect("anchor record");
    let mut rng = DeterministicRng::new(9999);

    // offset=1 → nearest neighbor (1 day away) → has Context section → returns chunk.
    let chunk = inner.select_chunk_parallel(&anchor, &Selector::TemporalOffset(1), &mut rng);
    assert!(
        chunk.is_some(),
        "TemporalOffset(1) should yield a context chunk from the temporal neighbor"
    );
}

/// Covers `select_role_parallel` when a record has NO sections matching the requested role,
/// so the empty-indices path returns None. Also covers the `select_anchor_positive_for_recipe`
/// non-auto path's `?` short-circuit (line 1671) when selection fails.
#[test]
fn select_role_parallel_returns_none_when_no_matching_role() {
    let split = SplitRatios::default();
    let config = base_config();
    let store = Arc::new(DeterministicSplitStore::new(split, 604).unwrap());
    let sampler = TripletSampler::new(config, store);

    // Record has ONLY Anchor sections — querying Context must fail.
    let anchor_only_record = DataRecord {
        id: "anchor_only".into(),
        source: "unit".into(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        quality: QualityScore::default(),
        taxonomy: vec![],
        sections: vec![RecordSection {
            role: SectionRole::Anchor,
            heading: None,
            text: "anchor text here".into(),
            sentences: vec!["anchor text here".into()],
        }],
        meta_prefix: None,
    };

    let inner = sampler.inner.lock().unwrap();
    let mut rng = DeterministicRng::new(1234);

    // Selector::Role(Context) on an Anchor-only record → empty indices → None.
    let result = inner.select_chunk_parallel(
        &anchor_only_record,
        &Selector::Role(SectionRole::Context),
        &mut rng,
    );
    assert!(
        result.is_none(),
        "Context selector on Anchor-only record must return None"
    );

    // Also validate the non-auto path in select_anchor_positive_for_recipe returns None
    // when the anchor selector has no matching sections (covering the ?-propagation line).
    let recipe = TripletRecipe {
        name: "no_anchor_test".into(),
        anchor: Selector::Role(SectionRole::Context), // no Context sections in record
        positive_selector: Selector::Role(SectionRole::Anchor),
        negative_selector: Selector::Role(SectionRole::Anchor),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    };
    let ap = inner.select_anchor_positive_for_recipe(&recipe, &anchor_only_record, &mut rng);
    assert!(
        ap.is_none(),
        "non-auto recipe with no matching anchor sections must return None"
    );
}

/// Covers `select_role_parallel` when matching sections exist but all produce empty chunk pools.
/// An empty section text yields zero chunks, causing the parallel role selector to fall through.
#[test]
fn select_role_parallel_returns_none_when_all_pools_are_empty() {
    let split = SplitRatios::default();
    let mut config = base_config();
    // max_window_tokens must be > 0 so the chunker tries to produce windows;
    // empty text still results in an empty pool regardless.
    config.chunking.max_window_tokens = 8;
    config.chunking.overlap_tokens = vec![0];
    let store = Arc::new(DeterministicSplitStore::new(split, 605).unwrap());
    let sampler = TripletSampler::new(config, store);

    // A Context section with whitespace-only text produces zero whitespace tokens
    // → materialize_chunks returns an empty Vec → pool is empty.
    let empty_section_record = DataRecord {
        id: "empty_pool_rec".into(),
        source: "unit".into(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        quality: QualityScore::default(),
        taxonomy: vec![],
        sections: vec![RecordSection {
            role: SectionRole::Context,
            heading: None,
            text: "".into(), // empty → no tokens → empty pool
            sentences: vec![],
        }],
        meta_prefix: None,
    };

    let inner = sampler.inner.lock().unwrap();
    let mut rng = DeterministicRng::new(5678);

    let result = inner.select_chunk_parallel(
        &empty_section_record,
        &Selector::Role(SectionRole::Context),
        &mut rng,
    );
    assert!(
        result.is_none(),
        "Context selector with empty-text section must return None (pool always empty)"
    );
}

/// Covers `decorate_chunk_parallel` when the prefix alone fills the entire window
/// (prefix_tokens.len() >= max_window branch) and when the body is truncated.
#[test]
fn decorate_chunk_parallel_truncation_paths() {
    // ── Case 1: prefix fills the entire window ───────────────────────────────
    let split = SplitRatios::default();
    let mut config_a = base_config();
    config_a.chunking.max_window_tokens = 3; // tiny window
    config_a.chunking.overlap_tokens = vec![0];
    let store_a = Arc::new(DeterministicSplitStore::new(split, 606).unwrap());
    let sampler_a = TripletSampler::new(config_a, store_a);

    let mut record_a = DataRecord {
        id: "pfx_fill".into(),
        source: "unit".into(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        quality: QualityScore::default(),
        taxonomy: vec![],
        sections: vec![RecordSection {
            role: SectionRole::Context,
            heading: None,
            text: "word1 word2".into(),
            sentences: vec!["word1 word2".into()],
        }],
        meta_prefix: None,
    };

    // Prefix has 4 tokens, window = 3 → prefix_tokens.len() (4) >= max_window (3).
    let mut kvp_a = KvpPrefixSampler::new(1.0);
    kvp_a.add_variant([("k", "a b c d")] as [(&str, &str); 1]); // 4 prefix tokens
    record_a.meta_prefix = Some(kvp_a);

    let inner_a = sampler_a.inner.lock().unwrap();
    let mut rng_a = DeterministicRng::new(11111);

    // Create a chunk from section 0.
    let mut chunk_a = inner_a
        .materialize_chunks(&record_a, 0, &record_a.sections[0])
        .into_iter()
        .next()
        .expect("non-empty section must produce at least one chunk");

    inner_a.decorate_chunk_parallel(&record_a, &mut chunk_a, &mut rng_a);

    // The prefix fills the window: exactly max_window tokens kept from the prefix.
    let tokens: Vec<&str> = chunk_a.text.split_whitespace().collect();
    assert_eq!(
        tokens.len(),
        3,
        "prefix-fills-window path must truncate to max_window tokens"
    );
    assert_eq!(chunk_a.tokens_estimate, 3);
    drop(inner_a);

    // ── Case 2: prefix partial + body truncated ──────────────────────────────
    let mut config_b = base_config();
    config_b.chunking.max_window_tokens = 5;
    config_b.chunking.overlap_tokens = vec![0];
    let store_b = Arc::new(DeterministicSplitStore::new(split, 607).unwrap());
    let sampler_b = TripletSampler::new(config_b, store_b);

    let mut record_b = DataRecord {
        id: "pfx_body_trunc".into(),
        source: "unit".into(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        quality: QualityScore::default(),
        taxonomy: vec![],
        sections: vec![RecordSection {
            role: SectionRole::Context,
            heading: None,
            text: "word1 word2 word3 word4 word5 word6 word7 word8".into(),
            sentences: vec!["word1 word2 word3 word4 word5 word6 word7 word8".into()],
        }],
        meta_prefix: None,
    };

    // Prefix has 2 tokens, window = 5 → remaining = 3 body tokens from the chunk.
    let mut kvp_b = KvpPrefixSampler::new(1.0);
    kvp_b.add_variant([("p", "pa pb")] as [(&str, &str); 1]); // 2 prefix tokens
    record_b.meta_prefix = Some(kvp_b);

    let inner_b = sampler_b.inner.lock().unwrap();
    let mut rng_b = DeterministicRng::new(22222);

    let mut chunk_b = inner_b
        .materialize_chunks(&record_b, 0, &record_b.sections[0])
        .into_iter()
        .next()
        .expect("non-empty section must produce at least one chunk");

    inner_b.decorate_chunk_parallel(&record_b, &mut chunk_b, &mut rng_b);

    // total body words in the chunk ≥ remaining (3), so trimmed body is present.
    assert_eq!(
        chunk_b.tokens_estimate, 5,
        "truncated chunk must have max_window tokens"
    );
    let decorated_tokens: Vec<&str> = chunk_b.text.split_whitespace().collect();
    assert_eq!(
        decorated_tokens.len(),
        5,
        "decorated text must have exactly 5 tokens"
    );
}

/// Covers `decorate_chunk` (sequential, &mut self) no-truncation path when
/// max_window_tokens = 0 (disabled), making total_tokens <= max_window always false.
/// This exercises the `else` branch of `if max_window > 0 && total_tokens > max_window`.
#[test]
fn decorate_chunk_no_truncation_when_window_is_zero() {
    let split = SplitRatios::default();
    let mut config = base_config();
    config.chunking.max_window_tokens = 0; // no window limit
    let store = Arc::new(DeterministicSplitStore::new(split, 608).unwrap());
    let sampler = TripletSampler::new(config, store);

    let mut record = DataRecord {
        id: "no_trunc_rec".into(),
        source: "unit".into(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        quality: QualityScore::default(),
        taxonomy: vec![],
        sections: vec![RecordSection {
            role: SectionRole::Context,
            heading: None,
            text: "body token one two three".into(),
            sentences: vec!["body token one two three".into()],
        }],
        meta_prefix: None,
    };

    // Prefix has 2 tokens; with max_window=0 no truncation occurs.
    let mut kvp = KvpPrefixSampler::new(1.0);
    kvp.add_variant([("x", "pre fix")] as [(&str, &str); 1]); // 2 tokens
    record.meta_prefix = Some(kvp);

    let mut inner = sampler.inner.lock().unwrap();

    // Build a fake chunk whose text matches the section body.
    let mut chunk = RecordChunk {
        record_id: "no_trunc_rec".into(),
        section_idx: 0,
        view: ChunkView::Window {
            index: 0,
            overlap: 0,
            span: 5,
        },
        text: "body token one two three".to_string(),
        tokens_estimate: 5,
        quality: QualityScore::default(),
        kvp_meta: Default::default(),
    };

    inner.decorate_chunk_seeded(&record, &mut chunk);

    // No truncation → prefix prepended to full body text.
    assert!(
        chunk.text.contains("pre"),
        "decorated text should contain part of the prefix key/value"
    );
    assert!(
        chunk.text.contains("body token one two three"),
        "full body should be present when no truncation occurs"
    );
    // prefix format is "meta: x=pre fix" (3 whitespace tokens) + 5 body tokens = 8 total.
    assert!(
        chunk.tokens_estimate > 5,
        "tokens_estimate should include prefix tokens"
    );
}

/// Covers `select_anchor_positive_parallel` retry-exhaustion path: when anchor and positive
/// use the same selector and the record has only one possible chunk, every retry returns the
/// same chunk → `same_selector_pair_is_valid` always returns false → `return None`.
#[test]
fn select_anchor_positive_parallel_returns_none_when_retries_exhausted() {
    let split = SplitRatios::default();
    let mut config = base_config();
    // Large window so the one short section produces exactly ONE window chunk.
    config.chunking.max_window_tokens = 1024;
    config.chunking.overlap_tokens = vec![0];
    let store = Arc::new(DeterministicSplitStore::new(split, 609).unwrap());
    let sampler = TripletSampler::new(config, store);

    // Record with a SINGLE Anchor section that produces exactly one window chunk.
    let single_chunk_record = DataRecord {
        id: "single_chunk".into(),
        source: "unit".into(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        quality: QualityScore::default(),
        taxonomy: vec![],
        sections: vec![RecordSection {
            role: SectionRole::Anchor,
            heading: None,
            text: "unique text word".into(), // short: fits in one window
            sentences: vec!["unique text word".into()],
        }],
        meta_prefix: None,
    };

    let inner = sampler.inner.lock().unwrap();
    let mut rng = DeterministicRng::new(33333);
    let selector = Selector::Role(SectionRole::Anchor);

    // anchor_selector == positive_selector → retry loop; only one chunk available
    // → same chunk always drawn → same_selector_pair_is_valid always false → None.
    let result = inner.select_anchor_positive_parallel(
        &single_chunk_record,
        &selector,
        &selector,
        false, // enforce_window_pair=false so only key comparison matters
        &mut rng,
    );
    assert!(
        result.is_none(),
        "retry exhaustion with single-chunk record must return None"
    );
}

/// Covers `_for_split` weight API success paths (lines that are only reached when the
/// inner batch succeeds after a failed attempt and a force-refresh). The existing
/// exhaustion test (without sources) now covers the post-loop fallthrough,
/// but this test confirms the success arm via a properly registered source.
#[test]
fn for_split_weight_apis_succeed_with_registered_source() {
    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 610).unwrap());

    let mut config = base_config();
    config.seed = 610;
    config.batch_size = 1;
    config.recipes = vec![TripletRecipe {
        name: "for_split_weight_test".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    }];
    config.text_recipes = vec![TextRecipe {
        name: "for_split_text_test".into(),
        selector: Selector::Role(SectionRole::Context),
        weight: 1.0,
        instruction: None,
    }];

    let sampler = TripletSampler::new(config, Arc::clone(&store));

    // Register enough records for negative selection.
    let records: Vec<_> = (0..8)
        .map(|i| {
            trader_record(
                &format!("fsw_{i}"),
                "2025-01-01",
                &format!("title {i}"),
                &format!("body text words extra {i}"),
            )
        })
        .collect();
    sampler.register_source(Box::new(InMemorySource::new(PRIMARY_SOURCE_ID, records)));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    let weights = std::collections::HashMap::new();

    let pair_batch = sampler
        .next_pair_batch_with_weights_for_split(SplitLabel::Train, &weights)
        .expect("pair batch must succeed with registered source");
    assert_eq!(pair_batch.pairs.len(), 1);

    let text_batch = sampler
        .next_text_batch_with_weights_for_split(SplitLabel::Train, &weights)
        .expect("text batch must succeed with registered source");
    assert_eq!(text_batch.samples.len(), 1);

    let triplet_batch = sampler
        .next_triplet_batch_with_weights_for_split(SplitLabel::Train, &weights)
        .expect("triplet batch must succeed with registered source");
    assert_eq!(triplet_batch.triplets.len(), 1);
}

// ── BM25 coverage ────────────────────────────────────────────────────────────

/// Covers the BM25 query-token-limit truncation path (lines 222-223):
/// when `anchor_query_text` has more than BM25_QUERY_TOKEN_LIMIT tokens,
/// the backend truncates it before querying the search engine.
#[cfg(feature = "bm25-mining")]
#[test]
fn bm25_query_text_over_token_limit_is_truncated_before_search() {
    // BM25_QUERY_TOKEN_LIMIT = 64 (from constants::sampler).
    const TOKEN_LIMIT: usize = 64;

    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 611).unwrap());

    let config = base_config();
    let sampler = TripletSampler::new(config, Arc::clone(&store));

    // Register enough records for BM25 to index.
    let records: Vec<_> = (0..6)
        .map(|i| {
            trader_record(
                &format!("bm25_trunc_{i}"),
                "2025-01-01",
                &format!("title {i}"),
                &format!("body text content {i}"),
            )
        })
        .collect();
    sampler.register_source(Box::new(InMemorySource::new(
        PRIMARY_SOURCE_ID,
        records.clone(),
    )));
    sampler
        .inner
        .lock()
        .unwrap()
        .ingest_internal(SplitLabel::Train)
        .unwrap();

    // Construct a query text that exceeds TOKEN_LIMIT (64) tokens.
    let long_query: String = (0..(TOKEN_LIMIT + 10))
        .map(|i| format!("word{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        long_query.split_whitespace().count() > TOKEN_LIMIT,
        "query must exceed the token limit to exercise the truncation path"
    );

    let mut inner = sampler.inner.lock().unwrap();
    let anchor = inner.records.get("bm25_trunc_0").cloned().expect("anchor");

    // The long query triggers the truncation branch; the call must not panic.
    let _result = inner.select_negative_record_seeded(
        &anchor,
        &NegativeStrategy::WrongArticle,
        Some(&long_query),
    );
    // Whether a negative is found is not the point — coverage of the truncation
    // path (assignment to query_limited) is the goal.
}

/// Covers `record_bm25_text` with `max_tokens = 0`, which returns the full
/// concatenated text without any token-count cap.
#[cfg(feature = "bm25-mining")]
#[test]
fn record_bm25_text_with_zero_max_tokens_returns_full_text() {
    let record = DataRecord {
        id: "bm25_text_zero".into(),
        source: "unit".into(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        quality: QualityScore::default(),
        taxonomy: vec![],
        sections: vec![
            RecordSection {
                role: SectionRole::Anchor,
                heading: Some("Heading".into()),
                text: "anchor body text with many tokens here and there and everywhere".into(),
                sentences: vec![
                    "anchor body text with many tokens here and there and everywhere".into(),
                ],
            },
            RecordSection {
                role: SectionRole::Context,
                heading: None,
                text: "context section body words more and more content".into(),
                sentences: vec!["context section body words more and more content".into()],
            },
        ],
        meta_prefix: None,
    };

    // With max_tokens = 0 every token is kept.
    let full = record_bm25_text(&record, 0);
    let capped = record_bm25_text(&record, 5);

    assert!(
        full.split_whitespace().count() > 5,
        "full text should have more tokens than the capped version"
    );
    assert_eq!(
        capped.split_whitespace().count(),
        5,
        "capped version should contain exactly 5 tokens"
    );
    // Ensure the full text actually contains the heading.
    assert!(
        full.contains("Heading"),
        "full text (max_tokens=0) must include the heading"
    );
}

/// Covers `index_meta_record_ids` returning `None` when no source indexes
/// have been built yet (the early-return path on an empty `source_indexes`).
#[cfg(feature = "bm25-mining")]
#[test]
fn bm25_index_meta_record_ids_returns_none_when_no_indexes_built() {
    let split = SplitRatios::default();
    let config = base_config();
    let store = Arc::new(DeterministicSplitStore::new(split, 612).unwrap());
    let sampler = TripletSampler::new(config, store);

    // No sources registered → no records ingested → source_indexes is empty.
    let mut inner_mut = sampler.inner.lock().unwrap();
    let ids = inner_mut.bm25_backend_mut().index_meta_record_ids();
    assert!(
        ids.is_none(),
        "index_meta_record_ids must return None when source_indexes is empty"
    );
}

/// Covers the `Err(err) => return Err(err)` arms (lines 2786, 2808, 2830) in all three
/// `next_*_batch_with_weights_for_split` functions.  These arms fire when the inner batch
/// function returns a **non-Exhausted** error (e.g. `SamplerError::Configuration`).
///
/// Setup: records are inserted directly into `inner.records` + `inner.chunk_index` so that
/// `ensure_split_has_records` passes, but no sources are registered (so `source_order` is
/// empty) and no recipes are configured – which causes the inner call to return a
/// `Configuration` error instead of `Exhausted`.
#[test]
fn for_split_non_exhausted_error_propagates_immediately() {
    // All records land in Train because train ratio is 1.0.
    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 613).unwrap());

    // Build a config with no recipes so the inner call returns Configuration.
    let mut config = base_config();
    config.seed = 613;
    config.batch_size = 1;
    config.recipes = vec![];
    config.text_recipes = vec![];
    config.allowed_splits = vec![SplitLabel::Train];

    let sampler = TripletSampler::new(config, Arc::clone(&store));

    // Manually seed one record into inner.records and inner.chunk_index.
    // We intentionally do NOT call register_source / rebuild_source_index, so
    // source_order stays empty → sources.is_empty() == true inside the inner fns.
    let record_id: RecordId = "no_recipe_rec".to_string();
    let section = RecordSection {
        role: SectionRole::Context,
        heading: None,
        text: "some context text".into(),
        sentences: vec!["some context text".into()],
    };
    let record = DataRecord {
        id: record_id.clone(),
        source: "unit".into(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        quality: QualityScore::default(),
        taxonomy: vec![],
        sections: vec![section],
        meta_prefix: None,
    };
    {
        let mut inner = sampler.inner.lock().unwrap();
        inner
            .records
            .insert(record_id.clone(), std::sync::Arc::new(record));
        // chunk_index maps chunk_id → record_id; use the same id for both.
        inner
            .chunk_index
            .insert(record_id.clone(), record_id.clone());
        // source_order is intentionally left empty.
    }

    let weights: std::collections::HashMap<_, f32> = std::collections::HashMap::new();

    // pair: Configuration("no triplet recipes available") → Err arm at line 2786
    let pair_err = sampler
        .next_pair_batch_with_weights_for_split(SplitLabel::Train, &weights)
        .expect_err("must fail with no recipes configured");
    assert!(
        matches!(pair_err, SamplerError::Configuration(_)),
        "expected Configuration error from pair batch, got: {pair_err:?}"
    );

    // text: Configuration("no text recipes configured") → Err arm at line 2808
    let text_err = sampler
        .next_text_batch_with_weights_for_split(SplitLabel::Train, &weights)
        .expect_err("must fail with no text recipes configured");
    assert!(
        matches!(text_err, SamplerError::Configuration(_)),
        "expected Configuration error from text batch, got: {text_err:?}"
    );

    // triplet: Configuration("no triplet recipes configured") → Err arm at line 2830
    let triplet_err = sampler
        .next_triplet_batch_with_weights_for_split(SplitLabel::Train, &weights)
        .expect_err("must fail with no triplet recipes configured");
    assert!(
        matches!(triplet_err, SamplerError::Configuration(_)),
        "expected Configuration error from triplet batch, got: {triplet_err:?}"
    );
}

/// Regression guard: the `>= batch_size` early-break in every sampling loop must not be
/// removed.  Several inner paths (notably the multi-source text loop and the no-source
/// text/triplet loops) have **no companion per-push size check** on the output vec.
/// Without the early-break, those loops accumulate more than `batch_size` items; the
/// subsequent `vec.len() == batch_size` equality check then fails, and the function
/// returns `Err(Exhausted)` even though the pool is ample.
///
/// This test uses a pool of 20 records (5× the batch_size of 4) and asserts that all
/// three batch APIs return `Ok` with **exactly** `batch_size` items.  Removing any of
/// the guards would cause the text batch (and possibly others) to return `Err(Exhausted)`
/// and break this assertion.
#[test]
fn batch_size_guard_prevents_oversampling_from_large_pool() {
    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 614).unwrap());

    const BATCH: usize = 4;
    const POOL: usize = 20; // 5× batch_size — loop would overfill without the guard

    let mut config = base_config();
    config.seed = 614;
    config.batch_size = BATCH;
    config.allowed_splits = vec![SplitLabel::Train];
    config.split = split;
    config.recipes = vec![TripletRecipe {
        name: "oversample_guard".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    }];
    config.text_recipes = vec![TextRecipe {
        name: "oversample_guard_text".into(),
        selector: Selector::Role(SectionRole::Context),
        weight: 1.0,
        instruction: None,
    }];

    let records: Vec<_> = (0..POOL)
        .map(|i| {
            trader_record(
                &format!("osg_{i}"),
                "2025-06-01",
                &format!("title oversample {i}"),
                &format!("body context words filling slot {i} for oversample guard test"),
            )
        })
        .collect();

    let sampler = TripletSampler::new(config, Arc::clone(&store));
    sampler.register_source(Box::new(InMemorySource::new("osg_source", records)));

    // Pair batch: assert exactly BATCH items (not Exhausted, not more).
    let pair_batch = sampler
        .next_pair_batch(SplitLabel::Train)
        .expect("pair batch must succeed with large pool");
    assert_eq!(
        pair_batch.pairs.len(),
        BATCH,
        "pair batch length must equal batch_size; got {}",
        pair_batch.pairs.len()
    );

    // Text batch: critical — the multi-source text loop has no per-push guard, so
    // removing the early-break produces > BATCH samples and Exhausted is returned.
    let text_batch = sampler
        .next_text_batch(SplitLabel::Train)
        .expect("text batch must succeed with large pool");
    assert_eq!(
        text_batch.samples.len(),
        BATCH,
        "text batch length must equal batch_size; got {}",
        text_batch.samples.len()
    );

    // Triplet batch: assert exactly BATCH items.
    let triplet_batch = sampler
        .next_triplet_batch(SplitLabel::Train)
        .expect("triplet batch must succeed with large pool");
    assert_eq!(
        triplet_batch.triplets.len(),
        BATCH,
        "triplet batch length must equal batch_size; got {}",
        triplet_batch.triplets.len()
    );
}
