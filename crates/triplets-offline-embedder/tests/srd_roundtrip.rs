//! Integration tests: verify data flows correctly through the ACTUAL scheduler
//! code path (run_interleaved_loop → step → accumulate → flush_pending →
//! SrdStoreAdapter::write_pairs → file on disk).

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tempfile::TempDir;

use triplets_core::SplitLabel;
use triplets_core::config::SamplerConfig;
use triplets_core::data::PairLabel;
use triplets_core::source::DataSource;
use triplets_offline_embedder::loop_runner::{LoopEvent, LoopHandler, run_interleaved_loop};
use triplets_offline_embedder::sampler_prefetcher::SamplerPrefetcher;
use triplets_offline_embedder::store_adapter::init_split_states_with_batch;
use triplets_offline_embedder::traits::*;
use triplets_srd_source::SrdSource;
use triplets_srd_source::srd_triplet::SrdMode;

// ---------------------------------------------------------------------------
// Mock infrastructure
// ---------------------------------------------------------------------------

/// Deterministic embedder: returns vecs where v[0] = hash of text.
struct TestEmbedder;

impl Embedder for TestEmbedder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let id: f32 = t.chars().map(|c| c as u32 as f32).sum();
                vec![id, 0.0, 0.0]
            })
            .collect())
    }
}

/// Returns N batches of `batch_size` entries per split, then exhausted.
struct TestProvider {
    batches_remaining: std::sync::atomic::AtomicUsize,
    batch_size: usize,
}

impl TestProvider {
    fn new(num_batches: usize, batch_size: usize) -> Self {
        Self {
            batches_remaining: std::sync::atomic::AtomicUsize::new(num_batches),
            batch_size,
        }
    }
}

impl BatchProvider for TestProvider {
    fn next_batch(&self, _split: SplitLabel) -> Result<Option<SamplerBatch>> {
        let remaining = self.batches_remaining.fetch_sub(1, Ordering::Relaxed);
        if remaining == 0 {
            return Ok(None);
        }
        let n = self.batch_size;
        let entries: Vec<PairEntry> = (0..n)
            .map(|i| PairEntry {
                anchor_text: format!("anchor_{remaining}_{i}"),
                candidate_text: format!("pos_{remaining}_{i}"),
                label: PairLabel::Positive,
            })
            .collect();
        Ok(Some(SamplerBatch::Pairs(entries)))
    }

    fn save_state(&self) -> Result<()> {
        Ok(())
    }
}

struct NoopHandler;

impl LoopHandler for NoopHandler {
    fn handle_event(&mut self, _event: &LoopEvent) {}
}

/// Split-aware provider: returns different text prefixes per split.
/// Each split gets its own batch counter so they don't interfere.
struct SplitAwareProvider {
    train_batches: std::sync::atomic::AtomicUsize,
    val_batches: std::sync::atomic::AtomicUsize,
    batch_size: usize,
}

impl SplitAwareProvider {
    fn new(num_batches_per_split: usize, batch_size: usize) -> Self {
        Self {
            train_batches: std::sync::atomic::AtomicUsize::new(num_batches_per_split),
            val_batches: std::sync::atomic::AtomicUsize::new(num_batches_per_split),
            batch_size,
        }
    }
}

impl BatchProvider for SplitAwareProvider {
    fn next_batch(&self, split: SplitLabel) -> Result<Option<SamplerBatch>> {
        let counter = match split {
            SplitLabel::Train => &self.train_batches,
            SplitLabel::Validation => &self.val_batches,
            _ => return Ok(None),
        };
        let remaining = counter.fetch_sub(1, Ordering::Relaxed);
        if remaining == 0 {
            return Ok(None);
        }
        let prefix = match split {
            SplitLabel::Train => "train",
            SplitLabel::Validation => "val",
            _ => "other",
        };
        let n = self.batch_size;
        let entries: Vec<PairEntry> = (0..n)
            .map(|i| PairEntry {
                anchor_text: format!("{prefix}_a{remaining}_{i}"),
                candidate_text: format!("{prefix}_p{remaining}_{i}"),
                label: PairLabel::Positive,
            })
            .collect();
        Ok(Some(SamplerBatch::Pairs(entries)))
    }

    fn save_state(&self) -> Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests through the REAL scheduler code path
// ---------------------------------------------------------------------------

#[test]
fn scheduler_writes_train_val_to_separate_files() {
    let dir = TempDir::new().unwrap();
    let split_descs = vec![
        (SplitLabel::Train, "train", 100u64, 0.8f32),
        (SplitLabel::Validation, "val", 50u64, 0.2f32),
    ];

    let mut states =
        init_split_states_with_batch(dir.path(), &split_descs, 4, SrdMode::Pair, 3, 2).unwrap();
    assert_eq!(states[0].name, "train");
    assert_eq!(states[1].name, "val");

    // 2 batches of 2 entries each → 4 entries per split, flushed at steps_per_batch=2
    let provider = Arc::new(SplitAwareProvider::new(2, 2));
    let embedder = TestEmbedder;
    let config = SchedulerConfig::new(2, 2, 3, 2);

    let mut prefetchers = HashMap::new();
    prefetchers.insert(
        SplitLabel::Train,
        SamplerPrefetcher::new(provider.clone(), SplitLabel::Train, 4),
    );
    prefetchers.insert(
        SplitLabel::Validation,
        SamplerPrefetcher::new(provider.clone(), SplitLabel::Validation, 4),
    );

    let stop = AtomicBool::new(false);
    run_interleaved_loop(
        &mut states,
        &mut prefetchers,
        &embedder,
        provider.as_ref(),
        &config,
        &stop,
        &mut NoopHandler,
    )
    .unwrap();

    // Files must exist.
    assert!(dir.path().join("train/data.srd").exists());
    assert!(dir.path().join("val/data.srd").exists());

    // Read train via SrdSource — goes through actual file, not mock.
    let train_source = SrdSource::open(
        &dir.path().join("train/data.srd"),
        "train",
        3,
        SrdMode::Pair,
    )
    .unwrap();
    let train_snap = train_source
        .refresh(&SamplerConfig::default(), None, None)
        .unwrap();
    assert!(!train_snap.records.is_empty(), "train must have data");

    // Read val via SrdSource — goes through actual file, not mock.
    let val_source =
        SrdSource::open(&dir.path().join("val/data.srd"), "val", 3, SrdMode::Pair).unwrap();
    let val_snap = val_source
        .refresh(&SamplerConfig::default(), None, None)
        .unwrap();
    assert!(
        !val_snap.records.is_empty(),
        "val must have data written by the scheduler"
    );

    // Cross-check: train must not contain val data.
    // (pos_* texts are legitimate — they're the positive texts from the sampler)
    let train_anchors: Vec<&str> = train_snap
        .records
        .iter()
        .map(|r| r.sections[0].text.as_str())
        .collect();
    let val_anchors: Vec<&str> = val_snap
        .records
        .iter()
        .map(|r| r.sections[0].text.as_str())
        .collect();
    // No anchor text should appear in both splits.
    for a in &train_anchors {
        assert!(
            !val_anchors.contains(a),
            "anchor '{a}' appears in both train and val"
        );
    }
}

#[test]
fn scheduler_flush_persists_to_disk() {
    let dir = TempDir::new().unwrap();
    let split_descs = vec![(SplitLabel::Train, "train", 0u64, 1.0f32)];

    let mut states =
        init_split_states_with_batch(dir.path(), &split_descs, 2, SrdMode::Pair, 3, 1).unwrap();

    // 3 batches of 2 → 6 entries, flush every 1 step (every batch)
    let provider = Arc::new(TestProvider::new(3, 2));
    let embedder = TestEmbedder;
    let config = SchedulerConfig::new(2, 2, 3, 1);

    let mut prefetchers = HashMap::new();
    prefetchers.insert(
        SplitLabel::Train,
        SamplerPrefetcher::new(provider.clone(), SplitLabel::Train, 4),
    );

    let stop = AtomicBool::new(false);
    run_interleaved_loop(
        &mut states,
        &mut prefetchers,
        &embedder,
        provider.as_ref(),
        &config,
        &stop,
        &mut NoopHandler,
    )
    .unwrap();

    // Data must be on disk — open a NEW SrdSource (not reusing state).
    let source = SrdSource::open(
        &dir.path().join("train/data.srd"),
        "verify",
        3,
        SrdMode::Pair,
    )
    .unwrap();
    let snap = source
        .refresh(&SamplerConfig::default(), None, None)
        .unwrap();
    assert_eq!(
        snap.records.len(),
        6,
        "scheduler must have written 6 entries to disk"
    );

    // Verify texts follow the pattern from TestProvider.
    // (Order may vary due to prefetcher threading, so just verify all entries
    // have valid anchor texts and total count is correct.)
    for rec in &snap.records {
        let text = &rec.sections[0].text;
        assert!(text.starts_with("anchor_"), "unexpected text: {text}");
        // Extract the slot index (second part after underscore).
        let parts: Vec<&str> = text.split('_').collect();
        assert_eq!(parts.len(), 3, "expected anchor_N_S format: {text}");
        let slot: usize = parts[2].parse().unwrap();
        assert!(slot < 2, "slot must be 0 or 1: {text}");
    }
}

#[test]
fn scheduler_resume_adds_to_existing_data() {
    let dir = TempDir::new().unwrap();
    let split_descs = vec![(SplitLabel::Train, "train", 0u64, 1.0f32)];

    // Session 1: write 2 entries.
    {
        let mut states =
            init_split_states_with_batch(dir.path(), &split_descs, 2, SrdMode::Pair, 3, 1).unwrap();
        let provider = Arc::new(TestProvider::new(1, 2));
        let embedder = TestEmbedder;
        let config = SchedulerConfig::new(2, 2, 3, 1);
        let mut prefetchers = HashMap::new();
        prefetchers.insert(
            SplitLabel::Train,
            SamplerPrefetcher::new(provider.clone(), SplitLabel::Train, 4),
        );
        let stop = AtomicBool::new(false);
        run_interleaved_loop(
            &mut states,
            &mut prefetchers,
            &embedder,
            provider.as_ref(),
            &config,
            &stop,
            &mut NoopHandler,
        )
        .unwrap();
    }

    // Session 2: write 2 more entries.
    {
        let mut states =
            init_split_states_with_batch(dir.path(), &split_descs, 2, SrdMode::Pair, 3, 1).unwrap();
        assert_eq!(states[0].total_written, 2, "must resume from session 1");
        let provider = Arc::new(TestProvider::new(1, 2));
        let embedder = TestEmbedder;
        let config = SchedulerConfig::new(2, 2, 3, 1);
        let mut prefetchers = HashMap::new();
        prefetchers.insert(
            SplitLabel::Train,
            SamplerPrefetcher::new(provider.clone(), SplitLabel::Train, 4),
        );
        let stop = AtomicBool::new(false);
        run_interleaved_loop(
            &mut states,
            &mut prefetchers,
            &embedder,
            provider.as_ref(),
            &config,
            &stop,
            &mut NoopHandler,
        )
        .unwrap();
    }

    // Verify 4 entries on disk, in order.
    let source = SrdSource::open(
        &dir.path().join("train/data.srd"),
        "resume",
        3,
        SrdMode::Pair,
    )
    .unwrap();
    let snap = source
        .refresh(&SamplerConfig::default(), None, None)
        .unwrap();
    assert_eq!(snap.records.len(), 4);

    // Session 1 entries (batch remaining=1 → anchor_1_0, anchor_1_1).
    assert_eq!(snap.records[0].sections[0].text, "anchor_1_0");
    assert_eq!(snap.records[1].sections[0].text, "anchor_1_1");
    // Session 2 entries (batch remaining=1 again → anchor_1_0, anchor_1_1).
    assert_eq!(snap.records[2].sections[0].text, "anchor_1_0");
    assert_eq!(snap.records[3].sections[0].text, "anchor_1_1");
}

#[test]
fn scheduler_ctrl_c_flushes_pending_to_disk() {
    let dir = TempDir::new().unwrap();
    let split_descs = vec![(SplitLabel::Train, "train", 0u64, 1.0f32)];

    let mut states =
        init_split_states_with_batch(dir.path(), &split_descs, 2, SrdMode::Pair, 3, 100).unwrap();

    // 5 batches, but stop after 1 step — pending data must be flushed.
    let provider = Arc::new(TestProvider::new(5, 2));
    let embedder = TestEmbedder;
    let config = SchedulerConfig::new(2, 2, 3, 100); // large flush interval

    let mut prefetchers = HashMap::new();
    prefetchers.insert(
        SplitLabel::Train,
        SamplerPrefetcher::new(provider.clone(), SplitLabel::Train, 4),
    );

    let stop = AtomicBool::new(true); // immediate Ctrl+C
    run_interleaved_loop(
        &mut states,
        &mut prefetchers,
        &embedder,
        provider.as_ref(),
        &config,
        &stop,
        &mut NoopHandler,
    )
    .unwrap();

    // Must have flushed entries from the step before Ctrl+C was processed.
    // (The prefetcher may have already buffered a batch, so exact text depends
    // on timing. Just verify data landed on disk.)
    let source = SrdSource::open(
        &dir.path().join("train/data.srd"),
        "ctrl_c",
        3,
        SrdMode::Pair,
    )
    .unwrap();
    let snap = source
        .refresh(&SamplerConfig::default(), None, None)
        .unwrap();
    assert!(
        !snap.records.is_empty(),
        "Ctrl+C must flush pending data to disk"
    );
    // Verify all records have valid anchor texts from our provider.
    for rec in &snap.records {
        assert!(
            rec.sections[0].text.starts_with("anchor_"),
            "unexpected text: {}",
            rec.sections[0].text
        );
    }
}

// ---------------------------------------------------------------------------
// Store-level roundtrip tests (direct DataStore + SrdSource)
// ---------------------------------------------------------------------------

use simd_r_drive::storage_engine::DataStore;
use simd_r_drive::storage_engine::traits::DataStoreReader;
use triplets_core::data::SectionRole;
use triplets_offline_embedder::traits::EmbedStore;
use triplets_srd_source::srd_triplet;

#[test]
fn pair_write_and_read_via_srd_source() {
    let dir = TempDir::new().unwrap();
    let store_path = dir.path().join("data.srd");

    // --- scope 1: write data, then drop the store handle ---
    {
        let store = DataStore::open(&store_path).unwrap();
        let anchor_vecs = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let anchor_texts = ["hello", "world", "foo"];
        let pos_vecs = [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]];
        let pos_texts = ["hi", "earth", "bar"];
        let labels = vec![PairLabel::Positive; 3];
        let entries: Vec<srd_triplet::SrdPairWriteEntry> = (0..3)
            .map(|i| srd_triplet::SrdPairWriteEntry {
                anchor_vec: &anchor_vecs[i],
                anchor_text: anchor_texts[i],
                candidate_vec: &pos_vecs[i],
                candidate_text: pos_texts[i],
                label: &labels[i],
            })
            .collect();
        srd_triplet::write_pair_entries(&store, 0, &entries).unwrap();
        assert_eq!(store.len().unwrap(), 3);
    }

    // --- scope 2: open as SrdSource and read back ---
    let source = SrdSource::open(&store_path, "test_pair", 3, SrdMode::Pair).unwrap();
    assert_eq!(source.mode(), SrdMode::Pair);

    let snapshot = source
        .refresh(&SamplerConfig::default(), None, None)
        .unwrap();
    assert_eq!(snapshot.records.len(), 3);

    // Each pair record has 2 sections: Anchor + Context (positive).
    for (i, rec) in snapshot.records.iter().enumerate() {
        assert_eq!(rec.sections.len(), 2, "record {i} should have 2 sections");
        assert_eq!(rec.sections[0].role, SectionRole::Anchor);
        assert_eq!(rec.sections[1].role, SectionRole::Context);
    }

    assert_eq!(snapshot.records[0].sections[0].text, "hello");
    assert_eq!(snapshot.records[0].sections[1].text, "hi");
    assert_eq!(snapshot.records[1].sections[0].text, "world");
    assert_eq!(snapshot.records[1].sections[1].text, "earth");
    assert_eq!(snapshot.records[2].sections[0].text, "foo");
    assert_eq!(snapshot.records[2].sections[1].text, "bar");
}

#[test]
fn pair_negative_label_roundtrip_via_srd_source() {
    let dir = TempDir::new().unwrap();
    let store_path = dir.path().join("data.srd");

    // --- scope 1: write pairs with Negative labels ---
    {
        let store = DataStore::open(&store_path).unwrap();
        let anchor_vecs = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let anchor_texts = ["a1", "a2"];
        let candidate_vecs = [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
        let candidate_texts = ["neg1", "neg2"];
        let labels = [PairLabel::Negative, PairLabel::Negative];
        let entries: Vec<srd_triplet::SrdPairWriteEntry> = (0..2)
            .map(|i| srd_triplet::SrdPairWriteEntry {
                anchor_vec: &anchor_vecs[i],
                anchor_text: anchor_texts[i],
                candidate_vec: &candidate_vecs[i],
                candidate_text: candidate_texts[i],
                label: &labels[i],
            })
            .collect();
        srd_triplet::write_pair_entries(&store, 0, &entries).unwrap();
        assert_eq!(store.len().unwrap(), 2);
    }

    // --- scope 2: open as SrdSource and read back ---
    let source = SrdSource::open(&store_path, "test_neg", 3, SrdMode::Pair).unwrap();
    let snapshot = source
        .refresh(&SamplerConfig::default(), None, None)
        .unwrap();
    assert_eq!(snapshot.records.len(), 2);

    // Verify texts are correct
    assert_eq!(snapshot.records[0].sections[0].text, "a1");
    assert_eq!(snapshot.records[0].sections[1].text, "neg1");
    assert_eq!(snapshot.records[1].sections[0].text, "a2");
    assert_eq!(snapshot.records[1].sections[1].text, "neg2");

    // Verify labels are preserved as Negative
    assert_eq!(
        snapshot.records[0].label,
        Some(PairLabel::Negative),
        "first record should have Negative label"
    );
    assert_eq!(
        snapshot.records[1].label,
        Some(PairLabel::Negative),
        "second record should have Negative label"
    );
}

// ---------------------------------------------------------------------------
// Triplet mode
// ---------------------------------------------------------------------------

#[test]
fn triplet_write_and_read_via_srd_source() {
    let dir = TempDir::new().unwrap();
    let store_path = dir.path().join("data.srd");

    {
        let store = DataStore::open(&store_path).unwrap();
        let anchor_vecs = [[1.0; 4], [2.0; 4]];
        let anchor_texts = ["anchor_a", "anchor_b"];
        let pos_vecs = [[3.0; 4], [4.0; 4]];
        let pos_texts = ["positive_a", "positive_b"];
        let neg_vecs = [[5.0; 4], [6.0; 4]];
        let neg_texts = ["negative_a", "negative_b"];
        let entries: Vec<srd_triplet::SrdTripletWriteEntry> = (0..2)
            .map(|i| srd_triplet::SrdTripletWriteEntry {
                anchor_vec: &anchor_vecs[i],
                anchor_text: anchor_texts[i],
                pos_vec: &pos_vecs[i],
                pos_text: pos_texts[i],
                neg_vec: &neg_vecs[i],
                neg_text: neg_texts[i],
            })
            .collect();
        srd_triplet::write_triplet_entries(&store, 0, &entries).unwrap();
        assert_eq!(store.len().unwrap(), 2);
    }

    let source = SrdSource::open(&store_path, "test_triplet", 4, SrdMode::Triplet).unwrap();
    assert_eq!(source.mode(), SrdMode::Triplet);

    let snapshot = source
        .refresh(&SamplerConfig::default(), None, None)
        .unwrap();
    assert_eq!(snapshot.records.len(), 2);

    // Each triplet record has 3 sections: Anchor + Context (positive) + Context (negative).
    for (i, rec) in snapshot.records.iter().enumerate() {
        assert_eq!(rec.sections.len(), 3, "record {i} should have 3 sections");
        assert_eq!(rec.sections[0].role, SectionRole::Anchor);
        assert_eq!(
            rec.sections[0].text,
            format!("anchor_{}", if i == 0 { 'a' } else { 'b' })
        );
        assert_eq!(rec.sections[1].role, SectionRole::Context);
        assert_eq!(
            rec.sections[1].text,
            format!("positive_{}", if i == 0 { 'a' } else { 'b' })
        );
        assert_eq!(rec.sections[2].role, SectionRole::Context);
        assert_eq!(
            rec.sections[2].text,
            format!("negative_{}", if i == 0 { 'a' } else { 'b' })
        );
    }
}

// ---------------------------------------------------------------------------
// Resume: write some, re-open, verify count
// ---------------------------------------------------------------------------

#[test]
fn resume_after_partial_write() {
    let dir = TempDir::new().unwrap();
    let store_path = dir.path().join("data.srd");

    // Write 2 entries.
    {
        let store = DataStore::open(&store_path).unwrap();
        let vecs = [[1.0; 3]; 2];
        let texts = ["a", "b"];
        let labels = vec![PairLabel::Positive; 2];
        let entries: Vec<srd_triplet::SrdPairWriteEntry> = (0..2)
            .map(|i| srd_triplet::SrdPairWriteEntry {
                anchor_vec: &vecs[i],
                anchor_text: texts[i],
                candidate_vec: &vecs[i],
                candidate_text: texts[i],
                label: &labels[i],
            })
            .collect();
        srd_triplet::write_pair_entries(&store, 0, &entries).unwrap();
    }

    // Re-open and verify we can read the 2 existing entries.
    let source = SrdSource::open(&store_path, "resume_test", 3, SrdMode::Pair).unwrap();
    let snapshot = source
        .refresh(&SamplerConfig::default(), None, None)
        .unwrap();
    assert_eq!(snapshot.records.len(), 2);
    assert_eq!(snapshot.records[0].sections[0].text, "a");
    assert_eq!(snapshot.records[1].sections[0].text, "b");
}

// ---------------------------------------------------------------------------
// SrdStoreAdapter roundtrip via init_split_states_with_batch
// ---------------------------------------------------------------------------

#[test]
fn adapter_roundtrip_through_split_state() {
    use triplets_core::SplitLabel;
    use triplets_offline_embedder::store_adapter::init_split_states_with_batch;

    let dir = TempDir::new().unwrap();
    let split_descs = vec![(SplitLabel::Train, "train", 0u64, 1.0f32)];

    // First call: creates the store, writes via the adapter.
    {
        let states =
            init_split_states_with_batch(dir.path(), &split_descs, 8, SrdMode::Pair, 3, 400)
                .unwrap();
        let vecs = vec![vec![1.0, 2.0, 3.0]; 4];
        let texts: Vec<&str> = vec!["alpha", "beta", "gamma", "delta"];
        let labels = vec![PairLabel::Positive; 4];
        let entries: Vec<PairWriteEntry> = (0..4)
            .map(|i| PairWriteEntry {
                anchor_vec: &vecs[i],
                anchor_text: texts[i],
                candidate_vec: &vecs[i],
                candidate_text: texts[i],
                label: &labels[i],
            })
            .collect();
        states[0]
            .store
            .write_pairs(0, &PairWriteArgs { entries: &entries })
            .unwrap();
        assert_eq!(states[0].store.len().unwrap(), 4);
    }

    // Second call: re-opens the same store file.
    let states =
        init_split_states_with_batch(dir.path(), &split_descs, 8, SrdMode::Pair, 3, 400).unwrap();
    assert_eq!(states[0].total_written, 4);
    assert_eq!(states[0].store.len().unwrap(), 4);

    // Verify data is readable through SrdSource.
    let source = SrdSource::open(
        &dir.path().join("train/data.srd"),
        "adapter_test",
        3,
        SrdMode::Pair,
    )
    .unwrap();
    let snapshot = source
        .refresh(&SamplerConfig::default(), None, None)
        .unwrap();
    assert_eq!(snapshot.records.len(), 4);
    assert_eq!(snapshot.records[0].sections[0].text, "alpha");
    assert_eq!(snapshot.records[3].sections[0].text, "delta");
}

// ---------------------------------------------------------------------------
// Multi-split tests through the REAL scheduler code path
// ---------------------------------------------------------------------------

#[test]
fn multi_split_train_val_separate_files() {
    let dir = TempDir::new().unwrap();
    let split_descs = vec![
        (SplitLabel::Train, "train", 100u64, 0.8f32),
        (SplitLabel::Validation, "val", 50u64, 0.2f32),
    ];

    let mut states =
        init_split_states_with_batch(dir.path(), &split_descs, 4, SrdMode::Pair, 3, 2).unwrap();

    // 2 batches of 2 → 4 entries per split, flush every 2 steps
    let provider = Arc::new(SplitAwareProvider::new(2, 2));
    let embedder = TestEmbedder;
    let config = SchedulerConfig::new(2, 2, 3, 2);

    let mut prefetchers = HashMap::new();
    prefetchers.insert(
        SplitLabel::Train,
        SamplerPrefetcher::new(provider.clone(), SplitLabel::Train, 4),
    );
    prefetchers.insert(
        SplitLabel::Validation,
        SamplerPrefetcher::new(provider.clone(), SplitLabel::Validation, 4),
    );

    let stop = AtomicBool::new(false);
    run_interleaved_loop(
        &mut states,
        &mut prefetchers,
        &embedder,
        provider.as_ref(),
        &config,
        &stop,
        &mut NoopHandler,
    )
    .unwrap();

    // Files must exist.
    assert!(dir.path().join("train/data.srd").exists());
    assert!(dir.path().join("val/data.srd").exists());

    // Read train via SrdSource.
    let train_source = SrdSource::open(
        &dir.path().join("train/data.srd"),
        "train",
        3,
        SrdMode::Pair,
    )
    .unwrap();
    let train_snap = train_source
        .refresh(&SamplerConfig::default(), None, None)
        .unwrap();
    assert!(!train_snap.records.is_empty(), "train must have data");

    // Read val via SrdSource.
    let val_source =
        SrdSource::open(&dir.path().join("val/data.srd"), "val", 3, SrdMode::Pair).unwrap();
    let val_snap = val_source
        .refresh(&SamplerConfig::default(), None, None)
        .unwrap();
    assert!(!val_snap.records.is_empty(), "val must have data");

    // Cross-check: every anchor in train must start with "train_", every
    // anchor in val must start with "val_".
    for rec in &train_snap.records {
        assert!(
            rec.sections[0].text.starts_with("train_"),
            "train contains wrong split data: {}",
            rec.sections[0].text
        );
    }
    for rec in &val_snap.records {
        assert!(
            rec.sections[0].text.starts_with("val_"),
            "val contains wrong split data: {}",
            rec.sections[0].text
        );
    }
}

#[test]
fn multi_split_interleaved_flush_preserves_separation() {
    let dir = TempDir::new().unwrap();
    let split_descs = vec![
        (SplitLabel::Train, "train", 100u64, 0.8f32),
        (SplitLabel::Validation, "val", 50u64, 0.2f32),
    ];

    let mut states =
        init_split_states_with_batch(dir.path(), &split_descs, 4, SrdMode::Pair, 3, 1).unwrap();

    // 4 batches of 2, flush every step → interleaved writes to train/val
    let provider = Arc::new(SplitAwareProvider::new(4, 2));
    let embedder = TestEmbedder;
    let config = SchedulerConfig::new(2, 2, 3, 1);

    let mut prefetchers = HashMap::new();
    prefetchers.insert(
        SplitLabel::Train,
        SamplerPrefetcher::new(provider.clone(), SplitLabel::Train, 4),
    );
    prefetchers.insert(
        SplitLabel::Validation,
        SamplerPrefetcher::new(provider.clone(), SplitLabel::Validation, 4),
    );

    let stop = AtomicBool::new(false);
    run_interleaved_loop(
        &mut states,
        &mut prefetchers,
        &embedder,
        provider.as_ref(),
        &config,
        &stop,
        &mut NoopHandler,
    )
    .unwrap();

    // Verify all train entries have train_ prefix.
    let train_source = SrdSource::open(
        &dir.path().join("train/data.srd"),
        "train_intl",
        3,
        SrdMode::Pair,
    )
    .unwrap();
    let train_snap = train_source
        .refresh(&SamplerConfig::default(), None, None)
        .unwrap();
    for rec in &train_snap.records {
        assert!(
            rec.sections[0].text.starts_with("train_"),
            "interleaved: train contains wrong data: {}",
            rec.sections[0].text
        );
    }

    // Verify all val entries have val_ prefix.
    let val_source = SrdSource::open(
        &dir.path().join("val/data.srd"),
        "val_intl",
        3,
        SrdMode::Pair,
    )
    .unwrap();
    let val_snap = val_source
        .refresh(&SamplerConfig::default(), None, None)
        .unwrap();
    for rec in &val_snap.records {
        assert!(
            rec.sections[0].text.starts_with("val_"),
            "interleaved: val contains wrong data: {}",
            rec.sections[0].text
        );
    }
}

#[test]
fn multi_split_resume_preserves_separation() {
    let dir = TempDir::new().unwrap();
    let split_descs = vec![
        (SplitLabel::Train, "train", 100u64, 0.8f32),
        (SplitLabel::Validation, "val", 50u64, 0.2f32),
    ];

    // Session 1: 2 batches of 2
    {
        let mut states =
            init_split_states_with_batch(dir.path(), &split_descs, 4, SrdMode::Pair, 3, 1).unwrap();
        let provider = Arc::new(SplitAwareProvider::new(2, 2));
        let embedder = TestEmbedder;
        let config = SchedulerConfig::new(2, 2, 3, 1);
        let mut prefetchers = HashMap::new();
        prefetchers.insert(
            SplitLabel::Train,
            SamplerPrefetcher::new(provider.clone(), SplitLabel::Train, 4),
        );
        prefetchers.insert(
            SplitLabel::Validation,
            SamplerPrefetcher::new(provider.clone(), SplitLabel::Validation, 4),
        );
        let stop = AtomicBool::new(false);
        run_interleaved_loop(
            &mut states,
            &mut prefetchers,
            &embedder,
            provider.as_ref(),
            &config,
            &stop,
            &mut NoopHandler,
        )
        .unwrap();
    }

    // Session 2: 2 more batches of 2
    {
        let mut states =
            init_split_states_with_batch(dir.path(), &split_descs, 4, SrdMode::Pair, 3, 1).unwrap();
        assert_eq!(
            states[0].total_written, 4,
            "train must resume from session 1"
        );
        assert_eq!(states[1].total_written, 4, "val must resume from session 1");
        let provider = Arc::new(SplitAwareProvider::new(2, 2));
        let embedder = TestEmbedder;
        let config = SchedulerConfig::new(2, 2, 3, 1);
        let mut prefetchers = HashMap::new();
        prefetchers.insert(
            SplitLabel::Train,
            SamplerPrefetcher::new(provider.clone(), SplitLabel::Train, 4),
        );
        prefetchers.insert(
            SplitLabel::Validation,
            SamplerPrefetcher::new(provider.clone(), SplitLabel::Validation, 4),
        );
        let stop = AtomicBool::new(false);
        run_interleaved_loop(
            &mut states,
            &mut prefetchers,
            &embedder,
            provider.as_ref(),
            &config,
            &stop,
            &mut NoopHandler,
        )
        .unwrap();
    }

    // Verify 8 entries per split on disk, all with correct prefix.
    let train_source = SrdSource::open(
        &dir.path().join("train/data.srd"),
        "train_r",
        3,
        SrdMode::Pair,
    )
    .unwrap();
    let train_snap = train_source
        .refresh(&SamplerConfig::default(), None, None)
        .unwrap();
    assert_eq!(train_snap.records.len(), 8);
    for rec in &train_snap.records {
        assert!(rec.sections[0].text.starts_with("train_"));
    }

    let val_source =
        SrdSource::open(&dir.path().join("val/data.srd"), "val_r", 3, SrdMode::Pair).unwrap();
    let val_snap = val_source
        .refresh(&SamplerConfig::default(), None, None)
        .unwrap();
    assert_eq!(val_snap.records.len(), 8);
    for rec in &val_snap.records {
        assert!(rec.sections[0].text.starts_with("val_"));
    }
}

#[test]
fn multi_split_vectors_not_swapped() {
    let dir = TempDir::new().unwrap();
    let split_descs = vec![
        (SplitLabel::Train, "train", 100u64, 0.8f32),
        (SplitLabel::Validation, "val", 50u64, 0.2f32),
    ];

    let mut states =
        init_split_states_with_batch(dir.path(), &split_descs, 2, SrdMode::Pair, 3, 1).unwrap();

    // 3 batches of 2, flush every step
    let provider = Arc::new(SplitAwareProvider::new(3, 2));
    let embedder = TestEmbedder;
    let config = SchedulerConfig::new(2, 2, 3, 1);

    let mut prefetchers = HashMap::new();
    prefetchers.insert(
        SplitLabel::Train,
        SamplerPrefetcher::new(provider.clone(), SplitLabel::Train, 4),
    );
    prefetchers.insert(
        SplitLabel::Validation,
        SamplerPrefetcher::new(provider.clone(), SplitLabel::Validation, 4),
    );

    let stop = AtomicBool::new(false);
    run_interleaved_loop(
        &mut states,
        &mut prefetchers,
        &embedder,
        provider.as_ref(),
        &config,
        &stop,
        &mut NoopHandler,
    )
    .unwrap();

    // Verify no anchor text appears in both splits.
    let train_source = SrdSource::open(
        &dir.path().join("train/data.srd"),
        "train_v",
        3,
        SrdMode::Pair,
    )
    .unwrap();
    let train_snap = train_source
        .refresh(&SamplerConfig::default(), None, None)
        .unwrap();

    let val_source =
        SrdSource::open(&dir.path().join("val/data.srd"), "val_v", 3, SrdMode::Pair).unwrap();
    let val_snap = val_source
        .refresh(&SamplerConfig::default(), None, None)
        .unwrap();

    let train_anchors: Vec<&str> = train_snap
        .records
        .iter()
        .map(|r| r.sections[0].text.as_str())
        .collect();
    let val_anchors: Vec<&str> = val_snap
        .records
        .iter()
        .map(|r| r.sections[0].text.as_str())
        .collect();
    for a in &train_anchors {
        assert!(
            !val_anchors.contains(a),
            "anchor '{a}' appears in both train and val"
        );
    }
    for a in &val_anchors {
        assert!(
            !train_anchors.contains(a),
            "anchor '{a}' appears in both val and train"
        );
    }

    // Verify prefixes.
    for rec in &train_snap.records {
        assert!(rec.sections[0].text.starts_with("train_"));
    }
    for rec in &val_snap.records {
        assert!(rec.sections[0].text.starts_with("val_"));
    }
}
