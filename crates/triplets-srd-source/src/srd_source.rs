use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use chrono::Utc;
use simd_r_drive::storage_engine::DataStore;
use simd_r_drive::storage_engine::traits::DataStoreReader;
use triplets::SamplerError;
use triplets::config::{NegativeStrategy, SamplerConfig, Selector, TripletRecipe};
use triplets::data::{DataRecord, QualityScore, RecordSection, SectionRole};
use triplets::source::{DataSource, SourceCursor, SourceSnapshot};
use triplets::tokenizer::{Tokenizer, WhitespaceTokenizer};
use triplets::types::RecordId;

use crate::error::SrdError;
use crate::srd_triplet::{self, SrdMode, SrdRecord};

/// A [`DataSource`] backed by a simd-r-drive embedding store.
///
/// Each entry in the store maps to a [`DataRecord`] with 2 sections (pair mode)
/// or 3 sections (triplet mode), determined by the entry's mode byte.
///
/// The `record_id` on each produced [`DataRecord`] is the entry's u64 index
/// (as a string). Entry indices are unique within one store; identity across
/// stores is the sampler's job (composite keys), not the id string's — so
/// nothing parses or decorates these ids. Entry vectors ride the
/// [`RecordSection`] embedding channel.
///
/// ## Sampling model: shuffled streaming windows, not file order
///
/// `refresh` draws **uniform-over-corpus without replacement** through a
/// deterministic permutation of the index domain `0..entry_count`, mirroring
/// [`triplets_core::source::IndexablePager`] / `ParquetRowSource`:
///
/// - The permutation seed is
///   `IndexablePager::seed_for_sampler(source_id, total, config.seed)`, where
///   `config.seed` is the epoch/step-mixed seed the ingestion manager already
///   threads through (`derive_epoch_seed(base, epoch) ^ step`). A fixed
///   `(source_id, total, seed, cursor)` therefore always yields the same
///   indices; different seeds/epochs/steps draw from across the corpus rather
///   than adjacent file offsets.
/// - The cursor `revision` is a position in *permutation-counter space*, not a
///   file offset. Consecutive refreshes with the same seed advance the counter
///   and return disjoint windows with no repeats until the counter completes
///   a full domain cycle (counter space is infinite — `next()` folds it modulo
///   the power-of-two domain — so the stream never stalls and cursors never
///   need rewinding). When the seed changes between refreshes (epoch/step
///   boundary) each window is an independent random subset, so a small chance
///   overlap of ~`window^2/total` rows is expected — negligible for large
///   stores, and never a coherent file-ordered block.
/// - Within one refresh the returned order is the permutation order (random),
///   not sorted file order.
///
/// ## Why the permutation is driven manually instead of via `IndexablePager`
///
/// `IndexablePager::refresh_with` materializes one `(index, cursor)` pair per
/// corpus row and fetches through per-index `record_at` calls. For the 10M+-
/// row SRD stores behind issue #192 that is hundreds of MB per refresh plus
/// one store read per row. Driving `IndexPermutation` directly lets `refresh`
/// draw only `max` indices and serve them with a single `batch_read_entries`
/// call. It also persists
/// [`triplets_core::source::IndexPermutation::raw_counter`] rather than the
/// folded [`triplets_core::source::IndexPermutation::cursor`], which is what
/// preserves without-replacement tiling when the row count is not a power of
/// two.
pub struct SrdSource {
    store: DataStore,
    source_id: String,
    entry_count: AtomicU64,
    mode: SrdMode,
    emb_dim: usize,
}

impl SrdSource {
    /// Open a simd-r-drive store.
    ///
    /// - `mode: None` detects the mode from entry 0 (single open, no probe
    ///   handle). Errors on empty stores — nothing to detect from.
    /// - `mode: Some(m)` pins the mode explicitly, required for empty stores
    ///   (schema cannot be inferred from nothing). Non-empty stores are
    ///   verified: a mismatch with `m` returns `SrdError::ModeMismatch`
    ///   instead of opening, so sampling can never silently run with the
    ///   wrong recipes.
    pub fn open(
        path: &Path,
        source_id: impl Into<String>,
        emb_dim: usize,
        mode: Option<SrdMode>,
    ) -> Result<Self, SrdError> {
        let store = DataStore::open_existing(path)?;
        let count = store.len()?;
        let detected = if count > 0 {
            let entries = srd_triplet::batch_read_entries(&store, &[0], emb_dim)?;
            match entries.into_iter().next() {
                Some(SrdRecord::Pair(_)) => Some(SrdMode::Pair),
                Some(SrdRecord::Triplet(_)) => Some(SrdMode::Triplet),
                None => None,
            }
        } else {
            None
        };
        let mode = match (mode, detected) {
            (Some(m), Some(d)) if m != d => {
                return Err(SrdError::ModeMismatch {
                    expected: m,
                    found: d,
                });
            }
            (Some(m), _) => m,
            (None, Some(d)) => d,
            (None, None) => return Err(SrdError::EmptyStore),
        };
        Ok(Self {
            store,
            source_id: source_id.into(),
            entry_count: AtomicU64::new(count as u64),
            mode,
            emb_dim,
        })
    }

    /// The detected mode of this store.
    pub fn mode(&self) -> SrdMode {
        self.mode
    }

    /// Number of entries in this store.
    pub fn len(&self) -> usize {
        self.entry_count.load(Ordering::Relaxed) as usize
    }

    /// Whether this store holds no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl DataSource for SrdSource {
    fn id(&self) -> &str {
        &self.source_id
    }

    fn refresh(
        &self,
        config: &SamplerConfig,
        cursor: Option<&SourceCursor>,
        limit: Option<usize>,
    ) -> Result<SourceSnapshot, SamplerError> {
        use triplets_core::source::{IndexPermutation, IndexablePager};

        let total_u64 = self.entry_count.load(Ordering::Relaxed);
        if total_u64 == 0 {
            return Ok(SourceSnapshot {
                records: Vec::new(),
                cursor: SourceCursor {
                    last_seen: Utc::now(),
                    revision: 0,
                },
            });
        }
        let total: usize = usize::try_from(total_u64).map_err(|e| {
            SamplerError::Configuration(format!(
                "SRD store entry count {total_u64} exceeds index range: {e}"
            ))
        })?;
        // The cursor revision is a raw permutation-counter position, not a
        // file offset: counter space is infinite (`next()` folds it modulo
        // the power-of-two domain), so no wrap/clamp is needed for the
        // infinite forward stream — arbitrarily large revisions just continue
        // the periodic walk.
        let start = cursor.map(|c| c.revision).unwrap_or(0);
        let max = limit.unwrap_or(256).min(total);
        if max == 0 {
            return Ok(SourceSnapshot {
                records: Vec::new(),
                cursor: SourceCursor {
                    last_seen: Utc::now(),
                    revision: start,
                },
            });
        }

        // Deterministic random sampling: the ingestion manager already mixes
        // `derive_epoch_seed(base_seed, epoch) ^ step` into `config.seed`, so
        // seeding the permutation from it honors (seed, epoch, step) without
        // the source tracking epochs itself.
        let seed = IndexablePager::seed_for_sampler(&self.source_id, total, config.seed);
        let mut permutation = IndexPermutation::new(total, seed, start);
        let indices: Vec<usize> = (0..max).map(|_| permutation.next()).collect();
        let entries = srd_triplet::batch_read_entries(&self.store, &indices, self.emb_dim)
            .map_err(|e| SamplerError::Configuration(e.to_string()))?;

        let now = Utc::now();
        let mut records = Vec::with_capacity(entries.len());
        for (offset, record) in indices.iter().zip(entries.iter()) {
            let id: RecordId = (*offset as u64).to_string();
            let (sections, label) = match record {
                SrdRecord::Pair(pair) => (
                    vec![
                        RecordSection {
                            role: SectionRole::Anchor,
                            heading: None,
                            text: pair.anchor_text.clone(),
                            sentences: vec![],
                            token_count: WhitespaceTokenizer.token_count(&pair.anchor_text),
                            embedding: Some(Arc::from(pair.anchor_emb.as_slice())),
                        },
                        RecordSection {
                            role: SectionRole::Context,
                            heading: None,
                            text: pair.candidate_text.clone(),
                            sentences: vec![],
                            token_count: WhitespaceTokenizer.token_count(&pair.candidate_text),
                            embedding: Some(Arc::from(pair.candidate_emb.as_slice())),
                        },
                    ],
                    Some(pair.label.clone()),
                ),
                SrdRecord::Triplet(triplet) => (
                    vec![
                        RecordSection {
                            role: SectionRole::Anchor,
                            heading: None,
                            text: triplet.anchor_text.clone(),
                            sentences: vec![],
                            token_count: WhitespaceTokenizer.token_count(&triplet.anchor_text),
                            embedding: Some(Arc::from(triplet.anchor_emb.as_slice())),
                        },
                        RecordSection {
                            role: SectionRole::Context,
                            heading: None,
                            text: triplet.pos_text.clone(),
                            sentences: vec![],
                            token_count: WhitespaceTokenizer.token_count(&triplet.pos_text),
                            embedding: Some(Arc::from(triplet.pos_emb.as_slice())),
                        },
                        RecordSection {
                            role: SectionRole::Context,
                            heading: None,
                            text: triplet.neg_text.clone(),
                            sentences: vec![],
                            token_count: WhitespaceTokenizer.token_count(&triplet.neg_text),
                            embedding: Some(Arc::from(triplet.neg_emb.as_slice())),
                        },
                    ],
                    None,
                ),
            };
            records.push(DataRecord {
                id,
                source: self.source_id.clone(),
                created_at: now,
                updated_at: now,
                quality: QualityScore::default(),
                taxonomy: vec![],
                sections,
                meta_prefix: None,
                label,
            });
        }

        let next_cursor = SourceCursor {
            last_seen: Utc::now(),
            // Persist the raw permutation counter (not `cursor() % total`):
            // only the raw counter preserves exact stream position across
            // refreshes, giving disjoint without-replacement windows until a
            // full domain cycle completes.
            revision: permutation.raw_counter(),
        };

        Ok(SourceSnapshot {
            records,
            cursor: next_cursor,
        })
    }

    fn reported_record_count(&self, _config: &SamplerConfig) -> Result<u128, SamplerError> {
        Ok(self.entry_count.load(Ordering::Relaxed) as u128)
    }

    fn default_triplet_recipes(&self) -> Vec<TripletRecipe> {
        match self.mode {
            SrdMode::Pair => vec![TripletRecipe {
                name: "srd_pair".into(),
                anchor: Selector::Paragraph(0),
                positive_selector: Selector::Paragraph(1),
                negative_selector: Selector::Role(SectionRole::Context),
                negative_strategy: NegativeStrategy::WrongArticle,
                weight: 1.0,
                instruction: None,
                allow_same_anchor_positive: false,
                retrieval_sampling: None,
            }],
            SrdMode::Triplet => vec![TripletRecipe {
                name: "srd_triplet".into(),
                anchor: Selector::Paragraph(0),
                positive_selector: Selector::Paragraph(1),
                negative_selector: Selector::Paragraph(2),
                negative_strategy: NegativeStrategy::SameRecord,
                weight: 1.0,
                instruction: None,
                allow_same_anchor_positive: false,
                retrieval_sampling: None,
            }],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use triplets::data::PairLabel;

    use crate::srd_triplet::{SrdPairWriteEntry, SrdTripletWriteEntry};

    /// Helper to create an SrdPairWriteEntry slice for tests.
    fn make_pair_entries<'a>(
        a_vecs: &'a [Vec<f32>],
        a_texts: &'a [&'a str],
        p_vecs: &'a [Vec<f32>],
        p_texts: &'a [&'a str],
    ) -> Vec<SrdPairWriteEntry<'a>> {
        a_vecs
            .iter()
            .zip(a_texts.iter())
            .zip(p_vecs.iter().zip(p_texts.iter()))
            .map(|((av, at), (pv, pt))| SrdPairWriteEntry {
                anchor_vec: av,
                anchor_text: at,
                candidate_vec: pv,
                candidate_text: pt,
                label: &PairLabel::Positive,
            })
            .collect()
    }

    /// Helper to create an SrdTripletWriteEntry slice for tests.
    fn make_triplet_entries<'a>(
        a_vecs: &'a [Vec<f32>],
        a_texts: &'a [&'a str],
        p_vecs: &'a [Vec<f32>],
        p_texts: &'a [&'a str],
        n_vecs: &'a [Vec<f32>],
        n_texts: &'a [&'a str],
    ) -> Vec<SrdTripletWriteEntry<'a>> {
        a_vecs
            .iter()
            .zip(a_texts.iter())
            .zip(p_vecs.iter().zip(p_texts.iter()))
            .zip(n_vecs.iter().zip(n_texts.iter()))
            .map(|(((av, at), (pv, pt)), (nv, nt))| SrdTripletWriteEntry {
                anchor_vec: av,
                anchor_text: at,
                pos_vec: pv,
                pos_text: pt,
                neg_vec: nv,
                neg_text: nt,
            })
            .collect()
    }
    use tempfile::TempDir;

    const TEST_EMB_DIM: usize = 768;

    fn make_pair_store(dir: &TempDir, n: usize) {
        let store = DataStore::open(&dir.path().join("data.srd")).unwrap();
        let vecs: Vec<Vec<f32>> = (0..n).map(|i| vec![i as f32; TEST_EMB_DIM]).collect();
        let texts: Vec<String> = (0..n).map(|i| format!("anchor_{i}")).collect();
        let pos_texts: Vec<String> = (0..n).map(|i| format!("positive_{i}")).collect();
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let pos_refs: Vec<&str> = pos_texts.iter().map(|s| s.as_str()).collect();
        let pair_entries = make_pair_entries(&vecs, &refs, &vecs, &pos_refs);
        srd_triplet::write_pair_entries(&store, 0, pair_entries.as_slice()).unwrap();
    }

    fn make_triplet_store(dir: &TempDir, n: usize) {
        let store = DataStore::open(&dir.path().join("data.srd")).unwrap();
        let vecs: Vec<Vec<f32>> = (0..n).map(|i| vec![i as f32; TEST_EMB_DIM]).collect();
        let a_texts: Vec<String> = (0..n).map(|i| format!("anchor_{i}")).collect();
        let p_texts: Vec<String> = (0..n).map(|i| format!("positive_{i}")).collect();
        let n_texts: Vec<String> = (0..n).map(|i| format!("negative_{i}")).collect();
        let a_refs: Vec<&str> = a_texts.iter().map(|s| s.as_str()).collect();
        let p_refs: Vec<&str> = p_texts.iter().map(|s| s.as_str()).collect();
        let n_refs: Vec<&str> = n_texts.iter().map(|s| s.as_str()).collect();
        let trip_entries = make_triplet_entries(&vecs, &a_refs, &vecs, &p_refs, &vecs, &n_refs);
        srd_triplet::write_triplet_entries(&store, 0, trip_entries.as_slice()).unwrap();
    }

    #[test]
    fn pair_mode_refresh_returns_two_sections() {
        let dir = TempDir::new().unwrap();
        make_pair_store(&dir, 3);
        let source = SrdSource::open(
            &dir.path().join("data.srd"),
            "test",
            TEST_EMB_DIM,
            Some(SrdMode::Pair),
        )
        .unwrap();
        assert_eq!(source.mode(), SrdMode::Pair);

        let config = SamplerConfig::default();
        let snapshot = source.refresh(&config, None, None).unwrap();
        assert_eq!(snapshot.records.len(), 3);

        for record in &snapshot.records {
            assert_eq!(record.sections.len(), 2);
            assert_eq!(record.sections[0].role, SectionRole::Anchor);
            assert_eq!(record.sections[1].role, SectionRole::Context);
        }
    }

    #[test]
    fn triplet_mode_refresh_returns_three_sections() {
        let dir = TempDir::new().unwrap();
        make_triplet_store(&dir, 3);
        let source = SrdSource::open(
            &dir.path().join("data.srd"),
            "test",
            TEST_EMB_DIM,
            Some(SrdMode::Triplet),
        )
        .unwrap();
        assert_eq!(source.mode(), SrdMode::Triplet);

        let config = SamplerConfig::default();
        let snapshot = source.refresh(&config, None, None).unwrap();
        assert_eq!(snapshot.records.len(), 3);

        for record in &snapshot.records {
            assert_eq!(record.sections.len(), 3);
            assert_eq!(record.sections[0].role, SectionRole::Anchor);
            assert_eq!(record.sections[1].role, SectionRole::Context);
            assert_eq!(record.sections[2].role, SectionRole::Context);
        }
    }

    #[test]
    fn record_id_matches_entry_index() {
        let dir = TempDir::new().unwrap();
        make_pair_store(&dir, 5);
        let source = SrdSource::open(
            &dir.path().join("data.srd"),
            "test",
            TEST_EMB_DIM,
            Some(SrdMode::Pair),
        )
        .unwrap();

        let config = SamplerConfig::default();
        let snapshot = source.refresh(&config, None, None).unwrap();

        // Sampling is permuted (see issue #192): ids cover the index domain
        // exactly once per cycle, in random order rather than file order.
        let mut ids: Vec<String> = snapshot.records.iter().map(|r| r.id.clone()).collect();
        ids.sort_unstable();
        let expected: Vec<String> = (0..5).map(|i| i.to_string()).collect();
        assert_eq!(ids, expected);
    }

    #[test]
    fn default_recipes_pair_mode() {
        let dir = TempDir::new().unwrap();
        make_pair_store(&dir, 1);
        let source = SrdSource::open(
            &dir.path().join("data.srd"),
            "test",
            TEST_EMB_DIM,
            Some(SrdMode::Pair),
        )
        .unwrap();

        let recipes = source.default_triplet_recipes();
        assert_eq!(recipes.len(), 1);
        assert_eq!(recipes[0].name, "srd_pair");
        assert_eq!(recipes[0].anchor, Selector::Paragraph(0));
        assert_eq!(recipes[0].positive_selector, Selector::Paragraph(1));
        assert!(matches!(
            recipes[0].negative_strategy,
            NegativeStrategy::WrongArticle
        ));
    }

    #[test]
    fn default_recipes_triplet_mode() {
        let dir = TempDir::new().unwrap();
        make_triplet_store(&dir, 1);
        let source = SrdSource::open(
            &dir.path().join("data.srd"),
            "test",
            TEST_EMB_DIM,
            Some(SrdMode::Triplet),
        )
        .unwrap();

        let recipes = source.default_triplet_recipes();
        assert_eq!(recipes.len(), 1);
        assert_eq!(recipes[0].name, "srd_triplet");
        assert_eq!(recipes[0].anchor, Selector::Paragraph(0));
        assert_eq!(recipes[0].positive_selector, Selector::Paragraph(1));
        assert_eq!(recipes[0].negative_selector, Selector::Paragraph(2));
        assert!(matches!(
            recipes[0].negative_strategy,
            NegativeStrategy::SameRecord
        ));
    }

    #[test]
    fn pagination_with_limit() {
        let dir = TempDir::new().unwrap();
        // n=4 is a power of two, so the permutation domain has no rejected
        // positions and two 2-row pages tile the corpus exactly.
        make_pair_store(&dir, 4);
        let source = SrdSource::open(
            &dir.path().join("data.srd"),
            "test",
            TEST_EMB_DIM,
            Some(SrdMode::Pair),
        )
        .unwrap();

        let config = SamplerConfig::default();

        // Windows stream forward through permutation-counter space: with a
        // fixed seed, consecutive pages are disjoint and tile the corpus
        // (order within/between pages is permuted, not file order).
        let snapshot1 = source.refresh(&config, None, Some(2)).unwrap();
        assert_eq!(snapshot1.records.len(), 2);

        let snapshot2 = source
            .refresh(&config, Some(&snapshot1.cursor), Some(2))
            .unwrap();
        assert_eq!(snapshot2.records.len(), 2);

        let mut all: Vec<String> = snapshot1
            .records
            .iter()
            .chain(snapshot2.records.iter())
            .map(|r| r.id.clone())
            .collect();
        all.sort_unstable();
        let expected: Vec<String> = (0..4).map(|i| i.to_string()).collect();
        assert_eq!(all, expected, "pages must tile the corpus without repeats");

        // Full cycle consumed: the next page wraps and restarts.
        let snapshot3 = source
            .refresh(&config, Some(&snapshot2.cursor), Some(2))
            .unwrap();
        let first_ids: Vec<&str> = snapshot1.records.iter().map(|r| r.id.as_str()).collect();
        let third_ids: Vec<&str> = snapshot3.records.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(third_ids, first_ids, "post-cycle page must restart");

        // Deterministic: replaying from the head with the same seed reproduces
        // the same first page.
        let replay = source.refresh(&config, None, Some(2)).unwrap();
        let replay_ids: Vec<&str> = replay.records.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(first_ids, replay_ids);
    }

    #[test]
    fn empty_store_returns_empty_records() {
        let dir = TempDir::new().unwrap();
        let store = DataStore::open(&dir.path().join("data.srd")).unwrap();
        // Don't write anything — store is empty
        drop(store);

        let source = SrdSource::open(
            &dir.path().join("data.srd"),
            "test",
            TEST_EMB_DIM,
            Some(SrdMode::Pair),
        )
        .unwrap();
        assert_eq!(source.mode(), SrdMode::Pair);

        let config = SamplerConfig::default();
        let snapshot = source.refresh(&config, None, None).unwrap();
        assert!(snapshot.records.is_empty());
    }

    #[test]
    fn reported_record_count_matches() {
        let dir = TempDir::new().unwrap();
        make_pair_store(&dir, 7);
        let source = SrdSource::open(
            &dir.path().join("data.srd"),
            "test",
            TEST_EMB_DIM,
            Some(SrdMode::Pair),
        )
        .unwrap();

        let config = SamplerConfig::default();
        let count = source.reported_record_count(&config).unwrap();
        assert_eq!(count, 7);
    }

    #[test]
    fn pair_mode_texts_match() {
        let dir = TempDir::new().unwrap();
        make_pair_store(&dir, 3);
        let source = SrdSource::open(
            &dir.path().join("data.srd"),
            "test",
            TEST_EMB_DIM,
            Some(SrdMode::Pair),
        )
        .unwrap();

        let config = SamplerConfig::default();
        let snapshot = source.refresh(&config, None, None).unwrap();

        // Each record's text must match its own entry index, wherever the
        // permutation placed it.
        for record in &snapshot.records {
            let idx: usize = record.id.parse().unwrap();
            assert_eq!(record.sections[0].text, format!("anchor_{idx}"));
            assert_eq!(record.sections[1].text, format!("positive_{idx}"));
        }
    }

    #[test]
    fn triplet_mode_texts_match() {
        let dir = TempDir::new().unwrap();
        make_triplet_store(&dir, 3);
        let source = SrdSource::open(
            &dir.path().join("data.srd"),
            "test",
            TEST_EMB_DIM,
            Some(SrdMode::Triplet),
        )
        .unwrap();

        let config = SamplerConfig::default();
        let snapshot = source.refresh(&config, None, None).unwrap();

        for record in &snapshot.records {
            let idx: usize = record.id.parse().unwrap();
            assert_eq!(record.sections[0].text, format!("anchor_{idx}"));
            assert_eq!(record.sections[1].text, format!("positive_{idx}"));
            assert_eq!(record.sections[2].text, format!("negative_{idx}"));
        }
    }

    #[test]
    fn source_id_is_set_on_records() {
        let dir = TempDir::new().unwrap();
        make_pair_store(&dir, 2);
        let source = SrdSource::open(
            &dir.path().join("data.srd"),
            "my_source",
            TEST_EMB_DIM,
            Some(SrdMode::Pair),
        )
        .unwrap();

        let config = SamplerConfig::default();
        let snapshot = source.refresh(&config, None, None).unwrap();

        for record in &snapshot.records {
            assert_eq!(record.source, "my_source");
        }
    }

    #[test]
    fn open_nonexistent_path_returns_error() {
        let result = SrdSource::open(
            std::path::Path::new("/nonexistent/path/data.srd"),
            "test",
            TEST_EMB_DIM,
            Some(SrdMode::Pair),
        );
        assert!(result.is_err());
    }

    #[test]
    fn refresh_cursor_past_end_wraps_to_head() {
        let dir = TempDir::new().unwrap();
        make_pair_store(&dir, 3);
        let source = SrdSource::open(
            &dir.path().join("data.srd"),
            "test",
            TEST_EMB_DIM,
            Some(SrdMode::Pair),
        )
        .unwrap();

        let config = SamplerConfig::default();
        let cursor = SourceCursor {
            last_seen: chrono::Utc::now(),
            revision: 100, // far past the end: counter space is periodic, so
                           // the walk just continues instead of stalling
        };
        // Infinite forward stream: large revisions keep drawing disjoint
        // windows instead of stalling on empty ones.
        let snapshot = source.refresh(&config, Some(&cursor), None).unwrap();
        assert_eq!(snapshot.records.len(), 3);
        let mut ids: Vec<String> = snapshot.records.iter().map(|r| r.id.clone()).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec!["0".to_string(), "1".to_string(), "2".to_string()]);
    }

    #[test]
    fn open_with_wrong_mode_is_rejected() {
        let dir = TempDir::new().unwrap();
        make_pair_store(&dir, 3);
        let err = match SrdSource::open(
            &dir.path().join("data.srd"),
            "test",
            TEST_EMB_DIM,
            Some(SrdMode::Triplet),
        ) {
            Ok(_) => panic!("wrong mode must fail loudly"),
            Err(e) => e,
        };
        assert!(
            matches!(err, SrdError::ModeMismatch { .. }),
            "wrong mode must fail loudly, got: {err:?}"
        );
    }

    #[test]
    fn open_detects_pair_and_triplet_modes() {
        let pair_dir = TempDir::new().unwrap();
        make_pair_store(&pair_dir, 3);
        let pair_src = SrdSource::open(
            &pair_dir.path().join("data.srd"),
            "test",
            TEST_EMB_DIM,
            None,
        )
        .unwrap();
        assert_eq!(pair_src.mode(), SrdMode::Pair);
        assert_eq!(pair_src.len(), 3);
        assert!(!pair_src.is_empty());

        let trip_dir = TempDir::new().unwrap();
        make_triplet_store(&trip_dir, 2);
        let trip_src = SrdSource::open(
            &trip_dir.path().join("data.srd"),
            "test",
            TEST_EMB_DIM,
            None,
        )
        .unwrap();
        assert_eq!(trip_src.mode(), SrdMode::Triplet);
        assert_eq!(trip_src.len(), 2);
    }

    #[test]
    fn non_power_of_two_windows_cover_corpus_before_repeating() {
        // total=5 resolves to a power-of-two domain of 8, so rejections occur
        // and the cursor must carry the raw permutation counter: resuming
        // from the folded `cursor() % total` replays already-seen indices
        // before the corpus is covered (verified: seeds 0, 1, 5 replay under
        // folded cursors). limit=2 neither tiles the corpus nor the domain,
        // so the cycle boundary falls mid-window. The assertions below hold
        // for every seed: from counter 0 the first five hits always land
        // inside the first domain cycle, hence distinct and complete.
        for seed in [0u64, 1, 7, 42, 77] {
            let dir = TempDir::new().unwrap();
            make_pair_store(&dir, 5);
            let source = SrdSource::open(
                &dir.path().join("data.srd"),
                "test",
                TEST_EMB_DIM,
                Some(SrdMode::Pair),
            )
            .unwrap();
            let config = config_with_seed(seed);

            let (p1, c1) = refresh_ids(&source, &config, None, 2);
            let (p2, c2) = refresh_ids(&source, &config, Some(&c1), 2);
            let (p3, _) = refresh_ids(&source, &config, Some(&c2), 2);

            // Pages 1-2 lie within the first domain cycle: strictly disjoint.
            {
                use std::collections::HashSet;
                let mut seen = HashSet::new();
                for id in p1.iter().chain(p2.iter()) {
                    assert!(seen.insert(id), "seed {seed}: pages 1-2 replay index {id}");
                }
            }

            // Across three pages the walk covers the whole corpus before any
            // index repeats: the first five draws are exactly 0..5.
            let flat: Vec<String> = p1.into_iter().chain(p2).chain(p3).collect();
            assert_eq!(flat.len(), 6);
            let mut first_five = flat[..5].to_vec();
            first_five.sort_unstable();
            let expected: Vec<String> = (0..5).map(|i| i.to_string()).collect();
            assert_eq!(
                first_five, expected,
                "seed {seed}: walk must cover the corpus before repeating"
            );
            // The sixth draw wraps into the next domain cycle, so it must repeat
            // an already-seen index (the cycle boundary fell mid-window).
            assert!(
                flat[..5].contains(&flat[5]),
                "seed {seed}: post-cycle draw must repeat an already-seen index"
            );
        }
    }

    #[test]
    fn open_rejects_empty_store_without_explicit_mode() {
        let dir = TempDir::new().unwrap();
        let store = DataStore::open(&dir.path().join("data.srd")).unwrap();
        drop(store);
        let err = match SrdSource::open(&dir.path().join("data.srd"), "test", TEST_EMB_DIM, None) {
            Ok(_) => panic!("empty store must fail loudly"),
            Err(e) => e,
        };
        assert!(
            matches!(err, SrdError::EmptyStore),
            "empty store must fail loudly, got: {err:?}"
        );
    }

    fn config_with_seed(seed: u64) -> SamplerConfig {
        SamplerConfig {
            seed,
            ..SamplerConfig::default()
        }
    }

    fn refresh_ids(
        source: &SrdSource,
        config: &SamplerConfig,
        cursor: Option<&SourceCursor>,
        limit: usize,
    ) -> (Vec<String>, SourceCursor) {
        let snapshot = source
            .refresh(config, cursor, Some(limit))
            .expect("refresh must succeed");
        (
            snapshot.records.iter().map(|r| r.id.clone()).collect(),
            snapshot.cursor,
        )
    }

    #[test]
    fn refresh_is_deterministic_for_same_seed_and_cursor() {
        // Issue #192 acceptance: two refreshes with the same seed produce
        // identical order.
        let dir = TempDir::new().unwrap();
        make_pair_store(&dir, 20);
        let source = SrdSource::open(
            &dir.path().join("data.srd"),
            "test",
            TEST_EMB_DIM,
            Some(SrdMode::Pair),
        )
        .unwrap();

        let config = config_with_seed(0xC0FFEE);
        let (first, cursor) = refresh_ids(&source, &config, None, 8);
        let (replay, replay_cursor) = refresh_ids(&source, &config, None, 8);
        assert_eq!(first, replay, "same seed+cursor must yield identical order");
        assert_eq!(cursor.revision, replay_cursor.revision);

        // Continuation from the same cursor is deterministic too.
        let (second, _) = refresh_ids(&source, &config, Some(&cursor), 8);
        let (second_replay, _) = refresh_ids(&source, &config, Some(&cursor), 8);
        assert_eq!(second, second_replay);
    }

    #[test]
    fn different_seeds_draw_from_across_corpus() {
        // Issue #192 acceptance: different seeds/epochs draw from across the
        // corpus rather than adjacent file offsets.
        let dir = TempDir::new().unwrap();
        make_pair_store(&dir, 64);
        let source = SrdSource::open(
            &dir.path().join("data.srd"),
            "test",
            TEST_EMB_DIM,
            Some(SrdMode::Pair),
        )
        .unwrap();

        let (ids_a, _) = refresh_ids(&source, &config_with_seed(11), None, 16);
        let (ids_b, _) = refresh_ids(&source, &config_with_seed(12), None, 16);
        assert_ne!(
            ids_a, ids_b,
            "different seeds must permute differently (seed is ignored)"
        );

        for (seed, ids) in [(11u64, &ids_a), (12u64, &ids_b)] {
            let file_order: Vec<String> = (0..16).map(|i| i.to_string()).collect();
            assert_ne!(
                ids, &file_order,
                "seed {seed}: window must not be the leading file-ordered slice"
            );
            let idx: Vec<usize> = ids.iter().map(|id| id.parse().unwrap()).collect();
            let spread = idx.iter().max().unwrap() - idx.iter().min().unwrap();
            assert!(
                spread >= 32,
                "seed {seed}: window spans only {spread} file positions, expected spread across the corpus"
            );
        }
    }

    #[test]
    fn consecutive_windows_with_same_seed_are_disjoint_at_scale() {
        // Stream contract (see AGENTS.md): with a fixed seed, consecutive
        // refreshes advance the permutation counter and return disjoint
        // windows with no repeats until the counter wraps past the corpus.
        // n=256 is a power of two so the permutation domain has no rejected
        // positions and four 64-row windows tile the corpus exactly.
        let dir = TempDir::new().unwrap();
        make_pair_store(&dir, 256);
        let source = SrdSource::open(
            &dir.path().join("data.srd"),
            "test",
            TEST_EMB_DIM,
            Some(SrdMode::Pair),
        )
        .unwrap();
        let config = config_with_seed(77);

        let (w0, c0) = refresh_ids(&source, &config, None, 64);
        let (w1, c1) = refresh_ids(&source, &config, Some(&c0), 64);
        let (w2, c2) = refresh_ids(&source, &config, Some(&c1), 64);
        let (w3, _) = refresh_ids(&source, &config, Some(&c2), 64);

        let mut seen = std::collections::HashSet::new();
        for (n, window) in [&w0, &w1, &w2, &w3].into_iter().enumerate() {
            assert_eq!(window.len(), 64, "window {n} must be full");
            for id in window {
                assert!(
                    seen.insert(id.clone()),
                    "window {n} replays id {id} from an earlier window"
                );
            }
        }
        assert_eq!(seen.len(), 256, "four windows must tile the corpus exactly");

        // Wrap-around: the next window restarts the cycle.
        let last_cursor = source.refresh(&config, Some(&c2), Some(64)).unwrap().cursor;
        let (w4, _) = refresh_ids(&source, &config, Some(&last_cursor), 64);
        assert_eq!(w4, w0, "post-cycle window must restart the permutation");
    }
}
