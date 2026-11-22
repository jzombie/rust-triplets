use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use jiff::Timestamp;
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
/// `refresh` draws a deterministic random window through a permutation of
/// the index domain `0..entry_count`, mirroring
/// [`triplets_core::source::IndexablePager`] / `ParquetRowSource`:
///
/// - The permutation seed is
///   `IndexablePager::seed_for_sampler(source_id, total, config.seed)`, where
///   `config.seed` is the epoch/step-mixed seed the ingestion manager already
///   threads through (`derive_epoch_seed(base, epoch) ^ step`). A fixed
///   `(source_id, total, seed, cursor)` therefore always yields the same
///   indices; different seeds/epochs/steps draw from across the corpus rather
///   than adjacent file offsets.
/// - The cursor `revision` is a position in the permutation walk, folded into
///   `[0, total)` exactly like every other pager in the crate — not a file
///   offset. Out-of-range revisions wrap to 0 so the infinite forward stream
///   never stalls. Like the other pagers, a refresh draws an independent
///   seed-derived window: cross-refresh non-repetition is not a cursor
///   contract (the corpus is modeled as an infinite stream), and overlapping
///   windows of ~`window^2/total` rows are expected — negligible for large
///   stores, and never a coherent file-ordered block.
/// - Within one refresh the returned order is the permutation order (random),
///   not sorted file order.
///
/// ## Why the permutation is driven manually instead of via `IndexablePager`
///
/// `IndexablePager::refresh_with` materializes one `(index, cursor)` pair
/// per corpus row. Driving `IndexPermutation` directly draws only `max`
/// indices and serves them with a single `batch_read_entries` call.
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
                    last_seen: Timestamp::now(),
                    revision: 0,
                },
            });
        }
        let total: usize = usize::try_from(total_u64).map_err(|e| {
            SamplerError::Configuration(format!(
                "SRD store entry count {total_u64} exceeds index range: {e}"
            ))
        })?;
        // The cursor revision is a position in the permutation walk, folded
        // into `[0, total)` like every other pager: out-of-range revisions
        // wrap to 0 so the infinite forward stream never stalls on empty
        // windows.
        let mut start = cursor.map(|c| c.revision).unwrap_or(0);
        if start >= total_u64 {
            start = 0;
        }
        let max = limit.unwrap_or(256).min(total);
        if max == 0 {
            return Ok(SourceSnapshot {
                records: Vec::new(),
                cursor: SourceCursor {
                    last_seen: Timestamp::now(),
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

        let now = Timestamp::now();
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
            last_seen: Timestamp::now(),
            // Persist the folded walk position, like every other pager:
            // revision stays in `[0, total)` and the next refresh continues
            // the walk from there.
            revision: permutation.cursor() as u64,
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
            last_seen: jiff::Timestamp::now(),
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
    fn non_power_of_two_refresh_is_deterministic_per_cursor() {
        // total=5 resolves to a power-of-two domain of 8, so domain rejections
        // occur. The cursor contract is per-position determinism — the same
        // (seed, cursor) always yields the same window — not cross-window
        // tiling: the corpus is an infinite stream and each refresh draws an
        // independent seed-derived window.
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

            // Replaying from the same cursors reproduces every page exactly.
            let (q1, d1) = refresh_ids(&source, &config, None, 2);
            let (q2, d2) = refresh_ids(&source, &config, Some(&d1), 2);
            let (q3, _) = refresh_ids(&source, &config, Some(&d2), 2);
            assert_eq!((&p1, &p2, &p3), (&q1, &q2, &q3));
            assert_eq!(c1.revision, d1.revision);
            assert_eq!(c2.revision, d2.revision);
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
        // n=256 is a power of two, so the permutation domain has no rejected
        // positions: the folded cursor coincides with the walk position and
        // four 64-row windows tile the corpus exactly. (For non-power-of-two
        // totals the cursor only promises per-position determinism — see
        // `non_power_of_two_refresh_is_deterministic_per_cursor`.)
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

    #[test]
    fn sampler_draws_triplet_batches_from_srd_source_end_to_end() {
        // The permuted windows must survive the full ingestion → split →
        // sampling pipeline, not just direct `refresh()`. Uses the source's
        // default recipes (empty config recipes) to prove the
        // default-recipe wiring too.
        use std::collections::HashMap;
        use triplets_core::{DeterministicSplitStore, SplitLabel, SplitRatios, TripletSampler};

        let dir = TempDir::new().unwrap();
        make_pair_store(&dir, 32);
        let source = SrdSource::open(
            &dir.path().join("data.srd"),
            "srd_e2e",
            TEST_EMB_DIM,
            Some(SrdMode::Pair),
        )
        .unwrap();

        let split = SplitRatios {
            train: 1.0,
            validation: 0.0,
            test: 0.0,
        };
        let store = std::sync::Arc::new(DeterministicSplitStore::new(split, 77).unwrap());
        let config = SamplerConfig {
            seed: 77,
            batch_size: 8,
            ..SamplerConfig::default()
        };
        let sampler = TripletSampler::new(config, store);
        sampler.register_source(Box::new(source)).unwrap();
        sampler.set_epoch(0).unwrap();

        let mut weights = HashMap::new();
        weights.insert("srd_e2e".to_string(), 1.0);
        let batch = sampler
            .next_triplet_batch_with_weights_for_split(SplitLabel::Train, &weights)
            .expect("triplet batch from SRD source must succeed");
        assert_eq!(batch.triplets.len(), 8);

        for triplet in &batch.triplets {
            assert_eq!(triplet.recipe, "srd_pair");
            for chunk in [&triplet.anchor, &triplet.positive, &triplet.negative] {
                assert_eq!(chunk.source, "srd_e2e");
                let idx: usize = chunk.record_id.parse().expect("entry index id");
                assert!(idx < 32, "chunk references unknown entry {idx}");
                let emb = chunk.embedding.as_ref().expect("section embedding");
                assert_eq!(emb.len(), TEST_EMB_DIM);
            }
            // Pair mode: positive is the candidate of the same entry,
            // WrongArticle negatives come from a different record.
            assert_eq!(triplet.positive.record_id, triplet.anchor.record_id);
            assert_ne!(triplet.anchor.record_id, triplet.negative.record_id);
        }

        // A second draw keeps working through the same pipeline (pool
        // advance, re-selection) with the same invariants.
        let batch2 = sampler
            .next_triplet_batch_with_weights_for_split(SplitLabel::Train, &weights)
            .expect("second triplet batch must succeed");
        assert_eq!(batch2.triplets.len(), 8);
        for triplet in &batch2.triplets {
            assert_eq!(triplet.recipe, "srd_pair");
            for chunk in [&triplet.anchor, &triplet.positive, &triplet.negative] {
                assert_eq!(chunk.source, "srd_e2e");
                let idx: usize = chunk.record_id.parse().expect("entry index id");
                assert!(idx < 32, "chunk references unknown entry {idx}");
            }
        }
    }

    #[test]
    fn epoch_derived_seeds_repermute_srd_windows() {
        // Pins the exact seed derivation `IngestionManager` performs per
        // refresh (`derive_epoch_seed(base, epoch) ^ step`): epoch/step
        // changes must move SRD windows, not replay them.
        use triplets_core::hash::derive_epoch_seed;

        let dir = TempDir::new().unwrap();
        make_pair_store(&dir, 64);
        let source = SrdSource::open(
            &dir.path().join("data.srd"),
            "test",
            TEST_EMB_DIM,
            Some(SrdMode::Pair),
        )
        .unwrap();

        let base = 99u64;
        let window = |epoch: u64, step: u64| {
            let seed = derive_epoch_seed(base, epoch) ^ step;
            refresh_ids(&source, &config_with_seed(seed), None, 16).0
        };
        // Note: seeds are `base ^ epoch ^ step`, so (epoch, step) pairs with
        // equal `epoch ^ step` share a seed by XOR construction (e.g. (0, 1)
        // and (1, 0)). The triples below use distinct combined values.
        let e0s0 = window(0, 0);
        let e0s1 = window(0, 1);
        let e1s2 = window(1, 2);
        assert_ne!(e0s0, e0s1, "step must repermute the window");
        assert_ne!(e0s0, e1s2, "epoch must repermute the window");
        assert_ne!(e0s1, e1s2, "epoch and step must combine");
    }

    #[test]
    fn seeded_windows_behave_as_independent_subsets_at_scale() {
        // Ensemble regression test over 64 fixed seeds (n=1024, w=128).
        // Each assertion fails loudly for a concrete known-bad
        // implementation while holding with wide margin for the real walk
        // (verified by running; bands below are theory-derived, not fitted):
        // - distinct-window count fails if seeds are ignored (all 64
        //   windows identical);
        // - spread fails if windows are file-ordered (spread would be w-1);
        // - mean overlap near w^2/n = 16 fails for replay (128) and for
        //   fixed partitions (0);
        // - joint coverage near n fails for regional bias (half the corpus
        //   dropped caps coverage at ~512);
        // - decile floor fails file-ordered reads (9 empty deciles) and
        //   half-corpus bias (5 empty); only the minimum is pinned since
        //   small-step walks legitimately concentrate draws (no ceiling).
        // Note: one walk is an arithmetic progression mod the domain, and a
        // walk is fully determined by seed bits 1..k (2^k distinct walks
        // for a 2^k domain), so among 64 seeds ~2 full-walk collisions are
        // EXPECTED by the birthday bound (observed: seeds 20 and 32) — that
        // is the design's key space, not a bug. Distinct-count (not
        // pairwise-distinctness) is therefore the correct assertion, and
        // the spread assertion tolerates small-step clustered windows by
        // design instead of demanding every window span the corpus.
        let dir = TempDir::new().unwrap();
        make_pair_store(&dir, 1024);
        let source = SrdSource::open(
            &dir.path().join("data.srd"),
            "test",
            TEST_EMB_DIM,
            Some(SrdMode::Pair),
        )
        .unwrap();

        let seeds: Vec<u64> = (1..=64).collect();
        let windows: Vec<Vec<String>> = seeds
            .iter()
            .map(|&s| refresh_ids(&source, &config_with_seed(s), None, 128).0)
            .collect();

        // Distinct windows: with 64 seeds sharing a 2^10 effective
        // permutation-key space, ~2 full-walk collisions are expected by the
        // birthday bound (observed: seeds 20 and 32), so expect ~62 distinct
        // windows, not 64. The floor only needs to catch a broken seed
        // thread, which collapses all 64 windows to 1.
        let mut distinct = std::collections::HashSet::new();
        for w in &windows {
            distinct.insert(w);
        }
        assert!(
            distinct.len() >= 58,
            "only {}/64 distinct windows (seed threading broken gives 1)",
            distinct.len()
        );

        // Spread: max-min entry index spanned by one 128-row window.
        // A file-ordered window always spans exactly w-1 = 127, so the 512
        // bar (half the corpus) cleanly separates shuffled windows from
        // sequential ones; a few small-step progressions legitimately fall
        // short, hence 56/64 instead of 64/64 (observed: 64/64).
        // Decile floor: aggregate share per tenth of this test's 1024 entry
        // indices, aggregated over all 64 windows (8192 draws, ~819
        // expected per decile). Only the
        // minimum is pinned, deliberately: small-step walks legitimately
        // concentrate draws, so a ceiling would punish healthy clustering,
        // while any gross bias empties deciles outright (file-ordered
        // reads leave 9 deciles near-empty, half-corpus bias 5). The 400
        // floor (~half of expected) is far below any healthy minimum and
        // far above any biased one.
        let mut deciles = [0usize; 10];
        for w in &windows {
            for id in w {
                let idx: usize = id.parse().unwrap();
                deciles[idx * 10 / 1024] += 1;
            }
        }
        let min_decile = *deciles.iter().min().unwrap();
        assert!(
            min_decile >= 400,
            "emptiest decile holds only {min_decile}/8192 draws (expected ~819)"
        );
        let mut wide = 0usize;
        for w in &windows {
            // One walk segment can never repeat an index (each domain
            // position is visited once per cycle), so every window is
            // internally duplicate-free regardless of seed.
            assert_eq!(
                w.iter().collect::<std::collections::HashSet<_>>().len(),
                w.len(),
                "window repeats an index within itself"
            );
            let idx: Vec<usize> = w.iter().map(|id| id.parse().unwrap()).collect();
            if idx.iter().max().unwrap() - idx.iter().min().unwrap() >= 512 {
                wide += 1;
            }
        }
        assert!(
            wide >= 56,
            "only {wide}/64 windows span the corpus (sequential would give 0)"
        );

        // Pairwise overlap = number of row ids shared by one pair of
        // windows, averaged over all 2016 pairs. Each window holds w/n =
        // 1/8 of the corpus, so for two INDEPENDENT windows a given row
        // lands in both with probability 1/8 * 1/8 = 1/64; over n = 1024
        // rows that is 1024/64 = 16 shared rows on average
        // (hypergeometric mean w^2/n). Averaging over all pairs squeezes
        // sampling noise to ~0.1, so [10, 22] is enormously wide for a
        // healthy shuffle (observed mean: 16.02) and only fails on
        // systematic breakage: ~128 means every window is identical (seed
        // ignored / replay), ~0 means rigid disjoint partitions.
        let sets: Vec<std::collections::HashSet<&str>> = windows
            .iter()
            .map(|w| w.iter().map(String::as_str).collect())
            .collect();
        let mut total_overlap = 0usize;
        let mut pairs = 0usize;
        for (i, a) in sets.iter().enumerate() {
            for b in sets.iter().skip(i + 1) {
                total_overlap += b.iter().filter(|id| a.contains(*id)).count();
                pairs += 1;
            }
        }
        let mean = total_overlap as f64 / pairs as f64;
        assert!(
            (10.0..=22.0).contains(&mean),
            "mean pairwise overlap {mean:.2} outside [10, 22] around expected 16"
        );

        // Joint coverage = distinct rows seen across all 64 windows. Each
        // window covers 1/8 of the corpus, so P(a given row is never drawn)
        // = (7/8)^64 ≈ 0.0002 and expect ~1023.8 of 1024 rows seen
        // (observed: 1023). The 1010 floor only needs to catch gross bias:
        // seed-ignored caps at 128, half-the-corpus bias at ~512.
        let mut covered = std::collections::HashSet::new();
        for w in &windows {
            for id in w {
                covered.insert(id);
            }
        }
        assert!(
            covered.len() >= 1010,
            "joint coverage only {}/1024 of the corpus",
            covered.len()
        );
    }

    #[test]
    fn refresh_with_zero_limit_returns_empty_and_holds_position() {
        let dir = TempDir::new().unwrap();
        make_pair_store(&dir, 10);
        let source = SrdSource::open(
            &dir.path().join("data.srd"),
            "test",
            TEST_EMB_DIM,
            Some(SrdMode::Pair),
        )
        .unwrap();

        let config = SamplerConfig::default();
        let snap = source.refresh(&config, None, Some(0)).unwrap();
        assert!(snap.records.is_empty());
        assert_eq!(snap.cursor.revision, 0);

        let cursor = SourceCursor {
            last_seen: jiff::Timestamp::now(),
            revision: 3,
        };
        let snap2 = source.refresh(&config, Some(&cursor), Some(0)).unwrap();
        assert!(snap2.records.is_empty());
        assert_eq!(snap2.cursor.revision, 3);
    }

    #[test]
    fn refresh_defaults_to_256_rows_on_large_stores() {
        let dir = TempDir::new().unwrap();
        make_pair_store(&dir, 300);
        let source = SrdSource::open(
            &dir.path().join("data.srd"),
            "test",
            TEST_EMB_DIM,
            Some(SrdMode::Pair),
        )
        .unwrap();

        let snapshot = source
            .refresh(&SamplerConfig::default(), None, None)
            .unwrap();
        assert_eq!(snapshot.records.len(), 256);
    }

    #[test]
    fn concurrent_refreshes_agree() {
        // The ingestion fan-out calls `refresh` from scoped threads sharing
        // one source: same inputs must yield identical outputs.
        let dir = TempDir::new().unwrap();
        make_pair_store(&dir, 64);
        let source = SrdSource::open(
            &dir.path().join("data.srd"),
            "test",
            TEST_EMB_DIM,
            Some(SrdMode::Pair),
        )
        .unwrap();
        let config = config_with_seed(5);

        std::thread::scope(|s| {
            let handles: Vec<_> = (0..8)
                .map(|_| s.spawn(|| refresh_ids(&source, &config, None, 64)))
                .collect();
            let mut results = Vec::with_capacity(handles.len());
            for h in handles {
                results.push(h.join().expect("worker must not panic"));
            }
            for (i, (ids, cursor)) in results.iter().enumerate() {
                assert_eq!(ids, &results[0].0, "worker {i} diverged");
                assert_eq!(
                    cursor.revision, results[0].1.revision,
                    "worker {i} cursor diverged"
                );
            }
        });
    }
}
