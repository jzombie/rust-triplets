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
/// [`RecordSection`](triplets::data::RecordSection) embedding channel.
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
        _config: &SamplerConfig,
        cursor: Option<&SourceCursor>,
        limit: Option<usize>,
    ) -> Result<SourceSnapshot, SamplerError> {
        let total = self.entry_count.load(Ordering::Relaxed);
        let start = cursor.map(|c| c.revision).unwrap_or(0);
        let batch_size = limit
            .unwrap_or(256)
            .min(total.saturating_sub(start) as usize);

        // Batch-read all entries in one call.
        let indices: Vec<usize> = (start..start + batch_size as u64)
            .filter(|&i| i < total)
            .map(|i| i as usize)
            .collect();
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
            revision: start + batch_size as u64,
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

        for (i, record) in snapshot.records.iter().enumerate() {
            assert_eq!(record.id, i.to_string());
        }
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
        make_pair_store(&dir, 5);
        let source = SrdSource::open(
            &dir.path().join("data.srd"),
            "test",
            TEST_EMB_DIM,
            Some(SrdMode::Pair),
        )
        .unwrap();

        let config = SamplerConfig::default();

        // First page: limit=2
        let snapshot1 = source.refresh(&config, None, Some(2)).unwrap();
        assert_eq!(snapshot1.records.len(), 2);
        assert_eq!(snapshot1.records[0].id, "0");
        assert_eq!(snapshot1.records[1].id, "1");

        // Second page: use cursor from first page
        let snapshot2 = source
            .refresh(&config, Some(&snapshot1.cursor), Some(2))
            .unwrap();
        assert_eq!(snapshot2.records.len(), 2);
        assert_eq!(snapshot2.records[0].id, "2");
        assert_eq!(snapshot2.records[1].id, "3");

        // Third page: remaining entry
        let snapshot3 = source
            .refresh(&config, Some(&snapshot2.cursor), Some(2))
            .unwrap();
        assert_eq!(snapshot3.records.len(), 1);
        assert_eq!(snapshot3.records[0].id, "4");
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

        for (i, record) in snapshot.records.iter().enumerate() {
            assert_eq!(record.sections[0].text, format!("anchor_{i}"));
            assert_eq!(record.sections[1].text, format!("positive_{i}"));
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

        for (i, record) in snapshot.records.iter().enumerate() {
            assert_eq!(record.sections[0].text, format!("anchor_{i}"));
            assert_eq!(record.sections[1].text, format!("positive_{i}"));
            assert_eq!(record.sections[2].text, format!("negative_{i}"));
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
    fn refresh_cursor_past_end_returns_empty() {
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
            revision: 100, // past the end
        };
        let snapshot = source.refresh(&config, Some(&cursor), None).unwrap();
        assert!(snapshot.records.is_empty());
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
}
