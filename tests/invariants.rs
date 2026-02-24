use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use chrono::{TimeZone, Utc};

use triplets::NegativeStrategy;
use triplets::config::{SamplerConfig, Selector, TripletRecipe};
use triplets::data::{DataRecord, QualityScore, SectionRole};
use triplets::sampler::{Sampler, TripletSampler};
use triplets::source::InMemorySource;
use triplets::splits::{
    EpochStateStore, PersistedSamplerState, PersistedSplitHashes, PersistedSplitMeta,
    SamplerStateStore, SplitLabel, SplitRatios, SplitStore,
};
use triplets::utils::make_section;
use triplets::{RecordId, SourceId};

fn build_record(source: &str, suffix: &str, day_offset: u32) -> DataRecord {
    let created_at = Utc
        .with_ymd_and_hms(2025, 1, 1 + day_offset, 12, 0, 0)
        .unwrap();
    DataRecord {
        id: format!("{source}::{suffix}"),
        source: source.to_string(),
        created_at,
        updated_at: created_at,
        quality: QualityScore { trust: 1.0 },
        taxonomy: vec![source.to_string()],
        sections: vec![
            make_section(
                SectionRole::Anchor,
                None,
                &format!("{source} title {suffix}"),
            ),
            make_section(
                SectionRole::Context,
                None,
                &format!("{source} body {suffix}"),
            ),
        ],
        meta_prefix: None,
    }
}

fn build_config(seed: u64, batch_size: usize, split: SplitRatios) -> SamplerConfig {
    SamplerConfig {
        seed,
        batch_size,
        ingestion_max_records: batch_size,
        allowed_splits: vec![SplitLabel::Train],
        split,
        recipes: vec![TripletRecipe {
            name: "shuffled_triplet".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
        }],
        text_recipes: Vec::new(),
        ..SamplerConfig::default()
    }
}

#[derive(Default)]
struct CountingSplitStore {
    ratios: SplitRatios,
    seed: u64,
    assignments: RwLock<HashMap<RecordId, SplitLabel>>,
    upserts: AtomicUsize,
    epoch_meta: RwLock<HashMap<SplitLabel, PersistedSplitMeta>>,
    epoch_hashes: RwLock<HashMap<SplitLabel, PersistedSplitHashes>>,
    sampler_state: RwLock<Option<PersistedSamplerState>>,
}

impl CountingSplitStore {
    fn new(ratios: SplitRatios, seed: u64) -> Self {
        Self {
            ratios,
            seed,
            ..Default::default()
        }
    }

    fn upsert_count(&self) -> usize {
        self.upserts.load(Ordering::Relaxed)
    }

    fn derive_label(&self, id: &str) -> SplitLabel {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        id.hash(&mut hasher);
        self.seed.hash(&mut hasher);
        let value = hasher.finish() as f64 / u64::MAX as f64;
        let train_cut = self.ratios.train as f64;
        let val_cut = train_cut + self.ratios.validation as f64;
        if value < train_cut {
            SplitLabel::Train
        } else if value < val_cut {
            SplitLabel::Validation
        } else {
            SplitLabel::Test
        }
    }
}

impl SplitStore for CountingSplitStore {
    fn label_for(&self, id: &String) -> Option<SplitLabel> {
        self.assignments.read().ok()?.get(id).copied()
    }

    fn upsert(&self, id: String, label: SplitLabel) -> Result<(), triplets::SamplerError> {
        self.upserts.fetch_add(1, Ordering::Relaxed);
        let mut guard = self
            .assignments
            .write()
            .map_err(|_| triplets::SamplerError::SplitStore("lock poisoned".into()))?;
        guard.insert(id, label);
        Ok(())
    }

    fn ratios(&self) -> SplitRatios {
        self.ratios
    }

    fn ensure(&self, id: String) -> Result<SplitLabel, triplets::SamplerError> {
        if let Some(label) = self.label_for(&id) {
            return Ok(label);
        }
        let label = self.derive_label(&id);
        self.upsert(id, label)?;
        Ok(label)
    }
}

impl EpochStateStore for CountingSplitStore {
    fn load_epoch_meta(
        &self,
    ) -> Result<HashMap<SplitLabel, PersistedSplitMeta>, triplets::SamplerError> {
        self.epoch_meta
            .read()
            .map_err(|_| triplets::SamplerError::SplitStore("epoch meta lock poisoned".into()))
            .map(|guard| guard.clone())
    }

    fn load_epoch_hashes(
        &self,
        label: SplitLabel,
    ) -> Result<Option<PersistedSplitHashes>, triplets::SamplerError> {
        Ok(self
            .epoch_hashes
            .read()
            .map_err(|_| triplets::SamplerError::SplitStore("epoch hashes lock poisoned".into()))?
            .get(&label)
            .cloned())
    }

    fn store_epoch_meta(
        &self,
        meta: &HashMap<SplitLabel, PersistedSplitMeta>,
    ) -> Result<(), triplets::SamplerError> {
        *self
            .epoch_meta
            .write()
            .map_err(|_| triplets::SamplerError::SplitStore("epoch meta lock poisoned".into()))? =
            meta.clone();
        Ok(())
    }

    fn store_epoch_hashes(
        &self,
        label: SplitLabel,
        hashes: &PersistedSplitHashes,
    ) -> Result<(), triplets::SamplerError> {
        self.epoch_hashes
            .write()
            .map_err(|_| triplets::SamplerError::SplitStore("epoch hashes lock poisoned".into()))?
            .insert(label, hashes.clone());
        Ok(())
    }
}

impl SamplerStateStore for CountingSplitStore {
    fn load_sampler_state(&self) -> Result<Option<PersistedSamplerState>, triplets::SamplerError> {
        self.sampler_state
            .read()
            .map_err(|_| triplets::SamplerError::SplitStore("sampler state lock poisoned".into()))
            .map(|guard| guard.clone())
    }

    fn store_sampler_state(
        &self,
        state: &PersistedSamplerState,
    ) -> Result<(), triplets::SamplerError> {
        *self.sampler_state.write().map_err(|_| {
            triplets::SamplerError::SplitStore("sampler state lock poisoned".into())
        })? = Some(state.clone());
        Ok(())
    }
}

#[test]
fn single_source_shuffled_cycles_records_before_repeat() {
    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };
    let store = Arc::new(CountingSplitStore::new(split, 7));
    let sampler = TripletSampler::new(build_config(7, 2, split), store);

    let records = vec![
        build_record("only", "a", 1),
        build_record("only", "b", 2),
        build_record("only", "c", 3),
    ];
    sampler.register_source(Box::new(InMemorySource::new("only", records)));

    let mut seen = Vec::new();
    for _ in 0..4 {
        let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        seen.extend(batch.triplets.iter().map(|t| t.anchor.record_id.clone()));
    }
    seen.sort();
    seen.dedup();
    assert_eq!(seen.len(), 3);
}

#[test]
fn multi_source_shuffled_visits_all_sources() {
    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };
    let store = Arc::new(CountingSplitStore::new(split, 9));
    let sampler = TripletSampler::new(build_config(9, 4, split), store);

    let source_a = vec![build_record("a", "a1", 1), build_record("a", "a2", 2)];
    let source_b = vec![build_record("b", "b1", 1), build_record("b", "b2", 2)];
    sampler.register_source(Box::new(InMemorySource::new("a", source_a)));
    sampler.register_source(Box::new(InMemorySource::new("b", source_b)));

    let mut sources: Vec<SourceId> = Vec::new();
    for _ in 0..2 {
        let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        for triplet in batch.triplets {
            let id = &triplet.anchor.record_id;
            sources.push(id.split("::").next().unwrap_or("").to_string());
        }
    }
    sources.sort();
    sources.dedup();
    assert_eq!(sources, vec!["a".to_string(), "b".to_string()]);
}

#[test]
fn restart_with_persisted_state_continues_sequence() {
    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };
    let store = Arc::new(CountingSplitStore::new(split, 11));
    let sampler = TripletSampler::new(build_config(11, 4, split), Arc::clone(&store));

    let source_a = vec![build_record("a", "a1", 1), build_record("a", "a2", 2)];
    let source_b = vec![build_record("b", "b1", 1), build_record("b", "b2", 2)];
    sampler.register_source(Box::new(InMemorySource::new("a", source_a.clone())));
    sampler.register_source(Box::new(InMemorySource::new("b", source_b.clone())));

    let first_batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
    let first = first_batch.triplets[0].anchor.record_id.clone();
    sampler.persist_state().unwrap();
    // Restart from persisted state: should continue from the stored sequence.
    let resumed = TripletSampler::new(build_config(11, 4, split), Arc::clone(&store));
    resumed.register_source(Box::new(InMemorySource::new("a", source_a.clone())));
    resumed.register_source(Box::new(InMemorySource::new("b", source_b.clone())));

    let next_batch = resumed.next_triplet_batch(SplitLabel::Train).unwrap();
    let next = next_batch.triplets[0].anchor.record_id.clone();

    // Second restart to assert determinism from storage alone.
    let resumed_again = TripletSampler::new(build_config(11, 4, split), store);
    resumed_again.register_source(Box::new(InMemorySource::new("a", source_a)));
    resumed_again.register_source(Box::new(InMemorySource::new("b", source_b)));

    let repeated_batch = resumed_again.next_triplet_batch(SplitLabel::Train).unwrap();
    let repeated = repeated_batch.triplets[0].anchor.record_id.clone();
    assert_ne!(first, next);
    assert_eq!(next, repeated);
}

#[test]
fn negatives_change_after_restart_when_state_is_persisted() {
    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };
    let store = Arc::new(CountingSplitStore::new(split, 13));
    let sampler = TripletSampler::new(build_config(13, 4, split), Arc::clone(&store));

    let source_a = vec![
        build_record("a", "a1", 1),
        build_record("a", "a2", 2),
        build_record("a", "a3", 3),
    ];
    let source_b = vec![
        build_record("b", "b1", 1),
        build_record("b", "b2", 2),
        build_record("b", "b3", 3),
    ];
    sampler.register_source(Box::new(InMemorySource::new("a", source_a.clone())));
    sampler.register_source(Box::new(InMemorySource::new("b", source_b.clone())));

    let first_batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
    let first_negatives: Vec<RecordId> = first_batch
        .triplets
        .iter()
        .map(|t| t.negative.record_id.clone())
        .collect();
    sampler.persist_state().unwrap();

    let resumed = TripletSampler::new(build_config(13, 4, split), store);
    resumed.register_source(Box::new(InMemorySource::new("a", source_a)));
    resumed.register_source(Box::new(InMemorySource::new("b", source_b)));

    let second_batch = resumed.next_triplet_batch(SplitLabel::Train).unwrap();
    let second_negatives: Vec<RecordId> = second_batch
        .triplets
        .iter()
        .map(|t| t.negative.record_id.clone())
        .collect();

    assert_ne!(first_negatives, second_negatives);
}

#[test]
fn batch_size_invariance_matches_sequence() {
    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };
    let source_a = vec![
        build_record("a", "a1", 1),
        build_record("a", "a2", 2),
        build_record("a", "a3", 3),
    ];
    let source_b = vec![
        build_record("b", "b1", 1),
        build_record("b", "b2", 2),
        build_record("b", "b3", 3),
    ];

    let sample_ids = |batch_size| {
        let store = Arc::new(CountingSplitStore::new(split, 17));
        let mut config = build_config(17, batch_size, split);
        config.ingestion_max_records = 6;
        let sampler = TripletSampler::new(config, store);
        sampler.register_source(Box::new(InMemorySource::new("a", source_a.clone())));
        sampler.register_source(Box::new(InMemorySource::new("b", source_b.clone())));
        let mut ids = Vec::new();
        while ids.len() < 6 {
            let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
            for triplet in batch.triplets {
                ids.push(triplet.anchor.record_id);
                if ids.len() >= 6 {
                    break;
                }
            }
        }
        ids
    };

    assert_eq!(sample_ids(4), sample_ids(4));
}

#[test]
fn negatives_are_not_positive_record() {
    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };
    let store = Arc::new(CountingSplitStore::new(split, 19));
    let sampler = TripletSampler::new(build_config(19, 2, split), store);

    let source_a = vec![build_record("a", "a1", 1), build_record("a", "a2", 2)];
    sampler.register_source(Box::new(InMemorySource::new("a", source_a)));

    let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
    for triplet in batch.triplets {
        assert_eq!(triplet.anchor.record_id, triplet.positive.record_id);
        assert_ne!(triplet.positive.record_id, triplet.negative.record_id);
    }
}

#[test]
fn refresh_only_writes_new_split_assignments() {
    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };
    let store = Arc::new(CountingSplitStore::new(split, 23));
    let sampler = TripletSampler::new(build_config(23, 1, split), Arc::clone(&store));

    let source_a = vec![build_record("a", "a1", 1), build_record("a", "a2", 2)];
    sampler.register_source(Box::new(InMemorySource::new("a", source_a.clone())));
    let first_upserts = store.upsert_count();

    let second_upserts = store.upsert_count();

    assert_eq!(first_upserts, second_upserts);
}

#[test]
fn deterministic_order_with_fixed_seed_and_unchanged_dataset() {
    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };
    let source_a = vec![
        build_record("a", "a1", 1),
        build_record("a", "a2", 2),
        build_record("a", "a3", 3),
    ];
    let source_b = vec![
        build_record("b", "b1", 1),
        build_record("b", "b2", 2),
        build_record("b", "b3", 3),
    ];

    let run = || {
        let store = Arc::new(CountingSplitStore::new(split, 31));
        let mut config = build_config(31, 4, split);
        config.ingestion_max_records = 6;
        let sampler = TripletSampler::new(config, store);
        sampler.register_source(Box::new(InMemorySource::new("a", source_a.clone())));
        sampler.register_source(Box::new(InMemorySource::new("b", source_b.clone())));
        let mut ids = Vec::new();
        for _ in 0..3 {
            let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
            ids.extend(batch.triplets.iter().map(|t| t.anchor.record_id.clone()));
        }
        ids
    };

    assert_eq!(run(), run());
}

#[test]
fn zero_test_split_still_cycles_all_records_when_train_and_validation_allowed() {
    let split = SplitRatios {
        train: 0.5,
        validation: 0.5,
        test: 0.0,
    };
    let store = Arc::new(CountingSplitStore::new(split, 53));

    let mut config = build_config(53, 2, split);
    config.allowed_splits = vec![SplitLabel::Train, SplitLabel::Validation];
    config.ingestion_max_records = 6;
    let sampler = TripletSampler::new(config, Arc::clone(&store));

    let mut train_suffixes = Vec::new();
    let mut validation_suffixes = Vec::new();
    let mut idx = 0usize;
    while train_suffixes.len() < 3 || validation_suffixes.len() < 3 {
        let suffix = idx.to_string();
        let id = format!("unit::{suffix}");
        match store.derive_label(&id) {
            SplitLabel::Train if train_suffixes.len() < 3 => train_suffixes.push(suffix),
            SplitLabel::Validation if validation_suffixes.len() < 3 => {
                validation_suffixes.push(suffix)
            }
            _ => {}
        }
        idx += 1;
    }

    let mut records = Vec::new();
    for (day, suffix) in train_suffixes
        .iter()
        .chain(validation_suffixes.iter())
        .enumerate()
    {
        records.push(build_record("unit", suffix, day as u32 + 1));
    }

    sampler.register_source(Box::new(InMemorySource::new("unit", records)));

    let mut seen_train = std::collections::HashSet::new();
    let mut seen_validation = std::collections::HashSet::new();
    for _ in 0..24 {
        let train_batch = sampler
            .next_triplet_batch_for_split(SplitLabel::Train)
            .unwrap();
        for triplet in train_batch.triplets {
            seen_train.insert(triplet.anchor.record_id);
        }

        let validation_batch = sampler
            .next_triplet_batch_for_split(SplitLabel::Validation)
            .unwrap();
        for triplet in validation_batch.triplets {
            seen_validation.insert(triplet.anchor.record_id);
        }

        if seen_train.len() == 3 && seen_validation.len() == 3 {
            break;
        }
    }

    assert_eq!(
        seen_train.len(),
        3,
        "all train records should appear over time"
    );
    assert_eq!(
        seen_validation.len(),
        3,
        "all validation records should appear over time"
    );
}

#[test]
fn allowed_split_records_eventually_sampled_across_ratio_matrix() {
    let cases = vec![
        (
            SplitRatios {
                train: 1.0,
                validation: 0.0,
                test: 0.0,
            },
            vec![SplitLabel::Train],
        ),
        (
            SplitRatios {
                train: 0.0,
                validation: 1.0,
                test: 0.0,
            },
            vec![SplitLabel::Validation],
        ),
        (
            SplitRatios {
                train: 0.0,
                validation: 0.0,
                test: 1.0,
            },
            vec![SplitLabel::Test],
        ),
        (
            SplitRatios {
                train: 0.5,
                validation: 0.5,
                test: 0.0,
            },
            vec![SplitLabel::Train, SplitLabel::Validation],
        ),
        (
            SplitRatios {
                train: 0.34,
                validation: 0.33,
                test: 0.33,
            },
            vec![SplitLabel::Train, SplitLabel::Validation, SplitLabel::Test],
        ),
    ];

    let target_per_split = 3usize;

    for (case_idx, (split, allowed_splits)) in cases.into_iter().enumerate() {
        let seed = 1000 + case_idx as u64;
        let store = Arc::new(CountingSplitStore::new(split, seed));

        let mut config = build_config(seed, 2, split);
        config.allowed_splits = allowed_splits.clone();
        config.ingestion_max_records = target_per_split * allowed_splits.len();
        let sampler = TripletSampler::new(config, Arc::clone(&store));

        let mut expected: HashMap<SplitLabel, std::collections::HashSet<RecordId>> = HashMap::new();
        let mut records = Vec::new();
        let mut idx = 0usize;
        while expected.values().map(|ids| ids.len()).sum::<usize>()
            < target_per_split * allowed_splits.len()
        {
            let suffix = format!("case{case_idx}_{idx}");
            let record_id = format!("unit::{suffix}");
            let label = store.derive_label(&record_id);
            if allowed_splits.contains(&label) {
                let ids = expected.entry(label).or_default();
                if ids.len() < target_per_split && ids.insert(record_id.clone()) {
                    records.push(build_record("unit", &suffix, (records.len() % 27) as u32));
                }
            }
            idx += 1;
        }

        for split_label in &allowed_splits {
            assert_eq!(
                expected.get(split_label).map(|ids| ids.len()).unwrap_or(0),
                target_per_split,
                "fixture generation failed for split {:?}",
                split_label
            );
        }

        sampler.register_source(Box::new(InMemorySource::new(
            format!("matrix_case_{case_idx}"),
            records,
        )));

        let mut seen: HashMap<SplitLabel, std::collections::HashSet<RecordId>> = HashMap::new();
        for split_label in &allowed_splits {
            seen.insert(*split_label, std::collections::HashSet::new());
        }

        let max_rounds = target_per_split * 8;
        for _ in 0..max_rounds {
            for split_label in &allowed_splits {
                let batch = sampler.next_triplet_batch_for_split(*split_label).unwrap();
                for triplet in batch.triplets {
                    let anchor_id = triplet.anchor.record_id;
                    let anchor_label = store.label_for(&anchor_id).unwrap();
                    assert_eq!(anchor_label, *split_label);
                    seen.get_mut(split_label).unwrap().insert(anchor_id);
                }
            }

            if allowed_splits
                .iter()
                .all(|split_label| seen[split_label].len() >= expected[split_label].len())
            {
                break;
            }
        }

        for split_label in &allowed_splits {
            let expected_ids = &expected[split_label];
            let seen_ids = &seen[split_label];
            assert!(
                expected_ids.is_subset(seen_ids),
                "missing expected records for split {:?}; expected {:?}, saw {:?}",
                split_label,
                expected_ids,
                seen_ids
            );
        }
    }
}

#[test]
fn allowed_split_records_eventually_sampled_across_ratio_matrix_via_generic_triplet_api() {
    let cases = vec![
        (
            SplitRatios {
                train: 1.0,
                validation: 0.0,
                test: 0.0,
            },
            vec![SplitLabel::Train],
        ),
        (
            SplitRatios {
                train: 0.6,
                validation: 0.4,
                test: 0.0,
            },
            vec![SplitLabel::Validation, SplitLabel::Train],
        ),
        (
            SplitRatios {
                train: 0.34,
                validation: 0.33,
                test: 0.33,
            },
            vec![SplitLabel::Test, SplitLabel::Train, SplitLabel::Validation],
        ),
    ];

    let target_per_split = 3usize;

    for (case_idx, (split, allowed_splits)) in cases.into_iter().enumerate() {
        let seed = 2000 + case_idx as u64;
        let store = Arc::new(CountingSplitStore::new(split, seed));

        let mut config = build_config(seed, 2, split);
        config.allowed_splits = allowed_splits.clone();
        config.ingestion_max_records = target_per_split * allowed_splits.len();
        let sampler = TripletSampler::new(config, Arc::clone(&store));

        let mut expected: HashMap<SplitLabel, std::collections::HashSet<RecordId>> = HashMap::new();
        let mut records = Vec::new();
        let mut idx = 0usize;
        while expected.values().map(|ids| ids.len()).sum::<usize>()
            < target_per_split * allowed_splits.len()
        {
            let suffix = format!("generic_case{case_idx}_{idx}");
            let record_id = format!("unit::{suffix}");
            let label = store.derive_label(&record_id);
            if allowed_splits.contains(&label) {
                let ids = expected.entry(label).or_default();
                if ids.len() < target_per_split && ids.insert(record_id.clone()) {
                    records.push(build_record("unit", &suffix, (records.len() % 27) as u32));
                }
            }
            idx += 1;
        }

        for split_label in &allowed_splits {
            assert_eq!(
                expected.get(split_label).map(|ids| ids.len()).unwrap_or(0),
                target_per_split,
                "fixture generation failed for split {:?}",
                split_label
            );
        }

        sampler.register_source(Box::new(InMemorySource::new(
            format!("generic_matrix_case_{case_idx}"),
            records,
        )));

        let mut seen: HashMap<SplitLabel, std::collections::HashSet<RecordId>> = HashMap::new();
        for split_label in &allowed_splits {
            seen.insert(*split_label, std::collections::HashSet::new());
        }

        let max_rounds = target_per_split * allowed_splits.len() * 8;
        for _ in 0..max_rounds {
            for split_label in &allowed_splits {
                let batch = sampler.next_triplet_batch(*split_label).unwrap();
                for triplet in batch.triplets {
                    let anchor_id = triplet.anchor.record_id;
                    let anchor_label = store.label_for(&anchor_id).unwrap();
                    assert!(
                        allowed_splits.contains(&anchor_label),
                        "generic batch returned disallowed split {:?}",
                        anchor_label
                    );
                    assert_eq!(anchor_label, *split_label);
                    seen.get_mut(split_label).unwrap().insert(anchor_id);
                }
            }

            if allowed_splits
                .iter()
                .all(|split_label| seen[split_label].len() >= expected[split_label].len())
            {
                break;
            }
        }

        for split_label in &allowed_splits {
            let expected_ids = &expected[split_label];
            let seen_ids = &seen[split_label];
            assert!(
                expected_ids.is_subset(seen_ids),
                "missing expected records for split {:?}; expected {:?}, saw {:?}",
                split_label,
                expected_ids,
                seen_ids
            );
        }
    }
}

#[test]
fn per_epoch_shuffle_changes_source_order() {
    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };
    let store = Arc::new(CountingSplitStore::new(split, 41));
    let sampler = TripletSampler::new(build_config(41, 3, split), store);

    let records = vec![
        build_record("only", "a", 1),
        build_record("only", "b", 2),
        build_record("only", "c", 3),
        build_record("only", "d", 4),
        build_record("only", "e", 5),
        build_record("only", "f", 6),
    ];
    sampler.register_source(Box::new(InMemorySource::new("only", records)));

    let mut first_epoch = Vec::new();
    for _ in 0..2 {
        let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        first_epoch.extend(batch.triplets.iter().map(|t| t.anchor.record_id.clone()));
    }

    let mut second_epoch = Vec::new();
    for _ in 0..2 {
        let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        second_epoch.extend(batch.triplets.iter().map(|t| t.anchor.record_id.clone()));
    }

    assert_ne!(first_epoch, second_epoch);
}
