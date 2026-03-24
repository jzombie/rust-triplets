use std::sync::Arc;

use chrono::{TimeZone, Utc};
use simd_r_drive::storage_engine::DataStore;
use simd_r_drive::storage_engine::traits::DataStoreWriter;

use triplets::source::InMemorySource;
use triplets::splits::{EpochStateStore, PersistedSamplerState, SamplerStateStore};
use triplets::utils::make_section;
use triplets::{
    DataRecord, FileSplitStore, NegativeStrategy, QualityScore, RecordId, Sampler, SamplerConfig,
    SectionRole, Selector, SplitLabel, SplitRatios, SplitStore, TripletRecipe, TripletSampler,
};

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

fn build_config(batch_size: usize, split: SplitRatios) -> SamplerConfig {
    SamplerConfig {
        seed: 999,
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
            allow_same_anchor_positive: false,
        }],
        text_recipes: Vec::new(),
        ..SamplerConfig::default()
    }
}

fn first_record_ids(store_path: &std::path::Path, batch_size: usize) -> Vec<RecordId> {
    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };
    let store = Arc::new(FileSplitStore::open(store_path, split, 73).unwrap());
    let sampler = TripletSampler::new(build_config(batch_size, split), store);

    let source_a = vec![
        build_record("source_a", "a1", 1),
        build_record("source_a", "a2", 2),
        build_record("source_a", "a3", 3),
    ];
    let source_b = vec![
        build_record("source_b", "b1", 1),
        build_record("source_b", "b2", 2),
        build_record("source_b", "b3", 3),
    ];

    sampler.register_source(Box::new(InMemorySource::new("source_a", source_a)));
    sampler.register_source(Box::new(InMemorySource::new("source_b", source_b)));

    let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
    sampler.save_sampler_state(None).unwrap();
    batch
        .triplets
        .iter()
        .map(|triplet| triplet.anchor.record_id.clone())
        .collect()
}

#[test]
fn shuffled_continues_across_runs_with_same_batch_size() {
    let temp = tempfile::tempdir().unwrap();
    let store_path = temp.path().join("rr_persist_store.bin");

    let first_run = first_record_ids(&store_path, 4);
    let second_run = first_record_ids(&store_path, 4);

    assert_ne!(first_run, second_run);

    let sources_first: Vec<&str> = first_run
        .iter()
        .map(|id| id.split("::").next().unwrap_or(""))
        .collect();
    let sources_second: Vec<&str> = second_run
        .iter()
        .map(|id| id.split("::").next().unwrap_or(""))
        .collect();

    assert_eq!(
        sources_first,
        vec!["source_a", "source_b", "source_a", "source_b"]
    );
    assert_eq!(
        sources_second,
        vec!["source_a", "source_b", "source_a", "source_b"]
    );
}

#[test]
fn negatives_persist_across_restart() {
    let temp = tempfile::tempdir().unwrap();
    let store_path = temp.path().join("rr_negative_persist.bin");
    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };

    let source_a = vec![
        build_record("source_a", "a1", 1),
        build_record("source_a", "a2", 2),
        build_record("source_a", "a3", 3),
        build_record("source_a", "a4", 4),
    ];
    let source_b = vec![
        build_record("source_b", "b1", 1),
        build_record("source_b", "b2", 2),
        build_record("source_b", "b3", 3),
        build_record("source_b", "b4", 4),
    ];

    let first_run_negatives = {
        let store = Arc::new(FileSplitStore::open(&store_path, split, 73).unwrap());
        let sampler = TripletSampler::new(build_config(4, split), store);
        sampler.register_source(Box::new(InMemorySource::new("source_a", source_a.clone())));
        sampler.register_source(Box::new(InMemorySource::new("source_b", source_b.clone())));

        let first = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        sampler.save_sampler_state(None).unwrap();
        first
            .triplets
            .iter()
            .map(|triplet| triplet.negative.record_id.clone())
            .collect::<Vec<_>>()
    };

    let restart_negatives = {
        let store = Arc::new(FileSplitStore::open(&store_path, split, 73).unwrap());
        let sampler = TripletSampler::new(build_config(4, split), store);
        sampler.register_source(Box::new(InMemorySource::new("source_a", source_a)));
        sampler.register_source(Box::new(InMemorySource::new("source_b", source_b)));

        let first_after_restart = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        sampler.save_sampler_state(None).unwrap();
        first_after_restart
            .triplets
            .iter()
            .map(|triplet| triplet.negative.record_id.clone())
            .collect::<Vec<_>>()
    };

    assert_ne!(first_run_negatives, restart_negatives);
}

/// `save(None)` must write to the sampler's own `save_path` only.  A
/// separately-published store at a different path must remain unchanged.
#[test]
fn save_none_writes_to_save_path_only() {
    let temp = tempfile::tempdir().unwrap();
    let store_path = temp.path().join("sampler_store.bin");
    let other_path = temp.path().join("other_store.bin");
    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };

    // Pre-publish a sentinel state to other_path so we can verify it is untouched
    // after the sampler writes to its own store_path.
    let sentinel = PersistedSamplerState {
        source_cycle_idx: 0xDEAD,
        source_record_cursors: vec![],
        source_epoch: 0xBEEF,
        rng_state: 0,
        triplet_recipe_rr_idx: 0,
        text_recipe_rr_idx: 0,
        source_stream_cursors: vec![],
    };
    FileSplitStore::open(&other_path, split, 73)
        .unwrap()
        .save_sampler_state(&sentinel, None)
        .unwrap();

    let store = Arc::new(FileSplitStore::open(&store_path, split, 73).unwrap());
    let sampler = TripletSampler::new(build_config(4, split), store);
    sampler.register_source(Box::new(InMemorySource::new(
        "source_a",
        vec![
            build_record("source_a", "a1", 1),
            build_record("source_a", "a2", 2),
            build_record("source_a", "a3", 3),
            build_record("source_a", "a4", 4),
        ],
    )));
    sampler.register_source(Box::new(InMemorySource::new(
        "source_b",
        vec![
            build_record("source_b", "b1", 1),
            build_record("source_b", "b2", 2),
            build_record("source_b", "b3", 3),
            build_record("source_b", "b4", 4),
        ],
    )));

    sampler.next_triplet_batch(SplitLabel::Train).unwrap();
    sampler.save_sampler_state(None).unwrap();

    // store_path must have the new sampler state.
    assert!(
        FileSplitStore::open(&store_path, split, 73)
            .unwrap()
            .load_sampler_state()
            .unwrap()
            .is_some(),
        "save(None) must publish state to save_path"
    );

    // other_path must still hold only the sentinel — save(None) must not touch it.
    let other_state = FileSplitStore::open(&other_path, split, 73)
        .unwrap()
        .load_sampler_state()
        .unwrap()
        .unwrap();
    assert_eq!(
        other_state.source_cycle_idx, 0xDEAD,
        "save(None) must not modify other store paths"
    );
    assert_eq!(
        other_state.source_epoch, 0xBEEF,
        "save(None) must not modify other store paths"
    );
}

#[test]
fn save_sampler_state_some_mirrors_to_new_store_path() {
    let temp = tempfile::tempdir().unwrap();
    let source_store_path = temp.path().join("source_store.bin");
    let mirror_store_path = temp.path().join("mirror_store.bin");
    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };

    let store = Arc::new(FileSplitStore::open(&source_store_path, split, 73).unwrap());
    let sampler = TripletSampler::new(build_config(4, split), store);
    sampler.register_source(Box::new(InMemorySource::new(
        "source_a",
        vec![
            build_record("source_a", "a1", 1),
            build_record("source_a", "a2", 2),
            build_record("source_a", "a3", 3),
            build_record("source_a", "a4", 4),
        ],
    )));
    sampler.register_source(Box::new(InMemorySource::new(
        "source_b",
        vec![
            build_record("source_b", "b1", 1),
            build_record("source_b", "b2", 2),
            build_record("source_b", "b3", 3),
            build_record("source_b", "b4", 4),
        ],
    )));

    sampler.next_triplet_batch(SplitLabel::Train).unwrap();
    // Publish current state to the canonical source store path, then snapshot
    // to the mirror.  Both calls see the same in-memory state so the resulting
    // files must be equivalent.
    sampler.save_sampler_state(None).unwrap();
    sampler
        .save_sampler_state(Some(mirror_store_path.as_path()))
        .unwrap();

    let source_store = FileSplitStore::open(&source_store_path, split, 73).unwrap();
    let mirror_store = FileSplitStore::open(&mirror_store_path, split, 73).unwrap();

    let source_state = source_store.load_sampler_state().unwrap().unwrap();
    let mirror_state = mirror_store.load_sampler_state().unwrap().unwrap();

    assert_eq!(mirror_state.source_cycle_idx, source_state.source_cycle_idx);
    assert_eq!(
        mirror_state.source_record_cursors,
        source_state.source_record_cursors
    );
    assert_eq!(mirror_state.source_epoch, source_state.source_epoch);
    assert_eq!(mirror_state.rng_state, source_state.rng_state);
    assert_eq!(
        mirror_state.triplet_recipe_rr_idx,
        source_state.triplet_recipe_rr_idx
    );
    assert_eq!(
        mirror_state.text_recipe_rr_idx,
        source_state.text_recipe_rr_idx
    );
    assert_eq!(
        mirror_state.source_stream_cursors,
        source_state.source_stream_cursors
    );

    let source_meta = source_store.load_epoch_meta().unwrap();
    let mirror_meta = mirror_store.load_epoch_meta().unwrap();
    assert_eq!(mirror_meta.len(), source_meta.len());
    for (label, expected) in source_meta {
        let actual = mirror_meta.get(&label).unwrap();
        assert_eq!(actual.epoch, expected.epoch);
        assert_eq!(actual.offset, expected.offset);
        assert_eq!(actual.hashes_checksum, expected.hashes_checksum);
    }
}

#[test]
fn save_sampler_state_some_preserves_split_assignments_in_mirror_store() {
    let temp = tempfile::tempdir().unwrap();
    let source_store_path = temp.path().join("source_with_assignments.bin");
    let mirror_store_path = temp.path().join("mirror_with_assignments.bin");
    let split = SplitRatios {
        train: 0.8,
        validation: 0.1,
        test: 0.1,
    };

    let probe_store = FileSplitStore::open(&source_store_path, split, 73).unwrap();

    let mut forced_id = None;
    for idx in 0..10_000 {
        let candidate = format!("forced_record_{idx}");
        if probe_store.label_for(&candidate) != Some(SplitLabel::Validation) {
            forced_id = Some(candidate);
            break;
        }
    }
    let forced_id = forced_id.expect("should find id whose derived label is not validation");

    let mut split_key = b"split:".to_vec();
    split_key.extend_from_slice(forced_id.as_bytes());
    let datastore = DataStore::open(&source_store_path).unwrap();
    datastore.write(&split_key, b"1").unwrap();

    let source_store = FileSplitStore::open(&source_store_path, split, 73).unwrap();

    assert_eq!(
        source_store.label_for(&forced_id),
        Some(SplitLabel::Validation)
    );

    let state = PersistedSamplerState {
        source_cycle_idx: 1,
        source_record_cursors: vec![("source_a".to_string(), 2)],
        source_epoch: 3,
        rng_state: 4,
        triplet_recipe_rr_idx: 5,
        text_recipe_rr_idx: 6,
        source_stream_cursors: vec![("source_a".to_string(), 7)],
    };

    source_store
        .save_sampler_state(&state, Some(mirror_store_path.as_path()))
        .unwrap();

    let mirror_store = FileSplitStore::open(&mirror_store_path, split, 73).unwrap();
    assert_eq!(
        mirror_store.label_for(&forced_id),
        Some(SplitLabel::Validation)
    );
}

#[test]
fn save_sampler_state_some_errors_if_destination_store_exists() {
    let temp = tempfile::tempdir().unwrap();
    let source_store_path = temp.path().join("source_store.bin");
    let existing_dest_path = temp.path().join("already_exists.bin");
    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };

    let source_store = FileSplitStore::open(&source_store_path, split, 73).unwrap();
    let existing_dest_store = FileSplitStore::open(&existing_dest_path, split, 73).unwrap();
    // Publish so the file exists on disk before we try to save over it.
    let dummy_state = PersistedSamplerState {
        source_cycle_idx: 0,
        source_record_cursors: vec![],
        source_epoch: 0,
        rng_state: 0,
        triplet_recipe_rr_idx: 0,
        text_recipe_rr_idx: 0,
        source_stream_cursors: vec![],
    };
    existing_dest_store
        .save_sampler_state(&dummy_state, None)
        .unwrap();
    drop(existing_dest_store);

    let state = PersistedSamplerState {
        source_cycle_idx: 1,
        source_record_cursors: vec![("source_a".to_string(), 2)],
        source_epoch: 3,
        rng_state: 4,
        triplet_recipe_rr_idx: 5,
        text_recipe_rr_idx: 6,
        source_stream_cursors: vec![("source_a".to_string(), 7)],
    };

    let err = source_store
        .save_sampler_state(&state, Some(existing_dest_path.as_path()))
        .unwrap_err();
    assert!(matches!(
        err,
        triplets::SamplerError::SplitStore(msg) if msg.contains("refusing to overwrite existing split store")
    ));
}

#[test]
fn save_sampler_state_some_creates_missing_parent_directories() {
    let temp = tempfile::tempdir().unwrap();
    let source_store_path = temp.path().join("source_store.bin");
    let deep_mirror_path = temp
        .path()
        .join("deep")
        .join("nested")
        .join("mirror")
        .join("state")
        .join("mirror_store.bin");
    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };

    assert!(!deep_mirror_path.exists());
    assert!(!deep_mirror_path.parent().unwrap().exists());

    let store = Arc::new(FileSplitStore::open(&source_store_path, split, 73).unwrap());
    let sampler = TripletSampler::new(build_config(4, split), store);
    sampler.register_source(Box::new(InMemorySource::new(
        "source_a",
        vec![
            build_record("source_a", "a1", 1),
            build_record("source_a", "a2", 2),
            build_record("source_a", "a3", 3),
            build_record("source_a", "a4", 4),
        ],
    )));
    sampler.register_source(Box::new(InMemorySource::new(
        "source_b",
        vec![
            build_record("source_b", "b1", 1),
            build_record("source_b", "b2", 2),
            build_record("source_b", "b3", 3),
            build_record("source_b", "b4", 4),
        ],
    )));

    sampler.next_triplet_batch(SplitLabel::Train).unwrap();
    sampler
        .save_sampler_state(Some(deep_mirror_path.as_path()))
        .unwrap();

    assert!(deep_mirror_path.parent().unwrap().exists());
    assert!(deep_mirror_path.exists());

    let mirror_store = FileSplitStore::open(&deep_mirror_path, split, 73).unwrap();
    assert!(mirror_store.load_sampler_state().unwrap().is_some());
}

#[test]
/// Loading from an existing store via `open_with_load_path` and then saving with `None`
/// must write to the declared `save_path` only.  The original `load_path` file must
/// remain byte-identical (never mutated).
fn open_with_load_path_and_save_none_writes_only_to_declared_save_path() {
    let temp = tempfile::tempdir().unwrap();
    let load_path = temp.path().join("original.bin");
    let save_path = temp.path().join("new.bin");
    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };

    // Create the source store with state.
    {
        let store = Arc::new(FileSplitStore::open(&load_path, split, 73).unwrap());
        let sampler = TripletSampler::new(build_config(4, split), store);
        sampler.register_source(Box::new(InMemorySource::new(
            "source_a",
            vec![
                build_record("source_a", "a1", 1),
                build_record("source_a", "a2", 2),
                build_record("source_a", "a3", 3),
                build_record("source_a", "a4", 4),
            ],
        )));
        sampler.register_source(Box::new(InMemorySource::new(
            "source_b",
            vec![
                build_record("source_b", "b1", 1),
                build_record("source_b", "b2", 2),
                build_record("source_b", "b3", 3),
                build_record("source_b", "b4", 4),
            ],
        )));
        sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        sampler.save_sampler_state(None).unwrap();
    }

    let load_path_size_before = std::fs::metadata(&load_path).unwrap().len();
    assert!(!save_path.exists());

    // Bootstrap from load_path into save_path and perform a save(None).
    {
        let store = Arc::new(
            FileSplitStore::open_with_load_path(Some(&load_path), &save_path, split, 73).unwrap(),
        );
        let sampler = TripletSampler::new(build_config(4, split), store);
        sampler.register_source(Box::new(InMemorySource::new(
            "source_a",
            vec![
                build_record("source_a", "a1", 1),
                build_record("source_a", "a2", 2),
                build_record("source_a", "a3", 3),
                build_record("source_a", "a4", 4),
            ],
        )));
        sampler.register_source(Box::new(InMemorySource::new(
            "source_b",
            vec![
                build_record("source_b", "b1", 1),
                build_record("source_b", "b2", 2),
                build_record("source_b", "b3", 3),
                build_record("source_b", "b4", 4),
            ],
        )));
        sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        // save(None) should go to save_path only.
        sampler.save_sampler_state(None).unwrap();
    }

    // original must not have changed.
    let load_path_size_after = std::fs::metadata(&load_path).unwrap().len();
    assert_eq!(
        load_path_size_before, load_path_size_after,
        "load_path was modified despite bootstrapped open"
    );

    // The new save_path must hold valid state.
    assert!(save_path.exists());
    let saved = FileSplitStore::open(&save_path, split, 73).unwrap();
    assert!(
        saved.load_sampler_state().unwrap().is_some(),
        "save_path should contain sampler state after save(None)"
    );
}

#[test]
/// Loading from an existing store via `open_with_load_path` and then saving with
/// `Some(other)` must write to `other` only.  Neither the original `load_path` nor
/// the declared `save_path` should be created or modified.
fn open_with_load_path_and_save_some_writes_only_to_explicit_path() {
    let temp = tempfile::tempdir().unwrap();
    let load_path = temp.path().join("original.bin");
    let save_path = temp.path().join("new.bin"); // canonical — must stay absent
    let other_path = temp.path().join("other.bin"); // explicit — receives the state
    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };

    // Create the source store with state.
    {
        let store = Arc::new(FileSplitStore::open(&load_path, split, 73).unwrap());
        let sampler = TripletSampler::new(build_config(4, split), store);
        sampler.register_source(Box::new(InMemorySource::new(
            "source_a",
            vec![
                build_record("source_a", "a1", 1),
                build_record("source_a", "a2", 2),
                build_record("source_a", "a3", 3),
                build_record("source_a", "a4", 4),
            ],
        )));
        sampler.register_source(Box::new(InMemorySource::new(
            "source_b",
            vec![
                build_record("source_b", "b1", 1),
                build_record("source_b", "b2", 2),
                build_record("source_b", "b3", 3),
                build_record("source_b", "b4", 4),
            ],
        )));
        sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        sampler.save_sampler_state(None).unwrap();
    }

    let load_path_size_before = std::fs::metadata(&load_path).unwrap().len();

    // Bootstrap and save to explicit other_path.
    {
        let store = Arc::new(
            FileSplitStore::open_with_load_path(Some(&load_path), &save_path, split, 73).unwrap(),
        );
        let sampler = TripletSampler::new(build_config(4, split), store);
        sampler.register_source(Box::new(InMemorySource::new(
            "source_a",
            vec![
                build_record("source_a", "a1", 1),
                build_record("source_a", "a2", 2),
                build_record("source_a", "a3", 3),
                build_record("source_a", "a4", 4),
            ],
        )));
        sampler.register_source(Box::new(InMemorySource::new(
            "source_b",
            vec![
                build_record("source_b", "b1", 1),
                build_record("source_b", "b2", 2),
                build_record("source_b", "b3", 3),
                build_record("source_b", "b4", 4),
            ],
        )));
        sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        sampler
            .save_sampler_state(Some(other_path.as_path()))
            .unwrap();
    }

    // original must not have changed.
    let load_path_size_after = std::fs::metadata(&load_path).unwrap().len();
    assert_eq!(
        load_path_size_before, load_path_size_after,
        "load_path was modified despite bootstrapped open"
    );

    // canonical save_path must not have been created.
    assert!(
        !save_path.exists(),
        "save_path must not be created when using save(Some(other))"
    );

    // The explicit other_path must hold valid state.
    assert!(other_path.exists());
    let saved = FileSplitStore::open(&other_path, split, 73).unwrap();
    assert!(
        saved.load_sampler_state().unwrap().is_some(),
        "other_path should contain sampler state after save(Some(other))"
    );
}
