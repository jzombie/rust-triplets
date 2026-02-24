use std::sync::Arc;

use chrono::{TimeZone, Utc};

use triplets::source::InMemorySource;
use triplets::utils::make_section;
use triplets::{
    DataRecord, FileSplitStore, NegativeStrategy, TripletSampler, QualityScore, RecordId, Sampler,
    SamplerConfig, SectionRole, Selector, SplitLabel, SplitRatios, TripletRecipe,
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
    sampler.persist_state().unwrap();
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
        sampler.persist_state().unwrap();
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
        sampler.persist_state().unwrap();
        first_after_restart
            .triplets
            .iter()
            .map(|triplet| triplet.negative.record_id.clone())
            .collect::<Vec<_>>()
    };

    assert_ne!(first_run_negatives, restart_negatives);
}
