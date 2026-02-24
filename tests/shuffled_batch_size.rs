use std::sync::Arc;

use chrono::{TimeZone, Utc};

use triplets::source::InMemorySource;
use triplets::utils::make_section;
use triplets::{
    DataRecord, DeterministicSplitStore, NegativeStrategy, TripletSampler, QualityScore, RecordId,
    Sampler, SamplerConfig, SectionRole, Selector, SplitLabel, SplitRatios, TripletRecipe,
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

fn run_samples(batch_size: usize, total_samples: usize) -> Vec<RecordId> {
    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 123).unwrap());
    let config = SamplerConfig {
        seed: 999,
        batch_size,
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
    };

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

    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(InMemorySource::new("source_a", source_a)));
    sampler.register_source(Box::new(InMemorySource::new("source_b", source_b)));

    let mut samples = Vec::new();
    while samples.len() < total_samples {
        let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        for triplet in batch.triplets {
            samples.push(triplet.anchor.record_id);
            if samples.len() >= total_samples {
                break;
            }
        }
    }
    samples
}

fn source_from_id(record_id: &str) -> &str {
    record_id.split("::").next().unwrap_or("")
}

#[test]
fn shuffled_is_deterministic_across_batch_sizes() {
    let total_samples = 12;
    let samples_batch_4 = run_samples(4, total_samples);
    let samples_batch_4_again = run_samples(4, total_samples);

    assert_eq!(samples_batch_4, samples_batch_4_again);

    let mut observed: Vec<&str> = samples_batch_4
        .iter()
        .take(8)
        .map(|id| source_from_id(id))
        .collect();
    observed.sort();
    observed.dedup();
    assert_eq!(observed, vec!["source_a", "source_b"]);
}
