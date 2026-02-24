use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{TimeZone, Utc};

use triplets::source::InMemorySource;
use triplets::{
    DataRecord, DeterministicSplitStore, NegativeStrategy, QualityScore, SamplerConfig,
    SectionRole, Selector, SplitLabel, SplitRatios, TripletRecipe, TripletSampler,
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
            triplets::utils::make_section(
                SectionRole::Anchor,
                None,
                &format!("{source} title {suffix}"),
            ),
            triplets::utils::make_section(
                SectionRole::Context,
                None,
                &format!("{source} body {suffix}"),
            ),
        ],
        meta_prefix: None,
    }
}

#[test]
fn prefetcher_yields_triplet_batches() {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.2,
        test: 0.1,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 77).unwrap());
    let config = SamplerConfig {
        seed: 42,
        batch_size: 2,
        allowed_splits: vec![SplitLabel::Train],
        split,
        recipes: vec![TripletRecipe {
            name: "prefetch_triplet".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
        }],
        ..SamplerConfig::default()
    };

    let records_a = vec![
        build_record("source_a", "a1", 1),
        build_record("source_a", "a2", 2),
        build_record("source_a", "a3", 3),
        build_record("source_a", "a4", 4),
    ];
    let records_b = vec![
        build_record("source_b", "b1", 1),
        build_record("source_b", "b2", 2),
        build_record("source_b", "b3", 3),
        build_record("source_b", "b4", 4),
    ];

    let sampler = Arc::new(TripletSampler::new(config, store));
    sampler.register_source(Box::new(InMemorySource::new("source_a", records_a)));
    sampler.register_source(Box::new(InMemorySource::new("source_b", records_b)));

    let prefetcher = Arc::clone(&sampler).prefetch_triplet_batches(SplitLabel::Train, 2);
    let start = Instant::now();
    while prefetcher.produced_count() < 2 {
        if start.elapsed() > Duration::from_millis(200) {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let first = prefetcher.next().unwrap();
    let second = prefetcher.next().unwrap();

    assert_eq!(first.triplets.len(), 2);
    assert_eq!(second.triplets.len(), 2);
    assert!(prefetcher.produced_count() >= 2);
    assert!(prefetcher.queue_len() <= 2);
    assert_eq!(prefetcher.error_count(), 0);
}
