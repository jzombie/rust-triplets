use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use chrono::{Duration, TimeZone, Utc};

use triplets::source::InMemorySource;
use triplets::utils::make_section;
use triplets::{
    DataRecord, DeterministicSplitStore, NegativeStrategy, PairSampler, QualityScore, RecordId,
    Sampler, SamplerConfig, SectionRole, Selector, SplitLabel, SplitRatios, TripletRecipe,
};

fn build_record(source: &str, idx: usize) -> DataRecord {
    let created_at =
        Utc.with_ymd_and_hms(2025, 1, 1, 12, 0, 0).unwrap() + Duration::days(idx as i64);
    DataRecord {
        id: format!("{source}::record_{idx}"),
        source: source.to_string(),
        created_at,
        updated_at: created_at,
        quality: QualityScore { trust: 1.0 },
        taxonomy: vec![source.to_string()],
        sections: vec![
            make_section(SectionRole::Anchor, None, &format!("title {idx}")),
            make_section(
                SectionRole::Context,
                None,
                &format!("body content for record {idx}"),
            ),
        ],
        meta_prefix: None,
    }
}

fn assert_diversity_for_seed_and_split(seed: u64, allowed_split: SplitLabel) {
    let split = SplitRatios {
        train: 0.7,
        validation: 0.15,
        test: 0.15,
    };
    let mut config = SamplerConfig {
        seed,
        batch_size: 8,
        ingestion_max_records: 64,
        ..SamplerConfig::default()
    };
    let ingestion_max_records = config.ingestion_max_records;
    config.allowed_splits = vec![allowed_split];
    config.split = split;
    config.recipes = vec![TripletRecipe {
        name: "diversity_triplet".into(),
        anchor: Selector::Role(SectionRole::Anchor),
        positive_selector: Selector::Role(SectionRole::Context),
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
    }];
    config.text_recipes = Vec::new();

    let records: Vec<DataRecord> = (0..300).map(|idx| build_record("unit", idx)).collect();
    let store = Arc::new(DeterministicSplitStore::new(split, 77).unwrap());
    let sampler = PairSampler::new(config, store);
    sampler.register_source(Box::new(InMemorySource::new("unit", records)));

    let mut anchor_counts: HashMap<RecordId, usize> = HashMap::new();
    let mut negative_counts: HashMap<RecordId, usize> = HashMap::new();
    let mut total_triplets = 0usize;

    for _ in 0..30 {
        let batch = sampler.next_triplet_batch(allowed_split).unwrap();
        for triplet in batch.triplets {
            *anchor_counts.entry(triplet.anchor.record_id).or_insert(0) += 1;
            *negative_counts
                .entry(triplet.negative.record_id)
                .or_insert(0) += 1;
            total_triplets += 1;
        }
    }

    // "Enough samples" should expose broad coverage, not collapse onto a tiny subset.
    assert!(
        anchor_counts.len() >= 30,
        "expected many unique anchors, got {}",
        anchor_counts.len()
    );
    if total_triplets >= ingestion_max_records {
        let mut windows = HashSet::new();
        for id in anchor_counts.keys() {
            if let Some(idx) = id.rsplit('_').next().and_then(|s| s.parse::<usize>().ok()) {
                windows.insert(idx / ingestion_max_records);
            }
        }
        assert!(
            windows.len() >= 2,
            "expected anchors to span multiple ingestion windows, got {}",
            windows.len()
        );
    }
    assert!(
        negative_counts.len() >= 24,
        "expected many unique negatives, got {}",
        negative_counts.len()
    );

    let max_negative_share = negative_counts
        .values()
        .copied()
        .max()
        .map(|count| count as f32 / total_triplets as f32)
        .unwrap_or(1.0);
    assert!(
        max_negative_share <= 0.15,
        "seed {} collapsed: max negative share {:.3} over {} triplets",
        seed,
        max_negative_share,
        total_triplets
    );
}

#[test]
fn triplet_sampling_stays_diverse_over_time() {
    assert_diversity_for_seed_and_split(12345, SplitLabel::Train);
    assert_diversity_for_seed_and_split(12345, SplitLabel::Validation);
    assert_diversity_for_seed_and_split(12345, SplitLabel::Test);
}

#[test]
fn triplet_sampling_stays_diverse_across_seeds() {
    // Run the same diversity checks across multiple deterministic seeds.
    for seed in [3_u64, 7, 11, 42, 29, 97] {
        for split in [SplitLabel::Train, SplitLabel::Validation, SplitLabel::Test] {
            assert_diversity_for_seed_and_split(seed, split);
        }
    }
}
