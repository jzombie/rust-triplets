use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::Arc;

use triplets::source::InMemorySource;
use triplets::source::indexing::file_corpus::FileCorpusIndex;
use triplets::splits::{FileSplitStore, SplitRatios, SplitStore};
use triplets::utils::make_section;
use triplets::{
    DataRecord, NegativeStrategy, TripletSampler, QualityScore, RecordId, Sampler, SamplerConfig,
    SectionRole, Selector, SourceId, SplitLabel, TripletRecipe,
};

fn write_qa_file(dir: &std::path::Path, name: &str, answer: &str) {
    let path = dir.join(name);
    fs::write(path, answer.as_bytes()).unwrap();
}

fn build_qa_record(
    root: &Path,
    source_id: &SourceId,
    path: &Path,
) -> Result<Option<DataRecord>, triplets::SamplerError> {
    let body = fs::read_to_string(path)?;
    let title = FileCorpusIndex::normalized_title_from_stem(path, source_id, true)?;
    let now = chrono::Utc::now();
    Ok(Some(DataRecord {
        id: FileCorpusIndex::source_scoped_record_id(source_id, root, path),
        source: source_id.clone(),
        created_at: now,
        updated_at: now,
        quality: QualityScore { trust: 1.0 },
        taxonomy: vec![source_id.clone()],
        sections: vec![
            make_section(SectionRole::Anchor, None, &title),
            make_section(SectionRole::Context, None, &body),
        ],
        meta_prefix: None,
    }))
}

fn ids_from_root(root: &Path, source_id: &SourceId) -> Vec<RecordId> {
    let index = FileCorpusIndex::new(root, source_id.clone()).with_sampler_seed(123);
    let snapshot = index
        .refresh_indexable(None, None, |path| build_qa_record(root, source_id, path))
        .unwrap();
    snapshot
        .records
        .into_iter()
        .map(|record| record.id)
        .collect()
}

fn build_record(source: &str, idx: usize) -> DataRecord {
    let created_at = chrono::Utc::now();
    DataRecord {
        id: format!("{source}::{idx}"),
        source: source.to_string(),
        created_at,
        updated_at: created_at,
        quality: QualityScore { trust: 1.0 },
        taxonomy: vec![source.to_string()],
        sections: vec![
            make_section(SectionRole::Anchor, None, &format!("{source} title {idx}")),
            make_section(SectionRole::Context, None, &format!("{source} body {idx}")),
        ],
        meta_prefix: None,
    }
}

fn build_config(batch_size: usize, split: SplitRatios) -> SamplerConfig {
    SamplerConfig {
        seed: 123,
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

#[test]
fn file_based_split_assignments_remain_stable_across_growth() {
    // Arrange: create initial QA files in a temp root.
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    write_qa_file(
        root,
        "What_is_alpha.txt",
        "Alpha measures risk-adjusted outperformance.",
    );
    write_qa_file(
        root,
        "What_is_beta.txt",
        "Beta measures sensitivity to the market.",
    );
    write_qa_file(
        root,
        "What_is_gamma.txt",
        "Gamma measures convexity of options.",
    );
    write_qa_file(
        root,
        "What_is_delta.txt",
        "Delta measures option price sensitivity.",
    );
    write_qa_file(root, "What_is_theta.txt", "Theta measures time decay.");
    write_qa_file(
        root,
        "What_is_vega.txt",
        "Vega measures volatility sensitivity.",
    );

    // Split ratios and store location for deterministic assignments.
    let split = SplitRatios {
        train: 0.5,
        validation: 0.25,
        test: 0.25,
    };
    let store_path = root.join("split_store.bin");
    let source_id: SourceId = "source_b".into();

    let (initial_ids, labels_initial) = {
        // First pass: index files and assign splits.
        let initial_ids = ids_from_root(root, &source_id);
        assert_eq!(initial_ids.len(), 6);

        let store = FileSplitStore::open(&store_path, split, 123).unwrap();
        let mut labels_initial = HashMap::new();
        for id in &initial_ids {
            labels_initial.insert(id.clone(), store.ensure(id.clone()).unwrap());
        }
        assert_eq!(labels_initial.len(), initial_ids.len());

        // Sanity-check that the initial IDs are unique.
        let mut union_initial = HashSet::new();
        for id in &initial_ids {
            union_initial.insert(id.clone());
        }
        assert_eq!(union_initial.len(), initial_ids.len());

        (initial_ids, labels_initial)
    };

    // Add new files after the initial split assignment.
    write_qa_file(
        root,
        "What_is_rho.txt",
        "Rho measures interest rate sensitivity.",
    );
    write_qa_file(root, "What_is_phi.txt", "Phi is not a common option Greek.");
    // Force the per-process file index to rebuild so the new files are visible.
    let index_path = FileCorpusIndex::file_index_store_path(root, &source_id);
    let _ = fs::remove_file(index_path);

    {
        // Second pass: re-index/store and ensure split stability.
        let all_ids = ids_from_root(root, &source_id);
        assert_eq!(all_ids.len(), 8);

        let store = FileSplitStore::open(&store_path, split, 123).unwrap();
        let mut labels_after = HashMap::new();
        for id in &all_ids {
            labels_after.insert(id.clone(), store.ensure(id.clone()).unwrap());
        }
        assert_eq!(labels_after.len(), all_ids.len());

        // Original files must retain their original split labels.
        for id in &initial_ids {
            assert_eq!(
                labels_after.get(id).copied().unwrap(),
                labels_initial.get(id).copied().unwrap()
            );
        }

        // New files must exist and be assigned to a split.
        let new_ids: Vec<RecordId> = all_ids
            .iter()
            .filter(|id| !labels_initial.contains_key(*id))
            .cloned()
            .collect();
        assert_eq!(new_ids.len(), 2);
        for id in &new_ids {
            assert!(labels_after.contains_key(id));
        }

        // Double-check: original IDs still map to the same splits.
        for id in &initial_ids {
            assert_eq!(
                labels_after.get(id).copied().unwrap(),
                labels_initial.get(id).copied().unwrap()
            );
        }
    }
}

#[test]
fn split_store_growth_stays_bounded_per_epoch() {
    // Baseline: measured on current store format, per-epoch growth is <= 256 bytes.
    // This guards against reintroducing per-record split writes.
    let temp = tempfile::tempdir().unwrap();
    let store_path = temp.path().join("split_store.bin");
    let split = SplitRatios::default();

    let records: Vec<DataRecord> = (0..64).map(|idx| build_record("unit", idx)).collect();
    let store = Arc::new(FileSplitStore::open(&store_path, split, 123).unwrap());
    let sampler = TripletSampler::new(build_config(8, split), store);
    sampler.register_source(Box::new(InMemorySource::new("unit", records)));

    let mut sizes = Vec::new();
    for _ in 0..5 {
        let _ = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        sampler.persist_state().unwrap();
        let size = fs::metadata(&store_path).unwrap().len();
        sizes.push(size);
    }

    let deltas: Vec<u64> = sizes
        .windows(2)
        .map(|pair| pair[1].saturating_sub(pair[0]))
        .collect();
    if deltas.is_empty() {
        return;
    }
    let max_delta = 256_u64;
    for delta in deltas {
        assert!(
            delta <= max_delta,
            "split store growth exceeded baseline: max={} delta={}",
            max_delta,
            delta
        );
    }
}
