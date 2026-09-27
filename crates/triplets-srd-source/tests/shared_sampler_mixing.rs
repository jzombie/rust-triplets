//! Proves the embedding channel end to end: every sampled chunk carries the
//! exact vector stored in its SRD entry — no side lookups, no attribution
//! plumbing. Embeddings ride the pipeline; consumers read them straight off
//! the sampled pairs.
//!
//! Also proves native source weighting: one shared sampler over two stores
//! with weights {0.75, 0.25} apportions anchors exactly 3:1 via deficit
//! round-robin (deterministic, seed-independent choice order).

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use simd_r_drive::storage_engine::DataStore;
use tempfile::TempDir;
use triplets::{DeterministicSplitStore, SamplerConfig, SplitLabel, SplitRatios, TripletSampler};
use triplets_srd_source::{
    PairLabel, SrdMode, SrdPairWriteEntry, SrdSource, SrdTripletWriteEntry, batch_read_entries,
    write_pair_entries, write_triplet_entries,
};

const DIM: usize = 8;
const N_ENTRIES: usize = 12;

/// Ground truth for one written store: anchor/candidate vectors + texts.
type StoreTruth = (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<String>, Vec<String>);

/// Write one pair-mode SRD store. Anchor/candidate texts AND vectors are
/// distinct per entry (the sampler rejects identical anchor/positive pairs),
/// and globally unique per dataset via `base`.
fn write_store(path: &Path, dataset: &str, base: f32) -> StoreTruth {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let store = DataStore::open(path).unwrap();
    let anchor_vecs: Vec<Vec<f32>> = (0..N_ENTRIES)
        .map(|i| vec![base + i as f32 * 0.01 + 0.1; DIM])
        .collect();
    let cand_vecs: Vec<Vec<f32>> = (0..N_ENTRIES)
        .map(|i| vec![base + i as f32 * 0.01 + 100.1; DIM])
        .collect();
    let anchor_texts: Vec<String> = (0..N_ENTRIES)
        .map(|i| format!("{dataset} anchor text {i}"))
        .collect();
    let cand_texts: Vec<String> = (0..N_ENTRIES)
        .map(|i| format!("{dataset} candidate text {i}"))
        .collect();
    let entries: Vec<SrdPairWriteEntry> = anchor_vecs
        .iter()
        .zip(anchor_texts.iter())
        .zip(cand_vecs.iter().zip(cand_texts.iter()))
        .map(|((av, at), (cv, ct))| SrdPairWriteEntry {
            anchor_vec: av,
            anchor_text: at,
            candidate_vec: cv,
            candidate_text: ct,
            label: &PairLabel::Positive,
        })
        .collect();
    write_pair_entries(&store, 0, &entries).unwrap();
    (anchor_vecs, cand_vecs, anchor_texts, cand_texts)
}

fn make_sampler(seed: u64, batch_size: usize) -> TripletSampler<DeterministicSplitStore> {
    let ratios = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };
    TripletSampler::new(
        SamplerConfig {
            seed,
            batch_size,
            allowed_splits: vec![SplitLabel::Train],
            split: ratios,
            ..Default::default()
        },
        Arc::new(DeterministicSplitStore::new(ratios, seed).unwrap()),
    )
}

#[test]
fn shared_sampler_apportions_by_weight_with_embeddings_on_chunks() {
    let tmp = TempDir::new().unwrap();
    let a_path = tmp.path().join("ds-a").join("data.srd");
    let b_path = tmp.path().join("ds-b").join("data.srd");
    let (a_anchor, a_cand, a_anchor_texts, a_cand_texts) = write_store(&a_path, "a", 0.0);
    let (b_anchor, b_cand, b_anchor_texts, b_cand_texts) = write_store(&b_path, "b", 1000.0);

    let sampler = make_sampler(7, 8);
    sampler
        .register_source(Box::new(
            SrdSource::open(&a_path, "ds-a", DIM, Some(SrdMode::Pair)).unwrap(),
        ))
        .unwrap();
    sampler
        .register_source(Box::new(
            SrdSource::open(&b_path, "ds-b", DIM, Some(SrdMode::Pair)).unwrap(),
        ))
        .unwrap();
    let weights: HashMap<String, f32> =
        [("ds-a".to_string(), 0.75), ("ds-b".to_string(), 0.25)].into();

    // Draw enough pairs for the uniform cycle to show; every chunk must
    // carry the exact store vector for its text (anchor/positive slots swap
    // ~50%, so attribute by text<->vector consistency, not by slot).
    let mut a_count = 0usize;
    let mut total = 0usize;
    for _ in 0..10 {
        let batch = sampler
            .next_pair_batch_with_weights_for_split(SplitLabel::Train, &weights)
            .unwrap();
        assert_eq!(batch.pairs.len(), 8);
        for p in &batch.pairs {
            for chunk in [&p.anchor, &p.positive] {
                total += 1;
                let emb: &[f32] = chunk
                    .embedding
                    .as_ref()
                    .expect("sampled chunk carries embedding");
                let from_a = a_anchor_texts
                    .iter()
                    .position(|t| t == &chunk.text)
                    .map(|i| a_anchor[i].as_slice() == emb)
                    .or_else(|| {
                        a_cand_texts
                            .iter()
                            .position(|t| t == &chunk.text)
                            .map(|i| a_cand[i].as_slice() == emb)
                    });
                let from_b = b_anchor_texts
                    .iter()
                    .position(|t| t == &chunk.text)
                    .map(|i| b_anchor[i].as_slice() == emb)
                    .or_else(|| {
                        b_cand_texts
                            .iter()
                            .position(|t| t == &chunk.text)
                            .map(|i| b_cand[i].as_slice() == emb)
                    });
                assert!(
                    from_a.unwrap_or(false) ^ from_b.unwrap_or(false),
                    "chunk text+vector match exactly one dataset entry, got {:?}",
                    chunk.text,
                );
                if from_a.unwrap_or(false) {
                    a_count += 1;
                }
            }
        }
    }
    // Anchor picks follow deficit round-robin exactly 3:1 (4 draws per
    // 8-row batch → 3 A + 1 B anchors → 6 A + 2 B rows; both chunks of a
    // pair belong to the anchor's dataset). Ten batches = 40 draws = 10
    // full DRR cycles → exactly 60/20.
    let share = a_count as f32 / total as f32;
    assert_eq!(
        (a_count, total),
        (120, 160),
        "DRR apportions exactly 3:1, ds-a chunk share = {share:.2}"
    );
}

#[test]
fn chunk_embeddings_match_store_ground_truth() {
    // Belt and braces: chunk vectors equal a fresh store read, entry by entry.
    let tmp = TempDir::new().unwrap();
    let a_path = tmp.path().join("ds-a").join("data.srd");
    let (a_anchor, a_cand, _, _) = write_store(&a_path, "a", 0.0);

    let sampler = make_sampler(3, 4);
    sampler
        .register_source(Box::new(
            SrdSource::open(&a_path, "ds-a", DIM, Some(SrdMode::Pair)).unwrap(),
        ))
        .unwrap();
    let batch = sampler
        .next_pair_batch_with_weights_for_split(SplitLabel::Train, &HashMap::new())
        .unwrap();
    assert!(!batch.pairs.is_empty());

    let store = DataStore::open_existing(&a_path).unwrap();
    for p in &batch.pairs {
        // Both slots verified: `section_idx` selects the entry side, the
        // bare entry index selects the entry.
        for chunk in [&p.anchor, &p.positive] {
            let idx: usize = chunk.record_id.parse().unwrap();
            let entries = batch_read_entries(&store, &[idx], DIM).unwrap();
            assert_eq!(entries.len(), 1);
            let emb: &[f32] = chunk.embedding.as_ref().expect("embedding present");
            let (expected_text, expected_vec) = if chunk.section_idx == 0 {
                (format!("a anchor text {idx}"), a_anchor[idx].as_slice())
            } else {
                assert_eq!(chunk.section_idx, 1);
                (format!("a candidate text {idx}"), a_cand[idx].as_slice())
            };
            assert_eq!(chunk.text, expected_text);
            assert_eq!(emb, expected_vec);
        }
    }
}

#[test]
fn same_seed_same_batches() {
    let tmp = TempDir::new().unwrap();
    let a_path = tmp.path().join("ds-a").join("data.srd");
    let b_path = tmp.path().join("ds-b").join("data.srd");
    write_store(&a_path, "a", 0.0);
    write_store(&b_path, "b", 1000.0);

    let build = || {
        let s = make_sampler(42, 8);
        s.register_source(Box::new(
            SrdSource::open(&a_path, "ds-a", DIM, Some(SrdMode::Pair)).unwrap(),
        ))
        .unwrap();
        s.register_source(Box::new(
            SrdSource::open(&b_path, "ds-b", DIM, Some(SrdMode::Pair)).unwrap(),
        ))
        .unwrap();
        s
    };
    let weights: HashMap<String, f32> =
        [("ds-a".to_string(), 0.5), ("ds-b".to_string(), 0.5)].into();
    let s1 = build();
    let s2 = build();
    for _ in 0..3 {
        let b1 = s1
            .next_pair_batch_with_weights_for_split(SplitLabel::Train, &weights)
            .unwrap();
        let b2 = s2
            .next_pair_batch_with_weights_for_split(SplitLabel::Train, &weights)
            .unwrap();
        let ids1: Vec<_> = b1.pairs.iter().map(|p| &p.anchor.record_id).collect();
        let ids2: Vec<_> = b2.pairs.iter().map(|p| &p.anchor.record_id).collect();
        assert_eq!(ids1, ids2);
        for (p1, p2) in b1.pairs.iter().zip(b2.pairs.iter()) {
            assert_eq!(
                p1.anchor.embedding.as_deref(),
                p2.anchor.embedding.as_deref()
            );
        }
    }
}

/// Write one triplet-mode SRD store with dataset-tagged texts on all three
/// slots (anchor slot may hold any of the three chunks after swaps).
fn write_triplet_store(path: &Path, dataset: &str, base: f32) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let store = DataStore::open(path).unwrap();
    let vecs = |b: f32| -> Vec<Vec<f32>> {
        (0..N_ENTRIES)
            .map(|i| vec![base + b + i as f32 * 0.01 + 0.1; DIM])
            .collect()
    };
    let texts = |tag: &str| -> Vec<String> {
        (0..N_ENTRIES)
            .map(|i| format!("{dataset} {tag} {i}"))
            .collect()
    };
    let (av, pv, nv) = (vecs(0.0), vecs(100.0), vecs(200.0));
    let (at, pt, nt) = (texts("t-anchor"), texts("t-pos"), texts("t-neg"));
    let entries: Vec<SrdTripletWriteEntry> = av
        .iter()
        .zip(at.iter())
        .zip(pv.iter().zip(pt.iter()))
        .zip(nv.iter().zip(nt.iter()))
        .map(
            |(((aav, aat), (ppv, ppt)), (nnv, nnt))| SrdTripletWriteEntry {
                anchor_vec: aav,
                anchor_text: aat,
                pos_vec: ppv,
                pos_text: ppt,
                neg_vec: nnv,
                neg_text: nnt,
            },
        )
        .collect();
    write_triplet_entries(&store, 0, &entries).unwrap();
}

#[test]
fn triplet_batches_apportion_anchors_by_weight() {
    let tmp = TempDir::new().unwrap();
    let a_path = tmp.path().join("ds-a").join("data.srd");
    let b_path = tmp.path().join("ds-b").join("data.srd");
    write_triplet_store(&a_path, "a", 0.0);
    write_triplet_store(&b_path, "b", 1000.0);

    let sampler = make_sampler(7, 8);
    sampler
        .register_source(Box::new(
            SrdSource::open(&a_path, "ds-a", DIM, Some(SrdMode::Triplet)).unwrap(),
        ))
        .unwrap();
    sampler
        .register_source(Box::new(
            SrdSource::open(&b_path, "ds-b", DIM, Some(SrdMode::Triplet)).unwrap(),
        ))
        .unwrap();
    let weights: HashMap<String, f32> =
        [("ds-a".to_string(), 0.75), ("ds-b".to_string(), 0.25)].into();

    // Anchor slots carry their dataset tag regardless of swaps (all three
    // slots tagged); count anchors per dataset over ten batches.
    let mut a_anchors = 0usize;
    let mut total = 0usize;
    for _ in 0..10 {
        let batch = sampler
            .next_triplet_batch_with_weights_for_split(SplitLabel::Train, &weights)
            .unwrap();
        assert_eq!(batch.triplets.len(), 8);
        for t in &batch.triplets {
            total += 1;
            assert!(
                t.anchor.embedding.is_some(),
                "anchor chunk carries embedding"
            );
            if t.anchor.text.starts_with("a ") {
                a_anchors += 1;
            } else {
                assert!(
                    t.anchor.text.starts_with("b "),
                    "unexpected anchor text {:?}",
                    t.anchor.text
                );
            }
        }
    }
    // 61/80 measured: DRR draws apportion exactly, but Phase 2/3 filtering
    // plus pad_with_reuse (duplicate fill to batch_size) adds +-1 noise.
    // The band pins weighted apportionment without overfitting the noise.
    let share = a_anchors as f32 / total as f32;
    assert!(
        (0.60..0.90).contains(&share),
        "DRR apportions ~3:1 anchors, ds-a share = {share:.2} ({a_anchors}/{total})"
    );
}
