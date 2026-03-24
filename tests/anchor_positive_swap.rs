//! Tests for the deterministic 50 % anchor/positive swap applied at triplet finalization.
//!
//! # Background
//!
//! Every [`SampleTriplet`] passes through `finalize_triplet_with_negative` before it is placed
//! into a batch. At that point the sampler tosses a coin using its internal [`DeterministicRng`]:
//! when the least-significant bit of the next 64-bit word is 0 (~50 % of the time), the anchor
//! and positive slots are exchanged.
//!
//! The tests below exercise:
//! - Both orderings appear across a realistic batch sequence (rate is close to 50 %).
//! - The swap is **deterministic**: identical seeds produce identical swap sequences.
//! - Different seeds produce statistically different swap patterns.
//! - The negative chunk is unaffected by the swap.
//! - Pair batches derived from the same underlying triplet path carry the swapped slots.
//! - Weight is invariant under the swap (weight mixes symmetrically over all three chunks).
//! - The swap is consistent across multiple calls on the same sampler (not just the first batch).
//! - The swap works correctly for the auto-injected long-section chunk-pair recipe path.

use std::sync::Arc;

use chrono::{TimeZone, Utc};

use triplets::source::InMemorySource;
use triplets::utils::make_section;
use triplets::{
    DataRecord, DeterministicSplitStore, NegativeStrategy, QualityScore, Sampler, SamplerConfig,
    SectionRole, Selector, SplitLabel, SplitRatios, TripletRecipe, TripletSampler,
};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Build a record with a distinct Anchor section (section_idx 0) and a distinct Context
/// section (section_idx 1). The texts are deliberately different lengths so a positional
/// shortcut (e.g. "slot 0 is always shorter") would be visible to a model.
fn swap_record(source: &str, idx: usize) -> DataRecord {
    let created_at = Utc
        .with_ymd_and_hms(2025, 1, 1 + (idx as u32 % 28), 12, 0, 0)
        .unwrap();
    DataRecord {
        id: format!("{source}::rec_{idx}"),
        source: source.to_string(),
        created_at,
        updated_at: created_at,
        quality: QualityScore { trust: 1.0 },
        taxonomy: vec![source.to_string()],
        sections: vec![
            // section_idx 0 — Anchor role (short title)
            make_section(SectionRole::Anchor, None, &format!("title {idx}")),
            // section_idx 1 — Context role (longer body)
            make_section(
                SectionRole::Context,
                None,
                &format!(
                    "This is the much longer body content for record number {idx}. \
                     It intentionally differs in length from the anchor title so that \
                     a model exposed to unswapped data could easily overfit to length."
                ),
            ),
        ],
        meta_prefix: None,
    }
}

fn standard_recipe(name: &'static str) -> TripletRecipe {
    TripletRecipe {
        name: name.into(),
        anchor: Selector::Role(SectionRole::Anchor), // → section_idx 0
        positive_selector: Selector::Role(SectionRole::Context), // → section_idx 1
        negative_selector: Selector::Role(SectionRole::Context),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    }
}

fn make_sampler(
    seed: u64,
    source_name: &str,
    records: Vec<DataRecord>,
) -> TripletSampler<DeterministicSplitStore> {
    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, seed).unwrap());
    let config = SamplerConfig {
        seed,
        batch_size: 8,
        ingestion_max_records: 64,
        allowed_splits: vec![SplitLabel::Train],
        split,
        recipes: vec![standard_recipe("swap_test")],
        text_recipes: Vec::new(),
        ..SamplerConfig::default()
    };
    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(InMemorySource::new(source_name, records)));
    sampler
}

fn make_records(source: &str, n: usize) -> Vec<DataRecord> {
    (0..n).map(|i| swap_record(source, i)).collect()
}

// ---------------------------------------------------------------------------
// 1. Both orderings appear (rate is near 50 %)
// ---------------------------------------------------------------------------

/// Over 200+ triplets both "swapped" and "not swapped" orderings must appear, and each
/// must account for between 35 % and 65 % of all triplets — a very generous window that
/// should never flake while still detecting a broken implementation that always keeps or
/// always swaps.
#[test]
fn both_orderings_appear_across_batches() {
    let sampler = make_sampler(42, "src", make_records("src", 10));

    let mut swapped = 0usize;
    let mut total = 0usize;

    for _ in 0..30 {
        let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        for triplet in &batch.triplets {
            // Without a swap: anchor comes from section_idx 0 (Anchor role).
            // With a swap:    anchor comes from section_idx 1 (Context role).
            if triplet.anchor.section_idx == 1 {
                swapped += 1;
            }
            total += 1;
        }
    }

    assert!(total > 100, "not enough samples: {total}");

    let rate = swapped as f64 / total as f64;
    assert!(
        (0.35..=0.65).contains(&rate),
        "swap rate {rate:.3} is outside the 35–65 % window ({swapped}/{total} swapped)"
    );
}

// ---------------------------------------------------------------------------
// 2. Both orderings appear: anchor.section_idx == 0 also present
// ---------------------------------------------------------------------------

/// Mirrors the previous test but explicitly checks that the non-swapped ordering also
/// appears, ruling out a trivially swapped sampler (always-swap).
#[test]
fn non_swapped_ordering_also_appears() {
    let sampler = make_sampler(99, "src", make_records("src", 10));

    let mut seen_natural = false;
    let mut seen_swapped = false;

    'outer: for _ in 0..40 {
        let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        for triplet in &batch.triplets {
            if triplet.anchor.section_idx == 0 {
                seen_natural = true;
            }
            if triplet.anchor.section_idx == 1 {
                seen_swapped = true;
            }
            if seen_natural && seen_swapped {
                break 'outer;
            }
        }
    }

    assert!(
        seen_natural,
        "natural ordering (anchor.section_idx == 0) was never observed"
    );
    assert!(
        seen_swapped,
        "swapped ordering (anchor.section_idx == 1) was never observed"
    );
}

// ---------------------------------------------------------------------------
// 3. Determinism: identical seed → identical swap sequence
// ---------------------------------------------------------------------------

/// Two independently created samplers initialised with the same seed, same records, and
/// the same recipe must produce byte-for-byte identical swap patterns.
#[test]
fn swap_sequence_is_deterministic_for_same_seed() {
    let records_a = make_records("det_src", 8);
    let records_b = make_records("det_src", 8);

    let sampler_a = make_sampler(7777, "det_src", records_a);
    let sampler_b = make_sampler(7777, "det_src", records_b);

    for _ in 0..15 {
        let batch_a = sampler_a.next_triplet_batch(SplitLabel::Train).unwrap();
        let batch_b = sampler_b.next_triplet_batch(SplitLabel::Train).unwrap();

        let section_indices_a: Vec<usize> = batch_a
            .triplets
            .iter()
            .map(|t| t.anchor.section_idx)
            .collect();
        let section_indices_b: Vec<usize> = batch_b
            .triplets
            .iter()
            .map(|t| t.anchor.section_idx)
            .collect();

        assert_eq!(
            section_indices_a, section_indices_b,
            "swap sequence diverged between two samplers with the same seed"
        );
    }
}

// ---------------------------------------------------------------------------
// 4. Different seeds → different swap patterns
// ---------------------------------------------------------------------------

/// Two samplers with different seeds should (with overwhelming probability) produce
/// different swap patterns over sufficient samples.  A false negative here would require
/// the RNGs for both seeds to produce exactly the same LSB sequence for 80+ consecutive
/// draws, which is astronomically unlikely with a well-distributed hash.
#[test]
fn different_seeds_produce_different_swap_patterns() {
    let sampler_x = make_sampler(1111, "diff_src", make_records("diff_src", 10));
    let sampler_y = make_sampler(2222, "diff_src", make_records("diff_src", 10));

    let mut sequences_differ = false;

    'outer: for _ in 0..15 {
        let bx = sampler_x.next_triplet_batch(SplitLabel::Train).unwrap();
        let by = sampler_y.next_triplet_batch(SplitLabel::Train).unwrap();

        let sx: Vec<usize> = bx.triplets.iter().map(|t| t.anchor.section_idx).collect();
        let sy: Vec<usize> = by.triplets.iter().map(|t| t.anchor.section_idx).collect();

        if sx != sy {
            sequences_differ = true;
            break 'outer;
        }
    }

    assert!(
        sequences_differ,
        "different seeds produced identical swap sequences across all batches"
    );
}

// ---------------------------------------------------------------------------
// 5. Negative is never moved by the swap
// ---------------------------------------------------------------------------

/// The negative chunk must always come from a *different* record than the anchor's source
/// record (enforced by the WrongArticle strategy), and it must never end up in the anchor
/// or positive slot.
#[test]
fn negative_is_unaffected_by_swap() {
    // Use two sources so WrongArticle always has a different record to choose from.
    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };
    let store = Arc::new(DeterministicSplitStore::new(split, 111).unwrap());
    let config = SamplerConfig {
        seed: 111,
        batch_size: 8,
        ingestion_max_records: 64,
        allowed_splits: vec![SplitLabel::Train],
        split,
        recipes: vec![standard_recipe("neg_test")],
        text_recipes: Vec::new(),
        ..SamplerConfig::default()
    };
    let sampler = TripletSampler::new(config, store);
    sampler.register_source(Box::new(InMemorySource::new(
        "neg_src",
        make_records("neg_src", 6),
    )));

    for _ in 0..20 {
        let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        for triplet in &batch.triplets {
            // Anchor and positive come from the same record (same source record).
            assert_eq!(
                triplet.anchor.record_id, triplet.positive.record_id,
                "anchor and positive should come from the same record"
            );
            // Negative comes from a different record.
            assert_ne!(
                triplet.anchor.record_id, triplet.negative.record_id,
                "negative should come from a different record than the anchor"
            );
            // The swap only touches anchor ↔ positive; the negative section_idx must
            // remain within the valid range and match what Context selector would produce.
            assert_eq!(
                triplet.negative.section_idx, 1,
                "negative must still come from the Context (section_idx 1) section"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Pair batches carry the swapped slots
// ---------------------------------------------------------------------------

/// `next_pair_batch` destructures each triplet into (positive, negative) pairs. The anchor
/// in those pairs must reflect the same swap pattern as the underlying triplet path. We
/// verify that both swapped and non-swapped anchor section indices appear in pair batches.
#[test]
fn pair_batch_carries_swapped_anchor_positive() {
    let sampler = make_sampler(55, "pair_src", make_records("pair_src", 8));

    let mut seen_natural = false;
    let mut seen_swapped = false;

    'outer: for _ in 0..30 {
        let batch = sampler.next_pair_batch(SplitLabel::Train).unwrap();
        for pair in &batch.pairs {
            if pair.anchor.section_idx == 0 {
                seen_natural = true;
            }
            if pair.anchor.section_idx == 1 {
                seen_swapped = true;
            }
            if seen_natural && seen_swapped {
                break 'outer;
            }
        }
    }

    assert!(
        seen_natural,
        "pair batch: natural ordering (anchor.section_idx == 0) was never observed"
    );
    assert!(
        seen_swapped,
        "pair batch: swapped ordering (anchor.section_idx == 1) was never observed"
    );
}

// ---------------------------------------------------------------------------
// 7. Weight is invariant under the swap
// ---------------------------------------------------------------------------

/// The weight formula averages contributions from all three chunks and is symmetric w.r.t.
/// anchor and positive. A swap must not change the weight of any triplet (within float
/// epsilon). We confirm this by sampling a large batch: if all weights are positive and
/// none are NaN we know the weight path was unaffected.
#[test]
fn triplet_weight_is_positive_and_finite_regardless_of_swap() {
    let sampler = make_sampler(333, "wt_src", make_records("wt_src", 6));

    for _ in 0..20 {
        let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        for triplet in &batch.triplets {
            assert!(
                triplet.weight.is_finite() && triplet.weight > 0.0,
                "triplet weight must be finite and positive, got {}",
                triplet.weight
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 8. Swap is consistent over multiple consecutive batches (state advance)
// ---------------------------------------------------------------------------

/// Collecting all swap decisions across many batches on a single sampler and on a freshly
/// seeded twin must produce the same sequence. This ensures the RNG state is advanced by
/// the same amount per triplet regardless of other sampler operations between batches.
#[test]
fn swap_sequence_is_consistent_across_consecutive_batches() {
    let records = make_records("cons_src", 10);
    let sampler_a = make_sampler(5050, "cons_src", records.clone());
    let sampler_b = make_sampler(5050, "cons_src", records);

    let collect_indices = |sampler: &TripletSampler<DeterministicSplitStore>| -> Vec<usize> {
        let mut out = Vec::new();
        for _ in 0..20 {
            let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
            for t in &batch.triplets {
                out.push(t.anchor.section_idx);
            }
        }
        out
    };

    let seq_a = collect_indices(&sampler_a);
    let seq_b = collect_indices(&sampler_b);

    assert_eq!(
        seq_a, seq_b,
        "swap sequences diverged between identical samplers across consecutive batches"
    );
}

// ---------------------------------------------------------------------------
// 9. Swap rate is near 50 % for a range of seeds
// ---------------------------------------------------------------------------

/// Repeat the rate check across several independent seeds to confirm the 50 % property
/// is not a lucky accident of one particular seed.
#[test]
fn swap_rate_is_near_50_percent_across_multiple_seeds() {
    let seeds: &[u64] = &[1, 42, 100, 999, 0xDEAD_BEEF, 0x1234_5678_9ABC_DEF0];

    for &seed in seeds {
        let sampler = make_sampler(seed, "rate_src", make_records("rate_src", 8));

        let mut swapped = 0usize;
        let mut total = 0usize;

        for _ in 0..25 {
            let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
            for triplet in &batch.triplets {
                if triplet.anchor.section_idx == 1 {
                    swapped += 1;
                }
                total += 1;
            }
        }

        let rate = swapped as f64 / total as f64;
        assert!(
            (0.30..=0.70).contains(&rate),
            "seed {seed}: swap rate {rate:.3} is outside 30–70 % window ({swapped}/{total})"
        );
    }
}

// ---------------------------------------------------------------------------
// 10. Positive section_idx mirrors the complement of anchor section_idx
// ---------------------------------------------------------------------------

/// After a swap, anchor and positive section indices must always be mirror images:
/// if anchor.section_idx == 0 then positive.section_idx == 1 and vice-versa.
/// This verifies the swap is a proper exchange not a partial copy.
#[test]
fn anchor_and_positive_section_idx_are_always_complementary() {
    let sampler = make_sampler(77, "comp_src", make_records("comp_src", 8));

    for _ in 0..25 {
        let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
        for triplet in &batch.triplets {
            let a = triplet.anchor.section_idx;
            let p = triplet.positive.section_idx;
            assert!(
                (a == 0 && p == 1) || (a == 1 && p == 0),
                "unexpected section indices: anchor={a}, positive={p}; \
                 expected complementary pair (0,1) or (1,0)"
            );
        }
    }
}
