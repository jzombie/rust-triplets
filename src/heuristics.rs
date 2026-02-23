use crate::config::{SamplerConfig, TextRecipe, TripletRecipe};
pub use crate::constants::heuristics::{
    EFFECTIVE_NEGATIVES_PER_ANCHOR, EFFECTIVE_POSITIVES_PER_ANCHOR,
};
use crate::splits::{SplitLabel, SplitRatios};

#[derive(Debug, Default, Clone, Copy)]
pub struct CapacityTotals {
    pub triplets: u128,
    pub effective_triplets: u128,
    pub pairs: u128,
    pub text_samples: u128,
}

pub fn estimate_source_split_capacity_from_counts(
    source_records_in_split: u128,
    triplet_recipes: &[TripletRecipe],
    text_recipes: &[TextRecipe],
) -> CapacityTotals {
    let mut totals = CapacityTotals::default();

    for _recipe in triplet_recipes {
        let anchor_positive_pairs = source_records_in_split;
        let negative_count_per_anchor = source_records_in_split.saturating_sub(1);
        if anchor_positive_pairs == 0 || negative_count_per_anchor == 0 {
            continue;
        }
        totals.triplets += anchor_positive_pairs.saturating_mul(negative_count_per_anchor);
        totals.effective_triplets += source_records_in_split
            .saturating_mul(EFFECTIVE_POSITIVES_PER_ANCHOR)
            .saturating_mul(EFFECTIVE_NEGATIVES_PER_ANCHOR);
        totals.pairs += anchor_positive_pairs.saturating_mul(1 + negative_count_per_anchor);
    }

    for _recipe in text_recipes {
        totals.text_samples += source_records_in_split;
    }

    totals
}

pub fn split_counts_for_total(total: u128, split: SplitRatios) -> [(SplitLabel, u128); 3] {
    let train = ((total as f64) * f64::from(split.train)).floor() as u128;
    let validation = ((total as f64) * f64::from(split.validation)).floor() as u128;
    let assigned = train.saturating_add(validation);
    let test = total.saturating_sub(assigned);
    [
        (SplitLabel::Train, train),
        (SplitLabel::Validation, validation),
        (SplitLabel::Test, test),
    ]
}

pub fn format_u128_with_commas(value: u128) -> String {
    let raw = value.to_string();
    let mut grouped_reversed = String::with_capacity(raw.len() + (raw.len() / 3));
    for (idx, ch) in raw.chars().rev().enumerate() {
        if idx > 0 && idx % 3 == 0 {
            grouped_reversed.push(',');
        }
        grouped_reversed.push(ch);
    }
    grouped_reversed.chars().rev().collect()
}

pub fn format_replay_factor(longest_records: u128, source_records: u128) -> String {
    if longest_records == 0 || source_records == 0 {
        return "n/a".to_string();
    }
    let factor = longest_records as f64 / source_records as f64;
    format!("{factor:.2}x")
}

pub fn resolve_text_recipes_for_source(
    config: &SamplerConfig,
    source_triplet_recipes: &[TripletRecipe],
) -> Vec<TextRecipe> {
    if !config.text_recipes.is_empty() {
        return config.text_recipes.clone();
    }
    if !config.recipes.is_empty() {
        return build_derived_text_recipes(&config.recipes);
    }
    build_derived_text_recipes(source_triplet_recipes)
}

pub fn build_derived_text_recipes(recipes: &[TripletRecipe]) -> Vec<TextRecipe> {
    let mut derived = Vec::new();
    for recipe in recipes {
        let base = recipe.name.as_ref();
        derived.push(TextRecipe {
            name: format!("{base}_anchor").into(),
            selector: recipe.anchor.clone(),
            weight: recipe.weight.max(0.0001),
            instruction: None,
        });
        derived.push(TextRecipe {
            name: format!("{base}_positive").into(),
            selector: recipe.positive_selector.clone(),
            weight: recipe.weight.max(0.0001),
            instruction: None,
        });
        derived.push(TextRecipe {
            name: format!("{base}_negative").into(),
            selector: recipe.negative_selector.clone(),
            weight: recipe.weight.max(0.0001),
            instruction: None,
        });
    }
    derived
}
