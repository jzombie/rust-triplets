use crate::data::SectionRole;
use crate::splits::{SplitLabel, SplitRatios};
use std::borrow::Cow;

/// Controls how long text sections are chunked and weighted.
#[derive(Clone, Debug)]
pub struct ChunkingStrategy {
    /// Max tokens per window when slicing a section into chunks.
    pub max_window_tokens: usize,
    /// Overlap sizes (in tokens) used when sliding windows across a section.
    pub overlap_tokens: Vec<usize>,
    /// Weight assigned to summary-fallback chunks (when generated).
    pub summary_fallback_weight: f32,
    /// Max tokens for summary-fallback chunks (0 disables fallback chunks).
    pub summary_fallback_tokens: usize,
    /// Floor applied to per-chunk weight after offset or summary fallback weighting.
    pub chunk_weight_floor: f32,
}

impl Default for ChunkingStrategy {
    fn default() -> Self {
        Self {
            max_window_tokens: 1024,
            overlap_tokens: vec![64],
            summary_fallback_weight: 0.35,
            summary_fallback_tokens: 512,
            chunk_weight_floor: 0.1,
        }
    }
}

/// Defines a triplet recipe (anchor/positive/negative selection + weighting).
///
/// ## Split-isolation contract
///
/// All three chunk slots (anchor, positive, negative) must resolve to records
/// whose IDs hash to the same split as the request split. The sampler enforces
/// this automatically for `Selector::Role`, `Selector::Paragraph`, and
/// `Selector::Random` — those selectors always read from the record that was
/// already confirmed to be in the correct split.
///
/// `Selector::TemporalOffset` crosses a record boundary (it picks a *different*
/// record by proximity in time) and the split check is re-applied inside
/// `select_temporal_neighbor`. No additional care is required on your side,
/// but you should be aware that in pools with few same-split neighbors the
/// selector will return `None` and fall back to skipping a slot rather than
/// contaminating splits.
///
/// ## Stable IDs
///
/// Record IDs must be stable across runs. Split assignment is derived
/// deterministically from the record ID and the sampler seed; changing an ID
/// changes its split assignment, which invalidates any persisted split state.
/// IDs should also be globally unique — if two records from different sources
/// share the same ID, only one will be kept in the sampler, and the discarded
/// record's split assignment silently goes with it.
#[derive(Clone, Debug)]
pub struct TripletRecipe {
    /// Unique name for this recipe.
    pub name: Cow<'static, str>,
    /// Selector used for anchor chunks.
    pub anchor: Selector,
    /// Selector used for positive chunks (same record).
    pub positive_selector: Selector,
    /// Selector used for negative chunks (different record).
    pub negative_selector: Selector,
    /// Strategy used to pick negatives.
    pub negative_strategy: NegativeStrategy,
    /// Relative weight used when sampling among recipes.
    pub weight: f32,
    /// Optional instruction text attached to samples from this recipe.
    pub instruction: Option<Cow<'static, str>>,
    /// Allow anchor and positive to carry identical text (SimCSE / dropout-trick mode).
    ///
    /// When `true`, the sampler will emit triplets even when the anchor and positive
    /// sections resolve to the same text.  This enables the unsupervised SimCSE
    /// training pattern: the same text string feeds both slots, and the model's
    /// dropout layers produce two slightly different embeddings at training time.
    ///
    /// Negatives are still required to differ from both anchor and positive.
    ///
    /// Defaults to `false`; set `true` only for recipes whose anchor and positive
    /// selectors intentionally resolve to the same content (e.g. text-only sources).
    pub allow_same_anchor_positive: bool,
}

impl Default for TripletRecipe {
    fn default() -> Self {
        Self {
            name: "".into(),
            anchor: Selector::Random,
            positive_selector: Selector::Random,
            negative_selector: Selector::Random,
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
            allow_same_anchor_positive: false,
        }
    }
}

/// Selector for choosing a section or neighboring record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Selector {
    /// Select a section by role.
    Role(SectionRole),
    /// Select a specific section by index.
    Paragraph(usize),
    /// Select a temporal neighbor record by offset days.
    ///
    /// Candidates are restricted to the same split as the requesting record;
    /// if no same-split neighbor exists the selector returns `None` for that
    /// slot rather than crossing split boundaries.
    TemporalOffset(i32),
    /// Select a random section.
    Random,
}

/// Defines how to build a text sample from a record.
#[derive(Clone, Debug)]
pub struct TextRecipe {
    /// Unique name for this recipe.
    pub name: Cow<'static, str>,
    /// Selector used for text chunks.
    pub selector: Selector,
    /// Relative weight used when sampling among text recipes.
    pub weight: f32,
    /// Optional instruction text attached to samples from this recipe.
    pub instruction: Option<Cow<'static, str>>,
}

/// Strategy for picking the negative *record* in a triplet.
///
/// Each variant defines the candidate pool from which the negative record is drawn.
/// By default all variants scope candidates to the same source as the anchor, so
/// negatives are hard relative to the source domain rather than trivially
/// cross-domain. A same-split fallback engages automatically when the in-source
/// pool is too small (for example a source with only one record in the split).
///
/// When the `bm25-mining` feature is enabled, BM25 lexical re-ranking is applied
/// on top of the strategy-filtered pool — BM25 re-orders candidates by keyword
/// overlap with the anchor, but does not widen or replace the strategy pool.
/// BM25 is a first-pass lexical ranker, not a semantic one; it is well-suited for
/// lifting average negative quality without an encoder at data-generation time.
/// Semantic or embedding-based re-ranking (iterative hard-negative mining with the
/// trained encoder, cross-encoder scoring, dense retrieval) is out of scope for
/// the data pipeline and can be integrated by pre-ranking negatives before
/// ingestion or by reweighting source batches in the training loop.
#[derive(Clone, Debug)]
pub enum NegativeStrategy {
    /// Choose a record with a different publication date from record metadata.
    ///
    /// This refers to metadata/taxonomy publication-date values (for example
    /// `META_FIELD_DATE`), not filesystem timestamps like mtime/ctime/atime.
    WrongPublicationDate,
    /// Choose a different record from the same source.
    ///
    /// Negatives are drawn from within the same source, making them hard relative
    /// to the source domain. This is appropriate when each source represents a
    /// coherent domain (a collection of finance articles, a physics paper set,
    /// etc.) where same-source records are already confusable. If your sources are
    /// not meaningful domain boundaries, the fallback path (same split, any source)
    /// is the relevant escape hatch.
    WrongArticle,
    /// Choose a mismatched Q/A pair.
    QuestionAnswerMismatch,
}

/// Top-level sampler configuration.
#[derive(Clone, Debug)]
pub struct SamplerConfig {
    /// RNG seed that controls deterministic sampling order.
    pub seed: u64,
    /// Target number of samples per batch.
    pub batch_size: usize,
    /// Max number of records kept in the ingestion cache for candidate sampling.
    ///
    /// This is intentionally decoupled from `batch_size` so anchors/negatives can
    /// be drawn from a broader rolling pool.
    ///
    /// Practical tuning: values above `batch_size` usually improve diversity and
    /// reduce short-horizon repetition; gains taper off as source/recipe/split
    /// constraints become the limiting factor. Higher values also increase memory.
    ///
    /// For remote shard-backed sources (for example Hugging Face), larger initial
    /// targets may require fetching more shards before the first batch, so startup
    /// latency can increase based on shard sizes and network throughput.
    pub ingestion_max_records: usize,
    /// Chunking behavior for long sections.
    pub chunking: ChunkingStrategy,
    /// Triplet recipes to use; empty means sources may provide defaults.
    pub recipes: Vec<TripletRecipe>,
    /// Text recipes to use; empty means derived from triplet recipes if available.
    pub text_recipes: Vec<TextRecipe>,
    /// Split ratios used when assigning records to train/val/test.
    pub split: SplitRatios,
    /// Splits allowed for sampling requests.
    pub allowed_splits: Vec<SplitLabel>,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            batch_size: 128,
            ingestion_max_records: 2048,
            chunking: ChunkingStrategy::default(),
            recipes: Vec::new(),
            text_recipes: Vec::new(),
            split: SplitRatios::default(),
            allowed_splits: vec![SplitLabel::Train],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunking_strategy_defaults_are_stable() {
        let cfg = ChunkingStrategy::default();
        assert_eq!(cfg.max_window_tokens, 1024);
        assert_eq!(cfg.overlap_tokens, vec![64]);
        assert_eq!(cfg.summary_fallback_weight, 0.35);
        assert_eq!(cfg.summary_fallback_tokens, 512);
        assert_eq!(cfg.chunk_weight_floor, 0.1);
    }

    #[test]
    fn sampler_config_defaults_are_expected() {
        let cfg = SamplerConfig::default();
        assert_eq!(cfg.seed, 42);
        assert_eq!(cfg.batch_size, 128);
        assert_eq!(cfg.ingestion_max_records, 2048);
        assert!(cfg.recipes.is_empty());
        assert!(cfg.text_recipes.is_empty());
        assert_eq!(cfg.allowed_splits, vec![SplitLabel::Train]);
        assert_eq!(cfg.chunking.max_window_tokens, 1024);
    }

    #[test]
    fn selector_variants_can_be_constructed() {
        let role = Selector::Role(SectionRole::Anchor);
        let paragraph = Selector::Paragraph(3);
        let temporal = Selector::TemporalOffset(-2);
        let random = Selector::Random;

        assert!(matches!(role, Selector::Role(SectionRole::Anchor)));
        assert!(matches!(paragraph, Selector::Paragraph(3)));
        assert!(matches!(temporal, Selector::TemporalOffset(-2)));
        assert!(matches!(random, Selector::Random));
    }

    #[test]
    fn triplet_recipe_default_is_expected() {
        let recipe = TripletRecipe::default();
        assert_eq!(recipe.name.as_ref(), "");
        assert!(matches!(recipe.anchor, Selector::Random));
        assert!(matches!(recipe.positive_selector, Selector::Random));
        assert!(matches!(recipe.negative_selector, Selector::Random));
        assert!(matches!(
            recipe.negative_strategy,
            NegativeStrategy::WrongArticle
        ));
        assert_eq!(recipe.weight, 1.0);
        assert!(recipe.instruction.is_none());
        assert!(!recipe.allow_same_anchor_positive);
    }
}
