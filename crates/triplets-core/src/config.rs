use crate::data::SectionRole;
use crate::splits::{SplitLabel, SplitRatios};
use std::borrow::Cow;
use std::sync::Arc;

/// Configuration for the OCR denoiser that filters digit-heavy text.
///
/// When enabled, text sections that are predominantly numerical (e.g. mangled OCR tables)
/// are stripped down to their alphabetical content on a line-by-line basis, or dropped
/// entirely when no alphabetical content remains.
#[derive(Clone, Debug)]
pub struct DenoiserConfig {
    /// Whether denoising is active. Defaults to `false` so existing behavior is unchanged.
    pub enabled: bool,
    /// Maximum ratio of digit characters to (digit + alphabetical) characters before a
    /// line is considered mangled OCR output. Range: `0.0`–`1.0`.
    ///
    /// A value of `0.35` means that if more than 35% of the alphanumeric characters on a
    /// line are digits, the line is treated as a mangled table row and stripped down to
    /// its alphabetical tokens.
    ///
    /// Defaults to `0.35`.
    pub max_digit_ratio: f32,
    /// Whether to strip common markdown formatting boundaries (e.g. pipe `|` table boundaries,
    /// dropping layout-only separator rows like `|---|---|`).
    ///
    /// Currently covers GFM tables; may expand to other structural markers in the future.
    /// Semantic text is preserved.
    ///
    /// Defaults to `true`.
    pub strip_markdown: bool,
}

impl Default for DenoiserConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_digit_ratio: 0.35,
            strip_markdown: true,
        }
    }
}

/// Controls how long text sections are chunked and weighted.
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
    /// Pluggable text preprocessors applied in order before chunking.
    pub(crate) preprocessors: Vec<Arc<dyn crate::preprocessor::TextPreprocessor>>,
}

impl Default for ChunkingStrategy {
    fn default() -> Self {
        Self {
            max_window_tokens: 1024,
            overlap_tokens: vec![64],
            summary_fallback_weight: 0.35,
            summary_fallback_tokens: 512,
            chunk_weight_floor: 0.1,
            preprocessors: Vec::new(),
        }
    }
}

impl Clone for ChunkingStrategy {
    fn clone(&self) -> Self {
        Self {
            max_window_tokens: self.max_window_tokens,
            overlap_tokens: self.overlap_tokens.clone(),
            summary_fallback_weight: self.summary_fallback_weight,
            summary_fallback_tokens: self.summary_fallback_tokens,
            chunk_weight_floor: self.chunk_weight_floor,
            preprocessors: self.preprocessors.clone(),
        }
    }
}

impl std::fmt::Debug for ChunkingStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChunkingStrategy")
            .field("max_window_tokens", &self.max_window_tokens)
            .field("overlap_tokens", &self.overlap_tokens)
            .field("summary_fallback_weight", &self.summary_fallback_weight)
            .field("summary_fallback_tokens", &self.summary_fallback_tokens)
            .field("chunk_weight_floor", &self.chunk_weight_floor)
            .field(
                "preprocessors",
                &format_args!("{} registered", self.preprocessors.len()),
            )
            .finish()
    }
}

impl ChunkingStrategy {
    /// Register a [`crate::preprocessor::TextPreprocessor`] to run before chunking.
    ///
    /// Preprocessors are applied in registration order; if any returns `None`
    /// the section is dropped and produces no chunks.
    pub fn register_preprocessor(
        &mut self,
        p: impl crate::preprocessor::TextPreprocessor + 'static,
    ) -> &mut Self {
        self.preprocessors.push(Arc::new(p));
        self
    }

    /// Return the slice of registered preprocessors.
    pub fn preprocessors(&self) -> &[Arc<dyn crate::preprocessor::TextPreprocessor>] {
        &self.preprocessors
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
    /// Relative weight controlling how often this recipe is selected versus other recipes.
    ///
    /// Each recipe with a positive weight receives a number of slots in the shuffled selection
    /// order proportional to `weight / min_positive_weight` across all active recipes, so a
    /// recipe with `weight = 2.0` is drawn approximately twice as often as one with `weight =
    /// 1.0`.  The weight also scales the `weight` field on every [`crate::SampleTriplet`]
    /// returned by this recipe, which the caller's training loop can use for loss weighting.
    ///
    /// Recipes with `weight <= 0.0` are excluded from selection entirely and no samples
    /// are produced for them.
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
    /// Per-lane retrieval hardness override (`None` = use global `SamplerConfig`).
    ///
    /// Rank-windowed BM25 sampling; cache partitioned by `(anchor, search_top_k)`.
    pub retrieval_sampling: Option<RetrievalSamplingConfig>,
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
            retrieval_sampling: None,
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
    /// Relative weight controlling how often this recipe is selected versus other recipes.
    ///
    /// Each recipe with a positive weight receives a number of slots in the shuffled selection
    /// order proportional to `weight / min_positive_weight` across all active recipes, so a
    /// recipe with `weight = 2.0` is drawn approximately twice as often as one with `weight =
    /// 1.0`.  The weight also scales the `weight` field on every [`crate::TextSample`]
    /// returned by this recipe, which the caller's training loop can use for loss weighting.
    ///
    /// Recipes with `weight <= 0.0` are excluded from selection entirely and no samples
    /// are produced for them.
    pub weight: f32,
    /// Optional instruction text attached to samples from this recipe.
    pub instruction: Option<Cow<'static, str>>,
}

/// Rank-windowed retrieval sampling configuration.
///
/// Industry-standard BM25 hard-negative mining: rank candidates by score
/// descending, skip `min_rank` false-positive heads, cap at `max_rank`,
/// uniform sample from `ranked[start..end]` with
/// `start = min_rank.min(len.saturating_sub(1))`,
/// `end = max_rank.clamp(start+1, len)`.
///
/// Defaults: `min_rank = 1` (skip rank-0 identical match),
/// `max_rank = 30` (standard hard), `search_top_k = 32`.
#[derive(Clone, Debug)]
pub struct RetrievalSamplingConfig {
    /// Ranks below this are never sampled (default 1 skips the rank-0
    /// identical match, often a false-positive paraphrase). Consumed in
    /// `Bm25Backend::select_hard_negative` as the window floor;
    /// ignored by `DefaultBackend`.
    pub min_rank: usize,
    /// Ranks at or above this are never sampled (default 30, industry
    /// standard hard cutoff; tail beyond is easy negatives). Consumed in
    /// `select_hard_negative` as the window ceiling; ignored by
    /// `DefaultBackend`.
    pub max_rank: usize,
    /// How many top-scored BM25 index hits to retrieve per anchor query
    /// (default 32). Consumed in `ranked_candidates` as the search depth
    /// before strategy-pool intersection; larger values cost more per
    /// query and grow the per-anchor cache. Ignored by `DefaultBackend`.
    pub search_top_k: usize,
}

impl Default for RetrievalSamplingConfig {
    fn default() -> Self {
        Self {
            min_rank: 1,
            max_rank: 30,
            #[cfg(feature = "bm25-mining")]
            search_top_k: crate::constants::sampler::BM25_SEARCH_TOP_K,
            #[cfg(not(feature = "bm25-mining"))]
            search_top_k: 32,
        }
    }
}

impl RetrievalSamplingConfig {
    /// Eager startup validation: rejects invalid states before runtime sampling.
    pub fn validate(&self) -> Result<(), String> {
        if self.search_top_k == 0 {
            return Err("retrieval_sampling.search_top_k must be > 0".to_string());
        }
        if self.max_rank == 0 {
            return Err("retrieval_sampling.max_rank must be > 0".to_string());
        }
        if !(self.min_rank < self.max_rank) {
            return Err(format!(
                "retrieval min_rank ({}) must be < max_rank ({})",
                self.min_rank, self.max_rank
            ));
        }
        Ok(())
    }
}

/// Strategy for picking the negative *record* in a triplet.
///
/// Each variant defines the candidate pool from which the negative record is drawn.
/// By default all variants scope candidates to the same source as the anchor, so
/// negatives are hard relative to the source domain rather than trivially
/// cross-domain. A same-split fallback engages automatically when the in-source
/// pool is too small (for example a source with only one record in the split).
///
/// When the `bm25-mining` feature is enabled, retrieval re-ranking
/// (see `RetrievalSamplingConfig`) is applied on top of the strategy-filtered
/// pool. BM25 lexical re-ranking is the current backend — BM25 re-orders candidates by keyword
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
    /// Choose a negative from the SAME record.
    ///
    /// Used for HuggingFace "dict" datasets where negatives are embedded
    /// in the same row as the anchor/positive pair.  The sampler returns
    /// the anchor record itself as the "negative record", and
    /// `select_chunk` picks from its Context sections.
    SameRecord,
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
    /// Global retrieval hardness default; per-recipe `retrieval_sampling` overrides.
    pub retrieval_sampling: RetrievalSamplingConfig,
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
            retrieval_sampling: RetrievalSamplingConfig::default(),
        }
    }
}

impl SamplerConfig {
    /// Eager startup validation for retrieval hardness configs.
    ///
    /// Validates the global default plus every per-recipe override. Returns
    /// `Err(SamplerError::Configuration)` on the first invalid entry so
    /// invalid states never penetrate the runtime sampling pipeline.
    pub fn validate(&self) -> Result<(), crate::errors::SamplerError> {
        self.retrieval_sampling
            .validate()
            .map_err(crate::errors::SamplerError::Configuration)?;
        for recipe in &self.recipes {
            if let Some(cfg) = recipe.retrieval_sampling.as_ref() {
                cfg.validate().map_err(|e| {
                    crate::errors::SamplerError::Configuration(format!(
                        "recipe '{}': {e}",
                        recipe.name
                    ))
                })?;
            }
        }
        Ok(())
    }

    /// Consuming builder to enable the built-in OCR/markdown denoiser on
    /// the sampler's chunking strategy.
    ///
    /// Chains denoiser setup during `SamplerConfig` construction. Works with
    /// struct update syntax to customize other fields at the same time:
    ///
    /// ```rust,no_run
    /// use triplets_core::{SamplerConfig, config::DenoiserConfig};
    ///
    /// // Enable denoiser with all other fields at their defaults:
    /// let config = SamplerConfig::default()
    ///     .with_denoiser(DenoiserConfig { enabled: true, ..DenoiserConfig::default() });
    ///
    /// // Or customize other fields first, then add the denoiser:
    /// let config = SamplerConfig { batch_size: 32, ..SamplerConfig::default() }
    ///     .with_denoiser(DenoiserConfig { enabled: true, ..DenoiserConfig::default() });
    /// ```
    pub fn with_denoiser(mut self, config: DenoiserConfig) -> Self {
        use crate::preprocessor::backends::denoiser_preprocessor::DenoiserPreprocessor;
        self.chunking
            .preprocessors
            .push(Arc::new(DenoiserPreprocessor::new(config)));
        self
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

    #[test]
    fn denoiser_config_defaults() {
        let cfg = DenoiserConfig::default();
        assert!(!cfg.enabled);
        assert_eq!(cfg.max_digit_ratio, 0.35);
        assert!(cfg.strip_markdown);
    }

    #[test]
    fn chunking_strategy_clone() {
        let cfg = ChunkingStrategy::default();
        let cloned = cfg.clone();
        assert_eq!(cloned.max_window_tokens, cfg.max_window_tokens);
        assert_eq!(cloned.overlap_tokens, cfg.overlap_tokens);
        assert_eq!(cloned.summary_fallback_weight, cfg.summary_fallback_weight);
        assert_eq!(cloned.summary_fallback_tokens, cfg.summary_fallback_tokens);
        assert_eq!(cloned.chunk_weight_floor, cfg.chunk_weight_floor);
    }

    #[test]
    fn chunking_strategy_debug() {
        let cfg = ChunkingStrategy::default();
        let debug_str = format!("{:?}", cfg);
        assert!(debug_str.contains("ChunkingStrategy"));
        assert!(debug_str.contains("1024"));
    }

    #[test]
    fn negative_strategy_variants() {
        let strategies = [
            NegativeStrategy::WrongPublicationDate,
            NegativeStrategy::WrongArticle,
            NegativeStrategy::QuestionAnswerMismatch,
            NegativeStrategy::SameRecord,
        ];
        for s in &strategies {
            let _ = format!("{:?}", s);
        }
    }

    #[test]
    fn retrieval_sampling_defaults_and_validation() {
        let cfg = RetrievalSamplingConfig::default();
        assert_eq!(cfg.min_rank, 1);
        assert_eq!(cfg.max_rank, 30);
        assert!(cfg.validate().is_ok());
        assert!(cfg.search_top_k > 0);
        let bad = RetrievalSamplingConfig {
            min_rank: 5,
            max_rank: 5,
            search_top_k: 32,
        };
        assert!(bad.validate().is_err());
        let bad2 = RetrievalSamplingConfig {
            min_rank: 10,
            max_rank: 5,
            search_top_k: 32,
        };
        assert!(bad2.validate().is_err());
        let bad3 = RetrievalSamplingConfig {
            min_rank: 0,
            max_rank: 0,
            search_top_k: 32,
        };
        assert!(bad3.validate().is_err());
        let bad4 = RetrievalSamplingConfig {
            min_rank: 1,
            max_rank: 30,
            search_top_k: 0,
        };
        assert!(bad4.validate().is_err());
        let mut global = SamplerConfig::default();
        assert!(global.validate().is_ok());
        global.recipes.push(TripletRecipe {
            retrieval_sampling: Some(RetrievalSamplingConfig {
                min_rank: 1,
                max_rank: 10,
                search_top_k: 100,
            }),
            ..TripletRecipe::default()
        });
        assert!(global.validate().is_ok());
    }

    #[test]
    fn selector_debug() {
        let s = Selector::Role(SectionRole::Anchor);
        let debug_str = format!("{:?}", s);
        assert!(debug_str.contains("Role"));
    }

    #[test]
    fn text_recipe_construction() {
        let recipe = TextRecipe {
            name: "test".into(),
            selector: Selector::Role(SectionRole::Anchor),
            weight: 2.0,
            instruction: Some("do something".into()),
        };
        assert_eq!(recipe.weight, 2.0);
        assert!(recipe.instruction.is_some());
    }

    #[test]
    fn sampler_config_with_denoiser() {
        let cfg = SamplerConfig::default().with_denoiser(DenoiserConfig {
            enabled: true,
            max_digit_ratio: 0.5,
            strip_markdown: false,
        });
        assert_eq!(cfg.chunking.preprocessors.len(), 1);
    }
}
