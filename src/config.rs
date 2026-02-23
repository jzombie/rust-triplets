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
            max_window_tokens: 4096,
            overlap_tokens: vec![64, 128],
            summary_fallback_weight: 0.35,
            summary_fallback_tokens: 512,
            chunk_weight_floor: 0.1,
        }
    }
}

/// Defines a triplet recipe (anchor/positive/negative selection + weighting).
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
}

/// Selector for choosing a section or neighboring record.
#[derive(Clone, Debug)]
pub enum Selector {
    /// Select a section by role.
    Role(SectionRole),
    /// Select a specific section by index.
    Paragraph(usize),
    /// Select a temporal neighbor record by offset days.
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

/// Strategy for picking negatives for triplets/pairs.
#[derive(Clone, Debug)]
pub enum NegativeStrategy {
    /// Choose a record with a different publication date.
    WrongPublicationDate,
    /// Choose a different record from the same source.
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
    /// This is intentionally decoupled from `batch_size` so negatives are sampled
    /// from a broader pool and repeated negatives are less likely.
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
