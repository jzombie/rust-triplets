use std::fmt;

use triplets_core::SplitLabel;
use triplets_core::data::PairLabel;

use triplets_srd_source::srd_triplet::{SrdPairWriteEntry, SrdTripletWriteEntry};

// ---------------------------------------------------------------------------
// Batch types (AoS — Array of Structs)
// ---------------------------------------------------------------------------

/// A single pair entry from the sampler: anchor + candidate + label.
#[derive(Debug, Clone)]
pub struct PairEntry {
    pub anchor_text: String,
    pub candidate_text: String,
    pub label: PairLabel,
}

/// A single triplet entry from the sampler: anchor + positive + negative.
#[derive(Debug, Clone)]
pub struct TripletEntry {
    pub anchor_text: String,
    pub pos_text: String,
    pub neg_text: String,
}

/// A batch of texts from the sampler, either pairs or triplets.
#[derive(Debug, Clone)]
pub enum SamplerBatch {
    Pairs(Vec<PairEntry>),
    Triplets(Vec<TripletEntry>),
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum SchedulerError {
    Io(std::io::Error),
    Msg(String),
}

impl fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::Msg(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for SchedulerError {}

impl From<std::io::Error> for SchedulerError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<String> for SchedulerError {
    fn from(s: String) -> Self {
        Self::Msg(s)
    }
}

impl From<&str> for SchedulerError {
    fn from(s: &str) -> Self {
        Self::Msg(s.to_string())
    }
}

pub type Result<T> = std::result::Result<T, SchedulerError>;

// ---------------------------------------------------------------------------
// Write boundaries (AoS)
// ---------------------------------------------------------------------------

/// A single pair entry ready for writing to the store.
#[derive(Debug, Clone)]
pub struct PairWriteEntry<'a> {
    pub anchor_text: &'a str,
    pub anchor_vec: &'a [f32],
    pub candidate_text: &'a str,
    pub candidate_vec: &'a [f32],
    pub label: &'a PairLabel,
}

#[derive(Debug, Clone)]
pub struct PairWriteArgs<'a> {
    pub entries: &'a [PairWriteEntry<'a>],
}

/// A single triplet entry ready for writing to the store.
#[derive(Debug, Clone)]
pub struct TripletWriteEntry<'a> {
    pub anchor_text: &'a str,
    pub anchor_vec: &'a [f32],
    pub pos_text: &'a str,
    pub pos_vec: &'a [f32],
    pub neg_text: &'a str,
    pub neg_vec: &'a [f32],
}

#[derive(Debug, Clone)]
pub struct TripletWriteArgs<'a> {
    pub entries: &'a [TripletWriteEntry<'a>],
}

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

pub trait EmbedStore: Send + Sync {
    /// Write pair entries starting at `start_idx`.
    fn write_pairs(&self, start_idx: u64, args: &PairWriteArgs<'_>) -> Result<()>;

    /// Write triplet entries (anchor + positive + negative) starting at `start_idx`.
    fn write_triplets(&self, start_idx: u64, args: &TripletWriteArgs<'_>) -> Result<()>;

    /// Current number of entries in the store.
    fn len(&self) -> Result<u64>;

    /// Returns `true` if the store contains no entries.
    fn is_empty(&self) -> Result<bool> {
        self.len().map(|n| n == 0)
    }
}

/// How to fetch the next batch of texts from the sampler.
pub trait BatchProvider: Send + Sync {
    /// Fetch the next batch for the given split.
    ///
    /// Returns `Ok(None)` when the split is exhausted.
    fn next_batch(&self, split: SplitLabel) -> Result<Option<SamplerBatch>>;

    /// Persist sampler state to disk.
    fn save_state(&self) -> Result<()>;
}

/// Embedding callback — the training crate implements this to route embed
/// calls to the remote teacher (or local model).
///
/// Returns `Vec<Vec<f32>>` for simplicity; a future change could parameterise
/// the element type, but would require a conversion at the storage boundary.
pub trait Embedder: Send + Sync {
    /// Embed a batch of texts. Returns vectors in the same order as input.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
}

/// Result of a single embed step.
#[derive(Debug)]
pub struct StepResult {
    /// Number of samples successfully processed and accumulated.
    pub samples_processed: u64,
    /// Number of samples dropped due to validation failures.
    pub samples_dropped: u64,
    /// Whether the pending buffer should be flushed to disk.
    pub should_flush: bool,
}

/// Configuration for the offline embedder pipeline.
#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    /// How many texts the sampler produces per call.
    pub sampler_batch_size: usize,
    /// How many texts the embedder processes per request.
    pub embed_batch_size: usize,
    /// Expected embedding dimension (for validation).
    pub emb_dim: usize,
    /// Number of steps between flushes.
    pub steps_per_batch: u64,
}

impl SchedulerConfig {
    /// Create a new config with the given parameters.
    pub fn new(
        sampler_batch_size: usize,
        embed_batch_size: usize,
        emb_dim: usize,
        steps_per_batch: u64,
    ) -> Self {
        Self {
            sampler_batch_size,
            embed_batch_size,
            emb_dim,
            steps_per_batch,
        }
    }
}
