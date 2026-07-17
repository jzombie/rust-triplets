use std::fmt;

use triplets_core::SplitLabel;
use triplets_core::data::PairLabel;

// ---------------------------------------------------------------------------
// Batch types (AoS — Array of Structs)
// ---------------------------------------------------------------------------

/// A single pair entry from the sampler: anchor + candidate + label.
#[derive(Debug, Clone)]
pub struct PairEntry {
    /// The anchor text for this pair.
    pub anchor_text: String,
    /// The candidate (positive/negative) text for this pair.
    pub candidate_text: String,
    /// The label indicating whether this is a positive or negative pair.
    pub label: PairLabel,
}

/// A single triplet entry from the sampler: anchor + positive + negative.
#[derive(Debug, Clone)]
pub struct TripletEntry {
    /// The anchor text for this triplet.
    pub anchor_text: String,
    /// The positive text for this triplet.
    pub pos_text: String,
    /// The negative text for this triplet.
    pub neg_text: String,
}

/// A batch of texts from the sampler, either pairs or triplets.
#[derive(Debug, Clone)]
pub enum SamplerBatch {
    /// A batch of pair entries.
    Pairs(Vec<PairEntry>),
    /// A batch of triplet entries.
    Triplets(Vec<TripletEntry>),
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur during the embedding pipeline.
#[derive(Debug)]
pub enum SchedulerError {
    /// An I/O error occurred.
    Io(std::io::Error),
    /// A semantic error with a message.
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

/// A specialized `Result` type for the embedding pipeline.
pub type Result<T> = std::result::Result<T, SchedulerError>;

// ---------------------------------------------------------------------------
// Write boundaries (AoS)
// ---------------------------------------------------------------------------

/// A single pair entry ready for writing to the store.
#[derive(Debug, Clone)]
pub struct PairWriteEntry<'a> {
    /// The anchor text (borrowed).
    pub anchor_text: &'a str,
    /// The anchor embedding vector (borrowed).
    pub anchor_vec: &'a [f32],
    /// The candidate text (borrowed).
    pub candidate_text: &'a str,
    /// The candidate embedding vector (borrowed).
    pub candidate_vec: &'a [f32],
    /// The pair label (borrowed).
    pub label: &'a PairLabel,
}

/// Arguments for writing a batch of pair entries.
#[derive(Debug, Clone)]
pub struct PairWriteArgs<'a> {
    /// The pair entries to write.
    pub entries: &'a [PairWriteEntry<'a>],
}

/// A single triplet entry ready for writing to the store.
#[derive(Debug, Clone)]
pub struct TripletWriteEntry<'a> {
    /// The anchor text (borrowed).
    pub anchor_text: &'a str,
    /// The anchor embedding vector (borrowed).
    pub anchor_vec: &'a [f32],
    /// The positive text (borrowed).
    pub pos_text: &'a str,
    /// The positive embedding vector (borrowed).
    pub pos_vec: &'a [f32],
    /// The negative text (borrowed).
    pub neg_text: &'a str,
    /// The negative embedding vector (borrowed).
    pub neg_vec: &'a [f32],
}

/// Arguments for writing a batch of triplet entries.
#[derive(Debug, Clone)]
pub struct TripletWriteArgs<'a> {
    /// The triplet entries to write.
    pub entries: &'a [TripletWriteEntry<'a>],
}

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// Trait for embedding stores that persist pair/triplet data.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn scheduler_error_display_io() {
        let err = SchedulerError::Io(io::Error::new(io::ErrorKind::NotFound, "file missing"));
        assert_eq!(err.to_string(), "file missing");
    }

    #[test]
    fn scheduler_error_display_msg() {
        let err = SchedulerError::Msg("something went wrong".to_string());
        assert_eq!(err.to_string(), "something went wrong");
    }

    #[test]
    fn scheduler_error_from_io_error() {
        let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "denied");
        let err: SchedulerError = io_err.into();
        assert!(matches!(err, SchedulerError::Io(_)));
        assert_eq!(err.to_string(), "denied");
    }

    #[test]
    fn scheduler_error_from_string() {
        let err: SchedulerError = "bad input".to_string().into();
        assert!(matches!(err, SchedulerError::Msg(_)));
        assert_eq!(err.to_string(), "bad input");
    }

    #[test]
    fn scheduler_error_from_str() {
        let err: SchedulerError = "static msg".into();
        assert!(matches!(err, SchedulerError::Msg(_)));
        assert_eq!(err.to_string(), "static msg");
    }

    #[test]
    fn scheduler_error_is_std_error() {
        let err: SchedulerError = "test".into();
        let _: &dyn std::error::Error = &err;
    }

    struct MockStore {
        count: u64,
    }

    impl EmbedStore for MockStore {
        fn write_pairs(&self, _start_idx: u64, _args: &PairWriteArgs<'_>) -> Result<()> {
            Ok(())
        }
        fn write_triplets(&self, _start_idx: u64, _args: &TripletWriteArgs<'_>) -> Result<()> {
            Ok(())
        }
        fn len(&self) -> Result<u64> {
            Ok(self.count)
        }
    }

    #[test]
    fn embed_store_is_empty_true() {
        let store = MockStore { count: 0 };
        assert!(store.is_empty().unwrap());
    }

    #[test]
    fn embed_store_is_empty_false() {
        let store = MockStore { count: 5 };
        assert!(!store.is_empty().unwrap());
    }

    #[test]
    fn scheduler_config_new() {
        let cfg = SchedulerConfig::new(32, 64, 768, 10);
        assert_eq!(cfg.sampler_batch_size, 32);
        assert_eq!(cfg.embed_batch_size, 64);
        assert_eq!(cfg.emb_dim, 768);
        assert_eq!(cfg.steps_per_batch, 10);
    }

    #[test]
    fn step_result_fields() {
        let r = StepResult {
            samples_processed: 100,
            samples_dropped: 3,
            should_flush: true,
        };
        assert_eq!(r.samples_processed, 100);
        assert_eq!(r.samples_dropped, 3);
        assert!(r.should_flush);
    }

    #[test]
    fn sampler_batch_variants() {
        let pairs = SamplerBatch::Pairs(vec![]);
        let triplets = SamplerBatch::Triplets(vec![]);
        assert!(matches!(pairs, SamplerBatch::Pairs(_)));
        assert!(matches!(triplets, SamplerBatch::Triplets(_)));
    }

    #[test]
    fn pair_entry_clone() {
        let e = PairEntry {
            anchor_text: "a".into(),
            candidate_text: "b".into(),
            label: PairLabel::Positive,
        };
        let e2 = e.clone();
        assert_eq!(e2.anchor_text, "a");
    }

    #[test]
    fn triplet_entry_clone() {
        let e = TripletEntry {
            anchor_text: "a".into(),
            pos_text: "p".into(),
            neg_text: "n".into(),
        };
        let e2 = e.clone();
        assert_eq!(e2.neg_text, "n");
    }
}
