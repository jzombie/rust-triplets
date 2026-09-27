use thiserror::Error;

/// Error type for simd-r-drive triplet encoding/decoding operations.
#[derive(Debug, Error)]
pub enum SrdError {
    /// Entry too short for the mode+flags header.
    #[error("entry too short for mode+flags")]
    EntryTooShort,

    /// Unknown SrdMode byte encountered.
    #[error("unknown SrdMode byte: {0}")]
    UnknownMode(u8),

    /// Truncated text length field.
    #[error("truncated text length field")]
    TruncatedTextLength,

    /// Entry data is shorter than expected for embeddings + texts.
    #[error("entry truncated: {actual} bytes < {expected}")]
    TruncatedEntry {
        /// Actual byte count available.
        actual: usize,
        /// Expected byte count.
        expected: usize,
    },

    /// Invalid UTF-8 in text payload.
    #[error("invalid UTF-8 in text")]
    InvalidUtf8(#[from] std::str::Utf8Error),

    /// Embedding dimension must be greater than zero.
    #[error("embedding dimension must be > 0")]
    ZeroDimension,

    /// Write batch length mismatch between vectors and texts.
    #[error("write batch length mismatch: {vec_count} embeddings for {text_count} texts")]
    BatchLengthMismatch {
        /// Number of embedding vectors provided.
        vec_count: usize,
        /// Number of text strings provided.
        text_count: usize,
    },

    /// Inconsistent embedding dimension within a batch.
    #[error("inconsistent embedding dimension at index {index}: got {got}, expected {expected}")]
    InconsistentDimension {
        /// Index of the offending vector.
        index: usize,
        /// Dimensionality encountered.
        got: usize,
        /// Expected dimensionality.
        expected: usize,
    },

    /// Non-finite embedding value detected.
    #[error("non-finite embedding value at index {index} component {component}")]
    NonFiniteEmbedding {
        /// Index of the offending vector.
        index: usize,
        /// Component index within the vector.
        component: usize,
    },

    /// Anchor and positive batch lengths differ.
    #[error("anchor/positive batch length mismatch")]
    PairLengthMismatch,

    /// Anchor, positive, and negative batch lengths differ.
    #[error("anchor/positive/negative batch length mismatch")]
    TripletLengthMismatch,

    /// Underlying I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
