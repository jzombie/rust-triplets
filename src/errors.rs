use std::io;

use thiserror::Error;

use crate::types::SourceId;

/// Error type for sampler configuration, IO, and persistence failures.
#[derive(Debug, Error)]
pub enum SamplerError {
    /// Source could not be refreshed or reached.
    #[error("data source '{source_id}' is unavailable: {reason}")]
    SourceUnavailable {
        /// Identifier of the failing source.
        source_id: SourceId,
        /// Human-readable failure reason.
        reason: String,
    },
    /// Source returned internally inconsistent state.
    #[error("data source '{source_id}' returned inconsistent state: {details}")]
    SourceInconsistent {
        /// Identifier of the inconsistent source.
        source_id: SourceId,
        /// Details about what invariant was violated.
        details: String,
    },
    /// Split-store persistence or decoding failure.
    #[error("split store failure: {0}")]
    SplitStore(String),
    /// Underlying IO error.
    #[error(transparent)]
    Io(#[from] io::Error),
    /// Invalid or conflicting configuration.
    #[error("configuration error: {0}")]
    Configuration(String),
    /// Sampling exhausted eligible candidates for a requested recipe/split.
    #[error("no eligible samples available for recipe '{0}'")]
    Exhausted(String),
}
