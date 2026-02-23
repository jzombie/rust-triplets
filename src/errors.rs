use std::io;

use thiserror::Error;

use crate::types::SourceId;

/// Error type for sampler configuration, IO, and persistence failures.
#[derive(Debug, Error)]
pub enum SamplerError {
    #[error("data source '{source_id}' is unavailable: {reason}")]
    SourceUnavailable { source_id: SourceId, reason: String },
    #[error("data source '{source_id}' returned inconsistent state: {details}")]
    SourceInconsistent {
        source_id: SourceId,
        details: String,
    },
    #[error("split store failure: {0}")]
    SplitStore(String),
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("configuration error: {0}")]
    Configuration(String),
    #[error("no eligible samples available for recipe '{0}'")]
    Exhausted(String),
}
