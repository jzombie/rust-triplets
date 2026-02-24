#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

/// Sampling configuration types.
pub mod config;
/// Centralized constants used across sampler, splits, and sources.
pub mod constants;
/// Data record and sample batch types.
pub mod data;
mod epoch;
/// Reusable example runners shared by downstream crates.
pub mod example_apps;
mod hash;
/// Capacity and sampling estimation helpers.
pub mod heuristics;
/// Background ingestion and caching infrastructure.
pub mod ingestion;
/// Key/value prefix sampling helpers.
pub mod kvp;
/// Metadata keys and helpers.
pub mod metadata;
/// Aggregate metrics helpers.
pub mod metrics;
/// Sampler implementations and public sampling API.
pub mod sampler;
/// Data source traits and built-in sources.
pub mod source;
/// Split stores and persistence helpers.
pub mod splits;
/// Input transports used by sources (filesystem today; DBs later).
pub mod transport;
/// Shared type aliases.
pub mod types;
/// Text normalization helpers.
pub mod utils;

mod errors;

pub use config::{
    ChunkingStrategy, NegativeStrategy, SamplerConfig, Selector, TextRecipe, TripletRecipe,
};
pub use data::{
    DataRecord, PairLabel, QualityScore, RecordChunk, SampleBatch, SamplePair, SampleTriplet,
    SectionRole, TextBatch, TextSample, TripletBatch,
};
pub use errors::SamplerError;
pub use ingestion::{IngestionManager, RecordCache};
pub use kvp::{KvpField, KvpPrefixSampler};
pub use sampler::{BatchPrefetcher, Sampler, TripletSampler};
pub use source::{DataSource, SourceCursor};
#[cfg(feature = "huggingface")]
pub use source::{HuggingFaceRowSource, HuggingFaceRowsConfig};
pub use splits::{DeterministicSplitStore, FileSplitStore, SplitLabel, SplitRatios, SplitStore};
pub use types::{
    CategoryId, HashPart, KvpValue, LogMessage, MetaValue, PathString, RecipeKey, RecordId,
    Sentence, SourceId, TaxonomyValue,
};
