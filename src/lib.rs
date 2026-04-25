#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

/// Pluggable chunking algorithms and default sliding-window implementation.
pub mod chunking;
/// Sampling configuration types.
pub mod config;
/// Centralized constants used across sampler, splits, and sources.
pub mod constants;
/// Data record and sample batch types.
pub mod data;
/// OCR denoising and markdown-table cleanup for text chunks.
pub mod denoiser;
mod epoch;
/// Reusable example runners shared by downstream crates.
pub mod example_apps;
/// Stable deterministic hashing utilities.
pub mod hash;
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

/// Structural text tokenizer trait and whitespace implementation.
pub mod tokenizer;
/// Shared type aliases.
pub mod types;
/// Text normalization helpers.
pub mod utils;

mod errors;

pub use chunking::{ChunkingAlgorithm, SlidingWindowChunker};
pub use config::{
    ChunkingStrategy, DenoiserConfig, NegativeStrategy, SamplerConfig, Selector, TextRecipe,
    TripletRecipe,
};
pub use data::{
    DataRecord, PairLabel, QualityScore, RecordChunk, SampleBatch, SamplePair, SampleTriplet,
    SectionRole, TextBatch, TextSample, TripletBatch,
};
pub use errors::SamplerError;
pub use hash::stable_hash_str;
pub use ingestion::{IngestionManager, RecordCache};
pub use kvp::{KvpField, KvpPrefixSampler};
pub use sampler::{BatchPrefetcher, Sampler, TripletSampler};
pub use source::backends::csv_source::{CsvSource, CsvSourceConfig};
#[cfg(feature = "huggingface")]
pub use source::backends::huggingface_source::{
    HfListRoots, HfSourceEntry, build_hf_sources, load_hf_sources_from_list,
    managed_hf_list_snapshot_dir, managed_hf_snapshot_dir, parse_csv_fields, parse_hf_source_line,
    parse_hf_uri, resolve_hf_list_roots,
};
pub use source::{DataSource, SourceCursor};
#[cfg(feature = "huggingface")]
pub use source::{HuggingFaceRowSource, HuggingFaceRowsConfig};
pub use splits::{DeterministicSplitStore, FileSplitStore, SplitLabel, SplitRatios, SplitStore};
pub use types::{
    CategoryId, HashPart, KvpValue, LogMessage, MetaValue, PathString, RecipeKey, RecordId,
    Sentence, SourceId, TaxonomyValue,
};
