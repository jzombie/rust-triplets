//! Pluggable chunking algorithms.
//!
//! [`ChunkingAlgorithm`] defines the extension point used by the sampler to
//! materialize chunk candidates from record sections.

mod algorithm;
mod sliding_window;

pub use algorithm::ChunkingAlgorithm;
pub use sliding_window::SlidingWindowChunker;
