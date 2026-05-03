//! Pluggable text preprocessor infrastructure.
//!
//! Preprocessors run as a sequential pipeline inside
//! [`crate::chunking::SlidingWindowChunker`] before tokenization.  Each
//! preprocessor receives the text of a section and returns either
//! `Some(transformed)` or `None`.  A `None` return from any stage
//! short-circuits the remainder and causes the entire section to be dropped —
//! no chunks are produced from it.
//!
//! ## Registration
//!
//! Preprocessors are registered on a [`crate::config::ChunkingStrategy`] via
//! [`crate::config::ChunkingStrategy::register_preprocessor`]:
//!
//! ```rust
//! use triplets_core::{ChunkingStrategy, DenoiserConfig, DenoiserPreprocessor};
//!
//! let mut strategy = ChunkingStrategy::default();
//! strategy.register_preprocessor(DenoiserPreprocessor::new(DenoiserConfig {
//!     enabled: true,
//!     max_digit_ratio: 0.35,
//!     strip_markdown: true,
//! }));
//! ```
//!
//! Multiple preprocessors run in registration order; the output of one feeds
//! the next.

/// Built-in preprocessor implementations.
pub mod backends;

/// Trait for pluggable text preprocessors.
///
/// Implement this trait to transform or filter section text before it is
/// tokenized and chunked.  The pipeline is sequential: the output of each
/// stage feeds the next.
///
/// # Implementing
///
/// ```rust
/// use triplets_core::TextPreprocessor;
///
/// struct UppercasePreprocessor;
///
/// impl TextPreprocessor for UppercasePreprocessor {
///     fn process(&self, text: &str) -> Option<String> {
///         Some(text.to_uppercase())
///     }
/// }
/// ```
pub trait TextPreprocessor: Send + Sync {
    /// Process a text block.
    ///
    /// Returns `Some(transformed)` with the (possibly modified) text, or
    /// `None` to signal that the section should be discarded entirely —
    /// no chunks will be produced from it.
    fn process(&self, text: &str) -> Option<String>;
}
