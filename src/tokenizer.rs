//! Tokenization primitives used across chunking, sampling, and BM25 indexing.
//!
//! # Structural tokenizers vs. model tokenizers
//!
//! The [`Tokenizer`] trait and its default implementation, [`WhitespaceTokenizer`],
//! are **structural** tokenizers — their token counts drive window sizing, prefix
//! budget arithmetic, and BM25 term-frequency scoring.  They are **not** the
//! subword tokenizers used by embedding or language models, which include:
//!
//! * **BPE** (Byte-Pair Encoding) — GPT-series, RoBERTa, most OpenAI encoders.
//! * **WordPiece** — BERT-family models.
//! * **SentencePiece / Unigram** — T5, LLaMA, Mistral, and most instruction-tuned LLMs.
//!
//! Subword tokenizers operate on a learned vocabulary and routinely split a
//! single word into multiple tokens.  Whitespace token counts are a *structural
//! estimate*, running roughly 0.75–1.3× the equivalent BPE token count depending
//! on vocabulary and language.  Exact model token counts are unnecessary and
//! prohibitively expensive to compute without a loaded tokenizer binary.

/// Tokenizer over text slices.
///
/// Implementations are expected to be cheap to construct — ideally zero-size —
/// and stateless.  Methods take `&self` to allow future implementations that
/// carry configuration (e.g. vocabulary, normalisation flags).
pub trait Tokenizer {
    /// Split `text` into tokens, returning slices into the original string.
    fn tokenize<'a>(&self, text: &'a str) -> Vec<&'a str>;

    /// Count the number of tokens in `text`.
    ///
    /// Implementations should override this when a direct count is cheaper
    /// than collecting tokens into a `Vec`.
    fn token_count(&self, text: &str) -> usize {
        self.tokenize(text).len()
    }
}

/// Unicode-scalar whitespace tokenizer.
///
/// Splits on any sequence of Unicode whitespace via [`str::split_whitespace`]
/// and discards empty spans.  Zero-size; free to copy.
///
/// Token counts produced by this type are a *structural estimate* — see the
/// [module documentation](self) for how they relate to subword model tokenizers.
///
/// # Performance
///
/// Both [`tokenize`](Tokenizer::tokenize) and [`token_count`](Tokenizer::token_count)
/// are O(n) single-pass scans with no internal allocation beyond the returned
/// `Vec`.  An LRU cache would add memory pressure and synchronisation overhead
/// that outweighs any benefit at these text sizes.
#[derive(Clone, Copy, Debug, Default)]
pub struct WhitespaceTokenizer;

impl Tokenizer for WhitespaceTokenizer {
    #[inline]
    fn tokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        text.split_whitespace().collect()
    }

    #[inline]
    fn token_count(&self, text: &str) -> usize {
        text.split_whitespace().count()
    }
}
