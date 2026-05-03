//! Tokenization primitives used across chunking, sampling, and BM25 indexing.
//!
//! # Structural tokenizers vs. model tokenizers
//!
//! The [`Tokenizer`](crate::tokenizer::Tokenizer) trait and its default implementation, [`WhitespaceTokenizer`](crate::tokenizer::WhitespaceTokenizer),
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

#[cfg(test)]
mod tests {
    use super::*;

    // --- Tokenizer::tokenize ---

    #[test]
    fn tokenize_splits_on_spaces() {
        let tokens = WhitespaceTokenizer.tokenize("hello world foo");
        assert_eq!(tokens, vec!["hello", "world", "foo"]);
    }

    #[test]
    fn tokenize_splits_on_tabs_and_newlines() {
        let tokens = WhitespaceTokenizer.tokenize("a\tb\nc");
        assert_eq!(tokens, vec!["a", "b", "c"]);
    }

    #[test]
    fn tokenize_collapses_runs_of_whitespace() {
        let tokens = WhitespaceTokenizer.tokenize("  foo   bar  ");
        assert_eq!(tokens, vec!["foo", "bar"]);
    }

    #[test]
    fn tokenize_empty_string_returns_empty() {
        assert!(WhitespaceTokenizer.tokenize("").is_empty());
    }

    #[test]
    fn tokenize_whitespace_only_returns_empty() {
        assert!(WhitespaceTokenizer.tokenize("   \t\n  ").is_empty());
    }

    #[test]
    fn tokenize_single_token_no_whitespace() {
        let tokens = WhitespaceTokenizer.tokenize("solo");
        assert_eq!(tokens, vec!["solo"]);
    }

    #[test]
    fn tokenize_returns_slices_into_original() {
        let text = String::from("alpha beta gamma");
        let tokens = WhitespaceTokenizer.tokenize(&text);
        // Pointers should point inside the original allocation.
        for token in &tokens {
            let token_ptr = token.as_ptr() as usize;
            let text_start = text.as_ptr() as usize;
            let text_end = text_start + text.len();
            assert!(token_ptr >= text_start && token_ptr < text_end);
        }
    }

    #[test]
    fn tokenize_unicode_whitespace_splits_correctly() {
        // U+3000 IDEOGRAPHIC SPACE is Unicode whitespace.
        let tokens = WhitespaceTokenizer.tokenize("東京\u{3000}大阪");
        assert_eq!(tokens, vec!["東京", "大阪"]);
    }

    // --- Tokenizer::token_count ---

    #[test]
    fn token_count_matches_tokenize_len() {
        let text = "one two three four";
        assert_eq!(
            WhitespaceTokenizer.token_count(text),
            WhitespaceTokenizer.tokenize(text).len()
        );
    }

    #[test]
    fn token_count_empty_is_zero() {
        assert_eq!(WhitespaceTokenizer.token_count(""), 0);
    }

    #[test]
    fn token_count_whitespace_only_is_zero() {
        assert_eq!(WhitespaceTokenizer.token_count("  \t\n "), 0);
    }

    #[test]
    fn token_count_single_word() {
        assert_eq!(WhitespaceTokenizer.token_count("word"), 1);
    }

    // --- Trait default method ---

    #[test]
    fn default_token_count_delegates_to_tokenize() {
        /// Tokenizer that always splits on '|' — exercises the default `token_count`.
        struct PipeTokenizer;
        impl Tokenizer for PipeTokenizer {
            fn tokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
                text.split('|').filter(|s| !s.is_empty()).collect()
            }
        }
        // token_count falls back to tokenize().len() since PipeTokenizer doesn't override it.
        assert_eq!(PipeTokenizer.token_count("a|b|c"), 3);
        assert_eq!(PipeTokenizer.token_count(""), 0);
    }

    // --- Derive traits ---

    #[test]
    fn whitespace_tokenizer_is_clone_copy_and_debug() {
        let t = WhitespaceTokenizer;
        let cloned = t;
        let copied = t;
        assert_eq!(format!("{:?}", cloned), "WhitespaceTokenizer");
        let _ = copied;
    }

    #[test]
    fn whitespace_tokenizer_default_is_usable() {
        let t = WhitespaceTokenizer;
        assert_eq!(t.token_count("x y"), 2);
    }
}
