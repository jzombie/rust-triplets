//! Text normalization helpers shared by source implementations.

use chrono::{DateTime, Utc};
use std::fs;
use std::path::Path;

use crate::data::{RecordSection, SectionRole};
use crate::types::Sentence;

/// Split `text` into whitespace-delimited tokens.
///
/// # What this tokenizer is — and is not
///
/// This is a **Unicode-scalar whitespace tokenizer** (equivalent to what NLP
/// literature calls a *space tokenizer* or *word tokenizer*).  It splits on
/// any sequence of Unicode whitespace and discards empty spans, matching the
/// semantics of [`str::split_whitespace`].
///
/// **This is not the subword tokenizer used by any of the models this library
/// serves data to.**  Embedding and language models typically use one of:
///
/// * **BPE** (Byte-Pair Encoding) — GPT-series, RoBERTa, most OpenAI encoders.
/// * **WordPiece** — BERT-family models.
/// * **SentencePiece / Unigram** — T5, LLaMA, Mistral, and most instruction-tuned LLMs.
///
/// Subword tokenizers operate on a learned vocabulary and routinely split a
/// single English word into multiple tokens (e.g. `"tokenization"` → 3–4
/// subword units in most BPE vocabularies).  As a result, the count returned
/// here is a *structural estimate*, not the exact token count a model would
/// produce.  In practice, whitespace token counts run roughly 0.75–1.3× the
/// equivalent BPE token count depending on vocabulary and language.
///
/// The whitespace granularity is intentional for **structural purposes** —
/// window sizing, prefix-budget arithmetic, BM25 term-frequency scoring — where
/// exact model token counts are unnecessary and prohibitively expensive to
/// compute without a loaded tokenizer binary.
///
/// # Performance
///
/// The implementation is a single O(n) scan with no heap allocation beyond the
/// returned `Vec`.  An LRU cache would add memory pressure and synchronisation
/// overhead that outweighs any benefit for these input sizes; the scan itself
/// is cheaper than a cache lookup under contention.
///
/// If the tokenization strategy ever needs to vary (e.g. to plug in a real BPE
/// encoder), introduce a `Tokenizer` trait at that point rather than
/// generalising prematurely here.
pub fn tokenize(text: &str) -> Vec<&str> {
    text.split_whitespace().collect()
}

/// Count whitespace-delimited tokens in `text` without allocating.
///
/// See [`tokenize`] for a full discussion of what this count represents and
/// how it relates to the subword token counts produced by downstream models.
pub fn token_count(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Collapse repeated whitespace in-place while preserving single spaces.
/// Collapse runs of whitespace into single spaces and trim.
pub fn normalize_inline_whitespace<T: AsRef<str>>(text: T) -> String {
    let mut normalized = String::new();
    let mut seen_space = false;
    for ch in text.as_ref().chars() {
        if ch.is_whitespace() {
            if !seen_space {
                normalized.push(' ');
                seen_space = true;
            }
        } else {
            normalized.push(ch);
            seen_space = false;
        }
    }
    normalized.trim().to_string()
}

/// Split a block of text into sentences, falling back to the whole string when needed.
/// Heuristic sentence splitter with tokenizer-friendly rules.
pub fn sentences(text: &str) -> Vec<Sentence> {
    let mut results = Vec::new();

    for block in text.split("\n\n") {
        if block.trim().is_empty() {
            continue;
        }
        let normalized = normalize_inline_whitespace(block);
        if normalized.is_empty() {
            continue;
        }
        push_block_sentences(&normalized, &mut results);
    }

    results
}

/// Convenience helper to construct a `RecordSection` with normalized text metadata.
/// Convenience helper to build a `RecordSection` with precomputed sentences.
pub fn make_section(role: SectionRole, heading: Option<&str>, text: &str) -> RecordSection {
    RecordSection {
        role,
        heading: heading.map(|h| h.to_string()),
        text: text.to_string(),
        sentences: sentences(text),
    }
}

fn push_block_sentences(block: &str, results: &mut Vec<Sentence>) {
    let chars: Vec<char> = block.chars().collect();
    let mut buffer = String::new();

    for (idx, ch) in chars.iter().enumerate() {
        buffer.push(*ch);
        if is_sentence_boundary(&chars, idx) {
            let trimmed = buffer.trim();
            if !trimmed.is_empty() {
                results.push(trimmed.to_string());
            }
            buffer.clear();
        }
    }

    let trailing = buffer.trim();
    if !trailing.is_empty() {
        results.push(trailing.to_string());
    }
}

fn is_sentence_boundary(chars: &[char], idx: usize) -> bool {
    match chars[idx] {
        '.' => is_dot_boundary(chars, idx),
        '!' | '?' => true,
        _ => false,
    }
}

fn is_dot_boundary(chars: &[char], idx: usize) -> bool {
    if is_decimal_middle(chars, idx) || is_ticker_middle(chars, idx) {
        return false;
    }
    if idx + 1 < chars.len() && chars[idx + 1] == '.' {
        return false;
    }
    true
}

fn is_decimal_middle(chars: &[char], idx: usize) -> bool {
    idx > 0
        && idx + 1 < chars.len()
        && chars[idx - 1].is_ascii_digit()
        && chars[idx + 1].is_ascii_digit()
}

fn is_ticker_middle(chars: &[char], idx: usize) -> bool {
    idx > 0
        && idx + 1 < chars.len()
        && is_ticker_char(chars[idx - 1])
        && is_ticker_char(chars[idx + 1])
}

fn is_ticker_char(ch: char) -> bool {
    ch.is_ascii_uppercase() || ch.is_ascii_digit()
}

// ---------------------------------------------------------------------------
// Filesystem helpers
// ---------------------------------------------------------------------------

/// True if the path has a `.txt` extension (case-insensitive).
pub fn is_text_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("txt"))
        .unwrap_or(false)
}

/// Best-effort file modified time.
pub fn file_mtime(path: &Path) -> Option<DateTime<Utc>> {
    let metadata = fs::metadata(path).ok()?;
    let modified = metadata.modified().ok()?;
    Some(system_time_to_utc(modified))
}

/// Best-effort (created_at, updated_at) pair for a file.
pub fn file_times(path: &Path) -> (DateTime<Utc>, DateTime<Utc>) {
    let metadata = fs::metadata(path).ok();
    let updated_at = metadata
        .as_ref()
        .and_then(|meta| meta.modified().ok())
        .map(system_time_to_utc)
        .unwrap_or_else(Utc::now);
    let created_at = metadata
        .and_then(|meta| meta.created().ok())
        .map(system_time_to_utc)
        .unwrap_or(updated_at);
    (created_at, updated_at)
}

fn system_time_to_utc(time: std::time::SystemTime) -> DateTime<Utc> {
    DateTime::<Utc>::from(time)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_inline_whitespace_collapses_runs() {
        let input = "Alpha\n\n  Beta\tGamma";
        assert_eq!(normalize_inline_whitespace(input), "Alpha Beta Gamma");
    }

    #[test]
    fn sentences_falls_back_to_full_text_when_needed() {
        let text = "   \n";
        let result = sentences(text);
        assert!(result.is_empty());

        let text2 = "Single block without punctuation";
        let result2 = sentences(text2);
        assert_eq!(
            result2,
            vec![String::from("Single block without punctuation")]
        );
    }

    #[test]
    fn make_section_populates_sentences() {
        let section = make_section(SectionRole::Context, Some("Summary"), "Line one. Line two!");
        assert_eq!(section.heading.as_deref(), Some("Summary"));
        assert_eq!(section.sentences.len(), 2);
        assert_eq!(section.role, SectionRole::Context);
    }

    #[test]
    fn sentences_keep_decimal_values_together() {
        let text = "Price closed at 3.14. Outlook improved.";
        let result = sentences(text);
        assert_eq!(result, vec!["Price closed at 3.14.", "Outlook improved."]);
    }

    #[test]
    fn sentences_keep_dot_tickers_together() {
        let text = "BRK.B rallied while RDS.A lagged.";
        let result = sentences(text);
        assert_eq!(result, vec!["BRK.B rallied while RDS.A lagged."]);
    }

    #[test]
    fn file_time_helpers_handle_existing_and_missing_paths() {
        use tempfile::tempdir;
        let temp = tempdir().unwrap();
        let existing = temp.path().join("exists.txt");
        std::fs::write(&existing, "hello").unwrap();

        assert!(file_mtime(&existing).is_some());
        let (created_at, updated_at) = file_times(&existing);
        assert!(updated_at >= created_at);

        let missing = temp.path().join("missing.txt");
        assert!(file_mtime(&missing).is_none());
        let (missing_created, missing_updated) = file_times(&missing);
        assert!(missing_updated >= missing_created);
    }

    #[test]
    fn sentences_treat_blank_line_as_boundary() {
        let text = "First line without punctuation\n\nSecond line with more context.";
        let result = sentences(text);
        assert_eq!(
            result,
            vec![
                "First line without punctuation".to_string(),
                "Second line with more context.".to_string()
            ]
        );
    }

    #[test]
    fn sentences_keep_ellipsis_together() {
        let text = "Wait... really? Yes.";
        let result = sentences(text);
        assert_eq!(result, vec!["Wait...", "really?", "Yes."]);
    }

    #[test]
    fn is_text_file_matches_txt_case_insensitively() {
        use std::path::PathBuf;
        assert!(is_text_file(&PathBuf::from("hello.txt")));
        assert!(is_text_file(&PathBuf::from("hello.TXT")));
        assert!(is_text_file(&PathBuf::from("hello.Txt")));
        assert!(!is_text_file(&PathBuf::from("hello.md")));
        assert!(!is_text_file(&PathBuf::from("hello")));
    }
}
