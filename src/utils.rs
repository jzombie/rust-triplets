//! Text normalization helpers shared by source implementations.

use crate::data::{RecordSection, SectionRole};
use crate::types::Sentence;

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
}
