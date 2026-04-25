//! OCR denoising and markdown-formatting cleanup for text chunks.
//!
//! The main entry point is [`denoise_text`], which applies a configurable set of
//! line-level (or whole-block) filters to strip digit-heavy OCR noise and
//! markdown table formatting that is useless for text embeddings.
//!
//! For use in a preprocessing pipeline, wrap a [`crate::config::DenoiserConfig`]
//! in a [`DenoiserPreprocessor`] and register it with
//! [`crate::config::ChunkingStrategy::register_preprocessor`].

use line_ending::LineEnding;

use crate::config::DenoiserConfig;
use crate::preprocessor::TextPreprocessor;

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Count ASCII digit and alphabetical characters in `text`.
fn count_digit_alpha(text: &str) -> (usize, usize) {
    let mut digits = 0usize;
    let mut alpha = 0usize;
    for ch in text.chars() {
        if ch.is_ascii_digit() {
            digits += 1;
        } else if ch.is_alphabetic() {
            alpha += 1;
        }
    }
    (digits, alpha)
}

/// Compute the ratio `digits / (digits + alpha)` for `text`.
///
/// Returns `0.0` when there are no alphanumeric characters.
fn digit_ratio(text: &str) -> f32 {
    let (digits, alpha) = count_digit_alpha(text);
    let total = digits + alpha;
    if total == 0 {
        0.0
    } else {
        digits as f32 / total as f32
    }
}

/// Keep tokens from a digit-heavy line while preserving numeric context.
///
/// ## Algorithm: iterative wave expansion from alpha-token seeds
///
/// Complexity: $O(N^2)$ worst case (N waves × O(N) scan per wave), but in
/// practice $O(N)$ with a small constant — lines are short (10–40 tokens)
/// and the ratio threshold terminates expansion after 2–5 waves.
///
/// 1. **Seed** the keep-set with all alpha-bearing tokens.
/// 2. **Each wave** rescues the immediate neighbors (±1 position) of every
///    currently-kept token that are not yet kept.  Before committing the
///    wave, the combined digit-ratio of the *new candidate set*
///    (current keep ∪ wave) is checked.  If the ratio stays ≤
///    `max_digit_ratio` the wave is accepted and expansion continues;
///    otherwise the wave is rejected and expansion stops.
/// 3. Repeat until no new neighbors exist or a wave is rejected.
///
/// Any token type (bare numbers, `—`, `$12.5M`, `+3%`, …) is eligible for
/// rescue — the ratio check is the sole gate.  Pure-symbol tokens adjacent
/// to alpha tokens can therefore be rescued when they carry contextual
/// meaning (e.g. `—` used as a minus sign, `&` in a company name).
///
/// Returns the space-joined result; may be empty.
fn strip_digit_tokens(line: &str, max_digit_ratio: f32) -> String {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.is_empty() {
        return String::new();
    }
    let n = tokens.len();

    let has_alpha: Vec<bool> = tokens
        .iter()
        .map(|t| t.chars().any(|c| c.is_alphabetic()))
        .collect();

    // Seed: all alpha-bearing tokens.
    let mut keep: Vec<bool> = has_alpha.clone();

    // Pre-compute (digits, alpha) per token for ratio checks.
    let char_counts: Vec<(usize, usize)> = tokens.iter().map(|t| count_digit_alpha(t)).collect();

    // Running totals for the current keep-set.
    let (mut d, mut a) = (0usize, 0usize);
    for (i, &k) in keep.iter().enumerate() {
        if k {
            d += char_counts[i].0;
            a += char_counts[i].1;
        }
    }

    // Wave expansion: each iteration rescues ±1 neighbors of kept tokens.
    loop {
        // Collect the indices of tokens that would be added in this wave.
        let wave: Vec<usize> = (0..n)
            .filter(|&i| !keep[i] && ((i > 0 && keep[i - 1]) || (i + 1 < n && keep[i + 1])))
            .collect();

        if wave.is_empty() {
            break;
        }

        // Compute the ratio if we were to accept this wave.
        let (wd, wa): (usize, usize) = wave.iter().fold((0, 0), |(ad, aa), &i| {
            (ad + char_counts[i].0, aa + char_counts[i].1)
        });
        let new_d = d + wd;
        let new_a = a + wa;
        let new_total = new_d + new_a;
        let new_ratio = if new_total == 0 {
            0.0
        } else {
            new_d as f32 / new_total as f32
        };

        if new_ratio > max_digit_ratio {
            break; // This wave would push ratio over threshold — stop.
        }

        // Accept the wave.
        for &i in &wave {
            keep[i] = true;
        }
        d = new_d;
        a = new_a;
    }

    tokens
        .iter()
        .enumerate()
        .filter(|&(i, _)| keep[i])
        .map(|(_, t)| *t)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Returns `true` when `line` is a GFM pipe-table row.
///
/// A line qualifies when its trimmed form starts with `'|'` and contains at
/// least one additional `'|'` (i.e. a two-column table or a single column with
/// a closing delimiter).  This covers:
///
/// - Header rows: `| Name | Age |`
/// - Separator rows: `|------|-----|`, `|:----:|:---:|`
/// - Data rows: `| Alice | 30 |`
fn is_markdown_table_line(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.starts_with('|') && trimmed.matches('|').count() >= 2
}

/// Returns `true` when `line` is a GFM pipe-table separator row.
///
/// A separator row contains only `|`, `-`, `:`, and whitespace (e.g.
/// `|------|-----|` or `|:----:|:---:|`).  These rows carry no textual
/// content and should be dropped entirely.
fn is_markdown_table_separator(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.starts_with('|')
        && trimmed.matches('|').count() >= 2
        && trimmed
            .chars()
            .all(|c| c == '|' || c == '-' || c == ':' || c == ' ' || c == '\t')
}

/// Strip GFM pipe-table delimiters from a header or data row and return the
/// concatenated cell text.
///
/// Each `|`-delimited cell is trimmed; empty cells (from leading/trailing
/// pipes) are discarded.  The surviving cells are joined with a single space.
///
/// Example: `"| Name | Age |"` → `"Name Age"`
fn strip_table_pipes(line: &str) -> String {
    line.split('|')
        .map(|cell| cell.trim())
        .filter(|cell| !cell.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Apply OCR denoising and markdown-table stripping to a block of text.
///
/// Returns `Some(cleaned)` with the (possibly stripped) text, or `None` when
/// the entire block should be dropped and no chunks should be produced.
///
/// Line endings are first normalized with [`LineEnding::normalize`].  Each
/// line is then evaluated through three gates in order:
///
/// 1. **Markdown table formatting** — GFM pipe-table rows (trimmed form
///    starts with `'|'` and contains at least one more `'|'`) are handled in
///    two ways.  *Separator rows* (containing only `|`, `-`, `:`, and
///    whitespace) carry no textual content and are dropped.  *Header and data
///    rows* have their pipe delimiters stripped and the extracted cell text is
///    evaluated by gates 2 and 3 like any other line, preserving useful text
///    from inside table cells.
///
/// 2. **No alphabetical characters** — lines that contain zero alphabetical
///    characters (all-numeric rows, symbol/dash-only rows, OCR separator
///    artifacts) are dropped.
///
/// 3. **High digit ratio** — lines whose `digit / (digit + alpha)` ratio
///    exceeds `config.max_digit_ratio` are *stripped* using iterative wave
///    expansion from alpha-token seeds.  neighboring tokens are rescued
///    progressively outward as long as the cumulative ratio stays ≤
///    `max_digit_ratio`.  Any token type — numbers, `—`, `+3%`, `$12B` —
///    is eligible; the ratio check is the sole gate.  If no tokens survive,
///    the line is dropped.
///
/// `None` is returned only when every line is removed.
///
/// When `config.enabled` is `false` the function returns `Some(text.to_string())`
/// unconditionally.
pub fn denoise_text(text: &str, config: &DenoiserConfig) -> Option<String> {
    if !config.enabled {
        return Some(text.to_string());
    }

    let normalized = LineEnding::normalize(text);
    let mut cleaned_lines: Vec<String> = Vec::new();
    for line in normalized.lines() {
        // Gate 1: markdown formatting.
        // If `strip_markdown` is active, separator rows (containing only |, -, :,
        // and whitespace) carry no textual content and are dropped. Header and data
        // rows have their pipe delimiters stripped; the extracted cell text then
        // passes through gates 2 and 3 like any other line.
        let table_stripped = if config.strip_markdown && is_markdown_table_line(line) {
            if is_markdown_table_separator(line) {
                continue;
            }
            Some(strip_table_pipes(line))
        } else {
            None
        };
        let effective = table_stripped.as_deref().unwrap_or(line);

        // Gate 2: no alphabetical characters → drop (all-numeric lines,
        //         symbol-only rows, OCR column-separator artifacts, etc.).
        let (_, alpha) = count_digit_alpha(effective);
        if alpha == 0 {
            continue;
        }

        // Gate 3: digit-heavy line → iterative wave expansion to rescue
        //         adjacent tokens within the ratio budget.
        if digit_ratio(effective) > config.max_digit_ratio {
            let retained = strip_digit_tokens(effective, config.max_digit_ratio);
            if !retained.is_empty() {
                cleaned_lines.push(retained);
            }
            // else: drop the line entirely
        } else {
            cleaned_lines.push(effective.to_string());
        }
    }
    if cleaned_lines.is_empty() {
        None
    } else {
        Some(cleaned_lines.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// DenoiserPreprocessor
// ---------------------------------------------------------------------------

/// A [`TextPreprocessor`] that applies OCR denoising and markdown-table
/// cleanup to section text before chunking.
///
/// Wraps a [`DenoiserConfig`] and delegates to [`denoise_text`].
///
/// # Example
///
/// ```rust
/// use triplets::{ChunkingStrategy, DenoiserConfig, DenoiserPreprocessor};
///
/// let mut strategy = ChunkingStrategy::default();
/// strategy.register_preprocessor(DenoiserPreprocessor::new(DenoiserConfig {
///     enabled: true,
///     max_digit_ratio: 0.35,
///     strip_markdown: true,
/// }));
/// ```
pub struct DenoiserPreprocessor {
    /// Configuration controlling the denoising behaviour.
    pub config: DenoiserConfig,
}

impl DenoiserPreprocessor {
    /// Create a new `DenoiserPreprocessor` with the given configuration.
    pub fn new(config: DenoiserConfig) -> Self {
        Self { config }
    }
}

impl TextPreprocessor for DenoiserPreprocessor {
    fn process(&self, text: &str) -> Option<String> {
        denoise_text(text, &self.config)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use indoc::indoc;

    fn denoiser_enabled() -> DenoiserConfig {
        DenoiserConfig {
            enabled: true,
            max_digit_ratio: 0.35,
            strip_markdown: true,
        }
    }

    // -----------------------------------------------------------------------
    // DenoiserPreprocessor trait impl
    // -----------------------------------------------------------------------

    #[test]
    fn denoiser_preprocessor_process_delegates_to_denoise_text() {
        let p = DenoiserPreprocessor::new(DenoiserConfig {
            enabled: true,
            max_digit_ratio: 0.35,
            strip_markdown: true,
        });
        let noisy = "42 524 10788 143 1995 190 394";
        assert_eq!(
            p.process(noisy),
            denoise_text(noisy, &p.config),
            "process() must delegate to denoise_text"
        );
    }

    #[test]
    fn denoiser_preprocessor_disabled_returns_text_unchanged() {
        let p = DenoiserPreprocessor::new(DenoiserConfig::default()); // enabled = false
        let input = "42 524 NOVEX INDUSTRIES 10,788.0 14.3";
        assert_eq!(p.process(input), Some(input.to_string()));
    }

    #[test]
    fn denoiser_preprocessor_drops_digit_heavy_block() {
        let p = DenoiserPreprocessor::new(DenoiserConfig {
            enabled: true,
            max_digit_ratio: 0.35,
            strip_markdown: true,
        });
        assert_eq!(
            p.process("42 524 10788 143 1995 190 394 13611 358 6444 266"),
            None
        );
    }

    // -----------------------------------------------------------------------
    // is_markdown_table_line helper
    // -----------------------------------------------------------------------

    #[test]
    fn markdown_table_line_detects_separator_row() {
        assert!(is_markdown_table_line("|------|-----|"));
        assert!(is_markdown_table_line("|:----:|:---:|"));
        assert!(is_markdown_table_line(
            "|----------------|-----------|----------|"
        ));
        assert!(
            is_markdown_table_line("  |---|---|  "),
            "leading/trailing spaces ok"
        );
    }

    #[test]
    fn markdown_table_line_detects_header_and_data_rows() {
        assert!(is_markdown_table_line("| Name | Age |"));
        assert!(is_markdown_table_line("| Annual revenue | $4.2B | +12% |"));
        assert!(is_markdown_table_line("| Alice | 30 | NYC |"));
    }

    #[test]
    fn markdown_table_line_detects_single_column_with_closing_pipe() {
        assert!(is_markdown_table_line("| Value |"));
        assert!(is_markdown_table_line("|---|"));
    }

    #[test]
    fn markdown_table_line_rejects_prose_with_single_pipe() {
        assert!(!is_markdown_table_line("Choose option A | option B"));
        assert!(!is_markdown_table_line("See foo | bar for details."));
    }

    #[test]
    fn markdown_table_line_rejects_line_without_leading_pipe() {
        assert!(!is_markdown_table_line("Name | Age | City"));
        assert!(!is_markdown_table_line("--- | --- | ---"));
    }

    #[test]
    fn markdown_table_line_rejects_plain_text() {
        assert!(!is_markdown_table_line("The quick brown fox."));
        assert!(!is_markdown_table_line("42 524 NOVEX INDUSTRIES"));
        assert!(!is_markdown_table_line(""));
    }

    // -----------------------------------------------------------------------
    // is_markdown_table_separator helper
    // -----------------------------------------------------------------------

    #[test]
    fn markdown_table_separator_detects_dash_and_colon_rows() {
        assert!(is_markdown_table_separator("|------|-----|"));
        assert!(is_markdown_table_separator("|:----:|:---:|"));
        assert!(is_markdown_table_separator("|:-----|-----:|"));
        assert!(is_markdown_table_separator(
            "|----------------|-----------|----------|"
        ));
        assert!(
            is_markdown_table_separator("| ---- | ---- |"),
            "dashes with spaces ok"
        );
        assert!(
            is_markdown_table_separator("  |---|---|  "),
            "leading/trailing spaces ok"
        );
    }

    #[test]
    fn markdown_table_separator_rejects_rows_with_text_or_digits() {
        assert!(!is_markdown_table_separator("| Name | Age |"), "has alpha");
        assert!(
            !is_markdown_table_separator("| 2023 | $3.8B | +10% |"),
            "has digit"
        );
        assert!(
            !is_markdown_table_separator("| Annual revenue | $4.2B |"),
            "has alpha"
        );
    }

    // -----------------------------------------------------------------------
    // strip_table_pipes helper
    // -----------------------------------------------------------------------

    #[test]
    fn strip_table_pipes_extracts_cell_text() {
        assert_eq!(strip_table_pipes("| Name | Age |"), "Name Age");
        assert_eq!(
            strip_table_pipes("| Annual revenue | $4.2B | +12% |"),
            "Annual revenue $4.2B +12%"
        );
        assert_eq!(strip_table_pipes("| Widget A |"), "Widget A");
        assert_eq!(
            strip_table_pipes("| Metric         | Value     | Change  |"),
            "Metric Value Change"
        );
    }

    // -----------------------------------------------------------------------
    // Disabled / no-op
    // -----------------------------------------------------------------------

    #[test]
    fn denoise_disabled_returns_text_unchanged() {
        let cfg = DenoiserConfig::default();
        let input = "42 524 NOVEX INDUSTRIES 10,788.0 14.3";
        assert_eq!(denoise_text(input, &cfg), Some(input.to_string()));
    }

    #[test]
    fn denoise_disabled_leaves_markdown_table_unchanged() {
        let cfg = DenoiserConfig::default();
        let input = "| Name | Age |\n|------|-----|\n| Alice | 30 |";
        assert_eq!(denoise_text(input, &cfg), Some(input.to_string()));
    }

    #[test]
    fn denoise_enabled_but_strip_markdown_false_leaves_tables_intact() {
        let mut cfg = denoiser_enabled();
        cfg.strip_markdown = false;

        let input = indoc! {"
            | Name | Age |
            |------|-----|
            | Alice | 30 |
        "};
        let expected = "| Name | Age |\n| Alice | 30 |";
        assert_eq!(denoise_text(input.trim(), &cfg), Some(expected.to_string()));
    }

    #[test]
    fn denoise_enabled_with_strip_markdown_strips_tables_and_preserves_headings() {
        let cfg = denoiser_enabled();
        let input = indoc! {"
            ### User Demographics
            
            | Name | Age |
            |------|-----|
            | Alice | 30 |
            
            Some bold **text** and `code` here.
        "};
        let expected = indoc! {"
            ### User Demographics
            Name Age
            Alice 30
            Some bold **text** and `code` here.
        "}
        .trim();
        assert_eq!(denoise_text(input.trim(), &cfg), Some(expected.to_string()));
    }

    // -----------------------------------------------------------------------
    // Single digit-heavy line / below-threshold pass-through
    // -----------------------------------------------------------------------

    #[test]
    fn denoise_drops_digit_heavy_single_line() {
        let cfg = denoiser_enabled();
        assert_eq!(
            denoise_text("42 524 10788 143 1995 190 394 13611 358 6444 266", &cfg),
            None
        );
    }

    #[test]
    fn denoise_below_threshold_preserves_numbers_and_symbols() {
        let cfg = denoiser_enabled();
        let input = "Q3 revenue grew 12% to $4.2B, up from $3.8B in Q2 (a 10.5% increase).";
        assert_eq!(denoise_text(input, &cfg), Some(input.to_string()));
    }

    #[test]
    fn denoise_pure_text_passes_through() {
        let cfg = denoiser_enabled();
        let input = "The quick brown fox jumps over the lazy dog";
        assert_eq!(denoise_text(input, &cfg), Some(input.to_string()));
    }

    // -----------------------------------------------------------------------
    // Line-level: basic drop / pass-through
    // -----------------------------------------------------------------------

    #[test]
    fn denoise_empty_input_returns_none_when_enabled() {
        let cfg = denoiser_enabled();
        assert_eq!(denoise_text("", &cfg), None);
    }

    #[test]
    fn denoise_line_level_returns_none_when_all_lines_dropped() {
        let cfg = denoiser_enabled();
        let input = indoc! {"
            42 524 10788
            143 1995 190
            394 13611 358
        "};
        assert_eq!(denoise_text(input.trim(), &cfg), None);
    }

    #[test]
    fn denoise_line_level_preserves_clean_lines_unchanged() {
        let cfg = denoiser_enabled();
        let clean = "Climate change drives ocean temperatures higher each decade.";
        assert_eq!(
            denoise_text(clean, &cfg).expect("clean text should be kept"),
            clean
        );
    }

    #[test]
    fn denoise_line_level_below_threshold_line_preserves_numbers_and_symbols() {
        let cfg = denoiser_enabled();
        let input = "See section 3.1 (page 42) for details on the Q2 results.";
        assert_eq!(
            denoise_text(input, &cfg).expect("below-threshold line must be kept"),
            input
        );
    }

    #[test]
    fn denoise_line_level_clean_lines_with_numbers_preserved_junk_stripped() {
        let cfg = denoiser_enabled();
        let input = indoc! {"
            Revenue grew 8% to $2.1B in FY2025 (vs $1.9B prior year).
            42 9871 3302 19283 4710 22913 5518 30021 6627 41132 7736 52243
            Net income rose 15% YoY, reaching $310M by Q4-2025.
        "};
        let result = denoise_text(input.trim(), &cfg).expect("should not be None");
        assert!(result.contains("Revenue grew 8% to $2.1B in FY2025 (vs $1.9B prior year)."));
        assert!(result.contains("Net income rose 15% YoY, reaching $310M by Q4-2025."));
        assert!(!result.contains("9871"));
        assert_eq!(result.lines().count(), 2);
    }

    // -----------------------------------------------------------------------
    // Line-level: mixed content
    // -----------------------------------------------------------------------

    #[test]
    fn denoise_line_level_mixed_content_same_line() {
        let cfg = denoiser_enabled();
        let input = indoc! {"
            42 524 NOVEX INDUSTRIES Springfield 10788 143 1995 190 394 13611 358
            343 294 ZETA POWER Riverside 10758 31 1283 267 189 45432 175
        "};
        let result = denoise_text(input.trim(), &cfg).expect("should not be None");
        for word in &[
            "NOVEX",
            "INDUSTRIES",
            "Springfield",
            "ZETA",
            "POWER",
            "Riverside",
        ] {
            assert!(result.contains(word), "'{word}' must survive");
        }
        assert!(result.contains("524"));
        assert!(result.contains("10788"));
        assert!(result.contains("294"));
        assert!(result.contains("10758"));
        assert!(!result.contains("45432"));
        assert!(!result.contains("13611"));
        assert_eq!(result.lines().count(), 2);
    }

    #[test]
    fn denoise_line_level_drops_lines_with_no_alpha_tokens() {
        let cfg = denoiser_enabled();
        let input = indoc! {"
            42 524 10788 143 1995
            — — (0.8) (203.5) 473
            NOVEX INDUSTRIES Springfield
        "};
        let result =
            denoise_text(input.trim(), &cfg).expect("should not be None — text line survives");
        assert!(result.contains("NOVEX"));
        assert!(!result.contains("10788"));
        assert!(!result.contains("(0.8)"));
        assert_eq!(result.lines().count(), 1);
    }

    #[test]
    fn denoise_line_level_retains_text_from_mixed_line() {
        let cfg = denoiser_enabled();
        let input = "42 524 NOVEX INDUSTRIES Springfield 10788 143 1995 190 394 13611 358 6444 266";
        let result = denoise_text(input, &cfg).expect("should not be None");
        for word in &["NOVEX", "INDUSTRIES", "Springfield"] {
            assert!(result.contains(word), "'{word}' must survive");
        }
        assert!(result.contains("10788"));
        assert!(!result.contains("13611"));
    }

    // -----------------------------------------------------------------------
    // Interleaved sequences
    // -----------------------------------------------------------------------

    #[test]
    fn denoise_line_level_text_sandwiched_between_junk_tokens() {
        let cfg = denoiser_enabled();
        let input = "42 NOVEX 524 INDUSTRIES 10788 143 1995 190";
        let result = denoise_text(input, &cfg).expect("should not be None");
        assert!(result.contains("NOVEX"));
        assert!(result.contains("INDUSTRIES"));
        assert!(!result.contains("10788"));
        assert!(!result.contains("524"));
    }

    #[test]
    fn denoise_line_level_repeated_junk_text_interleaving() {
        let cfg = denoiser_enabled();
        let input = "42 ZETA 524 POWER 10758 Riverside 31 GRID 1283 GROUP 267 Holdings 45432 Corp";
        let result = denoise_text(input, &cfg).expect("should not be None");
        let text_tokens = [
            "ZETA",
            "POWER",
            "Riverside",
            "GRID",
            "GROUP",
            "Holdings",
            "Corp",
        ];
        let num_tokens = ["42", "524", "10758", "31", "1283", "267", "45432"];
        for word in &text_tokens {
            assert!(result.contains(word));
        }
        for num in &num_tokens {
            assert!(!result.contains(num));
        }
        let mut last_pos = 0usize;
        for word in &text_tokens {
            let pos = result.find(word).unwrap();
            assert!(pos >= last_pos);
            last_pos = pos;
        }
    }

    // -----------------------------------------------------------------------
    // Symbol-heavy edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn denoise_line_level_parenthesized_negatives_and_dashes_stripped() {
        let cfg = denoiser_enabled();
        let input = "345 397 DELTA CORP Detroit, Mich. 10689 (0.8) 1069 302 — 18214 336 17590 182";
        let result = denoise_text(input, &cfg).expect("should not be None");
        assert!(result.contains("DELTA"));
        assert!(result.contains("CORP"));
        assert!(result.contains("Detroit"));
        assert!(result.contains("Mich."));
        assert!(result.contains("397"));
        assert!(result.contains("10689"));
        assert!(!result.contains("(0.8)"));
        assert!(!result.contains("18214"));
        assert_eq!(result, "397 DELTA CORP Detroit, Mich. 10689");
    }

    #[test]
    fn denoise_line_level_comma_formatted_numbers_stripped() {
        let cfg = denoiser_enabled();
        let input =
            "42 524 NOVEX INDUSTRIES Springfield 10,788.0 14.3 1,995.0 190 39.4 13,611.0 358";
        let result = denoise_text(input, &cfg).expect("should not be None");
        for word in &["NOVEX", "INDUSTRIES", "Springfield"] {
            assert!(result.contains(word));
        }
        assert!(result.contains("10,788.0"));
        for num in &["1,995.0", "13,611.0"] {
            assert!(!result.contains(num));
        }
    }

    #[test]
    fn denoise_neighbor_rescue_falls_back_when_ratio_still_exceeds_threshold() {
        let cfg = denoiser_enabled();
        let input = "1234 word 5678";
        let result = denoise_text(input, &cfg).expect("should not be None");
        assert_eq!(result, "word");
        assert!(!result.contains("1234"));
        assert!(!result.contains("5678"));
    }

    #[test]
    fn denoise_line_level_symbol_only_line_is_dropped() {
        let cfg = denoiser_enabled();
        assert_eq!(denoise_text("— — — (0.8) (203.5) 473 42 524", &cfg), None);
    }

    #[test]
    fn denoise_line_level_ordinal_tokens_are_kept() {
        let cfg = denoiser_enabled();
        let input = "3rd Quarter performance review 2nd half summary";
        let result = denoise_text(input, &cfg).expect("should not be None");
        assert!(result.contains("3rd"));
        assert!(result.contains("2nd"));
        assert!(result.contains("Quarter"));
    }

    #[test]
    fn denoise_line_level_dense_interleave_with_symbols() {
        let cfg = denoiser_enabled();
        let input = "42 (524) ZETA 10,758.0 — POWER 31.5 Riverside, 1283 Corp.";
        let result = denoise_text(input, &cfg).expect("should not be None");
        for word in &["ZETA", "POWER", "Riverside,", "Corp."] {
            assert!(result.contains(word));
        }
        for junk in &["42", "(524)", "10,758.0", "1283"] {
            assert!(!result.contains(junk));
        }
        assert_eq!(result, "ZETA POWER Riverside, Corp.");
    }

    #[test]
    fn denoise_line_level_multiple_em_dashes_all_stripped() {
        let cfg = denoiser_enabled();
        let input = "— 42 NOVEX — 524 INDUSTRIES — 10789 —";
        let result = denoise_text(input, &cfg).expect("should not be None");
        assert!(!result.contains("10789"));
        assert!(result.contains("42"));
        assert!(result.contains("524"));
        assert_eq!(result, "42 NOVEX — 524 INDUSTRIES —");
    }

    #[test]
    fn denoise_line_level_multiple_parenthesized_values_rescued() {
        let cfg = denoiser_enabled();
        let input = "(0.8) NOVEX (1.2) INDUSTRIES (3.4) 10789";
        let result = denoise_text(input, &cfg).expect("should not be None");
        assert!(result.contains("(0.8)"));
        assert!(result.contains("(1.2)"));
        assert!(result.contains("(3.4)"));
        assert!(!result.contains("10789"));
        assert_eq!(result, "(0.8) NOVEX (1.2) INDUSTRIES (3.4)");
    }

    #[test]
    fn denoise_line_level_mixed_symbol_trash_repeated() {
        let cfg = denoiser_enabled();
        let input = "— (0.8) 42 ZETA — (1.5) 524 POWER — (2.3) 10758 Corp —";
        let result = denoise_text(input, &cfg).expect("should not be None");
        assert!(!result.contains('—'));
        assert!(!result.contains("(0.8)"));
        assert!(!result.contains("(1.5)"));
        assert!(!result.contains("(2.3)"));
        assert!(!result.contains("10758"));
        assert_eq!(result, "ZETA POWER Corp");
    }

    #[test]
    fn denoise_line_level_multiple_symbol_trash_multiline_exact_output() {
        let cfg = denoiser_enabled();
        let input = indoc! {"
            — 42 NOVEX — 524 INDUSTRIES — 10789 —
            (0.8) ZETA (1.2) POWER (3.4) 10758
        "};
        let result = denoise_text(input.trim(), &cfg).expect("should not be None");
        assert_eq!(result, "42 NOVEX — 524 INDUSTRIES —\nZETA POWER");
    }

    // -----------------------------------------------------------------------
    // Markdown table handling
    // -----------------------------------------------------------------------

    #[test]
    fn denoise_line_level_pure_markdown_table_separator_dropped_text_extracted() {
        let cfg = denoiser_enabled();
        let input = indoc! {"
            | Metric         | Value     | Change  |
            |----------------|-----------|----------|
            | Annual revenue | $4.2B     | +12%     |
            | Operating cost | $2.1B     | +8%      |
            | Net income     | $310M     | +15%     |
        "};
        let result = denoise_text(input.trim(), &cfg).expect("cell text should survive");
        assert_eq!(
            result,
            "Metric Value Change\nAnnual revenue $4.2B +12%\nOperating cost $2.1B +8%\nNet income $310M +15%"
        );
    }

    #[test]
    fn denoise_line_level_single_markdown_table_row_pipes_stripped() {
        let cfg = denoiser_enabled();
        assert_eq!(
            denoise_text("|----------------|-----------|----------|", &cfg),
            None
        );
        assert_eq!(
            denoise_text("| Metric | Value | Change |", &cfg),
            Some("Metric Value Change".to_string())
        );
        assert_eq!(
            denoise_text("| Annual revenue | $4.2B | +12% |", &cfg),
            Some("Annual revenue $4.2B +12%".to_string())
        );
    }

    #[test]
    fn denoise_line_level_markdown_table_embedded_in_prose() {
        let cfg = denoiser_enabled();
        let input = indoc! {"
            Revenue grew steadily over the past three fiscal years.
            | Year | Revenue | Growth |
            |------|---------|--------|
            | 2023 | $3.8B   | +10%   |
            | 2024 | $4.2B   | +12%   |
            Management expects the trend to continue.
        "};
        let result = denoise_text(input.trim(), &cfg).expect("should not be None");
        assert!(result.contains("Revenue grew"));
        assert!(result.contains("Management expects"));
        assert!(result.contains("Year Revenue Growth"));
        assert!(!result.contains("---|"));
        assert_eq!(result.lines().count(), 5);
    }

    #[test]
    fn denoise_line_level_markdown_table_various_separator_styles() {
        let cfg = denoiser_enabled();
        let input = indoc! {"
            Only this prose line should survive.
            |------|------|
            |:----:|:----:|
            |:-----|-----:|
            | ---- | ---- |
        "};
        let result = denoise_text(input.trim(), &cfg).expect("should not be None");
        assert!(result.contains("Only this prose line"));
        assert!(!result.contains("---"));
        assert_eq!(result.lines().count(), 1);
    }

    #[test]
    fn denoise_line_level_markdown_table_numeric_cells_dropped() {
        let cfg = denoiser_enabled();
        let input = indoc! {"
            | ID   | Score | Rank |
            |------|-------|------|
            | 1001 | 98.5  | 1    |
            | 1002 | 87.3  | 2    |
            | 1003 | 76.0  | 3    |
        "};
        let result = denoise_text(input.trim(), &cfg).expect("header row text must survive");
        assert_eq!(result, "ID Score Rank");
    }

    #[test]
    fn denoise_line_level_markdown_table_single_column() {
        let cfg = denoiser_enabled();
        let input = indoc! {"
            Plain sentence before the table.
            | Item       |
            |------------|
            | Widget A   |
            | Widget B   |
            Plain sentence after the table.
        "};
        let result = denoise_text(input.trim(), &cfg).expect("prose and cell text must survive");
        assert!(result.contains("Plain sentence before"));
        assert!(result.contains("Plain sentence after"));
        assert!(result.contains("Item"));
        assert!(result.contains("Widget A"));
        assert!(result.contains("Widget B"));
        assert!(!result.contains("---"));
        assert!(!result.contains('|'));
        assert_eq!(result.lines().count(), 5);
    }

    #[test]
    fn denoise_line_level_single_pipe_in_prose_is_not_a_table_row() {
        let cfg = denoiser_enabled();
        let input = "Use the syntax foo | bar to combine options.";
        let result = denoise_text(input, &cfg).expect("should not be None");
        assert_eq!(result, input);
    }

    #[test]
    fn denoise_line_level_borderless_table_separator_dropped_data_survives() {
        let cfg = denoiser_enabled();
        let input = indoc! {"
            Name | Age | City
            -----|-----|------
            Alice | 30 | Denver
            Bob | 42 | Tulsa
        "};
        let result = denoise_text(input.trim(), &cfg).expect("should not be None");
        assert!(!result.contains("-----"));
        assert!(result.contains("Name"));
        assert!(result.contains("Alice"));
        assert!(result.contains("Bob"));
    }

    // -----------------------------------------------------------------------
    // Full OCR table block
    // -----------------------------------------------------------------------

    #[test]
    fn denoise_full_table_block_retains_company_names() {
        let cfg = denoiser_enabled();
        let input = indoc! {"
            42 524 NOVEX INDUSTRIES Springfield 10788 143 1995 190 394 13611 358 6444 266
            343 294 ZETA POWER Riverside 10758 31 1283 267 189 45432 175 8675 235
            344 442 OCEAN FORGE Denver 10707 699 3910 403 13380 361 5851 285
            345 397 DELTA FINANCIAL Detroit 10689 8 1069 302 1820 18214 336 17590 182
            346 397 APEX HOLDINGS Brentwood 10648 458 2035 473 6728 450 8775 450
            347 379 VEGA SYSTEMS Tulsa 10627 377 1517 231 6190 4672 172 11423 194
            348 225 CREST BRANDS Atlanta 10589 128 5720 369 14179 349 4091 328
            349 555 TITAN CHEMICAL Kingsport 10476 236 8570 332 793 15159 334 5704 288
            350 540 AIR PRODUCTS & LOGISTICS Allentown 10323 166 20991 182 113 26859 252 13539 169
            351 399 NORTHLAND FINANCIAL FOR MEMBERS Minneapolis 10312 265 25302 155 2972 116524 79 13694 165
        "};
        let result = denoise_text(input.trim(), &cfg).expect("block should not be dropped");
        for name in &[
            "NOVEX",
            "INDUSTRIES",
            "ZETA",
            "POWER",
            "OCEAN",
            "FORGE",
            "DELTA",
            "FINANCIAL",
            "APEX",
            "HOLDINGS",
            "VEGA",
            "SYSTEMS",
            "CREST",
            "BRANDS",
            "TITAN",
            "CHEMICAL",
            "AIR",
            "PRODUCTS",
            "LOGISTICS",
            "NORTHLAND",
            "MEMBERS",
        ] {
            assert!(result.contains(name));
        }
        for loc in &[
            "Springfield",
            "Riverside",
            "Denver",
            "Detroit",
            "Brentwood",
            "Tulsa",
            "Atlanta",
            "Kingsport",
            "Allentown",
            "Minneapolis",
        ] {
            assert!(result.contains(loc));
        }
        for junk in &["45432", "13539", "116524"] {
            assert!(!result.contains(junk));
        }
        assert!(result.contains("PRODUCTS & LOGISTICS"));
        assert!(result.contains("10788"));
        assert!(result.contains("10312"));
        assert_eq!(result.lines().count(), input.trim().lines().count());
        assert_eq!(
            result,
            indoc! {"
                42 524 NOVEX INDUSTRIES Springfield 10788 143
                294 ZETA POWER Riverside 10758
                442 OCEAN FORGE Denver 10707
                397 DELTA FINANCIAL Detroit 10689
                397 APEX HOLDINGS Brentwood 10648
                379 VEGA SYSTEMS Tulsa 10627
                225 CREST BRANDS Atlanta 10589
                555 TITAN CHEMICAL Kingsport 10476
                350 540 AIR PRODUCTS & LOGISTICS Allentown 10323 166
                351 399 NORTHLAND FINANCIAL FOR MEMBERS Minneapolis 10312 265 25302"
            }
        );
    }

    // -----------------------------------------------------------------------
    // Financial punctuation: +, -, —, $, %
    // -----------------------------------------------------------------------

    #[test]
    fn denoise_financial_punctuation_below_threshold_passes_through_unchanged() {
        let cfg = denoiser_enabled();
        let input = "Operating cash: $4.2B (+12% YoY) — net debt fell -$1.1B; margin: 23%.";
        let result = denoise_text(input, &cfg).expect("should not be None");
        assert_eq!(result, input);
    }

    #[test]
    fn denoise_em_dash_operator_on_gate3_line_is_rescued() {
        let cfg = denoiser_enabled();
        let input = "REVENUE — COSTS NET 42 524 10788";
        let result = denoise_text(input, &cfg).expect("should not be None");
        assert!(result.contains('—'));
        assert!(result.contains("42"));
        assert!(result.contains("524"));
        assert!(!result.contains("10788"));
        assert_eq!(result, "REVENUE — COSTS NET 42 524");
    }

    #[test]
    fn denoise_sign_percent_tokens_on_gate3_line_are_rescued() {
        let cfg = denoiser_enabled();
        let input = "REVENUE GROWTH +12% EARNINGS -8% COSTS 42 524 10788 5520 3918";
        let result = denoise_text(input, &cfg).expect("should not be None");
        assert!(result.contains("+12%"));
        assert!(result.contains("-8%"));
        assert!(result.contains("42"));
        assert!(result.contains("524"));
        assert!(result.contains("10788"));
        assert!(!result.contains("5520"));
        assert!(!result.contains("3918"));
        assert_eq!(
            result,
            "REVENUE GROWTH +12% EARNINGS -8% COSTS 42 524 10788"
        );
    }

    // -----------------------------------------------------------------------
    // Linearized XBRL passthrough
    // -----------------------------------------------------------------------

    #[test]
    fn linearized_xbrl_single_metric_line_passes_through_unchanged() {
        let line = concat!(
            "label=Net income | dir=up | traj=non_monotonic | path=mostly_upward",
            " | recent=up_bias | reg=growth_with_resets | cons=erratic | turn=high_turn",
            " | run=clustered_runs | end=recovering_off_peak | rec=weak_recovery | dd=extreme",
            " | shock=repeated_shock | pol=upside_shocks | flip=false | sig=UUDUUUDU-t4",
            " | first=30.00B | last=42.10B | filing_quality=score=100.0 grade=A transitions=12",
            " scale_issues=0 uom_issues=0",
        );
        let cfg = denoiser_enabled();
        assert_eq!(denoise_text(line, &cfg), Some(line.to_string()));
    }

    #[test]
    fn linearized_xbrl_full_aapl_block_content_preserved() {
        let input = indoc! {"
            ### AAPL
            periods=2026Q1,2025Q4,2025Q3,2025Q2,2025Q1,2024Q4,2024Q3,2024Q2,2024Q1,2023Q4,2023Q3,2023Q2,2023Q1

            label=Net income | dir=up | traj=non_monotonic | path=mostly_upward | recent=up_bias | reg=growth_with_resets | cons=erratic | turn=high_turn | run=clustered_runs | end=recovering_off_peak | rec=weak_recovery | dd=extreme | shock=repeated_shock | pol=upside_shocks | flip=false | sig=UUDUUUDU-t4 | first=30.00B | last=42.10B | filing_quality=score=100.0 grade=A transitions=12 scale_issues=0 uom_issues=0

            label=Operating income | dir=up | traj=non_monotonic | path=mostly_upward | recent=up_bias | reg=growth_with_resets | cons=erratic | turn=high_turn | run=clustered_runs | end=recovering_off_peak | rec=weak_recovery | dd=severe | shock=repeated_shock | pol=upside_shocks | flip=false | sig=UUDUUUDU-t4 | first=36.02B | last=50.85B | filing_quality=score=100.0 grade=A transitions=12 scale_issues=0 uom_issues=0"
        };
        let cfg = denoiser_enabled();
        assert_eq!(
            denoise_text(input, &cfg),
            Some(indoc! {"
                ### AAPL
                periods=2026Q1,2025Q4,2025Q3,2025Q2,2025Q1,2024Q4,2024Q3,2024Q2,2024Q1,2023Q4,2023Q3,2023Q2,2023Q1
                label=Net income | dir=up | traj=non_monotonic | path=mostly_upward | recent=up_bias | reg=growth_with_resets | cons=erratic | turn=high_turn | run=clustered_runs | end=recovering_off_peak | rec=weak_recovery | dd=extreme | shock=repeated_shock | pol=upside_shocks | flip=false | sig=UUDUUUDU-t4 | first=30.00B | last=42.10B | filing_quality=score=100.0 grade=A transitions=12 scale_issues=0 uom_issues=0
                label=Operating income | dir=up | traj=non_monotonic | path=mostly_upward | recent=up_bias | reg=growth_with_resets | cons=erratic | turn=high_turn | run=clustered_runs | end=recovering_off_peak | rec=weak_recovery | dd=severe | shock=repeated_shock | pol=upside_shocks | flip=false | sig=UUDUUUDU-t4 | first=36.02B | last=50.85B | filing_quality=score=100.0 grade=A transitions=12 scale_issues=0 uom_issues=0"
            }.to_string())
        );
    }
}
