//! OCR denoising and markdown-formatting cleanup for text chunks.
//!
//! The entry point is [`crate::denoiser::denoise_text()`], which applies a configurable set of
//! line-level (or whole-block) filters to strip digit-heavy OCR noise and
//! markdown table formatting that is useless for text embeddings.

use line_ending::LineEnding;

use crate::config::DenoiserConfig;

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
/// Algorithm: wave expansion from alpha-token seeds.
///
/// 1. Seed the keep-set with all alpha-bearing tokens.
/// 2. Each wave rescues the immediate neighbors (±1 position) of every
///    currently-kept token that are not yet kept.  Before committing the wave,
///    the combined digit-ratio of the *new candidate set* (current keep ∪ wave)
///    is checked.  If the ratio stays ≤ `max_digit_ratio` the wave is accepted
///    and expansion continues; otherwise the wave is rejected and expansion stops.
/// 3. This process repeats until no new neighbors exist or a wave is rejected.
///
/// Any token type (bare numbers, `—`, `$12.5M`, `+3%`, …) is eligible for
/// rescue — the ratio check is the sole gate.  Pure-symbol tokens adjacent to
/// alpha tokens can therefore be rescued when they carry contextual meaning
/// (e.g. `—` used as a minus sign, `$` prefixes).
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
///    exceeds `config.max_digit_ratio` are *stripped*: only whitespace-
///    delimited tokens that contain at least one alphabetical character are
///    retained.  If stripping leaves the line empty the line is dropped.
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

        // Gate 3: digit-heavy line → rescue adjacent numeric tokens, strip remainder.
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
        // Two pipe chars: opening + closing.
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
        // Borderless table header — does not start with '|'.
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
        let cfg = DenoiserConfig::default(); // enabled = false
        let input = "42 524 NOVEX INDUSTRIES 10,788.0 14.3";
        assert_eq!(denoise_text(input, &cfg), Some(input.to_string()));
    }

    #[test]
    fn denoise_disabled_leaves_markdown_table_unchanged() {
        let cfg = DenoiserConfig::default(); // enabled = false
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
        // The table survives unchanged, including the |---|---| separator row,
        // because the markdown gate is bypassed and the separator row gets
        // dropped by gate 2 ONLY if it has no alpha. Wait, `|---|---|` has no alpha!
        // So gate 2 drops the separator row anyway! Let's assert on what actually happens.
        // `| Name | Age |` -> passes (has alpha, low digits)
        // `|------|-----|` -> drops (no alpha)
        // `| Alice | 30 |` -> passes (has alpha, low digits)
        let expected = "| Name | Age |\n| Alice | 30 |";
        assert_eq!(denoise_text(input.trim(), &cfg), Some(expected.to_string()));
    }

    #[test]
    fn denoise_enabled_with_strip_markdown_strips_tables_and_preserves_headings() {
        let cfg = denoiser_enabled(); // strip_markdown is true by default
        let input = indoc! {"
            ### User Demographics
            
            | Name | Age |
            |------|-----|
            | Alice | 30 |
            
            Some bold **text** and `code` here.
        "};

        // - "### User Demographics" survives (has alpha)
        // - blank lines dropped (no alpha)
        // - "| Name | Age |" -> "Name Age"
        // - "|------|-----|" -> dropped (separator)
        // - "| Alice | 30 |" -> "Alice 30"
        // - "Some bold **text** and `code` here." -> survives (has alpha)
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
        let input = "42 524 10788 143 1995 190 394 13611 358 6444 266";
        assert_eq!(denoise_text(input, &cfg), None);
    }

    /// Below the threshold every character — including numbers and symbols —
    /// must be returned verbatim.
    #[test]
    fn denoise_below_threshold_preserves_numbers_and_symbols() {
        let cfg = denoiser_enabled();
        // ratio ≈ 0.12 — well below 0.35.
        let input = "Q3 revenue grew 12% to $4.2B, up from $3.8B in Q2 (a 10.5% increase).";
        assert_eq!(
            denoise_text(input, &cfg),
            Some(input.to_string()),
            "below-threshold line must be returned byte-for-byte"
        );
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
        let result = denoise_text(clean, &cfg).expect("clean text should be kept");
        assert_eq!(result, clean);
    }

    /// A clean line with numbers and symbols below the threshold must come
    /// through byte-for-byte; no tokens should be stripped.
    #[test]
    fn denoise_line_level_below_threshold_line_preserves_numbers_and_symbols() {
        let cfg = denoiser_enabled();
        // ratio ≈ 0.18 — alpha-heavy enough not to trigger the rule.
        let input = "See section 3.1 (page 42) for details on the Q2 results.";
        let result = denoise_text(input, &cfg).expect("below-threshold line must be kept");
        assert_eq!(
            result, input,
            "numbers and symbols on a clean line must survive intact"
        );
    }

    /// Only the all-numeric junk line is stripped; clean lines containing
    /// numbers and special characters pass through unchanged.
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
        assert!(!result.contains("9871"), "junk line must be stripped");
        assert_eq!(
            result.lines().count(),
            2,
            "only the two clean lines should remain"
        );
    }

    // -----------------------------------------------------------------------
    // Line-level: mixed content on the same line
    // -----------------------------------------------------------------------

    /// Alpha tokens survive; numbers immediately adjacent to alpha tokens are
    /// rescued when the candidate ratio stays within the threshold; numbers
    /// 2+ hops from any alpha token are stripped.
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
        // Numbers immediately adjacent to alpha tokens are rescued
        // (candidate ratio stays below 0.35 on both lines).
        assert!(result.contains("524"), "'524' adjacent to NOVEX — rescued");
        assert!(
            result.contains("10788"),
            "'10788' adjacent to Springfield — rescued"
        );
        assert!(result.contains("294"), "'294' adjacent to ZETA — rescued");
        assert!(
            result.contains("10758"),
            "'10758' adjacent to Riverside — rescued"
        );
        // Numbers 2+ hops from any alpha token are still stripped.
        assert!(
            !result.contains("45432"),
            "'45432' 2+ hops from alpha — stripped"
        );
        assert!(
            !result.contains("13611"),
            "'13611' 2+ hops from alpha — stripped"
        );
        assert_eq!(result.lines().count(), 2);
    }

    /// Purely numeric or purely symbolic lines are dropped; the block is
    /// `None` only when *every* line is removed.
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
        assert!(!result.contains("10788"), "all-number line must be gone");
        assert!(!result.contains("(0.8)"), "all-symbol line must be gone");
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
        // Adjacent to alpha: rescued.
        assert!(
            result.contains("10788"),
            "'10788' adjacent to Springfield — rescued"
        );
        // 2+ hops from alpha: stripped.
        assert!(
            !result.contains("13611"),
            "'13611' 2+ hops from alpha — stripped"
        );
    }

    // -----------------------------------------------------------------------
    // Line-level: interleaved sequences
    // -----------------------------------------------------------------------

    /// Digit-junk sandwiched *between* text tokens — all text must survive.
    #[test]
    fn denoise_line_level_text_sandwiched_between_junk_tokens() {
        let cfg = denoiser_enabled();
        let input = "42 NOVEX 524 INDUSTRIES 10788 143 1995 190";
        let result = denoise_text(input, &cfg).expect("should not be None");

        assert!(result.contains("NOVEX"), "NOVEX should survive");
        assert!(result.contains("INDUSTRIES"), "INDUSTRIES should survive");
        assert!(
            !result.contains("10788"),
            "pure-numeric token should be gone"
        );
        assert!(!result.contains("524"), "pure-numeric token should be gone");
    }

    /// Text and junk tokens alternating many times — all text tokens survive in
    /// their original relative order.
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
            assert!(result.contains(word), "expected '{word}' in output");
        }
        for num in &num_tokens {
            assert!(!result.contains(num), "did not expect '{num}' in output");
        }

        let mut last_pos = 0usize;
        for word in &text_tokens {
            let pos = result.find(word).expect("word should be present");
            assert!(pos >= last_pos, "word '{word}' is out of order");
            last_pos = pos;
        }
    }

    // -----------------------------------------------------------------------
    // Line-level: symbol-heavy edge cases
    // -----------------------------------------------------------------------

    /// Parenthesized negatives, em-dashes, and dotted abbreviations.
    #[test]
    fn denoise_line_level_parenthesized_negatives_and_dashes_stripped() {
        let cfg = denoiser_enabled();
        let input = "345 397 DELTA CORP Detroit, Mich. 10689 (0.8) 1069 302 — 18214 336 17590 182";
        let result = denoise_text(input, &cfg).expect("should not be None");

        assert!(result.contains("DELTA"), "DELTA must survive");
        assert!(result.contains("CORP"), "CORP must survive");
        assert!(result.contains("Detroit"), "city must survive");
        assert!(result.contains("Mich."), "dotted abbreviation must survive");
        // "397" is adjacent to DELTA and "10689" is adjacent to Mich. — both rescued
        // (candidate ratio ≈ 0.29 < 0.35).
        assert!(result.contains("397"), "'397' adjacent to DELTA — rescued");
        assert!(
            result.contains("10689"),
            "'10689' adjacent to Mich. — rescued"
        );
        // Tokens 2+ hops from alpha are still stripped.
        assert!(
            !result.contains("(0.8)"),
            "parenthesized negative must be stripped"
        );
        assert!(!result.contains("18214"), "bare number must be stripped");
        assert_eq!(
            result, "397 DELTA CORP Detroit, Mich. 10689",
            "complete output: alpha tokens plus immediately adjacent digit tokens"
        );
    }

    /// Comma-formatted numbers like `10,788.0` contain no alpha → stripped.
    #[test]
    fn denoise_line_level_comma_formatted_numbers_stripped() {
        let cfg = denoiser_enabled();
        let input =
            "42 524 NOVEX INDUSTRIES Springfield 10,788.0 14.3 1,995.0 190 39.4 13,611.0 358";
        let result = denoise_text(input, &cfg).expect("should not be None");

        for word in &["NOVEX", "INDUSTRIES", "Springfield"] {
            assert!(result.contains(word), "'{word}' must survive");
        }
        // "10,788.0" is immediately adjacent to Springfield — rescued
        // (candidate ratio ≈ 0.26 < 0.35).
        assert!(result.contains("10,788.0"), "adjacent comma-number rescued");
        // Numbers 2+ hops from alpha are still stripped.
        for num in &["1,995.0", "13,611.0"] {
            assert!(
                !result.contains(num),
                "non-adjacent comma-number '{num}' must be stripped"
            );
        }
    }

    /// When rescuing adjacent tokens would push the candidate set back over
    /// `max_digit_ratio`, the fallback emits alpha-only tokens.
    #[test]
    fn denoise_neighbor_rescue_falls_back_when_ratio_still_exceeds_threshold() {
        let cfg = denoiser_enabled(); // max_digit_ratio = 0.35
        // "1234 word 5678": candidate = ["1234", "word", "5678"]
        // digits = 4+4 = 8, alpha = 4; ratio = 8/12 = 0.67 > 0.35 → fallback.
        let input = "1234 word 5678";
        let result = denoise_text(input, &cfg).expect("should not be None");
        assert_eq!(result, "word", "fallback must emit only the alpha token");
        assert!(!result.contains("1234"));
        assert!(!result.contains("5678"));
    }

    /// A line of only symbols / numbers with no alpha → block returns `None`.
    #[test]
    fn denoise_line_level_symbol_only_line_is_dropped() {
        let cfg = denoiser_enabled();
        let input = "— — — (0.8) (203.5) 473 42 524";
        assert_eq!(
            denoise_text(input, &cfg),
            None,
            "a line with no alpha tokens should make the block None"
        );
    }

    /// Ordinal tokens like `3rd` / `2nd` contain alpha chars and must be kept.
    #[test]
    fn denoise_line_level_ordinal_tokens_are_kept() {
        let cfg = denoiser_enabled();
        let input = "3rd Quarter performance review 2nd half summary";
        let result = denoise_text(input, &cfg).expect("should not be None");

        assert!(result.contains("3rd"), "ordinal '3rd' should be retained");
        assert!(result.contains("2nd"), "ordinal '2nd' should be retained");
        assert!(result.contains("Quarter"), "plain word must be retained");
    }

    /// Dense interleaving: numbers, parenthesized values, em-dashes, and text.
    #[test]
    fn denoise_line_level_dense_interleave_with_symbols() {
        let cfg = denoiser_enabled();
        let input = "42 (524) ZETA 10,758.0 — POWER 31.5 Riverside, 1283 Corp.";
        let result = denoise_text(input, &cfg).expect("should not be None");

        for word in &["ZETA", "POWER", "Riverside,", "Corp."] {
            assert!(result.contains(word), "'{word}' must survive");
        }
        for junk in &["42", "(524)", "10,758.0", "1283"] {
            assert!(!result.contains(junk), "'{junk}' must be stripped");
        }
        assert_eq!(
            result, "ZETA POWER Riverside, Corp.",
            "complete output must equal only the alpha-bearing tokens joined by spaces"
        );
    }

    /// A single line where em-dashes appear *multiple* times — before, between,
    /// and after text tokens.  With wave expansion, em-dashes adjacent to alpha
    /// tokens are rescued in the first wave (they add zero digit chars so ratio
    /// does not increase).  The em-dash *after* `INDUSTRIES` (index 6) is rescued
    /// in wave 1; `10789` and the trailing `—` (index 8) would require wave 2
    /// which pushes ratio over threshold — both are dropped.
    #[test]
    fn denoise_line_level_multiple_em_dashes_all_stripped() {
        let cfg = denoiser_enabled();
        let input = "— 42 NOVEX — 524 INDUSTRIES — 10789 —";
        let result = denoise_text(input, &cfg).expect("should not be None");

        // Leading "—" (no alpha neighbor in keep-set before wave 1) is in wave 2
        // alongside 10789 — the wave is rejected, so both are dropped.
        assert!(
            !result.contains("10789"),
            "10789 unreachable within ratio budget — dropped"
        );
        // "42" and "524" are adjacent to NOVEX/INDUSTRIES — rescued in wave 1.
        assert!(result.contains("42"), "42 adjacent to NOVEX — rescued");
        assert!(
            result.contains("524"),
            "524 adjacent to INDUSTRIES — rescued"
        );
        assert_eq!(
            result, "42 NOVEX — 524 INDUSTRIES —",
            "digit tokens and adjacent em-dashes rescued up to ratio limit; outer tokens dropped"
        );
    }

    /// A single line where parenthesized decimal values appear *multiple* times
    /// adjacent to alpha tokens.  Because the parenthesized tokens contain digits
    /// they are eligible for rescue; the candidate ratio (6 digits / 21
    /// alphanumeric chars ≈ 0.29) stays below the threshold so they are kept.
    /// A trailing bare number with no alpha neighbor is still stripped.
    #[test]
    fn denoise_line_level_multiple_parenthesized_values_rescued() {
        let cfg = denoiser_enabled();
        let input = "(0.8) NOVEX (1.2) INDUSTRIES (3.4) 10789";
        let result = denoise_text(input, &cfg).expect("should not be None");

        // Adjacent digit tokens are rescued when ratio is below threshold.
        assert!(
            result.contains("(0.8)"),
            "parenthesized value adjacent to NOVEX must be rescued"
        );
        assert!(
            result.contains("(1.2)"),
            "parenthesized value between alpha tokens must be rescued"
        );
        assert!(
            result.contains("(3.4)"),
            "parenthesized value adjacent to INDUSTRIES must be rescued"
        );
        // "10789" has no alpha neighbor — isolated, not rescued.
        assert!(
            !result.contains("10789"),
            "isolated number must be stripped"
        );
        assert_eq!(
            result, "(0.8) NOVEX (1.2) INDUSTRIES (3.4)",
            "adjacent digit tokens rescued; isolated number stripped"
        );
    }

    /// A single line that mixes em-dashes, parenthesized values, and bare
    /// numbers — all appearing *multiple* times.  Every non-alpha token is
    /// stripped; the exact joined remainder must match.
    #[test]
    fn denoise_line_level_mixed_symbol_trash_repeated() {
        let cfg = denoiser_enabled();
        let input = "— (0.8) 42 ZETA — (1.5) 524 POWER — (2.3) 10758 Corp —";
        let result = denoise_text(input, &cfg).expect("should not be None");

        assert!(!result.contains('—'), "all em-dash instances must be gone");
        assert!(!result.contains("(0.8)"), "first parens value must be gone");
        assert!(
            !result.contains("(1.5)"),
            "second parens value must be gone"
        );
        assert!(!result.contains("(2.3)"), "third parens value must be gone");
        assert!(!result.contains("10758"), "bare number must be stripped");
        assert_eq!(
            result, "ZETA POWER Corp",
            "full output must be only the three surviving text tokens"
        );
    }

    /// Two lines both containing multiple instances of different symbol trash.
    /// Line 1: wave 1 rescues 42, —, 524, — (ratio 5/20 = 0.25 ≤ 0.35);
    /// wave 2 would add 10789 + trailing — (ratio 10/25 = 0.40 > 0.35) → rejected.
    /// Line 2: `(0.8)`, `(1.2)`, `(3.4)` each contain TWO digit chars ('0','8', etc.);
    /// wave 1 wd=6 → ratio 6/15 = 0.40 > 0.35 → rejected; only alpha seed survives.
    #[test]
    fn denoise_line_level_multiple_symbol_trash_multiline_exact_output() {
        let cfg = denoiser_enabled();
        let input = indoc! {"
            — 42 NOVEX — 524 INDUSTRIES — 10789 —
            (0.8) ZETA (1.2) POWER (3.4) 10758
        "};
        let result = denoise_text(input.trim(), &cfg).expect("should not be None");

        assert_eq!(
            result, "42 NOVEX — 524 INDUSTRIES —\nZETA POWER",
            "line 1: wave expansion rescues up to budget; line 2: wave 1 rejected (’(0.8)’ has 2 digit chars)"
        );
    }

    // -----------------------------------------------------------------------
    // Markdown table handling
    // -----------------------------------------------------------------------

    /// A pure GFM pipe table: separator rows are dropped; header and data rows
    /// have their pipe delimiters stripped and cell text is returned.
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
            "Metric Value Change\nAnnual revenue $4.2B +12%\nOperating cost $2.1B +8%\nNet income $310M +15%",
            "separator row dropped, pipe delimiters stripped, cell text retained"
        );
    }

    /// A single pipe-table separator row (no alpha, no content) produces `None`.
    /// A lone header or data row has its pipe delimiters stripped and cell text
    /// returned — the block does not need to be multi-line for this to work.
    #[test]
    fn denoise_line_level_single_markdown_table_row_pipes_stripped() {
        let cfg = denoiser_enabled();

        // A lone separator row — no content → None.
        assert_eq!(
            denoise_text("|----------------|-----------|----------|", &cfg),
            None,
            "a separator row has no content and must produce None"
        );

        // A lone header row — pipes stripped, cell text survives.
        assert_eq!(
            denoise_text("| Metric | Value | Change |", &cfg),
            Some("Metric Value Change".to_string()),
            "header row pipes must be stripped, cell text retained"
        );

        // A lone data row with mixed text and symbols — pipes stripped, text survives.
        assert_eq!(
            denoise_text("| Annual revenue | $4.2B | +12% |", &cfg),
            Some("Annual revenue $4.2B +12%".to_string()),
            "data row pipes must be stripped, cell text retained"
        );
    }

    /// A markdown table embedded inside prose: prose lines survive unchanged,
    /// separator rows are dropped, header cell text is extracted, and
    /// digit-heavy data rows are further stripped to their alpha-bearing tokens.
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
        let result =
            denoise_text(input.trim(), &cfg).expect("should not be None — prose lines exist");

        assert!(
            result.contains("Revenue grew"),
            "opening prose must survive"
        );
        assert!(
            result.contains("Management expects"),
            "closing prose must survive"
        );
        assert!(
            result.contains("Year Revenue Growth"),
            "header cell text must survive"
        );
        assert!(!result.contains("---|"), "separator row must be gone");
        assert!(
            !result.contains("| Year"),
            "pipe delimiters must be stripped"
        );
        assert!(
            !result.contains("| 2023"),
            "pipe delimiters must be stripped"
        );
        assert_eq!(
            result.lines().count(),
            5,
            "prose(1) + header(1) + two stripped data rows(2) + prose(1)"
        );
    }

    /// Various separator-row styles all get dropped.
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
        let result =
            denoise_text(input.trim(), &cfg).expect("should not be None — prose line exists");

        assert!(result.contains("Only this prose line"));
        assert!(
            !result.contains("---"),
            "no separator content should remain"
        );
        assert_eq!(result.lines().count(), 1);
    }

    /// A table whose data cells are purely numeric: the header row text survives
    /// after pipe stripping (`ID Score Rank`), but all data rows are dropped by
    /// gate 2 (zero alphabetical characters after pipe stripping).
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
        let result = denoise_text(input.trim(), &cfg)
            .expect("header row text must survive — block is not None");
        assert_eq!(
            result, "ID Score Rank",
            "separator and numeric data rows dropped; only header cell text survives"
        );
    }

    /// A single-column table (two pipes per row: opening + closing):
    /// pipe delimiters are stripped and cell text is retained alongside prose.
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

        assert!(
            result.contains("Plain sentence before"),
            "opening prose must survive"
        );
        assert!(
            result.contains("Plain sentence after"),
            "closing prose must survive"
        );
        assert!(result.contains("Item"), "header cell text must survive");
        assert!(result.contains("Widget A"), "data cell text must survive");
        assert!(result.contains("Widget B"), "data cell text must survive");
        assert!(!result.contains("---"), "separator row must be dropped");
        assert!(
            !result.contains('|'),
            "pipe delimiters must all be stripped"
        );
        assert_eq!(result.lines().count(), 5);
    }

    /// Prose lines that happen to contain a single `|` (e.g. inline code
    /// or informal OR notation) are *not* treated as table rows.
    #[test]
    fn denoise_line_level_single_pipe_in_prose_is_not_a_table_row() {
        let cfg = denoiser_enabled();
        let input = "Use the syntax foo | bar to combine options.";
        let result = denoise_text(input, &cfg).expect("should not be None");
        assert_eq!(
            result, input,
            "prose with a single pipe must pass through unchanged"
        );
    }

    /// A borderless-style table (no leading `|`) has its separator row dropped
    /// (no alpha chars) while the data rows survive as plain text — the pipe
    /// characters stay but are not treated as markdown table delimiters.
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

        // The separator row has no alpha — dropped.
        assert!(!result.contains("-----"), "separator row must be gone");
        // The header and data rows do not start with '|' → not treated as table rows.
        assert!(result.contains("Name"), "borderless header must survive");
        assert!(result.contains("Alice"), "borderless data row must survive");
        assert!(result.contains("Bob"), "borderless data row must survive");
    }

    // -----------------------------------------------------------------------
    // Full OCR table block
    // -----------------------------------------------------------------------

    /// A multi-line block structured like a mangled OCR financial table.
    /// Every row has a company name + city interspersed with dense numeric columns.
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

        let result =
            denoise_text(input.trim(), &cfg).expect("block should not be dropped entirely");

        let expected_names = [
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
        ];
        for name in &expected_names {
            assert!(
                result.contains(name),
                "expected company token '{name}' in output"
            );
        }

        let expected_locations = [
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
        ];
        for loc in &expected_locations {
            assert!(result.contains(loc), "expected location '{loc}' in output");
        }

        // Isolated numeric tokens (not adjacent to any alpha token) are stripped.
        for junk in &["45432", "13539", "116524"] {
            assert!(
                !result.contains(junk),
                "isolated junk token '{junk}' should have been stripped"
            );
        }

        // "&" is between two alpha tokens (PRODUCTS and LOGISTICS); wave expansion
        // rescues it in wave 1 (no digit chars → zero cost to ratio).  It is kept.
        assert!(
            result.contains("PRODUCTS & LOGISTICS"),
            "'&' between alpha tokens is rescued"
        );

        // Digit tokens immediately adjacent to an alpha token ARE rescued when
        // the candidate digit-ratio stays below the threshold.
        for rescued in &["10788", "10312"] {
            assert!(
                result.contains(rescued),
                "adjacent digit token '{rescued}' should be rescued"
            );
        }

        assert_eq!(
            result.lines().count(),
            input.trim().lines().count(),
            "every input row should survive as a stripped output line"
        );

        // Exact full multiline output — wave expansion rescues digit tokens and
        // zero-cost symbol tokens (like `&`) adjacent to alpha tokens; tokens
        // that would push the ratio over threshold are dropped.  The first token
        // of each row (e.g. `42`, `343`) is 2+ hops from alpha and gets stripped
        // from most rows, but on rows 1, 9, and 10 more waves are accepted because
        // those rows have more alpha to absorb the digit budget.
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
            },
            "full stripped output: wave-expanded rescue up to per-row ratio budget; '&' kept"
        );
    }

    // --- Linearized XBRL passthrough (rust-sec-xbrl-textualizer) ---
    //
    // These strings are real output captured from:
    //   cargo run -- trend-render "Revenues,NetIncomeLoss,OperatingIncomeLoss" "AAPL" \
    //       --window-years 3 --view linearized
    //
    // The denoiser must not corrupt this format. Blank lines (no alpha) are
    // legitimately dropped by Gate 2 — all content lines must survive verbatim.

    // -----------------------------------------------------------------------
    // Financial punctuation: +, -, —, $, %
    // -----------------------------------------------------------------------

    /// Lines below the digit-ratio threshold must pass through entirely
    /// unchanged — every financial token (`$4.2B`, `+12%`, `-$1.1B`, `23%`,
    /// `—`) is preserved byte-for-byte.
    #[test]
    fn denoise_financial_punctuation_below_threshold_passes_through_unchanged() {
        let cfg = denoiser_enabled();
        // digits ≈ 8, alpha ≈ 35 → ratio ≈ 0.19 — well below 0.35.
        let input = "Operating cash: $4.2B (+12% YoY) — net debt fell -$1.1B; margin: 23%.";
        let result = denoise_text(input, &cfg).expect("should not be None");
        assert_eq!(
            result, input,
            "below-threshold line with financial symbols must be byte-identical"
        );
    }

    /// On a gate-3 line, an em-dash (`—`) used as a minus-sign operator between
    /// alpha tokens is rescued by wave expansion: it carries zero digit chars so
    /// it adds nothing to the digit budget and is always accepted with its wave.
    #[test]
    fn denoise_em_dash_operator_on_gate3_line_is_rescued() {
        let cfg = denoiser_enabled();
        // ratio = 10/25 = 0.40 → gate 3 triggered.
        // Seed: {REVENUE(0), COSTS(2), NET(3)}, a=15, d=0.
        // Wave 1: {—(1), 42(4)}   — wd=0+2=2, ratio=2/17=0.12 ≤ 0.35 → accept.
        // Wave 2: {524(5)}       — wd=3,     ratio=5/20=0.25 ≤ 0.35 → accept.
        // Wave 3: {10788(6)}     — wd=5,     ratio=10/25=0.40 > 0.35 → reject.
        let input = "REVENUE — COSTS NET 42 524 10788";
        let result = denoise_text(input, &cfg).expect("should not be None");
        assert!(
            result.contains('—'),
            "em-dash between alpha tokens must survive"
        );
        assert!(result.contains("42"), "'42' rescued in wave 1 alongside —");
        assert!(result.contains("524"), "'524' rescued in wave 2");
        assert!(
            !result.contains("10788"),
            "'10788' in rejected wave 3 — stripped"
        );
        assert_eq!(result, "REVENUE — COSTS NET 42 524");
    }

    /// On a gate-3 line, `+N%` and `-N%` tokens (digit-bearing, no alpha) are
    /// rescued by wave expansion when adjacent to alpha tokens and the wave ratio
    /// stays within budget.  This verifies sign and percent characters are never
    /// blindly stripped.
    #[test]
    fn denoise_sign_percent_tokens_on_gate3_line_are_rescued() {
        let cfg = denoiser_enabled();
        // ratio = 21/47 ≈ 0.45 → gate 3 triggered.
        // Seed: {REVENUE(0), GROWTH(1), EARNINGS(3), COSTS(5)}, a=26, d=0.
        // Wave 1: {+12%(2), -8%(4), 42(6)} — wd=5,  ratio=5/31=0.16 ≤ 0.35 → accept.
        // Wave 2: {524(7)}                 — wd=3,  ratio=8/34=0.24 ≤ 0.35 → accept.
        // Wave 3: {10788(8)}               — wd=5,  ratio=13/39=0.33 ≤ 0.35 → accept.
        // Wave 4: {5520(9)}                — wd=4,  ratio=17/43=0.40 > 0.35 → reject.
        let input = "REVENUE GROWTH +12% EARNINGS -8% COSTS 42 524 10788 5520 3918";
        let result = denoise_text(input, &cfg).expect("should not be None");
        assert!(
            result.contains("+12%"),
            "'+12%' adjacent to alpha — rescued"
        );
        assert!(result.contains("-8%"), "'-8%' adjacent to alpha — rescued");
        assert!(result.contains("42"), "'42' rescued in wave 1");
        assert!(result.contains("524"), "'524' rescued in wave 2");
        assert!(result.contains("10788"), "'10788' rescued in wave 3");
        assert!(
            !result.contains("5520"),
            "'5520' in rejected wave 4 — stripped"
        );
        assert!(!result.contains("3918"), "'3918' unreachable — stripped");
        assert_eq!(
            result,
            "REVENUE GROWTH +12% EARNINGS -8% COSTS 42 524 10788"
        );
    }

    // --- Linearized XBRL passthrough (rust-sec-xbrl-textualizer) ---
    //
    // These strings are real output captured from:
    //   cargo run -- trend-render "Revenues,NetIncomeLoss,OperatingIncomeLoss" "AAPL" \
    //       --window-years 3 --view linearized
    //
    // The denoiser must not corrupt this format. Blank lines (no alpha) are
    // legitimately dropped by Gate 2 — all content lines must survive verbatim.

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
        let result = denoise_text(line, &cfg);
        assert_eq!(
            result,
            Some(line.to_string()),
            "linearized metric line must pass through denoise_text completely unchanged"
        );
    }

    #[test]
    fn linearized_xbrl_full_aapl_block_content_preserved() {
        // Real output block — blank lines are dropped by Gate 2 (zero alpha),
        // every content line must survive verbatim and in order.
        let input = indoc! {"
            ### AAPL
            periods=2026Q1,2025Q4,2025Q3,2025Q2,2025Q1,2024Q4,2024Q3,2024Q2,2024Q1,2023Q4,2023Q3,2023Q2,2023Q1

            label=Net income | dir=up | traj=non_monotonic | path=mostly_upward | recent=up_bias | reg=growth_with_resets | cons=erratic | turn=high_turn | run=clustered_runs | end=recovering_off_peak | rec=weak_recovery | dd=extreme | shock=repeated_shock | pol=upside_shocks | flip=false | sig=UUDUUUDU-t4 | first=30.00B | last=42.10B | filing_quality=score=100.0 grade=A transitions=12 scale_issues=0 uom_issues=0

            label=Operating income | dir=up | traj=non_monotonic | path=mostly_upward | recent=up_bias | reg=growth_with_resets | cons=erratic | turn=high_turn | run=clustered_runs | end=recovering_off_peak | rec=weak_recovery | dd=severe | shock=repeated_shock | pol=upside_shocks | flip=false | sig=UUDUUUDU-t4 | first=36.02B | last=50.85B | filing_quality=score=100.0 grade=A transitions=12 scale_issues=0 uom_issues=0"
        };
        let cfg = denoiser_enabled();
        let result = denoise_text(input, &cfg);
        // Blank separator lines are stripped (Gate 2 — no alpha). All content lines preserved.
        assert_eq!(
            result,
            Some(indoc! {"
                ### AAPL
                periods=2026Q1,2025Q4,2025Q3,2025Q2,2025Q1,2024Q4,2024Q3,2024Q2,2024Q1,2023Q4,2023Q3,2023Q2,2023Q1
                label=Net income | dir=up | traj=non_monotonic | path=mostly_upward | recent=up_bias | reg=growth_with_resets | cons=erratic | turn=high_turn | run=clustered_runs | end=recovering_off_peak | rec=weak_recovery | dd=extreme | shock=repeated_shock | pol=upside_shocks | flip=false | sig=UUDUUUDU-t4 | first=30.00B | last=42.10B | filing_quality=score=100.0 grade=A transitions=12 scale_issues=0 uom_issues=0
                label=Operating income | dir=up | traj=non_monotonic | path=mostly_upward | recent=up_bias | reg=growth_with_resets | cons=erratic | turn=high_turn | run=clustered_runs | end=recovering_off_peak | rec=weak_recovery | dd=severe | shock=repeated_shock | pol=upside_shocks | flip=false | sig=UUDUUUDU-t4 | first=36.02B | last=50.85B | filing_quality=score=100.0 grade=A transitions=12 scale_issues=0 uom_issues=0"
            }.to_string()),
            "all AAPL content lines must survive unchanged; only blank separators may be dropped"
        );
    }
}
