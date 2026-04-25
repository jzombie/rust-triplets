//! OCR denoising and markdown-formatting cleanup for text chunks.
//!
//! The entry point is [`denoise_text`], which applies a configurable set of
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

/// Keep only whitespace-delimited tokens that contain at least one alphabetical
/// character.  Returns the space-joined result; may be empty.
fn strip_digit_tokens(line: &str) -> String {
    line.split_whitespace()
        .filter(|token| token.chars().any(|ch| ch.is_alphabetic()))
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

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Apply OCR denoising and markdown-table stripping to a block of text.
///
/// Returns `Some(cleaned)` with the (possibly stripped) text, or `None` when
/// the entire block should be dropped and no chunks should be produced.
///
/// # Line-level mode (`config.line_level == true`)
///
/// Line endings are first normalised with [`LineEnding::normalize`].  Each
/// line is then evaluated through three gates in order:
///
/// 1. **Markdown table rows** — lines whose trimmed form starts with `'|'` and
///    contains at least one more `'|'` are dropped unconditionally.  This
///    removes GFM pipe-table headers, separator rows, and data rows, all of
///    which carry no semantic value for text embeddings.
///
/// 2. **No alphabetical characters** — lines that contain zero alphabetical
///    characters (all-numeric rows, symbol/dash-only rows, OCR separator
///    artefacts) are dropped.
///
/// 3. **High digit ratio** — lines whose `digit / (digit + alpha)` ratio
///    exceeds `config.max_digit_ratio` are *stripped*: only whitespace-
///    delimited tokens that contain at least one alphabetical character are
///    retained.  If stripping leaves the line empty the line is dropped.
///
/// `None` is returned only when every line is removed.
///
/// # Whole-block mode (`config.line_level == false`)
///
/// The ratio is computed for the entire block.  If it exceeds the threshold
/// the whole block is dropped (`None`); otherwise the block is returned
/// unchanged.
///
/// When `config.enabled` is `false` the function returns `Some(text.to_string())`
/// unconditionally.
pub fn denoise_text(text: &str, config: &DenoiserConfig) -> Option<String> {
    if !config.enabled {
        return Some(text.to_string());
    }

    if config.line_level {
        let normalized = LineEnding::normalize(text);
        let mut cleaned_lines: Vec<String> = Vec::new();
        for line in normalized.lines() {
            // Gate 1: markdown table formatting rows → drop entirely.
            if is_markdown_table_line(line) {
                continue;
            }

            // Gate 2: no alphabetical characters → drop (all-numeric lines,
            //         symbol-only rows, OCR column-separator artefacts, etc.).
            let (_, alpha) = count_digit_alpha(line);
            if alpha == 0 {
                continue;
            }

            // Gate 3: digit-heavy line → strip to alpha-bearing tokens only.
            if digit_ratio(line) > config.max_digit_ratio {
                let retained = strip_digit_tokens(line);
                if !retained.is_empty() {
                    cleaned_lines.push(retained);
                }
                // else: drop the line entirely
            } else {
                cleaned_lines.push(line.to_string());
            }
        }
        if cleaned_lines.is_empty() {
            None
        } else {
            Some(cleaned_lines.join("\n"))
        }
    } else {
        if digit_ratio(text) > config.max_digit_ratio {
            None
        } else {
            Some(text.to_string())
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use indoc::indoc;

    fn denoiser_enabled(line_level: bool) -> DenoiserConfig {
        DenoiserConfig {
            enabled: true,
            max_digit_ratio: 0.35,
            line_level,
        }
    }

    // -----------------------------------------------------------------------
    // is_markdown_table_line helper
    // -----------------------------------------------------------------------

    #[test]
    fn markdown_table_line_detects_separator_row() {
        assert!(is_markdown_table_line("|------|-----|"));
        assert!(is_markdown_table_line("|:----:|:---:|"));
        assert!(is_markdown_table_line("|----------------|-----------|----------|"));
        assert!(is_markdown_table_line("  |---|---|  "), "leading/trailing spaces ok");
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

    // -----------------------------------------------------------------------
    // Whole-block mode
    // -----------------------------------------------------------------------

    #[test]
    fn denoise_whole_block_drops_digit_heavy_text() {
        let cfg = denoiser_enabled(false);
        let input = "42 524 10788 143 1995 190 394 13611 358 6444 266";
        assert_eq!(denoise_text(input, &cfg), None);
    }

    /// Below the threshold every character — including numbers and symbols —
    /// must be returned verbatim.
    #[test]
    fn denoise_whole_block_below_threshold_preserves_numbers_and_symbols() {
        let cfg = denoiser_enabled(false);
        // ratio ≈ 0.12 — well below 0.35.
        let input = "Q3 revenue grew 12% to $4.2B, up from $3.8B in Q2 (a 10.5% increase).";
        assert_eq!(
            denoise_text(input, &cfg),
            Some(input.to_string()),
            "below-threshold block must be returned byte-for-byte"
        );
    }

    #[test]
    fn denoise_whole_block_keeps_text_light_on_digits() {
        let cfg = denoiser_enabled(false);
        let input = "The quick brown fox jumps over the lazy dog";
        assert_eq!(denoise_text(input, &cfg), Some(input.to_string()));
    }

    // -----------------------------------------------------------------------
    // Line-level: basic drop / pass-through
    // -----------------------------------------------------------------------

    #[test]
    fn denoise_empty_input_returns_none_when_enabled() {
        let cfg = denoiser_enabled(true);
        assert_eq!(denoise_text("", &cfg), None);
    }

    #[test]
    fn denoise_line_level_returns_none_when_all_lines_dropped() {
        let cfg = denoiser_enabled(true);
        let input = indoc! {"
            42 524 10788
            143 1995 190
            394 13611 358
        "};
        assert_eq!(denoise_text(input.trim(), &cfg), None);
    }

    #[test]
    fn denoise_line_level_preserves_clean_lines_unchanged() {
        let cfg = denoiser_enabled(true);
        let clean = "Climate change drives ocean temperatures higher each decade.";
        let result = denoise_text(clean, &cfg).expect("clean text should be kept");
        assert_eq!(result, clean);
    }

    /// A clean line with numbers and symbols below the threshold must come
    /// through byte-for-byte; no tokens should be stripped.
    #[test]
    fn denoise_line_level_below_threshold_line_preserves_numbers_and_symbols() {
        let cfg = denoiser_enabled(true);
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
        let cfg = denoiser_enabled(true);
        let input = indoc! {"
            Revenue grew 8% to $2.1B in FY2025 (vs $1.9B prior year).
            42 9871 3302 19283 4710 22913 5518 30021 6627 41132 7736 52243
            Net income rose 15% YoY, reaching $310M by Q4-2025.
        "};
        let result = denoise_text(input.trim(), &cfg).expect("should not be None");

        assert!(result.contains("Revenue grew 8% to $2.1B in FY2025 (vs $1.9B prior year)."));
        assert!(result.contains("Net income rose 15% YoY, reaching $310M by Q4-2025."));
        assert!(!result.contains("9871"), "junk line must be stripped");
        assert_eq!(result.lines().count(), 2, "only the two clean lines should remain");
    }

    // -----------------------------------------------------------------------
    // Line-level: mixed content on the same line
    // -----------------------------------------------------------------------

    /// Every line has text and numbers interleaved; digit-bearing tokens are
    /// stripped while the text tokens survive.
    #[test]
    fn denoise_line_level_mixed_content_same_line() {
        let cfg = denoiser_enabled(true);
        let input = indoc! {"
            42 524 NOVEX INDUSTRIES Springfield 10788 143 1995 190 394 13611 358
            343 294 ZETA POWER Riverside 10758 31 1283 267 189 45432 175
        "};
        let result = denoise_text(input.trim(), &cfg).expect("should not be None");

        for word in &["NOVEX", "INDUSTRIES", "Springfield", "ZETA", "POWER", "Riverside"] {
            assert!(result.contains(word), "'{word}' must survive line-level strip");
        }
        for num in &["10788", "45432"] {
            assert!(!result.contains(num), "'{num}' must be stripped");
        }
        assert_eq!(result.lines().count(), 2);
    }

    /// Purely numeric or purely symbolic lines are dropped; the block is
    /// `None` only when *every* line is removed.
    #[test]
    fn denoise_line_level_drops_lines_with_no_alpha_tokens() {
        let cfg = denoiser_enabled(true);
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
        let cfg = denoiser_enabled(true);
        let input =
            "42 524 NOVEX INDUSTRIES Springfield 10788 143 1995 190 394 13611 358 6444 266";
        let result = denoise_text(input, &cfg).expect("should not be None");

        for word in &["NOVEX", "INDUSTRIES", "Springfield"] {
            assert!(result.contains(word), "'{word}' must survive");
        }
        for num in &["10788", "13611"] {
            assert!(!result.contains(num), "'{num}' must be stripped");
        }
    }

    // -----------------------------------------------------------------------
    // Line-level: interleaved sequences
    // -----------------------------------------------------------------------

    /// Digit-junk sandwiched *between* text tokens — all text must survive.
    #[test]
    fn denoise_line_level_text_sandwiched_between_junk_tokens() {
        let cfg = denoiser_enabled(true);
        let input = "42 NOVEX 524 INDUSTRIES 10788 143 1995 190";
        let result = denoise_text(input, &cfg).expect("should not be None");

        assert!(result.contains("NOVEX"),      "NOVEX should survive");
        assert!(result.contains("INDUSTRIES"), "INDUSTRIES should survive");
        assert!(!result.contains("10788"),     "pure-numeric token should be gone");
        assert!(!result.contains("524"),       "pure-numeric token should be gone");
    }

    /// Text and junk tokens alternating many times — all text tokens survive in
    /// their original relative order.
    #[test]
    fn denoise_line_level_repeated_junk_text_interleaving() {
        let cfg = denoiser_enabled(true);
        let input =
            "42 ZETA 524 POWER 10758 Riverside 31 GRID 1283 GROUP 267 Holdings 45432 Corp";
        let result = denoise_text(input, &cfg).expect("should not be None");

        let text_tokens = ["ZETA", "POWER", "Riverside", "GRID", "GROUP", "Holdings", "Corp"];
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
        let cfg = denoiser_enabled(true);
        let input =
            "345 397 DELTA CORP Detroit, Mich. 10689 (0.8) 1069 302 — 18214 336 17590 182";
        let result = denoise_text(input, &cfg).expect("should not be None");

        assert!(result.contains("DELTA"),    "DELTA must survive");
        assert!(result.contains("CORP"),     "CORP must survive");
        assert!(result.contains("Detroit"),  "city must survive");
        assert!(result.contains("Mich."),    "dotted abbreviation must survive");
        assert!(!result.contains("10689"),   "bare number must be stripped");
        assert!(!result.contains("(0.8)"),   "parenthesized negative must be stripped");
        assert!(!result.contains("18214"),   "bare number must be stripped");
    }

    /// Comma-formatted numbers like `10,788.0` contain no alpha → stripped.
    #[test]
    fn denoise_line_level_comma_formatted_numbers_stripped() {
        let cfg = denoiser_enabled(true);
        let input =
            "42 524 NOVEX INDUSTRIES Springfield 10,788.0 14.3 1,995.0 190 39.4 13,611.0 358";
        let result = denoise_text(input, &cfg).expect("should not be None");

        for word in &["NOVEX", "INDUSTRIES", "Springfield"] {
            assert!(result.contains(word), "'{word}' must survive");
        }
        for num in &["10,788.0", "1,995.0", "13,611.0"] {
            assert!(!result.contains(num), "comma-number '{num}' must be stripped");
        }
    }

    /// A line of only symbols / numbers with no alpha → block returns `None`.
    #[test]
    fn denoise_line_level_symbol_only_line_is_dropped() {
        let cfg = denoiser_enabled(true);
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
        let cfg = denoiser_enabled(true);
        let input = "3rd Quarter performance review 2nd half summary";
        let result = denoise_text(input, &cfg).expect("should not be None");

        assert!(result.contains("3rd"),     "ordinal '3rd' should be retained");
        assert!(result.contains("2nd"),     "ordinal '2nd' should be retained");
        assert!(result.contains("Quarter"), "plain word must be retained");
    }

    /// Dense interleaving: numbers, parenthesized values, em-dashes, and text.
    #[test]
    fn denoise_line_level_dense_interleave_with_symbols() {
        let cfg = denoiser_enabled(true);
        let input = "42 (524) ZETA 10,758.0 — POWER 31.5 Riverside, 1283 Corp.";
        let result = denoise_text(input, &cfg).expect("should not be None");

        for word in &["ZETA", "POWER", "Riverside,", "Corp."] {
            assert!(result.contains(word), "'{word}' must survive");
        }
        for junk in &["42", "(524)", "10,758.0", "1283"] {
            assert!(!result.contains(junk), "'{junk}' must be stripped");
        }
    }

    // -----------------------------------------------------------------------
    // Markdown table handling
    // -----------------------------------------------------------------------

    /// A pure GFM pipe table — all rows start with `|` — produces `None`
    /// because every line is identified as a markdown table row and dropped.
    #[test]
    fn denoise_line_level_pure_markdown_table_returns_none() {
        let cfg = denoiser_enabled(true);
        let input = indoc! {"
            | Metric         | Value     | Change  |
            |----------------|-----------|----------|
            | Annual revenue | $4.2B     | +12%     |
            | Operating cost | $2.1B     | +8%      |
            | Net income     | $310M     | +15%     |
        "};
        assert_eq!(
            denoise_text(input.trim(), &cfg),
            None,
            "all pipe-table rows should be dropped, making the block None"
        );
    }

    /// A markdown table embedded inside prose: only the prose survives.
    #[test]
    fn denoise_line_level_markdown_table_embedded_in_prose() {
        let cfg = denoiser_enabled(true);
        let input = indoc! {"
            Revenue grew steadily over the past three fiscal years.
            | Year | Revenue | Growth |
            |------|---------|--------|
            | 2023 | $3.8B   | +10%   |
            | 2024 | $4.2B   | +12%   |
            Management expects the trend to continue.
        "};
        let result = denoise_text(input.trim(), &cfg).expect("should not be None — prose lines exist");

        assert!(result.contains("Revenue grew"),            "opening prose must survive");
        assert!(result.contains("Management expects"),      "closing prose must survive");
        assert!(!result.contains("---|"),                   "separator row must be gone");
        assert!(!result.contains("| Year"),                 "header row must be gone");
        assert!(!result.contains("| 2023"),                 "data row must be gone");
        assert!(!result.contains("| 2024"),                 "data row must be gone");
        assert_eq!(result.lines().count(), 2, "only the two prose lines should remain");
    }

    /// Various separator-row styles all get dropped.
    #[test]
    fn denoise_line_level_markdown_table_various_separator_styles() {
        let cfg = denoiser_enabled(true);
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
        assert!(!result.contains("---"),    "no separator content should remain");
        assert_eq!(result.lines().count(), 1);
    }

    /// A table whose data cells are numeric-heavy: every row is still a
    /// markdown table row and must be dropped regardless of cell content.
    #[test]
    fn denoise_line_level_markdown_table_numeric_cells_dropped() {
        let cfg = denoiser_enabled(true);
        let input = indoc! {"
            | ID   | Score | Rank |
            |------|-------|------|
            | 1001 | 98.5  | 1    |
            | 1002 | 87.3  | 2    |
            | 1003 | 76.0  | 3    |
        "};
        assert_eq!(
            denoise_text(input.trim(), &cfg),
            None,
            "pipe-table rows with numeric cells must still be dropped"
        );
    }

    /// A single-column table (two pipes per row: opening + closing).
    #[test]
    fn denoise_line_level_markdown_table_single_column() {
        let cfg = denoiser_enabled(true);
        let input = indoc! {"
            Plain sentence before the table.
            | Item       |
            |------------|
            | Widget A   |
            | Widget B   |
            Plain sentence after the table.
        "};
        let result = denoise_text(input.trim(), &cfg).expect("prose lines must survive");

        assert!(result.contains("Plain sentence before"));
        assert!(result.contains("Plain sentence after"));
        assert!(!result.contains("Widget"),    "table data rows must be dropped");
        assert!(!result.contains("---"),       "separator row must be dropped");
        assert_eq!(result.lines().count(), 2);
    }

    /// Prose lines that happen to contain a single `|` (e.g. inline code
    /// or informal OR notation) are *not* treated as table rows.
    #[test]
    fn denoise_line_level_single_pipe_in_prose_is_not_a_table_row() {
        let cfg = denoiser_enabled(true);
        let input = "Use the syntax foo | bar to combine options.";
        let result = denoise_text(input, &cfg).expect("should not be None");
        assert_eq!(result, input, "prose with a single pipe must pass through unchanged");
    }

    /// A borderless-style table (no leading `|`) has its separator row dropped
    /// (no alpha chars) while the data rows survive as plain text — the pipe
    /// characters stay but are not treated as markdown table delimiters.
    #[test]
    fn denoise_line_level_borderless_table_separator_dropped_data_survives() {
        let cfg = denoiser_enabled(true);
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
        assert!(result.contains("Name"),  "borderless header must survive");
        assert!(result.contains("Alice"), "borderless data row must survive");
        assert!(result.contains("Bob"),   "borderless data row must survive");
    }

    // -----------------------------------------------------------------------
    // Full OCR table block
    // -----------------------------------------------------------------------

    /// A multi-line block structured like a mangled OCR financial table.
    /// Every row has a company name + city interspersed with dense numeric columns.
    #[test]
    fn denoise_full_table_block_retains_company_names() {
        let cfg = denoiser_enabled(true);

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
            "NOVEX", "INDUSTRIES", "ZETA", "POWER", "OCEAN", "FORGE",
            "DELTA", "FINANCIAL", "APEX", "HOLDINGS", "VEGA", "SYSTEMS",
            "CREST", "BRANDS", "TITAN", "CHEMICAL", "AIR", "PRODUCTS",
            "LOGISTICS", "NORTHLAND", "MEMBERS",
        ];
        for name in &expected_names {
            assert!(result.contains(name), "expected company token '{name}' in output");
        }

        let expected_locations = [
            "Springfield", "Riverside", "Denver", "Detroit", "Brentwood",
            "Tulsa", "Atlanta", "Kingsport", "Allentown", "Minneapolis",
        ];
        for loc in &expected_locations {
            assert!(result.contains(loc), "expected location '{loc}' in output");
        }

        for junk in &["10788", "45432", "13539", "116524", "10312"] {
            assert!(!result.contains(junk), "junk token '{junk}' should have been stripped");
        }

        // "&" has no alpha chars → stripped.
        assert!(!result.contains(" & "), "'&' token should be stripped");

        assert_eq!(
            result.lines().count(),
            input.trim().lines().count(),
            "every input row should survive as a stripped output line"
        );
    }
}
