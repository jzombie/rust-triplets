use chrono::{DateTime, Utc};
use std::path::PathBuf;

use crate::config::{NegativeStrategy, SamplerConfig, Selector, TripletRecipe};
use crate::data::{DataRecord, QualityScore, SectionRole};
use crate::errors::SamplerError;
use crate::source::{DataSource, IndexablePager, IndexableSource, SourceCursor, SourceSnapshot};
use crate::types::SourceId;
use crate::utils::{file_times, make_section, normalize_inline_whitespace};

const CSV_RECIPE_ANCHOR_POSITIVE_WRONG_ARTICLE: &str = "csv_anchor_positive_wrong_article";
const CSV_RECIPE_ANCHOR_ANCHOR_WRONG_ARTICLE: &str = "csv_anchor_anchor_wrong_article";
/// Default CSV text-columns-mode SimCSE-style recipe name.
pub const CSV_RECIPE_TEXT_SIMCSE_WRONG_ARTICLE: &str = "csv_text_simcse_wrong_article";

/// Configuration for a CSV-backed data source.
///
/// Two modes are supported:
///
/// - **Role mode** — `anchor_column` is set (with an optional `positive_column`).
///   Each row produces an `Anchor` section from `anchor_column` and a `Context`
///   section from `positive_column` (or the anchor text when `positive_column` is
///   absent).
///
/// - **Text mode** — only `text_column` is set.  Each row produces both an
///   `Anchor` and a `Context` section from the same column (SimCSE-style).
///
/// `anchor_column` and `text_column` are mutually exclusive.
#[derive(Clone, Debug)]
pub struct CsvSourceConfig {
    /// Stable source identifier used in records and persistence keys.
    pub source_id: SourceId,
    /// Path to the CSV file.
    pub path: PathBuf,
    /// Column name for anchor text.  Enables role mode when set.
    ///
    /// Mutually exclusive with `text_column`.
    pub anchor_column: Option<String>,
    /// Column name for positive/context text.  Used with `anchor_column`.
    ///
    /// When absent in role mode, the anchor text is reused as the context
    /// (identical-positive fallback, suitable for contrastive pre-training).
    pub positive_column: Option<String>,
    /// Column name for single-text mode.
    ///
    /// Mutually exclusive with `anchor_column`.
    pub text_column: Option<String>,
    /// Trust/quality score assigned to every record from this source.
    pub trust: f32,
    /// Whether the CSV file has a header row.  Defaults to `true`.
    pub has_headers: bool,
}

impl CsvSourceConfig {
    /// Create a config for a CSV source with the given identifier and path.
    pub fn new(source_id: impl Into<SourceId>, path: impl Into<PathBuf>) -> Self {
        Self {
            source_id: source_id.into(),
            path: path.into(),
            anchor_column: None,
            positive_column: None,
            text_column: None,
            trust: 0.85,
            has_headers: true,
        }
    }

    /// Set the column used as the anchor (enables role mode).
    pub fn with_anchor_column(mut self, column: impl Into<String>) -> Self {
        self.anchor_column = Some(column.into());
        self
    }

    /// Set the column used as the positive/context (role mode only).
    pub fn with_positive_column(mut self, column: impl Into<String>) -> Self {
        self.positive_column = Some(column.into());
        self
    }

    /// Set the column used as the single text field (enables text mode).
    pub fn with_text_column(mut self, column: impl Into<String>) -> Self {
        self.text_column = Some(column.into());
        self
    }

    /// Override the default trust score.
    pub fn with_trust(mut self, trust: f32) -> Self {
        self.trust = trust;
        self
    }

    /// Set whether the CSV file has a header row.
    pub fn with_headers(mut self, has_headers: bool) -> Self {
        self.has_headers = has_headers;
        self
    }

    fn is_role_mode(&self) -> bool {
        self.anchor_column.is_some()
    }

    fn validate(&self) -> Result<(), SamplerError> {
        if self.anchor_column.is_some() && self.text_column.is_some() {
            return Err(SamplerError::Configuration(
                "CsvSourceConfig: `anchor_column` and `text_column` are mutually exclusive"
                    .to_string(),
            ));
        }
        if self.anchor_column.is_none() && self.text_column.is_none() {
            return Err(SamplerError::Configuration(
                "CsvSourceConfig: one of `anchor_column` or `text_column` must be set".to_string(),
            ));
        }
        if self.positive_column.is_some() && self.anchor_column.is_none() {
            return Err(SamplerError::Configuration(
                "CsvSourceConfig: `positive_column` requires `anchor_column` to be set".to_string(),
            ));
        }
        Ok(())
    }
}

/// Column-mapped CSV data source.
///
/// Reads all rows from a CSV file at construction and exposes them as
/// [`DataRecord`]s.  Suitable for small-to-medium datasets that fit comfortably
/// in memory.
///
/// ## Modes
///
/// Configure the source with either anchor/positive columns (role mode) or a
/// single text column (text mode):
///
/// ```rust,no_run
/// use triplets::source::{CsvSource, CsvSourceConfig};
///
/// // Role mode: explicit anchor + positive columns.
/// let config = CsvSourceConfig::new("my_qna", "data/qna.csv")
///     .with_anchor_column("question")
///     .with_positive_column("answer")
///     .with_trust(0.9);
/// let source = CsvSource::new(config).unwrap();
///
/// // Text mode: single text column.
/// let config2 = CsvSourceConfig::new("my_corpus", "data/corpus.csv")
///     .with_text_column("text");
/// let source2 = CsvSource::new(config2).unwrap();
/// ```
#[derive(Debug)]
pub struct CsvSource {
    config: CsvSourceConfig,
    records: Vec<DataRecord>,
}

impl CsvSource {
    /// Load a CSV source from the given configuration.
    ///
    /// Returns a `SamplerError::Configuration` error if the config is invalid,
    /// or a `SamplerError::SourceUnavailable` error if the CSV file cannot be
    /// opened or parsed.
    pub fn new(config: CsvSourceConfig) -> Result<Self, SamplerError> {
        config.validate()?;
        let records = Self::load_records(&config)?;
        Ok(Self { config, records })
    }

    fn load_records(config: &CsvSourceConfig) -> Result<Vec<DataRecord>, SamplerError> {
        let (created_at, updated_at) = file_times(&config.path);

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(config.has_headers)
            .flexible(false)
            .trim(csv::Trim::All)
            .from_path(&config.path)
            .map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!("failed to open CSV file '{}': {err}", config.path.display()),
            })?;

        // When has_headers is false the csv crate still owns a "headers" slot
        // but it will be empty; column lookup by name is unavailable so we
        // require named headers in that case. Build the header map once.
        let headers = reader
            .headers()
            .map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!(
                    "failed to read CSV headers in '{}': {err}",
                    config.path.display()
                ),
            })?
            .clone();

        // Pre-resolve column indices so we error early on bad config rather
        // than silently skipping every row.
        let anchor_idx = if let Some(col) = &config.anchor_column {
            Some(column_index(&headers, col).ok_or_else(|| {
                SamplerError::Configuration(format!(
                    "anchor_column '{}' not found in CSV headers of '{}'",
                    col,
                    config.path.display()
                ))
            })?)
        } else {
            None
        };

        let positive_idx = if let Some(col) = &config.positive_column {
            Some(column_index(&headers, col).ok_or_else(|| {
                SamplerError::Configuration(format!(
                    "positive_column '{}' not found in CSV headers of '{}'",
                    col,
                    config.path.display()
                ))
            })?)
        } else {
            None
        };

        let text_idx = if let Some(col) = &config.text_column {
            Some(column_index(&headers, col).ok_or_else(|| {
                SamplerError::Configuration(format!(
                    "text_column '{}' not found in CSV headers of '{}'",
                    col,
                    config.path.display()
                ))
            })?)
        } else {
            None
        };

        let mut records = Vec::new();

        let cols = ColumnIndices {
            anchor: anchor_idx,
            positive: positive_idx,
            text: text_idx,
        };

        for (row_idx, result) in reader.records().enumerate() {
            let row = result.map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!(
                    "failed to read row {} in '{}': {err}",
                    row_idx,
                    config.path.display()
                ),
            })?;

            if let Some(record) = build_record(config, &row, row_idx, &cols, created_at, updated_at)
            {
                records.push(record);
            }
        }

        Ok(records)
    }
}

/// Resolve a column name to its zero-based index in a header record.
fn column_index(headers: &csv::StringRecord, name: &str) -> Option<usize> {
    headers.iter().position(|h| h.eq_ignore_ascii_case(name))
}

/// Pre-resolved column indices for a CSV source.
struct ColumnIndices {
    anchor: Option<usize>,
    positive: Option<usize>,
    text: Option<usize>,
}

/// Build a [`DataRecord`] from a single CSV row.
///
/// Returns `None` when required column values are empty or missing.
fn build_record(
    config: &CsvSourceConfig,
    row: &csv::StringRecord,
    row_idx: usize,
    cols: &ColumnIndices,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
) -> Option<DataRecord> {
    let id = format!("{}::row_{}", config.source_id, row_idx);

    let sections = if config.is_role_mode() {
        // Role mode: anchor + optional positive
        let anchor_raw = cols.anchor.and_then(|i| row.get(i)).unwrap_or("");
        let anchor_text = normalize_inline_whitespace(anchor_raw);
        if anchor_text.is_empty() {
            return None;
        }

        let positive_text = if let Some(pidx) = cols.positive {
            let raw = row.get(pidx).unwrap_or("");
            let normalized = normalize_inline_whitespace(raw);
            if normalized.is_empty() {
                return None;
            }
            normalized
        } else {
            // Fall back to anchor text as positive when no positive column is set.
            anchor_text.clone()
        };

        let anchor_heading = config.anchor_column.as_deref();
        let positive_heading = config
            .positive_column
            .as_deref()
            .or(config.anchor_column.as_deref());

        vec![
            make_section(SectionRole::Anchor, anchor_heading, &anchor_text),
            make_section(SectionRole::Context, positive_heading, &positive_text),
        ]
    } else {
        // Text mode: single column used for both Anchor and Context (SimCSE pattern).
        let raw = cols.text.and_then(|i| row.get(i)).unwrap_or("");
        let text = normalize_inline_whitespace(raw);
        if text.is_empty() {
            return None;
        }

        let heading = config.text_column.as_deref();
        vec![
            make_section(SectionRole::Anchor, heading, &text),
            make_section(SectionRole::Context, heading, &text),
        ]
    };

    Some(DataRecord {
        id,
        source: config.source_id.clone(),
        created_at,
        updated_at,
        quality: QualityScore {
            trust: config.trust,
        },
        taxonomy: vec![config.source_id.clone()],
        sections,
        meta_prefix: None,
    })
}

impl IndexableSource for CsvSource {
    fn id(&self) -> &str {
        &self.config.source_id
    }

    fn len_hint(&self) -> Option<usize> {
        Some(self.records.len())
    }

    fn record_at(&self, idx: usize) -> Result<Option<DataRecord>, SamplerError> {
        Ok(self.records.get(idx).cloned())
    }
}

impl DataSource for CsvSource {
    fn id(&self) -> &str {
        &self.config.source_id
    }

    fn refresh(
        &self,
        _config: &SamplerConfig,
        cursor: Option<&SourceCursor>,
        limit: Option<usize>,
    ) -> Result<SourceSnapshot, SamplerError> {
        IndexablePager::new(&self.config.source_id).refresh(self, cursor, limit)
    }

    fn reported_record_count(&self, _config: &SamplerConfig) -> Result<u128, SamplerError> {
        Ok(self.records.len() as u128)
    }

    fn default_triplet_recipes(&self) -> Vec<TripletRecipe> {
        if !self.config.is_role_mode() {
            // Text mode: SimCSE-style recipe that allows same anchor/positive text.
            // Dropout noise provides the necessary embedding variation between
            // the two identical slots; the negative comes from a different record.
            return vec![TripletRecipe {
                name: CSV_RECIPE_TEXT_SIMCSE_WRONG_ARTICLE.into(),
                anchor: Selector::Role(SectionRole::Anchor),
                positive_selector: Selector::Role(SectionRole::Context),
                negative_selector: Selector::Role(SectionRole::Context),
                negative_strategy: NegativeStrategy::WrongArticle,
                weight: 1.0,
                instruction: None,
                allow_same_anchor_positive: true,
            }];
        }

        vec![
            // Primary lane: context (positive) negatives for broad coverage.
            TripletRecipe {
                name: CSV_RECIPE_ANCHOR_POSITIVE_WRONG_ARTICLE.into(),
                anchor: Selector::Role(SectionRole::Anchor),
                positive_selector: Selector::Role(SectionRole::Context),
                negative_selector: Selector::Role(SectionRole::Context),
                negative_strategy: NegativeStrategy::WrongArticle,
                weight: 0.75,
                instruction: None,
                allow_same_anchor_positive: false,
            },
            // Medium-hard lane: anchor-as-negative for discrimination pressure.
            TripletRecipe {
                name: CSV_RECIPE_ANCHOR_ANCHOR_WRONG_ARTICLE.into(),
                anchor: Selector::Role(SectionRole::Anchor),
                positive_selector: Selector::Role(SectionRole::Context),
                negative_selector: Selector::Role(SectionRole::Anchor),
                negative_strategy: NegativeStrategy::WrongArticle,
                weight: 0.25,
                instruction: None,
                allow_same_anchor_positive: false,
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SamplerConfig;
    use crate::source::DataSource;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_csv(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        write!(f, "{content}").unwrap();
        f
    }

    fn sampler_config() -> SamplerConfig {
        SamplerConfig {
            seed: 42,
            ..SamplerConfig::default()
        }
    }

    // ──────────────────────────────────────────────────────────── construction

    #[test]
    fn rejects_anchor_and_text_columns_together() {
        let f = write_csv("anchor,text\nhello,world\n");
        let err = CsvSource::new(
            CsvSourceConfig::new("src", f.path())
                .with_anchor_column("anchor")
                .with_text_column("text"),
        )
        .unwrap_err();
        assert!(
            matches!(err, SamplerError::Configuration(_)),
            "expected Configuration error, got {err:?}"
        );
    }

    #[test]
    fn rejects_missing_column_spec() {
        let f = write_csv("anchor,text\nhello,world\n");
        let err = CsvSource::new(CsvSourceConfig::new("src", f.path())).unwrap_err();
        assert!(matches!(err, SamplerError::Configuration(_)));
    }

    #[test]
    fn rejects_positive_without_anchor() {
        let f = write_csv("anchor,text\nhello,world\n");
        let err =
            CsvSource::new(CsvSourceConfig::new("src", f.path()).with_positive_column("text"))
                .unwrap_err();
        assert!(matches!(err, SamplerError::Configuration(_)));
    }

    #[test]
    fn rejects_missing_anchor_column_in_file() {
        let f = write_csv("question,answer\nhello,world\n");
        let err =
            CsvSource::new(CsvSourceConfig::new("src", f.path()).with_anchor_column("missing_col"))
                .unwrap_err();
        assert!(matches!(err, SamplerError::Configuration(_)));
    }

    #[test]
    fn rejects_missing_text_column_in_file() {
        let f = write_csv("question,answer\nhello,world\n");
        let err =
            CsvSource::new(CsvSourceConfig::new("src", f.path()).with_text_column("missing_col"))
                .unwrap_err();
        assert!(matches!(err, SamplerError::Configuration(_)));
    }

    #[test]
    fn rejects_missing_positive_column_in_file() {
        let f = write_csv("question,answer\nhello,world\n");
        let err = CsvSource::new(
            CsvSourceConfig::new("src", f.path())
                .with_anchor_column("question")
                .with_positive_column("missing_col"),
        )
        .unwrap_err();
        assert!(matches!(err, SamplerError::Configuration(_)));
    }

    // ──────────────────────────────────────────────────────────── role mode

    #[test]
    fn role_mode_anchor_and_positive() {
        let f = write_csv("question,answer\nWhat is Rust?,A systems language.\n");
        let source = CsvSource::new(
            CsvSourceConfig::new("qna", f.path())
                .with_anchor_column("question")
                .with_positive_column("answer"),
        )
        .unwrap();

        let snapshot = source.refresh(&sampler_config(), None, None).unwrap();
        assert_eq!(snapshot.records.len(), 1);
        let record = &snapshot.records[0];
        assert_eq!(record.source, "qna");
        assert_eq!(record.sections.len(), 2);
        assert_eq!(record.sections[0].role, SectionRole::Anchor);
        assert_eq!(record.sections[0].text, "What is Rust?");
        assert_eq!(record.sections[1].role, SectionRole::Context);
        assert_eq!(record.sections[1].text, "A systems language.");
    }

    #[test]
    fn role_mode_anchor_only_duplicates_to_context() {
        let f = write_csv("sentence\nHello world\n");
        let source = CsvSource::new(
            CsvSourceConfig::new("anchors", f.path()).with_anchor_column("sentence"),
        )
        .unwrap();

        let snapshot = source.refresh(&sampler_config(), None, None).unwrap();
        assert_eq!(snapshot.records.len(), 1);
        let record = &snapshot.records[0];
        assert_eq!(record.sections.len(), 2);
        assert_eq!(record.sections[0].role, SectionRole::Anchor);
        assert_eq!(record.sections[1].role, SectionRole::Context);
        // Context must mirror the anchor text.
        assert_eq!(record.sections[0].text, record.sections[1].text);
    }

    #[test]
    fn role_mode_skips_rows_with_empty_anchor() {
        let f = write_csv(
            "question,answer\n\
             What is Rust?,A systems language.\n\
             ,Missing anchor\n\
             What is Go?,A concurrent language.\n",
        );
        let source = CsvSource::new(
            CsvSourceConfig::new("qna", f.path())
                .with_anchor_column("question")
                .with_positive_column("answer"),
        )
        .unwrap();
        let snapshot = source.refresh(&sampler_config(), None, None).unwrap();
        assert_eq!(snapshot.records.len(), 2);
    }

    #[test]
    fn role_mode_skips_rows_with_empty_positive() {
        let f = write_csv(
            "question,answer\n\
             What is Rust?,A systems language.\n\
             What is Go?,\n",
        );
        let source = CsvSource::new(
            CsvSourceConfig::new("qna", f.path())
                .with_anchor_column("question")
                .with_positive_column("answer"),
        )
        .unwrap();
        let snapshot = source.refresh(&sampler_config(), None, None).unwrap();
        assert_eq!(snapshot.records.len(), 1);
    }

    // ──────────────────────────────────────────────────────────── text mode

    #[test]
    fn text_mode_produces_identical_anchor_and_context() {
        let f = write_csv("text\nThe quick brown fox\n");
        let source =
            CsvSource::new(CsvSourceConfig::new("corpus", f.path()).with_text_column("text"))
                .unwrap();

        let snapshot = source.refresh(&sampler_config(), None, None).unwrap();
        assert_eq!(snapshot.records.len(), 1);
        let record = &snapshot.records[0];
        assert_eq!(record.sections.len(), 2);
        assert_eq!(record.sections[0].role, SectionRole::Anchor);
        assert_eq!(record.sections[1].role, SectionRole::Context);
        assert_eq!(record.sections[0].text, record.sections[1].text);
    }

    #[test]
    fn text_mode_skips_empty_rows() {
        let f = write_csv("text\nHello\n\nWorld\n");
        let source =
            CsvSource::new(CsvSourceConfig::new("corpus", f.path()).with_text_column("text"))
                .unwrap();
        let snapshot = source.refresh(&sampler_config(), None, None).unwrap();
        assert_eq!(snapshot.records.len(), 2);
    }

    // ──────────────────────────────────────────────────────── quality / trust

    #[test]
    fn applies_trust_score() {
        let f = write_csv("text\nHello world\n");
        let source = CsvSource::new(
            CsvSourceConfig::new("corpus", f.path())
                .with_text_column("text")
                .with_trust(0.7),
        )
        .unwrap();
        let snapshot = source.refresh(&sampler_config(), None, None).unwrap();
        assert_eq!(snapshot.records[0].quality.trust, 0.7);
    }

    // ──────────────────────────────────────────────────────── default recipes

    #[test]
    fn text_mode_default_recipes_is_simcse() {
        let f = write_csv("text\nHello\n");
        let source =
            CsvSource::new(CsvSourceConfig::new("corpus", f.path()).with_text_column("text"))
                .unwrap();
        let recipes = source.default_triplet_recipes();
        assert_eq!(recipes.len(), 1);
        assert_eq!(recipes[0].name, CSV_RECIPE_TEXT_SIMCSE_WRONG_ARTICLE);
        assert!(
            recipes[0].allow_same_anchor_positive,
            "SimCSE recipe must allow same anchor/positive"
        );
    }

    #[test]
    fn role_mode_default_recipes_returns_two_recipes() {
        let f = write_csv("question,answer\nQ,A\n");
        let source = CsvSource::new(
            CsvSourceConfig::new("qna", f.path())
                .with_anchor_column("question")
                .with_positive_column("answer"),
        )
        .unwrap();
        let recipes = source.default_triplet_recipes();
        assert_eq!(recipes.len(), 2);
        let names: Vec<&str> = recipes.iter().map(|r| r.name.as_ref()).collect();
        assert!(names.contains(&CSV_RECIPE_ANCHOR_POSITIVE_WRONG_ARTICLE));
        assert!(names.contains(&CSV_RECIPE_ANCHOR_ANCHOR_WRONG_ARTICLE));
        assert!(
            recipes.iter().all(|r| !r.allow_same_anchor_positive),
            "role-mode recipes must not allow same anchor/positive"
        );
    }

    // ──────────────────────────────────────────────────────── IndexableSource

    #[test]
    fn len_hint_matches_loaded_record_count() {
        let f = write_csv("text\nAlpha\nBeta\nGamma\n");
        let source =
            CsvSource::new(CsvSourceConfig::new("corpus", f.path()).with_text_column("text"))
                .unwrap();
        assert_eq!(source.len_hint(), Some(3));
    }

    #[test]
    fn record_at_returns_correct_record() {
        let f = write_csv("question,answer\nFirst?,Yes.\nSecond?,No.\n");
        let source = CsvSource::new(
            CsvSourceConfig::new("qna", f.path())
                .with_anchor_column("question")
                .with_positive_column("answer"),
        )
        .unwrap();
        let r0 = source.record_at(0).unwrap().unwrap();
        let r1 = source.record_at(1).unwrap().unwrap();
        assert_eq!(r0.sections[0].text, "First?");
        assert_eq!(r1.sections[0].text, "Second?");
        assert!(source.record_at(99).unwrap().is_none());
    }

    // ──────────────────────────────────────────────────────── reported count

    #[test]
    fn reported_record_count_matches_loaded_records() {
        let f = write_csv("text\nAlpha\nBeta\n");
        let source =
            CsvSource::new(CsvSourceConfig::new("corpus", f.path()).with_text_column("text"))
                .unwrap();
        let count = source.reported_record_count(&sampler_config()).unwrap();
        assert_eq!(count, 2);
    }

    // ──────────────────────────────────────────────────────── stable record IDs

    #[test]
    fn record_ids_are_stable_across_refreshes() {
        let f = write_csv("text\nAlpha\nBeta\n");
        let source =
            CsvSource::new(CsvSourceConfig::new("corpus", f.path()).with_text_column("text"))
                .unwrap();
        let ids_a: Vec<_> = source
            .refresh(&sampler_config(), None, None)
            .unwrap()
            .records
            .iter()
            .map(|r| r.id.clone())
            .collect();
        let ids_b: Vec<_> = source
            .refresh(&sampler_config(), None, None)
            .unwrap()
            .records
            .iter()
            .map(|r| r.id.clone())
            .collect();
        // IDs must be the same set (order may differ due to pager permutation).
        let mut sorted_a = ids_a.clone();
        let mut sorted_b = ids_b.clone();
        sorted_a.sort();
        sorted_b.sort();
        assert_eq!(sorted_a, sorted_b);
    }

    // ──────────────────────────────────────────────────────── source id

    #[test]
    fn source_id_is_propagated_to_records() {
        let f = write_csv("text\nHello\n");
        let source =
            CsvSource::new(CsvSourceConfig::new("my_source", f.path()).with_text_column("text"))
                .unwrap();
        assert_eq!(DataSource::id(&source), "my_source");
        let snapshot = source.refresh(&sampler_config(), None, None).unwrap();
        assert_eq!(snapshot.records[0].source, "my_source");
    }

    // ──────────────────────────────────────────────────── column name trimming

    #[test]
    fn column_lookup_is_case_insensitive() {
        let f = write_csv("Question,Answer\nWhat is Rust?,A systems language.\n");
        // Lower-case lookup against mixed-case headers.
        let source = CsvSource::new(
            CsvSourceConfig::new("qna", f.path())
                .with_anchor_column("question")
                .with_positive_column("answer"),
        )
        .unwrap();
        let snapshot = source.refresh(&sampler_config(), None, None).unwrap();
        assert_eq!(snapshot.records.len(), 1);
        assert_eq!(snapshot.records[0].sections[0].text, "What is Rust?");
    }

    // ──────────────────────────────────────────────────── multi-row paging

    #[test]
    fn refresh_with_limit_returns_at_most_limit_records() {
        let f = write_csv("text\nA\nB\nC\nD\nE\n");
        let source =
            CsvSource::new(CsvSourceConfig::new("corpus", f.path()).with_text_column("text"))
                .unwrap();
        let snapshot = source.refresh(&sampler_config(), None, Some(3)).unwrap();
        assert!(
            snapshot.records.len() <= 3,
            "expected at most 3 records, got {}",
            snapshot.records.len()
        );
    }

    // ──────────────────────────────────────────────────────── with_headers

    #[test]
    fn with_headers_false_is_exercised_and_fails_column_lookup() {
        // The csv crate returns an empty header record when has_headers=false, so
        // any named-column lookup fails with a Configuration error.  This test
        // ensures the with_headers builder method body is reached.
        let f = write_csv("Hello world\n");
        let err = CsvSource::new(
            CsvSourceConfig::new("hdr", f.path())
                .with_headers(false)
                .with_text_column("text"),
        )
        .unwrap_err();
        assert!(
            matches!(err, SamplerError::Configuration(_)),
            "expected Configuration error, got {err:?}"
        );
    }

    // ──────────────────────────────────────────── validate: third error branch

    #[test]
    fn rejects_positive_and_text_column_without_anchor() {
        // positive_column + text_column (but no anchor_column) must reach the
        // third validate() check after passing the first two guards.
        let f = write_csv("text,answer\nhello,world\n");
        let err = CsvSource::new(
            CsvSourceConfig::new("src", f.path())
                .with_text_column("text")
                .with_positive_column("answer"),
        )
        .unwrap_err();
        assert!(
            matches!(err, SamplerError::Configuration(_)),
            "expected Configuration error, got {err:?}"
        );
    }

    // ───────────────────────────────────────────────── file open failure path

    #[test]
    fn returns_source_unavailable_for_nonexistent_file() {
        // Exercises the from_path error closure.
        let err = CsvSource::new(
            CsvSourceConfig::new("src", "/nonexistent/does-not-exist.csv").with_text_column("text"),
        )
        .unwrap_err();
        assert!(
            matches!(err, SamplerError::SourceUnavailable { .. }),
            "expected SourceUnavailable, got {err:?}"
        );
    }

    // ──────────────────────────────────────────────────── row parse error path

    #[test]
    fn returns_source_unavailable_for_malformed_row() {
        // With flexible(false), a data row that has more columns than the header
        // triggers a csv parse error, which must map to SourceUnavailable.
        let f = write_csv("question,answer\nWhat is Rust?,Good language.,extra_column\n");
        let err = CsvSource::new(
            CsvSourceConfig::new("src", f.path())
                .with_anchor_column("question")
                .with_positive_column("answer"),
        )
        .unwrap_err();
        assert!(
            matches!(err, SamplerError::SourceUnavailable { .. }),
            "expected SourceUnavailable for malformed row, got {err:?}"
        );
    }

    // ─────────────────────────────── text mode: whitespace-only cell is skipped

    #[test]
    fn text_mode_skips_whitespace_only_cells() {
        // A cell containing only spaces is trimmed to "" by the csv reader
        // (Trim::All).  Our normalize_inline_whitespace("") returns "", so
        // the record is skipped via the empty-text guard in build_record.
        // Blank lines (just "\n") are silently dropped by the csv crate before
        // reaching build_record, so we need an actual whitespace-valued cell.
        let f = write_csv("text\nHello\n   \nWorld\n");
        let source =
            CsvSource::new(CsvSourceConfig::new("corpus", f.path()).with_text_column("text"))
                .unwrap();
        let snapshot = source.refresh(&sampler_config(), None, None).unwrap();
        assert_eq!(
            snapshot.records.len(),
            2,
            "whitespace-only cell should be skipped"
        );
    }

    // ─────────────────────────────────────── IndexableSource::id() is reachable

    #[test]
    fn indexable_source_id_matches_config() {
        // IndexableSource::id() is only invoked through the trait
        // object; call it directly to ensure the implementation is exercised.
        let f = write_csv("text\nHello\n");
        let source =
            CsvSource::new(CsvSourceConfig::new("explicit_id", f.path()).with_text_column("text"))
                .unwrap();
        assert_eq!(IndexableSource::id(&source), "explicit_id");
    }
}
