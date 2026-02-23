use chrono::{DateTime, Utc};
use std::collections::HashSet;

use super::{DataSource, IndexablePager, SourceCursor, SourceSnapshot};
use crate::config::{NegativeStrategy, Selector, TripletRecipe};
use crate::data::{DataRecord, QualityScore, SectionRole};
use crate::errors::SamplerError;
use crate::utils::make_section;

/// A named text field in a row-like record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextField {
    /// Stable column/field name.
    pub name: String,
    /// Text value for this field.
    pub text: String,
}

/// Source-agnostic row contract for columnar or structured backends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowView {
    /// Stable row identifier, if provided by the backend.
    pub row_id: Option<String>,
    /// Optional canonical timestamp for ordering/metadata.
    pub timestamp: Option<DateTime<Utc>>,
    /// Named text fields extracted from the row.
    pub text_fields: Vec<TextField>,
}

/// Source-agnostic text contract for plain-text backends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextView {
    /// Stable record identifier.
    pub record_id: String,
    /// Full text payload.
    pub text: String,
    /// Optional canonical timestamp.
    pub timestamp: Option<DateTime<Utc>>,
}

/// Index-addressable source that yields `RowView` records.
pub trait RowViewSource: Send + Sync {
    /// Stable source identifier.
    fn id(&self) -> &str;
    /// Current index domain size, typically `Some(total_records)`.
    fn len_hint(&self) -> Option<usize>;
    /// Return row at index `idx`, or `None` when missing/sparse.
    fn row_at(&self, idx: usize) -> Result<Option<RowView>, SamplerError>;
}

/// Index-addressable source that yields plain `TextView` records.
pub trait TextViewSource: Send + Sync {
    /// Stable source identifier.
    fn id(&self) -> &str;
    /// Current index domain size, typically `Some(total_records)`.
    fn len_hint(&self) -> Option<usize>;
    /// Return text record at index `idx`, or `None` when missing/sparse.
    fn text_at(&self, idx: usize) -> Result<Option<TextView>, SamplerError>;
}

/// Adapter that converts source-agnostic `RowView` values into `DataRecord`s.
pub trait RowViewAdapter: Send + Sync {
    /// Stable source id associated with this adapter.
    fn source_id(&self) -> &str;
    /// Convert one row into a sampler-ready `DataRecord`.
    fn row_to_record(
        &self,
        row: &RowView,
        row_index: u64,
    ) -> Result<Option<DataRecord>, SamplerError>;
    /// Source-provided default triplet recipes derived from adapter behavior.
    fn default_triplet_recipes(&self) -> Vec<TripletRecipe>;
}

/// Generic bridge from `RowViewSource` + `RowViewAdapter` to `DataSource`.
pub struct RowViewDataSourceAdapter<T, A> {
    source: T,
    adapter: A,
}

impl<T, A> RowViewDataSourceAdapter<T, A>
where
    T: RowViewSource,
    A: RowViewAdapter,
{
    /// Create a new bridge adapter.
    pub fn new(source: T, adapter: A) -> Self {
        Self { source, adapter }
    }

    /// Access the wrapped row source.
    pub fn source(&self) -> &T {
        &self.source
    }

    /// Access the wrapped row adapter.
    pub fn adapter(&self) -> &A {
        &self.adapter
    }
}

impl<T, A> DataSource for RowViewDataSourceAdapter<T, A>
where
    T: RowViewSource,
    A: RowViewAdapter,
{
    fn id(&self) -> &str {
        self.source.id()
    }

    fn refresh(
        &self,
        cursor: Option<&SourceCursor>,
        limit: Option<usize>,
    ) -> Result<SourceSnapshot, SamplerError> {
        if self.adapter.source_id() != self.source.id() {
            return Err(SamplerError::Configuration(format!(
                "row view adapter source id mismatch: source='{}' adapter='{}'",
                self.source.id(),
                self.adapter.source_id()
            )));
        }

        let total = self
            .source
            .len_hint()
            .ok_or_else(|| SamplerError::SourceInconsistent {
                source_id: self.source.id().to_string(),
                details: "row source did not provide len_hint".to_string(),
            })?;
        let pager = IndexablePager::new(self.source.id());
        pager.refresh_with(total, cursor, limit, |idx| {
            match self.source.row_at(idx)? {
                Some(row) => self.adapter.row_to_record(&row, idx as u64),
                None => Ok(None),
            }
        })
    }

    fn reported_record_count(&self) -> Option<u128> {
        self.source.len_hint().map(|count| count as u128)
    }

    fn default_triplet_recipes(&self) -> Vec<TripletRecipe> {
        self.adapter.default_triplet_recipes()
    }
}

/// Policy for selecting text fields from a `RowView`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TextFieldPolicy {
    /// Select exactly these field names.
    Explicit(Vec<String>),
    /// Select all available text fields.
    AllTextColumns,
    /// Select all available text fields except the provided names.
    AllTextColumnsMinus(Vec<String>),
    /// Select remaining fields that have not already been consumed.
    RemainingTextColumns,
}

/// Field-selection mapping from a row into anchor/positive/context sets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowFieldMapping {
    /// Policy used to resolve anchor fields.
    pub anchor: TextFieldPolicy,
    /// Policy used to resolve positive fields.
    pub positive: TextFieldPolicy,
    /// Policy used to resolve context fields.
    pub context: TextFieldPolicy,
    /// Optional field names excluded globally from all sets.
    pub exclude_fields: Vec<String>,
    /// Match field names case-sensitively when true.
    pub case_sensitive: bool,
}

/// Concrete adapter that maps resolved row fields into record sections.
#[derive(Clone, Debug)]
pub struct MappedRowViewAdapter {
    source_id: String,
    mapping: RowFieldMapping,
    require_anchor: bool,
    require_positive: bool,
}

impl MappedRowViewAdapter {
    /// Build an adapter with explicit source id and field mapping policy.
    pub fn new(source_id: impl Into<String>, mapping: RowFieldMapping) -> Self {
        Self {
            source_id: source_id.into(),
            mapping,
            require_anchor: true,
            require_positive: true,
        }
    }

    /// Require at least one resolved anchor field to emit a record.
    pub fn with_require_anchor(mut self, require_anchor: bool) -> Self {
        self.require_anchor = require_anchor;
        self
    }

    /// Require at least one resolved positive field to emit a record.
    pub fn with_require_positive(mut self, require_positive: bool) -> Self {
        self.require_positive = require_positive;
        self
    }

    fn record_id_for(&self, row: &RowView, row_index: u64) -> String {
        let local = row
            .row_id
            .as_ref()
            .cloned()
            .unwrap_or_else(|| format!("row_{row_index}"));
        format!("{}::{local}", self.source_id)
    }

    fn selected_texts<'a>(&self, row: &'a RowView, names: &[String]) -> Vec<(&'a str, &'a str)> {
        let selected = self
            .mapping
            .normalized_set(names.iter().map(String::as_str));
        row.text_fields
            .iter()
            .filter(|field| selected.contains(&self.mapping.normalize(&field.name)))
            .map(|field| (field.name.as_str(), field.text.as_str()))
            .collect()
    }
}

impl RowViewAdapter for MappedRowViewAdapter {
    fn source_id(&self) -> &str {
        &self.source_id
    }

    fn row_to_record(
        &self,
        row: &RowView,
        row_index: u64,
    ) -> Result<Option<DataRecord>, SamplerError> {
        let resolved = self.mapping.resolve(row);
        if self.require_anchor && resolved.anchor.is_empty() {
            return Err(SamplerError::SourceInconsistent {
                source_id: self.source_id.clone(),
                details: "row mapping resolved no anchor fields".to_string(),
            });
        }
        if self.require_positive && resolved.positive.is_empty() {
            return Err(SamplerError::SourceInconsistent {
                source_id: self.source_id.clone(),
                details: "row mapping resolved no positive fields".to_string(),
            });
        }

        let mut sections = Vec::new();
        for (name, text) in self.selected_texts(row, &resolved.anchor) {
            sections.push(make_section(SectionRole::Anchor, Some(name), text));
        }
        for (name, text) in self.selected_texts(row, &resolved.positive) {
            sections.push(make_section(SectionRole::Context, Some(name), text));
        }
        for (name, text) in self.selected_texts(row, &resolved.context) {
            sections.push(make_section(SectionRole::Context, Some(name), text));
        }

        if sections.is_empty() {
            return Ok(None);
        }

        let timestamp = row.timestamp.unwrap_or(DateTime::<Utc>::UNIX_EPOCH);
        Ok(Some(DataRecord {
            id: self.record_id_for(row, row_index),
            source: self.source_id.clone(),
            created_at: timestamp,
            updated_at: timestamp,
            quality: QualityScore::default(),
            taxonomy: Vec::new(),
            sections,
            meta_prefix: None,
        }))
    }

    fn default_triplet_recipes(&self) -> Vec<TripletRecipe> {
        vec![TripletRecipe {
            name: "rowview_anchor_context".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
        }]
    }
}

impl Default for RowFieldMapping {
    fn default() -> Self {
        Self {
            anchor: TextFieldPolicy::Explicit(vec!["anchor".to_string()]),
            positive: TextFieldPolicy::Explicit(vec!["positive".to_string()]),
            context: TextFieldPolicy::RemainingTextColumns,
            exclude_fields: Vec::new(),
            case_sensitive: false,
        }
    }
}

/// Resolved field-name sets produced by `RowFieldMapping::resolve`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedFieldSets {
    /// Resolved anchor field names.
    pub anchor: Vec<String>,
    /// Resolved positive field names.
    pub positive: Vec<String>,
    /// Resolved context field names.
    pub context: Vec<String>,
}

impl RowFieldMapping {
    /// Resolve anchor/positive/context field names for `row`.
    pub fn resolve(&self, row: &RowView) -> ResolvedFieldSets {
        let excludes = self.normalized_set(self.exclude_fields.iter().map(String::as_str));
        let mut consumed = HashSet::new();

        let anchor = self.resolve_policy(&self.anchor, row, &excludes, &consumed);
        consumed.extend(anchor.iter().map(|field| self.normalize(field)));

        let positive = self.resolve_policy(&self.positive, row, &excludes, &consumed);
        consumed.extend(positive.iter().map(|field| self.normalize(field)));

        let context = self.resolve_policy(&self.context, row, &excludes, &consumed);

        ResolvedFieldSets {
            anchor,
            positive,
            context,
        }
    }

    fn resolve_policy(
        &self,
        policy: &TextFieldPolicy,
        row: &RowView,
        excludes: &HashSet<String>,
        consumed: &HashSet<String>,
    ) -> Vec<String> {
        let explicit = match policy {
            TextFieldPolicy::Explicit(names) => Some(names),
            TextFieldPolicy::AllTextColumnsMinus(names) => Some(names),
            TextFieldPolicy::AllTextColumns | TextFieldPolicy::RemainingTextColumns => None,
        };
        let explicit_set =
            explicit.map(|names| self.normalized_set(names.iter().map(String::as_str)));

        row.text_fields
            .iter()
            .map(|field| field.name.clone())
            .filter(|field_name| {
                let normalized = self.normalize(field_name);
                if excludes.contains(&normalized) {
                    return false;
                }

                match policy {
                    TextFieldPolicy::Explicit(_) => explicit_set
                        .as_ref()
                        .map(|set| set.contains(&normalized))
                        .unwrap_or(false),
                    TextFieldPolicy::AllTextColumns => true,
                    TextFieldPolicy::AllTextColumnsMinus(_) => explicit_set
                        .as_ref()
                        .map(|set| !set.contains(&normalized))
                        .unwrap_or(true),
                    TextFieldPolicy::RemainingTextColumns => !consumed.contains(&normalized),
                }
            })
            .collect()
    }

    fn normalized_set<'a>(&self, values: impl Iterator<Item = &'a str>) -> HashSet<String> {
        values.map(|value| self.normalize(value)).collect()
    }

    fn normalize(&self, value: &str) -> String {
        if self.case_sensitive {
            value.to_string()
        } else {
            value.to_ascii_lowercase()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[derive(Clone)]
    struct MemoryRowSource {
        id: String,
        rows: Vec<RowView>,
    }

    impl RowViewSource for MemoryRowSource {
        fn id(&self) -> &str {
            &self.id
        }

        fn len_hint(&self) -> Option<usize> {
            Some(self.rows.len())
        }

        fn row_at(&self, idx: usize) -> Result<Option<RowView>, SamplerError> {
            Ok(self.rows.get(idx).cloned())
        }
    }

    fn row(names: &[&str]) -> RowView {
        RowView {
            row_id: Some("r1".to_string()),
            timestamp: None,
            text_fields: names
                .iter()
                .map(|name| TextField {
                    name: (*name).to_string(),
                    text: format!("value for {name}"),
                })
                .collect(),
        }
    }

    #[test]
    fn default_mapping_uses_anchor_positive_and_remaining_context() {
        let mapping = RowFieldMapping::default();
        let resolved = mapping.resolve(&row(&["anchor", "positive", "title", "body"]));

        assert_eq!(resolved.anchor, vec!["anchor".to_string()]);
        assert_eq!(resolved.positive, vec!["positive".to_string()]);
        assert_eq!(
            resolved.context,
            vec!["title".to_string(), "body".to_string()]
        );
    }

    #[test]
    fn explicit_and_all_text_minus_support_columnar_and_all_columns_modes() {
        let mapping = RowFieldMapping {
            anchor: TextFieldPolicy::Explicit(vec!["question".to_string()]),
            positive: TextFieldPolicy::Explicit(vec!["answer".to_string()]),
            context: TextFieldPolicy::AllTextColumnsMinus(vec![
                "question".to_string(),
                "answer".to_string(),
            ]),
            exclude_fields: vec!["id".to_string()],
            case_sensitive: false,
        };

        let resolved = mapping.resolve(&row(&["id", "question", "answer", "title", "body"]));
        assert_eq!(resolved.anchor, vec!["question".to_string()]);
        assert_eq!(resolved.positive, vec!["answer".to_string()]);
        assert_eq!(
            resolved.context,
            vec!["title".to_string(), "body".to_string()]
        );
    }

    #[test]
    fn mapped_adapter_converts_row_to_record_with_expected_sections() {
        let mapping = RowFieldMapping {
            anchor: TextFieldPolicy::Explicit(vec!["question".to_string()]),
            positive: TextFieldPolicy::Explicit(vec!["answer".to_string()]),
            context: TextFieldPolicy::RemainingTextColumns,
            exclude_fields: vec![],
            case_sensitive: false,
        };
        let adapter = MappedRowViewAdapter::new("hf", mapping);
        let input = row(&["question", "answer", "title", "body"]);

        let record = adapter.row_to_record(&input, 7).unwrap().unwrap();
        assert_eq!(record.id, "hf::r1");
        assert_eq!(record.source, "hf");
        assert_eq!(record.sections.len(), 4);
        assert!(matches!(record.sections[0].role, SectionRole::Anchor));
        assert!(matches!(record.sections[1].role, SectionRole::Context));
    }

    #[test]
    fn mapped_adapter_supports_all_text_columns_mode() {
        let mapping = RowFieldMapping {
            anchor: TextFieldPolicy::AllTextColumns,
            positive: TextFieldPolicy::AllTextColumns,
            context: TextFieldPolicy::RemainingTextColumns,
            exclude_fields: vec![],
            case_sensitive: false,
        };
        let adapter = MappedRowViewAdapter::new("all", mapping)
            .with_require_anchor(false)
            .with_require_positive(false);
        let input = row(&["c1", "c2", "c3"]);
        let record = adapter.row_to_record(&input, 1).unwrap().unwrap();
        assert_eq!(record.sections.len(), 6);
    }

    #[test]
    fn mapped_adapter_errors_when_required_fields_missing() {
        let mapping = RowFieldMapping::default();
        let adapter = MappedRowViewAdapter::new("hf", mapping);
        let input = row(&["body_only"]);
        let err = adapter.row_to_record(&input, 0).unwrap_err();
        assert!(matches!(
            err,
            SamplerError::SourceInconsistent { details, .. } if details.contains("anchor")
        ));
    }

    #[test]
    fn mapped_adapter_exposes_default_recipe() {
        let recipes =
            MappedRowViewAdapter::new("hf", RowFieldMapping::default()).default_triplet_recipes();
        assert_eq!(recipes.len(), 1);
        assert_eq!(recipes[0].name, "rowview_anchor_context");
    }

    #[test]
    fn row_view_data_source_adapter_refreshes_with_deterministic_paging() {
        let ts = Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap();
        let source = MemoryRowSource {
            id: "hf".to_string(),
            rows: vec![
                RowView {
                    row_id: Some("1".to_string()),
                    timestamp: Some(ts),
                    text_fields: vec![
                        TextField {
                            name: "anchor".to_string(),
                            text: "a1".to_string(),
                        },
                        TextField {
                            name: "positive".to_string(),
                            text: "p1".to_string(),
                        },
                        TextField {
                            name: "extra".to_string(),
                            text: "e1".to_string(),
                        },
                    ],
                },
                RowView {
                    row_id: Some("2".to_string()),
                    timestamp: Some(ts),
                    text_fields: vec![
                        TextField {
                            name: "anchor".to_string(),
                            text: "a2".to_string(),
                        },
                        TextField {
                            name: "positive".to_string(),
                            text: "p2".to_string(),
                        },
                    ],
                },
            ],
        };
        let adapter = MappedRowViewAdapter::new("hf", RowFieldMapping::default());
        let ds = RowViewDataSourceAdapter::new(source, adapter);

        let first = ds.refresh(None, Some(1)).unwrap();
        assert_eq!(first.records.len(), 1);
        assert!(first.records[0].id.starts_with("hf::"));
        let second = ds.refresh(Some(&first.cursor), Some(1)).unwrap();
        assert_eq!(second.records.len(), 1);
        assert_ne!(first.records[0].id, second.records[0].id);
        assert_eq!(ds.reported_record_count(), Some(2));
        assert_eq!(ds.default_triplet_recipes().len(), 1);
    }

    #[test]
    fn row_view_data_source_adapter_rejects_source_id_mismatch() {
        let source = MemoryRowSource {
            id: "source_a".to_string(),
            rows: vec![row(&["anchor", "positive"])],
        };
        let adapter = MappedRowViewAdapter::new("source_b", RowFieldMapping::default());
        let ds = RowViewDataSourceAdapter::new(source, adapter);
        let err = ds.refresh(None, Some(1)).unwrap_err();
        assert!(matches!(
            err,
            SamplerError::Configuration(msg) if msg.contains("source id mismatch")
        ));
    }
}
