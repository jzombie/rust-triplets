use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::config::TripletRecipe;
use crate::data::{DataRecord, QualityScore, RecordSection, SectionRole};
use crate::errors::SamplerError;
use crate::source::utilities::file_corpus::FileCorpusIndex;
use crate::source::{DataSource, SourceCursor, SourceSnapshot};
use crate::transport::fs::{file_times, is_text_file};
use crate::types::{CategoryId, SourceId, TaxonomyValue};
use crate::utils::{make_section, normalize_inline_whitespace};

/// Builds taxonomy values from a root path and file path.
pub type TaxonomyBuilder =
    Arc<dyn Fn(&Path, &Path, &SourceId) -> Vec<TaxonomyValue> + Send + Sync + 'static>;

/// Builds record sections from a normalized title and body.
pub type SectionBuilder = Arc<dyn Fn(&str, &str) -> Vec<RecordSection> + Send + Sync + 'static>;

/// Configuration for a generic filesystem-backed data source.
#[derive(Clone)]
pub struct FileSourceConfig {
    /// Stable source identifier used in records and persistence keys.
    pub source_id: SourceId,
    /// Root directory containing source files.
    pub root: PathBuf,
    /// Default quality trust score applied to generated records.
    pub trust: f32,
    /// Optional trust overrides keyed by taxonomy segment.
    pub category_trust: HashMap<CategoryId, f32>,
    /// Whether to follow symlinks during index walking.
    pub follow_links: bool,
    /// Whether indexing should include only text files.
    pub text_files_only: bool,
    /// Whether deterministic directory grouping is enabled.
    pub group_by_directory: bool,
    /// Whether title extraction should replace underscores with spaces.
    pub title_replace_underscores: bool,
    /// Optional default recipes returned by this source.
    pub default_triplet_recipes: Vec<TripletRecipe>,
    /// Taxonomy builder invoked per file.
    pub taxonomy_builder: TaxonomyBuilder,
    /// Section builder invoked per file.
    pub section_builder: SectionBuilder,
}

impl FileSourceConfig {
    /// Create a config for a filesystem source with explicit id and root.
    pub fn new(source_id: impl Into<SourceId>, root: impl Into<PathBuf>) -> Self {
        Self {
            source_id: source_id.into(),
            root: root.into(),
            trust: 0.85,
            category_trust: HashMap::new(),
            follow_links: true,
            text_files_only: false,
            group_by_directory: true,
            title_replace_underscores: true,
            default_triplet_recipes: Vec::new(),
            taxonomy_builder: Arc::new(taxonomy_from_path),
            section_builder: Arc::new(anchor_context_sections),
        }
    }

    /// Override default trust score.
    pub fn with_trust(mut self, trust: f32) -> Self {
        self.trust = trust;
        self
    }

    /// Add a taxonomy-segment trust override.
    pub fn with_category_trust(mut self, category: impl Into<String>, trust: f32) -> Self {
        self.category_trust
            .insert(category.into().to_lowercase(), trust);
        self
    }

    /// Override whether symlinks are followed during index walk.
    pub fn with_follow_links(mut self, follow_links: bool) -> Self {
        self.follow_links = follow_links;
        self
    }

    /// Override whether index walk includes only text files.
    pub fn with_text_files_only(mut self, text_files_only: bool) -> Self {
        self.text_files_only = text_files_only;
        self
    }

    /// Enable or disable deterministic directory grouping.
    pub fn with_directory_grouping(mut self, group_by_directory: bool) -> Self {
        self.group_by_directory = group_by_directory;
        self
    }

    /// Set whether title extraction replaces underscores with spaces.
    pub fn with_title_replace_underscores(mut self, replace_underscores: bool) -> Self {
        self.title_replace_underscores = replace_underscores;
        self
    }

    /// Set source-provided default triplet recipes.
    pub fn with_default_triplet_recipes(mut self, recipes: Vec<TripletRecipe>) -> Self {
        self.default_triplet_recipes = recipes;
        self
    }

    /// Set a custom taxonomy builder.
    pub fn with_taxonomy_builder(mut self, taxonomy_builder: TaxonomyBuilder) -> Self {
        self.taxonomy_builder = taxonomy_builder;
        self
    }

    /// Set a custom section builder.
    pub fn with_section_builder(mut self, section_builder: SectionBuilder) -> Self {
        self.section_builder = section_builder;
        self
    }
}

/// Generic filesystem-backed source with configurable taxonomy and section mapping.
pub struct FileSource {
    config: FileSourceConfig,
}

impl FileSource {
    /// Create a generic file source from configuration.
    pub fn new(config: FileSourceConfig) -> Self {
        Self { config }
    }

    fn file_corpus_index(&self) -> FileCorpusIndex {
        FileCorpusIndex::new(&self.config.root, &self.config.source_id)
            .with_follow_links(self.config.follow_links)
            .with_text_files_only(self.config.text_files_only)
            .with_directory_grouping(self.config.group_by_directory)
    }

    fn trust_for_taxonomy(&self, taxonomy: &[String]) -> f32 {
        for segment in taxonomy.iter().skip(1) {
            if let Some(weight) = self.config.category_trust.get(&segment.to_lowercase()) {
                return *weight;
            }
        }
        self.config.trust
    }

    fn build_record(&self, path: &Path) -> Result<Option<DataRecord>, SamplerError> {
        if !is_text_file(path) {
            return Ok(None);
        }
        let title = FileCorpusIndex::normalized_title_from_stem(
            path,
            &self.config.source_id,
            self.config.title_replace_underscores,
        )?;
        if title.is_empty() {
            return Ok(None);
        }

        let body_raw = std::fs::read_to_string(path)?;
        let body = normalize_inline_whitespace(body_raw);
        if body.is_empty() {
            return Ok(None);
        }

        let taxonomy =
            (self.config.taxonomy_builder)(&self.config.root, path, &self.config.source_id);
        let sections = (self.config.section_builder)(&title, &body);
        let trust = self.trust_for_taxonomy(&taxonomy);
        let (created_at, updated_at) = file_times(path);

        Ok(Some(DataRecord {
            id: FileCorpusIndex::source_scoped_record_id(
                &self.config.source_id,
                &self.config.root,
                path,
            ),
            source: self.config.source_id.clone(),
            created_at,
            updated_at,
            quality: QualityScore { trust },
            taxonomy,
            sections,
            meta_prefix: None,
        }))
    }
}

impl DataSource for FileSource {
    fn id(&self) -> &str {
        &self.config.source_id
    }

    fn refresh(
        &self,
        cursor: Option<&SourceCursor>,
        limit: Option<usize>,
    ) -> Result<SourceSnapshot, SamplerError> {
        self.file_corpus_index()
            .refresh_indexable(cursor, limit, |path| self.build_record(path))
    }

    fn reported_record_count(&self) -> Result<u128, SamplerError> {
        self.file_corpus_index()
            .indexed_record_count()
            .map(|count| count as u128)
    }

    fn default_triplet_recipes(&self) -> Vec<TripletRecipe> {
        self.config.default_triplet_recipes.clone()
    }
}

/// Build default taxonomy from the file path relative to `root`.
///
/// Output shape is `[source_id, <parent segments...>]`.
pub fn taxonomy_from_path(root: &Path, path: &Path, source_id: &SourceId) -> Vec<TaxonomyValue> {
    let mut taxonomy = vec![source_id.to_string()];
    if let Ok(rel) = path.strip_prefix(root)
        && let Some(parent) = rel.parent()
    {
        for segment in parent.iter() {
            taxonomy.push(segment.to_string_lossy().to_string());
        }
    }
    taxonomy
}

/// Build a default two-section payload of title anchor and body context.
pub fn anchor_context_sections(title: &str, body: &str) -> Vec<RecordSection> {
    vec![
        make_section(SectionRole::Anchor, None, title),
        make_section(SectionRole::Context, None, body),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{NegativeStrategy, Selector};
    use tempfile::tempdir;

    #[test]
    fn reads_records_without_default_source_id() {
        let temp = tempdir().unwrap();
        let category = temp.path().join("factual");
        std::fs::create_dir_all(&category).unwrap();
        std::fs::write(
            category.join("What_is_alpha.txt"),
            "Alpha measures risk-adjusted outperformance.",
        )
        .unwrap();

        let source = FileSource::new(FileSourceConfig::new("qa_custom", temp.path()));
        let snapshot = source.refresh(None, None).unwrap();

        assert_eq!(snapshot.records.len(), 1);
        assert_eq!(snapshot.records[0].source, "qa_custom");
    }

    #[test]
    fn applies_category_trust_overrides() {
        let temp = tempdir().unwrap();
        let factual = temp.path().join("factual");
        let opinion = temp.path().join("opinionated");
        std::fs::create_dir_all(&factual).unwrap();
        std::fs::create_dir_all(&opinion).unwrap();
        std::fs::write(
            factual.join("What_is_beta.txt"),
            "Beta compares volatility.",
        )
        .unwrap();
        std::fs::write(
            opinion.join("Will_rates_fall.txt"),
            "Probably not this year.",
        )
        .unwrap();

        let source = FileSource::new(
            FileSourceConfig::new("qa_weighted", temp.path())
                .with_category_trust("factual", 0.95)
                .with_category_trust("opinionated", 0.6),
        );
        let snapshot = source.refresh(None, None).unwrap();

        let factual_record = snapshot
            .records
            .iter()
            .find(|record| record.taxonomy.iter().any(|value| value == "factual"))
            .unwrap();
        let opinion_record = snapshot
            .records
            .iter()
            .find(|record| record.taxonomy.iter().any(|value| value == "opinionated"))
            .unwrap();
        assert_eq!(factual_record.quality.trust, 0.95);
        assert_eq!(opinion_record.quality.trust, 0.6);
    }

    #[test]
    fn supports_custom_sections_and_default_recipes() {
        let temp = tempdir().unwrap();
        std::fs::write(
            temp.path().join("What_is_gamma.txt"),
            "Gamma measures convexity.",
        )
        .unwrap();

        let sections: SectionBuilder = Arc::new(|question, answer| {
            vec![
                make_section(SectionRole::Anchor, Some("Question"), question),
                make_section(SectionRole::Context, Some("Answer"), answer),
            ]
        });

        let recipes = vec![TripletRecipe {
            name: "question_answer".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::QuestionAnswerMismatch,
            weight: 1.0,
            instruction: None,
        }];

        let source = FileSource::new(
            FileSourceConfig::new("qa_sections", temp.path())
                .with_section_builder(sections)
                .with_default_triplet_recipes(recipes.clone()),
        );

        let snapshot = source.refresh(None, None).unwrap();
        assert_eq!(snapshot.records.len(), 1);
        assert_eq!(snapshot.records[0].sections.len(), 2);
        assert_eq!(source.default_triplet_recipes().len(), recipes.len());
    }

    #[test]
    fn taxonomy_from_path_handles_nested_and_non_descendant_paths() {
        let temp = tempdir().unwrap();
        let root = temp.path().join("root");
        std::fs::create_dir_all(root.join("topic/subtopic")).unwrap();

        let nested = root.join("topic/subtopic/doc.txt");
        let taxonomy = taxonomy_from_path(&root, &nested, &"qa_tax".to_string());
        assert_eq!(taxonomy, vec!["qa_tax", "topic", "subtopic"]);

        let outside = temp.path().join("outside.txt");
        let outside_taxonomy = taxonomy_from_path(&root, &outside, &"qa_tax".to_string());
        assert_eq!(outside_taxonomy, vec!["qa_tax"]);
    }

    #[test]
    fn anchor_context_sections_build_expected_roles_and_text() {
        let sections = anchor_context_sections("What is delta", "Delta is change over time.");
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].role, SectionRole::Anchor);
        assert_eq!(sections[0].text, "What is delta");
        assert_eq!(sections[1].role, SectionRole::Context);
        assert_eq!(sections[1].text, "Delta is change over time.");
    }

    #[test]
    fn title_replace_underscores_toggle_changes_anchor_title_text() {
        let temp = tempdir().unwrap();
        std::fs::write(
            temp.path().join("What_is_delta.txt"),
            "Delta captures directional change.",
        )
        .unwrap();

        let source_default =
            FileSource::new(FileSourceConfig::new("qa_title_default", temp.path()));
        let default_snapshot = source_default.refresh(None, Some(1)).unwrap();
        assert_eq!(default_snapshot.records.len(), 1);
        assert_eq!(
            default_snapshot.records[0].sections[0].text,
            "What is delta"
        );

        let source_preserve = FileSource::new(
            FileSourceConfig::new("qa_title_preserve", temp.path())
                .with_title_replace_underscores(false),
        );
        let preserve_snapshot = source_preserve.refresh(None, Some(1)).unwrap();
        assert_eq!(preserve_snapshot.records.len(), 1);
        assert_eq!(
            preserve_snapshot.records[0].sections[0].text,
            "What_is_delta"
        );
    }

    #[test]
    fn refresh_skips_non_txt_files_even_when_text_only_disabled() {
        let temp = tempdir().unwrap();
        std::fs::write(temp.path().join("notes.md"), "markdown should be skipped").unwrap();
        std::fs::write(temp.path().join("doc.txt"), "plain text should be indexed").unwrap();

        let source = FileSource::new(
            FileSourceConfig::new("qa_filtering", temp.path()).with_text_files_only(false),
        );
        let snapshot = source.refresh(None, None).unwrap();
        assert_eq!(snapshot.records.len(), 1);
        assert!(snapshot.records[0].id.contains("doc.txt"));
    }

    #[test]
    fn trust_falls_back_to_default_and_count_and_id_are_exposed() {
        let temp = tempdir().unwrap();
        let docs = temp.path().join("docs");
        std::fs::create_dir_all(&docs).unwrap();
        std::fs::write(docs.join("alpha.txt"), "Alpha body.").unwrap();

        let source = FileSource::new(
            FileSourceConfig::new("qa_count", temp.path())
                .with_trust(0.42)
                .with_category_trust("factual", 0.95)
                .with_taxonomy_builder(Arc::new(|_, _, source_id| {
                    vec![source_id.clone(), "UNMATCHED".to_string()]
                })),
        );

        let snapshot = source.refresh(None, None).unwrap();
        assert_eq!(snapshot.records.len(), 1);
        assert_eq!(snapshot.records[0].quality.trust, 0.42);
        assert_eq!(source.id(), "qa_count");
        assert_eq!(source.reported_record_count().unwrap(), 1);
    }
}
