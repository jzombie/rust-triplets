use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use triplets::config::TripletRecipe;
use triplets::data::{DataRecord, QualityScore, SectionRole};
use triplets::metadata::META_FIELD_DATE;
use triplets::source::indexing::file_corpus::FileCorpusIndex;
use triplets::source::{DataSource, InMemorySource, SourceCursor, SourceSnapshot};
use triplets::types::SourceId;
use triplets::utils::{make_section, normalize_inline_whitespace};
use walkdir::WalkDir;

#[cfg(feature = "huggingface")]
use triplets::{HuggingFaceRowSource, HuggingFaceRowsConfig};

#[derive(Debug, Clone)]
pub struct SourceRoots {
    pub sources: Vec<PathBuf>,
}

pub fn resolve_source_roots(source_overrides: Vec<String>) -> Result<SourceRoots, Box<dyn Error>> {
    if source_overrides.is_empty() {
        let defaults = vec![
            resolve_root(
                None,
                &["TRIPLETS_EXAMPLE_SOURCE_A_DIR"],
                &["example_dataset/source_a", "../example_dataset/source_a"],
                "Could not locate example source_a directory.",
            )?,
            resolve_root(
                None,
                &["TRIPLETS_EXAMPLE_SOURCE_B_DIR"],
                &["example_dataset/source_b", "../example_dataset/source_b"],
                "Could not locate example source_b directory.",
            )?,
            resolve_root(
                None,
                &["TRIPLETS_EXAMPLE_SOURCE_C_DIR"],
                &["example_dataset/source_c", "../example_dataset/source_c"],
                "Could not locate example source_c directory.",
            )?,
            resolve_root(
                None,
                &["TRIPLETS_EXAMPLE_SOURCE_D_DIR"],
                &["example_dataset/source_d", "../example_dataset/source_d"],
                "Could not locate example source_d directory.",
            )?,
            resolve_root(
                None,
                &["TRIPLETS_EXAMPLE_SOURCE_E_DIR"],
                &["example_dataset/source_e", "../example_dataset/source_e"],
                "Could not locate example source_e directory.",
            )?,
            resolve_root(
                None,
                &["TRIPLETS_EXAMPLE_SOURCE_F_DIR"],
                &["example_dataset/source_f", "../example_dataset/source_f"],
                "Could not locate example source_f directory.",
            )?,
        ];
        return Ok(SourceRoots { sources: defaults });
    }

    let mut sources = Vec::with_capacity(source_overrides.len());
    for value in source_overrides {
        if value.is_empty() {
            continue;
        }
        let root = PathBuf::from(value);
        if !root.is_dir() {
            return Err(format!("Directory not found: {}", root.display()).into());
        }
        sources.push(root);
    }

    if sources.is_empty() {
        return Err("no valid --source-root values were provided".into());
    }

    Ok(SourceRoots { sources })
}

pub fn build_default_sources(roots: &SourceRoots) -> Vec<Box<dyn DataSource + 'static>> {
    let mut sources = roots
        .sources
        .iter()
        .enumerate()
        .map(|(idx, root)| {
            Box::new(ExampleFileSource::from_root(
                &format!("source_{}", idx + 1),
                root,
            )) as Box<dyn DataSource + 'static>
        })
        .collect::<Vec<_>>();

    if let Some(source) = maybe_huggingface_source() {
        sources.push(source);
    }

    sources
}

struct ExampleFileSource {
    id: SourceId,
    inner: InMemorySource,
    reported_records: u128,
    triplet_recipes: Vec<TripletRecipe>,
}

impl ExampleFileSource {
    fn from_root(id: &str, root: &Path) -> Self {
        let records = load_records(id, root);
        let reported_records = records.len() as u128;
        let inner = InMemorySource::new(id, records);
        let triplet_recipes = FileCorpusIndex::default_title_summary_triplet_recipes();
        Self {
            id: id.into(),
            inner,
            reported_records,
            triplet_recipes,
        }
    }
}

impl DataSource for ExampleFileSource {
    fn id(&self) -> &str {
        &self.id
    }

    fn refresh(
        &self,
        config: &triplets::SamplerConfig,
        cursor: Option<&SourceCursor>,
        limit: Option<usize>,
    ) -> Result<SourceSnapshot, triplets::SamplerError> {
        self.inner.refresh(config, cursor, limit)
    }

    fn reported_record_count(
        &self,
        _config: &triplets::SamplerConfig,
    ) -> Result<u128, triplets::SamplerError> {
        Ok(self.reported_records)
    }

    fn default_triplet_recipes(&self) -> Vec<TripletRecipe> {
        self.triplet_recipes.clone()
    }
}

#[cfg(feature = "huggingface")]
fn maybe_huggingface_source() -> Option<Box<dyn DataSource + 'static>> {
    let source_id = "hf_rows".to_string();
    let dataset = "HuggingFaceFW/fineweb".to_string();
    let config_name = "default".to_string();
    let split_name = "train".to_string();
    let snapshot_dir = PathBuf::from(".hf-snapshots")
        .join(dataset.replace('/', "__"))
        .join(&config_name)
        .join(&split_name);

    let mut hf =
        HuggingFaceRowsConfig::new(source_id, dataset, config_name, split_name, snapshot_dir);
    hf.anchor_column = Some("text".to_string());
    hf.positive_column = Some("text".to_string());
    hf.context_columns = Vec::new();

    match HuggingFaceRowSource::new(hf) {
        Ok(source) => Some(Box::new(source)),
        Err(err) => {
            eprintln!(
                "Skipping Hugging Face source initialization for multi_source_demo: {}",
                err
            );
            None
        }
    }
}

#[cfg(not(feature = "huggingface"))]
fn maybe_huggingface_source() -> Option<Box<dyn DataSource + 'static>> {
    None
}

fn load_records(source_id: &str, root: &Path) -> Vec<DataRecord> {
    let mut records = Vec::new();
    for entry in WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
    {
        let path = entry.path();
        let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
            continue;
        };
        if !ext.eq_ignore_ascii_case("txt") {
            continue;
        }
        if let Some(record) = build_record(source_id, root, path) {
            records.push(record);
        }
    }
    records.sort_by(|a, b| a.id.cmp(&b.id));
    records
}

fn build_record(source_id: &str, root: &Path, path: &Path) -> Option<DataRecord> {
    let rel = path.strip_prefix(root).ok()?;
    let raw = fs::read_to_string(path).ok()?;
    let normalized = normalize_inline_whitespace(raw);
    if normalized.is_empty() {
        return None;
    }

    let title = normalized
        .split('.')
        .next()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .unwrap_or("Untitled sample");

    let date_text = extract_date_text(rel).unwrap_or_else(|| "2025-01-01".to_string());
    let dt = parse_date_utc(&date_text).unwrap_or_else(Utc::now);
    let record_id = format!("{}::{}", source_id, rel.to_string_lossy());

    Some(DataRecord {
        id: record_id,
        source: source_id.into(),
        created_at: dt,
        updated_at: dt,
        quality: QualityScore { trust: 1.0 },
        taxonomy: vec![source_id.into(), META_FIELD_DATE.encode(&date_text)],
        sections: vec![
            make_section(SectionRole::Anchor, Some("title"), title),
            make_section(SectionRole::Context, Some("body"), &normalized),
        ],
        meta_prefix: None,
    })
}

fn parse_date_utc(raw: &str) -> Option<DateTime<Utc>> {
    let date = NaiveDate::parse_from_str(raw, "%Y-%m-%d").ok()?;
    let naive = date.and_hms_opt(0, 0, 0)?;
    Some(Utc.from_utc_datetime(&naive))
}

fn extract_date_text(rel: &Path) -> Option<String> {
    for component in rel.components() {
        let text = component.as_os_str().to_string_lossy();
        if NaiveDate::parse_from_str(&text, "%Y-%m-%d").is_ok() {
            return Some(text.to_string());
        }
    }
    None
}

fn resolve_root(
    override_arg: Option<String>,
    env_keys: &[&str],
    default_candidates: &[&str],
    not_found_message: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    if let Some(path) = override_arg {
        let root = PathBuf::from(path);
        if root.is_dir() {
            return Ok(root);
        }
        return Err(format!("Directory not found: {}", root.display()).into());
    }

    if let Some(env_path) = env_path(env_keys) {
        return Ok(env_path);
    }

    if let Some(path) = first_existing_dir(default_candidates) {
        return Ok(path);
    }

    Err(not_found_message.into())
}

fn env_path(keys: &[&str]) -> Option<PathBuf> {
    for key in keys {
        if let Ok(value) = env::var(key) {
            let path = PathBuf::from(value);
            if path.is_dir() {
                return Some(path);
            }
        }
    }
    None
}

fn first_existing_dir(candidates: &[&str]) -> Option<PathBuf> {
    candidates
        .iter()
        .map(PathBuf::from)
        .find(|path| path.is_dir())
}
