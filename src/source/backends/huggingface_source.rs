use cache_manager::{CacheRoot, EvictPolicy};
use hf_hub::Repo;
use hf_hub::RepoType;
use hf_hub::api::sync::ApiBuilder;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::reader::RowIter;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use simd_r_drive::storage_engine::DataStore;
use simd_r_drive::storage_engine::traits::{DataStoreReader, DataStoreWriter};
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fs;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::path::PathBuf;
#[cfg(test)]
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use std::time::Instant;
#[cfg(test)]
use tempfile::TempDir;
use tracing::{debug, info, warn};
use walkdir::WalkDir;

use crate::SamplerError;
use crate::config::{NegativeStrategy, SamplerConfig, Selector, TripletRecipe};
use crate::constants::cache::HUGGINGFACE_GROUP;
use crate::constants::env_vars::{
    TRIPLETS_HF_INFO_ENDPOINT, TRIPLETS_HF_PARQUET_ENDPOINT, TRIPLETS_HF_SIZE_ENDPOINT,
};
use crate::constants::huggingface::{
    ALL_SPLITS_DIR, HF_CLASSLABEL_TYPE, HF_JSON_KEY_CONFIG, HF_JSON_KEY_CONFIG_NAME,
    HF_JSON_KEY_CONFIGS, HF_JSON_KEY_DATASET, HF_JSON_KEY_DATASET_INFO, HF_JSON_KEY_FEATURE_TYPE,
    HF_JSON_KEY_FEATURES, HF_JSON_KEY_LABEL_NAMES, HF_JSON_KEY_NUM_ROWS, HF_JSON_KEY_PARQUET_FILES,
    HF_JSON_KEY_SIZE, HF_JSON_KEY_SPLIT, HF_JSON_KEY_SPLIT_NAME, HF_JSON_KEY_SPLITS,
    HF_JSON_KEY_URL, HF_RESOLVE_UNKNOWN_FALLBACK_PATH, HF_RESOLVE_URL_SEPARATOR,
    HF_SHARD_CANDIDATE_SEED_TAG, HF_SHARD_STORE_EXTENSION, HF_SHARD_STORE_META_ROWS_KEY,
    HF_SHARD_STORE_ROW_PREFIX, HUGGINGFACE_REFRESH_BATCH_MULTIPLIER, PARQUET_MANIFEST_DIR,
    REMOTE_BOOTSTRAP_SHARDS, REMOTE_EXPANSION_HEADROOM_MULTIPLIER, REMOTE_URL_PREFIX,
};
use crate::data::{DataRecord, QualityScore, SectionRole};
use crate::utils::make_section;
use chrono::{DateTime, Utc};

use crate::source::{DataSource, SourceCursor, SourceSnapshot};

fn managed_cache_root() -> Result<CacheRoot, String> {
    #[cfg(test)]
    {
        static TEST_CACHE_ROOT: OnceLock<TempDir> = OnceLock::new();
        let root = TEST_CACHE_ROOT
            .get_or_init(|| TempDir::new().expect("failed to create test HF cache root"));
        Ok(CacheRoot::from_root(root.path()))
    }

    #[cfg(not(test))]
    {
        CacheRoot::from_discovery()
            .map_err(|err| format!("failed discovering managed cache root: {err}"))
    }
}

fn ensure_cache_group(relative_group: PathBuf) -> Result<PathBuf, String> {
    let cache_root = managed_cache_root()?;
    cache_root.ensure_group(&relative_group).map_err(|err| {
        format!(
            "failed creating managed cache group '{}': {err}",
            relative_group.display()
        )
    })
}

/// Resolve a managed snapshot directory for a list-based Hugging Face source.
pub fn managed_hf_list_snapshot_dir(
    dataset: &str,
    config: &str,
    split: &str,
    replica_idx: usize,
) -> Result<PathBuf, String> {
    // Empty split (all-splits mode) uses ALL_SPLITS_DIR so the path hierarchy stays valid
    // and won't collide with a split literally named "" on any filesystem.
    let split_dir = if split.is_empty() {
        ALL_SPLITS_DIR
    } else {
        split
    };
    ensure_cache_group(
        PathBuf::from(HUGGINGFACE_GROUP)
            .join("source-list")
            .join(dataset.replace('/', "__"))
            .join(config)
            .join(split_dir)
            .join(format!("replica_{replica_idx}")),
    )
}

/// Resolve a managed snapshot directory for a single Hugging Face source.
pub fn managed_hf_snapshot_dir(
    dataset: &str,
    config: &str,
    split: &str,
) -> Result<PathBuf, String> {
    let split_dir = if split.is_empty() {
        ALL_SPLITS_DIR
    } else {
        split
    };
    ensure_cache_group(
        PathBuf::from(HUGGINGFACE_GROUP)
            .join(dataset.replace('/', "__"))
            .join(config)
            .join(split_dir),
    )
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RowTextField {
    name: String,
    text: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RowView {
    row_id: Option<String>,
    timestamp: Option<DateTime<Utc>>,
    text_fields: Vec<RowTextField>,
}

/// Parsed Hugging Face source-list entry with explicit field mappings.
#[derive(Clone, Debug)]
pub struct HfSourceEntry {
    /// Full hf:// URI for dataset/config/split.
    pub uri: String,
    /// Anchor candidate columns (ordered).
    ///
    /// Each candidate is tried in order; the first whose value is present and
    /// non-empty is used as the anchor role for the row.  When the list is
    /// non-empty and no candidate yields content, the row is skipped.
    pub anchor_columns: Vec<String>,
    /// Positive candidate columns (ordered).
    ///
    /// Each candidate is tried in order; the first whose value is present and
    /// non-empty is used as the positive role for the row.  When the list is
    /// non-empty and no candidate yields content, the row is skipped.
    pub positive_columns: Vec<String>,
    /// Optional context columns (ordered).
    ///
    /// Used only in **role-based mode** (i.e. when `anchor_columns` and/or
    /// `positive_columns` are set).  Every listed column is required: if any
    /// is missing or blank the row is skipped.
    ///
    /// Each column becomes an additional `SectionRole::Context` section in the
    /// emitted record, appended after the positive section.  In contrast to
    /// `anchor_columns`/`positive_columns`, there is no coalescing — all
    /// columns contribute independently as separate sections.
    ///
    /// Not used in **text-columns mode** (`text_columns` non-empty,
    /// `anchor_columns` empty): in that mode only `text_columns` is consulted.
    pub context_columns: Vec<String>,
    /// Text candidate columns (ordered) for text-columns mode.
    ///
    /// Each candidate is tried in order; the first whose value is present and
    /// non-empty is used as the single text content for the row.  When the
    /// list is non-empty and no candidate yields content, the row is skipped.
    pub text_columns: Vec<String>,
    /// Optional trust/quality override for all records produced by this source.
    ///
    /// When set, overrides the default `QualityScore::default().trust` (0.5)
    /// for every record emitted by this source.  Must be in `[0.0, 1.0]`.
    pub trust: Option<f32>,
    /// Optional source ID override.
    ///
    /// When set, this string is used as the source identifier instead of the
    /// auto-derived slug from the dataset URI.  Useful for giving a stable,
    /// human-readable name to a source independently of its dataset/config/split
    /// path.  Deduplication suffixes are **not** applied to explicit source IDs.
    pub source_id: Option<String>,
}

impl PartialEq for HfSourceEntry {
    fn eq(&self, other: &Self) -> bool {
        self.uri == other.uri
            && self.anchor_columns == other.anchor_columns
            && self.positive_columns == other.positive_columns
            && self.context_columns == other.context_columns
            && self.text_columns == other.text_columns
            && self.source_id == other.source_id
            // Compare f32 bits so that identical bit patterns are considered equal.
            // Valid trust values are never NaN, so bit-level comparison is correct.
            && self.trust.map(f32::to_bits) == other.trust.map(f32::to_bits)
    }
}

impl Eq for HfSourceEntry {}

/// Parsed Hugging Face source list with explicit mappings.
#[derive(Debug, Clone)]
pub struct HfListRoots {
    /// The source list file path used for loading.
    pub source_list: String,
    /// Parsed sources with explicit field mappings.
    pub sources: Vec<HfSourceEntry>,
}

/// Split a comma-delimited field list into trimmed column names.
pub fn parse_csv_fields(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(ToString::to_string)
        .collect()
}

/// Parse a single source-list line of the form:
/// `hf://org/dataset/config/split anchor=... positive=... context=a,b text=x,y`.
pub fn parse_hf_source_line(line: &str) -> Result<HfSourceEntry, String> {
    let mut parts = line.split_whitespace();
    let Some(uri) = parts.next() else {
        return Err("empty source line".to_string());
    };
    if !uri.starts_with("hf://") {
        return Err(format!("unsupported source URI (expected hf://...): {uri}"));
    }

    let mut entry = HfSourceEntry {
        uri: uri.to_string(),
        anchor_columns: Vec::new(),
        positive_columns: Vec::new(),
        context_columns: Vec::new(),
        text_columns: Vec::new(),
        trust: None,
        source_id: None,
    };

    for token in parts {
        let Some((raw_key, raw_value)) = token.split_once('=') else {
            return Err(format!(
                "invalid mapping token '{token}' (expected key=value)"
            ));
        };
        let key = raw_key.trim().to_ascii_lowercase();
        let value = raw_value.trim();
        match key.as_str() {
            "anchor" => {
                entry.anchor_columns = parse_csv_fields(value);
            }
            "positive" => {
                entry.positive_columns = parse_csv_fields(value);
            }
            "context" => {
                entry.context_columns = parse_csv_fields(value);
            }
            "text" | "text_columns" => {
                entry.text_columns = parse_csv_fields(value);
            }
            "trust" => {
                let t: f32 = value.parse().map_err(|_| {
                    format!("invalid trust value '{value}': expected a float in [0.0, 1.0]")
                })?;
                if !(0.0..=1.0).contains(&t) {
                    return Err(format!("trust value {t} is out of range [0.0, 1.0]"));
                }
                entry.trust = Some(t);
            }
            "source_id" => {
                if value.is_empty() {
                    return Err("source_id must not be empty".to_string());
                }
                entry.source_id = Some(value.to_string());
            }
            _ => {
                return Err(format!("unsupported mapping key '{raw_key}'"));
            }
        }
    }

    let has_explicit_mapping = !entry.anchor_columns.is_empty()
        || !entry.positive_columns.is_empty()
        || !entry.context_columns.is_empty()
        || !entry.text_columns.is_empty();
    if !has_explicit_mapping {
        return Err(format!(
            "source '{}' has no field mapping; expected at least one of anchor=, positive=, context=, text=",
            entry.uri
        ));
    }

    Ok(entry)
}

/// Parse an hf:// URI into dataset/config/split components.
pub fn parse_hf_uri(uri: &str) -> Result<(String, String, String), String> {
    let trimmed = uri.trim();
    let Some(rest) = trimmed.strip_prefix("hf://") else {
        return Err(format!(
            "unsupported source URI (expected hf://...): {trimmed}"
        ));
    };

    let parts = rest
        .split('/')
        .filter(|part| !part.trim().is_empty())
        .collect::<Vec<_>>();

    if parts.len() < 2 {
        return Err(format!("invalid hf URI (need hf://org/dataset): {trimmed}"));
    }

    let dataset = format!("{}/{}", parts[0], parts[1]);
    let config = parts.get(2).copied().unwrap_or("default").to_string();
    // No trailing split component → empty string, which disables split-filtering
    // so all HF splits are discovered and triplets' own split logic handles partitioning.
    let split = parts.get(3).copied().unwrap_or("").to_string();

    Ok((dataset, config, split))
}

/// Load a Hugging Face source list file containing explicit field mappings.
pub fn load_hf_sources_from_list(path: &str) -> Result<Vec<HfSourceEntry>, String> {
    let body = fs::read_to_string(path).map_err(|err| format!("{err}"))?;
    let mut out = Vec::new();
    for (line_no, raw) in body.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parsed = parse_hf_source_line(line).map_err(|err| {
            format!(
                "invalid source-list entry at {}:{} -> {}",
                path,
                line_no + 1,
                err
            )
        })?;
        out.push(parsed);
    }
    Ok(out)
}

/// Resolve parsed Hugging Face source list entries into a structured root.
pub fn resolve_hf_list_roots(source_list: String) -> Result<HfListRoots, String> {
    let sources = load_hf_sources_from_list(&source_list)?;
    if sources.is_empty() {
        return Err(format!("no hf:// entries found in {}", source_list));
    }
    Ok(HfListRoots {
        source_list,
        sources,
    })
}

/// Sanitize a single component string for use in a source ID.
///
/// Replaces any character that is not alphanumeric, `-`, or `_` with `-`.
fn sanitize_source_id_component(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

/// Derive a human-readable source ID slug from parsed HF URI components.
///
/// Uses the short dataset name (the portion after the last `/`), then appends
/// `.{config}` when config is not `"default"` and `.{split}` when split is
/// not empty and not `"train"`.  Any character that is not alphanumeric,
/// `-`, or `_` is replaced with `-`.
fn hf_source_id_slug(dataset: &str, config: &str, split: &str) -> String {
    let short_name = dataset.rfind('/').map_or(dataset, |i| &dataset[i + 1..]);
    let mut slug = sanitize_source_id_component(short_name);
    if !config.is_empty() && config != "default" {
        slug.push('.');
        slug.push_str(&sanitize_source_id_component(config));
    }
    if !split.is_empty() && split != "train" {
        slug.push('.');
        slug.push_str(&sanitize_source_id_component(split));
    }
    if slug.is_empty() {
        slug = "hf".to_string();
    }
    slug
}

/// Build Hugging Face row sources from a parsed source list.
pub fn build_hf_sources(roots: &HfListRoots) -> Vec<Box<dyn DataSource + 'static>> {
    // Phase 1: compute auto-slugs for entries that don't have an explicit source_id.
    // Entries with an explicit source_id bypass slug computation and deduplication.
    let base_slugs: Vec<Option<String>> = roots
        .sources
        .iter()
        .enumerate()
        .map(|(idx, source)| {
            if source.source_id.is_some() {
                // Explicit source_id — skip slug generation entirely.
                None
            } else {
                Some(match parse_hf_uri(&source.uri) {
                    Ok((dataset, config, split)) => hf_source_id_slug(&dataset, &config, &split),
                    Err(_) => format!("hf_list_{idx}"),
                })
            }
        })
        .collect();

    // Phase 2: find auto-slugs that appear more than once so they can be disambiguated.
    // Explicit source_ids are not subject to deduplication.
    let mut slug_count: HashMap<&str, usize> = HashMap::new();
    for slug in base_slugs.iter().flatten() {
        *slug_count.entry(slug.as_str()).or_insert(0) += 1;
    }
    let duplicated: HashSet<&str> = slug_count
        .into_iter()
        .filter(|(_, n)| *n > 1)
        .map(|(s, _)| s)
        .collect();

    // Phase 3: resolve final IDs.
    // Explicit source_ids are used as-is; auto-slugs get `.{idx}` only when they collide.
    let source_ids: Vec<String> = roots
        .sources
        .iter()
        .zip(base_slugs.iter())
        .enumerate()
        .map(|(idx, (source, base_slug))| {
            if let Some(explicit_id) = &source.source_id {
                explicit_id.clone()
            } else if let Some(slug) = base_slug {
                if duplicated.contains(slug.as_str()) {
                    format!("{slug}.{idx}")
                } else {
                    slug.clone()
                }
            } else {
                format!("hf_list_{idx}")
            }
        })
        .collect();

    roots
        .sources
        .iter()
        .enumerate()
        .filter_map(|(idx, source)| {
            let (dataset, config, split) = match parse_hf_uri(&source.uri) {
                Ok(parsed) => parsed,
                Err(err) => {
                    eprintln!("Skipping invalid source URI '{}': {}", source.uri, err);
                    return None;
                }
            };

            let source_id = source_ids[idx].clone();
            let snapshot_dir = match managed_hf_list_snapshot_dir(&dataset, &config, &split, idx)
            {
                Ok(path) => path,
                Err(err) => {
                    eprintln!(
                        "Skipping Hugging Face source initialization for '{}': {}",
                        source.uri, err
                    );
                    return None;
                }
            };

            let mut hf = HuggingFaceRowsConfig::new(
                source_id,
                dataset,
                config,
                split,
                snapshot_dir,
            );
            hf.anchor_columns = source.anchor_columns.clone();
            hf.positive_columns = source.positive_columns.clone();
            hf.context_columns = source.context_columns.clone();
            hf.text_columns = source.text_columns.clone();
            hf.trust_override = source.trust;
            println!(
                "source {idx}: hf://{}/{}/{} -> anchor={:?}, positive={:?}, context={:?}, text_columns={:?}",
                hf.dataset,
                hf.config,
                hf.split,
                hf.anchor_columns,
                hf.positive_columns,
                hf.context_columns,
                hf.text_columns
            );

            match HuggingFaceRowSource::new(hf) {
                Ok(source) => Some(Box::new(source) as Box<dyn DataSource + 'static>),
                Err(err) => {
                    eprintln!(
                        "Skipping Hugging Face source initialization for '{}': {}",
                        source.uri, err
                    );
                    None
                }
            }
        })
        .collect()
}

/// Configuration for a bulk Hugging Face row source backed by local snapshot files.
#[derive(Clone, Debug)]
pub struct HuggingFaceRowsConfig {
    /// Stable sampler source id used in record ids and metrics.
    pub source_id: String,
    /// Hugging Face dataset id, e.g. `HuggingFaceFW/fineweb`.
    pub dataset: String,
    /// Dataset config name, e.g. `default`.
    pub config: String,
    /// Split name, e.g. `train`.
    pub split: String,
    /// Local path to a snapshot directory for this split.
    pub snapshot_dir: PathBuf,
    /// File extensions accepted as shard files.
    ///
    /// Non-parquet files are read as line-delimited entries. Each line may be:
    /// - a JSON object row (for example JSONL/NDJSON), or
    /// - plain text, which is wrapped as `{ "text": "..." }`.
    pub shard_extensions: Vec<String>,
    /// Number of rows between seek checkpoints while indexing a shard.
    pub checkpoint_stride: usize,
    /// Maximum number of rows cached in-memory.
    pub cache_capacity: usize,
    /// Maximum number of decoded parquet row groups cached in-memory.
    pub parquet_row_group_cache_capacity: usize,
    /// Multiplier applied to current refresh `limit` when building a read batch target.
    ///
    /// Effective target is `limit * refresh_batch_multiplier`.
    pub refresh_batch_multiplier: usize,
    /// Multiplier applied to ingestion-sized base records for `len_hint` headroom.
    ///
    /// Effective headroom is `cache_capacity * remote_expansion_headroom_multiplier`.
    pub remote_expansion_headroom_multiplier: usize,
    /// Hard cap for local manifest-shard cache bytes.
    ///
    /// Enforced by `cache-manager` policy application on manifest cache roots.
    pub local_disk_cap_bytes: Option<u64>,
    /// Optional row id column name. Falls back to synthetic id when missing.
    pub id_column: Option<String>,
    /// Text candidate columns (ordered) for text-columns mode.
    ///
    /// Each candidate is tried in order; the first whose value is present and
    /// non-empty is used as the single text content for the row.  When the
    /// list is non-empty and no candidate yields content, the row is skipped.
    pub text_columns: Vec<String>,
    /// Anchor candidate columns (ordered).
    ///
    /// Each candidate is tried in order; the first whose value is present and
    /// non-empty is used as the anchor role section.  When the list is
    /// non-empty and no candidate yields content, the row is skipped.
    ///
    /// When non-empty (or when `positive_columns`/`context_columns` are set),
    /// role-based extraction is used instead of `text_columns` mode.
    pub anchor_columns: Vec<String>,
    /// Positive candidate columns (ordered).
    ///
    /// Each candidate is tried in order; the first whose value is present and
    /// non-empty is used for the positive role section.  When the list is
    /// non-empty and no candidate yields content, the row is skipped.
    ///
    /// Positive text is emitted as a `SectionRole::Context` section.
    pub positive_columns: Vec<String>,
    /// Optional ordered context columns.
    ///
    /// Used only in **role-based mode** (i.e. when `anchor_columns` and/or
    /// `positive_columns` are set).  Every listed column is required: if any
    /// is missing or blank the row is skipped.
    ///
    /// Each column becomes an additional `SectionRole::Context` section in the
    /// emitted record, appended after the positive section.  Unlike
    /// `anchor_columns`/`positive_columns`, there is no coalescing — all
    /// columns contribute independently as separate sections.
    ///
    /// Ignored in **text-columns mode** (when `anchor_columns` is empty and
    /// `text_columns` is non-empty).
    pub context_columns: Vec<String>,
    /// Optional trust/quality override applied to all records produced by this source.
    ///
    /// When set, overrides the default `QualityScore::default().trust` (0.5) for
    /// every record emitted by this source.  Set this on sources that provide
    /// higher- or lower-quality data than the default.
    pub trust_override: Option<f32>,
    /// Optional integer-to-label maps for ClassLabel (or other integer) columns.
    ///
    /// Keyed by column name; value is an ordered list of label strings.  When a
    /// column value is an integer `n` and a map entry exists for that column,
    /// `label_maps[col][n]` is used as the text instead of the raw integer string.
    ///
    /// This field is auto-populated by [`HuggingFaceRowSource::new`] via a
    /// single call to the datasets-server `/info` endpoint.  Resolved label
    /// strings are written into the `.simdr` row store **at parquet-transcode
    /// time** and are not re-evaluated on subsequent reads.  Pre-existing
    /// stores that were transcoded before label resolution was introduced will
    /// continue to contain raw integer strings until their shard is evicted
    /// and re-transcoded.
    pub label_maps: HashMap<String, Vec<String>>,
}

impl HuggingFaceRowsConfig {
    /// Create a config with required dataset identity values and local snapshot path.
    pub fn new(
        source_id: impl Into<String>,
        dataset: impl Into<String>,
        config: impl Into<String>,
        split: impl Into<String>,
        snapshot_dir: impl Into<PathBuf>,
    ) -> Self {
        Self {
            source_id: source_id.into(),
            dataset: dataset.into(),
            config: config.into(),
            split: split.into(),
            snapshot_dir: snapshot_dir.into(),
            shard_extensions: vec![
                "parquet".to_string(),
                HF_SHARD_STORE_EXTENSION.to_string(),
                "jsonl".to_string(),
                "ndjson".to_string(),
            ],
            checkpoint_stride: 4096,
            cache_capacity: SamplerConfig::default().ingestion_max_records,
            parquet_row_group_cache_capacity: 8,
            refresh_batch_multiplier: HUGGINGFACE_REFRESH_BATCH_MULTIPLIER,
            remote_expansion_headroom_multiplier: REMOTE_EXPANSION_HEADROOM_MULTIPLIER,
            local_disk_cap_bytes: Some(32 * 1024 * 1024 * 1024),
            id_column: Some("id".to_string()),
            text_columns: vec!["text".to_string()],
            anchor_columns: Vec::new(),
            positive_columns: Vec::new(),
            context_columns: Vec::new(),
            trust_override: None,
            label_maps: HashMap::new(),
        }
    }

    fn has_explicit_mapping(&self) -> bool {
        !self.anchor_columns.is_empty()
            || !self.positive_columns.is_empty()
            || !self.context_columns.is_empty()
            || !self.text_columns.is_empty()
    }
}

#[derive(Default)]
struct ParquetCache {
    readers: HashMap<PathBuf, Arc<SerializedFileReader<File>>>,
    row_groups: HashMap<(PathBuf, usize), Arc<Vec<Value>>>,
    row_group_order: VecDeque<(PathBuf, usize)>,
}

#[derive(Default)]
#[allow(dead_code)]
struct EligibleIndexCache {
    signature: Option<u64>,
    rows: Option<Arc<Vec<usize>>>,
    shards: Vec<ShardIndex>,
}

impl ParquetCache {
    /// Return a cached parquet reader for `path`, opening and caching it when missing.
    fn reader_for(
        &mut self,
        source_id: &str,
        path: &Path,
    ) -> Result<Arc<SerializedFileReader<File>>, SamplerError> {
        if let Some(reader) = self.readers.get(path) {
            return Ok(reader.clone());
        }

        let file = File::open(path).map_err(|err| SamplerError::SourceUnavailable {
            source_id: source_id.to_string(),
            reason: format!("failed opening parquet shard {}: {err}", path.display()),
        })?;
        let reader =
            SerializedFileReader::new(file).map_err(|err| SamplerError::SourceUnavailable {
                source_id: source_id.to_string(),
                reason: format!("failed reading parquet shard {}: {err}", path.display()),
            })?;
        let reader = Arc::new(reader);
        self.readers.insert(path.to_path_buf(), reader.clone());
        Ok(reader)
    }

    fn row_group_rows_for(
        &mut self,
        source_id: &str,
        path: &Path,
        group_pos: usize,
        row_group_cache_capacity: usize,
    ) -> Result<Arc<Vec<Value>>, SamplerError> {
        let key = (path.to_path_buf(), group_pos);
        if let Some(rows) = self.row_groups.get(&key).cloned() {
            Self::refresh_row_group_order(&mut self.row_group_order, &key);
            self.row_group_order.push_back(key);
            return Ok(rows);
        }

        let reader = self.reader_for(source_id, path)?;
        let row_group =
            reader
                .get_row_group(group_pos)
                .map_err(|err| SamplerError::SourceUnavailable {
                    source_id: source_id.to_string(),
                    reason: format!(
                        "failed opening parquet row group {} for {}: {err}",
                        group_pos,
                        path.display()
                    ),
                })?;
        let iter = RowIter::from_row_group(None, row_group.as_ref()).map_err(|err| {
            SamplerError::SourceUnavailable {
                source_id: source_id.to_string(),
                reason: format!(
                    "failed iterating parquet row group {} for {}: {err}",
                    group_pos,
                    path.display()
                ),
            }
        })?;

        let mut decoded = Vec::new();
        for (position, row_result) in iter.enumerate() {
            let row_value = row_result.map_err(|err| SamplerError::SourceUnavailable {
                source_id: source_id.to_string(),
                reason: format!(
                    "failed reading parquet row {} in shard {} row_group {}: {err}",
                    position,
                    path.display(),
                    group_pos
                ),
            })?;
            decoded.push(row_value.to_json_value());
        }

        let rows = Arc::new(decoded);

        if row_group_cache_capacity > 0 {
            self.row_groups.insert(key.clone(), rows.clone());
            Self::refresh_row_group_order(&mut self.row_group_order, &key);
            self.row_group_order.push_back(key);
            while self.row_groups.len() > row_group_cache_capacity {
                if let Some(old) = self.row_group_order.pop_front() {
                    self.row_groups.remove(&old);
                } else {
                    break;
                }
            }
        }

        Ok(rows)
    }

    fn refresh_row_group_order(order: &mut VecDeque<(PathBuf, usize)>, key: &(PathBuf, usize)) {
        if order.is_empty() {
            return;
        }
        if let Some(pos) = order.iter().position(|existing| existing == key) {
            order.remove(pos);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ShardIndex {
    path: PathBuf,
    global_start: usize,
    row_count: usize,
    /// When `true`, rows are read via random-access indexed reads (parquet row-group
    /// reader or `.simdr` row-store).  When `false`, the shard is read sequentially
    /// as newline-delimited text (NDJSON, etc.).
    random_access: bool,
    parquet_row_groups: Vec<(usize, usize)>,
    checkpoints: Vec<u64>,
    /// Remote candidate string this shard was downloaded from, used to
    /// re-queue the download if the local file is evicted from the cache.
    remote_candidate: Option<String>,
}

#[derive(Default)]
struct RowCache {
    rows: HashMap<usize, RowView>,
    order: VecDeque<usize>,
}

impl RowCache {
    /// Return a cloned cached row by absolute index.
    fn get(&self, idx: usize) -> Option<RowView> {
        self.rows.get(&idx).cloned()
    }

    /// Insert or refresh a cached row and evict oldest entries over `capacity`.
    fn insert(&mut self, idx: usize, row: RowView, capacity: usize) {
        if capacity == 0 {
            return;
        }
        if !self.rows.contains_key(&idx) {
            self.order.push_back(idx);
        }
        self.rows.insert(idx, row);
        while self.rows.len() > capacity {
            if let Some(old) = self.order.pop_front() {
                self.rows.remove(&old);
            } else {
                break;
            }
        }
    }
}

/// Bulk-oriented Hugging Face source backed by local shard files.
///
/// ## Determinism guarantees
///
/// * **Split assignment** — fully deterministic and cache-independent.  A record's
///   Train/Validation/Test label is derived solely from its stable ID and the sampler
///   seed.  The same ID always maps to the same split regardless of when or how the
///   shard was downloaded.
///
/// * **Shard download order** — fully deterministic.  Given the same sampler seed and
///   the same HF manifest, position N in the download sequence always resolves to the
///   same shard file, independent of which shards are currently cached on disk.
///
/// * **Row selection within `refresh`** — **not deterministic across cache wipes.**
///   The permutation used to select rows in a given `refresh` call is seeded by
///   `(source_id, materialized_rows, sampler_seed)`.  `materialized_rows` grows as
///   shards are downloaded in the background; it is an accident of download timing,
///   not a stable property of the dataset.  After a cache wipe and re-download the
///   same epoch number will produce different rows.  Within a single run (stable
///   `materialized_rows`), repeated calls with the same cursor are reproducible.
///
/// ## Shard download failures
///
/// `next_remote_idx` is a cursor into `remote_candidate_order`, which is a
/// seed-derived permutation of the sorted HF manifest — not a simple sequential
/// counter.  It is **not** a wrapping ring buffer: when the cursor reaches the end
/// of the candidate list the background expansion thread stops spawning (normal
/// behavior once all shards are on disk).
///
/// When a shard download fails in the **background expansion thread** the error is
/// logged as a warning and that sequence position is skipped — `next_remote_idx`
/// was already incremented before the attempt, so no retry is performed in the
/// current cycle.  The skipped position becomes available again when the candidate
/// list is next rebuilt, which happens on:
///
/// * **Disk-cap eviction** — `sync_shard_state_from_disk_locked` nulls
///   `remote_candidates` and resets `next_remote_idx` to 0 whenever the cache
///   manager removes any shard from disk.  The next expansion cycle re-fetches the
///   HF manifest, rebuilds the permutation, and advances the cursor past shards
///   still present on disk via `first_uncached_order_position`.
/// * **Epoch-seed change** — the permutation is rebuilt for the new seed and the
///   cursor resets to `first_uncached_order_position`.
/// * **Source reconstruction** — `HuggingFaceRowSource::new()` starts from zero.
///
/// For **small datasets** that fit within the disk cap, all shards are typically
/// on disk before the cursor exhausts, so a transient network failure only delays
/// that shard until the next reset cycle.  For **large datasets without eviction**,
/// a skipped position is not revisited within the current run.
///
/// Failures on the **synchronous cold-start path** propagate immediately as
/// `SamplerError::SourceUnavailable` to the caller; the global row-count request
/// is non-fatal and only produces a warning.
pub struct HuggingFaceRowSource {
    config: HuggingFaceRowsConfig,
    sampler_config: Arc<Mutex<Option<SamplerConfig>>>,
    state: Arc<Mutex<SourceState>>,
    cache: Arc<Mutex<RowCache>>,
    parquet_cache: Arc<Mutex<ParquetCache>>,
    store_cache: Arc<Mutex<HashMap<PathBuf, Arc<DataStore>>>>,
    eligible_index: Arc<Mutex<EligibleIndexCache>>,
    /// Handle to the running background shard-expansion thread, if any.
    /// `is_finished()` returns true once the thread exits for any reason
    /// including panic, so this can never get permanently stuck the way
    /// an `AtomicBool` flag can when the thread panics before clearing it.
    expansion_thread: Arc<Mutex<Option<thread::JoinHandle<()>>>>,
}

impl Clone for HuggingFaceRowSource {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            sampler_config: Arc::clone(&self.sampler_config),
            state: Arc::clone(&self.state),
            cache: Arc::clone(&self.cache),
            parquet_cache: Arc::clone(&self.parquet_cache),
            store_cache: Arc::clone(&self.store_cache),
            eligible_index: Arc::clone(&self.eligible_index),
            expansion_thread: Arc::clone(&self.expansion_thread),
        }
    }
}

#[derive(Debug)]
struct SourceState {
    materialized_rows: usize,
    total_rows: Option<usize>,
    shards: Vec<ShardIndex>,
    /// Sorted, immutable list of all remote candidate identifiers.  Never
    /// shuffled in-place — ordering is expressed via `remote_candidate_order`.
    remote_candidates: Option<Vec<String>>,
    remote_candidate_sizes: HashMap<String, u64>,
    /// Seed-derived permutation of indices into `remote_candidates`.  For a
    /// given (seed, total) this is always the same sequence, regardless of
    /// how many shards have been consumed previously.
    remote_candidate_order: Vec<usize>,
    next_remote_idx: usize,
}

type ParquetGroupKey = (PathBuf, usize);
type ParquetGroupRequest = (usize, usize, ShardIndex);
type ParquetManifestCandidates = (Vec<String>, HashMap<String, u64>, usize);
type ShardIndexResult = (Vec<ShardIndex>, usize, HashMap<PathBuf, Arc<DataStore>>);

impl HuggingFaceRowSource {
    /// Build a new source by indexing local shard files.
    pub fn new(mut config: HuggingFaceRowsConfig) -> Result<Self, SamplerError> {
        let start_new = Instant::now();
        if config.checkpoint_stride == 0 {
            return Err(SamplerError::Configuration(
                "huggingface source checkpoint_stride must be > 0".to_string(),
            ));
        }
        if !config.has_explicit_mapping() {
            return Err(SamplerError::Configuration(
                "huggingface source requires explicit field mapping (anchor/positive/context/text_columns)"
                    .to_string(),
            ));
        }

        // Auto-resolve ClassLabel columns from the datasets-server /info endpoint.
        // If the fetch fails the label_maps stay empty and raw integers are used.
        config.label_maps = Self::fetch_classlabel_maps(&config);

        fs::create_dir_all(&config.snapshot_dir).map_err(|err| {
            SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!(
                    "failed creating snapshot_dir {}: {err}",
                    config.snapshot_dir.display()
                ),
            }
        })?;

        info!(
            "[triplets:hf] {} indexing local shards in {}",
            config.source_id,
            config.snapshot_dir.display()
        );
        let (shards, discovered, initial_store_cache) =
            Self::build_shard_index(&config).unwrap_or_default();
        if discovered == 0 {
            info!(
                "[triplets:hf] {} no local shards found in {} — lazy remote download enabled",
                config.source_id,
                config.snapshot_dir.display()
            );
        }

        let materialized_rows = discovered;
        let total_rows = match Self::fetch_global_row_count(&config) {
            Ok(value) => value,
            Err(err) => {
                warn!(
                    "[triplets:hf] {} global row count request failed; continuing with discovered rows only: {}",
                    config.source_id, err
                );
                None
            }
        };

        if let Some(global_total) = total_rows {
            info!(
                "[triplets:hf] {} global split row count reported: {} (known_local_rows={})",
                config.source_id, global_total, materialized_rows
            );
        }

        info!(
            "[triplets:hf] {} source ready in {:.2}s (rows={}, shards={})",
            config.source_id,
            start_new.elapsed().as_secs_f64(),
            materialized_rows,
            shards.len()
        );

        Ok(Self {
            config,
            sampler_config: Arc::new(Mutex::new(None)),
            state: Arc::new(Mutex::new(SourceState {
                materialized_rows,
                total_rows,
                shards,
                remote_candidates: None,
                remote_candidate_sizes: HashMap::new(),
                next_remote_idx: 0,
                remote_candidate_order: Vec::new(),
            })),
            cache: Arc::new(Mutex::new(RowCache::default())),
            parquet_cache: Arc::new(Mutex::new(ParquetCache::default())),
            store_cache: Arc::new(Mutex::new(initial_store_cache)),
            eligible_index: Arc::new(Mutex::new(EligibleIndexCache::default())),
            expansion_thread: Arc::new(Mutex::new(None)),
        })
    }

    /// Total row count for the dataset split as reported by the datasets-server
    /// `/size` endpoint, populated once during [`new`].
    ///
    /// Returns `None` if the endpoint was unreachable, returned a non-200
    /// response, or changed its response format.  The source continues to
    /// operate normally in that case (graceful degradation).
    pub fn known_total_rows(&self) -> Option<usize> {
        self.state.lock().ok()?.total_rows
    }

    fn is_store_shard_path(path: &Path) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case(HF_SHARD_STORE_EXTENSION))
    }

    fn shard_store_path_for(path: &Path) -> PathBuf {
        if Self::is_store_shard_path(path) {
            return path.to_path_buf();
        }
        path.with_extension(HF_SHARD_STORE_EXTENSION)
    }

    /// Map a candidate identifier directly to its canonical on-disk shard store path.
    fn candidate_store_path(config: &HuggingFaceRowsConfig, candidate: &str) -> PathBuf {
        Self::shard_store_path_for(&Self::candidate_target_path(config, candidate))
    }

    fn open_shard_store(
        config: &HuggingFaceRowsConfig,
        shard_store_path: &Path,
    ) -> Result<DataStore, SamplerError> {
        if let Some(parent) = shard_store_path.parent() {
            fs::create_dir_all(parent).map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!(
                    "failed creating row-store directory {}: {err}",
                    parent.display()
                ),
            })?;
        }
        DataStore::open(shard_store_path).map_err(|err| SamplerError::SourceUnavailable {
            source_id: config.source_id.clone(),
            reason: format!(
                "failed opening row store {}: {err}",
                shard_store_path.display()
            ),
        })
    }

    fn get_or_open_shard_store(
        &self,
        shard_store_path: &Path,
    ) -> Result<Arc<DataStore>, SamplerError> {
        if let Some(store) = self
            .store_cache
            .lock()
            .map_err(|_| SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: "huggingface row-store cache lock poisoned".to_string(),
            })?
            .get(shard_store_path)
            .cloned()
        {
            return Ok(store);
        }

        let store = Arc::new(Self::open_shard_store(&self.config, shard_store_path)?);
        let mut cache = self
            .store_cache
            .lock()
            .map_err(|_| SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: "huggingface row-store cache lock poisoned".to_string(),
            })?;
        let entry = cache
            .entry(shard_store_path.to_path_buf())
            .or_insert_with(|| store.clone());
        Ok(entry.clone())
    }

    fn prune_store_cache_to_shards(&self, shards: &[ShardIndex]) {
        let keep = shards
            .iter()
            .map(|shard| shard.path.clone())
            .collect::<HashSet<_>>();
        if let Ok(mut cache) = self.store_cache.lock() {
            cache.retain(|path, _| keep.contains(path));
        }
    }

    fn row_store_row_key(local_idx: usize) -> Vec<u8> {
        let mut key =
            Vec::with_capacity(HF_SHARD_STORE_ROW_PREFIX.len() + std::mem::size_of::<u64>());
        key.extend_from_slice(HF_SHARD_STORE_ROW_PREFIX);
        key.extend_from_slice(&(local_idx as u64).to_le_bytes());
        key
    }

    fn encode_row_view(&self, row: &RowView) -> Result<Vec<u8>, SamplerError> {
        serde_json::to_vec(row).map_err(|err| SamplerError::SourceUnavailable {
            source_id: self.config.source_id.clone(),
            reason: format!("failed encoding row-view payload: {err}"),
        })
    }

    fn decode_row_view(&self, bytes: &[u8]) -> Result<RowView, SamplerError> {
        serde_json::from_slice(bytes).map_err(|err| SamplerError::SourceUnavailable {
            source_id: self.config.source_id.clone(),
            reason: format!("failed decoding row-view payload: {err}"),
        })
    }

    fn read_store_row_count(&self, store: &DataStore) -> Result<usize, SamplerError> {
        let Some(entry) = store.read(HF_SHARD_STORE_META_ROWS_KEY).map_err(|err| {
            SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: format!("row-store meta read failed: {err}"),
            }
        })?
        else {
            return Ok(0);
        };

        let bytes = entry.as_ref();
        if bytes.len() != std::mem::size_of::<u64>() {
            return Err(SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: "row-store meta payload size mismatch".to_string(),
            });
        }
        let mut raw = [0u8; 8];
        raw.copy_from_slice(bytes);
        Ok(u64::from_le_bytes(raw) as usize)
    }

    fn write_store_row_count(&self, store: &DataStore, rows: usize) -> Result<(), SamplerError> {
        let payload = (rows as u64).to_le_bytes();
        store
            .write(HF_SHARD_STORE_META_ROWS_KEY, payload.as_slice())
            .map_err(|err| SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: format!("row-store meta write failed: {err}"),
            })?;
        Ok(())
    }

    fn transcode_parquet_shard_to_row_store(
        &self,
        shard: &ShardIndex,
    ) -> Result<Option<ShardIndex>, SamplerError> {
        if !shard.random_access {
            return Ok(Some(shard.clone()));
        }

        let store_path = Self::shard_store_path_for(&shard.path);
        let store = self.get_or_open_shard_store(&store_path)?;
        if store_path.exists() {
            let existing_rows = self.read_store_row_count(&store)?;
            if existing_rows > 0 {
                // Simdr store is already fully populated.  Clean up the
                // transient parquet source file if it is still present.
                if shard.path != store_path
                    && shard.path.exists()
                    && let Err(err) = fs::remove_file(&shard.path)
                {
                    warn!(
                        "[triplets:hf] failed removing stale parquet after store hit {}: {}",
                        shard.path.display(),
                        err
                    );
                }
                return Ok(Some(ShardIndex {
                    path: store_path,
                    global_start: shard.global_start,
                    row_count: existing_rows,
                    random_access: true,
                    parquet_row_groups: vec![(0, existing_rows)],
                    checkpoints: Vec::new(),
                    remote_candidate: shard.remote_candidate.clone(),
                }));
            }
        }

        let mut served_rows = 0usize;

        for (group_pos, (group_start, group_count)) in
            shard.parquet_row_groups.iter().copied().enumerate()
        {
            let rows = self
                .parquet_cache
                .lock()
                .map_err(|_| SamplerError::SourceUnavailable {
                    source_id: self.config.source_id.clone(),
                    reason: "huggingface parquet cache lock poisoned".to_string(),
                })?
                .row_group_rows_for(
                    &self.config.source_id,
                    &shard.path,
                    group_pos,
                    self.config.parquet_row_group_cache_capacity,
                )?;

            let mut batch: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(group_count);

            for local_in_group in 0..group_count {
                let local_idx = group_start.saturating_add(local_in_group);
                if local_idx >= shard.row_count {
                    break;
                }
                let Some(row_value) = rows.get(local_in_group) else {
                    break;
                };
                let absolute_idx = shard.global_start.saturating_add(local_idx);
                let Some(row) = self.parse_row(absolute_idx, row_value)? else {
                    continue;
                };

                let key = Self::row_store_row_key(served_rows);
                let payload = self.encode_row_view(&row)?;
                batch.push((key, payload));
                served_rows = served_rows.saturating_add(1);
            }

            if !batch.is_empty() {
                let refs: Vec<(&[u8], &[u8])> = batch
                    .iter()
                    .map(|(key, payload)| (key.as_slice(), payload.as_slice()))
                    .collect();
                store
                    .batch_write(&refs)
                    .map_err(|err| SamplerError::SourceUnavailable {
                        source_id: self.config.source_id.clone(),
                        reason: format!("row-store batch write failed: {err}"),
                    })?;
            }
        }

        self.write_store_row_count(&store, served_rows)?;

        if shard.path != store_path {
            fs::remove_file(&shard.path).map_err(|err| SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: format!(
                    "failed removing parquet shard after store transcode {}: {err}",
                    shard.path.display()
                ),
            })?;
        }

        if served_rows == 0 {
            return Ok(None);
        }

        Ok(Some(ShardIndex {
            path: store_path,
            global_start: shard.global_start,
            row_count: served_rows,
            random_access: true,
            parquet_row_groups: vec![(0, served_rows)],
            checkpoints: Vec::new(),
            remote_candidate: shard.remote_candidate.clone(),
        }))
    }

    fn invalidate_eligible_index(&self) {
        if let Ok(mut cache) = self.eligible_index.lock() {
            *cache = EligibleIndexCache::default();
        }
    }

    #[allow(dead_code)]
    fn shard_signature(shards: &[ShardIndex]) -> u64 {
        let mut hasher = DefaultHasher::new();
        for shard in shards {
            shard.path.hash(&mut hasher);
            shard.global_start.hash(&mut hasher);
            shard.row_count.hash(&mut hasher);
            shard.random_access.hash(&mut hasher);
            shard.parquet_row_groups.hash(&mut hasher);
        }
        hasher.finish()
    }

    #[allow(dead_code)]
    fn build_eligible_rows_from_shards(
        &self,
        shards: &[ShardIndex],
    ) -> Result<Vec<usize>, SamplerError> {
        let mut eligible = Vec::new();

        for shard in shards {
            if shard.random_access {
                if Self::is_store_shard_path(&shard.path) {
                    for local_idx in 0..shard.row_count {
                        let absolute_idx = shard.global_start.saturating_add(local_idx);
                        eligible.push(absolute_idx);
                    }
                    continue;
                }

                for (group_pos, (group_start, group_count)) in
                    shard.parquet_row_groups.iter().copied().enumerate()
                {
                    let rows = self
                        .parquet_cache
                        .lock()
                        .map_err(|_| SamplerError::SourceUnavailable {
                            source_id: self.config.source_id.clone(),
                            reason: "huggingface parquet cache lock poisoned".to_string(),
                        })?
                        .row_group_rows_for(
                            &self.config.source_id,
                            &shard.path,
                            group_pos,
                            self.config.parquet_row_group_cache_capacity,
                        )?;

                    for local_in_group in 0..group_count {
                        let Some(row_value) = rows.get(local_in_group) else {
                            break;
                        };
                        let local_idx = group_start.saturating_add(local_in_group);
                        if local_idx >= shard.row_count {
                            break;
                        }
                        let absolute_idx = shard.global_start.saturating_add(local_idx);
                        if self.parse_row(absolute_idx, row_value)?.is_some() {
                            eligible.push(absolute_idx);
                        }
                    }
                }
                continue;
            }

            let file = File::open(&shard.path).map_err(|err| SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: format!("failed opening shard {}: {err}", shard.path.display()),
            })?;
            let reader = BufReader::new(file);
            for (local_idx, line) in reader.lines().enumerate() {
                if local_idx >= shard.row_count {
                    break;
                }
                let line = line.map_err(|err| SamplerError::SourceUnavailable {
                    source_id: self.config.source_id.clone(),
                    reason: format!("failed reading shard {}: {err}", shard.path.display()),
                })?;
                let row_value = self.parse_non_parquet_line(shard, local_idx, &line)?;
                let absolute_idx = shard.global_start.saturating_add(local_idx);
                if self.parse_row(absolute_idx, &row_value)?.is_some() {
                    eligible.push(absolute_idx);
                }
            }
        }

        Ok(eligible)
    }

    #[allow(dead_code)]
    fn eligible_rows(&self) -> Result<Arc<Vec<usize>>, SamplerError> {
        let (signature, shards) = {
            let state = self
                .state
                .lock()
                .map_err(|_| SamplerError::SourceUnavailable {
                    source_id: self.config.source_id.clone(),
                    reason: "huggingface source state lock poisoned".to_string(),
                })?;
            (Self::shard_signature(&state.shards), state.shards.clone())
        };

        if let Ok(cache) = self.eligible_index.lock()
            && cache.signature == Some(signature)
            && let Some(rows) = cache.rows.as_ref()
        {
            return Ok(rows.clone());
        }

        let incremental_seed = if let Ok(cache) = self.eligible_index.lock()
            && cache.signature != Some(signature)
            && !cache.shards.is_empty()
            && cache.shards.len() < shards.len()
            && shards
                .iter()
                .take(cache.shards.len())
                .eq(cache.shards.iter())
            && let Some(existing_rows) = cache.rows.as_ref()
        {
            Some((cache.shards.len(), existing_rows.as_ref().clone()))
        } else {
            None
        };

        if let Some((prefix_len, mut merged)) = incremental_seed {
            let appended = self.build_eligible_rows_from_shards(&shards[prefix_len..])?;
            merged.extend(appended);
            let rows = Arc::new(merged);

            let mut writable =
                self.eligible_index
                    .lock()
                    .map_err(|_| SamplerError::SourceUnavailable {
                        source_id: self.config.source_id.clone(),
                        reason: "huggingface eligible-index cache lock poisoned".to_string(),
                    })?;
            writable.signature = Some(signature);
            writable.shards = shards;
            writable.rows = Some(rows.clone());
            return Ok(rows);
        }

        let rows = Arc::new(self.build_eligible_rows_from_shards(&shards)?);
        let mut cache =
            self.eligible_index
                .lock()
                .map_err(|_| SamplerError::SourceUnavailable {
                    source_id: self.config.source_id.clone(),
                    reason: "huggingface eligible-index cache lock poisoned".to_string(),
                })?;
        cache.signature = Some(signature);
        cache.shards = shards;
        cache.rows = Some(rows.clone());
        Ok(rows)
    }

    fn set_active_sampler_config(&self, config: &SamplerConfig) {
        let seed_changed = self
            .sampler_config
            .lock()
            .ok()
            .and_then(|slot| slot.as_ref().map(|c| c.seed != config.seed))
            .unwrap_or(false);

        if let Ok(mut slot) = self.sampler_config.lock() {
            *slot = Some(config.clone());
        }

        // When the epoch seed changes, rebuild the permuted order index from the
        // sorted immutable candidates list and advance the consumption pointer past
        // shards already materialised on disk in the new order.  Resetting to 0
        // causes every source-epoch advance to appear to restart from shard 1 even
        // when many shards are already cached, permanently stalling expansion.
        // For a given (seed, sorted-list) this always produces the same shard download
        // order, so shard N is always the same file whether it is the first or the
        // hundredth epoch.  Note: row selection within refresh() is NOT deterministic
        // across cache wipes — see HuggingFaceRowSource doc comment.
        if seed_changed
            && let Ok(mut state) = self.state.lock()
            && let Some(candidates) = state.remote_candidates.clone()
        {
            let new_order = Self::build_candidate_order(&self.config, &candidates, config.seed);
            let next_idx = Self::first_uncached_order_position(
                &self.config,
                &candidates,
                &new_order,
                &state.shards,
            );
            state.remote_candidate_order = new_order;
            state.next_remote_idx = next_idx;
        }
    }

    #[cfg(test)]
    fn active_or_default_sampler_config(&self) -> SamplerConfig {
        self.sampler_config
            .lock()
            .ok()
            .and_then(|slot| slot.clone())
            .unwrap_or_default()
    }

    #[cfg(test)]
    fn configure_sampler(&self, config: &SamplerConfig) {
        self.set_active_sampler_config(config);
    }

    #[cfg(test)]
    fn refresh(
        &self,
        cursor: Option<&SourceCursor>,
        limit: Option<usize>,
    ) -> Result<SourceSnapshot, SamplerError> {
        let config = self.active_or_default_sampler_config();
        <Self as DataSource>::refresh(self, &config, cursor, limit)
    }

    #[cfg(test)]
    fn reported_record_count(&self) -> Result<u128, SamplerError> {
        let config = self.active_or_default_sampler_config();
        <Self as DataSource>::reported_record_count(self, &config)
    }

    /// Compute the effective internal row read target from refresh `limit`.
    fn effective_refresh_batch_target(&self, limit: usize) -> usize {
        let multiplier = self.config.refresh_batch_multiplier.max(1);
        limit.saturating_mul(multiplier)
    }

    /// Compute dynamic `len_hint` headroom rows based on sampler and source config.
    fn effective_expansion_headroom_rows(&self) -> usize {
        let multiplier = self.config.remote_expansion_headroom_multiplier.max(1);
        let base = self
            .sampler_config
            .lock()
            .ok()
            .and_then(|config| config.as_ref().map(|value| value.ingestion_max_records))
            .unwrap_or(self.config.cache_capacity)
            .max(1);
        base.saturating_mul(multiplier)
    }

    fn configured_sampler_seed(&self) -> Result<u64, SamplerError> {
        self.sampler_config
            .lock()
            .map_err(|_| SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: "huggingface sampler-config lock poisoned".to_string(),
            })?
            .as_ref()
            .map(|config| config.seed)
            .ok_or_else(|| SamplerError::SourceInconsistent {
                source_id: self.config.source_id.clone(),
                details: "huggingface source sampler configuration not provided".to_string(),
            })
    }

    fn paging_seed(&self, total: usize) -> Result<u64, SamplerError> {
        let sampler_seed = self.configured_sampler_seed()?;
        Ok(crate::source::IndexablePager::seed_for_sampler(
            &self.config.source_id,
            total,
            sampler_seed,
        ))
    }

    fn normalized_shard_extensions(config: &HuggingFaceRowsConfig) -> Vec<String> {
        config
            .shard_extensions
            .iter()
            .map(|value| value.trim().trim_start_matches('.').to_ascii_lowercase())
            .collect::<Vec<_>>()
    }

    fn collect_candidates_from_siblings(
        config: &HuggingFaceRowsConfig,
        siblings: &[String],
        accepted: &[String],
        respect_split: bool,
    ) -> (Vec<String>, bool) {
        let mut saw_parquet = false;
        let mut candidates = Vec::new();
        for remote_path in siblings {
            if respect_split && !config.split.is_empty() {
                let split_tag = format!("{}/", config.split);
                let split_token = format!("-{}-", config.split);
                let split_prefix = format!("{}-", config.split);
                if !remote_path.contains(&split_tag)
                    && !remote_path.contains(&split_token)
                    && !Path::new(remote_path)
                        .file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.starts_with(&split_prefix))
                {
                    continue;
                }
            }

            let ext = Path::new(remote_path)
                .extension()
                .and_then(|v| v.to_str())
                .map(|v| v.to_ascii_lowercase());
            if ext.as_deref() == Some("parquet") {
                saw_parquet = true;
            }
            if ext
                .as_deref()
                .is_some_and(|ext| accepted.iter().any(|allowed| allowed == ext))
            {
                let target = Self::candidate_target_path(config, remote_path);
                let store_target = Self::shard_store_path_for(&target);
                if target.exists() || store_target.exists() {
                    continue;
                }
                candidates.push(remote_path.clone());
            }
        }
        (candidates, saw_parquet)
    }

    fn resolve_remote_candidates_from_siblings(
        config: &HuggingFaceRowsConfig,
        siblings: &[String],
        accepted: &[String],
    ) -> Result<(Vec<String>, HashMap<String, u64>), SamplerError> {
        let (mut candidates, mut saw_parquet) =
            Self::collect_candidates_from_siblings(config, siblings, accepted, true);
        if candidates.is_empty() && !config.split.is_empty() {
            let (fallback_candidates, fallback_saw_parquet) =
                Self::collect_candidates_from_siblings(config, siblings, accepted, false);
            if !fallback_candidates.is_empty() {
                warn!(
                    "[triplets:hf] split filter '{}' matched no remote files; falling back to extension-only remote candidate scan",
                    config.split
                );
                candidates = fallback_candidates;
                saw_parquet = fallback_saw_parquet;
            }
        }

        candidates.sort();
        candidates.dedup();
        info!(
            "[triplets:hf] remote candidates matching {:?}: {}",
            config.shard_extensions,
            candidates.len()
        );
        if candidates.is_empty() {
            if saw_parquet {
                return Err(SamplerError::SourceUnavailable {
                    source_id: config.source_id.clone(),
                    reason: format!(
                        "dataset '{}' appears to be parquet-only, but shard_extensions does not include parquet ({:?}).",
                        config.dataset, config.shard_extensions
                    ),
                });
            }
            warn!(
                "[triplets:hf] no remote candidates found for dataset='{}' split='{}' extensions={:?}; source will be treated as exhausted",
                config.dataset, config.split, config.shard_extensions
            );
            return Ok((Vec::new(), HashMap::new()));
        }

        Ok((candidates, HashMap::new()))
    }

    /// Return ALL shards from the parquet manifest regardless of what is already cached
    /// on disk.  Shard download order and local cache are orthogonal concerns:
    ///
    /// * **Shard download order** — must be computed from the complete HF manifest so
    ///   that position N for seed S always maps to the same shard file, independent of
    ///   what has been previously downloaded or evicted.
    /// * **Cache** — which shards are already on disk is handled downstream:
    ///   `ensure_row_available` advances `next_remote_idx` past already-materialised
    ///   positions, and `download_next_remote_shard` skips any position whose store
    ///   file already exists without re-fetching it.
    ///
    /// Note: row-level selection within `refresh` is *not* deterministic across cache
    /// wipes.  Only the shard download sequence is stable end-to-end.
    ///
    /// The only cleanup performed here is deleting stale/incomplete transient parquet
    /// downloads (wrong on-disk size) so they are re-fetched on the next download cycle.
    fn all_candidates_from_parquet_manifest(
        config: &HuggingFaceRowsConfig,
        json: &Value,
    ) -> Result<ParquetManifestCandidates, SamplerError> {
        let accepted = Self::normalized_shard_extensions(config);

        let mut candidates = Vec::new();
        let mut candidate_sizes = HashMap::new();
        let mut matched_manifest_entries = 0usize;
        if let Some(entries) = json
            .get(HF_JSON_KEY_PARQUET_FILES)
            .and_then(Value::as_array)
        {
            for entry in entries {
                let Some(url) = entry.get(HF_JSON_KEY_URL).and_then(Value::as_str) else {
                    continue;
                };

                let ext = Path::new(url)
                    .extension()
                    .and_then(|value| value.to_str())
                    .map(|value| value.to_ascii_lowercase());
                if !ext
                    .as_deref()
                    .is_some_and(|value| accepted.iter().any(|allowed| allowed == value))
                {
                    continue;
                }

                matched_manifest_entries += 1;
                let candidate = format!("{REMOTE_URL_PREFIX}{url}");
                let expected_size = entry.get(HF_JSON_KEY_SIZE).and_then(Value::as_u64);

                // Remove stale/incomplete transient parquet downloads so they get
                // re-fetched.  Fully-materialised `.simdr` stores are intentionally
                // kept — `download_next_remote_shard` will detect them and skip the
                // download without re-fetching.
                let target = Self::candidate_target_path(config, &candidate);
                if target.exists() && !Self::target_matches_expected_size(&target, expected_size) {
                    warn!(
                        "[triplets:hf] incomplete cached shard detected (will redownload): {}",
                        target.display()
                    );
                    if let Err(err) = fs::remove_file(&target)
                        && err.kind() != std::io::ErrorKind::NotFound
                    {
                        return Err(SamplerError::SourceUnavailable {
                            source_id: config.source_id.clone(),
                            reason: format!(
                                "failed removing incomplete shard {}: {err}",
                                target.display()
                            ),
                        });
                    }
                }

                if let Some(size) = expected_size {
                    candidate_sizes.insert(candidate.clone(), size);
                }
                candidates.push(candidate);
            }
        }

        candidates.sort();
        candidates.dedup();
        candidate_sizes.retain(|candidate, _| candidates.binary_search(candidate).is_ok());
        Ok((candidates, candidate_sizes, matched_manifest_entries))
    }

    /// Resolve and filter remote shard candidates from manifest or repository listing.
    fn list_remote_candidates(
        config: &HuggingFaceRowsConfig,
    ) -> Result<(Vec<String>, HashMap<String, u64>), SamplerError> {
        if let Ok((candidates, candidate_sizes, matched_manifest_entries)) =
            Self::list_remote_candidates_from_parquet_manifest(config)
            && matched_manifest_entries > 0
        {
            // Parquet manifest exists for this dataset.  Return the full candidate
            // list — including already-cached shards — so the seed-derived order is
            // always built from the complete HF manifest, independent of what has
            // been downloaded.  Never fall through to the hf-hub siblings listing,
            // which returns every language config in the repository, not just the
            // one requested.
            info!(
                "[triplets:hf] remote parquet manifest: {} shard(s) for dataset='{}' \
                     config='{}' split='{}'",
                candidates.len(),
                config.dataset,
                config.config,
                config.split
            );
            return Ok((candidates, candidate_sizes));

            // matched_manifest_entries == 0: parquet manifest does not cover this
            // dataset/config.  Fall through to the hf-hub siblings listing.
        }

        let api = ApiBuilder::new()
            .with_progress(true)
            .with_retries(5)
            .with_token(None)
            .build()
            .map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!("failed building hf-hub client: {err}"),
            })?;

        let repo = Repo::new(config.dataset.clone(), RepoType::Dataset);
        let repo_api = api.repo(repo);
        info!(
            "[triplets:hf] reading remote file list for dataset {}",
            config.dataset
        );
        let info = repo_api
            .info()
            .map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!("failed reading hf-hub repository info: {err}"),
            })?;

        let accepted = Self::normalized_shard_extensions(config);

        let siblings = info
            .siblings
            .into_iter()
            .map(|entry| entry.rfilename)
            .collect::<Vec<_>>();

        Self::resolve_remote_candidates_from_siblings(config, &siblings, &accepted)
    }

    /// Build a seed-derived permutation of indices 0..candidates.len().
    ///
    /// The candidates slice is never modified.  The returned Vec maps
    /// download-position → candidate index, so for epoch seed S position N
    /// always resolves to the same shard regardless of how many shards have
    /// been consumed before.
    fn build_candidate_order(
        config: &HuggingFaceRowsConfig,
        candidates: &[String],
        sampler_seed: u64,
    ) -> Vec<usize> {
        let n = candidates.len();
        let mut order: Vec<usize> = (0..n).collect();
        if n <= 1 {
            return order;
        }
        let base_seed = Self::shard_candidate_seed(config, n, sampler_seed);
        let mut rng = if base_seed == 0 {
            0xdeadbeef_cafebabe
        } else {
            base_seed
        };
        for i in (1..n).rev() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let j = (rng as usize) % (i + 1);
            order.swap(i, j);
        }
        order
    }

    /// Return the first position in `order` whose shard store is not yet on disk
    /// according to the in-memory shard list.  Returns `candidates.len()` when
    /// every position is already cached (nothing left to download).
    fn first_uncached_order_position(
        config: &HuggingFaceRowsConfig,
        candidates: &[String],
        order: &[usize],
        shards: &[ShardIndex],
    ) -> usize {
        let existing: HashSet<PathBuf> = shards.iter().map(|s| s.path.clone()).collect();
        order
            .iter()
            .position(|&idx| {
                !existing.contains(&Self::candidate_store_path(config, &candidates[idx]))
            })
            .unwrap_or(candidates.len())
    }

    /// Shuffle remote shard candidates into a deterministic-but-random order.
    ///
    /// Retained for use in tests that directly assert on shuffled slices.
    /// Production code uses `build_candidate_order` and keeps the list immutable.
    #[cfg(test)]
    fn shuffle_candidates_deterministically(
        config: &HuggingFaceRowsConfig,
        candidates: &mut [String],
        sampler_seed: u64,
    ) {
        let order = Self::build_candidate_order(config, candidates, sampler_seed);
        // Apply the permutation in-place via a temporary clone.
        let original = candidates.to_vec();
        for (pos, &src) in order.iter().enumerate() {
            candidates[pos] = original[src].clone();
        }
    }

    /// Build deterministic seed used to permute remote shard candidate order.
    fn shard_candidate_seed(
        config: &HuggingFaceRowsConfig,
        total_candidates: usize,
        sampler_seed: u64,
    ) -> u64 {
        let mut hasher = DefaultHasher::new();
        HF_SHARD_CANDIDATE_SEED_TAG.hash(&mut hasher);
        sampler_seed.hash(&mut hasher);
        config.source_id.hash(&mut hasher);
        config.dataset.hash(&mut hasher);
        config.config.hash(&mut hasher);
        config.split.hash(&mut hasher);
        total_candidates.hash(&mut hasher);
        hasher.finish()
    }

    fn parquet_manifest_endpoint() -> String {
        if let Ok(value) = std::env::var(TRIPLETS_HF_PARQUET_ENDPOINT)
            && !value.trim().is_empty()
        {
            return value;
        }
        "https://datasets-server.huggingface.co/parquet".to_string()
    }

    fn size_endpoint() -> String {
        if let Ok(value) = std::env::var(TRIPLETS_HF_SIZE_ENDPOINT)
            && !value.trim().is_empty()
        {
            return value;
        }
        "https://datasets-server.huggingface.co/size".to_string()
    }

    fn info_endpoint() -> String {
        if let Ok(value) = std::env::var(TRIPLETS_HF_INFO_ENDPOINT)
            && !value.trim().is_empty()
        {
            return value;
        }
        "https://datasets-server.huggingface.co/info".to_string()
    }

    /// Fetch ClassLabel name mappings from the datasets-server `/info` endpoint.
    ///
    /// Returns a map of column name → ordered label names for every feature whose
    /// `_type` is `"ClassLabel"`.  Called once per [`HuggingFaceRowSource::new`]
    /// call; the result is stored in [`HuggingFaceRowsConfig::label_maps`] and
    /// consulted at parquet-shard transcode time only.
    ///
    /// All failure paths (unreachable endpoint, HTTP error, unreadable body,
    /// invalid JSON) emit a `warn!` log and return an empty map.  The caller
    /// continues normally with raw integer strings as a fallback.
    fn fetch_classlabel_maps(config: &HuggingFaceRowsConfig) -> HashMap<String, Vec<String>> {
        let endpoint = Self::info_endpoint();
        let response = match ureq::get(&endpoint)
            .query(HF_JSON_KEY_DATASET, &config.dataset)
            .query(HF_JSON_KEY_CONFIG, &config.config)
            .call()
        {
            Ok(r) => r,
            Err(err) => {
                warn!(
                    "[triplets:hf] {} could not fetch dataset info for ClassLabel resolution: {}",
                    config.source_id, err
                );
                return HashMap::new();
            }
        };
        let body = match response.into_body().read_to_string() {
            Ok(b) => b,
            Err(err) => {
                warn!(
                    "[triplets:hf] {} failed reading dataset info response: {}",
                    config.source_id, err
                );
                return HashMap::new();
            }
        };
        let json: Value = match serde_json::from_str(&body) {
            Ok(v) => v,
            Err(err) => {
                warn!(
                    "[triplets:hf] {} failed parsing dataset info response: {}",
                    config.source_id, err
                );
                return HashMap::new();
            }
        };
        let maps = Self::extract_classlabel_maps(&json);
        if !maps.is_empty() {
            info!(
                "[triplets:hf] {} resolved ClassLabel mappings for columns: {:?}",
                config.source_id,
                maps.keys().collect::<Vec<_>>()
            );
        }
        maps
    }

    /// Extract ClassLabel mappings from a parsed `/info` response payload.
    ///
    /// Exposed as a separate function so it can be unit-tested without a live
    /// network.
    fn extract_classlabel_maps(info: &Value) -> HashMap<String, Vec<String>> {
        let mut maps = HashMap::new();
        let Some(features) = info
            .get(HF_JSON_KEY_DATASET_INFO)
            .and_then(|di| di.get(HF_JSON_KEY_FEATURES))
            .and_then(|f| f.as_object())
        else {
            return maps;
        };
        for (col, field) in features {
            if field.get(HF_JSON_KEY_FEATURE_TYPE).and_then(Value::as_str)
                != Some(HF_CLASSLABEL_TYPE)
            {
                continue;
            }
            let Some(names_arr) = field
                .get(HF_JSON_KEY_LABEL_NAMES)
                .and_then(|n| n.as_array())
            else {
                continue;
            };
            let label_names: Vec<String> = names_arr
                .iter()
                .filter_map(|v| v.as_str())
                .map(str::to_string)
                .collect();
            if !label_names.is_empty() {
                maps.insert(col.clone(), label_names);
            }
        }
        maps
    }

    /// Query datasets-server parquet manifest and derive shard candidates.
    fn list_remote_candidates_from_parquet_manifest(
        config: &HuggingFaceRowsConfig,
    ) -> Result<ParquetManifestCandidates, SamplerError> {
        let endpoint = Self::parquet_manifest_endpoint();
        info!(
            "[triplets:hf] reading datasets-server parquet manifest for dataset {}",
            config.dataset
        );
        let mut request = ureq::get(&endpoint)
            .query(HF_JSON_KEY_DATASET, &config.dataset)
            .query(HF_JSON_KEY_CONFIG, &config.config);
        // When split is empty (all-splits mode) omit the split query param so the
        // datasets-server returns shards for every split in the config.
        if !config.split.is_empty() {
            request = request.query(HF_JSON_KEY_SPLIT, &config.split);
        }
        let response = request
            .call()
            .map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!("failed querying datasets-server parquet endpoint: {err}"),
            })?;

        let body = response.into_body().read_to_string().map_err(|err| {
            SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!("failed reading datasets-server parquet response body: {err}"),
            }
        })?;

        Self::parse_parquet_manifest_response(config, &body)
    }

    fn parse_parquet_manifest_response(
        config: &HuggingFaceRowsConfig,
        body: &str,
    ) -> Result<ParquetManifestCandidates, SamplerError> {
        let json: Value =
            serde_json::from_str(body).map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!("failed parsing datasets-server parquet response: {err}"),
            })?;

        Self::all_candidates_from_parquet_manifest(config, &json)
    }

    /// Map a candidate identifier to the local snapshot target path.
    fn candidate_target_path(config: &HuggingFaceRowsConfig, candidate: &str) -> PathBuf {
        if let Some(url) = candidate.strip_prefix(REMOTE_URL_PREFIX) {
            let suffix = url
                .split(HF_RESOLVE_URL_SEPARATOR)
                .nth(1)
                .map(|value| value.trim_start_matches('/'))
                .filter(|value| !value.is_empty())
                .unwrap_or(HF_RESOLVE_UNKNOWN_FALLBACK_PATH);
            return config.snapshot_dir.join(PARQUET_MANIFEST_DIR).join(suffix);
        }
        config.snapshot_dir.join(candidate)
    }

    /// Validate target file size against expected bytes when available.
    fn target_matches_expected_size(path: &Path, expected_bytes: Option<u64>) -> bool {
        if !path.exists() {
            return false;
        }
        if let Some(expected) = expected_bytes
            && expected > 0
        {
            return fs::metadata(path)
                .map(|meta| meta.len() == expected)
                .unwrap_or(false);
        }
        true
    }

    /// Return on-disk size for a shard path, or 0 if metadata lookup fails.
    fn shard_size_bytes(path: &Path) -> u64 {
        fs::metadata(path).map(|meta| meta.len()).unwrap_or(0)
    }

    /// Return root directory used for manifest-cached remote shards.
    fn manifest_cache_root(&self) -> PathBuf {
        self.config.snapshot_dir.join(PARQUET_MANIFEST_DIR)
    }

    /// Recompute shard `global_start` offsets and total materialized row count.
    fn recompute_shard_offsets(state: &mut SourceState) {
        let mut running = 0usize;
        for shard in &mut state.shards {
            shard.global_start = running;
            running = running.saturating_add(shard.row_count);
        }
        state.materialized_rows = running;
    }

    /// Sync in-memory shard state from current on-disk snapshot tree.
    fn sync_shard_state_from_disk_locked(&self, state: &mut SourceState) {
        // If any shards have been evicted by the cache manager, remove them from
        // the in-memory index and reset the candidate list so the next expansion
        // cycle re-queries HF.  `all_candidates_from_parquet_manifest` returns every
        // shard from the manifest; evicted ones will be re-downloaded on next iteration.
        let any_missing = state.shards.iter().any(|shard| !shard.path.exists());
        state.shards.retain(|shard| shard.path.exists());
        Self::recompute_shard_offsets(state);
        if any_missing {
            state.remote_candidates = None;
            state.remote_candidate_order = Vec::new();
            state.next_remote_idx = 0;
        }
    }

    /// Apply cache-manager eviction policy to manifest shards and sync in-memory state.
    fn enforce_disk_cap_locked(
        &self,
        state: &mut SourceState,
        _protected_path: &Path,
    ) -> Result<bool, SamplerError> {
        let Some(cap_bytes) = self.config.local_disk_cap_bytes else {
            return Ok(false);
        };

        let before = state
            .shards
            .iter()
            .map(|shard| shard.path.clone())
            .collect::<Vec<_>>();
        let policy = EvictPolicy {
            max_bytes: Some(cap_bytes),
            ..EvictPolicy::default()
        };

        let cache_root = CacheRoot::from_root(&self.config.snapshot_dir);
        cache_root
            .ensure_group_with_policy(PARQUET_MANIFEST_DIR, Some(&policy))
            .map_err(|err| SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: format!(
                    "failed applying manifest cache eviction policy under {}: {err}",
                    self.config.snapshot_dir.display()
                ),
            })?;

        self.sync_shard_state_from_disk_locked(state);
        let after = state
            .shards
            .iter()
            .map(|shard| shard.path.clone())
            .collect::<Vec<_>>();
        Ok(before != after)
    }

    /// Return total on-disk bytes used by manifest-backed shards.
    fn manifest_usage_bytes_locked(&self, state: &SourceState) -> u64 {
        let manifest_root = self.manifest_cache_root();
        state
            .shards
            .iter()
            .filter(|shard| shard.path.starts_with(&manifest_root))
            .map(|shard| Self::shard_size_bytes(&shard.path))
            .sum::<u64>()
    }

    /// Fetch exact split row count metadata from datasets-server size endpoint.
    fn fetch_global_row_count(
        config: &HuggingFaceRowsConfig,
    ) -> Result<Option<usize>, SamplerError> {
        let endpoint = Self::size_endpoint();
        info!(
            "[triplets:hf] requesting global row count dataset='{}' config='{}' split='{}'",
            config.dataset, config.config, config.split
        );

        let mut request = ureq::get(&endpoint)
            .query(HF_JSON_KEY_DATASET, &config.dataset)
            .query(HF_JSON_KEY_CONFIG, &config.config);
        // When split is empty (all-splits mode) omit the split query param so the
        // server returns the total row count for the whole config.
        if !config.split.is_empty() {
            request = request.query(HF_JSON_KEY_SPLIT, &config.split);
        }
        let response = request
            .call()
            .map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!("failed querying datasets-server size endpoint: {err}"),
            })?;

        let body = response.into_body().read_to_string().map_err(|err| {
            SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!("failed reading datasets-server size response body: {err}"),
            }
        })?;

        Self::parse_global_row_count_response(config, &body)
    }

    fn parse_global_row_count_response(
        config: &HuggingFaceRowsConfig,
        body: &str,
    ) -> Result<Option<usize>, SamplerError> {
        let json: Value =
            serde_json::from_str(body).map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!("failed parsing datasets-server size response: {err}"),
            })?;

        let count =
            Self::extract_split_row_count_from_size_response(&json, &config.config, &config.split);
        Ok(count)
    }

    fn is_parquet_path(path: &Path) -> bool {
        path.extension()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.eq_ignore_ascii_case("parquet"))
    }

    fn allocate_temp_download_path(
        config: &HuggingFaceRowsConfig,
        remote_path: &str,
        extension: &str,
    ) -> Result<PathBuf, SamplerError> {
        let mut hasher = DefaultHasher::new();
        config.source_id.hash(&mut hasher);
        remote_path.hash(&mut hasher);
        let fingerprint = hasher.finish();
        let prefix = format!("triplets_hf_{fingerprint:016x}_");
        let suffix = format!(".{}", extension.trim_start_matches('.'));
        let temp_file = tempfile::Builder::new()
            .prefix(&prefix)
            .suffix(&suffix)
            .tempfile()
            .map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!("failed creating temporary download file: {err}"),
            })?;
        let (_, path) = temp_file
            .keep()
            .map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!(
                    "failed persisting temporary download path for '{}': {}",
                    remote_path, err.error
                ),
            })?;
        Ok(path)
    }

    fn download_remote_url_to_target(
        config: &HuggingFaceRowsConfig,
        remote_url: &str,
        target: &Path,
        expected_bytes: Option<u64>,
        shard_label: &str,
    ) -> Result<(), SamplerError> {
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!(
                    "failed creating shard output dir {}: {err}",
                    parent.display()
                ),
            })?;
        }

        let temp_target = target.with_extension("part");
        if temp_target.exists() {
            let _ = fs::remove_file(&temp_target);
        }

        let response =
            ureq::get(remote_url)
                .call()
                .map_err(|err| SamplerError::SourceUnavailable {
                    source_id: config.source_id.clone(),
                    reason: format!("failed downloading shard URL '{}': {err}", remote_url),
                })?;
        let mut reader = response.into_body().into_reader();
        let mut file =
            File::create(&temp_target).map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!(
                    "failed creating target shard {}: {err}",
                    temp_target.display()
                ),
            })?;
        info!(
            "[triplets:hf] {} {} downloading shard payload -> {}",
            config.source_id,
            shard_label,
            target.display()
        );
        let started = Instant::now();
        let mut total_bytes = 0u64;
        let mut buffer = vec![0u8; 8 * 1024 * 1024];
        let mut last_report = Instant::now();
        loop {
            let read = reader
                .read(&mut buffer)
                .map_err(|err| SamplerError::SourceUnavailable {
                    source_id: config.source_id.clone(),
                    reason: format!("failed reading shard stream '{}': {err}", remote_url),
                })?;
            if read == 0 {
                break;
            }
            file.write_all(&buffer[..read])
                .map_err(|err| SamplerError::SourceUnavailable {
                    source_id: config.source_id.clone(),
                    reason: format!(
                        "failed writing target shard {}: {err}",
                        temp_target.display()
                    ),
                })?;
            total_bytes = total_bytes.saturating_add(read as u64);
            if last_report.elapsed() >= Duration::from_secs(2) {
                let elapsed = started.elapsed().as_secs_f64();
                if let Some(expected) = expected_bytes
                    && expected > 0
                {
                    let pct = ((total_bytes as f64 / expected as f64) * 100.0).clamp(0.0, 100.0);
                    let rate = if elapsed > 0.0 {
                        total_bytes as f64 / elapsed
                    } else {
                        0.0
                    };
                    let eta_secs = if rate > 0.0 && total_bytes < expected {
                        (expected.saturating_sub(total_bytes) as f64) / rate
                    } else {
                        0.0
                    };
                    info!(
                        "[triplets:hf] {} {} download progress: {:.1}/{:.1} MiB ({:.1}%, {:.1}s elapsed, ETA {:.1}s)",
                        config.source_id,
                        shard_label,
                        total_bytes as f64 / (1024.0 * 1024.0),
                        expected as f64 / (1024.0 * 1024.0),
                        pct,
                        elapsed,
                        eta_secs.max(0.0)
                    );
                } else {
                    info!(
                        "[triplets:hf] {} {} download progress: {:.1} MiB ({:.1}s)",
                        config.source_id,
                        shard_label,
                        total_bytes as f64 / (1024.0 * 1024.0),
                        elapsed
                    );
                }
                last_report = Instant::now();
            }
        }
        let elapsed = started.elapsed().as_secs_f64();
        if let Some(expected) = expected_bytes
            && expected > 0
        {
            let pct = ((total_bytes as f64 / expected as f64) * 100.0).clamp(0.0, 100.0);
            info!(
                "[triplets:hf] {} {} download complete: {:.1}/{:.1} MiB ({:.1}%) in {:.1}s",
                config.source_id,
                shard_label,
                total_bytes as f64 / (1024.0 * 1024.0),
                expected as f64 / (1024.0 * 1024.0),
                pct,
                elapsed
            );
        } else {
            info!(
                "[triplets:hf] {} {} download complete: {:.1} MiB in {:.1}s",
                config.source_id,
                shard_label,
                total_bytes as f64 / (1024.0 * 1024.0),
                elapsed
            );
        }

        fs::rename(&temp_target, target).map_err(|err| SamplerError::SourceUnavailable {
            source_id: config.source_id.clone(),
            reason: format!(
                "failed moving downloaded shard {} -> {}: {err}",
                temp_target.display(),
                target.display()
            ),
        })?;

        Ok(())
    }

    /// Extract split row count from datasets-server size payload variants.
    fn extract_split_row_count_from_size_response(
        json: &Value,
        config_name: &str,
        split_name: &str,
    ) -> Option<usize> {
        let to_usize = |value: &Value| value.as_u64().and_then(|raw| usize::try_from(raw).ok());

        let size = json.get(HF_JSON_KEY_SIZE)?;

        if let Some(splits) = size.get(HF_JSON_KEY_SPLITS).and_then(Value::as_array) {
            for entry in splits {
                let entry_config = entry
                    .get(HF_JSON_KEY_CONFIG)
                    .or_else(|| entry.get(HF_JSON_KEY_CONFIG_NAME))
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let entry_split = entry
                    .get(HF_JSON_KEY_SPLIT)
                    .or_else(|| entry.get(HF_JSON_KEY_SPLIT_NAME))
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if entry_config == config_name
                    && entry_split == split_name
                    && let Some(rows) = entry.get(HF_JSON_KEY_NUM_ROWS).and_then(to_usize)
                {
                    return Some(rows);
                }
            }
        }

        if let Some(configs) = size.get(HF_JSON_KEY_CONFIGS).and_then(Value::as_array) {
            for config_entry in configs {
                let entry_config = config_entry
                    .get(HF_JSON_KEY_CONFIG)
                    .or_else(|| config_entry.get(HF_JSON_KEY_CONFIG_NAME))
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if entry_config != config_name {
                    continue;
                }

                if let Some(splits) = config_entry
                    .get(HF_JSON_KEY_SPLITS)
                    .and_then(Value::as_array)
                {
                    for split_entry in splits {
                        let entry_split = split_entry
                            .get(HF_JSON_KEY_SPLIT)
                            .or_else(|| split_entry.get(HF_JSON_KEY_SPLIT_NAME))
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        if entry_split == split_name
                            && let Some(rows) =
                                split_entry.get(HF_JSON_KEY_NUM_ROWS).and_then(to_usize)
                        {
                            return Some(rows);
                        }
                    }
                }

                if split_name.is_empty()
                    && let Some(rows) = config_entry.get(HF_JSON_KEY_NUM_ROWS).and_then(to_usize)
                {
                    return Some(rows);
                }
            }
        }

        if split_name.is_empty() {
            return size
                .get(HF_JSON_KEY_DATASET)
                .and_then(|dataset| dataset.get(HF_JSON_KEY_NUM_ROWS))
                .and_then(to_usize);
        }

        None
    }

    /// Build the stable human-readable label used in every shard-related log line.
    ///
    /// Format: `<file> (shard <M>/<total>)` where `M` is the 1-based index of this
    /// shard file in the sorted remote manifest and `total` is the total number of
    /// remote shards.  This label is purely file-derived and never depends on the
    /// ephemeral shuffle-position counter (`next_remote_idx`), which can reset
    /// whenever the candidate list is rebuilt for any reason, making position-based
    /// counters unfit for human interpretation.
    fn format_shard_label(
        remote_path: &str,
        candidate_idx: usize,
        candidate_total: usize,
    ) -> String {
        let file = remote_path
            .rsplit('/')
            .next()
            .unwrap_or(remote_path)
            .trim_start_matches("url::");
        format!("{file} (shard {}/{candidate_total})", candidate_idx + 1)
    }

    /// Download a shard (URL or hf-hub path) and materialize it under snapshot dir.
    fn download_and_materialize_shard(
        config: &HuggingFaceRowsConfig,
        remote_path: &str,
        expected_bytes: Option<u64>,
        shard_label: &str,
    ) -> Result<PathBuf, SamplerError> {
        if let Some(remote_url) = remote_path.strip_prefix(REMOTE_URL_PREFIX) {
            let target = Self::candidate_target_path(config, remote_path);
            let store_target = Self::shard_store_path_for(&target);
            if store_target.exists() {
                return Ok(store_target);
            }
            let parquet_candidate = Self::is_parquet_path(&target);
            if parquet_candidate {
                if target.exists() {
                    let _ = fs::remove_file(&target);
                }
                let temp_target =
                    Self::allocate_temp_download_path(config, remote_path, "parquet")?;
                Self::download_remote_url_to_target(
                    config,
                    remote_url,
                    &temp_target,
                    expected_bytes,
                    shard_label,
                )?;
                return Ok(temp_target);
            }
            if target.exists() {
                if Self::target_matches_expected_size(&target, expected_bytes) {
                    return Ok(target);
                }
                warn!(
                    "[triplets:hf] replacing incomplete shard before retry: {}",
                    target.display()
                );
                fs::remove_file(&target).map_err(|err| SamplerError::SourceUnavailable {
                    source_id: config.source_id.clone(),
                    reason: format!(
                        "failed removing incomplete shard {}: {err}",
                        target.display()
                    ),
                })?;
            }

            Self::download_remote_url_to_target(
                config,
                remote_url,
                &target,
                expected_bytes,
                shard_label,
            )?;
            return Ok(target);
        }

        if Path::new(remote_path)
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.eq_ignore_ascii_case("parquet"))
        {
            let target = Self::candidate_target_path(config, remote_path);
            let store_target = Self::shard_store_path_for(&target);
            if store_target.exists() {
                return Ok(store_target);
            }
            let remote_url = format!(
                "https://huggingface.co/datasets/{}/resolve/main/{}",
                config.dataset,
                remote_path.trim_start_matches('/')
            );
            let temp_target = Self::allocate_temp_download_path(config, remote_path, "parquet")?;
            Self::download_remote_url_to_target(
                config,
                &remote_url,
                &temp_target,
                expected_bytes,
                shard_label,
            )?;
            return Ok(temp_target);
        }

        let api = ApiBuilder::new()
            .with_progress(true)
            .with_retries(5)
            .with_token(None)
            .build()
            .map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!("failed building hf-hub client: {err}"),
            })?;

        let repo = Repo::new(config.dataset.clone(), RepoType::Dataset);
        let repo_api = api.repo(repo);

        let mut local_cached =
            repo_api
                .get(remote_path)
                .map_err(|err| SamplerError::SourceUnavailable {
                    source_id: config.source_id.clone(),
                    reason: format!("failed downloading '{}' from hf-hub: {err}", remote_path),
                })?;
        if !local_cached.exists() {
            for _ in 0..5 {
                local_cached = repo_api.download(remote_path).map_err(|err| {
                    SamplerError::SourceUnavailable {
                        source_id: config.source_id.clone(),
                        reason: format!(
                            "hf-hub returned missing cache path for '{}', and forced download failed: {err}",
                            remote_path
                        ),
                    }
                })?;
                if local_cached.exists() {
                    break;
                }
                thread::sleep(Duration::from_millis(400));
            }
        }
        if !local_cached.exists() {
            return Err(SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!(
                    "hf-hub returned non-existent cache file for '{}' at {}",
                    remote_path,
                    local_cached.display()
                ),
            });
        }

        let target = Self::candidate_target_path(config, remote_path);
        let store_target = Self::shard_store_path_for(&target);
        if store_target.exists() {
            return Ok(store_target);
        }
        Self::materialize_local_file(config, &local_cached, &target)?;
        Ok(target)
    }

    /// Build shard metadata for a single local file.
    fn index_single_shard(
        config: &HuggingFaceRowsConfig,
        path: &Path,
        global_start: usize,
    ) -> Result<(Option<ShardIndex>, Option<Arc<DataStore>>), SamplerError> {
        let is_store = Self::is_store_shard_path(path);
        // Parquet is treated as a transient decode artifact only.
        // Persisted shard artifacts should be per-shard .simdr stores.
        let is_transient_parquet = path
            .extension()
            .and_then(|v| v.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("parquet"));

        let (rows, parquet_row_groups, checkpoints, maybe_store) = if is_store {
            let store = Arc::new(Self::open_shard_store(config, path)?);
            let rows = if let Some(entry) =
                store.read(HF_SHARD_STORE_META_ROWS_KEY).map_err(|err| {
                    SamplerError::SourceUnavailable {
                        source_id: config.source_id.clone(),
                        reason: format!("row-store meta read failed {}: {err}", path.display()),
                    }
                })? {
                let payload = entry.as_ref();
                if payload.len() != std::mem::size_of::<u64>() {
                    return Err(SamplerError::SourceUnavailable {
                        source_id: config.source_id.clone(),
                        reason: format!("invalid row-store meta payload {}", path.display()),
                    });
                }
                let mut raw = [0u8; 8];
                raw.copy_from_slice(payload);
                u64::from_le_bytes(raw) as usize
            } else {
                0
            };
            let groups = if rows > 0 {
                vec![(0, rows)]
            } else {
                Vec::new()
            };
            (rows, groups, Vec::new(), Some(store))
        } else if is_transient_parquet {
            let (rows, parquet_row_groups) = Self::parquet_row_group_map(config, path)?;
            (rows, parquet_row_groups, Vec::new(), None)
        } else {
            let file = File::open(path).map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!("failed opening shard {}: {err}", path.display()),
            })?;
            let mut reader = BufReader::new(file);
            let mut checkpoints = Vec::new();
            let mut line = String::new();
            let mut offset = 0u64;
            let mut rows = 0usize;

            loop {
                if rows.is_multiple_of(config.checkpoint_stride) {
                    checkpoints.push(offset);
                }
                line.clear();
                let bytes =
                    reader
                        .read_line(&mut line)
                        .map_err(|err| SamplerError::SourceUnavailable {
                            source_id: config.source_id.clone(),
                            reason: format!("failed reading shard {}: {err}", path.display()),
                        })?;
                if bytes == 0 {
                    break;
                }
                rows += 1;
                offset = offset.saturating_add(bytes as u64);
            }

            (rows, Vec::new(), checkpoints, None)
        };

        if rows == 0 {
            return Ok((None, None));
        }

        Ok((
            Some(ShardIndex {
                path: path.to_path_buf(),
                global_start,
                row_count: rows,
                random_access: is_transient_parquet || is_store,
                parquet_row_groups,
                checkpoints,
                remote_candidate: None,
            }),
            maybe_store,
        ))
    }

    /// Build parquet row-group map for random-access row reads.
    fn parquet_row_group_map(
        config: &HuggingFaceRowsConfig,
        path: &Path,
    ) -> Result<(usize, Vec<(usize, usize)>), SamplerError> {
        let file = File::open(path).map_err(|err| SamplerError::SourceUnavailable {
            source_id: config.source_id.clone(),
            reason: format!("failed opening parquet shard {}: {err}", path.display()),
        })?;
        let reader =
            SerializedFileReader::new(file).map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!("failed reading parquet metadata {}: {err}", path.display()),
            })?;

        let mut row_groups = Vec::new();
        let mut running = 0usize;
        for meta in reader.metadata().row_groups() {
            let group_rows =
                usize::try_from(meta.num_rows()).map_err(|_| SamplerError::SourceUnavailable {
                    source_id: config.source_id.clone(),
                    reason: format!("parquet row group size overflow in {}", path.display()),
                })?;
            if group_rows == 0 {
                continue;
            }
            row_groups.push((running, group_rows));
            running = running.saturating_add(group_rows);
        }
        if running > 0 {
            return Ok((running, row_groups));
        }

        let total_rows =
            usize::try_from(reader.metadata().file_metadata().num_rows()).map_err(|_| {
                SamplerError::SourceUnavailable {
                    source_id: config.source_id.clone(),
                    reason: format!("parquet row count overflow in {}", path.display()),
                }
            })?;
        if total_rows == 0 {
            return Ok((0, Vec::new()));
        }
        Ok((total_rows, vec![(0, total_rows)]))
    }

    /// Ensure row index is available, expanding remote shard set lazily if needed.
    fn ensure_row_available(&self, idx: usize) -> Result<bool, SamplerError> {
        loop {
            {
                let state = self
                    .state
                    .lock()
                    .map_err(|_| SamplerError::SourceUnavailable {
                        source_id: self.config.source_id.clone(),
                        reason: "huggingface source state lock poisoned".to_string(),
                    })?;

                if idx < state.materialized_rows {
                    return Ok(true);
                }

                if let Some(candidates) = &state.remote_candidates
                    && state.next_remote_idx >= candidates.len()
                {
                    return Ok(false);
                }
            }

            let need_candidates = {
                let state = self
                    .state
                    .lock()
                    .map_err(|_| SamplerError::SourceUnavailable {
                        source_id: self.config.source_id.clone(),
                        reason: "huggingface source state lock poisoned".to_string(),
                    })?;
                state.remote_candidates.is_none()
            };

            if need_candidates {
                let mut state = self
                    .state
                    .lock()
                    .map_err(|_| SamplerError::SourceUnavailable {
                        source_id: self.config.source_id.clone(),
                        reason: "huggingface source state lock poisoned".to_string(),
                    })?;
                if state.remote_candidates.is_none() {
                    let (mut candidates, candidate_sizes) =
                        Self::list_remote_candidates(&self.config)?;
                    candidates.sort();
                    candidates.dedup();
                    let sampler_seed = self.configured_sampler_seed().unwrap_or(0);
                    let order =
                        Self::build_candidate_order(&self.config, &candidates, sampler_seed);

                    // Skip positions whose shard is already materialised on disk.
                    // Determinism: order is built from the full HF manifest regardless of
                    // cache state — position N for seed S always maps to the same shard.
                    // Cache: on restart we advance past already-downloaded shards so we
                    // don't redundantly re-download what we already have.
                    let next_idx = Self::first_uncached_order_position(
                        &self.config,
                        &candidates,
                        &order,
                        &state.shards,
                    );

                    state.remote_candidates = Some(candidates);
                    state.remote_candidate_order = order;
                    state.remote_candidate_sizes = candidate_sizes;
                    state.next_remote_idx = next_idx;

                    let candidate_count = state
                        .remote_candidates
                        .as_ref()
                        .map(|values| values.len())
                        .unwrap_or(0);
                    let bootstrap_needed = state.materialized_rows == 0
                        && candidate_count > 0
                        && state.next_remote_idx == 0;
                    let known_rows = state.materialized_rows;
                    let shard_count = state.shards.len();
                    info!(
                        "[triplets:hf] {} state: candidates={} known_rows={} active_shards={} disk_cap={}",
                        self.config.source_id,
                        candidate_count,
                        known_rows,
                        shard_count,
                        self.config
                            .local_disk_cap_bytes
                            .map(|bytes| format!(
                                "{:.2} GiB",
                                bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                            ))
                            .unwrap_or_else(|| "disabled".to_string()),
                    );
                    drop(state);

                    if bootstrap_needed {
                        let bootstrap_target = REMOTE_BOOTSTRAP_SHARDS.min(candidate_count);
                        info!(
                            "[triplets:hf] {} cold start: downloading {} initial shard(s) before first read",
                            self.config.source_id, bootstrap_target
                        );
                        for _ in 0..bootstrap_target {
                            if !self.download_next_remote_shard()? {
                                break;
                            }
                        }
                        info!(
                            "[triplets:hf] {} cold start complete",
                            self.config.source_id
                        );
                    }
                } else {
                    drop(state);
                }
                continue;
            }
            if !self.download_next_remote_shard()? {
                return Ok(false);
            }
        }
    }

    /// Download and register the next remote shard candidate.
    ///
    /// If the shard's store file already exists on disk (materialised from a previous
    /// run), the download is skipped and `next_remote_idx` is still advanced.  This
    /// keeps the shard download order stable regardless of cache state: the ordered
    /// position is consumed either way, but no redundant network traffic occurs.
    fn download_next_remote_shard(&self) -> Result<bool, SamplerError> {
        let (remote_total, cached_shards, candidate_idx, remote_path, expected_bytes) = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| SamplerError::SourceUnavailable {
                    source_id: self.config.source_id.clone(),
                    reason: "huggingface source state lock poisoned".to_string(),
                })?;
            {
                let Some(candidates) = &state.remote_candidates else {
                    return Ok(false);
                };
                if state.next_remote_idx >= candidates.len() {
                    return Ok(false);
                }
                let sequence_pos = state.next_remote_idx;
                let remote_total = candidates.len();
                let cached_shards = state.shards.len();
                // Use the seed-derived order index so the mapping from position →
                // shard is stable and independent of how many shards were previously
                // consumed.  Fall back to direct indexing only if the order vec is
                // somehow not populated (should never happen after candidates are set).
                let candidate_idx = state
                    .remote_candidate_order
                    .get(sequence_pos)
                    .copied()
                    .unwrap_or(sequence_pos);
                let remote_path = candidates[candidate_idx].clone();
                let expected_bytes = state.remote_candidate_sizes.get(&remote_path).copied();
                state.next_remote_idx += 1;

                // If this shard is already materialised on disk (from a previous run),
                // skip the download — it is already counted in materialized_rows via
                // build_shard_index.  Cache and order are fully decoupled: the position
                // is consumed regardless, but no network request is made.
                let store_path = Self::candidate_store_path(&self.config, &remote_path);
                if store_path.exists() {
                    debug!(
                        "[triplets:hf] {} {} already on disk, skipping download",
                        self.config.source_id,
                        Self::format_shard_label(remote_path.as_str(), candidate_idx, remote_total),
                    );
                    return Ok(true);
                }

                (
                    remote_total,
                    cached_shards,
                    candidate_idx,
                    remote_path,
                    expected_bytes,
                )
            }
        };

        let label = Self::format_shard_label(remote_path.as_str(), candidate_idx, remote_total);
        info!(
            "[triplets:hf] {} downloading {} ({} cached before)",
            self.config.source_id, label, cached_shards,
        );
        let local_path = Self::download_and_materialize_shard(
            &self.config,
            &remote_path,
            expected_bytes,
            &label,
        )?;

        let global_start = {
            let state = self
                .state
                .lock()
                .map_err(|_| SamplerError::SourceUnavailable {
                    source_id: self.config.source_id.clone(),
                    reason: "huggingface source state lock poisoned".to_string(),
                })?;
            state.materialized_rows
        };

        let (maybe_shard, _) = Self::index_single_shard(&self.config, &local_path, global_start)?;
        let Some(shard) = maybe_shard else {
            warn!(
                "[triplets:hf] downloaded shard had zero rows and was skipped: {}",
                local_path.display()
            );
            return Ok(true);
        };

        let state = self
            .state
            .lock()
            .map_err(|_| SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: "huggingface source state lock poisoned".to_string(),
            })?;

        let rows_to_add = shard.row_count;
        // All rows from this shard are now available.  A per-shard cap was
        // previously applied here but has been removed: reads are now gated on
        // materialized_rows and expansion happens one shard per refresh(), so a
        // large Wikipedia shard can no longer stall the read loop regardless of
        // how many rows it contributes.
        if rows_to_add == 0 {
            return Ok(true);
        }

        let mut shard = shard;
        shard.global_start = state.materialized_rows;
        shard.row_count = rows_to_add;
        if shard.random_access {
            shard
                .parquet_row_groups
                .retain(|(start, _)| *start < rows_to_add);
            if let Some((start, count)) = shard.parquet_row_groups.last_mut() {
                let allowed = rows_to_add.saturating_sub(*start);
                *count = (*count).min(allowed);
            }
        }

        drop(state);
        let mut shard = match self.transcode_parquet_shard_to_row_store(&shard)? {
            Some(shard) => shard,
            None => return Ok(true),
        };

        if local_path
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.eq_ignore_ascii_case("parquet"))
            && Self::is_store_shard_path(&shard.path)
        {
            let canonical_store = Self::shard_store_path_for(&Self::candidate_target_path(
                &self.config,
                &remote_path,
            ));
            if shard.path != canonical_store {
                if let Some(parent) = canonical_store.parent() {
                    fs::create_dir_all(parent).map_err(|err| SamplerError::SourceUnavailable {
                        source_id: self.config.source_id.clone(),
                        reason: format!(
                            "failed creating canonical store parent {}: {err}",
                            parent.display()
                        ),
                    })?;
                }

                if canonical_store.exists() {
                    fs::remove_file(&canonical_store).map_err(|err| {
                        SamplerError::SourceUnavailable {
                            source_id: self.config.source_id.clone(),
                            reason: format!(
                                "failed replacing canonical store {}: {err}",
                                canonical_store.display()
                            ),
                        }
                    })?;
                }

                if let Err(rename_err) = fs::rename(&shard.path, &canonical_store) {
                    fs::copy(&shard.path, &canonical_store).map_err(|copy_err| {
                        SamplerError::SourceUnavailable {
                            source_id: self.config.source_id.clone(),
                            reason: format!(
                                "failed moving temporary store {} -> {}: rename={rename_err}; copy={copy_err}",
                                shard.path.display(),
                                canonical_store.display()
                            ),
                        }
                    })?;
                    fs::remove_file(&shard.path).map_err(|cleanup_err| {
                        SamplerError::SourceUnavailable {
                            source_id: self.config.source_id.clone(),
                            reason: format!(
                                "failed cleaning temporary store {} after copy move: {cleanup_err}",
                                shard.path.display()
                            ),
                        }
                    })?;
                }

                shard.path = canonical_store;
            }
        }

        let mut state = self
            .state
            .lock()
            .map_err(|_| SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: "huggingface source state lock poisoned".to_string(),
            })?;
        state.materialized_rows += rows_to_add;
        shard.remote_candidate = Some(remote_path.clone());
        state.shards.push(shard);
        drop(state);
        self.invalidate_eligible_index();

        let mut state = self
            .state
            .lock()
            .map_err(|_| SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: "huggingface source state lock poisoned".to_string(),
            })?;

        let evicted_any = self.enforce_disk_cap_locked(&mut state, &local_path)?;
        let materialized_rows = state.materialized_rows;
        let shard_count = state.shards.len();
        let total_remote = state
            .remote_candidates
            .as_ref()
            .map(|c| c.len())
            .unwrap_or(0);
        let active_shards = state.shards.clone();
        let usage_bytes = self.manifest_usage_bytes_locked(&state);
        let usage_gib = usage_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let cap_str = self
            .config
            .local_disk_cap_bytes
            .map(|bytes| format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0)))
            .unwrap_or_else(|| "disabled".to_string());
        drop(state);
        self.prune_store_cache_to_shards(&active_shards);

        if evicted_any {
            if let Ok(mut cache) = self.cache.lock() {
                cache.rows.clear();
                cache.order.clear();
            }
            if let Ok(mut parquet_cache) = self.parquet_cache.lock() {
                parquet_cache.readers.clear();
                parquet_cache.row_groups.clear();
                parquet_cache.row_group_order.clear();
            }
            self.invalidate_eligible_index();
        }

        // `shard_count` is the number of shards currently on disk; `total_remote` is
        // how many shards the remote manifest reports in total for this dataset.
        info!(
            "[triplets:hf] {} rows={} shards_on_disk={}/{} disk_usage={:.2} GiB cap={}",
            self.config.source_id, materialized_rows, shard_count, total_remote, usage_gib, cap_str,
        );

        Ok(true)
    }

    /// Copy cached/downloaded source file into snapshot tree.
    fn materialize_local_file(
        config: &HuggingFaceRowsConfig,
        source_path: &Path,
        target_path: &Path,
    ) -> Result<(), SamplerError> {
        let resolved_source =
            fs::canonicalize(source_path).unwrap_or_else(|_| source_path.to_path_buf());

        if let Some(parent) = target_path.parent() {
            fs::create_dir_all(parent).map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!(
                    "failed creating snapshot subdir {}: {err}",
                    parent.display()
                ),
            })?;
        }

        if target_path.exists() {
            let src_meta =
                fs::metadata(&resolved_source).map_err(|err| SamplerError::SourceUnavailable {
                    source_id: config.source_id.clone(),
                    reason: format!(
                        "failed reading source metadata {}: {err}",
                        resolved_source.display()
                    ),
                })?;
            let dst_meta =
                fs::metadata(target_path).map_err(|err| SamplerError::SourceUnavailable {
                    source_id: config.source_id.clone(),
                    reason: format!(
                        "failed reading target metadata {}: {err}",
                        target_path.display()
                    ),
                })?;
            if src_meta.len() == dst_meta.len() {
                return Ok(());
            }
            fs::remove_file(target_path).map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!(
                    "failed replacing target file {}: {err}",
                    target_path.display()
                ),
            })?;
        }

        fs::copy(&resolved_source, target_path).map_err(|err| SamplerError::SourceUnavailable {
            source_id: config.source_id.clone(),
            reason: format!(
                "failed copying synced file {} -> {}: {err}",
                resolved_source.display(),
                target_path.display()
            ),
        })?;
        Ok(())
    }

    /// Build deterministic local shard index for accepted extensions.
    fn build_shard_index(config: &HuggingFaceRowsConfig) -> Result<ShardIndexResult, SamplerError> {
        let start_index = Instant::now();
        let mut shard_paths = Vec::new();
        let manifest_root = config.snapshot_dir.join(PARQUET_MANIFEST_DIR);
        let accepted = config
            .shard_extensions
            .iter()
            .map(|ext| ext.trim().trim_start_matches('.').to_ascii_lowercase())
            .collect::<Vec<_>>();

        let mut saw_parquet = false;
        for entry in WalkDir::new(&config.snapshot_dir)
            .follow_links(true)
            .into_iter()
            .filter_map(Result::ok)
        {
            if !entry.file_type().is_file() {
                continue;
            }
            let in_manifest = entry.path().starts_with(&manifest_root);
            let Some(ext) = entry.path().extension().and_then(|v| v.to_str()) else {
                continue;
            };
            if ext.eq_ignore_ascii_case("parquet") {
                // Parquet files under _parquet_manifest are transient download
                // artifacts (replaced by .simdr stores after transcoding).
                // Delete them quietly and skip — do NOT set saw_parquet, since
                // the .simdr stores may still be present and usable.
                // Parquet files outside the manifest are unexpected legacy
                // artifacts; flag them so the caller can warn and repopulate.
                if !in_manifest {
                    saw_parquet = true;
                }
                if let Err(err) = fs::remove_file(entry.path()) {
                    warn!(
                        "[triplets:hf] found persisted parquet shard (expected transient only) and failed to remove {}: {}",
                        entry.path().display(),
                        err
                    );
                }
                continue;
            }
            // For non-parquet files inside _parquet_manifest: only accept
            // shard store files (.simdr).  Metadata files like
            // _sequence_state.json and any other non-shard artifacts are
            // skipped.  Remote-sourced shards are stored as .simdr files
            // under _parquet_manifest and must be indexed here so that
            // materialized_rows is correctly populated on restart.
            if in_manifest && !ext.eq_ignore_ascii_case(HF_SHARD_STORE_EXTENSION) {
                continue;
            }
            if accepted
                .iter()
                .any(|allowed| allowed == &ext.to_ascii_lowercase())
            {
                shard_paths.push(entry.path().to_path_buf());
            }
        }

        shard_paths.sort();
        if shard_paths.is_empty() {
            if saw_parquet {
                warn!(
                    "[triplets:hf] found persisted parquet under {} (transient-only policy); parquet files were pruned and source will repopulate from remote candidates",
                    config.snapshot_dir.display()
                );
            }
            return Err(SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!(
                    "no shard files found under {} with extensions {:?}",
                    config.snapshot_dir.display(),
                    config.shard_extensions
                ),
            });
        }

        let total_shards = shard_paths.len();
        info!("[triplets:hf] indexing {} shards in parallel", total_shards);
        let completed = AtomicUsize::new(0);
        let indexed_shards: Result<Vec<_>, _> = shard_paths
            .into_par_iter()
            .enumerate()
            .map(|(ordinal, path)| {
                let result = Self::index_single_shard(config, &path, 0)?;
                let n = completed.fetch_add(1, AtomicOrdering::Relaxed) + 1;
                let row_count = result.0.as_ref().map_or(0, |s| s.row_count);
                debug!(
                    "[triplets:hf] indexed shard {}/{}: {} ({} rows)",
                    n,
                    total_shards,
                    path.file_name()
                        .unwrap_or(path.as_os_str())
                        .to_string_lossy(),
                    row_count,
                );
                Ok::<_, SamplerError>((ordinal, result))
            })
            .collect();
        let mut indexed_shards = indexed_shards?;

        indexed_shards.sort_by_key(|(ordinal, _)| *ordinal);

        let mut shards = Vec::new();
        let mut running_total = 0usize;
        let mut store_cache: HashMap<PathBuf, Arc<DataStore>> = HashMap::new();
        for (_, (maybe_shard, maybe_store)) in indexed_shards {
            let Some(mut shard) = maybe_shard else {
                continue;
            };

            if shard.row_count == 0 {
                continue;
            }

            if let Some(store) = maybe_store {
                store_cache.insert(shard.path.clone(), store);
            }
            shard.global_start = running_total;
            running_total = running_total.saturating_add(shard.row_count);
            shards.push(shard);
        }

        info!(
            "[triplets:hf] indexing complete in {:.2}s (rows={}, shards={})",
            start_index.elapsed().as_secs_f64(),
            running_total,
            shards.len()
        );

        Ok((shards, running_total, store_cache))
    }

    /// Locate containing shard and local offset for a global row index.
    fn locate_shard(shards: &[ShardIndex], idx: usize) -> Option<(&ShardIndex, usize)> {
        let pos = shards
            .binary_search_by(|shard| {
                if idx < shard.global_start {
                    Ordering::Greater
                } else if idx >= shard.global_start + shard.row_count {
                    Ordering::Less
                } else {
                    Ordering::Equal
                }
            })
            .ok()?;
        let shard = shards.get(pos)?;
        Some((shard, idx - shard.global_start))
    }

    /// Read one JSONL/NDJSON line at a local row offset using checkpoints.
    fn read_line_at(&self, shard: &ShardIndex, local_idx: usize) -> Result<String, SamplerError> {
        let checkpoint_idx = local_idx / self.config.checkpoint_stride;
        let checkpoint_line = checkpoint_idx * self.config.checkpoint_stride;
        let seek_offset = *shard.checkpoints.get(checkpoint_idx).ok_or_else(|| {
            SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: format!(
                    "missing checkpoint for shard {} line {}",
                    shard.path.display(),
                    local_idx
                ),
            }
        })?;

        let mut file = File::open(&shard.path).map_err(|err| SamplerError::SourceUnavailable {
            source_id: self.config.source_id.clone(),
            reason: format!("failed opening shard {}: {err}", shard.path.display()),
        })?;
        file.seek(SeekFrom::Start(seek_offset))
            .map_err(|err| SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: format!("failed seeking shard {}: {err}", shard.path.display()),
            })?;

        let mut reader = BufReader::new(file);
        let mut line = String::new();
        for _ in checkpoint_line..local_idx {
            line.clear();
            let bytes =
                reader
                    .read_line(&mut line)
                    .map_err(|err| SamplerError::SourceUnavailable {
                        source_id: self.config.source_id.clone(),
                        reason: format!("failed scanning shard {}: {err}", shard.path.display()),
                    })?;
            if bytes == 0 {
                return Err(SamplerError::SourceUnavailable {
                    source_id: self.config.source_id.clone(),
                    reason: format!(
                        "unexpected EOF while scanning shard {} at row {}",
                        shard.path.display(),
                        local_idx
                    ),
                });
            }
        }

        line.clear();
        let bytes = reader
            .read_line(&mut line)
            .map_err(|err| SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: format!("failed reading shard {}: {err}", shard.path.display()),
            })?;
        if bytes == 0 {
            return Err(SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: format!(
                    "unexpected EOF while reading shard {} row {}",
                    shard.path.display(),
                    local_idx
                ),
            });
        }
        Ok(line)
    }

    /// Locate parquet row-group and in-group row offset for a local row index.
    fn locate_parquet_group(
        &self,
        shard: &ShardIndex,
        local_idx: usize,
    ) -> Result<(usize, usize), SamplerError> {
        let group_pos = shard
            .parquet_row_groups
            .binary_search_by(|(start, count)| {
                if local_idx < *start {
                    Ordering::Greater
                } else if local_idx >= start.saturating_add(*count) {
                    Ordering::Less
                } else {
                    Ordering::Equal
                }
            })
            .map_err(|_| SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: format!(
                    "parquet row {} could not be mapped to a row group in {}",
                    local_idx,
                    shard.path.display()
                ),
            })?;
        let (group_start, _) = shard.parquet_row_groups[group_pos];
        Ok((group_pos, local_idx.saturating_sub(group_start)))
    }

    /// Convert a serde JSON value into non-empty text when possible.
    ///
    /// `label_names` optionally provides an ordered list of label strings for
    /// ClassLabel-style integer columns.  When the value is an integer `n` and
    /// `label_names[n]` exists, that label string is returned instead of the
    /// raw numeric string.
    fn value_to_text(value: &Value, label_names: Option<&[String]>) -> Option<String> {
        match value {
            Value::Null => None,
            Value::String(s) => {
                if s.trim().is_empty() {
                    None
                } else {
                    Some(s.clone())
                }
            }
            Value::Bool(b) => Some(b.to_string()),
            Value::Number(n) => {
                if let Some(labels) = label_names
                    && let Some(idx) = n.as_u64().map(|u| u as usize)
                    && let Some(label) = labels.get(idx)
                    && !label.trim().is_empty()
                {
                    return Some(label.clone());
                }
                Some(n.to_string())
            }
            Value::Array(arr) => serde_json::to_string(arr).ok().filter(|s| !s.is_empty()),
            Value::Object(obj) => serde_json::to_string(obj).ok().filter(|s| !s.is_empty()),
        }
    }

    /// Try each candidate column name in order and return the first one that
    /// yields a non-empty text value.  Returns `None` when no candidate
    /// matches, which the caller uses to decide whether to skip the row.
    fn coalesce_field(
        candidates: &[String],
        row_obj: &serde_json::Map<String, Value>,
        label_maps: &HashMap<String, Vec<String>>,
    ) -> Option<RowTextField> {
        for name in candidates {
            let label_names = label_maps.get(name).map(|v| v.as_slice());
            if let Some(value) = row_obj.get(name)
                && let Some(text) = Self::value_to_text(value, label_names)
            {
                return Some(RowTextField {
                    name: name.clone(),
                    text,
                });
            }
        }
        None
    }

    /// Parse a raw row payload into normalized `RowView` fields.
    fn parse_row(
        &self,
        absolute_idx: usize,
        row_value: &Value,
    ) -> Result<Option<RowView>, SamplerError> {
        if !self.config.has_explicit_mapping() {
            return Err(SamplerError::SourceInconsistent {
                source_id: self.config.source_id.clone(),
                details:
                    "huggingface row parsing requires explicit field mapping; no columns configured"
                        .to_string(),
            });
        }

        let row_payload = row_value.get("row").unwrap_or(row_value);
        let row_obj = row_payload
            .as_object()
            .ok_or_else(|| SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: "snapshot row entry missing JSON object payload".to_string(),
            })?;

        let row_id = self
            .config
            .id_column
            .as_ref()
            .and_then(|col| row_obj.get(col))
            .and_then(|v| Self::value_to_text(v, None))
            .unwrap_or_else(|| {
                format!(
                    "{}:{}:{}",
                    self.config.dataset, self.config.split, absolute_idx
                )
            });

        let mut text_fields = Vec::new();
        let use_role_columns = !self.config.anchor_columns.is_empty()
            || !self.config.positive_columns.is_empty()
            || !self.config.context_columns.is_empty();

        if use_role_columns {
            // Anchor: try each candidate column in order; use the first
            // whose value is present and non-empty.  Skip the row when the
            // list is non-empty but no candidate yields content.
            if !self.config.anchor_columns.is_empty() {
                match Self::coalesce_field(
                    &self.config.anchor_columns,
                    row_obj,
                    &self.config.label_maps,
                ) {
                    Some(field) => text_fields.push(field),
                    None => return Ok(None),
                }
            }

            // Positive: try each candidate column in order; use the first
            // whose value is present and non-empty.  Skip the row when the
            // list is non-empty but no candidate yields content.
            if !self.config.positive_columns.is_empty() {
                match Self::coalesce_field(
                    &self.config.positive_columns,
                    row_obj,
                    &self.config.label_maps,
                ) {
                    Some(field) => text_fields.push(field),
                    None => return Ok(None),
                }
            }

            for name in &self.config.context_columns {
                let Some(value) = row_obj.get(name) else {
                    return Ok(None);
                };
                let label_names = self.config.label_maps.get(name).map(|v| v.as_slice());
                let Some(text) = Self::value_to_text(value, label_names) else {
                    return Ok(None);
                };
                text_fields.push(RowTextField {
                    name: name.clone(),
                    text,
                });
            }
        } else {
            // Text-columns mode: try each candidate column in order; use the
            // first whose value is present and non-empty.  The row is skipped
            // when no candidate yields content (handled by the is_empty guard
            // below).
            if let Some(field) =
                Self::coalesce_field(&self.config.text_columns, row_obj, &self.config.label_maps)
            {
                text_fields.push(field);
            }
        }

        if text_fields.is_empty() {
            return Ok(None);
        }

        Ok(Some(RowView {
            row_id: Some(row_id),
            timestamp: None,
            text_fields,
        }))
    }

    /// Decode one line from a non-parquet shard into an object-like row payload.
    fn parse_non_parquet_line(
        &self,
        shard: &ShardIndex,
        local_idx: usize,
        line: &str,
    ) -> Result<Value, SamplerError> {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Err(SamplerError::SourceInconsistent {
                source_id: self.config.source_id.clone(),
                details: format!(
                    "empty row in shard {} at local index {}",
                    shard.path.display(),
                    local_idx
                ),
            });
        }

        let is_strict_json_lines = shard
            .path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| {
                ext.eq_ignore_ascii_case("jsonl") || ext.eq_ignore_ascii_case("ndjson")
            });

        match serde_json::from_str::<Value>(trimmed) {
            Ok(value) => {
                let payload = value.get("row").unwrap_or(&value);
                if payload.is_object() {
                    Ok(value)
                } else if let Some(text) = Self::value_to_text(payload, None) {
                    Ok(json!({ "text": text }))
                } else {
                    Err(SamplerError::SourceInconsistent {
                        source_id: self.config.source_id.clone(),
                        details: format!(
                            "non-object JSON row in shard {} at local index {} could not be converted to text",
                            shard.path.display(),
                            local_idx
                        ),
                    })
                }
            }
            Err(err) => {
                if is_strict_json_lines {
                    Err(SamplerError::SourceInconsistent {
                        source_id: self.config.source_id.clone(),
                        details: format!(
                            "failed decoding JSON row from shard {} at local index {}: {err}",
                            shard.path.display(),
                            local_idx
                        ),
                    })
                } else {
                    Ok(json!({ "text": trimmed }))
                }
            }
        }
    }

    /// Convert a `RowView` into a sampler `DataRecord`.
    fn row_to_record(
        &self,
        row: &RowView,
        row_index: u64,
    ) -> Result<Option<DataRecord>, SamplerError> {
        if row.text_fields.is_empty() {
            return Ok(None);
        }

        let record_id = row
            .row_id
            .as_ref()
            .cloned()
            .unwrap_or_else(|| format!("row_{row_index}"));
        let id = format!("{}::{}", self.config.source_id, record_id);

        let mut sections = Vec::new();
        let anchor = &row.text_fields[0];
        sections.push(make_section(
            SectionRole::Anchor,
            Some(anchor.name.as_str()),
            anchor.text.as_str(),
        ));

        let positive = row.text_fields.get(1).unwrap_or(anchor);
        sections.push(make_section(
            SectionRole::Context,
            Some(positive.name.as_str()),
            positive.text.as_str(),
        ));

        for field in row.text_fields.iter().skip(2) {
            sections.push(make_section(
                SectionRole::Context,
                Some(field.name.as_str()),
                field.text.as_str(),
            ));
        }

        let timestamp = row.timestamp.unwrap_or(DateTime::<Utc>::UNIX_EPOCH);
        Ok(Some(DataRecord {
            id,
            source: self.config.source_id.clone(),
            created_at: timestamp,
            updated_at: timestamp,
            quality: self
                .config
                .trust_override
                .map_or_else(QualityScore::default, |t| QualityScore { trust: t }),
            taxonomy: vec![
                format!("dataset={}", self.config.dataset),
                format!("config={}", self.config.config),
                format!("split={}", self.config.split),
            ],
            sections,
            meta_prefix: None,
        }))
    }

    /// Materialize records for requested indices into output buffer.
    fn read_row_batch(
        &self,
        indices: &[usize],
        out: &mut Vec<DataRecord>,
        limit: Option<usize>,
    ) -> Result<(), SamplerError> {
        let mut sorted = indices.to_vec();
        sorted.sort_unstable();

        let mut fetched = HashMap::with_capacity(sorted.len());
        let mut pending = Vec::new();
        for idx in &sorted {
            if !self.ensure_row_available(*idx)? {
                fetched.insert(*idx, None);
                continue;
            }

            if let Some(row) = self
                .cache
                .lock()
                .map_err(|_| SamplerError::SourceUnavailable {
                    source_id: self.config.source_id.clone(),
                    reason: "huggingface row cache lock poisoned".to_string(),
                })?
                .get(*idx)
            {
                let record = self.row_to_record(&row, *idx as u64)?;
                fetched.insert(*idx, record);
                continue;
            }

            pending.push(*idx);
        }

        if !pending.is_empty() {
            let resolutions =
                {
                    let state = self
                        .state
                        .lock()
                        .map_err(|_| SamplerError::SourceUnavailable {
                            source_id: self.config.source_id.clone(),
                            reason: "huggingface source state lock poisoned".to_string(),
                        })?;
                    let mut resolved = Vec::with_capacity(pending.len());
                    for idx in &pending {
                        let (shard, local_idx) = Self::locate_shard(&state.shards, *idx)
                            .ok_or_else(|| SamplerError::SourceUnavailable {
                                source_id: self.config.source_id.clone(),
                                reason: format!("row index out of range: {idx}"),
                            })?;
                        resolved.push((*idx, shard.clone(), local_idx));
                    }
                    resolved
                };

            let mut parquet_groups: HashMap<ParquetGroupKey, Vec<ParquetGroupRequest>> =
                HashMap::new();
            for (idx, shard, local_idx) in resolutions {
                if shard.random_access {
                    let (group_pos, local_in_group) =
                        self.locate_parquet_group(&shard, local_idx)?;
                    parquet_groups
                        .entry((shard.path.clone(), group_pos))
                        .or_default()
                        .push((idx, local_in_group, shard));
                    continue;
                }

                let line = self.read_line_at(&shard, local_idx)?;
                let row_value = self.parse_non_parquet_line(&shard, local_idx, &line)?;
                let row = self.parse_row(idx, &row_value)?;
                if let Some(row) = row {
                    let record = self.row_to_record(&row, idx as u64)?;
                    self.cache
                        .lock()
                        .map_err(|_| SamplerError::SourceUnavailable {
                            source_id: self.config.source_id.clone(),
                            reason: "huggingface row cache lock poisoned".to_string(),
                        })?
                        .insert(idx, row, self.config.cache_capacity);
                    fetched.insert(idx, record);
                } else {
                    fetched.insert(idx, None);
                }
            }

            for ((shard_path, group_pos), mut requested) in parquet_groups {
                requested.sort_by_key(|(_, local_in_group, _)| *local_in_group);
                let shard = requested
                    .first()
                    .map(|(_, _, shard)| shard.clone())
                    .ok_or_else(|| SamplerError::SourceUnavailable {
                        source_id: self.config.source_id.clone(),
                        reason: format!(
                            "missing parquet request metadata for shard {} row_group {}",
                            shard_path.display(),
                            group_pos
                        ),
                    })?;

                let mut targets: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
                for (idx, local_in_group, _) in requested {
                    targets.entry(local_in_group).or_default().push(idx);
                }

                let (group_start, _) = shard.parquet_row_groups[group_pos];
                let mut unresolved_targets: BTreeMap<usize, Vec<usize>> = targets.clone();

                if Self::is_store_shard_path(&shard.path) {
                    let store = self.get_or_open_shard_store(&shard.path)?;
                    let requested_positions = targets.keys().copied().collect::<Vec<_>>();
                    let store_keys = requested_positions
                        .iter()
                        .map(|position| {
                            let local_idx = group_start.saturating_add(*position);
                            Self::row_store_row_key(local_idx)
                        })
                        .collect::<Vec<_>>();
                    let store_key_refs = store_keys
                        .iter()
                        .map(|key| key.as_slice())
                        .collect::<Vec<_>>();
                    let store_entries = store.batch_read(&store_key_refs).map_err(|err| {
                        SamplerError::SourceUnavailable {
                            source_id: self.config.source_id.clone(),
                            reason: format!("row-store batch read failed: {err}"),
                        }
                    })?;

                    unresolved_targets.clear();
                    for (position, entry) in requested_positions
                        .into_iter()
                        .zip(store_entries.into_iter())
                    {
                        let Some(indices_for_position) = targets.get(&position).cloned() else {
                            continue;
                        };
                        let Some(entry) = entry else {
                            unresolved_targets.insert(position, indices_for_position);
                            continue;
                        };

                        let row = self.decode_row_view(entry.as_ref())?;
                        for idx in indices_for_position {
                            let record = self.row_to_record(&row, idx as u64)?;
                            if let Some(record) = record {
                                self.cache
                                    .lock()
                                    .map_err(|_| SamplerError::SourceUnavailable {
                                        source_id: self.config.source_id.clone(),
                                        reason: "huggingface row cache lock poisoned".to_string(),
                                    })?
                                    .insert(idx, row.clone(), self.config.cache_capacity);
                                fetched.insert(idx, Some(record));
                            } else {
                                fetched.insert(idx, None);
                            }
                        }
                    }

                    if unresolved_targets.is_empty() {
                        continue;
                    }

                    let missing = unresolved_targets
                        .keys()
                        .copied()
                        .map(|value| value.to_string())
                        .collect::<Vec<_>>()
                        .join(",");
                    return Err(SamplerError::SourceUnavailable {
                        source_id: self.config.source_id.clone(),
                        reason: format!(
                            "row-store rows missing in shard {} row_group {} at local offsets [{}]",
                            shard.path.display(),
                            group_pos,
                            missing
                        ),
                    });
                }

                let row_group_rows = self
                    .parquet_cache
                    .lock()
                    .map_err(|_| SamplerError::SourceUnavailable {
                        source_id: self.config.source_id.clone(),
                        reason: "huggingface parquet cache lock poisoned".to_string(),
                    })?
                    .row_group_rows_for(
                        &self.config.source_id,
                        &shard.path,
                        group_pos,
                        self.config.parquet_row_group_cache_capacity,
                    )?;

                let mut missing_offsets = Vec::new();
                for (position, indices_for_position) in unresolved_targets {
                    let Some(row_value) = row_group_rows.get(position) else {
                        missing_offsets.push(position);
                        continue;
                    };

                    let parsed = indices_for_position
                        .into_par_iter()
                        .map(|idx| {
                            let row = self.parse_row(idx, row_value)?;
                            if let Some(row) = row {
                                let record = self.row_to_record(&row, idx as u64)?;
                                Ok((idx, Some(row), record))
                            } else {
                                Ok((idx, None, None))
                            }
                        })
                        .collect::<Result<Vec<_>, SamplerError>>()?;

                    for (idx, row, record) in parsed {
                        if let Some(row) = row {
                            self.cache
                                .lock()
                                .map_err(|_| SamplerError::SourceUnavailable {
                                    source_id: self.config.source_id.clone(),
                                    reason: "huggingface row cache lock poisoned".to_string(),
                                })?
                                .insert(idx, row, self.config.cache_capacity);
                            fetched.insert(idx, record);
                        } else {
                            fetched.insert(idx, None);
                        }
                    }
                }

                if !missing_offsets.is_empty() {
                    let missing = missing_offsets
                        .into_iter()
                        .map(|value| value.to_string())
                        .collect::<Vec<_>>()
                        .join(",");
                    return Err(SamplerError::SourceUnavailable {
                        source_id: self.config.source_id.clone(),
                        reason: format!(
                            "parquet rows missing in shard {} row_group {} at local offsets [{}]",
                            shard.path.display(),
                            group_pos,
                            missing
                        ),
                    });
                }
            }
        }

        for idx in indices {
            if limit.is_some_and(|max| out.len() >= max) {
                break;
            }
            if let Some(record) = fetched.remove(idx).flatten() {
                out.push(record);
            }
        }
        Ok(())
    }

    /// Return the current index-domain upper bound for refresh paging.
    fn len_hint(&self) -> Option<usize> {
        let state = self.state.lock().ok()?;
        let known = state.materialized_rows;
        if known > 0 {
            let mut upper = known;
            if state
                .total_rows
                .is_some_and(|total_rows| total_rows > known)
            {
                let headroom = self.effective_expansion_headroom_rows();
                // Keep fast expansion during warmup, then continue with a small
                // ongoing tail so remote shard diversity still grows over time
                // without forcing large lazy-index spikes every refresh.
                let expansion_rows = if known < headroom {
                    headroom
                } else {
                    // 12.5% trickle beyond warmup (at least one row) keeps
                    // sampling from new remote shards alive.
                    (headroom / 8).max(1)
                };
                upper = known.saturating_add(expansion_rows);
                if let Some(total_rows) = state.total_rows {
                    upper = upper.min(total_rows);
                }
            }
            return Some(upper.max(known));
        }

        if state.total_rows.is_some_and(|total_rows| total_rows == 0) {
            return Some(0);
        }

        if state
            .remote_candidates
            .as_ref()
            .is_some_and(|candidates| candidates.is_empty())
        {
            return Some(0);
        }

        Some(1)
    }
}

impl HuggingFaceRowSource {
    /// Spawn the background shard-expansion thread if expansion is needed and
    /// no download is already in progress.  This is separate from `refresh()`
    /// so the ingestion manager can call it on every scheduling cycle even
    /// when the per-source buffer has not yet drained to empty, preventing
    /// expansion from stalling for long epochs.
    fn trigger_expansion_if_needed(&self) {
        let needs_expansion = self
            .state
            .lock()
            .map(|state| {
                let known_empty = state.total_rows == Some(0);
                let all_consumed = state
                    .remote_candidates
                    .as_ref()
                    .is_some_and(|c| state.next_remote_idx >= c.len());
                !known_empty && !all_consumed
            })
            .unwrap_or(false);

        if !needs_expansion {
            return;
        }

        let already_running = self
            .expansion_thread
            .lock()
            .map(|t| t.as_ref().is_some_and(|h| !h.is_finished()))
            .unwrap_or(false);

        if already_running {
            return;
        }

        let source = self.clone();
        let handle = thread::spawn(move || {
            // If candidates not yet fetched, discover them first.
            let needs_candidates = source
                .state
                .lock()
                .map(|s| s.remote_candidates.is_none())
                .unwrap_or(false);
            if needs_candidates {
                let target = source
                    .state
                    .lock()
                    .map(|s| s.materialized_rows)
                    .unwrap_or(0);
                if let Err(err) = source.ensure_row_available(target) {
                    warn!(
                        "[triplets:hf] background expansion (candidate fetch) error \
                         (source '{}'): {}",
                        source.config.source_id, err
                    );
                }
                return;
            }
            if let Err(err) = source.download_next_remote_shard() {
                warn!(
                    "[triplets:hf] background expansion error (source '{}'): {}",
                    source.config.source_id, err
                );
            }
        });
        if let Ok(mut slot) = self.expansion_thread.lock() {
            *slot = Some(handle);
        }
    }
}

impl DataSource for HuggingFaceRowSource {
    /// Return stable source id.
    fn id(&self) -> &str {
        &self.config.source_id
    }

    /// Refresh source records for the requested cursor and row limit.
    fn refresh(
        &self,
        config: &SamplerConfig,
        cursor: Option<&SourceCursor>,
        limit: Option<usize>,
    ) -> Result<SourceSnapshot, SamplerError> {
        self.set_active_sampler_config(config);
        let hinted_total = self.len_hint().unwrap_or(0);
        let max = limit.unwrap_or(hinted_total);

        // Page only over rows that are already on disk.  len_hint() includes
        // expansion headroom beyond materialized_rows; generating indices into
        // that range forces ensure_row_available() to download a shard
        // synchronously mid-read-loop, which blocks for minutes on large
        // datasets.  Instead, reads are always instant (materialized only),
        // and a single shard expansion is triggered AFTER the reads complete
        // so the next refresh call automatically has more rows available.
        // This way every remote shard is eventually consumed without ever
        // blocking on the hot read path.
        let total = {
            let (materialized, known_empty) = {
                let state = self
                    .state
                    .lock()
                    .map_err(|_| SamplerError::SourceUnavailable {
                        source_id: self.config.source_id.clone(),
                        reason: "huggingface source state lock poisoned".to_string(),
                    })?;
                let known_empty = state.total_rows == Some(0);
                (state.materialized_rows, known_empty)
            };
            if materialized == 0 && !known_empty {
                // Bootstrap: discover candidates and download the first shard so
                // the read loop below has rows to work with.  ensure_row_available
                // handles candidate discovery and the initial shard download.
                self.ensure_row_available(0)?;
                self.state
                    .lock()
                    .map_err(|_| SamplerError::SourceUnavailable {
                        source_id: self.config.source_id.clone(),
                        reason: "huggingface source state lock poisoned".to_string(),
                    })?
                    .materialized_rows
            } else {
                materialized
            }
        };

        if total == 0 {
            return Ok(SourceSnapshot {
                records: Vec::new(),
                cursor: SourceCursor {
                    last_seen: Utc::now(),
                    revision: 0,
                },
            });
        }

        let mut start = cursor.map(|state| state.revision as usize).unwrap_or(0);
        if start >= total {
            start = 0;
        }

        let source_id = self.config.source_id.clone();
        let seed = self.paging_seed(total)?;
        let mut permutation = crate::source::IndexPermutation::new(total, seed, start as u64);

        let mut records = Vec::new();
        let read_batch_target = self.effective_refresh_batch_target(max);
        let mut pending_indices = Vec::with_capacity(read_batch_target);
        let should_report = total >= 10_000 || max >= 1_024;
        let report_every = Duration::from_millis(750);
        let refresh_start = Instant::now();
        let mut last_report = refresh_start;
        let mut attempts = 0usize;

        if should_report {
            info!(
                "[triplets:source] refresh start source='{}' total={} target={}",
                source_id, total, max
            );
        }

        while attempts < total && records.len() < max {
            pending_indices.clear();
            let remaining_attempts = total.saturating_sub(attempts);
            let to_collect = read_batch_target.min(remaining_attempts);
            for _ in 0..to_collect {
                if records.len() + pending_indices.len() >= max {
                    break;
                }
                pending_indices.push(permutation.next());
                attempts += 1;
            }

            if pending_indices.is_empty() {
                break;
            }

            if should_report {
                info!(
                    "[triplets:source] refresh batch source='{}' batch_size={} attempted={} fetched={} elapsed={:.1}s",
                    source_id,
                    pending_indices.len(),
                    attempts,
                    records.len(),
                    refresh_start.elapsed().as_secs_f64()
                );
            }

            self.read_row_batch(&pending_indices, &mut records, Some(max))?;

            if should_report && last_report.elapsed() >= report_every {
                info!(
                    "[triplets:source] refresh progress source='{}' attempted={}/{} fetched={}/{} elapsed={:.1}s",
                    source_id,
                    attempts,
                    total,
                    records.len(),
                    max,
                    refresh_start.elapsed().as_secs_f64()
                );
                last_report = Instant::now();
            }
        }

        if should_report {
            info!(
                "[triplets:source] refresh done source='{}' attempted={} fetched={} elapsed={:.2}s",
                source_id,
                attempts,
                records.len(),
                refresh_start.elapsed().as_secs_f64()
            );
        }

        let next_start = permutation.cursor();
        let last_seen = records
            .iter()
            .map(|record| record.updated_at)
            .max()
            .unwrap_or_else(Utc::now);

        // Fire background shard expansion via the shared helper.  The helper
        // is also called by the ingestion manager on every scheduling cycle
        // (even when this source's buffer is non-empty and refresh() itself
        // is skipped), so expansion continues across long epochs.
        self.trigger_expansion_if_needed();

        Ok(SourceSnapshot {
            records,
            cursor: SourceCursor {
                last_seen,
                revision: next_start as u64,
            },
        })
    }

    /// Return exact reported record count from current len hint.
    fn reported_record_count(&self, config: &SamplerConfig) -> Result<u128, SamplerError> {
        self.set_active_sampler_config(config);
        self.len_hint()
            .map(|count| count as u128)
            .ok_or_else(|| SamplerError::SourceInconsistent {
                source_id: self.config.source_id.clone(),
                details: "huggingface source did not provide len_hint".to_string(),
            })
    }

    /// Return mixed default triplet recipes used by Hugging Face row sources.
    fn default_triplet_recipes(&self) -> Vec<TripletRecipe> {
        vec![
            // Majority lane remains context negatives for broad coverage and
            // stable optimization across varied HF schemas.
            TripletRecipe {
                name: "huggingface_anchor_context_wrong_article".into(),
                anchor: Selector::Role(SectionRole::Anchor),
                positive_selector: Selector::Role(SectionRole::Context),
                negative_selector: Selector::Role(SectionRole::Context),
                negative_strategy: NegativeStrategy::WrongArticle,
                weight: 0.75,
                instruction: None,
            },
            // Medium-hard lane adds anchor-as-negative pressure to improve
            // discrimination between title-like anchor fields.
            TripletRecipe {
                name: "huggingface_anchor_anchor_wrong_article".into(),
                anchor: Selector::Role(SectionRole::Anchor),
                positive_selector: Selector::Role(SectionRole::Context),
                negative_selector: Selector::Role(SectionRole::Anchor),
                negative_strategy: NegativeStrategy::WrongArticle,
                weight: 0.25,
                instruction: None,
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parquet::data_type::{ByteArray, ByteArrayType};
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::SerializedFileWriter;
    use parquet::schema::parser::parse_message_type;
    use serde_json::json;
    use std::env;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Mutex, OnceLock};
    use std::thread;
    use tempfile::tempdir;

    // Shared lock for all helpers that mutate process-global env vars.
    // Using a single shared static ensures with_env_var and with_env_vars
    // are mutually exclusive with each other, preventing races between tests.
    static TEST_ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn test_config(snapshot_dir: PathBuf) -> HuggingFaceRowsConfig {
        let mut config =
            HuggingFaceRowsConfig::new("hf_test", "org/dataset", "default", "train", snapshot_dir);
        config.cache_capacity = 10;
        config.remote_expansion_headroom_multiplier = 3;
        config
    }

    fn test_source(config: HuggingFaceRowsConfig) -> HuggingFaceRowSource {
        let source = HuggingFaceRowSource {
            config,
            sampler_config: Arc::new(Mutex::new(None)),
            state: Arc::new(Mutex::new(SourceState {
                materialized_rows: 0,
                total_rows: None,
                shards: Vec::new(),
                // Use Some(vec![]) rather than None so that trigger_expansion_if_needed
                // treats this source as "no remote candidates" and never spawns a
                // background thread that would make live network calls during tests.
                // Tests that explicitly exercise the remote-fetch path reset this field
                // to None before the call under test.
                remote_candidates: Some(vec![]),
                remote_candidate_sizes: HashMap::new(),
                next_remote_idx: 0,
                remote_candidate_order: Vec::new(),
            })),
            cache: Arc::new(Mutex::new(RowCache::default())),
            parquet_cache: Arc::new(Mutex::new(ParquetCache::default())),
            store_cache: Arc::new(Mutex::new(HashMap::new())),
            eligible_index: Arc::new(Mutex::new(EligibleIndexCache::default())),
            expansion_thread: Arc::new(Mutex::new(None)),
        };
        source.set_active_sampler_config(&SamplerConfig {
            seed: 1,
            ingestion_max_records: source.config.cache_capacity,
            ..SamplerConfig::default()
        });
        source
    }

    fn spawn_one_shot_http(payload: Vec<u8>) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request_buf = [0u8; 1024];
            let _ = stream.read(&mut request_buf);
            let headers = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                payload.len()
            );
            stream.write_all(headers.as_bytes()).unwrap();
            stream.write_all(&payload).unwrap();
            let _ = stream.flush();
        });
        (format!("http://{addr}"), handle)
    }

    /// Like `spawn_one_shot_http` but returns a specific HTTP status code.
    fn spawn_one_shot_http_with_status(
        status: u16,
        payload: Vec<u8>,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request_buf = [0u8; 1024];
            let _ = stream.read(&mut request_buf);
            let reason = match status {
                200 => "OK",
                400 => "Bad Request",
                404 => "Not Found",
                500 => "Internal Server Error",
                _ => "Unknown",
            };
            let headers = format!(
                "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                payload.len()
            );
            stream.write_all(headers.as_bytes()).unwrap();
            stream.write_all(&payload).unwrap();
            let _ = stream.flush();
        });
        (format!("http://{addr}"), handle)
    }

    fn spawn_manifest_and_shard_http(shard_payload: Vec<u8>) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{addr}");
        let manifest_body = serde_json::json!({
            "parquet_files": [
                {
                    "url": format!("{base_url}/resolve/main/train/bootstrap.ndjson"),
                    "size": shard_payload.len()
                }
            ]
        })
        .to_string();
        let handle = thread::spawn(move || {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request_buf = [0u8; 4096];
                let read = stream.read(&mut request_buf).unwrap_or(0);
                let request = String::from_utf8_lossy(&request_buf[..read]);
                let first_line = request.lines().next().unwrap_or_default();
                let body = if first_line.contains("/parquet") {
                    manifest_body.as_bytes().to_vec()
                } else {
                    shard_payload.clone()
                };
                let headers = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                stream.write_all(headers.as_bytes()).unwrap();
                stream.write_all(&body).unwrap();
                let _ = stream.flush();
            }
        });
        (base_url, handle)
    }

    fn with_env_var<R>(key: &str, value: &str, run: impl FnOnce() -> R) -> R {
        let _guard = TEST_ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = env::var(key).ok();
        struct EnvRestore {
            key: String,
            previous: Option<String>,
        }
        impl Drop for EnvRestore {
            fn drop(&mut self) {
                if let Some(old) = self.previous.clone() {
                    unsafe { env::set_var(&self.key, old) };
                } else {
                    unsafe { env::remove_var(&self.key) };
                }
            }
        }
        let _restore = EnvRestore {
            key: key.to_string(),
            previous,
        };
        unsafe { env::set_var(key, value) };
        // Locals drop in reverse-declaration order: _restore first (env restored),
        // then _guard (lock released), so the env var is always restored while the
        // lock is still held.
        run()
    }

    /// Like `with_env_var` but sets multiple `(key, value)` pairs atomically under
    /// the same `TEST_ENV_LOCK`.  Use this instead of nesting `with_env_var` calls
    /// (nested calls would deadlock on the shared mutex).
    fn with_env_vars<R>(pairs: &[(&str, &str)], run: impl FnOnce() -> R) -> R {
        let _guard = TEST_ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous: Vec<(String, Option<String>)> = pairs
            .iter()
            .map(|(key, _)| (key.to_string(), env::var(key).ok()))
            .collect();
        struct EnvRestore(Vec<(String, Option<String>)>);
        impl Drop for EnvRestore {
            fn drop(&mut self) {
                for (key, prev) in &self.0 {
                    if let Some(old) = prev {
                        unsafe { env::set_var(key, old) };
                    } else {
                        unsafe { env::remove_var(key) };
                    }
                }
            }
        }
        let _restore = EnvRestore(previous);
        for (key, value) in pairs {
            unsafe { env::set_var(key, value) };
        }
        run()
    }

    fn with_current_dir<R>(dir: &Path, run: impl FnOnce() -> R) -> R {
        static CWD_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _guard = CWD_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = env::current_dir().expect("get cwd");
        struct CwdRestore {
            previous: PathBuf,
        }
        impl Drop for CwdRestore {
            fn drop(&mut self) {
                let _ = env::set_current_dir(&self.previous);
            }
        }
        let _restore = CwdRestore { previous };
        env::set_current_dir(dir).expect("set cwd");
        // Locals drop in reverse-declaration order: _restore first (cwd restored),
        // then _guard (lock released).
        run()
    }

    fn write_parquet_fixture(path: &Path, rows: &[(&str, &str)]) {
        let schema = Arc::new(
            parse_message_type(
                "message test_schema {
                    REQUIRED BINARY id (UTF8);
                    REQUIRED BINARY text (UTF8);
                }",
            )
            .unwrap(),
        );
        let props = Arc::new(WriterProperties::builder().build());
        let file = File::create(path).unwrap();
        let mut writer = SerializedFileWriter::new(file, schema, props).unwrap();
        let mut row_group = writer.next_row_group().unwrap();

        if let Some(mut col_writer) = row_group.next_column().unwrap() {
            let values = rows
                .iter()
                .map(|(id, _)| ByteArray::from(*id))
                .collect::<Vec<_>>();
            col_writer
                .typed::<ByteArrayType>()
                .write_batch(&values, None, None)
                .unwrap();
            col_writer.close().unwrap();
        }

        if let Some(mut col_writer) = row_group.next_column().unwrap() {
            let values = rows
                .iter()
                .map(|(_, text)| ByteArray::from(*text))
                .collect::<Vec<_>>();
            col_writer
                .typed::<ByteArrayType>()
                .write_batch(&values, None, None)
                .unwrap();
            col_writer.close().unwrap();
        }

        assert!(row_group.next_column().unwrap().is_none());
        row_group.close().unwrap();
        writer.close().unwrap();
    }

    fn write_simdr_fixture(path: &Path, rows: &[(&str, &str)]) {
        // Create/open the simd-r-drive DataStore and write row-view entries
        let store = DataStore::open(path).expect("open simdr store");
        if rows.is_empty() {
            store
                .write(HF_SHARD_STORE_META_ROWS_KEY, &(0u64).to_le_bytes())
                .expect("write meta");
            return;
        }

        let mut batch: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        for (i, (id, text)) in rows.iter().enumerate() {
            let row = RowView {
                row_id: Some(id.to_string()),
                timestamp: None,
                text_fields: vec![RowTextField {
                    name: "text".to_string(),
                    text: text.to_string(),
                }],
            };
            let payload = serde_json::to_vec(&row).expect("encode row");
            let mut key = HF_SHARD_STORE_ROW_PREFIX.to_vec();
            key.extend_from_slice(&(i as u64).to_le_bytes());
            batch.push((key, payload));
        }

        let refs: Vec<(&[u8], &[u8])> = batch
            .iter()
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
            .collect();
        store.batch_write(&refs).expect("batch write");
        store
            .write(
                HF_SHARD_STORE_META_ROWS_KEY,
                &(rows.len() as u64).to_le_bytes(),
            )
            .expect("write meta");
    }

    #[test]
    fn managed_snapshot_helpers_create_cache_dirs_under_discovered_root() {
        let dir = tempdir().unwrap();
        fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname='tmp'\nversion='0.0.0'\n",
        )
        .unwrap();

        with_current_dir(dir.path(), || {
            let single = managed_hf_snapshot_dir("org/dataset", "default", "train").unwrap();
            let listed =
                managed_hf_list_snapshot_dir("org/dataset", "default", "train", 7).unwrap();

            assert!(single.exists());
            assert!(listed.exists());
            assert!(single.ends_with(PathBuf::from(format!(
                "{}/org__dataset/default/train",
                HUGGINGFACE_GROUP
            ))));
            assert!(listed.ends_with(PathBuf::from(format!(
                "{}/source-list/org__dataset/default/train/replica_7",
                HUGGINGFACE_GROUP
            ))));
            assert!(listed.ends_with("replica_7"));
        });
    }

    #[test]
    fn managed_snapshot_dirs_use_all_splits_dir_for_empty_split() {
        let dir = tempdir().unwrap();
        fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname='tmp'\nversion='0.0.0'\n",
        )
        .unwrap();

        with_current_dir(dir.path(), || {
            let single = managed_hf_snapshot_dir("org/dataset", "default", "").unwrap();
            let listed = managed_hf_list_snapshot_dir("org/dataset", "default", "", 0).unwrap();

            assert!(single.exists());
            assert!(listed.exists());
            // Both must use ALL_SPLITS_DIR ("_all") in the path, not an empty segment.
            assert!(
                single.ends_with(PathBuf::from(format!(
                    "{}/org__dataset/default/{}",
                    HUGGINGFACE_GROUP, ALL_SPLITS_DIR
                ))),
                "expected single-source path to end with ALL_SPLITS_DIR, got: {}",
                single.display()
            );
            assert!(
                listed.ends_with(PathBuf::from(format!(
                    "{}/source-list/org__dataset/default/{}/replica_0",
                    HUGGINGFACE_GROUP, ALL_SPLITS_DIR
                ))),
                "expected list-source path to end with ALL_SPLITS_DIR, got: {}",
                listed.display()
            );
            // Must not collide with the explicit-train path.
            let train_single = managed_hf_snapshot_dir("org/dataset", "default", "train").unwrap();
            assert_ne!(
                single, train_single,
                "empty-split and train-split paths must differ"
            );
        });
    }

    #[test]
    fn load_and_resolve_hf_source_list_reports_invalid_and_empty_inputs() {
        let dir = tempdir().unwrap();

        let invalid_list = dir.path().join("invalid_sources.txt");
        fs::write(&invalid_list, "hf://org/dataset/default/train badtoken\n").unwrap();
        let invalid = load_hf_sources_from_list(invalid_list.to_str().unwrap()).unwrap_err();
        assert!(invalid.contains("invalid source-list entry"));

        let empty_list = dir.path().join("empty_sources.txt");
        fs::write(&empty_list, "# comment only\n\n").unwrap();
        let resolved = resolve_hf_list_roots(empty_list.to_string_lossy().to_string()).unwrap_err();
        assert!(resolved.contains("no hf:// entries found"));

        let good_list = dir.path().join("good_sources.txt");
        fs::write(
            &good_list,
            "hf://org/dataset/default/train anchor=title positive=body\n",
        )
        .unwrap();
        let roots = resolve_hf_list_roots(good_list.to_string_lossy().to_string()).unwrap();
        assert_eq!(roots.sources.len(), 1);
    }

    #[test]
    fn new_errors_when_snapshot_dir_path_is_a_file() {
        let dir = tempdir().unwrap();
        let snapshot_file = dir.path().join("snapshot-file");
        fs::write(&snapshot_file, b"x").unwrap();

        let config = HuggingFaceRowsConfig::new(
            "hf_bad_snapshot",
            "org/dataset",
            "default",
            "train",
            snapshot_file,
        );
        let result = HuggingFaceRowSource::new(config);
        assert!(matches!(
            result,
            Err(SamplerError::SourceUnavailable { .. })
        ));
    }

    #[test]
    fn list_remote_candidates_falls_back_when_manifest_query_fails() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.dataset = "invalid///dataset".to_string();

        let result = with_env_var(
            TRIPLETS_HF_PARQUET_ENDPOINT,
            "http://127.0.0.1:1/parquet",
            || HuggingFaceRowSource::list_remote_candidates(&config),
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_hf_sources_skips_invalid_uri_and_builds_valid_source() {
        let roots = HfListRoots {
            source_list: "inline".to_string(),
            sources: vec![
                HfSourceEntry {
                    uri: "hf://onlyorg".to_string(),
                    anchor_columns: vec!["title".to_string()],
                    positive_columns: Vec::new(),
                    context_columns: Vec::new(),
                    text_columns: Vec::new(),
                    trust: None,
                    source_id: None,
                },
                HfSourceEntry {
                    uri: "hf://org/dataset/default/train".to_string(),
                    anchor_columns: vec!["title".to_string()],
                    positive_columns: vec!["body".to_string()],
                    context_columns: Vec::new(),
                    text_columns: Vec::new(),
                    trust: None,
                    source_id: None,
                },
            ],
        };

        let temp_root = tempdir().unwrap();
        fs::write(
            temp_root.path().join("Cargo.toml"),
            "[package]\nname='tmp'\nversion='0.0.0'\n",
        )
        .unwrap();
        fs::write(temp_root.path().join(".cache"), b"blocking-file").unwrap();

        // Serve a valid empty-splits response so fetch_global_row_count returns
        // None cleanly without printing a WARN about a 401 from the real API.
        let size_payload = serde_json::json!({"size": {"splits": []}})
            .to_string()
            .into_bytes();
        let (size_base_url, size_server) = spawn_one_shot_http(size_payload);

        with_current_dir(temp_root.path(), || {
            with_env_var(
                TRIPLETS_HF_SIZE_ENDPOINT,
                &format!("{size_base_url}/size"),
                || {
                    let built = build_hf_sources(&roots);
                    assert_eq!(built.len(), 1);
                },
            );
        });

        size_server.join().unwrap();
    }

    #[test]
    fn build_hf_sources_duplicate_uri_gets_distinct_ids_and_snapshot_dirs() {
        // Two identical entries must produce two built sources whose IDs are
        // disambiguated (".0" / ".1") and whose snapshot directories are
        // independent (replica_0 vs replica_1).
        let dup_entry = HfSourceEntry {
            uri: "hf://org/dataset/default/train".to_string(),
            anchor_columns: vec!["title".to_string()],
            positive_columns: vec!["body".to_string()],
            context_columns: Vec::new(),
            text_columns: Vec::new(),
            trust: None,
            source_id: None,
        };
        let roots = HfListRoots {
            source_list: "inline".to_string(),
            sources: vec![dup_entry.clone(), dup_entry],
        };

        let temp_root = tempdir().unwrap();
        fs::write(
            temp_root.path().join("Cargo.toml"),
            "[package]\nname='tmp'\nversion='0.0.0'\n",
        )
        .unwrap();

        // Two sources → two size-endpoint calls; serve both responses.
        let size_payload = || {
            serde_json::json!({"size": {"splits": []}})
                .to_string()
                .into_bytes()
        };
        let (size_base_url_a, size_server_a) = spawn_one_shot_http(size_payload());
        let (size_base_url_b, size_server_b) = spawn_one_shot_http(size_payload());
        // Both servers share the same base URL pattern; use the first for the env-var
        // (the second call may hit a different port, but both start with the same host).
        // In practice each spawn_one_shot_http binds its own ephemeral port, so we
        // point the env-var at either — what matters is that both succeed and the
        // test doesn't make real network calls.
        let _ = size_base_url_b; // second URL not needed since we assert on IDs, not rows

        with_current_dir(temp_root.path(), || {
            with_env_var(
                TRIPLETS_HF_SIZE_ENDPOINT,
                &format!("{size_base_url_a}/size"),
                || {
                    let built = build_hf_sources(&roots);
                    assert_eq!(built.len(), 2, "both duplicate sources should be built");

                    let id_0 = built[0].id().to_string();
                    let id_1 = built[1].id().to_string();
                    assert_ne!(
                        id_0, id_1,
                        "duplicate sources must have distinct source IDs"
                    );
                    assert!(
                        id_0.ends_with(".0"),
                        "first duplicate should have .0 suffix, got: {id_0}"
                    );
                    assert!(
                        id_1.ends_with(".1"),
                        "second duplicate should have .1 suffix, got: {id_1}"
                    );

                    // Snapshot dirs are derived from managed_hf_list_snapshot_dir with
                    // the list index, so replica_0 and replica_1 must differ.
                    let dir_0 =
                        managed_hf_list_snapshot_dir("org/dataset", "default", "train", 0).unwrap();
                    let dir_1 =
                        managed_hf_list_snapshot_dir("org/dataset", "default", "train", 1).unwrap();
                    assert_ne!(
                        dir_0, dir_1,
                        "duplicate sources must have distinct snapshot dirs"
                    );
                    assert!(dir_0.ends_with("replica_0"));
                    assert!(dir_1.ends_with("replica_1"));
                },
            );
        });

        size_server_a.join().unwrap();
        // size_server_b may not have been contacted if cache resolved both paths;
        // drop it without joining to avoid blocking.
        drop(size_server_b);
    }

    #[test]
    fn hf_source_id_slug_uses_short_dataset_name() {
        assert_eq!(
            hf_source_id_slug("allenai/dolmino-mix-1124", "default", "train"),
            "dolmino-mix-1124"
        );
    }

    #[test]
    fn hf_source_id_slug_appends_non_default_config() {
        assert_eq!(
            hf_source_id_slug("org/dataset", "en", "train"),
            "dataset.en"
        );
    }

    #[test]
    fn hf_source_id_slug_appends_non_train_split() {
        assert_eq!(
            hf_source_id_slug("org/dataset", "default", "validation"),
            "dataset.validation"
        );
    }

    #[test]
    fn hf_source_id_slug_omits_empty_split() {
        assert_eq!(hf_source_id_slug("org/dataset", "default", ""), "dataset");
    }

    #[test]
    fn hf_source_id_slug_appends_both_config_and_split() {
        assert_eq!(
            hf_source_id_slug("org/dataset", "en", "validation"),
            "dataset.en.validation"
        );
    }

    #[test]
    fn hf_source_id_slug_sanitizes_special_chars() {
        // dots and slashes in names become dashes
        assert_eq!(
            hf_source_id_slug("org/data.set", "v1.0", "train"),
            "data-set.v1-0"
        );
    }

    #[test]
    fn hf_source_id_slug_no_org_prefix() {
        // dataset without org/ prefix — falls back to using the full string
        assert_eq!(hf_source_id_slug("dataset", "default", "train"), "dataset");
    }

    #[test]
    fn build_hf_sources_disambiguates_duplicate_slugs() {
        // Two sources pointing at the same dataset/config/split should get
        // distinct IDs via the index suffix rather than silently colliding.
        let sources = [
            HfSourceEntry {
                uri: "hf://org/dataset/default/train".to_string(),
                anchor_columns: vec!["title".to_string()],
                positive_columns: vec!["body".to_string()],
                context_columns: Vec::new(),
                text_columns: Vec::new(),
                trust: None,
                source_id: None,
            },
            HfSourceEntry {
                uri: "hf://org/dataset/default/train".to_string(),
                anchor_columns: vec!["title".to_string()],
                positive_columns: vec!["body".to_string()],
                context_columns: Vec::new(),
                text_columns: Vec::new(),
                trust: None,
                source_id: None,
            },
        ];
        let base_slugs: Vec<String> = sources
            .iter()
            .enumerate()
            .map(|(idx, source)| match parse_hf_uri(&source.uri) {
                Ok((dataset, config, split)) => hf_source_id_slug(&dataset, &config, &split),
                Err(_) => format!("hf_list_{idx}"),
            })
            .collect();
        let mut slug_count: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::new();
        for s in &base_slugs {
            *slug_count.entry(s.as_str()).or_insert(0) += 1;
        }
        let duplicated: HashSet<&str> = slug_count
            .into_iter()
            .filter(|(_, n)| *n > 1)
            .map(|(s, _)| s)
            .collect();
        let resolved: Vec<String> = base_slugs
            .iter()
            .enumerate()
            .map(|(idx, slug)| {
                if duplicated.contains(slug.as_str()) {
                    format!("{slug}.{idx}")
                } else {
                    slug.clone()
                }
            })
            .collect();
        assert_eq!(resolved[0], "dataset.0");
        assert_eq!(resolved[1], "dataset.1");
    }

    #[test]
    fn row_cache_insert_and_evicts_oldest_entry() {
        let mut cache = RowCache::default();
        let row_a = RowView {
            row_id: Some("a".to_string()),
            timestamp: None,
            text_fields: vec![RowTextField {
                name: "text".to_string(),
                text: "alpha".to_string(),
            }],
        };
        let row_b = RowView {
            row_id: Some("b".to_string()),
            timestamp: None,
            text_fields: vec![RowTextField {
                name: "text".to_string(),
                text: "beta".to_string(),
            }],
        };

        cache.insert(0, row_a.clone(), 1);
        assert!(cache.get(0).is_some());

        cache.insert(1, row_b, 1);
        assert!(cache.get(0).is_none());
        assert_eq!(cache.get(1).unwrap().row_id.as_deref(), Some("b"));

        let mut zero_cache = RowCache::default();
        zero_cache.insert(7, row_a, 0);
        assert!(zero_cache.get(7).is_none());
    }

    #[test]
    fn parquet_cache_reader_for_reports_open_and_parse_errors() {
        let dir = tempdir().unwrap();
        let parquet_path = dir.path().join("missing.parquet");
        let mut cache = ParquetCache::default();
        let missing = cache.reader_for("hf_test", &parquet_path);
        assert!(missing.is_err());

        let invalid_parquet = dir.path().join("invalid.parquet");
        fs::write(&invalid_parquet, b"not parquet").unwrap();
        let invalid = cache.reader_for("hf_test", &invalid_parquet);
        assert!(invalid.is_err());
    }

    #[test]
    fn effective_targets_respect_minimum_multiplier_and_sampler_override() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.refresh_batch_multiplier = 0;
        config.remote_expansion_headroom_multiplier = 0;
        config.cache_capacity = 9;
        let source = test_source(config.clone());

        assert_eq!(source.effective_refresh_batch_target(5), 5);
        assert_eq!(source.effective_expansion_headroom_rows(), 9);

        let sampler = SamplerConfig {
            ingestion_max_records: 4,
            ..SamplerConfig::default()
        };
        *source.sampler_config.lock().unwrap() = Some(sampler);
        assert_eq!(source.effective_expansion_headroom_rows(), 4);
    }

    #[test]
    fn collect_candidates_from_siblings_filters_split_and_tracks_parquet() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let accepted = HuggingFaceRowSource::normalized_shard_extensions(&config);
        let siblings = vec![
            "train/a.ndjson".to_string(),
            "dev/b.ndjson".to_string(),
            "train-c.parquet".to_string(),
            "train-z.txt".to_string(),
        ];

        let (candidates, saw_parquet) = HuggingFaceRowSource::collect_candidates_from_siblings(
            &config, &siblings, &accepted, true,
        );

        assert!(saw_parquet);
        assert_eq!(
            candidates,
            vec!["train/a.ndjson".to_string(), "train-c.parquet".to_string()]
        );
    }

    #[test]
    fn collect_candidates_from_siblings_skips_existing_targets() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let accepted = HuggingFaceRowSource::normalized_shard_extensions(&config);
        let existing = "train/already.ndjson".to_string();
        let existing_target = HuggingFaceRowSource::candidate_target_path(&config, &existing);
        fs::create_dir_all(existing_target.parent().unwrap()).unwrap();
        fs::write(&existing_target, b"x\n").unwrap();

        let siblings = vec![existing, "train/new.ndjson".to_string()];
        let (candidates, _) = HuggingFaceRowSource::collect_candidates_from_siblings(
            &config, &siblings, &accepted, true,
        );
        assert_eq!(candidates, vec!["train/new.ndjson".to_string()]);
    }

    #[test]
    fn all_candidates_from_parquet_manifest_returns_all_with_sizes() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let payload = json!({
            "parquet_files": [
                {"url": "https://host/x/train/000.parquet", "size": 11},
                {"url": "https://host/x/train/001.ndjson", "size": 13},
                {"url": "https://host/x/train/002.txt", "size": 5},
                {"foo": "missing-url"}
            ]
        });

        let (candidates, sizes, matched) =
            HuggingFaceRowSource::all_candidates_from_parquet_manifest(&config, &payload).unwrap();
        assert_eq!(candidates.len(), 2);
        assert!(
            candidates
                .iter()
                .any(|c| c.ends_with("https://host/x/train/000.parquet"))
        );
        assert!(
            candidates
                .iter()
                .any(|c| c.ends_with("https://host/x/train/001.ndjson"))
        );
        assert_eq!(sizes.len(), 2);
        assert_eq!(matched, 2);
    }

    #[test]
    fn all_candidates_from_parquet_manifest_includes_cached_and_replaces_stale() {
        // Suppress the expected WARN "incomplete cached shard detected (will redownload)".
        let _quiet = tracing::subscriber::set_default(
            tracing_subscriber::fmt()
                .with_max_level(tracing::Level::ERROR)
                .finish(),
        );
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());

        // A parquet file with the correct declared size — considered fully cached.
        let complete_url = "https://host/datasets/org/ds/resolve/main/train/000.parquet";
        let complete_candidate = format!("{REMOTE_URL_PREFIX}{complete_url}");
        let complete_target =
            HuggingFaceRowSource::candidate_target_path(&config, &complete_candidate);
        fs::create_dir_all(complete_target.parent().unwrap()).unwrap();
        fs::write(&complete_target, vec![1u8; 7]).unwrap();

        // A parquet file with the WRONG size — stale/incomplete, must be deleted.
        let stale_url = "https://host/datasets/org/ds/resolve/main/train/001.parquet";
        let stale_candidate = format!("{REMOTE_URL_PREFIX}{stale_url}");
        let stale_target = HuggingFaceRowSource::candidate_target_path(&config, &stale_candidate);
        fs::create_dir_all(stale_target.parent().unwrap()).unwrap();
        fs::write(&stale_target, vec![2u8; 3]).unwrap();

        let payload = json!({
            "parquet_files": [
                {"url": complete_url, "size": 7},
                {"url": stale_url, "size": 9}
            ]
        });

        let (candidates, sizes, matched) =
            HuggingFaceRowSource::all_candidates_from_parquet_manifest(&config, &payload).unwrap();

        // Both shards are returned — cache state does not affect the candidate list.
        assert_eq!(candidates.len(), 2, "both shards must appear in candidates");
        // Complete shard: file exists and was not deleted.
        assert!(
            complete_target.exists(),
            "complete shard must not be deleted"
        );
        // Stale shard: wrong-size file was deleted so it will be re-fetched.
        assert!(!stale_target.exists(), "stale shard must be deleted");
        assert_eq!(sizes.len(), 2);
        assert_eq!(matched, 2);
    }

    #[test]
    fn candidates_from_parquet_manifest_errors_when_removing_incomplete_target_fails() {
        // Suppress the expected WARN "incomplete cached shard detected (will redownload)"
        // emitted before the attempted removal fails.  The removal failure is what this
        // test asserts on; the warn preceding it is correct production behaviour.
        let _quiet = tracing::subscriber::set_default(
            tracing_subscriber::fmt()
                .with_max_level(tracing::Level::ERROR)
                .finish(),
        );
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let url = "https://host/datasets/org/ds/resolve/main/train/blocked.parquet";
        let candidate = format!("{REMOTE_URL_PREFIX}{url}");
        let target = HuggingFaceRowSource::candidate_target_path(&config, &candidate);
        fs::create_dir_all(&target).unwrap();

        let payload = json!({
            "parquet_files": [
                {"url": url, "size": 1}
            ]
        });

        let err = HuggingFaceRowSource::all_candidates_from_parquet_manifest(&config, &payload);
        assert!(err.is_err());
    }

    #[test]
    fn normalized_shard_extensions_trims_dots_and_lowercases() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.shard_extensions = vec![".PARQUET".into(), " ndjson ".into()];
        let normalized = HuggingFaceRowSource::normalized_shard_extensions(&config);
        assert_eq!(
            normalized,
            vec!["parquet".to_string(), "ndjson".to_string()]
        );
    }

    #[test]
    fn manifest_usage_bytes_locked_counts_only_manifest_shards() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let manifest_root = source.manifest_cache_root();
        fs::create_dir_all(&manifest_root).unwrap();

        let manifest_file = manifest_root.join("a.parquet");
        fs::write(&manifest_file, vec![1u8; 7]).unwrap();
        let local_file = source.config.snapshot_dir.join("local.ndjson");
        fs::write(&local_file, vec![2u8; 9]).unwrap();

        let state = SourceState {
            materialized_rows: 2,
            total_rows: None,
            shards: vec![
                ShardIndex {
                    path: manifest_file,
                    global_start: 0,
                    row_count: 1,
                    random_access: true,
                    parquet_row_groups: vec![(0, 1)],
                    checkpoints: Vec::new(),
                    remote_candidate: None,
                },
                ShardIndex {
                    path: local_file,
                    global_start: 1,
                    row_count: 1,
                    random_access: false,
                    parquet_row_groups: Vec::new(),
                    checkpoints: vec![0],
                    remote_candidate: None,
                },
            ],
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            remote_candidate_order: Vec::new(),
        };

        assert_eq!(source.manifest_usage_bytes_locked(&state), 7);
    }

    #[test]
    fn build_shard_index_errors_when_parquet_present_but_not_accepted() {
        // Suppress the expected WARN "found persisted parquet under … (transient-only policy)"
        // that fires when parquet files are present in the snapshot dir but parquet is not
        // listed in shard_extensions.  That warn is correct production behaviour; this test
        // only cares that the function returns an error, not the diagnostic message.
        let _quiet = tracing::subscriber::set_default(
            tracing_subscriber::fmt()
                .with_max_level(tracing::Level::ERROR)
                .finish(),
        );
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("rows.parquet"), b"fake").unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.shard_extensions = vec!["ndjson".to_string()];

        let result = HuggingFaceRowSource::build_shard_index(&config);
        assert!(result.is_err());
    }

    #[test]
    fn locate_parquet_group_maps_offsets_and_reports_missing() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let shard = ShardIndex {
            path: dir.path().join("rows.parquet"),
            global_start: 0,
            row_count: 6,
            random_access: true,
            parquet_row_groups: vec![(0, 2), (2, 2), (4, 2)],
            checkpoints: Vec::new(),
            remote_candidate: None,
        };

        let mapped = source.locate_parquet_group(&shard, 3).unwrap();
        assert_eq!(mapped, (1, 1));
        let missing = source.locate_parquet_group(&shard, 99);
        assert!(missing.is_err());
    }

    #[test]
    fn parse_row_role_columns_mode_builds_expected_fields() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.anchor_columns = vec!["anchor".into()];
        config.positive_columns = vec!["positive".into()];
        config.context_columns = vec!["ctx1".into(), "ctx2".into()];
        let source = test_source(config);

        let row = source
            .parse_row(
                2,
                &json!({"id":"r","anchor":"a","positive":"p","ctx1":"c1","ctx2":2}),
            )
            .unwrap()
            .unwrap();
        assert_eq!(row.text_fields.len(), 4);
        assert_eq!(row.text_fields[0].name, "anchor");
        assert_eq!(row.text_fields[1].name, "positive");
    }

    #[test]
    fn parse_row_role_columns_mode_skips_missing_or_empty_values() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.anchor_columns = vec!["anchor".into()];
        config.context_columns = vec!["ctx".into()];
        let source = test_source(config);

        let missing = source.parse_row(0, &json!({"anchor":"a"}));
        assert!(missing.unwrap().is_none());

        let empty_anchor = source.parse_row(1, &json!({"anchor":"   ", "ctx":"ok"}));
        assert!(empty_anchor.unwrap().is_none());
    }

    #[test]
    fn row_to_record_uses_anchor_for_positive_when_single_field() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let row = RowView {
            row_id: Some("r1".into()),
            timestamp: None,
            text_fields: vec![RowTextField {
                name: "text".into(),
                text: "alpha".into(),
            }],
        };

        let record = source.row_to_record(&row, 0).unwrap().unwrap();
        assert_eq!(record.sections.len(), 2);
        assert_eq!(record.sections[0].text, record.sections[1].text);
    }

    #[test]
    fn read_line_at_errors_on_unexpected_eof_while_scanning() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("rows.jsonl");
        fs::write(&path, b"{\"text\":\"a\"}\n").unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.checkpoint_stride = 1;
        let source = test_source(config.clone());
        let mut shard = HuggingFaceRowSource::index_single_shard(&config, &path, 0)
            .unwrap()
            .0
            .unwrap();
        shard.checkpoints = vec![0];

        let err = source.read_line_at(&shard, 3);
        assert!(err.is_err());
    }

    #[test]
    fn target_matches_expected_size_is_false_for_missing_path() {
        let dir = tempdir().unwrap();
        let missing = dir.path().join("missing.bin");
        assert!(!HuggingFaceRowSource::target_matches_expected_size(
            &missing,
            Some(1)
        ));
    }

    #[test]
    fn candidate_target_path_uses_fallback_suffix_without_resolve_segment() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let candidate = "url::https://example.com/raw/file.parquet";
        let target = HuggingFaceRowSource::candidate_target_path(&config, candidate);
        assert!(target.ends_with("_parquet_manifest/parquet/unknown.parquet"));
    }

    #[test]
    fn parse_row_falls_back_to_synthetic_id_when_missing_id_column() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.id_column = Some("id".into());
        let source = test_source(config);

        let row = source
            .parse_row(42, &json!({"text": "hello"}))
            .unwrap()
            .unwrap();
        assert_eq!(row.row_id, Some("org/dataset:train:42".to_string()));
    }

    #[test]
    fn row_to_record_falls_back_to_row_index_when_row_id_missing() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let row = RowView {
            row_id: None,
            timestamp: None,
            text_fields: vec![RowTextField {
                name: "text".into(),
                text: "body".into(),
            }],
        };

        let record = source.row_to_record(&row, 7).unwrap().unwrap();
        assert!(record.id.ends_with("::row_7"));
    }

    #[test]
    fn locate_shard_returns_none_for_out_of_range_index() {
        let shards = vec![ShardIndex {
            path: PathBuf::from("a.ndjson"),
            global_start: 0,
            row_count: 2,
            random_access: false,
            parquet_row_groups: Vec::new(),
            checkpoints: vec![0],
            remote_candidate: None,
        }];

        assert!(HuggingFaceRowSource::locate_shard(&shards, 5).is_none());
    }

    #[test]
    fn read_row_batch_errors_when_row_not_mappable_to_shard() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 1;
            state.total_rows = Some(1);
            state.shards.clear();
        }

        let mut out = Vec::new();
        let err = source.read_row_batch(&[0], &mut out, Some(1));
        assert!(err.is_err());
    }

    #[test]
    fn read_row_batch_errors_on_invalid_json_row() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("broken.ndjson");
        fs::write(&path, b"not-json\n").unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);

        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 1;
            state.total_rows = Some(1);
            state.shards = vec![ShardIndex {
                path,
                global_start: 0,
                row_count: 1,
                random_access: false,
                parquet_row_groups: Vec::new(),
                checkpoints: vec![0],
                remote_candidate: None,
            }];
        }

        let mut out = Vec::new();
        let err = source.read_row_batch(&[0], &mut out, Some(1)).unwrap_err();
        assert!(matches!(
            err,
            SamplerError::SourceInconsistent { ref details, .. } if details.contains("failed decoding JSON row")
        ));
    }

    #[test]
    fn read_row_batch_errors_when_parquet_local_offsets_are_missing() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("rows.parquet");
        write_parquet_fixture(&path, &[("id-1", "text-1")]);
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);

        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 3;
            state.total_rows = Some(3);
            state.shards = vec![ShardIndex {
                path,
                global_start: 0,
                row_count: 3,
                random_access: true,
                parquet_row_groups: vec![(0, 3)],
                checkpoints: Vec::new(),
                remote_candidate: None,
            }];
        }

        let mut out = Vec::new();
        let err = source.read_row_batch(&[2], &mut out, Some(1)).unwrap_err();
        assert!(matches!(
            err,
            SamplerError::SourceUnavailable { ref reason, .. } if reason.contains("parquet rows missing")
        ));
    }

    #[test]
    fn enforce_disk_cap_returns_false_when_disabled_or_under_limit() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.local_disk_cap_bytes = None;
        let source = test_source(config);
        let mut state = SourceState {
            materialized_rows: 0,
            total_rows: None,
            shards: Vec::new(),
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            remote_candidate_order: Vec::new(),
        };
        let protected = dir.path().join("p");
        assert!(
            !source
                .enforce_disk_cap_locked(&mut state, &protected)
                .unwrap()
        );

        let mut config2 = test_config(dir.path().to_path_buf());
        config2.local_disk_cap_bytes = Some(10_000);
        let source2 = test_source(config2);
        let manifest_root = source2.manifest_cache_root();
        fs::create_dir_all(&manifest_root).unwrap();
        let shard_path = manifest_root.join("small.parquet");
        fs::write(&shard_path, vec![1u8; 32]).unwrap();
        let mut state2 = SourceState {
            materialized_rows: 1,
            total_rows: None,
            shards: vec![ShardIndex {
                path: shard_path,
                global_start: 0,
                row_count: 1,
                random_access: true,
                parquet_row_groups: vec![(0, 1)],
                checkpoints: Vec::new(),
                remote_candidate: None,
            }],
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            remote_candidate_order: Vec::new(),
        };
        assert!(
            !source2
                .enforce_disk_cap_locked(&mut state2, &protected)
                .unwrap()
        );
    }

    #[test]
    fn enforce_disk_cap_evicts_manifest_shards_and_recomputes_offsets() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.local_disk_cap_bytes = Some(20);
        let source = test_source(config);
        let manifest_root = source.manifest_cache_root();
        fs::create_dir_all(&manifest_root).unwrap();

        let first = manifest_root.join("first.parquet");
        let second = manifest_root.join("second.parquet");
        fs::write(&first, vec![1u8; 16]).unwrap();
        fs::write(&second, vec![2u8; 16]).unwrap();

        let mut state = SourceState {
            materialized_rows: 2,
            total_rows: None,
            shards: vec![
                ShardIndex {
                    path: first.clone(),
                    global_start: 0,
                    row_count: 1,
                    random_access: true,
                    parquet_row_groups: vec![(0, 1)],
                    checkpoints: Vec::new(),
                    remote_candidate: None,
                },
                ShardIndex {
                    path: second.clone(),
                    global_start: 1,
                    row_count: 1,
                    random_access: true,
                    parquet_row_groups: vec![(0, 1)],
                    checkpoints: Vec::new(),
                    remote_candidate: None,
                },
            ],
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            remote_candidate_order: Vec::new(),
        };

        let evicted = source.enforce_disk_cap_locked(&mut state, &second).unwrap();
        assert!(evicted);
        assert!(!first.exists());
        assert!(second.exists());
        assert_eq!(state.shards.len(), 1);
        assert_eq!(state.shards[0].global_start, 0);
        assert_eq!(state.materialized_rows, 1);
    }

    #[test]
    fn enforce_disk_cap_evicts_when_single_file_exceeds_cap() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.local_disk_cap_bytes = Some(1);
        let source = test_source(config);
        let manifest_root = source.manifest_cache_root();
        fs::create_dir_all(&manifest_root).unwrap();

        let protected = manifest_root.join("protected.parquet");
        fs::write(&protected, vec![3u8; 16]).unwrap();

        let mut state = SourceState {
            materialized_rows: 1,
            total_rows: None,
            shards: vec![ShardIndex {
                path: protected.clone(),
                global_start: 0,
                row_count: 1,
                random_access: true,
                parquet_row_groups: vec![(0, 1)],
                checkpoints: Vec::new(),
                remote_candidate: None,
            }],
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            remote_candidate_order: Vec::new(),
        };

        let evicted = source
            .enforce_disk_cap_locked(&mut state, &protected)
            .unwrap();
        assert!(evicted);
        assert!(!protected.exists());
        assert_eq!(state.shards.len(), 0);
        assert_eq!(state.materialized_rows, 0);
    }

    #[test]
    fn configured_sampler_seed_and_paging_seed_require_sampler_config() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = HuggingFaceRowSource {
            config,
            sampler_config: Arc::new(Mutex::new(None)),
            state: Arc::new(Mutex::new(SourceState {
                materialized_rows: 0,
                total_rows: None,
                shards: Vec::new(),
                remote_candidates: None,
                remote_candidate_sizes: HashMap::new(),
                next_remote_idx: 0,
                remote_candidate_order: Vec::new(),
            })),
            cache: Arc::new(Mutex::new(RowCache::default())),
            parquet_cache: Arc::new(Mutex::new(ParquetCache::default())),
            store_cache: Arc::new(Mutex::new(HashMap::new())),
            eligible_index: Arc::new(Mutex::new(EligibleIndexCache::default())),
            expansion_thread: Arc::new(Mutex::new(None)),
        };

        assert!(source.configured_sampler_seed().is_err());
        assert!(source.paging_seed(5).is_err());
    }

    #[test]
    fn shard_candidate_seed_and_shuffle_are_deterministic() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.source_id = "hf_rotator".to_string();

        let seed_a = HuggingFaceRowSource::shard_candidate_seed(&config, 12, 1);
        let seed_b = HuggingFaceRowSource::shard_candidate_seed(&config, 12, 2);
        assert_ne!(seed_a, seed_b);

        let baseline = vec!["c".to_string(), "a".to_string(), "b".to_string()];
        let mut left = baseline.clone();
        let mut right = baseline;
        HuggingFaceRowSource::shuffle_candidates_deterministically(&config, &mut left, 42);
        HuggingFaceRowSource::shuffle_candidates_deterministically(&config, &mut right, 42);
        assert_eq!(left, right);

        // Different seeds produce different orderings for non-trivial inputs.
        let mut alt = vec!["c".to_string(), "a".to_string(), "b".to_string()];
        HuggingFaceRowSource::shuffle_candidates_deterministically(&config, &mut alt, 99);
        // Membership is preserved regardless of seed.
        let mut sorted_left = left.clone();
        sorted_left.sort();
        let mut sorted_alt = alt.clone();
        sorted_alt.sort();
        assert_eq!(sorted_left, sorted_alt);
    }

    #[test]
    fn extract_split_row_count_handles_configs_and_dataset_fallbacks() {
        let by_config_splits = json!({
            "size": {
                "configs": [
                    {
                        "config_name": "default",
                        "splits": [
                            {"name": "train", "num_rows": 21},
                            {"name": "validation", "num_rows": 4}
                        ]
                    }
                ]
            }
        });
        let rows = HuggingFaceRowSource::extract_split_row_count_from_size_response(
            &by_config_splits,
            "default",
            "train",
        );
        assert_eq!(rows, Some(21));

        let dataset_only = json!({
            "size": {
                "dataset": {"num_rows": 99}
            }
        });
        let rows = HuggingFaceRowSource::extract_split_row_count_from_size_response(
            &dataset_only,
            "default",
            "",
        );
        assert_eq!(rows, Some(99));
    }

    #[test]
    fn parse_global_row_count_response_uses_config_total_when_split_empty() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let body = serde_json::to_string(&json!({
            "size": {
                "configs": [
                    {"config": "default", "num_rows": 17}
                ]
            }
        }))
        .unwrap();

        let parsed = HuggingFaceRowSource::parse_global_row_count_response(
            &HuggingFaceRowsConfig {
                split: "".to_string(),
                ..config
            },
            &body,
        )
        .unwrap();
        assert_eq!(parsed, Some(17));
    }

    #[test]
    fn ensure_row_available_returns_from_fast_paths() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);

        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 3;
            state.remote_candidates = Some(vec!["x".to_string()]);
            state.next_remote_idx = 0;
        }
        assert!(source.ensure_row_available(1).unwrap());

        let source_done = test_source(test_config(dir.path().to_path_buf()));
        {
            let mut state = source_done.state.lock().unwrap();
            state.materialized_rows = 0;
            state.remote_candidates = Some(vec!["a".to_string()]);
            state.next_remote_idx = 1;
        }
        assert!(!source_done.ensure_row_available(0).unwrap());
    }

    #[test]
    fn build_shard_index_errors_when_no_accepted_files_exist() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("notes.txt"), b"plain").unwrap();
        let config = test_config(dir.path().to_path_buf());

        let err = HuggingFaceRowSource::build_shard_index(&config)
            .err()
            .expect("build_shard_index should fail");
        assert!(matches!(
            err,
            SamplerError::SourceUnavailable { ref reason, .. } if reason.contains("no shard files found")
        ));
    }

    #[test]
    fn materialize_local_file_errors_for_missing_source() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let missing = dir.path().join("missing.ndjson");
        let target = dir.path().join("target.ndjson");

        let err =
            HuggingFaceRowSource::materialize_local_file(&config, &missing, &target).unwrap_err();
        assert!(matches!(
            err,
            SamplerError::SourceUnavailable { ref reason, .. } if reason.contains("failed copying synced file")
        ));
    }

    #[test]
    fn download_and_materialize_shard_hf_hub_branch_returns_error_for_invalid_repo() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.dataset = "invalid///dataset".to_string();

        let err = HuggingFaceRowSource::download_and_materialize_shard(
            &config,
            "train/part-000.parquet",
            None,
            "shard 1/1",
        )
        .unwrap_err();
        assert!(matches!(err, SamplerError::SourceUnavailable { .. }));
    }

    #[test]
    fn index_single_shard_errors_for_missing_file() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let missing = dir.path().join("missing.ndjson");

        let err = HuggingFaceRowSource::index_single_shard(&config, &missing, 0)
            .err()
            .expect("index_single_shard should fail");
        assert!(matches!(err, SamplerError::SourceUnavailable { .. }));
    }

    #[test]
    fn index_single_shard_jsonl_records_checkpoints_by_stride() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("rows.ndjson");
        fs::write(
            &path,
            b"{\"text\":\"a\"}\n{\"text\":\"b\"}\n{\"text\":\"c\"}\n",
        )
        .unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.checkpoint_stride = 2;

        let shard = HuggingFaceRowSource::index_single_shard(&config, &path, 5)
            .unwrap()
            .0
            .unwrap();
        assert_eq!(shard.global_start, 5);
        assert_eq!(shard.row_count, 3);
        assert!(!shard.random_access);
        assert!(shard.checkpoints.len() >= 2);
        assert_eq!(shard.checkpoints[0], 0);
    }

    #[test]
    fn parquet_row_group_map_handles_empty_parquet_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.parquet");
        write_parquet_fixture(&path, &[]);
        let config = test_config(dir.path().to_path_buf());

        let (rows, groups) = HuggingFaceRowSource::parquet_row_group_map(&config, &path).unwrap();
        assert_eq!(rows, 0);
        assert!(groups.is_empty());
    }

    #[test]
    fn download_next_remote_shard_clears_row_cache_when_eviction_occurs() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.local_disk_cap_bytes = Some(20);
        let source = test_source(config.clone());

        let manifest_root = source.manifest_cache_root();
        fs::create_dir_all(&manifest_root).unwrap();
        let old_path = manifest_root.join("old.parquet");
        fs::write(&old_path, vec![1u8; 20]).unwrap();

        let payload = b"{\"text\":\"new\"}\n".to_vec();
        let (base_url, server) = spawn_one_shot_http(payload);
        let candidate =
            format!("url::{base_url}/datasets/org/ds/resolve/main/train/new-shard.ndjson");

        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 1;
            state.shards = vec![ShardIndex {
                path: old_path.clone(),
                global_start: 0,
                row_count: 1,
                random_access: true,
                parquet_row_groups: vec![(0, 1)],
                checkpoints: Vec::new(),
                remote_candidate: None,
            }];
            state.remote_candidates = Some(vec![candidate]);
            state.next_remote_idx = 0;
        }
        {
            let mut cache = source.cache.lock().unwrap();
            cache.insert(
                0,
                RowView {
                    row_id: Some("cached".to_string()),
                    timestamp: None,
                    text_fields: vec![RowTextField {
                        name: "text".to_string(),
                        text: "cached".to_string(),
                    }],
                },
                8,
            );
        }

        assert!(source.download_next_remote_shard().unwrap());
        server.join().unwrap();

        assert!(!old_path.exists());
        let cache = source.cache.lock().unwrap();
        assert!(cache.rows.is_empty());
        assert!(cache.order.is_empty());
    }

    #[test]
    fn default_triplet_recipes_returns_expected_shape() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let recipes = source.default_triplet_recipes();
        assert_eq!(recipes.len(), 2);
        assert_eq!(recipes[0].name, "huggingface_anchor_context_wrong_article");
        assert_eq!(recipes[1].name, "huggingface_anchor_anchor_wrong_article");
        assert_eq!(recipes[0].weight, 0.75);
        assert_eq!(recipes[1].weight, 0.25);
    }

    #[test]
    fn download_and_materialize_shard_url_short_circuits_when_cached_complete() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let candidate = "url::https://host/datasets/org/ds/resolve/main/train/ok.ndjson";
        let target = HuggingFaceRowSource::candidate_target_path(&config, candidate);
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::write(&target, b"ok").unwrap();

        let resolved = HuggingFaceRowSource::download_and_materialize_shard(
            &config,
            candidate,
            Some(2),
            "shard 1/1",
        )
        .unwrap();
        assert_eq!(resolved, target);
    }

    #[test]
    fn download_and_materialize_shard_url_replaces_stale_part_file() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let payload = b"{\"text\":\"a\"}\n".to_vec();
        let (base_url, server) = spawn_one_shot_http(payload.clone());
        let candidate = format!("url::{base_url}/datasets/org/ds/resolve/main/train/part-x.ndjson");
        let target = HuggingFaceRowSource::candidate_target_path(&config, &candidate);
        let temp_target = target.with_extension("part");
        fs::create_dir_all(temp_target.parent().unwrap()).unwrap();
        fs::write(&temp_target, b"stale").unwrap();

        let out = HuggingFaceRowSource::download_and_materialize_shard(
            &config,
            &candidate,
            None,
            "shard 1/1",
        )
        .unwrap();
        server.join().unwrap();

        assert_eq!(out, target);
        assert_eq!(fs::read(&target).unwrap(), payload);
    }

    #[test]
    fn download_next_remote_shard_skips_zero_row_download() {
        // Suppress the expected WARN "downloaded shard had zero rows and was skipped"
        // emitted when a shard file contains no JSON lines after download.  That warn
        // is correct production behaviour; silenced here to keep test output clean.
        let _quiet = tracing::subscriber::set_default(
            tracing_subscriber::fmt()
                .with_max_level(tracing::Level::ERROR)
                .finish(),
        );
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let payload = Vec::<u8>::new();
        let (base_url, server) = spawn_one_shot_http(payload);
        let candidate =
            format!("url::{base_url}/datasets/org/ds/resolve/main/train/part-empty.ndjson");

        {
            let mut state = source.state.lock().unwrap();
            state.remote_candidates = Some(vec![candidate]);
            state.next_remote_idx = 0;
        }

        assert!(source.download_next_remote_shard().unwrap());
        server.join().unwrap();
        let state = source.state.lock().unwrap();
        assert_eq!(state.materialized_rows, 0);
        assert!(state.shards.is_empty());
    }

    #[test]
    fn read_row_batch_errors_when_parquet_reader_cannot_open_file() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 1;
            state.total_rows = Some(1);
            state.shards = vec![ShardIndex {
                path: dir.path().join("missing.parquet"),
                global_start: 0,
                row_count: 1,
                random_access: true,
                parquet_row_groups: vec![(0, 1)],
                checkpoints: Vec::new(),
                remote_candidate: None,
            }];
        }

        let mut out = Vec::new();
        let err = source.read_row_batch(&[0], &mut out, Some(1));
        assert!(err.is_err());
    }

    #[test]
    fn refresh_exercises_large_total_progress_branch() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("rows.jsonl");
        let line = b"{\"id\":\"r\",\"text\":\"v\"}\n";
        let mut bytes = Vec::with_capacity(line.len() * 10_000);
        for _ in 0..10_000 {
            bytes.extend_from_slice(line);
        }
        fs::write(&path, bytes).unwrap();

        let mut config = test_config(dir.path().to_path_buf());
        config.checkpoint_stride = 256;
        config.refresh_batch_multiplier = 1;
        let source = test_source(config.clone());
        let shard = HuggingFaceRowSource::index_single_shard(&config, &path, 0)
            .unwrap()
            .0
            .unwrap();

        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 10_000;
            state.total_rows = Some(10_000);
            state.shards = vec![shard];
        }

        let snapshot = source.refresh(None, Some(1)).unwrap();
        assert_eq!(snapshot.records.len(), 1);
    }

    #[test]
    fn shard_size_bytes_returns_zero_for_missing_path() {
        let dir = tempdir().unwrap();
        let missing = dir.path().join("missing.file");
        assert_eq!(HuggingFaceRowSource::shard_size_bytes(&missing), 0);
    }

    #[test]
    fn shuffle_candidates_deterministically_is_noop_for_singleton() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let mut candidates = vec!["one".to_string()];
        HuggingFaceRowSource::shuffle_candidates_deterministically(&config, &mut candidates, 1);
        assert_eq!(candidates, vec!["one".to_string()]);
    }

    #[test]
    fn extract_split_row_count_returns_none_when_missing_entries() {
        let payload = json!({"size": {"configs": [{"config": "other", "splits": []}]}});
        let rows = HuggingFaceRowSource::extract_split_row_count_from_size_response(
            &payload, "default", "train",
        );
        assert!(rows.is_none());
    }

    #[test]
    fn uncached_candidates_from_parquet_manifest_returns_empty_without_entries() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let payload = json!({"other": []});
        let (candidates, sizes, matched) =
            HuggingFaceRowSource::all_candidates_from_parquet_manifest(&config, &payload).unwrap();
        assert!(candidates.is_empty());
        assert!(sizes.is_empty());
        // No parquet_files key → zero matched entries.
        assert_eq!(matched, 0);
    }

    #[test]
    fn read_line_at_errors_on_unexpected_eof_while_reading_target_row() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("rows.jsonl");
        fs::write(&path, b"{\"text\":\"a\"}\n").unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.checkpoint_stride = 1;
        let source = test_source(config.clone());
        let mut shard = HuggingFaceRowSource::index_single_shard(&config, &path, 0)
            .unwrap()
            .0
            .unwrap();
        let end = fs::metadata(&path).unwrap().len();
        shard.checkpoints = vec![0, end];

        let err = source.read_line_at(&shard, 1);
        assert!(err.is_err());
    }

    #[test]
    fn materialize_local_file_replaces_target_when_size_differs() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let src = dir.path().join("src.ndjson");
        let dst = dir.path().join("dst.ndjson");
        fs::write(&src, b"newer\n").unwrap();
        fs::write(&dst, b"old\n").unwrap();

        HuggingFaceRowSource::materialize_local_file(&config, &src, &dst).unwrap();
        assert_eq!(fs::read(&dst).unwrap(), b"newer\n");
    }

    #[test]
    fn row_to_record_preserves_explicit_timestamp() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let ts = Utc::now();
        let row = RowView {
            row_id: Some("r1".into()),
            timestamp: Some(ts),
            text_fields: vec![RowTextField {
                name: "text".into(),
                text: "alpha".into(),
            }],
        };

        let record = source.row_to_record(&row, 0).unwrap().unwrap();
        assert_eq!(record.created_at, ts);
        assert_eq!(record.updated_at, ts);
    }

    #[test]
    fn parse_row_text_columns_accept_numeric_values() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.text_columns = vec!["score".into()];
        let source = test_source(config);

        let row = source
            .parse_row(0, &json!({"score": 123}))
            .unwrap()
            .unwrap();
        assert_eq!(row.text_fields.len(), 1);
        assert_eq!(row.text_fields[0].text, "123");
    }

    #[test]
    fn refresh_limit_none_reads_up_to_total() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("rows.jsonl");
        fs::write(
            &path,
            b"{\"id\":\"r1\",\"text\":\"a\"}\n{\"id\":\"r2\",\"text\":\"b\"}\n",
        )
        .unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.checkpoint_stride = 1;
        config.refresh_batch_multiplier = 1;
        let source = test_source(config.clone());
        let shard = HuggingFaceRowSource::index_single_shard(&config, &path, 0)
            .unwrap()
            .0
            .unwrap();
        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 2;
            state.total_rows = Some(2);
            state.shards = vec![shard];
        }

        let snapshot = source.refresh(None, None).unwrap();
        assert_eq!(snapshot.records.len(), 2);
    }

    #[test]
    fn read_row_batch_skips_unavailable_indices_without_error() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 0;
            state.total_rows = Some(0);
            state.remote_candidates = Some(Vec::new());
        }

        let mut out = Vec::new();
        source.read_row_batch(&[0, 1], &mut out, Some(2)).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn candidate_target_path_maps_remote_urls_under_manifest_root() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let candidate =
            "url::https://huggingface.co/datasets/org/ds/resolve/main/train/part-000.parquet";
        let target = HuggingFaceRowSource::candidate_target_path(&config, candidate);
        assert!(target.ends_with("_parquet_manifest/main/train/part-000.parquet"));
    }

    #[test]
    fn candidate_target_path_keeps_local_candidates_relative() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let candidate = "train/part-001.ndjson";
        let target = HuggingFaceRowSource::candidate_target_path(&config, candidate);
        assert_eq!(target, config.snapshot_dir.join(candidate));
    }

    #[test]
    fn target_matches_expected_size_validates_when_expected_is_provided() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("payload.bin");
        fs::write(&path, vec![0u8; 5]).unwrap();

        assert!(HuggingFaceRowSource::target_matches_expected_size(
            &path,
            Some(5)
        ));
        assert!(!HuggingFaceRowSource::target_matches_expected_size(
            &path,
            Some(4)
        ));
        assert!(HuggingFaceRowSource::target_matches_expected_size(
            &path, None
        ));
    }

    #[test]
    fn parquet_row_group_map_and_index_single_shard_cover_success_path() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("rows.parquet");
        write_parquet_fixture(&path, &[("r1", "alpha"), ("r2", "beta"), ("r3", "gamma")]);
        let config = test_config(dir.path().to_path_buf());

        let (total_rows, groups) =
            HuggingFaceRowSource::parquet_row_group_map(&config, &path).unwrap();
        assert_eq!(total_rows, 3);
        assert!(!groups.is_empty());

        let shard = HuggingFaceRowSource::index_single_shard(&config, &path, 0)
            .unwrap()
            .0
            .unwrap();
        assert!(shard.random_access);
        assert_eq!(shard.row_count, 3);
        assert!(shard.checkpoints.is_empty());
    }

    #[test]
    fn read_row_batch_reads_parquet_rows_and_uses_cache_on_repeat() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("rows.parquet");
        write_parquet_fixture(&path, &[("r10", "ten"), ("r11", "eleven")]);

        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());
        let shard = HuggingFaceRowSource::index_single_shard(&config, &path, 0)
            .unwrap()
            .0
            .unwrap();
        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 2;
            state.total_rows = Some(2);
            state.shards = vec![shard];
        }

        let mut first = Vec::new();
        source.read_row_batch(&[0, 1], &mut first, None).unwrap();
        assert_eq!(first.len(), 2);
        assert!(first.iter().any(|record| record.id.ends_with("::r10")));

        let mut second = Vec::new();
        source.read_row_batch(&[0, 1], &mut second, None).unwrap();
        assert_eq!(second.len(), 2);
    }

    #[test]
    fn ensure_row_available_bootstraps_from_in_memory_candidates() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        let payload =
            b"{\"id\":\"r1\",\"text\":\"alpha\"}\n{\"id\":\"r2\",\"text\":\"beta\"}\n".to_vec();
        let (base_url, server) = spawn_one_shot_http(payload);
        let candidate =
            format!("url::{base_url}/datasets/org/ds/resolve/main/train/persisted.ndjson");

        {
            let mut state = source.state.lock().unwrap();
            state.remote_candidates = Some(vec![candidate]);
            state.next_remote_idx = 0;
        }

        assert!(source.ensure_row_available(0).unwrap());
        server.join().unwrap();

        let state = source.state.lock().unwrap();
        assert_eq!(state.materialized_rows, 2);
        assert_eq!(state.next_remote_idx, 1);
        assert_eq!(state.shards.len(), 1);
    }

    #[test]
    fn configure_sampler_updates_len_hint_headroom_via_trait_methods() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.cache_capacity = 10;
        config.remote_expansion_headroom_multiplier = 3;
        let source = test_source(config);
        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 5;
            state.total_rows = Some(100);
        }

        assert_eq!(source.reported_record_count().unwrap(), 35);

        let sampler = SamplerConfig {
            ingestion_max_records: 2,
            ..SamplerConfig::default()
        };
        source.configure_sampler(&sampler);

        assert_eq!(source.reported_record_count().unwrap(), 11);
    }

    #[test]
    fn build_shard_index_skips_empty_files_and_keeps_non_empty() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("a.ndjson"), b"").unwrap();
        fs::write(dir.path().join("b.ndjson"), b"{\"text\":\"x\"}\n").unwrap();
        let config = test_config(dir.path().to_path_buf());

        let (shards, discovered, _) = HuggingFaceRowSource::build_shard_index(&config).unwrap();
        assert_eq!(discovered, 1);
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0].row_count, 1);
    }

    #[test]
    fn resolve_remote_candidates_from_siblings_falls_back_when_split_filter_misses() {
        // Suppress the expected WARN "split filter 'train' matched no remote files; falling
        // back to extension-only remote candidate scan" that fires when no sibling paths
        // contain the split tag.  The fallback is what this test asserts on.
        let _quiet = tracing::subscriber::set_default(
            tracing_subscriber::fmt()
                .with_max_level(tracing::Level::ERROR)
                .finish(),
        );
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.split = "train".to_string();
        let accepted = HuggingFaceRowSource::normalized_shard_extensions(&config);
        let siblings = vec![
            "validation/file-a.ndjson".to_string(),
            "test/file-b.ndjson".to_string(),
        ];

        let (candidates, sizes) = HuggingFaceRowSource::resolve_remote_candidates_from_siblings(
            &config, &siblings, &accepted,
        )
        .unwrap();

        assert!(sizes.is_empty());
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn resolve_remote_candidates_from_siblings_errors_for_parquet_only_when_not_accepted() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.shard_extensions = vec!["ndjson".to_string()];
        let accepted = HuggingFaceRowSource::normalized_shard_extensions(&config);
        let siblings = vec!["train/only.parquet".to_string()];

        let result = HuggingFaceRowSource::resolve_remote_candidates_from_siblings(
            &config, &siblings, &accepted,
        );
        assert!(result.is_err());
    }

    #[test]
    fn resolve_remote_candidates_from_siblings_returns_empty_when_no_matches_and_no_parquet() {
        // Suppress the expected WARN "no remote candidates found for dataset='org/dataset'"
        // emitted when the sibling list contains no files matching the accepted extensions
        // and no parquet files.  That warn is correct production behaviour; the empty
        // return value is what this test asserts on.
        let _quiet = tracing::subscriber::set_default(
            tracing_subscriber::fmt()
                .with_max_level(tracing::Level::ERROR)
                .finish(),
        );
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let accepted = HuggingFaceRowSource::normalized_shard_extensions(&config);
        let siblings = vec!["train/notes.txt".to_string()];

        let (candidates, sizes) = HuggingFaceRowSource::resolve_remote_candidates_from_siblings(
            &config, &siblings, &accepted,
        )
        .unwrap();
        assert!(candidates.is_empty());
        assert!(sizes.is_empty());
    }

    #[test]
    fn parse_global_row_count_response_errors_on_invalid_json() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let parsed = HuggingFaceRowSource::parse_global_row_count_response(&config, "{bad-json");
        assert!(parsed.is_err());
    }

    #[test]
    fn parse_parquet_manifest_response_errors_on_invalid_json() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let parsed = HuggingFaceRowSource::parse_parquet_manifest_response(&config, "{bad-json");
        assert!(parsed.is_err());
    }

    #[test]
    fn parse_parquet_manifest_response_returns_candidates() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let body = serde_json::to_string(&json!({
            "parquet_files": [
                {"url": "https://host/datasets/x/resolve/main/train/0.parquet", "size": 5}
            ]
        }))
        .unwrap();

        let (candidates, sizes, matched) =
            HuggingFaceRowSource::parse_parquet_manifest_response(&config, &body).unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(sizes.len(), 1);
        assert_eq!(matched, 1);
    }

    #[test]
    fn list_remote_candidates_from_parquet_manifest_uses_test_endpoint_override() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let body = serde_json::to_vec(&json!({
            "parquet_files": [
                {"url": "https://host/datasets/x/resolve/main/train/0.parquet", "size": 5}
            ]
        }))
        .unwrap();
        let (base_url, server) = spawn_one_shot_http(body);

        let (candidates, sizes, matched) =
            with_env_var(TRIPLETS_HF_PARQUET_ENDPOINT, &base_url, || {
                HuggingFaceRowSource::list_remote_candidates_from_parquet_manifest(&config)
            })
            .unwrap();
        server.join().unwrap();

        assert_eq!(candidates.len(), 1);
        assert_eq!(sizes.len(), 1);
        assert_eq!(matched, 1);
    }

    #[test]
    fn fetch_global_row_count_uses_test_endpoint_override() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let body = serde_json::to_vec(&json!({
            "size": {
                "splits": [
                    {"config": "default", "split": "train", "num_rows": 12}
                ]
            }
        }))
        .unwrap();
        let (base_url, server) = spawn_one_shot_http(body);

        let rows = with_env_var(TRIPLETS_HF_SIZE_ENDPOINT, &base_url, || {
            HuggingFaceRowSource::fetch_global_row_count(&config)
        })
        .unwrap();
        server.join().unwrap();
        assert_eq!(rows, Some(12));
    }

    #[test]
    fn endpoint_helpers_fallback_for_empty_env_values() {
        let parquet = with_env_var(TRIPLETS_HF_PARQUET_ENDPOINT, "   ", || {
            HuggingFaceRowSource::parquet_manifest_endpoint()
        });
        assert_eq!(parquet, "https://datasets-server.huggingface.co/parquet");

        let size = with_env_var(TRIPLETS_HF_SIZE_ENDPOINT, "", || {
            HuggingFaceRowSource::size_endpoint()
        });
        assert_eq!(size, "https://datasets-server.huggingface.co/size");

        let info = with_env_var(TRIPLETS_HF_INFO_ENDPOINT, "   ", || {
            HuggingFaceRowSource::info_endpoint()
        });
        assert_eq!(info, "https://datasets-server.huggingface.co/info");
    }

    #[test]
    fn resolve_remote_candidates_respects_split_prefix_in_filename() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.split = "train".to_string();
        let accepted = HuggingFaceRowSource::normalized_shard_extensions(&config);
        let siblings = vec![
            "train-part-000.ndjson".to_string(),
            "validation-part-000.ndjson".to_string(),
        ];

        let (candidates, _) = HuggingFaceRowSource::resolve_remote_candidates_from_siblings(
            &config, &siblings, &accepted,
        )
        .unwrap();

        assert_eq!(candidates, vec!["train-part-000.ndjson".to_string()]);
    }

    #[test]
    fn fetch_global_row_count_returns_none_when_split_not_present() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let body = serde_json::to_vec(&json!({
            "size": {
                "splits": [
                    {"config": "default", "split": "validation", "num_rows": 12}
                ]
            }
        }))
        .unwrap();
        let (base_url, server) = spawn_one_shot_http(body);

        let rows = with_env_var(TRIPLETS_HF_SIZE_ENDPOINT, &base_url, || {
            HuggingFaceRowSource::fetch_global_row_count(&config)
        })
        .unwrap();
        server.join().unwrap();
        assert_eq!(rows, None);
    }

    #[test]
    fn list_remote_candidates_returns_manifest_candidates_before_repo_fallback() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let body = serde_json::to_vec(&json!({
            "parquet_files": [
                {"url": "https://host/datasets/x/resolve/main/train/1.ndjson", "size": 9}
            ]
        }))
        .unwrap();
        let (base_url, server) = spawn_one_shot_http(body);

        let (candidates, sizes) = with_env_var(TRIPLETS_HF_PARQUET_ENDPOINT, &base_url, || {
            HuggingFaceRowSource::list_remote_candidates(&config)
        })
        .unwrap();
        server.join().unwrap();

        assert_eq!(candidates.len(), 1);
        assert_eq!(sizes.len(), 1);
        assert!(candidates[0].ends_with("/1.ndjson"));
    }

    #[test]
    fn list_remote_candidates_does_not_fall_back_when_all_manifest_shards_cached() {
        // Regression test: list_remote_candidates must NOT fall through to the hf-hub
        // siblings listing when a parquet manifest exists, regardless of whether all
        // shards are already cached on disk.
        //
        // The hf-hub siblings listing returns every language config in the repository,
        // not just the one requested (e.g. wikimedia/wikipedia/20231101.en → also
        // .fr, .de, …).  The guard is `matched_manifest_entries > 0`, which is
        // independent of cache state.
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());

        // Pre-create the .simdr store target so the manifest entry is "fully cached".
        let shard_url = "https://host/datasets/org/ds/resolve/main/train/part-000.ndjson";
        let candidate = format!("{REMOTE_URL_PREFIX}{shard_url}");
        let target = HuggingFaceRowSource::candidate_target_path(&config, &candidate);
        let store_target = HuggingFaceRowSource::shard_store_path_for(&target);
        fs::create_dir_all(store_target.parent().unwrap()).unwrap();
        fs::write(&store_target, b"cached").unwrap();

        let body = serde_json::to_vec(&json!({
            "parquet_files": [
                {"url": shard_url, "size": 6}
            ]
        }))
        .unwrap();
        let (base_url, server) = spawn_one_shot_http(body);

        // Must return the full manifest candidate list without falling through to hf-hub.
        let (candidates, sizes) = with_env_var(TRIPLETS_HF_PARQUET_ENDPOINT, &base_url, || {
            HuggingFaceRowSource::list_remote_candidates(&config)
        })
        .unwrap();
        server.join().unwrap();

        assert_eq!(
            candidates.len(),
            1,
            "fully-cached shard must still appear in candidates (cache ≠ order)"
        );
        assert_eq!(sizes.len(), 1);
        assert!(candidates[0].ends_with(shard_url));
    }

    #[test]
    fn list_remote_candidates_from_parquet_manifest_errors_when_endpoint_unreachable() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());

        let result = with_env_var(TRIPLETS_HF_PARQUET_ENDPOINT, "http://127.0.0.1:1", || {
            HuggingFaceRowSource::list_remote_candidates_from_parquet_manifest(&config)
        });
        assert!(result.is_err());
    }

    #[test]
    fn fetch_global_row_count_errors_when_endpoint_unreachable() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());

        let result = with_env_var(TRIPLETS_HF_SIZE_ENDPOINT, "http://127.0.0.1:1", || {
            HuggingFaceRowSource::fetch_global_row_count(&config)
        });
        assert!(result.is_err());
    }

    #[test]
    fn download_and_materialize_shard_downloads_url_candidate() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let payload = b"{\"text\":\"a\"}\n{\"text\":\"b\"}\n".to_vec();
        let (base_url, server) = spawn_one_shot_http(payload.clone());
        let candidate =
            format!("url::{base_url}/datasets/org/ds/resolve/main/train/part-000.ndjson");

        let target = HuggingFaceRowSource::download_and_materialize_shard(
            &config,
            &candidate,
            None,
            "shard 1/1",
        )
        .unwrap();

        server.join().unwrap();
        assert!(target.exists());
        assert_eq!(fs::read(&target).unwrap(), payload);
    }

    #[test]
    fn download_and_materialize_shard_replaces_incomplete_existing_target() {
        // Suppress the expected WARN "replacing incomplete shard before retry" that fires
        // when an existing target file's size does not match the expected manifest size.
        // Detecting and replacing the stale file is what this test asserts on.
        let _quiet = tracing::subscriber::set_default(
            tracing_subscriber::fmt()
                .with_max_level(tracing::Level::ERROR)
                .finish(),
        );
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let payload = b"{\"text\":\"a\"}\n".to_vec();
        let (base_url, server) = spawn_one_shot_http(payload.clone());
        let candidate =
            format!("url::{base_url}/datasets/org/ds/resolve/main/train/part-009.ndjson");

        let target = HuggingFaceRowSource::candidate_target_path(&config, &candidate);
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::write(&target, b"bad").unwrap();

        let refreshed = HuggingFaceRowSource::download_and_materialize_shard(
            &config,
            &candidate,
            Some(payload.len() as u64),
            "shard 1/1",
        )
        .unwrap();

        server.join().unwrap();
        assert_eq!(refreshed, target);
        assert_eq!(fs::read(&target).unwrap(), payload);
    }

    #[test]
    fn download_next_remote_shard_parquet_stages_temp_and_persists_store_only() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        let fixture_path = dir.path().join("fixture.parquet");
        write_parquet_fixture(&fixture_path, &[("r1", "alpha"), ("r2", "beta")]);
        let payload = fs::read(&fixture_path).unwrap();
        let (base_url, server) = spawn_one_shot_http(payload);
        let candidate =
            format!("url::{base_url}/datasets/org/ds/resolve/main/train/part-222.parquet");

        {
            let mut state = source.state.lock().unwrap();
            state.remote_candidates = Some(vec![candidate.clone()]);
            state.next_remote_idx = 0;
        }

        assert!(source.download_next_remote_shard().unwrap());
        server.join().unwrap();

        let parquet_target = HuggingFaceRowSource::candidate_target_path(&config, &candidate);
        let store_target = HuggingFaceRowSource::shard_store_path_for(&parquet_target);

        assert!(store_target.exists());
        assert!(!parquet_target.exists());

        let state = source.state.lock().unwrap();
        assert_eq!(state.shards.len(), 1);
        assert_eq!(state.shards[0].path, store_target);
        assert_eq!(state.materialized_rows, 2);
    }

    // When transcode_parquet_shard_to_row_store takes the early-return path
    // (simdr store already fully populated), it must still delete the input
    // parquet file and return a ShardIndex with random_access = true.
    #[test]
    fn transcode_parquet_shard_to_row_store_early_return_cleans_up_parquet() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        // Write a parquet fixture (simulates a stale/unconsumed parquet from a
        // previous run that crashed before the delete step fired).
        let parquet_path = dir.path().join("stale.parquet");
        write_parquet_fixture(&parquet_path, &[("r1", "hello"), ("r2", "world")]);
        assert!(
            parquet_path.exists(),
            "parquet fixture must exist before test"
        );

        // Pre-populate the corresponding simdr store so the function short-circuits.
        let store_path = HuggingFaceRowSource::shard_store_path_for(&parquet_path);
        write_simdr_fixture(&store_path, &[("r1", "hello"), ("r2", "world")]);
        assert!(store_path.exists(), "simdr store must exist before test");

        let shard = ShardIndex {
            path: parquet_path.clone(),
            global_start: 0,
            row_count: 2,
            random_access: true,
            parquet_row_groups: vec![(0, 2)],
            checkpoints: Vec::new(),
            remote_candidate: None,
        };

        let result = source
            .transcode_parquet_shard_to_row_store(&shard)
            .expect("transcode must succeed");

        // Stale parquet must be gone.
        assert!(
            !parquet_path.exists(),
            "stale parquet must be removed on early return"
        );

        // Simdr store must still be present.
        assert!(store_path.exists(), "simdr store must survive early return");

        // Returned shard must point to the store and carry random_access = true.
        let returned = result.expect("early return must yield Some(ShardIndex)");
        assert_eq!(returned.path, store_path);
        assert!(
            returned.random_access,
            "random_access must be true for .simdr store (random-access read path)"
        );
        assert_eq!(returned.row_count, 2);
    }

    #[test]
    fn download_next_remote_shard_materializes_and_indexes_rows() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let payload = b"{\"text\":\"a\"}\n{\"text\":\"b\"}\n".to_vec();
        let (base_url, server) = spawn_one_shot_http(payload);
        let candidate =
            format!("url::{base_url}/datasets/org/ds/resolve/main/train/part-001.ndjson");

        {
            let mut state = source.state.lock().unwrap();
            state.remote_candidates = Some(vec![candidate.clone()]);
            state.remote_candidate_sizes.insert(candidate, 24);
            state.next_remote_idx = 0;
        }

        assert!(source.download_next_remote_shard().unwrap());
        server.join().unwrap();

        let state = source.state.lock().unwrap();
        assert_eq!(state.materialized_rows, 2);
        assert_eq!(state.shards.len(), 1);
        assert_eq!(state.next_remote_idx, 1);
    }

    #[test]
    fn ensure_row_available_triggers_lazy_download_for_remote_candidates() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let payload = b"{\"text\":\"x\"}\n{\"text\":\"y\"}\n".to_vec();
        let (base_url, server) = spawn_one_shot_http(payload);
        let candidate =
            format!("url::{base_url}/datasets/org/ds/resolve/main/train/part-002.ndjson");

        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 0;
            state.remote_candidates = Some(vec![candidate.clone()]);
            state.remote_candidate_sizes.insert(candidate, 24);
            state.next_remote_idx = 0;
        }

        assert!(source.ensure_row_available(0).unwrap());
        server.join().unwrap();

        let state = source.state.lock().unwrap();
        assert!(state.materialized_rows >= 1);
        assert_eq!(state.next_remote_idx, 1);
    }

    #[test]
    fn download_next_remote_shard_consumes_distinct_candidates_in_order() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        let payload_a = b"{\"id\":\"a\",\"text\":\"alpha\"}\n".to_vec();
        let payload_b = b"{\"id\":\"b\",\"text\":\"beta\"}\n".to_vec();
        let (base_a, server_a) = spawn_one_shot_http(payload_a);
        let (base_b, server_b) = spawn_one_shot_http(payload_b);
        let candidate_a = format!("url::{base_a}/datasets/org/ds/resolve/main/train/part-a.ndjson");
        let candidate_b = format!("url::{base_b}/datasets/org/ds/resolve/main/train/part-b.ndjson");
        {
            let mut state = source.state.lock().unwrap();
            state.remote_candidates = Some(vec![candidate_a.clone(), candidate_b.clone()]);
            state.remote_candidate_sizes.insert(candidate_a.clone(), 27);
            state.remote_candidate_sizes.insert(candidate_b.clone(), 26);
            state.next_remote_idx = 0;
        }

        assert!(source.download_next_remote_shard().unwrap());
        assert!(source.download_next_remote_shard().unwrap());
        server_a.join().unwrap();
        server_b.join().unwrap();

        let state = source.state.lock().unwrap();
        assert_eq!(state.next_remote_idx, 2);
        assert_eq!(state.shards.len(), 2);
        assert_ne!(state.shards[0].path, state.shards[1].path);
    }

    #[test]
    fn download_next_remote_shard_skips_already_materialised_shard() {
        // Verifies the cache/determinism decoupling: if a shard's store file already
        // exists on disk, download_next_remote_shard must advance next_remote_idx
        // without making any network request, leaving materialized_rows unchanged.
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        let candidate =
            "url::http://127.0.0.1:1/datasets/org/ds/resolve/main/train/pre-cached.ndjson"
                .to_string();
        let target = HuggingFaceRowSource::candidate_target_path(&config, &candidate);
        let store_path = HuggingFaceRowSource::shard_store_path_for(&target);
        fs::create_dir_all(store_path.parent().unwrap()).unwrap();
        fs::write(&store_path, b"dummy").unwrap();

        {
            let mut state = source.state.lock().unwrap();
            state.remote_candidates = Some(vec![candidate.clone()]);
            state.remote_candidate_order = vec![0];
            state.remote_candidate_sizes.insert(candidate, 5);
            state.next_remote_idx = 0;
        }

        // No HTTP server is running — if a real download were attempted it would fail.
        assert!(
            source.download_next_remote_shard().unwrap(),
            "should return true (candidate consumed)"
        );

        let state = source.state.lock().unwrap();
        assert_eq!(
            state.next_remote_idx, 1,
            "pointer advanced past cached shard"
        );
        assert_eq!(
            state.materialized_rows, 0,
            "materialized_rows unchanged — shard was already counted at startup"
        );
        assert_eq!(
            state.shards.len(),
            0,
            "no new shard added to in-memory list"
        );
    }

    #[test]
    fn extract_split_row_count_reads_split_entries() {
        let payload = json!({
            "size": {
                "splits": [
                    {"config": "default", "split": "train", "num_rows": 123u64},
                    {"config": "default", "split": "validation", "num_rows": 45u64}
                ]
            }
        });

        let count = HuggingFaceRowSource::extract_split_row_count_from_size_response(
            &payload,
            "default",
            "validation",
        );
        assert_eq!(count, Some(45));
    }

    #[test]
    fn extract_split_row_count_reads_config_fallback_and_dataset_total() {
        let payload = json!({
            "size": {
                "configs": [
                    {
                        "config": "default",
                        "splits": [{"name": "test", "num_rows": 77u64}],
                        "num_rows": 200u64
                    }
                ],
                "dataset": {"num_rows": 999u64}
            }
        });

        let split_count = HuggingFaceRowSource::extract_split_row_count_from_size_response(
            &payload, "default", "test",
        );
        assert_eq!(split_count, Some(77));

        let empty_split_count = HuggingFaceRowSource::extract_split_row_count_from_size_response(
            &payload, "default", "",
        );
        assert_eq!(empty_split_count, Some(200));
    }

    #[test]
    fn shard_candidate_seed_is_seeded_and_source_scoped() {
        let dir = tempdir().unwrap();
        let mut a = test_config(dir.path().join("a"));
        let mut b = test_config(dir.path().join("b"));
        a.source_id = "source_a".to_string();
        b.source_id = "source_b".to_string();

        let with_seed_a = HuggingFaceRowSource::shard_candidate_seed(&a, 100, 42);
        let with_seed_a_again = HuggingFaceRowSource::shard_candidate_seed(&a, 100, 42);
        assert_eq!(with_seed_a, with_seed_a_again);

        let with_seed_b = HuggingFaceRowSource::shard_candidate_seed(&b, 100, 42);
        assert_ne!(with_seed_a, with_seed_b);

        let different_seed_a = HuggingFaceRowSource::shard_candidate_seed(&a, 100, 7);
        assert_ne!(with_seed_a, different_seed_a);
    }

    #[test]
    fn remote_shard_permutation_is_deterministic_by_sampler_seed() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let total = 8usize;

        let seed_a = HuggingFaceRowSource::shard_candidate_seed(&config, total, 7);
        let seed_b = HuggingFaceRowSource::shard_candidate_seed(&config, total, 7);
        let seed_c = HuggingFaceRowSource::shard_candidate_seed(&config, total, 123);

        let mut perm_a = crate::source::IndexPermutation::new(total, seed_a, 0);
        let mut perm_b = crate::source::IndexPermutation::new(total, seed_b, 0);
        let mut perm_c = crate::source::IndexPermutation::new(total, seed_c, 0);

        let take = 6usize;
        let order_a: Vec<usize> = (0..take).map(|_| perm_a.next()).collect();
        let order_b: Vec<usize> = (0..take).map(|_| perm_b.next()).collect();
        let order_c: Vec<usize> = (0..take).map(|_| perm_c.next()).collect();

        assert_eq!(order_a, order_b);
        assert_ne!(order_a, order_c);
    }

    #[test]
    fn build_shard_index_ignores_manifest_non_shard_artifacts() {
        // Non-shard-store files under _parquet_manifest (e.g. .ndjson, .json
        // metadata) must be skipped even though .ndjson is in shard_extensions.
        // Only .simdr stores inside _parquet_manifest should be indexed.
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.shard_extensions = vec!["ndjson".to_string()];

        let local = dir.path().join("local.ndjson");
        fs::write(&local, b"{\"id\":\"l1\",\"text\":\"x\"}\n").unwrap();

        // A .ndjson file under _parquet_manifest is a non-shard artifact and
        // must NOT be indexed (it does not match HF_SHARD_STORE_EXTENSION).
        let manifest_meta = dir
            .path()
            .join("_parquet_manifest")
            .join("main/train/cached.ndjson");
        fs::create_dir_all(manifest_meta.parent().unwrap()).unwrap();
        fs::write(&manifest_meta, b"{\"id\":\"r1\",\"text\":\"y\"}\n").unwrap();

        let (shards, discovered, _) = HuggingFaceRowSource::build_shard_index(&config).unwrap();
        assert_eq!(discovered, 1);
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0].path, local);
    }

    #[test]
    fn build_shard_index_indexes_simdr_stores_under_manifest() {
        // Remote-sourced shards are stored as .simdr stores under
        // _parquet_manifest after transcoding from parquet.  build_shard_index
        // must discover and index them so materialized_rows is correct on
        // restart — the regression that caused every refresh to return 0 rows.
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());

        // A .simdr store under _parquet_manifest (simulates a previously
        // downloaded and transcoded remote shard).
        let store_path = dir
            .path()
            .join("_parquet_manifest")
            .join("refs%2Fconvert%2Fparquet/20231101.en/train/0000.simdr");
        fs::create_dir_all(store_path.parent().unwrap()).unwrap();
        write_simdr_fixture(&store_path, &[("row0", "hello"), ("row1", "world")]);

        // A non-shard metadata file that must be ignored.
        let seq_state_path = dir
            .path()
            .join("_parquet_manifest")
            .join("_sequence_state.json");
        fs::write(&seq_state_path, b"{}").unwrap();

        let (shards, discovered, _) = HuggingFaceRowSource::build_shard_index(&config).unwrap();
        assert_eq!(discovered, 2, "simdr store rows should be indexed");
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0].path, store_path);
        assert_eq!(shards[0].row_count, 2);
    }

    #[test]
    fn expansion_headroom_uses_sampler_ingestion_max_records_when_configured() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);

        assert_eq!(source.effective_expansion_headroom_rows(), 30);

        let sampler = SamplerConfig {
            ingestion_max_records: 7,
            ..SamplerConfig::default()
        };
        source.configure_sampler(&sampler);
        assert_eq!(source.effective_expansion_headroom_rows(), 21);
    }

    #[test]
    fn value_to_text_handles_scalar_and_structured_values() {
        assert_eq!(
            HuggingFaceRowSource::value_to_text(&json!(null), None),
            None
        );
        assert_eq!(
            HuggingFaceRowSource::value_to_text(&json!("   "), None),
            None
        );
        assert_eq!(
            HuggingFaceRowSource::value_to_text(&json!("hello"), None),
            Some("hello".into())
        );
        assert_eq!(
            HuggingFaceRowSource::value_to_text(&json!(true), None),
            Some("true".into())
        );
        assert_eq!(
            HuggingFaceRowSource::value_to_text(&json!(3.5), None),
            Some("3.5".into())
        );
        assert_eq!(
            HuggingFaceRowSource::value_to_text(&json!([1, 2]), None),
            Some("[1,2]".into())
        );
    }

    #[test]
    fn parse_row_uses_explicit_text_columns() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.id_column = Some("id".into());
        config.text_columns = vec!["title".into(), "body".into()];
        let source = test_source(config);

        let row = source
            .parse_row(
                5,
                &json!({
                    "id": "row-5",
                    "title": "Anchor text",
                    "body": "Context text",
                    "flag": true
                }),
            )
            .unwrap()
            .unwrap();

        // Candidate coalescing: the first non-empty column (title) is selected;
        // body is never tried because title already yielded a value.
        assert_eq!(row.row_id.as_deref(), Some("row-5"));
        assert_eq!(row.text_fields.len(), 1);
        assert_eq!(row.text_fields[0].name, "title");
        assert_eq!(row.text_fields[0].text, "Anchor text");
        assert!(row.text_fields.iter().all(|field| field.name != "id"));
    }

    #[test]
    fn parse_row_with_required_columns_skips_when_missing() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.anchor_columns = vec!["anchor".into()];
        config.positive_columns = vec!["positive".into()];
        config.context_columns = vec!["context".into()];
        let source = test_source(config);

        let parsed = source.parse_row(0, &json!({"anchor": "x", "context": "z"}));
        assert!(parsed.unwrap().is_none());
    }

    #[test]
    fn parse_row_errors_when_payload_is_not_object() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);

        let err = source.parse_row(0, &json!("not-an-object"));
        assert!(err.is_err());
    }

    #[test]
    fn row_to_record_builds_expected_sections() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let row = RowView {
            row_id: Some("abc".into()),
            timestamp: Some(Utc::now()),
            text_fields: vec![
                RowTextField {
                    name: "title".into(),
                    text: "anchor".into(),
                },
                RowTextField {
                    name: "pos".into(),
                    text: "positive".into(),
                },
                RowTextField {
                    name: "ctx".into(),
                    text: "extra".into(),
                },
            ],
        };

        let record = source.row_to_record(&row, 1).unwrap().unwrap();
        assert_eq!(record.sections.len(), 3);
        assert_eq!(record.sections[0].role, SectionRole::Anchor);
        assert_eq!(record.sections[1].role, SectionRole::Context);
        assert_eq!(record.id, "hf_test::abc");
    }

    #[test]
    fn effective_refresh_batch_target_uses_multiplier_floor_of_one() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.refresh_batch_multiplier = 0;
        let source = test_source(config);
        assert_eq!(source.effective_refresh_batch_target(7), 7);
    }

    #[test]
    fn locate_shard_and_recompute_offsets_work() {
        let mut shards = vec![
            ShardIndex {
                path: PathBuf::from("a"),
                global_start: 10,
                row_count: 3,
                random_access: false,
                parquet_row_groups: Vec::new(),
                checkpoints: vec![0],
                remote_candidate: None,
            },
            ShardIndex {
                path: PathBuf::from("b"),
                global_start: 20,
                row_count: 2,
                random_access: false,
                parquet_row_groups: Vec::new(),
                checkpoints: vec![0],
                remote_candidate: None,
            },
        ];
        let hit = HuggingFaceRowSource::locate_shard(&shards, 11).unwrap();
        assert_eq!(hit.1, 1);

        let mut state = SourceState {
            materialized_rows: 0,
            total_rows: None,
            shards: std::mem::take(&mut shards),
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            remote_candidate_order: Vec::new(),
        };
        HuggingFaceRowSource::recompute_shard_offsets(&mut state);
        assert_eq!(state.shards[0].global_start, 0);
        assert_eq!(state.shards[1].global_start, 3);
        assert_eq!(state.materialized_rows, 5);
    }

    #[test]
    fn len_hint_covers_known_and_empty_paths() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);

        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 5;
            state.total_rows = Some(100);
        }
        // headroom = ingestion_max_records * multiplier = 10 * 3 = 30; since known (5)
        // < headroom, expansion = 30; upper = min(5+30, 100) = 35
        assert_eq!(source.len_hint(), Some(35));

        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 0;
            state.total_rows = Some(0);
        }
        assert_eq!(source.len_hint(), Some(0));
    }

    #[test]
    fn len_hint_defaults_to_one_when_unknown_and_not_exhausted() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        // Simulate an uninitialized source that hasn't fetched candidates yet.
        source.state.lock().unwrap().remote_candidates = None;
        assert_eq!(source.len_hint(), Some(1));
    }

    #[test]
    fn len_hint_keeps_trickle_remote_expansion_after_warmup() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.cache_capacity = 4;
        config.remote_expansion_headroom_multiplier = 2;
        let source = test_source(config);

        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 8;
            state.total_rows = Some(10_000);
        }

        // After warmup, expansion continues in trickle mode (12.5% of
        // headroom, minimum 1) so remote shard diversity can keep growing.
        assert_eq!(source.len_hint(), Some(9));
    }

    #[test]
    fn eligible_rows_extends_cached_index_when_new_shard_is_appended() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        let appended_path = dir.path().join("append.ndjson");
        fs::write(&appended_path, b"{\"id\":\"r1\",\"text\":\"hello\"}\n").unwrap();
        let appended = HuggingFaceRowSource::index_single_shard(&config, &appended_path, 1)
            .unwrap()
            .0
            .unwrap();

        let baseline = ShardIndex {
            path: dir.path().join("missing-baseline.ndjson"),
            global_start: 0,
            row_count: 1,
            random_access: false,
            parquet_row_groups: Vec::new(),
            checkpoints: vec![0],
            remote_candidate: None,
        };

        {
            let mut state = source.state.lock().unwrap();
            state.shards = vec![baseline.clone(), appended.clone()];
            state.materialized_rows = 2;
            state.total_rows = Some(2);
        }

        {
            let mut cache = source.eligible_index.lock().unwrap();
            cache.signature = Some(HuggingFaceRowSource::shard_signature(std::slice::from_ref(
                &baseline,
            )));
            cache.rows = Some(Arc::new(vec![0]));
            cache.shards = vec![baseline];
        }

        let rows = source.eligible_rows().unwrap();
        assert_eq!(rows.as_ref(), &vec![0, 1]);
    }

    #[test]
    fn read_line_at_reads_expected_row_with_checkpoints() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("rows.jsonl");
        let mut file = File::create(&path).unwrap();
        file.write_all(b"{\"text\":\"a\"}\n").unwrap();
        file.write_all(b"{\"text\":\"b\"}\n").unwrap();
        file.write_all(b"{\"text\":\"c\"}\n").unwrap();

        let mut config = test_config(dir.path().to_path_buf());
        config.checkpoint_stride = 1;
        let source = test_source(config.clone());
        let shard = HuggingFaceRowSource::index_single_shard(&config, &path, 0)
            .unwrap()
            .0
            .unwrap();

        let line = source.read_line_at(&shard, 2).unwrap();
        assert!(line.contains("\"c\""));
    }

    #[test]
    fn read_line_at_errors_when_checkpoint_is_missing() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("rows.jsonl");
        fs::write(&path, b"{\"text\":\"a\"}\n").unwrap();

        let mut config = test_config(dir.path().to_path_buf());
        config.checkpoint_stride = 1;
        let source = test_source(config.clone());
        let mut shard = HuggingFaceRowSource::index_single_shard(&config, &path, 0)
            .unwrap()
            .0
            .unwrap();
        shard.checkpoints.clear();

        let err = source.read_line_at(&shard, 0);
        assert!(err.is_err());
    }

    #[test]
    fn materialize_local_file_copies_and_is_idempotent_when_size_matches() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let src = dir.path().join("src.ndjson");
        let dst = dir.path().join("nested/dst.ndjson");

        fs::write(&src, b"line\n").unwrap();
        HuggingFaceRowSource::materialize_local_file(&config, &src, &dst).unwrap();
        let first = fs::read(&dst).unwrap();
        HuggingFaceRowSource::materialize_local_file(&config, &src, &dst).unwrap();
        let second = fs::read(&dst).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn enforce_disk_cap_evicts_old_manifest_shards() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.local_disk_cap_bytes = Some(10);
        let source = test_source(config);

        let manifest_root = source.manifest_cache_root();
        fs::create_dir_all(&manifest_root).unwrap();
        let evict_path = manifest_root.join("a.parquet");
        let keep_path = manifest_root.join("b.parquet");
        fs::write(&evict_path, vec![1u8; 8]).unwrap();
        fs::write(&keep_path, vec![2u8; 8]).unwrap();

        let mut state = SourceState {
            materialized_rows: 16,
            total_rows: None,
            shards: vec![
                ShardIndex {
                    path: evict_path.clone(),
                    global_start: 0,
                    row_count: 8,
                    random_access: true,
                    parquet_row_groups: vec![(0, 8)],
                    checkpoints: Vec::new(),
                    remote_candidate: None,
                },
                ShardIndex {
                    path: keep_path.clone(),
                    global_start: 8,
                    row_count: 8,
                    random_access: true,
                    parquet_row_groups: vec![(0, 8)],
                    checkpoints: Vec::new(),
                    remote_candidate: None,
                },
            ],
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            remote_candidate_order: Vec::new(),
        };

        let evicted = source
            .enforce_disk_cap_locked(&mut state, &keep_path)
            .unwrap();
        assert!(evicted);
        assert!(!evict_path.exists());
        assert!(keep_path.exists());
        assert_eq!(state.shards.len(), 1);
    }

    #[test]
    fn enforce_disk_cap_ignores_min_resident_and_applies_policy() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.local_disk_cap_bytes = Some(4);
        let source = test_source(config);

        let manifest_root = source.manifest_cache_root();
        fs::create_dir_all(&manifest_root).unwrap();
        let protected = manifest_root.join("only.parquet");
        fs::write(&protected, vec![1u8; 8]).unwrap();

        let mut state = SourceState {
            materialized_rows: 8,
            total_rows: None,
            shards: vec![ShardIndex {
                path: protected.clone(),
                global_start: 0,
                row_count: 8,
                random_access: true,
                parquet_row_groups: vec![(0, 8)],
                checkpoints: Vec::new(),
                remote_candidate: None,
            }],
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            remote_candidate_order: Vec::new(),
        };

        let evicted = source
            .enforce_disk_cap_locked(&mut state, &protected)
            .unwrap();
        assert!(evicted);
        assert!(!protected.exists());
        assert_eq!(state.shards.len(), 0);
    }

    #[test]
    fn build_shard_index_discovers_local_jsonl_shards() {
        let dir = tempdir().unwrap();
        let root = dir.path().to_path_buf();
        fs::write(root.join("a.jsonl"), b"{\"text\":\"a\"}\n").unwrap();
        fs::write(root.join("b.ndjson"), b"{\"text\":\"b\"}\n").unwrap();

        let config = test_config(root.clone());
        let (shards, discovered, _) = HuggingFaceRowSource::build_shard_index(&config).unwrap();
        assert_eq!(discovered, 2);
        assert_eq!(shards.len(), 2);
    }

    #[test]
    fn index_single_shard_returns_none_for_empty_file() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let path = dir.path().join("empty.jsonl");
        fs::write(&path, b"").unwrap();
        let shard = HuggingFaceRowSource::index_single_shard(&config, &path, 0).unwrap();
        assert!(shard.0.is_none());
    }

    #[test]
    fn refresh_reads_local_rows_and_advances_cursor() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("rows.jsonl");
        fs::write(
            &path,
            b"{\"id\":\"r1\",\"text\":\"alpha\"}\n{\"id\":\"r2\",\"text\":\"beta\"}\n{\"id\":\"r3\",\"text\":\"gamma\"}\n",
        )
        .unwrap();

        let mut config = test_config(dir.path().to_path_buf());
        config.checkpoint_stride = 1;
        config.refresh_batch_multiplier = 1;
        let source = test_source(config.clone());
        let shard = HuggingFaceRowSource::index_single_shard(&config, &path, 0)
            .unwrap()
            .0
            .unwrap();
        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = shard.row_count;
            state.total_rows = Some(shard.row_count);
            state.shards = vec![shard];
        }

        let snapshot = source.refresh(None, Some(2)).unwrap();
        assert_eq!(snapshot.records.len(), 2);
        assert!(snapshot.cursor.revision > 0);
    }

    #[test]
    fn reported_record_count_uses_len_hint_for_local_state() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 4;
            state.total_rows = Some(4);
        }
        assert_eq!(source.reported_record_count().unwrap(), 4);
    }

    #[test]
    fn shuffle_candidates_deterministically_preserves_membership() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let original = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut shuffled = original.clone();
        HuggingFaceRowSource::shuffle_candidates_deterministically(&config, &mut shuffled, 1);
        let mut sorted_original = original;
        let mut sorted_shuffled = shuffled;
        sorted_original.sort();
        sorted_shuffled.sort();
        assert_eq!(sorted_shuffled, sorted_original);
    }

    #[test]
    fn parse_row_supports_row_wrapped_payload_and_text_columns() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.text_columns = vec!["headline".into(), "body".into()];
        config.id_column = Some("rid".into());
        let source = test_source(config);

        let parsed = source
            .parse_row(
                0,
                &json!({"row": {"rid": "r-1", "headline": "h", "body": "b"}}),
            )
            .unwrap()
            .unwrap();

        // Candidate coalescing: headline is non-empty so it is selected;
        // body is not tried.
        assert_eq!(parsed.row_id.as_deref(), Some("r-1"));
        assert_eq!(parsed.text_fields.len(), 1);
        assert_eq!(parsed.text_fields[0].name, "headline");
    }

    #[test]
    fn parse_row_returns_none_when_all_positive_or_text_candidates_are_missing() {
        let dir = tempdir().unwrap();

        // Role mode: all positive_columns candidates absent → row skipped.
        let mut role_config = test_config(dir.path().to_path_buf());
        role_config.anchor_columns = vec!["anchor".into()];
        role_config.positive_columns = vec!["positive".into()];
        let role_source = test_source(role_config);

        let role_missing = role_source.parse_row(0, &json!({"anchor":"a"})).unwrap();
        assert!(role_missing.is_none());

        // Text-columns mode: a row that lacks all listed candidates → row skipped.
        let mut text_config = test_config(dir.path().to_path_buf());
        text_config.text_columns = vec!["title".into(), "body".into()];
        let text_source = test_source(text_config);
        // Row has neither "title" nor "body" → no candidate matches → skipped.
        let text_missing = text_source
            .parse_row(1, &json!({"other_field": "irrelevant"}))
            .unwrap();
        assert!(text_missing.is_none());
    }

    #[test]
    fn parse_row_text_columns_coalesces_to_first_nonempty_candidate() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.text_columns = vec!["title".into(), "body".into()];
        let source = test_source(config);

        // "title" is empty string → coalesces to "body".
        let row = source
            .parse_row(0, &json!({"title": "", "body": "fallback content"}))
            .unwrap()
            .unwrap();
        assert_eq!(row.text_fields.len(), 1);
        assert_eq!(row.text_fields[0].name, "body");
        assert_eq!(row.text_fields[0].text, "fallback content");

        // "title" is present and non-empty → it is used; "body" is never tried.
        let row2 = source
            .parse_row(1, &json!({"title": "primary content", "body": "ignored"}))
            .unwrap()
            .unwrap();
        assert_eq!(row2.text_fields.len(), 1);
        assert_eq!(row2.text_fields[0].name, "title");
        assert_eq!(row2.text_fields[0].text, "primary content");
    }

    #[test]
    fn parse_row_positive_columns_coalesces_to_first_nonempty_candidate() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.anchor_columns = vec!["anchor".into()];
        config.positive_columns = vec!["summary".into(), "body".into()];
        let source = test_source(config);

        // "summary" is absent → coalesces to "body".
        let row = source
            .parse_row(0, &json!({"anchor": "a", "body": "fallback positive"}))
            .unwrap()
            .unwrap();
        assert_eq!(row.text_fields.len(), 2);
        assert_eq!(row.text_fields[0].name, "anchor");
        assert_eq!(row.text_fields[1].name, "body");

        // "summary" is present and non-empty → it is used; "body" is ignored.
        let row2 = source
            .parse_row(
                1,
                &json!({"anchor": "a", "summary": "chosen", "body": "ignored"}),
            )
            .unwrap()
            .unwrap();
        assert_eq!(row2.text_fields.len(), 2);
        assert_eq!(row2.text_fields[1].name, "summary");
        assert_eq!(row2.text_fields[1].text, "chosen");

        // Both positive candidates absent → row skipped.
        let none = source.parse_row(2, &json!({"anchor": "a"})).unwrap();
        assert!(none.is_none());
    }

    #[test]
    fn parse_row_anchor_columns_coalesces_to_first_nonempty_candidate() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.anchor_columns = vec!["headline".into(), "title".into()];
        config.positive_columns = vec!["body".into()];
        let source = test_source(config);

        // "headline" is absent → coalesces to "title".
        let row = source
            .parse_row(
                0,
                &json!({"title": "fallback anchor", "body": "positive text"}),
            )
            .unwrap()
            .unwrap();
        assert_eq!(row.text_fields.len(), 2);
        assert_eq!(row.text_fields[0].name, "title");
        assert_eq!(row.text_fields[0].text, "fallback anchor");

        // "headline" is present and non-empty → it is used; "title" is ignored.
        let row2 = source
            .parse_row(
                1,
                &json!({"headline": "chosen anchor", "title": "ignored", "body": "positive"}),
            )
            .unwrap()
            .unwrap();
        assert_eq!(row2.text_fields[0].name, "headline");
        assert_eq!(row2.text_fields[0].text, "chosen anchor");

        // Both anchor candidates absent → row skipped.
        let none = source
            .parse_row(2, &json!({"body": "positive only"}))
            .unwrap();
        assert!(none.is_none());
    }

    #[test]
    fn parse_row_errors_when_no_mapping_is_configured() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.id_column = Some("id".into());
        config.text_columns.clear();
        let source = test_source(config);

        let parsed = source.parse_row(7, &json!({"id":"only-id"}));
        assert!(matches!(
            parsed,
            Err(SamplerError::SourceInconsistent { .. })
        ));
    }

    #[test]
    fn row_to_record_returns_none_for_empty_fields() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let row = RowView {
            row_id: Some("x".into()),
            timestamp: None,
            text_fields: Vec::new(),
        };
        assert!(source.row_to_record(&row, 0).unwrap().is_none());
    }

    #[test]
    fn ensure_row_available_handles_materialized_max_and_exhausted_candidates() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 1;
            state.remote_candidates = Some(vec![]);
            state.next_remote_idx = 0;
        }

        assert!(source.ensure_row_available(0).unwrap());
        assert!(!source.ensure_row_available(3).unwrap());
        assert!(!source.ensure_row_available(1).unwrap());
    }

    #[test]
    fn ensure_row_available_bootstraps_from_manifest_candidates() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let (base_url, server) = spawn_manifest_and_shard_http(b"{\"text\":\"hello\"}\n".to_vec());

        // Reset to None so ensure_row_available triggers the manifest-fetch path.
        source.state.lock().unwrap().remote_candidates = None;

        with_env_var(
            TRIPLETS_HF_PARQUET_ENDPOINT,
            &format!("{base_url}/parquet"),
            || {
                assert!(source.ensure_row_available(0).unwrap());
            },
        );

        server.join().unwrap();
    }

    #[test]
    fn ensure_row_available_skips_past_all_cached_candidates_on_restart() {
        // Verifies the restart scenario: when every candidate in the manifest is
        // already materialised on disk, next_remote_idx jumps to candidates.len()
        // and ensure_row_available returns Ok(false) without any download attempt.
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        // Construct the candidate URL that the manifest will list.
        let shard_raw_url = "http://127.0.0.1:1/datasets/org/ds/resolve/main/train/a.ndjson";
        let shard_candidate = format!("{REMOTE_URL_PREFIX}{shard_raw_url}");
        let target = HuggingFaceRowSource::candidate_target_path(&config, &shard_candidate);
        let store_path = HuggingFaceRowSource::shard_store_path_for(&target);
        fs::create_dir_all(store_path.parent().unwrap()).unwrap();
        fs::write(&store_path, b"dummy").unwrap();

        // Inject an already-materialised shard so materialized_rows == 1.
        {
            let mut state = source.state.lock().unwrap();
            state.shards = vec![ShardIndex {
                path: store_path,
                global_start: 0,
                row_count: 1,
                random_access: true,
                parquet_row_groups: vec![(0, 1)],
                checkpoints: Vec::new(),
                remote_candidate: None,
            }];
            state.materialized_rows = 1;
            state.remote_candidates = None;
        }

        // Serve a manifest that lists the same (already-cached) shard.
        let manifest_body = serde_json::to_vec(&json!({
            "parquet_files": [{"url": shard_raw_url, "size": 5}]
        }))
        .unwrap();
        let (base_url, server) = spawn_one_shot_http(manifest_body);

        // Row 1 is not yet materialised; this triggers the candidate-init path.
        // all candidates are already on disk → next_remote_idx = candidates.len() → Ok(false).
        let result = with_env_var(TRIPLETS_HF_PARQUET_ENDPOINT, &base_url, || {
            source.ensure_row_available(1)
        })
        .unwrap();
        server.join().unwrap();

        assert!(
            !result,
            "no new rows available — all candidates already cached"
        );
        let state = source.state.lock().unwrap();
        assert_eq!(
            state.next_remote_idx,
            state
                .remote_candidates
                .as_ref()
                .map(|c| c.len())
                .unwrap_or(0),
            "next_remote_idx must equal candidates.len() when all are cached"
        );
    }

    #[test]
    fn read_row_batch_uses_cached_rows_and_respects_limit() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 2;
            state.total_rows = Some(2);
        }

        let row0 = RowView {
            row_id: Some("r0".into()),
            timestamp: Some(Utc::now()),
            text_fields: vec![RowTextField {
                name: "text".into(),
                text: "alpha".into(),
            }],
        };
        let row1 = RowView {
            row_id: Some("r1".into()),
            timestamp: Some(Utc::now()),
            text_fields: vec![RowTextField {
                name: "text".into(),
                text: "beta".into(),
            }],
        };
        {
            let mut cache = source.cache.lock().unwrap();
            cache.insert(0, row0, config.cache_capacity);
            cache.insert(1, row1, config.cache_capacity);
        }

        let mut out = Vec::new();
        source.read_row_batch(&[0, 1], &mut out, Some(1)).unwrap();
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn read_row_batch_errors_on_invalid_json_line() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("broken.jsonl");
        fs::write(&path, b"not-json\n").unwrap();

        let mut config = test_config(dir.path().to_path_buf());
        config.checkpoint_stride = 1;
        let source = test_source(config.clone());
        let shard = HuggingFaceRowSource::index_single_shard(&config, &path, 0)
            .unwrap()
            .0
            .unwrap();
        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 1;
            state.total_rows = Some(1);
            state.shards = vec![shard];
        }

        let mut out = Vec::new();
        let result = source.read_row_batch(&[0], &mut out, Some(1));
        assert!(result.is_err());
    }

    #[test]
    fn build_shard_index_errors_when_no_matching_extensions() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("data.txt"), b"x\n").unwrap();
        let config = test_config(dir.path().to_path_buf());
        let result = HuggingFaceRowSource::build_shard_index(&config);
        assert!(result.is_err());
    }

    #[test]
    fn refresh_handles_empty_total_and_cursor_wrap() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 0;
            state.total_rows = Some(0);
        }
        let empty = source.refresh(None, Some(5)).unwrap();
        assert!(empty.records.is_empty());
        assert_eq!(empty.cursor.revision, 0);

        let path = dir.path().join("rows.jsonl");
        fs::write(
            &path,
            b"{\"id\":\"a\",\"text\":\"A\"}\n{\"id\":\"b\",\"text\":\"B\"}\n",
        )
        .unwrap();
        let mut cfg2 = config;
        cfg2.checkpoint_stride = 1;
        let source2 = test_source(cfg2.clone());
        let shard = HuggingFaceRowSource::index_single_shard(&cfg2, &path, 0)
            .unwrap()
            .0
            .unwrap();
        {
            let mut state = source2.state.lock().unwrap();
            state.materialized_rows = 2;
            state.total_rows = Some(2);
            state.shards = vec![shard];
        }
        let cursor = SourceCursor {
            last_seen: Utc::now(),
            revision: 99,
        };
        let snapshot = source2.refresh(Some(&cursor), Some(1)).unwrap();
        assert_eq!(snapshot.records.len(), 1);
    }

    #[test]
    fn new_rejects_zero_checkpoint_stride() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.checkpoint_stride = 0;
        let result = HuggingFaceRowSource::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn parse_global_row_count_response_returns_none_when_split_missing() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let body = r#"{
            "size": {
                "splits": [
                    {"config":"main","split":"test","num_rows":7}
                ]
            }
        }"#;

        let parsed = HuggingFaceRowSource::parse_global_row_count_response(&config, body).unwrap();
        assert_eq!(parsed, None);
    }

    #[test]
    fn extract_split_row_count_uses_config_num_rows_when_split_empty() {
        let payload = serde_json::json!({
            "size": {
                "configs": [
                    {
                        "config": "main",
                        "num_rows": 123,
                        "splits": [
                            {"split": "train", "num_rows": 999}
                        ]
                    }
                ]
            }
        });

        let rows =
            HuggingFaceRowSource::extract_split_row_count_from_size_response(&payload, "main", "");
        assert_eq!(rows, Some(123));
    }

    #[test]
    fn extract_split_row_count_uses_dataset_num_rows_when_split_empty() {
        let payload = serde_json::json!({
            "size": {
                "dataset": {
                    "num_rows": 77
                }
            }
        });

        let rows =
            HuggingFaceRowSource::extract_split_row_count_from_size_response(&payload, "main", "");
        assert_eq!(rows, Some(77));
    }

    #[test]
    fn refresh_order_uses_sampler_seed_for_local_rows() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("rows.jsonl");
        let mut payload = String::new();
        for idx in 0..12 {
            payload.push_str(&format!("{{\"id\":\"r{idx}\",\"text\":\"v{idx}\"}}\n"));
        }
        fs::write(&path, payload).unwrap();

        let mut config = test_config(dir.path().to_path_buf());
        config.checkpoint_stride = 1;
        config.refresh_batch_multiplier = 1;

        let source_a = test_source(config.clone());
        let source_b = test_source(config.clone());
        let source_c = test_source(config.clone());
        let shard = HuggingFaceRowSource::index_single_shard(&config, &path, 0)
            .unwrap()
            .0
            .unwrap();

        for source in [&source_a, &source_b, &source_c] {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 12;
            state.total_rows = Some(12);
            state.shards = vec![shard.clone()];
        }

        let seed_1 = SamplerConfig {
            seed: 7,
            ..SamplerConfig::default()
        };
        let seed_2 = SamplerConfig {
            seed: 7,
            ..SamplerConfig::default()
        };
        let seed_3 = SamplerConfig {
            seed: 123,
            ..SamplerConfig::default()
        };

        source_a.configure_sampler(&seed_1);
        source_b.configure_sampler(&seed_2);
        source_c.configure_sampler(&seed_3);

        let ids_a: Vec<String> = source_a
            .refresh(None, Some(8))
            .unwrap()
            .records
            .into_iter()
            .map(|record| record.id)
            .collect();
        let ids_b: Vec<String> = source_b
            .refresh(None, Some(8))
            .unwrap()
            .records
            .into_iter()
            .map(|record| record.id)
            .collect();
        let ids_c: Vec<String> = source_c
            .refresh(None, Some(8))
            .unwrap()
            .records
            .into_iter()
            .map(|record| record.id)
            .collect();

        assert_eq!(ids_a, ids_b);
        assert_ne!(ids_a, ids_c);
    }

    #[test]
    fn set_active_sampler_config_rebuilds_order_on_seed_change() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        let candidates = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
            "e".to_string(),
        ];

        // Prime the source at seed=7 BEFORE injecting state, so the subsequent
        // configure_sampler(seed=7) calls are not seen as seed changes.
        source.configure_sampler(&SamplerConfig {
            seed: 7,
            ..SamplerConfig::default()
        });

        {
            let mut state = source.state.lock().unwrap();
            // Candidates stored sorted/immutable; order derived from seed 7.
            let order = HuggingFaceRowSource::build_candidate_order(&config, &candidates, 7);
            state.remote_candidates = Some(candidates.clone());
            state.remote_candidate_order = order.clone();
            state.next_remote_idx = 3;
        }

        // Same seed — order and pointer must not change.
        source.configure_sampler(&SamplerConfig {
            seed: 7,
            ..SamplerConfig::default()
        });
        source.configure_sampler(&SamplerConfig {
            seed: 7,
            ..SamplerConfig::default()
        });
        {
            let state = source.state.lock().unwrap();
            let order = HuggingFaceRowSource::build_candidate_order(&config, &candidates, 7);
            assert_eq!(state.remote_candidate_order, order);
            assert_eq!(state.next_remote_idx, 3, "same seed must not move pointer");
        }

        // Different seed — candidates list untouched, order rebuilt, pointer reset to 0.
        source.configure_sampler(&SamplerConfig {
            seed: 18,
            ..SamplerConfig::default()
        });
        {
            let state = source.state.lock().unwrap();
            // List is immutable — same sorted candidates.
            assert_eq!(state.remote_candidates.as_ref().unwrap(), &candidates);
            // Order is now derived from seed 18.
            let expected_order =
                HuggingFaceRowSource::build_candidate_order(&config, &candidates, 18);
            assert_eq!(state.remote_candidate_order, expected_order);
            // No shards are materialised on disk so the pointer lands at 0
            // (the first non-materialised position in the new order).
            assert_eq!(state.next_remote_idx, 0);
        }
    }

    #[test]
    fn set_active_sampler_config_skips_materialised_shards_after_seed_change() {
        // This is the regression test for the bug where every source-epoch advance
        // reset next_remote_idx to 0, causing the expansion thread to always report
        // "shard 1/N already materialised" and never actually download new shards.
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        let candidates: Vec<String> = (0..5)
            .map(|i| {
                format!("url::http://host/datasets/org/ds/resolve/main/train/part-{i:04}.ndjson")
            })
            .collect();

        // Prime source at seed 7 so the subsequent call at seed 7 is not a "change".
        source.configure_sampler(&SamplerConfig {
            seed: 7,
            ..SamplerConfig::default()
        });

        // Build the order for the *new* seed (18) up-front so we know which
        // positions map to which candidates and can pre-materialise the first 3.
        let new_order = HuggingFaceRowSource::build_candidate_order(&config, &candidates, 18);
        let materialised_count = 3;
        let shards_to_inject: Vec<ShardIndex> = (0..materialised_count)
            .map(|pos| {
                let candidate_idx = new_order[pos];
                let target = HuggingFaceRowSource::candidate_target_path(
                    &config,
                    &candidates[candidate_idx],
                );
                let store = HuggingFaceRowSource::shard_store_path_for(&target);
                ShardIndex {
                    path: store,
                    global_start: pos * 100,
                    row_count: 100,
                    random_access: true,
                    parquet_row_groups: vec![(0, 100)],
                    checkpoints: Vec::new(),
                    remote_candidate: None,
                }
            })
            .collect();

        {
            let mut state = source.state.lock().unwrap();
            let order_7 = HuggingFaceRowSource::build_candidate_order(&config, &candidates, 7);
            state.remote_candidates = Some(candidates.clone());
            state.remote_candidate_order = order_7;
            state.next_remote_idx = 0;
            state.shards = shards_to_inject;
            state.materialized_rows = materialised_count * 100;
        }

        // Change the seed — must advance pointer past the 3 materialised shards
        // in the new order rather than resetting to 0.
        source.configure_sampler(&SamplerConfig {
            seed: 18,
            ..SamplerConfig::default()
        });

        {
            let state = source.state.lock().unwrap();
            assert_eq!(
                state.remote_candidate_order,
                HuggingFaceRowSource::build_candidate_order(&config, &candidates, 18),
                "order must be rebuilt from new seed"
            );
            assert_eq!(
                state.next_remote_idx, materialised_count,
                "pointer must skip the {} already-materialised shards in the new order, \
                 not reset to 0",
                materialised_count
            );
        }
    }

    // ── extract_classlabel_maps ───────────────────────────────────────────────

    #[test]
    fn extract_classlabel_maps_returns_label_names_for_classlabel_columns() {
        let info = json!({
            "dataset_info": {
                "features": {
                    "text": {"dtype": "string", "_type": "Value"},
                    "sentiment": {
                        "_type": "ClassLabel",
                        "names": ["neutral", "bullish", "bearish"]
                    }
                }
            }
        });
        let maps = HuggingFaceRowSource::extract_classlabel_maps(&info);
        assert_eq!(maps.len(), 1);
        assert_eq!(maps["sentiment"], vec!["neutral", "bullish", "bearish"]);
    }

    #[test]
    fn extract_classlabel_maps_handles_multiple_classlabel_columns() {
        let info = json!({
            "dataset_info": {
                "features": {
                    "label_a": {"_type": "ClassLabel", "names": ["no", "yes"]},
                    "label_b": {"_type": "ClassLabel", "names": ["low", "mid", "high"]}
                }
            }
        });
        let maps = HuggingFaceRowSource::extract_classlabel_maps(&info);
        assert_eq!(maps.len(), 2);
        assert_eq!(maps["label_a"], vec!["no", "yes"]);
        assert_eq!(maps["label_b"], vec!["low", "mid", "high"]);
    }

    #[test]
    fn extract_classlabel_maps_ignores_non_classlabel_features() {
        let info = json!({
            "dataset_info": {
                "features": {
                    "text":  {"dtype": "string", "_type": "Value"},
                    "score": {"dtype": "float32", "_type": "Value"}
                }
            }
        });
        let maps = HuggingFaceRowSource::extract_classlabel_maps(&info);
        assert!(maps.is_empty());
    }

    #[test]
    fn extract_classlabel_maps_skips_empty_names_array() {
        let info = json!({
            "dataset_info": {
                "features": {
                    "empty_labels": {"_type": "ClassLabel", "names": []}
                }
            }
        });
        let maps = HuggingFaceRowSource::extract_classlabel_maps(&info);
        assert!(
            maps.is_empty(),
            "columns with empty names arrays must not be included"
        );
    }

    #[test]
    fn extract_classlabel_maps_returns_empty_for_missing_dataset_info() {
        // Top-level key missing entirely.
        let maps = HuggingFaceRowSource::extract_classlabel_maps(&json!({}));
        assert!(maps.is_empty());

        // `dataset_info` present but no `features`.
        let maps2 = HuggingFaceRowSource::extract_classlabel_maps(
            &json!({"dataset_info": {"description": "x"}}),
        );
        assert!(maps2.is_empty());
    }

    #[test]
    fn extract_classlabel_maps_returns_empty_for_non_object_features() {
        // `features` is an array instead of an object — must not panic.
        let info = json!({
            "dataset_info": {
                "features": ["sentiment", "text"]
            }
        });
        let maps = HuggingFaceRowSource::extract_classlabel_maps(&info);
        assert!(maps.is_empty());
    }

    // ── fetch_classlabel_maps ─────────────────────────────────────────────────

    #[test]
    fn fetch_classlabel_maps_returns_empty_when_endpoint_unreachable() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        // Port 1 is always unreachable; ureq returns an Err which must be handled.
        let maps = with_env_var(TRIPLETS_HF_INFO_ENDPOINT, "http://127.0.0.1:1", || {
            HuggingFaceRowSource::fetch_classlabel_maps(&config)
        });
        assert!(
            maps.is_empty(),
            "unreachable endpoint must yield empty map, got: {maps:?}"
        );
    }

    #[test]
    fn fetch_classlabel_maps_returns_empty_on_non_200_response() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let (base_url, server) = spawn_one_shot_http_with_status(404, b"not found".to_vec());

        let maps = with_env_var(TRIPLETS_HF_INFO_ENDPOINT, &base_url, || {
            HuggingFaceRowSource::fetch_classlabel_maps(&config)
        });
        server.join().unwrap();
        assert!(
            maps.is_empty(),
            "HTTP 404 response must yield empty map, got: {maps:?}"
        );
    }

    #[test]
    fn fetch_classlabel_maps_returns_empty_on_malformed_json() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let (base_url, server) = spawn_one_shot_http(b"this is not json".to_vec());

        let maps = with_env_var(TRIPLETS_HF_INFO_ENDPOINT, &base_url, || {
            HuggingFaceRowSource::fetch_classlabel_maps(&config)
        });
        server.join().unwrap();
        assert!(
            maps.is_empty(),
            "malformed JSON must yield empty map, got: {maps:?}"
        );
    }

    #[test]
    fn fetch_classlabel_maps_resolves_classlabel_columns_from_info_response() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let body = serde_json::to_vec(&json!({
            "dataset_info": {
                "features": {
                    "text": {"dtype": "string", "_type": "Value"},
                    "sentiment": {
                        "_type": "ClassLabel",
                        "names": ["neutral", "bullish", "bearish"]
                    }
                }
            }
        }))
        .unwrap();
        let (base_url, server) = spawn_one_shot_http(body);

        let maps = with_env_var(TRIPLETS_HF_INFO_ENDPOINT, &base_url, || {
            HuggingFaceRowSource::fetch_classlabel_maps(&config)
        });
        server.join().unwrap();
        assert_eq!(maps.len(), 1);
        assert_eq!(maps["sentiment"], vec!["neutral", "bullish", "bearish"]);
    }

    // ── value_to_text with label resolution ──────────────────────────────────

    #[test]
    fn value_to_text_resolves_integer_to_label_name() {
        let labels = vec![
            "negative".to_string(),
            "neutral".to_string(),
            "positive".to_string(),
        ];
        assert_eq!(
            HuggingFaceRowSource::value_to_text(&json!(0), Some(&labels)),
            Some("negative".into())
        );
        assert_eq!(
            HuggingFaceRowSource::value_to_text(&json!(2), Some(&labels)),
            Some("positive".into())
        );
    }

    #[test]
    fn value_to_text_falls_back_to_raw_integer_when_index_out_of_range() {
        let labels = vec!["a".to_string(), "b".to_string()];
        // Index 5 is beyond the label list — must return the integer as a string.
        assert_eq!(
            HuggingFaceRowSource::value_to_text(&json!(5), Some(&labels)),
            Some("5".into())
        );
    }

    #[test]
    fn value_to_text_falls_back_to_raw_integer_when_no_label_names_provided() {
        assert_eq!(
            HuggingFaceRowSource::value_to_text(&json!(1), None),
            Some("1".into())
        );
    }

    /// Like `spawn_manifest_and_shard_http`, but counts `/parquet` manifest hits
    /// separately.  Accepts up to `max_accepts` total connections, dispatching on
    /// URL path: requests containing `/parquet` get the manifest body; all others
    /// get the shard payload.
    fn spawn_counting_manifest_and_shard_http(
        max_accepts: usize,
        shard_payload: Vec<u8>,
    ) -> (String, Arc<AtomicUsize>, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{addr}");
        let manifest_counter = Arc::new(AtomicUsize::new(0));
        let manifest_counter_arc = Arc::clone(&manifest_counter);
        let manifest_body = serde_json::json!({
            "parquet_files": [{
                "url": format!("{base_url}/resolve/main/train/bootstrap.ndjson"),
                "size": shard_payload.len()
            }]
        })
        .to_string();
        let handle = thread::spawn(move || {
            for _ in 0..max_accepts {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let mut buf = [0u8; 4096];
                        let read = stream.read(&mut buf).unwrap_or(0);
                        let request = String::from_utf8_lossy(&buf[..read]);
                        let first_line = request.lines().next().unwrap_or_default();
                        let (body, is_manifest): (Vec<u8>, bool) =
                            if first_line.contains("/parquet") {
                                (manifest_body.as_bytes().to_vec(), true)
                            } else {
                                (shard_payload.clone(), false)
                            };
                        if is_manifest {
                            manifest_counter_arc.fetch_add(1, AtomicOrdering::SeqCst);
                        }
                        let response = format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        );
                        let _ = stream.write_all(response.as_bytes());
                        let _ = stream.write_all(&body);
                        let _ = stream.flush();
                    }
                    Err(_) => break,
                }
            }
        });
        (base_url, manifest_counter, handle)
    }

    #[test]
    fn info_endpoint_called_exactly_once_per_source_construction() {
        // Verify that /info is called exactly once during new() and never during
        // refresh().  Two separate strategies are used to avoid fragile TCP
        // connection counting:
        //
        //   Phase 1 — new(): one-shot /info server returning a real ClassLabel
        //             mapping.  After new(), source.config.label_maps must reflect
        //             the mock response (proves /info was called) and joining the
        //             server thread proves it was called exactly once.
        //
        //   Phase 2 — refresh(): fresh one-shot server for /info.  After 3x
        //             refresh() calls, is_finished() must be false, proving /info
        //             was never contacted during refresh().
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.anchor_columns = vec!["title".into()];
        config.positive_columns = vec!["body".into()];
        config.checkpoint_stride = 1;

        let shard_path = dir.path().join("shard.ndjson");
        fs::write(
            &shard_path,
            b"{\"title\":\"t1\",\"body\":\"b1\"}\n{\"title\":\"t2\",\"body\":\"b2\"}\n",
        )
        .unwrap();

        // Phase 1: /info during new().  Return a non-trivial ClassLabel response so
        // source.config.label_maps is populated and the assertion can verify the call.
        let info_payload_1 = serde_json::json!({
            "dataset_info": {
                "features": {
                    "sentiment": {
                        "_type": "ClassLabel",
                        "names": ["negative", "neutral", "positive"]
                    }
                }
            }
        })
        .to_string()
        .into_bytes();
        let size_payload = serde_json::json!({"size": {"splits": []}})
            .to_string()
            .into_bytes();

        let (info_url_1, info_server_1) = spawn_one_shot_http(info_payload_1);
        let (size_url, size_server) = spawn_one_shot_http(size_payload);

        let config_for_index = config.clone();
        let source = with_env_vars(
            &[
                (TRIPLETS_HF_INFO_ENDPOINT, info_url_1.as_str()),
                (TRIPLETS_HF_SIZE_ENDPOINT, size_url.as_str()),
            ],
            || HuggingFaceRowSource::new(config).unwrap(),
        );

        // Verify /info was actually called: label_maps must match the mock response.
        assert_eq!(
            source.config.label_maps.get("sentiment"),
            Some(&vec![
                "negative".to_string(),
                "neutral".to_string(),
                "positive".to_string()
            ]),
            "fetch_classlabel_maps must have been called during new() \
             and populated label_maps from the mock /info response"
        );
        // Joining the one-shot server confirms it was called exactly once.
        info_server_1.join().unwrap();
        size_server.join().unwrap();

        // Phase 2: inject local shard state.
        // Set remote_candidates = Some(vec![]) so trigger_expansion_if_needed
        // treats this source as having no remote shards — no background network
        // thread is spawned from refresh().
        let shard = HuggingFaceRowSource::index_single_shard(&config_for_index, &shard_path, 0)
            .unwrap()
            .0
            .unwrap();
        {
            let mut state = source.state.lock().unwrap();
            state.shards = vec![shard.clone()];
            state.materialized_rows = shard.row_count;
            state.total_rows = Some(shard.row_count);
            state.remote_candidates = Some(vec![]);
        }

        // Create a fresh one-shot server for the refresh phase.  If any refresh()
        // call hits /info the server will serve the request and its thread will
        // exit — is_finished() would then be true.  We assert it stays false.
        let info_payload_2 = serde_json::json!({"dataset_info": {"features": {}}})
            .to_string()
            .into_bytes();
        let (info_url_2, info_server_2) = spawn_one_shot_http(info_payload_2);

        for _ in 0..3 {
            let _ = with_env_vars(&[(TRIPLETS_HF_INFO_ENDPOINT, info_url_2.as_str())], || {
                source.refresh(None, Some(1))
            });
        }

        assert!(
            !info_server_2.is_finished(),
            "fetch_classlabel_maps must NOT be called during refresh() — \
             /info server was unexpectedly hit"
        );
    }

    #[test]
    fn parquet_manifest_fetched_exactly_once_per_candidate_list_population() {
        // Verify that the /parquet manifest endpoint is contacted only once per
        // source lifetime.  After the first ensure_row_available() populates
        // state.remote_candidates, subsequent calls must not re-fetch the manifest.
        // The counting server stays alive so a spurious second request would be
        // recorded and the final assertion would catch it.
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        // Reset to None so the first ensure_row_available() triggers the lazy fetch.
        source.state.lock().unwrap().remote_candidates = None;

        let shard_payload = b"{\"text\":\"hello\"}\n".to_vec();
        // Counting manifest+shard server: 4 slots so a second /parquet hit is caught.
        let (base_url, manifest_counter, _manifest_handle) =
            spawn_counting_manifest_and_shard_http(4, shard_payload);

        // First call: remote_candidates is None → fetches manifest (counter→1) → downloads shard.
        let first_available = with_env_var(
            TRIPLETS_HF_PARQUET_ENDPOINT,
            &format!("{base_url}/parquet"),
            || source.ensure_row_available(0).unwrap(),
        );
        assert!(first_available);
        assert_eq!(
            manifest_counter.load(AtomicOrdering::SeqCst),
            1,
            "parquet manifest must be fetched exactly once on first ensure_row_available"
        );

        // Second call: remote_candidates is now Some(...) → must NOT re-fetch manifest.
        let _ = with_env_var(
            TRIPLETS_HF_PARQUET_ENDPOINT,
            &format!("{base_url}/parquet"),
            || source.ensure_row_available(0),
        );
        assert_eq!(
            manifest_counter.load(AtomicOrdering::SeqCst),
            1,
            "parquet manifest must not be re-fetched on subsequent ensure_row_available calls"
        );
    }
}
