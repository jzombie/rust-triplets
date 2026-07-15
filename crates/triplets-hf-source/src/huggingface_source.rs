use cache_manager::{CacheRoot, EvictPolicy};
use flate2::read::GzDecoder;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::reader::RowIter;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use simd_r_drive::storage_engine::DataStore;
use simd_r_drive::storage_engine::traits::{DataStoreReader, DataStoreWriter};
use siphasher::sip::SipHasher;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fs;
use std::fs::File;
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::path::PathBuf;
#[cfg(test)]
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
#[cfg(test)]
use std::time::SystemTime;
use std::time::{Duration, Instant};
#[cfg(test)]
use tempfile::TempDir;
use tracing::{debug, info, warn};
use walkdir::WalkDir;

use crate::constants::{
    ENV_TRIPLETS_HF_TOKEN, ENV_TRIPLETS_HF_WHOAMI_ENDPOINT, HF_ALL_SPLITS_DIR,
    HF_DATASETS_BASE_URL, HF_GROUP, HF_HTTP_CONNECT_TIMEOUT_SECS, HF_HTTP_REQUEST_TIMEOUT_SECS,
    HF_LOCAL_DISK_CAP_BYTES, HF_PARQUET_DEFAULT_ENDPOINT, HF_PARQUET_MANIFEST_DIR,
    HF_REFRESH_BATCH_MULTIPLIER, HF_REMOTE_BOOTSTRAP_SHARDS,
    HF_REMOTE_EXPANSION_HEADROOM_MULTIPLIER, HF_REMOTE_URL_PREFIX, HF_RESOLVE_URL_SEPARATOR,
    HF_SHARD_CANDIDATE_SEED_TAG, HF_SHARD_STORE_EXTENSION, HF_SHARD_STORE_META_ROWS_KEY,
    HF_SHARD_STORE_ROW_PREFIX, HF_SHARD_STORE_SOURCE_SIZE_KEY, HF_SHARED_RUNTIME_WORKER_THREADS,
    HF_WHOAMI_DEFAULT_ENDPOINT,
};
#[cfg(not(debug_assertions))]
use crate::constants::{
    HF_THROTTLE_ADAPTIVE_JITTER_MS, HF_THROTTLE_BASE_DELAY_MS, HF_THROTTLE_MAX_CONCURRENT,
    HF_THROTTLE_MAX_RETRIES,
};
use chrono::{DateTime, Utc};
use triplets_core::SamplerError;
use triplets_core::config::{NegativeStrategy, SamplerConfig, Selector, TripletRecipe};
use triplets_core::data::{DataRecord, QualityScore, SectionRole};
use triplets_core::utils::make_section;

use reqwest_drive::ClientWithMiddleware;

use triplets_core::source::{DataSource, SourceCursor, SourceSnapshot};

const HF_SOURCE_KEY_ANCHOR: &str = "anchor";
const HF_SOURCE_KEY_POSITIVE: &str = "positive";
const HF_SOURCE_KEY_NEGATIVE: &str = "negative";
const HF_SOURCE_KEY_CONTEXT: &str = "context";
const HF_SOURCE_KEY_TEXT: &str = "text";
const HF_SOURCE_KEY_TEXT_COLUMNS: &str = "text_columns";
const HF_SOURCE_KEY_TRUST: &str = "trust";
const HF_SOURCE_KEY_WEIGHT: &str = "weight";
const HF_SOURCE_KEY_SOURCE_ID: &str = "source_id";

/// Default HF text-columns-mode SimCSE-style recipe name.
pub const HF_RECIPE_TEXT_SIMCSE_WRONG_ARTICLE: &str = "huggingface_text_simcse_wrong_article";
const HF_RECIPE_ANCHOR_CONTEXT_WRONG_ARTICLE: &str = "huggingface_anchor_context_wrong_article";
const HF_RECIPE_ANCHOR_ANCHOR_WRONG_ARTICLE: &str = "huggingface_anchor_anchor_wrong_article";

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
    // Empty split (all-splits mode) uses HF_ALL_SPLITS_DIR so the path hierarchy stays valid
    // and won't collide with a split literally named "" on any filesystem.
    let split_dir = if split.is_empty() {
        HF_ALL_SPLITS_DIR
    } else {
        split
    };
    ensure_cache_group(
        PathBuf::from(HF_GROUP)
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
        HF_ALL_SPLITS_DIR
    } else {
        split
    };
    ensure_cache_group(
        PathBuf::from(HF_GROUP)
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
    /// Negative candidate columns (ordered).
    ///
    /// Used only in **role-based mode**.  Every listed column is required: if
    /// any is missing or blank the row is skipped.
    ///
    /// When a column value is a list (e.g. `["neg1", "neg2"]`), each element
    /// is expanded into a separate `SectionRole::Context` section.  This
    /// supports HuggingFace "dict" datasets where negatives are provided as
    /// a list within the same row.
    pub negative_columns: Vec<String>,
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
    /// Optional weight for weighted source scheduling.
    ///
    /// When set, used by [`build_hf_sources_with_weights`] to populate a
    /// per-source weight map that callers pass to
    /// `Sampler::next_triplet_batch_with_weights` for weighted scheduling.
    /// Must be `> 0.0`.
    pub weight: Option<f32>,
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
            && self.negative_columns == other.negative_columns
            && self.context_columns == other.context_columns
            && self.text_columns == other.text_columns
            && self.source_id == other.source_id
            // Compare f32 bits so that identical bit patterns are considered equal.
            // Valid trust values are never NaN, so bit-level comparison is correct.
            && self.trust.map(f32::to_bits) == other.trust.map(f32::to_bits)
            && self.weight.map(f32::to_bits) == other.weight.map(f32::to_bits)
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
        negative_columns: Vec::new(),
        context_columns: Vec::new(),
        text_columns: Vec::new(),
        trust: None,
        weight: None,
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
            HF_SOURCE_KEY_ANCHOR => {
                entry.anchor_columns = parse_csv_fields(value);
            }
            HF_SOURCE_KEY_POSITIVE => {
                entry.positive_columns = parse_csv_fields(value);
            }
            HF_SOURCE_KEY_NEGATIVE => {
                entry.negative_columns = parse_csv_fields(value);
            }
            HF_SOURCE_KEY_CONTEXT => {
                entry.context_columns = parse_csv_fields(value);
            }
            HF_SOURCE_KEY_TEXT | HF_SOURCE_KEY_TEXT_COLUMNS => {
                entry.text_columns = parse_csv_fields(value);
            }
            HF_SOURCE_KEY_TRUST => {
                let t: f32 = value.parse().map_err(|_| {
                    format!("invalid trust value '{value}': expected a float in [0.0, 1.0]")
                })?;
                if !(0.0..=1.0).contains(&t) {
                    return Err(format!("trust value {t} is out of range [0.0, 1.0]"));
                }
                entry.trust = Some(t);
            }
            HF_SOURCE_KEY_SOURCE_ID => {
                if value.is_empty() {
                    return Err("source_id must not be empty".to_string());
                }
                entry.source_id = Some(value.to_string());
            }
            HF_SOURCE_KEY_WEIGHT => {
                let w: f32 = value.parse().map_err(|_| {
                    format!("invalid weight value '{value}': expected a positive float")
                })?;
                if w <= 0.0 {
                    return Err(format!("weight value {w} must be > 0.0"));
                }
                entry.weight = Some(w);
            }
            _ => {
                return Err(format!("unsupported mapping key '{raw_key}'"));
            }
        }
    }

    let has_explicit_mapping = !entry.anchor_columns.is_empty()
        || !entry.positive_columns.is_empty()
        || !entry.negative_columns.is_empty()
        || !entry.context_columns.is_empty()
        || !entry.text_columns.is_empty();
    if !has_explicit_mapping {
        return Err(format!(
            "source '{}' has no field mapping; expected at least one of anchor=, positive=, negative=, context=, text=",
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

/// Extract the inner format extension from a file path, handling compound
/// extensions like `.jsonl.gz`. Returns the innermost recognized format.
///
/// Examples:
/// - `file.jsonl.gz` → `Some("jsonl")`
/// - `file.parquet` → `Some("parquet")`
/// - `file.gz` → `None` (no inner format)
/// - `file.simdr` → `Some("simdr")`
fn resolve_inner_extension(path: &Path) -> Option<String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())?
        .to_ascii_lowercase();

    // Handle compression extensions
    if ext == "gz" {
        let stem_ext = path
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|stem| Path::new(stem).extension())
            .and_then(|e| e.to_str())?
            .to_ascii_lowercase();
        Some(stem_ext)
    } else {
        Some(ext)
    }
}

/// Check if a file path has a gzip compression extension.
fn is_gzip_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gz"))
}

/// Check if a file path is a text-based format that should be transcoded to .simdr.
fn is_transient_text(path: &Path) -> bool {
    resolve_inner_extension(path)
        .is_some_and(|ext| ext == "jsonl" || ext == "ndjson" || ext == "json" || ext == "txt")
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

    let mut shared_client: Option<ClientWithMiddleware> = None;

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
            let snapshot_dir = match managed_hf_list_snapshot_dir(&dataset, &config, &split, idx) {
                Ok(path) => path,
                Err(err) => {
                    eprintln!(
                        "Skipping Hugging Face source initialization for '{}': {}",
                        source.uri, err
                    );
                    return None;
                }
            };

            let mut hf =
                HuggingFaceRowsConfig::new(source_id, dataset, config, split, snapshot_dir);
            hf.anchor_columns = source.anchor_columns.clone();
            hf.positive_columns = source.positive_columns.clone();
            hf.negative_columns = source.negative_columns.clone();
            hf.context_columns = source.context_columns.clone();
            hf.text_columns = source.text_columns.clone();
            hf.trust_override = source.trust;
            if shared_client.is_none() {
                shared_client = HuggingFaceRowSource::build_http_client(&hf).ok();
            }
            hf.http_client = shared_client.clone();
            println!(
                "source {idx}: hf://{}/{}/{} -> anchor={:?}, positive={:?}, negative={:?}, context={:?}, text_columns={:?}",
                hf.dataset_name,
                hf.config_name,
                hf.split_name,
                hf.anchor_columns,
                hf.positive_columns,
                hf.negative_columns,
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

/// Build Hugging Face row sources from a parsed source list, returning
/// both the sources and a per-source weight map.
///
/// Entries with a `weight=` value in their URI are included in the returned
/// `HashMap<String, f32>` (keyed by source ID).  Callers pass this map to
/// `Sampler::next_triplet_batch_with_weights` for weighted scheduling.
pub fn build_hf_sources_with_weights(
    roots: &HfListRoots,
) -> (Vec<Box<dyn DataSource + 'static>>, HashMap<String, f32>) {
    let mut weights = HashMap::new();

    // Phase 1: compute auto-slugs for entries that don't have an explicit source_id.
    let base_slugs: Vec<Option<String>> = roots
        .sources
        .iter()
        .enumerate()
        .map(|(idx, source)| {
            if source.source_id.is_some() {
                None
            } else {
                Some(match parse_hf_uri(&source.uri) {
                    Ok((dataset, config, split)) => hf_source_id_slug(&dataset, &config, &split),
                    Err(_) => format!("hf_list_{idx}"),
                })
            }
        })
        .collect();

    // Phase 2: find duplicate slugs for disambiguation.
    let mut slug_count: HashMap<&str, usize> = HashMap::new();
    for slug in base_slugs.iter().flatten() {
        *slug_count.entry(slug.as_str()).or_insert(0) += 1;
    }
    let mut slug_idx: HashMap<&str, usize> = HashMap::new();

    let shared_client = std::sync::OnceLock::<ClientWithMiddleware>::new();

    let sources: Vec<Box<dyn DataSource + 'static>> = roots
        .sources
        .iter()
        .enumerate()
        .filter_map(|(idx, source)| {
            let (dataset, config, split) = match parse_hf_uri(&source.uri) {
                Ok(v) => v,
                Err(err) => {
                    eprintln!("Skipping Hugging Face source '{}': {}", source.uri, err);
                    return None;
                }
            };

            let source_id = if let Some(ref sid) = source.source_id {
                sid.clone()
            } else {
                let base = base_slugs[idx].as_deref().unwrap_or("hf");
                let count = slug_count.get(base).copied().unwrap_or(0);
                if count > 1 {
                    let i = slug_idx.entry(base).or_insert(0);
                    let id = format!("{base}.{i}");
                    *i += 1;
                    id
                } else {
                    base.to_string()
                }
            };

            let snapshot_dir = match managed_hf_list_snapshot_dir(&dataset, &config, &split, idx) {
                Ok(dir) => dir,
                Err(err) => {
                    eprintln!("Skipping Hugging Face source '{}': {}", source.uri, err);
                    return None;
                }
            };

            let mut hf =
                HuggingFaceRowsConfig::new(source_id, dataset, config, split, snapshot_dir);
            hf.anchor_columns = source.anchor_columns.clone();
            hf.positive_columns = source.positive_columns.clone();
            hf.negative_columns = source.negative_columns.clone();
            hf.context_columns = source.context_columns.clone();
            hf.text_columns = source.text_columns.clone();
            hf.trust_override = source.trust;
            let client = shared_client.get_or_init(|| {
                HuggingFaceRowSource::build_http_client(&hf).unwrap_or_else(|_| {
                    HuggingFaceRowSource::build_http_client(&hf).expect("http client")
                })
            });
            hf.http_client = Some(client.clone());

            // Record weight if set.
            if let Some(w) = source.weight {
                weights.insert(hf.source_id.clone(), w);
            }

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
        .collect();

    (sources, weights)
}

/// Shared handle to the open-store cache.  Stored on `HuggingFaceRowsConfig`
/// so all methods have access without passing it separately.
#[derive(Clone)]
pub struct StoreCache(pub(crate) Arc<Mutex<HashMap<PathBuf, Arc<DataStore>>>>);

impl std::fmt::Debug for StoreCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoreCache").finish_non_exhaustive()
    }
}

impl StoreCache {
    fn new() -> Self {
        StoreCache(Arc::new(Mutex::new(HashMap::new())))
    }

    fn lock(&self) -> Result<MutexGuard<'_, HashMap<PathBuf, Arc<DataStore>>>, SamplerError> {
        self.0.lock().map_err(|_| SamplerError::SourceUnavailable {
            source_id: "store_cache".to_string(),
            reason: "row-store cache lock poisoned".to_string(),
        })
    }

    fn lock_ok(&self) -> Option<MutexGuard<'_, HashMap<PathBuf, Arc<DataStore>>>> {
        self.0.lock().ok()
    }
}

/// Configuration for a bulk Hugging Face row source backed by local snapshot files.
#[derive(Clone, Debug)]
pub struct HuggingFaceRowsConfig {
    /// Stable sampler source id used in record ids and metrics.
    pub source_id: String,
    /// Hugging Face dataset id, e.g. `HuggingFaceFW/fineweb`.
    pub dataset_name: String,
    /// Dataset config name, e.g. `default`.
    pub config_name: String,
    /// Split name, e.g. `train`.
    pub split_name: String,
    /// Local path to a snapshot directory for this split.
    pub snapshot_dir: PathBuf,
    /// File extensions accepted as shard files.
    ///
    /// Non-parquet files are read as line-delimited entries. Each line may be:
    /// - a JSON object row (for example JSONL/NDJSON), or
    /// - plain text, which is wrapped as `{ "text": "..." }`.
    pub shard_extensions: Vec<String>,

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
    /// Negative candidate columns (ordered).
    ///
    /// Used only in **role-based mode**.  When a column value is a JSON array,
    /// each element is expanded into a separate `SectionRole::Context` section.
    /// This supports HuggingFace "dict" datasets where negatives are embedded
    /// as a list in the same row (e.g. `embedding-data/QQP_triplets`).
    pub negative_columns: Vec<String>,
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
    /// Optional Hugging Face API token for authenticating private dataset access.
    ///
    /// When set, sent as `Authorization: Bearer <token>` on datasets-server API
    /// requests for shard downloads.  Populated
    /// automatically from the `HF_TOKEN` environment variable at construction
    /// time; callers may also set this field directly.
    pub hf_token: Option<String>,
    /// Resolved datasets-server parquet manifest endpoint URL.
    /// Populated at construction time from `TRIPLETS_HF_PARQUET_ENDPOINT` env var
    /// or `HF_PARQUET_DEFAULT_ENDPOINT`.
    pub parquet_endpoint: String,
    /// In-memory cache of opened `DataStore` instances, keyed by shard path.
    /// Populated lazily as shards are accessed and cleared when the cache grows
    /// beyond the configured capacity.
    pub store_cache: StoreCache,
    /// Optional pre-built HTTP client.  When set, [`HuggingFaceRowSource::new`]
    /// uses this client instead of building a new one.  This allows callers
    /// such as [`build_hf_sources`] to share a single connection pool and
    /// throttle state across many sources.
    pub(crate) http_client: Option<ClientWithMiddleware>,
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
            dataset_name: dataset.into(),
            config_name: config.into(),
            split_name: split.into(),
            snapshot_dir: snapshot_dir.into(),
            shard_extensions: vec![
                "parquet".to_string(),
                HF_SHARD_STORE_EXTENSION.to_string(),
                "jsonl".to_string(),
                "ndjson".to_string(),
                "json".to_string(),
            ],
            cache_capacity: SamplerConfig::default().ingestion_max_records,
            parquet_row_group_cache_capacity: 8,
            refresh_batch_multiplier: HF_REFRESH_BATCH_MULTIPLIER,
            remote_expansion_headroom_multiplier: HF_REMOTE_EXPANSION_HEADROOM_MULTIPLIER,
            local_disk_cap_bytes: Some(HF_LOCAL_DISK_CAP_BYTES),
            id_column: Some("id".to_string()),
            text_columns: vec!["text".to_string()],
            anchor_columns: Vec::new(),
            positive_columns: Vec::new(),
            negative_columns: Vec::new(),
            context_columns: Vec::new(),
            trust_override: None,
            hf_token: std::env::var(ENV_TRIPLETS_HF_TOKEN)
                .ok()
                .filter(|t| !t.trim().is_empty()),
            parquet_endpoint: HF_PARQUET_DEFAULT_ENDPOINT.to_string(),
            store_cache: StoreCache::new(),
            http_client: None,
        }
    }

    fn has_explicit_mapping(&self) -> bool {
        !self.anchor_columns.is_empty()
            || !self.positive_columns.is_empty()
            || !self.negative_columns.is_empty()
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
    parquet_row_groups: Vec<(usize, usize)>,
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
    pub(crate) config: HuggingFaceRowsConfig,
    http_runtime: Arc<tokio::runtime::Runtime>,
    /// Reusable HTTP client with pooled TCP/TLS connections, throttling, and
    /// exponential backoff for retries.
    ///
    /// Built once in [`HuggingFaceRowSource::new`] and cloned (internally
    /// reference-counted) wherever an HTTP request is issued.  Sharing the
    /// same client across all downloads, HEAD checks, and API queries avoids
    /// the per-request TCP/TLS handshake overhead and the noisy
    /// `CloseNotify`/`BrokenPipe` DEBUG traces from dropping short-lived
    /// connection pools.  The embedded [`reqwest_drive::DriveThrottleBackoff`]
    /// applies configurable backoff when HF responds with 429 or transient
    /// failures.
    ///
    /// **Note:** The throttle/backoff middleware is only compiled in
    /// **release builds** (`#[cfg(not(debug_assertions))]`).  Debug builds
    /// (including `cargo test`) skip it so that tests against mock servers
    /// aren't slowed by retry delays.  If you run a debug binary in
    /// production, you won't get automatic retry — compile with `--release`
    /// to enable it.
    http_client: ClientWithMiddleware,
    sampler_config: Arc<Mutex<Option<SamplerConfig>>>,
    state: Arc<Mutex<SourceState>>,
    cache: Arc<Mutex<RowCache>>,
    parquet_cache: Arc<Mutex<ParquetCache>>,
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
            http_runtime: Arc::clone(&self.http_runtime),
            http_client: self.http_client.clone(),
            sampler_config: Arc::clone(&self.sampler_config),
            state: Arc::clone(&self.state),
            cache: Arc::clone(&self.cache),
            parquet_cache: Arc::clone(&self.parquet_cache),
            eligible_index: Arc::clone(&self.eligible_index),
            expansion_thread: Arc::clone(&self.expansion_thread),
        }
    }
}

#[derive(Debug)]
struct SourceState {
    materialized_rows: usize,
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
type ShardIndexResult = (Vec<ShardIndex>, usize);

impl HuggingFaceRowSource {
    /// Return a reference to the process-wide shared multi-threaded tokio
    /// runtime, lazily initialized on first access.
    ///
    /// All `HuggingFaceRowSource` instances use this single runtime so that
    /// HTTP connections established by one source can be safely reused by
    /// another source via the shared `reqwest::Client` connection pool.
    fn shared_runtime() -> Arc<tokio::runtime::Runtime> {
        use std::sync::OnceLock;
        static RUNTIME: OnceLock<Arc<tokio::runtime::Runtime>> = OnceLock::new();
        RUNTIME
            .get_or_init(|| {
                Arc::new(
                    tokio::runtime::Builder::new_multi_thread()
                        .worker_threads(HF_SHARED_RUNTIME_WORKER_THREADS)
                        .enable_all()
                        .build()
                        .expect(
                            "failed building shared tokio runtime for Hugging Face HTTP requests",
                        ),
                )
            })
            .clone()
    }

    /// Build a new source by indexing local shard files.
    pub fn new(mut config: HuggingFaceRowsConfig) -> Result<Self, SamplerError> {
        let start_new = Instant::now();
        let http_runtime = Self::shared_runtime();
        let http_client = config
            .http_client
            .take()
            .map(Ok)
            .unwrap_or_else(|| Self::build_http_client(&config))?;

        if !config.has_explicit_mapping() {
            return Err(SamplerError::Configuration(
                "huggingface source requires explicit field mapping (anchor/positive/context/text_columns)"
                    .to_string(),
            ));
        }

        // Validate the token up-front so callers get a clear error immediately
        // rather than silent degradation on later API calls.
        if config.hf_token.is_some() {
            Self::validate_token_with_runtime(&http_client, &config, &http_runtime)?;
        }

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
        let (shards, discovered) = Self::build_shard_index(&config).unwrap_or_default();
        if discovered == 0 {
            info!(
                "[triplets:hf] {} no local shards found in {} — lazy remote download enabled",
                config.source_id,
                config.snapshot_dir.display()
            );
        }

        let materialized_rows = discovered;

        info!(
            "[triplets:hf] {} source ready in {:.2}s (rows={}, shards={})",
            config.source_id,
            start_new.elapsed().as_secs_f64(),
            materialized_rows,
            shards.len()
        );

        let source = Self {
            config,
            http_runtime,
            http_client,
            sampler_config: Arc::new(Mutex::new(None)),
            state: Arc::new(Mutex::new(SourceState {
                materialized_rows,
                shards,
                remote_candidates: None,
                remote_candidate_sizes: HashMap::new(),
                next_remote_idx: 0,
                remote_candidate_order: Vec::new(),
            })),
            cache: Arc::new(Mutex::new(RowCache::default())),
            parquet_cache: Arc::new(Mutex::new(ParquetCache::default())),
            eligible_index: Arc::new(Mutex::new(EligibleIndexCache::default())),
            expansion_thread: Arc::new(Mutex::new(None)),
        };

        // Post-initialization: transcode any local transient files to .simdr stores
        // with pre-corrected global offsets to prevent synthetic ID collisions.
        {
            let mut state = source
                .state
                .lock()
                .map_err(|_| SamplerError::SourceUnavailable {
                    source_id: source.config.source_id.clone(),
                    reason: "huggingface source state lock poisoned".to_string(),
                })?;
            let mut running_total = 0usize;
            let mut transcoded_shards = Vec::with_capacity(state.shards.len());
            for mut shard in state.shards.drain(..) {
                shard.global_start = running_total;
                if let Some(transcoded) = source.transcode_transient_shard_to_store(&shard)? {
                    running_total = running_total.saturating_add(transcoded.row_count);
                    transcoded_shards.push(transcoded);
                }
            }
            state.shards = transcoded_shards;
            state.materialized_rows = running_total;
        }

        Ok(source)
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
        let mut cache = self.config.store_cache.lock()?;
        if let Some(store) = cache.get(shard_store_path).cloned() {
            return Ok(store);
        }
        let store = Arc::new(Self::open_shard_store(&self.config, shard_store_path)?);
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
        if let Some(mut cache) = self.config.store_cache.lock_ok() {
            cache.retain(|path, _| keep.contains(path));
        }
    }

    /// Get or open a store through the shared cache.  Never opens a duplicate
    /// handle — if the path is already in `store_cache`, returns the cached
    /// `Arc`; otherwise opens and inserts it.
    fn open_store_via_cache(
        config: &HuggingFaceRowsConfig,
        path: &Path,
    ) -> Result<Arc<DataStore>, SamplerError> {
        // Fast path: check the cache while holding the lock briefly.
        {
            let cache = config.store_cache.lock()?;
            if let Some(store) = cache.get(path).cloned() {
                return Ok(store);
            }
        }
        // Open the store outside the lock so that concurrent calls (e.g. from
        // rayon's parallel iteration in build_shard_index) can proceed in
        // parallel instead of being serialized on the mutex.
        let store = Arc::new(Self::open_shard_store(config, path)?);
        // Re-acquire the lock and insert into the cache.  If another thread
        // already inserted the same path, our duplicate handle is harmless
        // (the cache retains the first one).  We return our handle either way
        // — both point to the same underlying file.
        let mut cache = config.store_cache.lock()?;
        cache
            .entry(path.to_path_buf())
            .or_insert_with(|| store.clone());
        Ok(store)
    }

    /// Evict a stale store from the cache and unlink the file so the shard
    /// gets re-downloaded on the next cycle.
    fn remove_stale_store(config: &HuggingFaceRowsConfig, path: &Path) {
        let _ = config
            .store_cache
            .lock_ok()
            .map(|mut cache| cache.remove(path));
        if let Err(err) = fs::remove_file(path) {
            warn!(
                "[triplets:hf] failed to remove stale store {}: {}",
                path.display(),
                err
            );
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

    fn transcode_transient_shard_to_store(
        &self,
        shard: &ShardIndex,
    ) -> Result<Option<ShardIndex>, SamplerError> {
        if Self::is_store_shard_path(&shard.path) {
            return Ok(Some(shard.clone()));
        }

        let store_path = Self::shard_store_path_for(&shard.path);
        let store = self.get_or_open_shard_store(&store_path)?;
        if store_path.exists() {
            let existing_rows = self.read_store_row_count(&store)?;
            if existing_rows > 0 {
                // Simdr store is already fully populated.  Clean up the
                // transient source file if it is still present.
                if shard.path != store_path
                    && shard.path.exists()
                    && let Err(err) = fs::remove_file(&shard.path)
                {
                    warn!(
                        "[triplets:hf] failed removing stale transient shard after store hit {}: {}",
                        shard.path.display(),
                        err
                    );
                }
                return Ok(Some(ShardIndex {
                    path: store_path,
                    global_start: shard.global_start,
                    row_count: existing_rows,
                    parquet_row_groups: vec![(0, existing_rows)],
                    remote_candidate: shard.remote_candidate.clone(),
                }));
            }
        }

        let mut served_rows = 0usize;

        if is_gzip_path(&shard.path) || is_transient_text(&shard.path) {
            // Transcode .jsonl.gz / .jsonl / .ndjson → .simdr store
            let file = File::open(&shard.path).map_err(|err| SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: format!(
                    "failed opening text shard for transcode {}: {err}",
                    shard.path.display()
                ),
            })?;
            let reader: Box<dyn BufRead> = if is_gzip_path(&shard.path) {
                Box::new(BufReader::new(GzDecoder::new(file)))
            } else {
                Box::new(BufReader::new(file))
            };
            let mut batch: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(1024);

            for (local_idx, line_result) in reader.lines().enumerate() {
                let line = line_result.map_err(|err| SamplerError::SourceUnavailable {
                    source_id: self.config.source_id.clone(),
                    reason: format!("failed reading text shard {}: {err}", shard.path.display()),
                })?;
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                // Use served_rows for absolute_idx to maintain bounded, non-overlapping IDs.
                // local_idx is only used for error reporting in parse_non_parquet_line.
                let absolute_idx = shard.global_start.saturating_add(served_rows);
                let line_value = self.parse_non_parquet_line(shard, local_idx, trimmed)?;

                let Some(row) = self.parse_row(absolute_idx, &line_value)? else {
                    continue;
                };

                let key = Self::row_store_row_key(served_rows);
                let payload = self.encode_row_view(&row)?;
                batch.push((key, payload));
                served_rows = served_rows.saturating_add(1);

                // Flush batch periodically
                if batch.len() >= 1024 {
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
                    batch.clear();
                }
            }

            // Flush remaining batch
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
        } else {
            // Parquet binary decoding
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
                    let absolute_idx = shard.global_start.saturating_add(served_rows);
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
        }

        self.write_store_row_count(&store, served_rows)?;

        // Only delete transient files inside the managed manifest root,
        // never delete user-provided local files.
        let in_manifest = shard.path.starts_with(self.manifest_cache_root());
        if shard.path != store_path && shard.path.exists() && in_manifest {
            fs::remove_file(&shard.path).map_err(|err| SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: format!(
                    "failed removing transient shard after store transcode {}: {err}",
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
            parquet_row_groups: vec![(0, served_rows)],
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
        let mut hasher = SipHasher::new();
        for shard in shards {
            shard.path.hash(&mut hasher);
            shard.global_start.hash(&mut hasher);
            shard.row_count.hash(&mut hasher);
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
        if let Ok(mut slot) = self.sampler_config.lock() {
            *slot = Some(config.clone());
        }

        // Rebuild the permuted order index whenever this is called (every
        // actually influences the shard download order.  Already-downloaded
        // shards are skipped via first_uncached_order_position regardless of
        // the permutation, so no re-downloads occur from re-ordering.
        if let Ok(mut state) = self.state.lock()
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
        Ok(triplets_core::source::IndexablePager::seed_for_sampler(
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
    /// Parse the Hub API tree endpoint response into shard candidates.
    ///
    /// The tree endpoint returns a JSON array of file objects:
    /// ```json
    /// [{"type":"file","path":"data/train-00000-of-00001.parquet","size":2648082,...}]
    /// ```
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

        // ── Hub API tree format: array of {"path": "...", "size": N} objects ────
        if let Some(arr) = json.as_array() {
            for entry in arr {
                let Some(file_path) = entry.get("path").and_then(Value::as_str) else {
                    continue;
                };
                let ext = resolve_inner_extension(Path::new(file_path));
                if !ext
                    .as_deref()
                    .is_some_and(|v| accepted.iter().any(|a| a == v))
                {
                    continue;
                }
                matched_manifest_entries += 1;
                let candidate = format!("{HF_REMOTE_URL_PREFIX}{file_path}");
                let expected_size = entry.get("size").and_then(Value::as_u64);

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
            if matched_manifest_entries > 0 {
                candidates.sort();
                candidates.dedup();
                candidate_sizes.retain(|candidate, _| candidates.binary_search(candidate).is_ok());
                return Ok((candidates, candidate_sizes, matched_manifest_entries));
            }
        }

        candidates.sort();
        candidates.dedup();
        candidate_sizes.retain(|candidate, _| candidates.binary_search(candidate).is_ok());
        Ok((candidates, candidate_sizes, matched_manifest_entries))
    }

    /// Resolve and filter remote shard candidates from manifest or repository listing.
    #[cfg(test)]
    fn list_remote_candidates(
        http_client: &ClientWithMiddleware,
        config: &HuggingFaceRowsConfig,
    ) -> Result<(Vec<String>, HashMap<String, u64>), SamplerError> {
        Self::list_remote_candidates_with_runtime(http_client, config, None)
    }

    /// Resolve and filter remote shard candidates from the Hub API tree endpoint.
    ///
    /// Returns an error when the parquet manifest is unavailable or has no entries
    /// for the requested dataset/config/split.
    pub fn list_remote_candidates_with_runtime(
        http_client: &ClientWithMiddleware,
        config: &HuggingFaceRowsConfig,
        runtime: Option<&tokio::runtime::Runtime>,
    ) -> Result<(Vec<String>, HashMap<String, u64>), SamplerError> {
        let manifest_result = Self::list_remote_candidates_from_parquet_manifest_with_runtime(
            http_client,
            config,
            runtime,
        );
        match &manifest_result {
            Ok((candidates, candidate_sizes, matched_manifest_entries))
                if *matched_manifest_entries > 0 =>
            {
                info!(
                    "[triplets:hf] remote parquet manifest: {} shard(s) for dataset='{}' \
                         config='{}' split='{}'",
                    candidates.len(),
                    config.dataset_name,
                    config.config_name,
                    config.split_name
                );
                Ok((candidates.clone(), candidate_sizes.clone()))
            }
            Ok(_) => Err(SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!(
                    "parquet manifest has no entries for dataset='{}' config='{}' split='{}'",
                    config.dataset_name, config.config_name, config.split_name
                ),
            }),
            Err(err) => Err(SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!(
                    "parquet manifest unavailable for dataset='{}': {err}",
                    config.dataset_name,
                ),
            }),
        }
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
        let mut hasher = SipHasher::new();
        HF_SHARD_CANDIDATE_SEED_TAG.hash(&mut hasher);
        sampler_seed.hash(&mut hasher);
        config.source_id.hash(&mut hasher);
        config.dataset_name.hash(&mut hasher);
        config.config_name.hash(&mut hasher);
        config.split_name.hash(&mut hasher);
        total_candidates.hash(&mut hasher);
        hasher.finish()
    }

    fn whoami_endpoint() -> String {
        if let Ok(value) = std::env::var(ENV_TRIPLETS_HF_WHOAMI_ENDPOINT)
            && !value.trim().is_empty()
        {
            return value;
        }
        HF_WHOAMI_DEFAULT_ENDPOINT.to_string()
    }

    /// Build a single-threaded tokio runtime for running async HTTP operations.
    ///
    /// The runtime enables all I/O and timer drivers.  Each source creates one
    /// such runtime at construction time and reuses it for all HTTP calls,
    /// avoiding the cost of building a new runtime per request.
    pub fn build_http_runtime(
        config: &HuggingFaceRowsConfig,
    ) -> Result<tokio::runtime::Runtime, SamplerError> {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!("failed building tokio runtime for Hugging Face HTTP path: {err}"),
            })
    }

    fn block_on_http_with_runtime<T>(
        runtime: Option<&tokio::runtime::Runtime>,
        config: &HuggingFaceRowsConfig,
        future: impl Future<Output = Result<T, SamplerError>>,
    ) -> Result<T, SamplerError> {
        if let Some(existing_runtime) = runtime {
            return existing_runtime.block_on(future);
        }
        let runtime = Self::build_http_runtime(config)?;
        runtime.block_on(future)
    }

    /// Validate a configured `hf_token` against the Hugging Face whoami endpoint.
    ///
    /// Called once during [`HuggingFaceRowSource::new`] when `config.hf_token` is
    /// `Some`.  Returns `Err(SamplerError::SourceUnavailable)` for any non-2xx
    /// response (including 401 Unauthorized for invalid/expired tokens) so that
    /// callers get a clear error at construction time rather than silent failures
    /// on later API calls.
    fn validate_token_with_runtime(
        http_client: &ClientWithMiddleware,
        config: &HuggingFaceRowsConfig,
        runtime: &tokio::runtime::Runtime,
    ) -> Result<(), SamplerError> {
        runtime.block_on(async {
            http_client
                .get(Self::whoami_endpoint())
                .send()
                .await
                .map_err(|err| SamplerError::SourceUnavailable {
                    source_id: config.source_id.clone(),
                    reason: format!("HF_TOKEN validation request failed: {err}"),
                })?
                .error_for_status()
                .map_err(|err| SamplerError::SourceUnavailable {
                    source_id: config.source_id.clone(),
                    reason: format!(
                        "HF_TOKEN is invalid or expired — \
                        the Hugging Face API rejected the credential ({err}). \
                        Generate a new token at https://huggingface.co/settings/tokens"
                    ),
                })?;
            Ok(())
        })
    }

    /// Build a throttled `ClientWithMiddleware` from the config's connection
    /// and auth settings, including exponential-backoff retry for rate limits
    /// and transient failures.
    pub(crate) fn build_http_client(
        config: &HuggingFaceRowsConfig,
    ) -> Result<ClientWithMiddleware, SamplerError> {
        use reqwest_drive::ClientBuilder;

        let mut builder = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(HF_HTTP_CONNECT_TIMEOUT_SECS))
            .timeout(Duration::from_secs(HF_HTTP_REQUEST_TIMEOUT_SECS));
        if let Some(token) = &config.hf_token {
            let header_value = reqwest::header::HeaderValue::from_str(&format!("Bearer {token}"))
                .map_err(|_| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: "HF_TOKEN contains characters invalid for an HTTP header value".to_string(),
            })?;
            let mut headers = reqwest::header::HeaderMap::new();
            headers.insert(reqwest::header::AUTHORIZATION, header_value);
            builder = builder.default_headers(headers);
        }
        let inner = builder
            .build()
            .map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!("failed building reqwest client: {err}"),
            })?;

        // Throttle/backoff middleware is skipped in debug builds
        // (cargo test, cargo build without --release).
        #[cfg(not(debug_assertions))]
        let throttle = {
            use reqwest_drive::{ThrottlePolicy, init_throttle};
            init_throttle(ThrottlePolicy {
                base_delay_ms: HF_THROTTLE_BASE_DELAY_MS,
                adaptive_jitter_ms: HF_THROTTLE_ADAPTIVE_JITTER_MS,
                max_concurrent: HF_THROTTLE_MAX_CONCURRENT,
                max_retries: HF_THROTTLE_MAX_RETRIES,
            })
        };

        let client = ClientBuilder::new(inner);
        #[cfg(not(debug_assertions))]
        let client = client.with_arc(throttle);
        Ok(client.build())
    }

    async fn fetch_http_body_text(
        http_client: &ClientWithMiddleware,
        source_id: &str,
        endpoint: &str,
        query: &[(&str, &str)],
        endpoint_label: &str,
    ) -> Result<String, SamplerError> {
        // Build the URL with query parameters, then build a fresh request
        // and execute it through the middleware client for throttling/retry.
        // We use url::Url::parse_with_params so the request is built without
        // creating a throwaway reqwest::Client (Client::new() has no timeouts).
        let url = reqwest::Url::parse_with_params(endpoint, query.iter().map(|&(k, v)| (k, v)))
            .map_err(|err| SamplerError::SourceUnavailable {
                source_id: source_id.to_string(),
                reason: format!("failed building URL for {endpoint_label}: {err}"),
            })?;
        let response = http_client
            .get(url)
            .send()
            .await
            .map_err(|err| SamplerError::SourceUnavailable {
                source_id: source_id.to_string(),
                reason: format!("failed querying {endpoint_label}: {err}"),
            })?
            .error_for_status()
            .map_err(|err| SamplerError::SourceUnavailable {
                source_id: source_id.to_string(),
                reason: format!("{endpoint_label} returned non-success response: {err}"),
            })?;

        response
            .text()
            .await
            .map_err(|err| SamplerError::SourceUnavailable {
                source_id: source_id.to_string(),
                reason: format!("failed reading {endpoint_label} response body: {err}"),
            })
    }

    /// Query datasets-server parquet manifest and derive shard candidates.
    #[cfg(test)]
    pub(crate) fn list_remote_candidates_from_parquet_manifest(
        http_client: &ClientWithMiddleware,
        config: &HuggingFaceRowsConfig,
    ) -> Result<ParquetManifestCandidates, SamplerError> {
        Self::list_remote_candidates_from_parquet_manifest_with_runtime(http_client, config, None)
    }

    fn list_remote_candidates_from_parquet_manifest_with_runtime(
        http_client: &ClientWithMiddleware,
        config: &HuggingFaceRowsConfig,
        runtime: Option<&tokio::runtime::Runtime>,
    ) -> Result<ParquetManifestCandidates, SamplerError> {
        let base = &config.parquet_endpoint;
        // Hub API: {base}/{dataset}/tree/main/{config}?recursive=true — lists
        // files under the config subdirectory, recursing into subdirectories.
        // Scoped to config_name so we don't pull files from other language
        // configs (e.g. wikimedia/wikipedia/20231101.en should not list .fr).
        let tree_path = if config.config_name.is_empty() || config.config_name == "default" {
            format!("{base}/{}/tree/main?recursive=true", config.dataset_name)
        } else {
            format!(
                "{base}/{}/tree/main/{}?recursive=true",
                config.dataset_name, config.config_name
            )
        };
        let url = tree_path;
        info!(
            "[triplets:hf] reading Hub API tree for dataset {}",
            config.dataset_name
        );
        let body = Self::block_on_http_with_runtime(
            runtime,
            config,
            Self::fetch_http_body_text(
                http_client,
                &config.source_id,
                &url,
                &[],
                "Hub tree endpoint",
            ),
        )?;

        Self::parse_parquet_manifest_response(config, &body)
    }

    fn parse_parquet_manifest_response(
        config: &HuggingFaceRowsConfig,
        body: &str,
    ) -> Result<ParquetManifestCandidates, SamplerError> {
        let json: Value =
            serde_json::from_str(body).map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!("failed parsing Hub API parquet response: {err}"),
            })?;

        Self::all_candidates_from_parquet_manifest(config, &json)
    }

    /// Map a candidate identifier to the local snapshot target path.
    ///
    /// Full CDN URLs (e.g. `https://huggingface.co/datasets/.../resolve/main/data/train.parquet`)
    /// are parsed to extract the relative path after `/resolve/`.  Bare relative
    /// paths (e.g. from the Hub API tree endpoint) are used directly as the suffix.
    fn candidate_target_path(config: &HuggingFaceRowsConfig, candidate: &str) -> PathBuf {
        if let Some(url) = candidate.strip_prefix(HF_REMOTE_URL_PREFIX) {
            // Full CDN URL: extract path after /resolve/
            if let Some(suffix) = url
                .split(HF_RESOLVE_URL_SEPARATOR)
                .nth(1)
                .map(|value| value.trim_start_matches('/'))
                .filter(|value| !value.is_empty())
            {
                return config
                    .snapshot_dir
                    .join(HF_PARQUET_MANIFEST_DIR)
                    .join(suffix);
            }
            // Bare relative path from tree endpoint (e.g. "train/000.parquet")
            return config
                .snapshot_dir
                .join(HF_PARQUET_MANIFEST_DIR)
                .join(url.trim_start_matches('/'));
        }
        config.snapshot_dir.join(candidate)
    }

    /// Validate target file size against expected bytes when available.
    ///
    /// When `expected_bytes` is `Some(N)` with N > 0, the local file must have
    /// exactly N bytes.  When `expected_bytes` is `None`, the file must exist
    /// and be non-zero bytes (rejects 0-byte corruption from failed downloads
    /// or chunked transfer encoding without Content-Length).
    fn target_matches_expected_size(path: &Path, expected_bytes: Option<u64>) -> bool {
        let Ok(metadata) = fs::metadata(path) else {
            return false;
        };
        let local_size = metadata.len();
        match expected_bytes {
            Some(expected) if expected > 0 => local_size == expected,
            None => local_size > 0,
            _ => true,
        }
    }

    /// Return on-disk size for a shard path, or 0 if metadata lookup fails.
    fn shard_size_bytes(path: &Path) -> u64 {
        fs::metadata(path).map(|meta| meta.len()).unwrap_or(0)
    }

    /// Build the full HTTP(S) remote URL for a candidate identifier.
    ///
    /// Candidates from the parquet manifest carry the `url::` prefix followed
    /// by a full URL or a repository-relative path.  This function always
    /// returns a valid absolute URL: if the value after stripping the prefix
    /// is already `http://` or `https://`, it is returned directly; otherwise
    /// it is resolved against the HuggingFace CDN.
    pub fn remote_url_for_candidate(config: &HuggingFaceRowsConfig, candidate: &str) -> String {
        let remainder = candidate
            .strip_prefix(HF_REMOTE_URL_PREFIX)
            .unwrap_or(candidate);
        if remainder.starts_with("http://") || remainder.starts_with("https://") {
            remainder.to_string()
        } else {
            format!(
                "{}/{}/resolve/main/{}",
                HF_DATASETS_BASE_URL,
                config.dataset_name,
                remainder.trim_start_matches('/')
            )
        }
    }

    /// Fetch the remote file size via an HTTP HEAD request.
    ///
    /// Returns `Ok(Some(size))` when the server responds with a `Content-Length`
    /// header, `Ok(None)` for non-2xx responses or missing `Content-Length`,
    /// and `Err` for network / configuration failures.
    pub fn fetch_remote_size_with_runtime(
        http_client: &ClientWithMiddleware,
        config: &HuggingFaceRowsConfig,
        remote_url: &str,
        runtime: &tokio::runtime::Runtime,
    ) -> Result<Option<u64>, SamplerError> {
        runtime.block_on(async {
            let response = http_client.head(remote_url).send().await.map_err(|err| {
                SamplerError::SourceUnavailable {
                    source_id: config.source_id.clone(),
                    reason: format!("HEAD request failed for shard URL '{}': {err}", remote_url),
                }
            })?;

            if !response.status().is_success() {
                return Ok(None);
            }

            Ok(response
                .headers()
                .get(reqwest::header::CONTENT_LENGTH)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok()))
        })
    }

    /// Return root directory used for manifest-cached remote shards.
    fn manifest_cache_root(&self) -> PathBuf {
        self.config.snapshot_dir.join(HF_PARQUET_MANIFEST_DIR)
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
            .ensure_group_with_policy(HF_PARQUET_MANIFEST_DIR, Some(&policy))
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
        let mut hasher = SipHasher::new();
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

    fn download_remote_url_to_target_with_runtime(
        http_client: &ClientWithMiddleware,
        config: &HuggingFaceRowsConfig,
        remote_url: &str,
        target: &Path,
        expected_bytes: Option<u64>,
        shard_label: &str,
        runtime: Option<&tokio::runtime::Runtime>,
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

        info!(
            "[triplets:hf] {} {} downloading shard payload -> {}",
            config.source_id,
            shard_label,
            target.display()
        );
        let http_client = http_client.clone();
        let (total_bytes, elapsed) = Self::block_on_http_with_runtime(runtime, config, async {
            use tokio::io::AsyncWriteExt;

            let mut response = http_client
                .get(remote_url)
                .send()
                .await
                .map_err(|err| SamplerError::SourceUnavailable {
                    source_id: config.source_id.clone(),
                    reason: format!("failed downloading shard URL '{}': {err}", remote_url),
                })?
                .error_for_status()
                .map_err(|err| SamplerError::SourceUnavailable {
                    source_id: config.source_id.clone(),
                    reason: format!(
                        "download URL '{}' returned non-success response: {err}",
                        remote_url
                    ),
                })?;

            let mut file = tokio::fs::File::create(&temp_target).await.map_err(|err| {
                SamplerError::SourceUnavailable {
                    source_id: config.source_id.clone(),
                    reason: format!(
                        "failed creating target shard {}: {err}",
                        temp_target.display()
                    ),
                }
            })?;
            let started = Instant::now();
            let mut total_bytes = 0u64;
            let mut last_report = Instant::now();
            while let Some(chunk) =
                response
                    .chunk()
                    .await
                    .map_err(|err| SamplerError::SourceUnavailable {
                        source_id: config.source_id.clone(),
                        reason: format!("failed reading shard stream '{}': {err}", remote_url),
                    })?
            {
                file.write_all(&chunk)
                    .await
                    .map_err(|err| SamplerError::SourceUnavailable {
                        source_id: config.source_id.clone(),
                        reason: format!(
                            "failed writing target shard {}: {err}",
                            temp_target.display()
                        ),
                    })?;
                total_bytes = total_bytes.saturating_add(chunk.len() as u64);
                if last_report.elapsed() >= Duration::from_secs(2) {
                    let elapsed = started.elapsed().as_secs_f64();
                    if let Some(expected) = expected_bytes
                        && expected > 0
                    {
                        let pct =
                            ((total_bytes as f64 / expected as f64) * 100.0).clamp(0.0, 100.0);
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
            file.flush()
                .await
                .map_err(|err| SamplerError::SourceUnavailable {
                    source_id: config.source_id.clone(),
                    reason: format!(
                        "failed flushing target shard {}: {err}",
                        temp_target.display()
                    ),
                })?;
            Ok((total_bytes, started.elapsed().as_secs_f64()))
        })?;

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

    /// Download a shard URL and materialize it under snapshot dir.
    #[cfg(test)]
    fn download_and_materialize_shard(
        http_client: &ClientWithMiddleware,
        config: &HuggingFaceRowsConfig,
        remote_path: &str,
        expected_bytes: Option<u64>,
        shard_label: &str,
    ) -> Result<PathBuf, SamplerError> {
        Self::download_and_materialize_shard_with_runtime(
            http_client,
            config,
            remote_path,
            expected_bytes,
            shard_label,
            None,
        )
    }

    fn download_and_materialize_shard_with_runtime(
        http_client: &ClientWithMiddleware,
        config: &HuggingFaceRowsConfig,
        remote_path: &str,
        expected_bytes: Option<u64>,
        shard_label: &str,
        runtime: Option<&tokio::runtime::Runtime>,
    ) -> Result<PathBuf, SamplerError> {
        // ── Path traversal guard ──────────────────────────────────────────────
        // Check the relative path that will determine the on-disk target location.
        // For full URLs (http/https), extract just the path portion after the host.
        // For bare relative paths, check the entire string.
        let clean_path = remote_path
            .strip_prefix(HF_REMOTE_URL_PREFIX)
            .unwrap_or(remote_path);
        let relative_for_guard = if let Some(after_proto) = clean_path
            .strip_prefix("https://")
            .or_else(|| clean_path.strip_prefix("http://"))
        {
            // "host/datasets/org/ds/resolve/main/data/train.parquet"
            // → "datasets/org/ds/resolve/main/data/train.parquet"
            after_proto
                .find('/')
                .map(|i| &after_proto[i..])
                .unwrap_or("/")
        } else {
            clean_path
        };
        for component in Path::new(relative_for_guard).components() {
            if matches!(
                component,
                std::path::Component::ParentDir | std::path::Component::Prefix(_)
            ) {
                return Err(SamplerError::SourceUnavailable {
                    source_id: config.source_id.clone(),
                    reason: format!(
                        "remote path contains traversal or absolute component, rejecting: {clean_path}"
                    ),
                });
            }
        }

        // ── Resolve full CDN URL ──────────────────────────────────────────────
        let resolved_url = Self::remote_url_for_candidate(config, remote_path);

        let target = Self::candidate_target_path(config, remote_path);
        let store_target = Self::shard_store_path_for(&target);
        if store_target.exists() {
            return Ok(store_target);
        }

        // ── Cache validation with HEAD fallback ───────────────────────────────
        if target.exists() {
            let effective_size = match expected_bytes {
                Some(bytes) => Some(bytes),
                None => runtime.and_then(|rt| {
                    Self::fetch_remote_size_with_runtime(http_client, config, &resolved_url, rt)
                        .ok()
                        .flatten()
                }),
            };
            if Self::target_matches_expected_size(&target, effective_size) {
                return Ok(target);
            }
            warn!(
                "[triplets:hf] replacing stale or incomplete shard before retry: {}",
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

        // ── Download ──────────────────────────────────────────────────────────
        let parquet_candidate = Self::is_parquet_path(&target);
        if parquet_candidate {
            let temp_target = Self::allocate_temp_download_path(config, remote_path, "parquet")?;
            Self::download_remote_url_to_target_with_runtime(
                http_client,
                config,
                &resolved_url,
                &temp_target,
                expected_bytes,
                shard_label,
                runtime,
            )?;
            return Ok(temp_target);
        }

        Self::download_remote_url_to_target_with_runtime(
            http_client,
            config,
            &resolved_url,
            &target,
            expected_bytes,
            shard_label,
            runtime,
        )?;
        Ok(target)
    }

    /// Build shard metadata for a single local file.  All store handles are
    /// fetched through `store_cache` (get-or-create), so there is never more
    /// than one `DataStore` handle open for the same path.
    #[cfg(test)]
    fn index_single_shard_for_test(
        config: &HuggingFaceRowsConfig,
        path: &Path,
        global_start: usize,
    ) -> Result<(Option<ShardIndex>, Option<Arc<DataStore>>), SamplerError> {
        Self::index_single_shard(config, path, global_start)
    }

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

        // .gz files (e.g. .jsonl.gz) are transient download artifacts that will
        // be transcoded to .simdr stores during the download phase.
        let is_transient_gz = is_gzip_path(path);

        let (rows, parquet_row_groups, _checkpoints, maybe_store) = if is_store {
            let store = Self::open_store_via_cache(config, path)?;
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

            // Integrity check: verify the last claimed row actually exists in
            // the store.  A corrupt store (partial write, truncated file) may
            // have the metadata key intact but be missing row data.  Delete it
            // so the shard is re-downloaded on the next expansion cycle.
            if rows > 0 {
                let last_key = Self::row_store_row_key(rows.saturating_sub(1));
                match store.batch_read(&[last_key.as_slice()]) {
                    Ok(entries) if entries[0].is_some() => {}
                    _ => {
                        warn!(
                            "[triplets:hf] corrupted store detected ({} rows claimed but last row missing), deleting: {}",
                            rows,
                            path.display()
                        );
                        Self::remove_stale_store(config, path);
                        return Ok((None, None));
                    }
                }
            }

            let groups = if rows > 0 {
                vec![(0, rows)]
            } else {
                Vec::new()
            };
            (rows, groups, Vec::<u64>::new(), Some(store))
        } else if is_transient_parquet {
            let (rows, parquet_row_groups) = Self::parquet_row_group_map(config, path)?;
            (rows, parquet_row_groups, Vec::<u64>::new(), None)
        } else if is_transient_gz || is_transient_text(path) {
            // All transient text/gzip files are transcoded to .simdr stores.
            // Validate file exists and is non-empty before returning dummy count.
            let metadata = fs::metadata(path).map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!("failed opening shard {}: {err}", path.display()),
            })?;
            if metadata.len() == 0 {
                return Ok((None, None));
            }
            return Ok((
                Some(ShardIndex {
                    path: path.to_path_buf(),
                    global_start,
                    row_count: 1,
                    parquet_row_groups: Vec::new(),
                    remote_candidate: None,
                }),
                None,
            ));
        } else {
            let file = File::open(path).map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!("failed opening shard {}: {err}", path.display()),
            })?;
            let reader = BufReader::new(file);
            let rows = reader.lines().count();

            (rows, Vec::new(), Vec::<u64>::new(), None)
        };

        if rows == 0 {
            return Ok((None, None));
        }

        Ok((
            Some(ShardIndex {
                path: path.to_path_buf(),
                global_start,
                row_count: rows,
                parquet_row_groups,
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
        // Track whether we have already fetched the remote candidate list during
        // this call.  Once candidates are fetched and a download is attempted, we
        // must NOT re-enter the candidate-fetch path even if the disk-cap eviction
        // inside download_next_remote_shard nulls `remote_candidates` again.
        // Doing so would create an infinite download loop:
        //
        //   1. Fetch candidate list from HF manifest
        //   2. Download shard N → evict old shard → candidates nulled
        //   3. Loop back → need_candidates=true → fetch manifest AGAIN
        //   4. Download shard M → evict → candidates nulled
        //   5. Repeat forever — a single expansion thread hammers HF every ~8s
        let mut fetched_candidates = false;
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
                if fetched_candidates {
                    // We already fetched candidates and downloaded a shard in a
                    // previous iteration.  Eviction inside download_next_remote_shard
                    // nulled remote_candidates again, but we are not re-fetching.
                    // The caller (expansion thread or refresh) will see that idx
                    // is still not available and may try again on the next cycle.
                    return Ok(true);
                }
                fetched_candidates = true;

                let mut state = self
                    .state
                    .lock()
                    .map_err(|_| SamplerError::SourceUnavailable {
                        source_id: self.config.source_id.clone(),
                        reason: "huggingface source state lock poisoned".to_string(),
                    })?;
                if state.remote_candidates.is_none() {
                    let (mut candidates, candidate_sizes) =
                        Self::list_remote_candidates_with_runtime(
                            &self.http_client,
                            &self.config,
                            Some(self.http_runtime.as_ref()),
                        )?;
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
                        let bootstrap_target = HF_REMOTE_BOOTSTRAP_SHARDS.min(candidate_count);
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
                //
                // However, if the remote manifest reports a source size and the cached
                // store carries a different stored value, the upstream shard was replaced
                // (newer version).  Delete the stale store so it gets redownloaded.
                //
                // When the manifest does not provide a size, a lightweight HTTP HEAD
                // size from `Content-Length` so that staleness detection still works
                // without depending on the datasets-server API.
                let store_path = Self::candidate_store_path(&self.config, &remote_path);
                if store_path.exists() {
                    // Resolve the expected remote size: prefer the manifest-provided
                    // value, but fall back to an HTTP HEAD request so staleness
                    // detection works even when the datasets-server is unavailable.
                    let effective_expected = if let Some(bytes) = expected_bytes {
                        Some(bytes)
                    } else {
                        let remote_url = Self::remote_url_for_candidate(&self.config, &remote_path);
                        match Self::fetch_remote_size_with_runtime(
                            &self.http_client,
                            &self.config,
                            &remote_url,
                            &self.http_runtime,
                        ) {
                            Ok(Some(size)) if size > 0 => Some(size),
                            Ok(_) => None,
                            Err(err) => {
                                warn!(
                                    "[triplets:hf] {} {} HEAD size stale check failed: {err}",
                                    self.config.source_id,
                                    Self::format_shard_label(
                                        remote_path.as_str(),
                                        candidate_idx,
                                        remote_total
                                    ),
                                );
                                None
                            }
                        }
                    };

                    if let Some(expected) = effective_expected {
                        // Only read the stored source size from the cache — never
                        // call get_or_open_shard_store purely to check, because it
                        // creates an empty store file if the path doesn't exist.
                        // If the handle isn't cached yet, the store was just loaded
                        // and we'll catch staleness on the next cycle.
                        let stale = self
                            .config
                            .store_cache
                            .lock_ok()
                            .and_then(|cache| cache.get(&store_path).cloned())
                            .and_then(|store| {
                                let entry = store.read(HF_SHARD_STORE_SOURCE_SIZE_KEY).ok()??;
                                let bytes = entry.as_ref();
                                if bytes.len() != std::mem::size_of::<u64>() {
                                    return None;
                                }
                                let mut raw = [0u8; 8];
                                raw.copy_from_slice(bytes);
                                Some(u64::from_le_bytes(raw))
                            });
                        if let Some(stale) = stale
                            && stale != expected
                        {
                            warn!(
                                "[triplets:hf] {} {} stale on disk (stored size {} ≠ expected {}), redownloading",
                                self.config.source_id,
                                Self::format_shard_label(
                                    remote_path.as_str(),
                                    candidate_idx,
                                    remote_total
                                ),
                                stale,
                                expected,
                            );
                            Self::remove_stale_store(&self.config, &store_path);
                        }
                    }

                    if store_path.exists() {
                        debug!(
                            "[triplets:hf] {} {} already on disk, skipping download",
                            self.config.source_id,
                            Self::format_shard_label(
                                remote_path.as_str(),
                                candidate_idx,
                                remote_total
                            ),
                        );
                        return Ok(true);
                    }
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
        let local_path = Self::download_and_materialize_shard_with_runtime(
            &self.http_client,
            &self.config,
            &remote_path,
            expected_bytes,
            &label,
            Some(self.http_runtime.as_ref()),
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
        shard
            .parquet_row_groups
            .retain(|(start, _)| *start < rows_to_add);
        if let Some((start, count)) = shard.parquet_row_groups.last_mut() {
            let allowed = rows_to_add.saturating_sub(*start);
            *count = (*count).min(allowed);
        }

        drop(state);
        let mut shard = match self.transcode_transient_shard_to_store(&shard)? {
            Some(shard) => shard,
            None => return Ok(true),
        };

        // Relocate any .simdr store to its canonical path in the manifest root
        if Self::is_store_shard_path(&shard.path) {
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

        // Persist the source shard's expected size from the remote manifest so
        // that future cycles can detect when the upstream shard was replaced.
        // When the manifest doesn't provide a size, use the actual downloaded
        let source_size = expected_bytes.unwrap_or_else(|| {
            fs::metadata(&local_path)
                .map(|meta| meta.len())
                .unwrap_or(0)
        });
        if source_size > 0 {
            // Write to the already-cached store handle (opened during transcode)
            // to avoid opening a second handle to the same file.
            if let Ok(cache) = self.config.store_cache.lock()
                && let Some(store) = cache.get(&shard.path)
            {
                let _ = store.write(HF_SHARD_STORE_SOURCE_SIZE_KEY, &source_size.to_le_bytes());
            }
        }

        let mut state = self
            .state
            .lock()
            .map_err(|_| SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: "huggingface source state lock poisoned".to_string(),
            })?;
        state.materialized_rows += shard.row_count;
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
    #[cfg(test)]
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
        let manifest_root = config.snapshot_dir.join(HF_PARQUET_MANIFEST_DIR);
        let accepted = config
            .shard_extensions
            .iter()
            .map(|ext| ext.trim().trim_start_matches('.').to_ascii_lowercase())
            .collect::<Vec<_>>();

        // .simdr must always be accepted regardless of user config, since
        // transient files may have already been transcoded to .simdr stores.
        let mut accepted = accepted;
        if !accepted.iter().any(|e| e == HF_SHARD_STORE_EXTENSION) {
            accepted.push(HF_SHARD_STORE_EXTENSION.to_string());
        }

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
            let is_parquet = ext.eq_ignore_ascii_case("parquet");
            let is_transient =
                is_transient_text(entry.path()) || is_gzip_path(entry.path()) || is_parquet;

            if is_transient {
                if in_manifest {
                    // Delete orphaned remote artifacts (truncated/crashed downloads)
                    let _ = fs::remove_file(entry.path());
                    continue;
                }
                // Local user files: skip if .simdr already exists (avoid double-indexing)
                let store_path = HuggingFaceRowSource::shard_store_path_for(entry.path());
                if store_path.exists() {
                    continue;
                }
                // Flow into shard_paths for initialization transcoding
                if is_parquet {
                    saw_parquet = true;
                }
                shard_paths.push(entry.path().to_path_buf());
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
        for (_, (maybe_shard, _maybe_store)) in indexed_shards {
            let Some(mut shard) = maybe_shard else {
                continue;
            };

            if shard.row_count == 0 {
                continue;
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

        Ok((shards, running_total))
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
    fn value_to_text(value: &Value) -> Option<String> {
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
            Value::Number(n) => Some(n.to_string()),
            Value::Array(arr) => {
                for element in arr {
                    if let Some(text) = Self::value_to_text(element) {
                        return Some(text);
                    }
                }
                None
            }
            Value::Object(obj) => serde_json::to_string(obj).ok().filter(|s| !s.is_empty()),
        }
    }

    /// Resolve a dot-separated column path against a JSON object.
    ///
    /// Splits `name` on `.` and walks the JSON hierarchy.  For example
    /// `"set.query"` resolves to `row["set"]["query"]`.  A bare name
    /// without dots (e.g. `"query"`) performs a simple top-level lookup.
    ///
    /// Returns `None` when any segment of the path is missing or when an
    /// intermediate value is not a JSON object.
    fn resolve_json_path(row_obj: &serde_json::Map<String, Value>, name: &str) -> Option<Value> {
        let mut current = Value::Object(row_obj.clone());
        for segment in name.split('.') {
            current = match current {
                Value::Object(map) => map.get(segment)?.clone(),
                _ => return None,
            };
        }
        Some(current)
    }

    /// Try each candidate column name in order and return the first one that
    /// yields a non-empty text value.  Returns `None` when no candidate
    /// matches, which the caller uses to decide whether to skip the row.
    fn coalesce_field(
        candidates: &[String],
        row_obj: &serde_json::Map<String, Value>,
    ) -> Option<RowTextField> {
        for name in candidates {
            if let Some(ref value) = Self::resolve_json_path(row_obj, name)
                && let Some(text) = Self::value_to_text(value)
            {
                return Some(RowTextField {
                    name: name.clone(),
                    text,
                });
            }
        }
        None
    }

    /// Extract all elements from a list-valued column, returning one
    /// `RowTextField` per element.
    ///
    /// For non-list values, returns a single-element vector (same behavior
    /// as `coalesce_field`).  For lists, each non-empty element becomes a
    /// separate field.  Empty/blank elements are skipped.  Returns `None`
    /// when no non-empty elements are found.
    fn coalesce_list_field(name: &str, value: &Value) -> Option<Vec<RowTextField>> {
        match value {
            Value::Array(arr) => {
                let fields: Vec<RowTextField> = arr
                    .iter()
                    .enumerate()
                    .filter_map(|(i, element)| {
                        Self::value_to_text(element).map(|text| RowTextField {
                            name: format!("{name}[{i}]"),
                            text,
                        })
                    })
                    .collect();
                if fields.is_empty() {
                    None
                } else {
                    Some(fields)
                }
            }
            other => Self::value_to_text(other).map(|text| {
                vec![RowTextField {
                    name: name.to_string(),
                    text,
                }]
            }),
        }
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
            .and_then(|col| Self::resolve_json_path(row_obj, col))
            .and_then(|v| Self::value_to_text(&v))
            .unwrap_or_else(|| {
                format!(
                    "{}:{}:{}",
                    self.config.dataset_name, self.config.split_name, absolute_idx
                )
            });

        let mut text_fields = Vec::new();
        let use_role_columns = !self.config.anchor_columns.is_empty()
            || !self.config.positive_columns.is_empty()
            || !self.config.negative_columns.is_empty()
            || !self.config.context_columns.is_empty();

        if use_role_columns {
            // Anchor: try each candidate column in order; use the first
            // whose value is present and non-empty.  Skip the row when the
            // list is non-empty but no candidate yields content.
            if !self.config.anchor_columns.is_empty() {
                match Self::coalesce_field(&self.config.anchor_columns, row_obj) {
                    Some(field) => text_fields.push(field),
                    None => return Ok(None),
                }
            }

            // Positive: try each candidate column in order; use the first
            // whose value is present and non-empty.  Skip the row when the
            // list is non-empty but no candidate yields content.
            if !self.config.positive_columns.is_empty() {
                match Self::coalesce_field(&self.config.positive_columns, row_obj) {
                    Some(field) => text_fields.push(field),
                    None => return Ok(None),
                }
            }

            for name in &self.config.context_columns {
                let Some(value) = Self::resolve_json_path(row_obj, name) else {
                    return Ok(None);
                };
                let Some(text) = Self::value_to_text(&value) else {
                    return Ok(None);
                };
                text_fields.push(RowTextField {
                    name: name.clone(),
                    text,
                });
            }

            // Negative columns: expand list values into multiple Context sections.
            for name in &self.config.negative_columns {
                let Some(value) = Self::resolve_json_path(row_obj, name) else {
                    return Ok(None);
                };
                let Some(fields) = Self::coalesce_list_field(name, &value) else {
                    return Ok(None);
                };
                text_fields.extend(fields);
            }
        } else {
            // Text-columns mode: try each candidate column in order; use the
            // first whose value is present and non-empty.  The row is skipped
            // when no candidate yields content (handled by the is_empty guard
            // below).
            if let Some(field) = Self::coalesce_field(&self.config.text_columns, row_obj) {
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

        let is_strict_json_lines = resolve_inner_extension(&shard.path)
            .is_some_and(|ext| ext == "jsonl" || ext == "ndjson");

        match serde_json::from_str::<Value>(trimmed) {
            Ok(value) => {
                let payload = value.get("row").unwrap_or(&value);
                if payload.is_object() {
                    Ok(value)
                } else if let Some(text) = Self::value_to_text(payload) {
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
                format!("dataset={}", self.config.dataset_name),
                format!("config={}", self.config.config_name),
                format!("split={}", self.config.split_name),
            ],
            sections,
            meta_prefix: None,
            label: None,
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
                let (group_pos, local_in_group) = self.locate_parquet_group(&shard, local_idx)?;
                parquet_groups
                    .entry((shard.path.clone(), group_pos))
                    .or_default()
                    .push((idx, local_in_group, shard));
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
                    for (position, entry) in requested_positions.into_iter().zip(store_entries) {
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
            if let Some(ref candidates) = state.remote_candidates
                && state.next_remote_idx < candidates.len()
            {
                let headroom = self.effective_expansion_headroom_rows();
                upper = known.saturating_add(headroom);
            }
            return Some(upper.max(known));
        }
        if state
            .remote_candidates
            .as_ref()
            .is_some_and(|c| c.is_empty())
        {
            return Some(0);
        }
        Some(1)
    }
}

/// Global gate that serializes expansion downloads across ALL HuggingFace
/// sources.  Only one source downloads a shard at any given time, preventing
/// bursts when multiple sources trigger expansion on the same cycle.
static EXPANSION_GATE: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();

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
                let all_consumed = state
                    .remote_candidates
                    .as_ref()
                    .is_some_and(|c| state.next_remote_idx >= c.len());
                !all_consumed
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
            // Acquire the global expansion gate so only one source downloads
            // a shard at a time across all Hugging Face sources.  The gate is
            // released when the thread exits (guard dropped).
            let _gate = EXPANSION_GATE
                .get_or_init(|| std::sync::Mutex::new(()))
                .lock()
                .expect("expansion gate not poisoned");

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
            let materialized = {
                let state = self
                    .state
                    .lock()
                    .map_err(|_| SamplerError::SourceUnavailable {
                        source_id: self.config.source_id.clone(),
                        reason: "huggingface source state lock poisoned".to_string(),
                    })?;
                state.materialized_rows
            };
            if materialized == 0 {
                // Bootstrap: discover candidates and download the first shard so
                // the read loop below has rows to work with.  ensure_row_available
                // handles candidate discovery and the initial shard download.  If
                // bootstrap fails (no remote dataset, all local rows skipped during
                // transcoding), return an empty snapshot rather than an error.
                match self.ensure_row_available(0) {
                    Ok(true) => {
                        self.state
                            .lock()
                            .map_err(|_| SamplerError::SourceUnavailable {
                                source_id: self.config.source_id.clone(),
                                reason: "huggingface source state lock poisoned".to_string(),
                            })?
                            .materialized_rows
                    }
                    Ok(false) | Err(_) => {
                        return Ok(SourceSnapshot {
                            records: Vec::new(),
                            cursor: SourceCursor {
                                last_seen: Utc::now(),
                                revision: 0,
                            },
                        });
                    }
                }
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
        let mut permutation =
            triplets_core::source::IndexPermutation::new(total, seed, start as u64);

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

        // Use the seed-derived permutation cursor as the returned revision
        // so the next refresh continues from where this one left off.
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
        // Dict-dataset mode: negative_columns is set, meaning negatives come
        // from the same row rather than different records.
        if !self.config.negative_columns.is_empty() {
            return vec![TripletRecipe {
                name: "huggingface_dict_anchor_positive_same_record".into(),
                anchor: Selector::Role(SectionRole::Anchor),
                positive_selector: Selector::Role(SectionRole::Context),
                negative_selector: Selector::Role(SectionRole::Context),
                negative_strategy: NegativeStrategy::SameRecord,
                weight: 1.0,
                instruction: None,
                allow_same_anchor_positive: false,
            }];
        }

        // Text-columns mode: anchor_columns is empty and only a plain text column is
        // mapped.  In this configuration every record's Anchor and Context sections
        // carry identical text (the single text field is duplicated by row_to_record).
        // Emit a single SimCSE-style recipe that deliberately allows same-text pairs:
        // the negative comes from a different record, and the model's dropout layers
        // provide the necessary embedding variation between the two identical slots.
        if self.config.anchor_columns.is_empty() {
            return vec![TripletRecipe {
                name: HF_RECIPE_TEXT_SIMCSE_WRONG_ARTICLE.into(),
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
            // Majority lane remains context negatives for broad coverage and
            // stable optimization across varied HF schemas.
            TripletRecipe {
                name: HF_RECIPE_ANCHOR_CONTEXT_WRONG_ARTICLE.into(),
                anchor: Selector::Role(SectionRole::Anchor),
                positive_selector: Selector::Role(SectionRole::Context),
                negative_selector: Selector::Role(SectionRole::Context),
                negative_strategy: NegativeStrategy::WrongArticle,
                weight: 0.75,
                instruction: None,
                allow_same_anchor_positive: false,
            },
            // Medium-hard lane adds anchor-as-negative pressure to improve
            // discrimination between title-like anchor fields.
            TripletRecipe {
                name: HF_RECIPE_ANCHOR_ANCHOR_WRONG_ARTICLE.into(),
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
    use parquet::data_type::{ByteArray, ByteArrayType};
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::SerializedFileWriter;
    use parquet::schema::parser::parse_message_type;
    use serde_json::json;
    use serial_test::serial;
    use std::env;
    use std::io::{Read, Write};
    use tempfile::tempdir;
    use triplets_core::splits::{PersistedSamplerState, SamplerStateStore};
    use triplets_core::utils::platform_newline;
    use triplets_core::{
        DeterministicSplitStore, Sampler, SplitLabel, SplitRatios, TripletSampler,
    };

    use crate::test_utils::{
        TEST_UNREACHABLE_URL, TestHttpServer, spawn_manifest_and_shard_http, spawn_one_shot_http,
    };

    fn test_config(snapshot_dir: PathBuf) -> HuggingFaceRowsConfig {
        let mut config =
            HuggingFaceRowsConfig::new("hf_test", "org/dataset", "default", "train", snapshot_dir);
        // Unit tests should be deterministic and fully mock-driven; ignore any
        // process-level HF_TOKEN that CI might inject.
        config.hf_token = None;
        config.cache_capacity = 10;
        config.remote_expansion_headroom_multiplier = 3;
        // Point endpoints to connection-refused so tests never wait on
        // real HF servers.  Tests that exercise HTTP against mock servers
        // override these in their own body.
        config.parquet_endpoint = TEST_UNREACHABLE_URL.to_string();
        config
    }

    fn test_http_client() -> ClientWithMiddleware {
        use reqwest_drive::ClientBuilder;

        let inner = reqwest::Client::builder()
            .connect_timeout(Duration::from_millis(500))
            .timeout(Duration::from_secs(1))
            .build()
            .expect("fast test reqwest client should build");
        ClientBuilder::new(inner).build()
    }

    fn test_source(config: HuggingFaceRowsConfig) -> SafeTestSource {
        let http_runtime = Arc::new(HuggingFaceRowSource::build_http_runtime(&config).unwrap());
        // Use a non-throttled client in tests — mock servers serve a single
        // request then shut down, so retry backoff would add unnecessary delay.
        let http_client = test_http_client();
        let source = HuggingFaceRowSource {
            config,
            http_runtime,
            http_client,
            sampler_config: Arc::new(Mutex::new(None)),
            state: Arc::new(Mutex::new(SourceState {
                materialized_rows: 0,
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
            eligible_index: Arc::new(Mutex::new(EligibleIndexCache::default())),
            expansion_thread: Arc::new(Mutex::new(None)),
        };
        source.set_active_sampler_config(&SamplerConfig {
            seed: 1,
            ingestion_max_records: source.config.cache_capacity,
            ..SamplerConfig::default()
        });
        SafeTestSource { source }
    }

    /// RAII wrapper that joins any leaked expansion thread on drop.
    /// Prevents zombie threads from holding `EXPANSION_GATE` and
    /// deadlocking subsequent tests in the CI runner.
    struct SafeTestSource {
        source: HuggingFaceRowSource,
    }

    impl std::ops::Deref for SafeTestSource {
        type Target = HuggingFaceRowSource;
        fn deref(&self) -> &Self::Target {
            &self.source
        }
    }

    impl std::ops::DerefMut for SafeTestSource {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.source
        }
    }

    impl Drop for SafeTestSource {
        fn drop(&mut self) {
            let mut lock = match self.source.expansion_thread.lock() {
                Ok(l) => l,
                Err(poisoned) => poisoned.into_inner(),
            };
            if let Some(handle) = lock.take() {
                let (tx, rx) = std::sync::mpsc::channel();
                std::thread::spawn(move || {
                    let _ = tx.send(handle.join());
                });
                let _ = rx
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .expect("Test teardown leaked a deadlocked expansion thread");
            }
        }
    }

    fn with_env_var<R>(key: &str, value: &str, run: impl FnOnce() -> R) -> R {
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
        run()
    }

    /// Sets multiple `(key, value)` pairs atomically, restoring originals on drop.
    /// Use this instead of nesting `with_env_var` calls.
    fn with_env_vars<R>(pairs: &[(&str, &str)], run: impl FnOnce() -> R) -> R {
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
    #[serial(global_state)]
    fn managed_snapshot_helpers_create_cache_dirs_under_discovered_root() {
        let dir = tempdir().unwrap();
        let nl = platform_newline();
        fs::write(
            dir.path().join("Cargo.toml"),
            format!("[package]{nl}name='tmp'{nl}version='0.0.0'{nl}"),
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
                HF_GROUP
            ))));
            assert!(listed.ends_with(PathBuf::from(format!(
                "{}/source-list/org__dataset/default/train/replica_7",
                HF_GROUP
            ))));
            assert!(listed.ends_with("replica_7"));
        });
    }

    #[test]
    #[serial(global_state)]
    fn managed_snapshot_dirs_use_all_splits_dir_for_empty_split() {
        let dir = tempdir().unwrap();
        let nl = platform_newline();
        fs::write(
            dir.path().join("Cargo.toml"),
            format!("[package]{nl}name='tmp'{nl}version='0.0.0'{nl}"),
        )
        .unwrap();

        with_current_dir(dir.path(), || {
            let single = managed_hf_snapshot_dir("org/dataset", "default", "").unwrap();
            let listed = managed_hf_list_snapshot_dir("org/dataset", "default", "", 0).unwrap();

            assert!(single.exists());
            assert!(listed.exists());
            // Both must use HF_ALL_SPLITS_DIR ("_all") in the path, not an empty segment.
            assert!(
                single.ends_with(PathBuf::from(format!(
                    "{}/org__dataset/default/{}",
                    HF_GROUP, HF_ALL_SPLITS_DIR
                ))),
                "expected single-source path to end with HF_ALL_SPLITS_DIR, got: {}",
                single.display()
            );
            assert!(
                listed.ends_with(PathBuf::from(format!(
                    "{}/source-list/org__dataset/default/{}/replica_0",
                    HF_GROUP, HF_ALL_SPLITS_DIR
                ))),
                "expected list-source path to end with HF_ALL_SPLITS_DIR, got: {}",
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
        let nl = platform_newline();

        let invalid_list = dir.path().join("invalid_sources.txt");
        fs::write(
            &invalid_list,
            format!("hf://org/dataset/default/train badtoken{nl}"),
        )
        .unwrap();
        let invalid = load_hf_sources_from_list(invalid_list.to_str().unwrap()).unwrap_err();
        assert!(invalid.contains("invalid source-list entry"));

        let empty_list = dir.path().join("empty_sources.txt");
        fs::write(&empty_list, format!("# comment only{nl}{nl}")).unwrap();
        let resolved = resolve_hf_list_roots(empty_list.to_string_lossy().to_string()).unwrap_err();
        assert!(resolved.contains("no hf:// entries found"));

        let good_list = dir.path().join("good_sources.txt");
        fs::write(
            &good_list,
            format!("hf://org/dataset/default/train anchor=title positive=body{nl}"),
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
    #[serial(global_state)]
    fn list_remote_candidates_falls_back_when_manifest_query_fails() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.dataset_name = "invalid///dataset".to_string();
        config.parquet_endpoint = TEST_UNREACHABLE_URL.to_string();

        let client = test_http_client();
        let result = HuggingFaceRowSource::list_remote_candidates(&client, &config);
        assert!(result.is_err());
    }

    // ── Token validation tests ──────────────────────────────────────────────

    #[test]
    fn http_client_builds_with_token() {
        let temp = tempdir().unwrap();
        let mut config = test_config(temp.path().to_path_buf());
        config.hf_token = Some("test-bearer-token".to_string());
        let result = HuggingFaceRowSource::build_http_client(&config);
        assert!(
            result.is_ok(),
            "build_http_client should succeed with a well-formed token string"
        );
    }

    #[test]
    #[serial(global_state)]
    fn validate_token_accepts_200_response() {
        let temp = tempdir().unwrap();
        let mut config = test_config(temp.path().to_path_buf());
        config.hf_token = Some("valid-test-token".to_string());
        let server = spawn_one_shot_http(b"{\"name\":\"testuser\"}".to_vec());
        let base_url = server.url().to_string();
        with_env_var(ENV_TRIPLETS_HF_WHOAMI_ENDPOINT, &base_url, || {
            let client = test_http_client();
            let runtime = HuggingFaceRowSource::build_http_runtime(&config).unwrap();
            let result =
                HuggingFaceRowSource::validate_token_with_runtime(&client, &config, &runtime);
            assert!(result.is_ok(), "200 response should pass token validation");
        });
    }

    #[test]
    #[serial(global_state)]
    fn validate_token_rejects_401_response() {
        let temp = tempdir().unwrap();
        let mut config = test_config(temp.path().to_path_buf());
        config.hf_token = Some("invalid-test-token".to_string());
        let server = TestHttpServer::new(401, b"Unauthorized".to_vec());
        let base_url = server.url().to_string();
        with_env_var(ENV_TRIPLETS_HF_WHOAMI_ENDPOINT, &base_url, || {
            let client = test_http_client();
            let runtime = HuggingFaceRowSource::build_http_runtime(&config).unwrap();
            let result =
                HuggingFaceRowSource::validate_token_with_runtime(&client, &config, &runtime);
            assert!(result.is_err(), "401 response should fail token validation");
            match result {
                Err(SamplerError::SourceUnavailable { reason, .. }) => {
                    assert!(
                        reason.contains("invalid or expired"),
                        "error should mention invalid/expired, got: {reason}"
                    );
                }
                _ => panic!("expected SamplerError::SourceUnavailable"),
            }
        });
    }

    #[test]
    #[serial(global_state)]
    fn build_hf_sources_skips_invalid_uri_and_builds_valid_source() {
        let roots = HfListRoots {
            source_list: "inline".to_string(),
            sources: vec![
                HfSourceEntry {
                    uri: "hf://onlyorg".to_string(),
                    anchor_columns: vec!["title".to_string()],
                    positive_columns: Vec::new(),
                    negative_columns: Vec::new(),
                    context_columns: Vec::new(),
                    text_columns: Vec::new(),
                    trust: None,
                    weight: None,
                    source_id: None,
                },
                HfSourceEntry {
                    uri: "hf://org/dataset/default/train".to_string(),
                    anchor_columns: vec!["title".to_string()],
                    positive_columns: vec!["body".to_string()],
                    negative_columns: Vec::new(),
                    context_columns: Vec::new(),
                    text_columns: Vec::new(),
                    trust: None,
                    weight: None,
                    source_id: None,
                },
            ],
        };

        let temp_root = tempdir().unwrap();
        let nl = platform_newline();
        fs::write(
            temp_root.path().join("Cargo.toml"),
            format!("[package]{nl}name='tmp'{nl}version='0.0.0'{nl}"),
        )
        .unwrap();
        fs::write(temp_root.path().join(".cache"), b"blocking-file").unwrap();

        with_current_dir(temp_root.path(), || {
            with_env_vars(&[(ENV_TRIPLETS_HF_TOKEN, "")], || {
                let built = build_hf_sources(&roots);
                assert_eq!(built.len(), 1);
            });
        });
    }

    #[test]
    #[serial(global_state)]
    fn build_hf_sources_duplicate_uri_gets_distinct_ids_and_snapshot_dirs() {
        // Two identical entries must produce two built sources whose IDs are
        // disambiguated (".0" / ".1") and whose snapshot directories are
        // independent (replica_0 vs replica_1).
        let dup_entry = HfSourceEntry {
            uri: "hf://org/dataset/default/train".to_string(),
            anchor_columns: vec!["title".to_string()],
            positive_columns: vec!["body".to_string()],
            negative_columns: Vec::new(),
            context_columns: Vec::new(),
            text_columns: Vec::new(),
            trust: None,
            weight: None,
            source_id: None,
        };
        let roots = HfListRoots {
            source_list: "inline".to_string(),
            sources: vec![dup_entry.clone(), dup_entry],
        };

        let temp_root = tempdir().unwrap();
        let nl = platform_newline();
        fs::write(
            temp_root.path().join("Cargo.toml"),
            format!("[package]{nl}name='tmp'{nl}version='0.0.0'{nl}"),
        )
        .unwrap();

        with_current_dir(temp_root.path(), || {
            with_env_vars(&[(ENV_TRIPLETS_HF_TOKEN, "")], || {
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
            });
        });
    }

    #[test]
    #[serial(global_state)]
    fn build_hf_sources_shares_http_client_across_entries() {
        // All sources produced by build_hf_sources must share a single HTTP
        // client so that connection pooling and throttle state apply to the
        // aggregate outbound traffic rather than per-source.
        let entries: Vec<HfSourceEntry> = (0..3)
            .map(|i| HfSourceEntry {
                uri: "hf://org/dataset/default/train".to_string(),
                anchor_columns: vec!["title".to_string()],
                positive_columns: vec!["body".to_string()],
                negative_columns: Vec::new(),
                context_columns: Vec::new(),
                text_columns: Vec::new(),
                trust: None,
                weight: None,
                source_id: Some(format!("src_{i}")),
            })
            .collect();
        let roots = HfListRoots {
            source_list: "inline".to_string(),
            sources: entries,
        };

        let temp_root = tempdir().unwrap();
        let nl = platform_newline();
        fs::write(
            temp_root.path().join("Cargo.toml"),
            format!("[package]{nl}name='tmp'{nl}version='0.0.0'{nl}"),
        )
        .unwrap();

        with_current_dir(temp_root.path(), || {
            with_env_vars(&[(ENV_TRIPLETS_HF_TOKEN, "")], || {
                let built = build_hf_sources(&roots);
                assert_eq!(built.len(), 3, "all three sources should build");
            });
        });
    }

    #[test]
    fn manual_http_client_sharing_works() {
        // Pre-building a client and setting it on multiple configs should
        // produce working sources that share the same connection pool.
        let dir = tempdir().unwrap();
        let client =
            HuggingFaceRowSource::build_http_client(&test_config(dir.path().to_path_buf()))
                .expect("build_http_client should succeed");

        for i in 0..3 {
            let mut config = test_config(dir.path().join(format!("src_{i}")));
            config.text_columns = vec!["text".to_string()];
            config.http_client = Some(client.clone());
            let source = HuggingFaceRowSource::new(config);
            assert!(source.is_ok(), "source {i} with shared client should build");
        }
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
                negative_columns: Vec::new(),
                context_columns: Vec::new(),
                text_columns: Vec::new(),
                trust: None,
                source_id: None,
                weight: None,
            },
            HfSourceEntry {
                uri: "hf://org/dataset/default/train".to_string(),
                anchor_columns: vec!["title".to_string()],
                positive_columns: vec!["body".to_string()],
                negative_columns: Vec::new(),
                context_columns: Vec::new(),
                text_columns: Vec::new(),
                trust: None,
                weight: None,
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
    fn read_store_row_count_validates_payload_size_and_roundtrips_written_value() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let store_path = dir.path().join("rows.simdr");
        let store = DataStore::open(&store_path).unwrap();

        store
            .write(HF_SHARD_STORE_META_ROWS_KEY, &[1u8, 2, 3])
            .unwrap();
        let err = source.read_store_row_count(&store).unwrap_err();
        let message = format!("{err}");
        assert!(message.contains("payload size mismatch"));

        source.write_store_row_count(&store, 7).unwrap();
        assert_eq!(source.read_store_row_count(&store).unwrap(), 7);
    }

    #[test]
    fn read_store_row_count_returns_zero_when_meta_key_is_absent() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let store_path = dir.path().join("empty.simdr");
        let store = DataStore::open(&store_path).unwrap();

        assert_eq!(source.read_store_row_count(&store).unwrap(), 0);
    }

    #[test]
    fn shard_store_path_for_passthrough_when_already_simdr() {
        let path = PathBuf::from("cache/shard.simdr");
        let mapped = HuggingFaceRowSource::shard_store_path_for(&path);
        assert_eq!(mapped, path);
    }

    #[test]
    fn get_or_open_shard_store_reuses_cached_handle_and_prune_keeps_active_only() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);

        let store_a = dir.path().join("a.simdr");
        let store_b = dir.path().join("b.simdr");

        let first = source.get_or_open_shard_store(&store_a).unwrap();
        let second = source.get_or_open_shard_store(&store_a).unwrap();
        assert!(Arc::ptr_eq(&first, &second));

        let _third = source.get_or_open_shard_store(&store_b).unwrap();
        {
            let cache = source.config.store_cache.lock().unwrap();
            assert!(cache.contains_key(&store_a));
            assert!(cache.contains_key(&store_b));
        }

        source.prune_store_cache_to_shards(&[ShardIndex {
            path: store_a.clone(),
            global_start: 0,
            row_count: 1,
            parquet_row_groups: vec![(0, 1)],
            remote_candidate: None,
        }]);

        let cache = source.config.store_cache.lock().unwrap();
        assert!(cache.contains_key(&store_a));
        assert!(!cache.contains_key(&store_b));
    }

    #[test]
    fn parquet_cache_row_group_rows_for_hits_cache_and_evicts_lru() {
        let dir = tempdir().unwrap();
        let path_a = dir.path().join("a.parquet");
        let path_b = dir.path().join("b.parquet");
        write_parquet_fixture(&path_a, &[("a1", "alpha")]);
        write_parquet_fixture(&path_b, &[("b1", "beta")]);

        let mut cache = ParquetCache::default();
        let rows_a_first = cache.row_group_rows_for("hf_test", &path_a, 0, 1).unwrap();
        let rows_a_second = cache.row_group_rows_for("hf_test", &path_a, 0, 1).unwrap();
        assert!(Arc::ptr_eq(&rows_a_first, &rows_a_second));

        let _rows_b = cache.row_group_rows_for("hf_test", &path_b, 0, 1).unwrap();
        assert_eq!(cache.row_groups.len(), 1);
        assert!(cache.row_groups.contains_key(&(path_b.clone(), 0)));
        assert!(!cache.row_groups.contains_key(&(path_a.clone(), 0)));
    }

    #[test]
    fn refresh_row_group_order_removes_existing_key_and_ignores_missing() {
        let key_a = (PathBuf::from("a.parquet"), 0usize);
        let key_b = (PathBuf::from("b.parquet"), 0usize);
        let mut order = VecDeque::from([key_a.clone(), key_b.clone(), key_a.clone()]);

        ParquetCache::refresh_row_group_order(&mut order, &key_a);
        assert_eq!(order, VecDeque::from([key_b.clone(), key_a.clone()]));

        let missing = (PathBuf::from("missing.parquet"), 0usize);
        ParquetCache::refresh_row_group_order(&mut order, &missing);
        assert_eq!(order, VecDeque::from([key_b, key_a]));
    }

    #[test]
    fn build_eligible_rows_from_store_shard_uses_global_offsets() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);

        let store_path = dir.path().join("eligible.simdr");
        write_simdr_fixture(&store_path, &[("r1", "alpha"), ("r2", "beta")]);

        let shards = vec![ShardIndex {
            path: store_path,
            global_start: 5,
            row_count: 2,
            parquet_row_groups: vec![(0, 2)],
            remote_candidate: None,
        }];

        let eligible = source.build_eligible_rows_from_shards(&shards).unwrap();
        assert_eq!(eligible, vec![5, 6]);
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
    fn all_candidates_from_parquet_manifest_returns_all_with_sizes() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        // Hub API tree endpoint format: array of {"path": "...", "size": N} objects
        let payload = json!([
            {"type": "file", "path": "train/000.parquet", "size": 100},
            {"type": "file", "path": "train/001.ndjson", "size": 200},
            {"type": "file", "path": "train/002.txt", "size": 50}
        ]);

        let (candidates, sizes, matched) =
            HuggingFaceRowSource::all_candidates_from_parquet_manifest(&config, &payload).unwrap();
        assert_eq!(candidates.len(), 2);
        assert!(candidates.iter().any(|c| c.ends_with("train/000.parquet")));
        assert!(candidates.iter().any(|c| c.ends_with("train/001.ndjson")));
        assert_eq!(sizes.len(), 2, "tree format provides sizes");
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
        let complete_candidate = format!("{HF_REMOTE_URL_PREFIX}train/000.parquet");
        let complete_target =
            HuggingFaceRowSource::candidate_target_path(&config, &complete_candidate);
        fs::create_dir_all(complete_target.parent().unwrap()).unwrap();
        fs::write(&complete_target, vec![1u8; 7]).unwrap();

        // A parquet file with the WRONG size — stale/incomplete, must be deleted.
        let stale_candidate = format!("{HF_REMOTE_URL_PREFIX}train/001.parquet");
        let stale_target = HuggingFaceRowSource::candidate_target_path(&config, &stale_candidate);
        fs::create_dir_all(stale_target.parent().unwrap()).unwrap();
        fs::write(&stale_target, vec![2u8; 3]).unwrap();

        let payload = json!([
            {"type": "file", "path": "train/000.parquet", "size": 7},
            {"type": "file", "path": "train/001.parquet", "size": 9}
        ]);

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
        let candidate = format!("{HF_REMOTE_URL_PREFIX}train/blocked.parquet");
        let target = HuggingFaceRowSource::candidate_target_path(&config, &candidate);
        fs::create_dir_all(&target).unwrap();

        let payload = json!([
            {"type": "file", "path": "train/blocked.parquet", "size": 1}
        ]);

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
            shards: vec![
                ShardIndex {
                    path: manifest_file,
                    global_start: 0,
                    row_count: 1,
                    parquet_row_groups: vec![(0, 1)],
                    remote_candidate: None,
                },
                ShardIndex {
                    path: local_file,
                    global_start: 1,
                    row_count: 1,
                    parquet_row_groups: Vec::new(),
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
            parquet_row_groups: vec![(0, 2), (2, 2), (4, 2)],
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
    fn target_matches_expected_size_is_false_for_missing_path() {
        let dir = tempdir().unwrap();
        let missing = dir.path().join("missing.bin");
        assert!(!HuggingFaceRowSource::target_matches_expected_size(
            &missing,
            Some(1)
        ));
    }

    #[test]
    fn candidate_target_path_uses_bare_path_when_no_resolve_segment() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        // Bare relative path from tree endpoint (no /resolve/ segment)
        let candidate = "url::train/000.parquet";
        let target = HuggingFaceRowSource::candidate_target_path(&config, candidate);
        assert!(target.ends_with("_parquet_manifest/train/000.parquet"));
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
            parquet_row_groups: Vec::new(),
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
            state.shards.clear();
        }

        let mut out = Vec::new();
        let err = source.read_row_batch(&[0], &mut out, Some(1));
        assert!(err.is_err());
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
            state.shards = vec![ShardIndex {
                path,
                global_start: 0,
                row_count: 3,
                parquet_row_groups: vec![(0, 3)],
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
            shards: vec![ShardIndex {
                path: shard_path,
                global_start: 0,
                row_count: 1,
                parquet_row_groups: vec![(0, 1)],
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
            shards: vec![
                ShardIndex {
                    path: first.clone(),
                    global_start: 0,
                    row_count: 1,
                    parquet_row_groups: vec![(0, 1)],
                    remote_candidate: None,
                },
                ShardIndex {
                    path: second.clone(),
                    global_start: 1,
                    row_count: 1,
                    parquet_row_groups: vec![(0, 1)],
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
            shards: vec![ShardIndex {
                path: protected.clone(),
                global_start: 0,
                row_count: 1,
                parquet_row_groups: vec![(0, 1)],
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
        let http_runtime = Arc::new(HuggingFaceRowSource::build_http_runtime(&config).unwrap());
        let http_client = test_http_client();
        let source = HuggingFaceRowSource {
            config,
            http_runtime,
            http_client,
            sampler_config: Arc::new(Mutex::new(None)),
            state: Arc::new(Mutex::new(SourceState {
                materialized_rows: 0,
                shards: Vec::new(),
                remote_candidates: None,
                remote_candidate_sizes: HashMap::new(),
                next_remote_idx: 0,
                remote_candidate_order: Vec::new(),
            })),
            cache: Arc::new(Mutex::new(RowCache::default())),
            parquet_cache: Arc::new(Mutex::new(ParquetCache::default())),
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
        fs::write(dir.path().join("notes.dat"), b"plain").unwrap();
        let config = test_config(dir.path().to_path_buf());

        let err = HuggingFaceRowSource::build_shard_index(&config)
            .expect_err("build_shard_index should fail");
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
    fn index_single_shard_errors_for_missing_file() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let missing = dir.path().join("missing.ndjson");

        let err = HuggingFaceRowSource::index_single_shard_for_test(&config, &missing, 0)
            .err()
            .expect("index_single_shard should fail");
        assert!(matches!(err, SamplerError::SourceUnavailable { .. }));
    }

    #[test]
    fn index_single_shard_detects_corrupted_store() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let store_path = dir.path().join("shard.simdr");

        // Write a store with 3 valid rows AND a metadata key claiming 5 rows.
        // The last claimed row (index 4) does not exist — corruption.
        write_simdr_fixture(&store_path, &[("r0", "zero"), ("r1", "one"), ("r2", "two")]);
        let store = DataStore::open(&store_path).expect("open store");
        store
            .write(HF_SHARD_STORE_META_ROWS_KEY, &(5u64).to_le_bytes())
            .expect("overwrite meta with inflated count");
        drop(store);

        // index_single_shard should detect the gap, delete the corrupt file,
        // and return None.
        let (maybe_shard, _) =
            HuggingFaceRowSource::index_single_shard_for_test(&config, &store_path, 0).expect(
                "corrupted store should not produce a hard error — it deletes and returns None",
            );
        assert!(maybe_shard.is_none(), "corrupt store should be skipped");
        assert!(!store_path.exists(), "corrupt store file should be deleted");
    }

    #[test]
    fn index_single_shard_jsonl_returns_dummy_count() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("rows.ndjson");
        fs::write(
            &path,
            b"{\"text\":\"a\"}\n{\"text\":\"b\"}\n{\"text\":\"c\"}\n",
        )
        .unwrap();
        let config = test_config(dir.path().to_path_buf());

        let shard = HuggingFaceRowSource::index_single_shard_for_test(&config, &path, 5)
            .unwrap()
            .0
            .unwrap();
        assert_eq!(shard.global_start, 5);
        assert_eq!(shard.row_count, 1); // Dummy count for transient text files
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
        // Cap small enough that old file alone exceeds it, triggering eviction,
        // but large enough that a single-row .simdr store (~4 KiB) fits.
        config.local_disk_cap_bytes = Some(6_144);
        let source = test_source(config.clone());

        let manifest_root = source.manifest_cache_root();
        fs::create_dir_all(&manifest_root).unwrap();
        let old_path = manifest_root.join("old.parquet");
        // Large enough to exceed the disk cap on its own.
        fs::write(&old_path, vec![1u8; 8_192]).unwrap();

        let payload = b"{\"text\":\"new\"}\n".to_vec();
        let server = spawn_one_shot_http(payload);
        let base_url = server.url().to_string();
        let candidate =
            format!("url::{base_url}/datasets/org/ds/resolve/main/train/new-shard.ndjson");
        let new_path = HuggingFaceRowSource::candidate_store_path(&config, &candidate);

        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 1;
            state.shards = vec![ShardIndex {
                path: old_path.clone(),
                global_start: 0,
                row_count: 1,
                parquet_row_groups: vec![(0, 1)],
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

        // Eviction removes at least one shard once disk cap is exceeded.
        // Which shard is removed can vary on filesystems with coarse mtime
        // resolution (tie-break is path order), so assert eviction semantics
        // rather than a specific filename.
        assert!(
            !(old_path.exists() && new_path.exists()),
            "expected at least one manifest shard to be evicted"
        );
        {
            let state = source.state.lock().unwrap();
            assert!(!state.shards.is_empty(), "at least one shard should remain");
        }
        let cache = source.cache.lock().unwrap();
        assert!(cache.rows.is_empty());
        assert!(cache.order.is_empty());
    }

    #[test]
    fn default_triplet_recipes_text_only_mode_returns_simcse_recipe() {
        // test_config() leaves anchor_columns empty → text-only mode.
        // A single SimCSE-style recipe with allow_same_anchor_positive must be returned.
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        assert!(
            config.anchor_columns.is_empty(),
            "test_config must be in text-only mode"
        );
        let source = test_source(config);
        let recipes = source.default_triplet_recipes();
        assert_eq!(recipes.len(), 1);
        assert_eq!(recipes[0].name, HF_RECIPE_TEXT_SIMCSE_WRONG_ARTICLE);
        assert!(
            recipes[0].allow_same_anchor_positive,
            "SimCSE recipe must allow same anchor/positive text"
        );
        assert_eq!(recipes[0].weight, 1.0);
    }

    #[test]
    fn default_triplet_recipes_role_mode_returns_two_recipes() {
        // When anchor_columns is non-empty the source is in role-based mode and
        // must return the two standard (anchor-context, anchor-anchor) recipes,
        // neither of which allows same anchor/positive text.
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.anchor_columns = vec!["title".to_string()];
        config.positive_columns = vec!["body".to_string()];
        let source = test_source(config);
        let recipes = source.default_triplet_recipes();
        assert_eq!(recipes.len(), 2);
        assert_eq!(recipes[0].name, HF_RECIPE_ANCHOR_CONTEXT_WRONG_ARTICLE);
        assert_eq!(recipes[1].name, HF_RECIPE_ANCHOR_ANCHOR_WRONG_ARTICLE);
        assert_eq!(recipes[0].weight, 0.75);
        assert_eq!(recipes[1].weight, 0.25);
        assert!(
            !recipes[0].allow_same_anchor_positive,
            "standard recipes must not allow same anchor/positive"
        );
        assert!(
            !recipes[1].allow_same_anchor_positive,
            "standard recipes must not allow same anchor/positive"
        );
    }

    #[test]
    fn download_and_materialize_shard_url_short_circuits_when_cached_complete() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let candidate = "url::https://host/datasets/org/ds/resolve/main/train/ok.ndjson";
        let target = HuggingFaceRowSource::candidate_target_path(&config, candidate);
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::write(&target, b"ok").unwrap();

        let client = test_http_client();
        let resolved = HuggingFaceRowSource::download_and_materialize_shard(
            &client,
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
        let server = spawn_one_shot_http(payload.clone());
        let base_url = server.url().to_string();
        let candidate = format!("url::{base_url}/datasets/org/ds/resolve/main/train/part-x.ndjson");
        let target = HuggingFaceRowSource::candidate_target_path(&config, &candidate);
        let temp_target = target.with_extension("part");
        fs::create_dir_all(temp_target.parent().unwrap()).unwrap();
        fs::write(&temp_target, b"stale").unwrap();

        let client = test_http_client();
        let out = HuggingFaceRowSource::download_and_materialize_shard(
            &client,
            &config,
            &candidate,
            None,
            "shard 1/1",
        )
        .unwrap();

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
        let server = spawn_one_shot_http(payload);
        let base_url = server.url().to_string();
        let candidate =
            format!("url::{base_url}/datasets/org/ds/resolve/main/train/part-empty.ndjson");

        {
            let mut state = source.state.lock().unwrap();
            state.remote_candidates = Some(vec![candidate]);
            state.next_remote_idx = 0;
        }

        assert!(source.download_next_remote_shard().unwrap());
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
            state.shards = vec![ShardIndex {
                path: dir.path().join("missing.parquet"),
                global_start: 0,
                row_count: 1,
                parquet_row_groups: vec![(0, 1)],
                remote_candidate: None,
            }];
        }

        let mut out = Vec::new();
        let err = source.read_row_batch(&[0], &mut out, Some(1));
        assert!(err.is_err());
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
        let simdr_path = dir.path().join("rows.simdr");
        write_simdr_fixture(&simdr_path, &[("r1", "a"), ("r2", "b")]);
        let mut config = test_config(dir.path().to_path_buf());
        config.refresh_batch_multiplier = 1;
        let source = test_source(config.clone());
        let shard = HuggingFaceRowSource::index_single_shard_for_test(&config, &simdr_path, 0)
            .unwrap()
            .0
            .unwrap();
        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 2;
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

        let shard = HuggingFaceRowSource::index_single_shard_for_test(&config, &path, 0)
            .unwrap()
            .0
            .unwrap();
        assert_eq!(shard.row_count, 3);
        // All shards are now .simdr stores with O(1) random access
    }

    #[test]
    fn read_row_batch_reads_parquet_rows_and_uses_cache_on_repeat() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("rows.parquet");
        write_parquet_fixture(&path, &[("r10", "ten"), ("r11", "eleven")]);

        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());
        let shard = HuggingFaceRowSource::index_single_shard_for_test(&config, &path, 0)
            .unwrap()
            .0
            .unwrap();
        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 2;
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
        let server = spawn_one_shot_http(payload);
        let base_url = server.url().to_string();
        let candidate =
            format!("url::{base_url}/datasets/org/ds/resolve/main/train/persisted.ndjson");

        {
            let mut state = source.state.lock().unwrap();
            state.remote_candidates = Some(vec![candidate]);
            state.next_remote_idx = 0;
        }

        assert!(source.ensure_row_available(0).unwrap());

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
            // Set up remote candidates so headroom is applied
            state.remote_candidates = Some(vec![
                "url::http://a/0.parquet".to_string(),
                "url::http://a/1.parquet".to_string(),
            ]);
            state.next_remote_idx = 0;
        }

        // headroom = ingestion_max_records * multiplier = 10 * 3 = 30
        // known (5) < headroom, expansion = 30; upper = 5 + 30 = 35
        assert_eq!(source.reported_record_count().unwrap(), 35);

        let sampler = SamplerConfig {
            ingestion_max_records: 2,
            ..SamplerConfig::default()
        };
        source.configure_sampler(&sampler);

        // headroom = 2 * 3 = 6; known (5) < headroom, expansion = 6; upper = 5 + 6 = 11
        assert_eq!(source.reported_record_count().unwrap(), 11);
    }

    #[test]
    fn build_shard_index_skips_empty_files_and_keeps_non_empty() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("a.ndjson"), b"").unwrap();
        fs::write(dir.path().join("b.ndjson"), b"{\"text\":\"x\"}\n").unwrap();
        let config = test_config(dir.path().to_path_buf());

        let (shards, discovered) = HuggingFaceRowSource::build_shard_index(&config).unwrap();
        assert_eq!(discovered, 1);
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0].row_count, 1);
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
        let body = serde_json::to_string(&json!([
            {"type": "file", "path": "https://host/datasets/x/resolve/main/train/0.parquet", "size": 100}
        ]))
        .unwrap();

        let (candidates, sizes, matched) =
            HuggingFaceRowSource::parse_parquet_manifest_response(&config, &body).unwrap();
        assert_eq!(candidates.len(), 1);
        assert!(!sizes.is_empty());
        assert_eq!(matched, 1);
    }

    #[test]
    #[serial(global_state)]
    fn list_remote_candidates_from_parquet_manifest_uses_test_endpoint_override() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        let body = serde_json::to_vec(&json!([
            {"type": "file", "path": "https://host/datasets/x/resolve/main/train/0.parquet", "size": 100}
        ]))
        .unwrap();
        let server = spawn_one_shot_http(body);
        let base_url = server.url().to_string();

        config.parquet_endpoint = base_url;
        let client = test_http_client();
        let (candidates, sizes, matched) =
            HuggingFaceRowSource::list_remote_candidates_from_parquet_manifest(&client, &config)
                .unwrap();

        assert_eq!(candidates.len(), 1);
        assert!(!sizes.is_empty());
        assert_eq!(matched, 1);
    }

    #[test]
    fn config_endpoint_fallback_for_empty_env_values() {
        // `test_config` overrides endpoints to `TEST_UNREACHABLE_URL` for
        // network isolation.  This test verifies the DEFAULT values from
        // `HuggingFaceRowsConfig::new()`, so we call it directly.
        let dir = tempdir().unwrap();

        let c =
            HuggingFaceRowsConfig::new("ep_test", "org/dataset", "default", "train", dir.path());
        assert_eq!(c.parquet_endpoint, HF_PARQUET_DEFAULT_ENDPOINT);
    }

    #[test]
    #[serial(global_state)]
    fn list_remote_candidates_returns_manifest_candidates() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        let body = serde_json::to_vec(&json!([
            {"type": "file", "path": "https://host/datasets/x/resolve/main/train/1.ndjson", "size": 100}
        ]))
        .unwrap();
        let server = spawn_one_shot_http(body);
        let base_url = server.url().to_string();

        config.parquet_endpoint = base_url;
        let client = test_http_client();
        let (candidates, sizes) =
            HuggingFaceRowSource::list_remote_candidates(&client, &config).unwrap();

        assert_eq!(candidates.len(), 1);
        assert!(!sizes.is_empty());
        assert!(candidates[0].ends_with("/1.ndjson"));
    }

    #[test]
    fn list_remote_candidates_scopes_tree_to_config_name() {
        // Verify that the tree endpoint URL is scoped to the config_name
        // subdirectory so we don't pull files from other configs
        // (e.g. wikimedia/wikipedia/20231101.en should not list .fr files).
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.config_name = "20231101.en".to_string();

        // Spawn a mock server that records the request path.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{addr}");
        let recorded_path: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let recorded_path_clone = Arc::clone(&recorded_path);
        let handle = std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 4096];
                let n = stream.read(&mut buf).unwrap_or(0);
                let request = String::from_utf8_lossy(&buf[..n]);
                let path = request
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .map(|s| s.to_string());
                *recorded_path_clone.lock().unwrap() = path;
                let body = br#"[]"#;
                let headers = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(headers.as_bytes());
                let _ = stream.write_all(body);
            }
        });

        config.parquet_endpoint = base_url;
        let client = test_http_client();
        let _ =
            HuggingFaceRowSource::list_remote_candidates_from_parquet_manifest(&client, &config);
        handle.join().unwrap();

        let path = recorded_path.lock().unwrap();
        let path = path.as_deref().expect("no request recorded");
        assert!(
            path.contains("/tree/main/20231101.en"),
            "tree endpoint URL must be scoped to config_name; got: {path}"
        );
        assert!(
            !path.contains("/tree/main?"),
            "tree endpoint must NOT use root path when config_name is set; got: {path}"
        );
    }

    #[test]
    fn list_remote_candidates_uses_root_tree_for_default_config() {
        // When config_name is "default", the tree endpoint should hit the repo root.
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.config_name = "default".to_string();

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{addr}");
        let recorded_path: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let recorded_path_clone = Arc::clone(&recorded_path);
        let handle = std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 4096];
                let n = stream.read(&mut buf).unwrap_or(0);
                let request = String::from_utf8_lossy(&buf[..n]);
                let path = request
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .map(|s| s.to_string());
                *recorded_path_clone.lock().unwrap() = path;
                let body = br#"[]"#;
                let headers = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(headers.as_bytes());
                let _ = stream.write_all(body);
            }
        });

        config.parquet_endpoint = base_url;
        let client = test_http_client();
        let _ =
            HuggingFaceRowSource::list_remote_candidates_from_parquet_manifest(&client, &config);
        handle.join().unwrap();

        let path = recorded_path.lock().unwrap();
        let path = path.as_deref().expect("no request recorded");
        assert!(
            path.contains("/tree/main?recursive=true"),
            "default config must hit repo root; got: {path}"
        );
    }

    #[test]
    #[serial(global_state)]
    fn list_remote_candidates_with_runtime_returns_manifest_candidates() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        let runtime = HuggingFaceRowSource::build_http_runtime(&config).unwrap();
        let body = serde_json::to_vec(&json!([
            {"type": "file", "path": "https://host/datasets/x/resolve/main/train/2.ndjson", "size": 100}
        ]))
        .unwrap();
        let server = spawn_one_shot_http(body);
        let base_url = server.url().to_string();

        config.parquet_endpoint = base_url;
        let client = test_http_client();
        let (candidates, sizes) = HuggingFaceRowSource::list_remote_candidates_with_runtime(
            &client,
            &config,
            Some(&runtime),
        )
        .unwrap();

        assert_eq!(candidates.len(), 1);
        assert!(!sizes.is_empty());
        assert!(candidates[0].ends_with("/2.ndjson"));
    }

    #[test]
    #[serial(global_state)]
    fn list_remote_candidates_does_not_fall_back_when_all_manifest_shards_cached() {
        // Regression test: list_remote_candidates must return the full manifest
        // candidate list when a parquet manifest exists, regardless of whether all
        // shards are already cached on disk.
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());

        // Pre-create the .simdr store target so the manifest entry is "fully cached".
        let shard_url = "https://host/datasets/org/ds/resolve/main/train/part-000.ndjson";
        let candidate = format!("{HF_REMOTE_URL_PREFIX}{shard_url}");
        let target = HuggingFaceRowSource::candidate_target_path(&config, &candidate);
        let store_target = HuggingFaceRowSource::shard_store_path_for(&target);
        fs::create_dir_all(store_target.parent().unwrap()).unwrap();
        fs::write(&store_target, b"cached").unwrap();

        let body = serde_json::to_vec(&json!([
            {"type": "file", "path": shard_url, "size": 100}
        ]))
        .unwrap();
        let server = spawn_one_shot_http(body);
        let base_url = server.url().to_string();

        // Must return the full manifest candidate list without falling through to hf-hub.
        config.parquet_endpoint = base_url;
        let client = test_http_client();
        let (candidates, sizes) =
            HuggingFaceRowSource::list_remote_candidates(&client, &config).unwrap();

        assert_eq!(
            candidates.len(),
            1,
            "fully-cached shard must still appear in candidates (cache ≠ order)"
        );
        assert!(!sizes.is_empty());
        assert!(candidates[0].ends_with(shard_url));
    }

    #[test]
    #[serial(global_state)]
    fn list_remote_candidates_from_parquet_manifest_errors_when_endpoint_unreachable() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.parquet_endpoint = TEST_UNREACHABLE_URL.to_string();

        let client = test_http_client();
        let result =
            HuggingFaceRowSource::list_remote_candidates_from_parquet_manifest(&client, &config);
        assert!(result.is_err());
    }

    #[test]
    fn download_and_materialize_shard_downloads_url_candidate() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let payload = b"{\"text\":\"a\"}\n{\"text\":\"b\"}\n".to_vec();
        let server = spawn_one_shot_http(payload.clone());
        let base_url = server.url().to_string();
        let candidate =
            format!("url::{base_url}/datasets/org/ds/resolve/main/train/part-000.ndjson");

        let client = test_http_client();
        let target = HuggingFaceRowSource::download_and_materialize_shard(
            &client,
            &config,
            &candidate,
            None,
            "shard 1/1",
        )
        .unwrap();

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
        let server = spawn_one_shot_http(payload.clone());
        let base_url = server.url().to_string();
        let candidate =
            format!("url::{base_url}/datasets/org/ds/resolve/main/train/part-009.ndjson");

        let target = HuggingFaceRowSource::candidate_target_path(&config, &candidate);
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::write(&target, b"bad").unwrap();

        let client = test_http_client();
        let refreshed = HuggingFaceRowSource::download_and_materialize_shard(
            &client,
            &config,
            &candidate,
            Some(payload.len() as u64),
            "shard 1/1",
        )
        .unwrap();

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
        let server = spawn_one_shot_http(payload);
        let base_url = server.url().to_string();
        let candidate =
            format!("url::{base_url}/datasets/org/ds/resolve/main/train/part-222.parquet");

        {
            let mut state = source.state.lock().unwrap();
            state.remote_candidates = Some(vec![candidate.clone()]);
            state.next_remote_idx = 0;
        }

        assert!(source.download_next_remote_shard().unwrap());

        let parquet_target = HuggingFaceRowSource::candidate_target_path(&config, &candidate);
        let store_target = HuggingFaceRowSource::shard_store_path_for(&parquet_target);

        assert!(store_target.exists());
        assert!(!parquet_target.exists());

        let state = source.state.lock().unwrap();
        assert_eq!(state.shards.len(), 1);
        assert_eq!(state.shards[0].path, store_target);
        assert_eq!(state.materialized_rows, 2);
    }

    // When transcode_transient_shard_to_store takes the early-return path
    // (simdr store already fully populated), it must still delete the input
    // transient file and return a ShardIndex
    #[test]
    fn transcode_transient_shard_to_store_early_return_cleans_up_transient() {
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
            parquet_row_groups: vec![(0, 2)],
            remote_candidate: None,
        };

        let result = source
            .transcode_transient_shard_to_store(&shard)
            .expect("transcode must succeed");

        // Stale parquet must be gone.
        assert!(
            !parquet_path.exists(),
            "stale parquet must be removed on early return"
        );

        // Simdr store must still be present.
        assert!(store_path.exists(), "simdr store must survive early return");

        // Returned shard must point to the store.
        let returned = result.expect("early return must yield Some(ShardIndex)");
        assert_eq!(returned.path, store_path);
        assert_eq!(returned.row_count, 2);
    }

    #[test]
    fn download_next_remote_shard_materializes_and_indexes_rows() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let payload = b"{\"text\":\"a\"}\n{\"text\":\"b\"}\n".to_vec();
        let server = spawn_one_shot_http(payload);
        let base_url = server.url().to_string();
        let candidate =
            format!("url::{base_url}/datasets/org/ds/resolve/main/train/part-001.ndjson");

        {
            let mut state = source.state.lock().unwrap();
            state.remote_candidates = Some(vec![candidate.clone()]);
            state.remote_candidate_sizes.insert(candidate, 24);
            state.next_remote_idx = 0;
        }

        assert!(source.download_next_remote_shard().unwrap());

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
        let server = spawn_one_shot_http(payload);
        let base_url = server.url().to_string();
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
        let server_a = spawn_one_shot_http(payload_a);
        let base_a = server_a.url().to_string();
        let server_b = spawn_one_shot_http(payload_b);
        let base_b = server_b.url().to_string();
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

        let candidate = format!(
            "url::{TEST_UNREACHABLE_URL}/datasets/org/ds/resolve/main/train/pre-cached.ndjson"
        );
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
    fn download_next_remote_shard_detects_stale_shard_by_size() {
        // When a cached store exists with a stored source_size that differs from
        // the current manifest's expected_bytes, the store should be deleted and
        // the shard re-downloaded.
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        // Create a real .simdr store with source_size = 100.
        let candidate =
            format!("url::{TEST_UNREACHABLE_URL}/datasets/org/ds/resolve/main/train/stale.ndjson");
        let store_path = HuggingFaceRowSource::candidate_store_path(&config, &candidate);
        fs::create_dir_all(store_path.parent().unwrap()).unwrap();
        write_simdr_fixture(&store_path, &[("r0", "row")]);
        {
            let store = DataStore::open(&store_path).unwrap();
            store
                .write(HF_SHARD_STORE_SOURCE_SIZE_KEY, &100u64.to_le_bytes())
                .unwrap();
        }

        // Manually populate the store cache with the handle so the stale
        // check can read source_size from it without opening a second handle.
        let cached_store = Arc::new(DataStore::open(&store_path).unwrap());
        source
            .config
            .store_cache
            .lock()
            .unwrap()
            .insert(store_path.clone(), cached_store);

        // Set up remote candidates with expected_bytes = 200 (≠ 100).
        {
            let mut state = source.state.lock().unwrap();
            state.remote_candidates = Some(vec![candidate.clone()]);
            state.remote_candidate_order = vec![0];
            state.remote_candidate_sizes.insert(candidate, 200);
            state.next_remote_idx = 0;
        }

        // The stale check should detect the mismatch, delete the store, and
        // attempt a download.  Since no HTTP server is running, the download
        // fails with SourceUnavailable — but the store should already be gone.
        let result = source.download_next_remote_shard();
        assert!(
            !store_path.exists(),
            "stale store file should be deleted before download attempt"
        );
        assert!(
            result.is_err(),
            "should fail with SourceUnavailable (no HTTP server for re-download)"
        );
        let err = result.unwrap_err();
        assert!(
            matches!(err, SamplerError::SourceUnavailable { .. }),
            "expected SourceUnavailable, got: {err:?}"
        );
    }

    #[test]
    fn download_next_remote_shard_preserves_fresh_shard_when_sizes_match() {
        // When a cached store exists and its stored source_size matches the
        // manifest's expected_bytes, the download is skipped normally.
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        // Create a store with source_size = 100 and set expected_bytes = 100.
        let candidate =
            format!("url::{TEST_UNREACHABLE_URL}/datasets/org/ds/resolve/main/train/fresh.ndjson");
        let store_path = HuggingFaceRowSource::candidate_store_path(&config, &candidate);
        fs::create_dir_all(store_path.parent().unwrap()).unwrap();
        write_simdr_fixture(&store_path, &[("r0", "row")]);
        {
            let store = DataStore::open(&store_path).unwrap();
            store
                .write(HF_SHARD_STORE_SOURCE_SIZE_KEY, &100u64.to_le_bytes())
                .unwrap();
        }

        // Populate the store cache for the stale check.
        let cached_store = Arc::new(DataStore::open(&store_path).unwrap());
        source
            .config
            .store_cache
            .lock()
            .unwrap()
            .insert(store_path.clone(), cached_store);

        {
            let mut state = source.state.lock().unwrap();
            state.remote_candidates = Some(vec![candidate.clone()]);
            state.remote_candidate_order = vec![0];
            state.remote_candidate_sizes.insert(candidate, 100);
            state.next_remote_idx = 0;
        }

        // Sizes match — should skip without any network call.
        assert!(
            source.download_next_remote_shard().unwrap(),
            "should return true (candidate consumed)"
        );
        assert!(store_path.exists(), "fresh store should NOT be deleted");
    }

    #[test]
    fn download_next_remote_shard_gz_materializes_true_row_count() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);

        // Create a 5-line .jsonl.gz payload
        use flate2::Compression;
        use flate2::write::GzEncoder;
        use std::io::Write;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        for i in 1..=5 {
            writeln!(encoder, r#"{{"id":"r{}","text":"line {}"}}"#, i, i).unwrap();
        }
        let gz_payload = encoder.finish().unwrap();

        let server = spawn_one_shot_http(gz_payload.clone());
        let base_url = server.url().to_string();
        let candidate = format!("url::{base_url}/datasets/org/ds/resolve/main/train/data.jsonl.gz");

        {
            let mut state = source.state.lock().unwrap();
            state.remote_candidates = Some(vec![candidate.clone()]);
            state
                .remote_candidate_sizes
                .insert(candidate, gz_payload.len() as u64);
            state.next_remote_idx = 0;
        }

        assert!(source.download_next_remote_shard().unwrap());

        let state = source.state.lock().unwrap();
        // Materialized rows must be 5 (true count), not 1 (dummy)
        assert_eq!(state.materialized_rows, 5);
        assert_eq!(state.shards.len(), 1);
    }

    #[test]
    fn download_next_remote_shard_gz_invalid_json_returns_error() {
        use flate2::Compression;
        use flate2::write::GzEncoder;
        use std::io::Write;

        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);

        // Create a .gz file with invalid JSON
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        writeln!(encoder, "this is not valid json").unwrap();
        let gz_payload = encoder.finish().unwrap();

        let server = spawn_one_shot_http(gz_payload.clone());
        let base_url = server.url().to_string();
        let candidate =
            format!("url::{base_url}/datasets/org/ds/resolve/main/train/invalid.jsonl.gz");

        {
            let mut state = source.state.lock().unwrap();
            state.remote_candidates = Some(vec![candidate.clone()]);
            state
                .remote_candidate_sizes
                .insert(candidate, gz_payload.len() as u64);
            state.next_remote_idx = 0;
        }

        let result = source.download_next_remote_shard();
        assert!(result.is_err());
        match result.unwrap_err() {
            SamplerError::SourceInconsistent { .. } => {} // Expected for invalid JSON
            other => panic!("expected SourceInconsistent, got: {other:?}"),
        }
    }

    #[test]
    fn download_next_remote_shard_gz_corrupt_stream_returns_error() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);

        // Corrupt gzip data
        let corrupt_payload = b"this is not valid gzip data".to_vec();

        let server = spawn_one_shot_http(corrupt_payload.clone());
        let base_url = server.url().to_string();
        let candidate =
            format!("url::{base_url}/datasets/org/ds/resolve/main/train/corrupt.jsonl.gz");

        {
            let mut state = source.state.lock().unwrap();
            state.remote_candidates = Some(vec![candidate.clone()]);
            state
                .remote_candidate_sizes
                .insert(candidate, corrupt_payload.len() as u64);
            state.next_remote_idx = 0;
        }

        let result = source.download_next_remote_shard();
        assert!(result.is_err());
        match result.unwrap_err() {
            SamplerError::SourceUnavailable { .. } => {} // Expected for corrupt stream
            other => panic!("expected SourceUnavailable, got: {other:?}"),
        }
    }

    #[test]
    fn remote_url_for_candidate_constructs_correct_urls() {
        // url:: prefix with full URL: returned as-is.
        let config = test_config(PathBuf::from("/tmp/snap"));
        let full_url =
            format!("url::{HF_DATASETS_BASE_URL}/org/ds/resolve/main/train/part-000.parquet");
        let result = HuggingFaceRowSource::remote_url_for_candidate(&config, &full_url);
        assert_eq!(
            result,
            format!("{HF_DATASETS_BASE_URL}/org/ds/resolve/main/train/part-000.parquet")
        );

        // url:: prefix with relative path (Hub API format): CDN prefix is constructed.
        let hub_relative = "url::data/train-00000-of-00001.parquet";
        let result = HuggingFaceRowSource::remote_url_for_candidate(&config, hub_relative);
        assert_eq!(
            result,
            format!(
                "{HF_DATASETS_BASE_URL}/org/dataset/resolve/main/data/train-00000-of-00001.parquet"
            )
        );

        // Bare path (hf-hub sibling fallback): CDN prefix is prepended.
        let bare_path = "data/train-00000-of-00001.parquet";
        let result = HuggingFaceRowSource::remote_url_for_candidate(&config, bare_path);
        assert_eq!(
            result,
            format!(
                "{HF_DATASETS_BASE_URL}/org/dataset/resolve/main/data/train-00000-of-00001.parquet"
            )
        );

        // Bare path with leading slash.
        let bare_path = "/data/train-00000-of-00001.parquet";
        let result = HuggingFaceRowSource::remote_url_for_candidate(&config, bare_path);
        assert_eq!(
            result,
            format!(
                "{HF_DATASETS_BASE_URL}/org/dataset/resolve/main/data/train-00000-of-00001.parquet"
            )
        );
    }

    #[test]
    #[serial(global_state)]
    fn fetch_remote_size_with_runtime_returns_content_length() {
        // A mock HTTP server that responds to HEAD with Content-Length.
        let payload = b"this is the shard content".to_vec();
        let server = TestHttpServer::new(200, payload.clone());
        let base_url = server.url().to_string();

        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.hf_token = None;

        let client = test_http_client();
        let runtime = HuggingFaceRowSource::build_http_runtime(&config).unwrap();
        let size = HuggingFaceRowSource::fetch_remote_size_with_runtime(
            &client, &config, &base_url, &runtime,
        )
        .unwrap();
        // Content-Length should match the payload size.
        assert_eq!(size, Some(payload.len() as u64));
    }

    #[test]
    #[serial(global_state)]
    fn fetch_remote_size_with_runtime_returns_none_on_non_success() {
        // A mock server returning 404 — HEAD should return Ok(None).
        let server = TestHttpServer::new(404, b"Not Found".to_vec());
        let base_url = server.url().to_string();

        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.hf_token = None;

        let client = test_http_client();
        let runtime = HuggingFaceRowSource::build_http_runtime(&config).unwrap();
        let size = HuggingFaceRowSource::fetch_remote_size_with_runtime(
            &client, &config, &base_url, &runtime,
        )
        .unwrap();
        assert_eq!(size, None, "non-2xx response should yield None");
    }

    #[test]
    #[serial(global_state)]
    fn download_next_remote_shard_detects_stale_shard_via_head() {
        // When the manifest does NOT provide expected_bytes (hf-hub sibling
        // fallback), but a cached store exists on disk with a stored source_size
        // that differs from the remote Content-Length (obtained via HTTP HEAD),
        // the store should be deleted so the shard gets re-downloaded.
        //
        // This test verifies that staleness is detected correctly by checking
        // the behaviour after the HEAD request:
        //   • stored source_size = 100
        //   • HEAD Content-Length = 200  (mismatch → store is deleted)
        //
        // The mock server serves the GET response body directly, so the
        // re-download may succeed.  The critical assertion is that the
        // original store is gone.
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        // Start a mock HTTP server — must stay alive for HEAD + any GET.
        // Use valid JSON so the transcoding pipeline succeeds.
        let payload = b"{\"text\":\"valid\",\"padding\":\"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\"}\n".to_vec();
        let server = TestHttpServer::new(200, payload);
        let base_url = server.url().to_string();

        // Candidate uses url:: prefix so the HEAD targets the mock server.
        let candidate = format!("url::{base_url}/resolve/main/train/stale-shard.ndjson");
        let store_path = HuggingFaceRowSource::candidate_store_path(&config, &candidate);
        fs::create_dir_all(store_path.parent().unwrap()).unwrap();
        write_simdr_fixture(&store_path, &[("r0", "row")]);
        {
            let store = DataStore::open(&store_path).unwrap();
            store
                .write(HF_SHARD_STORE_SOURCE_SIZE_KEY, &100u64.to_le_bytes())
                .unwrap();
        }

        // Snapshot the on-disk content so we can detect replacement.
        let original_content = fs::read(&store_path).unwrap();

        let cached_store = Arc::new(DataStore::open(&store_path).unwrap());
        source
            .config
            .store_cache
            .lock()
            .unwrap()
            .insert(store_path.clone(), cached_store);

        {
            let mut state = source.state.lock().unwrap();
            state.remote_candidates = Some(vec![candidate.clone()]);
            state.remote_candidate_order = vec![0];
            // No expected_bytes in remote_candidate_sizes — simulates the
            // hf-hub sibling fallback where sizes are unknown.
            state.next_remote_idx = 0;
        }

        // The HEAD request returns Content-Length: 200.  The stored
        // source_size is 100, so the staleness check should delete the
        // store and re-download.  The old store MUST be gone.
        let result = source.download_next_remote_shard();

        // The old store was deleted (HEAD detected mismatch).
        // A new store may or may not have been created depending on
        // whether the GET download + transcode succeeded.
        assert!(
            fs::read(&store_path).ok().as_deref() != Some(&original_content),
            "stale store content should have been replaced (HEAD detected size mismatch)"
        );

        // The candidate should have been consumed either way.
        assert!(
            result.is_ok(),
            "download may fail or succeed; the candidate should be consumed: {err:?}",
            err = result.as_ref().unwrap_err()
        );
    }

    #[test]
    #[serial(global_state)]
    fn download_next_remote_shard_preserves_fresh_shard_via_head() {
        // When the manifest does NOT provide expected_bytes, but a cached
        // store exists with a stored size that matches the remote Content-Length
        // from HEAD, the store should be preserved and the download skipped.
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        // Create a store with source_size matching the mock server's payload.
        let server = TestHttpServer::new(200, vec![0u8; 100]);
        let base_url = server.url().to_string();

        let candidate = format!("url::{base_url}/resolve/main/train/fresh-shard.ndjson");
        let store_path = HuggingFaceRowSource::candidate_store_path(&config, &candidate);
        fs::create_dir_all(store_path.parent().unwrap()).unwrap();
        write_simdr_fixture(&store_path, &[("r0", "row")]);
        {
            let store = DataStore::open(&store_path).unwrap();
            store
                .write(HF_SHARD_STORE_SOURCE_SIZE_KEY, &100u64.to_le_bytes())
                .unwrap();
        }

        let cached_store = Arc::new(DataStore::open(&store_path).unwrap());
        source
            .config
            .store_cache
            .lock()
            .unwrap()
            .insert(store_path.clone(), cached_store);

        {
            let mut state = source.state.lock().unwrap();
            state.remote_candidates = Some(vec![candidate.clone()]);
            state.remote_candidate_order = vec![0];
            // No expected_bytes in remote_candidate_sizes.
            state.next_remote_idx = 0;
        }

        // Sizes match (100 == 100) — should skip without download.
        let result = source.download_next_remote_shard();
        assert!(
            result.is_ok(),
            "expected Ok, got: {err:?}",
            err = result.as_ref().unwrap_err()
        );
        assert!(result.unwrap(), "should return true (candidate consumed)");
        assert!(
            store_path.exists(),
            "fresh store should NOT be deleted when sizes match via HEAD"
        );
    }

    #[test]
    #[serial(global_state)]
    fn download_next_remote_shard_keeps_store_when_head_returns_error() {
        // When the manifest does NOT provide expected_bytes (hf-hub sibling
        // fallback) AND the HTTP HEAD request fails (network error), the
        // staleness check is skipped and the cached store is preserved as-is.
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        // Candidate pointing at an unreachable address — HEAD will Err.
        let candidate = format!("url::{TEST_UNREACHABLE_URL}/resolve/main/train/head-err.ndjson");
        let store_path = HuggingFaceRowSource::candidate_store_path(&config, &candidate);
        fs::create_dir_all(store_path.parent().unwrap()).unwrap();
        write_simdr_fixture(&store_path, &[("r0", "row")]);
        {
            let store = DataStore::open(&store_path).unwrap();
            store
                .write(HF_SHARD_STORE_SOURCE_SIZE_KEY, &100u64.to_le_bytes())
                .unwrap();
        }

        let original_content = fs::read(&store_path).unwrap();

        let cached_store = Arc::new(DataStore::open(&store_path).unwrap());
        source
            .config
            .store_cache
            .lock()
            .unwrap()
            .insert(store_path.clone(), cached_store);

        {
            let mut state = source.state.lock().unwrap();
            state.remote_candidates = Some(vec![candidate.clone()]);
            state.remote_candidate_order = vec![0];
            // No expected_bytes — simulates hf-hub fallback.
            state.next_remote_idx = 0;
        }

        // HEAD fails (Err) → effective_expected = None → stale check
        // skipped → store preserved as-is.
        let result = source.download_next_remote_shard();
        assert!(
            result.is_ok(),
            "expected Ok even when HEAD fails, got: {err:?}",
            err = result.as_ref().unwrap_err()
        );
        assert!(
            fs::read(&store_path).ok().as_deref() == Some(&original_content),
            "store should be preserved when HEAD fails"
        );
    }

    #[test]
    #[serial(global_state)]
    fn download_next_remote_shard_keeps_store_when_head_returns_none() {
        // When the manifest does NOT provide expected_bytes AND the HEAD
        // request returns Ok(None) (e.g. 500 status, or missing
        // Content-Length), the staleness check is skipped and the cached
        // store is preserved.
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        // Mock server returning 500 — HEAD will succeed but return
        // Ok(None) because the status is not 2xx.
        let server = TestHttpServer::new(500, b"Internal Server Error".to_vec());
        let base_url = server.url().to_string();

        let candidate = format!("url::{base_url}/resolve/main/train/head-none.ndjson");
        let store_path = HuggingFaceRowSource::candidate_store_path(&config, &candidate);
        fs::create_dir_all(store_path.parent().unwrap()).unwrap();
        write_simdr_fixture(&store_path, &[("r0", "row")]);
        {
            let store = DataStore::open(&store_path).unwrap();
            store
                .write(HF_SHARD_STORE_SOURCE_SIZE_KEY, &100u64.to_le_bytes())
                .unwrap();
        }

        let original_content = fs::read(&store_path).unwrap();

        let cached_store = Arc::new(DataStore::open(&store_path).unwrap());
        source
            .config
            .store_cache
            .lock()
            .unwrap()
            .insert(store_path.clone(), cached_store);

        {
            let mut state = source.state.lock().unwrap();
            state.remote_candidates = Some(vec![candidate.clone()]);
            state.remote_candidate_order = vec![0];
            // No expected_bytes — simulates hf-hub fallback.
            state.next_remote_idx = 0;
        }

        // HEAD returns 500 → fetch_remote_size_with_runtime returns
        // Ok(None) → effective_expected = None → stale check skipped.
        let result = source.download_next_remote_shard();
        assert!(
            result.is_ok(),
            "expected Ok even when HEAD returns None, got: {err:?}",
            err = result.as_ref().unwrap_err()
        );
        assert!(
            fs::read(&store_path).ok().as_deref() == Some(&original_content),
            "store should be preserved when HEAD returns None"
        );
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
    fn shard_candidate_seed_changes_with_sampler_seed() {
        // Verifies that different sampler_seed values (which in production
        // include the epoch_step XOR from IngestionManager) produce
        // different shard permutations, while the same seed is deterministic.
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());

        // Different sampler seeds → different shard candidate seeds.
        let seed_1 = HuggingFaceRowSource::shard_candidate_seed(&config, 100, 1);
        let seed_2 = HuggingFaceRowSource::shard_candidate_seed(&config, 100, 2);
        assert_ne!(
            seed_1, seed_2,
            "different seeds must produce different shard seeds"
        );

        // Same sampler seed → deterministic.
        let seed_1_again = HuggingFaceRowSource::shard_candidate_seed(&config, 100, 1);
        assert_eq!(seed_1, seed_1_again, "same seed must be deterministic");

        // Verify the permutation itself changes with seed.
        let candidates: Vec<String> = (0..10).map(|i| format!("shard-{i:02}")).collect();
        let order_1 = HuggingFaceRowSource::build_candidate_order(&config, &candidates, 1);
        let order_2 = HuggingFaceRowSource::build_candidate_order(&config, &candidates, 2);
        assert_ne!(
            order_1, order_2,
            "different seeds must produce different shard orders"
        );

        // Same seed produces same order.
        let order_1_again = HuggingFaceRowSource::build_candidate_order(&config, &candidates, 1);
        assert_eq!(order_1, order_1_again, "same seed must produce same order");
    }

    #[test]
    fn remote_shard_permutation_is_deterministic_by_sampler_seed() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let total = 8usize;

        let seed_a = HuggingFaceRowSource::shard_candidate_seed(&config, total, 7);
        let seed_b = HuggingFaceRowSource::shard_candidate_seed(&config, total, 7);
        let seed_c = HuggingFaceRowSource::shard_candidate_seed(&config, total, 10);

        let mut perm_a = triplets_core::source::IndexPermutation::new(total, seed_a, 0);
        let mut perm_b = triplets_core::source::IndexPermutation::new(total, seed_b, 0);
        let mut perm_c = triplets_core::source::IndexPermutation::new(total, seed_c, 0);

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

        let (shards, discovered) = HuggingFaceRowSource::build_shard_index(&config).unwrap();
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

        let (shards, discovered) = HuggingFaceRowSource::build_shard_index(&config).unwrap();
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
        assert_eq!(HuggingFaceRowSource::value_to_text(&json!(null)), None);
        assert_eq!(HuggingFaceRowSource::value_to_text(&json!("   ")), None);
        assert_eq!(
            HuggingFaceRowSource::value_to_text(&json!("hello")),
            Some("hello".into())
        );
        assert_eq!(
            HuggingFaceRowSource::value_to_text(&json!(true)),
            Some("true".into())
        );
        assert_eq!(
            HuggingFaceRowSource::value_to_text(&json!(3.5)),
            Some("3.5".into())
        );
        assert_eq!(
            HuggingFaceRowSource::value_to_text(&json!([1, 2])),
            Some("1".into())
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
                parquet_row_groups: Vec::new(),
                remote_candidate: None,
            },
            ShardIndex {
                path: PathBuf::from("b"),
                global_start: 20,
                row_count: 2,
                parquet_row_groups: Vec::new(),
                remote_candidate: None,
            },
        ];
        let hit = HuggingFaceRowSource::locate_shard(&shards, 11).unwrap();
        assert_eq!(hit.1, 1);

        let mut state = SourceState {
            materialized_rows: 0,
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
            // Set up remote candidates so headroom is applied
            state.remote_candidates = Some(vec![
                "url::http://a/0.parquet".to_string(),
                "url::http://a/1.parquet".to_string(),
            ]);
            state.next_remote_idx = 0;
        }
        // headroom = ingestion_max_records * multiplier = 10 * 3 = 30; since known (5)
        // < headroom, expansion = 30; upper = 5 + 30 = 35
        assert_eq!(source.len_hint(), Some(35));

        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 0;
            state.remote_candidates = Some(vec!["url::http://a/0.parquet".to_string()]);
            state.next_remote_idx = 0;
        }
        assert_eq!(source.len_hint(), Some(1));
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
            // Set up remote candidates so headroom is applied
            state.remote_candidates = Some(vec![
                "url::http://a/0.parquet".to_string(),
                "url::http://a/1.parquet".to_string(),
            ]);
            state.next_remote_idx = 0;
        }

        // headroom = cache_capacity * multiplier = 4 * 2 = 8; upper = 8 + 8 = 16
        assert_eq!(source.len_hint(), Some(16));
    }

    #[test]
    fn eligible_rows_extends_cached_index_when_new_shard_is_appended() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        let appended_simdr = dir.path().join("append.simdr");
        write_simdr_fixture(&appended_simdr, &[("r1", "hello")]);
        let appended =
            HuggingFaceRowSource::index_single_shard_for_test(&config, &appended_simdr, 1)
                .unwrap()
                .0
                .unwrap();

        let baseline = ShardIndex {
            path: dir.path().join("missing-baseline.simdr"),
            global_start: 0,
            row_count: 1,
            parquet_row_groups: Vec::new(),
            remote_candidate: None,
        };

        {
            let mut state = source.state.lock().unwrap();
            state.shards = vec![baseline.clone(), appended.clone()];
            state.materialized_rows = 2;
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
            shards: vec![
                ShardIndex {
                    path: evict_path.clone(),
                    global_start: 0,
                    row_count: 8,
                    parquet_row_groups: vec![(0, 8)],
                    remote_candidate: None,
                },
                ShardIndex {
                    path: keep_path.clone(),
                    global_start: 8,
                    row_count: 8,
                    parquet_row_groups: vec![(0, 8)],
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
            shards: vec![ShardIndex {
                path: protected.clone(),
                global_start: 0,
                row_count: 8,
                parquet_row_groups: vec![(0, 8)],
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
        let (shards, discovered) = HuggingFaceRowSource::build_shard_index(&config).unwrap();
        assert_eq!(discovered, 2);
        assert_eq!(shards.len(), 2);
    }

    #[test]
    fn index_single_shard_returns_none_for_empty_file() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let path = dir.path().join("empty.jsonl");
        fs::write(&path, b"").unwrap();
        let shard = HuggingFaceRowSource::index_single_shard_for_test(&config, &path, 0).unwrap();
        assert!(shard.0.is_none());
    }

    #[test]
    fn refresh_reads_local_rows_and_advances_cursor() {
        let dir = tempdir().unwrap();
        let simdr_path = dir.path().join("rows.simdr");
        write_simdr_fixture(
            &simdr_path,
            &[("r1", "alpha"), ("r2", "beta"), ("r3", "gamma")],
        );

        let mut config = test_config(dir.path().to_path_buf());
        config.refresh_batch_multiplier = 1;
        let source = test_source(config.clone());
        let shard = HuggingFaceRowSource::index_single_shard_for_test(&config, &simdr_path, 0)
            .unwrap()
            .0
            .unwrap();
        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = shard.row_count;
            state.shards = vec![shard];
        }

        let snapshot = source.refresh(None, Some(2)).unwrap();
        assert_eq!(snapshot.records.len(), 2);
        assert!(snapshot.cursor.revision > 0);
    }

    #[test]
    fn next_text_batch_produces_distinct_cursor_values_per_call() {
        // Proves successive next_text_batch calls produce different cursor
        // values — the cursor is NOT stuck at the initial value.
        let dir = tempdir().unwrap();

        // 3 shards so we get enough distinct cursor positions.
        let simdr_path = dir.path().join("rows.simdr");
        let rows: Vec<(String, String)> = (0..20)
            .map(|i| (format!("r{i}"), format!("text-{i}")))
            .collect();
        let row_refs: Vec<(&str, &str)> =
            rows.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
        write_simdr_fixture(&simdr_path, &row_refs);

        let mut config =
            HuggingFaceRowsConfig::new("cursor_test", "org/ds", "default", "train", dir.path());
        config.hf_token = None;
        config.cache_capacity = 10;
        config.remote_expansion_headroom_multiplier = 1;
        config.parquet_endpoint = TEST_UNREACHABLE_URL.to_string();

        let source = HuggingFaceRowSource::new(config).unwrap();
        let shard_idx =
            HuggingFaceRowSource::index_single_shard_for_test(&source.config, &simdr_path, 0)
                .unwrap()
                .0
                .unwrap();
        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = shard_idx.row_count;
            state.shards = vec![shard_idx];
        }

        // Pre-load a PersistedSamplerState with cursor=0.
        let split_state = PersistedSamplerState {
            source_cycle_idx: 0,
            source_record_cursors: vec![("cursor_test".to_string(), 0)],
            epoch: 0,
            epoch_step: 0,
            rng_state: 0,
            triplet_recipe_rr_idx: 0,
            text_recipe_rr_idx: 0,
            source_stream_cursors: vec![("cursor_test".to_string(), 0)],
        };
        let split_store =
            Arc::new(DeterministicSplitStore::new(SplitRatios::default(), 777).unwrap());
        split_store.save_sampler_state(&split_state, None).unwrap();

        let sampler = TripletSampler::new(
            SamplerConfig {
                seed: 1,
                ingestion_max_records: 1,
                batch_size: 1,
                ..SamplerConfig::default()
            },
            split_store,
        );
        sampler.register_source(Box::new(source.clone())).unwrap();

        // Collect cursor_revision values over several next_text_batch calls.
        // Multiple next_text_batch calls should succeed (each triggers a
        // refresh with a distinct seed, proving the step counter influences
        // the shard order without crashing).
        let mut count = 0;
        for _ in 0..20 {
            match sampler.next_text_batch(SplitLabel::Train) {
                Ok(batch) => count += batch.samples.len(),
                Err(_) => break,
            }
        }
        assert!(count > 0, "expected at least 1 text sample across calls");
    }

    #[test]
    fn next_batch_methods_rebuild_shard_order_with_step() {
        // Each batch method gets its own source+sampler with a unique
        // snapshot directory.  The first refresh increments epoch_step
        // 0→1, XORs into the seed (42^0^1=43), set_active_sampler_config
        // rebuilds the order differently from the initial seed=0.
        let shard_rows: Vec<(String, String)> = (0..10)
            .map(|i| (format!("r{i}"), "t".to_string()))
            .collect();

        // Helper: create source+sampler in a fresh tempdir with a shard file.
        let setup = || -> (HuggingFaceRowSource, TripletSampler<DeterministicSplitStore>, tempfile::TempDir) {
            let tmp = tempdir().unwrap();
            let simdr_path = tmp.path().join("shard.simdr");
            let row_refs: Vec<(&str, &str)> = shard_rows.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
            write_simdr_fixture(&simdr_path, &row_refs);
            let mut cfg = HuggingFaceRowsConfig::new("t", "o/d", "d", "train", tmp.path());
            cfg.hf_token = None;
            cfg.parquet_endpoint = TEST_UNREACHABLE_URL.to_string();
            let source = HuggingFaceRowSource::new(cfg).unwrap();
            let idx = HuggingFaceRowSource::index_single_shard_for_test(
                &source.config, &simdr_path, 0).unwrap().0.unwrap();
            {
                let mut st = source.state.lock().unwrap();
                st.materialized_rows = idx.row_count;
                st.shards = vec![idx];
                let cand: Vec<String> = (0..5).map(|i|
                    format!("url::http://h/d/resolve/main/train/p-{i:04}.ndjson")
                ).collect();
                st.remote_candidates = Some(cand.clone());
                st.remote_candidate_order =
                    HuggingFaceRowSource::build_candidate_order(&source.config, &cand, 0);
                st.next_remote_idx = 0;
            }
            let split = Arc::new(
                DeterministicSplitStore::new(SplitRatios { train: 1.0, validation: 0.0, test: 0.0 }, 777).unwrap());
            let sampler = TripletSampler::new(SamplerConfig {
                seed: 42, ingestion_max_records: 10, batch_size: 1,
                ..SamplerConfig::default()
            }, split);
            sampler.register_source(Box::new(source.clone())).unwrap();
            (source, sampler, tmp)
        };

        // next_text_batch
        let (source, sampler, _tmp) = setup();
        let before = source.state.lock().unwrap().remote_candidate_order.clone();
        sampler.next_text_batch(SplitLabel::Train).unwrap();
        assert_ne!(
            before,
            source.state.lock().unwrap().remote_candidate_order,
            "next_text_batch must change shard order"
        );

        // next_pair_batch
        let (source, sampler, _tmp) = setup();
        let before = source.state.lock().unwrap().remote_candidate_order.clone();
        let _ = sampler.next_pair_batch(SplitLabel::Train);
        assert_ne!(
            before,
            source.state.lock().unwrap().remote_candidate_order,
            "next_pair_batch must change shard order"
        );

        // next_triplet_batch
        let (source, sampler, _tmp) = setup();
        let before = source.state.lock().unwrap().remote_candidate_order.clone();
        let _ = sampler.next_triplet_batch(SplitLabel::Train);
        assert_ne!(
            before,
            source.state.lock().unwrap().remote_candidate_order,
            "next_triplet_batch must change shard order"
        );
    }

    #[test]
    fn reported_record_count_uses_len_hint_for_local_state() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 4;
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
    #[serial(global_state)]
    fn ensure_row_available_bootstraps_from_manifest_candidates() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let mut source = test_source(config);
        let (base_url, _, server) =
            spawn_manifest_and_shard_http(2, b"{\"text\":\"hello\"}\n".to_vec());

        // Reset to None so ensure_row_available triggers the manifest-fetch path.
        source.state.lock().unwrap().remote_candidates = None;
        source.config.parquet_endpoint = base_url.to_string();

        assert!(source.ensure_row_available(0).unwrap());

        server.join().unwrap();
    }

    #[test]
    #[serial(global_state)]
    fn ensure_row_available_skips_past_all_cached_candidates_on_restart() {
        // Verifies the restart scenario: when every candidate in the manifest is
        // already materialised on disk, next_remote_idx jumps to candidates.len()
        // and ensure_row_available returns Ok(false) without any download attempt.
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let mut source = test_source(config.clone());

        // Construct the candidate URL that the manifest will list.
        let shard_raw_url =
            format!("{TEST_UNREACHABLE_URL}/datasets/org/ds/resolve/main/train/a.ndjson");
        let shard_candidate = format!("{HF_REMOTE_URL_PREFIX}{shard_raw_url}");
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
                parquet_row_groups: vec![(0, 1)],
                remote_candidate: None,
            }];
            state.materialized_rows = 1;
            state.remote_candidates = None;
        }

        // Serve a manifest that lists the same (already-cached) shard.
        let manifest_body = serde_json::to_vec(&json!([
            {"type": "file", "path": shard_raw_url, "size": 100}
        ]))
        .unwrap();
        let server = spawn_one_shot_http(manifest_body);
        let base_url = server.url().to_string();

        // Row 1 is not yet materialised; this triggers the candidate-init path.
        // all candidates are already on disk → next_remote_idx = candidates.len() → Ok(false).
        source.config.parquet_endpoint = base_url;
        let result = source.ensure_row_available(1).unwrap();

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

        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());
        let shard = HuggingFaceRowSource::index_single_shard_for_test(&config, &path, 0)
            .unwrap()
            .0
            .unwrap();
        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 1;
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
        // .txt is now a recognized transient text format, so build_shard_index
        // should succeed (the file is treated as a transient shard).
        assert!(result.is_ok());
    }

    #[test]
    fn refresh_handles_empty_total_and_cursor_wrap() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 0;
        }
        let empty = source.refresh(None, Some(5)).unwrap();
        assert!(empty.records.is_empty());
        assert_eq!(empty.cursor.revision, 0);

        let simdr_path = dir.path().join("rows.simdr");
        write_simdr_fixture(&simdr_path, &[("a", "A"), ("b", "B")]);
        let cfg2 = config;
        let source2 = test_source(cfg2.clone());
        let shard = HuggingFaceRowSource::index_single_shard_for_test(&cfg2, &simdr_path, 0)
            .unwrap()
            .0
            .unwrap();
        {
            let mut state = source2.state.lock().unwrap();
            state.materialized_rows = 2;
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
    fn refresh_order_uses_sampler_seed_for_local_rows() {
        let dir = tempdir().unwrap();
        let simdr_path = dir.path().join("rows.simdr");
        let rows: Vec<(String, String)> = (0..12)
            .map(|idx| (format!("r{idx}"), format!("v{idx}")))
            .collect();
        let row_refs: Vec<(&str, &str)> =
            rows.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
        write_simdr_fixture(&simdr_path, &row_refs);

        let mut config = test_config(dir.path().to_path_buf());
        config.refresh_batch_multiplier = 1;

        let source_a = test_source(config.clone());
        let source_b = test_source(config.clone());
        let source_c = test_source(config.clone());
        let shard = HuggingFaceRowSource::index_single_shard_for_test(&config, &simdr_path, 0)
            .unwrap()
            .0
            .unwrap();

        for source in [&source_a, &source_b, &source_c] {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 12;
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
            seed: 10,
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
            // Candidates stored sorted/immutable; order derived from seed 7, cursor 0.
            let order = HuggingFaceRowSource::build_candidate_order(&config, &candidates, 7);
            state.remote_candidates = Some(candidates.clone());
            state.remote_candidate_order = order.clone();
            state.next_remote_idx = 3;
        }

        // Order is rebuilt every call; pointer advances to first uncached position.
        source.configure_sampler(&SamplerConfig {
            seed: 7,
            ..SamplerConfig::default()
        });
        {
            let state = source.state.lock().unwrap();
            let order = HuggingFaceRowSource::build_candidate_order(&config, &candidates, 7);
            assert_eq!(state.remote_candidate_order, order);
            assert_eq!(
                state.next_remote_idx, 0,
                "order rebuilt every call: pointer lands at first uncached (no shards on disk)"
            );
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
            // Order is now derived from seed 18 (cursor_revision still 0).
            let expected_order =
                HuggingFaceRowSource::build_candidate_order(&config, &candidates, 18);
            assert_eq!(state.remote_candidate_order, expected_order);
            // No shards are materialised on disk so the pointer lands at 0
            // (the first non-materialised position in the new order).
            assert_eq!(state.next_remote_idx, 0);
        }
    }

    #[test]
    fn set_active_sampler_config_rebuilds_order_every_call() {
        // Proves that set_active_sampler_config rebuilds the shard
        // permutation every time it's called with a different seed
        // (the seed changes every call due to epoch_step XOR in
        // IngestionManager).
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        let candidates: Vec<String> = (0..5)
            .map(|i| {
                format!("url::http://host/datasets/org/ds/resolve/main/train/part-{i:04}.ndjson")
            })
            .collect();

        // Prime with seed 1.
        source.configure_sampler(&SamplerConfig {
            seed: 1,
            ..SamplerConfig::default()
        });

        {
            let mut state = source.state.lock().unwrap();
            state.remote_candidates = Some(candidates.clone());
            state.remote_candidate_order = Vec::new();
            state.next_remote_idx = 0;
        }

        // Call with seed 1 — order is built.
        source.configure_sampler(&SamplerConfig {
            seed: 1,
            ..SamplerConfig::default()
        });
        let order_seed1: Vec<usize>;
        {
            let state = source.state.lock().unwrap();
            let expected = HuggingFaceRowSource::build_candidate_order(&config, &candidates, 1);
            assert_eq!(state.remote_candidate_order, expected, "seed=1 order");
            order_seed1 = state.remote_candidate_order.clone();
        }

        // Call with seed 2 — order changes.
        source.configure_sampler(&SamplerConfig {
            seed: 2,
            ..SamplerConfig::default()
        });
        {
            let state = source.state.lock().unwrap();
            let expected = HuggingFaceRowSource::build_candidate_order(&config, &candidates, 2);
            assert_eq!(state.remote_candidate_order, expected, "seed=2 order");
            assert_ne!(
                state.remote_candidate_order, order_seed1,
                "different seed must produce different order"
            );
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
                    parquet_row_groups: vec![(0, 100)],
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

    // ── value_to_text ──────────────────────────────────────────────────────────

    #[test]
    #[serial(global_state)]
    fn parquet_manifest_fetched_exactly_once_per_candidate_list_population() {
        // Verify that the /parquet manifest endpoint is contacted only once per
        // source lifetime.  After the first ensure_row_available() populates
        // state.remote_candidates, subsequent calls must not re-fetch the manifest.
        // The counting server stays alive so a spurious second request would be
        // recorded and the final assertion would catch it.
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let mut source = test_source(config);
        // Reset to None so the first ensure_row_available() triggers the lazy fetch.
        source.state.lock().unwrap().remote_candidates = None;

        let shard_payload = b"{\"text\":\"hello\"}\n".to_vec();
        // Counting manifest+shard server: 4 slots so a second /parquet hit is caught.
        let (base_url, manifest_counter, _manifest_handle) =
            spawn_manifest_and_shard_http(4, shard_payload);

        // First call: remote_candidates is None → fetches manifest (counter→1) → downloads shard.
        source.config.parquet_endpoint = base_url.to_string();
        let first_available = source.ensure_row_available(0).unwrap();
        assert!(first_available);
        assert_eq!(
            manifest_counter.load(AtomicOrdering::SeqCst),
            1,
            "parquet manifest must be fetched exactly once on first ensure_row_available"
        );

        // Second call: remote_candidates is now Some(...) → must NOT re-fetch manifest.
        let _ = source.ensure_row_available(0);
        assert_eq!(
            manifest_counter.load(AtomicOrdering::SeqCst),
            1,
            "parquet manifest must not be re-fetched on subsequent ensure_row_available calls"
        );
    }

    // --- datasets viewer disabled (501) scenario ---
    //
    // Some HF datasets have the datasets viewer disabled.  In that case the
    // /size, /info, and /parquet datasets-server endpoints all return HTTP 501
    // with {"error":"Not supported: dataset viewer is disabled ..."}.
    //
    // The expected behaviour:
    //   * /size   → fetch_global_row_count returns Ok(None), not Err.
    //   * /info   → fetch_classlabel_maps returns an empty map, not an error.
    //   * /parquet → list_remote_candidates_from_parquet_manifest returns Err,
    //                which causes list_remote_candidates_with_runtime to fall
    //                through to the hf-hub repository listing path.

    #[test]
    #[serial(global_state)]
    fn ensure_row_available_does_not_loop_on_eviction() {
        // Regression: ensure the fetched_candidates guard prevents infinite
        // manifest re-fetching when eviction nulls remote_candidates mid-execution.
        //
        // On Windows, fs::remove_file fails for files with active handles (the
        // DataStore keeps .simdr open). We backdate existing.stub's mtime so the
        // cache manager evicts it FIRST (no open handle → deletable on all
        // platforms). After deletion, sync_shard_state_from_disk_locked detects
        // the missing shard and nulls remote_candidates, triggering the guard.
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        // Tight cap: existing shard fills it, so every new download triggers eviction.
        config.local_disk_cap_bytes = Some(10);
        let mut source = test_source(config.clone());

        // Create an existing shard on disk that fills the entire cap.
        let manifest_root = source.manifest_cache_root();
        fs::create_dir_all(&manifest_root).unwrap();
        let existing_path = manifest_root.join("existing.stub");
        fs::write(&existing_path, vec![1u8; 10]).unwrap();

        // Backdate existing.stub so it's always the LRU eviction target.
        let yesterday = SystemTime::now() - Duration::from_secs(86400);
        filetime::set_file_mtime(
            &existing_path,
            filetime::FileTime::from_system_time(yesterday),
        )
        .unwrap();

        {
            let mut state = source.state.lock().unwrap();
            state.shards = vec![ShardIndex {
                path: existing_path,
                global_start: 0,
                row_count: 1,
                parquet_row_groups: vec![(0, 1)],
                remote_candidate: None,
            }];
            state.materialized_rows = 1;
            // None triggers the candidate-fetch path on first ensure_row_available call.
            state.remote_candidates = None;
        }

        // Use a single multi-accept mock server that serves both the
        // parquet manifest (/parquet) and shard payloads (everything else).
        // This avoids the flakiness of separate one-shot servers where the
        // manifest re-fetch could point back to an already-consumed server
        // (the eviction order is non-deterministic across platforms).
        let shard_payload = b"{\"text\":\"new\"}\n".to_vec();
        let (base_url, manifest_counter, server) =
            spawn_manifest_and_shard_http(2, shard_payload.clone());

        // Call ensure_row_available with idx == materialized_rows so the
        // first download does not satisfy idx < materialized_rows.
        // Append /parquet so spawn_manifest_and_shard_http can route the
        // manifest re-fetch to the manifest body (see first_line.contains("/parquet")).
        source.config.parquet_endpoint = base_url.to_string();

        // ensure_row_available(1) must:
        //   1. Fetch manifest (remote_candidates = None)
        //   2. Download shard (materialized_rows 1→2)
        //   3. Eviction deletes existing.stub → remote_candidates = None
        //   4. fetched_candidates guard fires → returns Ok(true)
        //   5. manifest_counter == 1 (no re-fetch)
        assert!(
            source.ensure_row_available(1).unwrap(),
            "ensure_row_available must return Ok(true)"
        );

        assert_eq!(
            manifest_counter.load(AtomicOrdering::SeqCst),
            1,
            "manifest must be fetched exactly once — fetched_candidates guard must prevent re-fetch"
        );
        server.join().unwrap();
    }

    #[test]
    #[serial(global_state)]
    fn list_remote_candidates_from_parquet_manifest_errors_on_501() {
        // A 501 from /parquet causes the manifest path to return Err, which
        // propagates to the caller (no fallback path).
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        let body =
            br#"{"error":"Not supported: dataset viewer is disabled in org/dataset configuration."}"#
                .to_vec();
        let server = TestHttpServer::new(501, body);
        let base_url = server.url().to_string();

        config.parquet_endpoint = base_url;
        let client = test_http_client();
        let result =
            HuggingFaceRowSource::list_remote_candidates_from_parquet_manifest(&client, &config);

        assert!(
            result.is_err(),
            "expected Err from 501 /parquet response, got {result:?}"
        );
    }

    #[test]
    fn hf_source_entry_partial_eq_compares_all_fields() {
        let base = HfSourceEntry {
            uri: "hf://org/ds/default/train".to_string(),
            anchor_columns: vec!["title".to_string()],
            positive_columns: vec!["body".to_string()],
            negative_columns: Vec::new(),
            context_columns: vec!["meta".to_string()],
            text_columns: Vec::new(),
            trust: Some(0.8),
            weight: None,
            source_id: None,
        };
        let same = HfSourceEntry { ..base.clone() };
        assert_eq!(base, same);
        let diff_uri = HfSourceEntry {
            uri: "hf://other".to_string(),
            ..base.clone()
        };
        assert_ne!(base, diff_uri);
        let diff_trust = HfSourceEntry {
            trust: Some(0.5),
            ..base.clone()
        };
        assert_ne!(base, diff_trust);
        let diff_sid = HfSourceEntry {
            source_id: Some("my-id".to_string()),
            ..base.clone()
        };
        assert_ne!(base, diff_sid);
        let no_trust = HfSourceEntry {
            trust: None,
            ..base.clone()
        };
        assert_ne!(base, no_trust);
    }

    #[test]
    fn open_shard_store_creates_parent_directories() {
        let dir = tempdir().unwrap();
        let nested = dir.path().join("a").join("b").join("c.simdr");
        assert!(!nested.parent().unwrap().exists());
        let config = test_config(dir.path().to_path_buf());
        let store = HuggingFaceRowSource::open_shard_store(&config, &nested).unwrap();
        assert!(nested.parent().unwrap().exists());
        drop(store);
    }

    #[test]
    fn open_shard_store_errors_when_base_path_is_a_file() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let file_path = dir.path().join("not-a-dir");
        fs::write(&file_path, b"not-a-dir").unwrap();
        let bad_path = file_path.join("store.simdr");
        let result = HuggingFaceRowSource::open_shard_store(&config, &bad_path);
        assert!(result.is_err());
    }

    #[test]
    fn remove_stale_store_evicts_from_cache_and_removes_file() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let store_path = dir.path().join("stale.simdr");
        fs::write(&store_path, b"stale-content").unwrap();
        config.store_cache.lock().unwrap().insert(
            store_path.clone(),
            Arc::new(DataStore::open(&store_path).unwrap()),
        );
        HuggingFaceRowSource::remove_stale_store(&config, &store_path);
        assert!(!store_path.exists(), "stale store file must be removed");
        assert!(!config.store_cache.lock().unwrap().contains_key(&store_path));
    }

    #[test]
    fn remove_stale_store_does_not_panic_when_file_missing() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let missing = dir.path().join("never-existed.simdr");
        HuggingFaceRowSource::remove_stale_store(&config, &missing);
        assert!(!missing.exists());
    }

    #[test]
    fn invalidate_eligible_index_resets_cache() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        {
            let mut cache = source.eligible_index.lock().unwrap();
            *cache = EligibleIndexCache {
                signature: Some(42),
                rows: Some(Arc::new(vec![0, 1, 2])),
                shards: vec![ShardIndex {
                    path: PathBuf::from("dummy.parquet"),
                    global_start: 0,
                    row_count: 3,
                    parquet_row_groups: vec![(0, 3)],
                    remote_candidate: None,
                }],
            };
        }
        source.invalidate_eligible_index();
        let cache = source.eligible_index.lock().unwrap();
        assert!(cache.signature.is_none());
        assert!(cache.rows.is_none());
        assert!(cache.shards.is_empty());
    }

    #[test]
    fn write_store_row_count_and_read_store_row_count_roundtrip() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let store_path = dir.path().join("roundtrip.simdr");
        let store = DataStore::open(&store_path).unwrap();
        source.write_store_row_count(&store, 42).unwrap();
        assert_eq!(source.read_store_row_count(&store).unwrap(), 42);
        source.write_store_row_count(&store, 99).unwrap();
        assert_eq!(source.read_store_row_count(&store).unwrap(), 99);
    }

    #[test]
    fn read_store_row_count_errors_on_payload_size_mismatch() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let store_path = dir.path().join("bad-meta.simdr");
        let store = DataStore::open(&store_path).unwrap();
        store.write(HF_SHARD_STORE_META_ROWS_KEY, b"abc").unwrap();
        match source.read_store_row_count(&store) {
            Err(SamplerError::SourceUnavailable { reason, .. }) => {
                assert!(reason.contains("payload size"));
            }
            other => panic!("expected SourceUnavailable error, got {other:?}"),
        }
    }

    #[test]
    // FIXME: Windows tests timeout here, theoretically due to the following reason:
    //
    // Testing live thread spawning combined with a deliberate fallback to an
    // unreachable dead port (127.0.0.1:1) binds your test suite's determinism
    // directly to OS-level TCP/IP implementation details. While Unix environments
    // typically reject connections to unbound low ports instantaneously (ECONNREFUSED),
    // the Windows Winsock layer behaves non-deterministically under parallel test
    // execution profiles, frequently caching socket state or delaying connection drops
    // to match synthetic connect timeouts.
    #[cfg_attr(windows, ignore)]
    #[serial(global_state)]
    fn trigger_expansion_if_needed_starts_background_thread() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let mut source = test_source(config);

        // Override with ultra-short timeouts to force immediate connection failure on Windows
        let inner_client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_millis(100))
            .timeout(std::time::Duration::from_millis(200))
            .build()
            .expect("failed to build ultra-short timeout client");
        source.http_client = reqwest_drive::ClientBuilder::new(inner_client).build();

        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 5;
            state.remote_candidates = Some(vec![
                "url::http://127.0.0.1:1/ds/resolve/main/train/000.ndjson".to_string(),
            ]);
            state.next_remote_idx = 0;
            state.remote_candidate_order = vec![0];
        }
        assert!(source.expansion_thread.lock().unwrap().is_none());
        source.trigger_expansion_if_needed();
        let handle = source.expansion_thread.lock().unwrap().take();
        assert!(handle.is_some());
        if let Some(h) = handle {
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let _ = tx.send(h.join());
            });
            let _ = rx
                .recv_timeout(std::time::Duration::from_secs(5))
                .expect("expansion thread hung or deadlocked: Timeout");
        }
    }

    #[test]
    fn trigger_expansion_if_needed_skips_when_all_remote_candidates_consumed() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 100;
            state.remote_candidates = Some(vec!["done".to_string()]);
            state.next_remote_idx = 1;
        }
        source.trigger_expansion_if_needed();
        assert!(source.expansion_thread.lock().unwrap().is_none());
    }

    #[test]
    fn trigger_expansion_if_needed_skips_when_total_rows_is_zero() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 0;
        }
        source.trigger_expansion_if_needed();
        assert!(source.expansion_thread.lock().unwrap().is_none());
    }

    #[test]
    fn trigger_expansion_if_needed_skips_when_already_running() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);

        // Inject a dummy thread that blocks until explicitly released.
        // No network I/O, no sleep, no global mutex contention.
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let dummy = std::thread::spawn(move || {
            let _ = rx.recv();
        });
        *source.expansion_thread.lock().unwrap() = Some(dummy);

        // Must skip: slot is already occupied.
        source.trigger_expansion_if_needed();
        assert!(source.expansion_thread.lock().unwrap().as_ref().is_some());

        // Release: signal the dummy to exit, join cleanly.
        drop(tx);
        let handle = source.expansion_thread.lock().unwrap().take().unwrap();
        handle.join().unwrap();
    }

    #[test]
    fn ensure_cache_group_reports_error() {
        let bad_group = PathBuf::from("bad\0group");
        let result = ensure_cache_group(bad_group);
        assert!(result.is_err());
    }

    // ── New tests for uncovered functions ────────────────────────────────

    #[test]
    fn new_rejects_missing_explicit_mapping() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.anchor_columns.clear();
        config.positive_columns.clear();
        config.context_columns.clear();
        config.text_columns.clear();
        let result = HuggingFaceRowSource::new(config);
        assert!(result.is_err());
        let err = result.map(|_| ()).unwrap_err();
        assert!(format!("{err:?}").contains("explicit field mapping"));
    }

    #[test]
    fn id_returns_source_id() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        assert_eq!(source.id(), "hf_test");
    }

    #[test]
    fn is_store_shard_path_detects_simdr_extension() {
        assert!(HuggingFaceRowSource::is_store_shard_path(Path::new(
            "shard.simdr"
        )));
        assert!(HuggingFaceRowSource::is_store_shard_path(Path::new(
            "shard.SIMDR"
        )));
        assert!(HuggingFaceRowSource::is_store_shard_path(Path::new(
            "shard.SimDr"
        )));
        assert!(!HuggingFaceRowSource::is_store_shard_path(Path::new(
            "shard.parquet"
        )));
        assert!(!HuggingFaceRowSource::is_store_shard_path(Path::new(
            "shard.ndjson"
        )));
        assert!(!HuggingFaceRowSource::is_store_shard_path(Path::new(
            "no-extension"
        )));
        assert!(!HuggingFaceRowSource::is_store_shard_path(Path::new(
            ".hidden"
        )));
    }

    #[test]
    fn resolve_inner_extension_handles_compound_extensions() {
        // Compound extensions
        assert_eq!(
            resolve_inner_extension(Path::new("file.jsonl.gz")),
            Some("jsonl".to_string())
        );
        assert_eq!(
            resolve_inner_extension(Path::new("file.ndjson.gz")),
            Some("ndjson".to_string())
        );
        assert_eq!(
            resolve_inner_extension(Path::new("file.txt.gz")),
            Some("txt".to_string())
        );

        // Simple extensions
        assert_eq!(
            resolve_inner_extension(Path::new("file.parquet")),
            Some("parquet".to_string())
        );
        assert_eq!(
            resolve_inner_extension(Path::new("file.simdr")),
            Some("simdr".to_string())
        );
        assert_eq!(
            resolve_inner_extension(Path::new("file.jsonl")),
            Some("jsonl".to_string())
        );
        assert_eq!(
            resolve_inner_extension(Path::new("file.ndjson")),
            Some("ndjson".to_string())
        );

        // No inner format (just .gz)
        assert_eq!(resolve_inner_extension(Path::new("file.gz")), None);

        // No extension
        assert_eq!(resolve_inner_extension(Path::new("no-extension")), None);

        // Case insensitive
        assert_eq!(
            resolve_inner_extension(Path::new("file.JSONL.GZ")),
            Some("jsonl".to_string())
        );
        assert_eq!(
            resolve_inner_extension(Path::new("file.PARQUET")),
            Some("parquet".to_string())
        );

        // Boundary cases
        assert_eq!(
            resolve_inner_extension(Path::new("file.tar.gz")),
            Some("tar".to_string())
        );
        assert_eq!(
            resolve_inner_extension(Path::new(".hidden.jsonl.gz")),
            Some("jsonl".to_string())
        );
        assert_eq!(
            resolve_inner_extension(Path::new("file.gz.bak")),
            Some("bak".to_string())
        );
        assert_eq!(resolve_inner_extension(Path::new("")), None);
    }

    #[test]
    fn is_gzip_path_detects_gz_extension() {
        assert!(is_gzip_path(Path::new("file.jsonl.gz")));
        assert!(is_gzip_path(Path::new("file.GZ")));
        assert!(is_gzip_path(Path::new("file.Gz")));
        assert!(is_gzip_path(Path::new("file.tar.gz")));
        assert!(!is_gzip_path(Path::new("file.parquet")));
        assert!(!is_gzip_path(Path::new("file.jsonl")));
        assert!(!is_gzip_path(Path::new("file.simdr")));
        assert!(!is_gzip_path(Path::new("no-extension")));
        assert!(!is_gzip_path(Path::new("file.gz.bak")));
        assert!(!is_gzip_path(Path::new("")));
    }

    #[test]
    fn shard_store_path_for_appends_simdr_extension() {
        let path = PathBuf::from("cache/shard.parquet");
        let mapped = HuggingFaceRowSource::shard_store_path_for(&path);
        assert_eq!(mapped, PathBuf::from("cache/shard.simdr"));
        let no_ext = PathBuf::from("cache/shard");
        let mapped2 = HuggingFaceRowSource::shard_store_path_for(&no_ext);
        assert_eq!(mapped2, PathBuf::from("cache/shard.simdr"));
    }

    #[test]
    fn candidate_store_path_maps_via_target_path() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let candidate = "url::https://host/ds/resolve/main/train/data-000.parquet";
        let target = HuggingFaceRowSource::candidate_target_path(&config, candidate);
        let store = HuggingFaceRowSource::candidate_store_path(&config, candidate);
        assert_eq!(store, target.with_extension("simdr"));
    }

    #[test]
    fn manifest_cache_root_joins_manifest_dir() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let root = source.manifest_cache_root();
        assert!(root.ends_with(HF_PARQUET_MANIFEST_DIR));
    }

    #[test]
    fn recompute_shard_offsets_sums_row_counts() {
        let mut state = SourceState {
            materialized_rows: 0,
            shards: vec![
                ShardIndex {
                    path: PathBuf::from("a.simdr"),
                    global_start: 0,
                    row_count: 10,
                    parquet_row_groups: vec![(0, 10)],
                    remote_candidate: None,
                },
                ShardIndex {
                    path: PathBuf::from("b.simdr"),
                    global_start: 0,
                    row_count: 20,
                    parquet_row_groups: vec![(0, 20)],
                    remote_candidate: None,
                },
            ],
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            remote_candidate_order: Vec::new(),
        };
        HuggingFaceRowSource::recompute_shard_offsets(&mut state);
        assert_eq!(state.materialized_rows, 30);
        assert_eq!(state.shards[0].global_start, 0);
        assert_eq!(state.shards[1].global_start, 10);
    }

    #[test]
    fn sync_shard_state_from_disk_removes_missing_shards() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let existing = dir.path().join("existing.simdr");
        fs::write(&existing, b"data").unwrap();
        let missing = dir.path().join("missing.simdr");
        let mut state = SourceState {
            materialized_rows: 100,
            shards: vec![
                ShardIndex {
                    path: existing.clone(),
                    global_start: 0,
                    row_count: 50,
                    parquet_row_groups: vec![(0, 50)],
                    remote_candidate: None,
                },
                ShardIndex {
                    path: missing.clone(),
                    global_start: 50,
                    row_count: 50,
                    parquet_row_groups: vec![(0, 50)],
                    remote_candidate: None,
                },
            ],
            remote_candidates: Some(vec!["candidate".to_string()]),
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            remote_candidate_order: vec![0],
        };
        source.sync_shard_state_from_disk_locked(&mut state);
        assert_eq!(state.shards.len(), 1);
        assert_eq!(state.shards[0].path, existing);
        assert_eq!(state.materialized_rows, 50);
        assert!(state.remote_candidates.is_none());
    }

    #[test]
    fn sync_shard_state_from_disk_preserves_candidates_when_all_present() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let sp = dir.path().join("shard.simdr");
        fs::write(&sp, b"data").unwrap();
        let mut state = SourceState {
            materialized_rows: 50,
            shards: vec![ShardIndex {
                path: sp.clone(),
                global_start: 0,
                row_count: 50,
                parquet_row_groups: vec![(0, 50)],
                remote_candidate: None,
            }],
            remote_candidates: Some(vec!["next".to_string()]),
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 1,
            remote_candidate_order: vec![0],
        };
        source.sync_shard_state_from_disk_locked(&mut state);
        assert_eq!(state.shards.len(), 1);
        assert_eq!(state.remote_candidates, Some(vec!["next".to_string()]));
        assert_eq!(state.next_remote_idx, 1);
    }

    #[test]
    #[serial(global_state)]
    fn build_hf_sources_collapses_uri_parse_error() {
        let roots = HfListRoots {
            source_list: "inline".to_string(),
            sources: vec![HfSourceEntry {
                uri: "hf://incomplete".to_string(),
                anchor_columns: vec!["title".to_string()],
                positive_columns: Vec::new(),
                negative_columns: Vec::new(),
                context_columns: Vec::new(),
                text_columns: Vec::new(),
                trust: None,
                weight: None,
                source_id: None,
            }],
        };
        let temp_root = tempdir().unwrap();
        let nl = platform_newline();
        fs::write(
            temp_root.path().join("Cargo.toml"),
            format!("[package]{nl}name='tmp'{nl}version='0.0.0'{nl}"),
        )
        .unwrap();
        with_current_dir(temp_root.path(), || {
            with_env_vars(&[(crate::constants::ENV_TRIPLETS_HF_TOKEN, "")], || {
                let built = build_hf_sources(&roots);
                assert_eq!(built.len(), 0);
            });
        });
    }

    #[test]
    #[serial(global_state)]
    fn managed_hf_snapshot_dir_resolves_without_replica() {
        let dir = tempdir().unwrap();
        let nl = platform_newline();
        fs::write(
            dir.path().join("Cargo.toml"),
            format!("[package]{nl}name='tmp'{nl}version='0.0.0'{nl}"),
        )
        .unwrap();
        with_current_dir(dir.path(), || {
            let r = managed_hf_snapshot_dir("org/dataset", "default", "train");
            assert!(r.is_ok());
            assert!(r.unwrap().ends_with("train"));
        });
    }

    #[test]
    #[serial(global_state)]
    fn managed_hf_snapshot_dir_uses_all_splits_for_empty() {
        let dir = tempdir().unwrap();
        let nl = platform_newline();
        fs::write(
            dir.path().join("Cargo.toml"),
            format!("[package]{nl}name='tmp'{nl}version='0.0.0'{nl}"),
        )
        .unwrap();
        with_current_dir(dir.path(), || {
            let r = managed_hf_snapshot_dir("org/dataset", "default", "");
            assert!(r.is_ok());
            let path = r.unwrap();
            assert!(
                path.to_string_lossy().contains(HF_ALL_SPLITS_DIR),
                "expected path to contain '{}', got: {}",
                HF_ALL_SPLITS_DIR,
                path.display()
            );
        });
    }

    #[test]
    #[serial(global_state)]
    fn managed_hf_list_snapshot_dir_uses_replica_suffix() {
        let dir = tempdir().unwrap();
        let nl = platform_newline();
        fs::write(
            dir.path().join("Cargo.toml"),
            format!("[package]{nl}name='tmp'{nl}version='0.0.0'{nl}"),
        )
        .unwrap();
        with_current_dir(dir.path(), || {
            let r = managed_hf_list_snapshot_dir("org/dataset", "default", "train", 0);
            assert!(r.is_ok());
            assert!(r.unwrap().ends_with("replica_0"));
        });
    }

    #[test]
    fn remote_url_for_candidate_builds_bare_urls() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let r1 =
            HuggingFaceRowSource::remote_url_for_candidate(&config, "url::https://server/parquet");
        assert_eq!(r1, "https://server/parquet");
        let r2 = HuggingFaceRowSource::remote_url_for_candidate(&config, "data/train-000.parquet");
        assert!(r2.contains("/resolve/main/"));
    }

    #[test]
    fn row_store_row_key_uses_expected_format() {
        let key = HuggingFaceRowSource::row_store_row_key(0);
        assert!(key.starts_with(HF_SHARD_STORE_ROW_PREFIX));
        assert_eq!(key.len(), HF_SHARD_STORE_ROW_PREFIX.len() + 8);
        let key_42 = HuggingFaceRowSource::row_store_row_key(42);
        assert!(key_42.starts_with(HF_SHARD_STORE_ROW_PREFIX));
    }

    #[test]
    fn effective_expansion_headroom_rows_uses_config_multiplier() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.cache_capacity = 100;
        config.remote_expansion_headroom_multiplier = 3;
        let source = test_source(config);
        assert_eq!(source.effective_expansion_headroom_rows(), 300);

        source.configure_sampler(&SamplerConfig {
            ingestion_max_records: 50,
            ..SamplerConfig::default()
        });
        assert_eq!(source.effective_expansion_headroom_rows(), 150);
    }

    #[test]
    fn open_store_via_cache_inserts_and_reuses() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let path = dir.path().join("store.simdr");
        let store = DataStore::open(&path).unwrap();
        drop(store);
        let first = HuggingFaceRowSource::open_store_via_cache(&config, &path).unwrap();
        let second = HuggingFaceRowSource::open_store_via_cache(&config, &path).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn reported_record_count_uses_len_hint() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 10;
            state.remote_candidates = None;
        }
        assert_eq!(source.reported_record_count().unwrap(), 10);
    }

    #[test]
    fn format_shard_label_includes_totals() {
        let label = HuggingFaceRowSource::format_shard_label("data/train-000.parquet", 0, 5);
        assert!(label.contains("1/5"));
        assert!(label.contains("train-000.parquet"));
    }

    #[test]
    fn effective_refresh_batch_target_uses_multiplier() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        assert!(source.effective_refresh_batch_target(100) >= 2);
    }

    #[test]
    fn remote_shard_permutation_is_deterministic() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let c = ["a", "b", "c", "d", "e"];
        let c1: Vec<String> = c.iter().map(|s| s.to_string()).collect();
        let c2: Vec<String> = c.iter().map(|s| s.to_string()).collect();
        let o1 = HuggingFaceRowSource::build_candidate_order(&config, &c1, 42);
        let o2 = HuggingFaceRowSource::build_candidate_order(&config, &c2, 42);
        assert_eq!(o1, o2);
        let o3 = HuggingFaceRowSource::build_candidate_order(&config, &c1, 99);
        assert_ne!(o1, o3);
    }

    // ── Dict dataset / nested dict / list expansion tests ──────────────────

    #[test]
    fn parse_hf_source_line_negative_key() {
        let entry = parse_hf_source_line(
            "hf://embedding-data/QQP_triplets anchor=query positive=pos negative=neg",
        )
        .unwrap();
        assert_eq!(entry.anchor_columns, vec!["query"]);
        assert_eq!(entry.positive_columns, vec!["pos"]);
        assert_eq!(entry.negative_columns, vec!["neg"]);
        assert!(entry.context_columns.is_empty());
    }

    #[test]
    fn resolve_json_path_top_level() {
        let row = json!({"query": "hello", "pos": ["p"], "neg": ["n"]});
        let obj = row.as_object().unwrap();
        assert_eq!(
            HuggingFaceRowSource::resolve_json_path(obj, "query"),
            Some(json!("hello"))
        );
    }

    #[test]
    fn resolve_json_path_nested_dict() {
        let row = json!({"set": {"query": "hello", "pos": ["p"], "neg": ["n"]}});
        let obj = row.as_object().unwrap();
        assert_eq!(
            HuggingFaceRowSource::resolve_json_path(obj, "set.query"),
            Some(json!("hello"))
        );
        assert_eq!(
            HuggingFaceRowSource::resolve_json_path(obj, "set.pos"),
            Some(json!(["p"]))
        );
    }

    #[test]
    fn resolve_json_path_missing_returns_none() {
        let row = json!({"set": {"query": "hello"}});
        let obj = row.as_object().unwrap();
        assert_eq!(
            HuggingFaceRowSource::resolve_json_path(obj, "missing"),
            None
        );
    }

    #[test]
    fn value_to_text_array_first_element() {
        assert_eq!(
            HuggingFaceRowSource::value_to_text(&json!(["single"])),
            Some("single".into())
        );
        assert_eq!(
            HuggingFaceRowSource::value_to_text(&json!(["a", "b", "c"])),
            Some("a".into())
        );
        assert_eq!(HuggingFaceRowSource::value_to_text(&json!([])), None);
        assert_eq!(
            HuggingFaceRowSource::value_to_text(&json!([null, "valid"])),
            Some("valid".into())
        );
    }

    #[test]
    fn parse_row_dict_dataset() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.anchor_columns = vec!["set.query".into()];
        config.positive_columns = vec!["set.pos".into()];
        config.negative_columns = vec!["set.neg".into()];
        config.id_column = None;
        let source = test_source(config);

        let row = source
            .parse_row(
                0,
                &json!({
                    "set": {
                        "query": "What is Rust?",
                        "pos": ["How does Rust work?"],
                        "neg": ["Is Python better?", "What is Java?", "Tell me a joke"]
                    }
                }),
            )
            .unwrap()
            .unwrap();

        // anchor = set.query, positive = first set.pos element, negatives = expanded set.neg elements
        assert_eq!(row.text_fields.len(), 5);
        assert_eq!(row.text_fields[0].name, "set.query");
        assert_eq!(row.text_fields[0].text, "What is Rust?");
        assert_eq!(row.text_fields[1].name, "set.pos");
        assert_eq!(row.text_fields[1].text, "How does Rust work?");
        assert_eq!(row.text_fields[2].name, "set.neg[0]");
        assert_eq!(row.text_fields[2].text, "Is Python better?");
        assert_eq!(row.text_fields[3].name, "set.neg[1]");
        assert_eq!(row.text_fields[3].text, "What is Java?");
        assert_eq!(row.text_fields[4].name, "set.neg[2]");
        assert_eq!(row.text_fields[4].text, "Tell me a joke");
    }

    #[test]
    fn parse_row_dict_flat_columns_unaffected() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.anchor_columns = vec!["anchor".into()];
        config.positive_columns = vec!["positive".into()];
        config.id_column = Some("id".into());
        let source = test_source(config);

        // Flat (non-dict) row — should work exactly as before.
        let row = source
            .parse_row(0, &json!({"id": "r1", "anchor": "a", "positive": "p"}))
            .unwrap()
            .unwrap();
        assert_eq!(row.text_fields.len(), 2);
        assert_eq!(row.text_fields[0].text, "a");
        assert_eq!(row.text_fields[1].text, "p");
    }

    #[test]
    fn row_to_record_dict_dataset_sections() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.anchor_columns = vec!["query".into()];
        config.positive_columns = vec!["pos".into()];
        config.negative_columns = vec!["neg".into()];
        let source = test_source(config);

        let row = RowView {
            row_id: Some("dict:0".into()),
            timestamp: None,
            text_fields: vec![
                RowTextField {
                    name: "query".into(),
                    text: "anchor text".into(),
                },
                RowTextField {
                    name: "pos".into(),
                    text: "positive text".into(),
                },
                RowTextField {
                    name: "neg[0]".into(),
                    text: "negative one".into(),
                },
                RowTextField {
                    name: "neg[1]".into(),
                    text: "negative two".into(),
                },
            ],
        };

        let record = source.row_to_record(&row, 0).unwrap().unwrap();
        // First field → Anchor, second → Context (positive), rest → Context (negatives)
        assert_eq!(record.sections.len(), 4);
        assert_eq!(record.sections[0].role, SectionRole::Anchor);
        assert_eq!(record.sections[0].text, "anchor text");
        assert_eq!(record.sections[1].role, SectionRole::Context);
        assert_eq!(record.sections[1].text, "positive text");
        assert_eq!(record.sections[2].role, SectionRole::Context);
        assert_eq!(record.sections[2].text, "negative one");
        assert_eq!(record.sections[3].role, SectionRole::Context);
        assert_eq!(record.sections[3].text, "negative two");
    }

    #[test]
    fn parse_hf_source_line_weight_key() {
        let entry = parse_hf_source_line("hf://org/ds anchor=t positive=p weight=0.7").unwrap();
        assert_eq!(entry.weight, Some(0.7));
    }

    #[test]
    fn parse_hf_source_line_weight_zero_rejected() {
        let err = parse_hf_source_line("hf://org/ds anchor=t positive=p weight=0").unwrap_err();
        assert!(err.contains("must be > 0.0"));
    }

    #[test]
    fn parse_hf_source_line_weight_negative_rejected() {
        let err = parse_hf_source_line("hf://org/ds anchor=t positive=p weight=-1.0").unwrap_err();
        assert!(err.contains("must be > 0.0"));
    }

    #[test]
    fn parse_hf_source_line_weight_invalid_rejected() {
        let err = parse_hf_source_line("hf://org/ds anchor=t positive=p weight=abc").unwrap_err();
        assert!(err.contains("invalid weight"));
    }

    #[test]
    fn resolve_json_path_non_object_intermediate_returns_none() {
        let row = json!({"set": "not-an-object"});
        let obj = row.as_object().unwrap();
        // "set" exists but is a string, not an object — inner.get should fail.
        assert_eq!(
            HuggingFaceRowSource::resolve_json_path(obj, "set.query"),
            None
        );
    }
}
