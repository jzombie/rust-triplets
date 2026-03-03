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
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use std::time::Instant;
#[cfg(test)]
use tempfile::TempDir;
use tracing::{info, warn};
use walkdir::WalkDir;

use crate::SamplerError;
use crate::config::{NegativeStrategy, SamplerConfig, Selector, TripletRecipe};
use crate::constants::cache::HUGGINGFACE_GROUP;
use crate::data::{DataRecord, QualityScore, SectionRole};
use crate::utils::make_section;
use chrono::{DateTime, Utc};

use crate::source::{DataSource, SourceCursor, SourceSnapshot};

const REMOTE_URL_PREFIX: &str = "url::";
/// Extra row-index headroom above currently materialized rows exposed via `len_hint`.
///
/// This is not a file count. It lets sampling look slightly past the local row
/// frontier so lazy remote expansion can continue without jumping to the full
/// global row domain at once.
/// Multiplies the sampler ingestion base (`SamplerConfig.ingestion_max_records`)
/// to compute `len_hint` expansion headroom rows.
const REMOTE_EXPANSION_HEADROOM_MULTIPLIER: usize = 4;
/// Number of initial remote shards to materialize when bootstrapping an empty
/// local snapshot before regular lazy expansion.
const REMOTE_BOOTSTRAP_SHARDS: usize = 1;
/// Multiplies the source `refresh` limit passed by `IngestionManager`
/// (`step.unwrap_or(max_records)`) to set this source's internal row-read
/// batch target for each refresh pass.
const HUGGINGFACE_REFRESH_BATCH_MULTIPLIER: usize = 8;
const SHARD_SEQUENCE_STATE_VERSION: u32 = 1;
const SHARD_SEQUENCE_STATE_FILE: &str = "_sequence_state.json";
const HF_SHARD_STORE_EXTENSION: &str = "simdr";
const HF_SHARD_STORE_ROW_PREFIX: &[u8] = b"rowv1|";
const HF_SHARD_STORE_META_ROWS_KEY: &[u8] = b"meta|rows";
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
    ensure_cache_group(
        PathBuf::from(HUGGINGFACE_GROUP)
            .join("source-list")
            .join(dataset.replace('/', "__"))
            .join(config)
            .join(split)
            .join(format!("replica_{replica_idx}")),
    )
}

/// Resolve a managed snapshot directory for a single Hugging Face source.
pub fn managed_hf_snapshot_dir(
    dataset: &str,
    config: &str,
    split: &str,
) -> Result<PathBuf, String> {
    ensure_cache_group(
        PathBuf::from(HUGGINGFACE_GROUP)
            .join(dataset.replace('/', "__"))
            .join(config)
            .join(split),
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HfSourceEntry {
    /// Full hf:// URI for dataset/config/split.
    pub uri: String,
    /// Optional anchor column name.
    pub anchor_column: Option<String>,
    /// Optional positive column name.
    pub positive_column: Option<String>,
    /// Optional context columns (ordered).
    pub context_columns: Vec<String>,
    /// Optional text columns (ordered) for text-columns mode.
    pub text_columns: Vec<String>,
}

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
        anchor_column: None,
        positive_column: None,
        context_columns: Vec::new(),
        text_columns: Vec::new(),
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
                entry.anchor_column = (!value.is_empty()).then(|| value.to_string());
            }
            "positive" => {
                entry.positive_column = (!value.is_empty()).then(|| value.to_string());
            }
            "context" => {
                entry.context_columns = parse_csv_fields(value);
            }
            "text" | "text_columns" => {
                entry.text_columns = parse_csv_fields(value);
            }
            _ => {
                return Err(format!("unsupported mapping key '{raw_key}'"));
            }
        }
    }

    let has_explicit_mapping = entry.anchor_column.is_some()
        || entry.positive_column.is_some()
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
    let split = parts.get(3).copied().unwrap_or("train").to_string();

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

/// Build Hugging Face row sources from a parsed source list.
pub fn build_hf_sources(roots: &HfListRoots) -> Vec<Box<dyn DataSource + 'static>> {
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

            let source_id = format!("hf_list_{idx}");
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
            hf.anchor_column = source.anchor_column.clone();
            hf.positive_column = source.positive_column.clone();
            hf.context_columns = source.context_columns.clone();
            hf.text_columns = source.text_columns.clone();
            println!(
                "source {idx}: hf://{}/{}/{} -> anchor={:?}, positive={:?}, context={:?}, text_columns={:?}",
                hf.dataset,
                hf.config,
                hf.split,
                hf.anchor_column,
                hf.positive_column,
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
    /// Text columns to extract explicitly.
    pub text_columns: Vec<String>,
    /// Optional column used for anchor text.
    ///
    /// When set (or when `positive_column`/`context_columns` are set), role-based
    /// extraction is used instead of `text_columns` mode.
    pub anchor_column: Option<String>,
    /// Optional column used for positive text.
    ///
    /// Positive text is emitted as a `SectionRole::Context` section.
    pub positive_column: Option<String>,
    /// Optional ordered context columns.
    ///
    /// Used only in role-based extraction mode.
    pub context_columns: Vec<String>,
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
            anchor_column: None,
            positive_column: None,
            context_columns: Vec::new(),
        }
    }

    fn has_explicit_mapping(&self) -> bool {
        self.anchor_column.is_some()
            || self.positive_column.is_some()
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
    is_parquet: bool,
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
    remote_candidates: Option<Vec<String>>,
    remote_candidate_sizes: HashMap<String, u64>,
    next_remote_idx: usize,
    /// Candidates whose local shard files were evicted from the cache and
    /// must be re-downloaded before new candidates are consumed.
    eviction_queue: VecDeque<String>,
}

type ParquetGroupKey = (PathBuf, usize);
type ParquetGroupRequest = (usize, usize, ShardIndex);

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PersistedShardSequence {
    version: u32,
    source_id: String,
    dataset: String,
    config: String,
    split: String,
    sampler_seed: u64,
    candidates: Vec<String>,
    candidate_sizes: HashMap<String, u64>,
    next_remote_idx: usize,
}

impl HuggingFaceRowSource {
    /// Build a new source by indexing local shard files.
    pub fn new(config: HuggingFaceRowsConfig) -> Result<Self, SamplerError> {
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
            "[triplets:hf] indexing local shards in {}",
            config.snapshot_dir.display()
        );
        let (shards, discovered) = Self::build_shard_index(&config).unwrap_or_default();
        if discovered == 0 {
            info!(
                "[triplets:hf] no local shards found in {} — lazy remote download enabled",
                config.snapshot_dir.display()
            );
        }

        let materialized_rows = discovered;
        let total_rows = match Self::fetch_global_row_count(&config) {
            Ok(value) => value,
            Err(err) => {
                warn!(
                    "[triplets:hf] global row count request failed; continuing with discovered rows only: {}",
                    err
                );
                None
            }
        };

        if let Some(global_total) = total_rows {
            info!(
                "[triplets:hf] global split row count reported: {} (known_local_rows={})",
                global_total, materialized_rows
            );
        }

        info!(
            "[triplets:hf] source ready in {:.2}s (rows={}, shards={})",
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
                eviction_queue: VecDeque::new(),
            })),
            cache: Arc::new(Mutex::new(RowCache::default())),
            parquet_cache: Arc::new(Mutex::new(ParquetCache::default())),
            store_cache: Arc::new(Mutex::new(HashMap::new())),
            eligible_index: Arc::new(Mutex::new(EligibleIndexCache::default())),
            expansion_thread: Arc::new(Mutex::new(None)),
        })
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
        if !shard.is_parquet {
            return Ok(Some(shard.clone()));
        }

        let store_path = Self::shard_store_path_for(&shard.path);
        let store = self.get_or_open_shard_store(&store_path)?;
        if store_path.exists() {
            let existing_rows = self.read_store_row_count(&store)?;
            if existing_rows > 0 {
                return Ok(Some(ShardIndex {
                    path: store_path,
                    global_start: shard.global_start,
                    row_count: existing_rows,
                    is_parquet: true,
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

            let mut batch: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
            batch.reserve(group_count);

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
            is_parquet: true,
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

    fn shard_signature(shards: &[ShardIndex]) -> u64 {
        let mut hasher = DefaultHasher::new();
        for shard in shards {
            shard.path.hash(&mut hasher);
            shard.global_start.hash(&mut hasher);
            shard.row_count.hash(&mut hasher);
            shard.is_parquet.hash(&mut hasher);
            shard.parquet_row_groups.hash(&mut hasher);
        }
        hasher.finish()
    }

    fn build_eligible_rows_from_shards(
        &self,
        shards: &[ShardIndex],
    ) -> Result<Vec<usize>, SamplerError> {
        let mut eligible = Vec::new();

        for shard in shards {
            if shard.is_parquet {
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

    fn candidates_from_parquet_manifest_json(
        config: &HuggingFaceRowsConfig,
        json: &Value,
    ) -> Result<(Vec<String>, HashMap<String, u64>), SamplerError> {
        let accepted = Self::normalized_shard_extensions(config);

        let mut candidates = Vec::new();
        let mut candidate_sizes = HashMap::new();
        if let Some(entries) = json.get("parquet_files").and_then(Value::as_array) {
            for entry in entries {
                let Some(url) = entry.get("url").and_then(Value::as_str) else {
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

                let candidate = format!("{REMOTE_URL_PREFIX}{url}");
                let expected_size = entry.get("size").and_then(Value::as_u64);
                let target = Self::candidate_target_path(config, &candidate);
                let store_target = Self::shard_store_path_for(&target);
                if store_target.exists() {
                    continue;
                }
                if target.exists() {
                    if Self::target_matches_expected_size(&target, expected_size) {
                        continue;
                    }
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
        Ok((candidates, candidate_sizes))
    }

    /// Resolve and filter remote shard candidates from manifest or repository listing.
    fn list_remote_candidates(
        config: &HuggingFaceRowsConfig,
    ) -> Result<(Vec<String>, HashMap<String, u64>), SamplerError> {
        if let Ok((candidates, candidate_sizes)) =
            Self::list_remote_candidates_from_parquet_manifest(config)
            && !candidates.is_empty()
        {
            info!(
                "[triplets:hf] remote parquet manifest candidates matching {:?}: {}",
                config.shard_extensions,
                candidates.len()
            );
            return Ok((candidates, candidate_sizes));
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

    /// Return the persistence file path for shard sequence state.
    fn shard_sequence_state_path(config: &HuggingFaceRowsConfig) -> PathBuf {
        config
            .snapshot_dir
            .join("_parquet_manifest")
            .join(SHARD_SEQUENCE_STATE_FILE)
    }

    /// Load persisted shard candidate sequence when metadata and sampler seed match.
    #[cfg(test)]
    fn load_persisted_shard_sequence(
        config: &HuggingFaceRowsConfig,
        current_sampler_seed: u64,
    ) -> Result<Option<PersistedShardSequence>, SamplerError> {
        let path = Self::shard_sequence_state_path(config);
        if !path.exists() {
            return Ok(None);
        }

        let raw = fs::read_to_string(&path).map_err(|err| SamplerError::SourceUnavailable {
            source_id: config.source_id.clone(),
            reason: format!(
                "failed reading shard-sequence state {}: {err}",
                path.display()
            ),
        })?;

        let mut persisted: PersistedShardSequence =
            serde_json::from_str(&raw).map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!(
                    "failed parsing shard-sequence state {}: {err}",
                    path.display()
                ),
            })?;

        if persisted.version != SHARD_SEQUENCE_STATE_VERSION
            || persisted.source_id != config.source_id
            || persisted.dataset != config.dataset
            || persisted.config != config.config
            || persisted.split != config.split
            || persisted.sampler_seed != current_sampler_seed
        {
            warn!(
                "[triplets:hf] shard-sequence state mismatch for {}; rebuilding candidate order",
                path.display()
            );
            return Ok(None);
        }

        if persisted.next_remote_idx > persisted.candidates.len() {
            persisted.next_remote_idx = persisted.candidates.len();
        }

        Ok(Some(persisted))
    }

    /// Persist current shard candidate sequence and position atomically.
    fn persist_shard_sequence_locked(&self, state: &SourceState) -> Result<(), SamplerError> {
        let Some(candidates) = state.remote_candidates.as_ref() else {
            return Ok(());
        };

        let path = Self::shard_sequence_state_path(&self.config);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|err| SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: format!(
                    "failed creating shard-sequence state dir {}: {err}",
                    parent.display()
                ),
            })?;
        }

        let persisted = PersistedShardSequence {
            version: SHARD_SEQUENCE_STATE_VERSION,
            source_id: self.config.source_id.clone(),
            dataset: self.config.dataset.clone(),
            config: self.config.config.clone(),
            split: self.config.split.clone(),
            sampler_seed: self.configured_sampler_seed()?,
            candidates: candidates.clone(),
            candidate_sizes: state.remote_candidate_sizes.clone(),
            next_remote_idx: state.next_remote_idx.min(candidates.len()),
        };

        let raw = serde_json::to_vec_pretty(&persisted).map_err(|err| {
            SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: format!(
                    "failed encoding shard-sequence state {}: {err}",
                    path.display()
                ),
            }
        })?;

        let tmp_path = path.with_extension("tmp");
        fs::write(&tmp_path, raw).map_err(|err| SamplerError::SourceUnavailable {
            source_id: self.config.source_id.clone(),
            reason: format!(
                "failed writing shard-sequence state temp {}: {err}",
                tmp_path.display()
            ),
        })?;
        fs::rename(&tmp_path, &path).map_err(|err| SamplerError::SourceUnavailable {
            source_id: self.config.source_id.clone(),
            reason: format!(
                "failed replacing shard-sequence state {}: {err}",
                path.display()
            ),
        })?;

        Ok(())
    }

    /// Rotate candidate ordering deterministically using source identity.
    /// Shuffle remote shard candidates into a deterministic-but-random order.
    ///
    /// A Fisher-Yates shuffle seeded from source identity and the sampler seed
    /// ensures that downloaded shards are spread broadly across the full
    /// positional space of the dataset rather than walking file-name order from
    /// a fixed offset.  The same seed always produces the same shuffle, so runs
    /// are reproducible.
    fn shuffle_candidates_deterministically(
        config: &HuggingFaceRowsConfig,
        candidates: &mut [String],
        sampler_seed: u64,
    ) {
        let n = candidates.len();
        if n <= 1 {
            return;
        }
        let base_seed = Self::shard_candidate_seed(config, n, sampler_seed);
        // xorshift64 — fast, no dependency, deterministic.
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
            candidates.swap(i, j);
        }
    }

    /// Build deterministic seed used to permute remote shard candidate order.
    fn shard_candidate_seed(
        config: &HuggingFaceRowsConfig,
        total_candidates: usize,
        sampler_seed: u64,
    ) -> u64 {
        let mut hasher = DefaultHasher::new();
        "hf_shard_candidate_sequence_v1".hash(&mut hasher);
        sampler_seed.hash(&mut hasher);
        config.source_id.hash(&mut hasher);
        config.dataset.hash(&mut hasher);
        config.config.hash(&mut hasher);
        config.split.hash(&mut hasher);
        total_candidates.hash(&mut hasher);
        hasher.finish()
    }

    fn parquet_manifest_endpoint() -> String {
        #[cfg(test)]
        if let Ok(value) = std::env::var("TRIPLETS_HF_PARQUET_ENDPOINT")
            && !value.trim().is_empty()
        {
            return value;
        }
        "https://datasets-server.huggingface.co/parquet".to_string()
    }

    fn size_endpoint() -> String {
        #[cfg(test)]
        if let Ok(value) = std::env::var("TRIPLETS_HF_SIZE_ENDPOINT")
            && !value.trim().is_empty()
        {
            return value;
        }
        "https://datasets-server.huggingface.co/size".to_string()
    }

    /// Query datasets-server parquet manifest and derive shard candidates.
    fn list_remote_candidates_from_parquet_manifest(
        config: &HuggingFaceRowsConfig,
    ) -> Result<(Vec<String>, HashMap<String, u64>), SamplerError> {
        let endpoint = Self::parquet_manifest_endpoint();
        info!(
            "[triplets:hf] reading datasets-server parquet manifest for dataset {}",
            config.dataset
        );
        let response = ureq::get(&endpoint)
            .query("dataset", &config.dataset)
            .query("config", &config.config)
            .query("split", &config.split)
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
    ) -> Result<(Vec<String>, HashMap<String, u64>), SamplerError> {
        let json: Value =
            serde_json::from_str(body).map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!("failed parsing datasets-server parquet response: {err}"),
            })?;

        Self::candidates_from_parquet_manifest_json(config, &json)
    }

    /// Map a candidate identifier to the local snapshot target path.
    fn candidate_target_path(config: &HuggingFaceRowsConfig, candidate: &str) -> PathBuf {
        if let Some(url) = candidate.strip_prefix(REMOTE_URL_PREFIX) {
            let suffix = url
                .split("/resolve/")
                .nth(1)
                .map(|value| value.trim_start_matches('/'))
                .filter(|value| !value.is_empty())
                .unwrap_or("parquet/unknown.parquet");
            return config.snapshot_dir.join("_parquet_manifest").join(suffix);
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

    /// Return root directory used for manifest-cached remote shards.
    fn manifest_cache_root(&self) -> PathBuf {
        self.config.snapshot_dir.join("_parquet_manifest")
    }

    /// Return on-disk size for a shard path, or 0 if metadata lookup fails.
    fn shard_size_bytes(path: &Path) -> u64 {
        fs::metadata(path).map(|meta| meta.len()).unwrap_or(0)
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
        // Collect remote candidates for shards whose files have been evicted so
        // they can be re-downloaded before new candidates are consumed.
        for shard in &state.shards {
            if !shard.path.exists() {
                if let Some(candidate) = &shard.remote_candidate {
                    if !state.eviction_queue.contains(candidate) {
                        state.eviction_queue.push_back(candidate.clone());
                    }
                }
            }
        }
        state.shards.retain(|shard| shard.path.exists());
        Self::recompute_shard_offsets(state);
    }

    /// Apply cache-manager eviction policy and sync in-memory shard state.
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
            .ensure_group_with_policy("_parquet_manifest", Some(&policy))
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

        let response = ureq::get(&endpoint)
            .query("dataset", &config.dataset)
            .query("config", &config.config)
            .query("split", &config.split)
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

        let mut count =
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
            "[triplets:hf] downloading shard payload -> {}",
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
                        "[triplets:hf] download progress {}: {:.1}/{:.1} MiB ({:.1}%, {:.1}s elapsed, ETA {:.1}s)",
                        target.display(),
                        total_bytes as f64 / (1024.0 * 1024.0),
                        expected as f64 / (1024.0 * 1024.0),
                        pct,
                        elapsed,
                        eta_secs.max(0.0)
                    );
                } else {
                    info!(
                        "[triplets:hf] download progress {}: {:.1} MiB ({:.1}s)",
                        target.display(),
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
                "[triplets:hf] download complete {}: {:.1}/{:.1} MiB ({:.1}%) in {:.1}s",
                target.display(),
                total_bytes as f64 / (1024.0 * 1024.0),
                expected as f64 / (1024.0 * 1024.0),
                pct,
                elapsed
            );
        } else {
            info!(
                "[triplets:hf] download complete {}: {:.1} MiB in {:.1}s",
                target.display(),
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

        let size = json.get("size")?;

        if let Some(splits) = size.get("splits").and_then(Value::as_array) {
            for entry in splits {
                let entry_config = entry
                    .get("config")
                    .or_else(|| entry.get("config_name"))
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let entry_split = entry
                    .get("split")
                    .or_else(|| entry.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if entry_config == config_name
                    && entry_split == split_name
                    && let Some(rows) = entry.get("num_rows").and_then(to_usize)
                {
                    return Some(rows);
                }
            }
        }

        if let Some(configs) = size.get("configs").and_then(Value::as_array) {
            for config_entry in configs {
                let entry_config = config_entry
                    .get("config")
                    .or_else(|| config_entry.get("config_name"))
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if entry_config != config_name {
                    continue;
                }

                if let Some(splits) = config_entry.get("splits").and_then(Value::as_array) {
                    for split_entry in splits {
                        let entry_split = split_entry
                            .get("split")
                            .or_else(|| split_entry.get("name"))
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        if entry_split == split_name
                            && let Some(rows) = split_entry.get("num_rows").and_then(to_usize)
                        {
                            return Some(rows);
                        }
                    }
                }

                if split_name.is_empty()
                    && let Some(rows) = config_entry.get("num_rows").and_then(to_usize)
                {
                    return Some(rows);
                }
            }
        }

        if split_name.is_empty() {
            return size
                .get("dataset")
                .and_then(|dataset| dataset.get("num_rows"))
                .and_then(to_usize);
        }

        None
    }

    /// Download a shard (URL or hf-hub path) and materialize it under snapshot dir.
    fn download_and_materialize_shard(
        config: &HuggingFaceRowsConfig,
        remote_path: &str,
        expected_bytes: Option<u64>,
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

            Self::download_remote_url_to_target(config, remote_url, &target, expected_bytes)?;
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
            Self::download_remote_url_to_target(config, &remote_url, &temp_target, expected_bytes)?;
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
    ) -> Result<Option<ShardIndex>, SamplerError> {
        let is_store = Self::is_store_shard_path(path);
        // Parquet is treated as a transient decode artifact only.
        // Persisted shard artifacts should be per-shard .simdr stores.
        let is_transient_parquet = path
            .extension()
            .and_then(|v| v.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("parquet"));

        let (rows, parquet_row_groups, checkpoints) = if is_store {
            let store = Self::open_shard_store(config, path)?;
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
            (rows, groups, Vec::new())
        } else if is_transient_parquet {
            let (rows, parquet_row_groups) = Self::parquet_row_group_map(config, path)?;
            (rows, parquet_row_groups, Vec::new())
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

            (rows, Vec::new(), checkpoints)
        };

        if rows == 0 {
            return Ok(None);
        }

        Ok(Some(ShardIndex {
            path: path.to_path_buf(),
            global_start,
            row_count: rows,
            is_parquet: is_transient_parquet || is_store,
            parquet_row_groups,
            checkpoints,
            remote_candidate: None,
        }))
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
                    let sampler_seed = self.configured_sampler_seed().unwrap_or(0);
                    Self::shuffle_candidates_deterministically(
                        &self.config,
                        &mut candidates,
                        sampler_seed,
                    );
                    state.remote_candidates = Some(candidates);
                    state.remote_candidate_sizes = candidate_sizes;
                    state.next_remote_idx = 0;

                    self.persist_shard_sequence_locked(&state)?;

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
                        "[triplets:hf] state: candidates={} known_rows={} active_shards={} disk_cap={}",
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
                            "[triplets:hf] bootstrapping remote shard diversity: target={} shard(s)",
                            bootstrap_target
                        );
                        for step in 0..bootstrap_target {
                            info!(
                                "[triplets:hf] bootstrap progress: {}/{}",
                                step + 1,
                                bootstrap_target
                            );
                            if !self.download_next_remote_shard()? {
                                break;
                            }
                        }
                        info!("[triplets:hf] bootstrap complete");
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
    fn download_next_remote_shard(&self) -> Result<bool, SamplerError> {
        let (is_recovery, remote_ordinal, remote_total, remote_path, expected_bytes) = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| SamplerError::SourceUnavailable {
                    source_id: self.config.source_id.clone(),
                    reason: "huggingface source state lock poisoned".to_string(),
                })?;
            // Drain the eviction queue first so evicted shards are recovered
            // before new candidates are consumed.
            if let Some(candidate) = state.eviction_queue.pop_front() {
                let expected = state.remote_candidate_sizes.get(&candidate).copied();
                (true, 0usize, 0usize, candidate, expected)
            } else {
                let Some(candidates) = &state.remote_candidates else {
                    return Ok(false);
                };
                if state.next_remote_idx >= candidates.len() {
                    return Ok(false);
                }
                let sequence_pos = state.next_remote_idx;
                let remote_ordinal = sequence_pos + 1;
                let remote_total = candidates.len();
                let candidate_idx = sequence_pos;
                let remote_path = candidates[candidate_idx].clone();
                let expected_bytes = state.remote_candidate_sizes.get(&remote_path).copied();
                state.next_remote_idx += 1;
                (
                    false,
                    remote_ordinal,
                    remote_total,
                    remote_path,
                    expected_bytes,
                )
            }
        };

        if is_recovery {
            info!(
                "[triplets:hf] re-downloading evicted shard: {}",
                remote_path.as_str()
            );
        } else {
            info!(
                "[triplets:hf] lazy downloading shard {}/{}: {}",
                remote_ordinal,
                remote_total,
                remote_path.as_str()
            );
        }
        let local_path =
            Self::download_and_materialize_shard(&self.config, &remote_path, expected_bytes)?;

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

        let Some(shard) = Self::index_single_shard(&self.config, &local_path, global_start)? else {
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

        let mut rows_to_add = shard.row_count;
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
        if shard.is_parquet {
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
        self.persist_shard_sequence_locked(&state)?;
        let materialized_rows = state.materialized_rows;
        let shard_count = state.shards.len();
        let active_shards = state.shards.clone();
        let remaining_candidates = state
            .remote_candidates
            .as_ref()
            .map(|candidates| candidates.len().saturating_sub(state.next_remote_idx))
            .unwrap_or(0);
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

        info!(
            "[triplets:hf] state: rows={} shards={} remaining_candidates={} disk_usage={:.2} GiB cap={}",
            materialized_rows, shard_count, remaining_candidates, usage_gib, cap_str,
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
    fn build_shard_index(
        config: &HuggingFaceRowsConfig,
    ) -> Result<(Vec<ShardIndex>, usize), SamplerError> {
        let start_index = Instant::now();
        let mut shard_paths = Vec::new();
        let manifest_root = config.snapshot_dir.join("_parquet_manifest");
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
            if entry.path().starts_with(&manifest_root) {
                continue;
            }
            let Some(ext) = entry.path().extension().and_then(|v| v.to_str()) else {
                continue;
            };
            if ext.eq_ignore_ascii_case("parquet") {
                saw_parquet = true;
                if let Err(err) = fs::remove_file(entry.path()) {
                    warn!(
                        "[triplets:hf] found persisted parquet shard (expected transient only) and failed to remove {}: {}",
                        entry.path().display(),
                        err
                    );
                }
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

        let mut indexed_shards = Vec::with_capacity(shard_paths.len());
        for (ordinal, path) in shard_paths.into_iter().enumerate() {
            info!(
                "[triplets:hf] indexing shard {}: {}",
                ordinal + 1,
                path.display()
            );
            let shard = Self::index_single_shard(config, &path, 0)?;
            indexed_shards.push((ordinal, shard));
        }

        indexed_shards.sort_by_key(|(ordinal, _)| *ordinal);

        let mut shards = Vec::new();
        let mut running_total = 0usize;
        for (_, maybe_shard) in indexed_shards {
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
            Value::Array(_) | Value::Object(_) => Some(value.to_string()),
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
            .and_then(|col| row_obj.get(col))
            .and_then(Self::value_to_text)
            .unwrap_or_else(|| {
                format!(
                    "{}:{}:{}",
                    self.config.dataset, self.config.split, absolute_idx
                )
            });

        let mut text_fields = Vec::new();
        let use_role_columns = self.config.anchor_column.is_some()
            || self.config.positive_column.is_some()
            || !self.config.context_columns.is_empty();

        if use_role_columns {
            if let Some(name) = &self.config.anchor_column {
                let Some(value) = row_obj.get(name) else {
                    return Ok(None);
                };
                let Some(text) = Self::value_to_text(value) else {
                    return Ok(None);
                };
                text_fields.push(RowTextField {
                    name: name.clone(),
                    text,
                });
            }

            if let Some(name) = &self.config.positive_column {
                let Some(value) = row_obj.get(name) else {
                    return Ok(None);
                };
                let Some(text) = Self::value_to_text(value) else {
                    return Ok(None);
                };
                text_fields.push(RowTextField {
                    name: name.clone(),
                    text,
                });
            }

            for name in &self.config.context_columns {
                let Some(value) = row_obj.get(name) else {
                    return Ok(None);
                };
                let Some(text) = Self::value_to_text(value) else {
                    return Ok(None);
                };
                text_fields.push(RowTextField {
                    name: name.clone(),
                    text,
                });
            }
        } else {
            for name in &self.config.text_columns {
                let Some(value) = row_obj.get(name) else {
                    return Ok(None);
                };
                let Some(text) = Self::value_to_text(value) else {
                    return Ok(None);
                };
                text_fields.push(RowTextField {
                    name: name.clone(),
                    text,
                });
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
            quality: QualityScore::default(),
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
                if shard.is_parquet {
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

    /// Forward expansion hint to background shard downloader.
    fn try_expand(&self, _config: &SamplerConfig) {
        self.trigger_expansion_if_needed();
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
    use std::sync::OnceLock;
    use std::thread;
    use tempfile::tempdir;

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
                remote_candidates: None,
                remote_candidate_sizes: HashMap::new(),
                next_remote_idx: 0,
                eviction_queue: VecDeque::new(),
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
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let guard = ENV_LOCK
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
        let result = run();
        drop(guard);
        result
    }

    fn with_current_dir<R>(dir: &Path, run: impl FnOnce() -> R) -> R {
        static CWD_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let guard = CWD_LOCK
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
        let result = run();
        drop(guard);
        result
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
    fn persist_shard_sequence_errors_when_manifest_dir_is_blocked_by_file() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        fs::write(config.snapshot_dir.join("_parquet_manifest"), b"blocked").unwrap();
        let source = test_source(config);
        let state = SourceState {
            materialized_rows: 0,
            total_rows: None,
            shards: Vec::new(),
            remote_candidates: Some(vec!["train/a.ndjson".to_string()]),
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            eviction_queue: VecDeque::new(),
        };

        let err = source.persist_shard_sequence_locked(&state);
        assert!(err.is_err());
    }

    #[test]
    fn list_remote_candidates_falls_back_when_manifest_query_fails() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.dataset = "invalid///dataset".to_string();

        let result = with_env_var(
            "TRIPLETS_HF_PARQUET_ENDPOINT",
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
                    anchor_column: Some("title".to_string()),
                    positive_column: None,
                    context_columns: Vec::new(),
                    text_columns: Vec::new(),
                },
                HfSourceEntry {
                    uri: "hf://org/dataset/default/train".to_string(),
                    anchor_column: Some("title".to_string()),
                    positive_column: Some("body".to_string()),
                    context_columns: Vec::new(),
                    text_columns: Vec::new(),
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

        with_current_dir(temp_root.path(), || {
            let built = build_hf_sources(&roots);
            assert_eq!(built.len(), 1);
        });
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
    fn candidates_from_parquet_manifest_json_filters_and_records_sizes() {
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

        let (candidates, sizes) =
            HuggingFaceRowSource::candidates_from_parquet_manifest_json(&config, &payload).unwrap();
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
    }

    #[test]
    fn candidates_from_parquet_manifest_skips_complete_cached_and_replaces_incomplete() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());

        let complete_url = "https://host/datasets/org/ds/resolve/main/train/000.parquet";
        let complete_candidate = format!("{REMOTE_URL_PREFIX}{complete_url}");
        let complete_target =
            HuggingFaceRowSource::candidate_target_path(&config, &complete_candidate);
        fs::create_dir_all(complete_target.parent().unwrap()).unwrap();
        fs::write(&complete_target, vec![1u8; 7]).unwrap();

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

        let (candidates, sizes) =
            HuggingFaceRowSource::candidates_from_parquet_manifest_json(&config, &payload).unwrap();
        assert_eq!(candidates.len(), 1);
        assert!(candidates[0].ends_with(stale_url));
        assert!(!stale_target.exists());
        assert_eq!(sizes[&candidates[0]], 9);
        assert!(complete_target.exists());
    }

    #[test]
    fn candidates_from_parquet_manifest_errors_when_removing_incomplete_target_fails() {
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

        let err = HuggingFaceRowSource::candidates_from_parquet_manifest_json(&config, &payload);
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
                    is_parquet: true,
                    parquet_row_groups: vec![(0, 1)],
                    checkpoints: Vec::new(),
                    remote_candidate: None,
                },
                ShardIndex {
                    path: local_file,
                    global_start: 1,
                    row_count: 1,
                    is_parquet: false,
                    parquet_row_groups: Vec::new(),
                    checkpoints: vec![0],
                    remote_candidate: None,
                },
            ],
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            eviction_queue: VecDeque::new(),
        };

        assert_eq!(source.manifest_usage_bytes_locked(&state), 7);
    }

    #[test]
    fn build_shard_index_errors_when_parquet_present_but_not_accepted() {
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
            is_parquet: true,
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
        config.anchor_column = Some("anchor".into());
        config.positive_column = Some("positive".into());
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
        config.anchor_column = Some("anchor".into());
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
    fn persist_shard_sequence_is_noop_without_remote_candidates() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());
        let state = SourceState {
            materialized_rows: 0,
            total_rows: None,
            shards: Vec::new(),
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            eviction_queue: VecDeque::new(),
        };

        source.persist_shard_sequence_locked(&state).unwrap();
        assert!(!HuggingFaceRowSource::shard_sequence_state_path(&config).exists());
    }

    #[test]
    fn load_persisted_shard_sequence_returns_none_for_identity_mismatch() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let state_path = HuggingFaceRowSource::shard_sequence_state_path(&config);
        fs::create_dir_all(state_path.parent().unwrap()).unwrap();
        fs::write(
            &state_path,
            serde_json::to_vec_pretty(&json!({
                "version": 1,
                "source_id": "different",
                "dataset": config.dataset,
                "config": config.config,
                "split": config.split,
                "sampler_seed": 1,
                "candidates": ["train/0.ndjson"],
                "candidate_sizes": {},
                "next_remote_idx": 0
            }))
            .unwrap(),
        )
        .unwrap();

        let loaded = HuggingFaceRowSource::load_persisted_shard_sequence(&config, 1).unwrap();
        assert!(loaded.is_none());
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
            is_parquet: false,
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
                is_parquet: false,
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
                is_parquet: true,
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
            eviction_queue: VecDeque::new(),
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
                is_parquet: true,
                parquet_row_groups: vec![(0, 1)],
                checkpoints: Vec::new(),
                remote_candidate: None,
            }],
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            eviction_queue: VecDeque::new(),
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
                    is_parquet: true,
                    parquet_row_groups: vec![(0, 1)],
                    checkpoints: Vec::new(),
                    remote_candidate: None,
                },
                ShardIndex {
                    path: second.clone(),
                    global_start: 1,
                    row_count: 1,
                    is_parquet: true,
                    parquet_row_groups: vec![(0, 1)],
                    checkpoints: Vec::new(),
                    remote_candidate: None,
                },
            ],
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            eviction_queue: VecDeque::new(),
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
                is_parquet: true,
                parquet_row_groups: vec![(0, 1)],
                checkpoints: Vec::new(),
                remote_candidate: None,
            }],
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            eviction_queue: VecDeque::new(),
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
                eviction_queue: VecDeque::new(),
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

        let err = HuggingFaceRowSource::build_shard_index(&config).unwrap_err();
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
        )
        .unwrap_err();
        assert!(matches!(err, SamplerError::SourceUnavailable { .. }));
    }

    #[test]
    fn index_single_shard_errors_for_missing_file() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let missing = dir.path().join("missing.ndjson");

        let err = HuggingFaceRowSource::index_single_shard(&config, &missing, 0).unwrap_err();
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
            .unwrap();
        assert_eq!(shard.global_start, 5);
        assert_eq!(shard.row_count, 3);
        assert!(!shard.is_parquet);
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
                is_parquet: true,
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
    fn load_persisted_shard_sequence_clamps_next_remote_index() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let state_path = HuggingFaceRowSource::shard_sequence_state_path(&config);
        fs::create_dir_all(state_path.parent().unwrap()).unwrap();
        fs::write(
            &state_path,
            serde_json::to_vec_pretty(&json!({
                "version": SHARD_SEQUENCE_STATE_VERSION,
                "source_id": config.source_id,
                "dataset": config.dataset,
                "config": config.config,
                "split": config.split,
                "sampler_seed": 7,
                "candidates": ["train/0.ndjson", "train/1.ndjson"],
                "candidate_sizes": {},
                "next_remote_idx": 99
            }))
            .unwrap(),
        )
        .unwrap();

        let loaded = HuggingFaceRowSource::load_persisted_shard_sequence(&config, 7)
            .unwrap()
            .unwrap();
        assert_eq!(loaded.candidates.len(), 2);
        assert_eq!(loaded.next_remote_idx, 2);
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

        let resolved =
            HuggingFaceRowSource::download_and_materialize_shard(&config, candidate, Some(2))
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

        let out = HuggingFaceRowSource::download_and_materialize_shard(&config, &candidate, None)
            .unwrap();
        server.join().unwrap();

        assert_eq!(out, target);
        assert_eq!(fs::read(&target).unwrap(), payload);
    }

    #[test]
    fn download_next_remote_shard_skips_zero_row_download() {
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
                is_parquet: true,
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
    fn load_persisted_shard_sequence_errors_on_invalid_json() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let path = HuggingFaceRowSource::shard_sequence_state_path(&config);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"{not-valid-json").unwrap();

        let loaded = HuggingFaceRowSource::load_persisted_shard_sequence(&config, 1);
        assert!(loaded.is_err());
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
    fn candidates_from_parquet_manifest_json_returns_empty_without_entries() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let payload = json!({"other": []});
        let (candidates, sizes) =
            HuggingFaceRowSource::candidates_from_parquet_manifest_json(&config, &payload).unwrap();
        assert!(candidates.is_empty());
        assert!(sizes.is_empty());
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
            .unwrap();
        let end = fs::metadata(&path).unwrap().len();
        shard.checkpoints = vec![0, end];

        let err = source.read_line_at(&shard, 1);
        assert!(err.is_err());
    }

    #[test]
    fn load_persisted_shard_sequence_returns_none_when_state_missing() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let loaded = HuggingFaceRowSource::load_persisted_shard_sequence(&config, 1).unwrap();
        assert!(loaded.is_none());
    }

    #[test]
    fn persist_shard_sequence_clamps_next_index_on_write() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());
        let state = SourceState {
            materialized_rows: 0,
            total_rows: None,
            shards: Vec::new(),
            remote_candidates: Some(vec!["a".into(), "b".into()]),
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 99,
            eviction_queue: VecDeque::new(),
        };

        source.persist_shard_sequence_locked(&state).unwrap();
        let raw =
            fs::read_to_string(HuggingFaceRowSource::shard_sequence_state_path(&config)).unwrap();
        let parsed: Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(
            parsed.get("next_remote_idx").and_then(Value::as_u64),
            Some(2)
        );
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
            .unwrap();
        assert!(shard.is_parquet);
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
    fn refresh_ignores_persisted_remote_sequence_state() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        let payload = b"{\"id\":\"rr\",\"text\":\"hello\"}\n".to_vec();
        let (base_url, server) = spawn_one_shot_http(payload);
        let candidate =
            format!("url::{base_url}/datasets/org/ds/resolve/main/train/refresh.ndjson");

        let state_path = HuggingFaceRowSource::shard_sequence_state_path(&config);
        fs::create_dir_all(state_path.parent().unwrap()).unwrap();
        fs::write(
            &state_path,
            serde_json::to_vec_pretty(&json!({
                "version": 1,
                "source_id": config.source_id,
                "dataset": config.dataset,
                "config": config.config,
                "split": config.split,
                "sampler_seed": 1,
                "candidates": [candidate],
                "candidate_sizes": {},
                "next_remote_idx": 1
            }))
            .unwrap(),
        )
        .unwrap();

        {
            let mut state = source.state.lock().unwrap();
            state.remote_candidates = Some(vec![format!(
                "url::{base_url}/datasets/org/ds/resolve/main/train/refresh.ndjson"
            )]);
            state.next_remote_idx = 0;
        }

        let snapshot = source.refresh(None, Some(1)).unwrap();
        server.join().unwrap();

        assert_eq!(snapshot.records.len(), 1);
        assert!(snapshot.records[0].id.contains("hf_test::rr"));
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
    fn resolve_remote_candidates_from_siblings_falls_back_when_split_filter_misses() {
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

        let (candidates, sizes) =
            HuggingFaceRowSource::parse_parquet_manifest_response(&config, &body).unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(sizes.len(), 1);
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

        let (candidates, sizes) = with_env_var("TRIPLETS_HF_PARQUET_ENDPOINT", &base_url, || {
            HuggingFaceRowSource::list_remote_candidates_from_parquet_manifest(&config)
        })
        .unwrap();
        server.join().unwrap();

        assert_eq!(candidates.len(), 1);
        assert_eq!(sizes.len(), 1);
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

        let rows = with_env_var("TRIPLETS_HF_SIZE_ENDPOINT", &base_url, || {
            HuggingFaceRowSource::fetch_global_row_count(&config)
        })
        .unwrap();
        server.join().unwrap();
        assert_eq!(rows, Some(12));
    }

    #[test]
    fn endpoint_helpers_fallback_for_empty_env_values() {
        let parquet = with_env_var("TRIPLETS_HF_PARQUET_ENDPOINT", "   ", || {
            HuggingFaceRowSource::parquet_manifest_endpoint()
        });
        assert_eq!(parquet, "https://datasets-server.huggingface.co/parquet");

        let size = with_env_var("TRIPLETS_HF_SIZE_ENDPOINT", "", || {
            HuggingFaceRowSource::size_endpoint()
        });
        assert_eq!(size, "https://datasets-server.huggingface.co/size");
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

        let rows = with_env_var("TRIPLETS_HF_SIZE_ENDPOINT", &base_url, || {
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

        let (candidates, sizes) = with_env_var("TRIPLETS_HF_PARQUET_ENDPOINT", &base_url, || {
            HuggingFaceRowSource::list_remote_candidates(&config)
        })
        .unwrap();
        server.join().unwrap();

        assert_eq!(candidates.len(), 1);
        assert_eq!(sizes.len(), 1);
        assert!(candidates[0].ends_with("/1.ndjson"));
    }

    #[test]
    fn list_remote_candidates_from_parquet_manifest_errors_when_endpoint_unreachable() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());

        let result = with_env_var("TRIPLETS_HF_PARQUET_ENDPOINT", "http://127.0.0.1:1", || {
            HuggingFaceRowSource::list_remote_candidates_from_parquet_manifest(&config)
        });
        assert!(result.is_err());
    }

    #[test]
    fn fetch_global_row_count_errors_when_endpoint_unreachable() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());

        let result = with_env_var("TRIPLETS_HF_SIZE_ENDPOINT", "http://127.0.0.1:1", || {
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

        let target =
            HuggingFaceRowSource::download_and_materialize_shard(&config, &candidate, None)
                .unwrap();

        server.join().unwrap();
        assert!(target.exists());
        assert_eq!(fs::read(&target).unwrap(), payload);
    }

    #[test]
    fn download_and_materialize_shard_replaces_incomplete_existing_target() {
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
    fn build_shard_index_ignores_manifest_cache_artifacts() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.shard_extensions = vec!["ndjson".to_string()];

        let local = dir.path().join("local.ndjson");
        fs::write(&local, b"{\"id\":\"l1\",\"text\":\"x\"}\n").unwrap();

        let manifest_cached = dir
            .path()
            .join("_parquet_manifest")
            .join("main/train/cached.ndjson");
        fs::create_dir_all(manifest_cached.parent().unwrap()).unwrap();
        fs::write(&manifest_cached, b"{\"id\":\"r1\",\"text\":\"y\"}\n").unwrap();

        let (shards, discovered) = HuggingFaceRowSource::build_shard_index(&config).unwrap();
        assert_eq!(discovered, 1);
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0].path, local);
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
    fn persisted_shard_sequence_roundtrip_respects_sampler_seed() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        {
            let sampler = SamplerConfig {
                seed: 4242,
                ..SamplerConfig::default()
            };
            source.configure_sampler(&sampler);
        }

        let mut state = SourceState {
            materialized_rows: 0,
            total_rows: None,
            shards: Vec::new(),
            remote_candidates: Some(vec![
                "url::https://x/resolve/main/train/000.parquet".to_string(),
                "url::https://x/resolve/main/train/001.parquet".to_string(),
            ]),
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 1,
            eviction_queue: VecDeque::new(),
        };
        state.remote_candidate_sizes.insert(
            "url::https://x/resolve/main/train/000.parquet".to_string(),
            10,
        );

        source.persist_shard_sequence_locked(&state).unwrap();

        let restored = HuggingFaceRowSource::load_persisted_shard_sequence(&config, 4242).unwrap();
        assert!(restored.is_some());
        let restored = restored.unwrap();
        assert_eq!(restored.next_remote_idx, 1);
        assert_eq!(restored.candidates.len(), 2);

        let rejected = HuggingFaceRowSource::load_persisted_shard_sequence(&config, 9999).unwrap();
        assert!(rejected.is_none());
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

        assert_eq!(row.row_id.as_deref(), Some("row-5"));
        assert_eq!(row.text_fields.len(), 2);
        assert_eq!(row.text_fields[0].name, "title");
        assert_eq!(row.text_fields[1].name, "body");
        assert!(row.text_fields.iter().all(|field| field.name != "id"));
    }

    #[test]
    fn parse_row_with_required_columns_skips_when_missing() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.anchor_column = Some("anchor".into());
        config.positive_column = Some("positive".into());
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
                is_parquet: false,
                parquet_row_groups: Vec::new(),
                checkpoints: vec![0],
                remote_candidate: None,
            },
            ShardIndex {
                path: PathBuf::from("b"),
                global_start: 20,
                row_count: 2,
                is_parquet: false,
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
            eviction_queue: VecDeque::new(),
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
            .unwrap();

        let baseline = ShardIndex {
            path: dir.path().join("missing-baseline.ndjson"),
            global_start: 0,
            row_count: 1,
            is_parquet: false,
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
            cache.signature = Some(HuggingFaceRowSource::shard_signature(&[baseline.clone()]));
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
            .unwrap();
        shard.checkpoints.clear();

        let err = source.read_line_at(&shard, 0);
        assert!(err.is_err());
    }

    #[test]
    fn load_persisted_shard_sequence_clamps_next_index_to_candidate_len() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let state_path = HuggingFaceRowSource::shard_sequence_state_path(&config);
        fs::create_dir_all(state_path.parent().unwrap()).unwrap();
        fs::write(
            &state_path,
            serde_json::to_vec_pretty(&json!({
                "version": 1,
                "source_id": config.source_id,
                "dataset": config.dataset,
                "config": config.config,
                "split": config.split,
                "sampler_seed": 1,
                "candidates": ["url::http://x/resolve/main/train/000.ndjson"],
                "candidate_sizes": {},
                "next_remote_idx": 99
            }))
            .unwrap(),
        )
        .unwrap();

        let loaded = HuggingFaceRowSource::load_persisted_shard_sequence(&config, 1)
            .unwrap()
            .unwrap();
        assert_eq!(loaded.next_remote_idx, 1);
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
                    is_parquet: true,
                    parquet_row_groups: vec![(0, 8)],
                    checkpoints: Vec::new(),
                    remote_candidate: None,
                },
                ShardIndex {
                    path: keep_path.clone(),
                    global_start: 8,
                    row_count: 8,
                    is_parquet: true,
                    parquet_row_groups: vec![(0, 8)],
                    checkpoints: Vec::new(),
                    remote_candidate: None,
                },
            ],
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            eviction_queue: VecDeque::new(),
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
                is_parquet: true,
                parquet_row_groups: vec![(0, 8)],
                checkpoints: Vec::new(),
                remote_candidate: None,
            }],
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            eviction_queue: VecDeque::new(),
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
        let shard = HuggingFaceRowSource::index_single_shard(&config, &path, 0).unwrap();
        assert!(shard.is_none());
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

        assert_eq!(parsed.row_id.as_deref(), Some("r-1"));
        assert_eq!(parsed.text_fields.len(), 2);
        assert_eq!(parsed.text_fields[0].name, "headline");
    }

    #[test]
    fn parse_row_returns_none_when_positive_or_text_columns_are_missing() {
        let dir = tempdir().unwrap();
        let mut role_config = test_config(dir.path().to_path_buf());
        role_config.anchor_column = Some("anchor".into());
        role_config.positive_column = Some("positive".into());
        let role_source = test_source(role_config);

        let role_missing = role_source.parse_row(0, &json!({"anchor":"a"})).unwrap();
        assert!(role_missing.is_none());

        let mut text_config = test_config(dir.path().to_path_buf());
        text_config.text_columns = vec!["title".into(), "body".into()];
        let text_source = test_source(text_config);
        let text_missing = text_source.parse_row(1, &json!({"title":"t"})).unwrap();
        assert!(text_missing.is_none());
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

        with_env_var(
            "TRIPLETS_HF_PARQUET_ENDPOINT",
            &format!("{base_url}/parquet"),
            || {
                assert!(source.ensure_row_available(0).unwrap());
            },
        );

        server.join().unwrap();
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
}
