use hf_hub::Repo;
use hf_hub::RepoType;
use hf_hub::api::sync::ApiBuilder;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::reader::RowIter;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use std::time::Instant;
use walkdir::WalkDir;

use crate::SamplerError;
use crate::config::{NegativeStrategy, SamplerConfig, Selector, TripletRecipe};
use crate::data::{DataRecord, QualityScore, SectionRole};
use crate::utils::make_section;
use chrono::{DateTime, Utc};

use super::{DataSource, SourceCursor, SourceSnapshot};

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
const REMOTE_BOOTSTRAP_SHARDS: usize = 4;
/// Multiplies the source `refresh` limit passed by `IngestionManager`
/// (`step.unwrap_or(max_records)`) to set this source's internal row-read
/// batch target for each refresh pass.
const HUGGINGFACE_REFRESH_BATCH_MULTIPLIER: usize = 32;
const SHARD_SEQUENCE_STATE_VERSION: u32 = 1;
const SHARD_SEQUENCE_STATE_FILE: &str = "_sequence_state.json";

#[derive(Clone, Debug)]
struct RowTextField {
    name: String,
    text: String,
}

#[derive(Clone, Debug)]
struct RowView {
    row_id: Option<String>,
    timestamp: Option<DateTime<Utc>>,
    text_fields: Vec<RowTextField>,
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
    /// Optional maximum row cap exposed by the source.
    pub max_rows: Option<usize>,
    /// Hard cap for local manifest-shard cache bytes.
    ///
    /// When exceeded, oldest cached manifest shards are evicted.
    pub local_disk_cap_bytes: Option<u64>,
    /// Minimum number of manifest shards to keep resident during eviction.
    pub min_resident_shards: usize,
    /// Optional row id column name. Falls back to synthetic id when missing.
    pub id_column: Option<String>,
    /// Text columns to extract. Empty means auto-detect textual scalar columns.
    pub text_columns: Vec<String>,
    /// Optional column used for anchor text.
    ///
    /// When set (or when `positive_column`/`context_columns` are set), role-based
    /// extraction is used instead of `text_columns`/auto-detect mode.
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
                "jsonl".to_string(),
                "ndjson".to_string(),
            ],
            checkpoint_stride: 4096,
            cache_capacity: SamplerConfig::default().ingestion_max_records,
            parquet_row_group_cache_capacity: 8,
            refresh_batch_multiplier: HUGGINGFACE_REFRESH_BATCH_MULTIPLIER,
            remote_expansion_headroom_multiplier: REMOTE_EXPANSION_HEADROOM_MULTIPLIER,
            max_rows: None,
            local_disk_cap_bytes: Some(32 * 1024 * 1024 * 1024),
            min_resident_shards: REMOTE_BOOTSTRAP_SHARDS,
            id_column: Some("id".to_string()),
            text_columns: Vec::new(),
            anchor_column: None,
            positive_column: None,
            context_columns: Vec::new(),
        }
    }
}

#[derive(Default)]
struct ParquetCache {
    readers: HashMap<PathBuf, Arc<SerializedFileReader<File>>>,
}

impl ParquetCache {
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
}

#[derive(Clone, Debug)]
struct ShardIndex {
    path: PathBuf,
    global_start: usize,
    row_count: usize,
    is_parquet: bool,
    parquet_row_groups: Vec<(usize, usize)>,
    checkpoints: Vec<u64>,
}

#[derive(Default)]
struct RowCache {
    rows: HashMap<usize, RowView>,
    order: VecDeque<usize>,
}

impl RowCache {
    fn get(&self, idx: usize) -> Option<RowView> {
        self.rows.get(&idx).cloned()
    }

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
    sampler_config: Mutex<Option<SamplerConfig>>,
    state: Mutex<SourceState>,
    cache: Mutex<RowCache>,
    parquet_cache: Mutex<ParquetCache>,
}

#[derive(Debug)]
struct SourceState {
    materialized_rows: usize,
    total_rows: Option<usize>,
    shards: Vec<ShardIndex>,
    remote_candidates: Option<Vec<String>>,
    remote_candidate_sizes: HashMap<String, u64>,
    next_remote_idx: usize,
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
    #[serde(default)]
    sampler_seed: Option<u64>,
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

        fs::create_dir_all(&config.snapshot_dir).map_err(|err| {
            SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!(
                    "failed creating snapshot_dir {}: {err}",
                    config.snapshot_dir.display()
                ),
            }
        })?;

        eprintln!(
            "[triplets:hf] indexing local shards in {}",
            config.snapshot_dir.display()
        );
        let (shards, discovered) = Self::build_shard_index(&config).unwrap_or_default();
        if discovered == 0 {
            eprintln!(
                "[triplets:hf] no local shards found in {} — lazy remote download enabled",
                config.snapshot_dir.display()
            );
        }

        let materialized_rows = config
            .max_rows
            .map(|cap| cap.min(discovered))
            .unwrap_or(discovered);
        let total_rows = match Self::fetch_global_row_count(&config) {
            Ok(value) => value,
            Err(err) => {
                eprintln!(
                    "[triplets:hf] global row count request failed; continuing with discovered rows only: {}",
                    err
                );
                None
            }
        };

        if let Some(global_total) = total_rows {
            eprintln!(
                "[triplets:hf] global split row count reported: {} (known_local_rows={})",
                global_total, materialized_rows
            );
        }

        eprintln!(
            "[triplets:hf] source ready in {:.2}s (rows={}, shards={})",
            start_new.elapsed().as_secs_f64(),
            materialized_rows,
            shards.len()
        );

        Ok(Self {
            config,
            sampler_config: Mutex::new(None),
            state: Mutex::new(SourceState {
                materialized_rows,
                total_rows,
                shards,
                remote_candidates: None,
                remote_candidate_sizes: HashMap::new(),
                next_remote_idx: 0,
            }),
            cache: Mutex::new(RowCache::default()),
            parquet_cache: Mutex::new(ParquetCache::default()),
        })
    }

    fn effective_refresh_batch_target(&self, limit: usize) -> usize {
        let multiplier = self.config.refresh_batch_multiplier.max(1);
        limit.saturating_mul(multiplier)
    }

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

    fn list_remote_candidates(
        config: &HuggingFaceRowsConfig,
    ) -> Result<(Vec<String>, HashMap<String, u64>), SamplerError> {
        if let Ok((candidates, candidate_sizes)) =
            Self::list_remote_candidates_from_parquet_manifest(config)
            && !candidates.is_empty()
        {
            eprintln!(
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
        eprintln!(
            "[triplets:hf] reading remote file list for dataset {}",
            config.dataset
        );
        let info = repo_api
            .info()
            .map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!("failed reading hf-hub repository info: {err}"),
            })?;

        let accepted = config
            .shard_extensions
            .iter()
            .map(|value| value.trim().trim_start_matches('.').to_ascii_lowercase())
            .collect::<Vec<_>>();

        let siblings = info
            .siblings
            .into_iter()
            .map(|entry| entry.rfilename)
            .collect::<Vec<_>>();

        let collect_candidates = |respect_split: bool| {
            let mut saw_parquet = false;
            let mut candidates = Vec::new();
            for remote_path in &siblings {
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
                    if target.exists() {
                        continue;
                    }
                    candidates.push(remote_path.clone());
                }
            }
            (candidates, saw_parquet)
        };

        let (mut candidates, mut saw_parquet) = collect_candidates(true);
        if candidates.is_empty() && !config.split.is_empty() {
            let (fallback_candidates, fallback_saw_parquet) = collect_candidates(false);
            if !fallback_candidates.is_empty() {
                eprintln!(
                    "[triplets:hf] split filter '{}' matched no remote files; falling back to extension-only remote candidate scan",
                    config.split
                );
                candidates = fallback_candidates;
                saw_parquet = fallback_saw_parquet;
            }
        }

        candidates.sort();
        eprintln!(
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
            eprintln!(
                "[triplets:hf] no remote candidates found for dataset='{}' split='{}' extensions={:?}; source will be treated as exhausted",
                config.dataset, config.split, config.shard_extensions
            );
            return Ok((Vec::new(), HashMap::new()));
        }

        Ok((candidates, HashMap::new()))
    }

    fn shard_sequence_state_path(config: &HuggingFaceRowsConfig) -> PathBuf {
        config
            .snapshot_dir
            .join("_parquet_manifest")
            .join(SHARD_SEQUENCE_STATE_FILE)
    }

    fn load_persisted_shard_sequence(
        config: &HuggingFaceRowsConfig,
        current_sampler_seed: Option<u64>,
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
            eprintln!(
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
            sampler_seed: self
                .sampler_config
                .lock()
                .ok()
                .and_then(|config| config.as_ref().map(|value| value.seed)),
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

    fn rotate_candidates_deterministically(
        config: &HuggingFaceRowsConfig,
        candidates: &mut [String],
    ) {
        if candidates.len() <= 1 {
            return;
        }
        let mut hasher = DefaultHasher::new();
        config.source_id.hash(&mut hasher);
        config.dataset.hash(&mut hasher);
        config.config.hash(&mut hasher);
        config.split.hash(&mut hasher);
        let offset = (hasher.finish() as usize) % candidates.len();
        candidates.rotate_left(offset);
    }

    fn shard_candidate_seed(
        config: &HuggingFaceRowsConfig,
        total_candidates: usize,
        sampler_seed: Option<u64>,
    ) -> u64 {
        let mut hasher = DefaultHasher::new();
        "hf_shard_candidate_sequence_v1".hash(&mut hasher);
        if let Some(seed) = sampler_seed {
            seed.hash(&mut hasher);
        } else {
            config.source_id.hash(&mut hasher);
            config.dataset.hash(&mut hasher);
            config.config.hash(&mut hasher);
            config.split.hash(&mut hasher);
        }
        total_candidates.hash(&mut hasher);
        hasher.finish()
    }

    fn list_remote_candidates_from_parquet_manifest(
        config: &HuggingFaceRowsConfig,
    ) -> Result<(Vec<String>, HashMap<String, u64>), SamplerError> {
        let endpoint = "https://datasets-server.huggingface.co/parquet";
        eprintln!(
            "[triplets:hf] reading datasets-server parquet manifest for dataset {}",
            config.dataset
        );
        let response = ureq::get(endpoint)
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

        let json: Value =
            serde_json::from_str(&body).map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!("failed parsing datasets-server parquet response: {err}"),
            })?;

        let accepted = config
            .shard_extensions
            .iter()
            .map(|value| value.trim().trim_start_matches('.').to_ascii_lowercase())
            .collect::<Vec<_>>();

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
                if target.exists() {
                    if Self::target_matches_expected_size(&target, expected_size) {
                        continue;
                    }
                    eprintln!(
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
        Ok((candidates, candidate_sizes))
    }

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

    fn manifest_cache_root(&self) -> PathBuf {
        self.config.snapshot_dir.join("_parquet_manifest")
    }

    fn shard_size_bytes(path: &Path) -> u64 {
        fs::metadata(path).map(|meta| meta.len()).unwrap_or(0)
    }

    fn recompute_shard_offsets(state: &mut SourceState) {
        let mut running = 0usize;
        for shard in &mut state.shards {
            shard.global_start = running;
            running = running.saturating_add(shard.row_count);
        }
        state.materialized_rows = running;
    }

    fn enforce_disk_cap_locked(
        &self,
        state: &mut SourceState,
        protected_path: &Path,
    ) -> Result<bool, SamplerError> {
        let Some(cap_bytes) = self.config.local_disk_cap_bytes else {
            return Ok(false);
        };

        let manifest_root = self.manifest_cache_root();
        let mut usage_bytes = state
            .shards
            .iter()
            .filter(|shard| shard.path.starts_with(&manifest_root))
            .map(|shard| Self::shard_size_bytes(&shard.path))
            .sum::<u64>();

        if usage_bytes <= cap_bytes {
            return Ok(false);
        }

        let mut evicted_any = false;
        loop {
            if usage_bytes <= cap_bytes {
                break;
            }

            let resident_manifest_count = state
                .shards
                .iter()
                .filter(|shard| shard.path.starts_with(&manifest_root))
                .count();
            if resident_manifest_count <= self.config.min_resident_shards {
                break;
            }

            let evict_pos = state.shards.iter().position(|shard| {
                shard.path.starts_with(&manifest_root) && shard.path != protected_path
            });
            let Some(pos) = evict_pos else {
                break;
            };

            let shard = state.shards.remove(pos);
            let shard_size = Self::shard_size_bytes(&shard.path);
            if let Err(err) = fs::remove_file(&shard.path)
                && err.kind() != std::io::ErrorKind::NotFound
            {
                return Err(SamplerError::SourceUnavailable {
                    source_id: self.config.source_id.clone(),
                    reason: format!(
                        "failed evicting shard {} under disk cap: {err}",
                        shard.path.display()
                    ),
                });
            }

            usage_bytes = usage_bytes.saturating_sub(shard_size);
            evicted_any = true;
            eprintln!(
                "[triplets:hf] evicted shard for disk cap: {} (usage={:.2} GiB cap={:.2} GiB)",
                shard.path.display(),
                usage_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                cap_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            );
        }

        if usage_bytes > cap_bytes {
            if protected_path.exists() {
                let _ = fs::remove_file(protected_path);
            }
            return Err(SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: format!(
                    "local disk cap exceeded and cannot evict further (usage={} bytes cap={} bytes)",
                    usage_bytes, cap_bytes
                ),
            });
        }

        if evicted_any {
            Self::recompute_shard_offsets(state);
        }
        Ok(evicted_any)
    }

    fn manifest_usage_bytes_locked(&self, state: &SourceState) -> u64 {
        let manifest_root = self.manifest_cache_root();
        state
            .shards
            .iter()
            .filter(|shard| shard.path.starts_with(&manifest_root))
            .map(|shard| Self::shard_size_bytes(&shard.path))
            .sum::<u64>()
    }

    fn fetch_global_row_count(
        config: &HuggingFaceRowsConfig,
    ) -> Result<Option<usize>, SamplerError> {
        let endpoint = "https://datasets-server.huggingface.co/size";
        eprintln!(
            "[triplets:hf] requesting global row count dataset='{}' config='{}' split='{}'",
            config.dataset, config.config, config.split
        );

        let response = ureq::get(endpoint)
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

        let json: Value =
            serde_json::from_str(&body).map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!("failed parsing datasets-server size response: {err}"),
            })?;

        let mut count =
            Self::extract_split_row_count_from_size_response(&json, &config.config, &config.split);
        if let (Some(max_rows), Some(rows)) = (config.max_rows, count) {
            count = Some(rows.min(max_rows));
        }
        Ok(count)
    }

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

    fn download_and_materialize_shard(
        config: &HuggingFaceRowsConfig,
        remote_path: &str,
        expected_bytes: Option<u64>,
    ) -> Result<PathBuf, SamplerError> {
        if let Some(remote_url) = remote_path.strip_prefix(REMOTE_URL_PREFIX) {
            let target = Self::candidate_target_path(config, remote_path);
            if target.exists() {
                if Self::target_matches_expected_size(&target, expected_bytes) {
                    return Ok(target);
                }
                eprintln!(
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

            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent).map_err(|err| SamplerError::SourceUnavailable {
                    source_id: config.source_id.clone(),
                    reason: format!(
                        "failed creating snapshot subdir {}: {err}",
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
            eprintln!(
                "[triplets:hf] downloading shard payload -> {}",
                target.display()
            );
            let started = Instant::now();
            let mut total_bytes = 0u64;
            let mut buffer = vec![0u8; 8 * 1024 * 1024];
            let mut last_report = Instant::now();
            loop {
                let read =
                    reader
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
                        eprintln!(
                            "[triplets:hf] download progress {}: {:.1}/{:.1} MiB ({:.1}%, {:.1}s elapsed, ETA {:.1}s)",
                            target.display(),
                            total_bytes as f64 / (1024.0 * 1024.0),
                            expected as f64 / (1024.0 * 1024.0),
                            pct,
                            elapsed,
                            eta_secs.max(0.0)
                        );
                    } else {
                        eprintln!(
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
                eprintln!(
                    "[triplets:hf] download complete {}: {:.1}/{:.1} MiB ({:.1}%) in {:.1}s",
                    target.display(),
                    total_bytes as f64 / (1024.0 * 1024.0),
                    expected as f64 / (1024.0 * 1024.0),
                    pct,
                    elapsed
                );
            } else {
                eprintln!(
                    "[triplets:hf] download complete {}: {:.1} MiB in {:.1}s",
                    target.display(),
                    total_bytes as f64 / (1024.0 * 1024.0),
                    elapsed
                );
            }

            fs::rename(&temp_target, &target).map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!(
                    "failed moving downloaded shard {} -> {}: {err}",
                    temp_target.display(),
                    target.display()
                ),
            })?;
            return Ok(target);
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
        Self::materialize_local_file(config, &local_cached, &target)?;
        Ok(target)
    }

    fn index_single_shard(
        config: &HuggingFaceRowsConfig,
        path: &Path,
        global_start: usize,
    ) -> Result<Option<ShardIndex>, SamplerError> {
        let is_parquet = path
            .extension()
            .and_then(|v| v.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("parquet"));

        let (rows, parquet_row_groups, checkpoints) = if is_parquet {
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
            is_parquet,
            parquet_row_groups,
            checkpoints,
        }))
    }

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

                if self.config.max_rows.is_some_and(|max_rows| idx >= max_rows) {
                    return Ok(false);
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
                    let sampler_seed = self
                        .sampler_config
                        .lock()
                        .ok()
                        .and_then(|config| config.as_ref().map(|value| value.seed));
                    if let Some(persisted) =
                        Self::load_persisted_shard_sequence(&self.config, sampler_seed)?
                    {
                        let candidate_count = persisted.candidates.len();
                        state.remote_candidates = Some(persisted.candidates);
                        state.remote_candidate_sizes = persisted.candidate_sizes;
                        state.next_remote_idx = persisted.next_remote_idx.min(candidate_count);
                        eprintln!(
                            "[triplets:hf] resumed shard sequence state: next={}/{}",
                            state.next_remote_idx, candidate_count
                        );
                    } else {
                        let (mut candidates, candidate_sizes) =
                            Self::list_remote_candidates(&self.config)?;
                        Self::rotate_candidates_deterministically(&self.config, &mut candidates);
                        state.remote_candidates = Some(candidates);
                        state.remote_candidate_sizes = candidate_sizes;
                        state.next_remote_idx = 0;
                    }

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
                    eprintln!(
                        "[triplets:hf] state: candidates={} known_rows={} active_shards={} disk_cap={} min_resident_shards={}",
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
                        self.config.min_resident_shards,
                    );
                    drop(state);

                    if bootstrap_needed {
                        let bootstrap_target = REMOTE_BOOTSTRAP_SHARDS.min(candidate_count);
                        eprintln!(
                            "[triplets:hf] bootstrapping remote shard diversity: target={} shard(s)",
                            bootstrap_target
                        );
                        for step in 0..bootstrap_target {
                            eprintln!(
                                "[triplets:hf] bootstrap progress: {}/{}",
                                step + 1,
                                bootstrap_target
                            );
                            if !self.download_next_remote_shard()? {
                                break;
                            }
                        }
                        eprintln!("[triplets:hf] bootstrap complete");
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

    fn download_next_remote_shard(&self) -> Result<bool, SamplerError> {
        let (remote_ordinal, remote_total, remote_path, expected_bytes) = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| SamplerError::SourceUnavailable {
                    source_id: self.config.source_id.clone(),
                    reason: "huggingface source state lock poisoned".to_string(),
                })?;
            let Some(candidates) = &state.remote_candidates else {
                return Ok(false);
            };
            if state.next_remote_idx >= candidates.len() {
                return Ok(false);
            }
            let sequence_pos = state.next_remote_idx;
            let remote_ordinal = sequence_pos + 1;
            let remote_total = candidates.len();
            let sampler_seed = self
                .sampler_config
                .lock()
                .ok()
                .and_then(|config| config.as_ref().map(|value| value.seed));
            let seed = Self::shard_candidate_seed(&self.config, remote_total, sampler_seed);
            let mut permutation =
                super::IndexPermutation::new(remote_total, seed, sequence_pos as u64);
            let candidate_idx = permutation.next();
            let remote_path = candidates[candidate_idx].clone();
            let expected_bytes = state.remote_candidate_sizes.get(&remote_path).copied();
            state.next_remote_idx += 1;
            (remote_ordinal, remote_total, remote_path, expected_bytes)
        };

        eprintln!(
            "[triplets:hf] lazy downloading shard {}/{}: {}",
            remote_ordinal, remote_total, remote_path
        );
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
            eprintln!(
                "[triplets:hf] downloaded shard had zero rows and was skipped: {}",
                local_path.display()
            );
            return Ok(true);
        };

        let mut state = self
            .state
            .lock()
            .map_err(|_| SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: "huggingface source state lock poisoned".to_string(),
            })?;

        if self
            .config
            .max_rows
            .is_some_and(|max_rows| state.materialized_rows >= max_rows)
        {
            return Ok(true);
        }

        let mut rows_to_add = shard.row_count;
        if let Some(max_rows) = self.config.max_rows {
            rows_to_add = rows_to_add.min(max_rows.saturating_sub(state.materialized_rows));
        }
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
        state.materialized_rows += rows_to_add;
        state.shards.push(shard);

        let evicted_any = self.enforce_disk_cap_locked(&mut state, &local_path)?;
        self.persist_shard_sequence_locked(&state)?;
        let materialized_rows = state.materialized_rows;
        let shard_count = state.shards.len();
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

        if evicted_any {
            if let Ok(mut cache) = self.cache.lock() {
                cache.rows.clear();
                cache.order.clear();
            }
            if let Ok(mut parquet_cache) = self.parquet_cache.lock() {
                parquet_cache.readers.clear();
            }
        }

        eprintln!(
            "[triplets:hf] state: rows={} shards={} remaining_candidates={} disk_usage={:.2} GiB cap={}",
            materialized_rows, shard_count, remaining_candidates, usage_gib, cap_str,
        );

        Ok(true)
    }

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

    fn build_shard_index(
        config: &HuggingFaceRowsConfig,
    ) -> Result<(Vec<ShardIndex>, usize), SamplerError> {
        let start_index = Instant::now();
        let mut shard_paths = Vec::new();
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
            let Some(ext) = entry.path().extension().and_then(|v| v.to_str()) else {
                continue;
            };
            if ext.eq_ignore_ascii_case("parquet") {
                saw_parquet = true;
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
            if saw_parquet && !accepted.iter().any(|value| value == "parquet") {
                return Err(SamplerError::SourceUnavailable {
                    source_id: config.source_id.clone(),
                    reason: format!(
                        "found parquet files under {}, but shard_extensions does not include parquet.",
                        config.snapshot_dir.display()
                    ),
                });
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

        let mut shards = Vec::new();
        let mut running_total = 0usize;
        for (ordinal, path) in shard_paths.into_iter().enumerate() {
            if let Some(max_rows) = config.max_rows
                && running_total >= max_rows
            {
                break;
            }

            eprintln!(
                "[triplets:hf] indexing shard {}: {}",
                ordinal + 1,
                path.display()
            );

            let is_parquet = path
                .extension()
                .and_then(|v| v.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("parquet"));

            let (rows, parquet_row_groups, checkpoints) = if is_parquet {
                let (mut rows, mut parquet_row_groups) =
                    Self::parquet_row_group_map(config, &path)?;
                if let Some(max_rows) = config.max_rows {
                    rows = rows.min(max_rows.saturating_sub(running_total));
                    let covered = parquet_row_groups
                        .last()
                        .map(|(start, count)| start.saturating_add(*count))
                        .unwrap_or(0);
                    if covered > rows {
                        parquet_row_groups.retain(|(start, _)| *start < rows);
                        if let Some((start, count)) = parquet_row_groups.last_mut() {
                            let allowed = rows.saturating_sub(*start);
                            *count = (*count).min(allowed);
                        }
                    }
                }
                (rows, parquet_row_groups, Vec::new())
            } else {
                let file = File::open(&path).map_err(|err| SamplerError::SourceUnavailable {
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
                    let bytes = reader.read_line(&mut line).map_err(|err| {
                        SamplerError::SourceUnavailable {
                            source_id: config.source_id.clone(),
                            reason: format!("failed reading shard {}: {err}", path.display()),
                        }
                    })?;
                    if bytes == 0 {
                        break;
                    }
                    rows += 1;
                    offset = offset.saturating_add(bytes as u64);

                    if let Some(max_rows) = config.max_rows
                        && running_total + rows >= max_rows
                    {
                        break;
                    }
                }
                (rows, Vec::new(), checkpoints)
            };

            if rows == 0 {
                continue;
            }

            shards.push(ShardIndex {
                path,
                global_start: running_total,
                row_count: rows,
                is_parquet,
                parquet_row_groups,
                checkpoints,
            });
            running_total += rows;
        }

        eprintln!(
            "[triplets:hf] indexing complete in {:.2}s (rows={}, shards={})",
            start_index.elapsed().as_secs_f64(),
            running_total,
            shards.len()
        );

        Ok((shards, running_total))
    }

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

    fn parse_row(&self, absolute_idx: usize, row_value: &Value) -> Result<RowView, SamplerError> {
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
                let value = row_obj
                    .get(name)
                    .ok_or_else(|| SamplerError::SourceInconsistent {
                        source_id: self.config.source_id.clone(),
                        details: format!("missing configured anchor column '{name}'"),
                    })?;
                let text =
                    Self::value_to_text(value).ok_or_else(|| SamplerError::SourceInconsistent {
                        source_id: self.config.source_id.clone(),
                        details: format!("configured anchor column '{name}' has null/empty value"),
                    })?;
                text_fields.push(RowTextField {
                    name: name.clone(),
                    text,
                });
            }

            if let Some(name) = &self.config.positive_column {
                let value = row_obj
                    .get(name)
                    .ok_or_else(|| SamplerError::SourceInconsistent {
                        source_id: self.config.source_id.clone(),
                        details: format!("missing configured positive column '{name}'"),
                    })?;
                let text =
                    Self::value_to_text(value).ok_or_else(|| SamplerError::SourceInconsistent {
                        source_id: self.config.source_id.clone(),
                        details: format!(
                            "configured positive column '{name}' has null/empty value"
                        ),
                    })?;
                text_fields.push(RowTextField {
                    name: name.clone(),
                    text,
                });
            }

            for name in &self.config.context_columns {
                let value = row_obj
                    .get(name)
                    .ok_or_else(|| SamplerError::SourceInconsistent {
                        source_id: self.config.source_id.clone(),
                        details: format!("missing configured context column '{name}'"),
                    })?;
                let text =
                    Self::value_to_text(value).ok_or_else(|| SamplerError::SourceInconsistent {
                        source_id: self.config.source_id.clone(),
                        details: format!("configured context column '{name}' has null/empty value"),
                    })?;
                text_fields.push(RowTextField {
                    name: name.clone(),
                    text,
                });
            }
        } else if self.config.text_columns.is_empty() {
            for (name, value) in row_obj {
                if self.config.id_column.as_ref().is_some_and(|id| id == name) {
                    continue;
                }
                if let Some(text) = Self::value_to_text(value) {
                    text_fields.push(RowTextField {
                        name: name.clone(),
                        text,
                    });
                }
            }
        } else {
            for name in &self.config.text_columns {
                let value = row_obj
                    .get(name)
                    .ok_or_else(|| SamplerError::SourceInconsistent {
                        source_id: self.config.source_id.clone(),
                        details: format!("missing configured text column '{name}'"),
                    })?;
                let text =
                    Self::value_to_text(value).ok_or_else(|| SamplerError::SourceInconsistent {
                        source_id: self.config.source_id.clone(),
                        details: format!("configured text column '{name}' has null/empty value"),
                    })?;
                text_fields.push(RowTextField {
                    name: name.clone(),
                    text,
                });
            }
        }

        if text_fields.is_empty() {
            return Err(SamplerError::SourceInconsistent {
                source_id: self.config.source_id.clone(),
                details: "row resolved to zero text fields".to_string(),
            });
        }

        Ok(RowView {
            row_id: Some(row_id),
            timestamp: None,
            text_fields,
        })
    }

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
                let row_value = serde_json::from_str::<Value>(line.trim()).map_err(|err| {
                    SamplerError::SourceInconsistent {
                        source_id: self.config.source_id.clone(),
                        details: format!(
                            "failed decoding JSON row from shard {} at local index {}: {err}",
                            shard.path.display(),
                            local_idx
                        ),
                    }
                })?;
                let row = self.parse_row(idx, &row_value)?;
                let record = self.row_to_record(&row, idx as u64)?;
                self.cache
                    .lock()
                    .map_err(|_| SamplerError::SourceUnavailable {
                        source_id: self.config.source_id.clone(),
                        reason: "huggingface row cache lock poisoned".to_string(),
                    })?
                    .insert(idx, row, self.config.cache_capacity);
                fetched.insert(idx, record);
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
                let max_target = targets.keys().next_back().copied().unwrap_or(0);

                let reader = self
                    .parquet_cache
                    .lock()
                    .map_err(|_| SamplerError::SourceUnavailable {
                        source_id: self.config.source_id.clone(),
                        reason: "huggingface parquet cache lock poisoned".to_string(),
                    })?
                    .reader_for(&self.config.source_id, &shard.path)?;

                let row_group = reader.get_row_group(group_pos).map_err(|err| {
                    SamplerError::SourceUnavailable {
                        source_id: self.config.source_id.clone(),
                        reason: format!(
                            "failed opening parquet row group {} for {}: {err}",
                            group_pos,
                            shard.path.display()
                        ),
                    }
                })?;
                let iter = RowIter::from_row_group(None, row_group.as_ref()).map_err(|err| {
                    SamplerError::SourceUnavailable {
                        source_id: self.config.source_id.clone(),
                        reason: format!(
                            "failed iterating parquet row group {} for {}: {err}",
                            group_pos,
                            shard.path.display()
                        ),
                    }
                })?;

                for (position, row_result) in iter.enumerate() {
                    if position > max_target {
                        break;
                    }
                    let Some(indices_for_position) = targets.remove(&position) else {
                        continue;
                    };
                    let row_value = row_result.map_err(|err| SamplerError::SourceUnavailable {
                        source_id: self.config.source_id.clone(),
                        reason: format!(
                            "failed reading parquet row {} in shard {} row_group {}: {err}",
                            position,
                            shard.path.display(),
                            group_pos
                        ),
                    })?;
                    let row_value = row_value.to_json_value();

                    for idx in indices_for_position {
                        let row = self.parse_row(idx, &row_value)?;
                        let record = self.row_to_record(&row, idx as u64)?;
                        self.cache
                            .lock()
                            .map_err(|_| SamplerError::SourceUnavailable {
                                source_id: self.config.source_id.clone(),
                                reason: "huggingface row cache lock poisoned".to_string(),
                            })?
                            .insert(idx, row, self.config.cache_capacity);
                        fetched.insert(idx, record);
                    }

                    if targets.is_empty() {
                        break;
                    }
                }

                if !targets.is_empty() {
                    let missing = targets
                        .into_keys()
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

    fn len_hint(&self) -> Option<usize> {
        let state = self.state.lock().ok()?;
        let known = state.materialized_rows;
        if known > 0 {
            let mut upper = known;
            if state.total_rows.is_some_and(|total_rows| total_rows > known) {
                let headroom = self.effective_expansion_headroom_rows();
                upper = known.saturating_add(headroom);
                if let Some(total_rows) = state.total_rows {
                    upper = upper.min(total_rows);
                }
            }
            if let Some(max_rows) = self.config.max_rows {
                upper = upper.min(max_rows);
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

        if self.config.max_rows.is_some_and(|max_rows| max_rows == 0) {
            return Some(0);
        }

        Some(1)
    }
}

impl DataSource for HuggingFaceRowSource {
    fn id(&self) -> &str {
        &self.config.source_id
    }

    fn configure_sampler(&self, config: &SamplerConfig) {
        if let Ok(mut slot) = self.sampler_config.lock() {
            *slot = Some(config.clone());
        }
    }

    fn refresh(
        &self,
        cursor: Option<&SourceCursor>,
        limit: Option<usize>,
    ) -> Result<SourceSnapshot, SamplerError> {
        let total = self
            .len_hint()
            .ok_or_else(|| SamplerError::SourceInconsistent {
                source_id: self.config.source_id.clone(),
                details: "huggingface source did not provide len_hint".to_string(),
            })?;

        if total == 0 {
            return Ok(SourceSnapshot {
                records: Vec::new(),
                cursor: SourceCursor {
                    last_seen: Utc::now(),
                    revision: 0,
                },
            });
        }

        let max = limit.unwrap_or(total);
        let mut start = cursor.map(|state| state.revision as usize).unwrap_or(0);
        if start >= total {
            start = 0;
        }

        let source_id = self.config.source_id.clone();
        let seed = super::IndexablePager::seed_for(&source_id, total);
        let mut permutation = super::IndexPermutation::new(total, seed, start as u64);

        let mut records = Vec::new();
        let read_batch_target = self.effective_refresh_batch_target(max);
        let mut pending_indices = Vec::with_capacity(read_batch_target);
        let should_report = total >= 10_000 || max >= 1_024;
        let report_every = Duration::from_millis(750);
        let refresh_start = Instant::now();
        let mut last_report = refresh_start;
        let mut attempts = 0usize;

        if should_report {
            eprintln!(
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
                eprintln!(
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
                eprintln!(
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
            eprintln!(
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

        Ok(SourceSnapshot {
            records,
            cursor: SourceCursor {
                last_seen,
                revision: next_start as u64,
            },
        })
    }

    fn reported_record_count(&self) -> Result<u128, SamplerError> {
        self.len_hint()
            .map(|count| count as u128)
            .ok_or_else(|| SamplerError::SourceInconsistent {
                source_id: self.config.source_id.clone(),
                details: "huggingface source did not provide len_hint".to_string(),
            })
    }

    fn default_triplet_recipes(&self) -> Vec<TripletRecipe> {
        vec![TripletRecipe {
            name: "huggingface_anchor_context".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Role(SectionRole::Context),
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;
    use tempfile::tempdir;

    fn test_config(snapshot_dir: PathBuf) -> HuggingFaceRowsConfig {
        let mut config = HuggingFaceRowsConfig::new(
            "hf_test",
            "org/dataset",
            "default",
            "train",
            snapshot_dir,
        );
        config.cache_capacity = 10;
        config.remote_expansion_headroom_multiplier = 3;
        config
    }

    fn test_source(config: HuggingFaceRowsConfig) -> HuggingFaceRowSource {
        HuggingFaceRowSource {
            config,
            sampler_config: Mutex::new(None),
            state: Mutex::new(SourceState {
                materialized_rows: 0,
                total_rows: None,
                shards: Vec::new(),
                remote_candidates: None,
                remote_candidate_sizes: HashMap::new(),
                next_remote_idx: 0,
            }),
            cache: Mutex::new(RowCache::default()),
            parquet_cache: Mutex::new(ParquetCache::default()),
        }
    }

    #[test]
    fn candidate_target_path_maps_remote_urls_under_manifest_root() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let candidate = "url::https://huggingface.co/datasets/org/ds/resolve/main/train/part-000.parquet";
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

        assert!(HuggingFaceRowSource::target_matches_expected_size(&path, Some(5)));
        assert!(!HuggingFaceRowSource::target_matches_expected_size(&path, Some(4)));
        assert!(HuggingFaceRowSource::target_matches_expected_size(&path, None));
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
            &payload,
            "default",
            "test",
        );
        assert_eq!(split_count, Some(77));

        let empty_split_count = HuggingFaceRowSource::extract_split_row_count_from_size_response(
            &payload,
            "default",
            "",
        );
        assert_eq!(empty_split_count, Some(200));
    }

    #[test]
    fn shard_candidate_seed_uses_sampler_seed_when_provided() {
        let dir = tempdir().unwrap();
        let mut a = test_config(dir.path().join("a"));
        let mut b = test_config(dir.path().join("b"));
        a.source_id = "source_a".to_string();
        b.source_id = "source_b".to_string();

        let with_seed_a = HuggingFaceRowSource::shard_candidate_seed(&a, 100, Some(42));
        let with_seed_b = HuggingFaceRowSource::shard_candidate_seed(&b, 100, Some(42));
        assert_eq!(with_seed_a, with_seed_b);

        let without_seed_a = HuggingFaceRowSource::shard_candidate_seed(&a, 100, None);
        let without_seed_b = HuggingFaceRowSource::shard_candidate_seed(&b, 100, None);
        assert_ne!(without_seed_a, without_seed_b);
    }

    #[test]
    fn expansion_headroom_uses_sampler_ingestion_max_records_when_configured() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);

        assert_eq!(source.effective_expansion_headroom_rows(), 30);

        let mut sampler = SamplerConfig::default();
        sampler.ingestion_max_records = 7;
        source.configure_sampler(&sampler);
        assert_eq!(source.effective_expansion_headroom_rows(), 21);
    }

    #[test]
    fn persisted_shard_sequence_roundtrip_respects_sampler_seed() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        {
            let mut sampler = SamplerConfig::default();
            sampler.seed = 4242;
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
        };
        state.remote_candidate_sizes.insert(
            "url::https://x/resolve/main/train/000.parquet".to_string(),
            10,
        );

        source.persist_shard_sequence_locked(&state).unwrap();

        let restored =
            HuggingFaceRowSource::load_persisted_shard_sequence(&config, Some(4242)).unwrap();
        assert!(restored.is_some());
        let restored = restored.unwrap();
        assert_eq!(restored.next_remote_idx, 1);
        assert_eq!(restored.candidates.len(), 2);

        let rejected =
            HuggingFaceRowSource::load_persisted_shard_sequence(&config, Some(9999)).unwrap();
        assert!(rejected.is_none());
    }

    #[test]
    fn value_to_text_handles_scalar_and_structured_values() {
        assert_eq!(HuggingFaceRowSource::value_to_text(&json!(null)), None);
        assert_eq!(HuggingFaceRowSource::value_to_text(&json!("   ")), None);
        assert_eq!(HuggingFaceRowSource::value_to_text(&json!("hello")), Some("hello".into()));
        assert_eq!(HuggingFaceRowSource::value_to_text(&json!(true)), Some("true".into()));
        assert_eq!(HuggingFaceRowSource::value_to_text(&json!(3.5)), Some("3.5".into()));
        assert_eq!(
            HuggingFaceRowSource::value_to_text(&json!([1, 2])),
            Some("[1,2]".into())
        );
    }

    #[test]
    fn parse_row_auto_detects_text_fields_and_skips_id() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.id_column = Some("id".into());
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
            .unwrap();

        assert_eq!(row.row_id.as_deref(), Some("row-5"));
        assert!(row.text_fields.iter().any(|f| f.name == "title"));
        assert!(row.text_fields.iter().any(|f| f.name == "body"));
        assert!(row.text_fields.iter().all(|f| f.name != "id"));
    }

    #[test]
    fn parse_row_with_required_columns_errors_when_missing() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.anchor_column = Some("anchor".into());
        config.positive_column = Some("positive".into());
        config.context_columns = vec!["context".into()];
        let source = test_source(config);

        let err = source.parse_row(0, &json!({"anchor": "x", "context": "z"}));
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
            },
            ShardIndex {
                path: PathBuf::from("b"),
                global_start: 20,
                row_count: 2,
                is_parquet: false,
                parquet_row_groups: Vec::new(),
                checkpoints: vec![0],
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
        };
        HuggingFaceRowSource::recompute_shard_offsets(&mut state);
        assert_eq!(state.shards[0].global_start, 0);
        assert_eq!(state.shards[1].global_start, 3);
        assert_eq!(state.materialized_rows, 5);
    }

    #[test]
    fn len_hint_covers_known_and_empty_paths() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.max_rows = Some(9);
        let source = test_source(config);

        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 5;
            state.total_rows = Some(100);
        }
        assert_eq!(source.len_hint(), Some(9));

        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 0;
            state.total_rows = Some(0);
        }
        assert_eq!(source.len_hint(), Some(0));
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
        config.min_resident_shards = 0;
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
                },
                ShardIndex {
                    path: keep_path.clone(),
                    global_start: 8,
                    row_count: 8,
                    is_parquet: true,
                    parquet_row_groups: vec![(0, 8)],
                    checkpoints: Vec::new(),
                },
            ],
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
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
    fn enforce_disk_cap_errors_when_min_resident_prevents_eviction() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.local_disk_cap_bytes = Some(4);
        config.min_resident_shards = 1;
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
            }],
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
        };

        let err = source.enforce_disk_cap_locked(&mut state, &protected);
        assert!(err.is_err());
        assert!(!protected.exists());
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
    fn rotate_candidates_deterministically_preserves_membership() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let original = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut rotated = original.clone();
        HuggingFaceRowSource::rotate_candidates_deterministically(&config, &mut rotated);
        let mut sorted_original = original;
        let mut sorted_rotated = rotated;
        sorted_original.sort();
        sorted_rotated.sort();
        assert_eq!(sorted_rotated, sorted_original);
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
            .unwrap();

        assert_eq!(parsed.row_id.as_deref(), Some("r-1"));
        assert_eq!(parsed.text_fields.len(), 2);
        assert_eq!(parsed.text_fields[0].name, "headline");
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
        let mut config = test_config(dir.path().to_path_buf());
        config.max_rows = Some(2);
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
    fn build_shard_index_honors_max_rows() {
        let dir = tempdir().unwrap();
        fs::write(
            dir.path().join("rows.jsonl"),
            b"{\"text\":\"1\"}\n{\"text\":\"2\"}\n{\"text\":\"3\"}\n",
        )
        .unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.max_rows = Some(2);

        let (_, discovered) = HuggingFaceRowSource::build_shard_index(&config).unwrap();
        assert_eq!(discovered, 2);
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
}
