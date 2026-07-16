use crate::config::HuggingFaceRowsConfig;
use crate::disk_cache::StoreCache;
use crate::disk_cache::ensure_cache_group;
use crate::types::ShardIndex;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::reader::RowIter;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use crate::constants::{
    ENV_TRIPLETS_HF_TOKEN, HF_ALL_SPLITS_DIR, HF_GROUP, HF_LOCAL_DISK_CAP_BYTES,
    HF_PARQUET_DEFAULT_ENDPOINT, HF_REFRESH_BATCH_MULTIPLIER,
    HF_REMOTE_EXPANSION_HEADROOM_MULTIPLIER, HF_SHARD_STORE_EXTENSION,
};
use chrono::{DateTime, Utc};
use triplets_core::SamplerError;
use triplets_core::config::SamplerConfig;

pub(crate) const HF_SOURCE_KEY_ANCHOR: &str = "anchor";
pub(crate) const HF_SOURCE_KEY_POSITIVE: &str = "positive";
pub(crate) const HF_SOURCE_KEY_NEGATIVE: &str = "negative";
pub(crate) const HF_SOURCE_KEY_CONTEXT: &str = "context";
pub(crate) const HF_SOURCE_KEY_TEXT: &str = "text";
pub(crate) const HF_SOURCE_KEY_TEXT_COLUMNS: &str = "text_columns";
pub(crate) const HF_SOURCE_KEY_TRUST: &str = "trust";
pub(crate) const HF_SOURCE_KEY_WEIGHT: &str = "weight";
pub(crate) const HF_SOURCE_KEY_SOURCE_ID: &str = "source_id";

/// Default HF text-columns-mode SimCSE-style recipe name.
pub const HF_RECIPE_TEXT_SIMCSE_WRONG_ARTICLE: &str = "huggingface_text_simcse_wrong_article";
pub(crate) const HF_RECIPE_ANCHOR_CONTEXT_WRONG_ARTICLE: &str =
    "huggingface_anchor_context_wrong_article";
pub(crate) const HF_RECIPE_ANCHOR_ANCHOR_WRONG_ARTICLE: &str =
    "huggingface_anchor_anchor_wrong_article";

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
pub(crate) struct RowTextField {
    pub(crate) name: String,
    pub(crate) text: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct RowView {
    pub(crate) row_id: Option<String>,
    pub(crate) timestamp: Option<DateTime<Utc>>,
    pub(crate) text_fields: Vec<RowTextField>,
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

    pub(crate) fn has_explicit_mapping(&self) -> bool {
        !self.anchor_columns.is_empty()
            || !self.positive_columns.is_empty()
            || !self.negative_columns.is_empty()
            || !self.context_columns.is_empty()
            || !self.text_columns.is_empty()
    }
}

#[derive(Default)]
pub(crate) struct ParquetCache {
    pub(crate) readers: HashMap<PathBuf, Arc<SerializedFileReader<File>>>,
    pub(crate) row_groups: HashMap<(PathBuf, usize), Arc<Vec<Value>>>,
    pub(crate) row_group_order: VecDeque<(PathBuf, usize)>,
}

#[derive(Default)]
#[allow(dead_code)]
pub(crate) struct EligibleIndexCache {
    pub(crate) signature: Option<u64>,
    pub(crate) rows: Option<Arc<Vec<usize>>>,
    pub(crate) shards: Vec<ShardIndex>,
}

impl ParquetCache {
    /// Return a cached parquet reader for `path`, opening and caching it when missing.
    pub(crate) fn reader_for(
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

    pub(crate) fn row_group_rows_for(
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

    pub(crate) fn refresh_row_group_order(
        order: &mut VecDeque<(PathBuf, usize)>,
        key: &(PathBuf, usize),
    ) {
        if order.is_empty() {
            return;
        }
        if let Some(pos) = order.iter().position(|existing| existing == key) {
            order.remove(pos);
        }
    }
}

#[derive(Default)]
pub(crate) struct RowCache {
    pub(crate) rows: HashMap<usize, RowView>,
    pub(crate) order: VecDeque<usize>,
}

impl RowCache {
    /// Return a cloned cached row by absolute index.
    pub(crate) fn get(&self, idx: usize) -> Option<RowView> {
        self.rows.get(&idx).cloned()
    }

    /// Insert or refresh a cached row and evict oldest entries over `capacity`.
    pub(crate) fn insert(&mut self, idx: usize, row: RowView, capacity: usize) {
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

pub(crate) type ParquetGroupKey = (PathBuf, usize);
pub(crate) type ParquetGroupRequest = (usize, usize, ShardIndex);
pub(crate) type ParquetManifestCandidates = (Vec<String>, HashMap<String, u64>, usize);

/// Global gate that serializes expansion downloads across ALL HuggingFace
/// sources.  Only one source downloads a shard at any given time, preventing
/// bursts when multiple sources trigger expansion on the same cycle.
pub(crate) static EXPANSION_GATE: std::sync::OnceLock<std::sync::Mutex<()>> =
    std::sync::OnceLock::new();
