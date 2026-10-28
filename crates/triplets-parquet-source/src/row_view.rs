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

use chrono::{DateTime, Utc};
use triplets_core::SamplerError;

/// One named text field extracted from a row (column name + text).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RowTextField {
    /// Column (or dotted path) this text came from.
    pub name: String,
    /// Extracted non-empty text.
    pub text: String,
}

/// Normalized view of one dataset row: id plus extracted text fields.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RowView {
    /// Stable row id (explicit column or synthetic fallback).
    pub row_id: Option<String>,
    /// Row timestamp when the source provides one.
    pub timestamp: Option<DateTime<Utc>>,
    /// Extracted text fields in mapping order.
    pub text_fields: Vec<RowTextField>,
}

/// Cache of opened parquet readers and decoded row groups.
#[derive(Default)]
pub struct ParquetCache {
    /// Open file readers by shard path.
    pub readers: HashMap<PathBuf, Arc<SerializedFileReader<File>>>,
    /// Decoded row-group rows by `(path, group)` with LRU order below.
    pub row_groups: HashMap<(PathBuf, usize), Arc<Vec<Value>>>,
    /// LRU order for `row_groups` eviction.
    pub row_group_order: VecDeque<(PathBuf, usize)>,
}

/// Cached eligible-row index keyed by shard-layout signature.
#[derive(Default)]
#[allow(dead_code)]
pub struct EligibleIndexCache {
    /// Shard-layout signature this index was built for.
    pub signature: Option<u64>,
    /// Eligible absolute row indices.
    pub rows: Option<Arc<Vec<usize>>>,
    /// Shard layout the cached rows were derived from.
    pub shards: Vec<ShardIndex>,
}

impl ParquetCache {
    /// Return a cached parquet reader for `path`, opening and caching it when missing.
    pub fn reader_for(
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

    /// Return decoded rows for a parquet row group, caching up to capacity.
    pub fn row_group_rows_for(
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

    /// Refresh LRU order for a row-group key (most-recently-used last).
    pub fn refresh_row_group_order(order: &mut VecDeque<(PathBuf, usize)>, key: &(PathBuf, usize)) {
        if order.is_empty() {
            return;
        }
        if let Some(pos) = order.iter().position(|existing| existing == key) {
            order.remove(pos);
        }
    }
}

/// Bounded decoded-row cache keyed by absolute row index.
#[derive(Default)]
pub struct RowCache {
    /// Cached rows by absolute index.
    pub rows: HashMap<usize, RowView>,
    /// Insertion order for oldest-first eviction.
    pub order: VecDeque<usize>,
}

impl RowCache {
    /// Return a cloned cached row by absolute index.
    pub fn get(&self, idx: usize) -> Option<RowView> {
        self.rows.get(&idx).cloned()
    }

    /// Insert or refresh a cached row and evict oldest entries over `capacity`.
    pub fn insert(&mut self, idx: usize, row: RowView, capacity: usize) {
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

/// Cache key for one parquet row-group request: shard path + group position.
pub type ParquetGroupKey = (PathBuf, usize);
/// One row-group read request: absolute index, in-group offset, shard.
pub type ParquetGroupRequest = (usize, usize, ShardIndex);
/// Remote manifest listing: candidates, per-candidate sizes, total rows.
pub type ParquetManifestCandidates = (Vec<String>, HashMap<String, u64>, usize);
