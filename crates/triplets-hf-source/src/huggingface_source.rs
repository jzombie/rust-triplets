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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::write_parquet_fixture;
    use std::collections::VecDeque;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tempfile::tempdir;

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
    fn row_cache_get_returns_cached_row() {
        let mut cache = RowCache::default();
        let row = RowView {
            row_id: Some("r1".to_string()),
            timestamp: None,
            text_fields: vec![RowTextField {
                name: "text".to_string(),
                text: "hello".to_string(),
            }],
        };
        cache.insert(0, row.clone(), 10);
        let retrieved = cache.get(0).expect("should return cached row");
        assert_eq!(retrieved.row_id, Some("r1".to_string()));
    }

    #[test]
    fn row_cache_get_returns_none_for_missing() {
        let cache = RowCache::default();
        assert!(cache.get(0).is_none());
    }

    #[test]
    fn row_cache_insert_evicts_oldest_when_full() {
        let mut cache = RowCache::default();
        let row1 = RowView {
            row_id: Some("r1".to_string()),
            timestamp: None,
            text_fields: vec![],
        };
        let row2 = RowView {
            row_id: Some("r2".to_string()),
            timestamp: None,
            text_fields: vec![],
        };
        let row3 = RowView {
            row_id: Some("r3".to_string()),
            timestamp: None,
            text_fields: vec![],
        };

        cache.insert(0, row1, 2);
        cache.insert(1, row2, 2);
        cache.insert(2, row3, 2);

        assert!(cache.get(0).is_none(), "oldest should be evicted");
        assert!(cache.get(1).is_some());
        assert!(cache.get(2).is_some());
    }

    #[test]
    fn row_cache_insert_zero_capacity_is_noop() {
        let mut cache = RowCache::default();
        let row = RowView {
            row_id: Some("r1".to_string()),
            timestamp: None,
            text_fields: vec![],
        };
        cache.insert(0, row, 0);
        assert!(cache.get(0).is_none());
    }

    #[test]
    fn parquet_cache_refresh_row_group_order_removes_existing() {
        use std::collections::VecDeque;

        let mut order = VecDeque::new();
        let key1 = (PathBuf::from("a.parquet"), 0);
        let key2 = (PathBuf::from("b.parquet"), 0);
        order.push_back(key1.clone());
        order.push_back(key2.clone());

        // refresh_row_group_order removes the existing entry
        crate::huggingface_source::ParquetCache::refresh_row_group_order(&mut order, &key1);

        assert_eq!(order.len(), 1);
        assert_eq!(order[0], key2);
    }

    #[test]
    fn parquet_cache_refresh_row_group_order_noop_for_missing_key() {
        use std::collections::VecDeque;

        let mut order = VecDeque::new();
        let key1 = (PathBuf::from("a.parquet"), 0);
        let key2 = (PathBuf::from("b.parquet"), 0);
        order.push_back(key1.clone());

        // Refreshing a key not in the order should be a noop
        crate::huggingface_source::ParquetCache::refresh_row_group_order(&mut order, &key2);

        assert_eq!(order.len(), 1);
        assert_eq!(order[0], key1);
    }

    #[test]
    fn parquet_cache_refresh_row_group_order_noop_for_empty() {
        use std::collections::VecDeque;

        let mut order = VecDeque::new();
        let key = (PathBuf::from("a.parquet"), 0);
        crate::huggingface_source::ParquetCache::refresh_row_group_order(&mut order, &key);
        assert!(order.is_empty());
    }
}
