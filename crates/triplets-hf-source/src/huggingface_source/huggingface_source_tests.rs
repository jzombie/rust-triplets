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
