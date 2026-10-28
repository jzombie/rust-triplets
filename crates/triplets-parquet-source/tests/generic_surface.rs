//! Public-surface coverage for the generic parquet reader:
//! constructors, accessors, refresh/reporting, recipes, disk-cap
//! orchestration, store cache, and fixture writers.
use std::fs;
use std::path::Path;

use simd_r_drive::storage_engine::DataStore;
use tempfile::tempdir;
use triplets_core::SamplerError;
use triplets_core::config::SamplerConfig;
use triplets_core::source::DataSource;
use triplets_parquet_source::fixtures::{
    TEST_TRANSIENT_DIR, test_config, test_source, write_gzip_fixture, write_parquet_fixture,
    write_simdr_fixture,
};
use triplets_parquet_source::rows::read_store_row_count;
use triplets_parquet_source::store_cache::remove_stale_store;
use triplets_parquet_source::{ParquetRowSource, ParquetRowsConfig, StoreCache};

fn sampler_with(pool: usize) -> SamplerConfig {
    SamplerConfig {
        seed: 7,
        per_source_max_pool: pool,
        ..SamplerConfig::default()
    }
}

#[test]
fn new_rejects_missing_column_mapping() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.text_columns.clear();
    assert!(
        matches!(
            ParquetRowSource::new(config),
            Err(SamplerError::Configuration(_))
        ),
        "expected Configuration error for empty mapping"
    );
}

#[test]
fn new_fails_when_snapshot_path_is_a_file() {
    let dir = tempdir().unwrap();
    let file = dir.path().join("not-a-dir");
    fs::write(&file, b"x").unwrap();
    let config = test_config(file);
    assert!(ParquetRowSource::new(config).is_err());
}

#[test]
fn empty_source_surface() {
    let dir = tempdir().unwrap();
    let source = ParquetRowSource::new_empty(test_config(dir.path().to_path_buf()));

    assert_eq!(source.source_id(), "parquet_test");
    assert_eq!(DataSource::id(&source), "parquet_test");
    assert_eq!(source.config().source_id, "parquet_test");
    assert_eq!(source.materialized_row_count(), 0);
    assert_eq!(source.shard_count(), 0);
    assert!(source.shards_snapshot().is_empty());
    assert_eq!(source.len_hint(), Some(0));
    assert_eq!(source.active_pool_size(), 10);
    assert!(source.sampler_config_snapshot().is_none());
    assert!(source.configured_sampler_seed().is_err());
    assert!(source.paging_seed(0).is_err());
    assert_eq!(source.effective_refresh_batch_target(5), 40);

    let snapshot = source
        .refresh(&SamplerConfig::default(), None, Some(10))
        .unwrap();
    assert!(snapshot.records.is_empty());
    assert_eq!(snapshot.cursor.revision, 0);
    assert_eq!(
        source
            .reported_record_count(&SamplerConfig::default())
            .unwrap(),
        0
    );
    assert_eq!(source.usage_bytes_under(dir.path()), 0);

    source.set_active_sampler_config(&sampler_with(4));
    assert_eq!(source.active_pool_size(), 4);
    assert!(source.sampler_config_snapshot().is_some());
    assert_eq!(source.configured_sampler_seed().unwrap(), 7);
    assert!(source.paging_seed(3).is_ok());
}

#[test]
fn default_recipes_cover_all_modes() {
    let dir = tempdir().unwrap();

    // Text-columns mode.
    let source = test_source(test_config(dir.path().to_path_buf()));
    let recipes = DataSource::default_triplet_recipes(&source);
    assert_eq!(recipes.len(), 1);
    assert!(recipes[0].allow_same_anchor_positive);

    // Role mode.
    let mut config = test_config(dir.path().to_path_buf());
    config.anchor_columns = vec!["title".to_string()];
    config.positive_columns = vec!["body".to_string()];
    let source = test_source(config);
    let recipes = DataSource::default_triplet_recipes(&source);
    assert_eq!(recipes.len(), 2);

    // Dict mode.
    let mut config = test_config(dir.path().to_path_buf());
    config.anchor_columns = vec!["q".to_string()];
    config.positive_columns = vec!["a".to_string()];
    config.negative_columns = vec!["negs".to_string()];
    let source = test_source(config);
    let recipes = DataSource::default_triplet_recipes(&source);
    assert_eq!(recipes.len(), 1);
    assert!(!recipes[0].allow_same_anchor_positive);
}

#[test]
fn enforce_disabled_without_cap_or_subdir() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.managed_subdir = None;
    config.local_disk_cap_bytes = Some(1);
    let source = test_source(config);
    assert!(!source.enforce_disk_cap().unwrap());

    let mut config = test_config(dir.path().to_path_buf());
    config.local_disk_cap_bytes = None;
    let source = test_source(config);
    assert!(!source.enforce_disk_cap().unwrap());
}

#[test]
fn enforce_evicts_and_usage_tracks_indexed_shards() {
    let dir = tempdir().unwrap();
    let transient = dir.path().join(TEST_TRANSIENT_DIR);
    fs::create_dir_all(&transient).unwrap();
    let first = transient.join("first.parquet");
    let second = transient.join("second.parquet");
    write_parquet_fixture(&first, &[("a", "A")]);
    write_parquet_fixture(&second, &[("b", "B")]);

    let mut config = test_config(dir.path().to_path_buf());
    config.local_disk_cap_bytes = Some(20);
    let source = test_source(config.clone());
    for path in [&first, &second] {
        let shard = triplets_parquet_source::shard_index::index_single_shard(&config, path, 0)
            .unwrap()
            .0
            .unwrap();
        source.ingest_shard(shard).unwrap();
    }
    assert_eq!(source.materialized_row_count(), 2);
    assert!(source.usage_bytes_under(&transient) > 0);
    assert_eq!(
        source.usage_bytes_under(dir.path().join("elsewhere").as_path()),
        0
    );

    assert!(source.enforce_disk_cap().unwrap());
    assert_eq!(source.materialized_row_count(), 0);
    assert!(source.shards_snapshot().is_empty());
    assert_eq!(source.usage_bytes_under(&transient), 0);
}

#[test]
fn store_cache_debug_default_lock_and_stale_removal() {
    let cache = StoreCache::new();
    let rendered = format!("{cache:?}");
    assert!(rendered.contains("StoreCache"));
    let _ = StoreCache::default();
    assert!(cache.lock().is_ok());
    assert!(cache.lock_ok().is_some());

    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    // Missing path: cache evict is a no-op, unlink warns but does not fail.
    remove_stale_store(&config, &dir.path().join("missing.simdr"));

    let store_path = dir.path().join("stale.simdr");
    write_simdr_fixture(&store_path, &[("r1", "v1")]);
    assert!(store_path.exists());
    remove_stale_store(&config, &store_path);
    assert!(!store_path.exists());
}

#[test]
fn fixture_writers_roundtrip() {
    let dir = tempdir().unwrap();

    // Gzip fixture decodes to the original bytes.
    let gz_path = dir.path().join("rows.jsonl.gz");
    let payload = br#"{"id":"g1","text":"gzipped"}"#.to_vec();
    write_gzip_fixture(&gz_path, &payload);
    let raw = fs::read(&gz_path).unwrap();
    assert_eq!(&raw[0..2], &[0x1f, 0x8b], "gzip magic");

    // Empty simdr fixture records a zero row count.
    let empty_path = dir.path().join("empty.simdr");
    write_simdr_fixture(&empty_path, &[]);
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    let store = DataStore::open(&empty_path).unwrap();
    assert_eq!(read_store_row_count(&source, &store).unwrap(), 0);
}

#[test]
fn parquet_config_helpers() {
    let dir = tempdir().unwrap();
    let mut config = ParquetRowsConfig::new("s", dir.path(), "ns", vec!["k=v".to_string()]);
    assert!(config.has_explicit_mapping());
    assert!(!config.in_transient_dir(Path::new("/elsewhere")));
    config.transient_dirs = vec![dir.path().join("_transient")];
    assert!(config.in_transient_dir(&dir.path().join("_transient")));

    let normalized = ParquetRowSource::normalized_shard_extensions(&[
        ".PARQUET".to_string(),
        " ndjson ".to_string(),
    ]);
    assert_eq!(
        normalized,
        vec!["parquet".to_string(), "ndjson".to_string()]
    );
}
