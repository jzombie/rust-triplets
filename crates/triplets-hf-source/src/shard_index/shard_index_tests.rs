use super::*;
use crate::constants::{
    HF_PARQUET_MANIFEST_DIR, HF_SHARD_STORE_EXTENSION, HF_SHARD_STORE_META_ROWS_KEY,
    HF_SHARD_STORE_ROW_PREFIX,
};
use crate::test_utils::write_parquet_fixture;
use crate::test_utils::{test_config, write_simdr_fixture};
use simd_r_drive::storage_engine::DataStore;
use simd_r_drive::storage_engine::traits::DataStoreWriter;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;
use triplets_core::SamplerError;

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

    let (shards, discovered) = build_shard_index(&config).unwrap();
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

    let (shards, discovered) = build_shard_index(&config).unwrap();
    assert_eq!(discovered, 2, "simdr store rows should be indexed");
    assert_eq!(shards.len(), 1);
    assert_eq!(shards[0].path, store_path);
    assert_eq!(shards[0].row_count, 2);
}

#[test]
fn build_shard_index_discovers_local_jsonl_shards() {
    let dir = tempdir().unwrap();
    let root = dir.path().to_path_buf();
    fs::write(root.join("a.jsonl"), b"{\"text\":\"a\"}\n").unwrap();
    fs::write(root.join("b.ndjson"), b"{\"text\":\"b\"}\n").unwrap();

    let config = test_config(root.clone());
    let (shards, discovered) = build_shard_index(&config).unwrap();
    assert_eq!(discovered, 2);
    assert_eq!(shards.len(), 2);
}

#[test]
fn build_shard_index_errors_when_no_matching_extensions() {
    let dir = tempdir().unwrap();
    fs::write(dir.path().join("data.txt"), b"x\n").unwrap();
    let config = test_config(dir.path().to_path_buf());
    let result = build_shard_index(&config);
    // .txt is now a recognized transient text format, so build_shard_index
    // should succeed (the file is treated as a transient shard).
    assert!(result.is_ok());
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

    let result = build_shard_index(&config);
    assert!(result.is_err());
}

#[test]
fn build_shard_index_errors_when_no_accepted_files_exist() {
    let dir = tempdir().unwrap();
    fs::write(dir.path().join("notes.dat"), b"plain").unwrap();
    let config = test_config(dir.path().to_path_buf());

    let err = build_shard_index(&config).expect_err("build_shard_index should fail");
    assert!(matches!(
        err,
        SamplerError::SourceUnavailable { ref reason, .. } if reason.contains("no shard files found")
    ));
}

#[test]
fn build_shard_index_skips_empty_files_and_keeps_non_empty() {
    let dir = tempdir().unwrap();
    fs::write(dir.path().join("a.ndjson"), b"").unwrap();
    fs::write(dir.path().join("b.ndjson"), b"{\"text\":\"x\"}\n").unwrap();
    let config = test_config(dir.path().to_path_buf());

    let (shards, discovered) = build_shard_index(&config).unwrap();
    assert_eq!(discovered, 1);
    assert_eq!(shards.len(), 1);
    assert_eq!(shards[0].row_count, 1);
}

#[test]
fn build_shard_index_prunes_orphaned_remote_transients() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());

    // Create a transient file inside _parquet_manifest/ (simulates a
    // crashed/truncated remote download).
    let manifest_root = dir.path().join(HF_PARQUET_MANIFEST_DIR);
    fs::create_dir_all(&manifest_root).unwrap();
    let orphan = manifest_root.join("orphaned_shard.ndjson");
    fs::write(&orphan, b"{\"id\":\"r1\",\"text\":\"x\"}\n").unwrap();
    assert!(orphan.exists(), "orphan must exist before indexing");

    // Also create a valid local shard so build_shard_index succeeds.
    let local = dir.path().join("local.ndjson");
    fs::write(&local, b"{\"id\":\"l1\",\"text\":\"y\"}\n").unwrap();

    let (shards, discovered) = build_shard_index(&config).unwrap();
    assert_eq!(discovered, 1, "only local shard should be discovered");
    assert_eq!(shards.len(), 1);
    assert!(!orphan.exists(), "orphaned transient must be pruned");
}

#[test]
fn is_store_shard_path_detects_simdr_extension() {
    assert!(is_store_shard_path(Path::new("shard.simdr")));
    assert!(is_store_shard_path(Path::new("shard.SIMDR")));
    assert!(is_store_shard_path(Path::new("shard.SimDr")));
    assert!(!is_store_shard_path(Path::new("shard.parquet")));
    assert!(!is_store_shard_path(Path::new("shard.ndjson")));
    assert!(!is_store_shard_path(Path::new("no-extension")));
    assert!(!is_store_shard_path(Path::new(".hidden")));
}

#[test]
fn is_parquet_path_recognizes_parquet_extension() {
    assert!(crate::source_core::HuggingFaceRowSource::is_parquet_path(
        Path::new("data/train.parquet")
    ));
    assert!(crate::source_core::HuggingFaceRowSource::is_parquet_path(
        Path::new("data/train.PARQUET")
    ));
    assert!(!crate::source_core::HuggingFaceRowSource::is_parquet_path(
        Path::new("data/train.jsonl")
    ));
    assert!(!crate::source_core::HuggingFaceRowSource::is_parquet_path(
        Path::new("data/train.txt")
    ));
}

#[test]
fn shard_store_path_for_passthrough_when_already_simdr() {
    let path = PathBuf::from("cache/shard.simdr");
    let mapped = shard_store_path_for(&path);
    assert_eq!(mapped, path);
}

#[test]
fn shard_store_path_for_appends_simdr_extension() {
    let path = PathBuf::from("cache/shard.parquet");
    let mapped = shard_store_path_for(&path);
    assert_eq!(mapped, PathBuf::from("cache/shard.simdr"));
    let no_ext = PathBuf::from("cache/shard");
    let mapped2 = shard_store_path_for(&no_ext);
    assert_eq!(mapped2, PathBuf::from("cache/shard.simdr"));
}

#[test]
fn index_single_shard_errors_for_missing_file() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let missing = dir.path().join("missing.ndjson");

    let err = index_single_shard(&config, &missing, 0)
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
    let (maybe_shard, _) = index_single_shard(&config, &store_path, 0)
        .expect("corrupted store should not produce a hard error — it deletes and returns None");
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

    let shard = index_single_shard(&config, &path, 5).unwrap().0.unwrap();
    assert_eq!(shard.global_start, 5);
    assert_eq!(shard.row_count, 1); // Dummy count for transient text files
}

#[test]
fn index_single_shard_returns_none_for_empty_file() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let path = dir.path().join("empty.jsonl");
    fs::write(&path, b"").unwrap();
    let shard = index_single_shard(&config, &path, 0).unwrap();
    assert!(shard.0.is_none());
}

#[test]
fn parquet_row_group_map_handles_empty_parquet_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("empty.parquet");
    write_parquet_fixture(&path, &[]);
    let config = test_config(dir.path().to_path_buf());

    let (rows, groups) = parquet_row_group_map(&config, &path).unwrap();
    assert_eq!(rows, 0);
    assert!(groups.is_empty());
}

#[test]
fn parquet_row_group_map_and_index_single_shard_cover_success_path() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("rows.parquet");
    write_parquet_fixture(&path, &[("r1", "alpha"), ("r2", "beta"), ("r3", "gamma")]);
    let config = test_config(dir.path().to_path_buf());

    let (total_rows, groups) = parquet_row_group_map(&config, &path).unwrap();
    assert_eq!(total_rows, 3);
    assert!(!groups.is_empty());

    let shard = index_single_shard(&config, &path, 0).unwrap().0.unwrap();
    assert_eq!(shard.row_count, 3);
    // All shards are now .simdr stores with O(1) random access
}

#[test]
fn row_store_row_key_uses_expected_format() {
    let key = row_store_row_key(0);
    assert!(key.starts_with(HF_SHARD_STORE_ROW_PREFIX));
    assert_eq!(key.len(), HF_SHARD_STORE_ROW_PREFIX.len() + 8);
    let key_42 = row_store_row_key(42);
    assert!(key_42.starts_with(HF_SHARD_STORE_ROW_PREFIX));
}

#[test]
fn build_shard_index_empty_directory_returns_error() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());

    let result = crate::shard_index::build_shard_index(&config);
    assert!(result.is_err(), "empty directory should return error");
}

#[test]
fn build_shard_index_prunes_orphaned_transients_in_manifest() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let manifest_dir = dir.path().join(HF_PARQUET_MANIFEST_DIR);
    fs::create_dir_all(&manifest_dir).unwrap();

    // Create an orphaned transient file in the manifest directory
    let orphan = manifest_dir.join("orphan.ndjson");
    fs::write(&orphan, b"{}").unwrap();
    assert!(orphan.exists());

    // build_shard_index should delete orphaned transients in manifest
    let result = crate::shard_index::build_shard_index(&config);
    // Should fail because there are no valid shards, but the orphan should be pruned
    assert!(result.is_err());
    assert!(
        !orphan.exists(),
        "orphaned transient in manifest should be pruned"
    );
}

#[test]
fn build_shard_index_skips_local_files_with_existing_simdr() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.shard_extensions = vec!["ndjson".to_string()];

    // Create a local ndjson file
    let ndjson = dir.path().join("data.ndjson");
    fs::write(&ndjson, b"{}").unwrap();

    // Create a corresponding .simdr file
    let simdr = shard_store_path_for(&ndjson);
    write_simdr_fixture(&simdr, &[("r1", "text")]);

    // build_shard_index should skip the ndjson because simdr exists
    let (shards, discovered) = build_shard_index(&config).unwrap();
    // The .simdr store is always accepted, and the .ndjson is skipped
    // because its .simdr already exists (avoid double-indexing).
    assert_eq!(discovered, 1);
    assert_eq!(shards.len(), 1);

    // The ndjson should not be double-indexed
    let ndjson_indexed = shards.iter().any(|s| s.path == ndjson);
    assert!(
        !ndjson_indexed,
        "ndjson should be skipped when simdr exists"
    );

    assert!(
        shards[0].path.extension().and_then(|e| e.to_str()) == Some(HF_SHARD_STORE_EXTENSION),
        "indexed shard should be the .simdr store, not the .ndjson"
    );
}
