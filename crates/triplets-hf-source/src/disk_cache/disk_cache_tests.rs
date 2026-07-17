use super::*;
use crate::constants::{HF_ALL_SPLITS_DIR, HF_GROUP};
use crate::disk_cache::ensure_cache_group;
use crate::test_utils::{test_config, with_current_dir};
use serial_test::serial;
use simd_r_drive::storage_engine::DataStore;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::tempdir;
use triplets_core::utils::platform_newline;

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
        let listed = managed_hf_list_snapshot_dir("org/dataset", "default", "train", 7).unwrap();

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
fn ensure_cache_group_reports_error() {
    let bad_group = PathBuf::from("bad\0group");
    let result = ensure_cache_group(bad_group);
    assert!(result.is_err());
}

#[test]
fn open_store_via_cache_inserts_and_reuses() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let path = dir.path().join("store.simdr");
    let store = DataStore::open(&path).unwrap();
    drop(store);
    let first = crate::disk_cache::open_store_via_cache(&config, &path).unwrap();
    let second = crate::disk_cache::open_store_via_cache(&config, &path).unwrap();
    assert!(Arc::ptr_eq(&first, &second));
}
