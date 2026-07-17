use crate::config::HuggingFaceRowsConfig;
use crate::constants::{HF_ALL_SPLITS_DIR, HF_GROUP};
use cache_manager::CacheRoot;
use simd_r_drive::DataStore;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use tracing::warn;
use triplets_core::SamplerError;

#[cfg(test)]
use std::sync::OnceLock;
#[cfg(test)]
use tempfile::TempDir;

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
    pub(crate) fn new() -> Self {
        StoreCache(Arc::new(Mutex::new(HashMap::new())))
    }

    pub(crate) fn lock(
        &self,
    ) -> Result<MutexGuard<'_, HashMap<PathBuf, Arc<DataStore>>>, SamplerError> {
        self.0.lock().map_err(|_| SamplerError::SourceUnavailable {
            source_id: "store_cache".to_string(),
            reason: "row-store cache lock poisoned".to_string(),
        })
    }

    pub(crate) fn lock_ok(&self) -> Option<MutexGuard<'_, HashMap<PathBuf, Arc<DataStore>>>> {
        self.0.lock().ok()
    }
}

pub(crate) fn managed_cache_root() -> Result<CacheRoot, String> {
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

pub(crate) fn ensure_cache_group(relative_group: PathBuf) -> Result<PathBuf, String> {
    let cache_root = managed_cache_root()?;
    cache_root.ensure_group(&relative_group).map_err(|err| {
        format!(
            "failed creating managed cache group '{}': {err}",
            relative_group.display()
        )
    })
}

/// Evict a stale store from the cache and unlink the file so the shard
/// gets re-downloaded on the next cycle.
pub(crate) fn remove_stale_store(config: &HuggingFaceRowsConfig, path: &Path) {
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

/// Get or open a store through the shared cache.  Never opens a duplicate
/// handle — if the path is already in `store_cache`, returns the cached
/// `Arc`; otherwise opens and inserts it.
pub(crate) fn open_store_via_cache(
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
    let store = Arc::new(crate::shard_indexing::open_shard_store(config, path)?);
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

#[cfg(test)]
mod tests {
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
}
