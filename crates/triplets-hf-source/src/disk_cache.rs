use crate::config::HuggingFaceRowsConfig;
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

// TODO: Finish refactor
// Apply cache-manager eviction policy to manifest shards and sync in-memory state.
// pub fn enforce_disk_cap_locked(
//     &self,
//     state: &mut SourceState,
//     _protected_path: &Path,
// ) -> Result<bool, SamplerError> {
//     let Some(cap_bytes) = self.config.local_disk_cap_bytes else {
//         return Ok(false);
//     };

//     let before = state
//         .shards
//         .iter()
//         .map(|shard| shard.path.clone())
//         .collect::<Vec<_>>();
//     let policy = EvictPolicy {
//         max_bytes: Some(cap_bytes),
//         ..EvictPolicy::default()
//     };

//     let cache_root = CacheRoot::from_root(&self.config.snapshot_dir);
//     cache_root
//         .ensure_group_with_policy(HF_PARQUET_MANIFEST_DIR, Some(&policy))
//         .map_err(|err| SamplerError::SourceUnavailable {
//             source_id: self.config.source_id.clone(),
//             reason: format!(
//                 "failed applying manifest cache eviction policy under {}: {err}",
//                 self.config.snapshot_dir.display()
//             ),
//         })?;

//     self.sync_shard_state_from_disk_locked(state);
//     let after = state
//         .shards
//         .iter()
//         .map(|shard| shard.path.clone())
//         .collect::<Vec<_>>();
//     Ok(before != after)
// }
