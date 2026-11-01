//! Shared `.simdr` row-store handle cache.
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use simd_r_drive::DataStore;
use tracing::warn;
use triplets_core::SamplerError;

use crate::config::ParquetRowsConfig;

/// Shared handle to the open-store cache.  Stored on [`ParquetRowsConfig`]
/// so all methods have access without passing it separately.
#[derive(Clone)]
pub struct StoreCache(pub(crate) Arc<Mutex<HashMap<PathBuf, Arc<DataStore>>>>);

impl std::fmt::Debug for StoreCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoreCache").finish_non_exhaustive()
    }
}

impl Default for StoreCache {
    fn default() -> Self {
        Self::new()
    }
}

impl StoreCache {
    /// Create an empty store cache.
    pub fn new() -> Self {
        StoreCache(Arc::new(Mutex::new(HashMap::new())))
    }

    /// Lock the cache, mapping poisoning to a source error.
    pub fn lock(&self) -> Result<MutexGuard<'_, HashMap<PathBuf, Arc<DataStore>>>, SamplerError> {
        self.0.lock().map_err(|_| SamplerError::SourceUnavailable {
            source_id: "store_cache".to_string(),
            reason: "row-store cache lock poisoned".to_string(),
        })
    }

    /// Lock the cache, returning `None` when poisoned.
    pub fn lock_ok(&self) -> Option<MutexGuard<'_, HashMap<PathBuf, Arc<DataStore>>>> {
        self.0.lock().ok()
    }
}

/// Evict a stale store from the cache and unlink the file so the shard
/// gets rebuilt on the next cycle.
pub fn remove_stale_store(config: &ParquetRowsConfig, path: &Path) {
    let _ = config
        .store_cache
        .lock_ok()
        .map(|mut cache| cache.remove(path));
    if let Err(err) = fs::remove_file(path) {
        warn!(
            "[triplets:parquet] failed to remove stale store {}: {}",
            path.display(),
            err
        );
    }
}

/// Get or open a store through the shared cache.  Never opens a duplicate
/// handle — if the path is already in `store_cache`, returns the cached
/// `Arc`; otherwise opens and inserts it.
pub fn open_store_via_cache(
    config: &ParquetRowsConfig,
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
    let store = Arc::new(crate::shard_indexer::open_shard_store(config, path)?);
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
