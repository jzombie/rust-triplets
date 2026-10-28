use crate::constants::{HF_ALL_SPLITS_DIR, HF_GROUP};
use cache_manager::CacheRoot;
use std::path::PathBuf;

#[cfg(test)]
use std::sync::OnceLock;
#[cfg(test)]
use tempfile::TempDir;

/// Shared handle to the open-store cache: re-exported from the generic
/// parquet crate, where the row-store layer now lives.
pub use triplets_parquet_source::StoreCache;

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
