pub mod disk_cache;
#[cfg(test)]
pub(crate) use disk_cache::ensure_cache_group;
pub use disk_cache::{StoreCache, managed_hf_list_snapshot_dir, managed_hf_snapshot_dir};
pub(crate) use disk_cache::{open_store_via_cache, remove_stale_store};

#[cfg(test)]
mod disk_cache_tests;
