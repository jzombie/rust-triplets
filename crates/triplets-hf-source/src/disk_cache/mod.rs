pub mod disk_cache;
#[cfg(test)]
pub(crate) use disk_cache::ensure_cache_group;
pub use disk_cache::{StoreCache, managed_hf_list_snapshot_dir, managed_hf_snapshot_dir};

#[cfg(test)]
mod disk_cache_tests;
