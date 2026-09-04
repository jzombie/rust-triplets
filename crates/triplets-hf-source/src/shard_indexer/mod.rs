pub mod shard_indexer;
#[cfg(test)]
pub(crate) use shard_indexer::{
    build_eligible_rows_from_shards, eligible_rows, recompute_shard_offsets, shard_signature,
    sync_shard_state_from_disk_locked,
};
pub(crate) use shard_indexer::{
    candidate_store_path, enforce_disk_cap_locked, get_or_open_shard_store,
    invalidate_eligible_index, locate_parquet_group, locate_shard, manifest_usage_bytes_locked,
    open_shard_store, prune_store_cache_to_shards,
};

#[cfg(test)]
mod shard_indexer_tests;
