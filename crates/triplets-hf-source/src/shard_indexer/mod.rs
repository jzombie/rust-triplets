pub mod shard_indexer;
pub(crate) use shard_indexer::{
    candidate_store_path, open_shard_store, get_or_open_shard_store,
    prune_store_cache_to_shards, invalidate_eligible_index,
    enforce_disk_cap_locked, manifest_usage_bytes_locked,
    locate_shard, locate_parquet_group,
};
#[cfg(test)]
pub(crate) use shard_indexer::{
    recompute_shard_offsets, sync_shard_state_from_disk_locked,
    shard_signature, build_eligible_rows_from_shards, eligible_rows,
};

#[cfg(test)]
mod shard_indexer_tests;
