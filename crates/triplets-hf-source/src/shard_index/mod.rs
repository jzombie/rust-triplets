pub mod shard_index;
#[cfg(test)]
pub(crate) use shard_index::parquet_row_group_map;
pub(crate) use shard_index::{
    build_shard_index, index_single_shard, is_store_shard_path, row_store_row_key,
    shard_store_path_for,
};

#[cfg(test)]
mod shard_index_tests;
