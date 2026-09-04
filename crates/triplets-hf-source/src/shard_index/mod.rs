pub mod shard_index;
pub(crate) use shard_index::{
    build_shard_index, index_single_shard, is_store_shard_path,
    shard_store_path_for, row_store_row_key,
};
#[cfg(test)]
pub(crate) use shard_index::parquet_row_group_map;

#[cfg(test)]
mod shard_index_tests;
