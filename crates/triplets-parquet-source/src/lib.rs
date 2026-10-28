#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

//! Generic local-parquet row source for the triplets data pipeline.
//!
//! See the crate-level README for the split from `triplets-hf-source`.

/// Configuration for a local parquet/row-store source.
pub mod config;
/// Store-format constants shared by the row-store cache.
pub mod constants;
/// Path helpers for shard file formats.
pub mod file_utils;
/// Shared test fixtures (temp-dir configs, sources, fixture writers).
pub mod fixtures;
/// Normalized row views and in-memory caches.
pub mod row_view;
/// Row parsing, record materialization, and shard transcoding.
pub mod rows;
/// Local shard discovery and parquet row-group maps.
pub mod shard_index;
/// Row-store handles, eligible-row index, and disk-cap enforcement.
pub mod shard_indexer;
/// The generic [`source::ParquetRowSource`] data source.
pub mod source;
/// Shared `.simdr` row-store handle cache.
pub mod store_cache;
/// Shard index and local-frontier state types.
pub mod types;

pub use config::ParquetRowsConfig;
pub use row_view::{ParquetCache, RowCache, RowTextField, RowView};
pub use source::ParquetRowSource;
pub use store_cache::StoreCache;
pub use types::{LocalShardState, ShardIndex};

#[cfg(test)]
#[path = "file_utils_tests.rs"]
mod file_utils_tests;
#[cfg(test)]
#[path = "row_view_tests.rs"]
mod row_view_tests;
#[cfg(test)]
#[path = "rows_tests.rs"]
mod rows_tests;
#[cfg(test)]
#[path = "shard_index_tests.rs"]
mod shard_index_tests;
#[cfg(test)]
#[path = "shard_indexer_tests.rs"]
mod shard_indexer_tests;
