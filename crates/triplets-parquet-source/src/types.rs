//! Shard index and local-frontier state types.
use std::path::PathBuf;

/// Index entry for one materialized local shard.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardIndex {
    /// Local shard path (parquet, text, or `.simdr` row store).
    pub path: PathBuf,
    /// Global row offset of this shard's first row.
    pub global_start: usize,
    /// Number of materialized rows in this shard.
    pub row_count: usize,
    /// Parquet `(group_start, group_count)` map for random-access reads.
    /// Single `(0, row_count)` entry for row-store shards.
    pub parquet_row_groups: Vec<(usize, usize)>,
    /// Opaque tag recording where this shard came from (e.g. the remote
    /// candidate string it was downloaded from, used to re-queue the
    /// download if the local file is evicted).  Ignored by the generic
    /// reader; set and interpreted by remote-backed wrappers.
    pub remote_candidate: Option<String>,
}

/// Local row frontier: indexed shards and total materialized row count.
///
/// Remote-backed wrappers (e.g. Hugging Face) keep their own remote
/// frontier separately and merge new shards in via
/// [`crate::source::ParquetRowSource::ingest_shard`].
#[derive(Debug, Default)]
pub struct LocalShardState {
    /// Total materialized rows across all shards.
    pub materialized_rows: usize,
    /// Indexed shards in global-index order.
    pub shards: Vec<ShardIndex>,
}
