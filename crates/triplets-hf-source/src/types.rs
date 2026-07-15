use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ShardIndex {
    pub(crate) path: PathBuf,
    pub(crate) global_start: usize,
    pub(crate) row_count: usize,
    pub(crate) parquet_row_groups: Vec<(usize, usize)>,
    /// Remote candidate string this shard was downloaded from, used to
    /// re-queue the download if the local file is evicted from the cache.
    pub(crate) remote_candidate: Option<String>,
}

#[derive(Debug)]
pub(crate) struct SourceState {
    pub(crate) materialized_rows: usize,
    pub(crate) shards: Vec<ShardIndex>,
    /// Sorted, immutable list of all remote candidate identifiers.  Never
    /// shuffled in-place — ordering is expressed via `remote_candidate_order`.
    pub(crate) remote_candidates: Option<Vec<String>>,
    pub(crate) remote_candidate_sizes: HashMap<String, u64>,
    /// Seed-derived permutation of indices into `remote_candidates`.  For a
    /// given (seed, total) this is always the same sequence, regardless of
    /// how many shards have been consumed previously.
    pub(crate) remote_candidate_order: Vec<usize>,
    pub(crate) next_remote_idx: usize,
}
