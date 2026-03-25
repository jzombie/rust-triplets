//! Default (uniform-random) negative selection backend.
//!
//! Used when the `bm25-mining` feature is disabled.  Selects uniformly at
//! random from the pre-filtered strategy pool.  Because this backend is
//! zero-sized, all randomness state lives in the sampler's own `rng` field.

use std::collections::HashSet;

use indexmap::IndexMap;

use crate::data::DataRecord;
use crate::splits::SplitLabel;
use crate::types::{RecordId, SourceId};

use super::NegativeBackend;

/// Stateless uniform-random negative selection.
///
/// Chooses uniformly at random from the pre-filtered strategy pool supplied by
/// `select_negative_record`.  The sampler's own `rng` field provides all
/// randomness — no state is maintained here.
// When bm25-mining is enabled, Bm25Backend is always used; DefaultBackend is
// only constructed in non-bm25 builds.  The dead_code lint fires in bm25 builds.
#[allow(dead_code)]
pub struct DefaultBackend;

impl NegativeBackend for DefaultBackend {
    fn choose_negative(
        &mut self,
        _anchor: &DataRecord,
        _anchor_split: SplitLabel,
        pool: Vec<DataRecord>,
        fallback_used: bool,
        rng: &mut dyn rand::RngCore,
    ) -> Option<(DataRecord, bool)> {
        use rand::prelude::IndexedRandom as _;
        pool.choose(rng)
            .cloned()
            .map(|record| (record, fallback_used))
    }

    fn on_sync_start(&mut self) {}

    fn on_records_refreshed(
        &mut self,
        _records: &IndexMap<RecordId, DataRecord>,
        _max_window_tokens: usize,
        _split_fn: &dyn Fn(&RecordId) -> Option<SplitLabel>,
        _refreshed_source_ids: &[SourceId],
    ) {
    }

    fn prune_cursors(&mut self, _valid_ids: &HashSet<RecordId>) {}

    fn cursors_empty(&self) -> bool {
        true
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
