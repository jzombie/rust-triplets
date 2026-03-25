//! Pluggable negative-example selection backends.
//!
//! [`NegativeBackend`] is the only interface `mod.rs` programs against.  The
//! two concrete implementations — `DefaultBackend` (always compiled) and
//! `Bm25Backend` (`bm25-mining` feature only) — are fully encapsulated;
//! their internal state is invisible to the sampler core.
//!
//! Adding a new backend: create `backends/my_backend.rs`, gate `pub(super) mod
//! my_backend;` below, implement `NegativeBackend`, and swap in the constructor
//! in `TripletSamplerInner::new`.

use std::collections::HashSet;

use indexmap::IndexMap;

use crate::data::DataRecord;
use crate::splits::SplitLabel;
use crate::types::{RecordId, SourceId};

#[cfg(feature = "bm25-mining")]
pub(super) mod bm25_backend;
mod default_backend;

#[cfg(feature = "bm25-mining")]
pub(super) use self::bm25_backend::Bm25Backend;
#[cfg(not(feature = "bm25-mining"))]
pub(super) use self::default_backend::DefaultBackend;

/// Contract for negative-example selection.
///
/// Each backend receives a strategy-filtered `pool` produced by
/// `select_negative_record` and is solely responsible for **ranking or
/// selecting** within it.  Strategy predicates (source isolation, split
/// isolation, date matching) have already been applied by the caller before
/// `choose_negative` is invoked.
pub(super) trait NegativeBackend: Send {
    /// Select a hard-negative record from `pool` for `anchor`.
    ///
    /// `anchor_query_text` is the rendered text of the anchor's already-selected
    /// chunk window.  When `Some`, backends that perform lexical ranking (BM25)
    /// use it as the query instead of re-deriving text from the full article,
    /// producing negatives that are hard relative to the specific window rather
    /// than the full article.  Pass `None` to fall back to full-article text.
    ///
    /// `rng` is the sampler's top-level RNG; backends that need randomness for
    /// tie-breaking or fallback should use it rather than maintaining their own.
    /// `fallback_used` threads through from the caller and is returned unchanged
    /// so the batch builder can record whether the negative is from a degraded pool.
    fn choose_negative(
        &mut self,
        anchor: &DataRecord,
        anchor_split: SplitLabel,
        pool: Vec<DataRecord>,
        fallback_used: bool,
        anchor_query_text: Option<&str>,
        rng: &mut dyn rand::RngCore,
    ) -> Option<(DataRecord, bool)>;

    /// Called at the start of each `sync_records_from_cache` cycle, before the
    /// record pool is updated.  Backends should reset any per-anchor cursor state
    /// that must not carry over across corpus snapshot boundaries.
    fn on_sync_start(&mut self);

    /// Called after `sync_records_from_cache` completes and new records are in
    /// place.  `refreshed_source_ids` lists the sources whose buffers refetched
    /// during this cycle; each backend rebuilds only the state that depends on
    /// those sources rather than performing a full global reset.
    fn on_records_refreshed(
        &mut self,
        records: &IndexMap<RecordId, DataRecord>,
        max_window_tokens: usize,
        split_fn: &dyn Fn(&RecordId) -> Option<SplitLabel>,
        refreshed_source_ids: &[SourceId],
    );

    /// Prune internal per-record state so that only entries whose IDs appear in
    /// `valid_ids` are retained.  Called after every record-pool refresh.
    fn prune_cursors(&mut self, valid_ids: &HashSet<RecordId>);

    /// Returns `true` when all internal cursor maps are empty.
    ///
    /// Used as a fast-path check in `prune_cursor_state` to skip building the
    /// `valid_ids` set when nothing needs pruning.
    fn cursors_empty(&self) -> bool;

    /// Expose as [`std::any::Any`] for test-only downcasting.
    ///
    /// Each concrete backend must implement this as `fn as_any_mut(&mut self)
    /// -> &mut dyn std::any::Any { self }`.
    #[cfg(test)]
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}
