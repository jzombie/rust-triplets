//! Default (uniform-random) negative selection backend.
//!
//! Used when the `bm25-mining` feature is disabled.  Selects uniformly at
//! random from the pre-filtered strategy pool.  Because this backend is
//! zero-sized, all randomness state lives in the sampler's own `rng` field.

use std::collections::HashSet;
use std::sync::Arc;

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
        pool: Vec<Arc<DataRecord>>,
        fallback_used: bool,
        _anchor_query_text: Option<&str>,
        rng: &mut dyn rand::RngCore,
    ) -> Option<(Arc<DataRecord>, bool)> {
        use rand::prelude::IndexedRandom as _;
        pool.choose(rng)
            .cloned()
            .map(|record| (record, fallback_used))
    }

    fn on_sync_start(&mut self) {}

    fn on_records_refreshed(
        &mut self,
        _records: &IndexMap<RecordId, Arc<DataRecord>>,
        _max_window_tokens: usize,
        _split_fn: &dyn Fn(&RecordId) -> Option<SplitLabel>,
        _refreshed_source_ids: &[SourceId],
    ) {
    }

    fn prune_cursors(&mut self, _valid_ids: &HashSet<RecordId>) {}

    fn cursors_empty(&self) -> bool {
        true
    }

    // Note: Even though the bm25 backend uses a different trait impl, this method
    // must still be defined on the default for it to be compiled.
    #[cfg(all(feature = "bm25-mining", feature = "extended-metrics"))]
    fn bm25_fallback_stats(&self) -> (u64, u64) {
        (0, 0)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{QualityScore, RecordSection, SectionRole};
    use chrono::Utc;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn record(id: &str) -> DataRecord {
        DataRecord {
            id: id.to_string(),
            source: "test_source".to_string(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            quality: QualityScore { trust: 1.0 },
            taxonomy: Vec::new(),
            sections: vec![RecordSection {
                role: SectionRole::Anchor,
                heading: None,
                text: format!("text for {id}"),
                sentences: vec![format!("text for {id}")],
            }],
            meta_prefix: None,
        }
    }

    #[test]
    fn default_backend_returns_none_for_empty_pool() {
        let mut backend = DefaultBackend;
        let mut rng = StdRng::seed_from_u64(7);
        let anchor = record("anchor");

        let selected = backend.choose_negative(
            &anchor,
            SplitLabel::Train,
            Vec::new(),
            false,
            None,
            &mut rng,
        );

        assert!(selected.is_none());
    }

    #[test]
    fn default_backend_selects_from_pool_and_preserves_fallback_flag() {
        let mut backend = DefaultBackend;
        let mut rng = StdRng::seed_from_u64(11);
        let anchor = record("anchor");
        let pool = vec![Arc::new(record("neg_a")), Arc::new(record("neg_b")), Arc::new(record("neg_c"))];

        let selected = backend.choose_negative(
            &anchor,
            SplitLabel::Train,
            pool,
            true,
            Some("anchor query text"),
            &mut rng,
        );

        let (record, fallback_used) = selected.expect("expected a record from non-empty pool");
        assert!(matches!(record.id.as_str(), "neg_a" | "neg_b" | "neg_c"));
        assert!(fallback_used);
    }

    #[test]
    fn default_backend_noop_methods_and_test_hooks_are_stable() {
        let mut backend = DefaultBackend;
        let records = IndexMap::from_iter([("r1".to_string(), Arc::new(record("r1")))]);
        let valid_ids = HashSet::from_iter(["r1".to_string()]);

        backend.on_sync_start();
        backend.on_records_refreshed(&records, 128, &|_| Some(SplitLabel::Train), &[]);
        backend.prune_cursors(&valid_ids);

        assert!(backend.cursors_empty());

        #[cfg(all(feature = "bm25-mining", feature = "extended-metrics"))]
        assert_eq!(backend.bm25_fallback_stats(), (0, 0));

        let any_ref = backend.as_any_mut();
        assert!(any_ref.downcast_mut::<DefaultBackend>().is_some());
    }
}
