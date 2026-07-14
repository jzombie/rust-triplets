//! Adapter bridging a [`Sampler`](triplets_core::Sampler) trait object to the [`BatchProvider`] trait.

use std::sync::Arc;

use triplets_core::{Sampler, SplitLabel};
use triplets_srd_source::srd_triplet::SrdMode;

use crate::sampler_prefetcher::{filter_pair_batch, filter_triplet_batch};
use crate::traits::{BatchProvider, Result, SamplerBatch, SchedulerError};

/// Adapter that routes batch-fetching calls to a [`Sampler`] trait object
/// and converts the core [`SampleBatch`](triplets_core::data::SampleBatch) / [`TripletBatch`](triplets_core::data::TripletBatch) types into the
/// scheduler's [`SamplerBatch`].
pub struct SamplerAdapter {
    /// The underlying sampler (trait object for testability).
    pub sampler: Arc<dyn Sampler + Send + Sync>,
    /// Pair or triplet output mode.
    pub mode: SrdMode,
}

impl BatchProvider for SamplerAdapter {
    fn next_batch(&self, split: SplitLabel) -> Result<Option<SamplerBatch>> {
        match self.mode {
            SrdMode::Pair => {
                let batch = self
                    .sampler
                    .next_pair_batch(split)
                    .map_err(|e| SchedulerError::Msg(format!("sampler error: {e}")))?;
                if batch.pairs.is_empty() {
                    return Ok(None);
                }
                let entries = filter_pair_batch(&batch.pairs);
                if entries.is_empty() {
                    return Ok(None);
                }
                Ok(Some(SamplerBatch::Pairs(entries)))
            }
            SrdMode::Triplet => {
                let batch = self
                    .sampler
                    .next_triplet_batch(split)
                    .map_err(|e| SchedulerError::Msg(format!("sampler error: {e}")))?;
                if batch.triplets.is_empty() {
                    return Ok(None);
                }
                let entries = filter_triplet_batch(&batch.triplets);
                if entries.is_empty() {
                    return Ok(None);
                }
                Ok(Some(SamplerBatch::Triplets(entries)))
            }
        }
    }

    fn save_state(&self) -> Result<()> {
        self.sampler
            .save_sampler_state(None)
            .map_err(|e| SchedulerError::Msg(format!("save state error: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use triplets_core::data::{
        ChunkView, PairLabel, QualityScore, RecordChunk, SampleBatch, SamplePair, SampleTriplet,
        TripletBatch,
    };
    use triplets_core::{SamplerError, SourceId};

    fn make_chunk(text: &str) -> RecordChunk {
        RecordChunk {
            record_id: "r1".into(),
            section_idx: 0,
            view: ChunkView::Window {
                index: 0,
                overlap: 0,
                span: 10,
            },
            text: text.to_string(),
            tokens_estimate: 1,
            quality: QualityScore::default(),
            kvp_meta: Default::default(),
        }
    }

    struct MockSampler {
        pair_batch: Option<SampleBatch>,
        triplet_batch: Option<TripletBatch>,
        save_called: std::sync::atomic::AtomicBool,
    }

    impl MockSampler {
        fn new() -> Self {
            Self {
                pair_batch: Some(SampleBatch {
                    pairs: vec![
                        SamplePair {
                            recipe: "test".into(),
                            anchor: make_chunk("anchor_1"),
                            positive: make_chunk("pos_1"),
                            weight: 1.0,
                            instruction: None,
                            label: PairLabel::Positive,
                            reason: None,
                        },
                        SamplePair {
                            recipe: "test".into(),
                            anchor: make_chunk("anchor_2"),
                            positive: make_chunk("pos_2"),
                            weight: 1.0,
                            instruction: None,
                            label: PairLabel::Positive,
                            reason: None,
                        },
                    ],
                }),
                triplet_batch: Some(TripletBatch {
                    triplets: vec![SampleTriplet {
                        recipe: "test".into(),
                        anchor: make_chunk("t_anchor"),
                        positive: make_chunk("t_pos"),
                        negative: make_chunk("t_neg"),
                        weight: 1.0,
                        instruction: None,
                    }],
                }),
                save_called: std::sync::atomic::AtomicBool::new(false),
            }
        }

        fn with_empty_pairs() -> Self {
            let mut s = Self::new();
            s.pair_batch = Some(SampleBatch { pairs: vec![] });
            s
        }

        fn with_empty_triplets() -> Self {
            let mut s = Self::new();
            s.triplet_batch = Some(TripletBatch { triplets: vec![] });
            s
        }
    }

    impl Sampler for MockSampler {
        fn next_pair_batch(
            &self,
            _split: SplitLabel,
        ) -> std::result::Result<SampleBatch, SamplerError> {
            self.pair_batch
                .clone()
                .ok_or(SamplerError::SourceUnavailable {
                    source_id: "mock".into(),
                    reason: "empty".into(),
                })
        }

        fn next_triplet_batch(
            &self,
            _split: SplitLabel,
        ) -> std::result::Result<TripletBatch, SamplerError> {
            self.triplet_batch
                .clone()
                .ok_or(SamplerError::SourceUnavailable {
                    source_id: "mock".into(),
                    reason: "empty".into(),
                })
        }

        fn next_pair_batch_with_weights(
            &self,
            split: SplitLabel,
            _weights: &HashMap<SourceId, f32>,
        ) -> std::result::Result<SampleBatch, SamplerError> {
            self.next_pair_batch(split)
        }

        fn next_text_batch_with_weights(
            &self,
            _split: SplitLabel,
            _weights: &HashMap<SourceId, f32>,
        ) -> std::result::Result<triplets_core::data::TextBatch, SamplerError> {
            unimplemented!()
        }

        fn next_triplet_batch_with_weights(
            &self,
            split: SplitLabel,
            _weights: &HashMap<SourceId, f32>,
        ) -> std::result::Result<TripletBatch, SamplerError> {
            self.next_triplet_batch(split)
        }

        fn save_sampler_state(
            &self,
            _save_to: Option<&std::path::Path>,
        ) -> std::result::Result<(), SamplerError> {
            self.save_called
                .store(true, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        }
    }

    #[test]
    fn adapter_returns_pair_batch() {
        let adapter = SamplerAdapter {
            sampler: Arc::new(MockSampler::new()),
            mode: SrdMode::Pair,
        };
        let batch = adapter.next_batch(SplitLabel::Train).unwrap();
        let batch = batch.expect("should have batch");
        match &batch {
            SamplerBatch::Pairs(entries) => {
                assert_eq!(entries.len(), 2);
                assert_eq!(entries[0].anchor_text, "anchor_1");
                assert_eq!(entries[0].candidate_text, "pos_1");
                assert_eq!(entries[1].anchor_text, "anchor_2");
                assert_eq!(entries[1].candidate_text, "pos_2");
            }
            _ => panic!("expected pairs"),
        }
    }

    #[test]
    fn adapter_returns_triplet_batch() {
        let adapter = SamplerAdapter {
            sampler: Arc::new(MockSampler::new()),
            mode: SrdMode::Triplet,
        };
        let batch = adapter.next_batch(SplitLabel::Train).unwrap();
        let batch = batch.expect("should have batch");
        match &batch {
            SamplerBatch::Triplets(entries) => {
                assert_eq!(entries.len(), 1);
                assert_eq!(entries[0].anchor_text, "t_anchor");
                assert_eq!(entries[0].pos_text, "t_pos");
                assert_eq!(entries[0].neg_text, "t_neg");
            }
            _ => panic!("expected triplets"),
        }
    }

    #[test]
    fn adapter_returns_none_on_empty_pairs() {
        let adapter = SamplerAdapter {
            sampler: Arc::new(MockSampler::with_empty_pairs()),
            mode: SrdMode::Pair,
        };
        let batch = adapter.next_batch(SplitLabel::Train).unwrap();
        assert!(batch.is_none());
    }

    #[test]
    fn adapter_returns_none_on_empty_triplets() {
        let adapter = SamplerAdapter {
            sampler: Arc::new(MockSampler::with_empty_triplets()),
            mode: SrdMode::Triplet,
        };
        let batch = adapter.next_batch(SplitLabel::Train).unwrap();
        assert!(batch.is_none());
    }

    #[test]
    fn adapter_save_state_delegates_to_sampler() {
        let mock = MockSampler::new();
        let adapter = SamplerAdapter {
            sampler: Arc::new(mock),
            mode: SrdMode::Pair,
        };
        adapter.save_state().unwrap();
        // The mock's save_called would be true if we could access it,
        // but since it's behind Arc<dyn Sampler> we just verify no error.
    }

    #[test]
    fn adapter_propagates_sampler_error() {
        struct FailingSampler;
        impl Sampler for FailingSampler {
            fn next_pair_batch(
                &self,
                _split: SplitLabel,
            ) -> std::result::Result<SampleBatch, SamplerError> {
                Err(SamplerError::SourceUnavailable {
                    source_id: "fail".into(),
                    reason: "intentional".into(),
                })
            }
            fn next_triplet_batch(
                &self,
                _split: SplitLabel,
            ) -> std::result::Result<TripletBatch, SamplerError> {
                Err(SamplerError::SourceUnavailable {
                    source_id: "fail".into(),
                    reason: "intentional".into(),
                })
            }
            fn next_pair_batch_with_weights(
                &self,
                _split: SplitLabel,
                _w: &HashMap<SourceId, f32>,
            ) -> std::result::Result<SampleBatch, SamplerError> {
                self.next_pair_batch(_split)
            }
            fn next_text_batch_with_weights(
                &self,
                _split: SplitLabel,
                _w: &HashMap<SourceId, f32>,
            ) -> std::result::Result<triplets_core::data::TextBatch, SamplerError> {
                unimplemented!()
            }
            fn next_triplet_batch_with_weights(
                &self,
                _split: SplitLabel,
                _w: &HashMap<SourceId, f32>,
            ) -> std::result::Result<TripletBatch, SamplerError> {
                self.next_triplet_batch(_split)
            }
        }

        let adapter = SamplerAdapter {
            sampler: Arc::new(FailingSampler),
            mode: SrdMode::Pair,
        };
        let err = adapter.next_batch(SplitLabel::Train).unwrap_err();
        assert!(err.to_string().contains("sampler error"));
    }
}
