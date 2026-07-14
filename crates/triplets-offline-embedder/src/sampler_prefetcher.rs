use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

use triplets_core::SplitLabel;
use triplets_core::data::{SamplePair, SampleTriplet};

use crate::split_scheduler::is_exhaustion_error;
use crate::traits::{BatchProvider, PairEntry, Result, SamplerBatch, SchedulerError, TripletEntry};

/// Jointly filter a pair batch: skip any pair where EITHER text is empty
/// so anchor and positive vectors always have the same length.
pub fn filter_pair_batch(pairs: &[SamplePair]) -> Vec<PairEntry> {
    pairs
        .iter()
        .filter(|p| !p.anchor.text.is_empty() && !p.positive.text.is_empty())
        .map(|p| PairEntry {
            anchor_text: p.anchor.text.clone(),
            candidate_text: p.positive.text.clone(),
            label: p.label.clone(),
        })
        .collect()
}

/// Jointly filter a triplet batch: skip any triplet where ANY text is empty
/// so anchor, positive, and negative vectors always have the same length.
pub fn filter_triplet_batch(triplets: &[SampleTriplet]) -> Vec<TripletEntry> {
    triplets
        .iter()
        .filter(|t| {
            !t.anchor.text.is_empty() && !t.positive.text.is_empty() && !t.negative.text.is_empty()
        })
        .map(|t| TripletEntry {
            anchor_text: t.anchor.text.clone(),
            pos_text: t.positive.text.clone(),
            neg_text: t.negative.text.clone(),
        })
        .collect()
}

/// Background-thread prefetcher that calls the sampler's pair or triplet API
/// in a background thread, sending batches through a bounded channel.
pub struct SamplerPrefetcher<P: BatchProvider + 'static> {
    rx: Option<mpsc::Receiver<Result<SamplerBatch>>>,
    handle: Option<std::thread::JoinHandle<()>>,
    stop: Arc<AtomicBool>,
    _provider: std::sync::Arc<P>,
}

impl<P: BatchProvider + 'static> SamplerPrefetcher<P> {
    /// Spawn a background prefetcher thread.
    ///
    /// * `provider` — the batch provider (must be `Send + Sync + 'static`).
    /// * `split` — which split to prefetch batches for.
    /// * `queue_cap` — max buffered batches (back-pressure, at least 1).
    pub fn new(provider: Arc<P>, split: SplitLabel, queue_cap: usize) -> Self {
        let (tx, rx) = mpsc::sync_channel(queue_cap.max(1));
        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = Arc::clone(&stop);
        let provider_clone = Arc::clone(&provider);

        let handle = std::thread::Builder::new()
            .name(format!("sampler-prefetch-{split:?}"))
            .spawn(move || {
                loop {
                    if stop_clone.load(Ordering::Relaxed) {
                        break;
                    }

                    match provider_clone.next_batch(split) {
                        Ok(Some(batch)) => {
                            if match &batch {
                                SamplerBatch::Pairs(v) => v.is_empty(),
                                SamplerBatch::Triplets(v) => v.is_empty(),
                            } {
                                let _ = tx.send(Err(SchedulerError::Msg("exhausted".into())));
                                break;
                            }
                            if tx.send(Ok(batch)).is_err() {
                                break;
                            }
                        }
                        Ok(None) => {
                            let _ = tx.send(Err(SchedulerError::Msg("exhausted".into())));
                            break;
                        }
                        Err(e) => {
                            let msg = e.to_string();
                            if is_exhaustion_error(&msg) {
                                let _ = tx.send(Err(SchedulerError::Msg("exhausted".into())));
                            } else {
                                let _ = tx.send(Err(e));
                            }
                            break;
                        }
                    }
                }
            })
            .expect("spawn sampler prefetcher");

        Self {
            rx: Some(rx),
            handle: Some(handle),
            stop,
            _provider: provider,
        }
    }

    /// Pop the next prefetched batch (blocking if none are ready yet).
    pub fn next(&self) -> Result<SamplerBatch> {
        let rx = self
            .rx
            .as_ref()
            .ok_or_else(|| SchedulerError::Msg("sampler prefetcher stopped".into()))?;
        match rx.recv() {
            Ok(Ok(batch)) => Ok(batch),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(SchedulerError::Msg(
                "sampler prefetcher channel closed".into(),
            )),
        }
    }
}

impl<P: BatchProvider + 'static> Drop for SamplerPrefetcher<P> {
    fn drop(&mut self) {
        // 1. Signal the background thread to stop.
        self.stop.store(true, Ordering::Relaxed);
        // 2. Drop receiver to unblock any pending send() in the background thread.
        self.rx.take();
        // 3. Join the thread — unwrap to propagate panics from the background thread.
        //    A panicked worker indicates a fatal pipeline state (OOM, provider panic).
        if let Some(handle) = self.handle.take() {
            handle.join().unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use triplets_core::data::{ChunkView, PairLabel, QualityScore, RecordChunk};

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

    struct MockProvider {
        batches: std::sync::Mutex<Vec<SamplerBatch>>,
        call_count: AtomicUsize,
    }

    impl MockProvider {
        fn new(batches: Vec<SamplerBatch>) -> Self {
            Self {
                batches: std::sync::Mutex::new(batches),
                call_count: AtomicUsize::new(0),
            }
        }
    }

    impl BatchProvider for MockProvider {
        fn next_batch(&self, _split: SplitLabel) -> Result<Option<SamplerBatch>> {
            let idx = self.call_count.fetch_add(1, Ordering::Relaxed);
            let mut batches = self.batches.lock().unwrap();
            if idx < batches.len() {
                Ok(Some(batches.remove(0)))
            } else {
                Ok(None)
            }
        }

        fn save_state(&self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn filter_pair_batch_skips_empty() {
        let pairs = vec![
            SamplePair {
                recipe: "test".into(),
                anchor: make_chunk("hello"),
                positive: make_chunk("world"),
                weight: 1.0,
                instruction: None,
                label: PairLabel::Positive,
                reason: None,
            },
            SamplePair {
                recipe: "test".into(),
                anchor: make_chunk(""),
                positive: make_chunk("world"),
                weight: 1.0,
                instruction: None,
                label: PairLabel::Positive,
                reason: None,
            },
        ];
        let pair_entries = filter_pair_batch(&pairs);
        let (anchors, positions): (Vec<_>, Vec<_>) = pair_entries
            .into_iter()
            .map(|e| (e.anchor_text, e.candidate_text))
            .unzip();
        assert_eq!(anchors, vec!["hello"]);
        assert_eq!(positions, vec!["world"]);
    }

    #[test]
    fn filter_triplet_batch_skips_empty() {
        let triplets = vec![
            SampleTriplet {
                recipe: "test".into(),
                anchor: make_chunk("a"),
                positive: make_chunk("p"),
                negative: make_chunk("n"),
                weight: 1.0,
                instruction: None,
            },
            SampleTriplet {
                recipe: "test".into(),
                anchor: make_chunk("a2"),
                positive: make_chunk(""),
                negative: make_chunk("n2"),
                weight: 1.0,
                instruction: None,
            },
        ];
        let entries = filter_triplet_batch(&triplets);
        let anchors: Vec<String> = entries.iter().map(|e| e.anchor_text.clone()).collect();
        let positions: Vec<String> = entries.iter().map(|e| e.pos_text.clone()).collect();
        let negatives: Vec<String> = entries.iter().map(|e| e.neg_text.clone()).collect();
        assert_eq!(anchors, vec!["a"]);
        assert_eq!(positions, vec!["p"]);
        assert_eq!(negatives, vec!["n"]);
    }

    #[test]
    fn prefetcher_returns_batches_then_exhausted() {
        let batch = SamplerBatch::Pairs(
            ["a".into()]
                .into_iter()
                .zip(["p".into()])
                .map(|(a, c)| PairEntry {
                    anchor_text: a,
                    candidate_text: c,
                    label: PairLabel::Positive,
                })
                .collect(),
        );
        let provider = Arc::new(MockProvider::new(vec![batch]));
        let prefetcher = SamplerPrefetcher::new(provider, SplitLabel::Train, 4);

        let result = prefetcher.next().unwrap();
        match result {
            SamplerBatch::Pairs(pairs) => {
                let anchor_texts: Vec<&str> =
                    pairs.iter().map(|e| e.anchor_text.as_str()).collect();
                assert_eq!(anchor_texts, vec!["a"]);
            }
            _ => panic!("expected Pairs batch"),
        }

        let err = prefetcher.next().unwrap_err();
        assert!(err.to_string().contains("exhausted"));
    }

    // -----------------------------------------------------------------------
    // Additional filter_pair_batch tests
    // -----------------------------------------------------------------------

    #[test]
    fn filter_pair_batch_skips_empty_positive() {
        let pairs = vec![SamplePair {
            recipe: "test".into(),
            anchor: make_chunk("a1"),
            positive: make_chunk(""),
            weight: 1.0,
            instruction: None,
            label: PairLabel::Positive,
            reason: None,
        }];
        let pair_entries = filter_pair_batch(&pairs);
        let (anchors, positions): (Vec<_>, Vec<_>) = pair_entries
            .into_iter()
            .map(|e| (e.anchor_text, e.candidate_text))
            .unzip();
        assert!(anchors.is_empty());
        assert!(positions.is_empty());
    }

    #[test]
    fn filter_pair_batch_skips_both_empty() {
        let pairs = vec![SamplePair {
            recipe: "test".into(),
            anchor: make_chunk(""),
            positive: make_chunk(""),
            weight: 1.0,
            instruction: None,
            label: PairLabel::Positive,
            reason: None,
        }];
        let pair_entries = filter_pair_batch(&pairs);
        let (anchors, positions): (Vec<_>, Vec<_>) = pair_entries
            .into_iter()
            .map(|e| (e.anchor_text, e.candidate_text))
            .unzip();
        assert!(anchors.is_empty());
        assert!(positions.is_empty());
    }

    #[test]
    fn filter_pair_batch_keeps_all_nonempty() {
        let pairs = vec![
            SamplePair {
                recipe: "test".into(),
                anchor: make_chunk("a1"),
                positive: make_chunk("p1"),
                weight: 1.0,
                instruction: None,
                label: PairLabel::Positive,
                reason: None,
            },
            SamplePair {
                recipe: "test".into(),
                anchor: make_chunk("a2"),
                positive: make_chunk("p2"),
                weight: 1.0,
                instruction: None,
                label: PairLabel::Positive,
                reason: None,
            },
        ];
        let pair_entries = filter_pair_batch(&pairs);
        let (anchors, positions): (Vec<_>, Vec<_>) = pair_entries
            .into_iter()
            .map(|e| (e.anchor_text, e.candidate_text))
            .unzip();
        assert_eq!(anchors, vec!["a1", "a2"]);
        assert_eq!(positions, vec!["p1", "p2"]);
    }

    #[test]
    fn filter_pair_batch_empty_input() {
        let pairs: Vec<SamplePair> = vec![];
        let pair_entries = filter_pair_batch(&pairs);
        let (anchors, positions): (Vec<_>, Vec<_>) = pair_entries
            .into_iter()
            .map(|e| (e.anchor_text, e.candidate_text))
            .unzip();
        assert!(anchors.is_empty());
        assert!(positions.is_empty());
    }

    // -----------------------------------------------------------------------
    // Additional filter_triplet_batch tests
    // -----------------------------------------------------------------------

    #[test]
    fn filter_triplet_batch_skips_empty_positive() {
        let triplets = vec![SampleTriplet {
            recipe: "test".into(),
            anchor: make_chunk("a"),
            positive: make_chunk(""),
            negative: make_chunk("n"),
            weight: 1.0,
            instruction: None,
        }];
        let entries = filter_triplet_batch(&triplets);
        let anchors: Vec<String> = entries.iter().map(|e| e.anchor_text.clone()).collect();
        let positions: Vec<String> = entries.iter().map(|e| e.pos_text.clone()).collect();
        let negatives: Vec<String> = entries.iter().map(|e| e.neg_text.clone()).collect();
        assert!(anchors.is_empty());
        assert!(positions.is_empty());
        assert!(negatives.is_empty());
    }

    #[test]
    fn filter_triplet_batch_skips_empty_negative() {
        let triplets = vec![SampleTriplet {
            recipe: "test".into(),
            anchor: make_chunk("a"),
            positive: make_chunk("p"),
            negative: make_chunk(""),
            weight: 1.0,
            instruction: None,
        }];
        let entries = filter_triplet_batch(&triplets);
        let anchors: Vec<String> = entries.iter().map(|e| e.anchor_text.clone()).collect();
        let positions: Vec<String> = entries.iter().map(|e| e.pos_text.clone()).collect();
        let negatives: Vec<String> = entries.iter().map(|e| e.neg_text.clone()).collect();
        assert!(anchors.is_empty());
        assert!(positions.is_empty());
        assert!(negatives.is_empty());
    }

    #[test]
    fn filter_triplet_batch_skips_all_empty() {
        let triplets = vec![SampleTriplet {
            recipe: "test".into(),
            anchor: make_chunk(""),
            positive: make_chunk(""),
            negative: make_chunk(""),
            weight: 1.0,
            instruction: None,
        }];
        let entries = filter_triplet_batch(&triplets);
        let anchors: Vec<String> = entries.iter().map(|e| e.anchor_text.clone()).collect();
        let positions: Vec<String> = entries.iter().map(|e| e.pos_text.clone()).collect();
        let negatives: Vec<String> = entries.iter().map(|e| e.neg_text.clone()).collect();
        assert!(anchors.is_empty());
        assert!(positions.is_empty());
        assert!(negatives.is_empty());
    }

    #[test]
    fn filter_triplet_batch_keeps_all_nonempty() {
        let triplets = vec![
            SampleTriplet {
                recipe: "test".into(),
                anchor: make_chunk("a1"),
                positive: make_chunk("p1"),
                negative: make_chunk("n1"),
                weight: 1.0,
                instruction: None,
            },
            SampleTriplet {
                recipe: "test".into(),
                anchor: make_chunk("a2"),
                positive: make_chunk("p2"),
                negative: make_chunk("n2"),
                weight: 1.0,
                instruction: None,
            },
        ];
        let entries = filter_triplet_batch(&triplets);
        let anchors: Vec<String> = entries.iter().map(|e| e.anchor_text.clone()).collect();
        let positions: Vec<String> = entries.iter().map(|e| e.pos_text.clone()).collect();
        let negatives: Vec<String> = entries.iter().map(|e| e.neg_text.clone()).collect();
        assert_eq!(anchors, vec!["a1", "a2"]);
        assert_eq!(positions, vec!["p1", "p2"]);
        assert_eq!(negatives, vec!["n1", "n2"]);
    }

    #[test]
    fn filter_triplet_batch_empty_input() {
        let triplets: Vec<SampleTriplet> = vec![];
        let entries = filter_triplet_batch(&triplets);
        let anchors: Vec<String> = entries.iter().map(|e| e.anchor_text.clone()).collect();
        let positions: Vec<String> = entries.iter().map(|e| e.pos_text.clone()).collect();
        let negatives: Vec<String> = entries.iter().map(|e| e.neg_text.clone()).collect();
        assert!(anchors.is_empty());
        assert!(positions.is_empty());
        assert!(negatives.is_empty());
    }

    #[test]
    fn prefetcher_drop_joins_thread() {
        use std::time::Duration;

        // Provider that returns 3 batches then exhausted.
        let batch = SamplerBatch::Pairs(
            ["a".into()]
                .into_iter()
                .zip(["p".into()])
                .map(|(a, c)| PairEntry {
                    anchor_text: a,
                    candidate_text: c,
                    label: PairLabel::Positive,
                })
                .collect(),
        );
        let provider = Arc::new(MockProvider::new(vec![batch.clone(), batch.clone(), batch]));
        let prefetcher = SamplerPrefetcher::new(provider, SplitLabel::Train, 4);

        // Consume one batch so the background thread is mid-work.
        let _ = prefetcher.next();

        // Drop the prefetcher — must complete within 1 second if thread is joined.
        let start = std::time::Instant::now();
        drop(prefetcher);
        let elapsed = start.elapsed();

        assert!(
            elapsed < Duration::from_secs(1),
            "prefetcher drop took {elapsed:?} — thread may not have been joined"
        );
    }
}
