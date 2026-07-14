//! The interleaved embedding loop: picks the most-behind split, fetches a
//! batch, embeds it, flushes periodically, and reports progress via a
//! [`LoopHandler`].

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use triplets_core::SplitLabel;

use crate::orchestration::{flush_all_pending_states, flush_batch, handle_split_exhausted};
use crate::sampler_prefetcher::SamplerPrefetcher;
use crate::split_scheduler::SplitScheduler;
use crate::split_scheduler::{
    at_sample_limit, compute_deficit_str, compute_global_in_flight, compute_samples_per_sec,
    is_exhaustion_error, steps_until_next_flush,
};
use crate::split_state::{PendingState, SplitState};
use crate::traits::{BatchProvider, EmbedStore, Embedder, Result, SamplerBatch, SchedulerConfig};

// ---------------------------------------------------------------------------
// Event types
// ---------------------------------------------------------------------------

/// A point-in-time snapshot of a split's counters, for reporting.
pub struct StateSnapshot {
    /// Split display name.
    pub name: &'static str,
    /// Samples durably written to disk.
    pub total_written: u64,
    /// Samples dropped due to embed/validation failures.
    pub dropped_samples: u64,
    /// Batches dropped due to embed/validation failures.
    pub dropped_batches: u64,
}

/// Events emitted by [`run_interleaved_loop`].
pub enum LoopEvent {
    /// The scheduler selected a new split.
    SplitChange {
        /// Split display name.
        name: &'static str,
        /// Pre-formatted announcement string from the caller.
        announcement: String,
    },
    /// A step completed — progress metrics available.
    Step {
        /// Split display name.
        name: &'static str,
        /// Current step number.
        step_num: u64,
        /// Samples in flight (written + pending).
        in_flight: u64,
        /// Target max (0 = unlimited).
        max: u64,
        /// Throughput estimate.
        samples_per_sec: f64,
        /// Pre-formatted ratio/deficit string from the caller.
        ratio_str: String,
        /// Steps until next scheduled flush.
        steps_until_flush: u64,
        /// Samples dropped this step (0 if none).
        dropped: u64,
    },
    /// Pending buffer was flushed to disk.
    BatchFlushed {
        /// Split name.
        name: &'static str,
        /// Flush sequence number.
        batch_num: u64,
        /// Number of samples flushed.
        flushed: u64,
        /// Total samples written so far.
        total: u64,
    },
    /// Sampler has no more data for this split.
    Exhausted {
        /// Split name.
        name: &'static str,
    },
    /// Ctrl+C received — the loop has flushed all pending and is returning.
    CtrlC {
        /// Per-split state snapshots at the time of interrupt.
        states: Vec<StateSnapshot>,
    },
    /// The loop finished normally — all splits done.
    Complete {
        /// Per-split state snapshots at completion.
        states: Vec<StateSnapshot>,
        /// Wall-clock seconds elapsed.
        elapsed_secs: f64,
    },
}

/// Handler for loop events.  Implement this to receive progress and
/// notification callbacks from [`run_interleaved_loop`].
pub trait LoopHandler {
    /// Called when the loop emits an event.
    fn handle_event(&mut self, event: &LoopEvent);
}

// ---------------------------------------------------------------------------
// Announcement helper
// ---------------------------------------------------------------------------

/// Computes the split-change announcement string.
///
/// Call this from your [`LoopHandler`] or before emitting
/// [`LoopEvent::SplitChange`] to produce the `announcement` field.
pub fn compute_split_announcement(
    name: &str,
    in_flight: u64,
    max: u64,
    ratio: f32,
    ratio_sum: f32,
    total_in_flight: u64,
) -> String {
    let fair_share = if ratio_sum > 0.0 {
        (total_in_flight as f64 * ratio as f64 / ratio_sum as f64).ceil() as u64
    } else {
        0
    };
    let deficit = fair_share.saturating_sub(in_flight);
    let cap_str = if max > 0 {
        format!("{}/{}", in_flight, max)
    } else {
        format!("{} written", in_flight)
    };
    let pct = if ratio_sum > 0.0 {
        ratio as f64 / ratio_sum as f64 * 100.0
    } else {
        0.0
    };
    if deficit > 0 {
        format!(
            "precompute: \u{2500}\u{2500} now filling split [{}] ({}, needs +{deficit} to reach {pct:.0}% target) \u{2500}\u{2500}",
            name, cap_str,
        )
    } else {
        format!(
            "precompute: \u{2500}\u{2500} now filling split [{}] ({}, at or ahead of {pct:.0}% target) \u{2500}\u{2500}",
            name, cap_str,
        )
    }
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

/// Run the interleaved embedding loop.
///
/// Each iteration the [`SplitScheduler`] picks whichever split is most behind
/// its proportional target.  The loop fetches a batch from the corresponding
/// prefetcher, embeds it via `embedder`, and flushes periodically.
///
/// Progress and lifecycle events are reported through `handler`.
///
/// The loop exits when:
/// - all splits are exhausted or at their limit, or
/// - `stop` is set (Ctrl+C) — in this case all pending buffers are flushed
///   before returning.
pub fn run_interleaved_loop<S, E, P>(
    states: &mut [SplitState<S>],
    prefetchers: &mut HashMap<SplitLabel, SamplerPrefetcher<P>>,
    embedder: &E,
    provider: &P,
    config: &SchedulerConfig,
    stop: &AtomicBool,
    handler: &mut dyn LoopHandler,
) -> Result<()>
where
    S: EmbedStore,
    E: Embedder,
    P: BatchProvider,
{
    let mut scheduler = SplitScheduler::new();
    let global_start = Instant::now();

    loop {
        // Recompute in-flight counts every iteration.
        let labels_vec: Vec<SplitLabel> = states.iter().map(|s| s.label).collect();
        let counts: Vec<u64> = states
            .iter()
            .map(|s| s.total_written + s.pending.len() as u64)
            .collect();
        let maxes: Vec<u64> = states.iter().map(|s| s.max).collect();
        let ratios_vec: Vec<f32> = states.iter().map(|s| s.ratio).collect();
        let exhausted_vec: Vec<bool> = states.iter().map(|s| s.exhausted).collect();

        let Some((label, newly_selected)) =
            scheduler.next(&labels_vec, &counts, &maxes, &ratios_vec, &exhausted_vec)
        else {
            break; // All splits done.
        };

        if newly_selected {
            let s = states
                .iter_mut()
                .find(|s| s.label == label)
                .expect("split label not found");
            s.on_stint_start();
            debug_assert!(
                s.pending.is_empty(),
                "pending must be empty when starting a new batch for split {}",
                s.name,
            );
            let in_flight = s.total_written;
            let total_in_flight: u64 = counts.iter().sum();
            let ratio_sum: f32 = ratios_vec.iter().sum();
            let announcement = compute_split_announcement(
                s.name,
                in_flight,
                s.max,
                s.ratio,
                ratio_sum,
                total_in_flight,
            );
            handler.handle_event(&LoopEvent::SplitChange {
                name: s.name,
                announcement,
            });
        }

        let s = states
            .iter_mut()
            .find(|s| s.label == label)
            .expect("split label not found");

        // Fetch next batch from the prefetcher.
        let batch = match prefetchers
            .get_mut(&s.label)
            .expect("prefetcher for split")
            .next()
        {
            Ok(b) => b,
            Err(e) => {
                let msg = e.to_string();
                if is_exhaustion_error(&msg) {
                    handler.handle_event(&LoopEvent::Exhausted { name: s.name });
                    handle_split_exhausted(s, provider, &mut scheduler)?;
                    continue;
                }
                return Err(e);
            }
        };

        if match &batch {
            SamplerBatch::Pairs(v) => v.is_empty(),
            SamplerBatch::Triplets(v) => v.is_empty(),
        } {
            handler.handle_event(&LoopEvent::Exhausted { name: s.name });
            handle_split_exhausted(s, provider, &mut scheduler)?;
            continue;
        }

        let step_result = s.step(batch, embedder, config)?;

        let ctrl_c = stop.load(Ordering::Relaxed);
        let hit_limit = at_sample_limit(s.max, s.total_written, s.pending.len() as u64);
        let should_flush = step_result.should_flush || ctrl_c || hit_limit;

        if should_flush {
            let flushed_count = flush_batch(s, provider, &mut scheduler)?;
            if flushed_count > 0 {
                handler.handle_event(&LoopEvent::BatchFlushed {
                    name: s.name,
                    batch_num: s.batch_num,
                    flushed: flushed_count,
                    total: s.total_written,
                });
            }
        }

        if ctrl_c {
            flush_all_pending_states(states, provider)?;
            let snapshots: Vec<StateSnapshot> = states
                .iter()
                .map(|s| StateSnapshot {
                    name: s.name,
                    total_written: s.total_written,
                    dropped_samples: s.dropped_samples,
                    dropped_batches: s.dropped_batches,
                })
                .collect();
            handler.handle_event(&LoopEvent::CtrlC { states: snapshots });
            return Ok(());
        }

        // Progress metrics.
        let elapsed = s.start.map_or(0.0, |t| t.elapsed().as_secs_f64());
        let in_flight = s.total_written + s.pending.len() as u64;
        let new_samples = in_flight.saturating_sub(s.segment_base);
        let samples_per_sec = compute_samples_per_sec(new_samples, elapsed);
        let steps_until_flush = steps_until_next_flush(s.step_num, config.steps_per_batch);

        let label_pos = labels_vec
            .iter()
            .position(|&l| l == label)
            .expect("split label not found");
        let global_in_flight = compute_global_in_flight(&counts, label_pos, in_flight);
        let ratio_sum: f32 = ratios_vec.iter().sum();
        let ratio_str = compute_deficit_str(in_flight, global_in_flight, s.ratio, ratio_sum);

        handler.handle_event(&LoopEvent::Step {
            name: s.name,
            step_num: s.step_num,
            in_flight,
            max: s.max,
            samples_per_sec,
            ratio_str,
            steps_until_flush,
            dropped: step_result.samples_dropped,
        });

        if hit_limit {
            s.exhausted = true;
            scheduler.unlock();
        }
    }

    // Post-loop: flush remaining.
    flush_all_pending_states(states, provider)?;
    let snapshots: Vec<StateSnapshot> = states
        .iter()
        .map(|s| StateSnapshot {
            name: s.name,
            total_written: s.total_written,
            dropped_samples: s.dropped_samples,
            dropped_batches: s.dropped_batches,
        })
        .collect();
    let elapsed_secs = global_start.elapsed().as_secs_f64();
    handler.handle_event(&LoopEvent::Complete {
        states: snapshots,
        elapsed_secs,
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler_prefetcher::SamplerPrefetcher;
    use crate::split_state::EmbedMode;
    use crate::traits::{PairEntry, PairWriteArgs, SamplerBatch, SchedulerError, TripletWriteArgs};
    use std::sync::Arc;
    use triplets_core::data::PairLabel;

    // -- Mock types --

    struct MockStore;

    impl EmbedStore for MockStore {
        fn write_pairs(&self, _start_idx: u64, _args: &PairWriteArgs<'_>) -> Result<()> {
            Ok(())
        }
        fn write_triplets(&self, _start_idx: u64, _args: &TripletWriteArgs<'_>) -> Result<()> {
            Ok(())
        }
        fn len(&self) -> Result<u64> {
            Ok(0)
        }
    }

    struct MockEmbedder;

    impl Embedder for MockEmbedder {
        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![1.0; 4]).collect())
        }
    }

    /// Returns N batches then exhausted.
    struct MockProvider {
        batches_remaining: std::sync::atomic::AtomicUsize,
        save_count: std::sync::atomic::AtomicUsize,
    }

    impl MockProvider {
        fn new(n: usize) -> Self {
            Self {
                batches_remaining: std::sync::atomic::AtomicUsize::new(n),
                save_count: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        fn saves(&self) -> usize {
            self.save_count.load(Ordering::Relaxed)
        }
    }

    impl BatchProvider for MockProvider {
        fn next_batch(&self, _split: SplitLabel) -> Result<Option<SamplerBatch>> {
            let remaining = self.batches_remaining.fetch_sub(1, Ordering::Relaxed);
            if remaining == 0 {
                return Ok(None);
            }
            Ok(Some(SamplerBatch::Pairs(vec![PairEntry {
                anchor_text: "hello".into(),
                candidate_text: "world".into(),
                label: PairLabel::Positive,
            }])))
        }
        fn save_state(&self) -> Result<()> {
            self.save_count.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    fn make_state(name: &'static str) -> SplitState<MockStore> {
        SplitState {
            label: SplitLabel::Train,
            name,
            store: MockStore,
            mode: EmbedMode::Pair,
            emb_dim: 4,
            max: 0,
            ratio: 1.0,
            total_written: 0,
            step_num: 0,
            batch_num: 0,
            pending: PendingState::Pairs(Vec::new()),
            exhausted: false,
            dropped_batches: 0,
            total_batches: 0,
            dropped_samples: 0,
            start: None,
            segment_base: 0,
        }
    }

    /// Collects all events into a Vec for assertions.
    struct EventCollector {
        events: Vec<String>,
    }

    impl EventCollector {
        fn new() -> Self {
            Self { events: Vec::new() }
        }
    }

    impl LoopHandler for EventCollector {
        fn handle_event(&mut self, event: &LoopEvent) {
            match event {
                LoopEvent::SplitChange { name, .. } => {
                    self.events.push(format!("split_change:{name}"));
                }
                LoopEvent::Step { name, dropped, .. } => {
                    if *dropped > 0 {
                        self.events.push(format!("step:{name}:dropped:{dropped}"));
                    } else {
                        self.events.push(format!("step:{name}:ok"));
                    }
                }
                LoopEvent::BatchFlushed { name, flushed, .. } => {
                    self.events.push(format!("flush:{name}:{flushed}"));
                }
                LoopEvent::Exhausted { name } => {
                    self.events.push(format!("exhausted:{name}"));
                }
                LoopEvent::CtrlC { .. } => {
                    self.events.push("ctrl_c".into());
                }
                LoopEvent::Complete { states, .. } => {
                    let total: u64 = states.iter().map(|s| s.total_written).sum();
                    self.events.push(format!("complete:{total}"));
                }
            }
        }
    }

    #[test]
    fn loop_runs_until_exhausted() {
        let mut states = vec![make_state("train")];
        let provider = Arc::new(MockProvider::new(3));
        let embedder = MockEmbedder;
        let config = SchedulerConfig::new(1, 1, 4, 100);

        let mut prefetchers = HashMap::new();
        prefetchers.insert(
            SplitLabel::Train,
            SamplerPrefetcher::new(provider.clone(), SplitLabel::Train, 2),
        );

        let stop = AtomicBool::new(false);
        let mut handler = EventCollector::new();

        run_interleaved_loop(
            &mut states,
            &mut prefetchers,
            &embedder,
            provider.as_ref(),
            &config,
            &stop,
            &mut handler,
        )
        .unwrap();

        assert!(handler.events.iter().any(|e| e.starts_with("exhausted:")));
        assert!(handler.events.iter().any(|e| e.starts_with("complete:")));
        // save_state called: 1x on exhaustion (handle_split_exhausted) + 1x on final flush
        assert!(
            provider.saves() >= 1,
            "save_state must be called at least once, got {}",
            provider.saves()
        );
    }

    #[test]
    fn loop_emits_split_change() {
        let mut states = vec![make_state("train")];
        let provider = Arc::new(MockProvider::new(1));
        let embedder = MockEmbedder;
        let config = SchedulerConfig::new(1, 1, 4, 100);

        let mut prefetchers = HashMap::new();
        prefetchers.insert(
            SplitLabel::Train,
            SamplerPrefetcher::new(provider.clone(), SplitLabel::Train, 2),
        );

        let stop = AtomicBool::new(false);
        let mut handler = EventCollector::new();

        run_interleaved_loop(
            &mut states,
            &mut prefetchers,
            &embedder,
            provider.as_ref(),
            &config,
            &stop,
            &mut handler,
        )
        .unwrap();

        assert!(handler.events.contains(&"split_change:train".into()));
    }

    #[test]
    fn loop_stops_on_ctrl_c() {
        let mut states = vec![make_state("train")];
        let provider = Arc::new(MockProvider::new(100)); // many batches
        let embedder = MockEmbedder;
        let config = SchedulerConfig::new(1, 1, 4, 100);

        let mut prefetchers = HashMap::new();
        prefetchers.insert(
            SplitLabel::Train,
            SamplerPrefetcher::new(provider.clone(), SplitLabel::Train, 2),
        );

        let stop = AtomicBool::new(true); // pre-set stop flag
        let mut handler = EventCollector::new();

        run_interleaved_loop(
            &mut states,
            &mut prefetchers,
            &embedder,
            provider.as_ref(),
            &config,
            &stop,
            &mut handler,
        )
        .unwrap();

        assert!(handler.events.contains(&"ctrl_c".into()));
        // Ctrl+C triggers flush_all_pending_states → save_state
        assert!(
            provider.saves() >= 1,
            "save_state must be called on Ctrl+C, got {}",
            provider.saves()
        );
    }

    #[test]
    fn compute_split_announcement_formats_correctly() {
        let msg = compute_split_announcement("train", 50, 100, 0.8, 0.9, 100);
        assert!(msg.contains("train"));
        assert!(msg.contains("50/100"));
    }

    // ------------------------------------------------------------------
    // Coverage gap tests
    // ------------------------------------------------------------------

    /// Provider that returns a single empty batch then exhausted.
    struct EmptyBatchProvider;

    impl BatchProvider for EmptyBatchProvider {
        fn next_batch(&self, _split: SplitLabel) -> Result<Option<SamplerBatch>> {
            Ok(Some(SamplerBatch::Pairs(vec![])))
        }
        fn save_state(&self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn loop_handles_empty_batch_from_prefetcher() {
        let mut states = vec![make_state("train")];
        let provider = Arc::new(EmptyBatchProvider);
        let embedder = MockEmbedder;
        let config = SchedulerConfig::new(1, 1, 4, 100);

        let mut prefetchers = HashMap::new();
        prefetchers.insert(
            SplitLabel::Train,
            SamplerPrefetcher::new(provider.clone(), SplitLabel::Train, 2),
        );

        let stop = AtomicBool::new(false);
        let mut handler = EventCollector::new();

        run_interleaved_loop(
            &mut states,
            &mut prefetchers,
            &embedder,
            provider.as_ref(),
            &config,
            &stop,
            &mut handler,
        )
        .unwrap();

        // Empty batch triggers exhaustion path
        assert!(handler.events.iter().any(|e| e.starts_with("exhausted:")));
        assert!(handler.events.iter().any(|e| e.starts_with("complete:")));
    }

    /// Provider that returns a non-exhaustion error.
    struct ErrorProvider;

    impl BatchProvider for ErrorProvider {
        fn next_batch(&self, _split: SplitLabel) -> Result<Option<SamplerBatch>> {
            Err(SchedulerError::Msg("network timeout".into()))
        }
        fn save_state(&self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn loop_returns_non_exhaustion_error() {
        let mut states = vec![make_state("train")];
        let provider = Arc::new(ErrorProvider);
        let embedder = MockEmbedder;
        let config = SchedulerConfig::new(1, 1, 4, 100);

        let mut prefetchers = HashMap::new();
        prefetchers.insert(
            SplitLabel::Train,
            SamplerPrefetcher::new(provider.clone(), SplitLabel::Train, 2),
        );

        let stop = AtomicBool::new(false);
        let mut handler = EventCollector::new();

        let result = run_interleaved_loop(
            &mut states,
            &mut prefetchers,
            &embedder,
            provider.as_ref(),
            &config,
            &stop,
            &mut handler,
        );

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("network timeout"));
    }

    /// Provider that returns batches indefinitely for testing hit_limit.
    struct InfiniteProvider;

    impl BatchProvider for InfiniteProvider {
        fn next_batch(&self, _split: SplitLabel) -> Result<Option<SamplerBatch>> {
            Ok(Some(SamplerBatch::Pairs(vec![PairEntry {
                anchor_text: "data".into(),
                candidate_text: "positive".into(),
                label: PairLabel::Positive,
            }])))
        }
        fn save_state(&self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn loop_marks_split_at_limit_exhausted() {
        let mut states = vec![make_state("train")];
        states[0].max = 3; // limit of 3 samples

        let provider = Arc::new(InfiniteProvider);
        let embedder = MockEmbedder;
        let config = SchedulerConfig::new(1, 1, 4, 100); // flush every 100 steps

        let mut prefetchers = HashMap::new();
        prefetchers.insert(
            SplitLabel::Train,
            SamplerPrefetcher::new(provider.clone(), SplitLabel::Train, 2),
        );

        let stop = AtomicBool::new(false);
        let mut handler = EventCollector::new();

        run_interleaved_loop(
            &mut states,
            &mut prefetchers,
            &embedder,
            provider.as_ref(),
            &config,
            &stop,
            &mut handler,
        )
        .unwrap();

        // Should complete with the split marked at limit
        assert!(handler.events.iter().any(|e| e.starts_with("complete:")));
        assert!(states[0].exhausted);
        assert!(states[0].total_written >= 3);
    }

    #[test]
    fn compute_split_announcement_no_limit() {
        // max=0 means unlimited
        let msg = compute_split_announcement("val", 100, 0, 0.2, 1.0, 500);
        assert!(msg.contains("val"));
        assert!(msg.contains("100 written"));
        assert!(!msg.contains("/"));
    }

    #[test]
    fn compute_split_announcement_deficit_positive() {
        let msg = compute_split_announcement("train", 10, 200, 0.8, 1.0, 100);
        assert!(msg.contains("needs"));
    }

    #[test]
    fn compute_split_announcement_ratio_sum_zero() {
        let msg = compute_split_announcement("test", 50, 100, 0.0, 0.0, 100);
        assert!(msg.contains("test"));
        assert!(msg.contains("50/100"));
    }

    #[test]
    fn save_state_called_at_every_flush_boundary() {
        // 6 batches, steps_per_batch=2 → flush at steps 2, 4, 6 + exhaustion + final
        // Verify save_state is called at each flush, not just at the end.
        let mut states = vec![make_state("train")];
        let provider = Arc::new(MockProvider::new(6));
        let embedder = MockEmbedder;
        let config = SchedulerConfig::new(1, 1, 4, 2); // flush every 2 steps

        let mut prefetchers = HashMap::new();
        prefetchers.insert(
            SplitLabel::Train,
            SamplerPrefetcher::new(provider.clone(), SplitLabel::Train, 2),
        );

        let stop = AtomicBool::new(false);
        let mut handler = EventCollector::new();

        run_interleaved_loop(
            &mut states,
            &mut prefetchers,
            &embedder,
            provider.as_ref(),
            &config,
            &stop,
            &mut handler,
        )
        .unwrap();

        let saves = provider.saves();
        // Expected saves:
        // - step 2: flush_batch → save_state
        // - step 4: flush_batch → save_state
        // - step 6: flush_batch → save_state
        // - exhaustion: handle_split_exhausted → save_state
        // - post-loop: flush_all_pending_states → save_state (if pending, else skipped)
        assert!(
            saves >= 3,
            "expected >=3 saves (flush boundaries), got {saves}"
        );

        // Also verify BatchFlushed events were emitted at the right cadence
        let flushes: Vec<_> = handler
            .events
            .iter()
            .filter(|e| e.starts_with("flush:"))
            .collect();
        assert!(
            flushes.len() >= 2,
            "expected >=2 flush events, got {}",
            flushes.len()
        );
    }

    #[test]
    fn save_state_called_on_ctrl_c_flush() {
        // Verify save_state is called when Ctrl+C triggers flush_all_pending_states
        let mut states = vec![make_state("train")];
        let provider = Arc::new(MockProvider::new(100));
        let embedder = MockEmbedder;
        let config = SchedulerConfig::new(1, 1, 4, 100); // large flush interval

        let mut prefetchers = HashMap::new();
        prefetchers.insert(
            SplitLabel::Train,
            SamplerPrefetcher::new(provider.clone(), SplitLabel::Train, 2),
        );

        let stop = AtomicBool::new(true); // immediate Ctrl+C
        let mut handler = EventCollector::new();

        run_interleaved_loop(
            &mut states,
            &mut prefetchers,
            &embedder,
            provider.as_ref(),
            &config,
            &stop,
            &mut handler,
        )
        .unwrap();

        assert!(handler.events.contains(&"ctrl_c".into()));
        assert!(
            provider.saves() >= 1,
            "save_state must be called on Ctrl+C, got {}",
            provider.saves()
        );
    }
}
