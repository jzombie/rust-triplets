//! High-level orchestration helpers that compose [`flush_pending`],
//! [`SplitScheduler`], and [`SplitState`] operations.

use crate::split_scheduler::SplitScheduler;
use crate::split_state::SplitState;
use crate::split_state::flush_pending;
use crate::traits::{BatchProvider, EmbedStore, Result};

/// Marks `state` as exhausted, flushes its pending buffer, and unlocks the
/// scheduler.
///
/// Called from both the sampler-error path (`is_exhaustion_error` matched) and
/// the empty-batch path so both share identical flush+unlock semantics.  The
/// caller prints the exhaustion message before calling this so the message text
/// can differ between the two paths.
pub fn handle_split_exhausted<S: EmbedStore, P: BatchProvider>(
    state: &mut SplitState<S>,
    provider: &P,
    scheduler: &mut SplitScheduler,
) -> Result<()> {
    state.exhausted = true;
    state.batch_num += 1;
    flush_pending(state, provider)?;
    scheduler.unlock();
    Ok(())
}

/// Performs a scheduled flush: increments `batch_num`, flushes the pending
/// buffer to disk, unlocks the scheduler, and returns the number of newly
/// flushed samples.
pub fn flush_batch<S: EmbedStore, P: BatchProvider>(
    state: &mut SplitState<S>,
    provider: &P,
    scheduler: &mut SplitScheduler,
) -> Result<u64> {
    let pre_flush = state.total_written;
    state.batch_num += 1;
    flush_pending(state, provider)?;
    let flushed_count = state.total_written - pre_flush;
    scheduler.unlock();
    Ok(flushed_count)
}

/// Flushes pending buffers for every split that has unflushed data.
pub fn flush_all_pending_states<S: EmbedStore, P: BatchProvider>(
    states: &mut [SplitState<S>],
    provider: &P,
) -> Result<()> {
    for s in states.iter_mut() {
        if !s.pending_anchor_vecs.is_empty() {
            s.batch_num += 1;
            flush_pending(s, provider)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::split_state::EmbedMode;
    use crate::traits::{PairWriteArgs, SamplerBatch, TripletWriteArgs};

    /// Minimal in-memory store for testing.
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

    /// Mock batch provider that always returns exhausted.
    struct MockProvider;

    impl BatchProvider for MockProvider {
        fn next_batch(&self, _split: triplets_core::SplitLabel) -> Result<Option<SamplerBatch>> {
            Ok(None)
        }

        fn save_state(&self) -> Result<()> {
            Ok(())
        }
    }

    fn make_test_state() -> SplitState<MockStore> {
        SplitState {
            label: triplets_core::SplitLabel::Train,
            name: "train",
            store: MockStore,
            mode: EmbedMode::Pair,
            emb_dim: 4,
            max: 0,
            ratio: 0.8,
            total_written: 0,
            step_num: 0,
            batch_num: 0,
            pending_anchor_vecs: Vec::new(),
            pending_anchor_texts: Vec::new(),
            pending_pos_vecs: Vec::new(),
            pending_pos_texts: Vec::new(),
            pending_neg_vecs: Vec::new(),
            pending_neg_texts: Vec::new(),
            exhausted: false,
            dropped_batches: 0,
            total_batches: 0,
            dropped_samples: 0,
            start: None,
            segment_base: 0,
        }
    }

    #[test]
    fn handle_split_exhausted_marks_and_unlocks() {
        let mut state = make_test_state();
        state.pending_anchor_vecs = vec![vec![1.0, 2.0, 3.0, 4.0]];
        state.pending_anchor_texts = vec!["text".into()];
        state.pending_pos_vecs = vec![vec![1.0, 2.0, 3.0, 4.0]];
        state.pending_pos_texts = vec!["text".into()];
        let provider = MockProvider;
        let mut scheduler = crate::SplitScheduler::new();

        let _ = scheduler.next(
            &[triplets_core::SplitLabel::Train],
            &[0],
            &[0],
            &[1.0],
            &[false],
        );
        assert!(scheduler.is_locked());

        handle_split_exhausted(&mut state, &provider, &mut scheduler).unwrap();
        assert!(state.exhausted);
        assert!(!scheduler.is_locked());
    }

    #[test]
    fn flush_batch_unlocks_scheduler() {
        let mut state = make_test_state();
        state.pending_anchor_vecs = vec![vec![1.0; 4]; 3];
        state.pending_anchor_texts = vec!["a".into(), "b".into(), "c".into()];
        state.pending_pos_vecs = vec![vec![1.0; 4]; 3];
        state.pending_pos_texts = vec!["a".into(), "b".into(), "c".into()];
        let provider = MockProvider;
        let mut scheduler = crate::SplitScheduler::new();

        let _ = scheduler.next(
            &[triplets_core::SplitLabel::Train],
            &[0],
            &[0],
            &[1.0],
            &[false],
        );

        let count = flush_batch(&mut state, &provider, &mut scheduler).unwrap();
        assert_eq!(count, 3);
        assert_eq!(state.batch_num, 1);
        assert!(!scheduler.is_locked());
    }

    #[test]
    fn flush_all_pending_states_flushes_dirty_splits() {
        let mut states = vec![make_test_state(), make_test_state()];
        states[0].pending_anchor_vecs = vec![vec![1.0; 4]];
        states[0].pending_anchor_texts = vec!["x".into()];
        states[0].pending_pos_vecs = vec![vec![1.0; 4]];
        states[0].pending_pos_texts = vec!["x".into()];
        let provider = MockProvider;

        flush_all_pending_states(&mut states, &provider).unwrap();
        assert_eq!(states[0].batch_num, 1);
        assert_eq!(states[1].batch_num, 0);
    }

    #[test]
    fn handle_split_exhausted_with_pending_flushes_and_unlocks() {
        let mut state = make_test_state();
        state.pending_anchor_vecs = vec![vec![1.0; 4], vec![2.0; 4]];
        state.pending_anchor_texts = vec!["a".into(), "b".into()];
        state.pending_pos_vecs = vec![vec![3.0; 4], vec![4.0; 4]];
        state.pending_pos_texts = vec!["c".into(), "d".into()];
        let provider = MockProvider;
        let mut scheduler = crate::SplitScheduler::new();

        let _ = scheduler.next(
            &[triplets_core::SplitLabel::Train],
            &[0],
            &[0],
            &[1.0],
            &[false],
        );
        assert!(scheduler.is_locked());

        handle_split_exhausted(&mut state, &provider, &mut scheduler).unwrap();
        assert!(state.exhausted);
        assert!(!scheduler.is_locked());
        assert!(state.is_pending_empty());
        assert_eq!(state.total_written, 2); // flushed 2 pending
        assert_eq!(state.batch_num, 1);
    }

    #[test]
    fn flush_batch_with_pending_returns_count() {
        let mut state = make_test_state();
        state.pending_anchor_vecs = vec![vec![1.0; 4]; 5];
        state.pending_anchor_texts = vec!["a".into(); 5];
        state.pending_pos_vecs = vec![vec![2.0; 4]; 5];
        state.pending_pos_texts = vec!["b".into(); 5];
        let provider = MockProvider;
        let mut scheduler = crate::SplitScheduler::new();

        let _ = scheduler.next(
            &[triplets_core::SplitLabel::Train],
            &[0],
            &[0],
            &[1.0],
            &[false],
        );

        let count = flush_batch(&mut state, &provider, &mut scheduler).unwrap();
        assert_eq!(count, 5);
        assert_eq!(state.total_written, 5);
        assert!(state.is_pending_empty());
        assert!(!scheduler.is_locked());
    }

    #[test]
    fn flush_batch_empty_pending_returns_zero() {
        let mut state = make_test_state();
        // No pending data
        let provider = MockProvider;
        let mut scheduler = crate::SplitScheduler::new();

        let count = flush_batch(&mut state, &provider, &mut scheduler).unwrap();
        assert_eq!(count, 0);
        assert_eq!(state.total_written, 0);
        assert!(!scheduler.is_locked());
    }

    #[test]
    fn flush_all_pending_multiple_dirty_states() {
        let mut states = vec![make_test_state(), make_test_state(), make_test_state()];

        // State 0: 3 pending
        states[0].pending_anchor_vecs = vec![vec![1.0; 4]; 3];
        states[0].pending_anchor_texts = vec!["a".into(); 3];
        states[0].pending_pos_vecs = vec![vec![2.0; 4]; 3];
        states[0].pending_pos_texts = vec!["b".into(); 3];

        // State 1: clean
        // State 2: 2 pending
        states[2].pending_anchor_vecs = vec![vec![3.0; 4]; 2];
        states[2].pending_anchor_texts = vec!["c".into(); 2];
        states[2].pending_pos_vecs = vec![vec![4.0; 4]; 2];
        states[2].pending_pos_texts = vec!["d".into(); 2];

        let provider = MockProvider;
        flush_all_pending_states(&mut states, &provider).unwrap();

        assert_eq!(states[0].batch_num, 1);
        assert_eq!(states[0].total_written, 3);
        assert!(states[0].is_pending_empty());

        assert_eq!(states[1].batch_num, 0);
        assert_eq!(states[1].total_written, 0);

        assert_eq!(states[2].batch_num, 1);
        assert_eq!(states[2].total_written, 2);
        assert!(states[2].is_pending_empty());
    }
}
