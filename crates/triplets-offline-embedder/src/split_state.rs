use std::time::Instant;

use triplets_core::SplitLabel;

use crate::traits::{
    BatchProvider, EmbedStore, Embedder, PairWriteArgs, Result, SamplerBatch, SchedulerConfig,
    SchedulerError, StepResult, TripletWriteArgs,
};

/// Output mode for the embedding store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbedMode {
    /// Pair mode: anchor + positive per entry.
    Pair,
    /// Triplet mode: anchor + positive + negative per entry.
    Triplet,
}

/// Per-split runtime state used by the main precompute loop.
///
/// Generic over `S: EmbedStore` so the storage backend is injected by the
/// caller rather than hardcoded.
pub struct SplitState<S: EmbedStore> {
    /// Split label for this state.
    pub label: SplitLabel,
    /// Human-readable name (e.g. "train", "val").
    pub name: &'static str,
    /// The embedding store for this split.
    pub store: S,
    /// Output mode for this store.
    pub mode: EmbedMode,
    /// Expected embedding dimension (for validation).
    pub emb_dim: usize,
    /// Target max samples (0 = no limit).
    pub max: u64,
    /// Ratio weight used by `next_split_to_fill`.
    pub ratio: f32,
    /// Running count of durably-written samples (updated after each flush).
    pub total_written: u64,
    /// Per-split step counter (# of embed HTTP calls for this split).
    pub step_num: u64,
    /// Per-split flush counter.
    pub batch_num: u64,
    // TODO: Refactor these pending things; what is preventing massive syncronization issues?
    /// Accumulation buffers between flush points (pair mode).
    pub pending_anchor_vecs: Vec<Vec<f32>>,
    /// Pending anchor texts.
    pub pending_anchor_texts: Vec<String>,
    /// Pending positive embedding vectors.
    pub pending_pos_vecs: Vec<Vec<f32>>,
    /// Pending positive texts.
    pub pending_pos_texts: Vec<String>,
    /// Additional buffers for triplet mode.
    pub pending_neg_vecs: Vec<Vec<f32>>,
    /// Pending negative texts.
    pub pending_neg_texts: Vec<String>,
    /// Sampler has no more data for this split.
    pub exhausted: bool,
    /// Number of batches dropped due to embed/validation failures.
    pub dropped_batches: u64,
    /// Total batches processed (including dropped ones) for circuit breaker.
    pub total_batches: u64,
    /// Number of input texts dropped due to embed/validation failures.
    pub dropped_samples: u64,
    /// Wall-clock start for throughput / ETA.
    pub start: Option<Instant>,
    /// Sample count at the start of the current segment (reset each time
    /// this split is newly selected).
    pub segment_base: u64,
}

impl<S: EmbedStore> SplitState<S> {
    /// Whether the pending buffers are empty.
    pub fn is_pending_empty(&self) -> bool {
        self.pending_anchor_vecs.is_empty()
    }

    /// Number of pending samples.
    pub fn pending_len(&self) -> u64 {
        self.pending_anchor_vecs.len() as u64
    }

    /// Total in-flight samples (written + pending).
    pub fn in_flight(&self) -> u64 {
        self.total_written + self.pending_len()
    }

    /// Reset the stint timer and segment base.
    pub fn on_stint_start(&mut self) {
        self.start = Some(Instant::now());
        self.segment_base = self.total_written;
    }

    /// Accumulate a batch of embeddings and texts into the pending buffers.
    pub fn accumulate(
        &mut self,
        anchor_vecs: Vec<Vec<f32>>,
        anchor_texts: &[String],
        pos_vecs: Vec<Vec<f32>>,
        pos_texts: &[String],
        neg_vecs: Option<Vec<Vec<f32>>>,
        neg_texts: Option<&[String]>,
    ) {
        self.pending_anchor_texts
            .extend(anchor_texts.iter().cloned());
        self.pending_anchor_vecs.extend(anchor_vecs);
        self.pending_pos_texts.extend(pos_texts.iter().cloned());
        self.pending_pos_vecs.extend(pos_vecs);
        if let (Some(neg_v), Some(neg_t)) = (neg_vecs, neg_texts) {
            self.pending_neg_texts.extend(neg_t.iter().cloned());
            self.pending_neg_vecs.extend(neg_v);
        }
    }

    /// Clear all pending buffers.
    pub fn clear_pending(&mut self) {
        self.pending_anchor_vecs.clear();
        self.pending_anchor_texts.clear();
        self.pending_pos_vecs.clear();
        self.pending_pos_texts.clear();
        self.pending_neg_vecs.clear();
        self.pending_neg_texts.clear();
    }

    /// Process one embed step: embed the batch, validate, accumulate.
    ///
    /// If `config.embed_batch_size < batch.anchor_texts.len()`, the batch is
    /// chunked and each chunk is embedded separately.
    ///
    /// Returns `StepResult` indicating how many samples were processed/dropped
    /// and whether a flush is needed.
    pub fn step(
        &mut self,
        batch: SamplerBatch,
        embedder: &dyn Embedder,
        config: &SchedulerConfig,
    ) -> Result<StepResult> {
        let anchor_count = batch.anchor_texts.len();
        if anchor_count == 0 {
            self.step_num += 1;
            return Ok(StepResult {
                samples_processed: 0,
                samples_dropped: 0,
                should_flush: false,
            });
        }

        // Latch start time on first real embed
        self.start.get_or_insert_with(std::time::Instant::now);

        // Chunk the batch if embed_batch_size < batch size
        let chunk_size = config.embed_batch_size.max(1);
        let mut total_processed = 0u64;
        let mut total_dropped = 0u64;

        for chunk_start in (0..anchor_count).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(anchor_count);
            let chunk_anchor = &batch.anchor_texts[chunk_start..chunk_end];
            let chunk_pos = &batch.pos_texts[chunk_start..chunk_end];
            let chunk_neg = batch.neg_texts.as_ref().map(|n| &n[chunk_start..chunk_end]);

            // Embed anchor texts
            let anchor_refs: Vec<&str> = chunk_anchor.iter().map(String::as_str).collect();
            let anchor_vecs = match embedder.embed(&anchor_refs) {
                Ok(v) => {
                    match validate_embed_response(&v, chunk_anchor.len(), self.emb_dim) {
                        Ok(()) => v, // zero-copy: validate in-place, keep ownership
                        Err(_e) => {
                            self.step_num += 1;
                            self.dropped_batches += 1;
                            self.total_batches += 1;
                            self.dropped_samples += chunk_anchor.len() as u64;
                            total_dropped += chunk_anchor.len() as u64;
                            continue;
                        }
                    }
                }
                Err(_e) => {
                    self.step_num += 1;
                    self.dropped_batches += 1;
                    self.total_batches += 1;
                    self.dropped_samples += chunk_anchor.len() as u64;
                    total_dropped += chunk_anchor.len() as u64;
                    continue;
                }
            };

            // Embed positive texts (only if different from anchor)
            let pos_vecs = if chunk_pos == chunk_anchor {
                anchor_vecs.clone()
            } else {
                let pos_refs: Vec<&str> = chunk_pos.iter().map(String::as_str).collect();
                match embedder.embed(&pos_refs) {
                    Ok(v) => {
                        match validate_embed_response(&v, chunk_pos.len(), self.emb_dim) {
                            Ok(()) => v, // zero-copy
                            Err(_e) => {
                                self.step_num += 1;
                                self.dropped_batches += 1;
                                self.total_batches += 1;
                                self.dropped_samples += chunk_pos.len() as u64;
                                total_dropped += chunk_pos.len() as u64;
                                continue;
                            }
                        }
                    }
                    Err(_e) => {
                        self.step_num += 1;
                        self.dropped_batches += 1;
                        self.total_batches += 1;
                        self.dropped_samples += chunk_pos.len() as u64;
                        total_dropped += chunk_pos.len() as u64;
                        continue;
                    }
                }
            };

            // Embed negative texts (triplet mode only)
            let neg_vecs = if let Some(neg_texts) = chunk_neg {
                if neg_texts == chunk_anchor {
                    Some(anchor_vecs.clone())
                } else if neg_texts == chunk_pos {
                    Some(pos_vecs.clone())
                } else {
                    let neg_refs: Vec<&str> = neg_texts.iter().map(String::as_str).collect();
                    match embedder.embed(&neg_refs) {
                        Ok(v) => {
                            match validate_embed_response(&v, neg_texts.len(), self.emb_dim) {
                                Ok(()) => Some(v), // zero-copy
                                Err(_e) => {
                                    self.step_num += 1;
                                    self.dropped_batches += 1;
                                    self.total_batches += 1;
                                    self.dropped_samples += neg_texts.len() as u64;
                                    total_dropped += neg_texts.len() as u64;
                                    continue;
                                }
                            }
                        }
                        Err(_e) => {
                            self.step_num += 1;
                            self.dropped_batches += 1;
                            self.total_batches += 1;
                            self.dropped_samples += neg_texts.len() as u64;
                            total_dropped += neg_texts.len() as u64;
                            continue;
                        }
                    }
                }
            } else {
                None
            };

            self.step_num += 1;
            self.total_batches += 1;
            total_processed += chunk_anchor.len() as u64;

            // Accumulate into pending buffers
            self.accumulate(
                anchor_vecs,
                chunk_anchor,
                pos_vecs,
                chunk_pos,
                neg_vecs,
                chunk_neg,
            );
        }

        // Circuit breaker: halt if sustained failure rate exceeds 5%
        if self.total_batches >= 10 {
            let drop_rate = self.dropped_batches as f64 / self.total_batches as f64;
            if drop_rate > 0.05 {
                return Err(SchedulerError::Msg(format!(
                    "circuit breaker: {:.1}% batch drop rate exceeds 5% threshold ({} dropped / {} total)",
                    drop_rate * 100.0,
                    self.dropped_batches,
                    self.total_batches
                )));
            }
        }

        let should_flush = self.step_num.is_multiple_of(config.steps_per_batch);
        Ok(StepResult {
            samples_processed: total_processed,
            samples_dropped: total_dropped,
            should_flush,
        })
    }
}

/// Validate an embedding response in-place: check count, dimension, and
/// non-finite values.  Returns `Ok(())` on success — the caller retains
/// ownership of the vectors without any clone.
pub fn validate_embed_response(
    vecs: &[Vec<f32>],
    expected_count: usize,
    expected_dim: usize,
) -> Result<()> {
    if vecs.len() != expected_count {
        return Err(SchedulerError::Msg(format!(
            "embedding count mismatch: got {}, expected {}",
            vecs.len(),
            expected_count
        )));
    }
    for (i, vec) in vecs.iter().enumerate() {
        if vec.len() != expected_dim {
            return Err(SchedulerError::Msg(format!(
                "embedding dimension mismatch at index {}: got {}, expected {}",
                i,
                vec.len(),
                expected_dim
            )));
        }
        if let Some(j) = vec.iter().position(|x| !x.is_finite()) {
            return Err(SchedulerError::Msg(format!(
                "non-finite embedding value at index {} component {}",
                i, j
            )));
        }
    }
    Ok(())
}

/// Flush accumulated samples into the store and persist sampler state.
///
/// No-op if the buffers are empty.
pub fn flush_pending<S: EmbedStore, P: BatchProvider>(
    state: &mut SplitState<S>,
    provider: &P,
) -> Result<()> {
    let pending_count = state.pending_anchor_vecs.len();
    if pending_count == 0 {
        return Ok(());
    }

    let anchor_texts: Vec<&str> = state
        .pending_anchor_texts
        .iter()
        .map(String::as_str)
        .collect();
    let pos_texts: Vec<&str> = state.pending_pos_texts.iter().map(String::as_str).collect();

    match state.mode {
        EmbedMode::Pair => {
            state.store.write_pairs(
                state.total_written,
                &PairWriteArgs {
                    anchor_vecs: &state.pending_anchor_vecs,
                    anchor_texts: &anchor_texts,
                    pos_vecs: &state.pending_pos_vecs,
                    pos_texts: &pos_texts,
                },
            )?;
        }
        EmbedMode::Triplet => {
            let neg_texts: Vec<&str> = state.pending_neg_texts.iter().map(String::as_str).collect();
            state.store.write_triplets(
                state.total_written,
                &TripletWriteArgs {
                    anchor_vecs: &state.pending_anchor_vecs,
                    anchor_texts: &anchor_texts,
                    pos_vecs: &state.pending_pos_vecs,
                    pos_texts: &pos_texts,
                    neg_vecs: &state.pending_neg_vecs,
                    neg_texts: &neg_texts,
                },
            )?;
        }
    }

    state.total_written += pending_count as u64;
    state.clear_pending();
    provider.save_state()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    type PairWrite = (u64, Vec<Vec<f32>>, Vec<Vec<f32>>);
    type TripletWrite = (u64, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>);

    struct MockStore {
        pair_writes: Mutex<Vec<PairWrite>>,
        triplet_writes: Mutex<Vec<TripletWrite>>,
        len: AtomicUsize,
    }

    impl MockStore {
        fn new() -> Self {
            Self {
                pair_writes: Mutex::new(Vec::new()),
                triplet_writes: Mutex::new(Vec::new()),
                len: AtomicUsize::new(0),
            }
        }
    }

    impl EmbedStore for MockStore {
        fn write_pairs(&self, start_idx: u64, args: &PairWriteArgs<'_>) -> Result<()> {
            self.pair_writes.lock().unwrap().push((
                start_idx,
                args.anchor_vecs.to_vec(),
                args.pos_vecs.to_vec(),
            ));
            self.len
                .fetch_add(args.anchor_vecs.len(), Ordering::Relaxed);
            Ok(())
        }

        fn write_triplets(&self, start_idx: u64, args: &TripletWriteArgs<'_>) -> Result<()> {
            self.triplet_writes.lock().unwrap().push((
                start_idx,
                args.anchor_vecs.to_vec(),
                args.pos_vecs.to_vec(),
                args.neg_vecs.to_vec(),
            ));
            self.len
                .fetch_add(args.anchor_vecs.len(), Ordering::Relaxed);
            Ok(())
        }

        fn len(&self) -> Result<u64> {
            Ok(self.len.load(Ordering::Relaxed) as u64)
        }
    }

    struct MockProvider;

    impl BatchProvider for MockProvider {
        fn next_batch(&self, _split: SplitLabel) -> Result<Option<crate::traits::SamplerBatch>> {
            Ok(None)
        }
        fn save_state(&self) -> Result<()> {
            Ok(())
        }
    }

    struct MockEmbedder {
        dim: usize,
    }

    impl MockEmbedder {
        fn new(dim: usize) -> Self {
            Self { dim }
        }
    }

    impl Embedder for MockEmbedder {
        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .enumerate()
                .map(|(i, _)| vec![i as f32; self.dim])
                .collect())
        }
    }

    struct FailingEmbedder;

    impl Embedder for FailingEmbedder {
        fn embed(&self, _texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Err(SchedulerError::Msg("embed failed".into()))
        }
    }

    fn make_state(store: MockStore, mode: EmbedMode, emb_dim: usize) -> SplitState<MockStore> {
        SplitState {
            label: SplitLabel::Train,
            name: "train",
            store,
            mode,
            emb_dim,
            max: 100,
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
    fn accumulate_and_flush_pair() {
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 2);

        state.accumulate(
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            &["a1".into(), "a2".into()],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
            &["p1".into(), "p2".into()],
            None,
            None,
        );

        assert_eq!(state.pending_len(), 2);
        assert_eq!(state.in_flight(), 2);

        let provider = MockProvider;
        flush_pending(&mut state, &provider).unwrap();

        assert_eq!(state.total_written, 2);
        assert!(state.is_pending_empty());

        let writes = state.store.pair_writes.lock().unwrap();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].0, 0); // start_idx
    }

    #[test]
    fn flush_pending_empty_is_noop() {
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 2);
        state.total_written = 5;

        let provider = MockProvider;
        flush_pending(&mut state, &provider).unwrap();
        assert_eq!(state.total_written, 5);
    }

    #[test]
    fn on_stint_start_resets_segment() {
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 2);
        state.total_written = 50;

        state.on_stint_start();
        assert!(state.start.is_some());
        assert_eq!(state.segment_base, 50);
    }

    #[test]
    fn step_pair_mode_accumulates() {
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 2);
        let embedder = MockEmbedder::new(2);
        let config = SchedulerConfig::new(2, 2, 2, 100);

        let batch = SamplerBatch {
            anchor_texts: vec!["a1".into(), "a2".into()],
            pos_texts: vec!["p1".into(), "p2".into()],
            neg_texts: None,
        };

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 2);
        assert_eq!(result.samples_dropped, 0);
        assert!(!result.should_flush);

        assert_eq!(state.pending_len(), 2);
        assert_eq!(state.step_num, 1);
    }

    #[test]
    fn step_triplet_mode_accumulates() {
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Triplet, 2);
        let embedder = MockEmbedder::new(2);
        let config = SchedulerConfig::new(1, 1, 2, 100);

        let batch = SamplerBatch {
            anchor_texts: vec!["a1".into()],
            pos_texts: vec!["p1".into()],
            neg_texts: Some(vec!["n1".into()]),
        };

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 1);
        assert_eq!(result.samples_dropped, 0);

        assert_eq!(state.pending_len(), 1);
        assert_eq!(state.pending_neg_vecs.len(), 1);
    }

    #[test]
    fn step_flush_at_boundary() {
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 2);
        state.step_num = 99; // next step will be 100
        let embedder = MockEmbedder::new(2);
        let config = SchedulerConfig::new(1, 1, 2, 100);

        let batch = SamplerBatch {
            anchor_texts: vec!["a1".into()],
            pos_texts: vec!["p1".into()],
            neg_texts: None,
        };

        let result = state.step(batch, &embedder, &config).unwrap();
        assert!(result.should_flush);
    }

    #[test]
    fn step_embed_failure_drops_batch() {
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 2);
        let embedder = FailingEmbedder;
        let config = SchedulerConfig::new(2, 2, 2, 100);

        let batch = SamplerBatch {
            anchor_texts: vec!["a1".into(), "a2".into()],
            pos_texts: vec!["p1".into(), "p2".into()],
            neg_texts: None,
        };

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 0);
        assert_eq!(result.samples_dropped, 2);
        assert_eq!(state.dropped_batches, 1);
        assert_eq!(state.dropped_samples, 2);
        assert!(state.is_pending_empty());
    }

    #[test]
    fn step_chunking_when_embed_batch_smaller() {
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 2);
        let embedder = MockEmbedder::new(2);
        let config = SchedulerConfig::new(4, 2, 2, 100); // sampler=4, embed=2

        let batch = SamplerBatch {
            anchor_texts: vec!["a1".into(), "a2".into(), "a3".into(), "a4".into()],
            pos_texts: vec!["p1".into(), "p2".into(), "p3".into(), "p4".into()],
            neg_texts: None,
        };

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 4);
        assert_eq!(result.samples_dropped, 0);
        assert_eq!(state.pending_len(), 4);
        // 4 texts / 2 chunk_size = 2 embed calls = 2 step increments
        assert_eq!(state.step_num, 2);
    }

    #[test]
    fn validate_embed_response_ok() {
        let vecs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        validate_embed_response(&vecs, 2, 2).unwrap();
    }

    #[test]
    fn validate_embed_response_count_mismatch() {
        let vecs = vec![vec![1.0, 2.0]];
        let err = validate_embed_response(&vecs, 2, 2).unwrap_err();
        assert!(err.to_string().contains("count mismatch"));
    }

    #[test]
    fn validate_embed_response_dim_mismatch() {
        let vecs = vec![vec![1.0], vec![3.0, 4.0]];
        let err = validate_embed_response(&vecs, 2, 2).unwrap_err();
        assert!(err.to_string().contains("dimension mismatch"));
    }

    #[test]
    fn validate_embed_response_non_finite() {
        let vecs = vec![vec![1.0, f32::NAN], vec![3.0, 4.0]];
        let err = validate_embed_response(&vecs, 2, 2).unwrap_err();
        assert!(err.to_string().contains("non-finite"));
    }

    // ------------------------------------------------------------------
    // Coverage gap tests
    // ------------------------------------------------------------------

    #[test]
    fn step_empty_batch_returns_zero() {
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 2);
        let embedder = MockEmbedder::new(2);
        let config = SchedulerConfig::new(1, 1, 2, 100);

        let batch = SamplerBatch {
            anchor_texts: vec![],
            pos_texts: vec![],
            neg_texts: None,
        };

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 0);
        assert_eq!(result.samples_dropped, 0);
        assert!(!result.should_flush);
        assert!(state.is_pending_empty());
        assert_eq!(state.step_num, 1); // incremented even for empty
    }

    #[test]
    fn step_pos_equals_anchor_reuses_vectors() {
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 2);
        // Embedder that tracks call count
        let embedder = MockEmbedder::new(2);
        let config = SchedulerConfig::new(1, 1, 2, 100);

        // anchor_texts == pos_texts → only 1 embed call (anchor), pos reuses
        let batch = SamplerBatch {
            anchor_texts: vec!["same".into()],
            pos_texts: vec!["same".into()],
            neg_texts: None,
        };

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 1);
        assert_eq!(state.pending_len(), 1);
        // MockEmbedder called once for anchor; pos reuses anchor_vecs
        // So pending_pos_vecs should equal pending_anchor_vecs
        assert_eq!(state.pending_anchor_vecs, state.pending_pos_vecs);
    }

    #[test]
    fn step_neg_equals_anchor_reuses_vectors() {
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Triplet, 2);
        let embedder = MockEmbedder::new(2);
        let config = SchedulerConfig::new(1, 1, 2, 100);

        let batch = SamplerBatch {
            anchor_texts: vec!["a".into()],
            pos_texts: vec!["p".into()],
            neg_texts: Some(vec!["a".into()]), // neg == anchor
        };

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 1);
        assert_eq!(state.pending_neg_vecs.len(), 1);
        // neg should reuse anchor vectors
        assert_eq!(state.pending_neg_vecs, state.pending_anchor_vecs);
    }

    #[test]
    fn step_neg_equals_pos_reuses_vectors() {
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Triplet, 2);
        let embedder = MockEmbedder::new(2);
        let config = SchedulerConfig::new(1, 1, 2, 100);

        let batch = SamplerBatch {
            anchor_texts: vec!["a".into()],
            pos_texts: vec!["p".into()],
            neg_texts: Some(vec!["p".into()]), // neg == pos
        };

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 1);
        // neg should reuse pos vectors
        assert_eq!(state.pending_neg_vecs, state.pending_pos_vecs);
    }

    #[test]
    fn step_neg_embed_failure_drops_batch() {
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Triplet, 2);
        let _embedder = MockEmbedder::new(2);
        let config = SchedulerConfig::new(1, 1, 2, 100);

        // Make neg different from anchor and pos so it triggers embed call
        let batch = SamplerBatch {
            anchor_texts: vec!["a".into()],
            pos_texts: vec!["p".into()],
            neg_texts: Some(vec!["n".into()]),
        };

        // Override embedder to fail on "n" but succeed on "a" and "p"
        struct NegFailingEmbedder;
        impl Embedder for NegFailingEmbedder {
            fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
                if texts.contains(&"n") {
                    Err(SchedulerError::Msg("neg embed failed".into()))
                } else {
                    Ok(texts.iter().map(|_| vec![1.0; 2]).collect())
                }
            }
        }

        let result = state.step(batch, &NegFailingEmbedder, &config).unwrap();
        assert_eq!(result.samples_processed, 0);
        assert_eq!(result.samples_dropped, 1);
        assert_eq!(state.dropped_batches, 1);
        assert_eq!(state.dropped_samples, 1);
    }

    #[test]
    fn accumulate_with_negatives() {
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Triplet, 2);

        state.accumulate(
            vec![vec![1.0, 2.0]],
            &["a".into()],
            vec![vec![3.0, 4.0]],
            &["p".into()],
            Some(vec![vec![5.0, 6.0]]),
            Some(&["n".into()]),
        );

        assert_eq!(state.pending_len(), 1);
        assert_eq!(state.pending_neg_vecs.len(), 1);
        assert_eq!(state.pending_neg_texts, vec!["n"]);
    }

    #[test]
    fn flush_pending_triplet_mode() {
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Triplet, 2);
        state.accumulate(
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            &["a1".into(), "a2".into()],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
            &["p1".into(), "p2".into()],
            Some(vec![vec![9.0, 10.0], vec![11.0, 12.0]]),
            Some(&["n1".into(), "n2".into()]),
        );

        let provider = MockProvider;
        flush_pending(&mut state, &provider).unwrap();

        assert_eq!(state.total_written, 2);
        assert!(state.is_pending_empty());

        let writes = state.store.triplet_writes.lock().unwrap();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].0, 0); // start_idx
        assert_eq!(writes[0].1.len(), 2); // anchor vecs
        assert_eq!(writes[0].2.len(), 2); // pos vecs
        assert_eq!(writes[0].3.len(), 2); // neg vecs
    }

    struct FailingProvider;

    impl BatchProvider for FailingProvider {
        fn next_batch(&self, _split: SplitLabel) -> Result<Option<crate::traits::SamplerBatch>> {
            Ok(None)
        }
        fn save_state(&self) -> Result<()> {
            Err(SchedulerError::Msg("save failed".into()))
        }
    }

    #[test]
    fn flush_pending_save_state_error_propagates() {
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 2);
        state.accumulate(
            vec![vec![1.0, 2.0]],
            &["a".into()],
            vec![vec![3.0, 4.0]],
            &["p".into()],
            None,
            None,
        );

        let provider = FailingProvider;
        let err = flush_pending(&mut state, &provider).unwrap_err();
        assert!(err.to_string().contains("save failed"));
        // Data was written to store (write_pairs succeeded), but state wasn't saved
        // In a real system this would need transactional semantics
    }

    #[test]
    fn validate_embed_response_infinity() {
        let vecs = vec![vec![1.0, f32::INFINITY], vec![3.0, 4.0]];
        let err = validate_embed_response(&vecs, 2, 2).unwrap_err();
        assert!(err.to_string().contains("non-finite"));
    }

    #[test]
    fn validate_embed_response_neg_infinity() {
        let vecs = vec![vec![1.0, 2.0], vec![f32::NEG_INFINITY, 4.0]];
        let err = validate_embed_response(&vecs, 2, 2).unwrap_err();
        assert!(err.to_string().contains("non-finite"));
    }

    // ===================================================================
    // ALIGNMENT TESTS — highest priority for data integrity
    //
    // When embed_batch_size < sampler_batch_size, texts are chunked and
    // embedded in separate calls.  A misalignment bug would silently
    // corrupt training data.  These tests verify that every text-vector
    // pair remains correctly matched across all chunks.
    // ===================================================================

    /// Deterministic embedder that returns a unique vector per text.
    /// Uses text content as the first component so alignment can be verified.
    /// The second component encodes the chunk-local position (0-based within
    /// each embed call) so we can detect off-by-one at chunk boundaries.
    struct AlignmentEmbedder {
        dim: usize,
    }

    impl AlignmentEmbedder {
        fn new(dim: usize) -> Self {
            Self { dim }
        }
    }

    impl Embedder for AlignmentEmbedder {
        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    // Encode text identity as a float in the first slot
                    let text_id = t.chars().map(|c| c as u32 as f32).sum::<f32>();
                    let mut v = vec![0.0; self.dim];
                    v[0] = text_id;
                    if self.dim > 1 {
                        v[1] = i as f32; // chunk-local position
                    }
                    v
                })
                .collect())
        }
    }

    #[test]
    fn alignment_sampler_larger_than_embed_batch_pair() {
        // sampler_batch=6, embed_batch=2 → 3 chunks
        // Verify all 6 entries are correctly aligned after chunking
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 4);
        let embedder = AlignmentEmbedder::new(4);
        let config = SchedulerConfig::new(6, 2, 4, 100);

        let batch = SamplerBatch {
            anchor_texts: vec![
                "a1".into(),
                "a2".into(),
                "a3".into(),
                "a4".into(),
                "a5".into(),
                "a6".into(),
            ],
            pos_texts: vec![
                "p1".into(),
                "p2".into(),
                "p3".into(),
                "p4".into(),
                "p5".into(),
                "p6".into(),
            ],
            neg_texts: None,
        };

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 6);
        assert_eq!(state.pending_len(), 6);
        // 6 texts / 2 chunk_size = 3 embed calls = 3 step increments
        assert_eq!(state.step_num, 3);

        // CRITICAL: verify alignment — each pending text must match its vector
        assert_eq!(state.pending_anchor_texts.len(), 6);
        assert_eq!(state.pending_anchor_vecs.len(), 6);
        assert_eq!(state.pending_pos_texts.len(), 6);
        assert_eq!(state.pending_pos_vecs.len(), 6);

        for i in 0..6 {
            let anchor_text = format!("a{}", i + 1);
            let pos_text = format!("p{}", i + 1);
            let expected_anchor_id: f32 = anchor_text.chars().map(|c| c as u32 as f32).sum();
            let expected_pos_id: f32 = pos_text.chars().map(|c| c as u32 as f32).sum();
            // vec[1] = chunk-local position (embed_batch=2, so resets every 2)
            let chunk_local_pos = (i % 2) as f32;

            assert_eq!(
                state.pending_anchor_texts[i], anchor_text,
                "anchor text mismatch at index {i}"
            );
            assert_eq!(
                state.pending_anchor_vecs[i][0], expected_anchor_id,
                "anchor vector text_id mismatch at index {i}"
            );
            assert_eq!(
                state.pending_anchor_vecs[i][1], chunk_local_pos,
                "anchor vector position mismatch at index {i}"
            );
            assert_eq!(
                state.pending_pos_texts[i], pos_text,
                "pos text mismatch at index {i}"
            );
            assert_eq!(
                state.pending_pos_vecs[i][0], expected_pos_id,
                "pos vector text_id mismatch at index {i}"
            );
            assert_eq!(
                state.pending_pos_vecs[i][1], chunk_local_pos,
                "pos vector position mismatch at index {i}"
            );
        }
    }

    #[test]
    fn alignment_sampler_larger_than_embed_batch_triplet() {
        // sampler_batch=5, embed_batch=2 → 3 chunks (2+2+1)
        // Verify anchor/positive/negative stay aligned across chunks
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Triplet, 4);
        let embedder = AlignmentEmbedder::new(4);
        let config = SchedulerConfig::new(5, 2, 4, 100);

        let batch = SamplerBatch {
            anchor_texts: vec![
                "a1".into(),
                "a2".into(),
                "a3".into(),
                "a4".into(),
                "a5".into(),
            ],
            pos_texts: vec![
                "p1".into(),
                "p2".into(),
                "p3".into(),
                "p4".into(),
                "p5".into(),
            ],
            neg_texts: Some(vec![
                "n1".into(),
                "n2".into(),
                "n3".into(),
                "n4".into(),
                "n5".into(),
            ]),
        };

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 5);
        assert_eq!(state.pending_len(), 5);

        // Verify all 5 entries aligned across all 3 text/vector buffers
        for i in 0..5 {
            let anchor_text = format!("a{}", i + 1);
            let pos_text = format!("p{}", i + 1);
            let neg_text = format!("n{}", i + 1);
            let expected_anchor_id: f32 = anchor_text.chars().map(|c| c as u32 as f32).sum();
            let expected_pos_id: f32 = pos_text.chars().map(|c| c as u32 as f32).sum();
            let expected_neg_id: f32 = neg_text.chars().map(|c| c as u32 as f32).sum();
            let chunk_local_pos = (i % 2) as f32;

            assert_eq!(
                state.pending_anchor_texts[i], anchor_text,
                "anchor text mismatch at {i}"
            );
            assert_eq!(
                state.pending_anchor_vecs[i][0], expected_anchor_id,
                "anchor vec text_id mismatch at {i}"
            );
            assert_eq!(
                state.pending_anchor_vecs[i][1], chunk_local_pos,
                "anchor vec position mismatch at {i}"
            );
            assert_eq!(
                state.pending_pos_texts[i], pos_text,
                "pos text mismatch at {i}"
            );
            assert_eq!(
                state.pending_pos_vecs[i][0], expected_pos_id,
                "pos vec text_id mismatch at {i}"
            );
            assert_eq!(
                state.pending_pos_vecs[i][1], chunk_local_pos,
                "pos vec position mismatch at {i}"
            );
            assert_eq!(
                state.pending_neg_texts[i], neg_text,
                "neg text mismatch at {i}"
            );
            assert_eq!(
                state.pending_neg_vecs[i][0], expected_neg_id,
                "neg vec text_id mismatch at {i}"
            );
            assert_eq!(
                state.pending_neg_vecs[i][1], chunk_local_pos,
                "neg vec position mismatch at {i}"
            );
        }
    }

    #[test]
    fn alignment_embed_larger_than_sampler_batch() {
        // embed_batch=10, sampler_batch=3 → 1 chunk of 3
        // Verify no duplication or truncation
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 4);
        let embedder = AlignmentEmbedder::new(4);
        let config = SchedulerConfig::new(3, 10, 4, 100);

        let batch = SamplerBatch {
            anchor_texts: vec!["x1".into(), "x2".into(), "x3".into()],
            pos_texts: vec!["y1".into(), "y2".into(), "y3".into()],
            neg_texts: None,
        };

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 3);
        assert_eq!(state.pending_len(), 3);

        for i in 0..3 {
            let text = format!("x{}", i + 1);
            let expected_id: f32 = text.chars().map(|c| c as u32 as f32).sum();
            assert_eq!(state.pending_anchor_texts[i], text);
            assert_eq!(state.pending_anchor_vecs[i][0], expected_id);
            assert_eq!(state.pending_pos_texts[i], format!("y{}", i + 1));
        }
    }

    #[test]
    fn alignment_exact_match_sampler_equals_embed() {
        // embed_batch=4, sampler_batch=4 → 1 chunk, no chunking
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 4);
        let embedder = AlignmentEmbedder::new(4);
        let config = SchedulerConfig::new(4, 4, 4, 100);

        let batch = SamplerBatch {
            anchor_texts: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            pos_texts: vec!["e".into(), "f".into(), "g".into(), "h".into()],
            neg_texts: None,
        };

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 4);
        assert_eq!(state.step_num, 1); // single chunk

        for (i, (at, pt)) in state
            .pending_anchor_texts
            .iter()
            .zip(state.pending_pos_texts.iter())
            .enumerate()
        {
            let expected_id = (b'a' + i as u8) as char;
            assert_eq!(at.as_str(), expected_id.to_string().as_str());
            assert_eq!(pt.as_str(), ((b'e' + i as u8) as char).to_string().as_str());
        }
    }

    #[test]
    fn alignment_uneven_chunk_remainder() {
        // sampler_batch=7, embed_batch=3 → 3 chunks (3+3+1)
        // The remainder chunk (1 item) must be correctly aligned
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 4);
        let embedder = AlignmentEmbedder::new(4);
        let config = SchedulerConfig::new(7, 3, 4, 100);

        let texts: Vec<String> = (0..7).map(|i| format!("t{i}")).collect();
        let batch = SamplerBatch {
            anchor_texts: texts.clone(),
            pos_texts: texts.clone(), // same as anchor → tests the pos==anchor reuse path
            neg_texts: None,
        };

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 7);
        assert_eq!(state.pending_len(), 7);
        assert_eq!(state.step_num, 3); // ceil(7/3) = 3

        // All 7 entries must be correctly aligned
        for i in 0..7 {
            let text = format!("t{i}");
            let expected_id: f32 = text.chars().map(|c| c as u32 as f32).sum();
            assert_eq!(state.pending_anchor_texts[i], text, "text mismatch at {i}");
            assert_eq!(
                state.pending_anchor_vecs[i][0], expected_id,
                "vector text_id mismatch at {i}"
            );
            // pos == anchor, so pos vectors should equal anchor vectors
            assert_eq!(
                state.pending_anchor_vecs[i], state.pending_pos_vecs[i],
                "pos/anchor vec mismatch at {i}"
            );
        }
    }

    #[test]
    fn alignment_multiple_steps_compound() {
        // Run 2 steps with different batch sizes to verify pending buffers
        // compound correctly without interleaving
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 4);
        let embedder = AlignmentEmbedder::new(4);
        let config = SchedulerConfig::new(3, 2, 4, 100);

        // Step 1: 3 texts → 2 chunks (2+1)
        let batch1 = SamplerBatch {
            anchor_texts: vec!["step1_a".into(), "step1_b".into(), "step1_c".into()],
            pos_texts: vec!["step1_p1".into(), "step1_p2".into(), "step1_p3".into()],
            neg_texts: None,
        };
        state.step(batch1, &embedder, &config).unwrap();
        assert_eq!(state.pending_len(), 3);

        // Step 2: 2 texts → 1 chunk
        let batch2 = SamplerBatch {
            anchor_texts: vec!["step2_a".into(), "step2_b".into()],
            pos_texts: vec!["step2_p1".into(), "step2_p2".into()],
            neg_texts: None,
        };
        state.step(batch2, &embedder, &config).unwrap();
        assert_eq!(state.pending_len(), 5);

        // Verify first 3 entries are from step1, last 2 from step2
        assert_eq!(state.pending_anchor_texts[0], "step1_a");
        assert_eq!(state.pending_anchor_texts[1], "step1_b");
        assert_eq!(state.pending_anchor_texts[2], "step1_c");
        assert_eq!(state.pending_anchor_texts[3], "step2_a");
        assert_eq!(state.pending_anchor_texts[4], "step2_b");

        // Verify vectors match their texts
        for i in 0..5 {
            let text = &state.pending_anchor_texts[i];
            let expected_id: f32 = text.chars().map(|c| c as u32 as f32).sum();
            assert_eq!(
                state.pending_anchor_vecs[i][0], expected_id,
                "vector mismatch at index {i} for text '{text}'"
            );
        }
    }

    #[test]
    fn alignment_triplet_chunk_boundary_no_swap() {
        // sampler_batch=4, embed_batch=2 → 2 chunks
        // The boundary between chunks must not swap negative vectors
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Triplet, 4);
        let embedder = AlignmentEmbedder::new(4);
        let config = SchedulerConfig::new(4, 2, 4, 100);

        let batch = SamplerBatch {
            anchor_texts: vec![
                "a_odd".into(),
                "a_even".into(),
                "a_third".into(),
                "a_fourth".into(),
            ],
            pos_texts: vec![
                "p_odd".into(),
                "p_even".into(),
                "p_third".into(),
                "p_fourth".into(),
            ],
            neg_texts: Some(vec![
                "n_odd".into(),
                "n_even".into(),
                "n_third".into(),
                "n_fourth".into(),
            ]),
        };

        state.step(batch, &embedder, &config).unwrap();

        // At index 0: anchor="a_odd", pos="p_odd", neg="n_odd"
        assert_eq!(state.pending_anchor_texts[0], "a_odd");
        assert_eq!(state.pending_pos_texts[0], "p_odd");
        assert_eq!(state.pending_neg_texts[0], "n_odd");

        // At index 2 (chunk boundary): anchor="a_third", pos="p_third", neg="n_third"
        assert_eq!(state.pending_anchor_texts[2], "a_third");
        assert_eq!(state.pending_pos_texts[2], "p_third");
        assert_eq!(state.pending_neg_texts[2], "n_third");

        // Verify neg vectors are NOT swapped at boundary
        let expected_neg2_id: f32 = "n_third".chars().map(|c| c as u32 as f32).sum();
        assert_eq!(
            state.pending_neg_vecs[2][0], expected_neg2_id,
            "negative vector swapped at chunk boundary!"
        );
    }

    // ===================================================================
    // EDGE CASE TESTS — empty texts, infinite/zero vectors, corruption
    // ===================================================================

    /// Embedder that returns all-zero vectors. Tests that alignment holds
    /// even when vectors carry no distinguishing information.
    struct ZeroEmbedder;

    impl Embedder for ZeroEmbedder {
        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![0.0; 4]).collect())
        }
    }

    /// Embedder that returns vectors filled with infinity.
    struct InfEmbedder;

    impl Embedder for InfEmbedder {
        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![f32::INFINITY; 4]).collect())
        }
    }

    #[test]
    fn alignment_with_zero_vectors_across_chunks() {
        // All vectors are [0,0,0,0] — alignment must still hold text→vector pairing
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 4);
        let embedder = ZeroEmbedder;
        let config = SchedulerConfig::new(5, 2, 4, 100);

        let batch = SamplerBatch {
            anchor_texts: vec![
                "z1".into(),
                "z2".into(),
                "z3".into(),
                "z4".into(),
                "z5".into(),
            ],
            pos_texts: vec![
                "q1".into(),
                "q2".into(),
                "q3".into(),
                "q4".into(),
                "q5".into(),
            ],
            neg_texts: None,
        };

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 5);
        assert_eq!(state.pending_len(), 5);

        // Texts must be aligned even if vectors are all zero
        for i in 0..5 {
            assert_eq!(state.pending_anchor_texts[i], format!("z{}", i + 1));
            assert_eq!(state.pending_pos_texts[i], format!("q{}", i + 1));
            assert_eq!(state.pending_anchor_vecs[i], vec![0.0; 4]);
            assert_eq!(state.pending_pos_vecs[i], vec![0.0; 4]);
        }
    }

    #[test]
    fn alignment_with_infinite_vectors_across_chunks() {
        // Vectors are all inf — validate_embed_response rejects non-finite,
        // so the batch should be dropped
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 4);
        let embedder = InfEmbedder;
        let config = SchedulerConfig::new(3, 2, 4, 100);

        let batch = SamplerBatch {
            anchor_texts: vec!["a1".into(), "a2".into(), "a3".into()],
            pos_texts: vec!["p1".into(), "p2".into(), "p3".into()],
            neg_texts: None,
        };

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 0);
        assert_eq!(result.samples_dropped, 3);
        assert!(state.is_pending_empty());
    }

    #[test]
    fn alignment_with_nan_vectors_across_chunks() {
        // Vectors contain NaN — should be rejected
        struct NanEmbedder;

        impl Embedder for NanEmbedder {
            fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
                Ok(texts
                    .iter()
                    .enumerate()
                    .map(|(i, _)| {
                        let mut v = vec![0.0; 4];
                        if i % 2 == 0 {
                            v[0] = f32::NAN;
                        }
                        v
                    })
                    .collect())
            }
        }

        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 4);
        let embedder = NanEmbedder;
        let config = SchedulerConfig::new(4, 2, 4, 100);

        let batch = SamplerBatch {
            anchor_texts: vec!["a1".into(), "a2".into(), "a3".into(), "a4".into()],
            pos_texts: vec!["p1".into(), "p2".into(), "p3".into(), "p4".into()],
            neg_texts: None,
        };

        let result = state.step(batch, &embedder, &config).unwrap();
        // First chunk [a1,a2] → a1 has NaN → entire chunk dropped
        // Second chunk [a3,a4] → a3 has NaN → entire chunk dropped
        assert_eq!(result.samples_processed, 0);
        assert_eq!(result.samples_dropped, 4);
        assert_eq!(state.dropped_batches, 2);
        assert!(state.is_pending_empty());
    }

    #[test]
    fn alignment_empty_anchor_texts_returns_zero() {
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 4);
        let embedder = AlignmentEmbedder::new(4);
        let config = SchedulerConfig::new(0, 2, 4, 100);

        let batch = SamplerBatch {
            anchor_texts: vec![],
            pos_texts: vec![],
            neg_texts: None,
        };

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 0);
        assert_eq!(result.samples_dropped, 0);
        assert!(state.is_pending_empty());
    }

    #[test]
    fn alignment_single_text_batch() {
        // Edge case: sampler_batch=1, embed_batch=1 → 1 chunk of 1
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 4);
        let embedder = AlignmentEmbedder::new(4);
        let config = SchedulerConfig::new(1, 1, 4, 100);

        let batch = SamplerBatch {
            anchor_texts: vec!["solo".into()],
            pos_texts: vec!["partner".into()],
            neg_texts: None,
        };

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 1);
        assert_eq!(state.pending_len(), 1);

        let expected_id: f32 = "solo".chars().map(|c| c as u32 as f32).sum();
        assert_eq!(state.pending_anchor_texts[0], "solo");
        assert_eq!(state.pending_anchor_vecs[0][0], expected_id);
        assert_eq!(state.pending_pos_texts[0], "partner");
    }

    #[test]
    fn alignment_all_chunks_dropped_no_residue() {
        // If ALL chunks fail validation, pending must be completely empty
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 4);
        let embedder = InfEmbedder; // returns inf → all chunks rejected
        let config = SchedulerConfig::new(6, 2, 4, 100);

        let batch = SamplerBatch {
            anchor_texts: vec![
                "a1".into(),
                "a2".into(),
                "a3".into(),
                "a4".into(),
                "a5".into(),
                "a6".into(),
            ],
            pos_texts: vec![
                "p1".into(),
                "p2".into(),
                "p3".into(),
                "p4".into(),
                "p5".into(),
                "p6".into(),
            ],
            neg_texts: None,
        };

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 0);
        assert_eq!(result.samples_dropped, 6);
        assert_eq!(state.dropped_batches, 3); // 6/2 = 3 chunks
        assert!(state.is_pending_empty());
        assert_eq!(state.pending_anchor_texts.len(), 0);
        assert_eq!(state.pending_anchor_vecs.len(), 0);
    }

    #[test]
    fn alignment_partial_drop_preserves_remaining() {
        // Chunk 0 fails, chunk 1 succeeds → only chunk 1's data in pending
        struct FirstChunkFailEmbedder {
            call_count: std::sync::atomic::AtomicUsize,
        }

        impl Embedder for FirstChunkFailEmbedder {
            fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
                let call = self.call_count.fetch_add(1, Ordering::Relaxed);
                if call == 0 {
                    // First embed call (chunk 0 anchor) returns wrong count
                    Err(SchedulerError::Msg("transient failure".into()))
                } else {
                    Ok(texts.iter().map(|_| vec![1.0; 4]).collect())
                }
            }
        }

        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 4);
        let embedder = FirstChunkFailEmbedder {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        };
        let config = SchedulerConfig::new(4, 2, 4, 100);

        let batch = SamplerBatch {
            anchor_texts: vec!["a1".into(), "a2".into(), "a3".into(), "a4".into()],
            pos_texts: vec!["p1".into(), "p2".into(), "p3".into(), "p4".into()],
            neg_texts: None,
        };

        let result = state.step(batch, &embedder, &config).unwrap();
        // Chunk 0 [a1,a2] failed → dropped
        // Chunk 1 [a3,a4] succeeded → accumulated
        assert_eq!(result.samples_processed, 2);
        assert_eq!(result.samples_dropped, 2);
        assert_eq!(state.pending_len(), 2);
        // Only chunk 1's data should be in pending
        assert_eq!(state.pending_anchor_texts[0], "a3");
        assert_eq!(state.pending_anchor_texts[1], "a4");
    }

    #[test]
    fn step_circuit_breaker_halt() {
        struct AlwaysFailEmbedder;
        impl Embedder for AlwaysFailEmbedder {
            fn embed(&self, _texts: &[&str]) -> Result<Vec<Vec<f32>>> {
                Err(SchedulerError::Msg("systemic failure".into()))
            }
        }

        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 4);
        let embedder = AlwaysFailEmbedder;
        // embed_batch_size=4 → 1 chunk per batch → 1 embed call per batch
        let config = SchedulerConfig::new(4, 4, 4, 100);

        // All batches fail. Circuit breaker fires when total_batches >= 10 AND drop_rate > 5%.
        // With 100% failure: fires at batch index 9 (the 10th batch, total_batches=10).
        for i in 0..12 {
            let batch = SamplerBatch {
                anchor_texts: vec![
                    format!("a{i}"),
                    format!("b{i}"),
                    format!("c{i}"),
                    format!("d{i}"),
                ],
                pos_texts: vec![
                    format!("p{i}"),
                    format!("q{i}"),
                    format!("r{i}"),
                    format!("s{i}"),
                ],
                neg_texts: None,
            };
            let result = state.step(batch, &embedder, &config);
            if i < 9 {
                // Batches 0-8: total_batches < 10 → no circuit breaker
                let sr = result.unwrap();
                assert_eq!(sr.samples_dropped, 4);
            } else {
                // Batch 9+: total_batches >= 10, drop_rate = 100% > 5% → Err
                let err = result.unwrap_err();
                assert!(
                    err.to_string().contains("circuit breaker"),
                    "expected circuit breaker error, got: {err}"
                );
            }
        }
        assert_eq!(state.total_batches, 12);
        assert_eq!(state.dropped_batches, 12);
    }

    #[test]
    fn step_circuit_breaker_no_halt_below_threshold() {
        // Fail only on the 40th embed call (batch 20 anchor). By then total_batches=20,
        // so 1 drop / 20 total = 5% exactly (not > 5%) → no circuit breaker.
        struct FailOnCall40Embedder {
            call_count: std::sync::atomic::AtomicUsize,
        }
        impl Embedder for FailOnCall40Embedder {
            fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
                let call = self.call_count.fetch_add(1, Ordering::Relaxed);
                if call == 40 {
                    Err(SchedulerError::Msg("transient".into()))
                } else {
                    Ok(texts.iter().map(|_| vec![1.0; 4]).collect())
                }
            }
        }

        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 4);
        let embedder = FailOnCall40Embedder {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        };
        // embed_batch_size=4 → 1 chunk per batch → 2 embed calls per batch (anchor + pos)
        let config = SchedulerConfig::new(4, 4, 4, 100);

        // 30 batches. Call 40 fails (batch 20 anchor). By then total_batches=20, rate=5% (not > 5%).
        for i in 0..30 {
            let batch = SamplerBatch {
                anchor_texts: vec![
                    format!("a{i}"),
                    format!("b{i}"),
                    format!("c{i}"),
                    format!("d{i}"),
                ],
                pos_texts: vec![
                    format!("p{i}"),
                    format!("q{i}"),
                    format!("r{i}"),
                    format!("s{i}"),
                ],
                neg_texts: None,
            };
            let result = state.step(batch, &embedder, &config);
            result.unwrap_or_else(|e| panic!("batch {i} should not trigger circuit breaker: {e}"));
        }
        assert_eq!(state.total_batches, 30);
        assert_eq!(state.dropped_batches, 1);
    }
}
