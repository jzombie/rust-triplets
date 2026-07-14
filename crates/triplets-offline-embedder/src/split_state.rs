use std::time::Instant;

use triplets_core::SplitLabel;

use triplets_core::data::PairLabel;

use crate::traits::{
    BatchProvider, EmbedStore, Embedder, PairEntry, PairWriteArgs, PairWriteEntry, Result,
    SamplerBatch, SchedulerConfig, SchedulerError, StepResult, TripletEntry, TripletWriteArgs,
    TripletWriteEntry,
};

/// Accumulated pair entry ready for flush.
#[derive(Debug, Clone)]
pub struct PendingPair {
    pub anchor_text: String,
    pub anchor_vec: Vec<f32>,
    pub candidate_text: String,
    pub candidate_vec: Vec<f32>,
    pub label: PairLabel,
}

/// Accumulated triplet entry ready for flush.
#[derive(Debug, Clone)]
pub struct PendingTriplet {
    pub anchor_text: String,
    pub anchor_vec: Vec<f32>,
    pub pos_text: String,
    pub pos_vec: Vec<f32>,
    pub neg_text: String,
    pub neg_vec: Vec<f32>,
}

/// Pending accumulation state — either pairs or triplets.
#[derive(Debug, Clone)]
pub enum PendingState {
    Pairs(Vec<PendingPair>),
    Triplets(Vec<PendingTriplet>),
}

impl PendingState {
    pub fn len(&self) -> usize {
        match self {
            PendingState::Pairs(v) => v.len(),
            PendingState::Triplets(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn clear(&mut self) {
        *self = match self {
            PendingState::Pairs(_) => PendingState::Pairs(Vec::new()),
            PendingState::Triplets(_) => PendingState::Triplets(Vec::new()),
        };
    }
    pub fn push_pair(&mut self, entry: PendingPair) {
        if let PendingState::Pairs(v) = self {
            v.push(entry);
        } else {
            unreachable!("push_pair called on Triplets state");
        }
    }
    pub fn push_triplet(&mut self, entry: PendingTriplet) {
        if let PendingState::Triplets(v) = self {
            v.push(entry);
        } else {
            unreachable!("push_triplet called on Pairs state");
        }
    }
}

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
    /// Pending entries (AoS — Pair or Triplet variant).
    pub pending: PendingState,
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
        self.pending.is_empty()
    }

    /// Number of pending samples.
    pub fn pending_len(&self) -> u64 {
        self.pending.len() as u64
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

    /// Clear all pending buffers.
    pub fn clear_pending(&mut self) {
        self.pending.clear();
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
        let anchor_count = match &batch {
            SamplerBatch::Pairs(v) => v.len(),
            SamplerBatch::Triplets(v) => v.len(),
        };
        if anchor_count == 0 {
            self.step_num += 1;
            return Ok(StepResult {
                samples_processed: 0,
                samples_dropped: 0,
                should_flush: false,
            });
        }

        self.start.get_or_insert_with(std::time::Instant::now);

        let chunk_size = config.embed_batch_size.max(1);
        let mut total_processed = 0u64;
        let mut total_dropped = 0u64;

        match batch {
            SamplerBatch::Pairs(entries) => {
                let mut all_anchor_vecs = Vec::with_capacity(entries.len());
                let mut all_candidate_vecs: Vec<Option<Vec<f32>>> = Vec::with_capacity(entries.len());
                let mut keep = vec![false; entries.len()];
                let mut chunk_start = 0;

                // Phase 1: Borrow and Embed (entries untouched)
                for chunk in entries.chunks(chunk_size) {
                    let chunk_end = chunk_start + chunk.len();

                    let anchor_refs: Vec<&str> =
                        chunk.iter().map(|e| e.anchor_text.as_str()).collect();
                    let anchor_vecs = match embedder.embed(&anchor_refs) {
                        Ok(v) => match validate_embed_response(&v, chunk.len(), self.emb_dim) {
                            Ok(()) => v,
                            Err(_e) => {
                                self.step_num += 1;
                                self.dropped_batches += 1;
                                self.total_batches += 1;
                                self.dropped_samples += chunk.len() as u64;
                                total_dropped += chunk.len() as u64;
                                chunk_start = chunk_end;
                                continue;
                            }
                        },
                        Err(_e) => {
                            self.step_num += 1;
                            self.dropped_batches += 1;
                            self.total_batches += 1;
                            self.dropped_samples += chunk.len() as u64;
                            total_dropped += chunk.len() as u64;
                            chunk_start = chunk_end;
                            continue;
                        }
                    };

                    // Embed candidates with dedup (reuse anchor embedding when same text)
                    let mut candidate_indices = Vec::with_capacity(chunk.len());
                    let mut unique_refs = Vec::new();
                    for entry in chunk {
                        if entry.candidate_text == entry.anchor_text {
                            candidate_indices.push(None);
                        } else {
                            candidate_indices.push(Some(unique_refs.len()));
                            unique_refs.push(entry.candidate_text.as_str());
                        }
                    }

                    let candidate_embs = if unique_refs.is_empty() {
                        Vec::new()
                    } else {
                        match embedder.embed(&unique_refs) {
                            Ok(v) => {
                                match validate_embed_response(&v, unique_refs.len(), self.emb_dim) {
                                    Ok(()) => v,
                                    Err(_e) => {
                                        self.step_num += 1;
                                        self.dropped_batches += 1;
                                        self.total_batches += 1;
                                        self.dropped_samples += chunk.len() as u64;
                                        total_dropped += chunk.len() as u64;
                                        chunk_start = chunk_end;
                                        continue;
                                    }
                                }
                            }
                            Err(_e) => {
                                self.step_num += 1;
                                self.dropped_batches += 1;
                                self.total_batches += 1;
                                self.dropped_samples += chunk.len() as u64;
                                total_dropped += chunk.len() as u64;
                                chunk_start = chunk_end;
                                continue;
                            }
                        }
                    };

                    self.step_num += 1;
                    self.total_batches += 1;
                    total_processed += chunk.len() as u64;

                    // Resolve candidate options (push None for identical, Some for unique)
                    for i in 0..chunk.len() {
                        let candidate_vec_opt = match candidate_indices[i] {
                            Some(idx) => Some(candidate_embs[idx].clone()),
                            None => None, // Defer anchor clone to Phase 2
                        };
                        all_candidate_vecs.push(candidate_vec_opt);
                    }

                    // Consume anchor_vecs exactly once
                    all_anchor_vecs.extend(anchor_vecs);

                    for idx in chunk_start..chunk_end {
                        keep[idx] = true;
                    }
                    chunk_start = chunk_end;
                }

                // Phase 2: Consume and Assemble (zero-copy string moves)
                let mut vec_iter = all_anchor_vecs.into_iter().zip(all_candidate_vecs.into_iter());

                for (i, entry) in entries.into_iter().enumerate() {
                    if !keep[i] {
                        continue;
                    }

                    let (a_vec, c_vec_opt) = vec_iter.next().expect("sync error in Phase 2");
                    let final_c_vec = c_vec_opt.unwrap_or_else(|| a_vec.clone());

                    self.pending.push_pair(PendingPair {
                        anchor_text: entry.anchor_text,       // MOVED
                        anchor_vec: a_vec,                    // MOVED
                        candidate_text: entry.candidate_text,  // MOVED
                        candidate_vec: final_c_vec,            // MOVED
                        label: entry.label,
                    });
                }
            }
            SamplerBatch::Triplets(entries) => {
                let mut all_anchor_vecs = Vec::with_capacity(entries.len());
                let mut all_pos_vecs = Vec::with_capacity(entries.len());
                let mut all_neg_vecs = Vec::with_capacity(entries.len());
                let mut keep = vec![false; entries.len()];
                let mut chunk_start = 0;

                // Phase 1: Borrow and Embed (entries untouched)
                for chunk in entries.chunks(chunk_size) {
                    let chunk_end = chunk_start + chunk.len();

                    // Collect unique texts across all 3 roles with index maps
                    let mut unique_texts = Vec::new();
                    let mut anchor_map = Vec::with_capacity(chunk.len());
                    let mut pos_map = Vec::with_capacity(chunk.len());
                    let mut neg_map = Vec::with_capacity(chunk.len());

                    for entry in chunk {
                        anchor_map.push(unique_texts.len());
                        unique_texts.push(entry.anchor_text.as_str());

                        if entry.pos_text == entry.anchor_text {
                            pos_map.push(anchor_map.last().copied().unwrap());
                        } else {
                            pos_map.push(unique_texts.len());
                            unique_texts.push(entry.pos_text.as_str());
                        }

                        if entry.neg_text == entry.anchor_text {
                            neg_map.push(anchor_map.last().copied().unwrap());
                        } else if entry.neg_text == entry.pos_text {
                            neg_map.push(pos_map.last().copied().unwrap());
                        } else {
                            neg_map.push(unique_texts.len());
                            unique_texts.push(entry.neg_text.as_str());
                        }
                    }

                    let embedded = match embedder.embed(&unique_texts) {
                        Ok(v) => match validate_embed_response(&v, unique_texts.len(), self.emb_dim) {
                            Ok(()) => v,
                            Err(_e) => {
                                self.step_num += 1;
                                self.dropped_batches += 1;
                                self.total_batches += 1;
                                self.dropped_samples += chunk.len() as u64;
                                total_dropped += chunk.len() as u64;
                                chunk_start = chunk_end;
                                continue;
                            }
                        },
                        Err(_e) => {
                            self.step_num += 1;
                            self.dropped_batches += 1;
                            self.total_batches += 1;
                            self.dropped_samples += chunk.len() as u64;
                            total_dropped += chunk.len() as u64;
                            chunk_start = chunk_end;
                            continue;
                        }
                    };

                    self.step_num += 1;
                    self.total_batches += 1;
                    total_processed += chunk.len() as u64;

                    for local_i in 0..chunk.len() {
                        all_anchor_vecs.push(embedded[anchor_map[local_i]].clone());
                        all_pos_vecs.push(embedded[pos_map[local_i]].clone());
                        all_neg_vecs.push(embedded[neg_map[local_i]].clone());
                    }

                    for idx in chunk_start..chunk_end {
                        keep[idx] = true;
                    }
                    chunk_start = chunk_end;
                }

                // Phase 2: Consume and Assemble (zero-copy string moves)
                let mut vec_iter = all_anchor_vecs.into_iter()
                    .zip(all_pos_vecs.into_iter())
                    .zip(all_neg_vecs.into_iter());

                for (i, entry) in entries.into_iter().enumerate() {
                    if !keep[i] {
                        continue;
                    }

                    let ((a_vec, p_vec), n_vec) = vec_iter.next().expect("sync error in Phase 2");

                    self.pending.push_triplet(PendingTriplet {
                        anchor_text: entry.anchor_text,  // MOVED
                        anchor_vec: a_vec,               // MOVED
                        pos_text: entry.pos_text,        // MOVED
                        pos_vec: p_vec,                  // MOVED
                        neg_text: entry.neg_text,        // MOVED
                        neg_vec: n_vec,                  // MOVED
                    });
                }
            }
        }

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
    let pending_count = state.pending.len();
    if pending_count == 0 {
        return Ok(());
    }

    match state.pending {
        PendingState::Pairs(ref pairs) => {
            let entries: Vec<PairWriteEntry> = pairs
                .iter()
                .map(|p| PairWriteEntry {
                    anchor_text: p.anchor_text.as_str(),
                    anchor_vec: &p.anchor_vec,
                    candidate_text: p.candidate_text.as_str(),
                    candidate_vec: &p.candidate_vec,
                    label: &p.label,
                })
                .collect();
            state
                .store
                .write_pairs(state.total_written, &PairWriteArgs { entries: &entries })?;
        }
        PendingState::Triplets(ref triplets) => {
            let entries: Vec<TripletWriteEntry> = triplets
                .iter()
                .map(|t| TripletWriteEntry {
                    anchor_text: t.anchor_text.as_str(),
                    anchor_vec: &t.anchor_vec,
                    pos_text: t.pos_text.as_str(),
                    pos_vec: &t.pos_vec,
                    neg_text: t.neg_text.as_str(),
                    neg_vec: &t.neg_vec,
                })
                .collect();
            state
                .store
                .write_triplets(state.total_written, &TripletWriteArgs { entries: &entries })?;
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
                args.entries.iter().map(|e| e.anchor_vec.to_vec()).collect(),
                args.entries
                    .iter()
                    .map(|e| e.candidate_vec.to_vec())
                    .collect(),
            ));
            self.len.fetch_add(args.entries.len(), Ordering::Relaxed);
            Ok(())
        }

        fn write_triplets(&self, start_idx: u64, args: &TripletWriteArgs<'_>) -> Result<()> {
            self.triplet_writes.lock().unwrap().push((
                start_idx,
                args.entries.iter().map(|e| e.anchor_vec.to_vec()).collect(),
                args.entries.iter().map(|e| e.pos_vec.to_vec()).collect(),
                args.entries.iter().map(|e| e.neg_vec.to_vec()).collect(),
            ));
            self.len.fetch_add(args.entries.len(), Ordering::Relaxed);
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

    // ---- helpers for AoS pending state assertions ----
    fn pending_pair_anchor_texts(state: &SplitState<MockStore>) -> Vec<String> {
        match &state.pending {
            PendingState::Pairs(v) => v.iter().map(|p| p.anchor_text.clone()).collect(),
            _ => panic!("expected Pair pending state"),
        }
    }
    fn pending_pair_candidate_texts(state: &SplitState<MockStore>) -> Vec<String> {
        match &state.pending {
            PendingState::Pairs(v) => v.iter().map(|p| p.candidate_text.clone()).collect(),
            _ => panic!("expected Pair pending state"),
        }
    }
    fn pending_pair_anchor_vecs(state: &SplitState<MockStore>) -> Vec<Vec<f32>> {
        match &state.pending {
            PendingState::Pairs(v) => v.iter().map(|p| p.anchor_vec.clone()).collect(),
            _ => panic!("expected Pair pending state"),
        }
    }
    fn pending_pair_candidate_vecs(state: &SplitState<MockStore>) -> Vec<Vec<f32>> {
        match &state.pending {
            PendingState::Pairs(v) => v.iter().map(|p| p.candidate_vec.clone()).collect(),
            _ => panic!("expected Pair pending state"),
        }
    }
    fn pending_triplet_anchor_texts(state: &SplitState<MockStore>) -> Vec<String> {
        match &state.pending {
            PendingState::Triplets(v) => v.iter().map(|t| t.anchor_text.clone()).collect(),
            _ => panic!("expected Triplet pending state"),
        }
    }
    fn pending_triplet_pos_texts(state: &SplitState<MockStore>) -> Vec<String> {
        match &state.pending {
            PendingState::Triplets(v) => v.iter().map(|t| t.pos_text.clone()).collect(),
            _ => panic!("expected Triplet pending state"),
        }
    }
    fn pending_triplet_neg_texts(state: &SplitState<MockStore>) -> Vec<String> {
        match &state.pending {
            PendingState::Triplets(v) => v.iter().map(|t| t.neg_text.clone()).collect(),
            _ => panic!("expected Triplet pending state"),
        }
    }
    fn pending_triplet_anchor_vecs(state: &SplitState<MockStore>) -> Vec<Vec<f32>> {
        match &state.pending {
            PendingState::Triplets(v) => v.iter().map(|t| t.anchor_vec.clone()).collect(),
            _ => panic!("expected Triplet pending state"),
        }
    }
    fn pending_triplet_pos_vecs(state: &SplitState<MockStore>) -> Vec<Vec<f32>> {
        match &state.pending {
            PendingState::Triplets(v) => v.iter().map(|t| t.pos_vec.clone()).collect(),
            _ => panic!("expected Triplet pending state"),
        }
    }
    fn pending_triplet_neg_vecs(state: &SplitState<MockStore>) -> Vec<Vec<f32>> {
        match &state.pending {
            PendingState::Triplets(v) => v.iter().map(|t| t.neg_vec.clone()).collect(),
            _ => panic!("expected Triplet pending state"),
        }
    }

    struct FailingEmbedder;

    impl Embedder for FailingEmbedder {
        fn embed(&self, _texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Err(SchedulerError::Msg("embed failed".into()))
        }
    }

    fn make_state(store: MockStore, mode: EmbedMode, emb_dim: usize) -> SplitState<MockStore> {
        let pending = match mode {
            EmbedMode::Pair => PendingState::Pairs(Vec::new()),
            EmbedMode::Triplet => PendingState::Triplets(Vec::new()),
        };
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
            pending,
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

        state.pending.push_pair(PendingPair {
            anchor_text: "a1".into(),
            anchor_vec: vec![1.0, 2.0],
            candidate_text: "p1".into(),
            candidate_vec: vec![5.0, 6.0],
            label: PairLabel::Positive,
        });
        state.pending.push_pair(PendingPair {
            anchor_text: "a2".into(),
            anchor_vec: vec![3.0, 4.0],
            candidate_text: "p2".into(),
            candidate_vec: vec![7.0, 8.0],
            label: PairLabel::Positive,
        });

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

        let batch = SamplerBatch::Pairs(vec![
            PairEntry {
                anchor_text: "a1".into(),
                candidate_text: "p1".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "a2".into(),
                candidate_text: "p2".into(),
                label: PairLabel::Positive,
            },
        ]);

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

        let batch = SamplerBatch::Triplets(vec![TripletEntry {
            anchor_text: "a1".into(),
            pos_text: "p1".into(),
            neg_text: "n1".into(),
        }]);

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 1);
        assert_eq!(result.samples_dropped, 0);

        assert_eq!(state.pending_len(), 1);
        assert_eq!(pending_triplet_neg_vecs(&state).len(), 1);
    }

    #[test]
    fn step_flush_at_boundary() {
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 2);
        state.step_num = 99; // next step will be 100
        let embedder = MockEmbedder::new(2);
        let config = SchedulerConfig::new(1, 1, 2, 100);

        let batch = SamplerBatch::Pairs(vec![PairEntry {
            anchor_text: "a1".into(),
            candidate_text: "p1".into(),
            label: PairLabel::Positive,
        }]);

        let result = state.step(batch, &embedder, &config).unwrap();
        assert!(result.should_flush);
    }

    #[test]
    fn step_embed_failure_drops_batch() {
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 2);
        let embedder = FailingEmbedder;
        let config = SchedulerConfig::new(2, 2, 2, 100);

        let batch = SamplerBatch::Pairs(vec![
            PairEntry {
                anchor_text: "a1".into(),
                candidate_text: "p1".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "a2".into(),
                candidate_text: "p2".into(),
                label: PairLabel::Positive,
            },
        ]);

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

        let batch = SamplerBatch::Pairs(vec![
            PairEntry {
                anchor_text: "a1".into(),
                candidate_text: "p1".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "a2".into(),
                candidate_text: "p2".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "a3".into(),
                candidate_text: "p3".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "a4".into(),
                candidate_text: "p4".into(),
                label: PairLabel::Positive,
            },
        ]);

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

        let batch = SamplerBatch::Pairs(vec![]);

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
        let batch = SamplerBatch::Pairs(vec![PairEntry {
            anchor_text: "same".into(),
            candidate_text: "same".into(),
            label: PairLabel::Positive,
        }]);

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 1);
        assert_eq!(state.pending_len(), 1);
        // MockEmbedder called once for anchor; pos reuses anchor_vecs
        // So pending_pos_vecs should equal pending_anchor_vecs
        assert_eq!(
            pending_pair_anchor_vecs(&state),
            pending_pair_candidate_vecs(&state)
        );
    }

    #[test]
    fn step_neg_equals_anchor_reuses_vectors() {
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Triplet, 2);
        let embedder = MockEmbedder::new(2);
        let config = SchedulerConfig::new(1, 1, 2, 100);

        let batch = SamplerBatch::Triplets(vec![TripletEntry {
            anchor_text: "a".into(),
            pos_text: "p".into(),
            neg_text: "a".into(),
        }]);

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 1);
        assert_eq!(pending_triplet_neg_vecs(&state).len(), 1);
        // neg should reuse anchor vectors
        assert_eq!(
            pending_triplet_neg_vecs(&state),
            pending_triplet_anchor_vecs(&state)
        );
    }

    #[test]
    fn step_neg_equals_pos_reuses_vectors() {
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Triplet, 2);
        let embedder = MockEmbedder::new(2);
        let config = SchedulerConfig::new(1, 1, 2, 100);

        let batch = SamplerBatch::Triplets(vec![TripletEntry {
            anchor_text: "a".into(),
            pos_text: "p".into(),
            neg_text: "p".into(),
        }]);

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 1);
        // neg should reuse pos vectors
        assert_eq!(
            pending_triplet_neg_vecs(&state),
            pending_triplet_pos_vecs(&state)
        );
    }

    #[test]
    fn step_neg_embed_failure_drops_batch() {
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Triplet, 2);
        let _embedder = MockEmbedder::new(2);
        let config = SchedulerConfig::new(1, 1, 2, 100);

        // Make neg different from anchor and pos so it triggers embed call
        let batch = SamplerBatch::Triplets(vec![TripletEntry {
            anchor_text: "a".into(),
            pos_text: "p".into(),
            neg_text: "n".into(),
        }]);

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
        // Test that push_pair works with PairLabel::Negative
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 2);

        state.pending.push_pair(PendingPair {
            anchor_text: "a".into(),
            anchor_vec: vec![1.0, 2.0],
            candidate_text: "p".into(),
            candidate_vec: vec![3.0, 4.0],
            label: PairLabel::Negative,
        });

        assert_eq!(state.pending_len(), 1);
    }

    #[test]
    fn flush_pending_triplet_mode() {
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Triplet, 2);
        state.pending = PendingState::Triplets(vec![
            PendingTriplet {
                anchor_text: "a1".into(),
                anchor_vec: vec![1.0, 2.0],
                pos_text: "p1".into(),
                pos_vec: vec![5.0, 6.0],
                neg_text: "n1".into(),
                neg_vec: vec![9.0, 10.0],
            },
            PendingTriplet {
                anchor_text: "a2".into(),
                anchor_vec: vec![3.0, 4.0],
                pos_text: "p2".into(),
                pos_vec: vec![7.0, 8.0],
                neg_text: "n2".into(),
                neg_vec: vec![11.0, 12.0],
            },
        ]);

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
        state.pending.push_pair(PendingPair {
            anchor_text: "a".into(),
            anchor_vec: vec![1.0, 2.0],
            candidate_text: "p".into(),
            candidate_vec: vec![3.0, 4.0],
            label: PairLabel::Positive,
        });

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

        let batch = SamplerBatch::Pairs(vec![
            PairEntry {
                anchor_text: "a1".into(),
                candidate_text: "p1".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "a2".into(),
                candidate_text: "p2".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "a3".into(),
                candidate_text: "p3".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "a4".into(),
                candidate_text: "p4".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "a5".into(),
                candidate_text: "p5".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "a6".into(),
                candidate_text: "p6".into(),
                label: PairLabel::Positive,
            },
        ]);

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 6);
        assert_eq!(state.pending_len(), 6);
        // 6 texts / 2 chunk_size = 3 embed calls = 3 step increments
        assert_eq!(state.step_num, 3);

        // CRITICAL: verify alignment — each pending text must match its vector
        assert_eq!(pending_pair_anchor_texts(&state).len(), 6);
        assert_eq!(pending_pair_anchor_vecs(&state).len(), 6);
        assert_eq!(pending_pair_candidate_texts(&state).len(), 6);
        assert_eq!(pending_pair_candidate_vecs(&state).len(), 6);

        for i in 0..6 {
            let anchor_text = format!("a{}", i + 1);
            let pos_text = format!("p{}", i + 1);
            let expected_anchor_id: f32 = anchor_text.chars().map(|c| c as u32 as f32).sum();
            let expected_pos_id: f32 = pos_text.chars().map(|c| c as u32 as f32).sum();
            // vec[1] = chunk-local position (embed_batch=2, so resets every 2)
            let chunk_local_pos = (i % 2) as f32;

            assert_eq!(
                pending_pair_anchor_texts(&state)[i],
                anchor_text,
                "anchor text mismatch at index {i}"
            );
            assert_eq!(
                pending_pair_anchor_vecs(&state)[i][0],
                expected_anchor_id,
                "anchor vector text_id mismatch at index {i}"
            );
            assert_eq!(
                pending_pair_anchor_vecs(&state)[i][1],
                chunk_local_pos,
                "anchor vector position mismatch at index {i}"
            );
            assert_eq!(
                pending_pair_candidate_texts(&state)[i],
                pos_text,
                "pos text mismatch at index {i}"
            );
            assert_eq!(
                pending_pair_candidate_vecs(&state)[i][0],
                expected_pos_id,
                "pos vector text_id mismatch at index {i}"
            );
            assert_eq!(
                pending_pair_candidate_vecs(&state)[i][1],
                chunk_local_pos,
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

        let batch = SamplerBatch::Triplets(vec![
            TripletEntry {
                anchor_text: "a1".into(),
                pos_text: "p1".into(),
                neg_text: "n1".into(),
            },
            TripletEntry {
                anchor_text: "a2".into(),
                pos_text: "p2".into(),
                neg_text: "n2".into(),
            },
            TripletEntry {
                anchor_text: "a3".into(),
                pos_text: "p3".into(),
                neg_text: "n3".into(),
            },
            TripletEntry {
                anchor_text: "a4".into(),
                pos_text: "p4".into(),
                neg_text: "n4".into(),
            },
            TripletEntry {
                anchor_text: "a5".into(),
                pos_text: "p5".into(),
                neg_text: "n5".into(),
            },
        ]);

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 5);
        assert_eq!(state.pending_len(), 5);

        // Verify texts and vectors are correctly aligned (exact text-identity match)
        let at = pending_triplet_anchor_texts(&state);
        let pt = pending_triplet_pos_texts(&state);
        let nt = pending_triplet_neg_texts(&state);
        let av = pending_triplet_anchor_vecs(&state);
        let pv = pending_triplet_pos_vecs(&state);
        let nv = pending_triplet_neg_vecs(&state);

        for i in 0..5 {
            let anchor_text = format!("a{}", i + 1);
            let pos_text = format!("p{}", i + 1);
            let neg_text = format!("n{}", i + 1);
            let expected_anchor_id: f32 = anchor_text.chars().map(|c| c as u32 as f32).sum();
            let expected_pos_id: f32 = pos_text.chars().map(|c| c as u32 as f32).sum();
            let expected_neg_id: f32 = neg_text.chars().map(|c| c as u32 as f32).sum();

            assert_eq!(at[i], anchor_text, "anchor text mismatch at {i}");
            assert_eq!(
                av[i][0], expected_anchor_id,
                "anchor vec text_id mismatch at {i}"
            );
            assert_eq!(pt[i], pos_text, "pos text mismatch at {i}");
            assert_eq!(pv[i][0], expected_pos_id, "pos vec text_id mismatch at {i}");
            assert_eq!(nt[i], neg_text, "neg text mismatch at {i}");
            assert_eq!(nv[i][0], expected_neg_id, "neg vec text_id mismatch at {i}");
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

        let batch = SamplerBatch::Pairs(vec![
            PairEntry {
                anchor_text: "x1".into(),
                candidate_text: "y1".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "x2".into(),
                candidate_text: "y2".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "x3".into(),
                candidate_text: "y3".into(),
                label: PairLabel::Positive,
            },
        ]);

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 3);
        assert_eq!(state.pending_len(), 3);

        for i in 0..3 {
            let text = format!("x{}", i + 1);
            let expected_id: f32 = text.chars().map(|c| c as u32 as f32).sum();
            assert_eq!(pending_pair_anchor_texts(&state)[i], text);
            assert_eq!(pending_pair_anchor_vecs(&state)[i][0], expected_id);
            assert_eq!(
                pending_pair_candidate_texts(&state)[i],
                format!("y{}", i + 1)
            );
        }
    }

    #[test]
    fn alignment_exact_match_sampler_equals_embed() {
        // embed_batch=4, sampler_batch=4 → 1 chunk, no chunking
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 4);
        let embedder = AlignmentEmbedder::new(4);
        let config = SchedulerConfig::new(4, 4, 4, 100);

        let batch = SamplerBatch::Pairs(vec![
            PairEntry {
                anchor_text: "a".into(),
                candidate_text: "e".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "b".into(),
                candidate_text: "f".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "c".into(),
                candidate_text: "g".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "d".into(),
                candidate_text: "h".into(),
                label: PairLabel::Positive,
            },
        ]);

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 4);
        assert_eq!(state.step_num, 1); // single chunk

        let at_vec = pending_pair_anchor_texts(&state);
        let pt_vec = pending_pair_candidate_texts(&state);
        for (i, (at, pt)) in at_vec.iter().zip(pt_vec.iter()).enumerate() {
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
        let batch = SamplerBatch::Pairs(
            texts
                .iter()
                .map(|t| PairEntry {
                    anchor_text: t.clone(),
                    candidate_text: t.clone(),
                    label: PairLabel::Positive,
                })
                .collect(),
        );

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 7);
        assert_eq!(state.pending_len(), 7);
        assert_eq!(state.step_num, 3); // ceil(7/3) = 3

        // All 7 entries must be correctly aligned
        for i in 0..7 {
            let text = format!("t{i}");
            let expected_id: f32 = text.chars().map(|c| c as u32 as f32).sum();
            assert_eq!(
                pending_pair_anchor_texts(&state)[i],
                text,
                "text mismatch at {i}"
            );
            assert_eq!(
                pending_pair_anchor_vecs(&state)[i][0],
                expected_id,
                "vector text_id mismatch at {i}"
            );
            // pos == anchor, so pos vectors should equal anchor vectors
            assert_eq!(
                pending_pair_anchor_vecs(&state)[i],
                pending_pair_candidate_vecs(&state)[i],
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
        let batch1 = SamplerBatch::Pairs(vec![
            PairEntry {
                anchor_text: "step1_a".into(),
                candidate_text: "step1_p1".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "step1_b".into(),
                candidate_text: "step1_p2".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "step1_c".into(),
                candidate_text: "step1_p3".into(),
                label: PairLabel::Positive,
            },
        ]);
        state.step(batch1, &embedder, &config).unwrap();
        assert_eq!(state.pending_len(), 3);

        // Step 2: 2 texts → 1 chunk
        let batch2 = SamplerBatch::Pairs(vec![
            PairEntry {
                anchor_text: "step2_a".into(),
                candidate_text: "step2_p1".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "step2_b".into(),
                candidate_text: "step2_p2".into(),
                label: PairLabel::Positive,
            },
        ]);
        state.step(batch2, &embedder, &config).unwrap();
        assert_eq!(state.pending_len(), 5);

        // Verify first 3 entries are from step1, last 2 from step2
        assert_eq!(pending_pair_anchor_texts(&state)[0], "step1_a");
        assert_eq!(pending_pair_anchor_texts(&state)[1], "step1_b");
        assert_eq!(pending_pair_anchor_texts(&state)[2], "step1_c");
        assert_eq!(pending_pair_anchor_texts(&state)[3], "step2_a");
        assert_eq!(pending_pair_anchor_texts(&state)[4], "step2_b");

        // Verify vectors match their texts
        for i in 0..5 {
            let text = &pending_pair_anchor_texts(&state)[i];
            let expected_id: f32 = text.chars().map(|c| c as u32 as f32).sum();
            assert_eq!(
                pending_pair_anchor_vecs(&state)[i][0],
                expected_id,
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

        let batch = SamplerBatch::Triplets(vec![
            TripletEntry {
                anchor_text: "a_odd".into(),
                pos_text: "p_odd".into(),
                neg_text: "n_odd".into(),
            },
            TripletEntry {
                anchor_text: "a_even".into(),
                pos_text: "p_even".into(),
                neg_text: "n_even".into(),
            },
            TripletEntry {
                anchor_text: "a_third".into(),
                pos_text: "p_third".into(),
                neg_text: "n_third".into(),
            },
            TripletEntry {
                anchor_text: "a_fourth".into(),
                pos_text: "p_fourth".into(),
                neg_text: "n_fourth".into(),
            },
        ]);

        state.step(batch, &embedder, &config).unwrap();

        // At index 0: anchor="a_odd", pos="p_odd", neg="n_odd"
        assert_eq!(pending_triplet_anchor_texts(&state)[0], "a_odd");
        assert_eq!(pending_triplet_pos_texts(&state)[0], "p_odd");
        assert_eq!(pending_triplet_neg_texts(&state)[0], "n_odd");

        // At index 2 (chunk boundary): anchor="a_third", pos="p_third", neg="n_third"
        assert_eq!(pending_triplet_anchor_texts(&state)[2], "a_third");
        assert_eq!(pending_triplet_pos_texts(&state)[2], "p_third");
        assert_eq!(pending_triplet_neg_texts(&state)[2], "n_third");

        // Verify neg vectors are NOT swapped at boundary
        let expected_neg2_id: f32 = "n_third".chars().map(|c| c as u32 as f32).sum();
        assert_eq!(
            pending_triplet_neg_vecs(&state)[2][0],
            expected_neg2_id,
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

        let batch = SamplerBatch::Pairs(vec![
            PairEntry {
                anchor_text: "z1".into(),
                candidate_text: "q1".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "z2".into(),
                candidate_text: "q2".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "z3".into(),
                candidate_text: "q3".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "z4".into(),
                candidate_text: "q4".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "z5".into(),
                candidate_text: "q5".into(),
                label: PairLabel::Positive,
            },
        ]);

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 5);
        assert_eq!(state.pending_len(), 5);

        // Texts must be aligned even if vectors are all zero
        for i in 0..5 {
            assert_eq!(pending_pair_anchor_texts(&state)[i], format!("z{}", i + 1));
            assert_eq!(
                pending_pair_candidate_texts(&state)[i],
                format!("q{}", i + 1)
            );
            assert_eq!(pending_pair_anchor_vecs(&state)[i], vec![0.0; 4]);
            assert_eq!(pending_pair_candidate_vecs(&state)[i], vec![0.0; 4]);
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

        let batch = SamplerBatch::Pairs(vec![
            PairEntry {
                anchor_text: "a1".into(),
                candidate_text: "p1".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "a2".into(),
                candidate_text: "p2".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "a3".into(),
                candidate_text: "p3".into(),
                label: PairLabel::Positive,
            },
        ]);

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

        let batch = SamplerBatch::Pairs(vec![
            PairEntry {
                anchor_text: "a1".into(),
                candidate_text: "p1".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "a2".into(),
                candidate_text: "p2".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "a3".into(),
                candidate_text: "p3".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "a4".into(),
                candidate_text: "p4".into(),
                label: PairLabel::Positive,
            },
        ]);

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

        let batch = SamplerBatch::Pairs(vec![]);

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

        let batch = SamplerBatch::Pairs(vec![PairEntry {
            anchor_text: "solo".into(),
            candidate_text: "partner".into(),
            label: PairLabel::Positive,
        }]);

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 1);
        assert_eq!(state.pending_len(), 1);

        let expected_id: f32 = "solo".chars().map(|c| c as u32 as f32).sum();
        assert_eq!(pending_pair_anchor_texts(&state)[0], "solo");
        assert_eq!(pending_pair_anchor_vecs(&state)[0][0], expected_id);
        assert_eq!(pending_pair_candidate_texts(&state)[0], "partner");
    }

    #[test]
    fn alignment_all_chunks_dropped_no_residue() {
        // If ALL chunks fail validation, pending must be completely empty
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 4);
        let embedder = InfEmbedder; // returns inf → all chunks rejected
        let config = SchedulerConfig::new(6, 2, 4, 100);

        let batch = SamplerBatch::Pairs(vec![
            PairEntry {
                anchor_text: "a1".into(),
                candidate_text: "p1".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "a2".into(),
                candidate_text: "p2".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "a3".into(),
                candidate_text: "p3".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "a4".into(),
                candidate_text: "p4".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "a5".into(),
                candidate_text: "p5".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "a6".into(),
                candidate_text: "p6".into(),
                label: PairLabel::Positive,
            },
        ]);

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 0);
        assert_eq!(result.samples_dropped, 6);
        assert_eq!(state.dropped_batches, 3); // 6/2 = 3 chunks
        assert!(state.is_pending_empty());
        assert_eq!(pending_pair_anchor_texts(&state).len(), 0);
        assert_eq!(pending_pair_anchor_vecs(&state).len(), 0);
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

        let batch = SamplerBatch::Pairs(vec![
            PairEntry {
                anchor_text: "a1".into(),
                candidate_text: "p1".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "a2".into(),
                candidate_text: "p2".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "a3".into(),
                candidate_text: "p3".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "a4".into(),
                candidate_text: "p4".into(),
                label: PairLabel::Positive,
            },
        ]);

        let result = state.step(batch, &embedder, &config).unwrap();
        // Chunk 0 [a1,a2] failed → dropped
        // Chunk 1 [a3,a4] succeeded → accumulated
        assert_eq!(result.samples_processed, 2);
        assert_eq!(result.samples_dropped, 2);
        assert_eq!(state.pending_len(), 2);
        // Only chunk 1's data should be in pending
        assert_eq!(pending_pair_anchor_texts(&state)[0], "a3");
        assert_eq!(pending_pair_anchor_texts(&state)[1], "a4");
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
            let batch = SamplerBatch::Pairs(vec![
                PairEntry {
                    anchor_text: format!("a{i}"),
                    candidate_text: format!("p{i}"),
                    label: PairLabel::Positive,
                },
                PairEntry {
                    anchor_text: format!("b{i}"),
                    candidate_text: format!("q{i}"),
                    label: PairLabel::Positive,
                },
                PairEntry {
                    anchor_text: format!("c{i}"),
                    candidate_text: format!("r{i}"),
                    label: PairLabel::Positive,
                },
                PairEntry {
                    anchor_text: format!("d{i}"),
                    candidate_text: format!("s{i}"),
                    label: PairLabel::Positive,
                },
            ]);
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
            let batch = SamplerBatch::Pairs(vec![
                PairEntry {
                    anchor_text: format!("a{i}"),
                    candidate_text: format!("p{i}"),
                    label: PairLabel::Positive,
                },
                PairEntry {
                    anchor_text: format!("b{i}"),
                    candidate_text: format!("q{i}"),
                    label: PairLabel::Positive,
                },
                PairEntry {
                    anchor_text: format!("c{i}"),
                    candidate_text: format!("r{i}"),
                    label: PairLabel::Positive,
                },
                PairEntry {
                    anchor_text: format!("d{i}"),
                    candidate_text: format!("s{i}"),
                    label: PairLabel::Positive,
                },
            ]);
            let result = state.step(batch, &embedder, &config);
            result.unwrap_or_else(|e| panic!("batch {i} should not trigger circuit breaker: {e}"));
        }
        assert_eq!(state.total_batches, 30);
        assert_eq!(state.dropped_batches, 1);
    }

    // ===================================================================
    // NEW TESTS — two-phase execution, push safety, label preservation
    // ===================================================================

    #[test]
    fn step_pair_candidate_dedup_defers_clone() {
        // When candidate_text == anchor_text, the candidate vector should equal
        // the anchor vector (clone happens in Phase 2, not Phase 1)
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 2);
        let embedder = MockEmbedder::new(2);
        let config = SchedulerConfig::new(1, 1, 2, 100);

        let batch = SamplerBatch::Pairs(vec![PairEntry {
            anchor_text: "same".into(),
            candidate_text: "same".into(),
            label: PairLabel::Positive,
        }]);

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 1);
        assert_eq!(state.pending_len(), 1);
        // Candidate vec should equal anchor vec (dedup defers clone to Phase 2)
        assert_eq!(
            pending_pair_anchor_vecs(&state),
            pending_pair_candidate_vecs(&state)
        );
        // Texts should be moved, not cloned
        assert_eq!(pending_pair_anchor_texts(&state), vec!["same"]);
        assert_eq!(pending_pair_candidate_texts(&state), vec!["same"]);
    }

    #[test]
    fn step_pair_partial_chunk_failure_alignment() {
        // Chunk 1 of 3 fails → chunks 0 and 2 must be correctly aligned
        struct FailMiddleChunkEmbedder {
            call_count: std::sync::atomic::AtomicUsize,
        }
        impl Embedder for FailMiddleChunkEmbedder {
            fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
                let call = self.call_count.fetch_add(1, Ordering::Relaxed);
                if call == 2 {
                    // Third embed call = chunk 1 anchor → fail
                    Err(SchedulerError::Msg("transient failure".into()))
                } else {
                    Ok(texts.iter().map(|_| vec![1.0; 4]).collect())
                }
            }
        }

        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 4);
        let embedder = FailMiddleChunkEmbedder {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        };
        // 6 entries, embed_batch=2 → 3 chunks: [0,1], [2,3], [4,5]
        // Chunk 1 (entries 2,3) anchor embed fails
        let config = SchedulerConfig::new(6, 2, 4, 100);

        let batch = SamplerBatch::Pairs(vec![
            PairEntry {
                anchor_text: "a0".into(),
                candidate_text: "p0".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "a1".into(),
                candidate_text: "p1".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "a2".into(),
                candidate_text: "p2".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "a3".into(),
                candidate_text: "p3".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "a4".into(),
                candidate_text: "p4".into(),
                label: PairLabel::Positive,
            },
            PairEntry {
                anchor_text: "a5".into(),
                candidate_text: "p5".into(),
                label: PairLabel::Positive,
            },
        ]);

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 4); // chunks 0 and 2 succeeded
        assert_eq!(result.samples_dropped, 2); // chunk 1 failed
        assert_eq!(state.pending_len(), 4);

        // Verify alignment: chunk 0 entries first, then chunk 2 entries
        assert_eq!(pending_pair_anchor_texts(&state)[0], "a0");
        assert_eq!(pending_pair_anchor_texts(&state)[1], "a1");
        assert_eq!(pending_pair_anchor_texts(&state)[2], "a4");
        assert_eq!(pending_pair_anchor_texts(&state)[3], "a5");
    }

    #[test]
    fn step_triplet_partial_chunk_failure_alignment() {
        // Chunk 1 of 3 fails → chunks 0 and 2 must be correctly aligned
        struct FailMiddleChunkEmbedder {
            call_count: std::sync::atomic::AtomicUsize,
        }
        impl Embedder for FailMiddleChunkEmbedder {
            fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
                let call = self.call_count.fetch_add(1, Ordering::Relaxed);
                if call == 1 {
                    // Second embed call = chunk 1 → fail
                    Err(SchedulerError::Msg("transient failure".into()))
                } else {
                    Ok(texts.iter().map(|_| vec![1.0; 4]).collect())
                }
            }
        }

        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Triplet, 4);
        let embedder = FailMiddleChunkEmbedder {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        };
        // 6 entries, embed_batch=2 → 3 chunks: [0,1], [2,3], [4,5]
        let config = SchedulerConfig::new(6, 2, 4, 100);

        let batch = SamplerBatch::Triplets(vec![
            TripletEntry {
                anchor_text: "a0".into(),
                pos_text: "p0".into(),
                neg_text: "n0".into(),
            },
            TripletEntry {
                anchor_text: "a1".into(),
                pos_text: "p1".into(),
                neg_text: "n1".into(),
            },
            TripletEntry {
                anchor_text: "a2".into(),
                pos_text: "p2".into(),
                neg_text: "n2".into(),
            },
            TripletEntry {
                anchor_text: "a3".into(),
                pos_text: "p3".into(),
                neg_text: "n3".into(),
            },
            TripletEntry {
                anchor_text: "a4".into(),
                pos_text: "p4".into(),
                neg_text: "n4".into(),
            },
            TripletEntry {
                anchor_text: "a5".into(),
                pos_text: "p5".into(),
                neg_text: "n5".into(),
            },
        ]);

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 4);
        assert_eq!(result.samples_dropped, 2);
        assert_eq!(state.pending_len(), 4);

        // Verify alignment: chunk 0 entries first, then chunk 2 entries
        assert_eq!(pending_triplet_anchor_texts(&state)[0], "a0");
        assert_eq!(pending_triplet_pos_texts(&state)[0], "p0");
        assert_eq!(pending_triplet_neg_texts(&state)[0], "n0");
        assert_eq!(pending_triplet_anchor_texts(&state)[2], "a4");
        assert_eq!(pending_triplet_pos_texts(&state)[2], "p4");
        assert_eq!(pending_triplet_neg_texts(&state)[2], "n4");
    }

    #[test]
    #[should_panic(expected = "push_pair called on Triplets state")]
    fn push_pair_on_triplet_state_panics() {
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Triplet, 2);
        state.pending.push_pair(PendingPair {
            anchor_text: "a".into(),
            anchor_vec: vec![1.0, 2.0],
            candidate_text: "p".into(),
            candidate_vec: vec![3.0, 4.0],
            label: PairLabel::Positive,
        });
    }

    #[test]
    #[should_panic(expected = "push_triplet called on Pairs state")]
    fn push_triplet_on_pair_state_panics() {
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 2);
        state.pending.push_triplet(PendingTriplet {
            anchor_text: "a".into(),
            anchor_vec: vec![1.0, 2.0],
            pos_text: "p".into(),
            pos_vec: vec![3.0, 4.0],
            neg_text: "n".into(),
            neg_vec: vec![5.0, 6.0],
        });
    }

    #[test]
    fn pair_label_preserved_through_pipeline() {
        let store = MockStore::new();
        let mut state = make_state(store, EmbedMode::Pair, 2);
        let embedder = MockEmbedder::new(2);
        let config = SchedulerConfig::new(1, 1, 2, 100);

        let batch = SamplerBatch::Pairs(vec![PairEntry {
            anchor_text: "a".into(),
            candidate_text: "p".into(),
            label: PairLabel::Negative,
        }]);

        let result = state.step(batch, &embedder, &config).unwrap();
        assert_eq!(result.samples_processed, 1);

        // Verify label is preserved in pending state
        match &state.pending {
            PendingState::Pairs(v) => {
                assert_eq!(v.len(), 1);
                assert!(matches!(v[0].label, PairLabel::Negative));
            }
            _ => panic!("expected Pair pending state"),
        }

        // Flush and verify label reaches the store
        let provider = MockProvider;
        flush_pending(&mut state, &provider).unwrap();

        // Verify the MockStore received the write (label is in the PairWriteEntry)
        let writes = state.store.pair_writes.lock().unwrap();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].0, 0); // start_idx
    }
}
