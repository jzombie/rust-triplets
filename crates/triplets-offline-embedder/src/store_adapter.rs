//! Adapter bridging [`simd_r_drive::storage_engine::DataStore`] to the
//! [`EmbedStore`] trait, plus helpers for initializing per-split stores.

use simd_r_drive::storage_engine::DataStore;
use simd_r_drive::storage_engine::traits::DataStoreReader;
use triplets_core::SplitLabel;
use triplets_srd_source::srd_triplet::{self, SrdMode, SrdPairWriteEntry, SrdTripletWriteEntry};

use crate::split_state::{EmbedMode, PendingState, SplitState};
use crate::traits::{
    EmbedStore, PairWriteArgs, PairWriteEntry, Result, SchedulerError, TripletWriteArgs,
    TripletWriteEntry,
};

/// Newtype wrapping [`DataStore`] to implement [`EmbedStore`].
///
/// `DataStore` has its own inherent methods (`len`, etc.) whose signatures
/// clash with the trait methods, so a newtype sidesteps the ambiguity.
pub struct SrdStoreAdapter(pub DataStore);

impl EmbedStore for SrdStoreAdapter {
    fn write_pairs(&self, start_idx: u64, args: &PairWriteArgs<'_>) -> Result<()> {
        let srd_entries: Vec<SrdPairWriteEntry> = args
            .entries
            .iter()
            .map(|e| SrdPairWriteEntry {
                anchor_vec: e.anchor_vec,
                anchor_text: e.anchor_text,
                candidate_vec: e.candidate_vec,
                candidate_text: e.candidate_text,
                label: e.label,
            })
            .collect();
        srd_triplet::write_pair_entries(&self.0, start_idx, &srd_entries)
            .map_err(|e| SchedulerError::Msg(format!("write error: {e}")))
    }

    fn write_triplets(&self, start_idx: u64, args: &TripletWriteArgs<'_>) -> Result<()> {
        let srd_entries: Vec<SrdTripletWriteEntry> = args
            .entries
            .iter()
            .map(|e| SrdTripletWriteEntry {
                anchor_vec: e.anchor_vec,
                anchor_text: e.anchor_text,
                pos_vec: e.pos_vec,
                pos_text: e.pos_text,
                neg_vec: e.neg_vec,
                neg_text: e.neg_text,
            })
            .collect();
        srd_triplet::write_triplet_entries(&self.0, start_idx, &srd_entries)
            .map_err(|e| SchedulerError::Msg(format!("write error: {e}")))
    }

    fn len(&self) -> Result<u64> {
        self.0
            .len()
            .map(|n| n as u64)
            .map_err(|e| SchedulerError::Msg(format!("len error: {e}")))
    }
}

impl std::ops::Deref for SrdStoreAdapter {
    type Target = DataStore;
    fn deref(&self) -> &DataStore {
        &self.0
    }
}

/// Descriptor for a single split: label, display name, max samples, ratio weight.
pub type SplitDesc = (SplitLabel, &'static str, u64, f32);

/// Open (or create) per-split data stores and initialize [`SplitState`] entries.
///
/// Each split gets its own subdirectory under `out_dir` containing a
/// `data.srd` store file.  If the store already exists (resume), the
/// current sample count is read and used to compute `step_num` /
/// `batch_num` so the scheduler picks up where it left off.
pub fn init_split_states_with_batch(
    out_dir: &std::path::Path,
    split_descs: &[SplitDesc],
    embed_batch_size: usize,
    mode: SrdMode,
    emb_dim: usize,
    steps_per_batch: u64,
) -> Result<Vec<SplitState<SrdStoreAdapter>>> {
    let embed_mode = match mode {
        SrdMode::Pair => EmbedMode::Pair,
        SrdMode::Triplet => EmbedMode::Triplet,
    };
    let mut v = Vec::with_capacity(split_descs.len());
    for &(label, name, max, ratio) in split_descs {
        let split_dir = out_dir.join(name);
        std::fs::create_dir_all(&split_dir)
            .map_err(|e| SchedulerError::Msg(format!("create dir: {e}")))?;
        let store = DataStore::open(&split_dir.join("data.srd"))
            .map_err(|e| SchedulerError::Msg(format!("open store: {e}")))?;
        let already_done = store
            .len()
            .map_err(|e| SchedulerError::Msg(format!("store len: {e}")))?
            as u64;
        let step_num = already_done / embed_batch_size.max(1) as u64;
        let batch_num = step_num / steps_per_batch;
        v.push(SplitState {
            label,
            name,
            store: SrdStoreAdapter(store),
            mode: embed_mode,
            emb_dim,
            max,
            ratio,
            total_written: already_done,
            step_num,
            batch_num,
            pending: PendingState::Pairs(Vec::new()),
            exhausted: false,
            dropped_batches: 0,
            total_batches: 0,
            dropped_samples: 0,
            start: None,
            segment_base: already_done,
        });
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use triplets_core::data::PairLabel;

    #[test]
    fn srd_store_adapter_write_and_len() {
        let dir = TempDir::new().unwrap();
        let ds = DataStore::open(&dir.path().join("test.srd")).unwrap();
        let adapter = SrdStoreAdapter(ds);

        assert_eq!(adapter.len().unwrap(), 0);

        let vecs = vec![vec![1.0f32, 2.0, 3.0]];
        let texts = vec!["hello"];
        let entries = vec![PairWriteEntry {
            anchor_vec: &vecs[0],
            anchor_text: texts[0],
            candidate_vec: &vecs[0],
            candidate_text: texts[0],
            label: &PairLabel::Positive,
        }];
        adapter
            .write_pairs(0, &PairWriteArgs { entries: &entries })
            .unwrap();

        assert_eq!(adapter.len().unwrap(), 1);
    }

    #[test]
    fn init_split_states_creates_dirs_and_stores() {
        let dir = TempDir::new().unwrap();
        let descs = vec![
            (SplitLabel::Train, "train", 100u64, 0.8f32),
            (SplitLabel::Validation, "val", 50, 0.1),
        ];
        let states =
            init_split_states_with_batch(dir.path(), &descs, 32, SrdMode::Pair, 384, 400).unwrap();

        assert_eq!(states.len(), 2);
        assert_eq!(states[0].label, SplitLabel::Train);
        assert_eq!(states[1].label, SplitLabel::Validation);
        assert_eq!(states[0].total_written, 0);
        assert!(dir.path().join("train").exists());
        assert!(dir.path().join("val").exists());
    }

    #[test]
    fn init_split_states_resumes_existing_store() {
        let dir = TempDir::new().unwrap();
        let descs = vec![(SplitLabel::Train, "train", 0u64, 1.0f32)];

        {
            let states =
                init_split_states_with_batch(dir.path(), &descs, 8, SrdMode::Pair, 4, 400).unwrap();
            let vecs = vec![vec![1.0f32, 2.0, 3.0, 4.0]; 5];
            let texts: Vec<&str> = vec!["a", "b", "c", "d", "e"];
            let entries: Vec<PairWriteEntry> = vecs
                .iter()
                .zip(texts.iter())
                .map(|(v, t)| PairWriteEntry {
                    anchor_vec: v,
                    anchor_text: t,
                    candidate_vec: v,
                    candidate_text: t,
                    label: &PairLabel::Positive,
                })
                .collect();
            states[0]
                .store
                .write_pairs(0, &PairWriteArgs { entries: &entries })
                .unwrap();
        }

        // Second call: should resume with correct total_written.
        let states =
            init_split_states_with_batch(dir.path(), &descs, 8, SrdMode::Pair, 4, 400).unwrap();
        assert_eq!(states[0].total_written, 5);
        assert_eq!(states[0].step_num, 0); // 5 / 8 = 0
    }

    #[test]
    fn srd_store_adapter_write_triplets_and_len() {
        let dir = TempDir::new().unwrap();
        let ds = DataStore::open(&dir.path().join("triplets.srd")).unwrap();
        let adapter = SrdStoreAdapter(ds);

        assert_eq!(adapter.len().unwrap(), 0);

        let anchor_vecs = vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let anchor_texts = vec!["a1", "a2"];
        let pos_vecs = vec![vec![7.0f32, 8.0, 9.0], vec![10.0, 11.0, 12.0]];
        let pos_texts = vec!["p1", "p2"];
        let neg_vecs = vec![vec![13.0f32, 14.0, 15.0], vec![16.0, 17.0, 18.0]];
        let neg_texts = vec!["n1", "n2"];

        let entries: Vec<TripletWriteEntry> = (0..2)
            .map(|i| TripletWriteEntry {
                anchor_vec: &anchor_vecs[i],
                anchor_text: anchor_texts[i],
                pos_vec: &pos_vecs[i],
                pos_text: pos_texts[i],
                neg_vec: &neg_vecs[i],
                neg_text: neg_texts[i],
            })
            .collect();

        adapter
            .write_triplets(0, &TripletWriteArgs { entries: &entries })
            .unwrap();

        assert_eq!(adapter.len().unwrap(), 2);
    }

    #[test]
    fn srd_store_adapter_deref_accesses_inner() {
        let dir = TempDir::new().unwrap();
        let ds = DataStore::open(&dir.path().join("deref.srd")).unwrap();
        let adapter = SrdStoreAdapter(ds);

        let inner_len = DataStoreReader::len(&*adapter).unwrap();
        assert_eq!(inner_len, 0);
    }

    #[test]
    fn init_split_states_triplet_mode() {
        let dir = TempDir::new().unwrap();
        let descs = vec![(SplitLabel::Train, "train", 100u64, 0.8f32)];

        let states =
            init_split_states_with_batch(dir.path(), &descs, 32, SrdMode::Triplet, 384, 400)
                .unwrap();

        assert_eq!(states[0].mode, EmbedMode::Triplet);
        assert_eq!(states[0].emb_dim, 384);
    }

    #[test]
    fn init_split_states_batch_num_computed() {
        let dir = TempDir::new().unwrap();
        let descs = vec![(SplitLabel::Train, "train", 0u64, 1.0f32)];

        {
            let states =
                init_split_states_with_batch(dir.path(), &descs, 8, SrdMode::Pair, 4, 4).unwrap();
            let vecs = vec![vec![1.0f32; 4]; 20];
            let texts: Vec<&str> = (0..20).map(|_i| "").collect();
            let text_refs: Vec<&str> = texts.iter().map(|s| &**s).collect();
            let entries: Vec<PairWriteEntry> = vecs
                .iter()
                .zip(text_refs.iter())
                .map(|(v, t)| PairWriteEntry {
                    anchor_vec: v,
                    anchor_text: t,
                    candidate_vec: v,
                    candidate_text: t,
                    label: &PairLabel::Positive,
                })
                .collect();
            states[0]
                .store
                .write_pairs(0, &PairWriteArgs { entries: &entries })
                .unwrap();
        }

        let states =
            init_split_states_with_batch(dir.path(), &descs, 8, SrdMode::Pair, 4, 4).unwrap();
        assert_eq!(states[0].total_written, 20);
        assert_eq!(states[0].step_num, 2);
        assert_eq!(states[0].batch_num, 0);
        assert_eq!(states[0].segment_base, 20);
    }
}
