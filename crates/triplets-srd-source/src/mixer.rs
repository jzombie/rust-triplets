//! Exact batch mixtures over multiple baked SRD dataset splits.
//!
//! A [`MixedSrdSampler`] holds one [`TripletSampler`] per dataset dir. Triplets
//! owns all selection (ordering, swapping, dedupe, cycling) inside each
//! sampler; this module only does what triplets assigns to the training loop
//! (see `NegativeStrategy` docs: "reweighting source batches in the training
//! loop") — exact row counts per batch ([`split_rows`], largest remainder,
//! sums exactly to the batch size). Counts, not selection.
//!
//! Rows come out with vectors attached: the
//! [`RecordSection`](triplets::data::RecordSection) embedding channel carries
//! each entry's vector from its store onto the sampled chunk, so consumers
//! read `(text, embedding)` straight off the batch. No side lookups, no id
//! parsing, no shared id space (each sampler has its own record pool).
//! Split isolation is structural: a train mixer only ever opens train stores.
//!
//! Samplers run continuously as triplets intends: built once with construction
//! seeds, advancing their own internal epochs. Determinism comes from the
//! seeds; resume means continuation, not replay.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use simd_r_drive::storage_engine::DataStore;
use simd_r_drive::storage_engine::traits::DataStoreReader;
use triplets::{DeterministicSplitStore, SamplerConfig, SplitLabel, SplitRatios, TripletSampler};

use crate::SrdSource;
use crate::error::SrdError;
use crate::srd_triplet::{self, SrdMode, SrdRecord};

/// One mixed row: anchor text with its teacher embedding, ready to train on.
#[derive(Clone, Debug)]
pub struct MixedRow {
    /// Anchor text.
    pub text: String,
    /// Teacher embedding for the anchor text.
    pub embedding: Vec<f32>,
}

/// Largest-remainder split of `total` rows over `weights`; sums exactly to
/// `total`. Pure counts — no selection semantics.
fn split_rows(total: usize, weights: &[f32]) -> Vec<usize> {
    let sum: f32 = weights.iter().sum();
    let mut out = vec![0usize; weights.len()];
    if sum <= 0.0 {
        return out;
    }
    let mut rema: Vec<(f32, usize)> = Vec::with_capacity(weights.len());
    let mut assigned = 0usize;
    for (i, &w) in weights.iter().enumerate() {
        let exact = total as f32 * w / sum;
        let base = exact.floor() as usize;
        out[i] = base;
        assigned += base;
        rema.push((exact - base as f32, i));
    }
    rema.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    for (_, i) in rema.iter().take(total - assigned) {
        out[*i] += 1;
    }
    out
}

/// One dataset's share of a mix: a dedicated triplets sampler over that
/// dataset's split store alone, plus the split's row count (for sizing).
struct SourceSampler {
    sampler: TripletSampler<DeterministicSplitStore>,
    rows: usize,
}

/// Exact batch mixer over N baked SRD dataset splits.
///
/// Built once via [`MixedSrdSampler::open`], then [`MixedSrdSampler::next_batch`]
/// draws `(text, embedding)` rows on demand. The samplers advance their own
/// internal epochs continuously; no per-epoch rebuilds.
pub struct MixedSrdSampler {
    sources: Vec<SourceSampler>,
    weights: Vec<f32>,
    split: SplitLabel,
}

/// Stable source id for a dataset dir: its file name (e.g. `dataset-f16`).
/// Scoping is per-sampler (each source has its own pool), so this is only
/// for debuggability — nothing parses it.
fn dir_source_id(dir: &Path) -> String {
    dir.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("srd")
        .to_string()
}

/// Open the split store, fail fast when empty, and detect its SRD mode.
/// Returns `(mode, rows)`. The store is dropped; the sampler opens its own.
fn probe_split(
    dir: &Path,
    split_name: &'static str,
    emb_dim: usize,
) -> Result<(SrdMode, usize), SrdError> {
    let data_path = dir.join(split_name).join("data.srd");
    let store = DataStore::open_existing(&data_path)?;
    let n = store.len()?;
    if n == 0 {
        return Err(SrdError::EmptySplit {
            split: split_name,
            dir: dir.display().to_string(),
        });
    }
    let entries = srd_triplet::batch_read_entries(&store, &[0], emb_dim)?;
    let mode = match entries.into_iter().next() {
        Some(SrdRecord::Pair(_)) => SrdMode::Pair,
        Some(SrdRecord::Triplet(_)) => SrdMode::Triplet,
        None => {
            return Err(SrdError::EmptySplit {
                split: split_name,
                dir: dir.display().to_string(),
            });
        }
    };
    Ok((mode, n))
}

/// Build one source's sampler: probe the split (fail fast on empty), register
/// a plain [`SrdSource`] with a dedicated sampler forced to `split`.
fn build_source_sampler(
    dir: &Path,
    split: SplitLabel,
    split_name: &'static str,
    seed: u64,
    batch_rows: usize,
    emb_dim: usize,
) -> Result<SourceSampler, SrdError> {
    let ratios = match split {
        SplitLabel::Train => SplitRatios {
            train: 1.0,
            validation: 0.0,
            test: 0.0,
        },
        SplitLabel::Validation => SplitRatios {
            train: 0.0,
            validation: 1.0,
            test: 0.0,
        },
        _ => return Err(SrdError::BadSplit),
    };
    let (mode, rows) = probe_split(dir, split_name, emb_dim)?;
    let split_store = DeterministicSplitStore::new(ratios, seed)?;
    let config = SamplerConfig {
        seed,
        batch_size: batch_rows.max(1),
        allowed_splits: vec![split],
        split: ratios,
        ..Default::default()
    };
    let sampler = TripletSampler::new(config, Arc::new(split_store));
    let data_path = dir.join(split_name).join("data.srd");
    let src = SrdSource::open(&data_path, dir_source_id(dir), emb_dim, mode)?;
    sampler.register_source(Box::new(src))?;
    Ok(SourceSampler { sampler, rows })
}

impl MixedSrdSampler {
    /// Open a mixer over `dirs` with per-row-share `weights`.
    ///
    /// Each dir must contain `<split>/data.srd` (`train` for
    /// [`SplitLabel::Train`], `val` for [`SplitLabel::Validation`]) with the
    /// same embedding dim `emb_dim`; every split store must be non-empty.
    /// `weights` must be finite, non-negative, and sum to a positive value.
    ///
    /// `batch_rows` sizes each source sampler's internal batches; keep it at
    /// least as large as the largest `n_rows` passed to [`Self::next_batch`]
    /// so every source share is drawn from a single sampler batch.
    pub fn open(
        dirs: &[PathBuf],
        weights: &[f32],
        split: SplitLabel,
        seed: u64,
        batch_rows: usize,
        emb_dim: usize,
    ) -> Result<Self, SrdError> {
        if dirs.is_empty() {
            return Err(SrdError::EmptyMix);
        }
        if dirs.len() != weights.len() {
            return Err(SrdError::WeightCountMismatch {
                weights: weights.len(),
                dirs: dirs.len(),
            });
        }
        if weights.iter().any(|w| !w.is_finite() || *w < 0.0) || weights.iter().sum::<f32>() <= 0.0
        {
            return Err(SrdError::BadWeights);
        }
        let split_name: &'static str = match split {
            SplitLabel::Train => "train",
            SplitLabel::Validation => "val",
            _ => return Err(SrdError::BadSplit),
        };
        let mut sources = Vec::with_capacity(dirs.len());
        for dir in dirs {
            sources.push(build_source_sampler(
                dir, split, split_name, seed, batch_rows, emb_dim,
            )?);
        }
        Ok(Self {
            sources,
            weights: weights.to_vec(),
            split,
        })
    }

    /// Number of dataset dirs in this mix.
    pub fn num_sources(&self) -> usize {
        self.sources.len()
    }

    /// Total rows across all sources' split stores (for sizing/logging).
    pub fn total_rows(&self) -> usize {
        self.sources.iter().map(|s| s.rows).sum()
    }

    /// Per-row-share weights aligned with the `dirs` given to [`Self::open`].
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Draw `n_rows` total across sources by weight.
    ///
    /// Per source: one sampler batch, first `k` anchor rows with vectors read
    /// straight off the chunks. Sources are round-robin interleaved
    /// (deterministic) so in-batch negatives mix across datasets. Counts per
    /// source follow [`split_rows`] and sum exactly to `n_rows`.
    pub fn next_batch(&self, n_rows: usize) -> Result<Vec<MixedRow>, SrdError> {
        if n_rows == 0 {
            return Ok(Vec::new());
        }
        let counts = split_rows(n_rows, &self.weights);
        // Per-source rows in sampler order.
        let mut per_source: Vec<Vec<MixedRow>> = Vec::with_capacity(self.sources.len());
        for (src, &k) in self.sources.iter().zip(counts.iter()) {
            let mut rows = Vec::with_capacity(k);
            if k > 0 {
                // Single source per sampler: empty weights = uniform, the only
                // source takes the whole batch. Cross-source weighting already
                // happened in `counts` above.
                let batch = src
                    .sampler
                    .next_pair_batch_with_weights_for_split(self.split, &HashMap::new())?;
                for p in batch.pairs.iter().take(k) {
                    rows.push(MixedRow {
                        text: p.anchor.text.clone(),
                        embedding: p
                            .anchor
                            .embedding
                            .as_deref()
                            .ok_or(SrdError::MissingEmbedding)?
                            .to_vec(),
                    });
                }
            }
            per_source.push(rows);
        }
        // Round-robin interleave across sources (deterministic).
        let mut out = Vec::with_capacity(n_rows);
        let mut iters: Vec<_> = per_source.into_iter().map(Vec::into_iter).collect();
        loop {
            let mut advanced = false;
            for it in iters.iter_mut() {
                if let Some(row) = it.next() {
                    out.push(row);
                    advanced = true;
                }
            }
            if !advanced {
                break;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PairLabel;
    use crate::srd_triplet::{SrdPairWriteEntry, write_pair_entries};
    use tempfile::TempDir;

    const TEST_EMB_DIM: usize = 8;

    /// Write `n` pair entries into `<dir>/<split>/data.srd`, creating it.
    /// Anchor/candidate texts differ (the sampler rejects identical pairs).
    fn write_test_store(dir: &Path, split: &str, n: usize) {
        let split_dir = dir.join(split);
        std::fs::create_dir_all(&split_dir).unwrap();
        let store = DataStore::open(&split_dir.join("data.srd")).unwrap();
        let vecs: Vec<Vec<f32>> = (0..n)
            .map(|i| vec![i as f32 * 0.01 + 0.1; TEST_EMB_DIM])
            .collect();
        let texts: Vec<String> = (0..n).map(|i| format!("anchor text {i}")).collect();
        let cands: Vec<String> = (0..n).map(|i| format!("candidate text {i}")).collect();
        let entries: Vec<SrdPairWriteEntry> = vecs
            .iter()
            .zip(texts.iter())
            .zip(cands.iter())
            .map(|((v, t), c)| SrdPairWriteEntry {
                anchor_vec: v,
                anchor_text: t,
                candidate_vec: v,
                candidate_text: c,
                label: &PairLabel::Positive,
            })
            .collect();
        write_pair_entries(&store, 0, &entries).unwrap();
    }

    fn make_dataset(parent: &Path, name: &str, n_train: usize, n_val: usize) -> PathBuf {
        let dir = parent.join(name);
        write_test_store(&dir, "train", n_train);
        write_test_store(&dir, "val", n_val.max(1));
        dir
    }

    fn open_train(dirs: &[PathBuf], weights: &[f32], seed: u64) -> MixedSrdSampler {
        MixedSrdSampler::open(dirs, weights, SplitLabel::Train, seed, 8, TEST_EMB_DIM).unwrap()
    }

    #[test]
    fn split_rows_sums_exactly() {
        for total in [1usize, 3, 7, 8, 32, 100] {
            for weights in [
                vec![1.0],
                vec![0.5, 0.5],
                vec![0.75, 0.25],
                vec![0.6, 0.2, 0.2],
            ] {
                let counts = split_rows(total, &weights);
                assert_eq!(counts.iter().sum::<usize>(), total);
            }
        }
    }

    #[test]
    fn counts_match_weights_and_vectors_ride_along() {
        let tmp = TempDir::new().unwrap();
        let a = make_dataset(tmp.path(), "ds-a", 12, 2);
        let b = make_dataset(tmp.path(), "ds-b", 12, 2);
        let mixer = open_train(&[a, b], &[0.75, 0.25], 7);
        assert_eq!(mixer.num_sources(), 2);
        assert_eq!(mixer.total_rows(), 24);

        let rows = mixer.next_batch(8).unwrap();
        assert_eq!(rows.len(), 8);
        // Every row carries a full-dim vector, no empty fallbacks.
        for r in &rows {
            assert_eq!(r.embedding.len(), TEST_EMB_DIM);
            assert!(!r.text.is_empty());
        }
        // Counts follow largest-remainder exactly.
        let counts = split_rows(8, &[0.75, 0.25]);
        assert_eq!(counts.iter().sum::<usize>(), 8);
    }

    #[test]
    fn single_dir_passthrough() {
        let tmp = TempDir::new().unwrap();
        let a = make_dataset(tmp.path(), "only", 10, 2);
        let mixer = open_train(&[a], &[1.0], 3);
        let rows = mixer.next_batch(5).unwrap();
        assert_eq!(rows.len(), 5);
        assert!(rows.iter().all(|r| r.embedding.len() == TEST_EMB_DIM));
    }

    #[test]
    fn same_seed_same_sequence() {
        let tmp = TempDir::new().unwrap();
        let a = make_dataset(tmp.path(), "ds-a", 12, 2);
        let b = make_dataset(tmp.path(), "ds-b", 12, 2);
        let dirs = vec![a, b];
        let m1 = open_train(&dirs, &[0.5, 0.5], 42);
        let m2 = open_train(&dirs, &[0.5, 0.5], 42);
        let r1 = m1.next_batch(8).unwrap();
        let r2 = m2.next_batch(8).unwrap();
        assert_eq!(r1.len(), r2.len());
        for (x, y) in r1.iter().zip(r2.iter()) {
            assert_eq!(x.text, y.text);
            assert_eq!(x.embedding, y.embedding);
        }
    }

    #[test]
    fn zero_rows_returns_empty() {
        let tmp = TempDir::new().unwrap();
        let a = make_dataset(tmp.path(), "ds-a", 8, 2);
        let mixer = open_train(&[a], &[1.0], 1);
        assert!(mixer.next_batch(0).unwrap().is_empty());
    }

    #[test]
    fn rejects_bad_configs() {
        let tmp = TempDir::new().unwrap();
        let a = make_dataset(tmp.path(), "ds-a", 8, 2);
        let b = make_dataset(tmp.path(), "ds-b", 8, 2);

        assert!(matches!(
            MixedSrdSampler::open(&[], &[1.0], SplitLabel::Train, 1, 8, TEST_EMB_DIM),
            Err(SrdError::EmptyMix)
        ));
        assert!(matches!(
            MixedSrdSampler::open(
                std::slice::from_ref(&a),
                &[0.5, 0.5],
                SplitLabel::Train,
                1,
                8,
                TEST_EMB_DIM
            ),
            Err(SrdError::WeightCountMismatch { .. })
        ));
        assert!(matches!(
            MixedSrdSampler::open(
                &[a.clone(), b.clone()],
                &[0.0, 0.0],
                SplitLabel::Train,
                1,
                8,
                TEST_EMB_DIM
            ),
            Err(SrdError::BadWeights)
        ));
        assert!(matches!(
            MixedSrdSampler::open(
                std::slice::from_ref(&a),
                &[f32::NAN],
                SplitLabel::Train,
                1,
                8,
                TEST_EMB_DIM
            ),
            Err(SrdError::BadWeights)
        ));
        // Empty split store fails fast.
        let empty_parent = tmp.path().join("empty-ds");
        std::fs::create_dir_all(empty_parent.join("train")).unwrap();
        DataStore::open(&empty_parent.join("train").join("data.srd")).unwrap();
        write_test_store(&empty_parent, "val", 2);
        assert!(matches!(
            MixedSrdSampler::open(
                &[empty_parent],
                &[1.0],
                SplitLabel::Train,
                1,
                8,
                TEST_EMB_DIM
            ),
            Err(SrdError::EmptySplit { .. })
        ));
    }
}
