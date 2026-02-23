# triplets

[![made-with-rust][rust-logo]][rust-src-page] [![crates.io][crates-badge]][crates-page] [![MIT licensed][mit-license-badge]][mit-license-page] [![Apache 2.0 licensed][apache-2.0-license-badge]][apache-2.0-license-page] [![Coverage][coveralls-badge]][coveralls-page]

**WORK IN PROGRESS**

Composable Rust crate for deterministic multi-source sampling and split persistence for ML/AI training data.

`triplets` is a reusable core for ML/AI training-data orchestration. It provides sampler primitives, split/state persistence, chunking and weighting mechanics, and source abstractions (`DataSource`, `DataRecord`) without tying behavior to proprietary corpora.

## At a glance

`triplets` is for building reproducible ML/AI training batches from multiple data sources.

Compared with a static prebuilt dataset, it lets you sample at runtime while preserving deterministic behavior.

Threading model: source refresh work is parallelized per sampling call, while batch assembly remains serialized and deterministic.

## Core capabilities

- **Source-agnostic sampling:** implement `DataSource` for filesystem, APIs, DBs, streams, etc.
- **Runtime example generation:** produce triplet/pair/text batches from recipe selectors.
- **Deterministic split assignment:** stable train/validation/test assignment from record IDs + seed.
- **Resume support:** persist sampler/split state and continue after restart.
- **Bounded ingestion:** refresh in controlled windows instead of loading full corpora into memory.
- **Per-source progression:** each source has its own cursor; sources do not need to advance in lockstep.
- **Per-call concurrency:** source refreshes run in parallel within a sampling call, then merge before batch assembly.

## Not included

- This crate does **not** do semantic mining/retrieval scoring by itself.
- This crate does **not** guarantee semantic hardness beyond your recipes and source metadata design.
- Sources can be finite or unbounded; infinite streaming is supported but not required.

## Getting started

Add `triplets` to a downstream crate:

```bash
cargo add triplets
```

To run the included examples in this repository (for exploration/contributor workflow):

```bash
cargo run --example multi_source_demo -- --help
```

For contributors (development check):

```bash
cargo test
```

Minimal shape:

1. Implement one or more `DataSource` backends.
2. Create `SamplerConfig` (chunking, recipes, split policy).
3. Open a split store (`DeterministicSplitStore` or `FileSplitStore`).
4. Construct `PairSampler` and register sources.
5. Call one of the batch APIs: `next_triplet_batch(split)`, `next_pair_batch(split)`, or `next_text_batch(split)`.
6. Call `persist_state()` when you want restart-resume behavior.

## Examples

From the `triplets` crate:

```bash
# sample triplet batches
cargo run --example multi_source_demo

# inspect CLI flags
cargo run --example multi_source_demo -- --help

# metadata-only capacity estimation
cargo run --example estimate_capacity -- --help
cargo run --example estimate_capacity
```

Source roots can be overridden with repeatable flags:

```bash
cargo run --example multi_source_demo -- \
  --source-root /path/to/source_1 \
  --source-root /path/to/source_2
```

### Split-store path configuration

The `multi_source_demo` example persists sampler/split state by default to:

- `.sampler_store/split_store.bin`

You can override persistence location with either:

- `--split-store-path <FILE>` for an explicit file path
- `--split-store-dir <DIR>` to keep filename `split_store.bin` in a custom directory

## Usage flow

Short version:

- Call **`sampler.next_*_batch(split)`** to sample batches (ingestion happens automatically).
- Call **`sampler.persist_state()`** when you want restart-resume behavior.
- Optionally call **`sampler.set_epoch(n)`** for explicit epoch control.

Step-by-step:

1. Build config + open the split store.
2. Register sources.
3. Call one of **`sampler.next_triplet_batch(split)`**, **`sampler.next_pair_batch(split)`**, or **`sampler.next_text_batch(split)`**.
4. Call **`sampler.persist_state()`** when you want to save progress (typically at the end of an epoch, or at explicit checkpoint boundaries).
5. Optionally call **`sampler.set_epoch(n)`** for explicit epoch replay/order.

Operational notes:

- File-backed indexing is rebuilt per process/run and stored in an OS temp-backed index store.
- Persisting sampler/split state is explicit and manual.
- One split-store file shares sampler/source cursor + RNG state unless you use separate store files.
- Batch calls are thread-safe but serialized; refresh work within a call can be parallelized per source.
- Source cursors advance independently per source, so one source can continue making progress even if another source is sparse or slower.
- Refresh concurrency is per call: source refreshes run in parallel for that call, then the sampler joins all refresh threads before merging buffers (not an always-on per-source background ingest loop).
- Prefetchers smooth latency by filling bounded queues from the existing batch APIs (`next_triplet_batch`, `next_pair_batch`, `next_text_batch`).
- New data from streaming sources is pulled in on the next batch call.
- `sampler.persist_state()` is manual; skipping it means no resume state after restart.
- `sampler.set_epoch(n)` is an advanced override and is not required for normal resume behavior.
- `IngestionManager::source_refresh_stats()` exposes per-source refresh duration/records/throughput/errors.
- `metrics::source_skew` summarizes per-source sample imbalance for a batch.

Example:

```rust,no_run
use std::sync::Arc;
use triplets::{
  DeterministicSplitStore, PairSampler, Sampler, SamplerConfig, SplitLabel, SplitRatios,
};

# let split = SplitRatios { train: 1.0, validation: 0.0, test: 0.0 };
# let store = Arc::new(DeterministicSplitStore::new(split, 123).unwrap());
# let config = SamplerConfig::default();
let sampler = Arc::new(PairSampler::new(config, store));
// register sources...

let prefetcher = Arc::clone(&sampler).prefetch_triplet_batches(SplitLabel::Train, 4);
let batch = prefetcher.next().unwrap();
let _ = batch;
```

- For per-call source weighting, use `next_triplet_batch_with_weights(...)`, `next_pair_batch_with_weights(...)`, or `next_text_batch_with_weights(...)`.
- Missing source ids default to `1.0`; `0.0` disables a source for that call.
- **Production readiness note**: if `len_hint` drifts in streaming/append-only sources, epoch order/coverage can repeat/skip records within an epoch, even though split assignment remains deterministic.

## Sampling behavior (current)

This reflects the built-in file-corpus helpers (`FileCorpusIndex`) used by filesystem-backed sources.

- **Ingestion**: `next_triplet_batch(split)`, `next_pair_batch(split)`, and `next_text_batch(split)` trigger refresh; per-source buffers refill when empty (or on force refresh).
- **Memory bound**: refresh/cache limits are bounded by `ingestion_max_records` with a floor at `batch_size`.
- **File indexing**: deterministic path ordering + deterministic index permutation for paging.
- **Source ordering**: round-robin by source, deterministic within-source ordering by seed/epoch.
- **Splits**: labels are deterministic from `record_id + seed + ratios`; split APIs enforce `allowed_splits`.
- **Coverage caveat**: if `len_hint` drifts mid-epoch in streaming backends, strict single-pass coverage is not guaranteed.
- **Weights**: recipe/source/chunk weights affect scaling, not deterministic ordering.
- **Scale note**: full scan/sort/index rebuild cost grows roughly linearly with file count and path bytes.
- **Order note**: index batching preserves permutation order; chunked index reads do not remove deterministic shuffling.
- **Manual epoch control**: `sampler.set_epoch(n)` resets per-source cursors and reshuffles deterministically for that epoch.
- **Persisted state scope**: epoch tracking is split-aware, but sampler/source cursors + RNG/round-robin state are persisted per store file.
- **Triplet recipe behavior**: per-source recipes are scanned from per-source round-robin hints until a match is found.
- **Pair batches**: derived from triplets and follow the same source/recipe selection behavior.
- **Text recipes**: follow per-source behavior when provided; otherwise config recipes are used.
- **Oversampling**: when sources run dry, cached records may be reused (no global no-repeat guarantee).

### New-source implementation pattern

For any new backend (file/API/DB/stream), centralize backend configuration/state access in one helper reused by both `refresh(...)` and `reported_record_count()`.

Why this matters: capacity estimates and runtime sampling stay aligned only when both methods represent the same logical corpus slice.

File-backed pattern:

```rust,ignore
fn source_index(&self) -> FileCorpusIndex {
  FileCorpusIndex::new(&self.root, &self.id)
    .with_follow_links(true)
    .with_text_files_only(true)
    .with_directory_grouping(true)
}

fn refresh(
  &self,
  cursor: Option<&SourceCursor>,
  limit: Option<usize>,
) -> Result<SourceSnapshot, SamplerError> {
  self.source_index()
    .refresh_indexable(cursor, limit, |path| self.build_record(path))
}

fn reported_record_count(&self) -> Option<u128> {
  self.source_index().indexed_record_count().ok().map(|n| n as u128)
}
```

If a source emits sequential IDs, implement indexable paging (`IndexableSource` + `IndexablePager` or `IndexableAdapter`) to avoid time-ordered ingestion bias.

Example hash-sorted refresh skeleton:

```rust
use chrono::Utc;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use triplets::data::DataRecord;
use triplets::source::{SourceCursor, SourceSnapshot};
use triplets::SamplerError;

struct MySource {
  ids: Vec<String>,
}

impl MySource {
  fn load_record(&self, _id: &str) -> Result<DataRecord, SamplerError> {
    todo!("load record from storage")
  }

  fn stable_hash(id: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    id.hash(&mut hasher);
    hasher.finish()
  }

  fn refresh(
    &self,
    cursor: Option<&SourceCursor>,
    limit: Option<usize>,
  ) -> Result<SourceSnapshot, SamplerError> {
    let mut ids = self.ids.clone();
    ids.sort_by_key(|id| Self::stable_hash(id));
    let total = ids.len();
    let mut start = cursor.map(|c| c.revision as usize).unwrap_or(0);
    if total > 0 && start >= total {
      start = 0;
    }
    let max = limit.unwrap_or(total);
    let mut records = Vec::new();
    for idx in 0..total {
      if records.len() >= max {
        break;
      }
      let pos = (start + idx) % total;
      records.push(self.load_record(&ids[pos])?);
    }
    let next_start = (start + records.len()) % total.max(1);
    Ok(SourceSnapshot {
      records,
      cursor: SourceCursor {
        last_seen: Utc::now(),
        revision: next_start as u64,
      },
    })
  }
}
```

## Capacity estimates

The estimate helpers compute metadata-only approximations from source-reported counts and recipe structure.

- They do not call source refresh.
- They are floor-like approximations for real chunked training.
- Effective triplet estimates use bounded assumptions (positives/negatives per anchor).

## Potential future directions (optional)

These are ideas, not commitments.

- Add more backend adapters in downstream crates (APIs, DBs, manifests, streams)
- Improve strict-coverage options for drifting/streaming corpora
- Add optional split-keyed sampler cursor state in a single store file
- Extend observability hooks for ingestion latency/skew/error diagnostics

## License

`triplets` is primarily distributed under the terms of both the MIT license and the Apache License (Version 2.0).

See [LICENSE-APACHE](./LICENSE-APACHE) and [LICENSE-MIT](./LICENSE-MIT) for details.

[rust-src-page]: https://www.rust-lang.org/
[rust-logo]: https://img.shields.io/badge/Made%20with-Rust-black

[crates-page]: https://crates.io/crates/rust-triplets
[crates-badge]: https://img.shields.io/crates/v/term-wm.svg

[mit-license-page]: ./LICENSE-MIT
[mit-license-badge]: https://img.shields.io/badge/license-MIT-blue.svg

[apache-2.0-license-page]: ./LICENSE-APACHE
[apache-2.0-license-badge]: https://img.shields.io/badge/license-Apache%202.0-blue.svg

[coveralls-page]: https://coveralls.io/github/jzombie/rust-triplets?branch=main
[coveralls-badge]: https://img.shields.io/coveralls/github/jzombie/term-wm
