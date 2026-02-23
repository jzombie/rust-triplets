# triplets

[![made-with-rust][rust-logo]][rust-src-page] [![crates.io][crates-badge]][crates-page] [![MIT licensed][mit-license-badge]][mit-license-page] [![Apache 2.0 licensed][apache-2.0-license-badge]][apache-2.0-license-page] [![Coverage][coveralls-badge]][coveralls-page]

**WORK IN PROGRESS**

Composable Rust crate for deterministic multi-source sampling and split persistence for ML/AI training data.

`triplets` is a reusable core for ML/AI training-data orchestration. It provides sampler primitives, split/state persistence, chunking and weighting mechanics, and source abstractions (`DataSource`, `DataRecord`) without tying behavior to proprietary corpora.

## Why this instead of a static dataset

Compared with a typical static dataset workflow, `triplets` is designed for deterministic runtime orchestration:

- **Online deterministic sampling:** sample from multiple sources at runtime instead of consuming one pre-materialized dump.
- **Stable split assignment + persistence:** keep train/validation/test membership reproducible across restarts and runs.
- **Bounded ingestion windows:** progress through large or changing corpora without loading everything at once.
- **Recipe-time generation:** build triplet/pair/text training examples during sampling rather than only reading pre-generated examples.
- **Per-call source weighting:** adjust source mixture without regenerating a static artifact.
- **Streaming-aware refresh:** incorporate newly available records on subsequent sampling calls.

Concurrency and source progression model:

- Each source has an independent cursor and buffer, so sources do not advance in lockstep.
- Source refreshes run concurrently within a sampling/refresh call.
- Synchronization happens at call boundaries: refresh threads are joined before buffer merge (not an always-on per-source ingest loop).

## Philosophy

You can think of `triplets` as a training-pipeline orchestrator:

- **Composability:** recipe-driven pair/triplet/text generation independent of storage backend.
- **Abstractions:** source backends (filesystem, SQL, APIs, streams) are decoupled from sampling logic.
- **Pipeline management:** deterministic split assignment, bounded ingestion, chunk weighting, and persisted resume state.

## Supply-chain mindset

- **Suppliers:** each `DataSource` is a supplier.
- **Manifests & traceability:** stable record IDs plus deterministic split hashing keep records glued to train/validation/test.
- **Inventory control:** per-source cursors bound memory and support large corpora.
- **Routing plan:** seed + epoch + chunking define deterministic ordering.
- **Packaged outputs:** recipes emit triplets/pairs/text batches without changing source backends.

## Highlights

1. **Data-source agnostic core** – implement `DataSource` for files, SQL, APIs, streams, etc.
2. **Semantic recipes** – define anchor/positive/negative selectors and mismatch strategies.
3. **Deterministic split manager** – reproducible split assignment and optional persisted state.
4. **Quality knobs** – per-record trust scores and chunk-level weighting.
5. **Chunk orchestration** – overlap-aware windows with summary fallbacks.
6. **Thread-safe batching** – serialized batch construction with multi-threaded source refresh.
7. **Prefetchers** – background queueing for triplet/pair/text batch pipelines.
8. **Capacity estimation helpers** – metadata-only split/pair/triplet/text estimates.

## What this does (and does not do)

- **Does**: deterministic paging, split assignment, state persistence, and reproducible batch assembly.
- **Does**: enforce bounded ingestion and explicit resume semantics.
- **Does**: support both finite/index-backed sources and unbounded streaming/append-only sources.
- **Does not**: perform semantic mining, topic modeling, or relevance scoring by itself.
- **Does not**: assume every source is infinite.
- **Does not**: guarantee semantic hardness beyond recipe and source metadata design.

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
5. Call `next_*_batch(split)` APIs.
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
3. Call **`sampler.next_*_batch(split)`**.
4. Call **`sampler.persist_state()`** when you want to save progress.
5. Optionally call **`sampler.set_epoch(n)`** for explicit epoch replay/order.

Operational notes:

- File-backed indexing is rebuilt per process/run and stored in an OS temp-backed index store.
- Persisting sampler/split state is explicit and manual.
- One split-store file shares sampler/source cursor + RNG state unless you use separate store files.
- Batch calls are thread-safe but serialized; refresh work within a call can be parallelized per source.
- Source cursors advance independently per source, so one source can continue making progress even if another source is sparse or slower.
- Refresh concurrency is per call: source refreshes run in parallel for that call, then the sampler joins all refresh threads before merging buffers (not an always-on per-source background ingest loop).
- Prefetchers smooth latency by filling bounded queues from existing `next_*_batch(split)` APIs.
- New data from streaming sources is pulled in on the next `next_*_batch(split)` call.
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

- For per-call source weighting, use `next_*_batch_with_weights(split, &HashMap<SourceId, f32>)`.
- Missing source ids default to `1.0`; `0.0` disables a source for that call.
- **Production readiness note**: if `len_hint` drifts in streaming/append-only sources, epoch order/coverage can repeat/skip records within an epoch, even though split assignment remains deterministic.

## Sampling behavior (current)

This reflects the built-in file-corpus helpers (`FileCorpusIndex`) used by filesystem-backed sources.

- **Ingestion**: `next_*_batch(split)` triggers refresh; per-source buffers refill when empty (or on force refresh).
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

[crates-page]: https://crates.io/crates/triplets
[crates-badge]: https://img.shields.io/crates/v/term-wm.svg

[mit-license-page]: ./LICENSE-MIT
[mit-license-badge]: https://img.shields.io/badge/license-MIT-blue.svg

[apache-2.0-license-page]: ./LICENSE-APACHE
[apache-2.0-license-badge]: https://img.shields.io/badge/license-Apache%202.0-blue.svg

[coveralls-page]: https://coveralls.io/github/jzombie/triplets?branch=main
[coveralls-badge]: https://img.shields.io/coveralls/github/jzombie/term-wm
