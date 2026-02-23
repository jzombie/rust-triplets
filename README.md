# triplets

[![made-with-rust][rust-logo]][rust-src-page] [![crates.io][crates-badge]][crates-page] [![MIT licensed][mit-license-badge]][mit-license-page] [![Apache 2.0 licensed][apache-2.0-license-badge]][apache-2.0-license-page] [![Coverage][coveralls-badge]][coveralls-page]

**WORK IN PROGRESS.**

Composable data sampling primitives for deterministic multi-source ML/AI training-data orchestration.

`triplets` is a reusable core for ML/AI training-data orchestration. It provides sampler primitives, split/state persistence, chunking and weighting mechanics, and source abstractions (`DataSource`, `DataRecord`) without tying behavior to proprietary corpora.

CI is configured to run tests/linting on macOS, Linux, and Windows.

## High-level features

- **Automatic deterministic splits** (train/validation/test) from record IDs + seed.
- **Runtime batch sampling** via `next_triplet_batch`, `next_pair_batch`, and `next_text_batch`.
- **Recipe-driven sample construction** for triplet/pair/text generation (anchor/positive/negative selectors).
- **Weight-aware sampling controls** across source weights, recipe weights, and chunk trust/quality weighting.
- **Resume support** via `persist_state()` and split-store persistence.
- **Source-agnostic backends** (`DataSource` or `IndexableSource` + `IndexableAdapter`).
- **Supply-chain style orchestration (core layer):** multi-source intake (`refresh`) with per-call parallel ingest, optional per-source weighting, staged buffering, deterministic split routing, and batch assembly into train-ready outputs.
- **Bounded ingestion** windows instead of loading full corpora into memory.
- **Per-call source threading**: during refresh, each source is fetched on its own short-lived thread, then merged deterministically for batch assembly.
- **Streaming-friendly**: sources can be finite or unbounded.

This crate does **not** perform semantic mining/retrieval scoring by itself; instead, it gives you deterministic, metadata-driven sampling primitives you can feed into your downstream mining/retrieval stack.

### Metadata-driven use (how to use it)

Use `triplets` to build deterministic training batches that carry metadata context:

- Put structural tags in `DataRecord.taxonomy` (source/date/category/etc.) for filtering and analysis.
- Use recipes/selectors to choose which sections become anchor/positive/negative text.
- Attach optional KVP metadata prefixes (below) so sampled text can include lightweight context headers.
- Keep split assignment deterministic while changing recipe or weighting behavior at runtime.

This gives you metadata-aware sampling orchestration, while semantic retrieval/mining logic stays in your downstream pipeline.

### KVP data decorator (how it works)

- Each `DataRecord` can carry an optional `meta_prefix` sampler (`KvpPrefixSampler`).
- At sample time, the sampler can prepend a header line to chunk text, formatted like: `meta: key=value | key2=value2`.
- `KvpField` supports multiple value renderings per key and optional per-field presence probability.
- `KvpPrefixSampler` supports variant selection and overall dropout (emit prefix sometimes, or always).
- This is designed to give the model useful context signals (date/source/category/etc.) without making a single rigid header pattern easy to memorize.
- Multi-render values, per-field presence control, field-order variation, and prefix dropout reduce shortcut learning and encourage reliance on the underlying content.
- KVP prefixes decorate sampled text; they do not change deterministic split assignment.

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

# let split = SplitRatios { train: 0.8, validation: 0.1, test: 0.1 };
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

Example (different source mix across consecutive batches):

```rust,no_run
use std::collections::HashMap;
use std::sync::Arc;
use triplets::{
  DeterministicSplitStore, PairSampler, Sampler, SamplerConfig, SplitLabel, SplitRatios,
};

# let split = SplitRatios { train: 0.8, validation: 0.1, test: 0.1 };
# let store = Arc::new(DeterministicSplitStore::new(split, 123).unwrap());
# let config = SamplerConfig::default();
# let sampler = Arc::new(PairSampler::new(config, store));

let mut weights_a = HashMap::new();
weights_a.insert("source_a".to_string(), 1.0);
weights_a.insert("source_b".to_string(), 0.2);

let mut weights_b = HashMap::new();
weights_b.insert("source_a".to_string(), 0.2);
weights_b.insert("source_b".to_string(), 1.0);

let batch_a = sampler
  .next_triplet_batch_with_weights(SplitLabel::Train, &weights_a)
  .unwrap();
let batch_b = sampler
  .next_triplet_batch_with_weights(SplitLabel::Train, &weights_b)
  .unwrap();

let _ = (batch_a, batch_b);
```

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

If your records are time-ordered (oldest → newest), use these APIs:

- `IndexableSource` (you provide `len_hint()` + `record_at(idx)`).
- `IndexableAdapter` (easiest: turns your `IndexableSource` into a `DataSource`).
- `IndexablePager` (use directly only if you are writing a custom `refresh(...)`).

That is the built-in path for shuffled paging + cursor resume.

Helper-based path (uses the APIs above):

```rust,ignore
use triplets::source::{IndexableAdapter, IndexableSource};
use triplets::{data::DataRecord, SamplerError};

struct MyIndexableSource {
  // Could be DB/API client, manifest reader, etc.
  // No in-memory ID list required.
  total_records: usize,
}

impl MyIndexableSource {
  fn load_record(&self, _idx: usize) -> Result<Option<DataRecord>, SamplerError> {
    // Fetch by numeric position from your backend.
    // `None` means "no record at this index".
    todo!("load one record by index")
  }
}

impl IndexableSource for MyIndexableSource {
  fn id(&self) -> &str { "my_source" }
  fn len_hint(&self) -> Option<usize> { Some(self.total_records) }
  fn record_at(&self, idx: usize) -> Result<Option<DataRecord>, SamplerError> {
    self.load_record(idx)
  }
}

// register as a normal DataSource:
// sampler.register_source(Box::new(IndexableAdapter::new(MyIndexableSource { total_records }))); 
```

Manual path (does NOT use `IndexableSource`/`IndexableAdapter` directly):

```rust
use chrono::Utc;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use triplets::data::DataRecord;
use triplets::source::{SourceCursor, SourceSnapshot};
use triplets::SamplerError;

struct MySource {
  // Canonical record IDs for this source.
  // We keep IDs separate from record payloads so refresh can page deterministically.
  ids: Vec<String>,
}

impl MySource {
  fn load_record(&self, _id: &str) -> Result<DataRecord, SamplerError> {
    // Put your real fetch logic here (database call, API request, file read, etc.).
    // The sampler expects each loaded item to be returned as a DataRecord.
    todo!("load record from storage")
  }

  fn stable_hash(id: &str) -> u64 {
    // Convert each ID to a repeatable number so ordering is the same every run.
    // This avoids "newest-first" bias when IDs are naturally time-ordered.
    let mut hasher = DefaultHasher::new();
    id.hash(&mut hasher);
    hasher.finish()
  }

  fn refresh(
    &self,
    cursor: Option<&SourceCursor>,
    limit: Option<usize>,
  ) -> Result<SourceSnapshot, SamplerError> {
    // Make a sorted copy of IDs so this call runs in a repeatable order.
    // Note: this copy holds all IDs in memory for this refresh call.
    let mut ids = self.ids.clone();
    ids.sort_by_key(|id| Self::stable_hash(id));

    // How many records exist right now.
    let total = ids.len();

    // `revision` means "where to resume next time".
    // No cursor yet means this is the first run, so start at index 0.
    let mut start = cursor.map(|c| c.revision as usize).unwrap_or(0);

    // If data size changed and start is now invalid, safely reset to the beginning.
    if total > 0 && start >= total {
      start = 0;
    }

    // Hard cap for this call.
    // - If `limit` is Some(n), we load at most `n` records this call.
    // - If `limit` is None, we allow one full pass (`total` records).
    let max = limit.unwrap_or(total);
    let mut records = Vec::new();

    // Load records one-by-one, starting at `start`, and wrap at the end.
    // We stop as soon as `records.len() == max`.
    // So this does NOT always load everything; it only loads up to `max`.
    for idx in 0..total {
      if records.len() >= max {
        break;
      }
      let pos = (start + idx) % total;
      records.push(self.load_record(&ids[pos])?);
    }

    // Save where the next call should continue.
    let next_start = (start + records.len()) % total.max(1);
    Ok(SourceSnapshot {
      records,
      cursor: SourceCursor {
        // Record when this refresh happened.
        last_seen: Utc::now(),
        // Store resume position for the next refresh call.
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
[crates-badge]: https://img.shields.io/crates/v/triplets.svg

[mit-license-page]: ./LICENSE-MIT
[mit-license-badge]: https://img.shields.io/badge/license-MIT-blue.svg

[apache-2.0-license-page]: ./LICENSE-APACHE
[apache-2.0-license-badge]: https://img.shields.io/badge/license-Apache%202.0-blue.svg

[coveralls-page]: https://coveralls.io/github/jzombie/rust-triplets?branch=main
[coveralls-badge]: https://img.shields.io/coveralls/github/jzombie/rust-triplets
