# triplets

[![made-with-rust][rust-logo]][rust-src-page] [![crates.io][crates-badge]][crates-page] [![MIT licensed][mit-license-badge]][mit-license-page] [![Apache 2.0 licensed][apache-2.0-license-badge]][apache-2.0-license-page] [![Coverage][coveralls-badge]][coveralls-page]

**WORK IN PROGRESS. THIS API IS BEING PROTOTYPED AND MAY CHANGE WITHOUT NOTICE.**

`triplets` is a reusable core of composable data sampling primitives for deterministic multi-source ML/AI training-data orchestration, with sampler primitives, split/state persistence, chunking and weighting mechanics, and source abstractions (`DataSource`, `DataRecord`) that avoid tying behavior to proprietary corpora.

**Note:** This crate is intended primarily for textual (or textualized) data — records that can be represented as text (for example: documents, QA pairs, logs, or metadata-prefixed chunks) suitable for language-model training, embedding/metric-learning workflows, and related text-model pipelines.

> _CI is configured to run tests/linting on macOS, Linux, and Windows._

## What are triplets?

In metric learning, a triplet is a training example composed of:

- **Anchor**: a reference example.
- **Positive**: another example that should be close to the anchor.
- **Negative**: an example that should be farther from the anchor.

```text
      Anchor
      /    \
 Positive Negative

 Triplet: (Anchor, Positive, Negative)
```

Training on many `(anchor, positive, negative)` groups helps a model learn useful embedding space structure (similar items closer together, dissimilar items farther apart).

In this crate, those triplets are built automatically from one or more data sources using metadata-driven, user-defined recipes/selectors for anchor/positive/negative section choice.

It is designed for multi-source training pipelines where each batch can mix records from several sources, while source contribution is controlled independently (for example, over/under-sampling frequency and trust/quality weighting per source) to rebalance representation and reduce source-driven bias. Because source weights can be set per batch call, they can be wired to training-time loss/metric signals and adjusted dynamically during training.

## High-level features

- **Automatic deterministic splits** (train/validation/test) from record IDs + seed.
- **Sampler-seed-driven source determinism** for built-in deterministic source ordering (file + Hugging Face).
- **Runtime batch sampling** via `next_triplet_batch`, `next_pair_batch`, and `next_text_batch`.
- **Recipe-driven sample construction** for triplet/pair/text generation (anchor/positive/negative selectors).
- **Deterministic long-section chunking**: short text stays as one chunk; long text becomes multiple chunk candidates (sliding windows) sampled over time. Chunks are not emitted as one grouped bundle; each sampled triplet/pair/text item uses one selected chunk at a time. Defaults are `max_window_tokens=4096`, `overlap_tokens=[64,128]`, and `summary_fallback_tokens=512` (all configurable via `SamplerConfig.chunking`).
- **Weight-aware sampling controls** across source weights, recipe weights, and chunk trust/quality weighting.
- **Anti-shortcut metadata-prefix variation** via `KvpPrefixSampler` (variant choice, per-field presence probabilities, field-order shuffle, and prefix dropout) to reduce rigid header-pattern dependence.
- **Per-source batch mixing controls** so multiple sources can contribute to the same batch, with independent source frequency controls (including over/under-sampling).
- **Per-source trust controls** to weight quality/trust independently by source/taxonomy and help mitigate bias from uneven source quality.
- **Per-batch dynamic source reweighting** so source weights can be changed across batches (for example from loss/metric feedback) while training.
- **Resume support** via `persist_state()` and split-store persistence.
- **Source-agnostic backends** (`DataSource` or `IndexableSource` + `IndexableAdapter`).
- **Supply-chain style orchestration (core layer):** multi-source intake (`refresh`) with per-call parallel ingest, optional per-source weighting, staged buffering, deterministic split routing, and batch assembly into train-ready outputs.
- **Bounded ingestion** windows instead of loading full corpora into memory.
- **Per-call source threading**: during refresh, each source is fetched on its own short-lived thread, then merged deterministically for batch assembly.
- **Streaming-friendly**: sources can be finite or unbounded.

This crate does **not** perform semantic mining/retrieval scoring by itself; instead, it gives you deterministic, metadata-driven sampling primitives you can feed into your downstream mining/retrieval stack.

## Using a source for sampling

Create a sampler, register your source, then ask for a batch:

```rust,no_run
use std::sync::Arc;

use chrono::Utc;
use triplets::{
  DataRecord, DeterministicSplitStore, Sampler, SamplerConfig, SplitLabel, SplitRatios,
  TripletSampler,
};
use triplets::source::InMemorySource;

let record = DataRecord {
  id: "r1".into(),
  source: "demo".into(),
  created_at: Utc::now(),
  updated_at: Utc::now(),
  quality: Default::default(),
  taxonomy: Vec::new(),
  sections: Vec::new(),
  meta_prefix: None,
};

let source = InMemorySource::new("demo", vec![record]);

let split = SplitRatios {
  train: 1.0,
  validation: 0.0,
  test: 0.0,
};
// Deterministic split seed; keep stable to preserve split assignments across runs.
let store = Arc::new(DeterministicSplitStore::new(split, 42)?);
let sampler = TripletSampler::new(SamplerConfig::default(), Arc::clone(&store));

sampler.register_source(Box::new(source));

let _batch = sampler.next_triplet_batch(SplitLabel::Train)?;
# Ok::<(), triplets::SamplerError>(())
```

> _`DataRecord` is the core sampling primitive, but this in-memory example is only for illustration and not a scalable or memory-efficient pattern. For real datasets, prefer the built-in integrated sources or an `IndexableSource` implementation._

## Integrated sources

`triplets` ships with two built-in sources; if you use either, deterministic paging is always enabled (`FileSource`, `HuggingFaceRowSource`).

- **File source (`FileSource`)**: local files and folders.
- **Hugging Face source (`HuggingFaceRowSource`)** *(feature: `huggingface`)*: HF dataset rows.

### Hugging Face source lists (recommended)

Define HF sources in a text file and pass it to the demo or your own loader. The `hf://` prefix is a `triplets`-specific shorthand used only in these lists:

```text
hf://org/dataset/config/split anchor=... positive=... context=a,b text=x,y
```

Rules:

- Lines are whitespace-delimited; comments start with `#`.
- `anchor=`, `positive=`, `context=`, and `text=` are the only accepted keys.
- At least one mapping key is required per line.
- `context=` and `text=` accept comma-delimited column lists.
- Rows with missing/blank required fields are skipped.

Example list (see [examples/common/hf_sources.txt](examples/common/hf_sources.txt)):

```text
# role columns
hf://labofsahil/hackernews-vector-search-dataset/default/train anchor=title positive=text
hf://wikimedia/wikipedia/20231101.en/train anchor=title positive=text

# explicit text-column mode
hf://pfox/71k-English-uncleaned-wordlist/default/train text=text
```

Row formats supported by the HF backend:

- `.parquet`
- `.jsonl` / `.ndjson` (one JSON object per line)
- plain text lines (each non-empty line becomes `{ "text": "..." }`)

## Adding new sources

Use one of these two paths:

- **Implement `DataSource`** when your backend has its own paging/cursor model.
- **Implement `IndexableSource`** when you can fetch rows by a stable integer index, then wrap with `IndexableAdapter`.

Minimal `IndexableSource` example:

```rust,no_run
use triplets::{DataRecord, SamplerError};
use triplets::source::{IndexableAdapter, IndexableSource};
use chrono::Utc;

struct MySource {
  id: String,
}

impl IndexableSource for MySource {
  fn id(&self) -> &str {
    &self.id
  }

  fn len_hint(&self) -> Option<usize> {
    Some(0)
  }

  fn record_at(&self, _idx: usize) -> Result<Option<DataRecord>, SamplerError> {
    Ok(Some(DataRecord {
      id: format!("{}::0", self.id),
      source: self.id.clone(),
      created_at: Utc::now(),
      updated_at: Utc::now(),
      quality: Default::default(),
      taxonomy: Vec::new(),
      sections: Vec::new(),
      meta_prefix: None,
    }))
  }
}

let source = IndexableAdapter::new(MySource { id: "my_source".into() });
# let _ = source;
```

Then register the source with your sampler and call `next_triplet_batch`, `next_pair_batch`, or `next_text_batch`.

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

- Call **`sampler.next_triplet_batch(split)`**, **`sampler.next_pair_batch(split)`**, or **`sampler.next_text_batch(split)`** to sample batches (ingestion happens automatically).
- Call **`sampler.persist_state()`** when you want restart-resume behavior.
- Optionally call **`sampler.set_epoch(n)`** for explicit epoch control.

Step-by-step:

1. Build config + open the split store.
2. Register sources.
3. Call one of **`sampler.next_triplet_batch(split)`**, **`sampler.next_pair_batch(split)`**, or **`sampler.next_text_batch(split)`**.
4. Call **`sampler.persist_state()`** when you want to write persisted sampler/split state (typically at the end of an epoch or at explicit checkpoint boundaries). **Do not call this every step.** Very frequent writes can create high I/O overhead and, at very large write counts (for example, tens of millions), can also adversely affect split-store initialization time.
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
  DeterministicSplitStore, TripletSampler, Sampler, SamplerConfig, SplitLabel, SplitRatios,
};

# let split = SplitRatios { train: 0.8, validation: 0.1, test: 0.1 };
# let store = Arc::new(DeterministicSplitStore::new(split, 123).unwrap());
# let config = SamplerConfig::default();
let sampler = Arc::new(TripletSampler::new(config, store));
// register sources...

let prefetcher = Arc::clone(&sampler).prefetch_triplet_batches(SplitLabel::Train, 4);
let batch = prefetcher.next().unwrap();
let _ = batch;
```

### Expected batch output (assertion-style)

The most useful checks are shape/invariants, not exact record order. `next_triplet_batch`, `next_pair_batch`, and `next_text_batch` return exactly `batch_size` samples.

A minimal assertion pattern:

```rust,no_run
use std::borrow::Cow;
use std::sync::Arc;

use chrono::Utc;
use triplets::data::RecordSection;
use triplets::source::InMemorySource;
use triplets::{
  DataRecord, DeterministicSplitStore, NegativeStrategy, PairLabel, Sampler, SamplerConfig,
  SectionRole, Selector, SplitLabel, SplitRatios, TripletRecipe, TripletSampler,
};

fn record(id: &str) -> DataRecord {
  DataRecord {
    id: id.into(),
    source: "demo".into(),
    created_at: Utc::now(),
    updated_at: Utc::now(),
    quality: Default::default(),
    taxonomy: Vec::new(),
    sections: vec![
      RecordSection {
        role: SectionRole::Anchor,
        heading: Some("title".into()),
        text: format!("anchor {id}"),
        sentences: vec![format!("anchor {id}")],
      },
      RecordSection {
        role: SectionRole::Context,
        heading: Some("body".into()),
        text: format!("context {id}"),
        sentences: vec![format!("context {id}")],
      },
    ],
    meta_prefix: None,
  }
}

let source = InMemorySource::new("demo", vec![record("r1"), record("r2"), record("r3")]);

let split = SplitRatios {
  train: 1.0,
  validation: 0.0,
  test: 0.0,
};
let store = Arc::new(DeterministicSplitStore::new(split, 42)?);

let mut config = SamplerConfig::default();
config.batch_size = 2;
config.recipes = vec![TripletRecipe {
  name: Cow::Borrowed("title_ctx"),
  anchor: Selector::Role(SectionRole::Anchor),
  positive_selector: Selector::Role(SectionRole::Context),
  negative_selector: Selector::Role(SectionRole::Context),
  negative_strategy: NegativeStrategy::WrongArticle,
  weight: 1.0,
  instruction: None,
}];

let sampler = TripletSampler::new(config, Arc::clone(&store));
sampler.register_source(Box::new(source));

let triplets = sampler.next_triplet_batch(SplitLabel::Train)?;
assert_eq!(triplets.triplets.len(), 2);
assert!(triplets.triplets.iter().all(|t| t.recipe == "title_ctx"));

let pairs = sampler.next_pair_batch(SplitLabel::Train)?;
assert_eq!(pairs.pairs.len(), 2);
assert!(pairs
  .pairs
  .iter()
  .all(|p| matches!(p.label, PairLabel::Positive | PairLabel::Negative)));

let text = sampler.next_text_batch(SplitLabel::Train)?;
assert_eq!(text.samples.len(), 2);
assert!(text.samples.iter().all(|s| s.recipe.starts_with("title_ctx_")));

# Ok::<(), triplets::SamplerError>(())
```

If a `next_*_batch` call fails to produce `batch_size` samples, the call returns an error.

- For per-call source weighting, use `next_triplet_batch_with_weights(...)`, `next_pair_batch_with_weights(...)`, or `next_text_batch_with_weights(...)`.
- Missing source ids default to `1.0`; `0.0` disables a source for that call.

Example (different source mix across consecutive batches):

```rust,no_run
use std::collections::HashMap;
use std::sync::Arc;
use triplets::{
  DeterministicSplitStore, TripletSampler, Sampler, SamplerConfig, SplitLabel, SplitRatios,
};

# let split = SplitRatios { train: 0.8, validation: 0.1, test: 0.1 };
# let store = Arc::new(DeterministicSplitStore::new(split, 123).unwrap());
# let config = SamplerConfig::default();
# let sampler = Arc::new(TripletSampler::new(config, store));

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
- **`ingestion_max_records` tuning**: setting this above `batch_size` usually improves sample diversity (broader anchor/negative candidate pool) and reduces near-term repetition, but returns diminish once source availability, split boundaries, and recipe constraints dominate. For remote backends such as Hugging Face, larger initial ingestion targets can require pulling more initial shards before the first batch, so startup latency can increase depending on shard sizes and network throughput.
- **File indexing**: deterministic path ordering + deterministic index permutation for paging.
- **Source ordering**: round-robin by source, deterministic within-source ordering by seed/epoch.
- **Splits**: labels are deterministic from `record_id + seed + ratios`; split APIs enforce `allowed_splits`.
- **Coverage caveat**: if `len_hint` drifts mid-epoch in streaming backends, strict single-pass coverage is not guaranteed.
- **Weights**: recipe/source/chunk weights affect scaling, not deterministic ordering.
- **Scale note**: full scan/sort/index rebuild cost grows roughly linearly with file count and path bytes.
- **Order note**: index batching preserves permutation order; chunked index reads do not remove deterministic shuffling.
- **Manual epoch control**: `sampler.set_epoch(n)` resets per-source cursors and reshuffles deterministically for that epoch.
- **Persisted state scope**: epoch tracking is split-aware, but sampler/source cursors + RNG/round-robin state are persisted per store file.
- **Triplet recipe behavior**: if `SamplerConfig.recipes` is non-empty, those recipes are used for all sources; otherwise each source's `default_triplet_recipes()` is used (if any).
- **Pair batches**: derived from triplets and follow the same source/recipe selection behavior.
- **Text recipe behavior**:
  - If `SamplerConfig.text_recipes` is non-empty, those are used directly.
  - Else if triplet recipes are configured/available, text recipes are derived as `{triplet_name}_anchor`, `{triplet_name}_positive`, `{triplet_name}_negative`.
  - Else per-source text recipes are used when available.
- **Chunk progression**: for each `(record, section)` the sampler keeps a deterministic rotating cursor over that section's chunk windows, so repeated calls spread windows across the run instead of always taking the first window.
- **Oversampling**: when sources run dry, cached records may be reused (no global no-repeat guarantee).

### Reducing shortcut learning

When you use `DataRecord.meta_prefix` / `KvpPrefixSampler`, prefer varied prefix rendering instead of a single rigid header format.

- Use multiple renderings per key (`KvpField` variants) and per-field presence/dropout.
- Vary field order and enable prefix dropout so headers are informative but not mandatory.
- This helps avoid narrow sampling regimes and model shortcuts tied to one repeated prefix pattern.
- Prefixes decorate sampled text only; they do not change deterministic split assignment.

### Advanced source implementation examples

For any new backend (file/API/DB/stream), centralize backend configuration/state access in one helper reused by both `refresh(...)` and `reported_record_count()`.

Why this matters: capacity estimates and runtime sampling stay aligned only when both methods represent the same logical corpus slice.

File-backed pattern:

```rust,ignore
fn source_index(&self, config: &SamplerConfig) -> Result<FileCorpusIndex, SamplerError> {
  let sampler_seed = config.seed;
  Ok(FileCorpusIndex::new(&self.root, &self.id)
    .with_sampler_seed(sampler_seed)
    .with_follow_links(true)
    .with_text_files_only(true)
    .with_directory_grouping(true))
}

fn refresh(
  &self,
  config: &SamplerConfig,
  cursor: Option<&SourceCursor>,
  limit: Option<usize>,
) -> Result<SourceSnapshot, SamplerError> {
  self.source_index(config)?
    .refresh_indexable(cursor, limit, |path| self.build_record(path))
}

fn reported_record_count(&self, config: &SamplerConfig) -> Result<u128, SamplerError> {
  self.source_index(config)?.indexed_record_count().map(|n| n as u128)
}
```

For time-ordered corpora, prefer the `IndexableSource` + `IndexableAdapter` path (and use `IndexablePager` directly only when you need a custom `refresh(...)`) for deterministic shuffled paging with cursor resume.

Helper-based example:

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
