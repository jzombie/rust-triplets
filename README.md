# triplets

[![made-with-rust][rust-logo]][rust-src-page] [![crates.io][crates-badge]][crates-page] [![MIT licensed][mit-license-badge]][mit-license-page] [![Apache 2.0 licensed][apache-2.0-license-badge]][apache-2.0-license-page] [![Coverage][coveralls-badge]][coveralls-page]

_Compose an effectively unlimited supply of training triplets from your existing corpus — rule-driven, reproducible, and multi-source. No hand-mining required._

**WORK IN PROGRESS. THIS API IS BEING PROTOTYPED AND MAY CHANGE WITHOUT NOTICE.**

`triplets` is a reusable core of composable data sampling primitives for deterministic multi-source ML/AI training-data orchestration, with sampler primitives, split/state persistence, chunking and weighting mechanics, and source abstractions (`DataSource`, `DataRecord`) that avoid tying behavior to proprietary corpora.

Because triplets are assembled from source record combinations at batch time — never precomputed — even a modestly-sized corpus can yield billions of unique training examples per source, with trillions achievable across multiple sources once recipes and chunk windows are factored in. Rather than hand-mining or exhaustively annotating every `(anchor, negative)` pair, rule-based recipes combined with a large, diverse corpus let the [Law of Large Numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers) handle what manual curation cannot scale to: with sufficient source variety, the volume of correctly-structured combinations far outweighs any occasional noise, and the aggregate gradient signal reliably shapes the intended embedding space.

Each source is also independent: sources can carry their own recipe rules tailored to their data shape, so a document corpus, a QA dataset, and a structured log stream can all participate in the same training run with distinct anchor/positive/negative strategies suited to each. When `SamplerConfig.recipes` is non-empty those recipes apply uniformly across all sources; when left empty, each source contributes its own `default_triplet_recipes()`. Batch contribution defaults to equal weighting across all registered sources, but per-batch source weights let you shift that balance at call time — boosting a higher-quality source for one batch, suppressing a noisier one for the next, or tying the mix to live training-loss signals — without restarting or reconfiguring the sampler. This source-level recipe independence, combined with per-batch weighting flexibility, multiplies the practical variety of training signal even further: different domains generate structurally different triplets, each governed by rules that fit the data, all converging into the same embedding space.

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

> Note on "hard triplet" mining: mining hard triplets can be performed purely via metadata and recipe-driven selectors (no semantic scorer required). Negatives are chosen dynamically per anchor/positive — across record fields, chunk windows, and recipe selectors — so triplets are typically unique even without changing weights or recipes. As a result, a training run naturally sees a mixture of easy and hard negatives as sampling and configuration play out. More broadly, the scale of the combinatorial space makes exhaustive hand-mining unnecessary: with sufficient source variety, the Law of Large Numbers ensures that any individual misaligned negative is statistically outweighed by the correctly-structured majority, and the model still converges toward the intended embedding geometry.

## High-level features

- **Automatic deterministic splits** (train/validation/test) from record IDs + seed.
- **Sampler-seed-driven source determinism** for built-in deterministic source ordering (file + Hugging Face).
- **Runtime batch sampling** via `next_triplet_batch`, `next_pair_batch`, and `next_text_batch`.
- **Recipe-driven sample construction** for triplet/pair/text generation (anchor/positive/negative selectors).
- **Per-source independent recipe rules**: when `SamplerConfig.recipes` is left empty, each source supplies its own `default_triplet_recipes()` so sources with different data shapes — documents, QA pairs, structured logs — can each use tailored anchor/positive/negative strategies without affecting one another. A global recipe set can still be provided to override all sources uniformly. Batch contribution defaults to equal weighting across all registered sources; per-call source weights let you shift that balance at any time without reconfiguring the sampler.
- **Combinatorial triplet supply from modest corpora**: Triplets are assembled from source record combinations at batch time — never precomputed. N records yield up to N×(N−1) raw combinations per recipe, multiplied across all configured recipes and chunk windows. Even a moderate corpus generates billions of valid, unique training examples without enumerating them; raw source availability is the only practical ceiling.
- **Law of Large Numbers over hard mining**: Rule-based recipes across a large, diverse corpus let statistical variety replace exhaustive pair annotation. With enough source samples, the volume of correctly-structured triplets far outweighs any individual misaligned negative — the aggregate gradient signal reliably pulls similar records together and pushes dissimilar ones apart, without manually auditing every `(anchor, negative)` pair.
- **Automatic long-section recipe injection**: for sources with sections longer than `chunking.max_window_tokens`, automatically adds `auto_injected_long_section_chunk_pair_wrong_article`, which builds anchor/positive from two different context windows of the same record and uses a context section from a different record as the negative.
- **Deterministic long-section chunking**: short text stays as one chunk; long text becomes multiple chunk candidates (sliding windows) sampled over time. Chunks are not emitted as one grouped bundle; each sampled triplet/pair/text item uses one selected chunk at a time. Defaults are `max_window_tokens=1024`, `overlap_tokens=[64]`, and `summary_fallback_tokens=512` (all configurable via `SamplerConfig.chunking`).
- **Weight-aware sampling controls** across source weights, recipe weights, and chunk trust/quality weighting.
- **Anti-shortcut metadata-prefix variation** via `KvpPrefixSampler` (variant choice, per-field presence probabilities, field-order shuffle, and prefix dropout) to reduce rigid header-pattern dependence.
- **Per-source batch mixing controls** so multiple sources can contribute to the same batch, with independent source frequency controls (including over/under-sampling).
- **Per-source trust controls** to weight quality/trust independently by source/taxonomy and help mitigate bias from uneven source quality.
- **Per-batch dynamic source reweighting** so source weights can be changed across batches (for example from loss/metric feedback) while training.
- **Resume support** via `save_sampler_state(save_to)` and split-store persistence.
- **Source-agnostic backends** (`DataSource` or `IndexableSource` + `IndexableAdapter`).
- **Supply-chain style orchestration (core layer):** multi-source intake (`refresh`) with per-call parallel ingest, optional per-source weighting, staged buffering, deterministic split routing, and batch assembly into train-ready outputs.
- **Bounded ingestion** windows instead of loading full corpora into memory.
- **Per-call source threading**: during refresh, each source is fetched on its own short-lived thread, then merged deterministically for batch assembly.
- **Background batch prefetching** via `BatchPrefetcher`: spawns its own dedicated background thread that drives a tight production loop — calling the batch-assembly function repeatedly and pushing results into a bounded channel queue. The training loop blocks only on `next()`, which returns the next pre-assembled batch without waiting for source I/O. Within each batch call that background thread makes, the sampler itself fans out to per-source threads for ingestion, so both layers of concurrency are active simultaneously: the prefetch thread keeps the queue warm while per-source threads fetch records in parallel.
- **Streaming-friendly**: sources can be finite or unbounded.

> This crate does **not** perform semantic mining/retrieval scoring by itself; instead, it gives you deterministic, metadata-driven sampling primitives you can feed into your downstream mining/retrieval stack.

## Sources

`triplets` is source-first: sampling begins with one or more registered `DataSource`s, then recipe selection controls how samples are assembled from each source's records/sections.

A **source** is any backend that yields `DataRecord`s (for example filesystem corpora, Hugging Face rows, or your own adapter). The sampler can mix multiple sources in the same run/batch.

Why this matters:

- You define rules (selectors, strategies, weights) once, and the sampler constructs triplets from those rules at runtime.
- You do **not** have to precompute or hand-author every `(anchor, positive, negative)` combination.
- Each source advances with its own cursor/progress, so sparse or slow sources do not block others.
- Sources can be over/under-sampled independently via source weights (including per-batch reweighting).
- When a source has limited fresh records, replay/oversampling can happen for that source without coupling all other sources to the same behavior.

Key weighting concepts:

- **Source weights** control how often each source contributes in a batch (`next_*_batch_with_weights`).
- **Trust weights** (`DataRecord.quality.trust`, optional taxonomy overrides) scale sample influence by source/record quality.
- **Recipe weights** (`TripletRecipe.weight`) control how often each recipe path is selected.
- **Chunk weights** apply after section chunking to modulate long/short-window contribution.

Source weight semantics:

- **Proportional values:** Weights are treated as proportional scalars — only their ratios matter. They do *not* need to sum to `1.0`. For example, `{A: 2.0, B: 1.0}` and `{A: 0.75, B: 0.375}` are equivalent in relative contribution.
- **Omitted sources default to 1.0:** If a source id is not present in a per-call weight map, it is treated as having weight `1.0` for that call.
- **Invalid entries cause errors:** Passing a weight map that references an unknown source id or contains a negative weight will cause the weight-aware ingestion APIs to return an error (`SamplerError::InvalidWeight`). Callers should handle the `Result` returned by the weight-aware methods (for example, the ingestion helpers `advance_with_weights`, `refresh_all_with_weights`, and `force_refresh_all_with_weights`, and the sampler paths that propagate their errors).
- **Zero weights are allowed:** A source weight of exactly `0.0` disables contribution from that source for the call (it is effectively skipped). If all provided weights are non-positive, the implementation falls back to equal weighting.

### Recipes

### What is a recipe?

A recipe defines how one training sample is assembled from eligible sections:

- For **triplets**: selector for anchor, selector for positive, selector for negative, plus negative strategy and recipe weight.
- For **pairs/text**: either derived from triplet recipes or explicitly configured text recipes.

Recipes are metadata-driven selection rules; they define *what can be sampled*, while runtime sampling/weights decide *how often* each eligible path is drawn.

Recipe origin can be user-defined, system-defined, or mixed in the same run.

Basic recipe example:

```rust,no_run
use std::borrow::Cow;
use triplets::{NegativeStrategy, SectionRole, Selector, TripletRecipe};

let recipe = TripletRecipe {
  name: Cow::Borrowed("title_context_wrong_article"),
  anchor: Selector::Role(SectionRole::Anchor),
  positive_selector: Selector::Role(SectionRole::Context),
  negative_selector: Selector::Role(SectionRole::Context),
  negative_strategy: NegativeStrategy::WrongArticle,
  weight: 1.0,
  instruction: None,
};
# let _ = recipe;
```

### How recipe selection works

- If `SamplerConfig.recipes` is non-empty, those triplet recipes are used for all sources.
- Otherwise, each source uses its own `default_triplet_recipes()` (if any).
- System-defined recipes (for example, auto-injected long-section recipes) can be appended to source recipe pools, so effective sampling can use both user-defined and system-defined recipes.
- Pair batches are derived from the selected triplet recipe stream.
- Text recipes are resolved in this order:
  - `SamplerConfig.text_recipes` (if explicitly set)
  - derived from triplet recipes (`{triplet_name}_anchor|positive|negative`)
  - source-provided text recipes (fallback)

### Auto-injected long-section recipe

- Auto recipe name: `auto_injected_long_section_chunk_pair_wrong_article`.
- It may be appended per source during normal ingest/cache sync.
- It is eligible when a source has at least one section longer than `chunking.max_window_tokens`.
- Recipe selectors: anchor=`Context`, positive=`Context`, negative=`Context` with `WrongArticle` negatives.
- It augments the source's recipe pool; it does not change `select_chunk` globally.
- Anchor and positive are two independent chunk draws (not concatenated text, not derived from each other).

### Using a source for sampling

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

### Integrated sources

`triplets` ships with two built-in sources; if you use either, deterministic paging is always enabled (`FileSource`, `HuggingFaceRowSource`).

- **File source (`FileSource`)**: local files and folders.
- **Hugging Face source (`HuggingFaceRowSource`)** *(feature: `huggingface`)*: HF dataset rows.

Built-in source defaults use a mixed-negative recipe pool when `SamplerConfig.recipes` is empty:

- `*_anchor_context_wrong_article` / `title_context_wrong_article` (context negatives): weight `0.75`
- `*_anchor_anchor_wrong_article` / `title_anchor_wrong_article` (anchor negatives): weight `0.25`

File source note:

- Date-aware defaults are **gated** by `FileSourceConfig::with_date_aware_default_recipe(true)`.
- When enabled, file-source date-aware recipes are:
  - `title_context_wrong_date` (context negatives): weight `0.30`
  - `title_anchor_wrong_date` (anchor negatives): weight `0.10`
  - `title_context_wrong_article` (context negatives): weight `0.35`
  - `title_anchor_wrong_article` (anchor negatives): weight `0.25`
- Default `FileSourceConfig::new(...)` leaves date-aware defaults disabled.
- Here, "date-aware" means publication date metadata (for example `META_FIELD_DATE` from taxonomy/record metadata), **not** filesystem modification/creation/access timestamps.

Hugging Face source defaults use:

- `*_anchor_context_wrong_article` (context negatives): weight `0.75`
- `*_anchor_anchor_wrong_article` (anchor negatives): weight `0.25`

#### Hugging Face source lists (recommended)

Define HF sources in a text file and pass it to the demo or your own loader. The `hf://` prefix is a `triplets`-specific shorthand used only in these lists:

```text
hf://org/dataset/config/split anchor=... positive=... context=a,b text=x,y
```

Rules:

- Lines are whitespace-delimited; comments start with `#`.
- `anchor=`, `positive=`, `context=`, and `text=` are the only accepted keys.
- At least one mapping key is required per line.
- HF row extraction is strict: no auto-detect fallback is used when mappings are absent.

**Two extraction modes — pick one per source line:**

| Mode             | When active                               | Keys used                          |
| ---------------- | ----------------------------------------- | ---------------------------------- |
| **Role-based**   | `anchor=` (and/or `positive=`) is present | `anchor=`, `positive=`, `context=` |
| **Text-columns** | `anchor=` is absent; only `text=` is set  | `text=`                            |

- **Role-based mode** maps dataset columns directly onto triplet roles.  `anchor=` becomes the anchor section; `positive=` becomes the positive/context section; `context=` adds extra context sections.  Use this when your dataset already has distinct role-labelled fields (title, body, summary, etc.).
- **Text-columns mode** is simpler: one column (or the first non-empty candidate) becomes the entire record content.  The sampler then builds anchor/positive/negative from different records or chunks of that content.  Use this for datasets that have a single text blob per row.

Column-list semantics:

- `context=` accepts a comma-delimited list of columns; **all** listed columns must be present and non-empty, or the row is skipped.  Each one becomes a separate context section.
- `anchor=`, `positive=`, and `text=` accept comma-delimited **candidate** column lists.  Candidates are tried in order; the **first** non-empty value found is used.  The row is skipped only when **no** candidate yields content.  Use multiple candidates when a field is sometimes absent (for example `anchor=title,text` uses `title` when present and falls back to `text`).

Example list (see [examples/common/hf_sources.txt](examples/common/hf_sources.txt)):

```text
# role columns
hf://labofsahil/hackernews-vector-search-dataset/default text=text
hf://wikimedia/wikipedia/20231101.en/train anchor=title positive=text

# explicit text-column mode
hf://pfox/71k-English-uncleaned-wordlist/default text=text
```

HF backend persistence and lookup model:

- Persisted shard format is per-shard `.simdr` row stores.
- `.parquet` is treated as transient decode input only and is removed after transcode.
- Store keys use a single canonical key type: positional local row offset (`rowv1|<u64-le>`).
- Reads use batch key lookups (`batch_read`) over these positional keys.

### Adding new sources

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

The `multi_source_demo` example persists sampler/split state by default to a
managed cache-group path under:

- `.cache/triplets/multi-source-demo/split_store.bin`

You can still set an explicit persistence file path:

- `--split-store-path <FILE>`

If you need explicit load/save control, use:

- `FileSplitStore::open_with_load_path(Some(load_from), save_to, ratios, seed)`

This loads from `load_from` only when `save_to` does not exist yet, then writes
to `save_to`. Passing `None` for `load_from` starts from an empty/new store.
Parent directories for `save_to` are created recursively when missing.

When using `sampler.save_sampler_state(Some(path.as_path()))`:

- `path` must not already exist (the call errors rather than overwriting).
- parent directories for `path` are created recursively when missing.
- later `sampler.save_sampler_state(None)` calls still write to the originally opened store file.

## Usage flow

Short version:

- Call **`sampler.next_triplet_batch(split)`**, **`sampler.next_pair_batch(split)`**, or **`sampler.next_text_batch(split)`** to sample batches (ingestion happens automatically).
- Call **`sampler.save_sampler_state(None)`** when you want restart-resume behavior.
- Call **`sampler.save_sampler_state(Some(path.as_path()))`** when you also want to persist current state to another file.
  - `Some(path)` requires `path` to not already exist; otherwise it returns an error instead of overwriting.
  - After `Some(path)`, subsequent `None` calls still save to the originally opened store path.
- Optionally call **`sampler.set_epoch(n)`** for explicit epoch control.

Step-by-step:

1. Build config + open the split store.
   - Use `FileSplitStore::open(path, ratios, seed)` for a single file, or
     `FileSplitStore::open_with_load_path(Some(load_from), save_to, ratios, seed)` for explicit load/save split.
2. Register sources.
3. Call one of **`sampler.next_triplet_batch(split)`**, **`sampler.next_pair_batch(split)`**, or **`sampler.next_text_batch(split)`**.
4. Call **`sampler.save_sampler_state(None)`** when you want to write persisted sampler/split state (typically at the end of an epoch or at explicit checkpoint boundaries). **Do not call this every step.** Very frequent writes can create high I/O overhead and, at very large write counts (for example, tens of millions), can also adversely affect split-store initialization time.
  - Use `sampler.save_sampler_state(Some(path.as_path()))` to additionally write to another file on demand.
  - `path` must be a new file path; if it already exists, the call fails rather than overwriting.
  - This on-demand write does not retarget the sampler; later `sampler.save_sampler_state(None)` calls still write to the original store file.
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
- `sampler.save_sampler_state(None)` is manual; skipping it means no resume state after restart.
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
- **Overlap materialization**: when multiple overlap values are configured, the sampler materializes windows for each configured overlap value and adds all of them to the chunk pool (in config order); it does not randomly choose a single overlap value.
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
