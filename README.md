# triplets

[![made-with-rust][rust-logo]][rust-src-page] [![crates.io][crates-badge]][crates-page] [![MIT licensed][mit-license-badge]][mit-license-page] [![Apache 2.0 licensed][apache-2.0-license-badge]][apache-2.0-license-page] [![Coverage][coveralls-badge]][coveralls-page]

**WORK IN PROGRESS. THIS API IS BEING PROTOTYPED AND MAY CHANGE WITHOUT NOTICE.**

Compose an effectively unlimited supply of [training triplets](https://en.wikipedia.org/wiki/Triplet_loss), pairs, or plaintext samples, from your existing corpus, with optional [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) hard-negative mining.

- Multiple input source mixing, rule-driven sampling recipes, using a`Rayon`-managed thread pool with optional multi-batch prebuffering for training.
- Optional per-recipe training instructions.
- Configurable source sampling weights, independent source cursors, source/record trust weighting, recipe weighting, and position-aware window weighting (`start_ratio`), so you can tune per-source sampling frequency and per-sample training weight.
- Automatic & deterministic data splits.
- Optional split-store state snapshotting: split assignments (train/validation/test) and sampler cursor state are persisted to a compact binary file. Only record IDs and lightweight metadata are stored — record text and payloads are never written to the split store, keeping snapshot files small even for large corpora. Resume a multi-epoch training run from any persisted checkpoint.
- Automatic source chunking (ensure all data is eventually consumed regardless of context window size).
- Anti-regime and diversity features: Anchor/positive swapping; negatives drawn from other anchors/positives; long anchor/positive sections are chunked into additional anchor/positive windows; deterministic pseudo-random ID sampling via IndexPermutation (affine/LCG-style permutation with cycle-walking); and hash-shuffled source cycling (epoch/cycle-seeded) layered over split-aware Round-Robin cursors to avoid fixed Round-Robin regimes.
- Combine any combination of text-based streaming and static data sources.
- Included adapters for HuggingFace and file-based sources. Included traits to roll your own data loaders from any source.
- Fast, reproducible baseline sampling (great for iteration/debug), with optional BM25 hard-negative mining when you want stricter lexical difficulty.
- Low memory footprint; quick to compile; written in Rust.
- [MIT][mit-license-page] and [Apache 2.0][apache-2.0-license-page] licensed.

> _The loss function and choice of ML framework is a separate concern; this crate only handles the data._

Jump to the [quick start](#quick-start).

View the [capabilities](#capabilities) for a deeper dive into the aforementioned list.

> _CI is configured to run tests/linting on macOS, Linux, and Windows._

**Note:** This crate is intended primarily for textual (or textualized) data — records that can be represented as text (for example: documents, QA pairs, logs, or metadata-prefixed chunks) suitable for language-model training, embedding/metric-learning workflows, and related text-model pipelines.

## What is a triplet?

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

Training on many `(anchor, positive, negative)` groups helps a model learn useful embedding space structure — see [triplet loss](https://en.wikipedia.org/wiki/Triplet_loss) for the learning objective.

> _This crate is responsible for constructing those triplets from your data; the loss function, optimizer, and training loop are outside its scope and remain yours to choose._

In this crate, triplets are built automatically from one or more data sources using metadata-driven, user-defined recipes/selectors for anchor/positive/negative section choice.

### What a triplet looks like

`SampleTriplet` is an **output** type — the sampler produces it from your sources and recipes, you only consume it. The fields are compile-checked below so this will fail if anything changes:

```rust,no_run
use triplets::{RecordChunk, SampleTriplet, QualityScore};
use triplets::data::ChunkView;

// Output type example; you do not have to type this
# let triplet = SampleTriplet {
#   recipe: "title_context_wrong_article".to_string(),
#   anchor: RecordChunk {
#     record_id: "source_a::article_a".to_string(),
#     section_idx: 0,
#     view: ChunkView::Window { index: 0, overlap: 0, span: 512, start_ratio: 0.0 },
#     text: "Researchers report a breakthrough in solar cell efficiency.".to_string(),
#     tokens_estimate: 9,
#     quality: QualityScore::default(),
#   },
#   positive: RecordChunk {
#     record_id: "source_a::article_a".to_string(),
#     section_idx: 1,
#     view: ChunkView::Window { index: 0, overlap: 0, span: 512, start_ratio: 0.0 },
#     text: "The team achieved 35% efficiency using perovskite layers.".to_string(),
#     tokens_estimate: 9,
#     quality: QualityScore::default(),
#   },
#   negative: RecordChunk {
#     record_id: "source_a::article_b".to_string(),
#     section_idx: 0,
#     view: ChunkView::Window { index: 0, overlap: 0, span: 512, start_ratio: 0.0 },
#     text: "Local council approves new zoning guidelines for downtown.".to_string(),
#     tokens_estimate: 8,
#     quality: QualityScore::default(),
#   },
#   weight: 1.0,
#   instruction: None,
# };

// Fields you access during training — the sampler fills these in:
let _anchor_text: &str         = &triplet.anchor.text;
let _pos_text:    &str         = &triplet.positive.text;
let _neg_text:    &str         = &triplet.negative.text;
let _recipe:      &str         = &triplet.recipe;           // which TripletRecipe was used
let _weight:      f32          = triplet.weight;
let _record_id:   &str         = &triplet.anchor.record_id; // back-reference to DataRecord.id
let _instruction: Option<&str> = triplet.instruction.as_deref();
```

## Sources

`triplets` is source-first: sampling begins with one or more registered `DataSource`s, then recipe selection controls how samples are assembled from each source's records/sections.

A **source** is any backend that yields `DataRecord`s (for example filesystem corpora, Hugging Face rows, or your own adapter). The sampler can mix multiple sources in the same run/batch.

> **Terminology note:** In this crate "source" and "loader" are interchangeable concepts. The `DataSource` trait is the crate's loader abstraction — any backend that can yield `DataRecord`s is a source. Both terms appear in the docs and examples; internally the crate uses "source" throughout.

Why this matters:

- You define rules (selectors, strategies, weights) once, and the sampler constructs triplets from those rules at runtime.
- You do **not** have to precompute or hand-author every `(anchor, positive, negative)` combination.
- Each source advances with its own cursor/progress, so sparse or slow sources do not block others.
- Sources can be over/under-sampled independently via source weights (including per-batch reweighting).
- When a source has limited fresh records, replay/oversampling can happen for that source without coupling all other sources to the same behavior.

Key weighting concepts:
- **Default trust** (`QualityScore::default().trust`): 0.5 (used when a record/source doesn't override trust).
- **Source weights** control how often each source contributes in a batch (`next_*_batch_with_weights`).
- **Trust weights** (`DataRecord.quality.trust`, optional taxonomy overrides) scale sample influence by source/record quality.
- **Recipe weights** (`TripletRecipe.weight`) control how often each recipe path is selected.
- **Chunk weights** apply after section chunking to modulate long/short-window contribution.

Source weight semantics:

- **Proportional values:** Weights are treated as proportional scalars — only their ratios matter. They do *not* need to sum to `1.0`. For example, `{A: 2.0, B: 1.0}` and `{A: 0.75, B: 0.375}` are equivalent in relative contribution.
- **Omitted sources default to 1.0:** If a source id is not present in a per-call weight map, it is treated as having weight `1.0` for that call.
- **Invalid entries cause errors:** Passing a weight map that references an unknown source id or contains a negative weight will cause the weight-aware ingestion APIs to return an error (`SamplerError::InvalidWeight`). Callers should handle the `Result` returned by the weight-aware methods (for example, the ingestion helpers `advance_with_weights`, `refresh_all_with_weights`, and `force_refresh_all_with_weights`, and the sampler paths that propagate their errors).
- **Zero weights are allowed:** A source weight of exactly `0.0` disables contribution from that source for the call (it is effectively skipped). If all provided weights are non-positive, the implementation falls back to equal weighting.

## Recipes

A recipe defines how one training sample is assembled from eligible sections:

- For **triplets**: selector for anchor, selector for positive, selector for negative, plus negative strategy and recipe weight.
- For **pairs/text**: either derived from triplet recipes or explicitly configured text recipes.

Recipes are metadata-driven selection rules; they define *what can be sampled*, while runtime sampling/weights decide *how often* each eligible path is drawn.

Recipe origin can be user-defined, system-defined, or mixed in the same run.

```rust,no_run
use std::borrow::Cow;
use triplets::{NegativeStrategy, SectionRole, Selector, TripletRecipe};

let _recipe = TripletRecipe {
  name: Cow::Borrowed("title_context_wrong_article"),
  anchor: Selector::Role(SectionRole::Anchor),
  positive_selector: Selector::Role(SectionRole::Context),
  negative_selector: Selector::Role(SectionRole::Context),
  negative_strategy: NegativeStrategy::WrongArticle,
  weight: 1.0,
  instruction: None,
  allow_same_anchor_positive: false,
};
```

**`TripletRecipe` field reference:**

| Field                                                | Description                                                                                                                                                                                                                                                                                                                                                                                                           |
|------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `name`                                               | Unique identifier; appears in `SampleTriplet.recipe` and derived text sample names.                                                                                                                                                                                                                                                                                                                                   |
| `anchor` / `positive_selector` / `negative_selector` | `Selector` rules: `Role(SectionRole::Anchor)` picks a section with that role; `Random` picks any section; etc.                                                                                                                                                                                                                                                                                                        |
| `negative_strategy`                                  | Candidate pool for the negative slot: `WrongArticle` draws from different records; `WrongDate` draws from records with a different date (file source only).                                                                                                                                                                                                                                                           |
| `weight`                                             | Relative sampling frequency among recipes; only the ratio between weights matters.                                                                                                                                                                                                                                                                                                                                    |
| `instruction`                                        | Optional instruction string propagated to every `SampleTriplet.instruction` and `TextSample.instruction` produced by this recipe. Use for instruction-tuning workflows where the encoder is prompted per task (`"Retrieve a relevant document."`, `"Classify the sentiment."`, etc.). `None` means no instruction is attached and the output field is `None`.                                                         |
| `allow_same_anchor_positive`                         | When `true`, allows triplets where anchor and positive carry identical text (SimCSE / dropout-trick training). The encoder sees the same text twice; dropout produces two slightly different embeddings at training time. The negative must still differ from both. Defaults to `false`. The HuggingFace source sets this `true` automatically in text-columns mode (when `text=` is mapped but `anchor=` is absent). |

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

## How it works

`triplets` is a reusable core of composable data sampling primitives for deterministic multi-source ML/AI training-data orchestration, with sampler primitives, split/state persistence, chunking and weighting mechanics, and source abstractions (`DataSource`, `DataRecord`) that avoid tying behavior to proprietary corpora.

Because triplets are assembled from source record combinations at batch time — not precomputed corpus-wide (though optional prebuffering does pre-assemble a bounded ahead-of-time queue) — even a modestly-sized corpus can yield billions of unique training examples per source, with trillions achievable across multiple sources once recipes and chunk windows are factored in. Rather than exhaustively annotating every `(anchor, negative)` pair, rule-based recipes provide broad coverage and deterministic behavior while keeping generation scalable. This works most naturally when each source is rooted in a coherent domain — negatives are mined per-source by default, making the crate a natural fit for fine-tuning on domain-specific data. General-purpose training is equally supported depending on how sources and recipes are constructed, and optional BM25 hard-negative mining can be enabled when required.

Each source is independent: sources can carry their own recipe rules tailored to their data shape, so a document corpus, a QA dataset, and a structured log stream can all participate in the same training run with distinct anchor/positive/negative strategies suited to each. When `SamplerConfig.recipes` is non-empty those recipes apply uniformly across all sources; when left empty, each source contributes its own `default_triplet_recipes()`. Batch contribution defaults to equal weighting across all registered sources, but per-batch source weights let you shift that balance at call time — boosting a higher-quality source for one batch, suppressing a noisier one for the next, or tying the mix to live training-loss signals — without restarting or reconfiguring the sampler.

> **A note on domain assumptions:** By default, negatives are mined *within* each source — an anchor's negative candidates are drawn from the same source that produced the anchor. This implicitly treats each source as a coherent domain (for example: a corpus of financial news, a physics paper collection, or a product-review dataset). Because of this, the crate is naturally well-suited for **fine-tuning** on domain-specific data, where in-source negatives are already meaningfully hard without a semantic scorer — a pre-trained model that already separates broad domains will find two articles from the same financial corpus far more confusable than a finance article paired with a physics paper. Whether this approach extends gracefully to general-purpose training depends entirely on how sources and recipes are constructed.
>
> **On in-source negative scope:** Scoping negatives to each source is a deliberate design choice for domain-coherent corpora, not a universal assumption. When a source is a well-defined domain (finance news, physics papers, product reviews), same-source negatives are already hard — they prevent the model from learning trivial domain surface features while still requiring it to separate topically similar content. If your sources are instead general-purpose heterogeneous archives where source boundaries are not meaningful semantic divisions, you may benefit from cross-source negatives. The existing fallback path in the sampler will draw from the full same-split pool when a source is too small to supply candidates on its own; whether promoting that to the primary path improves your model depends on your corpus layout and training objective.
>
> **On negative hardness: a deliberate diversity-first design:** This crate does not attempt to mine the hardest-possible negative for every triplet, and it intentionally imposes no artificial floor or ceiling on negative difficulty. The resulting negative pool is a varied mix — some hard, some medium, some soft — shaped by recipe selector constraints, in-source candidate availability, and (optionally) BM25 lexical re-ranking within each strategy-defined pool.
>
> This is a conscious tradeoff. Broad recipe pools combined with lexical ranking produce **high negative variety** and reduce representation collapse, but do not guarantee that every individual triplet is maximally challenging. The consequence is stronger diversity and lower risk of overfitting to narrow hard-negative patterns, at the cost of weaker per-triplet hardness consistency — particularly for anchor-heavy recipes where average query representation may be noisy.
>
> **This approach is a good fit when your goal is:** robust broad generalization, high output variety, or avoiding the pathologies of always-hardest mining (mode collapse, training instability, sensitivity to outlier anchors).
>
> **It is a less natural fit when your goal is:** consistently high semantic hardness per triplet — for example, fine-grained metric learning where every negative must be a near-miss. For that regime, tighter per-recipe candidate definitions and selector-aware ranking text (or an external embedding-based re-ranker pre-populating the source or the negative pool) are better starting points.
>
> **On BM25 re-ranking specifically:** The optional `bm25-mining` feature adds BM25-based lexical ranking within strategy-defined candidate pools — BM25 scores are used only to re-rank candidates already selected by your recipe strategy, not as a global filter or hardness gate. It shifts the pool toward lexically harder negatives while preserving diversity mechanics. BM25 is a keyword-overlap ranker, not a semantic one; it is best understood as an inexpensive first-pass that lifts average negative quality without requiring an encoder at training-data generation time. For embedding-based re-ranking (online iterative mining with the trained encoder, cross-encoder scoring, or dense-retrieval-based hard-negative refresh), those integrate naturally via pre-ranked sources or per-batch source reweighting and represent the appropriate next step once the model has been initialized on BM25-ranked candidates.

## Features

| Feature            | What it enables                                                                                                                                                                                                                                                                                                                                                            | Default |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| `huggingface`      | `HuggingFaceRowSource` — streaming download and sampling from Hugging Face dataset repositories (parquet/ndjson shards, ClassLabel resolution, disk-cap eviction). Adds `hf-hub`, `parquet`, `ureq`, `rayon`, `serde_json`.                                                                                                                                                | No      |
| `bm25-mining`      | BM25 hard-negative ranking within strategy-defined candidate pools. Adds a `bm25` dependency. Rule-based strategy selection always runs first to define the eligible pool; BM25 re-ranks within that pool when this feature is enabled. When absent, candidate selection within each strategy pool is uniform (no re-ranking step).                                        | No      |
| `extended-metrics` | Enables additional per-triplet similarity diagnostics in the `multi_source_demo` output. Currently prints the Jaccard similarity (word-token overlap) between the anchor and each of the positive and negative chunks for every triplet in a batch. Adds no dependencies. Intended for manual inspection and debugging of sampling quality, not for use in training loops. | No      |

```toml
[dependencies]
triplets = { version = "*", features = ["huggingface", "bm25-mining"] }
```

Neither feature is on by default; enable them independently or together.

## Quick start

```bash
# sample triplet batches from the example dataset
cargo run --example multi_source_demo

# sample with extended per-triplet similarity metrics (Jaccard anchor↔positive and anchor↔negative)
cargo run --features extended-metrics --example multi_source_demo

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

Minimal working snippet:

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

## Using the sampler

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

### Split-store path configuration

**What the split store contains:** Record IDs and their assigned split labels (train/validation/test), plus a small amount of sampler cursor state (epoch counter, round-robin index, RNG state). It does **not** store record text, embeddings, or section payloads — only the identifiers needed for deterministic re-assignment and checkpoint resume. Even for corpora with millions of records, snapshot files remain compact (a few megabytes rather than gigabytes).

The `multi_source_demo` example persists sampler/split state by default to a managed cache-group path:

- `.cache/triplets/multi-source-demo/split_store.bin`

You can set an explicit persistence file path with `--split-store-path <FILE>`.

If you need explicit load/save control, use:

- `FileSplitStore::open_with_load_path(Some(load_from), save_to, ratios, seed)`

This loads from `load_from` only when `save_to` does not exist yet, then writes to `save_to`. Passing `None` for `load_from` starts from an empty/new store. Parent directories for `save_to` are created recursively when missing.

When using `sampler.save_sampler_state(Some(path.as_path()))`:

- `path` must not already exist (the call errors rather than overwriting).
- parent directories for `path` are created recursively when missing.
- later `sampler.save_sampler_state(None)` calls still write to the originally opened store file.

### Prefetcher

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
let _batch = prefetcher.next().unwrap();
```

### Expected batch output

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
  allow_same_anchor_positive: false,
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

let _batch_a = sampler
  .next_triplet_batch_with_weights(SplitLabel::Train, &weights_a)
  .unwrap();
let _batch_b = sampler
  .next_triplet_batch_with_weights(SplitLabel::Train, &weights_b)
  .unwrap();
```

- **Production readiness note**: if `len_hint` drifts in streaming/append-only sources, epoch order/coverage can repeat/skip records within an epoch, even though split assignment remains deterministic.

## Source backends

### FileSource

`FileSource` provides fully deterministic paging — same seed and corpus always produces the same row order.

Default recipes when `SamplerConfig.recipes` is empty:

- `*_anchor_context_wrong_article` / `title_context_wrong_article` (context negatives): weight `0.75`
- `*_anchor_anchor_wrong_article` / `title_anchor_wrong_article` (anchor negatives): weight `0.25`

Date-aware defaults (opt-in via `FileSourceConfig::with_date_aware_default_recipe(true)`):

- `title_context_wrong_date` (context negatives): weight `0.30`
- `title_anchor_wrong_date` (anchor negatives): weight `0.10`
- `title_context_wrong_article` (context negatives): weight `0.35`
- `title_anchor_wrong_article` (anchor negatives): weight `0.25`

"Date-aware" means publication date metadata (for example `META_FIELD_DATE` from taxonomy/record metadata), **not** filesystem modification/creation/access timestamps. `FileSourceConfig::new(...)` leaves date-aware defaults disabled.

### Hugging Face source *(feature: `huggingface`)*

`HuggingFaceRowSource` streams parquet/ndjson shards from Hugging Face dataset repositories. Shard download order is deterministic by seed (same seed + same HF manifest → same download sequence). Row-level selection within each `refresh` call is seeded by the number of locally materialized rows, so it is **not reproducible across cache wipes**. Split assignment (Train/Val/Test) remains fully deterministic and cache-independent.

Shard download failures are logged as warnings and the sequence position is skipped — no automatic retry. The skipped position becomes reachable again on disk-cap eviction, an epoch-seed change, or source reconstruction. For small datasets that fit within the disk cap, all shards are typically on disk before the cursor exhausts, so a transient failure only delays that shard until the next reset.

Default recipes when `SamplerConfig.recipes` is empty:

- `*_anchor_context_wrong_article` (context negatives): weight `0.75`
- `*_anchor_anchor_wrong_article` (anchor negatives): weight `0.25`

#### Source lists (recommended)

Define HF sources in a text file and pass it to the demo or your own loader. The `hf://` prefix is a `triplets`-specific shorthand used only in these lists:

```text
hf://org/dataset/config/split anchor=... positive=... context=a,b text=x,y [trust=<f32>] [source_id=<name>]
```

**URI path components — `config` and `split`:**

| Components supplied             | Behaviour                                                                            |
|---------------------------------|--------------------------------------------------------------------------------------|
| `hf://org/dataset`              | Config defaults to `default`; all splits discovered automatically.                   |
| `hf://org/dataset/config`       | Specified config; all splits discovered automatically.                               |
| `hf://org/dataset/config/split` | Specified config **and** split — only shards belonging to that split are downloaded. |

The way `split` filters shards depends on how the repository is laid out:

- **Sharded layout** (most large datasets): shards follow the convention
  `<split>/shard-NNNNN.parquet` or `<split>-NNNNN-of-MMMMM.parquet`.
  The split name is matched against the directory prefix, a `-split-` token,
  or a `split-` filename prefix.
- **Flat-table layout**: some repositories store each logical table as a
  single top-level file whose stem is the table name — e.g.
  `data/stock_news.parquet`.  When the split component of the URI exactly
  matches the file stem (`stock_news`), that file and only that file is
  selected.  The match is exact, so `train` does **not** accidentally select
  `training.parquet`.
- **No split specified** (omitted or empty): all parquet/ndjson files in the
  config are downloaded.  For multi-table flat repositories this means every
  table is ingested together, which is usually not what you want — supply the
  explicit table name as the split component to target a single table.

Rules:

- Lines are whitespace-delimited; comments start with `#`.
- `anchor=`, `positive=`, `context=`, `text=`, `trust=`, and `source_id=` are the accepted keys.
- **Any other key is an error.** Unknown or misspelled keys (for example `positve=`, `iajfaijww=`) are rejected immediately with `"unsupported mapping key '<key>'"` — the parser never silently ignores unrecognised tokens, so typos surface at load time rather than producing silently empty mappings.
- At least one of `anchor=`, `positive=`, `context=`, or `text=` is required per line; a line with a valid URI but no mapping keys is also an error.
- HF row extraction is strict: no auto-detect fallback is used when mappings are absent.

**Two extraction modes — pick one per source line:**

| Mode             | When active                               | Keys used                          |
|------------------|-------------------------------------------|------------------------------------|
| **Role-based**   | `anchor=` (and/or `positive=`) is present | `anchor=`, `positive=`, `context=` |
| **Text-columns** | `anchor=` is absent; only `text=` is set  | `text=`                            |

- **Role-based mode** maps dataset columns directly onto triplet roles.  `anchor=` becomes the anchor section; `positive=` becomes the positive/context section; `context=` adds extra context sections.  Use this when your dataset already has distinct role-labelled fields (title, body, summary, etc.).
- **Text-columns mode** is simpler: one column (or the first non-empty candidate) becomes the entire record content.  The sampler then builds anchor/positive/negative from different records or chunks of that content.  Use this for datasets that have a single text blob per row.

Column-list semantics:

- `context=` accepts a comma-delimited list of columns; **all** listed columns must be present and non-empty, or the row is skipped.  Each one becomes a separate context section.
- `anchor=`, `positive=`, and `text=` accept comma-delimited **candidate** column lists.  Candidates are tried in order; the **first** non-empty value found is used.  The row is skipped only when **no** candidate yields content.  Use multiple candidates when a field is sometimes absent (for example `anchor=title,text` uses `title` when present and falls back to `text`).

**Optional per-source overrides:**

- `trust=<f32>` — overrides the default trust score (`0.5`) for every record produced by this source.  Must be in `[0.0, 1.0]`.  Use higher values for high-quality authoritative sources and lower values for noisier ones.
- `source_id=<name>` — overrides the auto-derived source identifier slug for this entry.  When set, the provided name is used verbatim (no deduplication suffix is applied).  Useful for giving a stable, human-readable name to a source regardless of its dataset/config/split path.

**ClassLabel auto-resolution:**

HuggingFace datasets frequently encode categorical columns (such as `sentiment`, `label`, or `category`) as integers using the `ClassLabel` feature type. `HuggingFaceRowSource` automatically resolves those integers to their human-readable string names by querying the datasets-server `/info` endpoint at source construction time. No user-side annotation or manual label mapping is required in the source list.

- Resolution happens once per source construction, before any rows are fetched.
- Resolved labels are written directly into the `.simdr` row stores as strings — there is no integer-to-label lookup at batch time.
- If the `/info` endpoint is unreachable or returns no feature metadata, the source falls back to raw integer strings (e.g. `"0"`, `"1"`, `"2"`) and continues normally.
- **Pre-existing stores are not retroactively updated.** Rows cached before ClassLabel resolution was available retain their raw integer values. Delete the affected shard cache to trigger a fresh transcode with label resolution.

Example list (see [examples/common/hf_sources.txt](examples/common/hf_sources.txt)):

```text
# role columns with default trust and auto slug
hf://labofsahil/hackernews-vector-search-dataset/default text=title,text
hf://wikimedia/wikipedia/20231101.en/train anchor=title positive=text

# high-trust source with an explicit stable name
hf://wikimedia/wikipedia/20231101.en/train anchor=title positive=text trust=0.9 source_id=wiki-en

# lower-trust noisy dataset, custom stable id
hf://org/noisy-web-crawl/default text=text trust=0.3 source_id=noisy-web

# explicit text-column mode
hf://pfox/71k-English-uncleaned-wordlist/default text=text

# ClassLabel column — integers auto-resolved to "neutral"/"bullish"/"bearish" via /info endpoint
hf://TimKoornstra/financial-tweets-sentiment anchor=tweet positive=sentiment

# flat-table layout — one parquet file per table; split selects a single table
hf://defeatbeta/yahoo-finance-data/default/stock_news anchor=title positive=news
```

HF backend persistence and lookup model:

- Persisted shard format is per-shard `.simdr` row stores.
- `.parquet` is treated as transient decode input only and is removed after transcode.
- Store keys use a single canonical key type: positional local row offset (`rowv1|<u64-le>`).
- Reads use batch key lookups (`batch_read`) over these positional keys.

Endpoint overrides:

Three environment variables let you redirect the datasets-server base URLs — useful for test doubles, local mirrors, or air-gapped / on-premises deployments:

| Environment variable           | Endpoint controlled                 |
|--------------------------------|-------------------------------------|
| `TRIPLETS_HF_PARQUET_ENDPOINT` | `/parquet` shard manifest           |
| `TRIPLETS_HF_SIZE_ENDPOINT`    | `/size` total row count             |
| `TRIPLETS_HF_INFO_ENDPOINT`    | `/info` ClassLabel feature metadata |

When set, the variable value replaces the default `https://datasets-server.huggingface.co` base URL for that endpoint. All three are checked at runtime so they work in both unit tests and integration tests.

#### Authentication and private datasets

**Only public HuggingFace datasets are currently supported.** There is no authentication support:

- The `hf-hub` client is constructed with `.with_token(None)` — the HF Hub token is explicitly disabled and `HF_TOKEN` (or any other credential source) is never consulted.
- The three `ureq` HTTP calls to the datasets-server (`/parquet`, `/size`, `/info`) are made without an `Authorization` header.

Datasets that require authentication (gated models, private repos, or organization-private datasets) will return HTTP 401/403 errors from both the datasets-server manifest endpoints and the `hf-hub` shard download path, and the source will fail to construct.

### Adding a custom source

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

let _source = IndexableAdapter::new(MySource { id: "my_source".into() });
```

Then register the source with your sampler and call `next_triplet_batch`, `next_pair_batch`, or `next_text_batch`.

## Negative-selection backends

The sampler's negative-mining step is delegated to a pluggable backend that implements the `NegativeBackend` trait (`src/sampler/backends/`). The active backend is selected at compile time via Cargo features; no runtime configuration is needed.

### How backends fit into the pipeline

Strategy filtering (source isolation, split isolation, date predicates) runs first and produces a candidate pool. The backend then **selects or ranks within that pool only** — it never sees candidates that violate strategy constraints.

### Built-in backends

| Backend          | Enabled when                              | Behaviour                                                                                                                                                                                                                                                                    |
|------------------|-------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `DefaultBackend` | `bm25-mining` feature is absent (default) | Uniform-random selection from the pre-filtered pool, using the sampler's top-level RNG. Zero heap overhead beyond the pool itself.                                                                                                                                           |
| `Bm25Backend`    | `bm25-mining` feature is active           | Rebuilds a BM25 index over the full record pool after each source-refresh cycle and re-ranks the strategy-filtered pool by lexical overlap with the anchor. Candidate selection rotates deterministically per anchor to avoid always returning the single hardest candidate. |

### `NegativeBackend` trait

```rust,ignore
pub(super) trait NegativeBackend: Send {
    /// Select a negative from the strategy-filtered `pool` for `anchor`.
    fn choose_negative(
        &mut self,
        anchor: &DataRecord,
        anchor_split: SplitLabel,
        pool: Vec<DataRecord>,
        fallback_used: bool,
        rng: &mut dyn rand::RngCore,
    ) -> Option<(DataRecord, bool)>;

    fn on_sync_start(&mut self);
    fn on_records_refreshed(
        &mut self,
        records: &IndexMap<RecordId, DataRecord>,
        max_window_tokens: usize,
        split_fn: &dyn Fn(&RecordId) -> Option<SplitLabel>,
        sources_refreshed: bool,
    );
    fn prune_cursors(&mut self, valid_ids: &HashSet<RecordId>);
    fn cursors_empty(&self) -> bool;
}
```

### Adding a custom backend

1. Create `src/sampler/backends/my_backend.rs` and implement `NegativeBackend`.
2. In `src/sampler/backends/mod.rs`, gate the module declaration and re-export behind your feature flag:
   ```rust,ignore
   #[cfg(feature = "my-feature")]
   pub(super) mod my_backend;
   #[cfg(feature = "my-feature")]
   pub(super) use self::my_backend::MyBackend;
   ```
3. In `TripletSamplerInner::new`, swap the constructor to produce a `Box<dyn NegativeBackend>` containing your type.

The sampler core only calls `NegativeBackend` methods — all backend-specific state (BM25 indices, cursor maps, etc.) stays fully encapsulated in the concrete type.

## Capabilities

- **Automatic deterministic splits** (train/validation/test) from record IDs + seed.
- **Sampler-seed-driven source determinism** for built-in deterministic source ordering (file source) and deterministic shard download order (Hugging Face). Note: HF row-level selection within a `refresh` call depends on how many shards are locally cached and is not reproducible across cache wipes — only split assignment and shard download order are stable end-to-end for HF sources.
- **Runtime batch sampling** via `next_triplet_batch`, `next_pair_batch`, and `next_text_batch`.
- **Recipe-driven sample construction** for triplet/pair/text generation (anchor/positive/negative selectors).
- **Per-source independent recipe rules**: when `SamplerConfig.recipes` is left empty, each source supplies its own `default_triplet_recipes()` so sources with different data shapes — documents, QA pairs, structured logs — can each use tailored anchor/positive/negative strategies without affecting one another. A global recipe set can still be provided to override all sources uniformly.
- **Optional per-recipe instruction strings** (`TripletRecipe.instruction` / `TextRecipe.instruction`): an `Option<Cow<'static, str>>` that, when set, is propagated verbatim to every `SampleTriplet.instruction` and `TextSample.instruction` produced by that recipe. Use this for instruction-tuning workflows where the encoder is prompted per task (e.g. `"Retrieve a relevant document."`, `"Classify the sentiment."`). When `None`, the output field is also `None` and no instruction overhead is paid. Instructions are per-recipe, so different recipes in the same sampler can carry different instructions simultaneously.
- **Combinatorial triplet supply from modest corpora**: triplets are assembled on demand from source record combinations at batch time, not precomputed corpus-wide. Optional prefetch/prebuffering only materializes a bounded queue of upcoming sampled batches. N records still yield up to N×(N−1) raw combinations per recipe, multiplied across configured recipes and chunk windows.
- **Optional BM25 hard-negative mining** (`bm25-mining` feature): ranks same-split candidates inside each strategy-defined pool by BM25 score. Rule-based sampling remains the default fast path; BM25 is a ranking layer on top of existing strategy pools, not a global filter. Because negatives are mined per-source by default, each source is still treated as a domain boundary.
- **Automatic long-section recipe injection**: for sources with sections longer than `chunking.max_window_tokens`, automatically adds `auto_injected_long_section_chunk_pair_wrong_article`, which builds anchor/positive from two different context windows of the same record and uses a context section from a different record as the negative.
- **Deterministic long-section chunking**: short text stays as one chunk; long text becomes multiple chunk candidates (sliding windows) sampled over time. Defaults are `max_window_tokens=1024`, `overlap_tokens=[64]`, and `summary_fallback_tokens=512` (all configurable via `SamplerConfig.chunking`).
- **Weight-aware sampling controls** across source weights, recipe weights, and chunk trust/quality weighting.
- **Anti-shortcut anchor/positive swap**: deterministic 50% coin-flip swaps anchor and positive slots at triplet finalization, so both orderings appear at equal frequency. Important for InfoNCE and other contrastive objectives where asymmetric slot distributions would otherwise provide a shortcut. Seeded by sampler RNG; fully reproducible and covered by state-persistence mechanics.
- **Anti-shortcut metadata-prefix variation** via `KvpPrefixSampler` (variant choice, per-field presence probabilities, field-order shuffle, and prefix dropout).
- **Per-source batch mixing controls** with independent source frequency controls (including over/under-sampling).
- **Per-source trust controls** to weight quality/trust independently by source/taxonomy.
- **Per-batch dynamic source reweighting** so source weights can be changed across batches (for example from loss/metric feedback) while training.
- **Resume support** via `save_sampler_state(save_to)` and split-store persistence.
- **Source-agnostic backends** (`DataSource` or `IndexableSource` + `IndexableAdapter`).
- **Supply-chain style orchestration**: multi-source intake (`refresh`) with per-call parallel ingest, optional per-source weighting, staged buffering, deterministic split routing, and batch assembly into train-ready outputs.
- **Bounded ingestion** windows instead of loading full corpora into memory.
- **Per-call source threading**: during refresh, each source is fetched on its own short-lived thread, then merged deterministically for batch assembly.
- **Background batch prefetching** via `BatchPrefetcher`: spawns a dedicated background thread that drives a tight production loop, pushing results into a bounded channel queue. The training loop blocks only on `next()`. Within each batch call the background thread makes, the sampler fans out to per-source threads for ingestion — both concurrency layers are active simultaneously. GPU-side throughput depends on queue depth and the number of workers feeding it; the prefetcher queue size should be tuned so the GPU never drains the buffer between steps (see `prefetch_triplet_batches(split, queue_depth)`).
- **Streaming-friendly**: sources can be finite or unbounded.

## Sampling behavior (current)

This reflects the built-in file-corpus helpers (`FileCorpusIndex`) used by filesystem-backed sources.

- **Ingestion**: `next_triplet_batch(split)`, `next_pair_batch(split)`, and `next_text_batch(split)` trigger refresh; per-source buffers refill when empty (or on force refresh).
- **Memory bound**: refresh/cache limits are bounded by `ingestion_max_records` with a floor at `batch_size`.
- **Per-source cache**: each registered source keeps its own independent LRU window of `ingestion_max_records` records. The total in-memory record pool across N sources is therefore N × `ingestion_max_records`. This means every source always has the full candidate budget available — candidate availability does not shrink as more sources are added, as it would with a single shared pool divided N ways.
- **`ingestion_max_records` tuning**: setting this above `batch_size` usually improves sample diversity (broader anchor/negative candidate pool) and reduces near-term repetition, but returns diminish once source availability, split boundaries, and recipe constraints dominate. For remote backends such as Hugging Face, larger initial ingestion targets can require pulling more initial shards before the first batch, so startup latency can increase depending on shard sizes and network throughput.
- **File indexing**: deterministic path ordering + deterministic index permutation for paging.
- **Source ordering**: round-robin by source, deterministic within-source ordering by seed/epoch (file source). For `HuggingFaceRowSource`, shard download order is deterministic by seed, but row selection within a refresh is not stable across cache wipes (see `HuggingFaceRowSource` doc).
- **Splits**: labels are deterministic from `record_id + seed + ratios`; split APIs enforce `allowed_splits`.
- **Coverage caveat**: if `len_hint` drifts mid-epoch in streaming backends, strict single-pass coverage is not guaranteed.
- **Weights**: recipe/source/chunk weights affect scaling, not deterministic ordering.
- **Scale note**: full scan/sort/index rebuild cost grows roughly linearly with file count and path bytes. This is intentional and appropriate for the target corpus scale (thousands to low millions of files). For LLM-scale pre-training across billions of tokens, the right tool is a format designed for that workload (Arrow/Parquet, WebDataset shards, or a dedicated high-throughput dataloader) — this crate targets iterative fine-tuning on domain-specific corpora, not bulk pre-training infrastructure.
- **Order note**: index batching preserves permutation order; chunked index reads do not remove deterministic shuffling.
- **Manual epoch control**: `sampler.set_epoch(n)` resets per-source cursors and reshuffles deterministically for that epoch.
- **Persisted state scope**: epoch tracking is split-aware, but sampler/source cursors + RNG/round-robin state are persisted per store file.
- **Triplet recipe behavior**: if `SamplerConfig.recipes` is non-empty, those recipes are used for all sources; otherwise each source's `default_triplet_recipes()` is used (if any).
- **Optional BM25 hard negatives**: with feature `bm25-mining`, a single global BM25 index is rebuilt on the main thread after each source refresh cycle (not on every drain-only ingest step; source I/O fetches run in parallel threads but the index build always happens on the caller's thread after those fetches join). The index covers the entire N × `ingestion_max_records` record pool — every candidate in every per-source cache contributes, with no cap on index size. Records are stored once in the primary record map; the BM25 index holds only lightweight per-record metadata (`record_id`, `source`, `split`, `date`) and tokenized text for ranking — no `DataRecord` copies. During sampling, BM25-ranked candidates are filtered to the strategy-selected pool, same-split constraints are enforced, and candidate selection rotates deterministically per anchor to improve diversity. BM25 shifts the pool toward lexically harder negatives; it does not replace the diversity-first baseline — the output mix remains varied (hard, medium, and soft), not exclusively hardest-first. BM25 is a keyword-overlap ranker and is well-suited as an inexpensive first pass for negative hardness; for semantic re-ranking (dense retrieval, cross-encoder scoring, iterative mining with the trained encoder), those are out of scope for the data pipeline and integrate via pre-ranked source construction or by reweighting source batches in the training loop.
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

## Advanced source examples

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

See [LICENSE-APACHE][apache-2.0-license-page] and [LICENSE-MIT][mit-license-page] for details.

[rust-src-page]: https://www.rust-lang.org/
[rust-logo]: https://img.shields.io/badge/Made%20with-Rust-black

[crates-page]: https://crates.io/crates/triplets
[crates-badge]: https://img.shields.io/crates/v/triplets.svg

[mit-license-page]: https://raw.githubusercontent.com/jzombie/rust-triplets/refs/heads/main/LICENSE-MIT
[mit-license-badge]: https://img.shields.io/badge/license-MIT-blue.svg

[apache-2.0-license-page]: https://raw.githubusercontent.com/jzombie/rust-triplets/refs/heads/main/LICENSE-APACHE
[apache-2.0-license-badge]: https://img.shields.io/badge/license-Apache%202.0-blue.svg

[coveralls-page]: https://coveralls.io/github/jzombie/rust-triplets?branch=main
[coveralls-badge]: https://img.shields.io/coveralls/github/jzombie/rust-triplets
