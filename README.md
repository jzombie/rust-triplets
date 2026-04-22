<p align="center">
  <h1 align="center">⛏️ triplets</h1>
  <p align="center"><strong>Composable data sampling primitives for deterministic multi-source ML/AI training-data orchestration.</strong></p>
  <p align="center">
    <a href="#getting-started">Getting Started</a> &middot;
    <a href="#cargo-features">Cargo Features</a> &middot;
    <a href="#configuring-sources">Sources</a> &middot;
    <a href="#sampling-and-mixing">Sampling &amp; Mixing</a> &middot;
    <a href="#epochs-and-determinism">Epochs</a> &middot;
    <a href="#license">License</a>
  </p>
  <p align="center">
    <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Made%20with-Rust-black" alt="Made with Rust"></a>
    <a href="https://crates.io/crates/triplets"><img src="https://img.shields.io/crates/v/triplets.svg" alt="crates.io"></a>
    <a href="https://github.com/jzombie/rust-triplets/blob/main/LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT licensed"></a>
    <a href="https://github.com/jzombie/rust-triplets/blob/main/LICENSE-APACHE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="Apache 2.0 licensed"></a>
    <a href="https://coveralls.io/github/jzombie/rust-triplets?branch=main"><img src="https://coveralls.io/repos/github/jzombie/rust-triplets/badge.svg?branch=main" alt="Coverage Status"></a>
    <br><sub><em>Tested on macOS, Linux, and Windows.</em></sub>
  </p>
</p>

---

Generate an effectively unlimited stream of [training triplets](https://en.wikipedia.org/wiki/Triplet_loss), pairs, or plaintext samples from your existing corpus. This crate handles ingestion, multi-source mixing, deterministic train/validation/test splitting, and optional [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) hard-negative mining.

**Designed as a data-pipeline layer for a training loop.**

> A training loop has two halves: the *data side* and the *model side*. `triplets` owns the data side — deterministic and reproducible train/validation/test splitting, seeded shuffling across epochs, weighted multi-source mixing, BM25 hard-negative mining, and static per-record KVP metadata for input conditioning. What it intentionally does *not* include is the model side: forward passes, loss computation, and optimizer steps. The design goal is that you plug this crate's output stream directly into your training framework (crates like [Candle](https://github.com/huggingface/candle), [burn](https://crates.io/crates/burn), [tch](https://crates.io/crates/tch), [PyO3](https://crates.io/crates/pyo3)) and it already handles the parts of the data pipeline that are hardest to get right — correctness, reproducibility, and scale.

**Work in progress.**

## Overview

In metric learning and language model training, a **triplet** consists of an **anchor**, a **positive** example (similar to the anchor), and a **negative** example (dissimilar to the anchor).

`triplets` provides a high-throughput streaming pipeline to:
1. **Ingest** data from local text/CSV files, Hugging Face, or custom backends.
2. **Mix** sources with configurable weights to balance your training data.
3. **Split** data deterministically into train, validation, and test sets.
4. **Sample** triplets or pairs using rule-based "recipes".
5. **Mine** hard negatives using BM25 to improve model discrimination.

```text
      Anchor
      /    \
 Positive Negative

 Triplet: (Anchor, Positive, Negative)
```

## Getting Started

A `TripletSampler` needs a `SplitStore` for record-to-split assignments and a `SamplerConfig` for runtime behavior.

```rust,no_run
use std::sync::Arc;
use triplets::{
    BatchPrefetcher, SamplerConfig, TripletSampler, TripletBatch,
    SplitRatios, DeterministicSplitStore, SplitLabel,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Define your train/validation/test ratios (e.g., 80/10/10).
    let ratios = SplitRatios { train: 0.8, validation: 0.1, test: 0.1 };

    // 2. Initialize a deterministic split store.
    // The seed ensures record IDs are always assigned to the same split.
    let seed = 42;
    let store = Arc::new(DeterministicSplitStore::new(ratios, seed)?);

    // 3. Create the sampler wrapped in Arc — required for prefetching.
    let sampler = Arc::new(TripletSampler::new(SamplerConfig::default(), store));

    // 4. Register one or more sources (CSV, text files, Hugging Face, or custom).
    //    See the [Configuring Sources](#configuring-sources) section for full examples.
    //    sampler.register_source(Box::new(my_source));

    // 5. Spawn a background prefetcher with a queue depth of 4.
    //    The worker thread starts filling the queue immediately; your training
    //    loop calls prefetcher.next() and blocks only when the queue is empty.
    let prefetcher: BatchPrefetcher<TripletBatch> =
        Arc::clone(&sampler).prefetch_triplet_batches(SplitLabel::Train, 4);

    // 6. Pull batches in your training loop.
    for _step in 0..10 {
        let batch = prefetcher.next()?;
        for triplet in batch.triplets {
            println!("anchor:   {}", triplet.anchor.text);
            println!("positive: {}", triplet.positive.text);
            println!("negative: {}", triplet.negative.text);
        }
    }
    // The prefetcher's background thread shuts down automatically when dropped.

    Ok(())
}
```

## Cargo Features

| Feature            | What it enables                                                               | Default |
|--------------------|-------------------------------------------------------------------------------|---------|
| `huggingface`      | [Streaming from Hugging Face dataset repositories.](#hugging-face-source)     | No      |
| `bm25-mining`      | [BM25 hard-negative ranking within strategy-defined pools.](#negative-mining) | No      |
| `extended-metrics` | Additional per-triplet diagnostics for debugging.                             | No      |

> _[CSV](#csv-source), [text file](#text-file-source), and [custom source](#custom-source) support are enabled in all builds._

## Configuring Sources

### Hugging Face Source

Streams rows directly from the Hugging Face Hub without requiring a full dataset download. Map dataset columns to anchor, positive, or plain-text roles the same way as the CSV source.

```rust,no_run
#[cfg(feature = "huggingface")]
{
    use std::sync::Arc;
    use triplets::{SamplerConfig, TripletSampler, SplitRatios, DeterministicSplitStore, Sampler};
    use triplets::{HuggingFaceRowSource, HuggingFaceRowsConfig};

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        let ratios = SplitRatios { train: 0.8, validation: 0.1, test: 0.1 };
        let store = Arc::new(DeterministicSplitStore::new(ratios, 42)?);
        let mut sampler = TripletSampler::new(SamplerConfig::default(), store);
        // Configure the source to pull the "train" split of a dataset.
        // Note: While we specify "train" here as the ingestion source, the crate
        // automatically handles its own deterministic split assignments (train/val/test)
        // at the record level across all loaded data.
        let config = HuggingFaceRowsConfig::new(
            "hf_finance",          // Source identifier
            "financial_phrasebank", // HF Dataset name
            "default",             // Dataset config
            "train",               // Dataset split
            "cache/hf_snapshots"   // Local cache for downloaded shards
        );

        let source = HuggingFaceRowSource::new(config)?;
        sampler.register_source(Box::new(source));
        Ok(())
    }
}
```

#### Column Mapping Modes

The HF source supports two exclusive extraction modes, selected by which fields are populated on `HuggingFaceRowsConfig`:

**Role mode** — activated when `anchor_columns`, `positive_columns`, or `context_columns` is non-empty. Each row produces a `DataRecord` with explicitly assigned section roles:

| Config field       | Coalesces? | `SectionRole` produced          | Behaviour when missing / empty                   |
|--------------------|------------|---------------------------------|--------------------------------------------------|
| `anchor_columns`   | Yes        | `Anchor`                        | Row is skipped                                   |
| `positive_columns` | Yes        | `Context`                       | Row is skipped                                   |
| `context_columns`  | No         | `Context` (one section per col) | Row is skipped if **any** column is absent/blank |

*Coalescing* means multiple candidate column names can be supplied; the first with a non-empty value is used and the rest are ignored. `context_columns` does **not** coalesce — every listed column is strictly required and each contributes its own independent section.

**Text mode** — used when `anchor_columns` is empty and `text_columns` is non-empty. The first non-empty candidate column supplies the sole content for the row. This is the SimCSE-style path where the model learns from augmented views of the same text.

##### Role mode: three-column datasets (question / answer / context)

Datasets that pair a question with both an answer and a passage of supporting context — common in RAG evaluation sets — can be ingested with a single source-list line:

```text
# in hf_sources.txt
hf://zeitgeist-ai/financial-rag-nvidia-sec/default/train anchor=question positive=answer context=context
```

Or programmatically via `context_columns`:

```rust,no_run
#[cfg(feature = "huggingface")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use triplets::{HuggingFaceRowSource, HuggingFaceRowsConfig};

    let mut config = HuggingFaceRowsConfig::new(
        "hf_fin_rag",
        "zeitgeist-ai/financial-rag-nvidia-sec",
        "default",
        "train",
        "cache/hf_snapshots",
    );
    config.anchor_columns   = vec!["question".to_string()];
    config.positive_columns = vec!["answer".to_string()];
    config.context_columns  = vec!["context".to_string()];

    let source = HuggingFaceRowSource::new(config)?;
    let _ = source;
    Ok(())
}

#[cfg(not(feature = "huggingface"))]
fn main() {}
```

Each ingested row produces a `DataRecord` with three sections in declaration order:

| Section | Source column | `SectionRole` |
|---------|---------------|---------------|
| 0       | `question`    | `Anchor`      |
| 1       | `answer`      | `Context`     |
| 2       | `context`     | `Context`     |

Because both the positive column and every context column are emitted as `SectionRole::Context` sections, a recipe using `Selector::Role(SectionRole::Context)` will see all of them as candidates.

> **Row-skipping**: if any column listed in `context_columns` is absent from a row or contains an empty string, that row is silently dropped. This hard requirement prevents partially-populated rows from appearing in training batches. `anchor_columns` and `positive_columns` behave the same way — a row is skipped if the coalesced result is empty.

Multiple context columns are supported and each produces its own section, in the order they are declared:

```text
hf://my-org/my-dataset/default/train anchor=title positive=summary context=body,tags
```

#### Source-list file format

When using `build_hf_sources` / `load_hf_sources_from_list`, sources are described one per line in a plain-text file. Lines starting with `#` are comments; blank lines are ignored.

```text
hf://<org>/<dataset>/<config>/<split>  key=value  [key=value ...]
```

Every accepted key and its semantics:

| Key                       | Value                       | Accepts commas? | Required?                                                              | Description                                                                                                                                                                              |
|---------------------------|-----------------------------|-----------------|------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `anchor=`                 | one or more column names    | Yes             | At least one of `anchor`, `positive`, `context`, or `text` is required | Activates role mode. Columns are tried in order; the first non-empty value is used as the `Anchor` section. Row skipped if all candidates are absent/empty.                              |
| `positive=`               | one or more column names    | Yes             | No                                                                     | Activates role mode. Columns are tried in order; the first non-empty value becomes a `Context` section. Row skipped if all candidates are absent/empty.                                  |
| `context=`                | one or more column names    | Yes             | No                                                                     | Activates role mode. Every listed column is required — if any is absent or blank the row is dropped. Each column becomes its own `Context` section, in declaration order. No coalescing. |
| `text=` / `text_columns=` | one or more column names    | Yes             | At least one mapping key is required                                   | Activates text mode (SimCSE). Columns are tried in order; the first non-empty value is the sole content of the record. Ignored when role mode is active. Both spellings are equivalent.  |
| `trust=`                  | float in `[0.0, 1.0]`       | No              | No (default: `0.5`)                                                    | Overrides the quality trust score stamped on every record produced by this source. Out-of-range values or non-float strings are hard errors at parse time.                               |
| `source_id=`              | non-empty identifier string | No              | No (auto-derived when absent)                                          | Overrides the automatically generated source identifier. Must not be empty.                                                                                                              |

**Auto-derived `source_id`**

When `source_id=` is omitted, an identifier is derived from the URI:

1. The short dataset name (the part after the last `/` in the org/dataset pair) is taken as the base.
2. If the config is not `"default"`, it is appended as `.config`.
3. If the split is not `"train"`, it is appended as `.split`.
4. Special characters are sanitized to underscores.
5. If two sources produce the same auto-slug, `.{index}` is appended to the second and subsequent collisions.

Examples: `hf://org/wikipedia/20231101.en/train` → `wikipedia.20231101_en`; `hf://org/dataset/default/validation` → `dataset.validation`.

**Error behaviour**

Unknown keys (including typos such as `positve=`) are hard errors — the parser rejects the line immediately rather than silently ignoring the key. This prevents misconfigured sources from being silently loaded with missing column mappings. A line with no recognised mapping key (`anchor=`, `positive=`, `context=`, or `text=`) is also rejected.

#### Authenticating with Private Datasets

To access private or gated datasets set the `HF_TOKEN` environment variable to a valid
Hugging Face API token. Tokens with at least **read** scope are sufficient and can be
generated at <https://huggingface.co/settings/tokens>.

When `HF_TOKEN` is set to a non-empty value, `HuggingFaceRowsConfig::new()` picks it up
automatically and sends it as a `Bearer` credential on every API request and shard
download. If the token is invalid or expired, `HuggingFaceRowSource::new()` returns an
error immediately rather than silently degrading later.

| Platform                 | Command                                                |
|--------------------------|--------------------------------------------------------|
| macOS / Linux            | `export HF_TOKEN="hf_..."`                             |
| Windows — Command Prompt | `set HF_TOKEN=hf_...`                                  |
| Windows — PowerShell     | `$env:HF_TOKEN = "hf_..."`                             |
| Windows — persistent     | *System Properties → Advanced → Environment Variables* |

The token can also be set programmatically on the config struct if you prefer not to rely on
the process environment:

```rust,no_run
#[cfg(feature = "huggingface")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use triplets::{HuggingFaceRowSource, HuggingFaceRowsConfig};

    let mut config = HuggingFaceRowsConfig::new(
        "private_dataset",
        "my-org/private-dataset",
        "default",
        "train",
        "cache/hf_snapshots",
    );
    // Override after construction (or set HF_TOKEN env var before calling new()).
    config.hf_token = Some("hf_...".to_string());
    // new() validates the token immediately; an invalid token returns an error.
    let source = HuggingFaceRowSource::new(config)?;
    let _ = source;
    Ok(())
}

#[cfg(not(feature = "huggingface"))]
fn main() {}
```

> **Security**: never commit tokens to source control. Use environment variables, a secrets
> manager, or a credential file listed in `.gitignore`.

### CSV Source

Load rows from a CSV file with explicit column mappings. The file **must have a named header row** — columns are always selected by name. Supports two modes:

- **Role mode** — map separate columns to anchor and positive (context) roles.
- **Text mode** — map a single column for SimCSE-style contrastive pre-training.

```rust,no_run
use std::sync::Arc;
use triplets::{SamplerConfig, TripletSampler, SplitRatios, DeterministicSplitStore};
use triplets::source::{CsvSource, CsvSourceConfig};

let ratios = SplitRatios { train: 0.8, validation: 0.1, test: 0.1 };
let store = Arc::new(DeterministicSplitStore::new(ratios, 42).unwrap());
let mut sampler = TripletSampler::new(SamplerConfig::default(), store);

// Role mode: map "question" → anchor, "answer" → positive.
let config = CsvSourceConfig::new("qna", "data/qna.csv")
    .with_anchor_column("question")
    .with_positive_column("answer")
    .with_trust(0.9);
let source = CsvSource::new(config).unwrap();
sampler.register_source(Box::new(source));

// Text mode (SimCSE): single column used for both anchor and context.
let config2 = CsvSourceConfig::new("corpus", "data/corpus.csv")
    .with_text_column("text");
let source2 = CsvSource::new(config2).unwrap();
sampler.register_source(Box::new(source2));
```

Rows with empty required fields are skipped. Column name matching is case-insensitive.

### Text File Source

Recursively indexes plain-text files from a directory. Each file's stem (filename without extension) becomes the **anchor** and its body content becomes the **context**. Useful for local corpora where files are already titled meaningfully.

```rust
use std::sync::Arc;
use triplets::{SamplerConfig, TripletSampler, SplitRatios, DeterministicSplitStore};
use triplets::source::{FileSource, FileSourceConfig};

let ratios = SplitRatios { train: 0.8, validation: 0.1, test: 0.1 };
let store = Arc::new(DeterministicSplitStore::new(ratios, 42).unwrap());
let mut sampler = TripletSampler::new(SamplerConfig::default(), store);
// Point at a directory; all text files are indexed recursively.
// The filename stem is the anchor; the file body is the context.
let config = FileSourceConfig::new("docs", "./data/corpus")
    .with_text_files_only(true)
    .with_trust(0.9); // Assign a quality score to this source

let source = FileSource::new(config);
sampler.register_source(Box::new(source));
```

### Custom Source

Implement the `IndexableSource` trait to integrate any backend that can fetch records by a stable integer index.

```rust
use std::sync::Arc;
use triplets::{SamplerConfig, TripletSampler, SplitRatios, DeterministicSplitStore};
use chrono::Utc;
use triplets::{DataRecord, SamplerError};
use triplets::data::{RecordSection, SectionRole};
use triplets::source::{IndexableSource, IndexableAdapter};

struct MyApiSource;

impl IndexableSource for MyApiSource {
    fn id(&self) -> &str { "api_source" }
    fn len_hint(&self) -> Option<usize> { Some(1000) }
    fn record_at(&self, idx: usize) -> Result<Option<DataRecord>, SamplerError> {
        // Fetch record 'idx' from your database or API.
        // Return Ok(None) to skip a record (e.g. deleted rows or filtered entries).
        Ok(Some(DataRecord {
            id: format!("api_{idx}"),
            source: self.id().into(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            quality: Default::default(),
            // Optional free-form tags for filtering or recipe targeting.
            // Examples: domain labels, year strings, content-type markers.
            taxonomy: vec!["finance".into(), "2025".into()],
            // Each section represents one logical view of the record's content.
            // SectionRole::Anchor  — the primary subject text (e.g. a question, title, or key passage).
            // SectionRole::Context — supporting or related text (e.g. an answer, body, or description).
            // Recipes select sections by role: Selector::Role(SectionRole::Anchor / Context).
            //
            // `sentences` is an optional pre-split list of individual sentences within `text`.
            // Providing it gives the chunker more accurate boundaries when creating token windows.
            // Leave it as vec![] and the chunker will split `text` automatically.
            sections: vec![
                RecordSection {
                    role: SectionRole::Anchor,
                    heading: Some("Title".into()),
                    text: format!("Primary content for record {idx}."),
                    sentences: vec![], // or: vec!["Sentence one.".into(), "Sentence two.".into()]
                },
                RecordSection {
                    role: SectionRole::Context,
                    heading: None,
                    text: format!("Supporting context for record {idx}."),
                    sentences: vec![],
                },
            ],
            // Optional: attach a KvpPrefixSampler to inject structured key-value
            // metadata into sampled chunk text at training time. For example:
            //
            //   meta: source=api | date=2025-01-01
            //   <actual chunk text>
            //
            // The sampler controls dropout (how often the prefix appears) and
            // per-field presence probability, so the model learns to handle both
            // prefixed and plain chunks. See the "Metadata Prefixes and Tag Dropout"
            // section for full usage.
            meta_prefix: None,
        }))
    }
}

let ratios = SplitRatios { train: 0.8, validation: 0.1, test: 0.1 };
let store = Arc::new(DeterministicSplitStore::new(ratios, 42).unwrap());
let mut sampler = TripletSampler::new(SamplerConfig::default(), store);
let adapter = IndexableAdapter::new(MyApiSource);
sampler.register_source(Box::new(adapter));
```

## Sampling and Mixing

### Weighted Sampling

Adjust per-source sampling frequency to handle class imbalance or dataset quality differences.

```rust,no_run
use std::sync::Arc;
use std::collections::HashMap;
use triplets::{SamplerConfig, TripletSampler, SplitRatios, DeterministicSplitStore, SplitLabel, Sampler};
use triplets::source::{CsvSource, CsvSourceConfig, FileSource, FileSourceConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ratios = SplitRatios { train: 0.8, validation: 0.1, test: 0.1 };
    let store = Arc::new(DeterministicSplitStore::new(ratios, 42)?);
    let mut sampler = TripletSampler::new(SamplerConfig::default(), store);

    // Source 1: structured Q&A pairs from a CSV file.
    // Each row maps a "question" column → anchor, "answer" column → positive.
    let csv_config = CsvSourceConfig::new("hf_finance", "data/finance_qa.csv")
        .with_anchor_column("question")
        .with_positive_column("answer")
        .with_trust(0.9);
    sampler.register_source(Box::new(CsvSource::new(csv_config)?));

    // Source 2: local plain-text corpus of internal documentation.
    // Files are indexed recursively; filename stem → anchor, body → context.
    let file_config = FileSourceConfig::new("docs", "./data/internal_docs")
        .with_text_files_only(true)
        .with_trust(0.7); // lower trust — unreviewed internal docs
    sampler.register_source(Box::new(FileSource::new(file_config)));

    // Override the mixing ratio for this batch: pull from the high-quality
    // CSV source 70% of the time and the local docs 30% of the time.
    // Sources not listed here fall back to uniform sampling.
    let mut weights = HashMap::new();
    weights.insert("hf_finance".to_string(), 0.7);
    weights.insert("docs".to_string(), 0.3);

    let batch = sampler.next_triplet_batch_with_weights(SplitLabel::Train, &weights)?;
    Ok(())
}
```

### Recipe Selection Weights

The `weight` field on `TripletRecipe` controls **how often a recipe is selected** relative to other active recipes. The sampler expands each recipe into a proportional number of selection slots, shuffles them, and cycles through — so a recipe with `weight = 3.0` is drawn approximately three times as often as one with `weight = 1.0`.

| `weight` value                            | Effect                                                                                                  |
|-------------------------------------------|---------------------------------------------------------------------------------------------------------|
| Equal across all recipes (e.g. all `1.0`) | Uniform round-robin — each recipe is selected equally often (default behavior).                         |
| `2.0` vs `1.0`                            | The `2.0` recipe is tried ~2× as often per batch.                                                       |
| `0.0` or negative                         | Recipe is **excluded entirely** — useful for disabling a recipe without removing it from configuration. |

```rust,no_run
use triplets::{SamplerConfig, TripletRecipe, NegativeStrategy, Selector, SectionRole};

let config = SamplerConfig {
    recipes: vec![
        // High-signal structured pairs: tried 3× as often as the fallback.
        TripletRecipe {
            name: "structured".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Random,
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 3.0,
            instruction: None, // See the Instruction Tuning section to attach a task prompt.
            allow_same_anchor_positive: false,
        },
        // Fallback recipe with random chunk selection.
        TripletRecipe {
            name: "random_fallback".into(),
            anchor: Selector::Random,
            positive_selector: Selector::Random,
            negative_selector: Selector::Random,
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
            allow_same_anchor_positive: false,
        },
        // Disabled recipe — excluded from sampling until weight is set above zero.
        TripletRecipe {
            name: "experimental".into(),
            anchor: Selector::Random,
            positive_selector: Selector::Random,
            negative_selector: Selector::Random,
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 0.0,
            instruction: None,
            allow_same_anchor_positive: false,
        },
    ],
    ..SamplerConfig::default()
};
```

> **Sampling frequency vs. output score**: `TripletRecipe::weight` controls how often the recipe is *selected*. It is also one factor in the output `SampleTriplet::weight`, but the two serve different roles — see [Output Format](#output-format) below.

### Instruction Tuning

The `instruction` field on `TripletRecipe` attaches a static task prompt to every triplet, pair, or text sample produced by that recipe. It is copied verbatim into `SampleTriplet::instruction` (and the equivalent field on `SamplePair` / `TextSample`) so your training loop can prepend it to the anchor text before passing it to the model.

This lets different recipes express different task hypotheses over the same underlying data — for example, a retrieval recipe and a similarity recipe can share the same source but carry different prompts:

```rust,no_run
use triplets::{SamplerConfig, TripletRecipe, NegativeStrategy, Selector, SectionRole};

let config = SamplerConfig {
    recipes: vec![
        // Retrieval recipe: every triplet from this recipe carries a task prompt.
        TripletRecipe {
            name: "retrieval".into(),
            anchor: Selector::Role(SectionRole::Anchor),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Random,
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: Some("Retrieve a passage that answers the question:".into()),
            allow_same_anchor_positive: false,
        },
        // Plain contrastive recipe: no prompt — model sees bare chunk text.
        TripletRecipe {
            name: "similarity".into(),
            anchor: Selector::Role(SectionRole::Context),
            positive_selector: Selector::Role(SectionRole::Context),
            negative_selector: Selector::Random,
            negative_strategy: NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
            allow_same_anchor_positive: false,
        },
    ],
    ..SamplerConfig::default()
};
```

In your training loop, prepend the instruction to the anchor when present:

```rust,no_run
use std::sync::Arc;
use triplets::{SamplerConfig, TripletSampler, SplitRatios, DeterministicSplitStore, SplitLabel, Sampler};
let ratios = SplitRatios { train: 0.8, validation: 0.1, test: 0.1 };
let store = Arc::new(DeterministicSplitStore::new(ratios, 42).unwrap());
let mut sampler = TripletSampler::new(SamplerConfig::default(), store);
let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
for triplet in batch.triplets {
    // Prepend the task instruction to the anchor when the recipe specifies one.
    // Recipes without an instruction pass the anchor text through unchanged.
    //
    // With instruction:    "Retrieve a passage that answers the question:\nWhat is X?"
    // Without instruction: "What is X?"
    let anchor_input = match &triplet.instruction {
        Some(instr) => format!("{instr}\n{}", triplet.anchor.text),
        None => triplet.anchor.text.clone(),
    };

    // The positive and negative slots are never prefixed with the instruction —
    // only the anchor carries the task prompt.
    let positive_input = triplet.positive.text.clone();
    let negative_input = triplet.negative.text.clone();

    // Pass all three to your model's embedding function and compute triplet loss.
    // let loss = model.triplet_loss(&anchor_input, &positive_input, &negative_input);
}
```

### Output Format

Each `SampleTriplet` contains the sampled text and a computed training score.

```rust,no_run
use std::sync::Arc;
use triplets::{SamplerConfig, TripletSampler, SplitRatios, DeterministicSplitStore, SplitLabel, Sampler};
let ratios = SplitRatios { train: 0.8, validation: 0.1, test: 0.1 };
let store = Arc::new(DeterministicSplitStore::new(ratios, 42).unwrap());
let mut sampler = TripletSampler::new(SamplerConfig::default(), store);
let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
for triplet in batch.triplets {
    // Primary content
    let anchor_text = &triplet.anchor.text;
    let pos_text    = &triplet.positive.text;
    let neg_text    = &triplet.negative.text;

    // Metadata
    let recipe      = &triplet.recipe;      // which recipe produced this triplet
    let weight      = triplet.weight;       // training score — see below
    let instruction = triplet.instruction;  // task prompt set on the recipe, if any — see Instruction Tuning
}
```

#### What `triplet.weight` means and how it is calculated

`SampleTriplet::weight` is a **per-triplet training score** in the range `(0.0, recipe.weight]`. Use it to scale each triplet's contribution to the loss — triplets that are more structurally coherent or come from higher-trust sources receive a higher score.

The value is computed as `triplet.weight = recipe.weight × chunk_quality`, where `chunk_quality` is the average of three per-slot signals (one per chunk: anchor, positive, negative). Each signal is the product of two independent factors:

| Factor                    | What it measures                                                                                                          | How it is set                                    |
|---------------------------|---------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------|
| **Window position score** | `1 / (window_index + 1)` — earlier chunks in a section score higher (1.0 at index 0, 0.5 at index 1, 0.25 at index 3, …). | Automatic.                                       |
| **Source trust**          | Configured quality signal for the originating source (clamped to `[0, 1]`).                                               | Set via `.with_trust(0.9)` on the source config. |

The resulting raw signal is clamped to `[chunk_weight_floor, 1.0]` (default floor: `0.1`) before averaging.

The anchor/positive pair additionally has a **proximity multiplier** applied: chunks that are closer together within the same section receive a higher multiplier (two adjacent windows score 1.0; the score decreases as window distance grows). This rewards pairs that share local context.

A practical reading: a triplet from a high-trust source where all three chunks come from the opening windows of their sections will have `chunk_quality ≈ 1.0`, so `triplet.weight ≈ recipe.weight`. A triplet with chunks deep in long documents from a lower-trust source will have a noticeably smaller score.

In a training loop pass the weight straight into your criterion:

```rust,no_run
use std::sync::Arc;
use triplets::{SamplerConfig, TripletSampler, SplitRatios, DeterministicSplitStore, SplitLabel, Sampler};
let ratios = SplitRatios { train: 0.8, validation: 0.1, test: 0.1 };
let store = Arc::new(DeterministicSplitStore::new(ratios, 42).unwrap());
let mut sampler = TripletSampler::new(SamplerConfig::default(), store);
let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
// Example: accumulate weighted loss over a batch.
let _weighted_loss: f32 = batch.triplets.iter().map(|t| {
    let triplet_loss = 0.0_f32; // replace with your model's per-triplet loss
    triplet_loss * t.weight
}).sum();
```

### Source Within a Source

Each `TripletRecipe` is an **independent code path** over the sections of a record. Two recipes registered against the same source can express completely different training hypotheses about the same underlying data — no second source registration needed.

The mechanism is straightforward:

- Populate each `DataRecord::sections` with as many `RecordSection` entries as your data has natural views.
- Assign each section a `SectionRole` (or let position carry the meaning with `Selector::Paragraph(n)`).
- Write one `TripletRecipe` per hypothesis; each recipe independently specifies which sections fill the anchor, positive, and negative slots.
- Sources declare their own recipes via `default_triplet_recipes()` so callers need no recipe configuration at all.

**Sparse sections — optional data in the same record pool**

Not every record needs to have all sections. If a recipe targets `Selector::Paragraph(2)` (the third section) and a record only has two sections, the sampler simply skips that record *for that recipe only* — the record continues to serve all other recipes normally. This lets you mix densely-covered and sparsely-covered training hypotheses in a single source without any record filtering logic in your data pipeline.

**Example — financial data source with two recipe strategies**

Imagine each record represents one publicly-traded company with up to three sections:

| Index | Role           | Content                                                       | Always present?                       |
|-------|----------------|---------------------------------------------------------------|---------------------------------------|
| 0     | `Anchor`       | Linearized financial metrics — view A (a random tag subset)   | Yes                                   |
| 1     | `Context`      | Linearized financial metrics — view B (a disjoint tag subset) | Yes                                   |
| 2     | *(positional)* | Earnings-call transcript for the same period                  | No — only when a transcript was found |

Two recipes target different aspects of the same records:

```rust,no_run
use triplets::config::{NegativeStrategy, Selector, TripletRecipe};
use triplets::data::SectionRole;

/// Cross-view recipe: both metric views are always present, so every record
/// participates. Teaches the model that two different linearized views of the
/// same company are semantically closer than any view of a different company.
fn metrics_cross_view_recipe() -> TripletRecipe {
    TripletRecipe {
        name: "metrics_cross_view".into(),
        // Anchor: metric view A.
        anchor: Selector::Role(SectionRole::Anchor),
        // Positive: metric view B — disjoint tags, same company and period.
        positive_selector: Selector::Role(SectionRole::Context),
        // Negative: metric view A of a different company.
        negative_selector: Selector::Role(SectionRole::Anchor),
        negative_strategy: NegativeStrategy::WrongArticle,
        weight: 1.0,
        instruction: None,
        allow_same_anchor_positive: false,
    }
}

/// Transcript recipe: targets an optional third section (index 2).
/// Records without a transcript are skipped for *this recipe only* —
/// they still serve the metrics_cross_view recipe above without any
/// record filtering logic in the data pipeline.
///
/// Lower weight reflects partial coverage: fewer records satisfy this
/// recipe, so letting it drive the same number of gradient steps as the
/// dense recipe would over-represent the companies with transcripts.
fn metrics_to_transcript_recipe() -> TripletRecipe {
    TripletRecipe {
        name: "metrics_to_transcript".into(),
        // Anchor: metric view A.
        anchor: Selector::Role(SectionRole::Anchor),
        // Positive: earnings-call transcript at section index 2.
        // Records that lack this section are skipped for this recipe.
        positive_selector: Selector::Paragraph(2),
        // Negative: metric view A of a different company.
        negative_selector: Selector::Role(SectionRole::Anchor),
        negative_strategy: NegativeStrategy::WrongArticle,
        // Half the weight of the dense recipe; adjust as transcript coverage grows.
        weight: 0.5,
        instruction: None,
        allow_same_anchor_positive: false,
    }
}
```

The source returns both recipes from `default_triplet_recipes()` so that no recipe configuration is needed at the call site:

```rust,no_run
use triplets::config::TripletRecipe;
use triplets::source::{DataSource, IndexablePager, IndexableSource, SourceCursor, SourceSnapshot};
use triplets::{DataRecord, SamplerConfig, SamplerError};

# use triplets::config::{NegativeStrategy, Selector};
# use triplets::data::SectionRole;
# fn metrics_cross_view_recipe() -> TripletRecipe { TripletRecipe { name: "".into(), anchor: Selector::Random, positive_selector: Selector::Random, negative_selector: Selector::Random, negative_strategy: NegativeStrategy::WrongArticle, weight: 1.0, instruction: None, allow_same_anchor_positive: false } }
# fn metrics_to_transcript_recipe() -> TripletRecipe { metrics_cross_view_recipe() }
struct FinancialReportsSource { /* store handle, symbol index, … */ }

impl IndexableSource for FinancialReportsSource {
    fn id(&self) -> &str { "financial_reports" }
    fn len_hint(&self) -> Option<usize> { Some(5000) }

    fn record_at(&self, _idx: usize) -> Result<Option<DataRecord>, SamplerError> {
        // Build a record with 2 or 3 sections depending on transcript availability.
        // Sparse records (None returns) are skipped entirely by the pager.
        Ok(None) // replace with real record construction
    }
}

impl DataSource for FinancialReportsSource {
    fn id(&self) -> &str { "financial_reports" }

    fn refresh(
        &self,
        _config: &SamplerConfig,
        cursor: Option<&SourceCursor>,
        limit: Option<usize>,
    ) -> Result<SourceSnapshot, SamplerError> {
        IndexablePager::new(DataSource::id(self)).refresh(self, cursor, limit)
    }

    fn reported_record_count(&self, _config: &SamplerConfig) -> Result<u128, SamplerError> {
        Ok(5000)
    }

    /// Source declares its own recipes — no recipe config required at call site.
    fn default_triplet_recipes(&self) -> Vec<TripletRecipe> {
        vec![
            metrics_cross_view_recipe(),      // dense: all records, weight 1.0
            metrics_to_transcript_recipe(),   // sparse: records with transcripts, weight 0.5
        ]
    }
}
```

When the sampler processes a record that has only two sections, it attempts each recipe in weighted order: `metrics_cross_view` succeeds (both `Role(Anchor)` and `Role(Context)` sections are present), while `metrics_to_transcript` returns no candidate for that slot (section index 2 is absent). The sampler moves on without any special handling in the data pipeline.

The same single `register_source` call enables both training hypotheses:

```rust,no_run
use std::sync::Arc;
use triplets::{SamplerConfig, TripletSampler, SplitRatios, DeterministicSplitStore, SplitLabel, Sampler};

# struct FinancialReportsSource;
# impl triplets::source::DataSource for FinancialReportsSource {
#   fn id(&self) -> &str { "financial_reports" }
#   fn refresh(&self, _: &SamplerConfig, _: Option<&triplets::source::SourceCursor>, _: Option<usize>) -> Result<triplets::source::SourceSnapshot, triplets::SamplerError> { unimplemented!() }
#   fn reported_record_count(&self, _: &SamplerConfig) -> Result<u128, triplets::SamplerError> { Ok(0) }
# }
let ratios = SplitRatios { train: 0.8, validation: 0.1, test: 0.1 };
let store = Arc::new(DeterministicSplitStore::new(ratios, 42).unwrap());
let mut sampler = TripletSampler::new(SamplerConfig::default(), store);

// One registration — the source provides both recipes.
sampler.register_source(Box::new(FinancialReportsSource { /* … */ }));

let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
// batch.triplets is a mix of "metrics_cross_view" and "metrics_to_transcript"
// samples, proportional to their configured weights and record coverage.
```

## Metadata Prefixes and Tag Dropout

`KvpPrefixSampler` attaches structured key-value metadata to a record. When a chunk is selected for training, the sampler may prepend a `meta:` line to the chunk text before it reaches the model. What that line looks like varies per sample — a variant is selected at random, each field picks one value from its declared list, and the field order within the line is shuffled:

```text
meta: source=daily-update | date=2025-01-01
<actual chunk content begins here>

# same record, different sample — different value, different field order:
meta: date=Jan 1, 2025 | source=daily-update
<actual chunk content begins here>
```

### Tag dropout

The `dropout` parameter controls how often the prefix is included at all:

| `dropout` | Effect                                                                              |
|-----------|-------------------------------------------------------------------------------------|
| `1.0`     | Prefix is **always** prepended.                                                     |
| `0.5`     | Prefix is prepended ~half the time; the rest of the time the model sees plain text. |
| `0.0`     | Prefix is **never** prepended.                                                      |

Training with `dropout < 1.0` teaches the model to handle both cases — chunks with metadata context and chunks without. This prevents the model from becoming dependent on the tags being present at inference time.

Individual fields also have their own **presence probability** controlled by `.with_presence(p)`. A field with `presence = 0.7` is omitted from a given prefix 30% of the time, independently of the sampler-level dropout.

```rust
use triplets::kvp::{KvpField, KvpPrefixSampler};

// dropout=0.8: 80% of chunks get a prefix, 20% see plain text.
let mut sampler = KvpPrefixSampler::new(0.8);

sampler.add_variant_fields([
    // "date" appears in every emitted prefix (presence=1.0 is the default).
    KvpField::many("date", ["2025-01-01", "Jan 1, 2025"]),
    // "source" is omitted from ~30% of emitted prefixes.
    KvpField::one("source", "daily-update").with_presence(0.7),
]);
```

The two value options for `date` are chosen at random each time the prefix is rendered, and — when a variant has more than one field — the order the fields appear in the line is also shuffled. The model therefore never sees a consistent positional signal for any individual tag.

You can call `add_variant` / `add_variant_fields` multiple times to register alternative field sets. One set is selected uniformly at random per sample — useful when you want to teach the model different metadata "views" of the same record:

```rust
use triplets::kvp::{KvpField, KvpPrefixSampler};

let mut sampler = KvpPrefixSampler::new(1.0);
// Variant A: structural tags
sampler.add_variant([("type", "earnings-call"), ("quarter", "Q1-2025")]);
// Variant B: temporal tags
sampler.add_variant_fields([KvpField::many("date", ["2025-01-15", "Jan 15, 2025"])]);
```

### Attaching a prefix to a record

Set `DataRecord::meta_prefix` on any record before registering it with a source:

```rust
use chrono::Utc;
use triplets::DataRecord;
use triplets::kvp::{KvpField, KvpPrefixSampler};

let mut prefix = KvpPrefixSampler::new(0.9);
prefix.add_variant_fields([
    KvpField::many("date", ["2025-01-01", "Jan 1, 2025"]),
    KvpField::one("source", "daily-update").with_presence(0.7),
]);

let record = DataRecord {
    id: "rec-001".into(),
    source: "news".into(),
    created_at: Utc::now(),
    updated_at: Utc::now(),
    quality: Default::default(),
    taxonomy: vec![],
    sections: vec![],
    meta_prefix: Some(prefix),
};
```

### Inspecting metadata on output chunks

Every `RecordChunk` carries a `kvp_meta: HashMap<String, Vec<String>>` field containing **all** declared keys and every possible value across all variants. This is populated unconditionally — even when dropout suppresses the prefix text for that particular chunk:

```rust,no_run
use std::sync::Arc;
use triplets::{SamplerConfig, TripletSampler, SplitRatios, DeterministicSplitStore, SplitLabel, Sampler};
let ratios = SplitRatios { train: 0.8, validation: 0.1, test: 0.1 };
let store = Arc::new(DeterministicSplitStore::new(ratios, 42).unwrap());
let mut sampler = TripletSampler::new(SamplerConfig::default(), store);
let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
for triplet in &batch.triplets {
    // All declared keys and values are here regardless of dropout.
    println!("{:?}", triplet.anchor.kvp_meta);
}
```

## Epochs and Determinism

### Iterating Epochs

```rust,no_run
use std::sync::Arc;
use triplets::{SamplerConfig, TripletSampler, SplitRatios, DeterministicSplitStore, SplitLabel, Sampler};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ratios = SplitRatios { train: 0.8, validation: 0.1, test: 0.1 };
    let store = Arc::new(DeterministicSplitStore::new(ratios, 42)?);
    let mut sampler = TripletSampler::new(SamplerConfig::default(), store);
    let mut batches_left = 1;
    let mut training_not_finished = || {
        let ret = batches_left > 0;
        batches_left -= 1;
        ret
    };
    // In your training loop:
    for epoch in 0..10 {
        sampler.set_epoch(epoch)?;

        while training_not_finished() {
            let batch = sampler.next_triplet_batch(SplitLabel::Train)?;
            // ... pass batch to your model ...
        }

        // Save state at the end of each epoch to allow resuming if training is interrupted.
        sampler.save_sampler_state(None)?;
    }

    Ok(())
}
```

### Deterministic Resuming

To resume training, initialize a `FileSplitStore` at the same path. The sampler automatically restores cursors, RNG state, and epoch progress from that store.

```rust,no_run
use std::sync::Arc;
use triplets::{SamplerConfig, TripletSampler, FileSplitStore, SplitRatios, Sampler};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ratios = SplitRatios { train: 0.8, validation: 0.1, test: 0.1 };
    let seed = 42;

    // Opening an existing FileSplitStore automatically loads its persisted state.
    let store = Arc::new(FileSplitStore::open("checkpoints/splits.bin", ratios, seed)?);

    // The sampler will resume from the exact record and recipe it was on.
    let mut sampler = TripletSampler::new(SamplerConfig::default(), store);
    Ok(())
}
```

> **Note**: Sampler state is intentionally lightweight. It persists source identifiers, integer record cursors, and compact RNG state vectors, not full data records. This keeps frequent checkpointing practical in long-running training jobs.

## Technical Details

### Threading Model

Concurrency is handled at multiple levels for high throughput:
- **Prefetching**: `BatchPrefetcher` runs a dedicated background worker thread that fills a bounded queue.
- **Parallel Ingestion**: Source refresh executes concurrently across registered sources during ingestion cycles.
- **Synchronous API**: Sampling calls are synchronous at the API boundary for straightforward training-loop integration.
- **Thread-Safe Shared Use**: `TripletSampler` is safe to share across threads (for example via `Arc`); concurrent calls are internally synchronized with a mutex, so a single sampler instance is callable from multiple threads without data races.

### Chunking and Windows

Long documents are handled through a pluggable `ChunkingAlgorithm`. The default `SlidingWindowChunker` splits sections into fixed-size token windows with configurable overlap, preserving full coverage of long text.

### Negative Mining

Negative selection is delegated to a pluggable backend.
- **DefaultBackend**: Uniform random selection from the candidate pool.
- **Bm25Backend**: (Requires `bm25-mining`) Ranks candidates by lexical overlap with the anchor to provide harder training examples.

## Capabilities

| Capability              | Description                                                                   |
|-------------------------|-------------------------------------------------------------------------------|
| **Source Agnostic**     | Implement `DataSource` or `IndexableSource` for any DB or API.                |
| **Weighted Sampling**   | Tune source and recipe frequencies to handle class imbalance.                 |
| **Epoch Shuffling**     | Deterministic pseudo-random shuffling that re-permutes per epoch.             |
| **Instruction Tuning**  | Attach task-specific prompts (e.g., "Summarize this...") to specific recipes. |
| **Metadata Decorators** | Inject structured prefixes into sampled text via `KvpPrefixSampler`.          |
| **Anti-Shortcut**       | Includes anchor/positive swapping to avoid asymmetric slot bias.              |

## License

`triplets` is distributed under both the MIT license and the Apache License (Version 2.0).

See [LICENSE-APACHE](https://github.com/jzombie/rust-triplets/blob/main/LICENSE-APACHE) and [LICENSE-MIT](https://github.com/jzombie/rust-triplets/blob/main/LICENSE-MIT) for details.
