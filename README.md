# triplets

[![made-with-rust][rust-logo]][rust-src-page] [![crates.io][crates-badge]][crates-page] [![MIT licensed][mit-license-badge]][mit-license-page] [![Apache 2.0 licensed][apache-2.0-license-badge]][apache-2.0-license-page]

Compose an effectively unlimited supply of [training triplets](https://en.wikipedia.org/wiki/Triplet_loss), pairs, or plaintext samples from your existing corpus. This crate handles ingestion, mixing multiple sources, deterministic train/validation/test splitting, and optional [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) hard-negative mining.

## Overview

In metric learning and language model training, a **triplet** consists of an **anchor**, a **positive** example (similar to the anchor), and a **negative** example (dissimilar to the anchor).

`triplets` provides a high-performance, streaming pipeline to:
1. **Ingest** data from local files, Hugging Face, or custom backends.
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

A `TripletSampler` requires a `SplitStore` to manage record-to-split assignments and a `SamplerConfig` for its operational parameters.

```rust
use std::sync::Arc;
use triplets::{
    SamplerConfig, TripletSampler, SplitRatios, 
    DeterministicSplitStore, SplitLabel, Sampler
};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
// 1. Define your train/validation/test ratios (e.g., 80/10/10).
let ratios = SplitRatios { train: 0.8, validation: 0.1, test: 0.1 };

// 2. Initialize a deterministic split store. 
// The seed ensures record IDs are always assigned to the same split.
let seed = 42;
let store = Arc::new(DeterministicSplitStore::new(ratios, seed)?);

// 3. Create the sampler.
let mut sampler = TripletSampler::new(SamplerConfig::default(), store);
# Ok(())
# }
```

## Features

| Feature            | What it enables                                           | Default |
| ------------------ | --------------------------------------------------------- | ------- |
| `huggingface`      | Streaming from Hugging Face dataset repositories.         | No      |
| `bm25-mining`      | BM25 hard-negative ranking within strategy-defined pools. | No      |
| `extended-metrics` | Additional per-triplet diagnostics for debugging.         | No      |

## Configuring Sources

### Local File Source
Recursively indexes text files from a directory. Ideal for local datasets or exported corpora.

```rust
# use std::sync::Arc;
# use triplets::{SamplerConfig, TripletSampler, SplitRatios, DeterministicSplitStore};
use triplets::source::{FileSource, FileSourceConfig};

# let ratios = SplitRatios { train: 0.8, validation: 0.1, test: 0.1 };
# let store = Arc::new(DeterministicSplitStore::new(ratios, 42).unwrap());
# let mut sampler = TripletSampler::new(SamplerConfig::default(), store);
// Create a source named "docs" targeting a local directory.
let config = FileSourceConfig::new("docs", "./data/corpus")
    .with_text_files_only(true)
    .with_trust(0.9); // Assign a quality score to this source

let source = FileSource::new(config);
sampler.register_source(Box::new(source));
```


### Hugging Face Source
Streams rows directly from the Hugging Face Hub without requiring full dataset downloads.

```rust,no_run
# #[cfg(feature = "huggingface")]
# {
# use std::sync::Arc;
# use triplets::{SamplerConfig, TripletSampler, SplitRatios, DeterministicSplitStore, Sampler};
use triplets::{HuggingFaceRowSource, HuggingFaceRowsConfig};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
# let ratios = SplitRatios { train: 0.8, validation: 0.1, test: 0.1 };
# let store = Arc::new(DeterministicSplitStore::new(ratios, 42)?);
# let mut sampler = TripletSampler::new(SamplerConfig::default(), store);
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
# Ok(())
# }
# }
```

### Custom Data Source
Implement the `IndexableSource` trait to integrate any backend that can fetch records by a stable integer index.

```rust
# use std::sync::Arc;
# use triplets::{SamplerConfig, TripletSampler, SplitRatios, DeterministicSplitStore};
use chrono::Utc;
use triplets::{DataRecord, SamplerError};
use triplets::source::{IndexableSource, IndexableAdapter};

struct MyApiSource;

impl IndexableSource for MyApiSource {
    fn id(&self) -> &str { "api_source" }
    fn len_hint(&self) -> Option<usize> { Some(1000) }
    fn record_at(&self, idx: usize) -> Result<Option<DataRecord>, SamplerError> {
        // Fetch record 'idx' from your database or API.
        Ok(Some(DataRecord {
            id: format!("api_{idx}"),
            source: self.id().into(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            quality: Default::default(),
            taxonomy: vec![],
            sections: vec![], // Add text content here
            meta_prefix: None,
        }))
    }
}

# let ratios = SplitRatios { train: 0.8, validation: 0.1, test: 0.1 };
# let store = Arc::new(DeterministicSplitStore::new(ratios, 42).unwrap());
# let mut sampler = TripletSampler::new(SamplerConfig::default(), store);
let adapter = IndexableAdapter::new(MyApiSource);
sampler.register_source(Box::new(adapter));
```


## Sampling and Mixing

### Weighted Sampling
Adjust the relative frequency of each source to handle class imbalance or dataset quality.

```rust,no_run
# use std::sync::Arc;
# use triplets::{SamplerConfig, TripletSampler, SplitRatios, DeterministicSplitStore, SplitLabel, Sampler};
use std::collections::HashMap;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
# let ratios = SplitRatios { train: 0.8, validation: 0.1, test: 0.1 };
# let store = Arc::new(DeterministicSplitStore::new(ratios, 42)?);
# let mut sampler = TripletSampler::new(SamplerConfig::default(), store);
// Pull from HF 70% of the time and local files 30% of the time.
let mut weights = HashMap::new();
weights.insert("hf_finance".to_string(), 0.7);
weights.insert("docs".to_string(), 0.3);

let batch = sampler.next_triplet_batch_with_weights(SplitLabel::Train, &weights)?;
# Ok(())
# }
```

### Output Format
The sampler produces `SampleTriplet` objects containing the sampled text and associated metadata.

```rust,no_run
# use std::sync::Arc;
# use triplets::{SamplerConfig, TripletSampler, SplitRatios, DeterministicSplitStore, SplitLabel, Sampler};
# let ratios = SplitRatios { train: 0.8, validation: 0.1, test: 0.1 };
# let store = Arc::new(DeterministicSplitStore::new(ratios, 42).unwrap());
# let mut sampler = TripletSampler::new(SamplerConfig::default(), store);
# let batch = sampler.next_triplet_batch(SplitLabel::Train).unwrap();
for triplet in batch.triplets {
    // Primary content
    let anchor_text = &triplet.anchor.text;
    let pos_text    = &triplet.positive.text;
    let neg_text    = &triplet.negative.text;
    
    // Metadata
    let recipe      = &triplet.recipe;      // which recipe was used
    let weight      = triplet.weight;       // training weight
    let instruction = triplet.instruction;  // optional instruction string
}
```

## Epochs and Determinism

### Iterating Epochs
In a typical training loop, you notify the sampler of a new epoch to reset its cursors and reshuffle sources.

```rust,no_run
# use std::sync::Arc;
# use triplets::{SamplerConfig, TripletSampler, SplitRatios, DeterministicSplitStore, SplitLabel, Sampler};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
# let ratios = SplitRatios { train: 0.8, validation: 0.1, test: 0.1 };
# let store = Arc::new(DeterministicSplitStore::new(ratios, 42)?);
# let mut sampler = TripletSampler::new(SamplerConfig::default(), store);
# let mut batches_left = 1;
# let mut training_not_finished = || {
#     let ret = batches_left > 0;
#     batches_left -= 1;
#     ret
# };
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
# Ok(())
# }
```

### Deterministic Resuming
To resume training, initialize a `FileSplitStore` at the same path. The sampler automatically restores its cursors, RNG state, and epoch progress from the store.

```rust,no_run
use std::sync::Arc;
use triplets::{SamplerConfig, TripletSampler, FileSplitStore, SplitRatios, Sampler};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let ratios = SplitRatios { train: 0.8, validation: 0.1, test: 0.1 };
let seed = 42;

// Opening an existing FileSplitStore automatically loads its persisted state.
let store = Arc::new(FileSplitStore::open("checkpoints/splits.bin", ratios, seed)?);

// The sampler will resume from the exact record and recipe it was on.
let mut sampler = TripletSampler::new(SamplerConfig::default(), store);
# Ok(())
# }
```

> **Note**: The sampler state is designed to be extremely lightweight. It only persists source identifiers, integer record cursors, and small RNG state vectors, rather than the data records themselves. This makes it efficient to checkpoint frequently during long-running training jobs.

## Technical Details

### Threading Model
Concurrency is handled at multiple levels to ensure high throughput:
- **Prefetching**: A `Rayon`-managed thread pool pre-samples batches into a bounded queue.
- **Parallel Ingestion**: Multiple sources can be refreshed in parallel.
- **Synchronous API**: The public sampling API is synchronous at its boundaries for ease of integration into training loops.

### Chunking and Windows
Long documents are automatically handled via a pluggable `ChunkingAlgorithm`. The default `SlidingWindowChunker` breaks sections into windows of a fixed token size with a configurable overlap, ensuring full coverage of long texts.

### Negative Mining
The sampler delegates negative selection to a pluggable backend.
- **DefaultBackend**: Uniform-random selection from the candidate pool.
- **Bm25Backend**: (Requires `bm25-mining` feature) Ranks candidates by lexical overlap with the anchor to provide "harder" training examples.

## Capabilities

| Capability              | Description                                                                   |
| ----------------------- | ----------------------------------------------------------------------------- |
| **Source Agnostic**     | Implement `DataSource` or `IndexableSource` for any DB or API.                |
| **Weighted Sampling**   | Tune source and recipe frequencies to handle class imbalance.                 |
| **Epoch Shuffling**     | Deterministic pseudo-random shuffling that re-permutes per epoch.             |
| **Instruction Tuning**  | Attach task-specific prompts (e.g., "Summarize this...") to specific recipes. |
| **Metadata Decorators** | Inject structured prefixes into sampled text via `KvpPrefixSampler`.          |
| **Anti-Shortcut**       | Includes anchor/positive swapping to avoid asymmetric slot bias.              |

## License

`triplets` is distributed under the terms of both the MIT license and the Apache License (Version 2.0).

See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) for details.

[rust-src-page]: https://www.rust-lang.org/
[rust-logo]: https://img.shields.io/badge/Made%20with-Rust-black
[crates-page]: https://crates.io/crates/triplets
[crates-badge]: https://img.shields.io/crates/v/triplets.svg
[mit-license-page]: https://github.com/jzombie/rust-triplets/blob/main/LICENSE-MIT
[mit-license-badge]: https://img.shields.io/badge/license-MIT-blue.svg
[apache-2.0-license-page]: https://github.com/jzombie/rust-triplets/blob/main/LICENSE-APACHE
[apache-2.0-license-badge]: https://img.shields.io/badge/license-Apache%202.0-blue.svg
