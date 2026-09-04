# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/) and this project adheres to
(or is loosely based on) Semantic Versioning.

## [Unreleased]

### Fixed
- JSON array `.json` files (e.g. `gbharti/finance-alpaca`) now stream-parse correctly
  instead of failing on newline splits.
- Large HuggingFace datasets with >1000 files (e.g. `shash42/forecast-news`) now
  paginate the Hub API tree endpoint instead of silently returning no shards.
- `build_hf_sources_with_weights` now returns partial successes with failure details
  instead of silently dropping failed sources.
- Weight map entries are only inserted after source initialization succeeds, preventing
  `InvalidWeight` errors when a source fails.
- `extract_next_link_url` no longer aborts early on malformed `Link` header segments.
- Updated `h2` from v0.4.13 to v0.4.19 to fix RUSTSEC-2026-0258 (unbounded empty
  DATA frames vulnerability).

## [0.26.0-alpha] - 2026-08-15

### Added
- `SamplerAdapter` (`triplets-offline-embedder`) now carries a per-source
  `weights` map and forwards it to the sampler's `next_*_batch_with_weights`
  APIs, so callers can enforce an explicit source mixture (e.g. weighted
  dataset ratios) through the offline embedder. An empty map degrades to the
  existing unweighted behavior.

### Changed
- Bumped `serde` from 1.0.228 to 1.0.229 (#145)
- Bumped `serde_json` from 1.0.150 to 1.0.151 (#142)
- Bumped `thiserror` from 2.0.18 to 2.0.19 (#144)
- Bumped `clap` from 4.6.1 to 4.6.2 (#143)
- Bumped `tokio` from 1.52.3 to 1.53.0 (#146)

### Deprecated
- Deprecated the unweighted batch-fetch methods on the `Sampler` trait
  (`next_pair_batch`, `next_text_batch`, `next_triplet_batch`) and on
  `TripletSampler` (`next_pair_batch_for_split`, `next_text_batch_for_split`,
  `next_triplet_batch_for_split`, `prefetch_pair_batches`, `prefetch_text_batches`,
  `prefetch_triplet_batches`). These sample all sources uniformly; use the
  `*_with_weights` variants with an explicit per-source weight map to honor a
  data mixture.

## [0.25.0-alpha] - 2026-07-17

### Added
- Expanded test coverage across `triplets-hf-source` and `triplets-core` (#138)
- Modular sub-crate structure for `triplets-hf-source`: builder, config,
  disk_cache, download, expansion, file_utils, parsing, rows, shard_index,
  shard_indexing, source_core modules

### Changed
- **Refactored `triplets-hf-source` sub-crate** (#136): decomposed monolithic
  `huggingface_source.rs` (11k+ lines) into focused modules with dedicated tests
- **Refactored sampler batch from SoA to AoS** (#129): replaced flat
  `SamplerBatch` (separate `anchor_texts`/`pos_texts`/`neg_texts` vectors)
  with `PairEntry`/`TripletEntry` structs for zero-copy string movement
- **Fixed PairLabel round-trip** (#133): split `SrdEntry` into
  `SrdPairRecord`/`SrdTripletRecord`/`SrdRecord` enum; added `label` field
  to `DataRecord` and label propagation through `SrdSource` and `Sampler`
- **Replaced `println!`/`eprintln!` with `tracing` crate** (#137)
- **Removed `hf-hub` dependency and legacy `datasets-server` fallback paths**
  (#130): replaced `/parquet`, `/size`, `/info` endpoints with Hub API tree
  endpoint (`/api/datasets/{dataset}/tree/main`)
- Bumped `parquet` from 58.3.0 to 59.1.0 (#126)
- Bumped `simd-r-drive` to v0.16.3-alpha and `reqwest-drive` to v0.13.4-alpha
- Updated README files

### Removed
- `hf-hub` crate dependency entirely (#130)
- Legacy `datasets-server` fallback paths: sibling-based candidate resolution,
  `ClassLabel` `/info` resolution, global row count `/size` queries (#130)

### Fixed
- Negative pair labels silently converted to Positive on SRD read-back (#133):
  negative labels were discarded because `DataRecord` had no label field and
  `SrdSource::refresh()` dropped `entry.label`
- Transient shard downloads incorrectly written into evictable managed cache (#135)

### Security
- Path traversal guard rejects `ParentDir`/`Prefix` components in HF source
  candidate resolution (#130)
