# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/) and this project adheres to
(or is loosely based on) Semantic Versioning.

## [0.28.0-alpha] - TBD

### Changed
- Bumped `parquet` from 59.1.0 to 60.0.0 (#166, semver-major)
- Bumped `clap` from 4.6.6 to 4.6.7 (#167)

### Added
- **Precomputed-embedding channel through the sampler.** `RecordSection.embedding`
  and `RecordChunk.embedding` (`Option<Arc<[f32]>>`, `#[serde(skip)]` — transient,
  re-ingested on resume, never persisted or compared). The sampler stamps every
  materialized chunk from its parent section in `materialize_chunks`, so chunking
  backends stay agnostic and vectors ride the pipeline as Arc pointer copies.
- **`SrdSource` populates all section embeddings** (anchor + candidate/positive/
  negative) from its store vectors, so sampled pairs carry exact store vectors
  on every chunk with no side lookups.
- **Single `SrdSource::open` with explicit mode choice.** `mode: None` detects
  from entry 0 in the same open (no probe handle); `mode: Some(m)` pins the
  mode for empty stores and is verified against entry 0 otherwise — a mismatch
  returns loud `SrdError::ModeMismatch` (empty stores give `EmptyStore`).
  One open per dataset, ever.
- **Native weighted source selection (deficit round-robin).** `TripletSampler`
  apportions batch picks across sources by the per-call weight map: each draw
  credits every eligible source its weight quantum and serves the highest
  deficit, deducting the round total. Deterministic (no RNG — same state, same
  choice), exact over each weight cycle (0.75/0.25 yields exactly 6/2 anchors
  per 8-row batch). Uniform/empty maps take the legacy uniform-cycling path
  bit-identically, so existing deterministic sequences are unchanged. One
  shared sampler with N registered sources is now the mixing topology — no
  outer wrappers, cross-source negatives and global dedup intact.
- **`RecordChunk.source` is required** (no `#[serde(default)]`): a chunk without
  a source resolves split lookups against the wrong identity, so deserialization
  without it fails loudly instead of defaulting to `""`.

### Changed
- **BREAKING: `FileSplitStore` storage version bumped 1 → 2.** Persisted split
  label keys are namespaced by source (`split:<source>\0<id>`). v1 stores are
  rejected loudly at open (version mismatch) instead of silently missing their
  explicit assignments. No dual-read fallback is kept (pre-1.0: rebuild the
  store rather than carrying legacy shims). Epoch/sampler-state formats
  unchanged. SRD baked-data files are unaffected (encoding untouched).

### Fixed
- **Record identity is composite (`RecordKey { source, id }`) throughout the
  sampler.** Pools, chunk index, split labels, dedup sets, ingestion deltas,
  and BM25 state key by source + id instead of the bare id string, so same-id
  records from different sources coexist instead of the later source silently
  overwriting the earlier one's pool entries. Backends keep their native ids
  (no decoration, no parsing — nothing touches the id string); the composite
  derives from each record's own `source` + `id` fields, and sampled chunks
  carry a structured `source` field for the same purpose. Split derivation,
  orderings, and cursor offsets hash the id part exactly as before, so
  deterministic sequences are unchanged for existing corpora (all 512 core
  golden tests pass unmodified in behavior). `SplitStore::label_for/upsert/
  ensure` now take `&RecordKey`/`RecordKey`; `FileSplitStore` persisted label
  keys are namespaced the same way (epoch/sampler-state formats unchanged).
- **Weight maps are validated up front on every batch call.** A non-empty map
  with an unregistered source id (or negative weight) returns loud
  `InvalidWeight` — including the uniform single-unknown case that used to
  slip through validation and sample as if unweighted.

## [0.27.1-alpha] - 2026-09-09

### Changed
- Bumped `reqwest` from 0.13.4 to 0.13.5 and `reqwest-drive` from 0.13.4-alpha to 0.13.5-alpha
- Bumped `parquet` (and `arrow-*`) from 59.2.0 to 59.3.0 (#164)
- Bumped `flate2` from 1.1.9 to 1.1.10 (#163)
- Bumped `indexmap` from 2.14.0 to 2.14.2 (#162)

## [0.27.0-alpha] - 2026-09-07

### Changed
- Bumped `simd-r-drive` from 0.16.3-alpha to 0.17.1-alpha
- Add throughput timing metrics (wall time spent waiting, embedding, and flushing, per step) to offline embedder loop events

### Fixed
- **Sampler ingestion sync is now incremental.** Steady-state cache advances use
  an O(Δ) delta sync (`IngestionManager::sync_delta` over a unified global
  version timeline) instead of cloning, sorting, and diffing the full record
  pool on every batch. This was ~41% of `sampler-prefetch` CPU in profiles.
- **Split-label lookups are cached persistently** (`split_labels` on the sampler
  with a `get_or_insert_split_label` helper), so per-record `DataStore::read()`
  calls drop to near-zero after the first sync instead of running 2–3× per
  record per sync.
- **Token counts are precomputed on `RecordSection`** (`#[serde(default)]`, so
  existing serialized data still loads) instead of tokenizing record text inside
  the sync hot path.
- **`RecordCache` stores `Arc<DataRecord>`**, eliminating deep-clone allocations
  on snapshot/sync.
- **`source_record_indices` is keyed by `RecordId`** instead of positional
  `usize` offsets into the record map, so evictions can no longer corrupt or
  shift the index.
- **Per-source index maintenance is incremental (strict O(Δ)).** Evictions do
  ordered removal and additions binary-search by `stable_hash_str` order instead
  of rebuilding + re-sorting the full index every batch; `chunk_index` is
  maintained incrementally and long-section eligibility uses per-source
  refcounts instead of a full-pool rescan. Force-refresh cycles (which replace
  cache contents wholesale) fall back to full-resync semantics.
- **Negative-backend `on_sync_start` now fires on every sync boundary**, fixing
  stale BM25 cursor state persisting across snapshot syncs.
- **`RecordCache::clear()` now logs evictions** (bounded, version-stamped) so
  delta syncs report pool replacements and cross-batch text-dedup state is
  pruned for records that truly left — fixing `Exhausted("text_recipes")`
  failures after force refreshes while preserving dedup for re-added records.
- Fixed 8 failing `triplets-core` tests covering the above (BM25 cursor reset,
  text/pair/triplet batch sequences, weighted-drain distribution).
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` is clean.
- **Per-batch split-map rebuild eliminated.** `records_by_split` output is cached
  on the sampler behind a `pool_generation` counter that advances only on actual
  pool-membership change (new IDs, true evictions, full resyncs); steady-state
  batches fetch an O(1) reference instead of rebuilding the full map twice per
  batch. The cached reference is passed to `EpochTracker::reconcile` every
  batch, preserving its state-machine hook while hitting the unchanged-population
  fast path.
- **BM25 refresh uses the in-memory split cache.** The `split_fn` closure passed
  to `on_records_refreshed` now reads `split_labels` instead of doing a
  `DataStore::read()` per record.

## [0.26.1-alpha] - 2026-09-04

### Changed
- Bumped `serial_test` from 3.5.0 to 4.0.1
- Bumped `parquet` from 59.1.0 to 59.2.0
- Bumped `thiserror` from 2.0.19 to 2.0.20
- Bumped `clap` from 4.6.2 to 4.6.6
- Bumped `tokio` from 1.53.0 to 1.53.1

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
