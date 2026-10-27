# Plan: Generic parquet reader (HF becomes the network layer)

Closes #59. Unrelated to #66 (KVP) — no KVP changes here.

**Status: IMPLEMENTED** on branch
`59-extract-parquet-reader-from-hf-source-to-use-independently--draft`.
Workspace green (`cargo test --workspace`), zero warnings in both
crates, `cargo clippy` adds no new lints, `cargo fmt` applied.

Follow-up for PR #183 CI (lint + coverage checks):
- `cargo sort --workspace` ordering fixed; `fmt --check` clean.
- Coverage back to baseline-or-better on all local metrics (lines
  94.98% vs 94.97%, regions 95.28% vs 95.23%, functions 92.20% vs
  91.99%): new `tests/generic_surface.rs` covers the generic public
  surface, plus row-skip tests and a `SafeTestSource` drop-join test.
  Remaining misses are pre-existing patterns (tracing-disabled log
  args, poison/error closures), identical in kind to `main`.

As-built deviations from the draft below:
- `ParquetRowsConfig` carries `row_id_namespace: String` + `taxonomy:
  Vec<String>` data fields instead of fn pointers (simpler, same
  generality; `From<&HuggingFaceRowsConfig>` fills them).
- `enforce_disk_cap_locked` dropped its unused `_protected_path` arg;
  eviction scope comes from `managed_subdir` (`None` = disabled).
- `sync_shard_state_from_disk_locked` returns `bool` (any missing); the
  HF wrapper resets its remote frontier on `true`.
- `ParquetRowSource::{local, cache}` are `pub` white-box fields for
  wrappers/tests; `config` stays behind the `config()` accessor.
- Generic `len_hint` = materialized rows only; HF keeps its 3-branch
  version (headroom while remotes remain).
- HF crate dropped `parquet`/`rayon`/`walkdir` deps (`flate2`/`filetime`
  → dev-deps); the new crate has no `reqwest`.
- Decisions resolved: name `triplets-parquet-source`, `.simdr` moved
  as-is, compat shims kept (`StoreCache`, store/recipe constants).

## Goal

New generic local-parquet row reader sharing HF's row mechanics
(column mapping → `RowView` → `DataRecord`, shard index, parquet
row-group reads, `.simdr` row-store cache). `triplets-hf-source`
keeps only HF concerns: hub download, manifest/expansion, token,
`dataset/config/split` identity — then delegates to the generic reader.

## Baseline (verified read-only)

- `triplets-hf-source` ≈ 14k lines incl. tests. Generic modules:
  `parsing/` (317), `huggingface_source/` caches + `RowView` (183),
  `rows/` core (963), `shard_index/` (361), `shard_indexer/` (339),
  `file_utils/`, `types.rs` (`ShardIndex`).
- HF-only: `download/` (967), `expansion/` (87), `builder/` (208),
  `managed_hf_*` cache roots, `source_core.rs` remote-expansion path (1240).
- Coupling points: `HuggingFaceRowsConfig` bakes `dataset/config/split`
  into record-id fallback (`rows/rows.rs:53-58`) and
  `taxonomy=[dataset=,config=,split=]` (`rows/rows.rs:181-184`);
  `HuggingFaceRowSource::new` (`source_core.rs:146`) builds HTTP client,
  validates token, indexes shards, transcodes transients;
  `ensure_row_available` (`source_core.rs:378`) interleaves remote download.
- Only consumer of the crate is the root `triplets` crate behind the
  `huggingface` feature (`Cargo.toml:80`); `triplets-srd-source`
  (4 files) is the template for a small source crate.

## Design

- New crate `triplets-parquet-source` (name TBD, see decisions).
  Deps: `triplets-core, parquet, serde, serde_json, rayon,
  simd-r-drive, walkdir, chrono, tracing, flate2` (for existing
  transcode paths). No `reqwest`, `reqwest-drive`, `cache-manager`.
- New `ParquetRowsConfig`: `source_id, snapshot_dir, shard_extensions,
  text/anchor/positive/negative/context_columns, id_column` +
  `id_fallback fn` + `taxonomy fn` (replacing hardcoded
  `dataset/config/split`), cache capacities, `trust_override`.
  `impl From<&HuggingFaceRowsConfig>` lives in the HF crate.
- New `ParquetRowSource` (moved `SourceState`, `RowCache`,
  `ParquetCache`, refresh/`read_row_batch`/transcode path, minus
  `http_runtime/http_client/expansion_thread/remote_candidates`).
  Implements `triplets_core::source::DataSource`.
- `HuggingFaceRowSource` becomes `{ inner: ParquetRowSource, hf:
  HttpAndExpansionState }`; `DataSource` impl ensures/downloads, then
  delegates. Public constructors (`new`, `build_hf_sources*`) keep
  signatures; `lib.rs` re-exports moved types (`RowView`,
  `HuggingFaceRowsConfig`, etc.) so downstream code doesn't break.

## Phases

1. **Scaffold**: new crate, workspace `members` + `workspace.dependencies`
   entries, `Cargo.lock` via `cargo metadata`; copy `triplets-srd-source`
   crate layout as template. Baseline:
   `cargo test -p triplets-hf-source` green before touching code.
2. **Move leaf modules** (no logic changes, fix imports only):
   `types` → `parsing` → `huggingface_source` (rename to `row_view`
   if desired) → `file_utils`. Move their unit tests with them.
3. **Move `rows/`**: generalize `parse_row` id-fallback and
   `row_to_record` taxonomy behind config fns; keep `RowView` wire
   format identical (`.simdr` compat). Move `rows_tests.rs`.
4. **Move `shard_index`/`shard_indexer`**: drop manifest-root/transient
   assumptions to plain local-dir walk; manifest-aware filtering stays
   in HF crate. Move tests.
5. **Split `source_core`** (addresses #59-review SEV-1: index sync):
   extract `ParquetRowSource`; HF keeps download/materialize/expansion
   (`download_and_materialize_shard`, `download_next_remote_shard`,
   `materialize_local_file`, token validation, shared client). Delegate
   `refresh`, `reported_record_count`, `default_triplet_recipes`.
   - **State ownership split.** Today one `state: Arc<Mutex<SourceState>>`
     holds both frontiers: local (`shards`, `materialized_rows`) and
     remote (`remote_candidates`, `remote_candidate_sizes`,
     `next_remote_idx`, `remote_candidate_order`), and
     `download_next_remote_shard` mutates both under that one lock
     (`source_core.rs:741-742,840-842`: `shard.global_start =
     state.materialized_rows; state.materialized_rows += ...;
     state.shards.push(...)`). After the split: `ParquetRowSource`
     owns the local-frontier lock (`shards`, `materialized_rows`,
     `RowCache`, `ParquetCache`); the HF wrapper owns the remote-frontier
     lock. `Clone` keeps sharing both via `Arc`, as today
     (`source_core.rs:128-142`); the background `expansion_thread` +
     `EXPANSION_GATE` (`expansion.rs`) stay in the HF crate operating on
     clones.
   - **Hot-reload interface (new API on `ParquetRowSource`, required —
     no silent "delegate").** The HF wrapper must be able to ingest a
     freshly downloaded + transcoded shard without tearing down caches:
     `pub fn ingest_shard(&self, shard: ShardIndex) -> Result<(),
     SamplerError>` — assigns `global_start = materialized_rows`,
     bumps `materialized_rows`, pushes to `shards`, all under the
     generic's local-frontier lock (mirroring today's second-lock
     reassignment at `source_core.rs:722-742`, which is what keeps
     concurrent downloads from colliding on `global_start`). Readers
     (`read_row_batch` via `locate_shard`) snapshot the shard vec under
     the same lock, so ingest-vs-read is race-safe.
   - Call sequence in HF `download_next_remote_shard` becomes:
     download → transcode-to-`.simdr` (generic helper, unchanged) →
     `inner.ingest_shard(shard)` → advance `next_remote_idx` under the
     HF remote lock. `ensure_row_available` keeps its gating role but
     the `idx < materialized_rows` check reads the generic's lock.
6. **Wire-up**: root `triplets` crate gets optional `parquet-source`
   passthrough if needed; keep `huggingface` feature behavior identical.
   Run full workspace `cargo test`, `cargo clippy`, `cargo fmt`.
7. **Docs**: crate README, update #59, note in hf-source README that
   parquet mechanics live in the new crate.

## Acceptance

- `cargo test --workspace` green; no `rows/` parsing behavior change
  (byte-identical `RowView` payloads for fixture shards).
- Local `*.parquet` dir readable with zero HF config/network objects
  constructed (new crate has no `reqwest` dep).
- Existing HF integration tests pass unmodified.
- New race test: concurrent `ingest_shard` + `read_row_batch` yields no
  `global_start` collisions and newly ingested rows become readable
  (covers the SEV-1 ingest-vs-read path).

## Non-goals

- #66 KVP mapping; download/expansion logic changes; `.simdr` format
  changes; public API breaks beyond re-export shims.

## Open decisions (need user)

1. Crate name: `triplets-parquet-source` vs `triplets-row-source`.
2. Move `.simdr` store layer as-is (identical mechanics, faster) vs
   simplify generic reader to direct parquet row-group reads (cleaner,
   more work).
3. Keep `triplets-hf-source` re-export shims or allow breaking import
   paths (crate is `0.30.1-alpha`, so breaking is cheap).
