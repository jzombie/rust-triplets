# AGENTS.md — rust-triplets

Deterministic data sampling for embedding-model training: triplet/text
batches, hash-based split assignment, weighted multi-source ingestion.

## The stream contract (load-bearing — read before touching the sampler)

Ingestion is an **infinite forward stream**, not an epoch-sized dataset:

- `set_epoch(n)` re-permutes draw **order** only. It MUST NEVER rewind
  ingestion stream cursors. Consecutive epochs draw disjoint windows from
  wherever the stream currently sits.
- A fresh sampler starts at the head. A resumed (fresh) process restarts
  at the head — that replays the prefix region, accepted because nothing
  is persisted. Do not "fix" this with cursor files or sidecars.
- Split assignment is a deterministic hash per record (`derive_label_for_key`);
  the stream position is global and split-agnostic. There is no per-split
  cursor to persist or rewind.

## History you **must not ever** repeat

- `reset_stream_cursors` (rewound the stream every epoch: ~82k-row windows
  replayed over 10M-row stores) was **deleted entirely** after costing a
  week of GPU training. Do not reintroduce any rewind primitive under any
  name. The regression test below fails the moment an epoch replays a
  prior window.
- The old test `sampler_resalt_per_epoch...` certified the bug on a
  240-row toy where a rewind is invisible. Scale is the point: regression
  tests must run at multi-window scale.

## Tests that enforce the contract

- `triplets-core`: `consecutive_epochs_draw_disjoint_windows_without_replay`
  (drains two epochs, asserts zero window overlap).
- Full suite: `cargo test -p triplets-core --lib`.

## Versions

- The workspace uses one shared version (single `version.workspace` in
  the root `Cargo.toml` + internal path deps). Consumers pin that same
  version — after any bump, verify every consumer manifest and lockfile
  resolves it before calling it done. Check, don't trust.
