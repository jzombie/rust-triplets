> **Disclaimer:** This crate is not affiliated with, endorsed by, or associated with Hugging Face. "Hugging Face" is a trademark of Hugging Face, Inc.

Hugging Face integration for the [triplets](https://crates.io/crates/triplets) data pipeline framework.

For full documentation on configuring sources, sampling, chunking, and pipeline orchestration,
see the [main triplets README](https://github.com/jzombie/rust-triplets/blob/main/README.md) on GitHub.

## Why Not Use the `hf-hub` Crate?

This crate uses a custom downloader rather than the official [`hf-hub`](https://crates.io/crates/hf-hub) crate, for several structural reasons:

- **Deterministic Permutation Shuffling**: The sampler uses seed-based shard consumption orderings (`build_candidate_order`) tied to training step reproducibility. `hf-hub` is designed for sequential workspace downloads or snapshot checkouts, not granular step-shuffled indexing.
- **Granular Network Throttling**: The pipeline shares a process-wide `ClientWithMiddleware` with custom connection parameters, debug assertion bypasses, and worker thread limits (`HF_SHARED_RUNTIME_WORKER_THREADS`).
- **Integrated Storage Scopes**: File verification is tightly coupled to internal `.simdr` structural checks (e.g., `HF_SHARD_STORE_META_ROWS_KEY` metadata pairs), which upstream caching clients would abstract away.
- **Cache Layout Independence**: Importing from `hf-hub` and then deleting files from its cache would break its content-addressed reference database (`blobs/` + `snapshots/` symlinks), cause global side-effects for concurrent HF tools on the same machine, and introduce unnecessary I/O thrashing.
