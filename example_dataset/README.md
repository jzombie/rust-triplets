# Triplets Example Dataset

This directory provides a small, synthetic, non-proprietary dataset used by
`triplets` examples.

Layout:
- `source_a` through `source_f` each contain dated text samples.
- Examples load these directories by default unless overridden with repeated `--source-root <PATH>` flags.
- For `multi_source_demo`, split-store persistence defaults to `.sampler_store/split_store.bin` and can be configured with `--split-store-path <FILE>` or `--split-store-dir <DIR>`.
