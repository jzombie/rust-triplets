# Triplets Example Dataset

This directory provides a small, synthetic, non-proprietary dataset used by
`triplets` examples.

Layout:
- `source_a` through `source_f` each contain dated text samples.
- Examples load these directories by default unless overridden with repeated `--source-root <PATH>` flags.
- For `multi_source_demo`, split-store persistence defaults to managed cache-group path `.cache/triplets/multi-source-demo/split_store.bin`.
- Use `--split-store-path <FILE>` to override the default demo path.
- For explicit load/save behavior in code, use `FileSplitStore::open_with_load_path(Some(load_from), save_to, ratios, seed)`.
