# triplets-parquet-source

Generic local-parquet row reader for the triplets data pipeline.

This crate owns the network-independent half of row ingestion, extracted
from `triplets-hf-source` (see issue #59):

- column mapping (`text` / `anchor` / `positive` / `negative` / `context`)
  into normalized `RowView` values (`rows`),
- deterministic shard indexing over a local directory, including parquet
  row-group maps (`shard_index`),
- `.simdr` row-store transcoding and caching for random-access reads
  (`shard_indexer`, `store_cache`),
- the `ParquetRowSource` data source (`source`), including the
  `ingest_shard` hot-reload hook used by remote-backed wrappers when a
  background download materializes a new local shard.

Remote concerns — hub downloads, manifests, tokens, expansion policy,
per-dataset identity — stay in the calling crate (e.g.
`triplets-hf-source`, where `HuggingFaceRowSource` wraps
`ParquetRowSource`). This crate has no network dependencies.

## Identity

Unlike the Hugging Face source, record-id fallback and taxonomy are not
baked in: `ParquetRowsConfig` carries a caller-provided
`row_id_namespace` (used as `{namespace}:{row_index}` when no id column
resolves) and a caller-provided `taxonomy` vector stamped on every
record.
