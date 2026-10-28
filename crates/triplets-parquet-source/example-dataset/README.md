# Example parquet dataset

Sample data for `cargo run -p triplets-parquet-source --example local_ingest`.
Regenerate with `cargo run -p triplets-parquet-source --example
make_sample_parquet -- crates/triplets-parquet-source/example-dataset`.

Columns: `id`, `title`, `body`, `date`.

- Text mode (default): `--text body` reads the body column.
- Role mode: `--anchor title --positive body` maps title/body to
  Anchor/Context sections.

Note: the first ingest transcodes each `.parquet` file into a sibling
`.simdr` row store that freezes the parsed rows. If you change the
column mapping, delete the `.simdr` files first (they are git-ignored
build artifacts and are rebuilt automatically).
