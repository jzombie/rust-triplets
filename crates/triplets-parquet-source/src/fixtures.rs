//! Shared test fixtures: configs, sources, and fixture-file writers.
//!
//! These helpers are public so downstream crates' tests (e.g. the
//! Hugging Face wrapper's) can reuse them. They build deterministic,
//! network-free fixtures under caller-provided temp directories.
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parquet::data_type::{ByteArray, ByteArrayType};
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;
use simd_r_drive::storage_engine::DataStore;
use simd_r_drive::storage_engine::traits::DataStoreWriter;
use triplets_core::config::SamplerConfig;

use crate::config::ParquetRowsConfig;
use crate::constants::{HF_SHARD_STORE_META_ROWS_KEY, HF_SHARD_STORE_ROW_PREFIX};
use crate::row_view::{RowTextField, RowView};
use crate::source::ParquetRowSource;

/// Build a deterministic test config rooted at `snapshot_dir`.
///
/// Defaults mirror production (`text_columns = ["text"]`,
/// `id_column = "id"`), with small caches, namespace `"test"`, and empty
/// taxonomy.  A `<snapshot_dir>/_transient` managed dir is preconfigured
/// (transient + eviction scope) so index/transcode/cap tests have a
/// crate-managed directory; eviction itself stays disabled
/// (`local_disk_cap_bytes = None`) unless the test sets a cap.
pub fn test_config(snapshot_dir: PathBuf) -> ParquetRowsConfig {
    let mut config = ParquetRowsConfig::new("parquet_test", snapshot_dir, "test", Vec::new());
    config.cache_capacity = 10;
    config.transient_dirs = vec![config.snapshot_dir.join("_transient")];
    config.managed_subdir = Some("_transient".to_string());
    config
}

/// Name of the managed transient subdirectory preconfigured by
/// [`test_config`].
pub const TEST_TRANSIENT_DIR: &str = "_transient";

/// Build a test source without indexing.
///
/// Unlike [`ParquetRowSource::new`], this does not touch the filesystem:
/// callers write fixtures first, then register shards via
/// [`ParquetRowSource::ingest_shard`] or the index helpers. The sampler
/// seed is fixed to `1` for determinism.
pub fn test_source(config: ParquetRowsConfig) -> ParquetRowSource {
    let source = ParquetRowSource::new_empty(config);
    source.set_active_sampler_config(&SamplerConfig {
        seed: 1,
        per_source_max_pool: source.config.cache_capacity,
        ..SamplerConfig::default()
    });
    source
}

/// Write a minimal two-column (`id`, `text`) parquet fixture.
pub fn write_parquet_fixture(path: &Path, rows: &[(&str, &str)]) {
    let schema = Arc::new(
        parse_message_type(
            "message test_schema {
                    REQUIRED BINARY id (UTF8);
                    REQUIRED BINARY text (UTF8);
                }",
        )
        .unwrap(),
    );
    let props = Arc::new(WriterProperties::builder().build());
    let file = File::create(path).unwrap();
    let mut writer = SerializedFileWriter::new(file, schema, props).unwrap();
    let mut row_group = writer.next_row_group().unwrap();

    if let Some(mut col_writer) = row_group.next_column().unwrap() {
        let values = rows
            .iter()
            .map(|(id, _)| ByteArray::from(*id))
            .collect::<Vec<_>>();
        col_writer
            .typed::<ByteArrayType>()
            .write_batch(&values, None, None)
            .unwrap();
        col_writer.close().unwrap();
    }

    if let Some(mut col_writer) = row_group.next_column().unwrap() {
        let values = rows
            .iter()
            .map(|(_, text)| ByteArray::from(*text))
            .collect::<Vec<_>>();
        col_writer
            .typed::<ByteArrayType>()
            .write_batch(&values, None, None)
            .unwrap();
        col_writer.close().unwrap();
    }

    assert!(row_group.next_column().unwrap().is_none());
    row_group.close().unwrap();
    writer.close().unwrap();
}

/// Write a gzipped fixture file.
pub fn write_gzip_fixture(path: &std::path::Path, content: &[u8]) {
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use std::io::Write;
    let file = std::fs::File::create(path).unwrap();
    let mut encoder = GzEncoder::new(file, Compression::default());
    encoder.write_all(content).unwrap();
    encoder.finish().unwrap();
}

/// Write a `.simdr` row-store fixture with `(id, text)` rows.
pub fn write_simdr_fixture(path: &Path, rows: &[(&str, &str)]) {
    // Create/open the simd-r-drive DataStore and write row-view entries
    let store = DataStore::open(path).expect("open simdr store");
    if rows.is_empty() {
        store
            .write(HF_SHARD_STORE_META_ROWS_KEY, &(0u64).to_le_bytes())
            .expect("write meta");
        return;
    }

    let mut batch: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    for (i, (id, text)) in rows.iter().enumerate() {
        let row = RowView {
            row_id: Some(id.to_string()),
            timestamp: None,
            text_fields: vec![RowTextField {
                name: "text".to_string(),
                text: text.to_string(),
            }],
        };
        let payload = serde_json::to_vec(&row).expect("encode row");
        let mut key = HF_SHARD_STORE_ROW_PREFIX.to_vec();
        key.extend_from_slice(&(i as u64).to_le_bytes());
        batch.push((key, payload));
    }

    let refs: Vec<(&[u8], &[u8])> = batch
        .iter()
        .map(|(k, v)| (k.as_slice(), v.as_slice()))
        .collect();
    store.batch_write(&refs).expect("batch write");
    store
        .write(
            HF_SHARD_STORE_META_ROWS_KEY,
            &(rows.len() as u64).to_le_bytes(),
        )
        .expect("write meta");
}
