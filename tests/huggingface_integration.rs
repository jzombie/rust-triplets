#![cfg(feature = "huggingface")]

use std::fs;
use std::path::Path;
use std::sync::Arc;

use parquet::data_type::{ByteArray, ByteArrayType};
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;
use triplets::{DataSource, HuggingFaceRowSource, HuggingFaceRowsConfig, SamplerConfig};

fn seeded_config(seed: u64) -> SamplerConfig {
    SamplerConfig {
        seed,
        ..SamplerConfig::default()
    }
}

fn write_lines(path: &std::path::Path, lines: &[&str]) {
    let mut body = lines.join("\n");
    body.push('\n');
    fs::write(path, body).expect("failed writing snapshot shard");
}

fn write_parquet_fixture(path: &Path, rows: &[(&str, &str)]) {
    let schema = Arc::new(
        parse_message_type(
            "message test_schema {
                REQUIRED BINARY id (UTF8);
                REQUIRED BINARY text (UTF8);
            }",
        )
        .expect("failed parsing parquet schema"),
    );
    let props = Arc::new(WriterProperties::builder().build());
    let file = std::fs::File::create(path).expect("failed creating parquet fixture");
    let mut writer =
        SerializedFileWriter::new(file, schema, props).expect("failed creating parquet writer");
    let mut row_group = writer
        .next_row_group()
        .expect("failed creating parquet row group");

    if let Some(mut col_writer) = row_group.next_column().expect("missing id column") {
        let values = rows
            .iter()
            .map(|(id, _)| ByteArray::from(*id))
            .collect::<Vec<_>>();
        col_writer
            .typed::<ByteArrayType>()
            .write_batch(&values, None, None)
            .expect("failed writing id values");
        col_writer.close().expect("failed closing id column");
    }

    if let Some(mut col_writer) = row_group.next_column().expect("missing text column") {
        let values = rows
            .iter()
            .map(|(_, text)| ByteArray::from(*text))
            .collect::<Vec<_>>();
        col_writer
            .typed::<ByteArrayType>()
            .write_batch(&values, None, None)
            .expect("failed writing text values");
        col_writer.close().expect("failed closing text column");
    }

    assert!(
        row_group
            .next_column()
            .expect("unexpected extra column")
            .is_none(),
        "fixture schema should have exactly two columns"
    );
    row_group.close().expect("failed closing row group");
    writer.close().expect("failed closing parquet writer");
}

#[test]
fn huggingface_reads_local_jsonl_snapshot() {
    let temp = tempfile::tempdir().expect("failed creating tempdir");
    let shard_path = temp.path().join("part-00000.jsonl");

    write_lines(
        &shard_path,
        &[
            r#"{"id":"r1","title":"anchor one","body":"context one"}"#,
            r#"{"id":"r2","title":"anchor two","body":"context two"}"#,
        ],
    );

    let mut config = HuggingFaceRowsConfig::new(
        "hf_local_jsonl",
        "local/test-dataset",
        "default",
        "train",
        temp.path(),
    );
    config.shard_extensions = vec!["jsonl".to_string()];
    config.text_columns = vec!["title".to_string(), "body".to_string()];
    config.max_rows = Some(2);

    let source = HuggingFaceRowSource::new(config).expect("failed creating huggingface source");
    let seed = seeded_config(7);

    let count = source
        .reported_record_count(&seed)
        .expect("reported_record_count should succeed");
    assert_eq!(count, 2);

    let snapshot = source
        .refresh(&seed, None, Some(2))
        .expect("refresh should read jsonl rows");

    assert_eq!(snapshot.records.len(), 2);
    assert!(
        snapshot
            .records
            .iter()
            .all(|record| record.source == "hf_local_jsonl")
    );
    assert!(
        snapshot
            .records
            .iter()
            .any(|record| record.id.ends_with("::r1") || record.id.ends_with("::r2"))
    );
}

#[test]
fn huggingface_reads_local_ndjson_snapshot() {
    let temp = tempfile::tempdir().expect("failed creating tempdir");
    let shard_path = temp.path().join("part-00000.ndjson");

    write_lines(
        &shard_path,
        &[
            r#"{"id":"n1","text":"hello ndjson"}"#,
            r#"{"id":"n2","text":"goodbye ndjson"}"#,
        ],
    );

    let mut config = HuggingFaceRowsConfig::new(
        "hf_local_ndjson",
        "local/test-dataset",
        "default",
        "train",
        temp.path(),
    );
    config.shard_extensions = vec!["ndjson".to_string()];
    config.text_columns = vec!["text".to_string()];
    config.max_rows = Some(2);

    let source = HuggingFaceRowSource::new(config).expect("failed creating huggingface source");
    let seed = seeded_config(13);

    let snapshot = source
        .refresh(&seed, None, Some(2))
        .expect("refresh should read ndjson rows");

    assert_eq!(snapshot.records.len(), 2);
    assert!(
        snapshot
            .records
            .iter()
            .all(|record| record.source == "hf_local_ndjson")
    );
    assert!(snapshot.records.iter().all(|record| {
        record
            .sections
            .iter()
            .any(|section| section.text.contains("ndjson"))
    }));
}

#[test]
fn huggingface_reads_local_text_lines_snapshot() {
    let temp = tempfile::tempdir().expect("failed creating tempdir");
    let shard_path = temp.path().join("part-00000.txt");

    write_lines(
        &shard_path,
        &[
            "plain text row one",
            "plain text row two",
        ],
    );

    let mut config = HuggingFaceRowsConfig::new(
        "hf_local_text",
        "local/test-dataset",
        "default",
        "train",
        temp.path(),
    );
    config.shard_extensions = vec!["txt".to_string()];
    config.max_rows = Some(2);

    let source = HuggingFaceRowSource::new(config).expect("failed creating huggingface source");
    let seed = seeded_config(17);

    let snapshot = source
        .refresh(&seed, None, Some(2))
        .expect("refresh should read text rows");

    assert_eq!(snapshot.records.len(), 2);
    assert!(snapshot.records.iter().all(|record| {
        record
            .sections
            .iter()
            .any(|section| section.text.contains("plain text row"))
    }));
}

#[test]
fn huggingface_role_columns_mode_and_synthetic_ids_work() {
    let temp = tempfile::tempdir().expect("failed creating tempdir");
    let shard_path = temp.path().join("part-00001.ndjson");

    write_lines(
        &shard_path,
        &[
            r#"{"anchor":"headline a","positive":"summary a","ctx1":"context a1","ctx2":5}"#,
            r#"{"anchor":"headline b","positive":"summary b","ctx1":"context b1","ctx2":6}"#,
        ],
    );

    let mut config = HuggingFaceRowsConfig::new(
        "hf_role_columns",
        "local/test-dataset",
        "default",
        "train",
        temp.path(),
    );
    config.shard_extensions = vec!["ndjson".to_string()];
    config.id_column = Some("id".to_string());
    config.anchor_column = Some("anchor".to_string());
    config.positive_column = Some("positive".to_string());
    config.context_columns = vec!["ctx1".to_string(), "ctx2".to_string()];
    config.max_rows = Some(2);

    let source = HuggingFaceRowSource::new(config).expect("failed creating huggingface source");
    let seed = seeded_config(19);
    let snapshot = source
        .refresh(&seed, None, Some(2))
        .expect("refresh should parse role columns");

    assert_eq!(snapshot.records.len(), 2);
    assert!(
        snapshot
            .records
            .iter()
            .all(|record| record.id.contains("::"))
    );
    assert!(snapshot.records.iter().all(|record| {
        record
            .sections
            .iter()
            .any(|section| matches!(section.role, triplets::SectionRole::Anchor))
    }));
    assert!(snapshot.records.iter().all(|record| {
        record
            .sections
            .iter()
            .filter(|section| matches!(section.role, triplets::SectionRole::Context))
            .count()
            >= 2
    }));
}

#[test]
fn huggingface_refresh_cursor_wraps_and_limit_none_reads_all() {
    let temp = tempfile::tempdir().expect("failed creating tempdir");
    let shard_path = temp.path().join("part-00002.ndjson");

    write_lines(
        &shard_path,
        &[
            r#"{"id":"w1","text":"alpha"}"#,
            r#"{"id":"w2","text":"beta"}"#,
            r#"{"id":"w3","text":"gamma"}"#,
        ],
    );

    let mut config = HuggingFaceRowsConfig::new(
        "hf_cursor_wrap",
        "local/test-dataset",
        "default",
        "train",
        temp.path(),
    );
    config.shard_extensions = vec!["ndjson".to_string()];
    config.text_columns = vec!["text".to_string()];
    config.max_rows = Some(3);

    let source = HuggingFaceRowSource::new(config).expect("failed creating huggingface source");
    let seed = seeded_config(23);

    let full = source
        .refresh(&seed, None, None)
        .expect("refresh should read all rows when limit is none");
    assert_eq!(full.records.len(), 3);

    let wrapped = source
        .refresh(
            &seed,
            Some(&triplets::SourceCursor {
                last_seen: chrono::Utc::now(),
                revision: 99,
            }),
            Some(1),
        )
        .expect("refresh should wrap cursor beyond total rows");
    assert_eq!(wrapped.records.len(), 1);
}

#[test]
fn huggingface_refresh_surfaces_invalid_json_rows_as_errors() {
    let temp = tempfile::tempdir().expect("failed creating tempdir");
    let shard_path = temp.path().join("part-00003.ndjson");
    write_lines(&shard_path, &["{not-json"]);

    let mut config = HuggingFaceRowsConfig::new(
        "hf_invalid_json",
        "local/test-dataset",
        "default",
        "train",
        temp.path(),
    );
    config.shard_extensions = vec!["ndjson".to_string()];
    config.text_columns = vec!["text".to_string()];
    config.max_rows = Some(1);

    let source = HuggingFaceRowSource::new(config).expect("failed creating huggingface source");
    let seed = seeded_config(29);
    let err = source
        .refresh(&seed, None, Some(1))
        .expect_err("refresh should fail on invalid JSON row");
    let message = err.to_string();
    assert!(message.contains("failed decoding JSON row") || message.contains("inconsistent"));
}

#[test]
fn huggingface_reported_count_respects_max_rows_cap() {
    let temp = tempfile::tempdir().expect("failed creating tempdir");
    let shard_path = temp.path().join("part-00004.ndjson");

    write_lines(
        &shard_path,
        &[
            r#"{"id":"c1","text":"x"}"#,
            r#"{"id":"c2","text":"y"}"#,
            r#"{"id":"c3","text":"z"}"#,
        ],
    );

    let mut config = HuggingFaceRowsConfig::new(
        "hf_count_cap",
        "local/test-dataset",
        "default",
        "train",
        temp.path(),
    );
    config.shard_extensions = vec!["ndjson".to_string()];
    config.text_columns = vec!["text".to_string()];
    config.max_rows = Some(2);

    let source = HuggingFaceRowSource::new(config).expect("failed creating huggingface source");
    let seed = seeded_config(31);
    let count = source
        .reported_record_count(&seed)
        .expect("reported count should succeed");
    assert_eq!(count, 2);
}

#[test]
fn huggingface_reads_local_parquet_snapshot() {
    let temp = tempfile::tempdir().expect("failed creating tempdir");
    let shard_path = temp.path().join("part-00005.parquet");
    write_parquet_fixture(
        &shard_path,
        &[
            ("p1", "parquet one"),
            ("p2", "parquet two"),
            ("p3", "parquet three"),
        ],
    );

    let mut config = HuggingFaceRowsConfig::new(
        "hf_local_parquet",
        "local/test-dataset",
        "default",
        "train",
        temp.path(),
    );
    config.shard_extensions = vec!["parquet".to_string()];
    config.text_columns = vec!["text".to_string()];
    config.max_rows = Some(3);

    let source = HuggingFaceRowSource::new(config).expect("failed creating huggingface source");
    let seed = seeded_config(37);
    let snapshot = source
        .refresh(&seed, None, Some(3))
        .expect("refresh should read parquet rows");

    assert_eq!(snapshot.records.len(), 3);
    assert!(snapshot.records.iter().all(|record| {
        record
            .sections
            .iter()
            .any(|section| section.text.contains("parquet"))
    }));
}

#[test]
fn huggingface_role_columns_mode_errors_when_context_missing() {
    let temp = tempfile::tempdir().expect("failed creating tempdir");
    let shard_path = temp.path().join("part-00006.ndjson");
    write_lines(
        &shard_path,
        &[r#"{"anchor":"headline only","positive":"summary only"}"#],
    );

    let mut config = HuggingFaceRowsConfig::new(
        "hf_role_columns_error",
        "local/test-dataset",
        "default",
        "train",
        temp.path(),
    );
    config.shard_extensions = vec!["ndjson".to_string()];
    config.anchor_column = Some("anchor".to_string());
    config.positive_column = Some("positive".to_string());
    config.context_columns = vec!["ctx".to_string()];
    config.max_rows = Some(1);

    let source = HuggingFaceRowSource::new(config).expect("failed creating huggingface source");
    let seed = seeded_config(41);
    let err = source
        .refresh(&seed, None, Some(1))
        .expect_err("refresh should fail when required context column is missing");
    let message = err.to_string();
    assert!(message.contains("missing") || message.contains("inconsistent"));
}

#[test]
#[ignore = "network integration test against live Hugging Face dataset"]
fn huggingface_reads_live_remote_dataset() {
    let temp = tempfile::tempdir().expect("failed creating tempdir");

    let mut config = HuggingFaceRowsConfig::new(
        "hf_live_rotten_tomatoes",
        "cornell-movie-review-data/rotten_tomatoes",
        "default",
        "train",
        temp.path(),
    );
    config.shard_extensions = vec!["parquet".to_string()];
    config.text_columns = vec!["text".to_string()];
    config.max_rows = Some(8);

    let source = HuggingFaceRowSource::new(config).expect("failed creating huggingface source");
    let seed = seeded_config(17);
    let snapshot = source
        .refresh(&seed, None, Some(4))
        .expect("refresh should download and read live huggingface rows");

    assert!(!snapshot.records.is_empty());
    assert!(snapshot.records.len() <= 4);
    assert!(
        snapshot
            .records
            .iter()
            .all(|record| record.source == "hf_live_rotten_tomatoes")
    );
}
