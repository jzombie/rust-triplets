#![cfg(feature = "huggingface")]

use std::fs;

use triplets::{DataSource, HuggingFaceRowSource, HuggingFaceRowsConfig, configured_source_with_seed};

fn write_lines(path: &std::path::Path, lines: &[&str]) {
    let mut body = lines.join("\n");
    body.push('\n');
    fs::write(path, body).expect("failed writing snapshot shard");
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

    let source = configured_source_with_seed(
        HuggingFaceRowSource::new(config).expect("failed creating huggingface source"),
        7,
    );

    let count = source
        .reported_record_count()
        .expect("reported_record_count should succeed");
    assert_eq!(count, 2);

    let snapshot = source
        .refresh(None, Some(2))
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

    let source = configured_source_with_seed(
        HuggingFaceRowSource::new(config).expect("failed creating huggingface source"),
        13,
    );

    let snapshot = source
        .refresh(None, Some(2))
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

    let source = configured_source_with_seed(
        HuggingFaceRowSource::new(config).expect("failed creating huggingface source"),
        17,
    );
    let snapshot = source
        .refresh(None, Some(4))
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
