#![cfg(feature = "huggingface")]

use serde::Serialize;
use simd_r_drive::storage_engine::DataStore;
use simd_r_drive::storage_engine::traits::DataStoreWriter;
use std::fs;
use std::path::Path;
use triplets::source::backends::huggingface_source::load_hf_sources_from_list;
use triplets::{
    DataSource, HfListRoots, HfSourceEntry, HuggingFaceRowSource, HuggingFaceRowsConfig,
    SamplerConfig, build_hf_sources, parse_csv_fields, parse_hf_source_line, parse_hf_uri,
    resolve_hf_list_roots,
};

const HF_SHARD_STORE_ROW_PREFIX: &[u8] = b"rowv1|";
const HF_SHARD_STORE_META_ROWS_KEY: &[u8] = b"meta|rows";

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

#[derive(Serialize)]
struct SimdrTextField {
    name: String,
    text: String,
}

#[derive(Serialize)]
struct SimdrRowView {
    row_id: Option<String>,
    timestamp: Option<String>,
    text_fields: Vec<SimdrTextField>,
}

fn row_store_row_key(local_idx: usize) -> Vec<u8> {
    let mut key = Vec::with_capacity(HF_SHARD_STORE_ROW_PREFIX.len() + std::mem::size_of::<u64>());
    key.extend_from_slice(HF_SHARD_STORE_ROW_PREFIX);
    key.extend_from_slice(&(local_idx as u64).to_le_bytes());
    key
}

fn write_simdr_fixture(path: &Path, rows: &[(&str, &str)]) {
    let store = DataStore::open(path).expect("failed opening simdr fixture store");
    for (local_idx, (id, text)) in rows.iter().enumerate() {
        let row = SimdrRowView {
            row_id: Some((*id).to_string()),
            timestamp: None,
            text_fields: vec![SimdrTextField {
                name: "text".to_string(),
                text: (*text).to_string(),
            }],
        };
        let payload = serde_json::to_vec(&row).expect("failed serializing simdr row");
        store
            .write(&row_store_row_key(local_idx), &payload)
            .expect("failed writing simdr row");
    }
    store
        .write(
            HF_SHARD_STORE_META_ROWS_KEY,
            &(rows.len() as u64).to_le_bytes(),
        )
        .expect("failed writing simdr row count");
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

    write_lines(&shard_path, &["plain text row one", "plain text row two"]);

    let mut config = HuggingFaceRowsConfig::new(
        "hf_local_text",
        "local/test-dataset",
        "default",
        "train",
        temp.path(),
    );
    config.shard_extensions = vec!["txt".to_string()];

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
fn huggingface_role_columns_mode_skips_missing_rows_and_keeps_valid() {
    let temp = tempfile::tempdir().expect("failed creating tempdir");
    let shard_path = temp.path().join("part-00001.ndjson");

    write_lines(
        &shard_path,
        &[
            r#"{"anchor":"headline a","positive":"summary a"}"#,
            r#"{"anchor":"headline b","positive":"summary b","ctx":"context b"}"#,
        ],
    );

    let mut config = HuggingFaceRowsConfig::new(
        "hf_role_columns_skip",
        "local/test-dataset",
        "default",
        "train",
        temp.path(),
    );
    config.shard_extensions = vec!["ndjson".to_string()];
    config.anchor_column = Some("anchor".to_string());
    config.positive_column = Some("positive".to_string());
    config.context_columns = vec!["ctx".to_string()];

    let source = HuggingFaceRowSource::new(config).expect("failed creating huggingface source");
    let seed = seeded_config(23);
    let snapshot = source
        .refresh(&seed, None, Some(2))
        .expect("refresh should skip missing required context rows");

    assert_eq!(snapshot.records.len(), 1);
    assert!(
        snapshot.records[0]
            .sections
            .iter()
            .any(|section| section.text.contains("context b"))
    );
}

#[test]
fn huggingface_parses_source_list_with_explicit_mappings() {
    let temp = tempfile::tempdir().expect("failed creating tempdir");
    let list_path = temp.path().join("hf_sources.txt");

    fs::write(
        &list_path,
        "# demo list\n\n"
            .to_string()
            + "hf://org/dataset/default/train anchor=title positive=text context=ctx1,ctx2 text=text\n",
    )
    .expect("failed writing source list");

    let entries = load_hf_sources_from_list(list_path.to_str().expect("utf8 path"))
        .expect("failed parsing source list");

    assert_eq!(entries.len(), 1);
    let entry = &entries[0];
    assert_eq!(entry.uri, "hf://org/dataset/default/train");
    assert_eq!(entry.anchor_column.as_deref(), Some("title"));
    assert_eq!(entry.positive_column.as_deref(), Some("text"));
    assert_eq!(
        entry.context_columns,
        vec!["ctx1".to_string(), "ctx2".to_string()]
    );
    assert_eq!(entry.text_columns, vec!["text".to_string()]);
}

#[test]
fn huggingface_helper_parsers_cover_success_and_error_paths() {
    assert_eq!(
        parse_csv_fields(" title, body , , tags "),
        vec!["title".to_string(), "body".to_string(), "tags".to_string()]
    );

    let parsed = parse_hf_source_line(
        "hf://org/dataset/default/train anchor=title positive=body context=c1,c2 text_columns=body",
    )
    .expect("line should parse");
    assert_eq!(parsed.uri, "hf://org/dataset/default/train");
    assert_eq!(parsed.anchor_column.as_deref(), Some("title"));
    assert_eq!(parsed.positive_column.as_deref(), Some("body"));
    assert_eq!(parsed.context_columns, vec!["c1", "c2"]);
    assert_eq!(parsed.text_columns, vec!["body"]);

    assert!(parse_hf_source_line("").is_err());
    assert!(parse_hf_source_line("file://foo anchor=a").is_err());
    assert!(parse_hf_source_line("hf://org/dataset/default/train badtoken").is_err());
    assert!(parse_hf_source_line("hf://org/dataset/default/train nope=field").is_err());
    assert!(parse_hf_source_line("hf://org/dataset/default/train").is_err());

    let defaults = parse_hf_uri("hf://org/dataset").expect("uri should parse with defaults");
    assert_eq!(
        defaults,
        (
            "org/dataset".to_string(),
            "default".to_string(),
            "train".to_string()
        )
    );
    assert!(parse_hf_uri("hf://org").is_err());
    assert!(parse_hf_uri("https://huggingface.co").is_err());
}

#[test]
fn huggingface_list_root_and_builder_helpers_cover_invalid_inputs() {
    let temp = tempfile::tempdir().expect("failed creating tempdir");
    let list_path = temp.path().join("hf_sources_invalid.txt");
    fs::write(&list_path, "# no sources\n\n").expect("failed writing list file");

    let err = resolve_hf_list_roots(list_path.to_str().expect("utf8 path").to_string())
        .expect_err("empty list should fail");
    assert!(err.contains("no hf:// entries found"));

    let roots = HfListRoots {
        source_list: "manual".to_string(),
        sources: vec![HfSourceEntry {
            uri: "hf://org".to_string(),
            anchor_column: Some("a".to_string()),
            positive_column: None,
            context_columns: Vec::new(),
            text_columns: Vec::new(),
        }],
    };
    let built = build_hf_sources(&roots);
    assert!(built.is_empty());
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

    let source = HuggingFaceRowSource::new(config).expect("failed creating huggingface source");
    let seed = seeded_config(29);
    let err = source
        .refresh(&seed, None, Some(1))
        .expect_err("refresh should fail on invalid JSON row");
    let message = err.to_string();
    assert!(message.contains("failed decoding JSON row") || message.contains("inconsistent"));
}

#[test]
fn huggingface_reported_count_returns_file_count() {
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

    let source = HuggingFaceRowSource::new(config).expect("failed creating huggingface source");
    let seed = seeded_config(31);
    let count = source
        .reported_record_count(&seed)
        .expect("reported count should succeed");
    assert_eq!(count, 3);
}

#[test]
fn huggingface_reads_local_parquet_snapshot() {
    let temp = tempfile::tempdir().expect("failed creating tempdir");
    let shard_path = temp.path().join("part-00005.simdr");
    write_simdr_fixture(
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
    config.shard_extensions = vec!["simdr".to_string()];
    config.text_columns = vec!["text".to_string()];

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
fn huggingface_role_columns_mode_skips_when_context_missing() {
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

    let source = HuggingFaceRowSource::new(config).expect("failed creating huggingface source");
    let seed = seeded_config(41);
    let snapshot = source
        .refresh(&seed, None, Some(1))
        .expect("refresh should skip rows with missing required context column");
    assert!(snapshot.records.is_empty());
}

#[test]
fn huggingface_text_columns_mode_skips_when_required_column_missing() {
    let temp = tempfile::tempdir().expect("failed creating tempdir");
    let shard_path = temp.path().join("part-00007.ndjson");
    write_lines(&shard_path, &[r#"{"id":"t1","title":"headline only"}"#]);

    let mut config = HuggingFaceRowsConfig::new(
        "hf_text_columns_skip",
        "local/test-dataset",
        "default",
        "train",
        temp.path(),
    );
    config.shard_extensions = vec!["ndjson".to_string()];
    config.text_columns = vec!["title".to_string(), "body".to_string()];

    let source = HuggingFaceRowSource::new(config).expect("failed creating huggingface source");
    let seed = seeded_config(43);
    let snapshot = source
        .refresh(&seed, None, Some(1))
        .expect("refresh should skip rows missing required text columns");

    assert!(snapshot.records.is_empty());
}

#[test]
fn huggingface_parquet_role_columns_skip_missing_context_without_error() {
    let temp = tempfile::tempdir().expect("failed creating tempdir");
    let shard_path = temp.path().join("part-00008.simdr");
    write_simdr_fixture(&shard_path, &[("p1", "parquet body")]);

    let mut config = HuggingFaceRowsConfig::new(
        "hf_parquet_role_skip",
        "local/test-dataset",
        "default",
        "train",
        temp.path(),
    );
    config.shard_extensions = vec!["simdr".to_string()];
    config.anchor_column = Some("text".to_string());
    config.positive_column = Some("text".to_string());
    config.context_columns = vec!["ctx".to_string()];

    let source = HuggingFaceRowSource::new(config).expect("failed creating huggingface source");
    let seed = seeded_config(47);
    let snapshot = source
        .refresh(&seed, None, Some(1))
        .expect("refresh should read simdr rows without role-column revalidation");

    assert_eq!(snapshot.records.len(), 1);
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
