use serde::Serialize;
use simd_r_drive::storage_engine::DataStore;
use simd_r_drive::storage_engine::traits::DataStoreWriter;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use triplets_core::constants::env_vars::TRIPLETS_SKIP_LIVE_TESTS;
use triplets_core::constants::sampler::AUTO_INJECTED_LONG_SECTION_CHUNK_PAIR_RECIPE_NAME;
use triplets_core::utils::platform_newline;
use triplets_core::{
    ChunkingStrategy, DataSource, DeterministicSplitStore, Sampler, SamplerConfig, SplitLabel,
    SplitRatios, TripletSampler,
};
use triplets_hf_source::{
    ENV_TRIPLETS_HF_INFO_ENDPOINT, ENV_TRIPLETS_HF_PARQUET_ENDPOINT, ENV_TRIPLETS_HF_SIZE_ENDPOINT,
    ENV_TRIPLETS_HF_TOKEN_TEST_DATASET, HF_RECIPE_TEXT_SIMCSE_WRONG_ARTICLE, HF_TOKEN, HfListRoots,
    HfSourceEntry, HuggingFaceRowSource, HuggingFaceRowsConfig, build_hf_sources,
    load_hf_sources_from_list, parse_csv_fields, parse_hf_source_line, parse_hf_uri,
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

/// Create a `HuggingFaceRowsConfig` with token auth explicitly disabled.
///
/// Integration tests that use local or public datasets must call this instead
/// of `HuggingFaceRowsConfig::new()` directly, so that an ambient `HF_TOKEN`
/// environment variable (present in CI when the private-dataset job runs) does
/// not trigger the whoami token-validation request against the live API and
/// cause every unrelated test to fail.
///
/// The `hf_token_private_dataset_access` test is the sole exception: it sets
/// `config.hf_token = Some(token)` explicitly after construction, which
/// overrides the `None` set here and restores the intended behaviour.
fn config_no_auth(
    source_id: &str,
    dataset: &str,
    config_name: &str,
    split: &str,
    snapshot_dir: impl Into<std::path::PathBuf>,
) -> HuggingFaceRowsConfig {
    let mut c = HuggingFaceRowsConfig::new(source_id, dataset, config_name, split, snapshot_dir);
    c.hf_token = None;
    c
}

fn write_lines(path: &std::path::Path, lines: &[&str]) {
    let nl = platform_newline();
    let mut body = lines.join(nl);
    body.push_str(nl);
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

    let mut config = config_no_auth(
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

    let mut config = config_no_auth(
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
fn huggingface_text_mode_triplets_can_use_different_anchor_positive_windows() {
    let temp = tempfile::tempdir().expect("failed creating tempdir");
    let shard_path = temp.path().join("part-00000.ndjson");

    write_lines(
        &shard_path,
        &[
            r#"{"id":"r1","text":"alpha beta gamma delta epsilon zeta eta theta iota kappa"}"#,
            r#"{"id":"r2","text":"one two three four five six seven eight nine ten"}"#,
        ],
    );

    let mut source_config = config_no_auth(
        "hf_text_windows",
        "local/test-dataset",
        "default",
        "train",
        temp.path(),
    );
    source_config.shard_extensions = vec!["ndjson".to_string()];
    source_config.text_columns = vec!["text".to_string()];
    let source = HuggingFaceRowSource::new(source_config).expect("failed creating source");

    let recipes = source.default_triplet_recipes();
    assert_eq!(recipes.len(), 1);
    assert_eq!(recipes[0].name, HF_RECIPE_TEXT_SIMCSE_WRONG_ARTICLE);
    assert!(recipes[0].allow_same_anchor_positive);

    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };
    let mut sampler_config = seeded_config(41);
    sampler_config.split = split;
    sampler_config.allowed_splits = vec![SplitLabel::Train];
    sampler_config.batch_size = 1;
    sampler_config.ingestion_max_records = 16;
    sampler_config.chunking = {
        let mut c = ChunkingStrategy::default();
        c.max_window_tokens = 3;
        c.overlap_tokens = vec![0];
        c.summary_fallback_weight = 0.0;
        c.summary_fallback_tokens = 0;
        c.chunk_weight_floor = 0.0;
        c
    };

    let store = Arc::new(DeterministicSplitStore::new(split, 777).expect("split store"));
    let sampler = TripletSampler::new(sampler_config, store);
    sampler.register_source(Box::new(source));

    let mut observed_hf_simcse_triplet = false;
    let mut observed_different_window_pair = false;
    for _ in 0..64 {
        let batch = sampler
            .next_triplet_batch(SplitLabel::Train)
            .expect("triplet batch");
        assert_eq!(batch.triplets.len(), 1);
        let triplet = &batch.triplets[0];

        if triplet.recipe != HF_RECIPE_TEXT_SIMCSE_WRONG_ARTICLE {
            // In text= mode the source contributes the SimCSE recipe, but the sampler may
            // additionally auto-inject the long-section chunk-pair recipe when sections
            // exceed the chunk window. Those auto-injected samples are out-of-scope for
            // this assertion, so we skip only that known recipe and treat any other recipe
            // as a regression.
            if triplet.recipe == AUTO_INJECTED_LONG_SECTION_CHUNK_PAIR_RECIPE_NAME {
                continue;
            }
            panic!("unexpected recipe in HF text-mode test: {}", triplet.recipe);
        }
        observed_hf_simcse_triplet = true;
        assert_eq!(triplet.anchor.record_id, triplet.positive.record_id);

        let anchor_window = match &triplet.anchor.view {
            triplets_core::data::ChunkView::Window { index, .. } => Some(*index),
            _ => None,
        };
        let positive_window = match &triplet.positive.view {
            triplets_core::data::ChunkView::Window { index, .. } => Some(*index),
            _ => None,
        };
        if let (Some(a), Some(p)) = (anchor_window, positive_window)
            && a != p
        {
            observed_different_window_pair = true;
            break;
        }
    }

    assert!(
        observed_hf_simcse_triplet,
        "expected to observe at least one HF SimCSE triplet in text= mode"
    );
    assert!(
        observed_different_window_pair,
        "expected HF text= mode to occasionally sample different window indices for anchor and positive"
    );
}

#[test]
fn huggingface_reads_local_text_lines_snapshot() {
    let temp = tempfile::tempdir().expect("failed creating tempdir");
    let shard_path = temp.path().join("part-00000.txt");

    write_lines(&shard_path, &["plain text row one", "plain text row two"]);

    let mut config = config_no_auth(
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

    let mut config = config_no_auth(
        "hf_role_columns",
        "local/test-dataset",
        "default",
        "train",
        temp.path(),
    );
    config.shard_extensions = vec!["ndjson".to_string()];
    config.id_column = Some("id".to_string());
    config.anchor_columns = vec!["anchor".to_string()];
    config.positive_columns = vec!["positive".to_string()];
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
            .any(|section| matches!(section.role, triplets_core::SectionRole::Anchor))
    }));
    assert!(snapshot.records.iter().all(|record| {
        record
            .sections
            .iter()
            .filter(|section| matches!(section.role, triplets_core::SectionRole::Context))
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

    let mut config = config_no_auth(
        "hf_role_columns_skip",
        "local/test-dataset",
        "default",
        "train",
        temp.path(),
    );
    config.shard_extensions = vec!["ndjson".to_string()];
    config.anchor_columns = vec!["anchor".to_string()];
    config.positive_columns = vec!["positive".to_string()];
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

    let nl = platform_newline();
    fs::write(
        &list_path,
        format!("# demo list{nl}{nl}hf://org/dataset/default/train anchor=title positive=text context=ctx1,ctx2 text=text{nl}"),
    )
    .expect("failed writing source list");

    let entries = load_hf_sources_from_list(list_path.to_str().expect("utf8 path"))
        .expect("failed parsing source list");

    assert_eq!(entries.len(), 1);
    let entry = &entries[0];
    assert_eq!(entry.uri, "hf://org/dataset/default/train");
    assert_eq!(entry.anchor_columns, vec!["title".to_string()]);
    assert_eq!(entry.positive_columns, vec!["text".to_string()]);
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
    assert_eq!(parsed.anchor_columns, vec!["title".to_string()]);
    assert_eq!(parsed.positive_columns, vec!["body".to_string()]);
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
            // No split component in the URI → empty string, meaning all-splits mode.
            // Triplets' own train/validation/test split logic handles partitioning.
            "".to_string()
        )
    );
    // Explicit split is preserved exactly.
    let explicit_split = parse_hf_uri("hf://org/dataset/default/train")
        .expect("uri with explicit split should parse");
    assert_eq!(explicit_split.2, "train");

    assert!(parse_hf_uri("hf://org").is_err());
    assert!(parse_hf_uri("https://huggingface.co").is_err());
}

#[test]
fn huggingface_list_root_and_builder_helpers_cover_invalid_inputs() {
    let temp = tempfile::tempdir().expect("failed creating tempdir");
    let list_path = temp.path().join("hf_sources_invalid.txt");
    let nl = platform_newline();
    fs::write(&list_path, format!("# no sources{nl}{nl}")).expect("failed writing list file");

    let err = resolve_hf_list_roots(list_path.to_str().expect("utf8 path").to_string())
        .expect_err("empty list should fail");
    assert!(err.contains("no hf:// entries found"));

    let roots = HfListRoots {
        source_list: "manual".to_string(),
        sources: vec![HfSourceEntry {
            uri: "hf://org".to_string(),
            anchor_columns: vec!["a".to_string()],
            positive_columns: Vec::new(),
            context_columns: Vec::new(),
            text_columns: Vec::new(),
            trust: None,
            source_id: None,
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

    let mut config = config_no_auth(
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
            Some(&triplets_core::SourceCursor {
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

    let mut config = config_no_auth(
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

    let mut config = config_no_auth(
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

    let mut config = config_no_auth(
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

    let mut config = config_no_auth(
        "hf_role_columns_error",
        "local/test-dataset",
        "default",
        "train",
        temp.path(),
    );
    config.shard_extensions = vec!["ndjson".to_string()];
    config.anchor_columns = vec!["anchor".to_string()];
    config.positive_columns = vec!["positive".to_string()];
    config.context_columns = vec!["ctx".to_string()];

    let source = HuggingFaceRowSource::new(config).expect("failed creating huggingface source");
    let seed = seeded_config(41);
    let snapshot = source
        .refresh(&seed, None, Some(1))
        .expect("refresh should skip rows with missing required context column");
    assert!(snapshot.records.is_empty());
}

#[test]
fn huggingface_text_columns_mode_skips_when_all_candidates_missing() {
    let temp = tempfile::tempdir().expect("failed creating tempdir");
    let shard_path = temp.path().join("part-00007.ndjson");
    // Row contains neither "title" nor "body" → coalescing finds no candidate.
    write_lines(
        &shard_path,
        &[r#"{"id":"t1","other":"irrelevant content"}"#],
    );

    let mut config = config_no_auth(
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
        .expect("refresh should skip rows where no text candidate matches");

    assert!(snapshot.records.is_empty());
}

#[test]
fn huggingface_text_columns_coalesces_to_first_nonempty_candidate() {
    let temp = tempfile::tempdir().expect("failed creating tempdir");
    let shard_path = temp.path().join("part-00009.ndjson");
    // "title" is an empty string → coalescing falls through to "text".
    write_lines(
        &shard_path,
        &[
            r#"{"id":"c1","title":"","text":"fallback content"}"#,
            r#"{"id":"c2","title":"primary content","text":"ignored"}"#,
        ],
    );

    let mut config = config_no_auth(
        "hf_text_coalesce",
        "local/test-dataset",
        "default",
        "train",
        temp.path(),
    );
    config.shard_extensions = vec!["ndjson".to_string()];
    config.text_columns = vec!["title".to_string(), "text".to_string()];

    let source = HuggingFaceRowSource::new(config).expect("failed creating huggingface source");
    let seed = seeded_config(53);
    let snapshot = source
        .refresh(&seed, None, Some(2))
        .expect("refresh should coalesce text column candidates");

    assert_eq!(snapshot.records.len(), 2);

    let c1 = snapshot
        .records
        .iter()
        .find(|r| r.id.ends_with("::c1"))
        .expect("record c1 should be present");
    assert!(
        c1.sections.iter().any(|s| s.text == "fallback content"),
        "c1 should use 'text' column because 'title' is empty"
    );

    let c2 = snapshot
        .records
        .iter()
        .find(|r| r.id.ends_with("::c2"))
        .expect("record c2 should be present");
    assert!(
        c2.sections.iter().any(|s| s.text == "primary content"),
        "c2 should use 'title' column because it is non-empty"
    );
}

#[test]
fn huggingface_positive_columns_coalesces_to_first_nonempty_candidate() {
    let temp = tempfile::tempdir().expect("failed creating tempdir");
    let shard_path = temp.path().join("part-00010.ndjson");
    // Row A: "summary" absent → falls through to "body".
    // Row B: "summary" present → "summary" is used; "body" is ignored.
    write_lines(
        &shard_path,
        &[
            r#"{"id":"p1","anchor":"anchor content","body":"fallback positive"}"#,
            r#"{"id":"p2","anchor":"anchor content","summary":"chosen positive","body":"ignored"}"#,
        ],
    );

    let mut config = config_no_auth(
        "hf_positive_coalesce",
        "local/test-dataset",
        "default",
        "train",
        temp.path(),
    );
    config.shard_extensions = vec!["ndjson".to_string()];
    config.anchor_columns = vec!["anchor".to_string()];
    config.positive_columns = vec!["summary".to_string(), "body".to_string()];

    let source = HuggingFaceRowSource::new(config).expect("failed creating huggingface source");
    let seed = seeded_config(59);
    let snapshot = source
        .refresh(&seed, None, Some(2))
        .expect("refresh should coalesce positive column candidates");

    assert_eq!(snapshot.records.len(), 2);

    let p1 = snapshot
        .records
        .iter()
        .find(|r| r.id.ends_with("::p1"))
        .expect("record p1 should be present");
    assert!(
        p1.sections.iter().any(|s| s.text == "fallback positive"),
        "p1 should use 'body' because 'summary' is absent"
    );

    let p2 = snapshot
        .records
        .iter()
        .find(|r| r.id.ends_with("::p2"))
        .expect("record p2 should be present");
    assert!(
        p2.sections.iter().any(|s| s.text == "chosen positive"),
        "p2 should use 'summary' because it is present and non-empty"
    );
}

#[test]
fn huggingface_parquet_role_columns_skip_missing_context_without_error() {
    let temp = tempfile::tempdir().expect("failed creating tempdir");
    let shard_path = temp.path().join("part-00008.simdr");
    write_simdr_fixture(&shard_path, &[("p1", "parquet body")]);

    let mut config = config_no_auth(
        "hf_parquet_role_skip",
        "local/test-dataset",
        "default",
        "train",
        temp.path(),
    );
    config.shard_extensions = vec!["simdr".to_string()];
    config.anchor_columns = vec!["text".to_string()];
    config.positive_columns = vec!["text".to_string()];
    config.context_columns = vec!["ctx".to_string()];

    let source = HuggingFaceRowSource::new(config).expect("failed creating huggingface source");
    let seed = seeded_config(47);
    let snapshot = source
        .refresh(&seed, None, Some(1))
        .expect("refresh should read simdr rows without role-column revalidation");

    assert_eq!(snapshot.records.len(), 1);
}

#[test]
fn huggingface_record_ids_are_stable_across_independent_source_instances() {
    // The same row in the same shard file must produce the same record ID
    // regardless of which source instance reads it. This is what makes split
    // store assignments survive a parquet cache deletion and re-download.
    let temp = tempfile::tempdir().expect("failed creating tempdir");
    write_lines(
        &temp.path().join("part-00000.ndjson"),
        &[
            r#"{"id":"stable1","text":"row one"}"#,
            r#"{"id":"stable2","text":"row two"}"#,
        ],
    );

    let make_source = || {
        let mut config = config_no_auth(
            "hf_stable_ids",
            "local/test-dataset",
            "default",
            "train",
            temp.path(),
        );
        config.shard_extensions = vec!["ndjson".to_string()];
        config.text_columns = vec!["text".to_string()];
        HuggingFaceRowSource::new(config).expect("failed creating source")
    };

    let seed = seeded_config(7);
    let snap1 = make_source()
        .refresh(&seed, None, Some(2))
        .expect("first instance refresh");
    let snap2 = make_source()
        .refresh(&seed, None, Some(2))
        .expect("second instance refresh");

    let ids1: std::collections::HashSet<String> =
        snap1.records.iter().map(|r| r.id.clone()).collect();
    let ids2: std::collections::HashSet<String> =
        snap2.records.iter().map(|r| r.id.clone()).collect();
    assert_eq!(
        ids1, ids2,
        "record IDs must be identical across independent source instances"
    );
}

#[test]
fn huggingface_cursor_advances_between_refreshes() {
    // The cursor returned from refresh() must cause the next call to start
    // from a different position, not restart from row 0 every time.
    let temp = tempfile::tempdir().expect("failed creating tempdir");
    write_lines(
        &temp.path().join("part-00000.ndjson"),
        &[
            r#"{"id":"adv1","text":"alpha"}"#,
            r#"{"id":"adv2","text":"beta"}"#,
            r#"{"id":"adv3","text":"gamma"}"#,
            r#"{"id":"adv4","text":"delta"}"#,
        ],
    );

    let mut config = config_no_auth(
        "hf_cursor_advance",
        "local/test-dataset",
        "default",
        "train",
        temp.path(),
    );
    config.shard_extensions = vec!["ndjson".to_string()];
    config.text_columns = vec!["text".to_string()];

    let source = HuggingFaceRowSource::new(config).expect("failed creating source");
    let seed = seeded_config(11);

    let snap1 = source.refresh(&seed, None, Some(2)).expect("first refresh");
    assert_eq!(snap1.records.len(), 2);

    let snap2 = source
        .refresh(&seed, Some(&snap1.cursor), Some(2))
        .expect("second refresh");
    assert_eq!(snap2.records.len(), 2);

    // The cursor must have advanced: the two batches should not be identical.
    let ids1: std::collections::HashSet<_> = snap1.records.iter().map(|r| &r.id).collect();
    let ids2: std::collections::HashSet<_> = snap2.records.iter().map(|r| &r.id).collect();
    assert_ne!(
        ids1, ids2,
        "cursor must advance between successive refreshes"
    );
}

#[test]
fn huggingface_refresh_limit_is_strictly_respected() {
    // refresh(..., Some(limit)) must never return more than `limit` records.
    let temp = tempfile::tempdir().expect("failed creating tempdir");
    write_lines(
        &temp.path().join("part-00000.ndjson"),
        &[
            r#"{"id":"lim1","text":"a"}"#,
            r#"{"id":"lim2","text":"b"}"#,
            r#"{"id":"lim3","text":"c"}"#,
            r#"{"id":"lim4","text":"d"}"#,
            r#"{"id":"lim5","text":"e"}"#,
        ],
    );

    let mut config = config_no_auth(
        "hf_limit",
        "local/test-dataset",
        "default",
        "train",
        temp.path(),
    );
    config.shard_extensions = vec!["ndjson".to_string()];
    config.text_columns = vec!["text".to_string()];

    let source = HuggingFaceRowSource::new(config).expect("failed creating source");
    let seed = seeded_config(13);

    let snap = source
        .refresh(&seed, None, Some(2))
        .expect("refresh should succeed");
    assert!(
        snap.records.len() <= 2,
        "refresh returned {} records but limit was 2",
        snap.records.len()
    );
}

#[test]
fn huggingface_both_local_shards_are_sampled() {
    // When two shard files exist, records from both must appear across a
    // full refresh so that shard expansion actually increases coverage.
    let temp = tempfile::tempdir().expect("failed creating tempdir");
    write_lines(
        &temp.path().join("part-00000.ndjson"),
        &[
            r#"{"id":"s1r1","text":"shard one row one"}"#,
            r#"{"id":"s1r2","text":"shard one row two"}"#,
        ],
    );
    write_lines(
        &temp.path().join("part-00001.ndjson"),
        &[
            r#"{"id":"s2r1","text":"shard two row one"}"#,
            r#"{"id":"s2r2","text":"shard two row two"}"#,
        ],
    );

    let mut config = config_no_auth(
        "hf_two_shards",
        "local/test-dataset",
        "default",
        "train",
        temp.path(),
    );
    config.shard_extensions = vec!["ndjson".to_string()];
    config.text_columns = vec!["text".to_string()];

    let source = HuggingFaceRowSource::new(config).expect("failed creating source");
    let seed = seeded_config(17);

    let snap = source
        .refresh(&seed, None, Some(100))
        .expect("refresh should read from both shards");

    let ids: std::collections::HashSet<String> =
        snap.records.iter().map(|r| r.id.clone()).collect();
    assert!(
        ids.iter().any(|id| id.contains("s1r")),
        "no records from shard 1 found in refresh output"
    );
    assert!(
        ids.iter().any(|id| id.contains("s2r")),
        "no records from shard 2 found in refresh output"
    );
}

#[test]
fn huggingface_refresh_wraps_correctly_after_all_rows_consumed() {
    // After reading all materialized rows, a second refresh must wrap the
    // cursor back to the start and return records — it must not panic or
    // return an empty snapshot.
    let temp = tempfile::tempdir().expect("failed creating tempdir");
    write_lines(
        &temp.path().join("part-00000.ndjson"),
        &[
            r#"{"id":"w1","text":"row one"}"#,
            r#"{"id":"w2","text":"row two"}"#,
        ],
    );

    let mut config = config_no_auth(
        "hf_wrap_after_exhaustion",
        "local/test-dataset",
        "default",
        "train",
        temp.path(),
    );
    config.shard_extensions = vec!["ndjson".to_string()];
    config.text_columns = vec!["text".to_string()];

    let source = HuggingFaceRowSource::new(config).expect("failed creating source");
    let seed = seeded_config(19);

    // First pass: consume all rows.
    let snap1 = source
        .refresh(&seed, None, Some(100))
        .expect("first refresh");
    assert_eq!(snap1.records.len(), 2);

    // Second pass with returning cursor: must wrap and return records.
    let snap2 = source
        .refresh(&seed, Some(&snap1.cursor), Some(1))
        .expect("second refresh after exhaustion must not fail");
    assert_eq!(
        snap2.records.len(),
        1,
        "refresh must still return records after cursor wraps"
    );
}

#[test]
fn huggingface_different_epoch_seeds_produce_different_record_orderings() {
    // Guarantee: calling refresh() with a seed derived from epoch N must yield
    // a different record ordering than epoch 0, end-to-end through the HF
    // source's paging path.
    //
    // How the epoch seed reaches here: IngestionManager calls
    // source.refresh(&epoch_config, ...) where epoch_config.seed =
    // derive_epoch_seed(base_seed, epoch) = base_seed ^ epoch.  This test
    // exercises that same contract directly on HuggingFaceRowSource so that
    // if someone ever breaks the paging_seed → IndexPermutation chain the
    // failure shows up specifically in the HF source, not just in the sampler.
    //
    // Assertions mirror the sampler-level test:
    //   (a) Both epochs return every record exactly once (none lost/gained).
    //   (b) The full ordered ID sequences differ.
    //   (c) The first-half record sets differ — real positional movement,
    //       not just tail re-ordering.
    //
    // 20 rows gives 20! / (20!/2) ≈ 1 in 2 chance any two permutations
    // collide by accident; with a fixed seed the result is fully deterministic.
    let temp = tempfile::tempdir().expect("failed creating tempdir");

    let n_records: usize = 20;
    let rows: Vec<String> = (0..n_records)
        .map(|i| format!(r#"{{"id":"epoch_row_{i:02}","text":"body {i}"}}"#))
        .collect();
    let row_refs: Vec<&str> = rows.iter().map(|s| s.as_str()).collect();
    write_lines(&temp.path().join("part-00000.ndjson"), &row_refs);

    let mut config = config_no_auth(
        "hf_epoch_order",
        "local/test-dataset",
        "default",
        "train",
        temp.path(),
    );
    config.shard_extensions = vec!["ndjson".to_string()];
    config.text_columns = vec!["text".to_string()];

    let source = HuggingFaceRowSource::new(config).expect("failed creating source");

    // epoch 0: derive_epoch_seed(base, 0) = base ^ 0 = base
    let base_seed: u64 = 0xC0FFEE;
    let epoch0_seed = base_seed;
    let epoch1_seed = base_seed ^ 1;

    let epoch0_snapshot = source
        .refresh(&seeded_config(epoch0_seed), None, Some(n_records))
        .expect("epoch-0 refresh must succeed");
    let epoch1_snapshot = source
        .refresh(&seeded_config(epoch1_seed), None, Some(n_records))
        .expect("epoch-1 refresh must succeed");

    let epoch0_ids: Vec<String> = epoch0_snapshot
        .records
        .iter()
        .map(|r| r.id.clone())
        .collect();
    let epoch1_ids: Vec<String> = epoch1_snapshot
        .records
        .iter()
        .map(|r| r.id.clone())
        .collect();

    // (a) Both epochs must cover exactly the same set of records.
    let mut epoch0_ids_sorted = epoch0_ids.clone();
    let mut epoch1_ids_sorted = epoch1_ids.clone();
    epoch0_ids_sorted.sort();
    epoch1_ids_sorted.sort();
    assert_eq!(
        epoch0_ids_sorted, epoch1_ids_sorted,
        "epoch 1 is missing or duplicating records relative to epoch 0"
    );

    // (b) The full ordered sequences must differ.
    assert_ne!(
        epoch0_ids, epoch1_ids,
        "epoch-0 and epoch-1 seeds produced identical record orderings — \
         the epoch seed has no effect on HuggingFaceRowSource paging"
    );

    // (c) The early draws must differ as a set.
    let half = n_records / 2;
    let epoch0_first_half: std::collections::HashSet<&str> =
        epoch0_ids[..half].iter().map(String::as_str).collect();
    let epoch1_first_half: std::collections::HashSet<&str> =
        epoch1_ids[..half].iter().map(String::as_str).collect();
    assert_ne!(
        epoch0_first_half, epoch1_first_half,
        "the first {half} records returned by epoch-0 and epoch-1 are the same set — \
         records are not actually changing position across epochs"
    );

    // (d) Each epoch seed must be deterministic: calling refresh() again with
    //     the same seed must return records in the exact same order.  Without
    //     this a random shuffle would satisfy (b) and (c) by accident.
    let epoch0_snapshot_replay = source
        .refresh(&seeded_config(epoch0_seed), None, Some(n_records))
        .expect("epoch-0 second refresh must succeed");
    let epoch1_snapshot_replay = source
        .refresh(&seeded_config(epoch1_seed), None, Some(n_records))
        .expect("epoch-1 second refresh must succeed");

    let epoch0_ids_replay: Vec<String> = epoch0_snapshot_replay
        .records
        .iter()
        .map(|r| r.id.clone())
        .collect();
    let epoch1_ids_replay: Vec<String> = epoch1_snapshot_replay
        .records
        .iter()
        .map(|r| r.id.clone())
        .collect();

    assert_eq!(
        epoch0_ids, epoch0_ids_replay,
        "epoch-0 seed must produce the same record ordering on every refresh call"
    );
    assert_eq!(
        epoch1_ids, epoch1_ids_replay,
        "epoch-1 seed must produce the same record ordering on every refresh call"
    );
}

#[test]
fn huggingface_empty_split_discovers_all_splits() {
    // Guarantee: when `split` is an empty string (no split component in the URI),
    // refresh() must return records from every split present in the snapshot
    // directory, not just "train".  Triplets' own train/validation/test split
    // logic is responsible for the actual train/val/test partitioning downstream.
    let temp = tempfile::tempdir().expect("failed creating tempdir");

    // Write shards that look like they belong to different splits.
    write_lines(
        &temp.path().join("train-part-000.ndjson"),
        &[
            r#"{"id":"train1","text":"training record one"}"#,
            r#"{"id":"train2","text":"training record two"}"#,
        ],
    );
    write_lines(
        &temp.path().join("validation-part-000.ndjson"),
        &[
            r#"{"id":"val1","text":"validation record one"}"#,
            r#"{"id":"val2","text":"validation record two"}"#,
        ],
    );
    write_lines(
        &temp.path().join("test-part-000.ndjson"),
        &[r#"{"id":"test1","text":"test record one"}"#],
    );

    // Empty split string = all-splits mode.
    let mut config = config_no_auth(
        "hf_empty_split",
        "local/test-dataset",
        "default",
        "",
        temp.path(),
    );
    config.shard_extensions = vec!["ndjson".to_string()];
    config.text_columns = vec!["text".to_string()];

    let source = HuggingFaceRowSource::new(config).expect("failed creating source");
    let seed = seeded_config(71);

    let snapshot = source
        .refresh(&seed, None, Some(100))
        .expect("empty-split refresh must succeed");

    let ids: std::collections::HashSet<String> =
        snapshot.records.iter().map(|r| r.id.clone()).collect();

    assert!(
        ids.iter()
            .any(|id| id.contains("train1") || id.contains("train2")),
        "no train records found in empty-split refresh"
    );
    assert!(
        ids.iter()
            .any(|id| id.contains("val1") || id.contains("val2")),
        "no validation records found in empty-split refresh"
    );
    assert!(
        ids.iter().any(|id| id.contains("test1")),
        "no test records found in empty-split refresh"
    );
}

#[test]
#[ignore = "network integration test against live Hugging Face dataset"]
fn huggingface_reads_live_remote_dataset() {
    let temp = tempfile::tempdir().expect("failed creating tempdir");

    let mut config = config_no_auth(
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

#[test]
#[ignore = "network integration test — verifies /size endpoint returns a plausible row count"]
fn huggingface_live_size_endpoint_reports_dataset_row_count() {
    // Exercises the datasets-server /size endpoint called inside new().
    // If the endpoint changes its response schema, fetch_global_row_count()
    // returns None and known_total_rows() will be None — the first assert fails.
    // If the endpoint returns an implausibly small count for a well-known dataset
    // the second assert fails.
    //
    // Each test owns its own tempdir so snapshots never pollute each other or
    // the library's own managed cache directory.
    let temp = tempfile::tempdir().expect("failed creating tempdir");

    let mut config = config_no_auth(
        "hf_live_size_endpoint",
        "cornell-movie-review-data/rotten_tomatoes",
        "default",
        "train",
        temp.path(),
    );
    config.text_columns = vec!["text".to_string()];

    let source = HuggingFaceRowSource::new(config).expect("failed creating source");

    let total = source.known_total_rows();
    assert!(
        total.is_some_and(|t| t > 0),
        "/size endpoint should report a positive row count for rotten_tomatoes; \
         got {:?} — endpoint may have changed response format or become unreachable",
        total
    );
    // The rotten_tomatoes train split has 8530 rows per its dataset card.
    // A significantly different number indicates the endpoint contract changed.
    let rows = total.unwrap();
    assert!(
        rows > 1_000,
        "rotten_tomatoes train reports {rows} rows; expected > 1000 — \
         /size endpoint response format may have changed"
    );
}

#[test]
#[ignore = "network integration test — verifies /info endpoint ClassLabel resolution end-to-end"]
fn huggingface_live_classlabel_resolution_maps_integers_to_label_strings() {
    // Exercises three live endpoints in sequence:
    //   1. /info  — called in new() to resolve ClassLabel column names
    //   2. /parquet — called in refresh() to obtain the shard manifest
    //   3. HF CDN — actual parquet shard download URL
    //
    // TimKoornstra/financial-tweets-sentiment has a `sentiment` column declared
    // as ClassLabel with names = ["neutral", "bullish", "bearish"].  In the
    // parquet file the column is stored as integer (0/1/2).
    //
    // With successful /info resolution the transcoded records will contain the
    // label strings.  Without it (endpoint unreachable or format changed) they
    // will contain raw integer strings.  The final assertion distinguishes the
    // two cases and fails on any regression in the /info endpoint contract.
    //
    // snapshot_dir is a fresh tempdir — no shared cache is used.
    let temp = tempfile::tempdir().expect("failed creating tempdir");

    let mut config = config_no_auth(
        "hf_live_classlabel",
        "TimKoornstra/financial-tweets-sentiment",
        "default",
        "train",
        temp.path(),
    );
    // Using sentiment as the sole text column so every record's text IS the
    // resolved label.  Without ClassLabel resolution the text would be "0",
    // "1", or "2" instead of a named label.
    config.text_columns = vec!["sentiment".to_string()];

    let source = HuggingFaceRowSource::new(config).expect("failed creating source");
    let seed = seeded_config(43);
    let snapshot = source
        .refresh(&seed, None, Some(5))
        .expect("refresh should download and read live rows");

    assert!(
        !snapshot.records.is_empty(),
        "expected records from TimKoornstra/financial-tweets-sentiment"
    );

    const KNOWN_LABELS: &[&str] = &["neutral", "bullish", "bearish"];
    for record in &snapshot.records {
        for section in &record.sections {
            let text = section.text.as_str();
            assert!(
                KNOWN_LABELS.contains(&text),
                "/info ClassLabel resolution failed: expected one of {KNOWN_LABELS:?} \
                 but got {text:?} — raw integers (\"0\"/\"1\"/\"2\") indicate the /info \
                 endpoint no longer returns a 'names' array for this ClassLabel column"
            );
        }
    }
}

// ── trust= and source_id= column tests ─────────────────────────────────────

#[test]
fn parse_hf_source_line_accepts_trust_override() {
    let entry =
        parse_hf_source_line("hf://org/dataset/default/train anchor=title positive=text trust=0.8")
            .expect("line with trust= should parse");
    assert_eq!(entry.uri, "hf://org/dataset/default/train");
    assert_eq!(entry.anchor_columns, vec!["title".to_string()]);
    assert_eq!(entry.positive_columns, vec!["text".to_string()]);
    let trust = entry.trust.expect("trust should be Some");
    assert!(
        (trust - 0.8_f32).abs() < f32::EPSILON,
        "trust should be 0.8, got {trust}"
    );
}

#[test]
fn parse_hf_source_line_accepts_trust_boundary_values() {
    let zero = parse_hf_source_line("hf://org/dataset/default/train text=body trust=0.0")
        .expect("trust=0.0 should parse");
    assert_eq!(zero.trust, Some(0.0_f32));

    let one = parse_hf_source_line("hf://org/dataset/default/train text=body trust=1.0")
        .expect("trust=1.0 should parse");
    assert_eq!(one.trust, Some(1.0_f32));
}

#[test]
fn parse_hf_source_line_rejects_trust_above_one() {
    let err = parse_hf_source_line("hf://org/dataset/default/train text=body trust=1.1")
        .expect_err("trust > 1.0 should be rejected");
    assert!(
        err.contains("out of range"),
        "expected 'out of range' in error message, got: {err}"
    );
}

#[test]
fn parse_hf_source_line_rejects_negative_trust() {
    let err = parse_hf_source_line("hf://org/dataset/default/train text=body trust=-0.1")
        .expect_err("negative trust should be rejected");
    assert!(
        err.contains("out of range"),
        "expected 'out of range' in error message, got: {err}"
    );
}

#[test]
fn parse_hf_source_line_rejects_non_float_trust() {
    let err = parse_hf_source_line("hf://org/dataset/default/train text=body trust=high")
        .expect_err("non-float trust value should be rejected");
    assert!(
        err.contains("invalid trust value"),
        "expected 'invalid trust value' in error message, got: {err}"
    );
}

#[test]
fn parse_hf_source_line_accepts_source_id_override() {
    let entry =
        parse_hf_source_line("hf://org/dataset/default/train text=body source_id=my_custom_source")
            .expect("line with source_id= should parse");
    assert_eq!(
        entry.source_id,
        Some("my_custom_source".to_string()),
        "source_id should be overridden"
    );
}

#[test]
fn parse_hf_source_line_rejects_empty_source_id() {
    let err = parse_hf_source_line("hf://org/dataset/default/train text=body source_id=")
        .expect_err("empty source_id should be rejected");
    assert!(
        err.contains("source_id must not be empty"),
        "expected 'source_id must not be empty' in error message, got: {err}"
    );
}

#[test]
fn parse_hf_source_line_accepts_trust_and_source_id_together() {
    let entry = parse_hf_source_line(
        "hf://org/dataset/default/train anchor=title positive=text trust=0.9 source_id=wiki-en",
    )
    .expect("line with both trust= and source_id= should parse");
    let trust = entry.trust.expect("trust should be Some");
    assert!(
        (trust - 0.9_f32).abs() < f32::EPSILON,
        "trust should be 0.9, got {trust}"
    );
    assert_eq!(entry.source_id, Some("wiki-en".to_string()));
}

#[test]
fn parse_hf_source_line_defaults_trust_and_source_id_to_none() {
    let entry = parse_hf_source_line("hf://org/dataset/default/train anchor=title")
        .expect("line without trust= or source_id= should parse");
    assert!(entry.trust.is_none(), "trust should default to None");
    assert!(
        entry.source_id.is_none(),
        "source_id should default to None"
    );
}

#[test]
fn parse_hf_source_line_rejects_unknown_key_typo() {
    // "positve" is a typo for "positive" — the parser must reject it, not silently ignore
    // or fall back to a default, since accepting unknown keys would mask configuration mistakes.
    let err = parse_hf_source_line("hf://org/dataset/default/train positve=body")
        .expect_err("typo key 'positve' should be rejected");
    assert!(
        err.contains("unsupported mapping key") && err.contains("positve"),
        "expected 'unsupported mapping key' mentioning 'positve', got: {err}"
    );
}

#[test]
fn parse_hf_source_line_rejects_arbitrary_unknown_key() {
    let err = parse_hf_source_line("hf://org/dataset/default/train iajfaijww=body")
        .expect_err("arbitrary unknown key 'iajfaijww' should be rejected");
    assert!(
        err.contains("unsupported mapping key") && err.contains("iajfaijww"),
        "expected 'unsupported mapping key' mentioning 'iajfaijww', got: {err}"
    );
}

#[test]
fn huggingface_rows_config_trust_override_propagates_to_records() {
    let temp = tempfile::tempdir().expect("failed creating tempdir");
    let shard_path = temp.path().join("part-00000.ndjson");
    write_lines(
        &shard_path,
        &[
            r#"{"id":"t1","text":"alpha content"}"#,
            r#"{"id":"t2","text":"beta content"}"#,
        ],
    );

    let mut config = config_no_auth(
        "hf_trust_override",
        "local/test-dataset",
        "default",
        "train",
        temp.path(),
    );
    config.shard_extensions = vec!["ndjson".to_string()];
    config.text_columns = vec!["text".to_string()];
    config.trust_override = Some(0.9);

    let source = HuggingFaceRowSource::new(config).expect("failed creating huggingface source");
    let seed = seeded_config(53);
    let snapshot = source
        .refresh(&seed, None, Some(2))
        .expect("refresh should succeed");

    assert_eq!(snapshot.records.len(), 2, "expected 2 records");
    for record in &snapshot.records {
        assert!(
            (record.quality.trust - 0.9_f32).abs() < f32::EPSILON,
            "expected trust 0.9, got {}",
            record.quality.trust
        );
    }
}

#[test]
fn huggingface_rows_config_without_trust_override_uses_default_trust() {
    let temp = tempfile::tempdir().expect("failed creating tempdir");
    let shard_path = temp.path().join("part-00000.ndjson");
    write_lines(&shard_path, &[r#"{"id":"u1","text":"content"}"#]);

    let mut config = config_no_auth(
        "hf_default_trust",
        "local/test-dataset",
        "default",
        "train",
        temp.path(),
    );
    config.shard_extensions = vec!["ndjson".to_string()];
    config.text_columns = vec!["text".to_string()];
    // trust_override not set → should default to QualityScore::default().trust (0.5)

    let source = HuggingFaceRowSource::new(config).expect("failed creating huggingface source");
    let seed = seeded_config(59);
    let snapshot = source
        .refresh(&seed, None, Some(1))
        .expect("refresh should succeed");

    assert_eq!(snapshot.records.len(), 1);
    assert!(
        (snapshot.records[0].quality.trust - 0.5_f32).abs() < f32::EPSILON,
        "expected default trust 0.5, got {}",
        snapshot.records[0].quality.trust
    );
}

#[test]
fn huggingface_source_list_file_parses_trust_and_source_id() {
    let temp = tempfile::tempdir().expect("failed creating tempdir");
    let list_path = temp.path().join("hf_sources_trust.txt");

    let nl = platform_newline();
    fs::write(
        &list_path,
        format!(
            "# sources with trust and source_id overrides{nl}\
             hf://org/dataset-a/default/train anchor=title positive=text trust=0.9 source_id=dataset-a-train{nl}\
             hf://org/dataset-b/default/train text=body trust=0.3{nl}\
             hf://org/dataset-c/default/train text=content source_id=my-corpus{nl}"
        ),
    )
    .expect("failed writing source list");

    let entries = load_hf_sources_from_list(list_path.to_str().expect("utf8 path"))
        .expect("failed parsing source list");

    assert_eq!(entries.len(), 3);

    // Entry 0: both trust and source_id
    let e0 = &entries[0];
    assert_eq!(e0.uri, "hf://org/dataset-a/default/train");
    assert_eq!(e0.source_id, Some("dataset-a-train".to_string()));
    let t0 = e0.trust.expect("entry 0 should have trust");
    assert!(
        (t0 - 0.9_f32).abs() < f32::EPSILON,
        "entry 0 trust should be 0.9, got {t0}"
    );

    // Entry 1: trust only
    let e1 = &entries[1];
    assert_eq!(e1.uri, "hf://org/dataset-b/default/train");
    assert!(e1.source_id.is_none(), "entry 1 source_id should be None");
    let t1 = e1.trust.expect("entry 1 should have trust");
    assert!(
        (t1 - 0.3_f32).abs() < f32::EPSILON,
        "entry 1 trust should be 0.3, got {t1}"
    );

    // Entry 2: source_id only
    let e2 = &entries[2];
    assert_eq!(e2.uri, "hf://org/dataset-c/default/train");
    assert_eq!(e2.source_id, Some("my-corpus".to_string()));
    assert!(e2.trust.is_none(), "entry 2 trust should be None");
}

// ── Live network test: private dataset authentication ─────────────────────────
//
// This test requires *both* of the following environment variables to be set:
//
//   HF_TOKEN                       — a Hugging Face API token with read scope.
//   TRIPLETS_HF_TOKEN_TEST_DATASET — the dataset repo to access, e.g.
//                                    "my-org/my-private-test-dataset".
//
// If either variable is absent the test *fails* — this is intentional so that
// the test is not accidentally omitted from a run that is expected to exercise
// live credentials.  To opt out explicitly (e.g. in a base CI job that has no
// HF credentials), set TRIPLETS_SKIP_LIVE_TESTS=1; the test will then be
// skipped silently rather than failing.
//
// ── Reproducing this test ─────────────────────────────────────────────────────
//
// 1. Create a private Hugging Face dataset with a Parquet shard that has at
//    least two string columns named `a` and `b`.  A minimal example using the
//    Python `datasets` library:
//
//      from datasets import Dataset
//      import pandas as pd
//
//      df = pd.DataFrame({
//          "a": ["hello world", "foo bar"],
//          "b": ["baz qux",    "quux corge"],
//      })
//      ds = Dataset.from_pandas(df)
//      ds.push_to_hub("my-org/my-private-test-dataset", private=True)
//
// 2. Generate a read-scoped token at https://huggingface.co/settings/tokens.
//
// 3. Set the environment variables and run the test:
//
//    macOS / Linux:
//      export HF_TOKEN="hf_..."
//      export TRIPLETS_HF_TOKEN_TEST_DATASET="my-org/my-private-test-dataset"
//      cargo test --features huggingface hf_token_private_dataset_access -- --nocapture
//
//    Windows PowerShell:
//      $env:HF_TOKEN = "hf_..."
//      $env:TRIPLETS_HF_TOKEN_TEST_DATASET = "my-org/my-private-test-dataset"
//      cargo test --features huggingface hf_token_private_dataset_access -- --nocapture
//
// To suppress the test in a CI environment that has no HF credentials, set:
//      TRIPLETS_SKIP_LIVE_TESTS=1
// The test will skip silently instead of failing.
//
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn hf_token_private_dataset_access() {
    // ── Guard: require env vars (or explicit opt-out) ────────────────────────
    //
    // If TRIPLETS_SKIP_LIVE_TESTS is set to any non-empty value, missing
    // credentials produce a silent skip.  Otherwise missing credentials are a
    // hard failure so the test cannot be accidentally omitted.

    let skip_live = std::env::var(TRIPLETS_SKIP_LIVE_TESTS)
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false);

    let token = match std::env::var(HF_TOKEN) {
        Ok(t) if !t.trim().is_empty() => t,
        _ => {
            if skip_live {
                eprintln!(
                    "[skip] HF_TOKEN not set and TRIPLETS_SKIP_LIVE_TESTS is active — \
                     skipping private dataset integration test."
                );
                return;
            }
            panic!(
                "HF_TOKEN is not set. This test requires a valid Hugging Face API token. \
                 Set HF_TOKEN to run it, or set TRIPLETS_SKIP_LIVE_TESTS=1 to skip it. \
                 See the comment above this test for setup instructions."
            );
        }
    };

    let dataset = match std::env::var(ENV_TRIPLETS_HF_TOKEN_TEST_DATASET) {
        Ok(d) if !d.trim().is_empty() => d,
        _ => {
            if skip_live {
                eprintln!(
                    "[skip] {} not set and TRIPLETS_SKIP_LIVE_TESTS is active — \
                     skipping private dataset integration test.",
                    ENV_TRIPLETS_HF_TOKEN_TEST_DATASET
                );
                return;
            }
            panic!(
                "{} is not set. This test requires a private HF dataset repo. \
                 Set it to run the test, or set TRIPLETS_SKIP_LIVE_TESTS=1 to skip it. \
                 See the comment above this test for setup instructions.",
                ENV_TRIPLETS_HF_TOKEN_TEST_DATASET
            );
        }
    };

    // ── Build configuration ───────────────────────────────────────────────────
    //
    // Use text mode: coalesce column "a" first, then fall back to "b".
    // Leave the split empty so all HF splits are discovered automatically.

    let temp = tempfile::tempdir().expect("failed creating tempdir");

    let mut config = config_no_auth(
        "hf_token_test",
        dataset.trim(),
        "default",
        "", // empty → discover all splits
        temp.path(),
    );
    config.text_columns = vec!["a".to_string(), "b".to_string()];
    // Supply the token explicitly so this test is self-contained regardless of
    // whether HF_TOKEN was already picked up by the constructor.
    config.hf_token = Some(token);

    // ── Construction: validates token immediately ─────────────────────────────
    //
    // `new()` calls the HF whoami endpoint before any data work.  A bad token
    // returns Err here, making the failure obvious rather than silent.

    let source = HuggingFaceRowSource::new(config).expect(
        "HuggingFaceRowSource::new should succeed with a valid token and accessible dataset",
    );

    // ── Data access ───────────────────────────────────────────────────────────

    let sampler_cfg = seeded_config(42);

    let count = source
        .reported_record_count(&sampler_cfg)
        .expect("reported_record_count should succeed for an accessible private dataset");
    assert!(
        count > 0,
        "expected at least one record from the test dataset, got {count}"
    );

    let snapshot = source
        .refresh(&sampler_cfg, None, Some(count.min(16) as usize))
        .expect("refresh should succeed for an accessible private dataset");
    assert!(
        !snapshot.records.is_empty(),
        "expected a non-empty snapshot from the test dataset"
    );

    // At least one record must have non-empty text in column a or b.
    let has_content = snapshot.records.iter().any(|record| {
        record
            .sections
            .iter()
            .any(|section| !section.text.trim().is_empty())
    });
    assert!(
        has_content,
        "expected at least one record with non-empty text content from columns a or b"
    );
}

#[test]
#[serial_test::serial]
fn sampler_next_text_batch_re_expands_after_cache_eviction() {
    // E2E test verifying that when the cache manager automatically evicts
    // shards during background expansion, the source re-fetches the HF
    // parquet manifest and re-downloads evicted shards on the next cycle.
    //
    // The test uses ONLY the public Sampler API (`next_text_batch`) and
    // exercises the full production code path:
    //   next_text_batch -> Sampler -> source.refresh -> trigger_expansion_if_needed
    //   -> expansion thread -> download_next_remote_shard
    //   -> enforce_disk_cap_locked -> sync_shard_state_from_disk_locked
    //   -> remote_candidates nulled -> next cycle re-fetches manifest
    //
    // The mock server's manifest counter proves the manifest is re-queried.

    let temp = tempfile::tempdir().expect("tempdir");

    // ── Mock HF server ───────────────────────────────────────────────────
    //
    // Returns a manifest listing 5 shards, each containing 1 row of text.
    // The tight cap (room for ~2 shards) causes automatic eviction once
    // more than 2 shards have been downloaded.
    //
    // Shard payload: {"id":"s{shard}_r{row}","text":"txt_{shard}_{row}"}
    // Roughly 33 bytes each.  2 shards ≈ 66 bytes, cap = 70 bytes.
    // The manifest server also counts how many times /parquet is queried.
    let server = triplets_hf_source::test_utils::HfMockServer::new(5, 1);

    // ── Env var guards ──────────────────────────────────────────────────
    //
    // Set env vars for the test duration.  The triggers MUST outlive
    // the source and sampler so that async expansion threads can still
    // resolve the mock endpoints when they make HTTP requests.
    let _parquet_guard = triplets_hf_source::test_utils::EnvGuard::set(
        ENV_TRIPLETS_HF_PARQUET_ENDPOINT,
        &format!("{}/parquet", server.url()),
    );
    // The /size and /info endpoints are NOT mocked; failing to query them
    // is non-fatal (warns and returns None).  Point them somewhere harmless.
    let _size_guard = triplets_hf_source::test_utils::EnvGuard::set(
        ENV_TRIPLETS_HF_SIZE_ENDPOINT,
        "http://127.0.0.1:1/unreachable",
    );
    let _info_guard = triplets_hf_source::test_utils::EnvGuard::set(
        ENV_TRIPLETS_HF_INFO_ENDPOINT,
        "http://127.0.0.1:1/unreachable",
    );

    // ── Source ───────────────────────────────────────────────────────────
    let mut config = HuggingFaceRowsConfig::new(
        "hf_re_expand_test",
        "org/dataset",
        "default",
        "train",
        temp.path(),
    );
    config.hf_token = None;
    config.text_columns = vec!["text".to_string()];
    // Tight cap: room for ~2 shards.  After 3+ shards are downloaded the
    // oldest get evicted by the cache manager.
    config.local_disk_cap_bytes = Some(70);
    // Small capacity so the in-memory row cache doesn't mask any eviction.
    config.cache_capacity = 2;

    let source = HuggingFaceRowSource::new(config).expect("failed creating HuggingFaceRowSource");

    // ── Sampler ──────────────────────────────────────────────────────────
    let split_store =
        Arc::new(DeterministicSplitStore::new(SplitRatios::default(), 42).expect("split store"));
    let sampler_config = SamplerConfig {
        batch_size: 1,
        ingestion_max_records: 10,
        seed: 1,
        allowed_splits: vec![SplitLabel::Train],
        ..SamplerConfig::default()
    };
    let sampler = TripletSampler::new(sampler_config, split_store);
    sampler.register_source(Box::new(source));

    // ── Drain shards through next_text_batch ────────────────────────────
    //
    // Each call to `next_text_batch`:
    //   1. Calls `source.refresh()` which bootsraps (materialized_rows == 0)
    //      → fetches manifest → downloads shard 0.
    //   2. Reads rows from materialized shards.
    //   3. Calls `trigger_expansion_if_needed()` which spawns a thread to
    //      download the next shard in the background → cap exceeded → old
    //      shard evicted → `remote_candidates` nulled.
    //
    // The expansion thread is async so we call many times, allowing the
    // background downloads to complete between iterations.
    let total_batches = 20;
    let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut empty_batches = 0usize;
    let mut errors = 0usize;

    for i in 0..total_batches {
        // Brief sleep so the async expansion thread from the PRIOR call
        // has time to finish downloading the shard before we read again.
        std::thread::sleep(std::time::Duration::from_millis(50));

        let batch = match sampler.next_text_batch(SplitLabel::Train) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("[iter {i}] next_text_batch error: {e}");
                errors += 1;
                continue;
            }
        };

        if batch.samples.is_empty() {
            empty_batches += 1;
            continue;
        }

        for sample in &batch.samples {
            seen_ids.insert(sample.chunk.record_id.to_string());
        }
    }

    // The sampler may return empty or error batches while the source is
    // expanding.  But we must have seen at least some unique record IDs
    // from eviction-surviving shards.
    let survivors = seen_ids.len();
    eprintln!(
        "results: {total_batches} iters, {survivors} unique ids, {empty_batches} empty, {errors} errors, ids={seen_ids:?}"
    );
    assert!(
        survivors >= 1,
        "no unique record IDs observed — expansion never produced surviving rows; \
         empty={empty_batches} errors={errors}"
    );

    // ── Verify the manifest was re-fetched after eviction ───────────────
    //
    // The counter increments every time a /parquet request lands on the
    // mock server.  The bootstrap path fetches it once; re-expansion after
    // eviction fetches it again.  We expect >= 2.
    let fetch_count = server.manifest_fetch_count();
    eprintln!("parquet manifest fetched {fetch_count} times (expected >= 2)");
    assert!(
        fetch_count >= 2,
        "parquet manifest must be re-fetched after eviction-driven re-expansion; \
         fetched {fetch_count} times, expected >= 2",
    );

    server.shut_down();
}
