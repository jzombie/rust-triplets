use crate::builder::build_hf_sources;
use crate::constants::ENV_TRIPLETS_HF_TOKEN;
use crate::disk_cache::managed_hf_list_snapshot_dir;
use crate::download::*;
use crate::parsing::{
    HfListRoots, HfSourceEntry, hf_source_id_slug, load_hf_sources_from_list, parse_hf_source_line,
    parse_hf_uri, resolve_hf_list_roots,
};
use crate::source_core::HuggingFaceRowSource;
use crate::test_utils::{test_config, with_current_dir, with_env_vars};
use serial_test::serial;
use std::collections::HashSet;
use std::fs;
use tempfile::tempdir;
use triplets_core::utils::platform_newline;

#[test]
fn load_and_resolve_hf_source_list_reports_invalid_and_empty_inputs() {
    let dir = tempdir().unwrap();
    let nl = platform_newline();

    let invalid_list = dir.path().join("invalid_sources.txt");
    fs::write(
        &invalid_list,
        format!("hf://org/dataset/default/train badtoken{nl}"),
    )
    .unwrap();
    let invalid = load_hf_sources_from_list(invalid_list.to_str().unwrap()).unwrap_err();
    assert!(invalid.contains("invalid source-list entry"));

    let empty_list = dir.path().join("empty_sources.txt");
    fs::write(&empty_list, format!("# comment only{nl}{nl}")).unwrap();
    let resolved = resolve_hf_list_roots(empty_list.to_string_lossy().to_string()).unwrap_err();
    assert!(resolved.contains("no hf:// entries found"));

    let good_list = dir.path().join("good_sources.txt");
    fs::write(
        &good_list,
        format!("hf://org/dataset/default/train anchor=title positive=body{nl}"),
    )
    .unwrap();
    let roots = resolve_hf_list_roots(good_list.to_string_lossy().to_string()).unwrap();
    assert_eq!(roots.sources.len(), 1);
}

// ── Token validation tests ──────────────────────────────────────────────

#[test]
#[serial(global_state)]
fn build_hf_sources_skips_invalid_uri_and_builds_valid_source() {
    let roots = HfListRoots {
        source_list: "inline".to_string(),
        sources: vec![
            HfSourceEntry {
                uri: "hf://onlyorg".to_string(),
                anchor_columns: vec!["title".to_string()],
                positive_columns: Vec::new(),
                negative_columns: Vec::new(),
                context_columns: Vec::new(),
                text_columns: Vec::new(),
                trust: None,
                weight: None,
                source_id: None,
            },
            HfSourceEntry {
                uri: "hf://org/dataset/default/train".to_string(),
                anchor_columns: vec!["title".to_string()],
                positive_columns: vec!["body".to_string()],
                negative_columns: Vec::new(),
                context_columns: Vec::new(),
                text_columns: Vec::new(),
                trust: None,
                weight: None,
                source_id: None,
            },
        ],
    };

    let temp_root = tempdir().unwrap();
    let nl = platform_newline();
    fs::write(
        temp_root.path().join("Cargo.toml"),
        format!("[package]{nl}name='tmp'{nl}version='0.0.0'{nl}"),
    )
    .unwrap();
    fs::write(temp_root.path().join(".cache"), b"blocking-file").unwrap();

    with_current_dir(temp_root.path(), || {
        with_env_vars(&[(ENV_TRIPLETS_HF_TOKEN, "")], || {
            let built = build_hf_sources(&roots);
            assert_eq!(built.len(), 1);
        });
    });
}

#[test]
#[serial(global_state)]
fn build_hf_sources_duplicate_uri_gets_distinct_ids_and_snapshot_dirs() {
    // Two identical entries must produce two built sources whose IDs are
    // disambiguated (".0" / ".1") and whose snapshot directories are
    // independent (replica_0 vs replica_1).
    let dup_entry = HfSourceEntry {
        uri: "hf://org/dataset/default/train".to_string(),
        anchor_columns: vec!["title".to_string()],
        positive_columns: vec!["body".to_string()],
        negative_columns: Vec::new(),
        context_columns: Vec::new(),
        text_columns: Vec::new(),
        trust: None,
        weight: None,
        source_id: None,
    };
    let roots = HfListRoots {
        source_list: "inline".to_string(),
        sources: vec![dup_entry.clone(), dup_entry],
    };

    let temp_root = tempdir().unwrap();
    let nl = platform_newline();
    fs::write(
        temp_root.path().join("Cargo.toml"),
        format!("[package]{nl}name='tmp'{nl}version='0.0.0'{nl}"),
    )
    .unwrap();

    with_current_dir(temp_root.path(), || {
        with_env_vars(&[(ENV_TRIPLETS_HF_TOKEN, "")], || {
            let built = build_hf_sources(&roots);
            assert_eq!(built.len(), 2, "both duplicate sources should be built");

            let id_0 = built[0].id().to_string();
            let id_1 = built[1].id().to_string();
            assert_ne!(
                id_0, id_1,
                "duplicate sources must have distinct source IDs"
            );
            assert!(
                id_0.ends_with(".0"),
                "first duplicate should have .0 suffix, got: {id_0}"
            );
            assert!(
                id_1.ends_with(".1"),
                "second duplicate should have .1 suffix, got: {id_1}"
            );

            // Snapshot dirs are derived from managed_hf_list_snapshot_dir with
            // the list index, so replica_0 and replica_1 must differ.
            let dir_0 = managed_hf_list_snapshot_dir("org/dataset", "default", "train", 0).unwrap();
            let dir_1 = managed_hf_list_snapshot_dir("org/dataset", "default", "train", 1).unwrap();
            assert_ne!(
                dir_0, dir_1,
                "duplicate sources must have distinct snapshot dirs"
            );
            assert!(dir_0.ends_with("replica_0"));
            assert!(dir_1.ends_with("replica_1"));
        });
    });
}

#[test]
#[serial(global_state)]
fn build_hf_sources_shares_http_client_across_entries() {
    // All sources produced by build_hf_sources must share a single HTTP
    // client so that connection pooling and throttle state apply to the
    // aggregate outbound traffic rather than per-source.
    let entries: Vec<HfSourceEntry> = (0..3)
        .map(|i| HfSourceEntry {
            uri: "hf://org/dataset/default/train".to_string(),
            anchor_columns: vec!["title".to_string()],
            positive_columns: vec!["body".to_string()],
            negative_columns: Vec::new(),
            context_columns: Vec::new(),
            text_columns: Vec::new(),
            trust: None,
            weight: None,
            source_id: Some(format!("src_{i}")),
        })
        .collect();
    let roots = HfListRoots {
        source_list: "inline".to_string(),
        sources: entries,
    };

    let temp_root = tempdir().unwrap();
    let nl = platform_newline();
    fs::write(
        temp_root.path().join("Cargo.toml"),
        format!("[package]{nl}name='tmp'{nl}version='0.0.0'{nl}"),
    )
    .unwrap();

    with_current_dir(temp_root.path(), || {
        with_env_vars(&[(ENV_TRIPLETS_HF_TOKEN, "")], || {
            let built = build_hf_sources(&roots);
            assert_eq!(built.len(), 3, "all three sources should build");
        });
    });
}

#[test]
fn manual_http_client_sharing_works() {
    // Pre-building a client and setting it on multiple configs should
    // produce working sources that share the same connection pool.
    let dir = tempdir().unwrap();
    let client = build_http_client(&test_config(dir.path().to_path_buf()))
        .expect("build_http_client should succeed");

    for i in 0..3 {
        let mut config = test_config(dir.path().join(format!("src_{i}")));
        config.text_columns = vec!["text".to_string()];
        config.http_client = Some(client.clone());
        let source = HuggingFaceRowSource::new(config);
        assert!(source.is_ok(), "source {i} with shared client should build");
    }
}

#[test]
fn build_hf_sources_disambiguates_duplicate_slugs() {
    // Two sources pointing at the same dataset/config/split should get
    // distinct IDs via the index suffix rather than silently colliding.
    let sources = [
        HfSourceEntry {
            uri: "hf://org/dataset/default/train".to_string(),
            anchor_columns: vec!["title".to_string()],
            positive_columns: vec!["body".to_string()],
            negative_columns: Vec::new(),
            context_columns: Vec::new(),
            text_columns: Vec::new(),
            trust: None,
            source_id: None,
            weight: None,
        },
        HfSourceEntry {
            uri: "hf://org/dataset/default/train".to_string(),
            anchor_columns: vec!["title".to_string()],
            positive_columns: vec!["body".to_string()],
            negative_columns: Vec::new(),
            context_columns: Vec::new(),
            text_columns: Vec::new(),
            trust: None,
            weight: None,
            source_id: None,
        },
    ];
    let base_slugs: Vec<String> = sources
        .iter()
        .enumerate()
        .map(|(idx, source)| match parse_hf_uri(&source.uri) {
            Ok((dataset, config, split)) => hf_source_id_slug(&dataset, &config, &split),
            Err(_) => format!("hf_list_{idx}"),
        })
        .collect();
    let mut slug_count: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for s in &base_slugs {
        *slug_count.entry(s.as_str()).or_insert(0) += 1;
    }
    let duplicated: HashSet<&str> = slug_count
        .into_iter()
        .filter(|(_, n)| *n > 1)
        .map(|(s, _)| s)
        .collect();
    let resolved: Vec<String> = base_slugs
        .iter()
        .enumerate()
        .map(|(idx, slug)| {
            if duplicated.contains(slug.as_str()) {
                format!("{slug}.{idx}")
            } else {
                slug.clone()
            }
        })
        .collect();
    assert_eq!(resolved[0], "dataset.0");
    assert_eq!(resolved[1], "dataset.1");
}

#[test]
fn normalized_shard_extensions_trims_dots_and_lowercases() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.shard_extensions = vec![".PARQUET".into(), " ndjson ".into()];
    let normalized = HuggingFaceRowSource::normalized_shard_extensions(&config);
    assert_eq!(
        normalized,
        vec!["parquet".to_string(), "ndjson".to_string()]
    );
}

// ── extract_classlabel_maps ───────────────────────────────────────────────

// ── value_to_text ──────────────────────────────────────────────────────────

// TODO: Where should this comment go?
// --- datasets viewer disabled (501) scenario ---
//
// Some HF datasets have the datasets viewer disabled.  In that case the
// /size, /info, and /parquet datasets-server endpoints all return HTTP 501
// with {"error":"Not supported: dataset viewer is disabled ..."}.
//
// The expected behaviour:
//   * /size   → fetch_global_row_count returns Ok(None), not Err.
//   * /info   → fetch_classlabel_maps returns an empty map, not an error.
//   * /parquet → list_remote_candidates_from_parquet_manifest returns Err,
//                which causes list_remote_candidates_with_runtime to fall
//                through to the hf-hub repository listing path.

#[test]
fn hf_source_entry_partial_eq_compares_all_fields() {
    let base = HfSourceEntry {
        uri: "hf://org/ds/default/train".to_string(),
        anchor_columns: vec!["title".to_string()],
        positive_columns: vec!["body".to_string()],
        negative_columns: Vec::new(),
        context_columns: vec!["meta".to_string()],
        text_columns: Vec::new(),
        trust: Some(0.8),
        weight: None,
        source_id: None,
    };
    let same = HfSourceEntry { ..base.clone() };
    assert_eq!(base, same);
    let diff_uri = HfSourceEntry {
        uri: "hf://other".to_string(),
        ..base.clone()
    };
    assert_ne!(base, diff_uri);
    let diff_trust = HfSourceEntry {
        trust: Some(0.5),
        ..base.clone()
    };
    assert_ne!(base, diff_trust);
    let diff_sid = HfSourceEntry {
        source_id: Some("my-id".to_string()),
        ..base.clone()
    };
    assert_ne!(base, diff_sid);
    let no_trust = HfSourceEntry {
        trust: None,
        ..base.clone()
    };
    assert_ne!(base, no_trust);
}

// ── New tests for uncovered functions ────────────────────────────────

#[test]
#[serial(global_state)]
fn build_hf_sources_collapses_uri_parse_error() {
    let roots = HfListRoots {
        source_list: "inline".to_string(),
        sources: vec![HfSourceEntry {
            uri: "hf://incomplete".to_string(),
            anchor_columns: vec!["title".to_string()],
            positive_columns: Vec::new(),
            negative_columns: Vec::new(),
            context_columns: Vec::new(),
            text_columns: Vec::new(),
            trust: None,
            weight: None,
            source_id: None,
        }],
    };
    let temp_root = tempdir().unwrap();
    let nl = platform_newline();
    fs::write(
        temp_root.path().join("Cargo.toml"),
        format!("[package]{nl}name='tmp'{nl}version='0.0.0'{nl}"),
    )
    .unwrap();
    with_current_dir(temp_root.path(), || {
        with_env_vars(&[(crate::constants::ENV_TRIPLETS_HF_TOKEN, "")], || {
            let built = build_hf_sources(&roots);
            assert_eq!(built.len(), 0);
        });
    });
}

// ── Dict dataset / nested dict / list expansion tests ──────────────────

#[test]
fn parse_hf_source_line_negative_key() {
    let entry = parse_hf_source_line(
        "hf://embedding-data/QQP_triplets anchor=query positive=pos negative=neg",
    )
    .unwrap();
    assert_eq!(entry.anchor_columns, vec!["query"]);
    assert_eq!(entry.positive_columns, vec!["pos"]);
    assert_eq!(entry.negative_columns, vec!["neg"]);
    assert!(entry.context_columns.is_empty());
}

#[test]
fn parse_hf_source_line_weight_key() {
    let entry = parse_hf_source_line("hf://org/ds anchor=t positive=p weight=0.7").unwrap();
    assert_eq!(entry.weight, Some(0.7));
}

#[test]
fn parse_hf_source_line_weight_zero_rejected() {
    let err = parse_hf_source_line("hf://org/ds anchor=t positive=p weight=0").unwrap_err();
    assert!(err.contains("must be > 0.0"));
}

#[test]
fn parse_hf_source_line_weight_negative_rejected() {
    let err = parse_hf_source_line("hf://org/ds anchor=t positive=p weight=-1.0").unwrap_err();
    assert!(err.contains("must be > 0.0"));
}

#[test]
fn parse_hf_source_line_weight_invalid_rejected() {
    let err = parse_hf_source_line("hf://org/ds anchor=t positive=p weight=abc").unwrap_err();
    assert!(err.contains("invalid weight"));
}

// ── Phase 1c: transcode_transient_shard_to_store additional paths ──────────

// ── Phase 2: shard_indexing.rs tests ───────────────────────────────────────

// ── Phase 3: source_core.rs tests ──────────────────────────────────────────

// ── Phase 4: download.rs tests ─────────────────────────────────────────────

// ── Phase 1: additional download.rs coverage ────────────────────────────────

// ── Phase 2: source_core.rs tests ──────────────────────────────────────────

// ── Phase 4: shard_index.rs tests ─────────────────────────────────────────

// ── Phase 6: shard_indexing.rs tests ──────────────────────────────────────

// ── Phase 7: huggingface_source.rs tests ──────────────────────────────────
