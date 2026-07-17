use crate::parsing::{
    HfSourceEntry, load_hf_sources_from_list, parse_hf_source_line, resolve_hf_list_roots,
};
use crate::source_core::HuggingFaceRowSource;
use crate::test_utils::test_config;
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
