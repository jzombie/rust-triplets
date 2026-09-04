use super::*;
use std::fs;
use tempfile::tempdir;
use triplets_core::utils::platform_newline;

#[test]
fn hf_source_id_slug_uses_short_dataset_name() {
    assert_eq!(
        hf_source_id_slug("allenai/dolmino-mix-1124", "default", "train"),
        "dolmino-mix-1124"
    );
}

#[test]
fn hf_source_id_slug_appends_non_default_config() {
    assert_eq!(
        hf_source_id_slug("org/dataset", "en", "train"),
        "dataset.en"
    );
}

#[test]
fn hf_source_id_slug_appends_non_train_split() {
    assert_eq!(
        hf_source_id_slug("org/dataset", "default", "validation"),
        "dataset.validation"
    );
}

#[test]
fn hf_source_id_slug_omits_empty_split() {
    assert_eq!(hf_source_id_slug("org/dataset", "default", ""), "dataset");
}

#[test]
fn hf_source_id_slug_appends_both_config_and_split() {
    assert_eq!(
        hf_source_id_slug("org/dataset", "en", "validation"),
        "dataset.en.validation"
    );
}

#[test]
fn hf_source_id_slug_sanitizes_special_chars() {
    // dots and slashes in names become dashes
    assert_eq!(
        hf_source_id_slug("org/data.set", "v1.0", "train"),
        "data-set.v1-0"
    );
}

#[test]
fn hf_source_id_slug_no_org_prefix() {
    // dataset without org/ prefix — falls back to using the full string
    assert_eq!(hf_source_id_slug("dataset", "default", "train"), "dataset");
}

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
