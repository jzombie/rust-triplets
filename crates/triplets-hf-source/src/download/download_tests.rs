use super::*;
use crate::constants::HF_DATASETS_BASE_URL;
use crate::test_utils::{
    TEST_UNREACHABLE_URL, TestHttpServer, spawn_one_shot_http, test_config, test_http_client,
    with_env_var, write_simdr_fixture,
};
use serde_json::json;
use serial_test::serial;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::Mutex;
use tempfile::tempdir;

#[test]
fn remote_url_for_candidate_constructs_correct_urls() {
    // url:: prefix with full URL: returned as-is.
    let config = test_config(PathBuf::from("/tmp/snap"));
    let full_url =
        format!("url::{HF_DATASETS_BASE_URL}/org/ds/resolve/main/train/part-000.parquet");
    let result = remote_url_for_candidate(&config, &full_url);
    assert_eq!(
        result,
        format!("{HF_DATASETS_BASE_URL}/org/ds/resolve/main/train/part-000.parquet")
    );

    // url:: prefix with relative path (Hub API format): CDN prefix is constructed.
    let hub_relative = "url::data/train-00000-of-00001.parquet";
    let result = remote_url_for_candidate(&config, hub_relative);
    assert_eq!(
        result,
        format!(
            "{HF_DATASETS_BASE_URL}/org/dataset/resolve/main/data/train-00000-of-00001.parquet"
        )
    );

    // Bare path (hf-hub sibling fallback): CDN prefix is prepended.
    let bare_path = "data/train-00000-of-00001.parquet";
    let result = remote_url_for_candidate(&config, bare_path);
    assert_eq!(
        result,
        format!(
            "{HF_DATASETS_BASE_URL}/org/dataset/resolve/main/data/train-00000-of-00001.parquet"
        )
    );

    // Bare path with leading slash.
    let bare_path = "/data/train-00000-of-00001.parquet";
    let result = remote_url_for_candidate(&config, bare_path);
    assert_eq!(
        result,
        format!(
            "{HF_DATASETS_BASE_URL}/org/dataset/resolve/main/data/train-00000-of-00001.parquet"
        )
    );
}

#[test]
fn remote_url_for_candidate_builds_bare_urls() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let r1 = remote_url_for_candidate(&config, "url::https://server/parquet");
    assert_eq!(r1, "https://server/parquet");
    let r2 = remote_url_for_candidate(&config, "data/train-000.parquet");
    assert!(r2.contains("/resolve/main/"));
}

#[test]
fn remote_url_for_candidate_bare_path_resolves_to_cdn() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());

    let url = remote_url_for_candidate(&config, "train/shard.ndjson");
    assert!(url.contains(HF_DATASETS_BASE_URL));
    assert!(url.contains("train/shard.ndjson"));
}

#[test]
fn remote_url_for_candidate_full_url_returned_directly() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());

    let url = remote_url_for_candidate(&config, "url::https://cdn.example.com/shard.parquet");
    assert_eq!(url, "https://cdn.example.com/shard.parquet");
}

#[test]
fn http_client_builds_with_token() {
    let temp = tempdir().unwrap();
    let mut config = test_config(temp.path().to_path_buf());
    config.hf_token = Some("test-bearer-token".to_string());
    let result = build_http_client(&config);
    assert!(
        result.is_ok(),
        "build_http_client should succeed with a well-formed token string"
    );
}

#[test]
#[serial(global_state)]
fn validate_token_accepts_200_response() {
    let temp = tempdir().unwrap();
    let mut config = test_config(temp.path().to_path_buf());
    config.hf_token = Some("valid-test-token".to_string());
    let server = spawn_one_shot_http(b"{\"name\":\"testuser\"}".to_vec());
    let base_url = server.url().to_string();
    with_env_var(ENV_TRIPLETS_HF_WHOAMI_ENDPOINT, &base_url, || {
        let client = test_http_client();
        let runtime = build_http_runtime(&config).unwrap();
        let result = validate_token_with_runtime(&client, &config, &runtime);
        assert!(result.is_ok(), "200 response should pass token validation");
    });
}

#[test]
#[serial(global_state)]
fn validate_token_rejects_401_response() {
    let temp = tempdir().unwrap();
    let mut config = test_config(temp.path().to_path_buf());
    config.hf_token = Some("invalid-test-token".to_string());
    let server = TestHttpServer::new(401, b"Unauthorized".to_vec());
    let base_url = server.url().to_string();
    with_env_var(ENV_TRIPLETS_HF_WHOAMI_ENDPOINT, &base_url, || {
        let client = test_http_client();
        let runtime = build_http_runtime(&config).unwrap();
        let result = validate_token_with_runtime(&client, &config, &runtime);
        assert!(result.is_err(), "401 response should fail token validation");
        match result {
            Err(SamplerError::SourceUnavailable { reason, .. }) => {
                assert!(
                    reason.contains("invalid or expired"),
                    "error should mention invalid/expired, got: {reason}"
                );
            }
            _ => panic!("expected SamplerError::SourceUnavailable"),
        }
    });
}

#[test]
fn all_candidates_from_parquet_manifest_returns_all_with_sizes() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    // Hub API tree endpoint format: array of {"path": "...", "size": N} objects
    let payload = json!([
        {"type": "file", "path": "train/000.parquet", "size": 100},
        {"type": "file", "path": "train/001.ndjson", "size": 200},
        {"type": "file", "path": "train/002.txt", "size": 50}
    ]);

    let (candidates, sizes, matched) =
        all_candidates_from_parquet_manifest(&config, &payload).unwrap();
    assert_eq!(candidates.len(), 2);
    assert!(candidates.iter().any(|c| c.ends_with("train/000.parquet")));
    assert!(candidates.iter().any(|c| c.ends_with("train/001.ndjson")));
    assert_eq!(sizes.len(), 2, "tree format provides sizes");
    assert_eq!(matched, 2);
}

#[test]
fn all_candidates_from_parquet_manifest_includes_cached_and_replaces_stale() {
    // Suppress the expected WARN "incomplete cached shard detected (will redownload)".
    let _quiet = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::ERROR)
            .finish(),
    );
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());

    // A parquet file with the correct declared size — considered fully cached.
    let complete_candidate = format!("{HF_REMOTE_URL_PREFIX}train/000.parquet");
    let complete_target = candidate_target_path(&config, &complete_candidate);
    fs::create_dir_all(complete_target.parent().unwrap()).unwrap();
    fs::write(&complete_target, vec![1u8; 7]).unwrap();

    // A parquet file with the WRONG size — stale/incomplete, must be deleted.
    let stale_candidate = format!("{HF_REMOTE_URL_PREFIX}train/001.parquet");
    let stale_target = candidate_target_path(&config, &stale_candidate);
    fs::create_dir_all(stale_target.parent().unwrap()).unwrap();
    fs::write(&stale_target, vec![2u8; 3]).unwrap();

    let payload = json!([
        {"type": "file", "path": "train/000.parquet", "size": 7},
        {"type": "file", "path": "train/001.parquet", "size": 9}
    ]);

    let (candidates, sizes, matched) =
        all_candidates_from_parquet_manifest(&config, &payload).unwrap();

    // Both shards are returned — cache state does not affect the candidate list.
    assert_eq!(candidates.len(), 2, "both shards must appear in candidates");
    // Complete shard: file exists and was not deleted.
    assert!(
        complete_target.exists(),
        "complete shard must not be deleted"
    );
    // Stale shard: wrong-size file was deleted so it will be re-fetched.
    assert!(!stale_target.exists(), "stale shard must be deleted");
    assert_eq!(sizes.len(), 2);
    assert_eq!(matched, 2);
}

#[test]
fn candidates_from_parquet_manifest_errors_when_removing_incomplete_target_fails() {
    // Suppress the expected WARN "incomplete cached shard detected (will redownload)"
    // emitted before the attempted removal fails.  The removal failure is what this
    // test asserts on; the warn preceding it is correct production behaviour.
    let _quiet = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::ERROR)
            .finish(),
    );
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let candidate = format!("{HF_REMOTE_URL_PREFIX}train/blocked.parquet");
    let target = candidate_target_path(&config, &candidate);
    fs::create_dir_all(&target).unwrap();

    let payload = json!([
        {"type": "file", "path": "train/blocked.parquet", "size": 1}
    ]);

    let err = all_candidates_from_parquet_manifest(&config, &payload);
    assert!(err.is_err());
}

#[test]
fn uncached_candidates_from_parquet_manifest_returns_empty_without_entries() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let payload = json!({"other": []});
    let (candidates, sizes, matched) =
        all_candidates_from_parquet_manifest(&config, &payload).unwrap();
    assert!(candidates.is_empty());
    assert!(sizes.is_empty());
    // No parquet_files key → zero matched entries.
    assert_eq!(matched, 0);
}

#[test]
fn all_candidates_from_parquet_manifest_empty_array() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let json = json!([]);
    let (candidates, sizes, matched) =
        crate::download::all_candidates_from_parquet_manifest(&config, &json).unwrap();
    assert!(candidates.is_empty());
    assert!(sizes.is_empty());
    assert_eq!(matched, 0);
}

#[test]
fn all_candidates_from_parquet_manifest_filters_non_parquet() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    // Only accept parquet files
    config.shard_extensions = vec!["parquet".to_string()];
    let json = json!([
        {"type": "file", "path": "data/train-000.parquet", "size": 100},
        {"type": "file", "path": "data/README.md", "size": 50},
        {"type": "file", "path": "data/train.jsonl", "size": 200}
    ]);
    let (candidates, sizes, matched) =
        crate::download::all_candidates_from_parquet_manifest(&config, &json).unwrap();
    // Only .parquet is accepted
    assert_eq!(candidates.len(), 1);
    assert!(candidates[0].contains("train-000.parquet"));
    assert_eq!(matched, 1);
    assert_eq!(sizes.get(&candidates[0]), Some(&100));
}

#[test]
fn all_candidates_from_parquet_manifest_skips_entries_without_path() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let json = json!([
        {"type": "file", "size": 100},
        {"type": "file", "path": "data/train-000.parquet", "size": 200}
    ]);
    let (candidates, _, matched) =
        crate::download::all_candidates_from_parquet_manifest(&config, &json).unwrap();
    assert_eq!(candidates.len(), 1);
    assert_eq!(matched, 1);
}

#[test]
fn all_candidates_from_parquet_manifest_handles_non_array() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let json = json!({"not": "an array"});
    let (candidates, _, matched) =
        crate::download::all_candidates_from_parquet_manifest(&config, &json).unwrap();
    assert!(candidates.is_empty());
    assert_eq!(matched, 0);
}

#[test]
fn all_candidates_from_parquet_manifest_deduplicates() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let json = json!([
        {"type": "file", "path": "data/train-000.parquet", "size": 100},
        {"type": "file", "path": "data/train-000.parquet", "size": 100}
    ]);
    let (candidates, _, matched) =
        crate::download::all_candidates_from_parquet_manifest(&config, &json).unwrap();
    assert_eq!(candidates.len(), 1);
    assert_eq!(matched, 2); // Both entries matched even though deduplicated
}

#[test]
fn all_candidates_from_parquet_manifest_no_size() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let json = json!([
        {"type": "file", "path": "data/train-000.parquet"}
    ]);
    let (candidates, sizes, matched) =
        crate::download::all_candidates_from_parquet_manifest(&config, &json).unwrap();
    assert_eq!(candidates.len(), 1);
    assert!(sizes.is_empty()); // No size provided
    assert_eq!(matched, 1);
}

#[test]
fn candidate_target_path_uses_bare_path_when_no_resolve_segment() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    // Bare relative path from tree endpoint (no /resolve/ segment)
    let candidate = "url::train/000.parquet";
    let target = candidate_target_path(&config, candidate);
    assert!(target.ends_with("_parquet_manifest/train/000.parquet"));
}

#[test]
fn candidate_target_path_maps_remote_urls_under_manifest_root() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let candidate =
        "url::https://huggingface.co/datasets/org/ds/resolve/main/train/part-000.parquet";
    let target = candidate_target_path(&config, candidate);
    assert!(target.ends_with("_parquet_manifest/main/train/part-000.parquet"));
}

#[test]
fn candidate_target_path_keeps_local_candidates_relative() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let candidate = "train/part-001.ndjson";
    let target = candidate_target_path(&config, candidate);
    assert_eq!(target, config.snapshot_dir.join(candidate));
}

#[test]
fn candidate_target_path_bare_relative_path() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());

    let path = candidate_target_path(&config, "train/0001.parquet");
    assert!(path.ends_with("train/0001.parquet"));
    assert!(path.starts_with(dir.path()));
}

#[test]
fn candidate_target_path_full_url_extracts_suffix() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());

    let path = candidate_target_path(
        &config,
        "url::https://host/datasets/org/ds/resolve/main/data/shard.ndjson",
    );
    assert!(path.ends_with("data/shard.ndjson"));
}

#[test]
fn target_matches_expected_size_is_false_for_missing_path() {
    let dir = tempdir().unwrap();
    let missing = dir.path().join("missing.bin");
    assert!(!target_matches_expected_size(&missing, Some(1)));
}

#[test]
fn target_matches_expected_size_validates_when_expected_is_provided() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("payload.bin");
    fs::write(&path, vec![0u8; 5]).unwrap();

    assert!(target_matches_expected_size(&path, Some(5)));
    assert!(!target_matches_expected_size(&path, Some(4)));
    assert!(target_matches_expected_size(&path, None));
}

#[test]
fn target_matches_expected_size_zero_expected_returns_true() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("payload.bin");
    fs::write(&path, vec![0u8; 5]).unwrap();
    // expected_bytes = Some(0) falls into the `_ => true` branch
    assert!(target_matches_expected_size(&path, Some(0)));
}

#[test]
fn target_matches_expected_size_none_requires_nonzero() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("empty.bin");
    fs::write(&path, vec![]).unwrap();
    assert!(!target_matches_expected_size(&path, None));
}

#[test]
fn target_matches_expected_size_mismatch_returns_false() {
    let dir = tempdir().unwrap();
    let file = dir.path().join("mismatch.bin");
    fs::write(&file, b"hello").unwrap();
    assert!(
        !target_matches_expected_size(&file, Some(100)),
        "size mismatch should return false"
    );
}

#[test]
fn target_matches_expected_size_missing_file_returns_false() {
    let dir = tempdir().unwrap();
    let file = dir.path().join("nonexistent.bin");
    assert!(
        !target_matches_expected_size(&file, Some(100)),
        "missing file should return false"
    );
}

#[test]
fn shard_candidate_seed_and_shuffle_are_deterministic() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.source_id = "hf_rotator".to_string();

    let seed_a = shard_candidate_seed(&config, 12, 1);
    let seed_b = shard_candidate_seed(&config, 12, 2);
    assert_ne!(seed_a, seed_b);

    let baseline = vec!["c".to_string(), "a".to_string(), "b".to_string()];
    let mut left = baseline.clone();
    let mut right = baseline;
    shuffle_candidates_deterministically(&config, &mut left, 42);
    shuffle_candidates_deterministically(&config, &mut right, 42);
    assert_eq!(left, right);

    // Different seeds produce different orderings for non-trivial inputs.
    let mut alt = vec!["c".to_string(), "a".to_string(), "b".to_string()];
    shuffle_candidates_deterministically(&config, &mut alt, 99);
    // Membership is preserved regardless of seed.
    let mut sorted_left = left.clone();
    sorted_left.sort();
    let mut sorted_alt = alt.clone();
    sorted_alt.sort();
    assert_eq!(sorted_left, sorted_alt);
}

#[test]
fn shard_candidate_seed_is_seeded_and_source_scoped() {
    let dir = tempdir().unwrap();
    let mut a = test_config(dir.path().join("a"));
    let mut b = test_config(dir.path().join("b"));
    a.source_id = "source_a".to_string();
    b.source_id = "source_b".to_string();

    let with_seed_a = shard_candidate_seed(&a, 100, 42);
    let with_seed_a_again = shard_candidate_seed(&a, 100, 42);
    assert_eq!(with_seed_a, with_seed_a_again);

    let with_seed_b = shard_candidate_seed(&b, 100, 42);
    assert_ne!(with_seed_a, with_seed_b);

    let different_seed_a = shard_candidate_seed(&a, 100, 7);
    assert_ne!(with_seed_a, different_seed_a);
}

#[test]
fn shard_candidate_seed_changes_with_sampler_seed() {
    // Verifies that different sampler_seed values (which in production
    // include the epoch_step XOR from IngestionManager) produce
    // different shard permutations, while the same seed is deterministic.
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());

    // Different sampler seeds → different shard candidate seeds.
    let seed_1 = shard_candidate_seed(&config, 100, 1);
    let seed_2 = shard_candidate_seed(&config, 100, 2);
    assert_ne!(
        seed_1, seed_2,
        "different seeds must produce different shard seeds"
    );

    // Same sampler seed → deterministic.
    let seed_1_again = shard_candidate_seed(&config, 100, 1);
    assert_eq!(seed_1, seed_1_again, "same seed must be deterministic");

    // Verify the permutation itself changes with seed.
    let candidates: Vec<String> = (0..10).map(|i| format!("shard-{i:02}")).collect();
    let order_1 = build_candidate_order(&config, &candidates, 1);
    let order_2 = build_candidate_order(&config, &candidates, 2);
    assert_ne!(
        order_1, order_2,
        "different seeds must produce different shard orders"
    );

    // Same seed produces same order.
    let order_1_again = build_candidate_order(&config, &candidates, 1);
    assert_eq!(order_1, order_1_again, "same seed must produce same order");
}

#[test]
fn shuffle_candidates_deterministically_is_noop_for_singleton() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let mut candidates = vec!["one".to_string()];
    shuffle_candidates_deterministically(&config, &mut candidates, 1);
    assert_eq!(candidates, vec!["one".to_string()]);
}

#[test]
fn shuffle_candidates_deterministically_preserves_membership() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let original = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let mut shuffled = original.clone();
    shuffle_candidates_deterministically(&config, &mut shuffled, 1);
    let mut sorted_original = original;
    let mut sorted_shuffled = shuffled;
    sorted_original.sort();
    sorted_shuffled.sort();
    assert_eq!(sorted_shuffled, sorted_original);
}

#[test]
fn build_candidate_order_single_element() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let candidates = vec!["a".to_string()];
    let order = build_candidate_order(&config, &candidates, 42);
    assert_eq!(order, vec![0]);
}

#[test]
fn build_candidate_order_empty() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let candidates: Vec<String> = vec![];
    let order = build_candidate_order(&config, &candidates, 42);
    assert!(order.is_empty());
}

#[test]
fn build_candidate_order_base_seed_zero_uses_fallback() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.source_id = "".to_string();
    config.dataset_name = "".to_string();
    config.config_name = "".to_string();
    config.split_name = "".to_string();
    let candidates = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    // With all-empty fields, shard_candidate_seed may return 0, triggering fallback
    let order = build_candidate_order(&config, &candidates, 0);
    assert_eq!(order.len(), 3);
    // All indices must be present
    let mut sorted = order.clone();
    sorted.sort();
    assert_eq!(sorted, vec![0, 1, 2]);
}

#[test]
fn remote_shard_permutation_is_deterministic_by_sampler_seed() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let total = 8usize;

    let seed_a = shard_candidate_seed(&config, total, 7);
    let seed_b = shard_candidate_seed(&config, total, 7);
    let seed_c = shard_candidate_seed(&config, total, 10);

    let mut perm_a = triplets_core::source::IndexPermutation::new(total, seed_a, 0);
    let mut perm_b = triplets_core::source::IndexPermutation::new(total, seed_b, 0);
    let mut perm_c = triplets_core::source::IndexPermutation::new(total, seed_c, 0);

    let take = 6usize;
    let order_a: Vec<usize> = (0..take).map(|_| perm_a.next()).collect();
    let order_b: Vec<usize> = (0..take).map(|_| perm_b.next()).collect();
    let order_c: Vec<usize> = (0..take).map(|_| perm_c.next()).collect();

    assert_eq!(order_a, order_b);
    assert_ne!(order_a, order_c);
}

#[test]
fn remote_shard_permutation_is_deterministic() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let c = ["a", "b", "c", "d", "e"];
    let c1: Vec<String> = c.iter().map(|s| s.to_string()).collect();
    let c2: Vec<String> = c.iter().map(|s| s.to_string()).collect();
    let o1 = build_candidate_order(&config, &c1, 42);
    let o2 = build_candidate_order(&config, &c2, 42);
    assert_eq!(o1, o2);
    let o3 = build_candidate_order(&config, &c1, 99);
    assert_ne!(o1, o3);
}

#[test]
fn format_shard_label_includes_totals() {
    let label = format_shard_label("data/train-000.parquet", 0, 5);
    assert!(label.contains("1/5"));
    assert!(label.contains("train-000.parquet"));
}

#[test]
fn format_shard_label_strips_url_prefix() {
    let label = format_shard_label("url::data/train-000.parquet", 2, 10);
    assert!(label.contains("3/10"));
    assert!(label.contains("train-000.parquet"));
}

#[test]
fn format_shard_label_handles_no_slash() {
    let label = format_shard_label("train.parquet", 0, 1);
    assert_eq!(label, "train.parquet (shard 1/1)");
}

#[test]
fn parse_parquet_manifest_response_errors_on_invalid_json() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let parsed = parse_parquet_manifest_response(&config, "{bad-json");
    assert!(parsed.is_err());
}

#[test]
fn parse_parquet_manifest_response_returns_candidates() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let body = serde_json::to_string(&json!([
             {"type": "file", "path": "https://host/datasets/x/resolve/main/train/0.parquet", "size": 100}
         ]))
         .unwrap();

    let (candidates, sizes, matched) = parse_parquet_manifest_response(&config, &body).unwrap();
    assert_eq!(candidates.len(), 1);
    assert!(!sizes.is_empty());
    assert_eq!(matched, 1);
}

#[test]
fn parse_parquet_manifest_response_invalid_json() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let err = crate::download::parse_parquet_manifest_response(&config, "not json").unwrap_err();
    assert!(matches!(err, SamplerError::SourceUnavailable { .. }));
}

#[test]
fn parse_parquet_manifest_response_valid_json() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let body = r#"[{"type":"file","path":"data/train-000.parquet","size":100}]"#;
    let (candidates, _, matched) =
        crate::download::parse_parquet_manifest_response(&config, body).unwrap();
    assert_eq!(candidates.len(), 1);
    assert_eq!(matched, 1);
}

#[test]
#[serial(global_state)]
fn list_remote_candidates_falls_back_when_manifest_query_fails() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.dataset_name = "invalid///dataset".to_string();
    config.parquet_endpoint = TEST_UNREACHABLE_URL.to_string();

    let client = test_http_client();
    let result = HuggingFaceRowSource::list_remote_candidates(&client, &config);
    assert!(result.is_err());
}

#[test]
#[serial(global_state)]
fn list_remote_candidates_from_parquet_manifest_uses_test_endpoint_override() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    let body = serde_json::to_vec(&json!([
             {"type": "file", "path": "https://host/datasets/x/resolve/main/train/0.parquet", "size": 100}
         ]))
         .unwrap();
    let server = spawn_one_shot_http(body);
    let base_url = server.url().to_string();

    config.parquet_endpoint = base_url;
    let client = test_http_client();
    let (candidates, sizes, matched) =
        HuggingFaceRowSource::list_remote_candidates_from_parquet_manifest(&client, &config)
            .unwrap();

    assert_eq!(candidates.len(), 1);
    assert!(!sizes.is_empty());
    assert_eq!(matched, 1);
}

#[test]
#[serial(global_state)]
fn list_remote_candidates_returns_manifest_candidates() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    let body = serde_json::to_vec(&json!([
        {"type": "file", "path": "https://host/datasets/x/resolve/main/train/1.ndjson", "size": 100}
    ]))
    .unwrap();
    let server = spawn_one_shot_http(body);
    let base_url = server.url().to_string();

    config.parquet_endpoint = base_url;
    let client = test_http_client();
    let (candidates, sizes) =
        HuggingFaceRowSource::list_remote_candidates(&client, &config).unwrap();

    assert_eq!(candidates.len(), 1);
    assert!(!sizes.is_empty());
    assert!(candidates[0].ends_with("/1.ndjson"));
}

#[test]
fn list_remote_candidates_scopes_tree_to_config_name() {
    // Verify that the tree endpoint URL is scoped to the config_name
    // subdirectory so we don't pull files from other configs
    // (e.g. wikimedia/wikipedia/20231101.en should not list .fr files).
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.config_name = "20231101.en".to_string();

    // Spawn a mock server that records the request path.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let base_url = format!("http://{addr}");
    let recorded_path: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let recorded_path_clone = Arc::clone(&recorded_path);
    let handle = std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 4096];
            let n = stream.read(&mut buf).unwrap_or(0);
            let request = String::from_utf8_lossy(&buf[..n]);
            let path = request
                .lines()
                .next()
                .and_then(|line| line.split_whitespace().nth(1))
                .map(|s| s.to_string());
            *recorded_path_clone.lock().unwrap() = path;
            let body = br#"[]"#;
            let headers = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(headers.as_bytes());
            let _ = stream.write_all(body);
        }
    });

    config.parquet_endpoint = base_url;
    let client = test_http_client();
    let _ = HuggingFaceRowSource::list_remote_candidates_from_parquet_manifest(&client, &config);
    handle.join().unwrap();

    let path = recorded_path.lock().unwrap();
    let path = path.as_deref().expect("no request recorded");
    assert!(
        path.contains("/tree/main/20231101.en"),
        "tree endpoint URL must be scoped to config_name; got: {path}"
    );
    assert!(
        !path.contains("/tree/main?"),
        "tree endpoint must NOT use root path when config_name is set; got: {path}"
    );
}

#[test]
fn list_remote_candidates_uses_root_tree_for_default_config() {
    // When config_name is "default", the tree endpoint should hit the repo root.
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.config_name = "default".to_string();

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let base_url = format!("http://{addr}");
    let recorded_path: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let recorded_path_clone = Arc::clone(&recorded_path);
    let handle = std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 4096];
            let n = stream.read(&mut buf).unwrap_or(0);
            let request = String::from_utf8_lossy(&buf[..n]);
            let path = request
                .lines()
                .next()
                .and_then(|line| line.split_whitespace().nth(1))
                .map(|s| s.to_string());
            *recorded_path_clone.lock().unwrap() = path;
            let body = br#"[]"#;
            let headers = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(headers.as_bytes());
            let _ = stream.write_all(body);
        }
    });

    config.parquet_endpoint = base_url;
    let client = test_http_client();
    let _ = HuggingFaceRowSource::list_remote_candidates_from_parquet_manifest(&client, &config);
    handle.join().unwrap();

    let path = recorded_path.lock().unwrap();
    let path = path.as_deref().expect("no request recorded");
    assert!(
        path.contains("/tree/main?recursive=true"),
        "default config must hit repo root; got: {path}"
    );
}

#[test]
#[serial(global_state)]
fn list_remote_candidates_with_runtime_returns_manifest_candidates() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    let runtime = build_http_runtime(&config).unwrap();
    let body = serde_json::to_vec(&json!([
        {"type": "file", "path": "https://host/datasets/x/resolve/main/train/2.ndjson", "size": 100}
    ]))
    .unwrap();
    let server = spawn_one_shot_http(body);
    let base_url = server.url().to_string();

    config.parquet_endpoint = base_url;
    let client = test_http_client();
    let (candidates, sizes) =
        list_remote_candidates_with_runtime(&client, &config, Some(&runtime)).unwrap();

    assert_eq!(candidates.len(), 1);
    assert!(!sizes.is_empty());
    assert!(candidates[0].ends_with("/2.ndjson"));
}

#[test]
#[serial(global_state)]
fn list_remote_candidates_does_not_fall_back_when_all_manifest_shards_cached() {
    // Regression test: list_remote_candidates must return the full manifest
    // candidate list when a parquet manifest exists, regardless of whether all
    // shards are already cached on disk.
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());

    // Pre-create the .simdr store target so the manifest entry is "fully cached".
    let shard_url = "https://host/datasets/org/ds/resolve/main/train/part-000.ndjson";
    let candidate = format!("{HF_REMOTE_URL_PREFIX}{shard_url}");
    let target = candidate_target_path(&config, &candidate);
    let store_target = shard_store_path_for(&target);
    fs::create_dir_all(store_target.parent().unwrap()).unwrap();
    fs::write(&store_target, b"cached").unwrap();

    let body = serde_json::to_vec(&json!([
        {"type": "file", "path": shard_url, "size": 100}
    ]))
    .unwrap();
    let server = spawn_one_shot_http(body);
    let base_url = server.url().to_string();

    // Must return the full manifest candidate list without falling through to hf-hub.
    config.parquet_endpoint = base_url;
    let client = test_http_client();
    let (candidates, sizes) =
        HuggingFaceRowSource::list_remote_candidates(&client, &config).unwrap();

    assert_eq!(
        candidates.len(),
        1,
        "fully-cached shard must still appear in candidates (cache ≠ order)"
    );
    assert!(!sizes.is_empty());
    assert!(candidates[0].ends_with(shard_url));
}

#[test]
#[serial(global_state)]
fn list_remote_candidates_from_parquet_manifest_errors_when_endpoint_unreachable() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.parquet_endpoint = TEST_UNREACHABLE_URL.to_string();

    let client = test_http_client();
    let result =
        HuggingFaceRowSource::list_remote_candidates_from_parquet_manifest(&client, &config);
    assert!(result.is_err());
}

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
#[serial(global_state)]
fn list_remote_candidates_from_parquet_manifest_errors_on_501() {
    // A 501 from /parquet causes the manifest path to return Err, which
    // propagates to the caller (no fallback path).
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    let body =
        br#"{"error":"Not supported: dataset viewer is disabled in org/dataset configuration."}"#
            .to_vec();
    let server = TestHttpServer::new(501, body);
    let base_url = server.url().to_string();

    config.parquet_endpoint = base_url;
    let client = test_http_client();
    let result =
        HuggingFaceRowSource::list_remote_candidates_from_parquet_manifest(&client, &config);

    assert!(
        result.is_err(),
        "expected Err from 501 /parquet response, got {result:?}"
    );
}

#[test]
fn list_remote_candidates_returns_error_on_invalid_json() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    let server = spawn_one_shot_http(b"this is not json at all".to_vec());
    config.parquet_endpoint = server.url().to_string();

    let client = test_http_client();
    let result = HuggingFaceRowSource::list_remote_candidates(&client, &config);
    assert!(result.is_err(), "invalid JSON should return error");
}

#[test]
fn list_remote_candidates_returns_error_on_non_success() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    let server = TestHttpServer::new(503, b"Service Unavailable".to_vec());
    config.parquet_endpoint = server.url().to_string();

    let client = test_http_client();
    let result = HuggingFaceRowSource::list_remote_candidates(&client, &config);
    assert!(result.is_err(), "503 response should return error");
}

#[test]
fn download_and_materialize_shard_url_short_circuits_when_cached_complete() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let candidate = "url::https://host/datasets/org/ds/resolve/main/train/ok.ndjson";
    let target = candidate_target_path(&config, candidate);
    fs::create_dir_all(target.parent().unwrap()).unwrap();
    fs::write(&target, b"ok").unwrap();

    let client = test_http_client();
    let resolved = HuggingFaceRowSource::download_and_materialize_shard(
        &client,
        &config,
        candidate,
        Some(2),
        "shard 1/1",
    )
    .unwrap();
    assert_eq!(resolved, target);
}

#[test]
fn download_and_materialize_shard_url_replaces_stale_part_file() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let payload = b"{\"text\":\"a\"}\n".to_vec();
    let server = spawn_one_shot_http(payload.clone());
    let base_url = server.url().to_string();
    let candidate = format!("url::{base_url}/datasets/org/ds/resolve/main/train/part-x.ndjson");
    let target = candidate_target_path(&config, &candidate);
    let temp_target = target.with_extension("part");
    fs::create_dir_all(temp_target.parent().unwrap()).unwrap();
    fs::write(&temp_target, b"stale").unwrap();

    let client = test_http_client();
    let out = HuggingFaceRowSource::download_and_materialize_shard(
        &client,
        &config,
        &candidate,
        None,
        "shard 1/1",
    )
    .unwrap();

    // Transient formats (ndjson) are staged to a temp path, not the cache target.
    assert_ne!(out, target, "transient download should go to temp path");
    assert!(out.exists(), "temp file should exist");
    assert_eq!(fs::read(&out).unwrap(), payload);
}

#[test]
fn download_and_materialize_shard_downloads_url_candidate() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let payload = b"{\"text\":\"a\"}\n{\"text\":\"b\"}\n".to_vec();
    let server = spawn_one_shot_http(payload.clone());
    let base_url = server.url().to_string();
    let candidate = format!("url::{base_url}/datasets/org/ds/resolve/main/train/part-000.ndjson");

    let client = test_http_client();
    let target = HuggingFaceRowSource::download_and_materialize_shard(
        &client,
        &config,
        &candidate,
        None,
        "shard 1/1",
    )
    .unwrap();

    assert!(target.exists());
    assert_eq!(fs::read(&target).unwrap(), payload);
}

#[test]
fn download_and_materialize_shard_replaces_incomplete_existing_target() {
    // Suppress the expected WARN "replacing incomplete shard before retry" that fires
    // when an existing target file's size does not match the expected manifest size.
    // Detecting and replacing the stale file is what this test asserts on.
    let _quiet = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::ERROR)
            .finish(),
    );
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let payload = b"{\"text\":\"a\"}\n".to_vec();
    let server = spawn_one_shot_http(payload.clone());
    let base_url = server.url().to_string();
    let candidate = format!("url::{base_url}/datasets/org/ds/resolve/main/train/part-009.ndjson");

    let target = candidate_target_path(&config, &candidate);
    fs::create_dir_all(target.parent().unwrap()).unwrap();
    fs::write(&target, b"bad").unwrap();

    let client = test_http_client();
    let refreshed = HuggingFaceRowSource::download_and_materialize_shard(
        &client,
        &config,
        &candidate,
        Some(payload.len() as u64),
        "shard 1/1",
    )
    .unwrap();

    // Transient formats (ndjson) are staged to a temp path, not the cache target.
    assert_ne!(
        refreshed, target,
        "transient download should go to temp path"
    );
    assert!(refreshed.exists(), "temp file should exist");
    assert_eq!(fs::read(&refreshed).unwrap(), payload);
}

#[test]
fn download_shard_rejects_path_traversal_double_dot() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let client = test_http_client();

    let remote_path = "url::http://evil.com/../../etc/passwd";
    let result = HuggingFaceRowSource::download_and_materialize_shard(
        &client,
        &config,
        remote_path,
        None,
        "traversal-test",
    );
    assert!(result.is_err(), "path traversal should be rejected");
    match result {
        Err(SamplerError::SourceUnavailable { reason, .. }) => {
            assert!(
                reason.contains("traversal"),
                "error should mention traversal, got: {reason}"
            );
        }
        other => panic!("expected SourceUnavailable error, got: {:?}", other),
    }
}

#[test]
fn download_shard_rejects_path_traversal_in_full_url() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let client = test_http_client();

    let remote_path = "url::https://host/datasets/org/ds/resolve/main/../../../etc/passwd";
    let result = HuggingFaceRowSource::download_and_materialize_shard(
        &client,
        &config,
        remote_path,
        None,
        "traversal-test-2",
    );
    assert!(
        result.is_err(),
        "path traversal in full URL should be rejected"
    );
}

#[test]
fn download_shard_store_already_exists_returns_cached() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let client = test_http_client();

    let remote_path =
        "url::http://mock.example.com/datasets/org/ds/resolve/main/train/shard.ndjson";
    let store_path = crate::shard_indexing::candidate_store_path(&config, remote_path);
    fs::create_dir_all(store_path.parent().unwrap()).unwrap();
    write_simdr_fixture(&store_path, &[("r1", "cached")]);

    let result = HuggingFaceRowSource::download_and_materialize_shard(
        &client,
        &config,
        remote_path,
        None,
        "cache-test",
    );
    let path = result.expect("should return Ok with store path");
    assert_eq!(path, store_path, "should return the existing store path");
}

#[test]
fn download_shard_rejects_bare_relative_traversal() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let client = test_http_client();

    let remote_path = "../etc/passwd";
    let result = HuggingFaceRowSource::download_and_materialize_shard(
        &client,
        &config,
        remote_path,
        None,
        "bare-traversal",
    );
    assert!(
        result.is_err(),
        "bare relative traversal should be rejected"
    );
}

#[test]
fn download_shard_rejects_url_encoded_traversal() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let client = test_http_client();

    let remote_path = "url::http://evil.com/datasets/%2e%2e/%2e%2e/etc/passwd";
    let result = HuggingFaceRowSource::download_and_materialize_shard(
        &client,
        &config,
        remote_path,
        None,
        "encoded-traversal",
    );
    assert!(result.is_err(), "URL-encoded traversal should be rejected");
}

#[test]
fn download_shard_rejects_http_traversal_without_url_prefix() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let client = test_http_client();

    let remote_path = "http://evil.com/../../../etc/passwd";
    let result = HuggingFaceRowSource::download_and_materialize_shard(
        &client,
        &config,
        remote_path,
        None,
        "http-traversal",
    );
    assert!(
        result.is_err(),
        "HTTP URL with traversal should be rejected"
    );
}

#[test]
#[serial(global_state)]
fn fetch_remote_size_with_runtime_returns_content_length() {
    // A mock HTTP server that responds to HEAD with Content-Length.
    let payload = b"this is the shard content".to_vec();
    let server = TestHttpServer::new(200, payload.clone());
    let base_url = server.url().to_string();

    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.hf_token = None;

    let client = test_http_client();
    let runtime = build_http_runtime(&config).unwrap();
    let size = fetch_remote_size_with_runtime(&client, &config, &base_url, &runtime).unwrap();
    // Content-Length should match the payload size.
    assert_eq!(size, Some(payload.len() as u64));
}

#[test]
#[serial(global_state)]
fn fetch_remote_size_with_runtime_returns_none_on_non_success() {
    // A mock server returning 404 — HEAD should return Ok(None).
    let server = TestHttpServer::new(404, b"Not Found".to_vec());
    let base_url = server.url().to_string();

    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.hf_token = None;

    let client = test_http_client();
    let runtime = build_http_runtime(&config).unwrap();
    let size = fetch_remote_size_with_runtime(&client, &config, &base_url, &runtime).unwrap();
    assert_eq!(size, None, "non-2xx response should yield None");
}

#[test]
fn fetch_remote_size_network_failure_returns_error() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let client = test_http_client();
    let runtime = build_http_runtime(&config).unwrap();

    let result = fetch_remote_size_with_runtime(
        &client,
        &config,
        &format!("{TEST_UNREACHABLE_URL}/shard.parquet"),
        &runtime,
    );
    assert!(result.is_err(), "HEAD to unreachable URL should fail");
}
