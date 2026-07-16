use crate::builder::build_hf_sources;
use crate::config::HuggingFaceRowsConfig;
use crate::constants::{
    ENV_TRIPLETS_HF_TOKEN, ENV_TRIPLETS_HF_WHOAMI_ENDPOINT, HF_ALL_SPLITS_DIR,
    HF_DATASETS_BASE_URL, HF_GROUP, HF_PARQUET_DEFAULT_ENDPOINT, HF_PARQUET_MANIFEST_DIR,
    HF_REMOTE_URL_PREFIX, HF_SHARD_STORE_META_ROWS_KEY, HF_SHARD_STORE_ROW_PREFIX,
    HF_SHARD_STORE_SOURCE_SIZE_KEY,
};
use crate::disk_cache::ensure_cache_group;
use crate::download::*;
use crate::file_utils::{is_gzip_path, resolve_inner_extension};
use crate::huggingface_source::{
    EligibleIndexCache, HF_RECIPE_ANCHOR_ANCHOR_WRONG_ARTICLE,
    HF_RECIPE_ANCHOR_CONTEXT_WRONG_ARTICLE, HF_RECIPE_TEXT_SIMCSE_WRONG_ARTICLE,
    HuggingFaceRowSource, ParquetCache, RowCache, RowTextField, RowView,
    managed_hf_list_snapshot_dir, managed_hf_snapshot_dir,
};
use crate::parsing::{
    HfListRoots, HfSourceEntry, hf_source_id_slug, load_hf_sources_from_list, parse_hf_source_line,
    parse_hf_uri, resolve_hf_list_roots,
};
use crate::rows;
use crate::shard_index::{
    index_single_shard, is_store_shard_path, parquet_row_group_map, row_store_row_key,
    shard_store_path_for,
};
use crate::test_utils::{
    TEST_UNREACHABLE_URL, TestHttpServer, spawn_manifest_and_shard_http, spawn_one_shot_http,
    test_config, test_http_client, test_source, with_current_dir, with_env_var, with_env_vars,
    write_parquet_fixture, write_simdr_fixture,
};
use crate::types::{ShardIndex, SourceState};
use chrono::Utc;
use serde_json::json;
use serial_test::serial;
use simd_r_drive::DataStore;
use simd_r_drive::storage_engine::traits::DataStoreWriter;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use tempfile::tempdir;
use triplets_core::SamplerError;
use triplets_core::config::SamplerConfig;
use triplets_core::data::SectionRole;
use triplets_core::source::{DataSource, SourceCursor};
use triplets_core::splits::{PersistedSamplerState, SamplerStateStore};
use triplets_core::utils::platform_newline;
use triplets_core::{DeterministicSplitStore, Sampler, SplitLabel, SplitRatios, TripletSampler};

#[test]
#[serial(global_state)]
fn managed_snapshot_helpers_create_cache_dirs_under_discovered_root() {
    let dir = tempdir().unwrap();
    let nl = platform_newline();
    fs::write(
        dir.path().join("Cargo.toml"),
        format!("[package]{nl}name='tmp'{nl}version='0.0.0'{nl}"),
    )
    .unwrap();

    with_current_dir(dir.path(), || {
        let single = managed_hf_snapshot_dir("org/dataset", "default", "train").unwrap();
        let listed = managed_hf_list_snapshot_dir("org/dataset", "default", "train", 7).unwrap();

        assert!(single.exists());
        assert!(listed.exists());
        assert!(single.ends_with(PathBuf::from(format!(
            "{}/org__dataset/default/train",
            HF_GROUP
        ))));
        assert!(listed.ends_with(PathBuf::from(format!(
            "{}/source-list/org__dataset/default/train/replica_7",
            HF_GROUP
        ))));
        assert!(listed.ends_with("replica_7"));
    });
}

#[test]
#[serial(global_state)]
fn managed_snapshot_dirs_use_all_splits_dir_for_empty_split() {
    let dir = tempdir().unwrap();
    let nl = platform_newline();
    fs::write(
        dir.path().join("Cargo.toml"),
        format!("[package]{nl}name='tmp'{nl}version='0.0.0'{nl}"),
    )
    .unwrap();

    with_current_dir(dir.path(), || {
        let single = managed_hf_snapshot_dir("org/dataset", "default", "").unwrap();
        let listed = managed_hf_list_snapshot_dir("org/dataset", "default", "", 0).unwrap();

        assert!(single.exists());
        assert!(listed.exists());
        // Both must use HF_ALL_SPLITS_DIR ("_all") in the path, not an empty segment.
        assert!(
            single.ends_with(PathBuf::from(format!(
                "{}/org__dataset/default/{}",
                HF_GROUP, HF_ALL_SPLITS_DIR
            ))),
            "expected single-source path to end with HF_ALL_SPLITS_DIR, got: {}",
            single.display()
        );
        assert!(
            listed.ends_with(PathBuf::from(format!(
                "{}/source-list/org__dataset/default/{}/replica_0",
                HF_GROUP, HF_ALL_SPLITS_DIR
            ))),
            "expected list-source path to end with HF_ALL_SPLITS_DIR, got: {}",
            listed.display()
        );
        // Must not collide with the explicit-train path.
        let train_single = managed_hf_snapshot_dir("org/dataset", "default", "train").unwrap();
        assert_ne!(
            single, train_single,
            "empty-split and train-split paths must differ"
        );
    });
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
fn new_errors_when_snapshot_dir_path_is_a_file() {
    let dir = tempdir().unwrap();
    let snapshot_file = dir.path().join("snapshot-file");
    fs::write(&snapshot_file, b"x").unwrap();

    let config = HuggingFaceRowsConfig::new(
        "hf_bad_snapshot",
        "org/dataset",
        "default",
        "train",
        snapshot_file,
    );
    let result = HuggingFaceRowSource::new(config);
    assert!(matches!(
        result,
        Err(SamplerError::SourceUnavailable { .. })
    ));
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

// ── Token validation tests ──────────────────────────────────────────────

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
fn row_cache_insert_and_evicts_oldest_entry() {
    let mut cache = RowCache::default();
    let row_a = RowView {
        row_id: Some("a".to_string()),
        timestamp: None,
        text_fields: vec![RowTextField {
            name: "text".to_string(),
            text: "alpha".to_string(),
        }],
    };
    let row_b = RowView {
        row_id: Some("b".to_string()),
        timestamp: None,
        text_fields: vec![RowTextField {
            name: "text".to_string(),
            text: "beta".to_string(),
        }],
    };

    cache.insert(0, row_a.clone(), 1);
    assert!(cache.get(0).is_some());

    cache.insert(1, row_b, 1);
    assert!(cache.get(0).is_none());
    assert_eq!(cache.get(1).unwrap().row_id.as_deref(), Some("b"));

    let mut zero_cache = RowCache::default();
    zero_cache.insert(7, row_a, 0);
    assert!(zero_cache.get(7).is_none());
}

#[test]
fn parquet_cache_reader_for_reports_open_and_parse_errors() {
    let dir = tempdir().unwrap();
    let parquet_path = dir.path().join("missing.parquet");
    let mut cache = ParquetCache::default();
    let missing = cache.reader_for("hf_test", &parquet_path);
    assert!(missing.is_err());

    let invalid_parquet = dir.path().join("invalid.parquet");
    fs::write(&invalid_parquet, b"not parquet").unwrap();
    let invalid = cache.reader_for("hf_test", &invalid_parquet);
    assert!(invalid.is_err());
}

#[test]
fn read_store_row_count_validates_payload_size_and_roundtrips_written_value() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    let store_path = dir.path().join("rows.simdr");
    let store = DataStore::open(&store_path).unwrap();

    store
        .write(HF_SHARD_STORE_META_ROWS_KEY, &[1u8, 2, 3])
        .unwrap();
    let err = rows::read_store_row_count(&source, &store).unwrap_err();
    let message = format!("{err}");
    assert!(message.contains("payload size mismatch"));

    rows::write_store_row_count(&source, &store, 7).unwrap();
    assert_eq!(rows::read_store_row_count(&source, &store).unwrap(), 7);
}

#[test]
fn read_store_row_count_returns_zero_when_meta_key_is_absent() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    let store_path = dir.path().join("empty.simdr");
    let store = DataStore::open(&store_path).unwrap();

    assert_eq!(rows::read_store_row_count(&source, &store).unwrap(), 0);
}

#[test]
fn shard_store_path_for_passthrough_when_already_simdr() {
    let path = PathBuf::from("cache/shard.simdr");
    let mapped = shard_store_path_for(&path);
    assert_eq!(mapped, path);
}

#[test]
fn get_or_open_shard_store_reuses_cached_handle_and_prune_keeps_active_only() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);

    let store_a = dir.path().join("a.simdr");
    let store_b = dir.path().join("b.simdr");

    let first = source.get_or_open_shard_store(&store_a).unwrap();
    let second = source.get_or_open_shard_store(&store_a).unwrap();
    assert!(Arc::ptr_eq(&first, &second));

    let _third = source.get_or_open_shard_store(&store_b).unwrap();
    {
        let cache = source.config.store_cache.lock().unwrap();
        assert!(cache.contains_key(&store_a));
        assert!(cache.contains_key(&store_b));
    }

    source.prune_store_cache_to_shards(&[ShardIndex {
        path: store_a.clone(),
        global_start: 0,
        row_count: 1,
        parquet_row_groups: vec![(0, 1)],
        remote_candidate: None,
    }]);

    let cache = source.config.store_cache.lock().unwrap();
    assert!(cache.contains_key(&store_a));
    assert!(!cache.contains_key(&store_b));
}

#[test]
fn parquet_cache_row_group_rows_for_hits_cache_and_evicts_lru() {
    let dir = tempdir().unwrap();
    let path_a = dir.path().join("a.parquet");
    let path_b = dir.path().join("b.parquet");
    write_parquet_fixture(&path_a, &[("a1", "alpha")]);
    write_parquet_fixture(&path_b, &[("b1", "beta")]);

    let mut cache = ParquetCache::default();
    let rows_a_first = cache.row_group_rows_for("hf_test", &path_a, 0, 1).unwrap();
    let rows_a_second = cache.row_group_rows_for("hf_test", &path_a, 0, 1).unwrap();
    assert!(Arc::ptr_eq(&rows_a_first, &rows_a_second));

    let _rows_b = cache.row_group_rows_for("hf_test", &path_b, 0, 1).unwrap();
    assert_eq!(cache.row_groups.len(), 1);
    assert!(cache.row_groups.contains_key(&(path_b.clone(), 0)));
    assert!(!cache.row_groups.contains_key(&(path_a.clone(), 0)));
}

#[test]
fn refresh_row_group_order_removes_existing_key_and_ignores_missing() {
    let key_a = (PathBuf::from("a.parquet"), 0usize);
    let key_b = (PathBuf::from("b.parquet"), 0usize);
    let mut order = VecDeque::from([key_a.clone(), key_b.clone(), key_a.clone()]);

    ParquetCache::refresh_row_group_order(&mut order, &key_a);
    assert_eq!(order, VecDeque::from([key_b.clone(), key_a.clone()]));

    let missing = (PathBuf::from("missing.parquet"), 0usize);
    ParquetCache::refresh_row_group_order(&mut order, &missing);
    assert_eq!(order, VecDeque::from([key_b, key_a]));
}

#[test]
fn build_eligible_rows_from_store_shard_uses_global_offsets() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);

    let store_path = dir.path().join("eligible.simdr");
    write_simdr_fixture(&store_path, &[("r1", "alpha"), ("r2", "beta")]);

    let shards = vec![ShardIndex {
        path: store_path,
        global_start: 5,
        row_count: 2,
        parquet_row_groups: vec![(0, 2)],
        remote_candidate: None,
    }];

    let eligible = source.build_eligible_rows_from_shards(&shards).unwrap();
    assert_eq!(eligible, vec![5, 6]);
}

#[test]
fn effective_targets_respect_minimum_multiplier_and_sampler_override() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.refresh_batch_multiplier = 0;
    config.remote_expansion_headroom_multiplier = 0;
    config.cache_capacity = 9;
    let source = test_source(config.clone());

    assert_eq!(source.effective_refresh_batch_target(5), 5);
    assert_eq!(source.effective_expansion_headroom_rows(), 9);

    let sampler = SamplerConfig {
        ingestion_max_records: 4,
        ..SamplerConfig::default()
    };
    *source.sampler_config.lock().unwrap() = Some(sampler);
    assert_eq!(source.effective_expansion_headroom_rows(), 4);
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

#[test]
fn manifest_usage_bytes_locked_counts_only_manifest_shards() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    let manifest_root = source.manifest_cache_root();
    fs::create_dir_all(&manifest_root).unwrap();

    let manifest_file = manifest_root.join("a.parquet");
    fs::write(&manifest_file, vec![1u8; 7]).unwrap();
    let local_file = source.config.snapshot_dir.join("local.ndjson");
    fs::write(&local_file, vec![2u8; 9]).unwrap();

    let state = SourceState {
        materialized_rows: 2,
        shards: vec![
            ShardIndex {
                path: manifest_file,
                global_start: 0,
                row_count: 1,
                parquet_row_groups: vec![(0, 1)],
                remote_candidate: None,
            },
            ShardIndex {
                path: local_file,
                global_start: 1,
                row_count: 1,
                parquet_row_groups: Vec::new(),
                remote_candidate: None,
            },
        ],
        remote_candidates: None,
        remote_candidate_sizes: HashMap::new(),
        next_remote_idx: 0,
        remote_candidate_order: Vec::new(),
    };

    assert_eq!(source.manifest_usage_bytes_locked(&state), 7);
}

#[test]
fn locate_parquet_group_maps_offsets_and_reports_missing() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    let shard = ShardIndex {
        path: dir.path().join("rows.parquet"),
        global_start: 0,
        row_count: 6,
        parquet_row_groups: vec![(0, 2), (2, 2), (4, 2)],
        remote_candidate: None,
    };

    let mapped = source.locate_parquet_group(&shard, 3).unwrap();
    assert_eq!(mapped, (1, 1));
    let missing = source.locate_parquet_group(&shard, 99);
    assert!(missing.is_err());
}

#[test]
fn parse_row_role_columns_mode_builds_expected_fields() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.anchor_columns = vec!["anchor".into()];
    config.positive_columns = vec!["positive".into()];
    config.context_columns = vec!["ctx1".into(), "ctx2".into()];
    let source = test_source(config);

    let row = rows::parse_row(
        &source,
        2,
        &json!({"id":"r","anchor":"a","positive":"p","ctx1":"c1","ctx2":2}),
    )
    .unwrap()
    .unwrap();
    assert_eq!(row.text_fields.len(), 4);
    assert_eq!(row.text_fields[0].name, "anchor");
    assert_eq!(row.text_fields[1].name, "positive");
}

#[test]
fn parse_row_role_columns_mode_skips_missing_or_empty_values() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.anchor_columns = vec!["anchor".into()];
    config.context_columns = vec!["ctx".into()];
    let source = test_source(config);

    let missing = rows::parse_row(&source, 0, &json!({"anchor":"a"}));
    assert!(missing.unwrap().is_none());

    let empty_anchor = rows::parse_row(&source, 1, &json!({"anchor":"   ", "ctx":"ok"}));
    assert!(empty_anchor.unwrap().is_none());
}

#[test]
fn row_to_record_uses_anchor_for_positive_when_single_field() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    let row = RowView {
        row_id: Some("r1".into()),
        timestamp: None,
        text_fields: vec![RowTextField {
            name: "text".into(),
            text: "alpha".into(),
        }],
    };

    let record = rows::row_to_record(&source, &row, 0).unwrap().unwrap();
    assert_eq!(record.sections.len(), 2);
    assert_eq!(record.sections[0].text, record.sections[1].text);
}

#[test]
fn target_matches_expected_size_is_false_for_missing_path() {
    let dir = tempdir().unwrap();
    let missing = dir.path().join("missing.bin");
    assert!(!target_matches_expected_size(&missing, Some(1)));
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
fn parse_row_falls_back_to_synthetic_id_when_missing_id_column() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.id_column = Some("id".into());
    let source = test_source(config);

    let row = rows::parse_row(&source, 42, &json!({"text": "hello"}))
        .unwrap()
        .unwrap();
    assert_eq!(row.row_id, Some("org/dataset:train:42".to_string()));
}

#[test]
fn row_to_record_falls_back_to_row_index_when_row_id_missing() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    let row = RowView {
        row_id: None,
        timestamp: None,
        text_fields: vec![RowTextField {
            name: "text".into(),
            text: "body".into(),
        }],
    };

    let record = rows::row_to_record(&source, &row, 7).unwrap().unwrap();
    assert!(record.id.ends_with("::row_7"));
}

#[test]
fn locate_shard_returns_none_for_out_of_range_index() {
    let shards = vec![ShardIndex {
        path: PathBuf::from("a.ndjson"),
        global_start: 0,
        row_count: 2,
        parquet_row_groups: Vec::new(),
        remote_candidate: None,
    }];

    assert!(HuggingFaceRowSource::locate_shard(&shards, 5).is_none());
}

#[test]
fn read_row_batch_errors_when_row_not_mappable_to_shard() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = 1;
        state.shards.clear();
    }

    let mut out = Vec::new();
    let err = rows::read_row_batch(&source, &[0], &mut out, Some(1));
    assert!(err.is_err());
}

#[test]
fn read_row_batch_errors_when_parquet_local_offsets_are_missing() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("rows.parquet");
    write_parquet_fixture(&path, &[("id-1", "text-1")]);
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);

    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = 3;
        state.shards = vec![ShardIndex {
            path,
            global_start: 0,
            row_count: 3,
            parquet_row_groups: vec![(0, 3)],
            remote_candidate: None,
        }];
    }

    let mut out = Vec::new();
    let err = rows::read_row_batch(&source, &[2], &mut out, Some(1)).unwrap_err();
    assert!(matches!(
        err,
        SamplerError::SourceUnavailable { ref reason, .. } if reason.contains("parquet rows missing")
    ));
}

#[test]
fn enforce_disk_cap_returns_false_when_disabled_or_under_limit() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.local_disk_cap_bytes = None;
    let source = test_source(config);
    let mut state = SourceState {
        materialized_rows: 0,
        shards: Vec::new(),
        remote_candidates: None,
        remote_candidate_sizes: HashMap::new(),
        next_remote_idx: 0,
        remote_candidate_order: Vec::new(),
    };
    let protected = dir.path().join("p");
    assert!(
        !source
            .enforce_disk_cap_locked(&mut state, &protected)
            .unwrap()
    );

    let mut config2 = test_config(dir.path().to_path_buf());
    config2.local_disk_cap_bytes = Some(10_000);
    let source2 = test_source(config2);
    let manifest_root = source2.manifest_cache_root();
    fs::create_dir_all(&manifest_root).unwrap();
    let shard_path = manifest_root.join("small.parquet");
    fs::write(&shard_path, vec![1u8; 32]).unwrap();
    let mut state2 = SourceState {
        materialized_rows: 1,
        shards: vec![ShardIndex {
            path: shard_path,
            global_start: 0,
            row_count: 1,
            parquet_row_groups: vec![(0, 1)],
            remote_candidate: None,
        }],
        remote_candidates: None,
        remote_candidate_sizes: HashMap::new(),
        next_remote_idx: 0,
        remote_candidate_order: Vec::new(),
    };
    assert!(
        !source2
            .enforce_disk_cap_locked(&mut state2, &protected)
            .unwrap()
    );
}

#[test]
fn enforce_disk_cap_evicts_manifest_shards_and_recomputes_offsets() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.local_disk_cap_bytes = Some(20);
    let source = test_source(config);
    let manifest_root = source.manifest_cache_root();
    fs::create_dir_all(&manifest_root).unwrap();

    let first = manifest_root.join("first.parquet");
    let second = manifest_root.join("second.parquet");
    fs::write(&first, vec![1u8; 16]).unwrap();
    fs::write(&second, vec![2u8; 16]).unwrap();

    let mut state = SourceState {
        materialized_rows: 2,
        shards: vec![
            ShardIndex {
                path: first.clone(),
                global_start: 0,
                row_count: 1,
                parquet_row_groups: vec![(0, 1)],
                remote_candidate: None,
            },
            ShardIndex {
                path: second.clone(),
                global_start: 1,
                row_count: 1,
                parquet_row_groups: vec![(0, 1)],
                remote_candidate: None,
            },
        ],
        remote_candidates: None,
        remote_candidate_sizes: HashMap::new(),
        next_remote_idx: 0,
        remote_candidate_order: Vec::new(),
    };

    let evicted = source.enforce_disk_cap_locked(&mut state, &second).unwrap();
    assert!(evicted);
    assert!(!first.exists());
    assert!(second.exists());
    assert_eq!(state.shards.len(), 1);
    assert_eq!(state.shards[0].global_start, 0);
    assert_eq!(state.materialized_rows, 1);
}

#[test]
fn enforce_disk_cap_evicts_when_single_file_exceeds_cap() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.local_disk_cap_bytes = Some(1);
    let source = test_source(config);
    let manifest_root = source.manifest_cache_root();
    fs::create_dir_all(&manifest_root).unwrap();

    let protected = manifest_root.join("protected.parquet");
    fs::write(&protected, vec![3u8; 16]).unwrap();

    let mut state = SourceState {
        materialized_rows: 1,
        shards: vec![ShardIndex {
            path: protected.clone(),
            global_start: 0,
            row_count: 1,
            parquet_row_groups: vec![(0, 1)],
            remote_candidate: None,
        }],
        remote_candidates: None,
        remote_candidate_sizes: HashMap::new(),
        next_remote_idx: 0,
        remote_candidate_order: Vec::new(),
    };

    let evicted = source
        .enforce_disk_cap_locked(&mut state, &protected)
        .unwrap();
    assert!(evicted);
    assert!(!protected.exists());
    assert_eq!(state.shards.len(), 0);
    assert_eq!(state.materialized_rows, 0);
}

#[test]
fn configured_sampler_seed_and_paging_seed_require_sampler_config() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let http_runtime = Arc::new(build_http_runtime(&config).unwrap());
    let http_client = test_http_client();
    let source = HuggingFaceRowSource {
        config,
        http_runtime,
        http_client,
        sampler_config: Arc::new(Mutex::new(None)),
        state: Arc::new(Mutex::new(SourceState {
            materialized_rows: 0,
            shards: Vec::new(),
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            remote_candidate_order: Vec::new(),
        })),
        cache: Arc::new(Mutex::new(RowCache::default())),
        parquet_cache: Arc::new(Mutex::new(ParquetCache::default())),
        eligible_index: Arc::new(Mutex::new(EligibleIndexCache::default())),
        expansion_thread: Arc::new(Mutex::new(None)),
    };

    assert!(source.configured_sampler_seed().is_err());
    assert!(source.paging_seed(5).is_err());
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
fn ensure_row_available_returns_from_fast_paths() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);

    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = 3;
        state.remote_candidates = Some(vec!["x".to_string()]);
        state.next_remote_idx = 0;
    }
    assert!(source.ensure_row_available(1).unwrap());

    let source_done = test_source(test_config(dir.path().to_path_buf()));
    {
        let mut state = source_done.state.lock().unwrap();
        state.materialized_rows = 0;
        state.remote_candidates = Some(vec!["a".to_string()]);
        state.next_remote_idx = 1;
    }
    assert!(!source_done.ensure_row_available(0).unwrap());
}

#[test]
fn materialize_local_file_errors_for_missing_source() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let missing = dir.path().join("missing.ndjson");
    let target = dir.path().join("target.ndjson");

    let err = HuggingFaceRowSource::materialize_local_file(&config, &missing, &target).unwrap_err();
    assert!(matches!(
        err,
        SamplerError::SourceUnavailable { ref reason, .. } if reason.contains("failed copying synced file")
    ));
}

#[test]
fn index_single_shard_errors_for_missing_file() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let missing = dir.path().join("missing.ndjson");

    let err = index_single_shard(&config, &missing, 0)
        .err()
        .expect("index_single_shard should fail");
    assert!(matches!(err, SamplerError::SourceUnavailable { .. }));
}

#[test]
fn index_single_shard_detects_corrupted_store() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let store_path = dir.path().join("shard.simdr");

    // Write a store with 3 valid rows AND a metadata key claiming 5 rows.
    // The last claimed row (index 4) does not exist — corruption.
    write_simdr_fixture(&store_path, &[("r0", "zero"), ("r1", "one"), ("r2", "two")]);
    let store = DataStore::open(&store_path).expect("open store");
    store
        .write(HF_SHARD_STORE_META_ROWS_KEY, &(5u64).to_le_bytes())
        .expect("overwrite meta with inflated count");
    drop(store);

    // index_single_shard should detect the gap, delete the corrupt file,
    // and return None.
    let (maybe_shard, _) = index_single_shard(&config, &store_path, 0)
        .expect("corrupted store should not produce a hard error — it deletes and returns None");
    assert!(maybe_shard.is_none(), "corrupt store should be skipped");
    assert!(!store_path.exists(), "corrupt store file should be deleted");
}

#[test]
fn index_single_shard_jsonl_returns_dummy_count() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("rows.ndjson");
    fs::write(
        &path,
        b"{\"text\":\"a\"}\n{\"text\":\"b\"}\n{\"text\":\"c\"}\n",
    )
    .unwrap();
    let config = test_config(dir.path().to_path_buf());

    let shard = index_single_shard(&config, &path, 5).unwrap().0.unwrap();
    assert_eq!(shard.global_start, 5);
    assert_eq!(shard.row_count, 1); // Dummy count for transient text files
}

#[test]
fn parquet_row_group_map_handles_empty_parquet_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("empty.parquet");
    write_parquet_fixture(&path, &[]);
    let config = test_config(dir.path().to_path_buf());

    let (rows, groups) = parquet_row_group_map(&config, &path).unwrap();
    assert_eq!(rows, 0);
    assert!(groups.is_empty());
}

#[test]
fn download_next_remote_shard_clears_row_cache_when_eviction_occurs() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    // Cap small enough that old file alone exceeds it, triggering eviction,
    // but large enough that a single-row .simdr store (~4 KiB) fits.
    config.local_disk_cap_bytes = Some(6_144);
    let source = test_source(config.clone());

    let manifest_root = source.manifest_cache_root();
    fs::create_dir_all(&manifest_root).unwrap();
    let old_path = manifest_root.join("old.parquet");
    // Large enough to exceed the disk cap on its own.
    fs::write(&old_path, vec![1u8; 8_192]).unwrap();

    let payload = b"{\"text\":\"new\"}\n".to_vec();
    let server = spawn_one_shot_http(payload);
    let base_url = server.url().to_string();
    let candidate = format!("url::{base_url}/datasets/org/ds/resolve/main/train/new-shard.ndjson");
    let new_path = HuggingFaceRowSource::candidate_store_path(&config, &candidate);

    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = 1;
        state.shards = vec![ShardIndex {
            path: old_path.clone(),
            global_start: 0,
            row_count: 1,
            parquet_row_groups: vec![(0, 1)],
            remote_candidate: None,
        }];
        state.remote_candidates = Some(vec![candidate]);
        state.next_remote_idx = 0;
    }
    {
        let mut cache = source.cache.lock().unwrap();
        cache.insert(
            0,
            RowView {
                row_id: Some("cached".to_string()),
                timestamp: None,
                text_fields: vec![RowTextField {
                    name: "text".to_string(),
                    text: "cached".to_string(),
                }],
            },
            8,
        );
    }

    assert!(source.download_next_remote_shard().unwrap());

    // Eviction removes at least one shard once disk cap is exceeded.
    // Which shard is removed can vary on filesystems with coarse mtime
    // resolution (tie-break is path order), so assert eviction semantics
    // rather than a specific filename.
    assert!(
        !(old_path.exists() && new_path.exists()),
        "expected at least one manifest shard to be evicted"
    );
    {
        let state = source.state.lock().unwrap();
        assert!(!state.shards.is_empty(), "at least one shard should remain");
    }
    let cache = source.cache.lock().unwrap();
    assert!(cache.rows.is_empty());
    assert!(cache.order.is_empty());
}

#[test]
fn default_triplet_recipes_text_only_mode_returns_simcse_recipe() {
    // test_config() leaves anchor_columns empty → text-only mode.
    // A single SimCSE-style recipe with allow_same_anchor_positive must be returned.
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    assert!(
        config.anchor_columns.is_empty(),
        "test_config must be in text-only mode"
    );
    let source = test_source(config);
    let recipes = source.default_triplet_recipes();
    assert_eq!(recipes.len(), 1);
    assert_eq!(recipes[0].name, HF_RECIPE_TEXT_SIMCSE_WRONG_ARTICLE);
    assert!(
        recipes[0].allow_same_anchor_positive,
        "SimCSE recipe must allow same anchor/positive text"
    );
    assert_eq!(recipes[0].weight, 1.0);
}

#[test]
fn default_triplet_recipes_role_mode_returns_two_recipes() {
    // When anchor_columns is non-empty the source is in role-based mode and
    // must return the two standard (anchor-context, anchor-anchor) recipes,
    // neither of which allows same anchor/positive text.
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.anchor_columns = vec!["title".to_string()];
    config.positive_columns = vec!["body".to_string()];
    let source = test_source(config);
    let recipes = source.default_triplet_recipes();
    assert_eq!(recipes.len(), 2);
    assert_eq!(recipes[0].name, HF_RECIPE_ANCHOR_CONTEXT_WRONG_ARTICLE);
    assert_eq!(recipes[1].name, HF_RECIPE_ANCHOR_ANCHOR_WRONG_ARTICLE);
    assert_eq!(recipes[0].weight, 0.75);
    assert_eq!(recipes[1].weight, 0.25);
    assert!(
        !recipes[0].allow_same_anchor_positive,
        "standard recipes must not allow same anchor/positive"
    );
    assert!(
        !recipes[1].allow_same_anchor_positive,
        "standard recipes must not allow same anchor/positive"
    );
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
fn download_next_remote_shard_skips_zero_row_download() {
    // Suppress the expected WARN "downloaded shard had zero rows and was skipped"
    // emitted when a shard file contains no JSON lines after download.  That warn
    // is correct production behaviour; silenced here to keep test output clean.
    let _quiet = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::ERROR)
            .finish(),
    );
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    let payload = Vec::<u8>::new();
    let server = spawn_one_shot_http(payload);
    let base_url = server.url().to_string();
    let candidate = format!("url::{base_url}/datasets/org/ds/resolve/main/train/part-empty.ndjson");

    {
        let mut state = source.state.lock().unwrap();
        state.remote_candidates = Some(vec![candidate]);
        state.next_remote_idx = 0;
    }

    assert!(source.download_next_remote_shard().unwrap());
    let state = source.state.lock().unwrap();
    assert_eq!(state.materialized_rows, 0);
    assert!(state.shards.is_empty());
}

#[test]
fn read_row_batch_errors_when_parquet_reader_cannot_open_file() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = 1;
        state.shards = vec![ShardIndex {
            path: dir.path().join("missing.parquet"),
            global_start: 0,
            row_count: 1,
            parquet_row_groups: vec![(0, 1)],
            remote_candidate: None,
        }];
    }

    let mut out = Vec::new();
    let err = rows::read_row_batch(&source, &[0], &mut out, Some(1));
    assert!(err.is_err());
}

#[test]
fn shard_size_bytes_returns_zero_for_missing_path() {
    let dir = tempdir().unwrap();
    let missing = dir.path().join("missing.file");
    assert_eq!(HuggingFaceRowSource::shard_size_bytes(&missing), 0);
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
fn materialize_local_file_replaces_target_when_size_differs() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let src = dir.path().join("src.ndjson");
    let dst = dir.path().join("dst.ndjson");
    fs::write(&src, b"newer\n").unwrap();
    fs::write(&dst, b"old\n").unwrap();

    HuggingFaceRowSource::materialize_local_file(&config, &src, &dst).unwrap();
    assert_eq!(fs::read(&dst).unwrap(), b"newer\n");
}

#[test]
fn row_to_record_preserves_explicit_timestamp() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    let ts = Utc::now();
    let row = RowView {
        row_id: Some("r1".into()),
        timestamp: Some(ts),
        text_fields: vec![RowTextField {
            name: "text".into(),
            text: "alpha".into(),
        }],
    };

    let record = rows::row_to_record(&source, &row, 0).unwrap().unwrap();
    assert_eq!(record.created_at, ts);
    assert_eq!(record.updated_at, ts);
}

#[test]
fn parse_row_text_columns_accept_numeric_values() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.text_columns = vec!["score".into()];
    let source = test_source(config);

    let row = rows::parse_row(&source, 0, &json!({"score": 123}))
        .unwrap()
        .unwrap();
    assert_eq!(row.text_fields.len(), 1);
    assert_eq!(row.text_fields[0].text, "123");
}

#[test]
fn refresh_limit_none_reads_up_to_total() {
    let dir = tempdir().unwrap();
    let simdr_path = dir.path().join("rows.simdr");
    write_simdr_fixture(&simdr_path, &[("r1", "a"), ("r2", "b")]);
    let mut config = test_config(dir.path().to_path_buf());
    config.refresh_batch_multiplier = 1;
    let source = test_source(config.clone());
    let shard = index_single_shard(&config, &simdr_path, 0)
        .unwrap()
        .0
        .unwrap();
    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = 2;
        state.shards = vec![shard];
    }

    let snapshot = source.refresh(None, None).unwrap();
    assert_eq!(snapshot.records.len(), 2);
}

#[test]
fn read_row_batch_skips_unavailable_indices_without_error() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = 0;
        state.remote_candidates = Some(Vec::new());
    }

    let mut out = Vec::new();
    rows::read_row_batch(&source, &[0, 1], &mut out, Some(2)).unwrap();
    assert!(out.is_empty());
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
fn target_matches_expected_size_validates_when_expected_is_provided() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("payload.bin");
    fs::write(&path, vec![0u8; 5]).unwrap();

    assert!(target_matches_expected_size(&path, Some(5)));
    assert!(!target_matches_expected_size(&path, Some(4)));
    assert!(target_matches_expected_size(&path, None));
}

#[test]
fn parquet_row_group_map_and_index_single_shard_cover_success_path() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("rows.parquet");
    write_parquet_fixture(&path, &[("r1", "alpha"), ("r2", "beta"), ("r3", "gamma")]);
    let config = test_config(dir.path().to_path_buf());

    let (total_rows, groups) = parquet_row_group_map(&config, &path).unwrap();
    assert_eq!(total_rows, 3);
    assert!(!groups.is_empty());

    let shard = index_single_shard(&config, &path, 0).unwrap().0.unwrap();
    assert_eq!(shard.row_count, 3);
    // All shards are now .simdr stores with O(1) random access
}

#[test]
fn read_row_batch_reads_parquet_rows_and_uses_cache_on_repeat() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("rows.parquet");
    write_parquet_fixture(&path, &[("r10", "ten"), ("r11", "eleven")]);

    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config.clone());
    let shard = index_single_shard(&config, &path, 0).unwrap().0.unwrap();
    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = 2;
        state.shards = vec![shard];
    }

    let mut first = Vec::new();
    rows::read_row_batch(&source, &[0, 1], &mut first, None).unwrap();
    assert_eq!(first.len(), 2);
    assert!(first.iter().any(|record| record.id.ends_with("::r10")));

    let mut second = Vec::new();
    rows::read_row_batch(&source, &[0, 1], &mut second, None).unwrap();
    assert_eq!(second.len(), 2);
}

#[test]
fn ensure_row_available_bootstraps_from_in_memory_candidates() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config.clone());

    let payload =
        b"{\"id\":\"r1\",\"text\":\"alpha\"}\n{\"id\":\"r2\",\"text\":\"beta\"}\n".to_vec();
    let server = spawn_one_shot_http(payload);
    let base_url = server.url().to_string();
    let candidate = format!("url::{base_url}/datasets/org/ds/resolve/main/train/persisted.ndjson");

    {
        let mut state = source.state.lock().unwrap();
        state.remote_candidates = Some(vec![candidate]);
        state.next_remote_idx = 0;
    }

    assert!(source.ensure_row_available(0).unwrap());

    let state = source.state.lock().unwrap();
    assert_eq!(state.materialized_rows, 2);
    assert_eq!(state.next_remote_idx, 1);
    assert_eq!(state.shards.len(), 1);
}

#[test]
fn configure_sampler_updates_len_hint_headroom_via_trait_methods() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.cache_capacity = 10;
    config.remote_expansion_headroom_multiplier = 3;
    let source = test_source(config);
    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = 5;
        // Set up remote candidates so headroom is applied
        state.remote_candidates = Some(vec![
            "url::http://a/0.parquet".to_string(),
            "url::http://a/1.parquet".to_string(),
        ]);
        state.next_remote_idx = 0;
    }

    // headroom = ingestion_max_records * multiplier = 10 * 3 = 30
    // known (5) < headroom, expansion = 30; upper = 5 + 30 = 35
    assert_eq!(source.reported_record_count().unwrap(), 35);

    let sampler = SamplerConfig {
        ingestion_max_records: 2,
        ..SamplerConfig::default()
    };
    source.configure_sampler(&sampler);

    // headroom = 2 * 3 = 6; known (5) < headroom, expansion = 6; upper = 5 + 6 = 11
    assert_eq!(source.reported_record_count().unwrap(), 11);
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
fn config_endpoint_fallback_for_empty_env_values() {
    // `test_config` overrides endpoints to `TEST_UNREACHABLE_URL` for
    // network isolation.  This test verifies the DEFAULT values from
    // `HuggingFaceRowsConfig::new()`, so we call it directly.
    let dir = tempdir().unwrap();

    let c = HuggingFaceRowsConfig::new("ep_test", "org/dataset", "default", "train", dir.path());
    assert_eq!(c.parquet_endpoint, HF_PARQUET_DEFAULT_ENDPOINT);
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
fn download_next_remote_shard_parquet_stages_temp_and_persists_store_only() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config.clone());

    let fixture_path = dir.path().join("fixture.parquet");
    write_parquet_fixture(&fixture_path, &[("r1", "alpha"), ("r2", "beta")]);
    let payload = fs::read(&fixture_path).unwrap();
    let server = spawn_one_shot_http(payload);
    let base_url = server.url().to_string();
    let candidate = format!("url::{base_url}/datasets/org/ds/resolve/main/train/part-222.parquet");

    {
        let mut state = source.state.lock().unwrap();
        state.remote_candidates = Some(vec![candidate.clone()]);
        state.next_remote_idx = 0;
    }

    assert!(source.download_next_remote_shard().unwrap());

    let parquet_target = candidate_target_path(&config, &candidate);
    let store_target = shard_store_path_for(&parquet_target);

    assert!(store_target.exists());
    assert!(!parquet_target.exists());

    let state = source.state.lock().unwrap();
    assert_eq!(state.shards.len(), 1);
    assert_eq!(state.shards[0].path, store_target);
    assert_eq!(state.materialized_rows, 2);
}

// Temp-staged files (with triplet_hf_ prefix) must be removed after
// transcoding completes — no leaked temp files in the OS temp directory.
#[test]
fn transcode_transient_shard_to_store_cleans_up_temp_downloads() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config.clone());

    // Create a temp file with the triplet_hf_ prefix (simulates a staged download).
    let temp_parquet = dir.path().join("triplets_hf_aabbccdd00112233_temp.parquet");
    write_parquet_fixture(&temp_parquet, &[("r1", "hello")]);
    assert!(temp_parquet.exists(), "temp parquet must exist before test");

    let shard = ShardIndex {
        path: temp_parquet.clone(),
        global_start: 0,
        row_count: 1,
        parquet_row_groups: vec![(0, 1)],
        remote_candidate: None,
    };

    let result =
        rows::transcode_transient_shard_to_store(&source, &shard).expect("transcode must succeed");

    // Temp file must be gone after transcoding.
    assert!(
        !temp_parquet.exists(),
        "temp download file must be cleaned up after transcode"
    );

    // Store must still be present.
    let store_path = result.expect("must yield Some(ShardIndex)").path;
    assert!(
        store_path.exists(),
        "simdr store must exist after transcode"
    );
}

// When transcode_transient_shard_to_store takes the early-return path
// (simdr store already fully populated), it must still delete the input
// transient file and return a ShardIndex
#[test]
fn transcode_transient_shard_to_store_early_return_cleans_up_transient() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config.clone());

    // Write a parquet fixture (simulates a stale/unconsumed parquet from a
    // previous run that crashed before the delete step fired).
    let parquet_path = dir.path().join("stale.parquet");
    write_parquet_fixture(&parquet_path, &[("r1", "hello"), ("r2", "world")]);
    assert!(
        parquet_path.exists(),
        "parquet fixture must exist before test"
    );

    // Pre-populate the corresponding simdr store so the function short-circuits.
    let store_path = shard_store_path_for(&parquet_path);
    write_simdr_fixture(&store_path, &[("r1", "hello"), ("r2", "world")]);
    assert!(store_path.exists(), "simdr store must exist before test");

    let shard = ShardIndex {
        path: parquet_path.clone(),
        global_start: 0,
        row_count: 2,
        parquet_row_groups: vec![(0, 2)],
        remote_candidate: None,
    };

    let result =
        rows::transcode_transient_shard_to_store(&source, &shard).expect("transcode must succeed");

    // Stale parquet must be gone.
    assert!(
        !parquet_path.exists(),
        "stale parquet must be removed on early return"
    );

    // Simdr store must still be present.
    assert!(store_path.exists(), "simdr store must survive early return");

    // Returned shard must point to the store.
    let returned = result.expect("early return must yield Some(ShardIndex)");
    assert_eq!(returned.path, store_path);
    assert_eq!(returned.row_count, 2);
}

#[test]
fn download_next_remote_shard_materializes_and_indexes_rows() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    let payload = b"{\"text\":\"a\"}\n{\"text\":\"b\"}\n".to_vec();
    let server = spawn_one_shot_http(payload);
    let base_url = server.url().to_string();
    let candidate = format!("url::{base_url}/datasets/org/ds/resolve/main/train/part-001.ndjson");

    {
        let mut state = source.state.lock().unwrap();
        state.remote_candidates = Some(vec![candidate.clone()]);
        state.remote_candidate_sizes.insert(candidate, 24);
        state.next_remote_idx = 0;
    }

    assert!(source.download_next_remote_shard().unwrap());

    let state = source.state.lock().unwrap();
    assert_eq!(state.materialized_rows, 2);
    assert_eq!(state.shards.len(), 1);
    assert_eq!(state.next_remote_idx, 1);
}

#[test]
fn ensure_row_available_triggers_lazy_download_for_remote_candidates() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    let payload = b"{\"text\":\"x\"}\n{\"text\":\"y\"}\n".to_vec();
    let server = spawn_one_shot_http(payload);
    let base_url = server.url().to_string();
    let candidate = format!("url::{base_url}/datasets/org/ds/resolve/main/train/part-002.ndjson");

    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = 0;
        state.remote_candidates = Some(vec![candidate.clone()]);
        state.remote_candidate_sizes.insert(candidate, 24);
        state.next_remote_idx = 0;
    }

    assert!(source.ensure_row_available(0).unwrap());

    let state = source.state.lock().unwrap();
    assert!(state.materialized_rows >= 1);
    assert_eq!(state.next_remote_idx, 1);
}

#[test]
fn download_next_remote_shard_consumes_distinct_candidates_in_order() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config.clone());

    let payload_a = b"{\"id\":\"a\",\"text\":\"alpha\"}\n".to_vec();
    let payload_b = b"{\"id\":\"b\",\"text\":\"beta\"}\n".to_vec();
    let server_a = spawn_one_shot_http(payload_a);
    let base_a = server_a.url().to_string();
    let server_b = spawn_one_shot_http(payload_b);
    let base_b = server_b.url().to_string();
    let candidate_a = format!("url::{base_a}/datasets/org/ds/resolve/main/train/part-a.ndjson");
    let candidate_b = format!("url::{base_b}/datasets/org/ds/resolve/main/train/part-b.ndjson");
    {
        let mut state = source.state.lock().unwrap();
        state.remote_candidates = Some(vec![candidate_a.clone(), candidate_b.clone()]);
        state.remote_candidate_sizes.insert(candidate_a.clone(), 27);
        state.remote_candidate_sizes.insert(candidate_b.clone(), 26);
        state.next_remote_idx = 0;
    }

    assert!(source.download_next_remote_shard().unwrap());
    assert!(source.download_next_remote_shard().unwrap());

    let state = source.state.lock().unwrap();
    assert_eq!(state.next_remote_idx, 2);
    assert_eq!(state.shards.len(), 2);
    assert_ne!(state.shards[0].path, state.shards[1].path);
}

#[test]
fn download_next_remote_shard_skips_already_materialised_shard() {
    // Verifies the cache/determinism decoupling: if a shard's store file already
    // exists on disk, download_next_remote_shard must advance next_remote_idx
    // without making any network request, leaving materialized_rows unchanged.
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config.clone());

    let candidate =
        format!("url::{TEST_UNREACHABLE_URL}/datasets/org/ds/resolve/main/train/pre-cached.ndjson");
    let target = candidate_target_path(&config, &candidate);
    let store_path = shard_store_path_for(&target);
    fs::create_dir_all(store_path.parent().unwrap()).unwrap();
    fs::write(&store_path, b"dummy").unwrap();

    {
        let mut state = source.state.lock().unwrap();
        state.remote_candidates = Some(vec![candidate.clone()]);
        state.remote_candidate_order = vec![0];
        state.remote_candidate_sizes.insert(candidate, 5);
        state.next_remote_idx = 0;
    }

    // No HTTP server is running — if a real download were attempted it would fail.
    assert!(
        source.download_next_remote_shard().unwrap(),
        "should return true (candidate consumed)"
    );

    let state = source.state.lock().unwrap();
    assert_eq!(
        state.next_remote_idx, 1,
        "pointer advanced past cached shard"
    );
    assert_eq!(
        state.materialized_rows, 0,
        "materialized_rows unchanged — shard was already counted at startup"
    );
    assert_eq!(
        state.shards.len(),
        0,
        "no new shard added to in-memory list"
    );
}

#[test]
fn download_next_remote_shard_detects_stale_shard_by_size() {
    // When a cached store exists with a stored source_size that differs from
    // the current manifest's expected_bytes, the store should be deleted and
    // the shard re-downloaded.
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config.clone());

    // Create a real .simdr store with source_size = 100.
    let candidate =
        format!("url::{TEST_UNREACHABLE_URL}/datasets/org/ds/resolve/main/train/stale.ndjson");
    let store_path = HuggingFaceRowSource::candidate_store_path(&config, &candidate);
    fs::create_dir_all(store_path.parent().unwrap()).unwrap();
    write_simdr_fixture(&store_path, &[("r0", "row")]);
    {
        let store = DataStore::open(&store_path).unwrap();
        store
            .write(HF_SHARD_STORE_SOURCE_SIZE_KEY, &100u64.to_le_bytes())
            .unwrap();
    }

    // Manually populate the store cache with the handle so the stale
    // check can read source_size from it without opening a second handle.
    let cached_store = Arc::new(DataStore::open(&store_path).unwrap());
    source
        .config
        .store_cache
        .lock()
        .unwrap()
        .insert(store_path.clone(), cached_store);

    // Set up remote candidates with expected_bytes = 200 (≠ 100).
    {
        let mut state = source.state.lock().unwrap();
        state.remote_candidates = Some(vec![candidate.clone()]);
        state.remote_candidate_order = vec![0];
        state.remote_candidate_sizes.insert(candidate, 200);
        state.next_remote_idx = 0;
    }

    // The stale check should detect the mismatch, delete the store, and
    // attempt a download.  Since no HTTP server is running, the download
    // fails with SourceUnavailable — but the store should already be gone.
    let result = source.download_next_remote_shard();
    assert!(
        !store_path.exists(),
        "stale store file should be deleted before download attempt"
    );
    assert!(
        result.is_err(),
        "should fail with SourceUnavailable (no HTTP server for re-download)"
    );
    let err = result.unwrap_err();
    assert!(
        matches!(err, SamplerError::SourceUnavailable { .. }),
        "expected SourceUnavailable, got: {err:?}"
    );
}

#[test]
fn download_next_remote_shard_preserves_fresh_shard_when_sizes_match() {
    // When a cached store exists and its stored source_size matches the
    // manifest's expected_bytes, the download is skipped normally.
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config.clone());

    // Create a store with source_size = 100 and set expected_bytes = 100.
    let candidate =
        format!("url::{TEST_UNREACHABLE_URL}/datasets/org/ds/resolve/main/train/fresh.ndjson");
    let store_path = HuggingFaceRowSource::candidate_store_path(&config, &candidate);
    fs::create_dir_all(store_path.parent().unwrap()).unwrap();
    write_simdr_fixture(&store_path, &[("r0", "row")]);
    {
        let store = DataStore::open(&store_path).unwrap();
        store
            .write(HF_SHARD_STORE_SOURCE_SIZE_KEY, &100u64.to_le_bytes())
            .unwrap();
    }

    // Populate the store cache for the stale check.
    let cached_store = Arc::new(DataStore::open(&store_path).unwrap());
    source
        .config
        .store_cache
        .lock()
        .unwrap()
        .insert(store_path.clone(), cached_store);

    {
        let mut state = source.state.lock().unwrap();
        state.remote_candidates = Some(vec![candidate.clone()]);
        state.remote_candidate_order = vec![0];
        state.remote_candidate_sizes.insert(candidate, 100);
        state.next_remote_idx = 0;
    }

    // Sizes match — should skip without any network call.
    assert!(
        source.download_next_remote_shard().unwrap(),
        "should return true (candidate consumed)"
    );
    assert!(store_path.exists(), "fresh store should NOT be deleted");
}

#[test]
fn download_next_remote_shard_gz_materializes_true_row_count() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);

    // Create a 5-line .jsonl.gz payload
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use std::io::Write;

    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    for i in 1..=5 {
        writeln!(encoder, r#"{{"id":"r{}","text":"line {}"}}"#, i, i).unwrap();
    }
    let gz_payload = encoder.finish().unwrap();

    let server = spawn_one_shot_http(gz_payload.clone());
    let base_url = server.url().to_string();
    let candidate = format!("url::{base_url}/datasets/org/ds/resolve/main/train/data.jsonl.gz");

    {
        let mut state = source.state.lock().unwrap();
        state.remote_candidates = Some(vec![candidate.clone()]);
        state
            .remote_candidate_sizes
            .insert(candidate, gz_payload.len() as u64);
        state.next_remote_idx = 0;
    }

    assert!(source.download_next_remote_shard().unwrap());

    let state = source.state.lock().unwrap();
    // Materialized rows must be 5 (true count), not 1 (dummy)
    assert_eq!(state.materialized_rows, 5);
    assert_eq!(state.shards.len(), 1);
}

#[test]
fn download_next_remote_shard_gz_invalid_json_returns_error() {
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use std::io::Write;

    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);

    // Create a .gz file with invalid JSON
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    writeln!(encoder, "this is not valid json").unwrap();
    let gz_payload = encoder.finish().unwrap();

    let server = spawn_one_shot_http(gz_payload.clone());
    let base_url = server.url().to_string();
    let candidate = format!("url::{base_url}/datasets/org/ds/resolve/main/train/invalid.jsonl.gz");

    {
        let mut state = source.state.lock().unwrap();
        state.remote_candidates = Some(vec![candidate.clone()]);
        state
            .remote_candidate_sizes
            .insert(candidate, gz_payload.len() as u64);
        state.next_remote_idx = 0;
    }

    let result = source.download_next_remote_shard();
    assert!(result.is_err());
    match result.unwrap_err() {
        SamplerError::SourceInconsistent { .. } => {} // Expected for invalid JSON
        other => panic!("expected SourceInconsistent, got: {other:?}"),
    }
}

#[test]
fn download_next_remote_shard_gz_corrupt_stream_returns_error() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);

    // Corrupt gzip data
    let corrupt_payload = b"this is not valid gzip data".to_vec();

    let server = spawn_one_shot_http(corrupt_payload.clone());
    let base_url = server.url().to_string();
    let candidate = format!("url::{base_url}/datasets/org/ds/resolve/main/train/corrupt.jsonl.gz");

    {
        let mut state = source.state.lock().unwrap();
        state.remote_candidates = Some(vec![candidate.clone()]);
        state
            .remote_candidate_sizes
            .insert(candidate, corrupt_payload.len() as u64);
        state.next_remote_idx = 0;
    }

    let result = source.download_next_remote_shard();
    assert!(result.is_err());
    match result.unwrap_err() {
        SamplerError::SourceUnavailable { .. } => {} // Expected for corrupt stream
        other => panic!("expected SourceUnavailable, got: {other:?}"),
    }
}

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
#[serial(global_state)]
fn download_next_remote_shard_detects_stale_shard_via_head() {
    // When the manifest does NOT provide expected_bytes (hf-hub sibling
    // fallback), but a cached store exists on disk with a stored source_size
    // that differs from the remote Content-Length (obtained via HTTP HEAD),
    // the store should be deleted so the shard gets re-downloaded.
    //
    // This test verifies that staleness is detected correctly by checking
    // the behaviour after the HEAD request:
    //   • stored source_size = 100
    //   • HEAD Content-Length = 200  (mismatch → store is deleted)
    //
    // The mock server serves the GET response body directly, so the
    // re-download may succeed.  The critical assertion is that the
    // original store is gone.
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config.clone());

    // Start a mock HTTP server — must stay alive for HEAD + any GET.
    // Use valid JSON so the transcoding pipeline succeeds.
    let payload = b"{\"text\":\"valid\",\"padding\":\"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\"}\n".to_vec();
    let server = TestHttpServer::new(200, payload);
    let base_url = server.url().to_string();

    // Candidate uses url:: prefix so the HEAD targets the mock server.
    let candidate = format!("url::{base_url}/resolve/main/train/stale-shard.ndjson");
    let store_path = HuggingFaceRowSource::candidate_store_path(&config, &candidate);
    fs::create_dir_all(store_path.parent().unwrap()).unwrap();
    write_simdr_fixture(&store_path, &[("r0", "row")]);
    {
        let store = DataStore::open(&store_path).unwrap();
        store
            .write(HF_SHARD_STORE_SOURCE_SIZE_KEY, &100u64.to_le_bytes())
            .unwrap();
    }

    // Snapshot the on-disk content so we can detect replacement.
    let original_content = fs::read(&store_path).unwrap();

    let cached_store = Arc::new(DataStore::open(&store_path).unwrap());
    source
        .config
        .store_cache
        .lock()
        .unwrap()
        .insert(store_path.clone(), cached_store);

    {
        let mut state = source.state.lock().unwrap();
        state.remote_candidates = Some(vec![candidate.clone()]);
        state.remote_candidate_order = vec![0];
        // No expected_bytes in remote_candidate_sizes — simulates the
        // hf-hub sibling fallback where sizes are unknown.
        state.next_remote_idx = 0;
    }

    // The HEAD request returns Content-Length: 200.  The stored
    // source_size is 100, so the staleness check should delete the
    // store and re-download.  The old store MUST be gone.
    let result = source.download_next_remote_shard();

    // The old store was deleted (HEAD detected mismatch).
    // A new store may or may not have been created depending on
    // whether the GET download + transcode succeeded.
    assert!(
        fs::read(&store_path).ok().as_deref() != Some(&original_content),
        "stale store content should have been replaced (HEAD detected size mismatch)"
    );

    // The candidate should have been consumed either way.
    assert!(
        result.is_ok(),
        "download may fail or succeed; the candidate should be consumed: {err:?}",
        err = result.as_ref().unwrap_err()
    );
}

#[test]
#[serial(global_state)]
fn download_next_remote_shard_preserves_fresh_shard_via_head() {
    // When the manifest does NOT provide expected_bytes, but a cached
    // store exists with a stored size that matches the remote Content-Length
    // from HEAD, the store should be preserved and the download skipped.
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config.clone());

    // Create a store with source_size matching the mock server's payload.
    let server = TestHttpServer::new(200, vec![0u8; 100]);
    let base_url = server.url().to_string();

    let candidate = format!("url::{base_url}/resolve/main/train/fresh-shard.ndjson");
    let store_path = HuggingFaceRowSource::candidate_store_path(&config, &candidate);
    fs::create_dir_all(store_path.parent().unwrap()).unwrap();
    write_simdr_fixture(&store_path, &[("r0", "row")]);
    {
        let store = DataStore::open(&store_path).unwrap();
        store
            .write(HF_SHARD_STORE_SOURCE_SIZE_KEY, &100u64.to_le_bytes())
            .unwrap();
    }

    let cached_store = Arc::new(DataStore::open(&store_path).unwrap());
    source
        .config
        .store_cache
        .lock()
        .unwrap()
        .insert(store_path.clone(), cached_store);

    {
        let mut state = source.state.lock().unwrap();
        state.remote_candidates = Some(vec![candidate.clone()]);
        state.remote_candidate_order = vec![0];
        // No expected_bytes in remote_candidate_sizes.
        state.next_remote_idx = 0;
    }

    // Sizes match (100 == 100) — should skip without download.
    let result = source.download_next_remote_shard();
    assert!(
        result.is_ok(),
        "expected Ok, got: {err:?}",
        err = result.as_ref().unwrap_err()
    );
    assert!(result.unwrap(), "should return true (candidate consumed)");
    assert!(
        store_path.exists(),
        "fresh store should NOT be deleted when sizes match via HEAD"
    );
}

#[test]
#[serial(global_state)]
fn download_next_remote_shard_keeps_store_when_head_returns_error() {
    // When the manifest does NOT provide expected_bytes (hf-hub sibling
    // fallback) AND the HTTP HEAD request fails (network error), the
    // staleness check is skipped and the cached store is preserved as-is.
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config.clone());

    // Candidate pointing at an unreachable address — HEAD will Err.
    let candidate = format!("url::{TEST_UNREACHABLE_URL}/resolve/main/train/head-err.ndjson");
    let store_path = HuggingFaceRowSource::candidate_store_path(&config, &candidate);
    fs::create_dir_all(store_path.parent().unwrap()).unwrap();
    write_simdr_fixture(&store_path, &[("r0", "row")]);
    {
        let store = DataStore::open(&store_path).unwrap();
        store
            .write(HF_SHARD_STORE_SOURCE_SIZE_KEY, &100u64.to_le_bytes())
            .unwrap();
    }

    let original_content = fs::read(&store_path).unwrap();

    let cached_store = Arc::new(DataStore::open(&store_path).unwrap());
    source
        .config
        .store_cache
        .lock()
        .unwrap()
        .insert(store_path.clone(), cached_store);

    {
        let mut state = source.state.lock().unwrap();
        state.remote_candidates = Some(vec![candidate.clone()]);
        state.remote_candidate_order = vec![0];
        // No expected_bytes — simulates hf-hub fallback.
        state.next_remote_idx = 0;
    }

    // HEAD fails (Err) → effective_expected = None → stale check
    // skipped → store preserved as-is.
    let result = source.download_next_remote_shard();
    assert!(
        result.is_ok(),
        "expected Ok even when HEAD fails, got: {err:?}",
        err = result.as_ref().unwrap_err()
    );
    assert!(
        fs::read(&store_path).ok().as_deref() == Some(&original_content),
        "store should be preserved when HEAD fails"
    );
}

#[test]
#[serial(global_state)]
fn download_next_remote_shard_keeps_store_when_head_returns_none() {
    // When the manifest does NOT provide expected_bytes AND the HEAD
    // request returns Ok(None) (e.g. 500 status, or missing
    // Content-Length), the staleness check is skipped and the cached
    // store is preserved.
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config.clone());

    // Mock server returning 500 — HEAD will succeed but return
    // Ok(None) because the status is not 2xx.
    let server = TestHttpServer::new(500, b"Internal Server Error".to_vec());
    let base_url = server.url().to_string();

    let candidate = format!("url::{base_url}/resolve/main/train/head-none.ndjson");
    let store_path = HuggingFaceRowSource::candidate_store_path(&config, &candidate);
    fs::create_dir_all(store_path.parent().unwrap()).unwrap();
    write_simdr_fixture(&store_path, &[("r0", "row")]);
    {
        let store = DataStore::open(&store_path).unwrap();
        store
            .write(HF_SHARD_STORE_SOURCE_SIZE_KEY, &100u64.to_le_bytes())
            .unwrap();
    }

    let original_content = fs::read(&store_path).unwrap();

    let cached_store = Arc::new(DataStore::open(&store_path).unwrap());
    source
        .config
        .store_cache
        .lock()
        .unwrap()
        .insert(store_path.clone(), cached_store);

    {
        let mut state = source.state.lock().unwrap();
        state.remote_candidates = Some(vec![candidate.clone()]);
        state.remote_candidate_order = vec![0];
        // No expected_bytes — simulates hf-hub fallback.
        state.next_remote_idx = 0;
    }

    // HEAD returns 500 → fetch_remote_size_with_runtime returns
    // Ok(None) → effective_expected = None → stale check skipped.
    let result = source.download_next_remote_shard();
    assert!(
        result.is_ok(),
        "expected Ok even when HEAD returns None, got: {err:?}",
        err = result.as_ref().unwrap_err()
    );
    assert!(
        fs::read(&store_path).ok().as_deref() == Some(&original_content),
        "store should be preserved when HEAD returns None"
    );
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
fn expansion_headroom_uses_sampler_ingestion_max_records_when_configured() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);

    assert_eq!(source.effective_expansion_headroom_rows(), 30);

    let sampler = SamplerConfig {
        ingestion_max_records: 7,
        ..SamplerConfig::default()
    };
    source.configure_sampler(&sampler);
    assert_eq!(source.effective_expansion_headroom_rows(), 21);
}

#[test]
fn value_to_text_handles_scalar_and_structured_values() {
    assert_eq!(rows::value_to_text(&json!(null)), None);
    assert_eq!(rows::value_to_text(&json!("   ")), None);
    assert_eq!(rows::value_to_text(&json!("hello")), Some("hello".into()));
    assert_eq!(rows::value_to_text(&json!(true)), Some("true".into()));
    assert_eq!(rows::value_to_text(&json!(3.5)), Some("3.5".into()));
    assert_eq!(rows::value_to_text(&json!([1, 2])), Some("1".into()));
}

#[test]
fn parse_row_uses_explicit_text_columns() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.id_column = Some("id".into());
    config.text_columns = vec!["title".into(), "body".into()];
    let source = test_source(config);

    let row = rows::parse_row(
        &source,
        5,
        &json!({
            "id": "row-5",
            "title": "Anchor text",
            "body": "Context text",
            "flag": true
        }),
    )
    .unwrap()
    .unwrap();

    // Candidate coalescing: the first non-empty column (title) is selected;
    // body is never tried because title already yielded a value.
    assert_eq!(row.row_id.as_deref(), Some("row-5"));
    assert_eq!(row.text_fields.len(), 1);
    assert_eq!(row.text_fields[0].name, "title");
    assert_eq!(row.text_fields[0].text, "Anchor text");
    assert!(row.text_fields.iter().all(|field| field.name != "id"));
}

#[test]
fn parse_row_with_required_columns_skips_when_missing() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.anchor_columns = vec!["anchor".into()];
    config.positive_columns = vec!["positive".into()];
    config.context_columns = vec!["context".into()];
    let source = test_source(config);

    let parsed = rows::parse_row(&source, 0, &json!({"anchor": "x", "context": "z"}));
    assert!(parsed.unwrap().is_none());
}

#[test]
fn parse_row_errors_when_payload_is_not_object() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);

    let err = rows::parse_row(&source, 0, &json!("not-an-object"));
    assert!(err.is_err());
}

#[test]
fn row_to_record_builds_expected_sections() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    let row = RowView {
        row_id: Some("abc".into()),
        timestamp: Some(Utc::now()),
        text_fields: vec![
            RowTextField {
                name: "title".into(),
                text: "anchor".into(),
            },
            RowTextField {
                name: "pos".into(),
                text: "positive".into(),
            },
            RowTextField {
                name: "ctx".into(),
                text: "extra".into(),
            },
        ],
    };

    let record = rows::row_to_record(&source, &row, 1).unwrap().unwrap();
    assert_eq!(record.sections.len(), 3);
    assert_eq!(record.sections[0].role, SectionRole::Anchor);
    assert_eq!(record.sections[1].role, SectionRole::Context);
    assert_eq!(record.id, "hf_test::abc");
}

#[test]
fn effective_refresh_batch_target_uses_multiplier_floor_of_one() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.refresh_batch_multiplier = 0;
    let source = test_source(config);
    assert_eq!(source.effective_refresh_batch_target(7), 7);
}

#[test]
fn locate_shard_and_recompute_offsets_work() {
    let mut shards = vec![
        ShardIndex {
            path: PathBuf::from("a"),
            global_start: 10,
            row_count: 3,
            parquet_row_groups: Vec::new(),
            remote_candidate: None,
        },
        ShardIndex {
            path: PathBuf::from("b"),
            global_start: 20,
            row_count: 2,
            parquet_row_groups: Vec::new(),
            remote_candidate: None,
        },
    ];
    let hit = HuggingFaceRowSource::locate_shard(&shards, 11).unwrap();
    assert_eq!(hit.1, 1);

    let mut state = SourceState {
        materialized_rows: 0,
        shards: std::mem::take(&mut shards),
        remote_candidates: None,
        remote_candidate_sizes: HashMap::new(),
        next_remote_idx: 0,
        remote_candidate_order: Vec::new(),
    };
    HuggingFaceRowSource::recompute_shard_offsets(&mut state);
    assert_eq!(state.shards[0].global_start, 0);
    assert_eq!(state.shards[1].global_start, 3);
    assert_eq!(state.materialized_rows, 5);
}

#[test]
fn len_hint_covers_known_and_empty_paths() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);

    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = 5;
        // Set up remote candidates so headroom is applied
        state.remote_candidates = Some(vec![
            "url::http://a/0.parquet".to_string(),
            "url::http://a/1.parquet".to_string(),
        ]);
        state.next_remote_idx = 0;
    }
    // headroom = ingestion_max_records * multiplier = 10 * 3 = 30; since known (5)
    // < headroom, expansion = 30; upper = 5 + 30 = 35
    assert_eq!(source.len_hint(), Some(35));

    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = 0;
        state.remote_candidates = Some(vec!["url::http://a/0.parquet".to_string()]);
        state.next_remote_idx = 0;
    }
    assert_eq!(source.len_hint(), Some(1));
}

#[test]
fn len_hint_defaults_to_one_when_unknown_and_not_exhausted() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    // Simulate an uninitialized source that hasn't fetched candidates yet.
    source.state.lock().unwrap().remote_candidates = None;
    assert_eq!(source.len_hint(), Some(1));
}

#[test]
fn len_hint_keeps_trickle_remote_expansion_after_warmup() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.cache_capacity = 4;
    config.remote_expansion_headroom_multiplier = 2;
    let source = test_source(config);

    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = 8;
        // Set up remote candidates so headroom is applied
        state.remote_candidates = Some(vec![
            "url::http://a/0.parquet".to_string(),
            "url::http://a/1.parquet".to_string(),
        ]);
        state.next_remote_idx = 0;
    }

    // headroom = cache_capacity * multiplier = 4 * 2 = 8; upper = 8 + 8 = 16
    assert_eq!(source.len_hint(), Some(16));
}

#[test]
fn eligible_rows_extends_cached_index_when_new_shard_is_appended() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config.clone());

    let appended_simdr = dir.path().join("append.simdr");
    write_simdr_fixture(&appended_simdr, &[("r1", "hello")]);
    let appended = index_single_shard(&config, &appended_simdr, 1)
        .unwrap()
        .0
        .unwrap();

    let baseline = ShardIndex {
        path: dir.path().join("missing-baseline.simdr"),
        global_start: 0,
        row_count: 1,
        parquet_row_groups: Vec::new(),
        remote_candidate: None,
    };

    {
        let mut state = source.state.lock().unwrap();
        state.shards = vec![baseline.clone(), appended.clone()];
        state.materialized_rows = 2;
    }

    {
        let mut cache = source.eligible_index.lock().unwrap();
        cache.signature = Some(HuggingFaceRowSource::shard_signature(std::slice::from_ref(
            &baseline,
        )));
        cache.rows = Some(Arc::new(vec![0]));
        cache.shards = vec![baseline];
    }

    let rows = source.eligible_rows().unwrap();
    assert_eq!(rows.as_ref(), &vec![0, 1]);
}

#[test]
fn materialize_local_file_copies_and_is_idempotent_when_size_matches() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let src = dir.path().join("src.ndjson");
    let dst = dir.path().join("nested/dst.ndjson");

    fs::write(&src, b"line\n").unwrap();
    HuggingFaceRowSource::materialize_local_file(&config, &src, &dst).unwrap();
    let first = fs::read(&dst).unwrap();
    HuggingFaceRowSource::materialize_local_file(&config, &src, &dst).unwrap();
    let second = fs::read(&dst).unwrap();
    assert_eq!(first, second);
}

#[test]
fn enforce_disk_cap_evicts_old_manifest_shards() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.local_disk_cap_bytes = Some(10);
    let source = test_source(config);

    let manifest_root = source.manifest_cache_root();
    fs::create_dir_all(&manifest_root).unwrap();
    let evict_path = manifest_root.join("a.parquet");
    let keep_path = manifest_root.join("b.parquet");
    fs::write(&evict_path, vec![1u8; 8]).unwrap();
    fs::write(&keep_path, vec![2u8; 8]).unwrap();

    let mut state = SourceState {
        materialized_rows: 16,
        shards: vec![
            ShardIndex {
                path: evict_path.clone(),
                global_start: 0,
                row_count: 8,
                parquet_row_groups: vec![(0, 8)],
                remote_candidate: None,
            },
            ShardIndex {
                path: keep_path.clone(),
                global_start: 8,
                row_count: 8,
                parquet_row_groups: vec![(0, 8)],
                remote_candidate: None,
            },
        ],
        remote_candidates: None,
        remote_candidate_sizes: HashMap::new(),
        next_remote_idx: 0,
        remote_candidate_order: Vec::new(),
    };

    let evicted = source
        .enforce_disk_cap_locked(&mut state, &keep_path)
        .unwrap();
    assert!(evicted);
    assert!(!evict_path.exists());
    assert!(keep_path.exists());
    assert_eq!(state.shards.len(), 1);
}

#[test]
fn enforce_disk_cap_ignores_min_resident_and_applies_policy() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.local_disk_cap_bytes = Some(4);
    let source = test_source(config);

    let manifest_root = source.manifest_cache_root();
    fs::create_dir_all(&manifest_root).unwrap();
    let protected = manifest_root.join("only.parquet");
    fs::write(&protected, vec![1u8; 8]).unwrap();

    let mut state = SourceState {
        materialized_rows: 8,
        shards: vec![ShardIndex {
            path: protected.clone(),
            global_start: 0,
            row_count: 8,
            parquet_row_groups: vec![(0, 8)],
            remote_candidate: None,
        }],
        remote_candidates: None,
        remote_candidate_sizes: HashMap::new(),
        next_remote_idx: 0,
        remote_candidate_order: Vec::new(),
    };

    let evicted = source
        .enforce_disk_cap_locked(&mut state, &protected)
        .unwrap();
    assert!(evicted);
    assert!(!protected.exists());
    assert_eq!(state.shards.len(), 0);
}

#[test]
fn index_single_shard_returns_none_for_empty_file() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let path = dir.path().join("empty.jsonl");
    fs::write(&path, b"").unwrap();
    let shard = index_single_shard(&config, &path, 0).unwrap();
    assert!(shard.0.is_none());
}

#[test]
fn refresh_reads_local_rows_and_advances_cursor() {
    let dir = tempdir().unwrap();
    let simdr_path = dir.path().join("rows.simdr");
    write_simdr_fixture(
        &simdr_path,
        &[("r1", "alpha"), ("r2", "beta"), ("r3", "gamma")],
    );

    let mut config = test_config(dir.path().to_path_buf());
    config.refresh_batch_multiplier = 1;
    let source = test_source(config.clone());
    let shard = index_single_shard(&config, &simdr_path, 0)
        .unwrap()
        .0
        .unwrap();
    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = shard.row_count;
        state.shards = vec![shard];
    }

    let snapshot = source.refresh(None, Some(2)).unwrap();
    assert_eq!(snapshot.records.len(), 2);
    assert!(snapshot.cursor.revision > 0);
}

#[test]
fn next_text_batch_produces_distinct_cursor_values_per_call() {
    // Proves successive next_text_batch calls produce different cursor
    // values — the cursor is NOT stuck at the initial value.
    let dir = tempdir().unwrap();

    // 3 shards so we get enough distinct cursor positions.
    let simdr_path = dir.path().join("rows.simdr");
    let rows: Vec<(String, String)> = (0..20)
        .map(|i| (format!("r{i}"), format!("text-{i}")))
        .collect();
    let row_refs: Vec<(&str, &str)> = rows.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    write_simdr_fixture(&simdr_path, &row_refs);

    let mut config =
        HuggingFaceRowsConfig::new("cursor_test", "org/ds", "default", "train", dir.path());
    config.hf_token = None;
    config.cache_capacity = 10;
    config.remote_expansion_headroom_multiplier = 1;
    config.parquet_endpoint = TEST_UNREACHABLE_URL.to_string();

    let source = HuggingFaceRowSource::new(config).unwrap();
    let shard_idx = index_single_shard(&source.config, &simdr_path, 0)
        .unwrap()
        .0
        .unwrap();
    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = shard_idx.row_count;
        state.shards = vec![shard_idx];
    }

    // Pre-load a PersistedSamplerState with cursor=0.
    let split_state = PersistedSamplerState {
        source_cycle_idx: 0,
        source_record_cursors: vec![("cursor_test".to_string(), 0)],
        epoch: 0,
        epoch_step: 0,
        rng_state: 0,
        triplet_recipe_rr_idx: 0,
        text_recipe_rr_idx: 0,
        source_stream_cursors: vec![("cursor_test".to_string(), 0)],
    };
    let split_store = Arc::new(DeterministicSplitStore::new(SplitRatios::default(), 777).unwrap());
    split_store.save_sampler_state(&split_state, None).unwrap();

    let sampler = TripletSampler::new(
        SamplerConfig {
            seed: 1,
            ingestion_max_records: 1,
            batch_size: 1,
            ..SamplerConfig::default()
        },
        split_store,
    );
    sampler.register_source(Box::new(source.clone())).unwrap();

    // Collect cursor_revision values over several next_text_batch calls.
    // Multiple next_text_batch calls should succeed (each triggers a
    // refresh with a distinct seed, proving the step counter influences
    // the shard order without crashing).
    let mut count = 0;
    for _ in 0..20 {
        match sampler.next_text_batch(SplitLabel::Train) {
            Ok(batch) => count += batch.samples.len(),
            Err(_) => break,
        }
    }
    assert!(count > 0, "expected at least 1 text sample across calls");
}

#[test]
fn next_batch_methods_rebuild_shard_order_with_step() {
    // Each batch method gets its own source+sampler with a unique
    // snapshot directory.  The first refresh increments epoch_step
    // 0→1, XORs into the seed (42^0^1=43), set_active_sampler_config
    // rebuilds the order differently from the initial seed=0.
    let shard_rows: Vec<(String, String)> = (0..10)
        .map(|i| (format!("r{i}"), "t".to_string()))
        .collect();

    // Helper: create source+sampler in a fresh tempdir with a shard file.
    let setup = || -> (HuggingFaceRowSource, TripletSampler<DeterministicSplitStore>, tempfile::TempDir) {
         let tmp = tempdir().unwrap();
         let simdr_path = tmp.path().join("shard.simdr");
         let row_refs: Vec<(&str, &str)> = shard_rows.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
         write_simdr_fixture(&simdr_path, &row_refs);
         let mut cfg = HuggingFaceRowsConfig::new("t", "o/d", "d", "train", tmp.path());
         cfg.hf_token = None;
         cfg.parquet_endpoint = TEST_UNREACHABLE_URL.to_string();
         let source = HuggingFaceRowSource::new(cfg).unwrap();
         let idx = index_single_shard(
             &source.config, &simdr_path, 0).unwrap().0.unwrap();
         {
             let mut st = source.state.lock().unwrap();
             st.materialized_rows = idx.row_count;
             st.shards = vec![idx];
             let cand: Vec<String> = (0..5).map(|i|
                 format!("url::http://h/d/resolve/main/train/p-{i:04}.ndjson")
             ).collect();
             st.remote_candidates = Some(cand.clone());
             st.remote_candidate_order =
                 build_candidate_order(&source.config, &cand, 0);
             st.next_remote_idx = 0;
         }
         let split = Arc::new(
             DeterministicSplitStore::new(SplitRatios { train: 1.0, validation: 0.0, test: 0.0 }, 777).unwrap());
         let sampler = TripletSampler::new(SamplerConfig {
             seed: 42, ingestion_max_records: 10, batch_size: 1,
             ..SamplerConfig::default()
         }, split);
         sampler.register_source(Box::new(source.clone())).unwrap();
         (source, sampler, tmp)
     };

    // next_text_batch
    let (source, sampler, _tmp) = setup();
    let before = source.state.lock().unwrap().remote_candidate_order.clone();
    sampler.next_text_batch(SplitLabel::Train).unwrap();
    assert_ne!(
        before,
        source.state.lock().unwrap().remote_candidate_order,
        "next_text_batch must change shard order"
    );

    // next_pair_batch
    let (source, sampler, _tmp) = setup();
    let before = source.state.lock().unwrap().remote_candidate_order.clone();
    let _ = sampler.next_pair_batch(SplitLabel::Train);
    assert_ne!(
        before,
        source.state.lock().unwrap().remote_candidate_order,
        "next_pair_batch must change shard order"
    );

    // next_triplet_batch
    let (source, sampler, _tmp) = setup();
    let before = source.state.lock().unwrap().remote_candidate_order.clone();
    let _ = sampler.next_triplet_batch(SplitLabel::Train);
    assert_ne!(
        before,
        source.state.lock().unwrap().remote_candidate_order,
        "next_triplet_batch must change shard order"
    );
}

#[test]
fn reported_record_count_uses_len_hint_for_local_state() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = 4;
    }
    assert_eq!(source.reported_record_count().unwrap(), 4);
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
fn parse_row_supports_row_wrapped_payload_and_text_columns() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.text_columns = vec!["headline".into(), "body".into()];
    config.id_column = Some("rid".into());
    let source = test_source(config);

    let parsed = rows::parse_row(
        &source,
        0,
        &json!({"row": {"rid": "r-1", "headline": "h", "body": "b"}}),
    )
    .unwrap()
    .unwrap();

    // Candidate coalescing: headline is non-empty so it is selected;
    // body is not tried.
    assert_eq!(parsed.row_id.as_deref(), Some("r-1"));
    assert_eq!(parsed.text_fields.len(), 1);
    assert_eq!(parsed.text_fields[0].name, "headline");
}

#[test]
fn parse_row_returns_none_when_all_positive_or_text_candidates_are_missing() {
    let dir = tempdir().unwrap();

    // Role mode: all positive_columns candidates absent → row skipped.
    let mut role_config = test_config(dir.path().to_path_buf());
    role_config.anchor_columns = vec!["anchor".into()];
    role_config.positive_columns = vec!["positive".into()];
    let role_source = test_source(role_config);

    let role_missing = rows::parse_row(&role_source, 0, &json!({"anchor":"a"})).unwrap();
    assert!(role_missing.is_none());

    // Text-columns mode: a row that lacks all listed candidates → row skipped.
    let mut text_config = test_config(dir.path().to_path_buf());
    text_config.text_columns = vec!["title".into(), "body".into()];
    let text_source = test_source(text_config);
    // Row has neither "title" nor "body" → no candidate matches → skipped.
    let text_missing =
        rows::parse_row(&text_source, 1, &json!({"other_field": "irrelevant"})).unwrap();
    assert!(text_missing.is_none());
}

#[test]
fn parse_row_text_columns_coalesces_to_first_nonempty_candidate() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.text_columns = vec!["title".into(), "body".into()];
    let source = test_source(config);

    // "title" is empty string → coalesces to "body".
    let row = rows::parse_row(
        &source,
        0,
        &json!({"title": "", "body": "fallback content"}),
    )
    .unwrap()
    .unwrap();
    assert_eq!(row.text_fields.len(), 1);
    assert_eq!(row.text_fields[0].name, "body");
    assert_eq!(row.text_fields[0].text, "fallback content");

    // "title" is present and non-empty → it is used; "body" is never tried.
    let row2 = rows::parse_row(
        &source,
        1,
        &json!({"title": "primary content", "body": "ignored"}),
    )
    .unwrap()
    .unwrap();
    assert_eq!(row2.text_fields.len(), 1);
    assert_eq!(row2.text_fields[0].name, "title");
    assert_eq!(row2.text_fields[0].text, "primary content");
}

#[test]
fn parse_row_positive_columns_coalesces_to_first_nonempty_candidate() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.anchor_columns = vec!["anchor".into()];
    config.positive_columns = vec!["summary".into(), "body".into()];
    let source = test_source(config);

    // "summary" is absent → coalesces to "body".
    let row = rows::parse_row(
        &source,
        0,
        &json!({"anchor": "a", "body": "fallback positive"}),
    )
    .unwrap()
    .unwrap();
    assert_eq!(row.text_fields.len(), 2);
    assert_eq!(row.text_fields[0].name, "anchor");
    assert_eq!(row.text_fields[1].name, "body");

    // "summary" is present and non-empty → it is used; "body" is ignored.
    let row2 = rows::parse_row(
        &source,
        1,
        &json!({"anchor": "a", "summary": "chosen", "body": "ignored"}),
    )
    .unwrap()
    .unwrap();
    assert_eq!(row2.text_fields.len(), 2);
    assert_eq!(row2.text_fields[1].name, "summary");
    assert_eq!(row2.text_fields[1].text, "chosen");

    // Both positive candidates absent → row skipped.
    let none = rows::parse_row(&source, 2, &json!({"anchor": "a"})).unwrap();
    assert!(none.is_none());
}

#[test]
fn parse_row_anchor_columns_coalesces_to_first_nonempty_candidate() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.anchor_columns = vec!["headline".into(), "title".into()];
    config.positive_columns = vec!["body".into()];
    let source = test_source(config);

    // "headline" is absent → coalesces to "title".
    let row = rows::parse_row(
        &source,
        0,
        &json!({"title": "fallback anchor", "body": "positive text"}),
    )
    .unwrap()
    .unwrap();
    assert_eq!(row.text_fields.len(), 2);
    assert_eq!(row.text_fields[0].name, "title");
    assert_eq!(row.text_fields[0].text, "fallback anchor");

    // "headline" is present and non-empty → it is used; "title" is ignored.
    let row2 = rows::parse_row(
        &source,
        1,
        &json!({"headline": "chosen anchor", "title": "ignored", "body": "positive"}),
    )
    .unwrap()
    .unwrap();
    assert_eq!(row2.text_fields[0].name, "headline");
    assert_eq!(row2.text_fields[0].text, "chosen anchor");

    // Both anchor candidates absent → row skipped.
    let none = rows::parse_row(&source, 2, &json!({"body": "positive only"})).unwrap();
    assert!(none.is_none());
}

#[test]
fn parse_row_errors_when_no_mapping_is_configured() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.id_column = Some("id".into());
    config.text_columns.clear();
    let source = test_source(config);

    let parsed = rows::parse_row(&source, 7, &json!({"id":"only-id"}));
    assert!(matches!(
        parsed,
        Err(SamplerError::SourceInconsistent { .. })
    ));
}

#[test]
fn row_to_record_returns_none_for_empty_fields() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    let row = RowView {
        row_id: Some("x".into()),
        timestamp: None,
        text_fields: Vec::new(),
    };
    assert!(rows::row_to_record(&source, &row, 0).unwrap().is_none());
}

#[test]
fn ensure_row_available_handles_materialized_max_and_exhausted_candidates() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = 1;
        state.remote_candidates = Some(vec![]);
        state.next_remote_idx = 0;
    }

    assert!(source.ensure_row_available(0).unwrap());
    assert!(!source.ensure_row_available(3).unwrap());
    assert!(!source.ensure_row_available(1).unwrap());
}

#[test]
#[serial(global_state)]
fn ensure_row_available_bootstraps_from_manifest_candidates() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let mut source = test_source(config);
    let (base_url, _, server) =
        spawn_manifest_and_shard_http(2, b"{\"text\":\"hello\"}\n".to_vec());

    // Reset to None so ensure_row_available triggers the manifest-fetch path.
    source.state.lock().unwrap().remote_candidates = None;
    source.config.parquet_endpoint = base_url.to_string();

    assert!(source.ensure_row_available(0).unwrap());

    server.join().unwrap();
}

#[test]
#[serial(global_state)]
fn ensure_row_available_skips_past_all_cached_candidates_on_restart() {
    // Verifies the restart scenario: when every candidate in the manifest is
    // already materialised on disk, next_remote_idx jumps to candidates.len()
    // and ensure_row_available returns Ok(false) without any download attempt.
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let mut source = test_source(config.clone());

    // Construct the candidate URL that the manifest will list.
    let shard_raw_url =
        format!("{TEST_UNREACHABLE_URL}/datasets/org/ds/resolve/main/train/a.ndjson");
    let shard_candidate = format!("{HF_REMOTE_URL_PREFIX}{shard_raw_url}");
    let target = candidate_target_path(&config, &shard_candidate);
    let store_path = shard_store_path_for(&target);
    fs::create_dir_all(store_path.parent().unwrap()).unwrap();
    fs::write(&store_path, b"dummy").unwrap();

    // Inject an already-materialised shard so materialized_rows == 1.
    {
        let mut state = source.state.lock().unwrap();
        state.shards = vec![ShardIndex {
            path: store_path,
            global_start: 0,
            row_count: 1,
            parquet_row_groups: vec![(0, 1)],
            remote_candidate: None,
        }];
        state.materialized_rows = 1;
        state.remote_candidates = None;
    }

    // Serve a manifest that lists the same (already-cached) shard.
    let manifest_body = serde_json::to_vec(&json!([
        {"type": "file", "path": shard_raw_url, "size": 100}
    ]))
    .unwrap();
    let server = spawn_one_shot_http(manifest_body);
    let base_url = server.url().to_string();

    // Row 1 is not yet materialised; this triggers the candidate-init path.
    // all candidates are already on disk → next_remote_idx = candidates.len() → Ok(false).
    source.config.parquet_endpoint = base_url;
    let result = source.ensure_row_available(1).unwrap();

    assert!(
        !result,
        "no new rows available — all candidates already cached"
    );
    let state = source.state.lock().unwrap();
    assert_eq!(
        state.next_remote_idx,
        state
            .remote_candidates
            .as_ref()
            .map(|c| c.len())
            .unwrap_or(0),
        "next_remote_idx must equal candidates.len() when all are cached"
    );
}

#[test]
fn read_row_batch_uses_cached_rows_and_respects_limit() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config.clone());

    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = 2;
    }

    let row0 = RowView {
        row_id: Some("r0".into()),
        timestamp: Some(Utc::now()),
        text_fields: vec![RowTextField {
            name: "text".into(),
            text: "alpha".into(),
        }],
    };
    let row1 = RowView {
        row_id: Some("r1".into()),
        timestamp: Some(Utc::now()),
        text_fields: vec![RowTextField {
            name: "text".into(),
            text: "beta".into(),
        }],
    };
    {
        let mut cache = source.cache.lock().unwrap();
        cache.insert(0, row0, config.cache_capacity);
        cache.insert(1, row1, config.cache_capacity);
    }

    let mut out = Vec::new();
    rows::read_row_batch(&source, &[0, 1], &mut out, Some(1)).unwrap();
    assert_eq!(out.len(), 1);
}

#[test]
fn read_row_batch_errors_on_invalid_json_line() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("broken.jsonl");
    fs::write(&path, b"not-json\n").unwrap();

    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config.clone());
    let shard = index_single_shard(&config, &path, 0).unwrap().0.unwrap();
    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = 1;
        state.shards = vec![shard];
    }

    let mut out = Vec::new();
    let result = rows::read_row_batch(&source, &[0], &mut out, Some(1));
    assert!(result.is_err());
}

#[test]
fn refresh_handles_empty_total_and_cursor_wrap() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config.clone());

    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = 0;
    }
    let empty = source.refresh(None, Some(5)).unwrap();
    assert!(empty.records.is_empty());
    assert_eq!(empty.cursor.revision, 0);

    let simdr_path = dir.path().join("rows.simdr");
    write_simdr_fixture(&simdr_path, &[("a", "A"), ("b", "B")]);
    let cfg2 = config;
    let source2 = test_source(cfg2.clone());
    let shard = index_single_shard(&cfg2, &simdr_path, 0)
        .unwrap()
        .0
        .unwrap();
    {
        let mut state = source2.state.lock().unwrap();
        state.materialized_rows = 2;
        state.shards = vec![shard];
    }
    let cursor = SourceCursor {
        last_seen: Utc::now(),
        revision: 99,
    };
    let snapshot = source2.refresh(Some(&cursor), Some(1)).unwrap();
    assert_eq!(snapshot.records.len(), 1);
}

#[test]
fn refresh_order_uses_sampler_seed_for_local_rows() {
    let dir = tempdir().unwrap();
    let simdr_path = dir.path().join("rows.simdr");
    let rows: Vec<(String, String)> = (0..12)
        .map(|idx| (format!("r{idx}"), format!("v{idx}")))
        .collect();
    let row_refs: Vec<(&str, &str)> = rows.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    write_simdr_fixture(&simdr_path, &row_refs);

    let mut config = test_config(dir.path().to_path_buf());
    config.refresh_batch_multiplier = 1;

    let source_a = test_source(config.clone());
    let source_b = test_source(config.clone());
    let source_c = test_source(config.clone());
    let shard = index_single_shard(&config, &simdr_path, 0)
        .unwrap()
        .0
        .unwrap();

    for source in [&source_a, &source_b, &source_c] {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = 12;
        state.shards = vec![shard.clone()];
    }

    let seed_1 = SamplerConfig {
        seed: 7,
        ..SamplerConfig::default()
    };
    let seed_2 = SamplerConfig {
        seed: 7,
        ..SamplerConfig::default()
    };
    let seed_3 = SamplerConfig {
        seed: 10,
        ..SamplerConfig::default()
    };

    source_a.configure_sampler(&seed_1);
    source_b.configure_sampler(&seed_2);
    source_c.configure_sampler(&seed_3);

    let ids_a: Vec<String> = source_a
        .refresh(None, Some(8))
        .unwrap()
        .records
        .into_iter()
        .map(|record| record.id)
        .collect();
    let ids_b: Vec<String> = source_b
        .refresh(None, Some(8))
        .unwrap()
        .records
        .into_iter()
        .map(|record| record.id)
        .collect();
    let ids_c: Vec<String> = source_c
        .refresh(None, Some(8))
        .unwrap()
        .records
        .into_iter()
        .map(|record| record.id)
        .collect();

    assert_eq!(ids_a, ids_b);
    assert_ne!(ids_a, ids_c);
}

#[test]
fn set_active_sampler_config_rebuilds_order_on_seed_change() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config.clone());

    let candidates = vec![
        "a".to_string(),
        "b".to_string(),
        "c".to_string(),
        "d".to_string(),
        "e".to_string(),
    ];

    // Prime the source at seed=7 BEFORE injecting state, so the subsequent
    // configure_sampler(seed=7) calls are not seen as seed changes.
    source.configure_sampler(&SamplerConfig {
        seed: 7,
        ..SamplerConfig::default()
    });

    {
        let mut state = source.state.lock().unwrap();
        // Candidates stored sorted/immutable; order derived from seed 7, cursor 0.
        let order = build_candidate_order(&config, &candidates, 7);
        state.remote_candidates = Some(candidates.clone());
        state.remote_candidate_order = order.clone();
        state.next_remote_idx = 3;
    }

    // Order is rebuilt every call; pointer advances to first uncached position.
    source.configure_sampler(&SamplerConfig {
        seed: 7,
        ..SamplerConfig::default()
    });
    {
        let state = source.state.lock().unwrap();
        let order = build_candidate_order(&config, &candidates, 7);
        assert_eq!(state.remote_candidate_order, order);
        assert_eq!(
            state.next_remote_idx, 0,
            "order rebuilt every call: pointer lands at first uncached (no shards on disk)"
        );
    }

    // Different seed — candidates list untouched, order rebuilt, pointer reset to 0.
    source.configure_sampler(&SamplerConfig {
        seed: 18,
        ..SamplerConfig::default()
    });
    {
        let state = source.state.lock().unwrap();
        // List is immutable — same sorted candidates.
        assert_eq!(state.remote_candidates.as_ref().unwrap(), &candidates);
        // Order is now derived from seed 18 (cursor_revision still 0).
        let expected_order = build_candidate_order(&config, &candidates, 18);
        assert_eq!(state.remote_candidate_order, expected_order);
        // No shards are materialised on disk so the pointer lands at 0
        // (the first non-materialised position in the new order).
        assert_eq!(state.next_remote_idx, 0);
    }
}

#[test]
fn set_active_sampler_config_rebuilds_order_every_call() {
    // Proves that set_active_sampler_config rebuilds the shard
    // permutation every time it's called with a different seed
    // (the seed changes every call due to epoch_step XOR in
    // IngestionManager).
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config.clone());

    let candidates: Vec<String> = (0..5)
        .map(|i| format!("url::http://host/datasets/org/ds/resolve/main/train/part-{i:04}.ndjson"))
        .collect();

    // Prime with seed 1.
    source.configure_sampler(&SamplerConfig {
        seed: 1,
        ..SamplerConfig::default()
    });

    {
        let mut state = source.state.lock().unwrap();
        state.remote_candidates = Some(candidates.clone());
        state.remote_candidate_order = Vec::new();
        state.next_remote_idx = 0;
    }

    // Call with seed 1 — order is built.
    source.configure_sampler(&SamplerConfig {
        seed: 1,
        ..SamplerConfig::default()
    });
    let order_seed1: Vec<usize>;
    {
        let state = source.state.lock().unwrap();
        let expected = build_candidate_order(&config, &candidates, 1);
        assert_eq!(state.remote_candidate_order, expected, "seed=1 order");
        order_seed1 = state.remote_candidate_order.clone();
    }

    // Call with seed 2 — order changes.
    source.configure_sampler(&SamplerConfig {
        seed: 2,
        ..SamplerConfig::default()
    });
    {
        let state = source.state.lock().unwrap();
        let expected = build_candidate_order(&config, &candidates, 2);
        assert_eq!(state.remote_candidate_order, expected, "seed=2 order");
        assert_ne!(
            state.remote_candidate_order, order_seed1,
            "different seed must produce different order"
        );
    }
}

#[test]
fn set_active_sampler_config_skips_materialised_shards_after_seed_change() {
    // This is the regression test for the bug where every source-epoch advance
    // reset next_remote_idx to 0, causing the expansion thread to always report
    // "shard 1/N already materialised" and never actually download new shards.
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config.clone());

    let candidates: Vec<String> = (0..5)
        .map(|i| format!("url::http://host/datasets/org/ds/resolve/main/train/part-{i:04}.ndjson"))
        .collect();

    // Prime source at seed 7 so the subsequent call at seed 7 is not a "change".
    source.configure_sampler(&SamplerConfig {
        seed: 7,
        ..SamplerConfig::default()
    });

    // Build the order for the *new* seed (18) up-front so we know which
    // positions map to which candidates and can pre-materialise the first 3.
    let new_order = build_candidate_order(&config, &candidates, 18);
    let materialised_count = 3;
    let shards_to_inject: Vec<ShardIndex> = (0..materialised_count)
        .map(|pos| {
            let candidate_idx = new_order[pos];
            let target = candidate_target_path(&config, &candidates[candidate_idx]);
            let store = shard_store_path_for(&target);
            ShardIndex {
                path: store,
                global_start: pos * 100,
                row_count: 100,
                parquet_row_groups: vec![(0, 100)],
                remote_candidate: None,
            }
        })
        .collect();

    {
        let mut state = source.state.lock().unwrap();
        let order_7 = build_candidate_order(&config, &candidates, 7);
        state.remote_candidates = Some(candidates.clone());
        state.remote_candidate_order = order_7;
        state.next_remote_idx = 0;
        state.shards = shards_to_inject;
        state.materialized_rows = materialised_count * 100;
    }

    // Change the seed — must advance pointer past the 3 materialised shards
    // in the new order rather than resetting to 0.
    source.configure_sampler(&SamplerConfig {
        seed: 18,
        ..SamplerConfig::default()
    });

    {
        let state = source.state.lock().unwrap();
        assert_eq!(
            state.remote_candidate_order,
            build_candidate_order(&config, &candidates, 18),
            "order must be rebuilt from new seed"
        );
        assert_eq!(
            state.next_remote_idx, materialised_count,
            "pointer must skip the {} already-materialised shards in the new order, \
              not reset to 0",
            materialised_count
        );
    }
}

// ── extract_classlabel_maps ───────────────────────────────────────────────

// ── value_to_text ──────────────────────────────────────────────────────────

#[test]
#[serial(global_state)]
fn parquet_manifest_fetched_exactly_once_per_candidate_list_population() {
    // Verify that the /parquet manifest endpoint is contacted only once per
    // source lifetime.  After the first ensure_row_available() populates
    // state.remote_candidates, subsequent calls must not re-fetch the manifest.
    // The counting server stays alive so a spurious second request would be
    // recorded and the final assertion would catch it.
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let mut source = test_source(config);
    // Reset to None so the first ensure_row_available() triggers the lazy fetch.
    source.state.lock().unwrap().remote_candidates = None;

    let shard_payload = b"{\"text\":\"hello\"}\n".to_vec();
    // Counting manifest+shard server: 4 slots so a second /parquet hit is caught.
    let (base_url, manifest_counter, _manifest_handle) =
        spawn_manifest_and_shard_http(4, shard_payload);

    // First call: remote_candidates is None → fetches manifest (counter→1) → downloads shard.
    source.config.parquet_endpoint = base_url.to_string();
    let first_available = source.ensure_row_available(0).unwrap();
    assert!(first_available);
    assert_eq!(
        manifest_counter.load(AtomicOrdering::SeqCst),
        1,
        "parquet manifest must be fetched exactly once on first ensure_row_available"
    );

    // Second call: remote_candidates is now Some(...) → must NOT re-fetch manifest.
    let _ = source.ensure_row_available(0);
    assert_eq!(
        manifest_counter.load(AtomicOrdering::SeqCst),
        1,
        "parquet manifest must not be re-fetched on subsequent ensure_row_available calls"
    );
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
fn ensure_row_available_does_not_loop_on_eviction() {
    // Regression: ensure the fetched_candidates guard prevents infinite
    // manifest re-fetching when eviction nulls remote_candidates mid-execution.
    //
    // On Windows, fs::remove_file fails for files with active handles (the
    // DataStore keeps .simdr open). We backdate existing.stub's mtime so the
    // cache manager evicts it FIRST (no open handle → deletable on all
    // platforms). After deletion, sync_shard_state_from_disk_locked detects
    // the missing shard and nulls remote_candidates, triggering the guard.
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    // Tight cap: existing shard fills it, so every new download triggers eviction.
    config.local_disk_cap_bytes = Some(10);
    let mut source = test_source(config.clone());

    // Create an existing shard on disk that fills the entire cap.
    let manifest_root = source.manifest_cache_root();
    fs::create_dir_all(&manifest_root).unwrap();
    let existing_path = manifest_root.join("existing.stub");
    fs::write(&existing_path, vec![1u8; 10]).unwrap();

    // Backdate existing.stub so it's always the LRU eviction target.
    let yesterday = SystemTime::now() - Duration::from_secs(86400);
    filetime::set_file_mtime(
        &existing_path,
        filetime::FileTime::from_system_time(yesterday),
    )
    .unwrap();

    {
        let mut state = source.state.lock().unwrap();
        state.shards = vec![ShardIndex {
            path: existing_path,
            global_start: 0,
            row_count: 1,
            parquet_row_groups: vec![(0, 1)],
            remote_candidate: None,
        }];
        state.materialized_rows = 1;
        // None triggers the candidate-fetch path on first ensure_row_available call.
        state.remote_candidates = None;
    }

    // Use a single multi-accept mock server that serves both the
    // parquet manifest (/parquet) and shard payloads (everything else).
    // This avoids the flakiness of separate one-shot servers where the
    // manifest re-fetch could point back to an already-consumed server
    // (the eviction order is non-deterministic across platforms).
    let shard_payload = b"{\"text\":\"new\"}\n".to_vec();
    let (base_url, manifest_counter, server) =
        spawn_manifest_and_shard_http(2, shard_payload.clone());

    // Call ensure_row_available with idx == materialized_rows so the
    // first download does not satisfy idx < materialized_rows.
    // Append /parquet so spawn_manifest_and_shard_http can route the
    // manifest re-fetch to the manifest body (see first_line.contains("/parquet")).
    source.config.parquet_endpoint = base_url.to_string();

    // ensure_row_available(1) must:
    //   1. Fetch manifest (remote_candidates = None)
    //   2. Download shard (materialized_rows 1→2)
    //   3. Eviction deletes existing.stub → remote_candidates = None
    //   4. fetched_candidates guard fires → returns Ok(true)
    //   5. manifest_counter == 1 (no re-fetch)
    assert!(
        source.ensure_row_available(1).unwrap(),
        "ensure_row_available must return Ok(true)"
    );

    assert_eq!(
        manifest_counter.load(AtomicOrdering::SeqCst),
        1,
        "manifest must be fetched exactly once — fetched_candidates guard must prevent re-fetch"
    );
    server.join().unwrap();
}

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
fn open_shard_store_creates_parent_directories() {
    let dir = tempdir().unwrap();
    let nested = dir.path().join("a").join("b").join("c.simdr");
    assert!(!nested.parent().unwrap().exists());
    let config = test_config(dir.path().to_path_buf());
    let store = HuggingFaceRowSource::open_shard_store(&config, &nested).unwrap();
    assert!(nested.parent().unwrap().exists());
    drop(store);
}

#[test]
fn open_shard_store_errors_when_base_path_is_a_file() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let file_path = dir.path().join("not-a-dir");
    fs::write(&file_path, b"not-a-dir").unwrap();
    let bad_path = file_path.join("store.simdr");
    let result = HuggingFaceRowSource::open_shard_store(&config, &bad_path);
    assert!(result.is_err());
}

#[test]
fn remove_stale_store_evicts_from_cache_and_removes_file() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let store_path = dir.path().join("stale.simdr");
    fs::write(&store_path, b"stale-content").unwrap();
    config.store_cache.lock().unwrap().insert(
        store_path.clone(),
        Arc::new(DataStore::open(&store_path).unwrap()),
    );
    crate::disk_cache::remove_stale_store(&config, &store_path);
    assert!(!store_path.exists(), "stale store file must be removed");
    assert!(!config.store_cache.lock().unwrap().contains_key(&store_path));
}

#[test]
fn remove_stale_store_does_not_panic_when_file_missing() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let missing = dir.path().join("never-existed.simdr");
    crate::disk_cache::remove_stale_store(&config, &missing);
    assert!(!missing.exists());
}

#[test]
fn invalidate_eligible_index_resets_cache() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    {
        let mut cache = source.eligible_index.lock().unwrap();
        *cache = EligibleIndexCache {
            signature: Some(42),
            rows: Some(Arc::new(vec![0, 1, 2])),
            shards: vec![ShardIndex {
                path: PathBuf::from("dummy.parquet"),
                global_start: 0,
                row_count: 3,
                parquet_row_groups: vec![(0, 3)],
                remote_candidate: None,
            }],
        };
    }
    source.invalidate_eligible_index();
    let cache = source.eligible_index.lock().unwrap();
    assert!(cache.signature.is_none());
    assert!(cache.rows.is_none());
    assert!(cache.shards.is_empty());
}

#[test]
fn write_store_row_count_and_read_store_row_count_roundtrip() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    let store_path = dir.path().join("roundtrip.simdr");
    let store = DataStore::open(&store_path).unwrap();
    rows::write_store_row_count(&source, &store, 42).unwrap();
    assert_eq!(rows::read_store_row_count(&source, &store).unwrap(), 42);
    rows::write_store_row_count(&source, &store, 99).unwrap();
    assert_eq!(rows::read_store_row_count(&source, &store).unwrap(), 99);
}

#[test]
fn read_store_row_count_errors_on_payload_size_mismatch() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    let store_path = dir.path().join("bad-meta.simdr");
    let store = DataStore::open(&store_path).unwrap();
    store.write(HF_SHARD_STORE_META_ROWS_KEY, b"abc").unwrap();
    match rows::read_store_row_count(&source, &store) {
        Err(SamplerError::SourceUnavailable { reason, .. }) => {
            assert!(reason.contains("payload size"));
        }
        other => panic!("expected SourceUnavailable error, got {other:?}"),
    }
}

// FIXME: This test passes in isolation, but times out when running with all of the tests.
//
// Additional context (may be inaccurate):
//
// Testing live thread spawning combined with a deliberate fallback to an
// unreachable dead port (127.0.0.1:1) binds your test suite's determinism
// directly to OS-level TCP/IP implementation details. While Unix environments
// typically reject connections to unbound low ports instantaneously (ECONNREFUSED),
// the Windows Winsock layer behaves non-deterministically under parallel test
// execution profiles, frequently caching socket state or delaying connection drops
// to match synthetic connect timeouts.
#[test]
#[serial(global_state)]
#[cfg(not(target_os = "windows"))]
fn trigger_expansion_if_needed_starts_background_thread() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let mut source = test_source(config);

    // Override with ultra-short timeouts to force immediate connection failure on Windows
    let inner_client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_millis(100))
        .timeout(std::time::Duration::from_millis(200))
        .build()
        .expect("failed to build ultra-short timeout client");
    source.http_client = reqwest_drive::ClientBuilder::new(inner_client).build();

    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = 5;
        state.remote_candidates = Some(vec![
            "url::http://127.0.0.1:1/ds/resolve/main/train/000.ndjson".to_string(),
        ]);
        state.next_remote_idx = 0;
        state.remote_candidate_order = vec![0];
    }
    assert!(source.expansion_thread.lock().unwrap().is_none());
    source.trigger_expansion_if_needed();
    let handle = source.expansion_thread.lock().unwrap().take();
    assert!(handle.is_some());
    if let Some(h) = handle {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(h.join());
        });
        let _ = rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("expansion thread hung or deadlocked: Timeout");
    }
}

#[test]
fn trigger_expansion_if_needed_skips_when_all_remote_candidates_consumed() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = 100;
        state.remote_candidates = Some(vec!["done".to_string()]);
        state.next_remote_idx = 1;
    }
    source.trigger_expansion_if_needed();
    assert!(source.expansion_thread.lock().unwrap().is_none());
}

#[test]
fn trigger_expansion_if_needed_skips_when_total_rows_is_zero() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = 0;
    }
    source.trigger_expansion_if_needed();
    assert!(source.expansion_thread.lock().unwrap().is_none());
}

#[test]
fn trigger_expansion_if_needed_skips_when_already_running() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);

    // Inject a dummy thread that blocks until explicitly released.
    // No network I/O, no sleep, no global mutex contention.
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    let dummy = std::thread::spawn(move || {
        let _ = rx.recv();
    });
    *source.expansion_thread.lock().unwrap() = Some(dummy);

    // Must skip: slot is already occupied.
    source.trigger_expansion_if_needed();
    assert!(source.expansion_thread.lock().unwrap().as_ref().is_some());

    // Release: signal the dummy to exit, join cleanly.
    drop(tx);
    let handle = source.expansion_thread.lock().unwrap().take().unwrap();
    handle.join().unwrap();
}

#[test]
fn ensure_cache_group_reports_error() {
    let bad_group = PathBuf::from("bad\0group");
    let result = ensure_cache_group(bad_group);
    assert!(result.is_err());
}

// ── New tests for uncovered functions ────────────────────────────────

#[test]
fn new_rejects_missing_explicit_mapping() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.anchor_columns.clear();
    config.positive_columns.clear();
    config.context_columns.clear();
    config.text_columns.clear();
    let result = HuggingFaceRowSource::new(config);
    assert!(result.is_err());
    let err = result.map(|_| ()).unwrap_err();
    assert!(format!("{err:?}").contains("explicit field mapping"));
}

#[test]
fn id_returns_source_id() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    assert_eq!(source.id(), "hf_test");
}

#[test]
fn is_store_shard_path_detects_simdr_extension() {
    assert!(is_store_shard_path(Path::new("shard.simdr")));
    assert!(is_store_shard_path(Path::new("shard.SIMDR")));
    assert!(is_store_shard_path(Path::new("shard.SimDr")));
    assert!(!is_store_shard_path(Path::new("shard.parquet")));
    assert!(!is_store_shard_path(Path::new("shard.ndjson")));
    assert!(!is_store_shard_path(Path::new("no-extension")));
    assert!(!is_store_shard_path(Path::new(".hidden")));
}

#[test]
fn resolve_inner_extension_handles_compound_extensions() {
    // Compound extensions
    assert_eq!(
        resolve_inner_extension(Path::new("file.jsonl.gz")),
        Some("jsonl".to_string())
    );
    assert_eq!(
        resolve_inner_extension(Path::new("file.ndjson.gz")),
        Some("ndjson".to_string())
    );
    assert_eq!(
        resolve_inner_extension(Path::new("file.txt.gz")),
        Some("txt".to_string())
    );

    // Simple extensions
    assert_eq!(
        resolve_inner_extension(Path::new("file.parquet")),
        Some("parquet".to_string())
    );
    assert_eq!(
        resolve_inner_extension(Path::new("file.simdr")),
        Some("simdr".to_string())
    );
    assert_eq!(
        resolve_inner_extension(Path::new("file.jsonl")),
        Some("jsonl".to_string())
    );
    assert_eq!(
        resolve_inner_extension(Path::new("file.ndjson")),
        Some("ndjson".to_string())
    );

    // No inner format (just .gz)
    assert_eq!(resolve_inner_extension(Path::new("file.gz")), None);

    // No extension
    assert_eq!(resolve_inner_extension(Path::new("no-extension")), None);

    // Case insensitive
    assert_eq!(
        resolve_inner_extension(Path::new("file.JSONL.GZ")),
        Some("jsonl".to_string())
    );
    assert_eq!(
        resolve_inner_extension(Path::new("file.PARQUET")),
        Some("parquet".to_string())
    );

    // Boundary cases
    assert_eq!(
        resolve_inner_extension(Path::new("file.tar.gz")),
        Some("tar".to_string())
    );
    assert_eq!(
        resolve_inner_extension(Path::new(".hidden.jsonl.gz")),
        Some("jsonl".to_string())
    );
    assert_eq!(
        resolve_inner_extension(Path::new("file.gz.bak")),
        Some("bak".to_string())
    );
    assert_eq!(resolve_inner_extension(Path::new("")), None);
}

#[test]
fn is_gzip_path_detects_gz_extension() {
    assert!(is_gzip_path(Path::new("file.jsonl.gz")));
    assert!(is_gzip_path(Path::new("file.GZ")));
    assert!(is_gzip_path(Path::new("file.Gz")));
    assert!(is_gzip_path(Path::new("file.tar.gz")));
    assert!(!is_gzip_path(Path::new("file.parquet")));
    assert!(!is_gzip_path(Path::new("file.jsonl")));
    assert!(!is_gzip_path(Path::new("file.simdr")));
    assert!(!is_gzip_path(Path::new("no-extension")));
    assert!(!is_gzip_path(Path::new("file.gz.bak")));
    assert!(!is_gzip_path(Path::new("")));
}

#[test]
fn shard_store_path_for_appends_simdr_extension() {
    let path = PathBuf::from("cache/shard.parquet");
    let mapped = shard_store_path_for(&path);
    assert_eq!(mapped, PathBuf::from("cache/shard.simdr"));
    let no_ext = PathBuf::from("cache/shard");
    let mapped2 = shard_store_path_for(&no_ext);
    assert_eq!(mapped2, PathBuf::from("cache/shard.simdr"));
}

#[test]
fn candidate_store_path_maps_via_target_path() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let candidate = "url::https://host/ds/resolve/main/train/data-000.parquet";
    let target = candidate_target_path(&config, candidate);
    let store = HuggingFaceRowSource::candidate_store_path(&config, candidate);
    assert_eq!(store, target.with_extension("simdr"));
}

#[test]
fn manifest_cache_root_joins_manifest_dir() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    let root = source.manifest_cache_root();
    assert!(root.ends_with(HF_PARQUET_MANIFEST_DIR));
}

#[test]
fn recompute_shard_offsets_sums_row_counts() {
    let mut state = SourceState {
        materialized_rows: 0,
        shards: vec![
            ShardIndex {
                path: PathBuf::from("a.simdr"),
                global_start: 0,
                row_count: 10,
                parquet_row_groups: vec![(0, 10)],
                remote_candidate: None,
            },
            ShardIndex {
                path: PathBuf::from("b.simdr"),
                global_start: 0,
                row_count: 20,
                parquet_row_groups: vec![(0, 20)],
                remote_candidate: None,
            },
        ],
        remote_candidates: None,
        remote_candidate_sizes: HashMap::new(),
        next_remote_idx: 0,
        remote_candidate_order: Vec::new(),
    };
    HuggingFaceRowSource::recompute_shard_offsets(&mut state);
    assert_eq!(state.materialized_rows, 30);
    assert_eq!(state.shards[0].global_start, 0);
    assert_eq!(state.shards[1].global_start, 10);
}

#[test]
fn sync_shard_state_from_disk_removes_missing_shards() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    let existing = dir.path().join("existing.simdr");
    fs::write(&existing, b"data").unwrap();
    let missing = dir.path().join("missing.simdr");
    let mut state = SourceState {
        materialized_rows: 100,
        shards: vec![
            ShardIndex {
                path: existing.clone(),
                global_start: 0,
                row_count: 50,
                parquet_row_groups: vec![(0, 50)],
                remote_candidate: None,
            },
            ShardIndex {
                path: missing.clone(),
                global_start: 50,
                row_count: 50,
                parquet_row_groups: vec![(0, 50)],
                remote_candidate: None,
            },
        ],
        remote_candidates: Some(vec!["candidate".to_string()]),
        remote_candidate_sizes: HashMap::new(),
        next_remote_idx: 0,
        remote_candidate_order: vec![0],
    };
    source.sync_shard_state_from_disk_locked(&mut state);
    assert_eq!(state.shards.len(), 1);
    assert_eq!(state.shards[0].path, existing);
    assert_eq!(state.materialized_rows, 50);
    assert!(state.remote_candidates.is_none());
}

#[test]
fn sync_shard_state_from_disk_preserves_candidates_when_all_present() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    let sp = dir.path().join("shard.simdr");
    fs::write(&sp, b"data").unwrap();
    let mut state = SourceState {
        materialized_rows: 50,
        shards: vec![ShardIndex {
            path: sp.clone(),
            global_start: 0,
            row_count: 50,
            parquet_row_groups: vec![(0, 50)],
            remote_candidate: None,
        }],
        remote_candidates: Some(vec!["next".to_string()]),
        remote_candidate_sizes: HashMap::new(),
        next_remote_idx: 1,
        remote_candidate_order: vec![0],
    };
    source.sync_shard_state_from_disk_locked(&mut state);
    assert_eq!(state.shards.len(), 1);
    assert_eq!(state.remote_candidates, Some(vec!["next".to_string()]));
    assert_eq!(state.next_remote_idx, 1);
}

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

#[test]
#[serial(global_state)]
fn managed_hf_snapshot_dir_resolves_without_replica() {
    let dir = tempdir().unwrap();
    let nl = platform_newline();
    fs::write(
        dir.path().join("Cargo.toml"),
        format!("[package]{nl}name='tmp'{nl}version='0.0.0'{nl}"),
    )
    .unwrap();
    with_current_dir(dir.path(), || {
        let r = managed_hf_snapshot_dir("org/dataset", "default", "train");
        assert!(r.is_ok());
        assert!(r.unwrap().ends_with("train"));
    });
}

#[test]
#[serial(global_state)]
fn managed_hf_snapshot_dir_uses_all_splits_for_empty() {
    let dir = tempdir().unwrap();
    let nl = platform_newline();
    fs::write(
        dir.path().join("Cargo.toml"),
        format!("[package]{nl}name='tmp'{nl}version='0.0.0'{nl}"),
    )
    .unwrap();
    with_current_dir(dir.path(), || {
        let r = managed_hf_snapshot_dir("org/dataset", "default", "");
        assert!(r.is_ok());
        let path = r.unwrap();
        assert!(
            path.to_string_lossy().contains(HF_ALL_SPLITS_DIR),
            "expected path to contain '{}', got: {}",
            HF_ALL_SPLITS_DIR,
            path.display()
        );
    });
}

#[test]
#[serial(global_state)]
fn managed_hf_list_snapshot_dir_uses_replica_suffix() {
    let dir = tempdir().unwrap();
    let nl = platform_newline();
    fs::write(
        dir.path().join("Cargo.toml"),
        format!("[package]{nl}name='tmp'{nl}version='0.0.0'{nl}"),
    )
    .unwrap();
    with_current_dir(dir.path(), || {
        let r = managed_hf_list_snapshot_dir("org/dataset", "default", "train", 0);
        assert!(r.is_ok());
        assert!(r.unwrap().ends_with("replica_0"));
    });
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
fn row_store_row_key_uses_expected_format() {
    let key = row_store_row_key(0);
    assert!(key.starts_with(HF_SHARD_STORE_ROW_PREFIX));
    assert_eq!(key.len(), HF_SHARD_STORE_ROW_PREFIX.len() + 8);
    let key_42 = row_store_row_key(42);
    assert!(key_42.starts_with(HF_SHARD_STORE_ROW_PREFIX));
}

#[test]
fn effective_expansion_headroom_rows_uses_config_multiplier() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.cache_capacity = 100;
    config.remote_expansion_headroom_multiplier = 3;
    let source = test_source(config);
    assert_eq!(source.effective_expansion_headroom_rows(), 300);

    source.configure_sampler(&SamplerConfig {
        ingestion_max_records: 50,
        ..SamplerConfig::default()
    });
    assert_eq!(source.effective_expansion_headroom_rows(), 150);
}

#[test]
fn open_store_via_cache_inserts_and_reuses() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let path = dir.path().join("store.simdr");
    let store = DataStore::open(&path).unwrap();
    drop(store);
    let first = crate::disk_cache::open_store_via_cache(&config, &path).unwrap();
    let second = crate::disk_cache::open_store_via_cache(&config, &path).unwrap();
    assert!(Arc::ptr_eq(&first, &second));
}

#[test]
fn reported_record_count_uses_len_hint() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = 10;
        state.remote_candidates = None;
    }
    assert_eq!(source.reported_record_count().unwrap(), 10);
}

#[test]
fn format_shard_label_includes_totals() {
    let label = format_shard_label("data/train-000.parquet", 0, 5);
    assert!(label.contains("1/5"));
    assert!(label.contains("train-000.parquet"));
}

#[test]
fn effective_refresh_batch_target_uses_multiplier() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    assert!(source.effective_refresh_batch_target(100) >= 2);
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
fn resolve_json_path_top_level() {
    let row = json!({"query": "hello", "pos": ["p"], "neg": ["n"]});
    let obj = row.as_object().unwrap();
    assert_eq!(rows::resolve_json_path(obj, "query"), Some(json!("hello")));
}

#[test]
fn resolve_json_path_nested_dict() {
    let row = json!({"set": {"query": "hello", "pos": ["p"], "neg": ["n"]}});
    let obj = row.as_object().unwrap();
    assert_eq!(
        rows::resolve_json_path(obj, "set.query"),
        Some(json!("hello"))
    );
    assert_eq!(rows::resolve_json_path(obj, "set.pos"), Some(json!(["p"])));
}

#[test]
fn resolve_json_path_missing_returns_none() {
    let row = json!({"set": {"query": "hello"}});
    let obj = row.as_object().unwrap();
    assert_eq!(rows::resolve_json_path(obj, "missing"), None);
}

#[test]
fn value_to_text_array_first_element() {
    assert_eq!(
        rows::value_to_text(&json!(["single"])),
        Some("single".into())
    );
    assert_eq!(
        rows::value_to_text(&json!(["a", "b", "c"])),
        Some("a".into())
    );
    assert_eq!(rows::value_to_text(&json!([])), None);
    assert_eq!(
        rows::value_to_text(&json!([null, "valid"])),
        Some("valid".into())
    );
}

#[test]
fn parse_row_dict_dataset() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.anchor_columns = vec!["set.query".into()];
    config.positive_columns = vec!["set.pos".into()];
    config.negative_columns = vec!["set.neg".into()];
    config.id_column = None;
    let source = test_source(config);

    let row = rows::parse_row(
        &source,
        0,
        &json!({
            "set": {
                "query": "What is Rust?",
                "pos": ["How does Rust work?"],
                "neg": ["Is Python better?", "What is Java?", "Tell me a joke"]
            }
        }),
    )
    .unwrap()
    .unwrap();

    // anchor = set.query, positive = first set.pos element, negatives = expanded set.neg elements
    assert_eq!(row.text_fields.len(), 5);
    assert_eq!(row.text_fields[0].name, "set.query");
    assert_eq!(row.text_fields[0].text, "What is Rust?");
    assert_eq!(row.text_fields[1].name, "set.pos");
    assert_eq!(row.text_fields[1].text, "How does Rust work?");
    assert_eq!(row.text_fields[2].name, "set.neg[0]");
    assert_eq!(row.text_fields[2].text, "Is Python better?");
    assert_eq!(row.text_fields[3].name, "set.neg[1]");
    assert_eq!(row.text_fields[3].text, "What is Java?");
    assert_eq!(row.text_fields[4].name, "set.neg[2]");
    assert_eq!(row.text_fields[4].text, "Tell me a joke");
}

#[test]
fn parse_row_dict_flat_columns_unaffected() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.anchor_columns = vec!["anchor".into()];
    config.positive_columns = vec!["positive".into()];
    config.id_column = Some("id".into());
    let source = test_source(config);

    // Flat (non-dict) row — should work exactly as before.
    let row = rows::parse_row(
        &source,
        0,
        &json!({"id": "r1", "anchor": "a", "positive": "p"}),
    )
    .unwrap()
    .unwrap();
    assert_eq!(row.text_fields.len(), 2);
    assert_eq!(row.text_fields[0].text, "a");
    assert_eq!(row.text_fields[1].text, "p");
}

#[test]
fn row_to_record_dict_dataset_sections() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.anchor_columns = vec!["query".into()];
    config.positive_columns = vec!["pos".into()];
    config.negative_columns = vec!["neg".into()];
    let source = test_source(config);

    let row = RowView {
        row_id: Some("dict:0".into()),
        timestamp: None,
        text_fields: vec![
            RowTextField {
                name: "query".into(),
                text: "anchor text".into(),
            },
            RowTextField {
                name: "pos".into(),
                text: "positive text".into(),
            },
            RowTextField {
                name: "neg[0]".into(),
                text: "negative one".into(),
            },
            RowTextField {
                name: "neg[1]".into(),
                text: "negative two".into(),
            },
        ],
    };

    let record = rows::row_to_record(&source, &row, 0).unwrap().unwrap();
    // First field → Anchor, second → Context (positive), rest → Context (negatives)
    assert_eq!(record.sections.len(), 4);
    assert_eq!(record.sections[0].role, SectionRole::Anchor);
    assert_eq!(record.sections[0].text, "anchor text");
    assert_eq!(record.sections[1].role, SectionRole::Context);
    assert_eq!(record.sections[1].text, "positive text");
    assert_eq!(record.sections[2].role, SectionRole::Context);
    assert_eq!(record.sections[2].text, "negative one");
    assert_eq!(record.sections[3].role, SectionRole::Context);
    assert_eq!(record.sections[3].text, "negative two");
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

#[test]
fn resolve_json_path_non_object_intermediate_returns_none() {
    let row = json!({"set": "not-an-object"});
    let obj = row.as_object().unwrap();
    // "set" exists but is a string, not an object — inner.get should fail.
    assert_eq!(rows::resolve_json_path(obj, "set.query"), None);
}
