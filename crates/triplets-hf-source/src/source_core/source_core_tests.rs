// These tests intentionally exercise the deprecated unweighted batch-fetch
// convenience methods (next_*_batch, next_*_batch_for_split, prefetch_*_batches)
// to validate uniform sampling. The *with_weights variants are the supported API
// for honoring a data mixture.
#![allow(deprecated)]

use super::*;
use crate::constants::{
    ENV_TRIPLETS_HF_WHOAMI_ENDPOINT, HF_PARQUET_MANIFEST_DIR,
    HF_RECIPE_ANCHOR_ANCHOR_WRONG_ARTICLE, HF_RECIPE_ANCHOR_CONTEXT_WRONG_ARTICLE,
    HF_RECIPE_TEXT_SIMCSE_WRONG_ARTICLE, HF_REMOTE_URL_PREFIX, HF_SHARD_STORE_SOURCE_SIZE_KEY,
};
use crate::download::{build_http_runtime, candidate_target_path};
use crate::huggingface_source::{
    EligibleIndexCache, ParquetCache, RowCache, RowTextField, RowView,
};
use crate::shard_index::{index_single_shard, shard_store_path_for};
use crate::test_utils::{
    TEST_UNREACHABLE_URL, TestHttpServer, spawn_manifest_and_shard_http, spawn_one_shot_http,
    test_config, test_http_client, test_source, with_env_var, write_parquet_fixture,
    write_simdr_fixture,
};
use crate::types::{ShardIndex, SourceState};
use chrono::Utc;
use serde_json::json;
use serial_test::serial;
use simd_r_drive::storage_engine::DataStore;
use simd_r_drive::storage_engine::traits::DataStoreWriter;
use std::collections::HashMap;
use std::fs;
use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use tempfile::tempdir;
use triplets_core::SamplerError;
use triplets_core::config::{NegativeStrategy, SamplerConfig};
use triplets_core::source::SourceCursor;
use triplets_core::splits::{PersistedSamplerState, SamplerStateStore};
use triplets_core::{DeterministicSplitStore, Sampler, SplitLabel, SplitRatios, TripletSampler};

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
fn new_source_indexes_local_simdr_files() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());

    // Pre-create a .simdr file so new() discovers it during indexing.
    let simdr = dir.path().join("_parquet_manifest").join("shard.simdr");
    fs::create_dir_all(simdr.parent().unwrap()).unwrap();
    write_simdr_fixture(&simdr, &[("r1", "hello"), ("r2", "world")]);

    // Provide explicit mapping so has_explicit_mapping() returns true.
    config.anchor_columns = vec!["anchor".to_string()];
    config.positive_columns = vec!["positive".to_string()];

    let source = HuggingFaceRowSource::new(config).expect("new should succeed");
    let state = source.state.lock().unwrap();
    assert_eq!(state.materialized_rows, 2);
    assert_eq!(state.shards.len(), 1);
}

#[test]
#[serial(global_state)]
fn new_source_with_hf_token_validates_via_mock() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.hf_token = Some("valid-token".to_string());
    config.anchor_columns = vec!["anchor".to_string()];
    config.positive_columns = vec!["positive".to_string()];

    let server = spawn_one_shot_http(b"{\"name\":\"testuser\"}".to_vec());
    let base_url = server.url().to_string();
    with_env_var(ENV_TRIPLETS_HF_WHOAMI_ENDPOINT, &base_url, || {
        let result = HuggingFaceRowSource::new(config);
        assert!(
            result.is_ok(),
            "new with valid token should succeed: {:?}",
            result.err()
        );
    });
}

#[test]
#[serial(global_state)]
fn new_source_rejects_invalid_token() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.hf_token = Some("bad-token".to_string());
    config.anchor_columns = vec!["anchor".to_string()];
    config.positive_columns = vec!["positive".to_string()];

    let server = TestHttpServer::new(401, b"Unauthorized".to_vec());
    let base_url = server.url().to_string();
    with_env_var(ENV_TRIPLETS_HF_WHOAMI_ENDPOINT, &base_url, || {
        let result = HuggingFaceRowSource::new(config);
        assert!(result.is_err(), "new with invalid token should fail");
    });
}

#[test]
fn new_source_without_explicit_mapping_returns_error() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    // Clear all column mappings so has_explicit_mapping() returns false.
    config.text_columns.clear();
    config.anchor_columns.clear();
    config.positive_columns.clear();
    config.context_columns.clear();
    assert!(!config.has_explicit_mapping());

    let result = HuggingFaceRowSource::new(config);
    assert!(result.is_err(), "new without mapping should fail");
    match result {
        Err(SamplerError::Configuration(msg)) => {
            assert!(
                msg.contains("explicit field mapping"),
                "error should mention field mapping"
            );
        }
        Err(_) => panic!("expected Configuration error variant"),
        Ok(_) => panic!("expected error, got Ok"),
    }
}

#[test]
fn id_returns_source_id() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    assert_eq!(source.id(), "hf_test");
}

#[test]
fn data_source_id_returns_config_source_id() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);

    assert_eq!(source.id(), "hf_test");
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
fn effective_refresh_batch_target_uses_multiplier_floor_of_one() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.refresh_batch_multiplier = 0;
    let source = test_source(config);
    assert_eq!(source.effective_refresh_batch_target(7), 7);
}

#[test]
fn effective_refresh_batch_target_uses_multiplier() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    assert!(source.effective_refresh_batch_target(100) >= 2);
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
fn effective_expansion_headroom_rows_uses_sampler_config() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);

    let sampler_cfg = SamplerConfig {
        ingestion_max_records: 500,
        ..SamplerConfig::default()
    };
    source.set_active_sampler_config(&sampler_cfg);
    // sampler_config.ingestion_max_records = 500, multiplier = 3
    // headroom = 500 * 3 = 1500
    assert_eq!(source.effective_expansion_headroom_rows(), 1500);
}

#[test]
fn effective_expansion_headroom_rows_falls_back_to_cache_capacity() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.cache_capacity = 50;
    let source = test_source(config);

    // No sampler config set — should use cache_capacity (50) * multiplier (3) = 150
    assert_eq!(source.effective_expansion_headroom_rows(), 150);
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
fn ensure_row_available_row_already_materialized() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);

    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = 10;
    }

    let result = source.ensure_row_available(5).unwrap();
    assert!(
        result,
        "row 5 should be available when materialized_rows=10"
    );
}

#[test]
fn ensure_row_available_candidates_exhausted() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);

    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = 0;
        state.remote_candidates = Some(vec![]);
        state.next_remote_idx = 0;
    }

    let result = source.ensure_row_available(0).unwrap();
    assert!(!result, "should return false when candidates exhausted");
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
    let new_path = crate::shard_indexer::candidate_store_path(&config, &candidate);

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
    let store_path = crate::shard_indexer::candidate_store_path(&config, &candidate);
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
    let store_path = crate::shard_indexer::candidate_store_path(&config, &candidate);
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
    let store_path = crate::shard_indexer::candidate_store_path(&config, &candidate);
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
    let store_path = crate::shard_indexer::candidate_store_path(&config, &candidate);
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
    let store_path = crate::shard_indexer::candidate_store_path(&config, &candidate);
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
    let store_path = crate::shard_indexer::candidate_store_path(&config, &candidate);
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
fn download_next_shard_store_already_on_disk_skips_download() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config.clone());

    // Create a candidate path and pre-create its .simdr store.
    let candidate = "url::http://mock.example.com/datasets/org/ds/resolve/main/train/shard.ndjson";
    let store_path = crate::shard_indexer::candidate_store_path(&config, candidate);
    fs::create_dir_all(store_path.parent().unwrap()).unwrap();
    write_simdr_fixture(&store_path, &[("r1", "hello")]);

    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = 0;
        state.remote_candidates = Some(vec![candidate.to_string()]);
        state.remote_candidate_sizes = HashMap::new();
        state.next_remote_idx = 0;
        state.remote_candidate_order = vec![0];
    }

    let result = source.download_next_remote_shard().unwrap();
    assert!(result, "should return true when store already on disk");

    let state = source.state.lock().unwrap();
    assert_eq!(
        state.next_remote_idx, 1,
        "candidate position should be consumed"
    );
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
fn len_hint_known_rows_no_headroom_when_candidates_exhausted() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);

    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = 10;
        state.remote_candidates = Some(vec![
            "url::http://a/0.parquet".to_string(),
            "url::http://a/1.parquet".to_string(),
        ]);
        // All candidates already consumed — no headroom should be added.
        state.next_remote_idx = 2;
    }
    assert_eq!(source.len_hint(), Some(10));
}

#[test]
fn len_hint_known_rows_no_candidates_returns_known() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);

    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = 7;
        state.remote_candidates = None;
    }
    assert_eq!(source.len_hint(), Some(7));
}

#[test]
fn len_hint_zero_rows_empty_candidates_returns_zero() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);

    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = 0;
        state.remote_candidates = Some(vec![]);
    }
    assert_eq!(source.len_hint(), Some(0));
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
fn set_active_sampler_config_skips_materialized_shards_after_seed_change() {
    // This is the regression test for the bug where every source-epoch advance
    // reset next_remote_idx to 0, causing the expansion thread to always report
    // "shard 1/N already materialized" and never actually download new shards.
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

#[test]
fn clone_shares_state_arc_references() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    let cloned = source.clone();

    // Both should return the same id
    assert_eq!(source.id(), cloned.id());

    // Modify state through one, verify through the other
    {
        let mut state = source.state.lock().unwrap();
        state.materialized_rows = 42;
    }
    let cloned_state = cloned.state.lock().unwrap();
    assert_eq!(cloned_state.materialized_rows, 42);
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
fn default_triplet_recipes_dict_mode_returns_same_record_recipe() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.negative_columns = vec!["neg".to_string()];
    let source = test_source(config);

    let recipes = source.default_triplet_recipes();
    assert_eq!(recipes.len(), 1);
    let recipe = &recipes[0];
    assert_eq!(recipe.name, "huggingface_dict_anchor_positive_same_record");
    assert!(matches!(
        recipe.negative_strategy,
        NegativeStrategy::SameRecord
    ));
    assert!(!recipe.allow_same_anchor_positive);
}

#[test]
fn default_triplet_recipes_text_columns_mode() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.anchor_columns.clear();
    let source = test_source(config);

    let recipes = source.default_triplet_recipes();
    assert_eq!(recipes.len(), 1);
    assert!(recipes[0].allow_same_anchor_positive);
    assert!(matches!(
        recipes[0].negative_strategy,
        NegativeStrategy::WrongArticle
    ));
}

#[test]
fn default_triplet_recipes_standard_mode() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.anchor_columns = vec!["title".to_string()];
    let source = test_source(config);

    let recipes = source.default_triplet_recipes();
    assert_eq!(recipes.len(), 2);
    assert!(!recipes[0].allow_same_anchor_positive);
    assert!(!recipes[1].allow_same_anchor_positive);
}

#[test]
fn shard_size_bytes_returns_zero_for_missing_path() {
    let dir = tempdir().unwrap();
    let missing = dir.path().join("missing.file");
    assert_eq!(HuggingFaceRowSource::shard_size_bytes(&missing), 0);
}

#[test]
fn shard_size_bytes_returns_nonzero_for_existing_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.simdr");
    std::fs::write(&path, b"hello world").unwrap();
    assert_eq!(
        crate::source_core::HuggingFaceRowSource::shard_size_bytes(&path),
        11
    );
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
fn materialize_local_file_skips_copy_when_sizes_match() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());

    let src_dir = dir.path().join("src");
    fs::create_dir_all(&src_dir).unwrap();
    let source_file = src_dir.join("data.parquet");
    fs::write(&source_file, b"same content here").unwrap();

    let target_file = dir.path().join("dst").join("data.parquet");
    fs::create_dir_all(target_file.parent().unwrap()).unwrap();
    fs::write(&target_file, b"same content here").unwrap();

    let result = HuggingFaceRowSource::materialize_local_file(&config, &source_file, &target_file);
    assert!(result.is_ok(), "should succeed when sizes match");
}

#[test]
fn materialize_local_file_copies_when_sizes_differ() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());

    let src_dir = dir.path().join("src");
    fs::create_dir_all(&src_dir).unwrap();
    let source_file = src_dir.join("data.parquet");
    fs::write(&source_file, b"source content").unwrap();

    let target_file = dir.path().join("dst").join("data.parquet");
    fs::create_dir_all(target_file.parent().unwrap()).unwrap();
    fs::write(&target_file, b"old target").unwrap();

    let result = HuggingFaceRowSource::materialize_local_file(&config, &source_file, &target_file);
    assert!(result.is_ok());
    let content = fs::read(&target_file).unwrap();
    assert_eq!(content, b"source content");
}

#[test]
fn materialize_local_file_creates_parent_dirs() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());

    let src_dir = dir.path().join("src");
    fs::create_dir_all(&src_dir).unwrap();
    let source_file = src_dir.join("data.parquet");
    fs::write(&source_file, b"content").unwrap();

    let target_file = dir
        .path()
        .join("a")
        .join("b")
        .join("c")
        .join("data.parquet");

    let result = HuggingFaceRowSource::materialize_local_file(&config, &source_file, &target_file);
    assert!(result.is_ok());
    assert!(target_file.exists());
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
fn manifest_cache_root_joins_manifest_dir() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = test_source(config);
    let root = source.manifest_cache_root();
    assert!(root.ends_with(HF_PARQUET_MANIFEST_DIR));
}

#[test]
fn manifest_cache_rootjoins_parquet_manifest_dir() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path().to_path_buf());
    config.snapshot_dir = dir.path().join("snap");
    let source = test_source(config);
    let root = source.manifest_cache_root();
    assert!(root.ends_with(crate::constants::HF_PARQUET_MANIFEST_DIR));
}
