use super::*;
use crate::constants::HF_PARQUET_DEFAULT_ENDPOINT;
use crate::source_core::HuggingFaceRowSource;
use crate::test_utils::test_config;
use tempfile::tempdir;

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
