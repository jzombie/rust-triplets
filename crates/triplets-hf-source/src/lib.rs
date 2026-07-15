#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

mod config;
mod constants;
#[cfg(test)]
mod core_tests;
mod disk_cache;
mod huggingface_source;
mod shard_index;
mod types;

/// Test utilities for Hugging Face source tests.
///
/// Used by unit tests inside the `triplets-hf` crate and integration tests
/// in `crates/triplets-hf/tests/huggingface_integration.rs`.
pub mod test_utils;

pub use config::HuggingFaceRowsConfig;
pub use constants::{
    ENV_TRIPLETS_HF_TOKEN, ENV_TRIPLETS_HF_TOKEN_TEST_DATASET, ENV_TRIPLETS_HF_WHOAMI_ENDPOINT,
    HF_BASE_URL, HF_DATASETS_BASE_URL, HF_GROUP, HF_PARQUET_DEFAULT_ENDPOINT,
    HF_PUBLIC_TEST_DATASET, HF_REMOTE_URL_PREFIX,
};
pub use huggingface_source::HuggingFaceRowSource;
pub use huggingface_source::{
    HF_RECIPE_TEXT_SIMCSE_WRONG_ARTICLE, HfListRoots, HfSourceEntry, build_hf_sources,
    build_hf_sources_with_weights, load_hf_sources_from_list, managed_hf_list_snapshot_dir,
    managed_hf_snapshot_dir, parse_csv_fields, parse_hf_source_line, parse_hf_uri,
    resolve_hf_list_roots,
};
