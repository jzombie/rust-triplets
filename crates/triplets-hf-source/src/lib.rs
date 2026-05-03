#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

mod constants;
mod huggingface_source;

/// Test utilities for HuggingFace source tests.
///
/// Used by unit tests inside the `triplets-hf` crate and integration tests
/// in `crates/triplets-hf/tests/huggingface_integration.rs`.
pub mod test_utils;

pub use constants::{
    ENV_TRIPLETS_HF_INFO_ENDPOINT, ENV_TRIPLETS_HF_PARQUET_ENDPOINT,
    ENV_TRIPLETS_HF_SIZE_ENDPOINT, ENV_TRIPLETS_HF_TOKEN_TEST_DATASET,
    ENV_TRIPLETS_HF_WHOAMI_ENDPOINT, HF_GROUP, HF_TOKEN,
};
pub use huggingface_source::{
    HF_RECIPE_TEXT_SIMCSE_WRONG_ARTICLE, HfListRoots, HfSourceEntry, build_hf_sources,
    load_hf_sources_from_list, managed_hf_list_snapshot_dir, managed_hf_snapshot_dir,
    parse_csv_fields, parse_hf_source_line, parse_hf_uri, resolve_hf_list_roots,
};
pub use huggingface_source::{HuggingFaceRowSource, HuggingFaceRowsConfig};
