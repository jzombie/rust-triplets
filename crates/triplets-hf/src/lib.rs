mod constants;
mod huggingface_source;

/// Test utilities for HuggingFace source tests.
///
/// Used by unit tests inside the `triplets-hf` crate and integration tests
/// in `crates/triplets-hf/tests/huggingface_integration.rs`.
pub mod test_utils;

pub use constants::{
    HF_TOKEN, HUGGINGFACE_GROUP, TRIPLETS_HF_INFO_ENDPOINT, TRIPLETS_HF_PARQUET_ENDPOINT,
    TRIPLETS_HF_SIZE_ENDPOINT, TRIPLETS_HF_TOKEN_TEST_DATASET, TRIPLETS_HF_WHOAMI_ENDPOINT,
};
pub use huggingface_source::{
    HF_RECIPE_TEXT_SIMCSE_WRONG_ARTICLE, HfListRoots, HfSourceEntry, build_hf_sources,
    load_hf_sources_from_list, managed_hf_list_snapshot_dir, managed_hf_snapshot_dir,
    parse_csv_fields, parse_hf_source_line, parse_hf_uri, resolve_hf_list_roots,
};
pub use huggingface_source::{HuggingFaceRowSource, HuggingFaceRowsConfig};
