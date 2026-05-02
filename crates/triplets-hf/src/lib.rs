mod constants;
mod huggingface_source;

/// Test utilities for HuggingFace source tests.
///
/// Used by both unit tests inside the `triplets-hf` crate and integration tests
/// in `tests/huggingface_integration.rs` in the main `triplets` crate.
pub mod test_utils;

pub use huggingface_source::{
    HF_RECIPE_TEXT_SIMCSE_WRONG_ARTICLE, HfListRoots, HfSourceEntry, build_hf_sources,
    load_hf_sources_from_list, managed_hf_list_snapshot_dir, managed_hf_snapshot_dir,
    parse_csv_fields, parse_hf_source_line, parse_hf_uri, resolve_hf_list_roots,
};
pub use huggingface_source::{HuggingFaceRowSource, HuggingFaceRowsConfig};
