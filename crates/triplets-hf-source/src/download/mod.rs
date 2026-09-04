/// Core download implementation for Hugging Face shard fetching.
pub mod download;
#[cfg(test)]
pub(crate) use download::{
    all_candidates_from_parquet_manifest, extract_next_link_url,
    list_remote_candidates_from_parquet_manifest_with_runtime, parse_parquet_manifest_response,
    shard_candidate_seed, shuffle_candidates_deterministically, target_matches_expected_size,
};
pub(crate) use download::{
    build_candidate_order, build_http_client, candidate_target_path,
    download_and_materialize_shard_with_runtime, first_uncached_order_position, format_shard_label,
    shared_http_runtime, validate_token_with_runtime,
};
pub use download::{
    build_http_runtime, fetch_remote_size_with_runtime, list_remote_candidates_with_runtime,
    remote_url_for_candidate,
};

#[cfg(test)]
mod download_tests;
