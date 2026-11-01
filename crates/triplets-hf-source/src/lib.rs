#![doc = include_str!("../README.md")]
#![warn(missing_docs)]
#![allow(clippy::module_inception)]

//! Hugging Face remote layer for the triplets data pipeline.
//!
//! Local parquet mechanics (column mapping, shard indexing, row-store
//! caching) live in `triplets-parquet-source`; this crate adds hub
//! downloads, manifest expansion, authentication, and per-dataset identity
//! on top of [`triplets_parquet_source::ParquetRowSource`].

/// Download utilities and HTTP helpers for Hugging Face shard fetching.
pub mod download;

/// Test utilities for Hugging Face source tests.
///
/// Used by unit tests inside the `triplets-hf` crate and integration
/// tests in `crates/triplets-hf/tests/huggingface_integration.rs`.
pub mod test_utils;

pub use builder::{BuildFailure, BuildResult, build_hf_sources, build_hf_sources_with_weights};
pub use config::HuggingFaceRowsConfig;
pub use constants::HF_RECIPE_TEXT_SIMCSE_WRONG_ARTICLE;
pub use constants::{
    ENV_TRIPLETS_HF_TOKEN, ENV_TRIPLETS_HF_TOKEN_TEST_DATASET, ENV_TRIPLETS_HF_WHOAMI_ENDPOINT,
    HF_BASE_URL, HF_DATASETS_BASE_URL, HF_GROUP, HF_PARQUET_DEFAULT_ENDPOINT,
    HF_PUBLIC_TEST_DATASET, HF_REMOTE_URL_PREFIX,
};
pub use disk_cache::{managed_hf_list_snapshot_dir, managed_hf_snapshot_dir};
pub use parsing::{
    HfListRoots, HfSourceEntry, HfUriParts, hf_source_id_slug, load_hf_sources_from_list,
    parse_csv_fields, parse_hf_source_line, parse_hf_uri, resolve_hf_list_roots,
};
pub use source_core::HuggingFaceRowSource;

mod builder;
mod config;
mod constants;
mod disk_cache;
mod expansion;
mod parsing;
mod source_core;
