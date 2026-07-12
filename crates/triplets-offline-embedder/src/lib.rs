#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

//! Interleaved embedding scheduler: fetches batches from a sampler,
//! embeds them via a callback, and writes results to per-split stores.

/// The interleaved embedding loop.
pub mod loop_runner;
/// High-level orchestration helpers (flush, announce, etc.).
pub mod orchestration;
/// [`TripletSampler`] adapter for the [`BatchProvider`] trait.
pub mod sampler_adapter;
/// Background-thread sampler prefetcher.
pub mod sampler_prefetcher;
/// Pure split-scheduling helpers.
pub mod split_scheduler;
/// Per-split runtime state and flush logic.
pub mod split_state;
/// [`DataStore`](simd_r_drive::storage_engine::DataStore) adapter and split-state initialisation.
pub mod store_adapter;
/// Core traits for the offline embedder.
pub mod traits;

pub use loop_runner::{
    LoopEvent, LoopHandler, StateSnapshot, compute_split_announcement, run_interleaved_loop,
};
pub use orchestration::{flush_all_pending_states, flush_batch, handle_split_exhausted};
pub use sampler_adapter::SamplerAdapter;
pub use sampler_prefetcher::{SamplerPrefetcher, filter_pair_batch, filter_triplet_batch};
pub use split_scheduler::{
    SplitScheduler, at_sample_limit, compute_deficit_str, compute_global_in_flight,
    compute_samples_per_sec, is_exhaustion_error, next_split_to_fill, scale_max, should_flush_now,
    steps_until_next_flush,
};
pub use split_state::{EmbedMode, SplitState, flush_pending, validate_embed_response};
pub use store_adapter::{SrdStoreAdapter, init_split_states_with_batch};
pub use traits::{
    BatchProvider, EmbedStore, Embedder, Result, SamplerBatch, SchedulerConfig, SchedulerError,
    StepResult,
};
pub use triplets_core::SplitLabel;
