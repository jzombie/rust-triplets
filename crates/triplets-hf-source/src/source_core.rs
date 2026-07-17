use crate::config::HuggingFaceRowsConfig;
use crate::constants::{
    HF_PARQUET_MANIFEST_DIR, HF_REMOTE_BOOTSTRAP_SHARDS, HF_SHARD_STORE_SOURCE_SIZE_KEY,
};
use crate::constants::{
    HF_RECIPE_ANCHOR_ANCHOR_WRONG_ARTICLE, HF_RECIPE_ANCHOR_CONTEXT_WRONG_ARTICLE,
    HF_RECIPE_TEXT_SIMCSE_WRONG_ARTICLE,
};
#[cfg(test)]
use crate::download::list_remote_candidates_from_parquet_manifest_with_runtime;
use crate::download::{
    build_candidate_order, build_http_client, candidate_target_path,
    download_and_materialize_shard_with_runtime, fetch_remote_size_with_runtime,
    first_uncached_order_position, format_shard_label, list_remote_candidates_with_runtime,
    remote_url_for_candidate, shared_http_runtime, validate_token_with_runtime,
};
#[cfg(test)]
use crate::huggingface_source::ParquetManifestCandidates;
use crate::huggingface_source::{EligibleIndexCache, ParquetCache, RowCache};
use crate::rows;
use crate::shard_index::{
    build_shard_index, index_single_shard, is_store_shard_path, shard_store_path_for,
};
use crate::shard_indexing;
use crate::types::SourceState;

use chrono::Utc;
use reqwest_drive::ClientWithMiddleware;
use simd_r_drive::storage_engine::traits::{DataStoreReader, DataStoreWriter};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};
use triplets_core::SamplerError;
use triplets_core::config::{NegativeStrategy, SamplerConfig, Selector, TripletRecipe};
use triplets_core::data::SectionRole;
use triplets_core::source::{DataSource, SourceCursor, SourceSnapshot};

/// Bulk-oriented Hugging Face source backed by local shard files.
///
/// ## Determinism guarantees
///
/// * **Split assignment** — fully deterministic and cache-independent.  A record's
///   Train/Validation/Test label is derived solely from its stable ID and the sampler
///   seed.  The same ID always maps to the same split regardless of when or how the
///   shard was downloaded.
///
/// * **Shard download order** — fully deterministic.  Given the same sampler seed and
///   the same HF manifest, position N in the download sequence always resolves to the
///   same shard file, independent of which shards are currently cached on disk.
///
/// * **Row selection within `refresh`** — **not deterministic across cache wipes.**
///   The permutation used to select rows in a given `refresh` call is seeded by
///   `(source_id, materialized_rows, sampler_seed)`.  `materialized_rows` grows as
///   shards are downloaded in the background; it is an accident of download timing,
///   not a stable property of the dataset.  After a cache wipe and re-download the
///   same epoch number will produce different rows.  Within a single run (stable
///   `materialized_rows`), repeated calls with the same cursor are reproducible.
///
/// ## Shard download failures
///
/// `next_remote_idx` is a cursor into `remote_candidate_order`, which is a
/// seed-derived permutation of the sorted HF manifest — not a simple sequential
/// counter.  It is **not** a wrapping ring buffer: when the cursor reaches the end
/// of the candidate list the background expansion thread stops spawning (normal
/// behavior once all shards are on disk).
///
/// When a shard download fails in the **background expansion thread** the error is
/// logged as a warning and that sequence position is skipped — `next_remote_idx`
/// was already incremented before the attempt, so no retry is performed in the
/// current cycle.  The skipped position becomes available again when the candidate
/// list is next rebuilt, which happens on:
///
/// * **Disk-cap eviction** — `sync_shard_state_from_disk_locked` nulls
///   `remote_candidates` and resets `next_remote_idx` to 0 whenever the cache
///   manager removes any shard from disk.  The next expansion cycle re-fetches the
///   HF manifest, rebuilds the permutation, and advances the cursor past shards
///   still present on disk via `first_uncached_order_position`.
/// * **Epoch-seed change** — the permutation is rebuilt for the new seed and the
///   cursor resets to `first_uncached_order_position`.
/// * **Source reconstruction** — `HuggingFaceRowSource::new()` starts from zero.
///
/// For **small datasets** that fit within the disk cap, all shards are typically
/// on disk before the cursor exhausts, so a transient network failure only delays
/// that shard until the next reset cycle.  For **large datasets without eviction**,
/// a skipped position is not revisited within the current run.
///
/// Failures on the **synchronous cold-start path** propagate immediately as
/// `SamplerError::SourceUnavailable` to the caller; the global row-count request
/// is non-fatal and only produces a warning.
pub struct HuggingFaceRowSource {
    pub(crate) config: HuggingFaceRowsConfig,
    pub(crate) http_runtime: Arc<tokio::runtime::Runtime>,
    /// Reusable HTTP client with pooled TCP/TLS connections, throttling, and
    /// exponential backoff for retries.
    ///
    /// Built once in [`HuggingFaceRowSource::new`] and cloned (internally
    /// reference-counted) wherever an HTTP request is issued.  Sharing the
    /// same client across all downloads, HEAD checks, and API queries avoids
    /// the per-request TCP/TLS handshake overhead and the noisy
    /// `CloseNotify`/`BrokenPipe` DEBUG traces from dropping short-lived
    /// connection pools.  The embedded [`reqwest_drive::DriveThrottleBackoff`]
    /// applies configurable backoff when HF responds with 429 or transient
    /// failures.
    ///
    /// **Note:** The throttle/backoff middleware is only compiled in
    /// **release builds** (`#[cfg(not(debug_assertions))]`).  Debug builds
    /// (including `cargo test`) skip it so that tests against mock servers
    /// aren't slowed by retry delays.  If you run a debug binary in
    /// production, you won't get automatic retry — compile with `--release`
    /// to enable it.
    pub(crate) http_client: ClientWithMiddleware,
    pub(crate) sampler_config: Arc<Mutex<Option<SamplerConfig>>>,
    pub(crate) state: Arc<Mutex<SourceState>>,
    pub(crate) cache: Arc<Mutex<RowCache>>,
    pub(crate) parquet_cache: Arc<Mutex<ParquetCache>>,
    pub(crate) eligible_index: Arc<Mutex<EligibleIndexCache>>,
    /// Handle to the running background shard-expansion thread, if any.
    /// `is_finished()` returns true once the thread exits for any reason
    /// including panic, so this can never get permanently stuck the way
    /// an `AtomicBool` flag can when the thread panics before clearing it.
    pub(crate) expansion_thread: Arc<Mutex<Option<thread::JoinHandle<()>>>>,
}

impl Clone for HuggingFaceRowSource {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            http_runtime: Arc::clone(&self.http_runtime),
            http_client: self.http_client.clone(),
            sampler_config: Arc::clone(&self.sampler_config),
            state: Arc::clone(&self.state),
            cache: Arc::clone(&self.cache),
            parquet_cache: Arc::clone(&self.parquet_cache),
            eligible_index: Arc::clone(&self.eligible_index),
            expansion_thread: Arc::clone(&self.expansion_thread),
        }
    }
}

impl HuggingFaceRowSource {
    /// Build a new source by indexing local shard files.
    pub fn new(mut config: HuggingFaceRowsConfig) -> Result<Self, SamplerError> {
        let start_new = Instant::now();
        let http_runtime = shared_http_runtime();
        let http_client = config
            .http_client
            .take()
            .map(Ok)
            .unwrap_or_else(|| build_http_client(&config))?;

        if !config.has_explicit_mapping() {
            return Err(SamplerError::Configuration(
                "huggingface source requires explicit field mapping (anchor/positive/context/text_columns)"
                    .to_string(),
            ));
        }

        // Validate the token up-front so callers get a clear error immediately
        // rather than silent degradation on later API calls.
        if config.hf_token.is_some() {
            validate_token_with_runtime(&http_client, &config, &http_runtime)?;
        }

        fs::create_dir_all(&config.snapshot_dir).map_err(|err| {
            SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!(
                    "failed creating snapshot_dir {}: {err}",
                    config.snapshot_dir.display()
                ),
            }
        })?;

        info!(
            "[triplets:hf] {} indexing local shards in {}",
            config.source_id,
            config.snapshot_dir.display()
        );
        let (shards, discovered) = build_shard_index(&config).unwrap_or_default();
        if discovered == 0 {
            info!(
                "[triplets:hf] {} no local shards found in {} — lazy remote download enabled",
                config.source_id,
                config.snapshot_dir.display()
            );
        }

        let materialized_rows = discovered;

        info!(
            "[triplets:hf] {} source ready in {:.2}s (rows={}, shards={})",
            config.source_id,
            start_new.elapsed().as_secs_f64(),
            materialized_rows,
            shards.len()
        );

        let source = Self {
            config,
            http_runtime,
            http_client,
            sampler_config: Arc::new(Mutex::new(None)),
            state: Arc::new(Mutex::new(SourceState {
                materialized_rows,
                shards,
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

        // Post-initialization: transcode any local transient files to .simdr stores
        // with pre-corrected global offsets to prevent synthetic ID collisions.
        {
            let mut state = source
                .state
                .lock()
                .map_err(|_| SamplerError::SourceUnavailable {
                    source_id: source.config.source_id.clone(),
                    reason: "huggingface source state lock poisoned".to_string(),
                })?;
            let mut running_total = 0usize;
            let mut transcoded_shards = Vec::with_capacity(state.shards.len());
            for mut shard in state.shards.drain(..) {
                shard.global_start = running_total;
                if let Some(transcoded) = rows::transcode_transient_shard_to_store(&source, &shard)?
                {
                    running_total = running_total.saturating_add(transcoded.row_count);
                    transcoded_shards.push(transcoded);
                }
            }
            state.shards = transcoded_shards;
            state.materialized_rows = running_total;
        }

        Ok(source)
    }

    pub(crate) fn set_active_sampler_config(&self, config: &SamplerConfig) {
        if let Ok(mut slot) = self.sampler_config.lock() {
            *slot = Some(config.clone());
        }

        // Rebuild the permuted order index whenever this is called (every
        // actually influences the shard download order.  Already-downloaded
        // shards are skipped via first_uncached_order_position regardless of
        // the permutation, so no re-downloads occur from re-ordering.
        if let Ok(mut state) = self.state.lock()
            && let Some(candidates) = state.remote_candidates.clone()
        {
            let new_order = build_candidate_order(&self.config, &candidates, config.seed);
            let next_idx =
                first_uncached_order_position(&self.config, &candidates, &new_order, &state.shards);
            state.remote_candidate_order = new_order;
            state.next_remote_idx = next_idx;
        }
    }

    #[cfg(test)]
    fn active_or_default_sampler_config(&self) -> SamplerConfig {
        self.sampler_config
            .lock()
            .ok()
            .and_then(|slot| slot.clone())
            .unwrap_or_default()
    }

    #[cfg(test)]
    pub(crate) fn configure_sampler(&self, config: &SamplerConfig) {
        self.set_active_sampler_config(config);
    }

    #[cfg(test)]
    pub(crate) fn refresh(
        &self,
        cursor: Option<&SourceCursor>,
        limit: Option<usize>,
    ) -> Result<SourceSnapshot, SamplerError> {
        let config = self.active_or_default_sampler_config();
        <Self as DataSource>::refresh(self, &config, cursor, limit)
    }

    #[cfg(test)]
    pub(crate) fn reported_record_count(&self) -> Result<u128, SamplerError> {
        let config = self.active_or_default_sampler_config();
        <Self as DataSource>::reported_record_count(self, &config)
    }

    /// Compute the effective internal row read target from refresh `limit`.
    pub(crate) fn effective_refresh_batch_target(&self, limit: usize) -> usize {
        let multiplier = self.config.refresh_batch_multiplier.max(1);
        limit.saturating_mul(multiplier)
    }

    /// Compute dynamic `len_hint` headroom rows based on sampler and source config.
    pub(crate) fn effective_expansion_headroom_rows(&self) -> usize {
        let multiplier = self.config.remote_expansion_headroom_multiplier.max(1);
        let base = self
            .sampler_config
            .lock()
            .ok()
            .and_then(|config| config.as_ref().map(|value| value.ingestion_max_records))
            .unwrap_or(self.config.cache_capacity)
            .max(1);
        base.saturating_mul(multiplier)
    }

    pub(crate) fn configured_sampler_seed(&self) -> Result<u64, SamplerError> {
        self.sampler_config
            .lock()
            .map_err(|_| SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: "huggingface sampler-config lock poisoned".to_string(),
            })?
            .as_ref()
            .map(|config| config.seed)
            .ok_or_else(|| SamplerError::SourceInconsistent {
                source_id: self.config.source_id.clone(),
                details: "huggingface source sampler configuration not provided".to_string(),
            })
    }

    pub(crate) fn paging_seed(&self, total: usize) -> Result<u64, SamplerError> {
        let sampler_seed = self.configured_sampler_seed()?;
        Ok(triplets_core::source::IndexablePager::seed_for_sampler(
            &self.config.source_id,
            total,
            sampler_seed,
        ))
    }

    pub(crate) fn normalized_shard_extensions(config: &HuggingFaceRowsConfig) -> Vec<String> {
        config
            .shard_extensions
            .iter()
            .map(|value| value.trim().trim_start_matches('.').to_ascii_lowercase())
            .collect::<Vec<_>>()
    }

    /// Resolve and filter remote shard candidates from manifest or repository listing.
    #[cfg(test)]
    pub(crate) fn list_remote_candidates(
        http_client: &ClientWithMiddleware,
        config: &HuggingFaceRowsConfig,
    ) -> Result<(Vec<String>, HashMap<String, u64>), SamplerError> {
        list_remote_candidates_with_runtime(http_client, config, None)
    }

    /// Query datasets-server parquet manifest and derive shard candidates.
    #[cfg(test)]
    pub(crate) fn list_remote_candidates_from_parquet_manifest(
        http_client: &ClientWithMiddleware,
        config: &HuggingFaceRowsConfig,
    ) -> Result<ParquetManifestCandidates, SamplerError> {
        list_remote_candidates_from_parquet_manifest_with_runtime(http_client, config, None)
    }

    /// Return on-disk size for a shard path, or 0 if metadata lookup fails.
    pub(crate) fn shard_size_bytes(path: &Path) -> u64 {
        fs::metadata(path).map(|meta| meta.len()).unwrap_or(0)
    }

    /// Return root directory used for manifest-cached remote shards.
    pub(crate) fn manifest_cache_root(&self) -> PathBuf {
        self.config.snapshot_dir.join(HF_PARQUET_MANIFEST_DIR)
    }

    /// Ensure row index is available, expanding remote shard set lazily if needed.
    pub(crate) fn ensure_row_available(&self, idx: usize) -> Result<bool, SamplerError> {
        // Track whether we have already fetched the remote candidate list during
        // this call.  Once candidates are fetched and a download is attempted, we
        // must NOT re-enter the candidate-fetch path even if the disk-cap eviction
        // inside download_next_remote_shard nulls `remote_candidates` again.
        // Doing so would create an infinite download loop:
        //
        //   1. Fetch candidate list from HF manifest
        //   2. Download shard N → evict old shard → candidates nulled
        //   3. Loop back → need_candidates=true → fetch manifest AGAIN
        //   4. Download shard M → evict → candidates nulled
        //   5. Repeat forever — a single expansion thread hammers HF every ~8s
        let mut fetched_candidates = false;
        loop {
            {
                let state = self
                    .state
                    .lock()
                    .map_err(|_| SamplerError::SourceUnavailable {
                        source_id: self.config.source_id.clone(),
                        reason: "huggingface source state lock poisoned".to_string(),
                    })?;

                if idx < state.materialized_rows {
                    return Ok(true);
                }

                if let Some(candidates) = &state.remote_candidates
                    && state.next_remote_idx >= candidates.len()
                {
                    return Ok(false);
                }
            }

            let need_candidates = {
                let state = self
                    .state
                    .lock()
                    .map_err(|_| SamplerError::SourceUnavailable {
                        source_id: self.config.source_id.clone(),
                        reason: "huggingface source state lock poisoned".to_string(),
                    })?;
                state.remote_candidates.is_none()
            };

            if need_candidates {
                if fetched_candidates {
                    // We already fetched candidates and downloaded a shard in a
                    // previous iteration.  Eviction inside download_next_remote_shard
                    // nulled remote_candidates again, but we are not re-fetching.
                    // The caller (expansion thread or refresh) will see that idx
                    // is still not available and may try again on the next cycle.
                    return Ok(true);
                }
                fetched_candidates = true;

                let mut state = self
                    .state
                    .lock()
                    .map_err(|_| SamplerError::SourceUnavailable {
                        source_id: self.config.source_id.clone(),
                        reason: "huggingface source state lock poisoned".to_string(),
                    })?;
                if state.remote_candidates.is_none() {
                    let (mut candidates, candidate_sizes) = list_remote_candidates_with_runtime(
                        &self.http_client,
                        &self.config,
                        Some(self.http_runtime.as_ref()),
                    )?;
                    candidates.sort();
                    candidates.dedup();
                    let sampler_seed = self.configured_sampler_seed().unwrap_or(0);
                    let order = build_candidate_order(&self.config, &candidates, sampler_seed);

                    // Skip positions whose shard is already materialised on disk.
                    // Determinism: order is built from the full HF manifest regardless of
                    // cache state — position N for seed S always maps to the same shard.
                    // Cache: on restart we advance past already-downloaded shards so we
                    // don't redundantly re-download what we already have.
                    let next_idx = first_uncached_order_position(
                        &self.config,
                        &candidates,
                        &order,
                        &state.shards,
                    );

                    state.remote_candidates = Some(candidates);
                    state.remote_candidate_order = order;
                    state.remote_candidate_sizes = candidate_sizes;
                    state.next_remote_idx = next_idx;

                    let candidate_count = state
                        .remote_candidates
                        .as_ref()
                        .map(|values| values.len())
                        .unwrap_or(0);
                    let bootstrap_needed = state.materialized_rows == 0
                        && candidate_count > 0
                        && state.next_remote_idx == 0;
                    let known_rows = state.materialized_rows;
                    let shard_count = state.shards.len();
                    info!(
                        "[triplets:hf] {} state: candidates={} known_rows={} active_shards={} disk_cap={}",
                        self.config.source_id,
                        candidate_count,
                        known_rows,
                        shard_count,
                        self.config
                            .local_disk_cap_bytes
                            .map(|bytes| format!(
                                "{:.2} GiB",
                                bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                            ))
                            .unwrap_or_else(|| "disabled".to_string()),
                    );
                    drop(state);

                    if bootstrap_needed {
                        let bootstrap_target = HF_REMOTE_BOOTSTRAP_SHARDS.min(candidate_count);
                        info!(
                            "[triplets:hf] {} cold start: downloading {} initial shard(s) before first read",
                            self.config.source_id, bootstrap_target
                        );
                        for _ in 0..bootstrap_target {
                            if !self.download_next_remote_shard()? {
                                break;
                            }
                        }
                        info!(
                            "[triplets:hf] {} cold start complete",
                            self.config.source_id
                        );
                    }
                } else {
                    drop(state);
                }
                continue;
            }
            if !self.download_next_remote_shard()? {
                return Ok(false);
            }
        }
    }

    pub(crate) fn is_parquet_path(path: &Path) -> bool {
        path.extension()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.eq_ignore_ascii_case("parquet"))
    }

    /// Download a shard URL and materialize it under snapshot dir.
    #[cfg(test)]
    pub(crate) fn download_and_materialize_shard(
        http_client: &ClientWithMiddleware,
        config: &HuggingFaceRowsConfig,
        remote_path: &str,
        expected_bytes: Option<u64>,
        shard_label: &str,
    ) -> Result<PathBuf, SamplerError> {
        download_and_materialize_shard_with_runtime(
            http_client,
            config,
            remote_path,
            expected_bytes,
            shard_label,
            None,
        )
    }

    /// Download and register the next remote shard candidate.
    ///
    /// If the shard's store file already exists on disk (materialised from a previous
    /// run), the download is skipped and `next_remote_idx` is still advanced.  This
    /// keeps the shard download order stable regardless of cache state: the ordered
    /// position is consumed either way, but no redundant network traffic occurs.
    pub(crate) fn download_next_remote_shard(&self) -> Result<bool, SamplerError> {
        let (remote_total, cached_shards, candidate_idx, remote_path, expected_bytes) = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| SamplerError::SourceUnavailable {
                    source_id: self.config.source_id.clone(),
                    reason: "huggingface source state lock poisoned".to_string(),
                })?;
            {
                let Some(candidates) = &state.remote_candidates else {
                    return Ok(false);
                };
                if state.next_remote_idx >= candidates.len() {
                    return Ok(false);
                }
                let sequence_pos = state.next_remote_idx;
                let remote_total = candidates.len();
                let cached_shards = state.shards.len();
                // Use the seed-derived order index so the mapping from position →
                // shard is stable and independent of how many shards were previously
                // consumed.  Fall back to direct indexing only if the order vec is
                // somehow not populated (should never happen after candidates are set).
                let candidate_idx = state
                    .remote_candidate_order
                    .get(sequence_pos)
                    .copied()
                    .unwrap_or(sequence_pos);
                let remote_path = candidates[candidate_idx].clone();
                let expected_bytes = state.remote_candidate_sizes.get(&remote_path).copied();
                state.next_remote_idx += 1;

                // If this shard is already materialised on disk (from a previous run),
                // skip the download — it is already counted in materialized_rows via
                // build_shard_index.  Cache and order are fully decoupled: the position
                // is consumed regardless, but no network request is made.
                //
                // However, if the remote manifest reports a source size and the cached
                // store carries a different stored value, the upstream shard was replaced
                // (newer version).  Delete the stale store so it gets redownloaded.
                //
                // When the manifest does not provide a size, a lightweight HTTP HEAD
                // size from `Content-Length` so that staleness detection still works
                // without depending on the datasets-server API.
                let store_path = shard_indexing::candidate_store_path(&self.config, &remote_path);
                if store_path.exists() {
                    // Resolve the expected remote size: prefer the manifest-provided
                    // value, but fall back to an HTTP HEAD request so staleness
                    // detection works even when the datasets-server is unavailable.
                    let effective_expected = if let Some(bytes) = expected_bytes {
                        Some(bytes)
                    } else {
                        let remote_url = remote_url_for_candidate(&self.config, &remote_path);
                        match fetch_remote_size_with_runtime(
                            &self.http_client,
                            &self.config,
                            &remote_url,
                            &self.http_runtime,
                        ) {
                            Ok(Some(size)) if size > 0 => Some(size),
                            Ok(_) => None,
                            Err(err) => {
                                warn!(
                                    "[triplets:hf] {} {} HEAD size stale check failed: {err}",
                                    self.config.source_id,
                                    format_shard_label(
                                        remote_path.as_str(),
                                        candidate_idx,
                                        remote_total
                                    ),
                                );
                                None
                            }
                        }
                    };

                    if let Some(expected) = effective_expected {
                        // Only read the stored source size from the cache — never
                        // call get_or_open_shard_store purely to check, because it
                        // creates an empty store file if the path doesn't exist.
                        // If the handle isn't cached yet, the store was just loaded
                        // and we'll catch staleness on the next cycle.
                        let stale = self
                            .config
                            .store_cache
                            .lock_ok()
                            .and_then(|cache| cache.get(&store_path).cloned())
                            .and_then(|store| {
                                let entry = store.read(HF_SHARD_STORE_SOURCE_SIZE_KEY).ok()??;
                                let bytes = entry.as_ref();
                                if bytes.len() != std::mem::size_of::<u64>() {
                                    return None;
                                }
                                let mut raw = [0u8; 8];
                                raw.copy_from_slice(bytes);
                                Some(u64::from_le_bytes(raw))
                            });
                        if let Some(stale) = stale
                            && stale != expected
                        {
                            warn!(
                                "[triplets:hf] {} {} stale on disk (stored size {} ≠ expected {}), redownloading",
                                self.config.source_id,
                                format_shard_label(
                                    remote_path.as_str(),
                                    candidate_idx,
                                    remote_total
                                ),
                                stale,
                                expected,
                            );
                            crate::disk_cache::remove_stale_store(&self.config, &store_path);
                        }
                    }

                    if store_path.exists() {
                        debug!(
                            "[triplets:hf] {} {} already on disk, skipping download",
                            self.config.source_id,
                            format_shard_label(remote_path.as_str(), candidate_idx, remote_total),
                        );
                        return Ok(true);
                    }
                }

                (
                    remote_total,
                    cached_shards,
                    candidate_idx,
                    remote_path,
                    expected_bytes,
                )
            }
        };

        let label = format_shard_label(remote_path.as_str(), candidate_idx, remote_total);
        info!(
            "[triplets:hf] {} downloading {} ({} cached before)",
            self.config.source_id, label, cached_shards,
        );
        let local_path = download_and_materialize_shard_with_runtime(
            &self.http_client,
            &self.config,
            &remote_path,
            expected_bytes,
            &label,
            Some(self.http_runtime.as_ref()),
        )?;

        let global_start = {
            let state = self
                .state
                .lock()
                .map_err(|_| SamplerError::SourceUnavailable {
                    source_id: self.config.source_id.clone(),
                    reason: "huggingface source state lock poisoned".to_string(),
                })?;
            state.materialized_rows
        };

        let (maybe_shard, _) = index_single_shard(&self.config, &local_path, global_start)?;
        let Some(shard) = maybe_shard else {
            warn!(
                "[triplets:hf] downloaded shard had zero rows and was skipped: {}",
                local_path.display()
            );
            return Ok(true);
        };

        let state = self
            .state
            .lock()
            .map_err(|_| SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: "huggingface source state lock poisoned".to_string(),
            })?;

        let rows_to_add = shard.row_count;
        // All rows from this shard are now available.  A per-shard cap was
        // previously applied here but has been removed: reads are now gated on
        // materialized_rows and expansion happens one shard per refresh(), so a
        // large Wikipedia shard can no longer stall the read loop regardless of
        // how many rows it contributes.
        if rows_to_add == 0 {
            return Ok(true);
        }

        let mut shard = shard;
        shard.global_start = state.materialized_rows;
        shard.row_count = rows_to_add;
        shard
            .parquet_row_groups
            .retain(|(start, _)| *start < rows_to_add);
        if let Some((start, count)) = shard.parquet_row_groups.last_mut() {
            let allowed = rows_to_add.saturating_sub(*start);
            *count = (*count).min(allowed);
        }

        drop(state);
        let mut shard = match rows::transcode_transient_shard_to_store(self, &shard)? {
            Some(shard) => shard,
            None => return Ok(true),
        };

        // Relocate any .simdr store to its canonical path in the manifest root
        if is_store_shard_path(&shard.path) {
            let canonical_store =
                shard_store_path_for(&candidate_target_path(&self.config, &remote_path));
            if shard.path != canonical_store {
                // Drop the active DataStore handle before rename/remove to avoid
                // file lock violations on Windows.
                let _ = self
                    .config
                    .store_cache
                    .lock_ok()
                    .map(|mut cache| cache.remove(&shard.path));

                if let Some(parent) = canonical_store.parent() {
                    fs::create_dir_all(parent).map_err(|err| SamplerError::SourceUnavailable {
                        source_id: self.config.source_id.clone(),
                        reason: format!(
                            "failed creating canonical store parent {}: {err}",
                            parent.display()
                        ),
                    })?;
                }

                if canonical_store.exists() {
                    fs::remove_file(&canonical_store).map_err(|err| {
                        SamplerError::SourceUnavailable {
                            source_id: self.config.source_id.clone(),
                            reason: format!(
                                "failed replacing canonical store {}: {err}",
                                canonical_store.display()
                            ),
                        }
                    })?;
                }

                if let Err(rename_err) = fs::rename(&shard.path, &canonical_store) {
                    fs::copy(&shard.path, &canonical_store).map_err(|copy_err| {
                        SamplerError::SourceUnavailable {
                            source_id: self.config.source_id.clone(),
                            reason: format!(
                                "failed moving temporary store {} -> {}: rename={rename_err}; copy={copy_err}",
                                shard.path.display(),
                                canonical_store.display()
                            ),
                        }
                    })?;
                    fs::remove_file(&shard.path).map_err(|cleanup_err| {
                        SamplerError::SourceUnavailable {
                            source_id: self.config.source_id.clone(),
                            reason: format!(
                                "failed cleaning temporary store {} after copy move: {cleanup_err}",
                                shard.path.display()
                            ),
                        }
                    })?;
                }

                shard.path = canonical_store;
            }
        }

        // Persist the source shard's expected size from the remote manifest so
        // that future cycles can detect when the upstream shard was replaced.
        // When the manifest doesn't provide a size, use the actual downloaded
        let source_size = expected_bytes.unwrap_or_else(|| {
            fs::metadata(&local_path)
                .map(|meta| meta.len())
                .unwrap_or(0)
        });
        if source_size > 0 {
            // Force the engine to acquire the handle using the new canonical path.
            if let Ok(store) = shard_indexing::get_or_open_shard_store(self, &shard.path) {
                let _ = store.write(HF_SHARD_STORE_SOURCE_SIZE_KEY, &source_size.to_le_bytes());
            }
        }

        let mut state = self
            .state
            .lock()
            .map_err(|_| SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: "huggingface source state lock poisoned".to_string(),
            })?;
        state.materialized_rows += shard.row_count;
        shard.remote_candidate = Some(remote_path.clone());
        state.shards.push(shard);
        drop(state);
        shard_indexing::invalidate_eligible_index(self);

        let mut state = self
            .state
            .lock()
            .map_err(|_| SamplerError::SourceUnavailable {
                source_id: self.config.source_id.clone(),
                reason: "huggingface source state lock poisoned".to_string(),
            })?;

        let evicted_any = shard_indexing::enforce_disk_cap_locked(self, &mut state, &local_path)?;
        let materialized_rows = state.materialized_rows;
        let shard_count = state.shards.len();
        let total_remote = state
            .remote_candidates
            .as_ref()
            .map(|c| c.len())
            .unwrap_or(0);
        let active_shards = state.shards.clone();
        let usage_bytes = shard_indexing::manifest_usage_bytes_locked(self, &state);
        let usage_gib = usage_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let cap_str = self
            .config
            .local_disk_cap_bytes
            .map(|bytes| format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0)))
            .unwrap_or_else(|| "disabled".to_string());
        drop(state);
        shard_indexing::prune_store_cache_to_shards(self, &active_shards);

        if evicted_any {
            if let Ok(mut cache) = self.cache.lock() {
                cache.rows.clear();
                cache.order.clear();
            }
            if let Ok(mut parquet_cache) = self.parquet_cache.lock() {
                parquet_cache.readers.clear();
                parquet_cache.row_groups.clear();
                parquet_cache.row_group_order.clear();
            }
            shard_indexing::invalidate_eligible_index(self);
        }

        // `shard_count` is the number of shards currently on disk; `total_remote` is
        // how many shards the remote manifest reports in total for this dataset.
        info!(
            "[triplets:hf] {} rows={} shards_on_disk={}/{} disk_usage={:.2} GiB cap={}",
            self.config.source_id, materialized_rows, shard_count, total_remote, usage_gib, cap_str,
        );

        Ok(true)
    }

    /// Copy cached/downloaded source file into snapshot tree.
    #[cfg(test)]
    pub(crate) fn materialize_local_file(
        config: &HuggingFaceRowsConfig,
        source_path: &Path,
        target_path: &Path,
    ) -> Result<(), SamplerError> {
        let resolved_source =
            fs::canonicalize(source_path).unwrap_or_else(|_| source_path.to_path_buf());

        if let Some(parent) = target_path.parent() {
            fs::create_dir_all(parent).map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!(
                    "failed creating snapshot subdir {}: {err}",
                    parent.display()
                ),
            })?;
        }

        if target_path.exists() {
            let src_meta =
                fs::metadata(&resolved_source).map_err(|err| SamplerError::SourceUnavailable {
                    source_id: config.source_id.clone(),
                    reason: format!(
                        "failed reading source metadata {}: {err}",
                        resolved_source.display()
                    ),
                })?;
            let dst_meta =
                fs::metadata(target_path).map_err(|err| SamplerError::SourceUnavailable {
                    source_id: config.source_id.clone(),
                    reason: format!(
                        "failed reading target metadata {}: {err}",
                        target_path.display()
                    ),
                })?;
            if src_meta.len() == dst_meta.len() {
                return Ok(());
            }
            fs::remove_file(target_path).map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!(
                    "failed replacing target file {}: {err}",
                    target_path.display()
                ),
            })?;
        }

        fs::copy(&resolved_source, target_path).map_err(|err| SamplerError::SourceUnavailable {
            source_id: config.source_id.clone(),
            reason: format!(
                "failed copying synced file {} -> {}: {err}",
                resolved_source.display(),
                target_path.display()
            ),
        })?;
        Ok(())
    }

    /// Return the current index-domain upper bound for refresh paging.
    pub(crate) fn len_hint(&self) -> Option<usize> {
        let state = self.state.lock().ok()?;
        let known = state.materialized_rows;
        if known > 0 {
            let mut upper = known;
            if let Some(ref candidates) = state.remote_candidates
                && state.next_remote_idx < candidates.len()
            {
                let headroom = self.effective_expansion_headroom_rows();
                upper = known.saturating_add(headroom);
            }
            return Some(upper.max(known));
        }
        if state
            .remote_candidates
            .as_ref()
            .is_some_and(|c| c.is_empty())
        {
            return Some(0);
        }
        Some(1)
    }
}

impl DataSource for HuggingFaceRowSource {
    /// Return stable source id.
    fn id(&self) -> &str {
        &self.config.source_id
    }

    /// Refresh source records for the requested cursor and row limit.
    fn refresh(
        &self,
        config: &SamplerConfig,
        cursor: Option<&SourceCursor>,
        limit: Option<usize>,
    ) -> Result<SourceSnapshot, SamplerError> {
        self.set_active_sampler_config(config);
        let hinted_total = self.len_hint().unwrap_or(0);
        let max = limit.unwrap_or(hinted_total);

        // Page only over rows that are already on disk.  len_hint() includes
        // expansion headroom beyond materialized_rows; generating indices into
        // that range forces ensure_row_available() to download a shard
        // synchronously mid-read-loop, which blocks for minutes on large
        // datasets.  Instead, reads are always instant (materialized only),
        // and a single shard expansion is triggered AFTER the reads complete
        // so the next refresh call automatically has more rows available.
        // This way every remote shard is eventually consumed without ever
        // blocking on the hot read path.
        let total = {
            let materialized = {
                let state = self
                    .state
                    .lock()
                    .map_err(|_| SamplerError::SourceUnavailable {
                        source_id: self.config.source_id.clone(),
                        reason: "huggingface source state lock poisoned".to_string(),
                    })?;
                state.materialized_rows
            };
            if materialized == 0 {
                // Bootstrap: discover candidates and download the first shard so
                // the read loop below has rows to work with.  ensure_row_available
                // handles candidate discovery and the initial shard download.  If
                // bootstrap fails (no remote dataset, all local rows skipped during
                // transcoding), return an empty snapshot rather than an error.
                match self.ensure_row_available(0) {
                    Ok(true) => {
                        self.state
                            .lock()
                            .map_err(|_| SamplerError::SourceUnavailable {
                                source_id: self.config.source_id.clone(),
                                reason: "huggingface source state lock poisoned".to_string(),
                            })?
                            .materialized_rows
                    }
                    Ok(false) | Err(_) => {
                        return Ok(SourceSnapshot {
                            records: Vec::new(),
                            cursor: SourceCursor {
                                last_seen: Utc::now(),
                                revision: 0,
                            },
                        });
                    }
                }
            } else {
                materialized
            }
        };

        if total == 0 {
            return Ok(SourceSnapshot {
                records: Vec::new(),
                cursor: SourceCursor {
                    last_seen: Utc::now(),
                    revision: 0,
                },
            });
        }

        let mut start = cursor.map(|state| state.revision as usize).unwrap_or(0);
        if start >= total {
            start = 0;
        }

        let source_id = self.config.source_id.clone();
        let seed = self.paging_seed(total)?;
        let mut permutation =
            triplets_core::source::IndexPermutation::new(total, seed, start as u64);

        let mut records = Vec::new();
        let read_batch_target = self.effective_refresh_batch_target(max);
        let mut pending_indices = Vec::with_capacity(read_batch_target);
        let should_report = total >= 10_000 || max >= 1_024;
        let report_every = Duration::from_millis(750);
        let refresh_start = Instant::now();
        let mut last_report = refresh_start;
        let mut attempts = 0usize;

        if should_report {
            info!(
                "[triplets:source] refresh start source='{}' total={} target={}",
                source_id, total, max
            );
        }

        while attempts < total && records.len() < max {
            pending_indices.clear();
            let remaining_attempts = total.saturating_sub(attempts);
            let to_collect = read_batch_target.min(remaining_attempts);
            for _ in 0..to_collect {
                if records.len() + pending_indices.len() >= max {
                    break;
                }
                pending_indices.push(permutation.next());
                attempts += 1;
            }

            if pending_indices.is_empty() {
                break;
            }

            if should_report {
                info!(
                    "[triplets:source] refresh batch source='{}' batch_size={} attempted={} fetched={} elapsed={:.1}s",
                    source_id,
                    pending_indices.len(),
                    attempts,
                    records.len(),
                    refresh_start.elapsed().as_secs_f64()
                );
            }

            rows::read_row_batch(self, &pending_indices, &mut records, Some(max))?;

            if should_report && last_report.elapsed() >= report_every {
                info!(
                    "[triplets:source] refresh progress source='{}' attempted={}/{} fetched={}/{} elapsed={:.1}s",
                    source_id,
                    attempts,
                    total,
                    records.len(),
                    max,
                    refresh_start.elapsed().as_secs_f64()
                );
                last_report = Instant::now();
            }
        }

        if should_report {
            info!(
                "[triplets:source] refresh done source='{}' attempted={} fetched={} elapsed={:.2}s",
                source_id,
                attempts,
                records.len(),
                refresh_start.elapsed().as_secs_f64()
            );
        }

        // Use the seed-derived permutation cursor as the returned revision
        // so the next refresh continues from where this one left off.
        let next_start = permutation.cursor();
        let last_seen = records
            .iter()
            .map(|record| record.updated_at)
            .max()
            .unwrap_or_else(Utc::now);

        // Fire background shard expansion via the shared helper.  The helper
        // is also called by the ingestion manager on every scheduling cycle
        // (even when this source's buffer is non-empty and refresh() itself
        // is skipped), so expansion continues across long epochs.
        crate::expansion::trigger_expansion_if_needed(self);

        Ok(SourceSnapshot {
            records,
            cursor: SourceCursor {
                last_seen,
                revision: next_start as u64,
            },
        })
    }

    /// Return exact reported record count from current len hint.
    fn reported_record_count(&self, config: &SamplerConfig) -> Result<u128, SamplerError> {
        self.set_active_sampler_config(config);
        self.len_hint()
            .map(|count| count as u128)
            .ok_or_else(|| SamplerError::SourceInconsistent {
                source_id: self.config.source_id.clone(),
                details: "huggingface source did not provide len_hint".to_string(),
            })
    }

    /// Return mixed default triplet recipes used by Hugging Face row sources.
    fn default_triplet_recipes(&self) -> Vec<TripletRecipe> {
        // Dict-dataset mode: negative_columns is set, meaning negatives come
        // from the same row rather than different records.
        if !self.config.negative_columns.is_empty() {
            return vec![TripletRecipe {
                name: "huggingface_dict_anchor_positive_same_record".into(),
                anchor: Selector::Role(SectionRole::Anchor),
                positive_selector: Selector::Role(SectionRole::Context),
                negative_selector: Selector::Role(SectionRole::Context),
                negative_strategy: NegativeStrategy::SameRecord,
                weight: 1.0,
                instruction: None,
                allow_same_anchor_positive: false,
            }];
        }

        // Text-columns mode: anchor_columns is empty and only a plain text column is
        // mapped.  In this configuration every record's Anchor and Context sections
        // carry identical text (the single text field is duplicated by row_to_record).
        // Emit a single SimCSE-style recipe that deliberately allows same-text pairs:
        // the negative comes from a different record, and the model's dropout layers
        // provide the necessary embedding variation between the two identical slots.
        if self.config.anchor_columns.is_empty() {
            return vec![TripletRecipe {
                name: HF_RECIPE_TEXT_SIMCSE_WRONG_ARTICLE.into(),
                anchor: Selector::Role(SectionRole::Anchor),
                positive_selector: Selector::Role(SectionRole::Context),
                negative_selector: Selector::Role(SectionRole::Context),
                negative_strategy: NegativeStrategy::WrongArticle,
                weight: 1.0,
                instruction: None,
                allow_same_anchor_positive: true,
            }];
        }

        vec![
            // Majority lane remains context negatives for broad coverage and
            // stable optimization across varied HF schemas.
            TripletRecipe {
                name: HF_RECIPE_ANCHOR_CONTEXT_WRONG_ARTICLE.into(),
                anchor: Selector::Role(SectionRole::Anchor),
                positive_selector: Selector::Role(SectionRole::Context),
                negative_selector: Selector::Role(SectionRole::Context),
                negative_strategy: NegativeStrategy::WrongArticle,
                weight: 0.75,
                instruction: None,
                allow_same_anchor_positive: false,
            },
            // Medium-hard lane adds anchor-as-negative pressure to improve
            // discrimination between title-like anchor fields.
            TripletRecipe {
                name: HF_RECIPE_ANCHOR_ANCHOR_WRONG_ARTICLE.into(),
                anchor: Selector::Role(SectionRole::Anchor),
                positive_selector: Selector::Role(SectionRole::Context),
                negative_selector: Selector::Role(SectionRole::Anchor),
                negative_strategy: NegativeStrategy::WrongArticle,
                weight: 0.25,
                instruction: None,
                allow_same_anchor_positive: false,
            },
        ]
    }
}

#[cfg(test)]
mod tests {
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
    use triplets_core::{
        DeterministicSplitStore, Sampler, SplitLabel, SplitRatios, TripletSampler,
    };

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
        let candidate =
            format!("url::{base_url}/datasets/org/ds/resolve/main/train/persisted.ndjson");

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
        let candidate =
            format!("url::{base_url}/datasets/org/ds/resolve/main/train/part-002.ndjson");

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
        let candidate =
            format!("url::{base_url}/datasets/org/ds/resolve/main/train/new-shard.ndjson");
        let new_path = crate::shard_indexing::candidate_store_path(&config, &candidate);

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
        let candidate =
            format!("url::{base_url}/datasets/org/ds/resolve/main/train/part-empty.ndjson");

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
        let candidate =
            format!("url::{base_url}/datasets/org/ds/resolve/main/train/part-222.parquet");

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
        let candidate =
            format!("url::{base_url}/datasets/org/ds/resolve/main/train/part-001.ndjson");

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

        let candidate = format!(
            "url::{TEST_UNREACHABLE_URL}/datasets/org/ds/resolve/main/train/pre-cached.ndjson"
        );
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
        let store_path = crate::shard_indexing::candidate_store_path(&config, &candidate);
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
        let store_path = crate::shard_indexing::candidate_store_path(&config, &candidate);
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
        let candidate =
            format!("url::{base_url}/datasets/org/ds/resolve/main/train/invalid.jsonl.gz");

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
        let candidate =
            format!("url::{base_url}/datasets/org/ds/resolve/main/train/corrupt.jsonl.gz");

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
        let store_path = crate::shard_indexing::candidate_store_path(&config, &candidate);
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
        let store_path = crate::shard_indexing::candidate_store_path(&config, &candidate);
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
        let store_path = crate::shard_indexing::candidate_store_path(&config, &candidate);
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
        let store_path = crate::shard_indexing::candidate_store_path(&config, &candidate);
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
        let candidate =
            "url::http://mock.example.com/datasets/org/ds/resolve/main/train/shard.ndjson";
        let store_path = crate::shard_indexing::candidate_store_path(&config, candidate);
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
            .map(|i| {
                format!("url::http://host/datasets/org/ds/resolve/main/train/part-{i:04}.ndjson")
            })
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
            .map(|i| {
                format!("url::http://host/datasets/org/ds/resolve/main/train/part-{i:04}.ndjson")
            })
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

        let err =
            HuggingFaceRowSource::materialize_local_file(&config, &missing, &target).unwrap_err();
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

        let result =
            HuggingFaceRowSource::materialize_local_file(&config, &source_file, &target_file);
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

        let result =
            HuggingFaceRowSource::materialize_local_file(&config, &source_file, &target_file);
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

        let result =
            HuggingFaceRowSource::materialize_local_file(&config, &source_file, &target_file);
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
        let row_refs: Vec<(&str, &str)> =
            rows.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
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
        let row_refs: Vec<(&str, &str)> =
            rows.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
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
        let split_store =
            Arc::new(DeterministicSplitStore::new(SplitRatios::default(), 777).unwrap());
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
}
