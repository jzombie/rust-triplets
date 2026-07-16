use crate::config::HuggingFaceRowsConfig;
use crate::constants::{
    HF_PARQUET_MANIFEST_DIR, HF_REMOTE_BOOTSTRAP_SHARDS, HF_SHARD_STORE_SOURCE_SIZE_KEY,
};
#[cfg(test)]
use crate::download::list_remote_candidates_from_parquet_manifest_with_runtime;
use crate::download::{
    build_candidate_order, build_http_client, candidate_target_path,
    download_and_materialize_shard_with_runtime, fetch_remote_size_with_runtime,
    first_uncached_order_position, format_shard_label, list_remote_candidates_with_runtime,
    remote_url_for_candidate, shared_runtime, validate_token_with_runtime,
};
#[cfg(test)]
use crate::huggingface_source::ParquetManifestCandidates;
use crate::huggingface_source::{
    EXPANSION_GATE, EligibleIndexCache, HF_RECIPE_ANCHOR_ANCHOR_WRONG_ARTICLE,
    HF_RECIPE_ANCHOR_CONTEXT_WRONG_ARTICLE, HF_RECIPE_TEXT_SIMCSE_WRONG_ARTICLE, ParquetCache,
    RowCache,
};
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
        let http_runtime = shared_runtime();
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

    /// Spawn the background shard-expansion thread if expansion is needed and
    /// no download is already in progress.  This is separate from `refresh()`
    /// so the ingestion manager can call it on every scheduling cycle even
    /// when the per-source buffer has not yet drained to empty, preventing
    /// expansion from stalling for long epochs.
    pub(crate) fn trigger_expansion_if_needed(&self) {
        let needs_expansion = self
            .state
            .lock()
            .map(|state| {
                let all_consumed = state
                    .remote_candidates
                    .as_ref()
                    .is_some_and(|c| state.next_remote_idx >= c.len());
                !all_consumed
            })
            .unwrap_or(false);

        if !needs_expansion {
            return;
        }

        let already_running = self
            .expansion_thread
            .lock()
            .map(|t| t.as_ref().is_some_and(|h| !h.is_finished()))
            .unwrap_or(false);

        if already_running {
            return;
        }

        let source = self.clone();
        let handle = thread::spawn(move || {
            // Acquire the global expansion gate so only one source downloads
            // a shard at a time across all Hugging Face sources.  The gate is
            // released when the thread exits (guard dropped).
            let _gate = EXPANSION_GATE
                .get_or_init(|| std::sync::Mutex::new(()))
                .lock()
                .expect("expansion gate not poisoned");

            // If candidates not yet fetched, discover them first.
            let needs_candidates = source
                .state
                .lock()
                .map(|s| s.remote_candidates.is_none())
                .unwrap_or(false);
            if needs_candidates {
                let target = source
                    .state
                    .lock()
                    .map(|s| s.materialized_rows)
                    .unwrap_or(0);
                if let Err(err) = source.ensure_row_available(target) {
                    warn!(
                        "[triplets:hf] background expansion (candidate fetch) error \
                         (source '{}'): {}",
                        source.config.source_id, err
                    );
                }
                return;
            }
            if let Err(err) = source.download_next_remote_shard() {
                warn!(
                    "[triplets:hf] background expansion error (source '{}'): {}",
                    source.config.source_id, err
                );
            }
        });
        if let Ok(mut slot) = self.expansion_thread.lock() {
            *slot = Some(handle);
        }
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
        self.trigger_expansion_if_needed();

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
