use crate::config::HuggingFaceRowsConfig;
use crate::constants::{
    ENV_TRIPLETS_HF_WHOAMI_ENDPOINT, HF_DATASETS_BASE_URL, HF_HTTP_CONNECT_TIMEOUT_SECS,
    HF_HTTP_REQUEST_TIMEOUT_SECS, HF_PARQUET_MANIFEST_DIR, HF_REMOTE_URL_PREFIX,
    HF_RESOLVE_URL_SEPARATOR, HF_SHARD_CANDIDATE_SEED_TAG, HF_SHARED_RUNTIME_WORKER_THREADS,
    HF_TEMP_DEFAULT_EXTENSION, HF_TEMP_DOWNLOAD_PREFIX, HF_WHOAMI_DEFAULT_ENDPOINT,
};
#[cfg(not(debug_assertions))]
use crate::constants::{
    HF_THROTTLE_ADAPTIVE_JITTER_MS, HF_THROTTLE_BASE_DELAY_MS, HF_THROTTLE_MAX_CONCURRENT,
    HF_THROTTLE_MAX_RETRIES,
};
use crate::file_utils::{is_gzip_path, is_transient_text, resolve_inner_extension};
use crate::huggingface_source::ParquetManifestCandidates;
use crate::shard_index::shard_store_path_for;
use crate::source_core::HuggingFaceRowSource;
use crate::types::ShardIndex;

use reqwest_drive::ClientWithMiddleware;
use serde_json::Value;
use siphasher::sip::SipHasher;
use tracing::{info, warn};
use triplets_core::SamplerError;

use std::collections::{HashMap, HashSet};
use std::fs;
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Return a reference to the process-wide shared multi-threaded tokio
/// runtime, lazily initialized on first access.
///
/// All `HuggingFaceRowSource` instances use this single runtime so that
/// HTTP connections established by one source can be safely reused by
/// another source via the shared `reqwest::Client` connection pool.
pub(crate) fn shared_http_runtime() -> Arc<tokio::runtime::Runtime> {
    use std::sync::OnceLock;
    static RUNTIME: OnceLock<Arc<tokio::runtime::Runtime>> = OnceLock::new();
    RUNTIME
        .get_or_init(|| {
            Arc::new(
                tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(HF_SHARED_RUNTIME_WORKER_THREADS)
                    .enable_all()
                    .build()
                    .expect("failed building shared tokio runtime for Hugging Face HTTP requests"),
            )
        })
        .clone()
}

/// Build a single-threaded tokio runtime for running async HTTP operations.
///
/// The runtime enables all I/O and timer drivers.  Each source creates one
/// such runtime at construction time and reuses it for all HTTP calls,
/// avoiding the cost of building a new runtime per request.
pub fn build_http_runtime(
    config: &HuggingFaceRowsConfig,
) -> Result<tokio::runtime::Runtime, SamplerError> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| SamplerError::SourceUnavailable {
            source_id: config.source_id.clone(),
            reason: format!("failed building tokio runtime for Hugging Face HTTP path: {err}"),
        })
}

fn block_on_http_with_runtime<T>(
    runtime: Option<&tokio::runtime::Runtime>,
    config: &HuggingFaceRowsConfig,
    future: impl Future<Output = Result<T, SamplerError>>,
) -> Result<T, SamplerError> {
    if let Some(existing_runtime) = runtime {
        return existing_runtime.block_on(future);
    }
    let runtime = build_http_runtime(config)?;
    runtime.block_on(future)
}

/// Build a throttled `ClientWithMiddleware` from the config's connection
/// and auth settings, including exponential-backoff retry for rate limits
/// and transient failures.
pub(crate) fn build_http_client(
    config: &HuggingFaceRowsConfig,
) -> Result<ClientWithMiddleware, SamplerError> {
    use reqwest_drive::ClientBuilder;

    let mut builder = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(HF_HTTP_CONNECT_TIMEOUT_SECS))
        .timeout(Duration::from_secs(HF_HTTP_REQUEST_TIMEOUT_SECS));
    if let Some(token) = &config.hf_token {
        let header_value = reqwest::header::HeaderValue::from_str(&format!("Bearer {token}"))
            .map_err(|_| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: "HF_TOKEN contains characters invalid for an HTTP header value".to_string(),
            })?;
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::AUTHORIZATION, header_value);
        builder = builder.default_headers(headers);
    }
    let inner = builder
        .build()
        .map_err(|err| SamplerError::SourceUnavailable {
            source_id: config.source_id.clone(),
            reason: format!("failed building reqwest client: {err}"),
        })?;

    // Throttle/backoff middleware is skipped in debug builds
    // (cargo test, cargo build without --release).
    #[cfg(not(debug_assertions))]
    let throttle = {
        use reqwest_drive::{ThrottlePolicy, init_throttle};
        init_throttle(ThrottlePolicy {
            base_delay_ms: HF_THROTTLE_BASE_DELAY_MS,
            adaptive_jitter_ms: HF_THROTTLE_ADAPTIVE_JITTER_MS,
            max_concurrent: HF_THROTTLE_MAX_CONCURRENT,
            max_retries: HF_THROTTLE_MAX_RETRIES,
        })
    };

    let client = ClientBuilder::new(inner);
    #[cfg(not(debug_assertions))]
    let client = client.with_arc(throttle);
    Ok(client.build())
}

/// Validate a configured `hf_token` against the Hugging Face whoami endpoint.
///
/// Called once during [`HuggingFaceRowSource::new`] when `config.hf_token` is
/// `Some`.  Returns `Err(SamplerError::SourceUnavailable)` for any non-2xx
/// response (including 401 Unauthorized for invalid/expired tokens) so that
/// callers get a clear error at construction time rather than silent failures
/// on later API calls.
pub(crate) fn validate_token_with_runtime(
    http_client: &ClientWithMiddleware,
    config: &HuggingFaceRowsConfig,
    runtime: &tokio::runtime::Runtime,
) -> Result<(), SamplerError> {
    runtime.block_on(async {
        http_client
            .get(whoami_endpoint())
            .send()
            .await
            .map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!("HF_TOKEN validation request failed: {err}"),
            })?
            .error_for_status()
            .map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!(
                    "HF_TOKEN is invalid or expired — \
                       the Hugging Face API rejected the credential ({err}). \
                       Generate a new token at https://huggingface.co/settings/tokens"
                ),
            })?;
        Ok(())
    })
}

fn whoami_endpoint() -> String {
    if let Ok(value) = std::env::var(ENV_TRIPLETS_HF_WHOAMI_ENDPOINT)
        && !value.trim().is_empty()
    {
        return value;
    }
    HF_WHOAMI_DEFAULT_ENDPOINT.to_string()
}

async fn fetch_http_body_text(
    http_client: &ClientWithMiddleware,
    source_id: &str,
    endpoint: &str,
    query: &[(&str, &str)],
    endpoint_label: &str,
) -> Result<String, SamplerError> {
    // Build the URL with query parameters, then build a fresh request
    // and execute it through the middleware client for throttling/retry.
    // We use url::Url::parse_with_params so the request is built without
    // creating a throwaway reqwest::Client (Client::new() has no timeouts).
    let url = reqwest::Url::parse_with_params(endpoint, query.iter().map(|&(k, v)| (k, v)))
        .map_err(|err| SamplerError::SourceUnavailable {
            source_id: source_id.to_string(),
            reason: format!("failed building URL for {endpoint_label}: {err}"),
        })?;
    let response = http_client
        .get(url)
        .send()
        .await
        .map_err(|err| SamplerError::SourceUnavailable {
            source_id: source_id.to_string(),
            reason: format!("failed querying {endpoint_label}: {err}"),
        })?
        .error_for_status()
        .map_err(|err| SamplerError::SourceUnavailable {
            source_id: source_id.to_string(),
            reason: format!("{endpoint_label} returned non-success response: {err}"),
        })?;

    response
        .text()
        .await
        .map_err(|err| SamplerError::SourceUnavailable {
            source_id: source_id.to_string(),
            reason: format!("failed reading {endpoint_label} response body: {err}"),
        })
}

/// Return ALL shards from the parquet manifest regardless of what is already cached
/// on disk.  Shard download order and local cache are orthogonal concerns:
///
/// * **Shard download order** — must be computed from the complete HF manifest so
///   that position N for seed S always maps to the same shard file, independent of
///   what has been previously downloaded or evicted.
/// * **Cache** — which shards are already on disk is handled downstream:
///   `ensure_row_available` advances `next_remote_idx` past already-materialised
///   positions, and `download_next_remote_shard` skips any position whose store
///   file already exists without re-fetching it.
///
/// Note: row-level selection within `refresh` is *not* deterministic across cache
/// wipes.  Only the shard download sequence is stable end-to-end.
///
/// Parse the Hub API tree endpoint response into shard candidates.
///
/// The tree endpoint returns a JSON array of file objects:
/// ```json
/// [{"type":"file","path":"data/train-00000-of-00001.parquet","size":2648082,...}]
/// ```
///
/// The only cleanup performed here is deleting stale/incomplete transient parquet
/// downloads (wrong on-disk size) so they are re-fetched on the next download cycle.
pub(crate) fn all_candidates_from_parquet_manifest(
    config: &HuggingFaceRowsConfig,
    json: &Value,
) -> Result<ParquetManifestCandidates, SamplerError> {
    let accepted = HuggingFaceRowSource::normalized_shard_extensions(config);

    let mut candidates = Vec::new();
    let mut candidate_sizes = HashMap::new();
    let mut matched_manifest_entries = 0usize;

    // ── Hub API tree format: array of {"path": "...", "size": N} objects ────
    if let Some(arr) = json.as_array() {
        for entry in arr {
            let Some(file_path) = entry.get("path").and_then(Value::as_str) else {
                continue;
            };
            let ext = resolve_inner_extension(Path::new(file_path));
            if !ext
                .as_deref()
                .is_some_and(|v| accepted.iter().any(|a| a == v))
            {
                continue;
            }
            matched_manifest_entries += 1;
            let candidate = format!("{HF_REMOTE_URL_PREFIX}{file_path}");
            let expected_size = entry.get("size").and_then(Value::as_u64);

            let target = candidate_target_path(config, &candidate);
            if target.exists() && !target_matches_expected_size(&target, expected_size) {
                warn!(
                    "[triplets:hf] incomplete cached shard detected (will redownload): {}",
                    target.display()
                );
                if let Err(err) = fs::remove_file(&target)
                    && err.kind() != std::io::ErrorKind::NotFound
                {
                    return Err(SamplerError::SourceUnavailable {
                        source_id: config.source_id.clone(),
                        reason: format!(
                            "failed removing incomplete shard {}: {err}",
                            target.display()
                        ),
                    });
                }
            }
            if let Some(size) = expected_size {
                candidate_sizes.insert(candidate.clone(), size);
            }
            candidates.push(candidate);
        }
        if matched_manifest_entries > 0 {
            candidates.sort();
            candidates.dedup();
            candidate_sizes.retain(|candidate, _| candidates.binary_search(candidate).is_ok());
            return Ok((candidates, candidate_sizes, matched_manifest_entries));
        }
    }

    candidates.sort();
    candidates.dedup();
    candidate_sizes.retain(|candidate, _| candidates.binary_search(candidate).is_ok());
    Ok((candidates, candidate_sizes, matched_manifest_entries))
}

/// Resolve and filter remote shard candidates from the Hub API tree endpoint.
///
/// Returns an error when the parquet manifest is unavailable or has no entries
/// for the requested dataset/config/split.
pub fn list_remote_candidates_with_runtime(
    http_client: &ClientWithMiddleware,
    config: &HuggingFaceRowsConfig,
    runtime: Option<&tokio::runtime::Runtime>,
) -> Result<(Vec<String>, HashMap<String, u64>), SamplerError> {
    let manifest_result =
        list_remote_candidates_from_parquet_manifest_with_runtime(http_client, config, runtime);
    match &manifest_result {
        Ok((candidates, candidate_sizes, matched_manifest_entries))
            if *matched_manifest_entries > 0 =>
        {
            info!(
                "[triplets:hf] remote parquet manifest: {} shard(s) for dataset='{}' \
                     config='{}' split='{}'",
                candidates.len(),
                config.dataset_name,
                config.config_name,
                config.split_name
            );
            Ok((candidates.clone(), candidate_sizes.clone()))
        }
        Ok(_) => Err(SamplerError::SourceUnavailable {
            source_id: config.source_id.clone(),
            reason: format!(
                "parquet manifest has no entries for dataset='{}' config='{}' split='{}'",
                config.dataset_name, config.config_name, config.split_name
            ),
        }),
        Err(err) => Err(SamplerError::SourceUnavailable {
            source_id: config.source_id.clone(),
            reason: format!(
                "parquet manifest unavailable for dataset='{}': {err}",
                config.dataset_name,
            ),
        }),
    }
}

pub(crate) fn list_remote_candidates_from_parquet_manifest_with_runtime(
    http_client: &ClientWithMiddleware,
    config: &HuggingFaceRowsConfig,
    runtime: Option<&tokio::runtime::Runtime>,
) -> Result<ParquetManifestCandidates, SamplerError> {
    let base = &config.parquet_endpoint;
    // Hub API: {base}/{dataset}/tree/main/{config}?recursive=true — lists
    // files under the config subdirectory, recursing into subdirectories.
    // Scoped to config_name so we don't pull files from other language
    // configs (e.g. wikimedia/wikipedia/20231101.en should not list .fr).
    let tree_path = if config.config_name.is_empty() || config.config_name == "default" {
        format!("{base}/{}/tree/main?recursive=true", config.dataset_name)
    } else {
        format!(
            "{base}/{}/tree/main/{}?recursive=true",
            config.dataset_name, config.config_name
        )
    };
    let url = tree_path;
    info!(
        "[triplets:hf] reading Hub API tree for dataset {}",
        config.dataset_name
    );
    let body = block_on_http_with_runtime(
        runtime,
        config,
        fetch_http_body_text(
            http_client,
            &config.source_id,
            &url,
            &[],
            "Hub tree endpoint",
        ),
    )?;

    parse_parquet_manifest_response(config, &body)
}

pub(crate) fn parse_parquet_manifest_response(
    config: &HuggingFaceRowsConfig,
    body: &str,
) -> Result<ParquetManifestCandidates, SamplerError> {
    let json: Value =
        serde_json::from_str(body).map_err(|err| SamplerError::SourceUnavailable {
            source_id: config.source_id.clone(),
            reason: format!("failed parsing Hub API parquet response: {err}"),
        })?;

    all_candidates_from_parquet_manifest(config, &json)
}

/// Map a candidate identifier to the local snapshot target path.
///
/// Full CDN URLs (e.g. `https://huggingface.co/datasets/.../resolve/main/data/train.parquet`)
/// are parsed to extract the relative path after `/resolve/`.  Bare relative
/// paths (e.g. from the Hub API tree endpoint) are used directly as the suffix.
pub(crate) fn candidate_target_path(config: &HuggingFaceRowsConfig, candidate: &str) -> PathBuf {
    if let Some(url) = candidate.strip_prefix(HF_REMOTE_URL_PREFIX) {
        // Full CDN URL: extract path after /resolve/
        if let Some(suffix) = url
            .split(HF_RESOLVE_URL_SEPARATOR)
            .nth(1)
            .map(|value| value.trim_start_matches('/'))
            .filter(|value| !value.is_empty())
        {
            return config
                .snapshot_dir
                .join(HF_PARQUET_MANIFEST_DIR)
                .join(suffix);
        }
        // Bare relative path from tree endpoint (e.g. "train/000.parquet")
        return config
            .snapshot_dir
            .join(HF_PARQUET_MANIFEST_DIR)
            .join(url.trim_start_matches('/'));
    }
    config.snapshot_dir.join(candidate)
}

/// Validate target file size against expected bytes when available.
///
/// When `expected_bytes` is `Some(N)` with N > 0, the local file must have
/// exactly N bytes.  When `expected_bytes` is `None`, the file must exist
/// and be non-zero bytes (rejects 0-byte corruption from failed downloads
/// or chunked transfer encoding without Content-Length).
pub(crate) fn target_matches_expected_size(path: &Path, expected_bytes: Option<u64>) -> bool {
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };
    let local_size = metadata.len();
    match expected_bytes {
        Some(expected) if expected > 0 => local_size == expected,
        None => local_size > 0,
        _ => true,
    }
}

/// Build the full HTTP(S) remote URL for a candidate identifier.
///
/// Candidates from the parquet manifest carry the `url::` prefix followed
/// by a full URL or a repository-relative path.  This function always
/// returns a valid absolute URL: if the value after stripping the prefix
/// is already `http://` or `https://`, it is returned directly; otherwise
/// it is resolved against the HuggingFace CDN.
pub fn remote_url_for_candidate(config: &HuggingFaceRowsConfig, candidate: &str) -> String {
    let remainder = candidate
        .strip_prefix(HF_REMOTE_URL_PREFIX)
        .unwrap_or(candidate);
    if remainder.starts_with("http://") || remainder.starts_with("https://") {
        remainder.to_string()
    } else {
        format!(
            "{}/{}/resolve/main/{}",
            HF_DATASETS_BASE_URL,
            config.dataset_name,
            remainder.trim_start_matches('/')
        )
    }
}

/// Fetch the remote file size via an HTTP HEAD request.
///
/// Returns `Ok(Some(size))` when the server responds with a `Content-Length`
/// header, `Ok(None)` for non-2xx responses or missing `Content-Length`,
/// and `Err` for network / configuration failures.
pub fn fetch_remote_size_with_runtime(
    http_client: &ClientWithMiddleware,
    config: &HuggingFaceRowsConfig,
    remote_url: &str,
    runtime: &tokio::runtime::Runtime,
) -> Result<Option<u64>, SamplerError> {
    runtime.block_on(async {
        let response = http_client.head(remote_url).send().await.map_err(|err| {
            SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!("HEAD request failed for shard URL '{}': {err}", remote_url),
            }
        })?;

        if !response.status().is_success() {
            return Ok(None);
        }

        Ok(response
            .headers()
            .get(reqwest::header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok()))
    })
}

/// Build a seed-derived permutation of indices 0..candidates.len().
///
/// The candidates slice is never modified.  The returned Vec maps
/// download-position → candidate index, so for epoch seed S position N
/// always resolves to the same shard regardless of how many shards have
/// been consumed before.
pub(crate) fn build_candidate_order(
    config: &HuggingFaceRowsConfig,
    candidates: &[String],
    sampler_seed: u64,
) -> Vec<usize> {
    let n = candidates.len();
    let mut order: Vec<usize> = (0..n).collect();
    if n <= 1 {
        return order;
    }
    let base_seed = shard_candidate_seed(config, n, sampler_seed);
    let mut rng = if base_seed == 0 {
        0xdeadbeef_cafebabe
    } else {
        base_seed
    };
    for i in (1..n).rev() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let j = (rng as usize) % (i + 1);
        order.swap(i, j);
    }
    order
}

/// Return the first position in `order` whose shard store is not yet on disk
/// according to the in-memory shard list.  Returns `candidates.len()` when
/// every position is already cached (nothing left to download).
pub(crate) fn first_uncached_order_position(
    config: &HuggingFaceRowsConfig,
    candidates: &[String],
    order: &[usize],
    shards: &[ShardIndex],
) -> usize {
    let existing: HashSet<PathBuf> = shards.iter().map(|s| s.path.clone()).collect();
    order
        .iter()
        .position(|&idx| {
            !existing.contains(&crate::shard_indexing::candidate_store_path(
                config,
                &candidates[idx],
            ))
        })
        .unwrap_or(candidates.len())
}

/// Build deterministic seed used to permute remote shard candidate order.
pub(crate) fn shard_candidate_seed(
    config: &HuggingFaceRowsConfig,
    total_candidates: usize,
    sampler_seed: u64,
) -> u64 {
    let mut hasher = SipHasher::new();
    HF_SHARD_CANDIDATE_SEED_TAG.hash(&mut hasher);
    sampler_seed.hash(&mut hasher);
    config.source_id.hash(&mut hasher);
    config.dataset_name.hash(&mut hasher);
    config.config_name.hash(&mut hasher);
    config.split_name.hash(&mut hasher);
    total_candidates.hash(&mut hasher);
    hasher.finish()
}

/// Shuffle remote shard candidates into a deterministic-but-random order.
///
/// Retained for use in tests that directly assert on shuffled slices.
/// Production code uses `build_candidate_order` and keeps the list immutable.
#[cfg(test)]
pub(crate) fn shuffle_candidates_deterministically(
    config: &HuggingFaceRowsConfig,
    candidates: &mut [String],
    sampler_seed: u64,
) {
    let order = build_candidate_order(config, candidates, sampler_seed);
    // Apply the permutation in-place via a temporary clone.
    let original = candidates.to_vec();
    for (pos, &src) in order.iter().enumerate() {
        candidates[pos] = original[src].clone();
    }
}

pub(crate) fn download_and_materialize_shard_with_runtime(
    http_client: &ClientWithMiddleware,
    config: &HuggingFaceRowsConfig,
    remote_path: &str,
    expected_bytes: Option<u64>,
    shard_label: &str,
    runtime: Option<&tokio::runtime::Runtime>,
) -> Result<PathBuf, SamplerError> {
    // ── Path traversal guard ──────────────────────────────────────────────
    // Check the relative path that will determine the on-disk target location.
    // For full URLs (http/https), extract just the path portion after the host.
    // For bare relative paths, check the entire string.
    let clean_path = remote_path
        .strip_prefix(HF_REMOTE_URL_PREFIX)
        .unwrap_or(remote_path);
    let relative_for_guard = if let Some(after_proto) = clean_path
        .strip_prefix("https://")
        .or_else(|| clean_path.strip_prefix("http://"))
    {
        // "host/datasets/org/ds/resolve/main/data/train.parquet"
        // → "datasets/org/ds/resolve/main/data/train.parquet"
        after_proto
            .find('/')
            .map(|i| &after_proto[i..])
            .unwrap_or("/")
    } else {
        clean_path
    };
    for component in Path::new(relative_for_guard).components() {
        if matches!(
            component,
            std::path::Component::ParentDir | std::path::Component::Prefix(_)
        ) {
            return Err(SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!(
                    "remote path contains traversal or absolute component, rejecting: {clean_path}"
                ),
            });
        }
    }

    // ── Resolve full CDN URL ──────────────────────────────────────────────
    let resolved_url = remote_url_for_candidate(config, remote_path);

    let target = candidate_target_path(config, remote_path);
    let store_target = shard_store_path_for(&target);
    if store_target.exists() {
        return Ok(store_target);
    }

    // ── Cache validation with HEAD fallback ───────────────────────────────
    if target.exists() {
        let effective_size = match expected_bytes {
            Some(bytes) => Some(bytes),
            None => runtime.and_then(|rt| {
                fetch_remote_size_with_runtime(http_client, config, &resolved_url, rt)
                    .ok()
                    .flatten()
            }),
        };
        if target_matches_expected_size(&target, effective_size) {
            return Ok(target);
        }
        warn!(
            "[triplets:hf] replacing stale or incomplete shard before retry: {}",
            target.display()
        );
        fs::remove_file(&target).map_err(|err| SamplerError::SourceUnavailable {
            source_id: config.source_id.clone(),
            reason: format!(
                "failed removing incomplete shard {}: {err}",
                target.display()
            ),
        })?;
    }

    // ── Download ──────────────────────────────────────────────────────────
    let is_transient = HuggingFaceRowSource::is_parquet_path(&target)
        || is_gzip_path(&target)
        || is_transient_text(&target);

    if is_transient {
        // Preserve full compound extension (e.g. "jsonl.gz") so transcoder
        // routing (is_gzip_path) still works on the staged temp file.
        let file_name = target.file_name().unwrap_or_default().to_string_lossy();
        let compound_ext = file_name
            .split_once('.')
            .map(|(_, e)| e)
            .unwrap_or(HF_TEMP_DEFAULT_EXTENSION);
        let temp_target = allocate_temp_download_path(config, remote_path, compound_ext)?;
        download_remote_url_to_target_with_runtime(
            http_client,
            config,
            &resolved_url,
            &temp_target,
            expected_bytes,
            shard_label,
            runtime,
        )?;
        return Ok(temp_target);
    }

    download_remote_url_to_target_with_runtime(
        http_client,
        config,
        &resolved_url,
        &target,
        expected_bytes,
        shard_label,
        runtime,
    )?;
    Ok(target)
}

fn download_remote_url_to_target_with_runtime(
    http_client: &ClientWithMiddleware,
    config: &HuggingFaceRowsConfig,
    remote_url: &str,
    target: &Path,
    expected_bytes: Option<u64>,
    shard_label: &str,
    runtime: Option<&tokio::runtime::Runtime>,
) -> Result<(), SamplerError> {
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent).map_err(|err| SamplerError::SourceUnavailable {
            source_id: config.source_id.clone(),
            reason: format!(
                "failed creating shard output dir {}: {err}",
                parent.display()
            ),
        })?;
    }

    let temp_target = target.with_extension("part");
    if temp_target.exists() {
        let _ = fs::remove_file(&temp_target);
    }

    info!(
        "[triplets:hf] {} {} downloading shard payload -> {}",
        config.source_id,
        shard_label,
        target.display()
    );
    let http_client = http_client.clone();
    let (total_bytes, elapsed) = block_on_http_with_runtime(runtime, config, async {
        use tokio::io::AsyncWriteExt;

        let mut response = http_client
            .get(remote_url)
            .send()
            .await
            .map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!("failed downloading shard URL '{}': {err}", remote_url),
            })?
            .error_for_status()
            .map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!(
                    "download URL '{}' returned non-success response: {err}",
                    remote_url
                ),
            })?;

        let mut file = tokio::fs::File::create(&temp_target).await.map_err(|err| {
            SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!(
                    "failed creating target shard {}: {err}",
                    temp_target.display()
                ),
            }
        })?;
        let started = Instant::now();
        let mut total_bytes = 0u64;
        let mut last_report = Instant::now();
        while let Some(chunk) =
            response
                .chunk()
                .await
                .map_err(|err| SamplerError::SourceUnavailable {
                    source_id: config.source_id.clone(),
                    reason: format!("failed reading shard stream '{}': {err}", remote_url),
                })?
        {
            file.write_all(&chunk)
                .await
                .map_err(|err| SamplerError::SourceUnavailable {
                    source_id: config.source_id.clone(),
                    reason: format!(
                        "failed writing target shard {}: {err}",
                        temp_target.display()
                    ),
                })?;
            total_bytes = total_bytes.saturating_add(chunk.len() as u64);
            if last_report.elapsed() >= Duration::from_secs(2) {
                let elapsed = started.elapsed().as_secs_f64();
                if let Some(expected) = expected_bytes
                    && expected > 0
                {
                    let pct = ((total_bytes as f64 / expected as f64) * 100.0).clamp(0.0, 100.0);
                    let rate = if elapsed > 0.0 {
                        total_bytes as f64 / elapsed
                    } else {
                        0.0
                    };
                    let eta_secs = if rate > 0.0 && total_bytes < expected {
                        (expected.saturating_sub(total_bytes) as f64) / rate
                    } else {
                        0.0
                    };
                    info!(
                        "[triplets:hf] {} {} download progress: {:.1}/{:.1} MiB ({:.1}%, {:.1}s elapsed, ETA {:.1}s)",
                        config.source_id,
                        shard_label,
                        total_bytes as f64 / (1024.0 * 1024.0),
                        expected as f64 / (1024.0 * 1024.0),
                        pct,
                        elapsed,
                        eta_secs.max(0.0)
                    );
                } else {
                    info!(
                        "[triplets:hf] {} {} download progress: {:.1} MiB ({:.1}s)",
                        config.source_id,
                        shard_label,
                        total_bytes as f64 / (1024.0 * 1024.0),
                        elapsed
                    );
                }
                last_report = Instant::now();
            }
        }
        file.flush()
            .await
            .map_err(|err| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!(
                    "failed flushing target shard {}: {err}",
                    temp_target.display()
                ),
            })?;
        Ok((total_bytes, started.elapsed().as_secs_f64()))
    })?;

    if let Some(expected) = expected_bytes
        && expected > 0
    {
        let pct = ((total_bytes as f64 / expected as f64) * 100.0).clamp(0.0, 100.0);
        info!(
            "[triplets:hf] {} {} download complete: {:.1}/{:.1} MiB ({:.1}%) in {:.1}s",
            config.source_id,
            shard_label,
            total_bytes as f64 / (1024.0 * 1024.0),
            expected as f64 / (1024.0 * 1024.0),
            pct,
            elapsed
        );
    } else {
        info!(
            "[triplets:hf] {} {} download complete: {:.1} MiB in {:.1}s",
            config.source_id,
            shard_label,
            total_bytes as f64 / (1024.0 * 1024.0),
            elapsed
        );
    }

    fs::rename(&temp_target, target).map_err(|err| SamplerError::SourceUnavailable {
        source_id: config.source_id.clone(),
        reason: format!(
            "failed moving downloaded shard {} -> {}: {err}",
            temp_target.display(),
            target.display()
        ),
    })?;

    Ok(())
}

/// Build the stable human-readable label used in every shard-related log line.
///
/// Format: `<file> (shard <M>/<total>)` where `M` is the 1-based index of this
/// shard file in the sorted remote manifest and `total` is the total number of
/// remote shards.  This label is purely file-derived and never depends on the
/// ephemeral shuffle-position counter (`next_remote_idx`), which can reset
/// whenever the candidate list is rebuilt for any reason, making position-based
/// counters unfit for human interpretation.
pub(crate) fn format_shard_label(
    remote_path: &str,
    candidate_idx: usize,
    candidate_total: usize,
) -> String {
    let file = remote_path
        .rsplit('/')
        .next()
        .unwrap_or(remote_path)
        .trim_start_matches("url::");
    format!("{file} (shard {}/{candidate_total})", candidate_idx + 1)
}

fn allocate_temp_download_path(
    config: &HuggingFaceRowsConfig,
    remote_path: &str,
    extension: &str,
) -> Result<PathBuf, SamplerError> {
    let mut hasher = SipHasher::new();
    config.source_id.hash(&mut hasher);
    remote_path.hash(&mut hasher);
    let fingerprint = hasher.finish();
    let prefix = format!("{HF_TEMP_DOWNLOAD_PREFIX}{fingerprint:016x}_");
    let suffix = format!(".{}", extension.trim_start_matches('.'));
    let temp_file = tempfile::Builder::new()
        .prefix(&prefix)
        .suffix(&suffix)
        .tempfile()
        .map_err(|err| SamplerError::SourceUnavailable {
            source_id: config.source_id.clone(),
            reason: format!("failed creating temporary download file: {err}"),
        })?;
    let (_, path) = temp_file
        .keep()
        .map_err(|err| SamplerError::SourceUnavailable {
            source_id: config.source_id.clone(),
            reason: format!(
                "failed persisting temporary download path for '{}': {}",
                remote_path, err.error
            ),
        })?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::HF_DATASETS_BASE_URL;
    use crate::test_utils::{
        TEST_UNREACHABLE_URL, TestHttpServer, spawn_one_shot_http, test_config, test_http_client,
        with_env_var, write_simdr_fixture,
    };
    use serde_json::json;
    use serial_test::serial;
    use std::io::{Read, Write};
    use std::path::PathBuf;
    use std::sync::Mutex;
    use tempfile::tempdir;

    #[test]
    fn remote_url_for_candidate_constructs_correct_urls() {
        // url:: prefix with full URL: returned as-is.
        let config = test_config(PathBuf::from("/tmp/snap"));
        let full_url =
            format!("url::{HF_DATASETS_BASE_URL}/org/ds/resolve/main/train/part-000.parquet");
        let result = remote_url_for_candidate(&config, &full_url);
        assert_eq!(
            result,
            format!("{HF_DATASETS_BASE_URL}/org/ds/resolve/main/train/part-000.parquet")
        );

        // url:: prefix with relative path (Hub API format): CDN prefix is constructed.
        let hub_relative = "url::data/train-00000-of-00001.parquet";
        let result = remote_url_for_candidate(&config, hub_relative);
        assert_eq!(
            result,
            format!(
                "{HF_DATASETS_BASE_URL}/org/dataset/resolve/main/data/train-00000-of-00001.parquet"
            )
        );

        // Bare path (hf-hub sibling fallback): CDN prefix is prepended.
        let bare_path = "data/train-00000-of-00001.parquet";
        let result = remote_url_for_candidate(&config, bare_path);
        assert_eq!(
            result,
            format!(
                "{HF_DATASETS_BASE_URL}/org/dataset/resolve/main/data/train-00000-of-00001.parquet"
            )
        );

        // Bare path with leading slash.
        let bare_path = "/data/train-00000-of-00001.parquet";
        let result = remote_url_for_candidate(&config, bare_path);
        assert_eq!(
            result,
            format!(
                "{HF_DATASETS_BASE_URL}/org/dataset/resolve/main/data/train-00000-of-00001.parquet"
            )
        );
    }

    #[test]
    fn remote_url_for_candidate_builds_bare_urls() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let r1 = remote_url_for_candidate(&config, "url::https://server/parquet");
        assert_eq!(r1, "https://server/parquet");
        let r2 = remote_url_for_candidate(&config, "data/train-000.parquet");
        assert!(r2.contains("/resolve/main/"));
    }

    #[test]
    fn remote_url_for_candidate_bare_path_resolves_to_cdn() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());

        let url = remote_url_for_candidate(&config, "train/shard.ndjson");
        assert!(url.contains(HF_DATASETS_BASE_URL));
        assert!(url.contains("train/shard.ndjson"));
    }

    #[test]
    fn remote_url_for_candidate_full_url_returned_directly() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());

        let url = remote_url_for_candidate(&config, "url::https://cdn.example.com/shard.parquet");
        assert_eq!(url, "https://cdn.example.com/shard.parquet");
    }

    #[test]
    fn http_client_builds_with_token() {
        let temp = tempdir().unwrap();
        let mut config = test_config(temp.path().to_path_buf());
        config.hf_token = Some("test-bearer-token".to_string());
        let result = build_http_client(&config);
        assert!(
            result.is_ok(),
            "build_http_client should succeed with a well-formed token string"
        );
    }

    #[test]
    #[serial(global_state)]
    fn validate_token_accepts_200_response() {
        let temp = tempdir().unwrap();
        let mut config = test_config(temp.path().to_path_buf());
        config.hf_token = Some("valid-test-token".to_string());
        let server = spawn_one_shot_http(b"{\"name\":\"testuser\"}".to_vec());
        let base_url = server.url().to_string();
        with_env_var(ENV_TRIPLETS_HF_WHOAMI_ENDPOINT, &base_url, || {
            let client = test_http_client();
            let runtime = build_http_runtime(&config).unwrap();
            let result = validate_token_with_runtime(&client, &config, &runtime);
            assert!(result.is_ok(), "200 response should pass token validation");
        });
    }

    #[test]
    #[serial(global_state)]
    fn validate_token_rejects_401_response() {
        let temp = tempdir().unwrap();
        let mut config = test_config(temp.path().to_path_buf());
        config.hf_token = Some("invalid-test-token".to_string());
        let server = TestHttpServer::new(401, b"Unauthorized".to_vec());
        let base_url = server.url().to_string();
        with_env_var(ENV_TRIPLETS_HF_WHOAMI_ENDPOINT, &base_url, || {
            let client = test_http_client();
            let runtime = build_http_runtime(&config).unwrap();
            let result = validate_token_with_runtime(&client, &config, &runtime);
            assert!(result.is_err(), "401 response should fail token validation");
            match result {
                Err(SamplerError::SourceUnavailable { reason, .. }) => {
                    assert!(
                        reason.contains("invalid or expired"),
                        "error should mention invalid/expired, got: {reason}"
                    );
                }
                _ => panic!("expected SamplerError::SourceUnavailable"),
            }
        });
    }

    #[test]
    fn all_candidates_from_parquet_manifest_returns_all_with_sizes() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        // Hub API tree endpoint format: array of {"path": "...", "size": N} objects
        let payload = json!([
            {"type": "file", "path": "train/000.parquet", "size": 100},
            {"type": "file", "path": "train/001.ndjson", "size": 200},
            {"type": "file", "path": "train/002.txt", "size": 50}
        ]);

        let (candidates, sizes, matched) =
            all_candidates_from_parquet_manifest(&config, &payload).unwrap();
        assert_eq!(candidates.len(), 2);
        assert!(candidates.iter().any(|c| c.ends_with("train/000.parquet")));
        assert!(candidates.iter().any(|c| c.ends_with("train/001.ndjson")));
        assert_eq!(sizes.len(), 2, "tree format provides sizes");
        assert_eq!(matched, 2);
    }

    #[test]
    fn all_candidates_from_parquet_manifest_includes_cached_and_replaces_stale() {
        // Suppress the expected WARN "incomplete cached shard detected (will redownload)".
        let _quiet = tracing::subscriber::set_default(
            tracing_subscriber::fmt()
                .with_max_level(tracing::Level::ERROR)
                .finish(),
        );
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());

        // A parquet file with the correct declared size — considered fully cached.
        let complete_candidate = format!("{HF_REMOTE_URL_PREFIX}train/000.parquet");
        let complete_target = candidate_target_path(&config, &complete_candidate);
        fs::create_dir_all(complete_target.parent().unwrap()).unwrap();
        fs::write(&complete_target, vec![1u8; 7]).unwrap();

        // A parquet file with the WRONG size — stale/incomplete, must be deleted.
        let stale_candidate = format!("{HF_REMOTE_URL_PREFIX}train/001.parquet");
        let stale_target = candidate_target_path(&config, &stale_candidate);
        fs::create_dir_all(stale_target.parent().unwrap()).unwrap();
        fs::write(&stale_target, vec![2u8; 3]).unwrap();

        let payload = json!([
            {"type": "file", "path": "train/000.parquet", "size": 7},
            {"type": "file", "path": "train/001.parquet", "size": 9}
        ]);

        let (candidates, sizes, matched) =
            all_candidates_from_parquet_manifest(&config, &payload).unwrap();

        // Both shards are returned — cache state does not affect the candidate list.
        assert_eq!(candidates.len(), 2, "both shards must appear in candidates");
        // Complete shard: file exists and was not deleted.
        assert!(
            complete_target.exists(),
            "complete shard must not be deleted"
        );
        // Stale shard: wrong-size file was deleted so it will be re-fetched.
        assert!(!stale_target.exists(), "stale shard must be deleted");
        assert_eq!(sizes.len(), 2);
        assert_eq!(matched, 2);
    }

    #[test]
    fn candidates_from_parquet_manifest_errors_when_removing_incomplete_target_fails() {
        // Suppress the expected WARN "incomplete cached shard detected (will redownload)"
        // emitted before the attempted removal fails.  The removal failure is what this
        // test asserts on; the warn preceding it is correct production behaviour.
        let _quiet = tracing::subscriber::set_default(
            tracing_subscriber::fmt()
                .with_max_level(tracing::Level::ERROR)
                .finish(),
        );
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let candidate = format!("{HF_REMOTE_URL_PREFIX}train/blocked.parquet");
        let target = candidate_target_path(&config, &candidate);
        fs::create_dir_all(&target).unwrap();

        let payload = json!([
            {"type": "file", "path": "train/blocked.parquet", "size": 1}
        ]);

        let err = all_candidates_from_parquet_manifest(&config, &payload);
        assert!(err.is_err());
    }

    #[test]
    fn uncached_candidates_from_parquet_manifest_returns_empty_without_entries() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let payload = json!({"other": []});
        let (candidates, sizes, matched) =
            all_candidates_from_parquet_manifest(&config, &payload).unwrap();
        assert!(candidates.is_empty());
        assert!(sizes.is_empty());
        // No parquet_files key → zero matched entries.
        assert_eq!(matched, 0);
    }

    #[test]
    fn all_candidates_from_parquet_manifest_empty_array() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let json = json!([]);
        let (candidates, sizes, matched) =
            crate::download::all_candidates_from_parquet_manifest(&config, &json).unwrap();
        assert!(candidates.is_empty());
        assert!(sizes.is_empty());
        assert_eq!(matched, 0);
    }

    #[test]
    fn all_candidates_from_parquet_manifest_filters_non_parquet() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        // Only accept parquet files
        config.shard_extensions = vec!["parquet".to_string()];
        let json = json!([
            {"type": "file", "path": "data/train-000.parquet", "size": 100},
            {"type": "file", "path": "data/README.md", "size": 50},
            {"type": "file", "path": "data/train.jsonl", "size": 200}
        ]);
        let (candidates, sizes, matched) =
            crate::download::all_candidates_from_parquet_manifest(&config, &json).unwrap();
        // Only .parquet is accepted
        assert_eq!(candidates.len(), 1);
        assert!(candidates[0].contains("train-000.parquet"));
        assert_eq!(matched, 1);
        assert_eq!(sizes.get(&candidates[0]), Some(&100));
    }

    #[test]
    fn all_candidates_from_parquet_manifest_skips_entries_without_path() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let json = json!([
            {"type": "file", "size": 100},
            {"type": "file", "path": "data/train-000.parquet", "size": 200}
        ]);
        let (candidates, _, matched) =
            crate::download::all_candidates_from_parquet_manifest(&config, &json).unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(matched, 1);
    }

    #[test]
    fn all_candidates_from_parquet_manifest_handles_non_array() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let json = json!({"not": "an array"});
        let (candidates, _, matched) =
            crate::download::all_candidates_from_parquet_manifest(&config, &json).unwrap();
        assert!(candidates.is_empty());
        assert_eq!(matched, 0);
    }

    #[test]
    fn all_candidates_from_parquet_manifest_deduplicates() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let json = json!([
            {"type": "file", "path": "data/train-000.parquet", "size": 100},
            {"type": "file", "path": "data/train-000.parquet", "size": 100}
        ]);
        let (candidates, _, matched) =
            crate::download::all_candidates_from_parquet_manifest(&config, &json).unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(matched, 2); // Both entries matched even though deduplicated
    }

    #[test]
    fn all_candidates_from_parquet_manifest_no_size() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let json = json!([
            {"type": "file", "path": "data/train-000.parquet"}
        ]);
        let (candidates, sizes, matched) =
            crate::download::all_candidates_from_parquet_manifest(&config, &json).unwrap();
        assert_eq!(candidates.len(), 1);
        assert!(sizes.is_empty()); // No size provided
        assert_eq!(matched, 1);
    }

    #[test]
    fn candidate_target_path_uses_bare_path_when_no_resolve_segment() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        // Bare relative path from tree endpoint (no /resolve/ segment)
        let candidate = "url::train/000.parquet";
        let target = candidate_target_path(&config, candidate);
        assert!(target.ends_with("_parquet_manifest/train/000.parquet"));
    }

    #[test]
    fn candidate_target_path_maps_remote_urls_under_manifest_root() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let candidate =
            "url::https://huggingface.co/datasets/org/ds/resolve/main/train/part-000.parquet";
        let target = candidate_target_path(&config, candidate);
        assert!(target.ends_with("_parquet_manifest/main/train/part-000.parquet"));
    }

    #[test]
    fn candidate_target_path_keeps_local_candidates_relative() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let candidate = "train/part-001.ndjson";
        let target = candidate_target_path(&config, candidate);
        assert_eq!(target, config.snapshot_dir.join(candidate));
    }

    #[test]
    fn candidate_target_path_bare_relative_path() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());

        let path = candidate_target_path(&config, "train/0001.parquet");
        assert!(path.ends_with("train/0001.parquet"));
        assert!(path.starts_with(dir.path()));
    }

    #[test]
    fn candidate_target_path_full_url_extracts_suffix() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());

        let path = candidate_target_path(
            &config,
            "url::https://host/datasets/org/ds/resolve/main/data/shard.ndjson",
        );
        assert!(path.ends_with("data/shard.ndjson"));
    }

    #[test]
    fn target_matches_expected_size_is_false_for_missing_path() {
        let dir = tempdir().unwrap();
        let missing = dir.path().join("missing.bin");
        assert!(!target_matches_expected_size(&missing, Some(1)));
    }

    #[test]
    fn target_matches_expected_size_validates_when_expected_is_provided() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("payload.bin");
        fs::write(&path, vec![0u8; 5]).unwrap();

        assert!(target_matches_expected_size(&path, Some(5)));
        assert!(!target_matches_expected_size(&path, Some(4)));
        assert!(target_matches_expected_size(&path, None));
    }

    #[test]
    fn target_matches_expected_size_zero_expected_returns_true() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("payload.bin");
        fs::write(&path, vec![0u8; 5]).unwrap();
        // expected_bytes = Some(0) falls into the `_ => true` branch
        assert!(target_matches_expected_size(&path, Some(0)));
    }

    #[test]
    fn target_matches_expected_size_none_requires_nonzero() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.bin");
        fs::write(&path, vec![]).unwrap();
        assert!(!target_matches_expected_size(&path, None));
    }

    #[test]
    fn target_matches_expected_size_mismatch_returns_false() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("mismatch.bin");
        fs::write(&file, b"hello").unwrap();
        assert!(
            !target_matches_expected_size(&file, Some(100)),
            "size mismatch should return false"
        );
    }

    #[test]
    fn target_matches_expected_size_missing_file_returns_false() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("nonexistent.bin");
        assert!(
            !target_matches_expected_size(&file, Some(100)),
            "missing file should return false"
        );
    }

    #[test]
    fn shard_candidate_seed_and_shuffle_are_deterministic() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.source_id = "hf_rotator".to_string();

        let seed_a = shard_candidate_seed(&config, 12, 1);
        let seed_b = shard_candidate_seed(&config, 12, 2);
        assert_ne!(seed_a, seed_b);

        let baseline = vec!["c".to_string(), "a".to_string(), "b".to_string()];
        let mut left = baseline.clone();
        let mut right = baseline;
        shuffle_candidates_deterministically(&config, &mut left, 42);
        shuffle_candidates_deterministically(&config, &mut right, 42);
        assert_eq!(left, right);

        // Different seeds produce different orderings for non-trivial inputs.
        let mut alt = vec!["c".to_string(), "a".to_string(), "b".to_string()];
        shuffle_candidates_deterministically(&config, &mut alt, 99);
        // Membership is preserved regardless of seed.
        let mut sorted_left = left.clone();
        sorted_left.sort();
        let mut sorted_alt = alt.clone();
        sorted_alt.sort();
        assert_eq!(sorted_left, sorted_alt);
    }

    #[test]
    fn shard_candidate_seed_is_seeded_and_source_scoped() {
        let dir = tempdir().unwrap();
        let mut a = test_config(dir.path().join("a"));
        let mut b = test_config(dir.path().join("b"));
        a.source_id = "source_a".to_string();
        b.source_id = "source_b".to_string();

        let with_seed_a = shard_candidate_seed(&a, 100, 42);
        let with_seed_a_again = shard_candidate_seed(&a, 100, 42);
        assert_eq!(with_seed_a, with_seed_a_again);

        let with_seed_b = shard_candidate_seed(&b, 100, 42);
        assert_ne!(with_seed_a, with_seed_b);

        let different_seed_a = shard_candidate_seed(&a, 100, 7);
        assert_ne!(with_seed_a, different_seed_a);
    }

    #[test]
    fn shard_candidate_seed_changes_with_sampler_seed() {
        // Verifies that different sampler_seed values (which in production
        // include the epoch_step XOR from IngestionManager) produce
        // different shard permutations, while the same seed is deterministic.
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());

        // Different sampler seeds → different shard candidate seeds.
        let seed_1 = shard_candidate_seed(&config, 100, 1);
        let seed_2 = shard_candidate_seed(&config, 100, 2);
        assert_ne!(
            seed_1, seed_2,
            "different seeds must produce different shard seeds"
        );

        // Same sampler seed → deterministic.
        let seed_1_again = shard_candidate_seed(&config, 100, 1);
        assert_eq!(seed_1, seed_1_again, "same seed must be deterministic");

        // Verify the permutation itself changes with seed.
        let candidates: Vec<String> = (0..10).map(|i| format!("shard-{i:02}")).collect();
        let order_1 = build_candidate_order(&config, &candidates, 1);
        let order_2 = build_candidate_order(&config, &candidates, 2);
        assert_ne!(
            order_1, order_2,
            "different seeds must produce different shard orders"
        );

        // Same seed produces same order.
        let order_1_again = build_candidate_order(&config, &candidates, 1);
        assert_eq!(order_1, order_1_again, "same seed must produce same order");
    }

    #[test]
    fn shuffle_candidates_deterministically_is_noop_for_singleton() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let mut candidates = vec!["one".to_string()];
        shuffle_candidates_deterministically(&config, &mut candidates, 1);
        assert_eq!(candidates, vec!["one".to_string()]);
    }

    #[test]
    fn shuffle_candidates_deterministically_preserves_membership() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let original = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut shuffled = original.clone();
        shuffle_candidates_deterministically(&config, &mut shuffled, 1);
        let mut sorted_original = original;
        let mut sorted_shuffled = shuffled;
        sorted_original.sort();
        sorted_shuffled.sort();
        assert_eq!(sorted_shuffled, sorted_original);
    }

    #[test]
    fn build_candidate_order_single_element() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let candidates = vec!["a".to_string()];
        let order = build_candidate_order(&config, &candidates, 42);
        assert_eq!(order, vec![0]);
    }

    #[test]
    fn build_candidate_order_empty() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let candidates: Vec<String> = vec![];
        let order = build_candidate_order(&config, &candidates, 42);
        assert!(order.is_empty());
    }

    #[test]
    fn build_candidate_order_base_seed_zero_uses_fallback() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.source_id = "".to_string();
        config.dataset_name = "".to_string();
        config.config_name = "".to_string();
        config.split_name = "".to_string();
        let candidates = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        // With all-empty fields, shard_candidate_seed may return 0, triggering fallback
        let order = build_candidate_order(&config, &candidates, 0);
        assert_eq!(order.len(), 3);
        // All indices must be present
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn remote_shard_permutation_is_deterministic_by_sampler_seed() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let total = 8usize;

        let seed_a = shard_candidate_seed(&config, total, 7);
        let seed_b = shard_candidate_seed(&config, total, 7);
        let seed_c = shard_candidate_seed(&config, total, 10);

        let mut perm_a = triplets_core::source::IndexPermutation::new(total, seed_a, 0);
        let mut perm_b = triplets_core::source::IndexPermutation::new(total, seed_b, 0);
        let mut perm_c = triplets_core::source::IndexPermutation::new(total, seed_c, 0);

        let take = 6usize;
        let order_a: Vec<usize> = (0..take).map(|_| perm_a.next()).collect();
        let order_b: Vec<usize> = (0..take).map(|_| perm_b.next()).collect();
        let order_c: Vec<usize> = (0..take).map(|_| perm_c.next()).collect();

        assert_eq!(order_a, order_b);
        assert_ne!(order_a, order_c);
    }

    #[test]
    fn remote_shard_permutation_is_deterministic() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let c = ["a", "b", "c", "d", "e"];
        let c1: Vec<String> = c.iter().map(|s| s.to_string()).collect();
        let c2: Vec<String> = c.iter().map(|s| s.to_string()).collect();
        let o1 = build_candidate_order(&config, &c1, 42);
        let o2 = build_candidate_order(&config, &c2, 42);
        assert_eq!(o1, o2);
        let o3 = build_candidate_order(&config, &c1, 99);
        assert_ne!(o1, o3);
    }

    #[test]
    fn format_shard_label_includes_totals() {
        let label = format_shard_label("data/train-000.parquet", 0, 5);
        assert!(label.contains("1/5"));
        assert!(label.contains("train-000.parquet"));
    }

    #[test]
    fn format_shard_label_strips_url_prefix() {
        let label = format_shard_label("url::data/train-000.parquet", 2, 10);
        assert!(label.contains("3/10"));
        assert!(label.contains("train-000.parquet"));
    }

    #[test]
    fn format_shard_label_handles_no_slash() {
        let label = format_shard_label("train.parquet", 0, 1);
        assert_eq!(label, "train.parquet (shard 1/1)");
    }

    #[test]
    fn parse_parquet_manifest_response_errors_on_invalid_json() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let parsed = parse_parquet_manifest_response(&config, "{bad-json");
        assert!(parsed.is_err());
    }

    #[test]
    fn parse_parquet_manifest_response_returns_candidates() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let body = serde_json::to_string(&json!([
             {"type": "file", "path": "https://host/datasets/x/resolve/main/train/0.parquet", "size": 100}
         ]))
         .unwrap();

        let (candidates, sizes, matched) = parse_parquet_manifest_response(&config, &body).unwrap();
        assert_eq!(candidates.len(), 1);
        assert!(!sizes.is_empty());
        assert_eq!(matched, 1);
    }

    #[test]
    fn parse_parquet_manifest_response_invalid_json() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let err =
            crate::download::parse_parquet_manifest_response(&config, "not json").unwrap_err();
        assert!(matches!(err, SamplerError::SourceUnavailable { .. }));
    }

    #[test]
    fn parse_parquet_manifest_response_valid_json() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let body = r#"[{"type":"file","path":"data/train-000.parquet","size":100}]"#;
        let (candidates, _, matched) =
            crate::download::parse_parquet_manifest_response(&config, body).unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(matched, 1);
    }

    #[test]
    #[serial(global_state)]
    fn list_remote_candidates_falls_back_when_manifest_query_fails() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.dataset_name = "invalid///dataset".to_string();
        config.parquet_endpoint = TEST_UNREACHABLE_URL.to_string();

        let client = test_http_client();
        let result = HuggingFaceRowSource::list_remote_candidates(&client, &config);
        assert!(result.is_err());
    }

    #[test]
    #[serial(global_state)]
    fn list_remote_candidates_from_parquet_manifest_uses_test_endpoint_override() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        let body = serde_json::to_vec(&json!([
             {"type": "file", "path": "https://host/datasets/x/resolve/main/train/0.parquet", "size": 100}
         ]))
         .unwrap();
        let server = spawn_one_shot_http(body);
        let base_url = server.url().to_string();

        config.parquet_endpoint = base_url;
        let client = test_http_client();
        let (candidates, sizes, matched) =
            HuggingFaceRowSource::list_remote_candidates_from_parquet_manifest(&client, &config)
                .unwrap();

        assert_eq!(candidates.len(), 1);
        assert!(!sizes.is_empty());
        assert_eq!(matched, 1);
    }

    #[test]
    #[serial(global_state)]
    fn list_remote_candidates_returns_manifest_candidates() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        let body = serde_json::to_vec(&json!([
            {"type": "file", "path": "https://host/datasets/x/resolve/main/train/1.ndjson", "size": 100}
        ]))
        .unwrap();
        let server = spawn_one_shot_http(body);
        let base_url = server.url().to_string();

        config.parquet_endpoint = base_url;
        let client = test_http_client();
        let (candidates, sizes) =
            HuggingFaceRowSource::list_remote_candidates(&client, &config).unwrap();

        assert_eq!(candidates.len(), 1);
        assert!(!sizes.is_empty());
        assert!(candidates[0].ends_with("/1.ndjson"));
    }

    #[test]
    fn list_remote_candidates_scopes_tree_to_config_name() {
        // Verify that the tree endpoint URL is scoped to the config_name
        // subdirectory so we don't pull files from other configs
        // (e.g. wikimedia/wikipedia/20231101.en should not list .fr files).
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.config_name = "20231101.en".to_string();

        // Spawn a mock server that records the request path.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{addr}");
        let recorded_path: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let recorded_path_clone = Arc::clone(&recorded_path);
        let handle = std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 4096];
                let n = stream.read(&mut buf).unwrap_or(0);
                let request = String::from_utf8_lossy(&buf[..n]);
                let path = request
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .map(|s| s.to_string());
                *recorded_path_clone.lock().unwrap() = path;
                let body = br#"[]"#;
                let headers = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(headers.as_bytes());
                let _ = stream.write_all(body);
            }
        });

        config.parquet_endpoint = base_url;
        let client = test_http_client();
        let _ =
            HuggingFaceRowSource::list_remote_candidates_from_parquet_manifest(&client, &config);
        handle.join().unwrap();

        let path = recorded_path.lock().unwrap();
        let path = path.as_deref().expect("no request recorded");
        assert!(
            path.contains("/tree/main/20231101.en"),
            "tree endpoint URL must be scoped to config_name; got: {path}"
        );
        assert!(
            !path.contains("/tree/main?"),
            "tree endpoint must NOT use root path when config_name is set; got: {path}"
        );
    }

    #[test]
    fn list_remote_candidates_uses_root_tree_for_default_config() {
        // When config_name is "default", the tree endpoint should hit the repo root.
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.config_name = "default".to_string();

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{addr}");
        let recorded_path: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let recorded_path_clone = Arc::clone(&recorded_path);
        let handle = std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 4096];
                let n = stream.read(&mut buf).unwrap_or(0);
                let request = String::from_utf8_lossy(&buf[..n]);
                let path = request
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .map(|s| s.to_string());
                *recorded_path_clone.lock().unwrap() = path;
                let body = br#"[]"#;
                let headers = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(headers.as_bytes());
                let _ = stream.write_all(body);
            }
        });

        config.parquet_endpoint = base_url;
        let client = test_http_client();
        let _ =
            HuggingFaceRowSource::list_remote_candidates_from_parquet_manifest(&client, &config);
        handle.join().unwrap();

        let path = recorded_path.lock().unwrap();
        let path = path.as_deref().expect("no request recorded");
        assert!(
            path.contains("/tree/main?recursive=true"),
            "default config must hit repo root; got: {path}"
        );
    }

    #[test]
    #[serial(global_state)]
    fn list_remote_candidates_with_runtime_returns_manifest_candidates() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        let runtime = build_http_runtime(&config).unwrap();
        let body = serde_json::to_vec(&json!([
            {"type": "file", "path": "https://host/datasets/x/resolve/main/train/2.ndjson", "size": 100}
        ]))
        .unwrap();
        let server = spawn_one_shot_http(body);
        let base_url = server.url().to_string();

        config.parquet_endpoint = base_url;
        let client = test_http_client();
        let (candidates, sizes) =
            list_remote_candidates_with_runtime(&client, &config, Some(&runtime)).unwrap();

        assert_eq!(candidates.len(), 1);
        assert!(!sizes.is_empty());
        assert!(candidates[0].ends_with("/2.ndjson"));
    }

    #[test]
    #[serial(global_state)]
    fn list_remote_candidates_does_not_fall_back_when_all_manifest_shards_cached() {
        // Regression test: list_remote_candidates must return the full manifest
        // candidate list when a parquet manifest exists, regardless of whether all
        // shards are already cached on disk.
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());

        // Pre-create the .simdr store target so the manifest entry is "fully cached".
        let shard_url = "https://host/datasets/org/ds/resolve/main/train/part-000.ndjson";
        let candidate = format!("{HF_REMOTE_URL_PREFIX}{shard_url}");
        let target = candidate_target_path(&config, &candidate);
        let store_target = shard_store_path_for(&target);
        fs::create_dir_all(store_target.parent().unwrap()).unwrap();
        fs::write(&store_target, b"cached").unwrap();

        let body = serde_json::to_vec(&json!([
            {"type": "file", "path": shard_url, "size": 100}
        ]))
        .unwrap();
        let server = spawn_one_shot_http(body);
        let base_url = server.url().to_string();

        // Must return the full manifest candidate list without falling through to hf-hub.
        config.parquet_endpoint = base_url;
        let client = test_http_client();
        let (candidates, sizes) =
            HuggingFaceRowSource::list_remote_candidates(&client, &config).unwrap();

        assert_eq!(
            candidates.len(),
            1,
            "fully-cached shard must still appear in candidates (cache ≠ order)"
        );
        assert!(!sizes.is_empty());
        assert!(candidates[0].ends_with(shard_url));
    }

    #[test]
    #[serial(global_state)]
    fn list_remote_candidates_from_parquet_manifest_errors_when_endpoint_unreachable() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.parquet_endpoint = TEST_UNREACHABLE_URL.to_string();

        let client = test_http_client();
        let result =
            HuggingFaceRowSource::list_remote_candidates_from_parquet_manifest(&client, &config);
        assert!(result.is_err());
    }

    // --- datasets viewer disabled (501) scenario ---
    //
    // Some HF datasets have the datasets viewer disabled.  In that case the
    // /size, /info, and /parquet datasets-server endpoints all return HTTP 501
    // with {"error":"Not supported: dataset viewer is disabled ..."}.
    //
    // The expected behaviour:
    //   * /size   → fetch_global_row_count returns Ok(None), not Err.
    //   * /info   → fetch_classlabel_maps returns an empty map, not an error.
    //   * /parquet → list_remote_candidates_from_parquet_manifest returns Err,
    //                which causes list_remote_candidates_with_runtime to fall
    //                through to the hf-hub repository listing path.

    #[test]
    #[serial(global_state)]
    fn list_remote_candidates_from_parquet_manifest_errors_on_501() {
        // A 501 from /parquet causes the manifest path to return Err, which
        // propagates to the caller (no fallback path).
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        let body =
            br#"{"error":"Not supported: dataset viewer is disabled in org/dataset configuration."}"#
                .to_vec();
        let server = TestHttpServer::new(501, body);
        let base_url = server.url().to_string();

        config.parquet_endpoint = base_url;
        let client = test_http_client();
        let result =
            HuggingFaceRowSource::list_remote_candidates_from_parquet_manifest(&client, &config);

        assert!(
            result.is_err(),
            "expected Err from 501 /parquet response, got {result:?}"
        );
    }

    #[test]
    fn list_remote_candidates_returns_error_on_invalid_json() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        let server = spawn_one_shot_http(b"this is not json at all".to_vec());
        config.parquet_endpoint = server.url().to_string();

        let client = test_http_client();
        let result = HuggingFaceRowSource::list_remote_candidates(&client, &config);
        assert!(result.is_err(), "invalid JSON should return error");
    }

    #[test]
    fn list_remote_candidates_returns_error_on_non_success() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        let server = TestHttpServer::new(503, b"Service Unavailable".to_vec());
        config.parquet_endpoint = server.url().to_string();

        let client = test_http_client();
        let result = HuggingFaceRowSource::list_remote_candidates(&client, &config);
        assert!(result.is_err(), "503 response should return error");
    }

    #[test]
    fn download_and_materialize_shard_url_short_circuits_when_cached_complete() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let candidate = "url::https://host/datasets/org/ds/resolve/main/train/ok.ndjson";
        let target = candidate_target_path(&config, candidate);
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::write(&target, b"ok").unwrap();

        let client = test_http_client();
        let resolved = HuggingFaceRowSource::download_and_materialize_shard(
            &client,
            &config,
            candidate,
            Some(2),
            "shard 1/1",
        )
        .unwrap();
        assert_eq!(resolved, target);
    }

    #[test]
    fn download_and_materialize_shard_url_replaces_stale_part_file() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let payload = b"{\"text\":\"a\"}\n".to_vec();
        let server = spawn_one_shot_http(payload.clone());
        let base_url = server.url().to_string();
        let candidate = format!("url::{base_url}/datasets/org/ds/resolve/main/train/part-x.ndjson");
        let target = candidate_target_path(&config, &candidate);
        let temp_target = target.with_extension("part");
        fs::create_dir_all(temp_target.parent().unwrap()).unwrap();
        fs::write(&temp_target, b"stale").unwrap();

        let client = test_http_client();
        let out = HuggingFaceRowSource::download_and_materialize_shard(
            &client,
            &config,
            &candidate,
            None,
            "shard 1/1",
        )
        .unwrap();

        // Transient formats (ndjson) are staged to a temp path, not the cache target.
        assert_ne!(out, target, "transient download should go to temp path");
        assert!(out.exists(), "temp file should exist");
        assert_eq!(fs::read(&out).unwrap(), payload);
    }

    #[test]
    fn download_and_materialize_shard_downloads_url_candidate() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let payload = b"{\"text\":\"a\"}\n{\"text\":\"b\"}\n".to_vec();
        let server = spawn_one_shot_http(payload.clone());
        let base_url = server.url().to_string();
        let candidate =
            format!("url::{base_url}/datasets/org/ds/resolve/main/train/part-000.ndjson");

        let client = test_http_client();
        let target = HuggingFaceRowSource::download_and_materialize_shard(
            &client,
            &config,
            &candidate,
            None,
            "shard 1/1",
        )
        .unwrap();

        assert!(target.exists());
        assert_eq!(fs::read(&target).unwrap(), payload);
    }

    #[test]
    fn download_and_materialize_shard_replaces_incomplete_existing_target() {
        // Suppress the expected WARN "replacing incomplete shard before retry" that fires
        // when an existing target file's size does not match the expected manifest size.
        // Detecting and replacing the stale file is what this test asserts on.
        let _quiet = tracing::subscriber::set_default(
            tracing_subscriber::fmt()
                .with_max_level(tracing::Level::ERROR)
                .finish(),
        );
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let payload = b"{\"text\":\"a\"}\n".to_vec();
        let server = spawn_one_shot_http(payload.clone());
        let base_url = server.url().to_string();
        let candidate =
            format!("url::{base_url}/datasets/org/ds/resolve/main/train/part-009.ndjson");

        let target = candidate_target_path(&config, &candidate);
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::write(&target, b"bad").unwrap();

        let client = test_http_client();
        let refreshed = HuggingFaceRowSource::download_and_materialize_shard(
            &client,
            &config,
            &candidate,
            Some(payload.len() as u64),
            "shard 1/1",
        )
        .unwrap();

        // Transient formats (ndjson) are staged to a temp path, not the cache target.
        assert_ne!(
            refreshed, target,
            "transient download should go to temp path"
        );
        assert!(refreshed.exists(), "temp file should exist");
        assert_eq!(fs::read(&refreshed).unwrap(), payload);
    }

    #[test]
    fn download_shard_rejects_path_traversal_double_dot() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let client = test_http_client();

        let remote_path = "url::http://evil.com/../../etc/passwd";
        let result = HuggingFaceRowSource::download_and_materialize_shard(
            &client,
            &config,
            remote_path,
            None,
            "traversal-test",
        );
        assert!(result.is_err(), "path traversal should be rejected");
        match result {
            Err(SamplerError::SourceUnavailable { reason, .. }) => {
                assert!(
                    reason.contains("traversal"),
                    "error should mention traversal, got: {reason}"
                );
            }
            other => panic!("expected SourceUnavailable error, got: {:?}", other),
        }
    }

    #[test]
    fn download_shard_rejects_path_traversal_in_full_url() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let client = test_http_client();

        let remote_path = "url::https://host/datasets/org/ds/resolve/main/../../../etc/passwd";
        let result = HuggingFaceRowSource::download_and_materialize_shard(
            &client,
            &config,
            remote_path,
            None,
            "traversal-test-2",
        );
        assert!(
            result.is_err(),
            "path traversal in full URL should be rejected"
        );
    }

    #[test]
    fn download_shard_store_already_exists_returns_cached() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let client = test_http_client();

        let remote_path =
            "url::http://mock.example.com/datasets/org/ds/resolve/main/train/shard.ndjson";
        let store_path = crate::shard_indexing::candidate_store_path(&config, remote_path);
        fs::create_dir_all(store_path.parent().unwrap()).unwrap();
        write_simdr_fixture(&store_path, &[("r1", "cached")]);

        let result = HuggingFaceRowSource::download_and_materialize_shard(
            &client,
            &config,
            remote_path,
            None,
            "cache-test",
        );
        let path = result.expect("should return Ok with store path");
        assert_eq!(path, store_path, "should return the existing store path");
    }

    #[test]
    fn download_shard_rejects_bare_relative_traversal() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let client = test_http_client();

        let remote_path = "../etc/passwd";
        let result = HuggingFaceRowSource::download_and_materialize_shard(
            &client,
            &config,
            remote_path,
            None,
            "bare-traversal",
        );
        assert!(
            result.is_err(),
            "bare relative traversal should be rejected"
        );
    }

    #[test]
    fn download_shard_rejects_url_encoded_traversal() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let client = test_http_client();

        let remote_path = "url::http://evil.com/datasets/%2e%2e/%2e%2e/etc/passwd";
        let result = HuggingFaceRowSource::download_and_materialize_shard(
            &client,
            &config,
            remote_path,
            None,
            "encoded-traversal",
        );
        assert!(result.is_err(), "URL-encoded traversal should be rejected");
    }

    #[test]
    fn download_shard_rejects_http_traversal_without_url_prefix() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let client = test_http_client();

        let remote_path = "http://evil.com/../../../etc/passwd";
        let result = HuggingFaceRowSource::download_and_materialize_shard(
            &client,
            &config,
            remote_path,
            None,
            "http-traversal",
        );
        assert!(
            result.is_err(),
            "HTTP URL with traversal should be rejected"
        );
    }

    #[test]
    #[serial(global_state)]
    fn fetch_remote_size_with_runtime_returns_content_length() {
        // A mock HTTP server that responds to HEAD with Content-Length.
        let payload = b"this is the shard content".to_vec();
        let server = TestHttpServer::new(200, payload.clone());
        let base_url = server.url().to_string();

        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.hf_token = None;

        let client = test_http_client();
        let runtime = build_http_runtime(&config).unwrap();
        let size = fetch_remote_size_with_runtime(&client, &config, &base_url, &runtime).unwrap();
        // Content-Length should match the payload size.
        assert_eq!(size, Some(payload.len() as u64));
    }

    #[test]
    #[serial(global_state)]
    fn fetch_remote_size_with_runtime_returns_none_on_non_success() {
        // A mock server returning 404 — HEAD should return Ok(None).
        let server = TestHttpServer::new(404, b"Not Found".to_vec());
        let base_url = server.url().to_string();

        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.hf_token = None;

        let client = test_http_client();
        let runtime = build_http_runtime(&config).unwrap();
        let size = fetch_remote_size_with_runtime(&client, &config, &base_url, &runtime).unwrap();
        assert_eq!(size, None, "non-2xx response should yield None");
    }

    #[test]
    fn fetch_remote_size_network_failure_returns_error() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let client = test_http_client();
        let runtime = build_http_runtime(&config).unwrap();

        let result = fetch_remote_size_with_runtime(
            &client,
            &config,
            &format!("{TEST_UNREACHABLE_URL}/shard.parquet"),
            &runtime,
        );
        assert!(result.is_err(), "HEAD to unreachable URL should fail");
    }
}
