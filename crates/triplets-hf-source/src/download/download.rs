use crate::config::HuggingFaceRowsConfig;
use crate::constants::{
    ENV_TRIPLETS_HF_WHOAMI_ENDPOINT, HF_DATASETS_BASE_URL, HF_HTTP_CONNECT_TIMEOUT_SECS,
    HF_HTTP_REQUEST_TIMEOUT_SECS, HF_MAX_PAGINATION_PAGES, HF_PARQUET_MANIFEST_DIR,
    HF_REMOTE_URL_PREFIX, HF_RESOLVE_URL_SEPARATOR, HF_SHARD_CANDIDATE_SEED_TAG,
    HF_SHARED_RUNTIME_WORKER_THREADS, HF_TEMP_DEFAULT_EXTENSION, HF_TEMP_DOWNLOAD_PREFIX,
    HF_WHOAMI_DEFAULT_ENDPOINT,
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

/// Extract the next pagination URL from `Link` header(s).
/// Uses `get_all()` to handle split Link headers across multiple HTTP header lines.
/// Resolves relative URIs against `request_url` to preserve custom endpoints.
pub(crate) fn extract_next_link_url(
    headers: &reqwest::header::HeaderMap,
    request_url: &str,
) -> Option<String> {
    for header in headers.get_all("link") {
        let Ok(link) = header.to_str() else {
            continue;
        };
        for part in link.split(',') {
            let lower = part.to_ascii_lowercase();
            if !(lower.contains("rel=\"next\"") || lower.contains("rel=next")) {
                continue;
            }
            // Use closure to isolate ? operators — malformed segment continues loop
            let parsed = (|| {
                let url_str = part.split('<').nth(1)?.split('>').next()?.trim();
                let base = reqwest::Url::parse(request_url).ok()?;
                let resolved = base.join(url_str).ok()?;
                Some(resolved.to_string())
            })();
            if parsed.is_some() {
                return parsed;
            }
        }
    }
    None
}

/// Fetch a single page and extract the next pagination URL from the Link header.
async fn fetch_page_with_next_url(
    http_client: &ClientWithMiddleware,
    source_id: &str,
    url: &str,
    endpoint_label: &str,
) -> Result<(String, Option<String>), SamplerError> {
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
            reason: format!("{endpoint_label} returned non-success: {err}"),
        })?;
    let next_url = extract_next_link_url(response.headers(), url);
    let body = response
        .text()
        .await
        .map_err(|err| SamplerError::SourceUnavailable {
            source_id: source_id.to_string(),
            reason: format!("failed reading {endpoint_label} body: {err}"),
        })?;
    Ok((body, next_url))
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
    let initial_url = if config.config_name.is_empty() || config.config_name == "default" {
        format!("{base}/{}/tree/main?recursive=true", config.dataset_name)
    } else {
        format!(
            "{base}/{}/tree/main/{}?recursive=true",
            config.dataset_name, config.config_name
        )
    };

    info!(
        "[triplets:hf] reading Hub API tree for dataset {}",
        config.dataset_name
    );

    let mut all_candidates = Vec::new();
    let mut all_sizes = HashMap::new();
    let mut total_matched = 0usize;
    let mut current_url = Some(initial_url);
    let mut page_count = 0usize;

    while let Some(url) = current_url {
        if page_count >= HF_MAX_PAGINATION_PAGES {
            return Err(SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!(
                    "pagination limit ({HF_MAX_PAGINATION_PAGES}) reached for dataset='{}'",
                    config.dataset_name
                ),
            });
        }
        let (body, next) = block_on_http_with_runtime(
            runtime,
            config,
            fetch_page_with_next_url(http_client, &config.source_id, &url, "Hub tree endpoint"),
        )?;
        let (page_candidates, page_sizes, matched) =
            parse_parquet_manifest_response(config, &body)?;
        all_candidates.extend(page_candidates);
        all_sizes.extend(page_sizes);
        total_matched += matched;
        current_url = next;
        page_count += 1;
    }

    // CRITICAL: sort after accumulation (required by build_candidate_order)
    all_candidates.sort();
    all_candidates.dedup();
    all_sizes.retain(|k, _| all_candidates.binary_search(k).is_ok());

    Ok((all_candidates, all_sizes, total_matched))
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
            !existing.contains(&crate::shard_indexer::candidate_store_path(
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
