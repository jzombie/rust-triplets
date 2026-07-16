/// Prefix added to remote URL shard identifiers to distinguish them from local paths.
/// Hard cap for local manifest-shard cache bytes (32 GiB).
/// Enforced by cache-manager policy application on manifest cache roots.
pub const HF_LOCAL_DISK_CAP_BYTES: u64 = 32 * 1024 * 1024 * 1024;

/// Prefix added to remote URL shard identifiers to distinguish them from local paths.
pub const HF_REMOTE_URL_PREFIX: &str = "url::";
/// Extra row-index headroom above currently materialized rows exposed via `len_hint`.
///
/// This is not a file count. It lets sampling look slightly past the local row
/// frontier so lazy remote expansion can continue without jumping to the full
/// global row domain at once.
/// Multiplies the sampler ingestion base (`SamplerConfig.ingestion_max_records`)
/// to compute `len_hint` expansion headroom rows.
pub const HF_REMOTE_EXPANSION_HEADROOM_MULTIPLIER: usize = 4;
/// Number of initial remote shards to materialize when bootstrapping an empty
/// local snapshot before regular lazy expansion.
pub const HF_REMOTE_BOOTSTRAP_SHARDS: usize = 1;
/// Multiplies the source `refresh` limit passed by `IngestionManager`
/// (`step.unwrap_or(max_records)`) to set this source's internal row-read
/// batch target for each refresh pass.
pub const HF_REFRESH_BATCH_MULTIPLIER: usize = 8;

/// Extension used by persisted per-shard row-store files.
pub const HF_SHARD_STORE_EXTENSION: &str = "simdr";
/// Key prefix for individual row payload entries in shard row stores.
pub const HF_SHARD_STORE_ROW_PREFIX: &[u8] = b"rowv1|";
/// Metadata key storing row counts in shard row stores.
pub const HF_SHARD_STORE_META_ROWS_KEY: &[u8] = b"meta|rows";
/// Metadata key storing the original source shard size from the remote
/// manifest at download time.  Compared against the current manifest on
/// subsequent cycles to detect out-of-date shards.
pub const HF_SHARD_STORE_SOURCE_SIZE_KEY: &[u8] = b"meta|source_size";
/// Directory segment used when no split is specified (all-splits mode).
/// Must not collide with any real HF split name; HF split names never start with `_`.
pub const HF_ALL_SPLITS_DIR: &str = "_all";
/// Sub-directory under `snapshot_dir` that holds manifest-cached remote shard files.
pub const HF_PARQUET_MANIFEST_DIR: &str = "_parquet_manifest";
/// Path separator component used to extract a local path suffix from HF CDN resolve URLs.
pub const HF_RESOLVE_URL_SEPARATOR: &str = "/resolve/";
/// Domain tag mixed into the shard-candidate permutation seed hash for forward isolation.
pub const HF_SHARD_CANDIDATE_SEED_TAG: &str = "hf_shard_candidate_sequence_v1";
/// Root base URL for the Hugging Face website.
pub const HF_BASE_URL: &str = "https://huggingface.co";

/// Base URL for the Hugging Face datasets-resolve endpoint.
///
/// Full download URLs are constructed as
/// `{base}/{dataset}/resolve/main/{relative_path}`.
pub const HF_DATASETS_BASE_URL: &str = "https://huggingface.co/datasets";

// ---------------------------------------------------------------------------
// ── Shared tokio runtime ────────────────────────────────────────────────────

/// Number of worker threads for the process-wide shared tokio multi-threaded
/// runtime used by Hugging Face HTTP operations.
pub const HF_SHARED_RUNTIME_WORKER_THREADS: usize = 2;

// ---------------------------------------------------------------------------
// ── HTTP client timeouts ────────────────────────────────────────────────────

/// TCP/TLS connect timeout for the shared `reqwest::Client` (seconds).
pub const HF_HTTP_CONNECT_TIMEOUT_SECS: u64 = 15;

/// Total request timeout for the shared `reqwest::Client` (seconds).
/// Covers the entire download of a single shard.
pub const HF_HTTP_REQUEST_TIMEOUT_SECS: u64 = 300;

// ---------------------------------------------------------------------------
// ── Throttle / backoff policy ───────────────────────────────────────────────

/// Base delay (milliseconds) before retrying a rate-limited or failed request.
#[allow(dead_code)]
pub const HF_THROTTLE_BASE_DELAY_MS: u64 = 200;

/// Maximum random jitter (milliseconds) added to the base delay to spread
/// retries across clients and avoid thundering-herd synchronization.
#[allow(dead_code)]
pub const HF_THROTTLE_ADAPTIVE_JITTER_MS: u64 = 100;

/// Maximum number of concurrent in-flight requests per source. Shard
/// downloads are serialised per source so this primarily limits overlapping
/// API calls (parquet manifest, size, info) during startup.
#[allow(dead_code)]
pub const HF_THROTTLE_MAX_CONCURRENT: usize = 4;

/// Maximum number of retry attempts for a single request before surfacing
/// the error to the caller.
#[allow(dead_code)]
pub const HF_THROTTLE_MAX_RETRIES: usize = 3;

/// Base URL for the Hub API tree endpoint.
///
/// Lists files in a dataset repository: `{base}/{dataset}/tree/main`.
pub const HF_PARQUET_DEFAULT_ENDPOINT: &str = "https://huggingface.co/api/datasets";

/// Public Hugging Face dataset used as a fallback in live integration tests.
///
/// When `TRIPLETS_HF_TOKEN_TEST_DATASET` is not set (or is empty), live tests
/// fall back to this public dataset.
pub const HF_PUBLIC_TEST_DATASET: &str = "TimKoornstra/financial-tweets-sentiment";

/// Endpoint used to validate a Hugging Face API token.
///
/// A GET to this URL with a valid `Authorization: Bearer <token>` header
/// returns `200 OK`; an invalid or expired token yields `401 Unauthorized`.
/// Used by `HuggingFaceRowSource::new()` to fail fast when an `HF_TOKEN`
/// is provided but cannot authenticate.
pub const HF_WHOAMI_DEFAULT_ENDPOINT: &str = "https://huggingface.co/api/whoami-v2";

// ---------------------------------------------------------------------------
// ── Environment variable name constants ─────────────────────────────────────

/// Hugging Face API token for authenticating with private datasets.
pub const ENV_TRIPLETS_HF_TOKEN: &str = "HF_TOKEN";
/// Dataset repo used by the live private-dataset integration test.
pub const ENV_TRIPLETS_HF_TOKEN_TEST_DATASET: &str = "TRIPLETS_HF_TOKEN_TEST_DATASET";
/// Overrides the Hugging Face whoami endpoint URL used for token validation.
pub const ENV_TRIPLETS_HF_WHOAMI_ENDPOINT: &str = "TRIPLETS_HF_WHOAMI_ENDPOINT";
/// Managed cache group for Hugging Face snapshot-backed sources.
pub const HF_GROUP: &str = "triplets/huggingface";

/// Prefix for temporary download files created by `allocate_temp_download_path`.
/// Used to identify temp files that can be safely deleted after transcoding.
pub const HF_TEMP_DOWNLOAD_PREFIX: &str = "triplets_hf_";

/// Default file extension when the original filename has no extension.
/// Used as fallback when building temp download paths.
pub const HF_TEMP_DEFAULT_EXTENSION: &str = "part";

// ---------------------------------------------------------------------------
// ── Source constants ─────────────────────────────────────

pub(crate) const HF_SOURCE_KEY_ANCHOR: &str = "anchor";
pub(crate) const HF_SOURCE_KEY_POSITIVE: &str = "positive";
pub(crate) const HF_SOURCE_KEY_NEGATIVE: &str = "negative";
pub(crate) const HF_SOURCE_KEY_CONTEXT: &str = "context";
pub(crate) const HF_SOURCE_KEY_TEXT: &str = "text";
pub(crate) const HF_SOURCE_KEY_TEXT_COLUMNS: &str = "text_columns";
pub(crate) const HF_SOURCE_KEY_TRUST: &str = "trust";
pub(crate) const HF_SOURCE_KEY_WEIGHT: &str = "weight";
pub(crate) const HF_SOURCE_KEY_SOURCE_ID: &str = "source_id";

// ---------------------------------------------------------------------------
// ── Recipe constants ─────────────────────────────────────

/// Default HF text-columns-mode SimCSE-style recipe name.
pub const HF_RECIPE_TEXT_SIMCSE_WRONG_ARTICLE: &str = "huggingface_text_simcse_wrong_article";
pub(crate) const HF_RECIPE_ANCHOR_CONTEXT_WRONG_ARTICLE: &str =
    "huggingface_anchor_context_wrong_article";
pub(crate) const HF_RECIPE_ANCHOR_ANCHOR_WRONG_ARTICLE: &str =
    "huggingface_anchor_anchor_wrong_article";
