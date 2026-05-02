/// Prefix added to remote URL shard identifiers to distinguish them from local paths.
pub const REMOTE_URL_PREFIX: &str = "url::";
/// Extra row-index headroom above currently materialized rows exposed via `len_hint`.
///
/// This is not a file count. It lets sampling look slightly past the local row
/// frontier so lazy remote expansion can continue without jumping to the full
/// global row domain at once.
/// Multiplies the sampler ingestion base (`SamplerConfig.ingestion_max_records`)
/// to compute `len_hint` expansion headroom rows.
pub const REMOTE_EXPANSION_HEADROOM_MULTIPLIER: usize = 4;
/// Number of initial remote shards to materialize when bootstrapping an empty
/// local snapshot before regular lazy expansion.
pub const REMOTE_BOOTSTRAP_SHARDS: usize = 1;
/// Multiplies the source `refresh` limit passed by `IngestionManager`
/// (`step.unwrap_or(max_records)`) to set this source's internal row-read
/// batch target for each refresh pass.
pub const HUGGINGFACE_REFRESH_BATCH_MULTIPLIER: usize = 8;
/// Version tag for persisted shard-sequence metadata payloads.
pub const SHARD_SEQUENCE_STATE_VERSION: u32 = 1;
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
pub const ALL_SPLITS_DIR: &str = "_all";
/// Sub-directory under `snapshot_dir` that holds manifest-cached remote shard files.
pub const PARQUET_MANIFEST_DIR: &str = "_parquet_manifest";
/// Path separator component used to extract a local path suffix from HF CDN resolve URLs.
pub const HF_RESOLVE_URL_SEPARATOR: &str = "/resolve/";
/// Fallback relative path used when a CDN resolve URL cannot yield a valid suffix.
pub const HF_RESOLVE_UNKNOWN_FALLBACK_PATH: &str = "parquet/unknown.parquet";
/// Domain tag mixed into the shard-candidate permutation seed hash for forward isolation.
pub const HF_SHARD_CANDIDATE_SEED_TAG: &str = "hf_shard_candidate_sequence_v1";
/// JSON field key for the parquet files array in the datasets-server manifest response.
pub const HF_JSON_KEY_PARQUET_FILES: &str = "parquet_files";
/// JSON field key for the shard URL within a parquet file manifest entry.
pub const HF_JSON_KEY_URL: &str = "url";
/// JSON field key for shard byte sizes (parquet entry) and the size response root object.
pub const HF_JSON_KEY_SIZE: &str = "size";
/// JSON field key for the splits array in size response objects.
pub const HF_JSON_KEY_SPLITS: &str = "splits";
/// JSON field key for the configs array in size response objects.
pub const HF_JSON_KEY_CONFIGS: &str = "configs";
/// JSON field key for config name in size response entries (primary form); also used as
/// the HTTP query parameter name sent to the datasets-server API.
pub const HF_JSON_KEY_CONFIG: &str = "config";
/// JSON field key for config name in size response entries (alternate form).
pub const HF_JSON_KEY_CONFIG_NAME: &str = "config_name";
/// JSON field key for split name in size response entries (primary form); also used as
/// the HTTP query parameter name sent to the datasets-server API.
pub const HF_JSON_KEY_SPLIT: &str = "split";
/// JSON field key for split name in size response entries (alternate form).
pub const HF_JSON_KEY_SPLIT_NAME: &str = "name";
/// JSON field key for the row count in size response entries.
pub const HF_JSON_KEY_NUM_ROWS: &str = "num_rows";
/// JSON field key for dataset-level size metrics in size response; also the HTTP query
/// parameter name for the dataset identifier sent to the datasets-server API.
pub const HF_JSON_KEY_DATASET: &str = "dataset";
/// JSON key for the top-level dataset info object in the /info response.
pub const HF_JSON_KEY_DATASET_INFO: &str = "dataset_info";
/// JSON key for the features map within the dataset info object.
pub const HF_JSON_KEY_FEATURES: &str = "features";
/// JSON key for the feature type discriminator within a feature entry.
pub const HF_JSON_KEY_FEATURE_TYPE: &str = "_type";
/// JSON key for the label names array within a ClassLabel feature entry.
pub const HF_JSON_KEY_LABEL_NAMES: &str = "names";
/// Feature type string that identifies a ClassLabel column.
pub const HF_CLASSLABEL_TYPE: &str = "ClassLabel";
/// Default base URL for the datasets-server parquet-manifest endpoint.
///
/// Can be overridden at runtime with the `TRIPLETS_HF_PARQUET_ENDPOINT`
/// environment variable (useful for test doubles or on-premises deployments).
pub const HF_PARQUET_DEFAULT_ENDPOINT: &str = "https://datasets-server.huggingface.co/parquet";

/// Default base URL for the datasets-server size endpoint.
///
/// Can be overridden at runtime with the `TRIPLETS_HF_SIZE_ENDPOINT`
/// environment variable.
pub const HF_SIZE_DEFAULT_ENDPOINT: &str = "https://datasets-server.huggingface.co/size";

/// Default base URL for the datasets-server info endpoint.
///
/// Can be overridden at runtime with the `TRIPLETS_HF_INFO_ENDPOINT`
/// environment variable.
pub const HF_INFO_DEFAULT_ENDPOINT: &str = "https://datasets-server.huggingface.co/info";

/// Endpoint used to validate a Hugging Face API token.
///
/// A GET to this URL with a valid `Authorization: Bearer <token>` header
/// returns `200 OK`; an invalid or expired token yields `401 Unauthorized`.
/// Used by `HuggingFaceRowSource::new()` to fail fast when an `HF_TOKEN`
/// is provided but cannot authenticate.
pub const HF_WHOAMI_ENDPOINT: &str = "https://huggingface.co/api/whoami-v2";
