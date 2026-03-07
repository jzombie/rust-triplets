use crate::metadata::MetadataKey;
use crate::splits::SplitLabel;

/// Environment variable names read at runtime to override default behaviour.
///
/// Keeping the strings here ensures every call site references the same name
/// and makes the full set of supported overrides easy to discover.
pub mod env_vars {
    /// Overrides the Hugging Face datasets-server parquet-manifest endpoint URL.
    ///
    /// When set to a non-blank value, `parquet_manifest_endpoint()` returns this
    /// value instead of the default `https://datasets-server.huggingface.co/parquet`.
    /// Intended for test doubles and air-gapped environments.
    #[cfg(test)]
    pub const TRIPLETS_HF_PARQUET_ENDPOINT: &str = "TRIPLETS_HF_PARQUET_ENDPOINT";

    /// Overrides the Hugging Face datasets-server size endpoint URL.
    ///
    /// When set to a non-blank value, `size_endpoint()` returns this value instead
    /// of the default `https://datasets-server.huggingface.co/size`.
    /// Intended for test doubles and air-gapped environments.
    #[cfg(test)]
    pub const TRIPLETS_HF_SIZE_ENDPOINT: &str = "TRIPLETS_HF_SIZE_ENDPOINT";
}

/// Constants used by capacity estimation heuristics.
pub mod heuristics {
    /// Effective positive examples sampled per anchor during bounded estimates.
    pub const EFFECTIVE_POSITIVES_PER_ANCHOR: u128 = 1;
    /// Effective negative examples sampled per anchor during bounded estimates.
    pub const EFFECTIVE_NEGATIVES_PER_ANCHOR: u128 = 4;
}

/// Constants used by metadata key encoding and canonical fields.
pub mod metadata {
    use super::MetadataKey;

    /// Separator used for serialized metadata entries (for example `date=2025-01-01`).
    pub const METADATA_DELIMITER: &str = "=";
    /// Canonical metadata field key used for publication dates.
    pub const META_FIELD_DATE: MetadataKey = MetadataKey::new("date");
}

/// Constants used by sampler runtime behavior and labeling.
pub mod sampler {
    /// Maximum number of forced refresh retries after an exhausted sampling pass.
    pub const EXHAUSTION_RETRY_LIMIT: usize = 2;
    /// Retry cap when trying to produce a valid anchor/positive pair from the same selector.
    pub const SAME_SELECTOR_PAIR_RETRY_LIMIT: usize = 8;
    /// Offset mixed into epoch RNG seed derivation for deterministic variation.
    pub const EPOCH_SEED_OFFSET: u64 = 0xB4C3_5EED;
    /// Label used for triplet recipe weight maps.
    pub const RECIPE_LABEL_TRIPLETS: &str = "triplet_recipes";
    /// Label used for text recipe weight maps.
    pub const RECIPE_LABEL_TEXT: &str = "text_recipes";
    /// Label identifying anchor-role sections in role-weight maps.
    pub const ROLE_LABEL_ANCHOR: &str = "anchor";
    /// Label identifying context-role sections in role-weight maps.
    pub const ROLE_LABEL_CONTEXT: &str = "context";
    /// Synthetic source id used in prefetcher failure reporting.
    pub const PREFETCHER_SOURCE_ID: &str = "prefetcher";
    /// Failure reason emitted when a prefetcher worker stops unexpectedly.
    pub const PREFETCHER_STOPPED_REASON: &str = "prefetcher stopped";
    /// Negative-pair reason tag for mismatched publication dates.
    pub const NEG_REASON_WRONG_DATE: &str = "wrong_publication_date";
    /// Negative-pair reason tag for mismatched article associations.
    pub const NEG_REASON_WRONG_ARTICLE: &str = "wrong_article";
    /// Negative-pair reason tag for mismatched question/answer pairings.
    pub const NEG_REASON_WRONG_QA: &str = "wrong_qa_pairing";
    /// Denominator used for the anchor/positive swap coin-flip (swap when `rng & mask == 0`).
    ///
    /// A value of `1` means the least-significant bit is tested, giving a uniform 50 % swap
    /// rate. This eliminates positional shortcuts — e.g. a model cannot learn to always treat
    /// the first slot as the "short" anchor — which is especially important for InfoNCE and
    /// similar contrastive objectives.
    pub const ANCHOR_POSITIVE_SWAP_MASK: u64 = 1;
}

/// Constants used by sampler test fixtures and determinism snapshots.
#[cfg(test)]
pub mod sampler_tests {
    /// Primary source id used by sampler unit tests.
    pub const PRIMARY_SOURCE_ID: &str = "source_a";
    /// Secondary source id used by sampler unit tests.
    pub const SECONDARY_SOURCE_ID: &str = "source_b";

    /// FNV-1a 64-bit offset basis used in snapshot hashing tests.
    pub const FNV1A64_OFFSET: u64 = 0xcbf29ce484222325;
    /// FNV-1a 64-bit prime used in snapshot hashing tests.
    pub const FNV1A64_PRIME: u64 = 0x100000001b3;

    /// Number of batches sampled for deterministic sequence hash assertions.
    pub const FULL_SEQUENCE_LEN: usize = 45;
    /// Expected hash for deterministic text batch sequence.
    pub const TEXT_BATCH_SEQUENCE_HASH: u64 = 16700524736973776041;
    /// Expected hash for deterministic triplet batch sequence.
    pub const TRIPLET_BATCH_SEQUENCE_HASH: u64 = 14618375830351864347;
    /// Expected hash for deterministic pair batch sequence.
    pub const PAIR_BATCH_SEQUENCE_HASH: u64 = 13133711470950174124;
    /// Expected hash for deterministic prefetch text batch sequence.
    pub const PREFETCH_TEXT_BATCH_SEQUENCE_HASH: u64 = 16740235391902546413;
    /// Expected hash for deterministic prefetch triplet batch sequence.
    pub const PREFETCH_TRIPLET_BATCH_SEQUENCE_HASH: u64 = 11709520942485223708;
    /// Expected hash for deterministic prefetch pair batch sequence.
    pub const PREFETCH_PAIR_BATCH_SEQUENCE_HASH: u64 = 2553049861003803008;
}

/// Constants used by split-store persistence and wire encoding.
pub mod splits {
    use super::SplitLabel;

    /// Version tag for persisted epoch metadata payloads.
    pub const EPOCH_STATE_VERSION: u8 = 1;
    /// Version tag for persisted sampler-state payloads.
    pub const SAMPLER_STATE_RECORD_VERSION: u8 = 1;
    /// Key used for storing sampler-state payloads.
    pub const SAMPLER_STATE_KEY: &[u8] = b"sampler_state";

    /// Key used for split-store global metadata.
    pub const META_KEY: &[u8] = b"__meta__";
    /// Key prefix for split label assignments.
    pub const SPLIT_PREFIX: &[u8] = b"split:";
    /// Key prefix for per-split epoch metadata records.
    pub const EPOCH_META_PREFIX: &[u8] = b"epoch_meta:";
    /// Key prefix for per-split epoch hash-list records.
    pub const EPOCH_HASHES_PREFIX: &[u8] = b"epoch_hashes:";
    /// Tombstone marker byte for clearing persisted epoch hashes.
    pub const EPOCH_RECORD_TOMBSTONE: u8 = b'-';
    /// Version tag for persisted epoch-meta records.
    pub const EPOCH_META_RECORD_VERSION: u8 = 1;
    /// Version tag for persisted epoch-hash records.
    pub const EPOCH_HASH_RECORD_VERSION: u8 = 1;
    /// Prefix marker for bitcode-encoded payloads.
    pub const BITCODE_PREFIX: u8 = b'B';
    /// Version tag for split-store metadata compatibility checks.
    pub const STORE_VERSION: u8 = 1;
    /// Canonical split iteration order used when storing/loading all splits.
    pub const ALL_SPLITS: [SplitLabel; 3] =
        [SplitLabel::Train, SplitLabel::Validation, SplitLabel::Test];
}

/// Constants used by file-corpus indexing and persisted index layout.
pub mod file_corpus {
    /// Metadata key for serialized file-index settings and entry count.
    pub const FILE_INDEX_META_KEY: &[u8] = b"meta";
    /// Prefix for serialized file-index path records.
    pub const FILE_INDEX_PATH_KEY_PREFIX: &[u8] = b"idx:";
    /// Default directory name for persisted file-index datastore.
    pub const FILE_INDEX_STORE_DIR: &str = "sampler_file_index";
    /// Internal datastore read chunk size for file-index lookups.
    ///
    /// This only controls how many index keys are fetched per `batch_read` call
    /// when scanning index metadata; it does **not** cap sampler/training
    /// `batch_size` values.
    pub const FILE_INDEX_READ_BATCH: usize = 256;
    /// Log message used when unreadable records are skipped.
    pub const SKIP_UNREADABLE_MSG: &str = "skipping unreadable file record";
}

/// Constants used by the Hugging Face row source backend.
pub mod huggingface {
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
}

/// Constants used for managed cache-root groups.
pub mod cache {
    /// Managed cache group for Hugging Face snapshot-backed sources.
    pub const HUGGINGFACE_GROUP: &str = "triplets/huggingface";
    /// Managed cache group for file-corpus index stores.
    pub const FILE_CORPUS_GROUP: &str = "triplets/file-corpus";
    /// Managed cache group for multi-source demo split-store persistence.
    pub const MULTI_SOURCE_DEMO_GROUP: &str = "triplets/multi-source-demo";
    /// Filename used by the demo app split-store persistence.
    pub const MULTI_SOURCE_DEMO_STORE_FILENAME: &str = "split_store.bin";
}
