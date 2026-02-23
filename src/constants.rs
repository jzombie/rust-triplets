use crate::metadata::MetadataKey;
use crate::splits::SplitLabel;

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
    pub const TRIPLET_BATCH_SEQUENCE_HASH: u64 = 5355337600689408051;
    /// Expected hash for deterministic pair batch sequence.
    pub const PAIR_BATCH_SEQUENCE_HASH: u64 = 8198096084611658104;
    /// Expected hash for deterministic prefetch text batch sequence.
    pub const PREFETCH_TEXT_BATCH_SEQUENCE_HASH: u64 = 16740235391902546413;
    /// Expected hash for deterministic prefetch triplet batch sequence.
    pub const PREFETCH_TRIPLET_BATCH_SEQUENCE_HASH: u64 = 17475118382069588204;
    /// Expected hash for deterministic prefetch pair batch sequence.
    pub const PREFETCH_PAIR_BATCH_SEQUENCE_HASH: u64 = 13723875325938529772;
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

    /// Default directory for persisted split-store files.
    pub const DEFAULT_STORE_DIR: &str = ".sampler_store";
    /// Default filename for persisted split-store files.
    pub const DEFAULT_STORE_FILENAME: &str = "split_store.bin";
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
