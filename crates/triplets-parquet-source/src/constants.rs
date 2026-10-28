//! Store-format and recipe constants for the generic parquet row source.
//!
//! Names keep their historical `HF_` prefix to avoid churning every use
//! site; values are the compatibility surface (on-disk `.simdr` format,
//! recipe names) and must not change.

// ---------------------------------------------------------------------------
// ── Row-store format ────────────────────────────────────────────────────────
// ---------------------------------------------------------------------------

/// Extension used by persisted per-shard row-store files.
pub const HF_SHARD_STORE_EXTENSION: &str = "simdr";
/// Key prefix for individual row payload entries in shard row stores.
pub const HF_SHARD_STORE_ROW_PREFIX: &[u8] = b"rowv1|";
/// Metadata key storing row counts in shard row stores.
pub const HF_SHARD_STORE_META_ROWS_KEY: &[u8] = b"meta|rows";
/// Metadata key storing the original source shard size recorded at
/// materialization time.  Compared on subsequent cycles to detect
/// out-of-date shards.
pub const HF_SHARD_STORE_SOURCE_SIZE_KEY: &[u8] = b"meta|source_size";

/// Filename prefix for in-flight temporary downloads.
///
/// Files with this prefix may be deleted once their store has been
/// materialized; never delete user-provided local files.
pub const HF_TEMP_DOWNLOAD_PREFIX: &str = "triplets_hf_";

// ---------------------------------------------------------------------------
// ── Refresh batching ────────────────────────────────────────────────────────
// ---------------------------------------------------------------------------

/// Multiplies the source `refresh` limit passed by the ingestion manager
/// to set this source's internal row-read batch target for each refresh
/// pass.
pub const HF_REFRESH_BATCH_MULTIPLIER: usize = 8;

// ---------------------------------------------------------------------------
// ── Default triplet recipes ─────────────────────────────────────────────────
// ---------------------------------------------------------------------------

/// Default recipe for text-columns mode (SimCSE-style, same-text pairs allowed).
pub const HF_RECIPE_TEXT_SIMCSE_WRONG_ARTICLE: &str = "huggingface_text_simcse_wrong_article";
/// Default majority-lane recipe for role-mapped sources.
pub const HF_RECIPE_ANCHOR_CONTEXT_WRONG_ARTICLE: &str = "huggingface_anchor_context_wrong_article";
/// Default recipe for dict mode (negatives from the same record).
pub const HF_RECIPE_DICT_ANCHOR_POSITIVE_SAME_RECORD: &str =
    "huggingface_dict_anchor_positive_same_record";
/// Default medium-hard-lane recipe for role-mapped sources.
pub const HF_RECIPE_ANCHOR_ANCHOR_WRONG_ARTICLE: &str = "huggingface_anchor_anchor_wrong_article";
