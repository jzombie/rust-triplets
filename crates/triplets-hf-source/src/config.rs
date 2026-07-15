use crate::disk_cache::StoreCache;
use reqwest_drive::ClientWithMiddleware;
use std::path::PathBuf;

/// Configuration for a bulk Hugging Face row source backed by local snapshot files.
#[derive(Clone, Debug)]
pub struct HuggingFaceRowsConfig {
    /// Stable sampler source id used in record ids and metrics.
    pub source_id: String,
    /// Hugging Face dataset id, e.g. `HuggingFaceFW/fineweb`.
    pub dataset_name: String,
    /// Dataset config name, e.g. `default`.
    pub config_name: String,
    /// Split name, e.g. `train`.
    pub split_name: String,
    /// Local path to a snapshot directory for this split.
    pub snapshot_dir: PathBuf,
    /// File extensions accepted as shard files.
    ///
    /// Non-parquet files are read as line-delimited entries. Each line may be:
    /// - a JSON object row (for example JSONL/NDJSON), or
    /// - plain text, which is wrapped as `{ "text": "..." }`.
    pub shard_extensions: Vec<String>,

    /// Maximum number of rows cached in-memory.
    pub cache_capacity: usize,
    /// Maximum number of decoded parquet row groups cached in-memory.
    pub parquet_row_group_cache_capacity: usize,
    /// Multiplier applied to current refresh `limit` when building a read batch target.
    ///
    /// Effective target is `limit * refresh_batch_multiplier`.
    pub refresh_batch_multiplier: usize,
    /// Multiplier applied to ingestion-sized base records for `len_hint` headroom.
    ///
    /// Effective headroom is `cache_capacity * remote_expansion_headroom_multiplier`.
    pub remote_expansion_headroom_multiplier: usize,
    /// Hard cap for local manifest-shard cache bytes.
    ///
    /// Enforced by `cache-manager` policy application on manifest cache roots.
    pub local_disk_cap_bytes: Option<u64>,
    /// Optional row id column name. Falls back to synthetic id when missing.
    pub id_column: Option<String>,
    /// Text candidate columns (ordered) for text-columns mode.
    ///
    /// Each candidate is tried in order; the first whose value is present and
    /// non-empty is used as the single text content for the row.  When the
    /// list is non-empty and no candidate yields content, the row is skipped.
    pub text_columns: Vec<String>,
    /// Anchor candidate columns (ordered).
    ///
    /// Each candidate is tried in order; the first whose value is present and
    /// non-empty is used as the anchor role section.  When the list is
    /// non-empty and no candidate yields content, the row is skipped.
    ///
    /// When non-empty (or when `positive_columns`/`context_columns` are set),
    /// role-based extraction is used instead of `text_columns` mode.
    pub anchor_columns: Vec<String>,
    /// Positive candidate columns (ordered).
    ///
    /// Each candidate is tried in order; the first whose value is present and
    /// non-empty is used for the positive role section.  When the list is
    /// non-empty and no candidate yields content, the row is skipped.
    ///
    /// Positive text is emitted as a `SectionRole::Context` section.
    pub positive_columns: Vec<String>,
    /// Negative candidate columns (ordered).
    ///
    /// Used only in **role-based mode**.  When a column value is a JSON array,
    /// each element is expanded into a separate `SectionRole::Context` section.
    /// This supports HuggingFace "dict" datasets where negatives are embedded
    /// as a list in the same row (e.g. `embedding-data/QQP_triplets`).
    pub negative_columns: Vec<String>,
    /// Optional ordered context columns.
    ///
    /// Used only in **role-based mode** (i.e. when `anchor_columns` and/or
    /// `positive_columns` are set).  Every listed column is required: if any
    /// is missing or blank the row is skipped.
    ///
    /// Each column becomes an additional `SectionRole::Context` section in the
    /// emitted record, appended after the positive section.  Unlike
    /// `anchor_columns`/`positive_columns`, there is no coalescing — all
    /// columns contribute independently as separate sections.
    ///
    /// Ignored in **text-columns mode** (when `anchor_columns` is empty and
    /// `text_columns` is non-empty).
    pub context_columns: Vec<String>,
    /// Optional trust/quality override applied to all records produced by this source.
    ///
    /// When set, overrides the default `QualityScore::default().trust` (0.5) for
    /// every record emitted by this source.  Set this on sources that provide
    /// higher- or lower-quality data than the default.
    pub trust_override: Option<f32>,
    /// Optional Hugging Face API token for authenticating private dataset access.
    ///
    /// When set, sent as `Authorization: Bearer <token>` on datasets-server API
    /// requests for shard downloads.  Populated
    /// automatically from the `HF_TOKEN` environment variable at construction
    /// time; callers may also set this field directly.
    pub hf_token: Option<String>,
    /// Resolved datasets-server parquet manifest endpoint URL.
    /// Populated at construction time from `TRIPLETS_HF_PARQUET_ENDPOINT` env var
    /// or `HF_PARQUET_DEFAULT_ENDPOINT`.
    pub parquet_endpoint: String,
    /// In-memory cache of opened `DataStore` instances, keyed by shard path.
    /// Populated lazily as shards are accessed and cleared when the cache grows
    /// beyond the configured capacity.
    pub store_cache: StoreCache,
    /// Optional pre-built HTTP client.  When set, [`crate::huggingface_source::HuggingFaceRowSource::new`]
    /// uses this client instead of building a new one.  This allows callers
    /// such as [`crate::huggingface_source::build_hf_sources`] to share a single connection pool and
    /// throttle state across many sources.
    pub(crate) http_client: Option<ClientWithMiddleware>,
}
