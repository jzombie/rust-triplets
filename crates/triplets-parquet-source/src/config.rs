//! Configuration for a generic local parquet/row-store source.
use std::path::PathBuf;

use crate::constants::HF_REFRESH_BATCH_MULTIPLIER;
use crate::store_cache::StoreCache;
use triplets_core::config::SamplerConfig;

/// Configuration for a local row source backed by snapshot files.
///
/// Unlike [`triplets_hf_source::HuggingFaceRowsConfig`](https://github.com/jzombie/rust-triplets),
/// dataset identity is caller-provided plain data: [`Self::row_id_namespace`]
/// feeds synthetic row ids and [`Self::taxonomy`] is stamped on every
/// record.  There are no network fields.
#[derive(Clone, Debug)]
pub struct ParquetRowsConfig {
    /// Stable sampler source id used in record ids and metrics.
    pub source_id: String,
    /// Local path to a snapshot directory for this source.
    pub snapshot_dir: PathBuf,
    /// File extensions accepted as shard files.
    ///
    /// Non-parquet files are read as line-delimited entries. Each line may be:
    /// - a JSON object row (for example JSONL/NDJSON), or
    /// - plain text, which is wrapped as `{ "text": "..." }`.
    pub shard_extensions: Vec<String>,
    /// Directories whose non-store files are crate-managed: they are
    /// transcoded to `.simdr` stores on indexing and the originals may be
    /// deleted afterwards.  Files outside these directories (user-provided
    /// local files) are never deleted.  Remote-backed wrappers pass their
    /// manifest/download dir here.
    pub transient_dirs: Vec<PathBuf>,
    /// Optional managed subdirectory (relative to `snapshot_dir`) that disk-cap
    /// eviction applies to.  `None` disables eviction entirely — the safe
    /// default for user-owned files, which must never be deleted.
    pub managed_subdir: Option<String>,
    /// Namespace prefix for synthetic row ids (`{namespace}:{row_index}`),
    /// used when no `id_column` value resolves.
    pub row_id_namespace: String,
    /// Taxonomy values stamped on every record produced by this source.
    pub taxonomy: Vec<String>,

    /// Maximum number of rows cached in-memory.
    pub cache_capacity: usize,
    /// Maximum number of decoded parquet row groups cached in-memory.
    pub parquet_row_group_cache_capacity: usize,
    /// Multiplier applied to current refresh `limit` when building a read batch target.
    ///
    /// Effective target is `limit * refresh_batch_multiplier`.
    pub refresh_batch_multiplier: usize,
    /// Hard cap for managed-shard cache bytes.
    ///
    /// Enforced by `cache-manager` policy application on `managed_subdir`.
    /// Ignored when `managed_subdir` is `None`.
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
    /// every record emitted by this source.
    pub trust_override: Option<f32>,
    /// In-memory cache of opened `DataStore` instances, keyed by shard path.
    /// Populated lazily as shards are accessed and cleared when the cache grows
    /// beyond the configured capacity.
    pub store_cache: StoreCache,
}

impl ParquetRowsConfig {
    /// Create a config with a source id, snapshot directory, id namespace,
    /// and taxonomy.  Column mappings default to `text_columns = ["text"]`
    /// with `id_column = Some("id")`; eviction defaults to disabled.
    pub fn new(
        source_id: impl Into<String>,
        snapshot_dir: impl Into<PathBuf>,
        row_id_namespace: impl Into<String>,
        taxonomy: Vec<String>,
    ) -> Self {
        Self {
            source_id: source_id.into(),
            snapshot_dir: snapshot_dir.into(),
            shard_extensions: vec![
                "parquet".to_string(),
                crate::constants::HF_SHARD_STORE_EXTENSION.to_string(),
                "jsonl".to_string(),
                "ndjson".to_string(),
                "json".to_string(),
            ],
            transient_dirs: Vec::new(),
            managed_subdir: None,
            row_id_namespace: row_id_namespace.into(),
            taxonomy,
            cache_capacity: SamplerConfig::default().per_source_max_pool,
            parquet_row_group_cache_capacity: 8,
            refresh_batch_multiplier: HF_REFRESH_BATCH_MULTIPLIER,
            local_disk_cap_bytes: None,
            id_column: Some("id".to_string()),
            text_columns: vec!["text".to_string()],
            anchor_columns: Vec::new(),
            positive_columns: Vec::new(),
            negative_columns: Vec::new(),
            context_columns: Vec::new(),
            trust_override: None,
            store_cache: StoreCache::new(),
        }
    }

    /// Returns `true` when at least one column mapping is configured.
    pub fn has_explicit_mapping(&self) -> bool {
        !self.anchor_columns.is_empty()
            || !self.positive_columns.is_empty()
            || !self.negative_columns.is_empty()
            || !self.context_columns.is_empty()
            || !self.text_columns.is_empty()
    }

    /// Returns `true` when `path` lives under a crate-managed transient dir.
    pub fn in_transient_dir(&self, path: &std::path::Path) -> bool {
        self.transient_dirs
            .iter()
            .any(|root| path.starts_with(root))
    }
}
