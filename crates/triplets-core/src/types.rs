/// Unique record identifier (stable across runs).
/// Example: `source_b::factual::definitions/What_is_tag_x.txt`
pub type RecordId = String;
/// Identifier for the source that produced a record.
/// Examples: `source_a`, `source_b::factual`, `source_c`
pub type SourceId = String;
/// Identifier for a category label (e.g., question-answer taxonomy buckets).
/// Examples: `factual`, `opinionated`
pub type CategoryId = String;
/// Normalized metadata values (e.g., date strings).
/// Examples: `2025-02-25`, `02/25/2025`, `Oct 15, 2024`
pub type MetaValue = String;
/// Normalized taxonomy values.
/// Examples: `source_b::factual`, `definitions`, `metaphors`
pub type TaxonomyValue = String;
/// Sentence text extracted from sections.
/// Example: `Amount of direct financing lease revenue.`
pub type Sentence = String;
/// Key for per-source recipe scheduling.
/// Examples: `source_b`, `source_b_anchor`, `source_b_positive`
pub type RecipeKey = String;
/// Warning/log message text.
/// Examples: `skipping unreadable file record`, `[data_sampler] source '...' refresh failed: ...`
pub type LogMessage = String;
/// Value for key-value metadata sampling.
/// Examples: `2025-02-25`, `factual`, `train`
pub type KvpValue = String;
/// File path strings used in transport tests.
/// Example: `factual/xbrl_definitions/What Does the Xbrl Tag Us-gaap:timedepositslessthan100000 Represent?.txt`
pub type PathString = String;
/// Deterministic grouping key for locality-aware ordering.
/// Example: `factual/xbrl_definitions`
pub type GroupKey = String;
/// Deterministic per-item ordering key used during grouping.
/// Example: `factual/xbrl_definitions/What Does the Xbrl Tag Us-gaap:timedepositslessthan100000 Represent?.txt`
pub type ItemOrderKey = String;
/// Components used to build snapshot hashes.
/// Example: `text|source_b_anchor|source_b::factual/alpha.txt|summary:head:24|1.000000`
pub type HashPart = String;

/// Composite record identity: the source that produced the record plus the
/// record's id within that source.
///
/// Record ids are only unique within one source (two stores can both emit
/// `"5"`). Every pool keyed by record identity — the sampler pool, chunk
/// index, split labels, dedup sets, ingestion caches — keys by this composite
/// instead of the bare id, so overlapping ids from different sources coexist
/// instead of silently overwriting each other. Backends never decorate their
/// ids; the composite is derived from the record's own `source` + `id`
/// fields wherever identity is needed.
#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
    bitcode::Encode,
    bitcode::Decode,
)]
pub struct RecordKey {
    /// Source that produced the record.
    pub source: SourceId,
    /// Record id within that source.
    pub id: RecordId,
}

impl RecordKey {
    /// Build an identity from parts.
    pub fn new(source: impl Into<SourceId>, id: impl Into<RecordId>) -> Self {
        Self {
            source: source.into(),
            id: id.into(),
        }
    }
}

impl From<&crate::data::DataRecord> for RecordKey {
    /// Build the identity of a record from its own `source` + `id` fields.
    fn from(record: &crate::data::DataRecord) -> Self {
        Self::new(record.source.clone(), record.id.clone())
    }
}

impl From<&std::sync::Arc<crate::data::DataRecord>> for RecordKey {
    /// Build the identity of a record from its own `source` + `id` fields.
    fn from(record: &std::sync::Arc<crate::data::DataRecord>) -> Self {
        Self::new(record.source.clone(), record.id.clone())
    }
}

impl From<std::sync::Arc<crate::data::DataRecord>> for RecordKey {
    /// Build the identity of a record from its own `source` + `id` fields.
    fn from(record: std::sync::Arc<crate::data::DataRecord>) -> Self {
        Self::new(record.source.clone(), record.id.clone())
    }
}

impl RecordKey {
    /// Build the identity of an `Arc`-wrapped record from its own `source` +
    /// `id` fields. Plain named fn (rather than `From`) so it coerces to a
    /// `map`/`sort_by_key` fn item; `From` hits higher-ranked lifetime
    /// inference limits in that position.
    pub fn of_arc(record: &std::sync::Arc<crate::data::DataRecord>) -> Self {
        Self::new(record.source.clone(), record.id.clone())
    }

    /// Build the identity of a chunk's parent record from the chunk's own
    /// `source` + `record_id` fields.
    pub fn of_chunk(chunk: &crate::data::RecordChunk) -> Self {
        Self::new(chunk.source.clone(), chunk.record_id.clone())
    }
}
