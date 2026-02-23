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
