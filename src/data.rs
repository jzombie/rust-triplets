use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::kvp::KvpPrefixSampler;

pub use crate::types::{RecordId, Sentence, SourceId, TaxonomyValue};

/// Trust/quality metadata for a record.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct QualityScore {
    /// Normalized 0-1 trust measure combining provenance, recency, and manual reviews.
    pub trust: f32,
}

impl Default for QualityScore {
    fn default() -> Self {
        Self {
            // Assume full trust unless a source overrides it.
            trust: 1.0,
        }
    }
}

/// Canonical record payload produced by a DataSource.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataRecord {
    /// Stable record identifier (used for splits and determinism).
    pub id: RecordId,
    /// Source identifier that produced this record.
    pub source: SourceId,
    /// Canonical creation time for the record (used for ordering/metadata).
    pub created_at: DateTime<Utc>,
    /// Last update time for the record (used for refresh decisions).
    pub updated_at: DateTime<Utc>,
    /// Trust/quality score used to weight sampling.
    pub quality: QualityScore,
    /// Free-form tags (e.g., source id, year, date) used for filtering/recipes.
    pub taxonomy: Vec<TaxonomyValue>,
    /// Structured content sections used by sampling recipes.
    pub sections: Vec<RecordSection>,
    /// Optional metadata prefix policy for KVP sampling (key-value headers injected into text).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meta_prefix: Option<KvpPrefixSampler>,
}

/// A structured section within a record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecordSection {
    pub role: SectionRole,
    pub heading: Option<String>,
    pub text: String,
    pub sentences: Vec<Sentence>,
}

/// Role label for a section.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SectionRole {
    Anchor,
    Context,
}

/// A chunked view over a section.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecordChunk {
    pub record_id: RecordId,
    pub section_idx: usize,
    pub view: ChunkView,
    pub text: String,
    pub tokens_estimate: usize,
    pub quality: QualityScore,
}

/// Chunk view metadata (window or summary).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ChunkView {
    Window {
        index: usize,
        overlap: usize,
        span: usize,
        /// Position of the window's start as a fraction of total tokens in the section (0.0 = first token, 1.0 = past the end).
        start_ratio: f32,
    },
    SummaryFallback {
        strategy: String,
        /// Precomputed base weight for summary-fallback chunks before trust/floor are applied.
        weight: f32,
    },
}

/// Sample pair (positive/negative) derived from a triplet.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SamplePair {
    /// Recipe name used to generate this pair.
    pub recipe: String,
    pub anchor: RecordChunk,
    pub positive: RecordChunk,
    pub weight: f32,
    pub instruction: Option<String>,
    pub label: PairLabel,
    pub reason: Option<String>,
}

/// Sample triplet (anchor/positive/negative).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SampleTriplet {
    /// Recipe name used to generate this triplet.
    pub recipe: String,
    pub anchor: RecordChunk,
    pub positive: RecordChunk,
    pub negative: RecordChunk,
    pub weight: f32,
    pub instruction: Option<String>,
}

/// Pair label for supervised pair batches.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum PairLabel {
    Positive,
    Negative,
}

/// Batch of pairs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SampleBatch {
    pub pairs: Vec<SamplePair>,
}

impl SampleBatch {
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}

/// Batch of triplets.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TripletBatch {
    pub triplets: Vec<SampleTriplet>,
}

impl TripletBatch {
    pub fn is_empty(&self) -> bool {
        self.triplets.is_empty()
    }
}

/// A single text sample (chunk + weight).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextSample {
    /// Recipe name used to generate this sample.
    pub recipe: String,
    pub chunk: RecordChunk,
    pub weight: f32,
    pub instruction: Option<String>,
}

/// Batch of text samples.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextBatch {
    pub samples: Vec<TextSample>,
}

impl TextBatch {
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}
