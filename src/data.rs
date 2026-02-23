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
    /// Semantic role used by selectors (for example, anchor vs context text).
    pub role: SectionRole,
    /// Optional short heading/title for this section.
    pub heading: Option<String>,
    /// Full section text.
    pub text: String,
    /// Sentence-level segmentation of `text` used by chunking strategies.
    pub sentences: Vec<Sentence>,
}

/// Role label for a section.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SectionRole {
    /// Primary section typically used as an anchor candidate.
    Anchor,
    /// Supporting/context section used for positives, negatives, or text samples.
    Context,
}

/// A chunked view over a section.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecordChunk {
    /// Parent record id this chunk belongs to.
    pub record_id: RecordId,
    /// Index of the source section in `DataRecord.sections`.
    pub section_idx: usize,
    /// Chunk view metadata (window position or summary fallback).
    pub view: ChunkView,
    /// Rendered chunk text (possibly with metadata prefix decoration).
    pub text: String,
    /// Approximate token count for scheduling/weighting heuristics.
    pub tokens_estimate: usize,
    /// Trust/quality inherited from the parent record.
    pub quality: QualityScore,
}

/// Chunk view metadata (window or summary).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ChunkView {
    /// Sliding-window chunk extracted directly from section text.
    Window {
        /// Zero-based window index within the section.
        index: usize,
        /// Overlap (in tokens) with the previous window.
        overlap: usize,
        /// Nominal window span in tokens.
        span: usize,
        /// Position of the window's start as a fraction of total tokens in the section (0.0 = first token, 1.0 = past the end).
        start_ratio: f32,
    },
    /// Summary fallback chunk used when window extraction is unavailable.
    SummaryFallback {
        /// Name of summary strategy that produced this fallback chunk.
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
    /// Anchor chunk used to build this supervised pair.
    pub anchor: RecordChunk,
    /// Candidate chunk paired with the anchor.
    pub positive: RecordChunk,
    /// Training weight for this pair.
    pub weight: f32,
    /// Optional instruction/prompt hint for this sample.
    pub instruction: Option<String>,
    /// Supervision label (positive or negative).
    pub label: PairLabel,
    /// Optional reason/annotation describing the label.
    pub reason: Option<String>,
}

/// Sample triplet (anchor/positive/negative).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SampleTriplet {
    /// Recipe name used to generate this triplet.
    pub recipe: String,
    /// Anchor chunk.
    pub anchor: RecordChunk,
    /// Positive chunk.
    pub positive: RecordChunk,
    /// Negative chunk.
    pub negative: RecordChunk,
    /// Training weight for this triplet.
    pub weight: f32,
    /// Optional instruction/prompt hint for this sample.
    pub instruction: Option<String>,
}

/// Pair label for supervised pair batches.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum PairLabel {
    /// Anchor and candidate are semantically aligned.
    Positive,
    /// Anchor and candidate are semantically mismatched.
    Negative,
}

/// Batch of pairs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SampleBatch {
    /// Pair samples contained in this batch.
    pub pairs: Vec<SamplePair>,
}

impl SampleBatch {
    /// Returns `true` when the batch has no pairs.
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}

/// Batch of triplets.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TripletBatch {
    /// Triplet samples contained in this batch.
    pub triplets: Vec<SampleTriplet>,
}

impl TripletBatch {
    /// Returns `true` when the batch has no triplets.
    pub fn is_empty(&self) -> bool {
        self.triplets.is_empty()
    }
}

/// A single text sample (chunk + weight).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextSample {
    /// Recipe name used to generate this sample.
    pub recipe: String,
    /// Chunk payload used for this text sample.
    pub chunk: RecordChunk,
    /// Training weight for this sample.
    pub weight: f32,
    /// Optional instruction/prompt hint for this sample.
    pub instruction: Option<String>,
}

/// Batch of text samples.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextBatch {
    /// Text samples contained in this batch.
    pub samples: Vec<TextSample>,
}

impl TextBatch {
    /// Returns `true` when the batch has no text samples.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}
