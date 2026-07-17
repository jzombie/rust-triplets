use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
            // Assume medium trust by default, allowing recipes to upweight or downweight based on other signals.
            trust: 0.5,
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
    /// Pair label for supervised pair entries (Positive/Negative).
    ///
    /// Only meaningful for pair-mode records written by the SRD pipeline.
    /// `None` for records from other sources or triplet-mode entries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<PairLabel>,
}

impl DataRecord {
    /// Create a record with a single [`SectionRole::Context`] section from a plain text string.
    ///
    /// The `id` and `source` are set to the same value. Use [`DataRecord::from_text_with_role`]
    /// to assign a different role, or construct the struct directly for full control.
    ///
    /// # Example
    ///
    /// ```
    /// use triplets_core::DataRecord;
    ///
    /// let record = DataRecord::from_text("doc-0", "my_corpus", "The quick brown fox.");
    /// assert_eq!(record.id.as_str(), "doc-0");
    /// assert_eq!(record.sections[0].text, "The quick brown fox.");
    /// ```
    pub fn from_text(
        id: impl Into<crate::types::RecordId>,
        source: impl Into<crate::types::SourceId>,
        text: impl Into<String>,
    ) -> Self {
        Self::from_text_with_role(id, source, text, SectionRole::Context)
    }

    /// Create a record with a single section of the given role from a plain text string.
    ///
    /// # Example
    ///
    /// ```
    /// use triplets_core::{DataRecord, SectionRole};
    ///
    /// let record = DataRecord::from_text_with_role(
    ///     "doc-0", "my_corpus", "What is the capital of France?", SectionRole::Anchor,
    /// );
    /// assert_eq!(record.sections[0].role, SectionRole::Anchor);
    /// ```
    pub fn from_text_with_role(
        id: impl Into<crate::types::RecordId>,
        source: impl Into<crate::types::SourceId>,
        text: impl Into<String>,
        role: SectionRole,
    ) -> Self {
        let now = chrono::Utc::now();
        Self {
            id: id.into(),
            source: source.into(),
            created_at: now,
            updated_at: now,
            quality: QualityScore::default(),
            taxonomy: vec![],
            sections: vec![RecordSection {
                role,
                heading: None,
                text: text.into(),
                sentences: vec![],
            }],
            meta_prefix: None,
            label: None,
        }
    }
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
    /// All KVP metadata defined on the source record's `meta_prefix`, exposed for
    /// downstream inspection and debugging. Contains every key with all its possible
    /// values across all variants — unaffected by presence probability, dropout, or
    /// which variant was sampled into this chunk's text.
    ///
    /// Populated unconditionally by the sampler during chunk decoration. Empty when the
    /// record has no `meta_prefix` configured.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub kvp_meta: HashMap<String, Vec<String>>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    fn sample_chunk(id: &str) -> RecordChunk {
        RecordChunk {
            record_id: id.to_string(),
            section_idx: 0,
            view: ChunkView::SummaryFallback {
                strategy: "test".to_string(),
                weight: 1.0,
            },
            text: "text".to_string(),
            tokens_estimate: 4,
            quality: QualityScore::default(),
            kvp_meta: Default::default(),
        }
    }

    #[test]
    fn quality_score_defaults_to_medium_trust() {
        let quality = QualityScore::default();
        assert!((quality.trust - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn batch_is_empty_helpers_match_contents() {
        let empty_pairs = SampleBatch { pairs: Vec::new() };
        assert!(empty_pairs.is_empty());

        let non_empty_pairs = SampleBatch {
            pairs: vec![SamplePair {
                recipe: "r".to_string(),
                anchor: sample_chunk("a"),
                positive: sample_chunk("b"),
                weight: 1.0,
                instruction: None,
                label: PairLabel::Positive,
                reason: Some("test".to_string()),
            }],
        };
        assert!(!non_empty_pairs.is_empty());

        let empty_triplets = TripletBatch {
            triplets: Vec::new(),
        };
        assert!(empty_triplets.is_empty());

        let non_empty_triplets = TripletBatch {
            triplets: vec![SampleTriplet {
                recipe: "r".to_string(),
                anchor: sample_chunk("a"),
                positive: sample_chunk("b"),
                negative: sample_chunk("c"),
                weight: 1.0,
                instruction: Some("hint".to_string()),
            }],
        };
        assert!(!non_empty_triplets.is_empty());

        let empty_text = TextBatch {
            samples: Vec::new(),
        };
        assert!(empty_text.is_empty());

        let non_empty_text = TextBatch {
            samples: vec![TextSample {
                recipe: "r".to_string(),
                chunk: sample_chunk("t"),
                weight: 1.0,
                instruction: None,
            }],
        };
        assert!(!non_empty_text.is_empty());
    }

    #[test]
    fn data_record_roundtrip_basics_are_constructible() {
        let now = Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap();
        let record = DataRecord {
            id: "source_a::1".to_string(),
            source: "source_a".to_string(),
            created_at: now,
            updated_at: now,
            quality: QualityScore { trust: 0.9 },
            taxonomy: vec!["topic:news".to_string()],
            sections: vec![RecordSection {
                role: SectionRole::Anchor,
                heading: Some("headline".to_string()),
                text: "body".to_string(),
                sentences: vec!["body".to_string()],
            }],
            meta_prefix: None,
            label: None,
        };

        assert_eq!(record.source, "source_a");
        assert_eq!(record.sections.len(), 1);
        assert!(matches!(record.sections[0].role, SectionRole::Anchor));
    }

    #[test]
    fn from_text_creates_record_with_context_role() {
        let record = DataRecord::from_text("id1", "src1", "hello world");
        assert_eq!(record.id.as_str(), "id1");
        assert_eq!(record.source.as_str(), "src1");
        assert_eq!(record.sections.len(), 1);
        assert_eq!(record.sections[0].text, "hello world");
        assert!(matches!(record.sections[0].role, SectionRole::Context));
    }

    #[test]
    fn from_text_with_role_creates_record_with_specified_role() {
        let record =
            DataRecord::from_text_with_role("id1", "src1", "anchor text", SectionRole::Anchor);
        assert!(matches!(record.sections[0].role, SectionRole::Anchor));
        assert_eq!(record.sections[0].text, "anchor text");
    }

    #[test]
    fn data_record_with_label() {
        let record = DataRecord {
            id: "id".into(),
            source: "src".into(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            quality: QualityScore::default(),
            taxonomy: vec![],
            sections: vec![],
            meta_prefix: None,
            label: Some(PairLabel::Positive),
        };
        assert_eq!(record.label, Some(PairLabel::Positive));
    }

    #[test]
    fn data_record_with_taxonomy() {
        let record = DataRecord {
            id: "id".into(),
            source: "src".into(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            quality: QualityScore::default(),
            taxonomy: vec!["year:2025".into(), "topic:ai".into()],
            sections: vec![],
            meta_prefix: None,
            label: None,
        };
        assert_eq!(record.taxonomy.len(), 2);
    }

    #[test]
    fn pair_label_variants() {
        let pos = PairLabel::Positive;
        let neg = PairLabel::Negative;
        assert_ne!(pos, neg);
        assert_eq!(pos, PairLabel::Positive);
        assert_eq!(neg, PairLabel::Negative);
    }

    #[test]
    fn chunk_view_variants() {
        let window = ChunkView::Window {
            index: 0,
            overlap: 10,
            span: 100,
        };
        let summary = ChunkView::SummaryFallback {
            strategy: "head".into(),
            weight: 0.5,
        };
        assert!(matches!(window, ChunkView::Window { .. }));
        assert!(matches!(summary, ChunkView::SummaryFallback { .. }));
    }

    #[test]
    fn record_section_with_heading() {
        let section = RecordSection {
            role: SectionRole::Context,
            heading: Some("Title".into()),
            text: "content".into(),
            sentences: vec!["content".into()],
        };
        assert_eq!(section.heading, Some("Title".to_string()));
    }

    #[test]
    fn sample_pair_with_reason() {
        let pair = SamplePair {
            recipe: "test".into(),
            anchor: sample_chunk("a"),
            positive: sample_chunk("b"),
            weight: 1.5,
            instruction: Some("instruction".into()),
            label: PairLabel::Negative,
            reason: Some("wrong topic".into()),
        };
        assert_eq!(pair.weight, 1.5);
        assert_eq!(pair.reason, Some("wrong topic".to_string()));
    }

    #[test]
    fn sample_triplet_construction() {
        let triplet = SampleTriplet {
            recipe: "triplet_recipe".into(),
            anchor: sample_chunk("a"),
            positive: sample_chunk("p"),
            negative: sample_chunk("n"),
            weight: 2.0,
            instruction: None,
        };
        assert_eq!(triplet.recipe, "triplet_recipe");
        assert_eq!(triplet.weight, 2.0);
    }

    #[test]
    fn text_sample_construction() {
        let sample = TextSample {
            recipe: "text_recipe".into(),
            chunk: sample_chunk("t"),
            weight: 1.0,
            instruction: Some("do this".into()),
        };
        assert_eq!(sample.recipe, "text_recipe");
        assert!(sample.instruction.is_some());
    }
}
