use std::collections::HashMap;

use crate::data::{ChunkView, RecordChunk};
use crate::types::SourceId;

/// Aggregate skew metrics for per-source sample counts.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceSkew {
    /// Total sample count across all sources.
    pub total: usize,
    /// Number of sources represented in the metric.
    pub sources: usize,
    /// Minimum per-source sample count.
    pub min: usize,
    /// Maximum per-source sample count.
    pub max: usize,
    /// Mean per-source sample count.
    pub mean: f64,
    /// Largest source share (`max / total`).
    pub max_share: f64,
    /// Smallest source share (`min / total`).
    pub min_share: f64,
    /// Imbalance ratio (`max / min`, or `inf` when `min == 0`).
    pub ratio: f64,
    /// Per-source counts and shares sorted descending by count.
    pub per_source: Vec<SourceShare>,
}

/// Per-source share of a batch for skew inspection.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceShare {
    /// Source identifier.
    pub source: SourceId,
    /// Number of samples drawn from this source.
    pub count: usize,
    /// Fraction of total samples contributed by this source.
    pub share: f64,
}

/// Compute skew metrics from per-source counts.
/// The map keys are source IDs (the `RecordId` prefix before `::`).
pub fn source_skew(counts: &HashMap<SourceId, usize>) -> Option<SourceSkew> {
    if counts.is_empty() {
        return None;
    }
    let total: usize = counts.values().sum();
    let sources = counts.len();
    let min = *counts.values().min().expect("counts non-empty");
    let max = *counts.values().max().expect("counts non-empty");
    let mean = total as f64 / sources as f64;
    let max_share = if total == 0 {
        0.0
    } else {
        max as f64 / total as f64
    };
    let min_share = if total == 0 {
        0.0
    } else {
        min as f64 / total as f64
    };
    let ratio = if min == 0 {
        f64::INFINITY
    } else {
        max as f64 / min as f64
    };
    let mut per_source: Vec<SourceShare> = counts
        .iter()
        .map(|(source, count)| SourceShare {
            source: source.clone(),
            count: *count,
            share: if total == 0 {
                0.0
            } else {
                *count as f64 / total as f64
            },
        })
        .collect();
    per_source.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.source.cmp(&b.source)));
    Some(SourceSkew {
        total,
        sources,
        min,
        max,
        mean,
        max_share,
        min_share,
        ratio,
        per_source,
    })
}

/// Compute normalized distance between two chunk windows from the same section.
///
/// Returns `Some(distance)` in `[0.0, 1.0]` when both chunks are `Window` views
/// from the same `(record_id, section_idx)`. Returns `None` when distance is not
/// comparable (different records/sections or non-window views).
pub fn window_chunk_distance(anchor: &RecordChunk, positive: &RecordChunk) -> Option<f32> {
    if anchor.record_id != positive.record_id || anchor.section_idx != positive.section_idx {
        return None;
    }
    match (&anchor.view, &positive.view) {
        (ChunkView::Window { index: left, .. }, ChunkView::Window { index: right, .. }) => {
            let delta = left.abs_diff(*right) as f32;
            Some(delta / (delta + 1.0))
        }
        _ => None,
    }
}

/// Convert chunk distance into a chunk proximity score in `[0.0, 1.0]`.
///
/// A higher score means anchor/positive chunks are closer in the document.
/// When distance cannot be computed, returns `1.0` (neutral multiplier).
pub fn chunk_proximity_score(anchor: &RecordChunk, positive: &RecordChunk) -> f32 {
    window_chunk_distance(anchor, positive)
        .map(|distance| 1.0 - distance)
        .unwrap_or(1.0)
}

/// Backward-compatible alias for `chunk_proximity_score`.
pub fn chunk_distance_relevance_score(anchor: &RecordChunk, positive: &RecordChunk) -> f32 {
    chunk_proximity_score(anchor, positive)
}

/// Proximity score of a window chunk to the section head (index 0).
///
/// Returns a value in `(0.0, 1.0]` using `1 / (index + 1)`.
/// - index `0` -> `1.0`
/// - index `1` -> `0.5`
/// - index `3` -> `0.25`
pub fn window_index_proximity(index: usize) -> f32 {
    1.0 / (index as f32 + 1.0)
}

/// Compute byte-level Jaccard and cosine similarity scores between two strings.
///
/// Uses raw UTF-8 byte occurrence frequencies (no tokenisation), so it is fast
/// and dependency-free. Returns `(jaccard, cosine)` each in `[0.0, 1.0]`;
/// both are `0.0` when either input is empty.
///
/// Used by BM25 ranking tests to verify top-ranked candidates beat the
/// uniform-pool baseline, and by the `extended-metrics` demo output.
#[cfg(any(feature = "bm25-mining", feature = "extended-metrics"))]
pub(crate) fn lexical_similarity_scores(left: &str, right: &str) -> (f32, f32) {
    if left.is_empty() || right.is_empty() {
        return (0.0, 0.0);
    }

    let mut left_freq = [0.0_f32; 256];
    let mut right_freq = [0.0_f32; 256];
    let mut left_bits = [0_u8; 32];
    let mut right_bits = [0_u8; 32];

    for byte in left.as_bytes() {
        let idx = *byte as usize;
        left_freq[idx] += 1.0;
        left_bits[idx / 8] |= 1_u8 << (idx % 8);
    }
    for byte in right.as_bytes() {
        let idx = *byte as usize;
        right_freq[idx] += 1.0;
        right_bits[idx / 8] |= 1_u8 << (idx % 8);
    }

    let dot: f32 = left_freq
        .iter()
        .zip(right_freq.iter())
        .map(|(a, b)| a * b)
        .sum();
    let left_norm_sq: f32 = left_freq.iter().map(|v| v * v).sum();
    let right_norm_sq: f32 = right_freq.iter().map(|v| v * v).sum();
    let cosine = if left_norm_sq > 0.0 && right_norm_sq > 0.0 {
        dot / (left_norm_sq.sqrt() * right_norm_sq.sqrt())
    } else {
        0.0
    };

    let mut intersection = 0_u32;
    let mut union = 0_u32;
    for i in 0..left_bits.len() {
        intersection += (left_bits[i] & right_bits[i]).count_ones();
        union += (left_bits[i] | right_bits[i]).count_ones();
    }
    let jaccard = if union > 0 {
        intersection as f32 / union as f32
    } else {
        0.0
    };

    (jaccard, cosine)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window_chunk(record_id: &str, section_idx: usize, index: usize) -> RecordChunk {
        RecordChunk {
            record_id: record_id.to_string(),
            section_idx,
            view: ChunkView::Window {
                index,
                overlap: 0,
                span: 16,
            },
            text: "x".to_string(),
            tokens_estimate: 1,
            quality: crate::data::QualityScore::default(),
        }
    }

    #[test]
    fn source_skew_returns_none_for_empty_counts() {
        let counts = HashMap::new();
        assert!(source_skew(&counts).is_none());
    }

    #[test]
    fn source_skew_reports_balance() {
        let mut counts = HashMap::new();
        counts.insert("A".to_string(), 2);
        counts.insert("B".to_string(), 2);
        let skew = source_skew(&counts).expect("skew");
        assert_eq!(skew.total, 4);
        assert_eq!(skew.sources, 2);
        assert_eq!(skew.min, 2);
        assert_eq!(skew.max, 2);
        assert!((skew.max_share - 0.5).abs() < 1e-6);
        assert!((skew.ratio - 1.0).abs() < 1e-6);
        assert_eq!(skew.per_source.len(), 2);
        assert!(
            skew.per_source
                .iter()
                .all(|entry| (entry.share - 0.5).abs() < 1e-6)
        );
    }

    #[test]
    fn source_skew_reports_imbalance() {
        let mut counts = HashMap::new();
        counts.insert("A".to_string(), 4);
        counts.insert("B".to_string(), 2);
        counts.insert("C".to_string(), 2);
        let skew = source_skew(&counts).expect("skew");
        assert_eq!(skew.total, 8);
        assert_eq!(skew.sources, 3);
        assert_eq!(skew.min, 2);
        assert_eq!(skew.max, 4);
        assert!((skew.max_share - 0.5).abs() < 1e-6);
        assert!((skew.ratio - 2.0).abs() < 1e-6);
        assert_eq!(skew.per_source[0].source, "A");
        assert_eq!(skew.per_source[0].count, 4);
    }

    #[test]
    fn source_skew_zero_totals_report_zero_shares_and_infinite_ratio() {
        let mut counts = HashMap::new();
        counts.insert("B".to_string(), 0);
        counts.insert("A".to_string(), 0);

        let skew = source_skew(&counts).expect("skew");
        assert_eq!(skew.total, 0);
        assert_eq!(skew.min, 0);
        assert_eq!(skew.max, 0);
        assert_eq!(skew.max_share, 0.0);
        assert_eq!(skew.min_share, 0.0);
        assert!(skew.ratio.is_infinite());
        assert_eq!(skew.per_source[0].source, "A");
        assert_eq!(skew.per_source[1].source, "B");
        assert!(skew.per_source.iter().all(|entry| entry.share == 0.0));
    }

    #[test]
    fn window_chunk_distance_uses_index_delta() {
        let a = window_chunk("record", 0, 1);
        let b = window_chunk("record", 0, 4);
        let distance = window_chunk_distance(&a, &b).expect("distance");
        assert!((distance - 0.75).abs() < 1e-6, "distance={distance}");
    }

    #[test]
    fn chunk_proximity_score_inverts_distance() {
        let a = window_chunk("record", 0, 1);
        let b = window_chunk("record", 0, 4);
        let proximity = chunk_proximity_score(&a, &b);
        assert!((proximity - 0.25).abs() < 1e-6, "proximity={proximity}");
    }

    #[test]
    fn chunk_proximity_score_is_neutral_when_not_comparable() {
        let a = window_chunk("record_a", 0, 1);
        let b = window_chunk("record_b", 0, 4);
        assert_eq!(window_chunk_distance(&a, &b), None);
        assert_eq!(chunk_proximity_score(&a, &b), 1.0);
    }

    #[test]
    fn chunk_distance_relevance_score_alias_matches_proximity() {
        let a = window_chunk("record", 0, 1);
        let b = window_chunk("record", 0, 4);
        assert_eq!(
            chunk_distance_relevance_score(&a, &b),
            chunk_proximity_score(&a, &b)
        );
    }

    #[test]
    fn window_index_proximity_scores_drop_with_index() {
        assert!((window_index_proximity(0) - 1.0).abs() < 1e-6);
        assert!((window_index_proximity(1) - 0.5).abs() < 1e-6);
        assert!((window_index_proximity(3) - 0.25).abs() < 1e-6);
    }

    #[cfg(any(feature = "bm25-mining", feature = "extended-metrics"))]
    #[test]
    fn lexical_similarity_identical_strings_score_one() {
        let (j, c) = lexical_similarity_scores("hello world", "hello world");
        assert!((j - 1.0).abs() < 1e-6, "jaccard={j}");
        assert!((c - 1.0).abs() < 1e-6, "cosine={c}");
    }

    #[cfg(any(feature = "bm25-mining", feature = "extended-metrics"))]
    #[test]
    fn lexical_similarity_empty_inputs_score_zero() {
        assert_eq!(lexical_similarity_scores("", "hello"), (0.0, 0.0));
        assert_eq!(lexical_similarity_scores("hello", ""), (0.0, 0.0));
        assert_eq!(lexical_similarity_scores("", ""), (0.0, 0.0));
    }

    #[cfg(any(feature = "bm25-mining", feature = "extended-metrics"))]
    #[test]
    fn lexical_similarity_scores_are_in_unit_range() {
        let cases = [
            ("foo bar baz", "qux quux"),
            ("abc", "abc def"),
            ("the quick brown fox", "jumped over the lazy dog"),
        ];
        for (a, b) in cases {
            let (j, c) = lexical_similarity_scores(a, b);
            assert!(
                (0.0..=1.0).contains(&j),
                "jaccard={j} out of range for ({a:?}, {b:?})"
            );
            assert!(
                (0.0..=1.0).contains(&c),
                "cosine={c} out of range for ({a:?}, {b:?})"
            );
        }
    }
}
