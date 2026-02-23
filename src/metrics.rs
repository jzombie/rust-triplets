use std::collections::HashMap;

use crate::types::SourceId;

/// Aggregate skew metrics for per-source sample counts.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceSkew {
    pub total: usize,
    pub sources: usize,
    pub min: usize,
    pub max: usize,
    pub mean: f64,
    pub max_share: f64,
    pub min_share: f64,
    pub ratio: f64,
    pub per_source: Vec<SourceShare>,
}

/// Per-source share of a batch for skew inspection.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceShare {
    pub source: SourceId,
    pub count: usize,
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
