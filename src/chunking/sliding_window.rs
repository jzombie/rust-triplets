use super::algorithm::ChunkingAlgorithm;
use crate::config::ChunkingStrategy;
use crate::data::{ChunkView, DataRecord, RecordChunk, RecordSection};
use crate::tokenizer::{Tokenizer, WhitespaceTokenizer};
use crate::denoiser::denoise_text;

/// Default sliding-window chunking algorithm.
#[derive(Clone, Copy, Debug, Default)]
pub struct SlidingWindowChunker;

impl ChunkingAlgorithm for SlidingWindowChunker {
    fn materialize(
        &self,
        strategy: &ChunkingStrategy,
        record: &DataRecord,
        section_idx: usize,
        section: &RecordSection,
    ) -> Vec<RecordChunk> {
        let raw_text = section.text.as_str();
        let denoised_owned: String;
        let text = match denoise_text(raw_text, &strategy.denoiser) {
            Some(cleaned) => {
                denoised_owned = cleaned;
                denoised_owned.as_str()
            }
            None => return Vec::new(),
        };
        let tokens: Vec<&str> = WhitespaceTokenizer.tokenize(text);
        if tokens.is_empty() {
            return Vec::new();
        }

        let mut chunks = Vec::new();
        let total_tokens = tokens.len();
        let span = strategy.max_window_tokens.min(total_tokens);
        if span == tokens.len() {
            let text = text.to_string();
            chunks.push(RecordChunk {
                record_id: record.id.clone(),
                section_idx,
                view: ChunkView::Window {
                    index: 0,
                    overlap: 0,
                    span,
                },
                text,
                tokens_estimate: span,
                quality: record.quality,
                kvp_meta: Default::default(),
            });
            return chunks;
        }

        for overlap in &strategy.overlap_tokens {
            let stride = span.saturating_sub(*overlap).max(1);
            let mut start = 0;
            let mut index = 0;
            while start < tokens.len() {
                let end = (start + span).min(tokens.len());
                let window = tokens[start..end].join(" ");
                chunks.push(RecordChunk {
                    record_id: record.id.clone(),
                    section_idx,
                    view: ChunkView::Window {
                        index,
                        overlap: *overlap,
                        span,
                    },
                    text: window,
                    tokens_estimate: end - start,
                    quality: record.quality,
                    kvp_meta: Default::default(),
                });
                if end == tokens.len() {
                    break;
                }
                start += stride;
                index += 1;
            }
        }

        if tokens.len() > strategy.max_window_tokens && strategy.summary_fallback_tokens > 0 {
            let fallback_cap = strategy
                .summary_fallback_tokens
                .min(strategy.max_window_tokens)
                .max(1);
            let fallback_len = tokens.len().min(fallback_cap);
            let summary_tokens = tokens
                .iter()
                .take(fallback_len)
                .copied()
                .collect::<Vec<_>>()
                .join(" ");
            chunks.push(RecordChunk {
                record_id: record.id.clone(),
                section_idx,
                view: ChunkView::SummaryFallback {
                    strategy: "head".into(),
                    weight: strategy.summary_fallback_weight,
                },
                text: summary_tokens,
                tokens_estimate: fallback_len,
                quality: record.quality,
                kvp_meta: Default::default(),
            });
        }

        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{QualityScore, RecordSection, SectionRole};
    use chrono::Utc;

    fn strategy() -> ChunkingStrategy {
        ChunkingStrategy {
            max_window_tokens: 4,
            overlap_tokens: vec![1],
            summary_fallback_weight: 0.3,
            summary_fallback_tokens: 2,
            chunk_weight_floor: 0.0,
            ..ChunkingStrategy::default()
        }
    }

    fn record(text: &str) -> DataRecord {
        DataRecord {
            id: "r1".into(),
            source: "unit".into(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            quality: QualityScore { trust: 1.0 },
            taxonomy: vec![],
            sections: vec![RecordSection {
                role: SectionRole::Context,
                heading: None,
                text: text.into(),
                sentences: vec![text.into()],
            }],
            meta_prefix: None,
        }
    }

    #[test]
    fn sliding_window_chunker_materializes_windows_and_summary() {
        let strategy = strategy();
        let record = record("one two three four five six seven");
        let section = &record.sections[0];
        let chunks = SlidingWindowChunker.materialize(&strategy, &record, 0, section);

        let window_count = chunks
            .iter()
            .filter(|chunk| matches!(chunk.view, ChunkView::Window { .. }))
            .count();
        let summary_count = chunks
            .iter()
            .filter(|chunk| matches!(chunk.view, ChunkView::SummaryFallback { .. }))
            .count();

        assert_eq!(window_count, 2);
        assert_eq!(summary_count, 1);
    }

    #[test]
    fn denoiser_disabled_produces_chunks_unchanged() {
        // Default strategy has denoiser disabled; numeric text should chunk normally.
        let strategy = strategy();
        let rec = record("42 524 10788 143 1995 190 394");
        let section = &rec.sections[0];
        let chunks = SlidingWindowChunker.materialize(&strategy, &rec, 0, section);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn denoiser_enabled_drops_pure_numeric_section() {
        use crate::config::DenoiserConfig;
        let mut strategy = strategy();
        strategy.denoiser = DenoiserConfig {
            enabled: true,
            max_digit_ratio: 0.35,
            line_level: false,
        };
        let rec = record("42 524 10788 143 1995 190 394 13611 358 6444 266");
        let section = &rec.sections[0];
        let chunks = SlidingWindowChunker.materialize(&strategy, &rec, 0, section);
        assert!(chunks.is_empty(), "digit-heavy section should produce no chunks");
    }

    #[test]
    fn denoiser_line_level_strips_noisy_lines_retaining_text() {
        use crate::config::DenoiserConfig;
        let mut strategy = strategy();
        strategy.denoiser = DenoiserConfig {
            enabled: true,
            max_digit_ratio: 0.35,
            line_level: true,
        };
        // The section contains one clean line and one numeric-heavy line.
        let rec = record("NOVEX INDUSTRIES Springfield\n42 524 10788 143 1995 190 394 13611 358");
        let section = &rec.sections[0];
        let chunks = SlidingWindowChunker.materialize(&strategy, &rec, 0, section);
        assert!(!chunks.is_empty(), "clean line should yield at least one chunk");
        let all_text: String = chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join(" ");
        assert!(all_text.contains("NOVEX") || all_text.contains("Springfield"));
    }
}
