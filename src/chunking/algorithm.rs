use crate::config::ChunkingStrategy;
use crate::data::{DataRecord, RecordChunk, RecordSection};

/// Pluggable algorithm for materializing chunks from a record section.
pub trait ChunkingAlgorithm: Send + Sync {
    /// Produce candidate chunks for one section of a record.
    fn materialize(
        &self,
        strategy: &ChunkingStrategy,
        record: &DataRecord,
        section_idx: usize,
        section: &RecordSection,
    ) -> Vec<RecordChunk>;
}
