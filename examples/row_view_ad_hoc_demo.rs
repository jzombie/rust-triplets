use chrono::{TimeZone, Utc};
use std::collections::BTreeMap;
use triplets::{
    DataSource, DeterministicSplitStore, MappedRowViewAdapter, PairSampler, RowFieldMapping,
    RowView, RowViewDataSourceAdapter, RowViewSource, Sampler, SamplerConfig, SplitLabel,
    SplitRatios, TextField, TextFieldPolicy,
};

#[derive(Clone)]
struct BackendRow {
    row_id: String,
    timestamp: chrono::DateTime<Utc>,
    fields: BTreeMap<String, String>,
}

trait RowBackend {
    fn total_rows(&self) -> usize;
    fn fetch_row(&self, idx: usize) -> Result<Option<BackendRow>, triplets::SamplerError>;
}

struct InMemoryBackend {
    rows: Vec<BackendRow>,
}

impl RowBackend for InMemoryBackend {
    fn total_rows(&self) -> usize {
        self.rows.len()
    }

    fn fetch_row(&self, idx: usize) -> Result<Option<BackendRow>, triplets::SamplerError> {
        Ok(self.rows.get(idx).cloned())
    }
}

struct AdHocRowSource<B: RowBackend> {
    id: String,
    backend: B,
    text_columns: Vec<String>,
}

fn preview(text: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for ch in text.chars().take(max_chars) {
        out.push(ch);
    }
    if text.chars().count() > max_chars {
        out.push_str("...");
    }
    out
}

fn print_chunk(label: &str, chunk: &triplets::RecordChunk) {
    let local_id = chunk
        .record_id
        .split_once("::")
        .map(|(_, local)| local)
        .unwrap_or(chunk.record_id.as_str());
    println!("  {label}");
    println!("    record id  : {}", chunk.record_id);
    println!("    local id   : {local_id}");
    println!("    section idx: {}", chunk.section_idx);
    println!("    text       : {}", preview(&chunk.text, 90));
}

impl<B: RowBackend + Send + Sync> RowViewSource for AdHocRowSource<B> {
    fn id(&self) -> &str {
        &self.id
    }

    fn len_hint(&self) -> Option<usize> {
        Some(self.backend.total_rows())
    }

    fn row_at(&self, idx: usize) -> Result<Option<RowView>, triplets::SamplerError> {
        let Some(row) = self.backend.fetch_row(idx)? else {
            return Ok(None);
        };

        let mut text_fields = Vec::with_capacity(self.text_columns.len());
        for column in &self.text_columns {
            let Some(value) = row.fields.get(column) else {
                return Err(triplets::SamplerError::SourceInconsistent {
                    source_id: self.id.clone(),
                    details: format!("missing configured text column '{column}' in row {}", row.row_id),
                });
            };
            text_fields.push(TextField {
                name: column.clone(),
                text: value.clone(),
            });
        }

        Ok(Some(RowView {
            row_id: Some(row.row_id),
            timestamp: Some(row.timestamp),
            text_fields,
        }))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let timestamp = Utc.with_ymd_and_hms(2025, 2, 23, 12, 0, 0).unwrap();
    let backend = InMemoryBackend {
        rows: vec![
            BackendRow {
                row_id: "row_1".to_string(),
                timestamp,
                fields: BTreeMap::from([
                    ("question".to_string(), "How does deterministic sampling work?".to_string()),
                    (
                        "answer".to_string(),
                        "Deterministic sampling uses stable IDs and a seed.".to_string(),
                    ),
                    ("title".to_string(), "Deterministic sampling basics".to_string()),
                    ("body".to_string(), "Extra context column from ad-hoc source.".to_string()),
                ]),
            },
            BackendRow {
                row_id: "row_2".to_string(),
                timestamp,
                fields: BTreeMap::from([
                    ("question".to_string(), "What is split persistence?".to_string()),
                    (
                        "answer".to_string(),
                        "It stores split labels and sampler progress across restarts.".to_string(),
                    ),
                    ("title".to_string(), "Persistence model".to_string()),
                    ("body".to_string(), "Another ad-hoc field that becomes context.".to_string()),
                ]),
            },
            BackendRow {
                row_id: "row_3".to_string(),
                timestamp,
                fields: BTreeMap::from([
                    ("question".to_string(), "Why use row-view adapters?".to_string()),
                    (
                        "answer".to_string(),
                        "They keep source-specific parsing separate from sampler logic.".to_string(),
                    ),
                    ("title".to_string(), "Adapter rationale".to_string()),
                    (
                        "body".to_string(),
                        "This source is in-memory but follows the same contract.".to_string(),
                    ),
                ]),
            },
        ],
    };

    let source = AdHocRowSource {
        id: "ad_hoc_row_source".to_string(),
        backend,
        text_columns: vec![
            "question".to_string(),
            "answer".to_string(),
            "title".to_string(),
            "body".to_string(),
        ],
    };
    let mapping = RowFieldMapping {
        anchor: TextFieldPolicy::Explicit(vec!["question".to_string()]),
        positive: TextFieldPolicy::Explicit(vec!["answer".to_string()]),
        context: TextFieldPolicy::RemainingTextColumns,
        ..RowFieldMapping::default()
    };
    let mapper = MappedRowViewAdapter::new(source.id().to_string(), mapping);
    let data_source: Box<dyn DataSource> = Box::new(RowViewDataSourceAdapter::new(source, mapper));

    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };
    let split_store = DeterministicSplitStore::new(split, 7)?;
    let config = SamplerConfig {
        seed: 7,
        batch_size: 2,
        split,
        allowed_splits: vec![SplitLabel::Train],
        ..SamplerConfig::default()
    };

    let sampler = PairSampler::new(config, std::sync::Arc::new(split_store));
    sampler.register_source(data_source);

    let batch = sampler.next_triplet_batch(SplitLabel::Train)?;
    println!("Generated {} triplets", batch.triplets.len());
    for (idx, triplet) in batch.triplets.iter().enumerate() {
        println!("triplet #{idx} (recipe: {})", triplet.recipe);
        print_chunk("anchor", &triplet.anchor);
        print_chunk("positive", &triplet.positive);
        print_chunk("negative", &triplet.negative);
        println!();
    }

    Ok(())
}
