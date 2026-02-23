use triplets::{
    DataSource, DeterministicSplitStore, HuggingFaceRowSource, HuggingFaceRowsConfig,
    MappedRowViewAdapter, PairSampler, RowFieldMapping, RowViewDataSourceAdapter, Sampler,
    SamplerConfig, SplitLabel, SplitRatios, TextFieldPolicy,
};
use std::path::PathBuf;

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source_id = "hf_rows".to_string();
    let dataset = "HuggingFaceFW/fineweb".to_string();
    let config_name = "default".to_string();
    let split_name = "train".to_string();
    let snapshot_default = PathBuf::from(".hf-snapshots")
        .join(dataset.replace('/', "__"))
        .join(&config_name)
        .join(&split_name);
    let snapshot_dir = snapshot_default;
    let text_columns = vec!["text".to_string(), "text".to_string()];

    let mut hf = HuggingFaceRowsConfig::new(
        source_id.clone(),
        dataset.clone(),
        config_name.clone(),
        split_name.clone(),
        snapshot_dir.clone(),
    );
    hf.shard_extensions = vec!["parquet".to_string(), "jsonl".to_string(), "ndjson".to_string()];
    hf.text_columns = text_columns.clone();

    println!(
        "Using Hugging Face bulk snapshot rows from: {} ({}) split={} dir={}",
        dataset,
        config_name,
        split_name,
        snapshot_dir.display()
    );

    let source = HuggingFaceRowSource::new(hf)?;

    let mapping = RowFieldMapping {
        anchor: TextFieldPolicy::Explicit(vec![text_columns[0].clone()]),
        positive: TextFieldPolicy::Explicit(vec![text_columns[1].clone()]),
        context: TextFieldPolicy::RemainingTextColumns,
        ..RowFieldMapping::default()
    };

    let mapper = MappedRowViewAdapter::new(source_id, mapping);
    let data_source: Box<dyn DataSource> = Box::new(RowViewDataSourceAdapter::new(source, mapper));

    let split = SplitRatios {
        train: 1.0,
        validation: 0.0,
        test: 0.0,
    };
    let split_store = DeterministicSplitStore::new(split, 13)?;
    let config = SamplerConfig {
        seed: 13,
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
