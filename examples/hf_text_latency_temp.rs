#![cfg_attr(not(feature = "huggingface"), allow(dead_code, unused_imports))]

use std::error::Error;

#[cfg(feature = "huggingface")]
use clap::{Parser, ValueEnum};
#[cfg(feature = "huggingface")]
use std::fs;
#[cfg(feature = "huggingface")]
use std::path::PathBuf;
#[cfg(feature = "huggingface")]
use std::sync::Arc;
#[cfg(feature = "huggingface")]
use std::time::Instant;
#[cfg(feature = "huggingface")]
use triplets::{
    FileSplitStore, Sampler, SamplerConfig, SplitLabel, SplitRatios, TripletSampler,
    build_hf_sources, resolve_hf_list_roots,
};

#[cfg(feature = "huggingface")]
const DEFAULT_SOURCE_LIST: &str = r#"# triplets Hugging Face source-list format:
# hf://org/dataset/config/split anchor=... positive=... context=a,b text=x,y
#
# Rules:
# - comments start with '#'
# - at least one mapping key is required per line
# - accepted keys: anchor= positive= context= text=
# - context/text accept comma-delimited columns

# hf://allenai/c4/default/train text=text

# Single words (probably *never* use because BGE was never optimized for this)
# hf://pfox/71k-English-uncleaned-wordlist/default/train text=text

# TODO: Add different types of transcriptions

# Anchor basic relationships
# hf://roneneldan/TinyStories/default/train text=text

# Conversational tech
hf://labofsahil/hackernews-vector-search-dataset/default/train anchor=title positive=text

# Skeleton of most embedding spaces
hf://wikimedia/wikipedia/20231101.en/train anchor=title positive=text

# The "All-in-One" cleaned giant
# hf://cerebras/SlimPajama-627B/default/train text=text
"#;

#[cfg(feature = "huggingface")]
#[derive(Clone, Copy, Debug, ValueEnum)]
enum SplitArg {
    Train,
    Validation,
    Test,
}

#[cfg(feature = "huggingface")]
impl From<SplitArg> for SplitLabel {
    fn from(value: SplitArg) -> Self {
        match value {
            SplitArg::Train => SplitLabel::Train,
            SplitArg::Validation => SplitLabel::Validation,
            SplitArg::Test => SplitLabel::Test,
        }
    }
}

#[cfg(feature = "huggingface")]
#[derive(Debug, Parser)]
#[command(
    name = "hf_text_latency_temp",
    disable_help_subcommand = true,
    about = "Temporary latency probe for repeated next_text_batch on HF source-list inputs"
)]
struct HfTextLatencyCli {
    #[arg(
        long,
        default_value_t = 64,
        value_name = "N",
        help = "Text batch size per next_text_batch call"
    )]
    batch_size: usize,
    #[arg(
        long,
        default_value_t = 1024,
        value_name = "N",
        help = "Sampler ingestion_max_records"
    )]
    ingestion_max_records: usize,
    #[arg(
        long,
        default_value_t = 20,
        value_name = "N",
        help = "Number of timed next_text_batch calls"
    )]
    iterations: usize,
    #[arg(long, default_value_t = 99, value_name = "N", help = "Sampler seed")]
    seed: u64,
    #[arg(
        long,
        value_enum,
        default_value = "train",
        help = "Split to sample from"
    )]
    split: SplitArg,
    #[arg(
        long = "max-rows-per-source",
        default_value_t = 2048,
        value_name = "N",
        help = "Per-source row cap while probing"
    )]
    max_rows_per_source: usize,
    #[arg(
        long = "source-list",
        value_name = "PATH",
        help = "Optional existing source-list path; if omitted, writes your requested list to .cache/triplets/tmp_hf_text_latency_sources.txt"
    )]
    source_list: Option<PathBuf>,
}

#[cfg(not(feature = "huggingface"))]
fn main() {
    eprintln!("hf_text_latency_temp requires --features huggingface");
}

#[cfg(feature = "huggingface")]
fn percentile(sorted_ms: &[f64], p: f64) -> f64 {
    if sorted_ms.is_empty() {
        return 0.0;
    }
    let p = p.clamp(0.0, 100.0);
    let idx = ((p / 100.0) * ((sorted_ms.len() - 1) as f64)).round() as usize;
    sorted_ms[idx]
}

#[cfg(feature = "huggingface")]
fn write_default_source_list() -> Result<PathBuf, Box<dyn Error>> {
    let path = PathBuf::from(".cache/triplets/tmp_hf_text_latency_sources.txt");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, DEFAULT_SOURCE_LIST)?;
    Ok(path)
}

#[cfg(feature = "huggingface")]
fn main() -> Result<(), Box<dyn Error>> {
    let cli = HfTextLatencyCli::parse();
    let split: SplitLabel = cli.split.into();

    let source_list_path = if let Some(path) = cli.source_list {
        path
    } else {
        write_default_source_list()?
    };

    println!("== hf_text_latency_temp ==");
    println!("source_list: {}", source_list_path.display());
    println!("batch_size: {}", cli.batch_size);
    println!("ingestion_max_records: {}", cli.ingestion_max_records);
    println!("iterations: {}", cli.iterations);
    println!("split: {:?}", split);
    println!("max_rows_per_source: {}", cli.max_rows_per_source);

    let roots = resolve_hf_list_roots(
        source_list_path.to_string_lossy().to_string(),
        Some(cli.max_rows_per_source),
    )
    .map_err(|err| -> Box<dyn Error> { err.into() })?;
    println!("resolved_sources: {}", roots.sources.len());

    let sources = build_hf_sources(&roots);

    let mut config = SamplerConfig::default();
    config.seed = cli.seed;
    config.batch_size = cli.batch_size;
    config.ingestion_max_records = cli.ingestion_max_records;
    config.split = SplitRatios::default();
    config.allowed_splits = vec![split];

    let split_store_path = PathBuf::from(".cache/triplets/tmp_hf_text_latency_split_store.bin");
    if let Some(parent) = split_store_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let split_store = Arc::new(FileSplitStore::open(
        &split_store_path,
        config.split,
        config.seed,
    )?);
    let sampler = TripletSampler::new(config, split_store.clone());
    for source in sources {
        sampler.register_source(source);
    }

    println!("warming up first next_text_batch...");
    let warmup_started = Instant::now();
    let warmup = sampler.next_text_batch(split)?;
    let warmup_ms = warmup_started.elapsed().as_secs_f64() * 1000.0;
    println!(
        "warmup_ms={:.2} samples={}",
        warmup_ms,
        warmup.samples.len()
    );

    let mut latencies_ms = Vec::with_capacity(cli.iterations);
    for iter in 0..cli.iterations {
        let started = Instant::now();
        let batch = sampler.next_text_batch(split)?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        latencies_ms.push(elapsed_ms);
        println!(
            "iter={:03} latency_ms={:.2} samples={}",
            iter + 1,
            elapsed_ms,
            batch.samples.len()
        );
    }

    if !latencies_ms.is_empty() {
        let mut sorted = latencies_ms.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let sum: f64 = latencies_ms.iter().sum();
        let mean = sum / (latencies_ms.len() as f64);
        let min = *sorted.first().unwrap_or(&0.0);
        let max = *sorted.last().unwrap_or(&0.0);

        println!("--- latency summary (ms) ---");
        println!("count={}", latencies_ms.len());
        println!("min={:.2}", min);
        println!("mean={:.2}", mean);
        println!("p50={:.2}", percentile(&sorted, 50.0));
        println!("p95={:.2}", percentile(&sorted, 95.0));
        println!("p99={:.2}", percentile(&sorted, 99.0));
        println!("max={:.2}", max);
    }

    sampler.save_sampler_state(None)?;
    Ok(())
}
