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

# hf://allenai/c4/default text=text

# Single words (probably *never* use because BGE was never optimized for this)
# hf://pfox/71k-English-uncleaned-wordlist/default text=text

# TODO: Add different types of transcriptions

# Anchor basic relationships
# hf://roneneldan/TinyStories/default text=text

# Conversational tech
hf://labofsahil/hackernews-vector-search-dataset/default anchor=title positive=text

# Skeleton of most embedding spaces
hf://wikimedia/wikipedia/20231101.en anchor=title positive=text

# The "All-in-One" cleaned giant
# hf://cerebras/SlimPajama-627B/default text=text
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
        default_value_t = 2048,
        value_name = "N",
        help = "Sampler ingestion_max_records"
    )]
    ingestion_max_records: usize,
    #[arg(
        long,
        default_value_t = 1000,
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

    let roots = resolve_hf_list_roots(source_list_path.to_string_lossy().to_string())
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

    println!("running {} iterations of next_text_batch (batch_size={})...", cli.iterations, cli.batch_size);
    println!("watch .cache/triplets/ for new .simdr shard files appearing");
    println!();

    let run_start = Instant::now();
    let mut total_samples = 0usize;

    for iter in 0..cli.iterations {
        let started = Instant::now();
        let batch = sampler.next_text_batch(split)?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        total_samples += batch.samples.len();

        println!(
            "[{:>7.1}s] iter={:04} latency_ms={:>7.2} samples={} total_samples={}",
            run_start.elapsed().as_secs_f64(),
            iter + 1,
            elapsed_ms,
            batch.samples.len(),
            total_samples,
        );
    }

    println!();
    println!("done: {} iterations, {} total samples in {:.1}s",
        cli.iterations,
        total_samples,
        run_start.elapsed().as_secs_f64(),
    );

    sampler.save_sampler_state(None)?;
    Ok(())
}
