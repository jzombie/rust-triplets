use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Once;

use clap::{Parser, ValueEnum, error::ErrorKind};

use crate::config::{ChunkingStrategy, SamplerConfig, TripletRecipe};
use crate::data::ChunkView;
use crate::heuristics::{
    CapacityTotals, EFFECTIVE_NEGATIVES_PER_ANCHOR, EFFECTIVE_POSITIVES_PER_ANCHOR,
    estimate_source_split_capacity_from_counts, format_replay_factor, format_u128_with_commas,
    resolve_text_recipes_for_source, split_counts_for_total,
};
use crate::metrics::source_skew;
use crate::sampler::chunk_weight;
use crate::source::DataSource;
use crate::splits::{FileSplitStore, SplitLabel, SplitRatios, SplitStore};
use crate::{
    RecordChunk, SampleBatch, Sampler, SamplerError, SourceId, TextBatch, TextRecipe, TripletBatch,
    TripletSampler,
};

type DynSource = Box<dyn DataSource + 'static>;

fn init_example_tracing() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("triplets=debug"));
        let _ = tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .try_init();
    });
}

#[derive(Debug, Clone, Copy, ValueEnum)]
/// CLI split selector mapped onto `SplitLabel`.
enum SplitArg {
    Train,
    Validation,
    Test,
}

impl From<SplitArg> for SplitLabel {
    fn from(value: SplitArg) -> Self {
        match value {
            SplitArg::Train => SplitLabel::Train,
            SplitArg::Validation => SplitLabel::Validation,
            SplitArg::Test => SplitLabel::Test,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "estimate_capacity",
    disable_help_subcommand = true,
    about = "Metadata-only capacity estimation",
    long_about = "Estimate record, pair, triplet, and text-sample capacity using source-reported counts only (no data refresh).",
    after_help = "Source roots are optional and resolved in order by explicit arg, environment variables, then project defaults."
)]
/// CLI arguments for metadata-only capacity estimation.
struct EstimateCapacityCli {
    #[arg(
        long,
        default_value_t = 99,
        help = "Deterministic seed used for split allocation"
    )]
    seed: u64,
    #[arg(
        long = "split-ratios",
        value_name = "TRAIN,VALIDATION,TEST",
        value_parser = parse_split_ratios_arg,
        default_value = "0.8,0.1,0.1",
        help = "Comma-separated split ratios that must sum to 1.0"
    )]
    split: SplitRatios,
    #[arg(
        long = "source-root",
        value_name = "PATH",
        help = "Optional source root override, repeat as needed in source order"
    )]
    source_roots: Vec<String>,
}

#[derive(Debug, Parser)]
#[command(
    name = "multi_source_demo",
    disable_help_subcommand = true,
    about = "Run sampled batches from multiple sources",
    long_about = "Sample triplet, pair, or text batches from multiple sources and persist split/epoch state.",
    after_help = "Source roots are optional and resolved in order by explicit arg, environment variables, then project defaults."
)]
/// CLI for `multi_source_demo`.
///
/// Common usage:
/// - Keep default persistence file location: `.sampler_store/split_store.bin`
/// - Set an explicit file path: `--split-store-path /tmp/split_store.bin`
/// - Set a custom directory and keep default filename: `--split-store-dir /tmp/sampler_store`
/// - Repeat `--source-root <PATH>` to override source roots in order
struct MultiSourceDemoCli {
    #[arg(
        long = "text-recipes",
        help = "Emit a text batch instead of a triplet batch"
    )]
    show_text_samples: bool,
    #[arg(
        long = "pair-batch",
        help = "Emit a pair batch instead of a triplet batch"
    )]
    show_pair_samples: bool,
    #[arg(
        long = "list-text-recipes",
        help = "Print registered text recipes and exit"
    )]
    list_text_recipes: bool,
    #[arg(
        long = "batch-size",
        default_value_t = 4,
        value_parser = parse_positive_usize,
        help = "Batch size used for sampling"
    )]
    batch_size: usize,
    #[arg(long, help = "Optional deterministic seed override")]
    seed: Option<u64>,
    #[arg(long, value_enum, help = "Target split to sample from")]
    split: Option<SplitArg>,
    #[arg(
        long = "source-root",
        value_name = "PATH",
        help = "Optional source root override, repeat as needed in source order"
    )]
    source_roots: Vec<String>,
    #[arg(
        long = "split-store-path",
        value_name = "SPLIT_STORE_PATH",
        help = "Optional path for persisted split/epoch state file"
    )]
    split_store_path: Option<PathBuf>,
    #[arg(
        long = "split-store-dir",
        value_name = "DIR",
        conflicts_with = "split_store_path",
        help = "Optional directory for persisted split/epoch state file (uses split_store.bin filename)"
    )]
    split_store_dir: Option<PathBuf>,
}

#[derive(Debug, Clone)]
/// Source-level inventory used by capacity estimation output.
struct SourceInventory {
    source_id: String,
    reported_records: u128,
    triplet_recipes: Vec<TripletRecipe>,
}

/// Run the capacity-estimation CLI with injectable root resolution/source builders.
///
/// `build_sources` is construction-only; sampler configuration is applied
/// centrally by this function before any source calls.
pub fn run_estimate_capacity<R, Resolve, Build, I>(
    args_iter: I,
    resolve_roots: Resolve,
    build_sources: Build,
) -> Result<(), Box<dyn Error>>
where
    Resolve: FnOnce(Vec<String>) -> Result<R, Box<dyn Error>>,
    Build: FnOnce(&R) -> Vec<DynSource>,
    I: Iterator<Item = String>,
{
    init_example_tracing();

    let Some(cli) = parse_cli::<EstimateCapacityCli, _>(
        std::iter::once("estimate_capacity".to_string()).chain(args_iter),
    )?
    else {
        return Ok(());
    };

    let roots = resolve_roots(cli.source_roots)?;

    let config = SamplerConfig {
        seed: cli.seed,
        split: cli.split,
        ..SamplerConfig::default()
    };

    let sources = build_sources(&roots);

    let mut inventories = Vec::new();
    for source in &sources {
        let recipes = if config.recipes.is_empty() {
            source.default_triplet_recipes()
        } else {
            config.recipes.clone()
        };
        let reported_records = source.reported_record_count(&config).map_err(|err| {
            format!(
                "source '{}' failed to report exact record count: {err}",
                source.id()
            )
        })?;
        inventories.push(SourceInventory {
            source_id: source.id().to_string(),
            reported_records,
            triplet_recipes: recipes,
        });
    }

    let mut per_source_split_counts: HashMap<(String, SplitLabel), u128> = HashMap::new();
    let mut split_record_counts: HashMap<SplitLabel, u128> = HashMap::new();

    for source in &inventories {
        let counts = split_counts_for_total(source.reported_records, cli.split);
        for (label, count) in counts {
            per_source_split_counts.insert((source.source_id.clone(), label), count);
            *split_record_counts.entry(label).or_insert(0) += count;
        }
    }

    let mut totals_by_split: HashMap<SplitLabel, CapacityTotals> = HashMap::new();
    let mut totals_by_source_and_split: HashMap<(String, SplitLabel), CapacityTotals> =
        HashMap::new();

    for split_label in [SplitLabel::Train, SplitLabel::Validation, SplitLabel::Test] {
        let mut totals = CapacityTotals::default();

        for source in &inventories {
            let source_split_records = per_source_split_counts
                .get(&(source.source_id.clone(), split_label))
                .copied()
                .unwrap_or(0);

            let triplet_recipes = &source.triplet_recipes;
            let text_recipes = resolve_text_recipes_for_source(&config, triplet_recipes);

            let capacity = estimate_source_split_capacity_from_counts(
                source_split_records,
                triplet_recipes,
                &text_recipes,
            );

            totals_by_source_and_split.insert((source.source_id.clone(), split_label), capacity);

            totals.triplets += capacity.triplets;
            totals.effective_triplets += capacity.effective_triplets;
            totals.pairs += capacity.pairs;
            totals.text_samples += capacity.text_samples;
        }

        totals_by_split.insert(split_label, totals);
    }

    println!("=== capacity estimate (length-only) ===");
    println!("mode: metadata-only (no source.refresh calls)");
    println!("classification: heuristic approximation (not exact)");
    println!("split seed: {}", cli.seed);
    println!(
        "split ratios: train={:.4}, validation={:.4}, test={:.4}",
        cli.split.train, cli.split.validation, cli.split.test
    );
    println!();

    println!("[SOURCES]");
    for source in &inventories {
        println!(
            "  {} => reported records: {}",
            source.source_id,
            format_u128_with_commas(source.reported_records)
        );
    }
    println!();

    println!("[PER SOURCE BREAKDOWN]");
    for source in &inventories {
        println!("  {}", source.source_id);
        let mut source_grand = CapacityTotals::default();
        let mut source_total_records = 0u128;
        for split_label in [SplitLabel::Train, SplitLabel::Validation, SplitLabel::Test] {
            let split_records = per_source_split_counts
                .get(&(source.source_id.clone(), split_label))
                .copied()
                .unwrap_or(0);
            source_total_records = source_total_records.saturating_add(split_records);
            let split_longest_records = inventories
                .iter()
                .map(|candidate| {
                    per_source_split_counts
                        .get(&(candidate.source_id.clone(), split_label))
                        .copied()
                        .unwrap_or(0)
                })
                .max()
                .unwrap_or(0);
            let totals = totals_by_source_and_split
                .get(&(source.source_id.clone(), split_label))
                .copied()
                .unwrap_or_default();
            source_grand.triplets += totals.triplets;
            source_grand.effective_triplets += totals.effective_triplets;
            source_grand.pairs += totals.pairs;
            source_grand.text_samples += totals.text_samples;
            println!("    [{:?}]", split_label);
            println!("      records: {}", format_u128_with_commas(split_records));
            println!(
                "      triplet combinations: {}",
                format_u128_with_commas(totals.triplets)
            );
            println!(
                "      effective sampled triplets (p={}, k={}): {}",
                EFFECTIVE_POSITIVES_PER_ANCHOR,
                EFFECTIVE_NEGATIVES_PER_ANCHOR,
                format_u128_with_commas(totals.effective_triplets)
            );
            println!(
                "      pair combinations:    {}",
                format_u128_with_commas(totals.pairs)
            );
            println!(
                "      text samples:         {}",
                format_u128_with_commas(totals.text_samples)
            );
            println!(
                "      replay factor vs longest source: {}",
                format_replay_factor(split_longest_records, split_records)
            );
        }
        let longest_source_total = inventories
            .iter()
            .map(|candidate| candidate.reported_records)
            .max()
            .unwrap_or(0);
        println!("    [ALL SPLITS FOR SOURCE]");
        println!(
            "      triplet combinations: {}",
            format_u128_with_commas(source_grand.triplets)
        );
        println!(
            "      effective sampled triplets (p={}, k={}): {}",
            EFFECTIVE_POSITIVES_PER_ANCHOR,
            EFFECTIVE_NEGATIVES_PER_ANCHOR,
            format_u128_with_commas(source_grand.effective_triplets)
        );
        println!(
            "      pair combinations:    {}",
            format_u128_with_commas(source_grand.pairs)
        );
        println!(
            "      text samples:         {}",
            format_u128_with_commas(source_grand.text_samples)
        );
        println!(
            "      replay factor vs longest source: {}",
            format_replay_factor(longest_source_total, source_total_records)
        );
        println!();
    }

    let mut grand = CapacityTotals::default();
    for split_label in [SplitLabel::Train, SplitLabel::Validation, SplitLabel::Test] {
        let record_count = split_record_counts.get(&split_label).copied().unwrap_or(0);
        let totals = totals_by_split
            .get(&split_label)
            .copied()
            .unwrap_or_default();

        grand.triplets += totals.triplets;
        grand.effective_triplets += totals.effective_triplets;
        grand.pairs += totals.pairs;
        grand.text_samples += totals.text_samples;

        println!("[{:?}]", split_label);
        println!("  records: {}", format_u128_with_commas(record_count));
        println!(
            "  triplet combinations: {}",
            format_u128_with_commas(totals.triplets)
        );
        println!(
            "  effective sampled triplets (p={}, k={}): {}",
            EFFECTIVE_POSITIVES_PER_ANCHOR,
            EFFECTIVE_NEGATIVES_PER_ANCHOR,
            format_u128_with_commas(totals.effective_triplets)
        );
        println!(
            "  pair combinations:    {}",
            format_u128_with_commas(totals.pairs)
        );
        println!(
            "  text samples:         {}",
            format_u128_with_commas(totals.text_samples)
        );
        println!();
    }

    println!("[ALL SPLITS TOTAL]");
    println!(
        "  triplet combinations: {}",
        format_u128_with_commas(grand.triplets)
    );
    println!(
        "  effective sampled triplets (p={}, k={}): {}",
        EFFECTIVE_POSITIVES_PER_ANCHOR,
        EFFECTIVE_NEGATIVES_PER_ANCHOR,
        format_u128_with_commas(grand.effective_triplets)
    );
    println!(
        "  pair combinations:    {}",
        format_u128_with_commas(grand.pairs)
    );
    println!(
        "  text samples:         {}",
        format_u128_with_commas(grand.text_samples)
    );
    println!();
    println!(
        "Note: counts are heuristic, length-based estimates from source-reported totals and recipe structure. They are approximate, not exact, and assume anchor-positive pairs=records (one positive per anchor by default), negatives=source_records_in_split-1 (anchor excluded as its own negative), and at most one chunk/window realization per sample. In real-world chunked sampling, practical combinations are often higher, so treat this as a floor-like baseline."
    );
    println!(
        "Effective sampled triplets apply a bounded training assumption: effective_triplets = records * p * k per triplet recipe, with defaults p={} positives per anchor and k={} negatives per anchor.",
        EFFECTIVE_POSITIVES_PER_ANCHOR, EFFECTIVE_NEGATIVES_PER_ANCHOR
    );
    println!(
        "Oversample loops are not inferred from this static report. To measure true oversampling (how many times sampling loops through the combination space), use observed sampled draw counts from an actual run."
    );

    Ok(())
}

/// Run the multi-source demo CLI with injectable root resolution/source builders.
///
/// `build_sources` is construction-only. Source sampler configuration is owned
/// by sampler registration (`TripletSampler::register_source`).
pub fn run_multi_source_demo<R, Resolve, Build, I>(
    args_iter: I,
    resolve_roots: Resolve,
    build_sources: Build,
) -> Result<(), Box<dyn Error>>
where
    Resolve: FnOnce(Vec<String>) -> Result<R, Box<dyn Error>>,
    Build: FnOnce(&R) -> Vec<DynSource>,
    I: Iterator<Item = String>,
{
    init_example_tracing();

    let Some(cli) = parse_cli::<MultiSourceDemoCli, _>(
        std::iter::once("multi_source_demo".to_string()).chain(args_iter),
    )?
    else {
        return Ok(());
    };

    let roots = resolve_roots(cli.source_roots)?;

    let mut config = SamplerConfig::default();
    config.seed = cli.seed.unwrap_or(config.seed);
    config.batch_size = cli.batch_size;
    config.chunking = Default::default();
    let selected_split = cli.split.map(Into::into).unwrap_or(SplitLabel::Train);
    config.split = SplitRatios::default();
    config.allowed_splits = vec![selected_split];
    let chunking = config.chunking.clone();

    let split_store_path = if let Some(path) = cli.split_store_path {
        path
    } else if let Some(dir) = cli.split_store_dir {
        FileSplitStore::default_path_in_dir(dir)
    } else {
        FileSplitStore::default_path()
    };

    println!(
        "Persisting split assignments and epoch state to {}",
        split_store_path.display()
    );
    let sources = build_sources(&roots);
    let split_store = Arc::new(FileSplitStore::open(&split_store_path, config.split, 99)?);
    let sampler = TripletSampler::new(config, split_store.clone());
    for source in sources {
        sampler.register_source(source);
    }

    if cli.show_pair_samples {
        match sampler.next_pair_batch(selected_split) {
            Ok(pair_batch) => {
                if pair_batch.pairs.is_empty() {
                    println!("Pair sampling produced no results.");
                } else {
                    print_pair_batch(&chunking, &pair_batch, split_store.as_ref());
                }
                sampler.persist_state()?;
            }
            Err(SamplerError::Exhausted(name)) => {
                eprintln!(
                    "Pair sampler exhausted recipe '{}'. Ensure both positive and negative examples exist.",
                    name
                );
            }
            Err(err) => return Err(err.into()),
        }
    } else if cli.show_text_samples {
        match sampler.next_text_batch(selected_split) {
            Ok(text_batch) => {
                if text_batch.samples.is_empty() {
                    println!(
                        "Text sampling produced no results. Ensure each source has eligible sections."
                    );
                } else {
                    print_text_batch(&chunking, &text_batch, split_store.as_ref());
                }
                sampler.persist_state()?;
            }
            Err(SamplerError::Exhausted(name)) => {
                eprintln!(
                    "Text sampler exhausted selector '{}'. Ensure matching sections exist.",
                    name
                );
            }
            Err(err) => return Err(err.into()),
        }
    } else if cli.list_text_recipes {
        let recipes = sampler.text_recipes();
        if recipes.is_empty() {
            println!(
                "No text recipes registered. Ensure your sources expose triplet selectors or configure text_recipes explicitly."
            );
        } else {
            print_text_recipes(&recipes);
        }
    } else {
        match sampler.next_triplet_batch(selected_split) {
            Ok(triplet_batch) => {
                if triplet_batch.triplets.is_empty() {
                    println!(
                        "Triplet sampling produced no results. Ensure multiple records per source exist."
                    );
                } else {
                    print_triplet_batch(&chunking, &triplet_batch, split_store.as_ref());
                }
                sampler.persist_state()?;
            }
            Err(SamplerError::Exhausted(name)) => {
                eprintln!(
                    "Triplet sampler exhausted recipe '{}'. Ensure both positive and negative examples exist.",
                    name
                );
            }
            Err(err) => return Err(err.into()),
        }
    }

    Ok(())
}

fn parse_positive_usize(raw: &str) -> Result<usize, String> {
    let parsed = raw.parse::<usize>().map_err(|_| {
        format!(
            "Could not parse --batch-size value '{}' as a positive integer",
            raw
        )
    })?;
    if parsed == 0 {
        return Err("--batch-size must be greater than zero".to_string());
    }
    Ok(parsed)
}

fn parse_cli<T, I>(args: I) -> Result<Option<T>, Box<dyn Error>>
where
    T: Parser,
    I: IntoIterator,
    I::Item: Into<std::ffi::OsString> + Clone,
{
    match T::try_parse_from(args) {
        Ok(cli) => Ok(Some(cli)),
        Err(err) => match err.kind() {
            ErrorKind::DisplayHelp | ErrorKind::DisplayVersion => {
                err.print()?;
                Ok(None)
            }
            _ => Err(err.into()),
        },
    }
}

fn parse_split_ratios_arg(raw: &str) -> Result<SplitRatios, String> {
    let parts: Vec<&str> = raw.split(',').collect();
    if parts.len() != 3 {
        return Err("--split-ratios expects exactly 3 comma-separated values".to_string());
    }
    let train = parts[0]
        .trim()
        .parse::<f32>()
        .map_err(|_| format!("invalid train ratio '{}': must be a float", parts[0].trim()))?;
    let validation = parts[1].trim().parse::<f32>().map_err(|_| {
        format!(
            "invalid validation ratio '{}': must be a float",
            parts[1].trim()
        )
    })?;
    let test = parts[2]
        .trim()
        .parse::<f32>()
        .map_err(|_| format!("invalid test ratio '{}': must be a float", parts[2].trim()))?;
    let ratios = SplitRatios {
        train,
        validation,
        test,
    };
    let sum = ratios.train + ratios.validation + ratios.test;
    if (sum - 1.0).abs() > 1e-5 {
        return Err(format!(
            "split ratios must sum to 1.0, got {:.6} (train={}, validation={}, test={})",
            sum, ratios.train, ratios.validation, ratios.test
        ));
    }
    if ratios.train < 0.0 || ratios.validation < 0.0 || ratios.test < 0.0 {
        return Err("split ratios must be non-negative".to_string());
    }
    Ok(ratios)
}

fn print_triplet_batch(
    strategy: &ChunkingStrategy,
    batch: &TripletBatch,
    split_store: &impl SplitStore,
) {
    println!("=== triplet batch ===");
    for (idx, triplet) in batch.triplets.iter().enumerate() {
        println!("--- triplet #{} ---", idx);
        println!("recipe       : {}", triplet.recipe);
        println!("sample_weight: {:.4}", triplet.weight);
        if let Some(instr) = &triplet.instruction {
            println!("instruction shown to model:\n{}\n", instr);
        }
        print_chunk_block("ANCHOR", &triplet.anchor, strategy, split_store);
        print_chunk_block("POSITIVE", &triplet.positive, strategy, split_store);
        print_chunk_block("NEGATIVE", &triplet.negative, strategy, split_store);
    }
    print_source_summary(
        "triplet anchors",
        batch
            .triplets
            .iter()
            .map(|triplet| triplet.anchor.record_id.as_str()),
    );
    print_recipe_summary_by_source(
        "triplet recipes by source",
        batch
            .triplets
            .iter()
            .map(|triplet| (triplet.anchor.record_id.as_str(), triplet.recipe.as_str())),
    );
}

fn print_text_batch(strategy: &ChunkingStrategy, batch: &TextBatch, split_store: &impl SplitStore) {
    println!("=== text batch ===");
    for (idx, sample) in batch.samples.iter().enumerate() {
        println!("--- sample #{} ---", idx);
        println!("recipe       : {}", sample.recipe);
        println!("sample_weight: {:.4}", sample.weight);
        if let Some(instr) = &sample.instruction {
            println!("instruction shown to model:\n{}\n", instr);
        }
        print_chunk_block("TEXT", &sample.chunk, strategy, split_store);
    }
    print_source_summary(
        "text samples",
        batch
            .samples
            .iter()
            .map(|sample| sample.chunk.record_id.as_str()),
    );
    print_recipe_summary_by_source(
        "text recipes by source",
        batch
            .samples
            .iter()
            .map(|sample| (sample.chunk.record_id.as_str(), sample.recipe.as_str())),
    );
}

fn print_pair_batch(
    strategy: &ChunkingStrategy,
    batch: &SampleBatch,
    split_store: &impl SplitStore,
) {
    println!("=== pair batch ===");
    for (idx, pair) in batch.pairs.iter().enumerate() {
        println!("--- pair #{} ---", idx);
        println!("recipe       : {}", pair.recipe);
        println!("label        : {:?}", pair.label);
        if let Some(reason) = &pair.reason {
            println!("reason       : {}", reason);
        }
        print_chunk_block("ANCHOR", &pair.anchor, strategy, split_store);
        print_chunk_block("OTHER", &pair.positive, strategy, split_store);
    }
    print_source_summary(
        "pair anchors",
        batch
            .pairs
            .iter()
            .map(|pair| pair.anchor.record_id.as_str()),
    );
    print_recipe_summary_by_source(
        "pair recipes by source",
        batch
            .pairs
            .iter()
            .map(|pair| (pair.anchor.record_id.as_str(), pair.recipe.as_str())),
    );
}

fn print_text_recipes(recipes: &[TextRecipe]) {
    println!("=== available text recipes ===");
    for recipe in recipes {
        println!(
            "- {} (weight: {:.3}) selector={:?}",
            recipe.name, recipe.weight, recipe.selector
        );
        if let Some(instr) = &recipe.instruction {
            println!("  instruction: {}", instr);
        }
    }
}

trait ChunkDebug {
    fn view_name(&self) -> String;
}

impl ChunkDebug for RecordChunk {
    fn view_name(&self) -> String {
        match &self.view {
            ChunkView::Window {
                index,
                span,
                overlap,
                start_ratio,
            } => format!(
                "window#index={} span={} overlap={} start_ratio={:.3} tokens={}",
                index, span, overlap, start_ratio, self.tokens_estimate
            ),
            ChunkView::SummaryFallback { strategy, .. } => {
                format!("summary:{} tokens={}", strategy, self.tokens_estimate)
            }
        }
    }
}

fn print_chunk_block(
    title: &str,
    chunk: &RecordChunk,
    strategy: &ChunkingStrategy,
    split_store: &impl SplitStore,
) {
    let chunk_weight = chunk_weight(strategy, chunk);
    let split = split_store
        .label_for(&chunk.record_id)
        .map(|label| format!("{:?}", label))
        .unwrap_or_else(|| "Unknown".to_string());
    println!("--- {} ---", title);
    println!("split        : {}", split);
    println!("view         : {}", chunk.view_name());
    println!("chunk_weight : {:.4}", chunk_weight);
    println!("record_id    : {}", chunk.record_id);
    println!("section_idx  : {}", chunk.section_idx);
    println!("token_est    : {}", chunk.tokens_estimate);
    println!("model_input (exact text sent to the model):");
    println!(
        "<<< BEGIN MODEL TEXT >>>\n{}\n<<< END MODEL TEXT >>>\n",
        chunk.text
    );
}

fn print_source_summary<'a, I>(label: &str, ids: I)
where
    I: Iterator<Item = &'a str>,
{
    let mut counts: HashMap<SourceId, usize> = HashMap::new();
    for id in ids {
        let source = extract_source(id);
        *counts.entry(source).or_insert(0) += 1;
    }
    if counts.is_empty() {
        return;
    }
    let skew = source_skew(&counts);
    let mut entries: Vec<(String, usize)> = counts.into_iter().collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    println!("--- {} by source ---", label);
    if let Some(skew) = skew {
        for entry in &skew.per_source {
            println!(
                "{}: count={} share={:.2}",
                entry.source, entry.count, entry.share
            );
        }
        println!(
            "skew: sources={} total={} min={} max={} mean={:.2} ratio={:.2}",
            skew.sources, skew.total, skew.min, skew.max, skew.mean, skew.ratio
        );
    } else {
        for (source, count) in &entries {
            println!("{source}: count={count}");
        }
    }
}

fn print_recipe_summary_by_source<'a, I>(label: &str, entries: I)
where
    I: Iterator<Item = (&'a str, &'a str)>,
{
    let mut counts: HashMap<SourceId, HashMap<String, usize>> = HashMap::new();
    for (record_id, recipe) in entries {
        let source = extract_source(record_id);
        let entry = counts
            .entry(source)
            .or_default()
            .entry(recipe.to_string())
            .or_insert(0);
        *entry += 1;
    }
    if counts.is_empty() {
        return;
    }
    let mut sources: Vec<(SourceId, HashMap<String, usize>)> = counts.into_iter().collect();
    sources.sort_by(|a, b| a.0.cmp(&b.0));
    println!("--- {} ---", label);
    for (source, recipes) in sources {
        println!("{source}");
        let mut entries: Vec<(String, usize)> = recipes.into_iter().collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        for (recipe, count) in entries {
            println!("  - {recipe}={count}");
        }
    }
}

fn extract_source(record_id: &str) -> SourceId {
    record_id
        .split_once("::")
        .map(|(source, _)| source.to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DeterministicSplitStore;
    use crate::data::SectionRole;
    use crate::source::{SourceCursor, SourceSnapshot};
    use chrono::Utc;
    use tempfile::tempdir;

    /// Minimal in-memory `DataSource` test double for example app tests.
    struct TestSource {
        id: String,
        count: Option<u128>,
        recipes: Vec<TripletRecipe>,
    }

    impl DataSource for TestSource {
        fn id(&self) -> &str {
            &self.id
        }

        fn refresh(
            &self,
            _config: &SamplerConfig,
            _cursor: Option<&SourceCursor>,
            _limit: Option<usize>,
        ) -> Result<SourceSnapshot, SamplerError> {
            Ok(SourceSnapshot {
                records: Vec::new(),
                cursor: SourceCursor {
                    last_seen: Utc::now(),
                    revision: 0,
                },
            })
        }

        fn reported_record_count(&self, _config: &SamplerConfig) -> Result<u128, SamplerError> {
            self.count.ok_or_else(|| SamplerError::SourceInconsistent {
                source_id: self.id.clone(),
                details: "test source has no configured exact count".to_string(),
            })
        }

        fn default_triplet_recipes(&self) -> Vec<TripletRecipe> {
            self.recipes.clone()
        }
    }

    struct ConfigRequiredSource {
        id: String,
        expected_seed: u64,
    }

    impl DataSource for ConfigRequiredSource {
        fn id(&self) -> &str {
            &self.id
        }

        fn refresh(
            &self,
            _config: &SamplerConfig,
            _cursor: Option<&SourceCursor>,
            _limit: Option<usize>,
        ) -> Result<SourceSnapshot, SamplerError> {
            Ok(SourceSnapshot {
                records: Vec::new(),
                cursor: SourceCursor {
                    last_seen: Utc::now(),
                    revision: 0,
                },
            })
        }

        fn reported_record_count(&self, config: &SamplerConfig) -> Result<u128, SamplerError> {
            if config.seed == self.expected_seed {
                Ok(1)
            } else {
                Err(SamplerError::SourceInconsistent {
                    source_id: self.id.clone(),
                    details: format!(
                        "expected sampler seed {} but got {}",
                        self.expected_seed, config.seed
                    ),
                })
            }
        }

        fn default_triplet_recipes(&self) -> Vec<TripletRecipe> {
            Vec::new()
        }
    }

    fn default_recipe(name: &str) -> TripletRecipe {
        TripletRecipe {
            name: name.to_string().into(),
            anchor: crate::config::Selector::Role(SectionRole::Anchor),
            positive_selector: crate::config::Selector::Role(SectionRole::Context),
            negative_selector: crate::config::Selector::Role(SectionRole::Context),
            negative_strategy: crate::config::NegativeStrategy::WrongArticle,
            weight: 1.0,
            instruction: None,
        }
    }

    #[test]
    fn parse_helpers_validate_inputs() {
        assert_eq!(parse_positive_usize("2").unwrap(), 2);
        assert!(parse_positive_usize("0").is_err());
        assert!(parse_positive_usize("abc").is_err());

        let split = parse_split_ratios_arg("0.8,0.1,0.1").unwrap();
        assert!((split.train - 0.8).abs() < 1e-6);
        assert!(parse_split_ratios_arg("0.8,0.1").is_err());
        assert!(parse_split_ratios_arg("1.0,0.0,0.1").is_err());
        assert!(parse_split_ratios_arg("-0.1,0.6,0.5").is_err());
    }

    #[test]
    fn parse_cli_handles_help_and_invalid_args() {
        let help = parse_cli::<EstimateCapacityCli, _>(["estimate_capacity", "--help"]).unwrap();
        assert!(help.is_none());

        let err = parse_cli::<EstimateCapacityCli, _>(["estimate_capacity", "--unknown"]);
        assert!(err.is_err());
    }

    #[test]
    fn run_estimate_capacity_succeeds_with_reported_counts() {
        let result = run_estimate_capacity(
            std::iter::empty::<String>(),
            |roots| {
                assert!(roots.is_empty());
                Ok(())
            },
            |_| {
                vec![Box::new(TestSource {
                    id: "source_a".into(),
                    count: Some(12),
                    recipes: vec![default_recipe("r1")],
                }) as DynSource]
            },
        );

        assert!(result.is_ok());
    }

    #[test]
    fn run_estimate_capacity_errors_when_source_count_missing() {
        let result = run_estimate_capacity(
            std::iter::empty::<String>(),
            |_| Ok(()),
            |_| {
                vec![Box::new(TestSource {
                    id: "source_missing".into(),
                    count: None,
                    recipes: vec![default_recipe("r1")],
                }) as DynSource]
            },
        );

        let err = result.unwrap_err().to_string();
        assert!(err.contains("failed to report exact record count"));
    }

    #[test]
    fn run_estimate_capacity_configures_sources_centrally_before_counting() {
        let result = run_estimate_capacity(
            std::iter::empty::<String>(),
            |_| Ok(()),
            |_| {
                vec![Box::new(ConfigRequiredSource {
                    id: "requires_config".into(),
                    expected_seed: 99,
                }) as DynSource]
            },
        );

        assert!(result.is_ok());
    }

    #[test]
    fn parse_multi_source_cli_handles_help_and_batch_size_validation() {
        let help = parse_cli::<MultiSourceDemoCli, _>(["multi_source_demo", "--help"]).unwrap();
        assert!(help.is_none());

        let err = parse_cli::<MultiSourceDemoCli, _>(["multi_source_demo", "--batch-size", "0"]);
        assert!(err.is_err());
    }

    #[test]
    fn run_multi_source_demo_list_text_recipes_path_succeeds() {
        let dir = tempdir().unwrap();
        let mut args = vec![
            "--list-text-recipes".to_string(),
            "--split-store-dir".to_string(),
            dir.path().to_string_lossy().to_string(),
        ];
        let result = run_multi_source_demo(
            args.drain(..),
            |_| Ok(()),
            |_| {
                vec![Box::new(TestSource {
                    id: "source_for_recipes".into(),
                    count: Some(10),
                    recipes: vec![default_recipe("recipe_a")],
                }) as DynSource]
            },
        );

        assert!(result.is_ok());
    }

    #[test]
    fn run_multi_source_demo_sampling_modes_handle_empty_sources() {
        for mode in [
            vec!["--pair-batch".to_string()],
            vec!["--text-recipes".to_string()],
            vec![],
        ] {
            let dir = tempdir().unwrap();
            let mut args = mode;
            args.push("--split-store-dir".to_string());
            args.push(dir.path().to_string_lossy().to_string());
            args.push("--split".to_string());
            args.push("validation".to_string());

            let result = run_multi_source_demo(
                args.into_iter(),
                |_| Ok(()),
                |_| {
                    vec![Box::new(TestSource {
                        id: "source_empty".into(),
                        count: Some(0),
                        recipes: vec![default_recipe("recipe_empty")],
                    }) as DynSource]
                },
            );

            assert!(result.is_ok());
        }
    }

    #[test]
    fn print_helpers_and_extract_source_cover_paths() {
        let split = SplitRatios::default();
        let store = DeterministicSplitStore::new(split, 42).unwrap();
        let strategy = ChunkingStrategy::default();

        let anchor = RecordChunk {
            record_id: "source_a::rec1".to_string(),
            section_idx: 0,
            view: ChunkView::Window {
                index: 1,
                overlap: 2,
                span: 12,
                start_ratio: 0.25,
            },
            text: "anchor text".to_string(),
            tokens_estimate: 8,
            quality: crate::data::QualityScore { trust: 0.9 },
        };
        let positive = RecordChunk {
            record_id: "source_a::rec2".to_string(),
            section_idx: 1,
            view: ChunkView::SummaryFallback {
                strategy: "summary".to_string(),
                weight: 0.7,
            },
            text: "positive text".to_string(),
            tokens_estimate: 6,
            quality: crate::data::QualityScore { trust: 0.8 },
        };
        let negative = RecordChunk {
            record_id: "source_b::rec3".to_string(),
            section_idx: 2,
            view: ChunkView::Window {
                index: 0,
                overlap: 0,
                span: 16,
                start_ratio: 0.0,
            },
            text: "negative text".to_string(),
            tokens_estimate: 7,
            quality: crate::data::QualityScore { trust: 0.5 },
        };

        let triplet_batch = TripletBatch {
            triplets: vec![crate::SampleTriplet {
                recipe: "triplet_recipe".to_string(),
                anchor: anchor.clone(),
                positive: positive.clone(),
                negative: negative.clone(),
                weight: 1.0,
                instruction: Some("triplet instruction".to_string()),
            }],
        };
        print_triplet_batch(&strategy, &triplet_batch, &store);

        let pair_batch = SampleBatch {
            pairs: vec![crate::SamplePair {
                recipe: "pair_recipe".to_string(),
                anchor: anchor.clone(),
                positive: positive.clone(),
                weight: 1.0,
                instruction: None,
                label: crate::PairLabel::Positive,
                reason: Some("same topic".to_string()),
            }],
        };
        print_pair_batch(&strategy, &pair_batch, &store);

        let text_batch = TextBatch {
            samples: vec![crate::TextSample {
                recipe: "text_recipe".to_string(),
                chunk: negative,
                weight: 0.8,
                instruction: Some("text instruction".to_string()),
            }],
        };
        print_text_batch(&strategy, &text_batch, &store);

        let recipes = vec![TextRecipe {
            name: "recipe_name".into(),
            selector: crate::config::Selector::Role(SectionRole::Context),
            instruction: Some("instruction".into()),
            weight: 1.0,
        }];
        print_text_recipes(&recipes);

        assert_eq!(extract_source("source_a::record"), "source_a");
        assert_eq!(extract_source("record-without-delimiter"), "unknown");
    }

    #[test]
    fn split_arg_conversion_and_version_parse_paths_are_covered() {
        assert!(matches!(
            SplitLabel::from(SplitArg::Train),
            SplitLabel::Train
        ));
        assert!(matches!(
            SplitLabel::from(SplitArg::Validation),
            SplitLabel::Validation
        ));
        assert!(matches!(SplitLabel::from(SplitArg::Test), SplitLabel::Test));
    }

    #[test]
    fn parse_split_ratios_reports_per_field_parse_errors() {
        assert!(
            parse_split_ratios_arg("x,0.1,0.9")
                .unwrap_err()
                .contains("invalid train ratio")
        );
        assert!(
            parse_split_ratios_arg("0.1,y,0.8")
                .unwrap_err()
                .contains("invalid validation ratio")
        );
        assert!(
            parse_split_ratios_arg("0.1,0.2,z")
                .unwrap_err()
                .contains("invalid test ratio")
        );
    }

    #[test]
    fn run_multi_source_demo_exhausted_paths_are_handled() {
        for mode in [
            vec!["--pair-batch".to_string()],
            vec!["--text-recipes".to_string()],
            Vec::new(),
        ] {
            let dir = tempdir().unwrap();
            let mut args = mode;
            args.push("--split-store-dir".to_string());
            args.push(dir.path().to_string_lossy().to_string());

            let result = run_multi_source_demo(
                args.into_iter(),
                |_| Ok(()),
                |_| {
                    vec![Box::new(TestSource {
                        id: "source_without_recipes".into(),
                        count: Some(1),
                        recipes: Vec::new(),
                    }) as DynSource]
                },
            );

            assert!(result.is_ok());
        }
    }
}
