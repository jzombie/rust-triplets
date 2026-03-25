// TODO: Consider extracting to a debug crate

use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Once;
use std::time::Instant;

use cache_manager::CacheRoot;
use clap::{Parser, ValueEnum, error::ErrorKind};

use crate::config::{ChunkingStrategy, SamplerConfig, TripletRecipe};
use crate::constants::cache::{MULTI_SOURCE_DEMO_GROUP, MULTI_SOURCE_DEMO_STORE_FILENAME};
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

fn managed_demo_split_store_path() -> Result<PathBuf, String> {
    let cache_root = CacheRoot::from_discovery()
        .map_err(|err| format!("failed discovering managed cache root: {err}"))?;
    let group = PathBuf::from(MULTI_SOURCE_DEMO_GROUP);
    let dir = cache_root.ensure_group(&group).map_err(|err| {
        format!(
            "failed creating managed demo cache group '{}': {err}",
            group.display()
        )
    })?;
    Ok(dir.join(MULTI_SOURCE_DEMO_STORE_FILENAME))
}

fn init_example_tracing() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("triplets=info"));
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
/// - Use managed cache-group default path (no flag)
/// - Set an explicit file path: `--split-store-path /tmp/split_store.bin`
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
        value_parser = parse_batch_size,
        help = "Batch size used for sampling"
    )]
    batch_size: usize,
    #[arg(
        long = "ingestion-max-records",
        default_value_t = default_ingestion_max_records(),
        value_parser = parse_ingestion_max_records,
        help = "Per-source ingestion buffer target used while refreshing records"
    )]
    ingestion_max_records: usize,
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
        help = "Optional explicit path for persisted split/epoch state file"
    )]
    split_store_path: Option<PathBuf>,
    #[arg(
        long = "reset",
        help = "Delete the persisted split/epoch state before sampling, restarting from epoch 0"
    )]
    reset: bool,
    #[arg(
        long = "batches",
        value_name = "N",
        value_parser = parse_batch_count,
        help = "Run N triplet batches in succession, printing a timing line per batch and (with --features extended-metrics) a per-source similarity summary at the end"
    )]
    batches: Option<usize>,
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

    let min_nonzero_records_by_split: HashMap<SplitLabel, u128> =
        [SplitLabel::Train, SplitLabel::Validation, SplitLabel::Test]
            .into_iter()
            .map(|split_label| {
                let min_nonzero = inventories
                    .iter()
                    .filter_map(|source| {
                        per_source_split_counts
                            .get(&(source.source_id.clone(), split_label))
                            .copied()
                    })
                    .filter(|&records| records > 0)
                    .min()
                    .unwrap_or(0);
                (split_label, min_nonzero)
            })
            .collect();

    let min_nonzero_records_all_splits = inventories
        .iter()
        .map(|source| source.reported_records)
        .filter(|&records| records > 0)
        .min()
        .unwrap_or(0);

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
            println!(
                "      suggested proportional-size batch weight (0-1): {:.4}",
                suggested_balancing_weight(split_longest_records, split_records)
            );
            let split_smallest_nonzero = min_nonzero_records_by_split
                .get(&split_label)
                .copied()
                .unwrap_or(0);
            println!(
                "      suggested small-source-boost batch weight (0-1): {:.4}",
                suggested_oversampling_weight(split_smallest_nonzero, split_records)
            );
            println!();
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
        println!(
            "      suggested proportional-size batch weight (0-1): {:.4}",
            suggested_balancing_weight(longest_source_total, source_total_records)
        );
        println!(
            "      suggested small-source-boost batch weight (0-1): {:.4}",
            suggested_oversampling_weight(min_nonzero_records_all_splits, source_total_records)
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
    println!();
    println!(
        "Effective sampled triplets apply a bounded training assumption: effective_triplets = records * p * k per triplet recipe, with defaults p={} positives per anchor and k={} negatives per anchor.",
        EFFECTIVE_POSITIVES_PER_ANCHOR, EFFECTIVE_NEGATIVES_PER_ANCHOR
    );
    println!();
    println!(
        "Oversample loops are not inferred from this static report. To measure true oversampling (how many times sampling loops through the combination space), use observed sampled draw counts from an actual run."
    );
    println!();
    println!(
        "Suggested proportional-size batch weight (0-1) is source/max_source by record count: 1.0 for the largest source in scope, smaller values for smaller sources."
    );
    println!();
    println!(
        "Suggested small-source-boost batch weight (0-1) is min_nonzero_source/source by record count: 1.0 for the smallest non-zero source in scope, smaller values for larger sources."
    );
    println!();
    println!(
        "When passed to next_*_batch_with_weights, higher weight means that source is sampled more often relative to lower-weight sources."
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
    config.ingestion_max_records = cli.ingestion_max_records;
    config.chunking = Default::default();
    let selected_split = cli.split.map(Into::into).unwrap_or(SplitLabel::Train);
    config.split = SplitRatios::default();
    config.allowed_splits = vec![selected_split];
    let chunking = config.chunking.clone();
    let config_snapshot = MultiSourceDemoConfigSnapshot {
        seed: config.seed,
        batch_size: config.batch_size,
        ingestion_max_records: config.ingestion_max_records,
        split: selected_split,
        split_ratios: config.split,
        max_window_tokens: config.chunking.max_window_tokens,
        overlap_tokens: config.chunking.overlap_tokens.clone(),
        summary_fallback_tokens: config.chunking.summary_fallback_tokens,
    };

    let split_store_path = if let Some(path) = cli.split_store_path {
        path
    } else {
        managed_demo_split_store_path().map_err(|err| {
            Box::<dyn Error>::from(format!("failed to resolve demo split-store path: {err}"))
        })?
    };

    if cli.reset && split_store_path.exists() {
        std::fs::remove_file(&split_store_path).map_err(|err| {
            Box::<dyn Error>::from(format!(
                "failed to remove split store '{}': {err}",
                split_store_path.display()
            ))
        })?;
        println!("Reset: removed {}", split_store_path.display());
    }
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
                sampler.save_sampler_state(None)?;
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
                sampler.save_sampler_state(None)?;
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
    } else if let Some(batch_count) = cli.batches {
        print_demo_config(&config_snapshot);
        println!("=== benchmark: {} triplet batches ===", batch_count);

        // source_id -> Vec<(pos_jaccard, pos_byte_cosine, neg_jaccard, neg_byte_cosine)>
        #[cfg(feature = "extended-metrics")]
        let mut source_metrics: HashMap<String, Vec<(f32, f32, f32, f32)>> = HashMap::new();

        for i in 0..batch_count {
            let t0 = Instant::now();
            match sampler.next_triplet_batch(selected_split) {
                Ok(batch) => {
                    let elapsed = t0.elapsed();
                    let n = batch.triplets.len();
                    println!(
                        "batch {:>4}  triplets={:<4}  elapsed={:>8.2}ms  per_triplet={:.2}ms",
                        i + 1,
                        n,
                        elapsed.as_secs_f64() * 1000.0,
                        if n > 0 {
                            elapsed.as_secs_f64() * 1000.0 / n as f64
                        } else {
                            0.0
                        },
                    );
                    #[cfg(feature = "extended-metrics")]
                    {
                        use crate::metrics::lexical_similarity_scores;
                        for triplet in &batch.triplets {
                            let (pj, pc) = lexical_similarity_scores(
                                &triplet.anchor.text,
                                &triplet.positive.text,
                            );
                            let (nj, nc) = lexical_similarity_scores(
                                &triplet.anchor.text,
                                &triplet.negative.text,
                            );
                            let source = extract_source(&triplet.anchor.record_id);
                            source_metrics
                                .entry(source)
                                .or_default()
                                .push((pj, pc, nj, nc));
                        }
                    }
                }
                Err(SamplerError::Exhausted(name)) => {
                    println!(
                        "batch {:>4}  exhausted recipe '{}' — stopping early",
                        i + 1,
                        name
                    );
                    break;
                }
                Err(err) => return Err(err.into()),
            }
        }

        sampler.save_sampler_state(None)?;

        #[cfg(feature = "extended-metrics")]
        if !source_metrics.is_empty() {
            println!();
            print_metric_summary(&source_metrics);
        }

        #[cfg(all(feature = "extended-metrics", feature = "bm25-mining"))]
        {
            let (fallback, total) = sampler.bm25_fallback_stats();
            if total > 0 {
                let pct = fallback as f64 / total as f64 * 100.0;
                println!("bm25 fallback rate : {}/{} ({:.1}%)", fallback, total, pct);
            }
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
                sampler.save_sampler_state(None)?;
                #[cfg(all(feature = "extended-metrics", feature = "bm25-mining"))]
                {
                    let (fallback, total) = sampler.bm25_fallback_stats();
                    if total > 0 {
                        let pct = fallback as f64 / total as f64 * 100.0;
                        println!("bm25 fallback rate : {}/{} ({:.1}%)", fallback, total, pct);
                    }
                }
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

struct MultiSourceDemoConfigSnapshot {
    seed: u64,
    batch_size: usize,
    ingestion_max_records: usize,
    split: SplitLabel,
    split_ratios: SplitRatios,
    max_window_tokens: usize,
    overlap_tokens: Vec<usize>,
    summary_fallback_tokens: usize,
}

fn print_demo_config(cfg: &MultiSourceDemoConfigSnapshot) {
    let overlaps: Vec<String> = cfg.overlap_tokens.iter().map(|t| t.to_string()).collect();
    println!("=== sampler config ===");
    println!("seed                 : {}", cfg.seed);
    println!("batch_size           : {}", cfg.batch_size);
    println!("ingestion_max_records: {}", cfg.ingestion_max_records);
    println!("split                : {:?}", cfg.split);
    println!(
        "split_ratios         : train={:.2} val={:.2} test={:.2}",
        cfg.split_ratios.train, cfg.split_ratios.validation, cfg.split_ratios.test
    );
    println!("max_window_tokens    : {}", cfg.max_window_tokens);
    println!("overlap_tokens       : [{}]", overlaps.join(", "));
    println!(
        "summary_fallback     : {} tokens (0 = disabled)",
        cfg.summary_fallback_tokens
    );
    println!();
}

fn default_ingestion_max_records() -> usize {
    SamplerConfig::default().ingestion_max_records
}

fn parse_positive_usize_flag(raw: &str, flag: &str) -> Result<usize, String> {
    let parsed = raw.parse::<usize>().map_err(|_| {
        format!(
            "Could not parse {} value '{}' as a positive integer",
            flag, raw
        )
    })?;
    if parsed == 0 {
        return Err(format!("{} must be greater than zero", flag));
    }
    Ok(parsed)
}

fn parse_batch_size(raw: &str) -> Result<usize, String> {
    parse_positive_usize_flag(raw, "--batch-size")
}

fn parse_ingestion_max_records(raw: &str) -> Result<usize, String> {
    parse_positive_usize_flag(raw, "--ingestion-max-records")
}

fn parse_batch_count(raw: &str) -> Result<usize, String> {
    parse_positive_usize_flag(raw, "--batches")
}

fn suggested_balancing_weight(max_baseline: u128, source_baseline: u128) -> f32 {
    if max_baseline == 0 || source_baseline == 0 {
        return 0.0;
    }
    (source_baseline as f64 / max_baseline as f64).clamp(0.0, 1.0) as f32
}

fn suggested_oversampling_weight(min_nonzero_baseline: u128, source_baseline: u128) -> f32 {
    if min_nonzero_baseline == 0 || source_baseline == 0 {
        return 0.0;
    }
    (min_nonzero_baseline as f64 / source_baseline as f64).clamp(0.0, 1.0) as f32
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
        #[cfg(feature = "extended-metrics")]
        let (pos_sim, neg_sim) = {
            use crate::metrics::lexical_similarity_scores;
            (
                Some(lexical_similarity_scores(
                    &triplet.anchor.text,
                    &triplet.positive.text,
                )),
                Some(lexical_similarity_scores(
                    &triplet.anchor.text,
                    &triplet.negative.text,
                )),
            )
        };
        #[cfg(not(feature = "extended-metrics"))]
        let (pos_sim, neg_sim): (Option<(f32, f32)>, Option<(f32, f32)>) = (None, None);
        print_chunk_block("ANCHOR", &triplet.anchor, strategy, split_store, None);
        print_chunk_block(
            "POSITIVE",
            &triplet.positive,
            strategy,
            split_store,
            pos_sim,
        );
        print_chunk_block(
            "NEGATIVE",
            &triplet.negative,
            strategy,
            split_store,
            neg_sim,
        );
    }
    print_source_summary(
        "triplet anchors",
        batch
            .triplets
            .iter()
            .map(|triplet| triplet.anchor.record_id.as_str()),
    );
    print_recipe_context_by_source(
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
        print_chunk_block("TEXT", &sample.chunk, strategy, split_store, None);
    }
    print_source_summary(
        "text samples",
        batch
            .samples
            .iter()
            .map(|sample| sample.chunk.record_id.as_str()),
    );
    print_recipe_context_by_source(
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
        print_chunk_block("ANCHOR", &pair.anchor, strategy, split_store, None);
        print_chunk_block("OTHER", &pair.positive, strategy, split_store, None);
    }
    print_source_summary(
        "pair anchors",
        batch
            .pairs
            .iter()
            .map(|pair| pair.anchor.record_id.as_str()),
    );
    print_recipe_context_by_source(
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

#[cfg(feature = "extended-metrics")]
fn metric_mean_median(vals: &mut [f32]) -> (f32, f32) {
    let mean = vals.iter().sum::<f32>() / vals.len() as f32;
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if vals.len() % 2 == 1 {
        vals[vals.len() / 2]
    } else {
        (vals[vals.len() / 2 - 1] + vals[vals.len() / 2]) / 2.0
    };
    (mean, median)
}

#[cfg(feature = "extended-metrics")]
fn print_metric_summary(source_data: &HashMap<String, Vec<(f32, f32, f32, f32)>>) {
    let total: usize = source_data.values().map(|v| v.len()).sum();
    let n_sources = source_data.len();
    println!(
        "=== extended metrics summary ({} triplets, {} {}) ===",
        total,
        n_sources,
        if n_sources == 1 { "source" } else { "sources" }
    );

    // Returns [pos, neg] as (mean, median) pairs for one metric across entries.
    fn metric_pair(
        entries: &[(f32, f32, f32, f32)],
        pos_idx: usize,
        neg_idx: usize,
    ) -> [(f32, f32); 2] {
        let extract = |idx: usize| -> Vec<f32> {
            entries
                .iter()
                .map(|e| match idx {
                    0 => e.0,
                    1 => e.1,
                    2 => e.2,
                    _ => e.3,
                })
                .collect()
        };
        let mut pos_vals = extract(pos_idx);
        let mut neg_vals = extract(neg_idx);
        [
            metric_mean_median(&mut pos_vals),
            metric_mean_median(&mut neg_vals),
        ]
    }

    fn print_metric_section(
        label: &str,
        sources: &[&String],
        source_data: &HashMap<String, Vec<(f32, f32, f32, f32)>>,
        pos_idx: usize,
        neg_idx: usize,
        total: usize,
        n_sources: usize,
    ) {
        const SEP: usize = 83;
        println!();
        println!("[{}]", label);
        println!(
            "{:<24} {:>5}  {:<16} {:<16} {:<16}",
            "source", "n", "positive", "negative", "gap (pos\u{2212}neg)"
        );
        println!(
            "{:<24} {:>5}  {:<16} {:<16} {:<16}",
            "", "", "mean / median", "mean / median", "mean / median"
        );
        println!("{}", "-".repeat(SEP));
        for source in sources {
            let entries = &source_data[*source];
            let [pos, neg] = metric_pair(entries, pos_idx, neg_idx);
            let gap_mean = pos.0 - neg.0;
            let gap_med = pos.1 - neg.1;
            println!(
                "{:<24} {:>5}  {:.3} / {:.3}     {:.3} / {:.3}     {:+.3} / {:+.3}",
                source,
                entries.len(),
                pos.0,
                pos.1,
                neg.0,
                neg.1,
                gap_mean,
                gap_med,
            );
        }
        if n_sources > 1 {
            let all: Vec<(f32, f32, f32, f32)> = source_data.values().flatten().copied().collect();
            let [pos, neg] = metric_pair(&all, pos_idx, neg_idx);
            let gap_mean = pos.0 - neg.0;
            let gap_med = pos.1 - neg.1;
            println!("{}", "-".repeat(SEP));
            println!(
                "{:<24} {:>5}  {:.3} / {:.3}     {:.3} / {:.3}     {:+.3} / {:+.3}",
                "ALL", total, pos.0, pos.1, neg.0, neg.1, gap_mean, gap_med,
            );
        }
    }

    let mut sources: Vec<&String> = source_data.keys().collect();
    sources.sort();

    print_metric_section(
        "jaccard \u{2194} anchor",
        &sources,
        source_data,
        0,
        2,
        total,
        n_sources,
    );
    print_metric_section(
        "byte-cos \u{2194} anchor",
        &sources,
        source_data,
        1,
        3,
        total,
        n_sources,
    );
    println!();
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
    anchor_sim: Option<(f32, f32)>,
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
    if let Some((j, c)) = anchor_sim {
        println!("jaccard(↔a)  : {:.4}  byte-cos(↔a): {:.4}", j, c);
    }
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
    }
}

fn print_recipe_context_by_source<'a, I>(label: &str, entries: I)
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
    use crate::DataRecord;
    use crate::DeterministicSplitStore;
    use crate::data::{QualityScore, RecordSection, SectionRole};
    use crate::source::{SourceCursor, SourceSnapshot};
    use crate::utils::make_section;
    use chrono::{TimeZone, Utc};
    use tempfile::tempdir;

    fn empty_dyn_sources(_: &()) -> Vec<DynSource> {
        Vec::new()
    }

    fn ok_unit_roots(_: Vec<String>) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    fn error_unit_roots(_: Vec<String>) -> Result<(), Box<dyn Error>> {
        Err("root-resolution-error".into())
    }

    struct ErrorRefreshSource {
        id: String,
    }

    impl DataSource for ErrorRefreshSource {
        fn id(&self) -> &str {
            &self.id
        }

        fn refresh(
            &self,
            _config: &SamplerConfig,
            _cursor: Option<&SourceCursor>,
            _limit: Option<usize>,
        ) -> Result<SourceSnapshot, SamplerError> {
            Err(SamplerError::SourceUnavailable {
                source_id: self.id.clone(),
                reason: "simulated refresh failure".to_string(),
            })
        }

        fn reported_record_count(&self, _config: &SamplerConfig) -> Result<u128, SamplerError> {
            Ok(1)
        }

        fn default_triplet_recipes(&self) -> Vec<TripletRecipe> {
            vec![default_recipe("error_refresh_recipe")]
        }
    }

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

    struct FixtureSource {
        id: String,
        records: Vec<DataRecord>,
        recipes: Vec<TripletRecipe>,
    }

    impl DataSource for FixtureSource {
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
                records: self.records.clone(),
                cursor: SourceCursor {
                    last_seen: Utc::now(),
                    revision: 0,
                },
            })
        }

        fn reported_record_count(&self, _config: &SamplerConfig) -> Result<u128, SamplerError> {
            Ok(self.records.len() as u128)
        }

        fn default_triplet_recipes(&self) -> Vec<TripletRecipe> {
            self.recipes.clone()
        }
    }

    struct IngestionConfigSource {
        expected_ingestion_max_records: usize,
        records: Vec<DataRecord>,
    }

    impl DataSource for IngestionConfigSource {
        fn id(&self) -> &str {
            "ingestion_config_source"
        }

        fn refresh(
            &self,
            config: &SamplerConfig,
            _cursor: Option<&SourceCursor>,
            _limit: Option<usize>,
        ) -> Result<SourceSnapshot, SamplerError> {
            if config.ingestion_max_records != self.expected_ingestion_max_records {
                return Err(SamplerError::SourceInconsistent {
                    source_id: self.id().to_string(),
                    details: format!(
                        "expected ingestion_max_records {} but got {}",
                        self.expected_ingestion_max_records, config.ingestion_max_records
                    ),
                });
            }
            Ok(SourceSnapshot {
                records: self.records.clone(),
                cursor: SourceCursor {
                    last_seen: Utc::now(),
                    revision: 0,
                },
            })
        }

        fn reported_record_count(&self, _config: &SamplerConfig) -> Result<u128, SamplerError> {
            Ok(self.records.len() as u128)
        }

        fn default_triplet_recipes(&self) -> Vec<TripletRecipe> {
            vec![default_recipe("ingestion_config_recipe")]
        }
    }

    fn fixture_record(
        source: &str,
        id_suffix: &str,
        day: u32,
        title: &str,
        body: &str,
    ) -> DataRecord {
        let now = Utc.with_ymd_and_hms(2025, 1, day, 12, 0, 0).unwrap();
        DataRecord {
            id: format!("{source}::{id_suffix}"),
            source: source.to_string(),
            created_at: now,
            updated_at: now,
            quality: QualityScore { trust: 1.0 },
            taxonomy: Vec::new(),
            sections: vec![
                make_section(SectionRole::Anchor, Some("title"), title),
                make_section(SectionRole::Context, Some("body"), body),
            ],
            meta_prefix: None,
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
            allow_same_anchor_positive: false,
        }
    }

    #[test]
    fn parse_helpers_validate_inputs() {
        assert_eq!(parse_batch_size("2").unwrap(), 2);
        assert!(parse_batch_size("0").is_err());
        assert!(parse_batch_size("abc").is_err());
        assert_eq!(parse_ingestion_max_records("16").unwrap(), 16);
        assert!(parse_ingestion_max_records("0").is_err());
        assert!(parse_batch_count("0").is_err());

        let split = parse_split_ratios_arg("0.8,0.1,0.1").unwrap();
        assert!((split.train - 0.8).abs() < 1e-6);
        assert!(parse_split_ratios_arg("0.8,0.1").is_err());
        assert!(parse_split_ratios_arg("1.0,0.0,0.1").is_err());
        assert!(parse_split_ratios_arg("-0.1,0.6,0.5").is_err());
    }

    #[test]
    fn fixture_and_ingestion_sources_trait_methods_cover_paths() {
        let records = vec![fixture_record("fixture_source", "r1", 1, "Title", "Body")];
        let recipes = vec![default_recipe("fixture_recipe")];
        let fixture = FixtureSource {
            id: "fixture_source".into(),
            records: records.clone(),
            recipes: recipes.clone(),
        };

        let snapshot = fixture
            .refresh(&SamplerConfig::default(), None, None)
            .expect("fixture refresh should succeed");
        assert_eq!(snapshot.records.len(), 1);
        assert_eq!(
            fixture
                .reported_record_count(&SamplerConfig::default())
                .unwrap(),
            1
        );
        assert_eq!(fixture.default_triplet_recipes().len(), 1);

        let source = IngestionConfigSource {
            expected_ingestion_max_records: 7,
            records,
        };
        let ok_cfg = SamplerConfig {
            ingestion_max_records: 7,
            ..SamplerConfig::default()
        };
        assert!(source.refresh(&ok_cfg, None, None).is_ok());
        assert_eq!(source.reported_record_count(&ok_cfg).unwrap(), 1);
        assert_eq!(source.default_triplet_recipes().len(), 1);

        let bad_cfg = SamplerConfig {
            ingestion_max_records: 8,
            ..SamplerConfig::default()
        };
        let err = source.refresh(&bad_cfg, None, None).unwrap_err();
        assert!(matches!(err, SamplerError::SourceInconsistent { .. }));
    }

    #[test]
    fn suggested_balancing_weight_is_longest_normalized_and_bounded() {
        assert!((suggested_balancing_weight(100, 100) - 1.0).abs() < 1e-6);
        assert!((suggested_balancing_weight(400, 100) - 0.25).abs() < 1e-6);
        assert!((suggested_balancing_weight(400, 400) - 1.0).abs() < 1e-6);
        assert_eq!(suggested_balancing_weight(0, 100), 0.0);
        assert_eq!(suggested_balancing_weight(100, 0), 0.0);
    }

    #[test]
    fn suggested_oversampling_weight_is_inverse_in_unit_interval() {
        assert!((suggested_oversampling_weight(100, 100) - 1.0).abs() < 1e-6);
        assert!((suggested_oversampling_weight(100, 400) - 0.25).abs() < 1e-6);
        assert!((suggested_oversampling_weight(100, 1000) - 0.1).abs() < 1e-6);
        assert_eq!(suggested_oversampling_weight(0, 100), 0.0);
        assert_eq!(suggested_oversampling_weight(100, 0), 0.0);
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
    fn run_estimate_capacity_propagates_root_resolution_error() {
        let result = run_estimate_capacity(
            std::iter::empty::<String>(),
            |_| Err("root resolution failed".into()),
            empty_dyn_sources,
        );

        let err = result.unwrap_err().to_string();
        assert!(err.contains("root resolution failed"));
    }

    #[test]
    fn run_estimate_capacity_allows_empty_source_list() {
        let result =
            run_estimate_capacity(std::iter::empty::<String>(), |_| Ok(()), empty_dyn_sources);

        assert!(result.is_ok());
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
    fn config_required_source_refresh_and_seed_mismatch_are_exercised() {
        let source = ConfigRequiredSource {
            id: "cfg-source".to_string(),
            expected_seed: 42,
        };

        let refreshed = source
            .refresh(&SamplerConfig::default(), None, None)
            .unwrap();
        assert!(refreshed.records.is_empty());

        let mismatched = source.reported_record_count(&SamplerConfig {
            seed: 7,
            ..SamplerConfig::default()
        });
        assert!(matches!(
            mismatched,
            Err(SamplerError::SourceInconsistent { .. })
        ));

        assert!(source.default_triplet_recipes().is_empty());
    }

    #[test]
    fn run_multi_source_demo_exhausted_paths_return_ok() {
        struct OneRecordSource;

        impl DataSource for OneRecordSource {
            fn id(&self) -> &str {
                "one_record"
            }

            fn refresh(
                &self,
                _config: &SamplerConfig,
                _cursor: Option<&SourceCursor>,
                _limit: Option<usize>,
            ) -> Result<SourceSnapshot, SamplerError> {
                let now = Utc::now();
                Ok(SourceSnapshot {
                    records: vec![DataRecord {
                        id: "one_record::r1".to_string(),
                        source: "one_record".to_string(),
                        created_at: now,
                        updated_at: now,
                        quality: QualityScore { trust: 1.0 },
                        taxonomy: Vec::new(),
                        sections: vec![
                            RecordSection {
                                role: SectionRole::Anchor,
                                heading: Some("title".to_string()),
                                text: "anchor".to_string(),
                                sentences: vec!["anchor".to_string()],
                            },
                            RecordSection {
                                role: SectionRole::Context,
                                heading: Some("body".to_string()),
                                text: "context".to_string(),
                                sentences: vec!["context".to_string()],
                            },
                        ],
                        meta_prefix: None,
                    }],
                    cursor: SourceCursor {
                        last_seen: now,
                        revision: 0,
                    },
                })
            }

            fn reported_record_count(&self, _config: &SamplerConfig) -> Result<u128, SamplerError> {
                Ok(1)
            }

            fn default_triplet_recipes(&self) -> Vec<TripletRecipe> {
                vec![default_recipe("single_record_recipe")]
            }
        }

        let one = OneRecordSource;
        assert_eq!(
            one.reported_record_count(&SamplerConfig::default())
                .unwrap(),
            1
        );
        assert_eq!(one.default_triplet_recipes().len(), 1);

        for mode in ["--pair-batch", "--text-recipes", ""] {
            let dir = tempdir().unwrap();
            let split_store_path = dir.path().join("split_store.bin");
            let mut args = vec![
                "--split-store-path".to_string(),
                split_store_path.to_string_lossy().to_string(),
            ];
            if !mode.is_empty() {
                args.push(mode.to_string());
            }

            let result = run_multi_source_demo(
                args.into_iter(),
                |_| Ok(()),
                |_| vec![Box::new(OneRecordSource) as DynSource],
            );
            assert!(result.is_ok());
        }
    }

    #[test]
    fn parse_multi_source_cli_handles_help_and_batch_size_validation() {
        let help = parse_cli::<MultiSourceDemoCli, _>(["multi_source_demo", "--help"]).unwrap();
        assert!(help.is_none());

        let err = parse_cli::<MultiSourceDemoCli, _>(["multi_source_demo", "--batch-size", "0"]);
        assert!(err.is_err());

        let err = parse_cli::<MultiSourceDemoCli, _>([
            "multi_source_demo",
            "--ingestion-max-records",
            "0",
        ]);
        assert!(err.is_err());

        let parsed = parse_cli::<MultiSourceDemoCli, _>(["multi_source_demo"]);
        assert!(parsed.is_ok());
    }

    #[test]
    fn run_example_apps_invalid_cli_args_return_errors() {
        let estimate = run_estimate_capacity(
            ["--unknown".to_string()].into_iter(),
            ok_unit_roots,
            empty_dyn_sources,
        );
        assert!(estimate.is_err());

        let demo = run_multi_source_demo(
            ["--unknown".to_string()].into_iter(),
            ok_unit_roots,
            empty_dyn_sources,
        );
        assert!(demo.is_err());
    }

    #[test]
    fn helper_and_error_refresh_source_methods_are_exercised() {
        assert!(ok_unit_roots(Vec::new()).is_ok());
        assert!(error_unit_roots(Vec::new()).is_err());

        let source = ErrorRefreshSource {
            id: "error_refresh_source".to_string(),
        };
        assert_eq!(
            source
                .reported_record_count(&SamplerConfig::default())
                .unwrap(),
            1
        );
        assert_eq!(source.default_triplet_recipes().len(), 1);
    }

    #[test]
    fn print_source_summary_handles_non_empty_ids() {
        let ids = [
            "source_a::r1",
            "source_a::r2",
            "source_b::r1",
            "source_without_delimiter",
        ];
        print_source_summary("non-empty summary", ids.into_iter());
    }

    #[test]
    fn run_multi_source_demo_refresh_failures_degrade_to_exhausted_paths() {
        for mode in [
            vec!["--pair-batch".to_string()],
            vec!["--text-recipes".to_string()],
            vec!["--batches".to_string(), "1".to_string()],
            Vec::new(),
        ] {
            let dir = tempdir().unwrap();
            let split_store_path = dir.path().join("error_modes_split_store.bin");
            let mut args = mode;
            args.push("--split-store-path".to_string());
            args.push(split_store_path.to_string_lossy().to_string());

            let result = run_multi_source_demo(
                args.into_iter(),
                |_| Ok(()),
                |_| {
                    vec![Box::new(ErrorRefreshSource {
                        id: "error_refresh_source".to_string(),
                    }) as DynSource]
                },
            );

            assert!(result.is_ok());
        }
    }

    #[test]
    fn run_multi_source_demo_batches_exhausted_path_returns_ok() {
        let dir = tempdir().unwrap();
        let split_store_path = dir.path().join("batches_exhausted_split_store.bin");
        let args = vec![
            "--batches".to_string(),
            "3".to_string(),
            "--split-store-path".to_string(),
            split_store_path.to_string_lossy().to_string(),
        ];

        let result = run_multi_source_demo(
            args.into_iter(),
            |_| Ok(()),
            |_| {
                vec![Box::new(FixtureSource {
                    id: "batches_exhausted_source".into(),
                    records: vec![fixture_record(
                        "batches_exhausted_source",
                        "r1",
                        1,
                        "Only one record",
                        "Single record body",
                    )],
                    recipes: vec![default_recipe("batches_exhausted_recipe")],
                }) as DynSource]
            },
        );

        assert!(result.is_ok());
    }

    #[test]
    fn run_multi_source_demo_default_triplet_success_path_returns_ok() {
        let dir = tempdir().unwrap();
        let split_store_path = dir.path().join("default_triplet_success_split_store.bin");
        let args = vec![
            "--split-store-path".to_string(),
            split_store_path.to_string_lossy().to_string(),
        ];

        let result = run_multi_source_demo(
            args.into_iter(),
            |_| Ok(()),
            |_| {
                vec![Box::new(FixtureSource {
                    id: "default_triplet_success_source".into(),
                    records: vec![
                        fixture_record(
                            "default_triplet_success_source",
                            "r1",
                            1,
                            "Title one",
                            "Body one",
                        ),
                        fixture_record(
                            "default_triplet_success_source",
                            "r2",
                            2,
                            "Title two",
                            "Body two",
                        ),
                        fixture_record(
                            "default_triplet_success_source",
                            "r3",
                            3,
                            "Title three",
                            "Body three",
                        ),
                    ],
                    recipes: vec![default_recipe("default_triplet_success_recipe")],
                }) as DynSource]
            },
        );

        assert!(result.is_ok());
    }

    #[test]
    fn run_multi_source_demo_passes_ingestion_max_records_to_sources() {
        let dir = tempdir().unwrap();
        let split_store_path = dir.path().join("ingestion_config_split_store.bin");
        let expected = 7;

        let result = run_multi_source_demo(
            [
                "--pair-batch".to_string(),
                "--ingestion-max-records".to_string(),
                expected.to_string(),
                "--split-store-path".to_string(),
                split_store_path.to_string_lossy().to_string(),
            ]
            .into_iter(),
            |_| Ok(()),
            |_| {
                vec![Box::new(IngestionConfigSource {
                    expected_ingestion_max_records: expected,
                    records: (1..=8)
                        .map(|day| {
                            fixture_record(
                                "ingestion_config_source",
                                &format!("r{day}"),
                                day,
                                &format!("Config headline {day}"),
                                &format!("Config body {day}"),
                            )
                        })
                        .collect(),
                }) as DynSource]
            },
        );

        assert!(result.is_ok());
    }

    #[test]
    fn parse_cli_handles_display_version_path() {
        #[derive(Debug, Parser)]
        #[command(name = "version_test", version = "1.0.0")]
        struct VersionCli {}

        let parsed = parse_cli::<VersionCli, _>(["version_test", "--version"]).unwrap();
        assert!(parsed.is_none());
    }

    #[test]
    fn run_multi_source_demo_list_text_recipes_path_succeeds() {
        let dir = tempdir().unwrap();
        let split_store_path = dir.path().join("recipes_split_store.bin");
        let mut args = vec![
            "--list-text-recipes".to_string(),
            "--split-store-path".to_string(),
            split_store_path.to_string_lossy().to_string(),
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
    fn run_multi_source_demo_list_text_recipes_uses_explicit_split_store_path() {
        let dir = tempdir().unwrap();
        let split_store_path = dir.path().join("custom_split_store.bin");
        let args = vec![
            "--list-text-recipes".to_string(),
            "--split-store-path".to_string(),
            split_store_path.to_string_lossy().to_string(),
        ];

        let result = run_multi_source_demo(
            args.into_iter(),
            |_| Ok(()),
            |_| {
                vec![Box::new(TestSource {
                    id: "source_without_text_recipes".into(),
                    count: Some(1),
                    recipes: Vec::new(),
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
            let split_store_path = dir.path().join("empty_sources_split_store.bin");
            let mut args = mode;
            args.push("--split-store-path".to_string());
            args.push(split_store_path.to_string_lossy().to_string());
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
    fn run_multi_source_demo_propagates_root_resolution_error() {
        let dir = tempdir().unwrap();
        let split_store_path = dir.path().join("root_resolution_error_store.bin");
        let result = run_multi_source_demo(
            [
                "--split-store-path".to_string(),
                split_store_path.to_string_lossy().to_string(),
            ]
            .into_iter(),
            |_| Err("demo root resolution failed".into()),
            empty_dyn_sources,
        );

        let err = result.unwrap_err().to_string();
        assert!(err.contains("demo root resolution failed"));
    }

    #[test]
    fn run_multi_source_demo_list_text_recipes_allows_empty_sources() {
        let dir = tempdir().unwrap();
        let split_store_path = dir.path().join("empty_source_list_recipes.bin");
        let result = run_multi_source_demo(
            [
                "--list-text-recipes".to_string(),
                "--split-store-path".to_string(),
                split_store_path.to_string_lossy().to_string(),
            ]
            .into_iter(),
            |_| Ok(()),
            empty_dyn_sources,
        );

        assert!(result.is_ok());
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
            let split_store_path = dir.path().join("exhausted_split_store.bin");
            let mut args = mode;
            args.push("--split-store-path".to_string());
            args.push(split_store_path.to_string_lossy().to_string());

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

    #[test]
    fn run_multi_source_demo_reset_recreates_split_store_and_samples() {
        let dir = tempdir().unwrap();
        let split_store_path = dir.path().join("reset_split_store.bin");
        std::fs::write(&split_store_path, b"stale-data").unwrap();

        let args = vec![
            "--reset".to_string(),
            "--pair-batch".to_string(),
            "--split-store-path".to_string(),
            split_store_path.to_string_lossy().to_string(),
        ];

        let result = run_multi_source_demo(
            args.into_iter(),
            |_| Ok(()),
            |_| {
                let recipes = vec![default_recipe("fixture_recipe")];
                let records: Vec<DataRecord> = (1..=8)
                    .map(|day| {
                        fixture_record(
                            "fixture_source",
                            &format!("r{day}"),
                            day,
                            &format!("Fixture headline {day}"),
                            &format!("Fixture body content for day {day}."),
                        )
                    })
                    .collect();
                vec![Box::new(FixtureSource {
                    id: "fixture_source".into(),
                    records,
                    recipes,
                }) as DynSource]
            },
        );

        assert!(result.is_ok());
        assert!(split_store_path.exists());
        let metadata = std::fs::metadata(&split_store_path).unwrap();
        assert!(metadata.len() > 0);
    }

    #[test]
    fn run_multi_source_demo_batches_mode_executes_multiple_batches() {
        let dir = tempdir().unwrap();
        let split_store_path = dir.path().join("batches_split_store.bin");
        let args = vec![
            "--batches".to_string(),
            "2".to_string(),
            "--split-store-path".to_string(),
            split_store_path.to_string_lossy().to_string(),
        ];

        let result = run_multi_source_demo(
            args.into_iter(),
            |_| Ok(()),
            |_| {
                let recipes = vec![default_recipe("batch_recipe")];
                vec![Box::new(FixtureSource {
                    id: "batch_source".into(),
                    records: vec![
                        fixture_record(
                            "batch_source",
                            "r1",
                            3,
                            "Inflation cools in latest report",
                            "Core inflation moderated compared with prior quarter.",
                        ),
                        fixture_record(
                            "batch_source",
                            "r2",
                            4,
                            "Labor market remains resilient",
                            "Job openings remain elevated despite slower growth.",
                        ),
                        fixture_record(
                            "batch_source",
                            "r3",
                            5,
                            "Manufacturing sentiment stabilizes",
                            "Survey data suggests output expectations are improving.",
                        ),
                    ],
                    recipes,
                }) as DynSource]
            },
        );

        assert!(result.is_ok());
        assert!(split_store_path.exists());
    }

    #[test]
    fn managed_demo_split_store_path_resolves_under_cache_group() {
        let path = managed_demo_split_store_path().unwrap();
        assert!(path.ends_with(MULTI_SOURCE_DEMO_STORE_FILENAME));
        let parent = path
            .parent()
            .expect("managed split-store path should have a parent");
        assert!(parent.ends_with(PathBuf::from(MULTI_SOURCE_DEMO_GROUP)));
    }

    #[test]
    fn run_multi_source_demo_help_returns_ok_without_work() {
        let no_help = run_multi_source_demo(
            std::iter::empty::<String>(),
            error_unit_roots,
            empty_dyn_sources,
        );
        assert!(
            no_help
                .expect_err("non-help path should attempt to resolve roots")
                .to_string()
                .contains("root-resolution-error")
        );

        let result = run_multi_source_demo(
            ["--help".to_string()].into_iter(),
            ok_unit_roots,
            empty_dyn_sources,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn run_estimate_capacity_help_returns_ok_without_work() {
        let result = run_estimate_capacity(
            ["--help".to_string()].into_iter(),
            ok_unit_roots,
            empty_dyn_sources,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn run_multi_source_demo_pair_exhausted_branch_returns_ok() {
        let dir = tempdir().unwrap();
        let split_store_path = dir.path().join("pair_exhausted_split_store.bin");
        let args = vec![
            "--pair-batch".to_string(),
            "--split-store-path".to_string(),
            split_store_path.to_string_lossy().to_string(),
        ];

        let result = run_multi_source_demo(
            args.into_iter(),
            |_| Ok(()),
            |_| {
                vec![Box::new(FixtureSource {
                    id: "pair_exhausted_source".into(),
                    records: vec![fixture_record(
                        "pair_exhausted_source",
                        "r1",
                        1,
                        "Single record title",
                        "Single record body",
                    )],
                    recipes: vec![default_recipe("pair_exhausted_recipe")],
                }) as DynSource]
            },
        );

        assert!(result.is_ok());
    }

    #[test]
    fn run_multi_source_demo_uses_managed_split_store_path_when_not_provided() {
        let result = run_multi_source_demo(
            ["--list-text-recipes".to_string()].into_iter(),
            |_| Ok(()),
            |_| {
                vec![Box::new(TestSource {
                    id: "managed_path_source".into(),
                    count: Some(2),
                    recipes: vec![default_recipe("managed_recipe")],
                }) as DynSource]
            },
        );

        assert!(result.is_ok());
    }

    #[test]
    fn run_multi_source_demo_reset_errors_when_target_is_directory() {
        let dir = tempdir().unwrap();
        let split_store_path = dir.path().join("split_store_dir");
        std::fs::create_dir(&split_store_path).unwrap();

        let result = run_multi_source_demo(
            [
                "--reset".to_string(),
                "--split-store-path".to_string(),
                split_store_path.to_string_lossy().to_string(),
            ]
            .into_iter(),
            |_| Ok(()),
            empty_dyn_sources,
        );

        let err = result.unwrap_err().to_string();
        assert!(err.contains("failed to remove split store"));
    }

    #[test]
    fn print_summary_helpers_accept_empty_iterators() {
        print_source_summary("empty summary", std::iter::empty::<&str>());
        print_recipe_context_by_source("empty recipe context", std::iter::empty::<(&str, &str)>());
    }

    #[cfg(feature = "extended-metrics")]
    #[test]
    fn metric_mean_median_handles_even_length_inputs() {
        let mut vals = [1.0, 4.0, 2.0, 3.0];
        let (mean, median) = metric_mean_median(&mut vals);
        assert!((mean - 2.5).abs() < 1e-6);
        assert!((median - 2.5).abs() < 1e-6);
    }

    #[cfg(feature = "extended-metrics")]
    #[test]
    fn metric_mean_median_handles_odd_length_inputs() {
        let mut vals = [3.0, 1.0, 2.0];
        let (mean, median) = metric_mean_median(&mut vals);
        assert!((mean - 2.0).abs() < 1e-6);
        assert!((median - 2.0).abs() < 1e-6);
    }

    #[cfg(feature = "extended-metrics")]
    #[test]
    fn print_metric_summary_includes_multi_source_aggregate() {
        let source_data = HashMap::from([
            (
                "source_a".to_string(),
                vec![(0.9, 0.8, 0.2, 0.1), (0.8, 0.7, 0.3, 0.2)],
            ),
            (
                "source_b".to_string(),
                vec![(0.7, 0.6, 0.4, 0.3), (0.6, 0.5, 0.5, 0.4)],
            ),
        ]);

        print_metric_summary(&source_data);
    }
}
