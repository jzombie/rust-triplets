//! Ingest a local directory of parquet files and print sampled records.
//!
//! Usage:
//! ```sh
//! cargo run -p triplets-parquet-source --example local_ingest -- <dir> [options]
//! ```
//!
//! Options (repeatable column flags; all configuration is CLI args,
//! no environment variables):
//! ```text
//! --schema           print file schemas (columns, rows, row groups) and exit
//! --text COL       text-columns mode candidate (default: body)
//! --anchor COL     role-mode anchor candidates (switches to role mode)
//! --positive COL   role-mode positive candidates
//! --context COL    role-mode context columns
//! --id COL         row id column (default: id)
//! --limit N        max records to fetch (default: 10)
//! --namespace NS   synthetic-id namespace (default: local)
//! ```
//!
//! Example:
//! ```sh
//! cargo run -p triplets-parquet-source --example local_ingest -- \
//!     crates/triplets-parquet-source/example-dataset
//! cargo run -p triplets-parquet-source --example local_ingest -- \
//!     crates/triplets-parquet-source/example-dataset --anchor title --positive body
//! ```

use std::error::Error;
use std::path::PathBuf;

use triplets_core::config::SamplerConfig;
use triplets_core::source::DataSource;
use triplets_parquet_source::{ParquetRowSource, ParquetRowsConfig};

#[derive(Default)]
struct Args {
    dir: PathBuf,
    schema_only: bool,
    text: Vec<String>,
    anchor: Vec<String>,
    positive: Vec<String>,
    context: Vec<String>,
    id: Option<String>,
    limit: usize,
    namespace: String,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut args = Args {
        limit: 10,
        namespace: "local".to_string(),
        ..Args::default()
    };
    let mut it = std::env::args().skip(1).peekable();
    let dir = it.next().ok_or("usage: local_ingest <dir> [options]")?;
    args.dir = PathBuf::from(dir);
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--schema" => args.schema_only = true,
            "--text" => args.text.push(it.next().ok_or("--text needs a value")?),
            "--anchor" => args.anchor.push(it.next().ok_or("--anchor needs a value")?),
            "--positive" => args.positive.push(it.next().ok_or("--positive needs a value")?),
            "--context" => args.context.push(it.next().ok_or("--context needs a value")?),
            "--id" => args.id = Some(it.next().ok_or("--id needs a value")?),
            "--limit" => {
                args.limit = it
                    .next()
                    .ok_or("--limit needs a value")?
                    .parse()
                    .map_err(|_| "--limit needs an integer")?
            }
            "--namespace" => args.namespace = it.next().ok_or("--namespace needs a value")?,
            other => return Err(format!("unknown flag: {other}").into()),
        }
    }
    if args.text.is_empty() && args.anchor.is_empty() && args.positive.is_empty() {
        args.text.push("body".to_string());
    }
    Ok(args)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;

    if args.schema_only {
        return print_schemas(&args.dir);
    }

    let mut config = ParquetRowsConfig::new(
        "local_example",
        args.dir.clone(),
        args.namespace.clone(),
        vec!["example=local_ingest".to_string()],
    );
    config.text_columns = args.text.clone();
    config.anchor_columns = args.anchor.clone();
    config.positive_columns = args.positive.clone();
    config.context_columns = args.context.clone();
    if let Some(id) = args.id.clone() {
        config.id_column = Some(id);
    }

    let source = ParquetRowSource::new(config)?;
    println!(
        "indexed {} rows in {} shards from {}",
        source.materialized_row_count(),
        source.shard_count(),
        args.dir.display()
    );

    let snapshot = source.refresh(&SamplerConfig::default(), None, Some(args.limit))?;
    println!("fetched {} records:", snapshot.records.len());
    for record in &snapshot.records {
        println!("== {}  taxonomy={:?}", record.id, record.taxonomy);
        for section in &record.sections {
            let preview: String = section.text.chars().take(120).collect();
            println!(
                "   [{:?} {}] {preview}",
                section.role,
                section.heading.as_deref().unwrap_or("-")
            );
        }
    }
    Ok(())
}

fn print_schemas(dir: &std::path::Path) -> Result<(), Box<dyn Error>> {
    use parquet::file::reader::{FileReader, SerializedFileReader};

    let mut paths: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("parquet"))
        })
        .collect();
    paths.sort();
    for path in paths {
        let file = std::fs::File::open(&path)?;
        let reader = SerializedFileReader::new(file)?;
        let meta = reader.metadata();
        println!("== {}", path.display());
        println!(
            "   bytes={} rows={} row_groups={}",
            std::fs::metadata(&path)?.len(),
            meta.file_metadata().num_rows(),
            meta.num_row_groups()
        );
        for field in meta.file_metadata().schema().get_fields() {
            println!("   col: {field:?}");
        }
    }
    Ok(())
}
