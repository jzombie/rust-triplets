#![cfg_attr(not(feature = "huggingface"), allow(dead_code, unused_imports))]

use std::error::Error;

#[cfg(feature = "huggingface")]
use clap::Parser;
#[cfg(feature = "huggingface")]
use triplets::{HfListRoots, build_hf_sources, parse_hf_source_line, resolve_hf_list_roots};

#[cfg(feature = "huggingface")]
#[derive(Debug, Parser)]
#[command(
    name = "hf_source_list_demo",
    disable_help_subcommand = true,
    about = "Run multi_source_demo using Hugging Face source-list roots",
    long_about = "Resolve Hugging Face source-list entries, then forward remaining args to multi_source_demo.\n\
                  Provide `--entry` to test a single source directly without a source-list file.",
    after_help = "Examples:\n  # Load sources from a file\n  hf_source_list_demo\n\n  # Paste a single source-line directly\n  hf_source_list_demo --entry 'hf://wikimedia/wikipedia/20231101.en/train anchor=title positive=text'\n\n  # With passthrough args\n  hf_source_list_demo --entry 'hf://labofsahil/hackernews-vector-search-dataset/default text=title,text' -- --batch-size 8\n\nNote: Use `--` before forwarded multi_source_demo args (for example: -- --batch-size 8 --split train)."
)]
struct HfSourceListDemoCli {
    #[arg(
        long = "source-list",
        default_value = "examples/common/hf_sources.txt",
        required = false,
        value_name = "PATH",
        help = "Path to Hugging Face source-list file (ignored when --entry is provided)"
    )]
    source_list: String,
    #[arg(
        long = "entry",
        required = false,
        value_name = "SOURCE_LINE",
        help = "Full source-line string exactly as it would appear in a source-list file\n\
                (e.g. 'hf://org/dataset/config/split text=title,text')."
    )]
    entry: Option<String>,
    #[arg(
        last = true,
        value_name = "ARGS",
        help = "Arguments forwarded to multi_source_demo after `--`"
    )]
    passthrough: Vec<String>,
}

#[cfg(not(feature = "huggingface"))]
fn main() {
    eprintln!("hf_source_list_demo requires --features huggingface");
}

#[cfg(feature = "huggingface")]
fn main() -> Result<(), Box<dyn Error>> {
    let parsed = HfSourceListDemoCli::parse();

    let roots = if let Some(entry) = &parsed.entry {
        // Paste a full source-line directly.
        let entry = parse_hf_source_line(entry).map_err(|err| format!("invalid source: {err}"))?;
        let roots = HfListRoots {
            source_list: entry.uri.clone(),
            sources: vec![entry],
        };
        println!("== hf_source_list_demo (direct entry mode) ==");
        roots
    } else {
        let roots = resolve_hf_list_roots(parsed.source_list.clone())
            .map_err(|err| -> Box<dyn Error> { err.into() })?;
        println!("== hf_source_list_demo (source-list mode) ==");
        println!("source_list: {}", roots.source_list);
        roots
    };

    println!("sources: {}", roots.sources.len());
    println!(
        "forwarding args to multi_source_demo: {:?}",
        parsed.passthrough
    );

    triplets::debug::run_multi_source_demo(
        parsed.passthrough.into_iter(),
        move |_source_roots| Ok::<HfListRoots, Box<dyn Error>>(roots.clone()),
        build_hf_sources,
    )?;

    Ok(())
}

#[cfg(all(test, feature = "huggingface"))]
mod tests {
    use super::*;

    #[test]
    fn parse_hf_source_list_demo_cli_parses_flags_and_passthrough() {
        let parsed = HfSourceListDemoCli::try_parse_from([
            "hf_source_list_demo",
            "--source-list",
            "examples/common/custom_hf.txt",
            "--",
            "--batch-size",
            "8",
        ])
        .expect("expected parsed CLI");

        assert_eq!(parsed.source_list, "examples/common/custom_hf.txt");
        assert!(parsed.entry.is_none());
        assert_eq!(
            parsed.passthrough,
            vec!["--batch-size".to_string(), "8".to_string()]
        );
    }

    #[test]
    fn parse_hf_source_list_demo_cli_parses_entry() {
        let parsed = HfSourceListDemoCli::try_parse_from([
            "hf_source_list_demo",
            "--entry",
            "hf://wikimedia/wikipedia/20231101.en/train anchor=title positive=text",
        ])
        .expect("expected parsed CLI");

        assert_eq!(
            parsed.entry.as_deref(),
            Some("hf://wikimedia/wikipedia/20231101.en/train anchor=title positive=text")
        );
    }

    #[test]
    fn parse_hf_source_list_demo_cli_parses_entry_with_commas() {
        let parsed = HfSourceListDemoCli::try_parse_from([
            "hf_source_list_demo",
            "--entry",
            "hf://labofsahil/hackernews-vector-search-dataset/default text=title,text",
        ])
        .expect("expected parsed CLI");

        assert_eq!(
            parsed.entry.as_deref(),
            Some("hf://labofsahil/hackernews-vector-search-dataset/default text=title,text")
        );
    }
}
