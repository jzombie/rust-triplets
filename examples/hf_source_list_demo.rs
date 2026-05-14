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
                  Provide a `--url` (with optional `--field` flags) to test a single source directly\n\
                  without a source-list file.",
    after_help = "Examples:\n  # Load sources from a file\n  hf_source_list_demo\n\n  # Test a single source directly\n  hf_source_list_demo --url hf://wikimedia/wikipedia/20231101.en/train --field anchor=title --field positive=text\n\n  # Single source with text columns and passthrough args\n  hf_source_list_demo --url hf://labofsahil/hackernews-vector-search-dataset/default \\\n      --field text=title,text -- --batch-size 8\n\nNote: Use `--` before forwarded multi_source_demo args (for example: -- --batch-size 8 --split train)."
)]
struct HfSourceListDemoCli {
    #[arg(
        long = "source-list",
        default_value = "examples/common/hf_sources.txt",
        required = false,
        value_name = "PATH",
        help = "Path to Hugging Face source-list file (ignored when --url is provided)"
    )]
    source_list: String,
    #[arg(
        long = "url",
        required = false,
        value_name = "HF_URI",
        help = "Single hf:// URI to test directly (e.g. hf://org/dataset/config/split)"
    )]
    url: Option<String>,
    #[arg(
        long = "field",
        required = false,
        value_name = "KEY=VALUE",
        help = "Field mapping for the source (e.g. text=title,text or anchor=title positive=text).\n\
                Can be specified multiple times. Only used when --url is provided."
    )]
    field: Vec<String>,
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

    let roots = if let Some(url) = &parsed.url {
        // Build a single source entry from --url and --field args.
        let field_str = if parsed.field.is_empty() {
            String::new()
        } else {
            format!(" {}", parsed.field.join(" "))
        };
        let line = format!("{url}{field_str}");
        let entry = parse_hf_source_line(&line).map_err(|err| format!("invalid source: {err}"))?;

        let roots = HfListRoots {
            source_list: url.clone(),
            sources: vec![entry],
        };
        println!("== hf_source_list_demo (direct URL mode) ==");
        println!("url: {url}");
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
        assert!(parsed.url.is_none());
        assert!(parsed.field.is_empty());
        assert_eq!(
            parsed.passthrough,
            vec!["--batch-size".to_string(), "8".to_string()]
        );
    }

    #[test]
    fn parse_hf_source_list_demo_cli_parses_url_and_field() {
        let parsed = HfSourceListDemoCli::try_parse_from([
            "hf_source_list_demo",
            "--url",
            "hf://wikimedia/wikipedia/20231101.en/train",
            "--field",
            "anchor=title",
            "--field",
            "positive=text",
        ])
        .expect("expected parsed CLI");

        assert_eq!(
            parsed.url.as_deref(),
            Some("hf://wikimedia/wikipedia/20231101.en/train")
        );
        assert_eq!(
            parsed.field,
            vec!["anchor=title".to_string(), "positive=text".to_string()]
        );
    }

    #[test]
    fn parse_hf_source_list_demo_cli_parses_url_with_commas() {
        let parsed = HfSourceListDemoCli::try_parse_from([
            "hf_source_list_demo",
            "--url",
            "hf://labofsahil/hackernews-vector-search-dataset/default",
            "--field",
            "text=title,text",
        ])
        .expect("expected parsed CLI");

        assert_eq!(
            parsed.url.as_deref(),
            Some("hf://labofsahil/hackernews-vector-search-dataset/default")
        );
        assert_eq!(parsed.field, vec!["text=title,text"]);
    }
}
