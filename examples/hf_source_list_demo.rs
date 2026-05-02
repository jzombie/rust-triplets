#![cfg_attr(not(feature = "huggingface"), allow(dead_code, unused_imports))]

use std::error::Error;

#[cfg(feature = "huggingface")]
use clap::Parser;
#[cfg(feature = "huggingface")]
use triplets::{HfListRoots, build_hf_sources, resolve_hf_list_roots};

#[cfg(feature = "huggingface")]
#[derive(Debug, Parser)]
#[command(
    name = "hf_source_list_demo",
    disable_help_subcommand = true,
    about = "Run multi_source_demo using Hugging Face source-list roots",
    long_about = "Resolve Hugging Face source-list entries, then forward remaining args to multi_source_demo.",
    after_help = "Use `--` before forwarded multi_source_demo args (for example: -- --batch-size 8 --split train)."
)]
struct HfSourceListDemoCli {
    #[arg(
        long = "source-list",
        default_value = "examples/common/hf_sources.txt",
        value_name = "PATH",
        help = "Path to Hugging Face source-list file"
    )]
    source_list: String,
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
    let roots = resolve_hf_list_roots(parsed.source_list.clone())
        .map_err(|err| -> Box<dyn Error> { err.into() })?;

    println!("== hf_source_list_demo (example_apps integration) ==");
    println!("source_list: {}", roots.source_list);
    println!("sources: {}", roots.sources.len());
    println!(
        "forwarding args to multi_source_demo: {:?}",
        parsed.passthrough
    );

    triplets::example_apps::run_multi_source_demo(
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
        assert_eq!(
            parsed.passthrough,
            vec!["--batch-size".to_string(), "8".to_string()]
        );
    }
}
