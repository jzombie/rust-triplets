#![cfg_attr(not(feature = "huggingface"), allow(dead_code, unused_imports))]

use std::error::Error;

#[cfg(feature = "huggingface")]
use clap::Parser;
#[cfg(feature = "huggingface")]
use triplets::source::backends::huggingface_source::{
    HfListRoots, build_hf_sources, resolve_hf_list_roots,
};

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
        long = "max-rows-per-source",
        default_value_t = 512,
        value_name = "N",
        help = "Per-source max rows cap for loaded HF list sources"
    )]
    max_rows_per_source: usize,
    #[arg(long = "no-max-rows-cap", help = "Disable per-source max rows cap")]
    no_max_rows_cap: bool,
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
    let max_rows_per_source = if parsed.no_max_rows_cap {
        None
    } else {
        Some(parsed.max_rows_per_source)
    };

    let roots = resolve_hf_list_roots(parsed.source_list.clone(), max_rows_per_source)
        .map_err(|err| -> Box<dyn Error> { err.into() })?;

    println!("== hf_source_list_demo (example_apps integration) ==");
    println!("source_list: {}", roots.source_list);
    println!("sources: {}", roots.sources.len());
    println!(
        "max_rows_per_source: {}",
        roots
            .max_rows_per_source
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string())
    );
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
            "--max-rows-per-source",
            "123",
            "--",
            "--batch-size",
            "8",
        ])
        .expect("expected parsed CLI");

        assert_eq!(parsed.source_list, "examples/common/custom_hf.txt");
        assert_eq!(parsed.max_rows_per_source, 123);
        assert_eq!(
            parsed.passthrough,
            vec!["--batch-size".to_string(), "8".to_string()]
        );

        let no_cap =
            HfSourceListDemoCli::try_parse_from(["hf_source_list_demo", "--no-max-rows-cap"])
                .expect("expected parsed CLI");
        assert!(no_cap.no_max_rows_cap);
    }
}
