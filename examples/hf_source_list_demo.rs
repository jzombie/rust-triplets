#![cfg_attr(not(feature = "huggingface"), allow(dead_code, unused_imports))]

use std::error::Error;

#[cfg(feature = "huggingface")]
use triplets::source::backends::huggingface_source::{
    HfListRoots, build_hf_sources, resolve_hf_list_roots,
};

#[cfg(not(feature = "huggingface"))]
fn main() {
    eprintln!("hf_source_list_demo requires --features huggingface");
}

#[cfg(feature = "huggingface")]
fn main() -> Result<(), Box<dyn Error>> {
    let raw_args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut source_list = "examples/common/hf_sources.txt".to_string();
    let mut max_rows_per_source: Option<usize> = Some(512);
    let mut passthrough = Vec::new();

    let mut idx = 0usize;
    while idx < raw_args.len() {
        match raw_args[idx].as_str() {
            "--source-list" => {
                if let Some(value) = raw_args.get(idx + 1) {
                    source_list = value.clone();
                    idx += 2;
                    continue;
                }
                return Err("--source-list requires a file path".into());
            }
            "--max-rows-per-source" => {
                if let Some(value) = raw_args.get(idx + 1) {
                    max_rows_per_source = Some(value.parse::<usize>().map_err(|_| {
                        format!(
                            "invalid value for --max-rows-per-source '{}': expected integer",
                            value
                        )
                    })?);
                    idx += 2;
                    continue;
                }
                return Err("--max-rows-per-source requires an integer value".into());
            }
            "--no-max-rows-cap" => {
                max_rows_per_source = None;
                idx += 1;
                continue;
            }
            _ => {
                passthrough.push(raw_args[idx].clone());
                idx += 1;
            }
        }
    }

    let roots = resolve_hf_list_roots(source_list.clone(), max_rows_per_source)
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
        passthrough
    );

    triplets::example_apps::run_multi_source_demo(
        passthrough.into_iter(),
        move |_source_roots| Ok::<HfListRoots, Box<dyn Error>>(roots.clone()),
        build_hf_sources,
    )?;

    Ok(())
}
