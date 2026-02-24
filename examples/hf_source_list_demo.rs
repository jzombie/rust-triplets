#![cfg(feature = "huggingface")]

use std::error::Error;
use std::fs;
use std::path::PathBuf;

use triplets::source::DataSource;
use triplets::{HuggingFaceRowSource, HuggingFaceRowsConfig};

#[derive(Debug, Clone)]
struct HfListRoots {
    source_list: String,
    sources: Vec<HfSourceEntry>,
    max_rows_per_source: Option<usize>,
}

#[derive(Debug, Clone)]
struct HfSourceEntry {
    uri: String,
    anchor_column: Option<String>,
    positive_column: Option<String>,
    context_columns: Vec<String>,
    text_columns: Vec<String>,
}

fn parse_csv_fields(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn parse_hf_source_line(line: &str) -> Result<HfSourceEntry, Box<dyn Error>> {
    let mut parts = line.split_whitespace();
    let Some(uri) = parts.next() else {
        return Err("empty source line".into());
    };
    if !uri.starts_with("hf://") {
        return Err(format!("unsupported source URI (expected hf://...): {uri}").into());
    }

    let mut entry = HfSourceEntry {
        uri: uri.to_string(),
        anchor_column: None,
        positive_column: None,
        context_columns: Vec::new(),
        text_columns: Vec::new(),
    };

    for token in parts {
        let Some((raw_key, raw_value)) = token.split_once('=') else {
            return Err(format!("invalid mapping token '{token}' (expected key=value)").into());
        };
        let key = raw_key.trim().to_ascii_lowercase();
        let value = raw_value.trim();
        match key.as_str() {
            "anchor" => {
                entry.anchor_column = (!value.is_empty()).then(|| value.to_string());
            }
            "positive" => {
                entry.positive_column = (!value.is_empty()).then(|| value.to_string());
            }
            "context" => {
                entry.context_columns = parse_csv_fields(value);
            }
            "text" | "text_columns" => {
                entry.text_columns = parse_csv_fields(value);
            }
            _ => {
                return Err(format!("unsupported mapping key '{raw_key}'").into());
            }
        }
    }

    let has_explicit_mapping = entry.anchor_column.is_some()
        || entry.positive_column.is_some()
        || !entry.context_columns.is_empty()
        || !entry.text_columns.is_empty();
    if !has_explicit_mapping {
        return Err(format!(
            "source '{}' has no field mapping; expected at least one of anchor=, positive=, context=, text=",
            entry.uri
        )
        .into());
    }

    Ok(entry)
}

fn parse_hf_uri(uri: &str) -> Result<(String, String, String), Box<dyn Error>> {
    let trimmed = uri.trim();
    let Some(rest) = trimmed.strip_prefix("hf://") else {
        return Err(format!("unsupported source URI (expected hf://...): {trimmed}").into());
    };

    let parts = rest
        .split('/')
        .filter(|part| !part.trim().is_empty())
        .collect::<Vec<_>>();

    if parts.len() < 2 {
        return Err(format!("invalid hf URI (need hf://org/dataset): {trimmed}").into());
    }

    let dataset = format!("{}/{}", parts[0], parts[1]);
    let config = parts.get(2).copied().unwrap_or("default").to_string();
    let split = parts.get(3).copied().unwrap_or("train").to_string();

    Ok((dataset, config, split))
}

fn load_hf_sources_from_list(path: &str) -> Result<Vec<HfSourceEntry>, Box<dyn Error>> {
    let body = fs::read_to_string(path)?;
    let mut out = Vec::new();
    for (line_no, raw) in body.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parsed = parse_hf_source_line(line).map_err(|err| {
            format!(
                "invalid source-list entry at {}:{} -> {}",
                path,
                line_no + 1,
                err
            )
        })?;
        out.push(parsed);
    }
    Ok(out)
}

fn resolve_hf_list_roots(
    source_list: String,
    max_rows_per_source: Option<usize>,
) -> Result<HfListRoots, Box<dyn Error>> {
    let sources = load_hf_sources_from_list(&source_list)?;
    if sources.is_empty() {
        return Err(format!("no hf:// entries found in {}", source_list).into());
    }
    Ok(HfListRoots {
        source_list,
        sources,
        max_rows_per_source,
    })
}

fn build_hf_sources(roots: &HfListRoots) -> Vec<Box<dyn DataSource + 'static>> {
    roots
        .sources
        .iter()
        .enumerate()
        .filter_map(|(idx, source)| {
            let (dataset, config, split) = match parse_hf_uri(&source.uri) {
                Ok(parsed) => parsed,
                Err(err) => {
                    eprintln!("Skipping invalid source URI '{}': {}", source.uri, err);
                    return None;
                }
            };

            let source_id = format!("hf_list_{idx}");
            let snapshot_dir = PathBuf::from(".hf-snapshots")
                .join("source-list-demo")
                .join(dataset.replace('/', "__"))
                .join(&config)
                .join(&split)
                .join(format!("replica_{idx}"));

            let mut hf = HuggingFaceRowsConfig::new(
                source_id,
                dataset,
                config,
                split,
                snapshot_dir,
            );
            hf.anchor_column = source.anchor_column.clone();
            hf.positive_column = source.positive_column.clone();
            hf.context_columns = source.context_columns.clone();
            hf.text_columns = source.text_columns.clone();
            println!(
                "source {idx}: hf://{}/{}/{} -> anchor={:?}, positive={:?}, context={:?}, text_columns={:?}",
                hf.dataset,
                hf.config,
                hf.split,
                hf.anchor_column,
                hf.positive_column,
                hf.context_columns,
                hf.text_columns
            );
            hf.max_rows = roots.max_rows_per_source;

            match HuggingFaceRowSource::new(hf) {
                Ok(source) => Some(Box::new(source) as Box<dyn DataSource + 'static>),
                Err(err) => {
                    eprintln!(
                        "Skipping Hugging Face source initialization for '{}': {}",
                        source.uri, err
                    );
                    None
                }
            }
        })
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let raw_args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut source_list = "hf_sources.txt".to_string();
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

    let roots = resolve_hf_list_roots(source_list.clone(), max_rows_per_source)?;

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
