use std::collections::{HashMap, HashSet};

use crate::config::HuggingFaceRowsConfig;
use crate::disk_cache::managed_hf_list_snapshot_dir;
use crate::download::build_http_client;
use crate::parsing::{HfListRoots, hf_source_id_slug, parse_hf_uri};
use crate::source_core::HuggingFaceRowSource;
use reqwest_drive::ClientWithMiddleware;
use triplets_core::source::DataSource;

// TODO: Wrap `build_hf_sources_with_weights`
/// Build Hugging Face row sources from a parsed source list.
pub fn build_hf_sources(roots: &HfListRoots) -> Vec<Box<dyn DataSource + 'static>> {
    // Phase 1: compute auto-slugs for entries that don't have an explicit source_id.
    // Entries with an explicit source_id bypass slug computation and deduplication.
    let base_slugs: Vec<Option<String>> = roots
        .sources
        .iter()
        .enumerate()
        .map(|(idx, source)| {
            if source.source_id.is_some() {
                // Explicit source_id — skip slug generation entirely.
                None
            } else {
                Some(match parse_hf_uri(&source.uri) {
                    Ok((dataset, config, split)) => hf_source_id_slug(&dataset, &config, &split),
                    Err(_) => format!("hf_list_{idx}"),
                })
            }
        })
        .collect();

    // Phase 2: find auto-slugs that appear more than once so they can be disambiguated.
    // Explicit source_ids are not subject to deduplication.
    let mut slug_count: HashMap<&str, usize> = HashMap::new();
    for slug in base_slugs.iter().flatten() {
        *slug_count.entry(slug.as_str()).or_insert(0) += 1;
    }
    let duplicated: HashSet<&str> = slug_count
        .into_iter()
        .filter(|(_, n)| *n > 1)
        .map(|(s, _)| s)
        .collect();

    // Phase 3: resolve final IDs.
    // Explicit source_ids are used as-is; auto-slugs get `.{idx}` only when they collide.
    let source_ids: Vec<String> = roots
        .sources
        .iter()
        .zip(base_slugs.iter())
        .enumerate()
        .map(|(idx, (source, base_slug))| {
            if let Some(explicit_id) = &source.source_id {
                explicit_id.clone()
            } else if let Some(slug) = base_slug {
                if duplicated.contains(slug.as_str()) {
                    format!("{slug}.{idx}")
                } else {
                    slug.clone()
                }
            } else {
                format!("hf_list_{idx}")
            }
        })
        .collect();

    let mut shared_client: Option<ClientWithMiddleware> = None;

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

            let source_id = source_ids[idx].clone();
            let snapshot_dir = match managed_hf_list_snapshot_dir(&dataset, &config, &split, idx) {
                Ok(path) => path,
                Err(err) => {
                    eprintln!(
                        "Skipping Hugging Face source initialization for '{}': {}",
                        source.uri, err
                    );
                    return None;
                }
            };

            let mut hf =
                HuggingFaceRowsConfig::new(source_id, dataset, config, split, snapshot_dir);
            hf.anchor_columns = source.anchor_columns.clone();
            hf.positive_columns = source.positive_columns.clone();
            hf.negative_columns = source.negative_columns.clone();
            hf.context_columns = source.context_columns.clone();
            hf.text_columns = source.text_columns.clone();
            hf.trust_override = source.trust;
            if shared_client.is_none() {
                shared_client = build_http_client(&hf).ok();
            }
            hf.http_client = shared_client.clone();
            println!(
                "source {idx}: hf://{}/{}/{} -> anchor={:?}, positive={:?}, negative={:?}, context={:?}, text_columns={:?}",
                hf.dataset_name,
                hf.config_name,
                hf.split_name,
                hf.anchor_columns,
                hf.positive_columns,
                hf.negative_columns,
                hf.context_columns,
                hf.text_columns
            );

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

/// Build Hugging Face row sources from a parsed source list, returning
/// both the sources and a per-source weight map.
///
/// Entries with a `weight=` value in their URI are included in the returned
/// `HashMap<String, f32>` (keyed by source ID).  Callers pass this map to
/// `Sampler::next_triplet_batch_with_weights` for weighted scheduling.
pub fn build_hf_sources_with_weights(
    roots: &HfListRoots,
) -> (Vec<Box<dyn DataSource + 'static>>, HashMap<String, f32>) {
    let mut weights = HashMap::new();

    // Phase 1: compute auto-slugs for entries that don't have an explicit source_id.
    let base_slugs: Vec<Option<String>> = roots
        .sources
        .iter()
        .enumerate()
        .map(|(idx, source)| {
            if source.source_id.is_some() {
                None
            } else {
                Some(match parse_hf_uri(&source.uri) {
                    Ok((dataset, config, split)) => hf_source_id_slug(&dataset, &config, &split),
                    Err(_) => format!("hf_list_{idx}"),
                })
            }
        })
        .collect();

    // Phase 2: find duplicate slugs for disambiguation.
    let mut slug_count: HashMap<&str, usize> = HashMap::new();
    for slug in base_slugs.iter().flatten() {
        *slug_count.entry(slug.as_str()).or_insert(0) += 1;
    }
    let mut slug_idx: HashMap<&str, usize> = HashMap::new();

    let shared_client = std::sync::OnceLock::<ClientWithMiddleware>::new();

    let sources: Vec<Box<dyn DataSource + 'static>> = roots
        .sources
        .iter()
        .enumerate()
        .filter_map(|(idx, source)| {
            let (dataset, config, split) = match parse_hf_uri(&source.uri) {
                Ok(v) => v,
                Err(err) => {
                    eprintln!("Skipping Hugging Face source '{}': {}", source.uri, err);
                    return None;
                }
            };

            let source_id = if let Some(ref sid) = source.source_id {
                sid.clone()
            } else {
                let base = base_slugs[idx].as_deref().unwrap_or("hf");
                let count = slug_count.get(base).copied().unwrap_or(0);
                if count > 1 {
                    let i = slug_idx.entry(base).or_insert(0);
                    let id = format!("{base}.{i}");
                    *i += 1;
                    id
                } else {
                    base.to_string()
                }
            };

            let snapshot_dir = match managed_hf_list_snapshot_dir(&dataset, &config, &split, idx) {
                Ok(dir) => dir,
                Err(err) => {
                    eprintln!("Skipping Hugging Face source '{}': {}", source.uri, err);
                    return None;
                }
            };

            let mut hf =
                HuggingFaceRowsConfig::new(source_id, dataset, config, split, snapshot_dir);
            hf.anchor_columns = source.anchor_columns.clone();
            hf.positive_columns = source.positive_columns.clone();
            hf.negative_columns = source.negative_columns.clone();
            hf.context_columns = source.context_columns.clone();
            hf.text_columns = source.text_columns.clone();
            hf.trust_override = source.trust;
            let client = shared_client.get_or_init(|| {
                build_http_client(&hf)
                    .unwrap_or_else(|_| build_http_client(&hf).expect("http client"))
            });
            hf.http_client = Some(client.clone());

            // Record weight if set.
            if let Some(w) = source.weight {
                weights.insert(hf.source_id.clone(), w);
            }

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
        .collect();

    (sources, weights)
}
