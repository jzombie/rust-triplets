use std::collections::HashMap;

use crate::config::HuggingFaceRowsConfig;
use crate::disk_cache::managed_hf_list_snapshot_dir;
use crate::download::build_http_client;
use crate::parsing::{HfListRoots, hf_source_id_slug, parse_hf_uri};
use crate::source_core::HuggingFaceRowSource;
use reqwest_drive::ClientWithMiddleware;
use tracing::{info, warn};
use triplets_core::source::DataSource;

/// Build Hugging Face row sources from a parsed source list.
pub fn build_hf_sources(roots: &HfListRoots) -> Vec<Box<dyn DataSource + 'static>> {
    build_hf_sources_with_weights(roots).0
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

    let mut shared_client: Option<ClientWithMiddleware> = None;

    let sources: Vec<Box<dyn DataSource + 'static>> = roots
        .sources
        .iter()
        .enumerate()
        .filter_map(|(idx, source)| {
            let (dataset, config, split) = match parse_hf_uri(&source.uri) {
                Ok(v) => v,
                Err(err) => {
                    warn!(uri = %source.uri, error = %err, "Skipping Hugging Face source (URI parse failure)");
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
                    warn!(uri = %source.uri, error = %err, "Skipping Hugging Face source (snapshot dir failure)");
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
                match build_http_client(&hf) {
                    Ok(client) => shared_client = Some(client),
                    Err(err) => {
                        warn!(
                            uri = %source.uri,
                            error = %err,
                            "Skipping source due to HTTP client initialization failure"
                        );
                        return None;
                    }
                }
            }
            hf.http_client = shared_client.clone();

            // Record weight if set.
            if let Some(w) = source.weight {
                weights.insert(hf.source_id.clone(), w);
            }

            info!(
                source_index = idx,
                dataset = %hf.dataset_name,
                config = %hf.config_name,
                split = %hf.split_name,
                anchor = ?hf.anchor_columns,
                positive = ?hf.positive_columns,
                negative = ?hf.negative_columns,
                context = ?hf.context_columns,
                text_columns = ?hf.text_columns,
                "Initialized Hugging Face source mapping"
            );

            match HuggingFaceRowSource::new(hf) {
                Ok(source) => Some(Box::new(source) as Box<dyn DataSource + 'static>),
                Err(err) => {
                    warn!(
                        uri = %source.uri,
                        error = %err,
                        "Skipping Hugging Face source initialization"
                    );
                    None
                }
            }
        })
        .collect();

    (sources, weights)
}
