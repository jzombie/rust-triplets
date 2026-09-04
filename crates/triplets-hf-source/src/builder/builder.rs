use std::collections::HashMap;

use crate::config::HuggingFaceRowsConfig;
use crate::disk_cache::managed_hf_list_snapshot_dir;
use crate::download::build_http_client;
use crate::parsing::{HfListRoots, hf_source_id_slug, parse_hf_uri};
use crate::source_core::HuggingFaceRowSource;
use reqwest_drive::ClientWithMiddleware;
use tracing::{info, warn};
use triplets_core::source::DataSource;

/// Result of building Hugging Face sources from a source list.
/// Always returns partial successes alongside failures.
pub struct BuildResult {
    /// Successfully initialized data sources.
    pub sources: Vec<Box<dyn DataSource + 'static>>,
    /// Per-source weights for weighted scheduling.
    pub weights: HashMap<String, f32>,
    /// Sources that failed to initialize.
    pub failures: Vec<BuildFailure>,
}

/// Record of a single source that failed to initialize.
pub struct BuildFailure {
    /// Index of the failed source in the source list.
    pub index: usize,
    /// The `hf://` URI of the failed source.
    pub uri: String,
    /// Human-readable error description.
    pub reason: String,
}

/// Build Hugging Face row sources from a parsed source list, returning
/// only the successfully initialized sources (legacy backward-compatible API).
/// Logs all failures for debugging.
pub fn build_hf_sources(roots: &HfListRoots) -> Vec<Box<dyn DataSource + 'static>> {
    let result = build_hf_sources_with_weights(roots);
    for f in &result.failures {
        warn!(
            "[triplets:hf] source skipped [{}]: {}: {}",
            f.index, f.uri, f.reason
        );
    }
    result.sources
}

/// Build Hugging Face row sources from a parsed source list, returning
/// ALL results: successfully initialized sources, weights, and failure details.
///
/// Entries with a `weight=` value in their URI are included in the returned
/// `HashMap<String, f32>` (keyed by source ID).  Callers pass this map to
/// `Sampler::next_triplet_batch_with_weights` for weighted scheduling.
///
/// Callers decide whether to abort or proceed with valid sources when
/// `failures` is non-empty.
pub fn build_hf_sources_with_weights(roots: &HfListRoots) -> BuildResult {
    let mut weights = HashMap::new();
    let mut failures = Vec::new();

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
                    failures.push(BuildFailure {
                        index: idx,
                        uri: source.uri.clone(),
                        reason: format!("URI parse failure: {err}"),
                    });
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
                    failures.push(BuildFailure {
                        index: idx,
                        uri: source.uri.clone(),
                        reason: format!("snapshot dir failure: {err}"),
                    });
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
                        failures.push(BuildFailure {
                            index: idx,
                            uri: source.uri.clone(),
                            reason: format!("HTTP client init failure: {err}"),
                        });
                        return None;
                    }
                }
            }
            hf.http_client = shared_client.clone();

            // Save weight for deferred insertion after successful init.
            let source_weight = source.weight;
            let source_id_for_weight = hf.source_id.clone();

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
                Ok(source) => {
                    // Only record weight AFTER successful initialization.
                    if let Some(w) = source_weight {
                        weights.insert(source_id_for_weight, w);
                    }
                    Some(Box::new(source) as Box<dyn DataSource + 'static>)
                }
                Err(err) => {
                    failures.push(BuildFailure {
                        index: idx,
                        uri: source.uri.clone(),
                        reason: format!("source init failure: {err}"),
                    });
                    None
                }
            }
        })
        .collect();

    if !failures.is_empty() {
        warn!(
            "[triplets:hf] {}/{} sources failed to initialize",
            failures.len(),
            roots.sources.len()
        );
    }

    BuildResult {
        sources,
        weights,
        failures,
    }
}
