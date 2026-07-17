use crate::config::HuggingFaceRowsConfig;
use crate::constants::{
    HF_PARQUET_MANIFEST_DIR, HF_SHARD_STORE_EXTENSION, HF_SHARD_STORE_META_ROWS_KEY,
    HF_SHARD_STORE_ROW_PREFIX,
};
use crate::disk_cache;
use crate::disk_cache::open_store_via_cache;
use crate::file_utils::{is_gzip_path, is_transient_text};
use crate::types::ShardIndex;
use parquet::file::reader::{FileReader, SerializedFileReader};
use rayon::prelude::*;
use simd_r_drive::storage_engine::DataStore;
use simd_r_drive::storage_engine::traits::DataStoreReader;
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::Instant;
use tracing::{debug, info, warn};
use triplets_core::SamplerError;
use walkdir::WalkDir;

type ShardIndexResult = (Vec<ShardIndex>, usize);

/// Build deterministic local shard index for accepted extensions.
pub(crate) fn build_shard_index(
    config: &HuggingFaceRowsConfig,
) -> Result<ShardIndexResult, SamplerError> {
    let start_index = Instant::now();
    let mut shard_paths = Vec::new();
    let manifest_root = config.snapshot_dir.join(HF_PARQUET_MANIFEST_DIR);
    let accepted = config
        .shard_extensions
        .iter()
        .map(|ext| ext.trim().trim_start_matches('.').to_ascii_lowercase())
        .collect::<Vec<_>>();

    // .simdr must always be accepted regardless of user config, since
    // transient files may have already been transcoded to .simdr stores.
    let mut accepted = accepted;
    if !accepted.iter().any(|e| e == HF_SHARD_STORE_EXTENSION) {
        accepted.push(HF_SHARD_STORE_EXTENSION.to_string());
    }

    let mut saw_parquet = false;
    for entry in WalkDir::new(&config.snapshot_dir)
        .follow_links(true)
        .into_iter()
        .filter_map(Result::ok)
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let in_manifest = entry.path().starts_with(&manifest_root);
        let Some(ext) = entry.path().extension().and_then(|v| v.to_str()) else {
            continue;
        };
        let is_parquet = ext.eq_ignore_ascii_case("parquet");
        let is_transient =
            is_transient_text(entry.path()) || is_gzip_path(entry.path()) || is_parquet;

        if is_transient {
            if in_manifest {
                // Delete orphaned remote artifacts (truncated/crashed downloads)
                let _ = fs::remove_file(entry.path());
                continue;
            }
            // Local user files: skip if .simdr already exists (avoid double-indexing)
            let store_path = shard_store_path_for(entry.path());
            if store_path.exists() {
                continue;
            }
            // Flow into shard_paths for initialization transcoding
            if is_parquet {
                saw_parquet = true;
            }
            shard_paths.push(entry.path().to_path_buf());
            continue;
        }
        // For non-parquet files inside _parquet_manifest: only accept
        // shard store files (.simdr).  Metadata files like
        // _sequence_state.json and any other non-shard artifacts are
        // skipped.  Remote-sourced shards are stored as .simdr files
        // under _parquet_manifest and must be indexed here so that
        // materialized_rows is correctly populated on restart.
        if in_manifest && !ext.eq_ignore_ascii_case(HF_SHARD_STORE_EXTENSION) {
            continue;
        }
        if accepted
            .iter()
            .any(|allowed| allowed == &ext.to_ascii_lowercase())
        {
            shard_paths.push(entry.path().to_path_buf());
        }
    }

    shard_paths.sort();
    if shard_paths.is_empty() {
        if saw_parquet {
            warn!(
                "[triplets:hf] found persisted parquet under {} (transient-only policy); parquet files were pruned and source will repopulate from remote candidates",
                config.snapshot_dir.display()
            );
        }
        return Err(SamplerError::SourceUnavailable {
            source_id: config.source_id.clone(),
            reason: format!(
                "no shard files found under {} with extensions {:?}",
                config.snapshot_dir.display(),
                config.shard_extensions
            ),
        });
    }

    let total_shards = shard_paths.len();
    info!("[triplets:hf] indexing {} shards in parallel", total_shards);
    let completed = AtomicUsize::new(0);
    let indexed_shards: Result<Vec<_>, _> = shard_paths
        .into_par_iter()
        .enumerate()
        .map(|(ordinal, path)| {
            let result = index_single_shard(config, &path, 0)?;
            let n = completed.fetch_add(1, AtomicOrdering::Relaxed) + 1;
            let row_count = result.0.as_ref().map_or(0, |s| s.row_count);
            debug!(
                "[triplets:hf] indexed shard {}/{}: {} ({} rows)",
                n,
                total_shards,
                path.file_name()
                    .unwrap_or(path.as_os_str())
                    .to_string_lossy(),
                row_count,
            );
            Ok::<_, SamplerError>((ordinal, result))
        })
        .collect();
    let mut indexed_shards = indexed_shards?;

    indexed_shards.sort_by_key(|(ordinal, _)| *ordinal);

    let mut shards = Vec::new();
    let mut running_total = 0usize;
    for (_, (maybe_shard, _maybe_store)) in indexed_shards {
        let Some(mut shard) = maybe_shard else {
            continue;
        };

        if shard.row_count == 0 {
            continue;
        }

        shard.global_start = running_total;
        running_total = running_total.saturating_add(shard.row_count);
        shards.push(shard);
    }

    info!(
        "[triplets:hf] indexing complete in {:.2}s (rows={}, shards={})",
        start_index.elapsed().as_secs_f64(),
        running_total,
        shards.len()
    );

    Ok((shards, running_total))
}

/// Build shard metadata for a single local file.  All store handles are
/// fetched through `store_cache` (get-or-create), so there is never more
/// than one `DataStore` handle open for the same path.
#[cfg(test)]
#[allow(dead_code)]
fn index_single_shard_for_test(
    config: &HuggingFaceRowsConfig,
    path: &Path,
    global_start: usize,
) -> Result<(Option<ShardIndex>, Option<Arc<DataStore>>), SamplerError> {
    index_single_shard(config, path, global_start)
}

pub(crate) fn is_store_shard_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case(HF_SHARD_STORE_EXTENSION))
}

pub(crate) fn index_single_shard(
    config: &HuggingFaceRowsConfig,
    path: &Path,
    global_start: usize,
) -> Result<(Option<ShardIndex>, Option<Arc<DataStore>>), SamplerError> {
    let is_store = is_store_shard_path(path);
    // Parquet is treated as a transient decode artifact only.
    // Persisted shard artifacts should be per-shard .simdr stores.
    let is_transient_parquet = path
        .extension()
        .and_then(|v| v.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("parquet"));

    // .gz files (e.g. .jsonl.gz) are transient download artifacts that will
    // be transcoded to .simdr stores during the download phase.
    let is_transient_gz = is_gzip_path(path);

    let (rows, parquet_row_groups, _checkpoints, maybe_store) = if is_store {
        let store = open_store_via_cache(config, path)?;
        let rows = if let Some(entry) = store.read(HF_SHARD_STORE_META_ROWS_KEY).map_err(|err| {
            SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!("row-store meta read failed {}: {err}", path.display()),
            }
        })? {
            let payload = entry.as_ref();
            if payload.len() != std::mem::size_of::<u64>() {
                return Err(SamplerError::SourceUnavailable {
                    source_id: config.source_id.clone(),
                    reason: format!("invalid row-store meta payload {}", path.display()),
                });
            }
            let mut raw = [0u8; 8];
            raw.copy_from_slice(payload);
            u64::from_le_bytes(raw) as usize
        } else {
            0
        };

        // Integrity check: verify the last claimed row actually exists in
        // the store.  A corrupt store (partial write, truncated file) may
        // have the metadata key intact but be missing row data.  Delete it
        // so the shard is re-downloaded on the next expansion cycle.
        if rows > 0 {
            let last_key = row_store_row_key(rows.saturating_sub(1));
            match store.batch_read(&[last_key.as_slice()]) {
                Ok(entries) if entries[0].is_some() => {}
                _ => {
                    warn!(
                        "[triplets:hf] corrupted store detected ({} rows claimed but last row missing), deleting: {}",
                        rows,
                        path.display()
                    );
                    disk_cache::remove_stale_store(config, path);
                    return Ok((None, None));
                }
            }
        }

        let groups = if rows > 0 {
            vec![(0, rows)]
        } else {
            Vec::new()
        };
        (rows, groups, Vec::<u64>::new(), Some(store))
    } else if is_transient_parquet {
        let (rows, parquet_row_groups) = parquet_row_group_map(config, path)?;
        (rows, parquet_row_groups, Vec::<u64>::new(), None)
    } else if is_transient_gz || is_transient_text(path) {
        // All transient text/gzip files are transcoded to .simdr stores.
        // Validate file exists and is non-empty before returning dummy count.
        let metadata = fs::metadata(path).map_err(|err| SamplerError::SourceUnavailable {
            source_id: config.source_id.clone(),
            reason: format!("failed opening shard {}: {err}", path.display()),
        })?;
        if metadata.len() == 0 {
            return Ok((None, None));
        }
        return Ok((
            Some(ShardIndex {
                path: path.to_path_buf(),
                global_start,
                row_count: 1,
                parquet_row_groups: Vec::new(),
                remote_candidate: None,
            }),
            None,
        ));
    } else {
        let file = File::open(path).map_err(|err| SamplerError::SourceUnavailable {
            source_id: config.source_id.clone(),
            reason: format!("failed opening shard {}: {err}", path.display()),
        })?;
        let reader = BufReader::new(file);
        let rows = reader.lines().count();

        (rows, Vec::new(), Vec::<u64>::new(), None)
    };

    if rows == 0 {
        return Ok((None, None));
    }

    Ok((
        Some(ShardIndex {
            path: path.to_path_buf(),
            global_start,
            row_count: rows,
            parquet_row_groups,
            remote_candidate: None,
        }),
        maybe_store,
    ))
}

pub(crate) fn row_store_row_key(local_idx: usize) -> Vec<u8> {
    let mut key = Vec::with_capacity(HF_SHARD_STORE_ROW_PREFIX.len() + std::mem::size_of::<u64>());
    key.extend_from_slice(HF_SHARD_STORE_ROW_PREFIX);
    key.extend_from_slice(&(local_idx as u64).to_le_bytes());
    key
}

/// Build parquet row-group map for random-access row reads.
pub(crate) fn parquet_row_group_map(
    config: &HuggingFaceRowsConfig,
    path: &Path,
) -> Result<(usize, Vec<(usize, usize)>), SamplerError> {
    let file = File::open(path).map_err(|err| SamplerError::SourceUnavailable {
        source_id: config.source_id.clone(),
        reason: format!("failed opening parquet shard {}: {err}", path.display()),
    })?;
    let reader =
        SerializedFileReader::new(file).map_err(|err| SamplerError::SourceUnavailable {
            source_id: config.source_id.clone(),
            reason: format!("failed reading parquet metadata {}: {err}", path.display()),
        })?;

    let mut row_groups = Vec::new();
    let mut running = 0usize;
    for meta in reader.metadata().row_groups() {
        let group_rows =
            usize::try_from(meta.num_rows()).map_err(|_| SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!("parquet row group size overflow in {}", path.display()),
            })?;
        if group_rows == 0 {
            continue;
        }
        row_groups.push((running, group_rows));
        running = running.saturating_add(group_rows);
    }
    if running > 0 {
        return Ok((running, row_groups));
    }

    let total_rows =
        usize::try_from(reader.metadata().file_metadata().num_rows()).map_err(|_| {
            SamplerError::SourceUnavailable {
                source_id: config.source_id.clone(),
                reason: format!("parquet row count overflow in {}", path.display()),
            }
        })?;
    if total_rows == 0 {
        return Ok((0, Vec::new()));
    }
    Ok((total_rows, vec![(0, total_rows)]))
}

pub(crate) fn shard_store_path_for(path: &Path) -> PathBuf {
    if is_store_shard_path(path) {
        return path.to_path_buf();
    }
    path.with_extension(HF_SHARD_STORE_EXTENSION)
}
