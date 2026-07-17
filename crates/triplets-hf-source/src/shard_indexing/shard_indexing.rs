use std::cmp::Ordering;
use std::collections::HashSet;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cache_manager::{CacheRoot, EvictPolicy};
use simd_r_drive::storage_engine::DataStore;
use siphasher::sip::SipHasher;

use crate::config::HuggingFaceRowsConfig;
use crate::constants::HF_PARQUET_MANIFEST_DIR;
use crate::download::candidate_target_path;
use crate::huggingface_source::EligibleIndexCache;
use crate::rows;
use crate::shard_index::{is_store_shard_path, shard_store_path_for};
use crate::source_core::HuggingFaceRowSource;
use crate::types::{ShardIndex, SourceState};
use triplets_core::SamplerError;

pub(crate) fn candidate_store_path(config: &HuggingFaceRowsConfig, candidate: &str) -> PathBuf {
    shard_store_path_for(&candidate_target_path(config, candidate))
}

pub(crate) fn open_shard_store(
    config: &HuggingFaceRowsConfig,
    shard_store_path: &Path,
) -> Result<DataStore, SamplerError> {
    if let Some(parent) = shard_store_path.parent() {
        fs::create_dir_all(parent).map_err(|err| SamplerError::SourceUnavailable {
            source_id: config.source_id.clone(),
            reason: format!(
                "failed creating row-store directory {}: {err}",
                parent.display()
            ),
        })?;
    }
    DataStore::open(shard_store_path).map_err(|err| SamplerError::SourceUnavailable {
        source_id: config.source_id.clone(),
        reason: format!(
            "failed opening row store {}: {err}",
            shard_store_path.display()
        ),
    })
}

pub(crate) fn get_or_open_shard_store(
    source: &HuggingFaceRowSource,
    shard_store_path: &Path,
) -> Result<Arc<DataStore>, SamplerError> {
    let mut cache = source.config.store_cache.lock()?;
    if let Some(store) = cache.get(shard_store_path).cloned() {
        return Ok(store);
    }
    let store = Arc::new(open_shard_store(&source.config, shard_store_path)?);
    let entry = cache
        .entry(shard_store_path.to_path_buf())
        .or_insert_with(|| store.clone());
    Ok(entry.clone())
}

pub(crate) fn prune_store_cache_to_shards(source: &HuggingFaceRowSource, shards: &[ShardIndex]) {
    let keep = shards
        .iter()
        .map(|shard| shard.path.clone())
        .collect::<HashSet<_>>();
    if let Some(mut cache) = source.config.store_cache.lock_ok() {
        cache.retain(|path, _| keep.contains(path));
    }
}

pub(crate) fn invalidate_eligible_index(source: &HuggingFaceRowSource) {
    if let Ok(mut cache) = source.eligible_index.lock() {
        *cache = EligibleIndexCache::default();
    }
}

#[allow(dead_code)]
pub(crate) fn shard_signature(shards: &[ShardIndex]) -> u64 {
    let mut hasher = SipHasher::new();
    for shard in shards {
        shard.path.hash(&mut hasher);
        shard.global_start.hash(&mut hasher);
        shard.row_count.hash(&mut hasher);
        shard.parquet_row_groups.hash(&mut hasher);
    }
    hasher.finish()
}
#[allow(dead_code)]
pub(crate) fn build_eligible_rows_from_shards(
    source: &HuggingFaceRowSource,
    shards: &[ShardIndex],
) -> Result<Vec<usize>, SamplerError> {
    let mut eligible = Vec::new();

    for shard in shards {
        if is_store_shard_path(&shard.path) {
            for local_idx in 0..shard.row_count {
                let absolute_idx = shard.global_start.saturating_add(local_idx);
                eligible.push(absolute_idx);
            }
            continue;
        }

        for (group_pos, (group_start, group_count)) in
            shard.parquet_row_groups.iter().copied().enumerate()
        {
            let rows = source
                .parquet_cache
                .lock()
                .map_err(|_| SamplerError::SourceUnavailable {
                    source_id: source.config.source_id.clone(),
                    reason: "huggingface parquet cache lock poisoned".to_string(),
                })?
                .row_group_rows_for(
                    &source.config.source_id,
                    &shard.path,
                    group_pos,
                    source.config.parquet_row_group_cache_capacity,
                )?;

            for local_in_group in 0..group_count {
                let Some(row_value) = rows.get(local_in_group) else {
                    break;
                };
                let local_idx = group_start.saturating_add(local_in_group);
                if local_idx >= shard.row_count {
                    break;
                }
                let absolute_idx = shard.global_start.saturating_add(local_idx);
                if rows::parse_row(source, absolute_idx, row_value)?.is_some() {
                    eligible.push(absolute_idx);
                }
            }
        }
    }

    Ok(eligible)
}

#[allow(dead_code)]
pub(crate) fn eligible_rows(
    source: &HuggingFaceRowSource,
) -> Result<Arc<Vec<usize>>, SamplerError> {
    let (signature, shards) = {
        let state = source
            .state
            .lock()
            .map_err(|_| SamplerError::SourceUnavailable {
                source_id: source.config.source_id.clone(),
                reason: "huggingface source state lock poisoned".to_string(),
            })?;
        (shard_signature(&state.shards), state.shards.clone())
    };

    if let Ok(cache) = source.eligible_index.lock()
        && cache.signature == Some(signature)
        && let Some(rows) = cache.rows.as_ref()
    {
        return Ok(rows.clone());
    }

    let incremental_seed = if let Ok(cache) = source.eligible_index.lock()
        && cache.signature != Some(signature)
        && !cache.shards.is_empty()
        && cache.shards.len() < shards.len()
        && shards
            .iter()
            .take(cache.shards.len())
            .eq(cache.shards.iter())
        && let Some(existing_rows) = cache.rows.as_ref()
    {
        Some((cache.shards.len(), existing_rows.as_ref().clone()))
    } else {
        None
    };

    if let Some((prefix_len, mut merged)) = incremental_seed {
        let appended = build_eligible_rows_from_shards(source, &shards[prefix_len..])?;
        merged.extend(appended);
        let rows = Arc::new(merged);

        let mut writable =
            source
                .eligible_index
                .lock()
                .map_err(|_| SamplerError::SourceUnavailable {
                    source_id: source.config.source_id.clone(),
                    reason: "huggingface eligible-index cache lock poisoned".to_string(),
                })?;
        writable.signature = Some(signature);
        writable.shards = shards;
        writable.rows = Some(rows.clone());
        return Ok(rows);
    }

    let rows = Arc::new(build_eligible_rows_from_shards(source, &shards)?);
    let mut cache = source
        .eligible_index
        .lock()
        .map_err(|_| SamplerError::SourceUnavailable {
            source_id: source.config.source_id.clone(),
            reason: "huggingface eligible-index cache lock poisoned".to_string(),
        })?;
    cache.signature = Some(signature);
    cache.shards = shards;
    cache.rows = Some(rows.clone());
    Ok(rows)
}

/// Recompute shard `global_start` offsets and total materialized row count.
pub(crate) fn recompute_shard_offsets(state: &mut SourceState) {
    let mut running = 0usize;
    for shard in &mut state.shards {
        shard.global_start = running;
        running = running.saturating_add(shard.row_count);
    }
    state.materialized_rows = running;
}

/// Sync in-memory shard state from current on-disk snapshot tree.
pub(crate) fn sync_shard_state_from_disk_locked(
    _source: &HuggingFaceRowSource,
    state: &mut SourceState,
) {
    // If any shards have been evicted by the cache manager, remove them from
    // the in-memory index and reset the candidate list so the next expansion
    // cycle re-queries HF.  `all_candidates_from_parquet_manifest` returns every
    // shard from the manifest; evicted ones will be re-downloaded on next iteration.
    let any_missing = state.shards.iter().any(|shard| !shard.path.exists());
    state.shards.retain(|shard| shard.path.exists());
    recompute_shard_offsets(state);
    if any_missing {
        state.remote_candidates = None;
        state.remote_candidate_order = Vec::new();
        state.next_remote_idx = 0;
    }
}

/// Apply cache-manager eviction policy to manifest shards and sync in-memory state.
pub(crate) fn enforce_disk_cap_locked(
    source: &HuggingFaceRowSource,
    state: &mut SourceState,
    _protected_path: &Path,
) -> Result<bool, SamplerError> {
    let Some(cap_bytes) = source.config.local_disk_cap_bytes else {
        return Ok(false);
    };

    let before = state
        .shards
        .iter()
        .map(|shard| shard.path.clone())
        .collect::<Vec<_>>();
    let policy = EvictPolicy {
        max_bytes: Some(cap_bytes),
        ..EvictPolicy::default()
    };

    let cache_root = CacheRoot::from_root(&source.config.snapshot_dir);
    cache_root
        .ensure_group_with_policy(HF_PARQUET_MANIFEST_DIR, Some(&policy))
        .map_err(|err| SamplerError::SourceUnavailable {
            source_id: source.config.source_id.clone(),
            reason: format!(
                "failed applying manifest cache eviction policy under {}: {err}",
                source.config.snapshot_dir.display()
            ),
        })?;

    sync_shard_state_from_disk_locked(source, state);
    let after = state
        .shards
        .iter()
        .map(|shard| shard.path.clone())
        .collect::<Vec<_>>();
    Ok(before != after)
}

/// Return total on-disk bytes used by manifest-backed shards.
pub(crate) fn manifest_usage_bytes_locked(
    source: &HuggingFaceRowSource,
    state: &SourceState,
) -> u64 {
    let manifest_root = source.manifest_cache_root();
    state
        .shards
        .iter()
        .filter(|shard| shard.path.starts_with(&manifest_root))
        .map(|shard| HuggingFaceRowSource::shard_size_bytes(&shard.path))
        .sum::<u64>()
}

/// Locate containing shard and local offset for a global row index.
pub(crate) fn locate_shard(shards: &[ShardIndex], idx: usize) -> Option<(&ShardIndex, usize)> {
    let pos = shards
        .binary_search_by(|shard| {
            if idx < shard.global_start {
                Ordering::Greater
            } else if idx >= shard.global_start + shard.row_count {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        })
        .ok()?;
    let shard = shards.get(pos)?;
    Some((shard, idx - shard.global_start))
}

/// Locate parquet row-group and in-group row offset for a local row index.
pub(crate) fn locate_parquet_group(
    source: &HuggingFaceRowSource,
    shard: &ShardIndex,
    local_idx: usize,
) -> Result<(usize, usize), SamplerError> {
    let group_pos = shard
        .parquet_row_groups
        .binary_search_by(|(start, count)| {
            if local_idx < *start {
                Ordering::Greater
            } else if local_idx >= start.saturating_add(*count) {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        })
        .map_err(|_| SamplerError::SourceUnavailable {
            source_id: source.config.source_id.clone(),
            reason: format!(
                "parquet row {} could not be mapped to a row group in {}",
                local_idx,
                shard.path.display()
            ),
        })?;
    let (group_start, _) = shard.parquet_row_groups[group_pos];
    Ok((group_pos, local_idx.saturating_sub(group_start)))
}
