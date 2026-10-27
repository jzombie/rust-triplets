use std::cmp::Ordering;
use std::collections::HashSet;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Arc;

use cache_manager::{CacheRoot, EvictPolicy};
use simd_r_drive::storage_engine::DataStore;
use siphasher::sip::SipHasher;

use crate::config::ParquetRowsConfig;
use crate::row_view::EligibleIndexCache;
use crate::rows;
use crate::shard_index::is_store_shard_path;
use crate::source::ParquetRowSource;
use crate::types::{LocalShardState, ShardIndex};
use triplets_core::SamplerError;

/// Open (creating parent dirs) the `.simdr` row store at a path.
pub fn open_shard_store(
    config: &ParquetRowsConfig,
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

/// Get a cached row-store handle or open and cache it.
pub fn get_or_open_shard_store(
    source: &ParquetRowSource,
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

/// Drop cached store handles for shards no longer indexed.
pub fn prune_store_cache_to_shards(source: &ParquetRowSource, shards: &[ShardIndex]) {
    let keep = shards
        .iter()
        .map(|shard| shard.path.clone())
        .collect::<HashSet<_>>();
    if let Some(mut cache) = source.config.store_cache.lock_ok() {
        cache.retain(|path, _| keep.contains(path));
    }
}

/// Clear the cached eligible-row index (call after index mutations).
pub fn invalidate_eligible_index(source: &ParquetRowSource) {
    if let Ok(mut cache) = source.eligible_index.lock() {
        *cache = EligibleIndexCache::default();
    }
}

#[allow(dead_code)]
/// Hash of the shard layout, used to key the eligible-row index cache.
pub fn shard_signature(shards: &[ShardIndex]) -> u64 {
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
/// Collect eligible absolute row indices from shards (parses rows).
pub fn build_eligible_rows_from_shards(
    source: &ParquetRowSource,
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
                    reason: "parquet cache lock poisoned".to_string(),
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
/// Cached eligible absolute row indices for the current shard layout.
pub fn eligible_rows(source: &ParquetRowSource) -> Result<Arc<Vec<usize>>, SamplerError> {
    let (signature, shards) = {
        let local = source
            .local
            .lock()
            .map_err(|_| SamplerError::SourceUnavailable {
                source_id: source.config.source_id.clone(),
                reason: "parquet row source state lock poisoned".to_string(),
            })?;
        (shard_signature(&local.shards), local.shards.clone())
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
                    reason: "eligible-index cache lock poisoned".to_string(),
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
            reason: "eligible-index cache lock poisoned".to_string(),
        })?;
    cache.signature = Some(signature);
    cache.shards = shards;
    cache.rows = Some(rows.clone());
    Ok(rows)
}

/// Recompute shard `global_start` offsets and total materialized row count.
pub fn recompute_shard_offsets(state: &mut LocalShardState) {
    let mut running = 0usize;
    for shard in &mut state.shards {
        shard.global_start = running;
        running = running.saturating_add(shard.row_count);
    }
    state.materialized_rows = running;
}

/// Sync in-memory shard state from current on-disk snapshot tree.
///
/// Drops indexed shards whose files no longer exist and recomputes offsets.
/// Returns `true` when any shard was missing, so remote-backed wrappers can
/// reset their download frontier (evicted shards must be re-fetched).
pub fn sync_shard_state_from_disk_locked(state: &mut LocalShardState) -> bool {
    let any_missing = state.shards.iter().any(|shard| !shard.path.exists());
    state.shards.retain(|shard| shard.path.exists());
    recompute_shard_offsets(state);
    any_missing
}

/// Apply cache-manager eviction policy to managed shards and sync in-memory state.
///
/// Eviction applies to `managed_subdir` under `snapshot_dir` when both it
/// and `local_disk_cap_bytes` are configured; otherwise returns `false`
/// without touching anything.
pub fn enforce_disk_cap_locked(
    source: &ParquetRowSource,
    state: &mut LocalShardState,
) -> Result<bool, SamplerError> {
    let (Some(cap_bytes), Some(managed_subdir)) = (
        source.config.local_disk_cap_bytes,
        source.config.managed_subdir.as_deref(),
    ) else {
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
        .ensure_group_with_policy(managed_subdir, Some(&policy))
        .map_err(|err| SamplerError::SourceUnavailable {
            source_id: source.config.source_id.clone(),
            reason: format!(
                "failed applying managed cache eviction policy under {}: {err}",
                source.config.snapshot_dir.display()
            ),
        })?;

    sync_shard_state_from_disk_locked(state);
    let after = state
        .shards
        .iter()
        .map(|shard| shard.path.clone())
        .collect::<Vec<_>>();
    Ok(before != after)
}

/// Return total on-disk bytes used by transient-managed shards.
pub fn managed_usage_bytes_locked(source: &ParquetRowSource, state: &LocalShardState) -> u64 {
    state
        .shards
        .iter()
        .filter(|shard| source.config.in_transient_dir(&shard.path))
        .map(|shard| ParquetRowSource::shard_size_bytes(&shard.path))
        .sum::<u64>()
}

/// Locate containing shard and local offset for a global row index.
pub fn locate_shard(shards: &[ShardIndex], idx: usize) -> Option<(&ShardIndex, usize)> {
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
pub fn locate_parquet_group(
    source: &ParquetRowSource,
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
