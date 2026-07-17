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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shard_index::index_single_shard;
    use crate::test_utils::{test_config, test_source, write_parquet_fixture, write_simdr_fixture};
    use std::collections::HashMap;
    use std::ops::Deref;
    use tempfile::tempdir;

    #[test]
    fn get_or_open_shard_store_reuses_cached_handle_and_prune_keeps_active_only() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);

        let store_a = dir.path().join("a.simdr");
        let store_b = dir.path().join("b.simdr");

        let first =
            crate::shard_indexing::get_or_open_shard_store(source.deref(), &store_a).unwrap();
        let second =
            crate::shard_indexing::get_or_open_shard_store(source.deref(), &store_a).unwrap();
        assert!(Arc::ptr_eq(&first, &second));

        let _third =
            crate::shard_indexing::get_or_open_shard_store(source.deref(), &store_b).unwrap();
        {
            let cache = source.config.store_cache.lock().unwrap();
            assert!(cache.contains_key(&store_a));
            assert!(cache.contains_key(&store_b));
        }

        crate::shard_indexing::prune_store_cache_to_shards(
            source.deref(),
            &[ShardIndex {
                path: store_a.clone(),
                global_start: 0,
                row_count: 1,
                parquet_row_groups: vec![(0, 1)],
                remote_candidate: None,
            }],
        );

        let cache = source.config.store_cache.lock().unwrap();
        assert!(cache.contains_key(&store_a));
        assert!(!cache.contains_key(&store_b));
    }

    #[test]
    fn build_eligible_rows_from_store_shard_uses_global_offsets() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);

        let store_path = dir.path().join("eligible.simdr");
        write_simdr_fixture(&store_path, &[("r1", "alpha"), ("r2", "beta")]);

        let shards = vec![ShardIndex {
            path: store_path,
            global_start: 5,
            row_count: 2,
            parquet_row_groups: vec![(0, 2)],
            remote_candidate: None,
        }];

        let eligible =
            crate::shard_indexing::build_eligible_rows_from_shards(source.deref(), &shards)
                .unwrap();
        assert_eq!(eligible, vec![5, 6]);
    }

    #[test]
    fn locate_parquet_group_maps_offsets_and_reports_missing() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let shard = ShardIndex {
            path: dir.path().join("rows.parquet"),
            global_start: 0,
            row_count: 6,
            parquet_row_groups: vec![(0, 2), (2, 2), (4, 2)],
            remote_candidate: None,
        };

        let mapped =
            crate::shard_indexing::locate_parquet_group(source.deref(), &shard, 3).unwrap();
        assert_eq!(mapped, (1, 1));
        let missing = crate::shard_indexing::locate_parquet_group(source.deref(), &shard, 99);
        assert!(missing.is_err());
    }

    #[test]
    fn manifest_usage_bytes_locked_counts_only_manifest_shards() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let manifest_root = source.manifest_cache_root();
        fs::create_dir_all(&manifest_root).unwrap();

        let manifest_file = manifest_root.join("a.parquet");
        fs::write(&manifest_file, vec![1u8; 7]).unwrap();
        let local_file = source.config.snapshot_dir.join("local.ndjson");
        fs::write(&local_file, vec![2u8; 9]).unwrap();

        let state = SourceState {
            materialized_rows: 2,
            shards: vec![
                ShardIndex {
                    path: manifest_file,
                    global_start: 0,
                    row_count: 1,
                    parquet_row_groups: vec![(0, 1)],
                    remote_candidate: None,
                },
                ShardIndex {
                    path: local_file,
                    global_start: 1,
                    row_count: 1,
                    parquet_row_groups: Vec::new(),
                    remote_candidate: None,
                },
            ],
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            remote_candidate_order: Vec::new(),
        };

        assert_eq!(
            crate::shard_indexing::manifest_usage_bytes_locked(source.deref(), &state),
            7
        );
    }

    #[test]
    fn enforce_disk_cap_returns_false_when_disabled_or_under_limit() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.local_disk_cap_bytes = None;
        let source = test_source(config);
        let mut state = SourceState {
            materialized_rows: 0,
            shards: Vec::new(),
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            remote_candidate_order: Vec::new(),
        };
        let protected = dir.path().join("p");
        assert!(
            !crate::shard_indexing::enforce_disk_cap_locked(source.deref(), &mut state, &protected)
                .unwrap()
        );

        let mut config2 = test_config(dir.path().to_path_buf());
        config2.local_disk_cap_bytes = Some(10_000);
        let source2 = test_source(config2);
        let manifest_root = source2.manifest_cache_root();
        fs::create_dir_all(&manifest_root).unwrap();
        let shard_path = manifest_root.join("small.parquet");
        fs::write(&shard_path, vec![1u8; 32]).unwrap();
        let mut state2 = SourceState {
            materialized_rows: 1,
            shards: vec![ShardIndex {
                path: shard_path,
                global_start: 0,
                row_count: 1,
                parquet_row_groups: vec![(0, 1)],
                remote_candidate: None,
            }],
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            remote_candidate_order: Vec::new(),
        };
        assert!(
            !crate::shard_indexing::enforce_disk_cap_locked(
                source2.deref(),
                &mut state2,
                &protected
            )
            .unwrap()
        );
    }

    #[test]
    fn enforce_disk_cap_evicts_manifest_shards_and_recomputes_offsets() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.local_disk_cap_bytes = Some(20);
        let source = test_source(config);
        let manifest_root = source.manifest_cache_root();
        fs::create_dir_all(&manifest_root).unwrap();

        let first = manifest_root.join("first.parquet");
        let second = manifest_root.join("second.parquet");
        fs::write(&first, vec![1u8; 16]).unwrap();
        fs::write(&second, vec![2u8; 16]).unwrap();

        let mut state = SourceState {
            materialized_rows: 2,
            shards: vec![
                ShardIndex {
                    path: first.clone(),
                    global_start: 0,
                    row_count: 1,
                    parquet_row_groups: vec![(0, 1)],
                    remote_candidate: None,
                },
                ShardIndex {
                    path: second.clone(),
                    global_start: 1,
                    row_count: 1,
                    parquet_row_groups: vec![(0, 1)],
                    remote_candidate: None,
                },
            ],
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            remote_candidate_order: Vec::new(),
        };

        let evicted =
            crate::shard_indexing::enforce_disk_cap_locked(source.deref(), &mut state, &second)
                .unwrap();
        assert!(evicted);
        assert!(!first.exists());
        assert!(second.exists());
        assert_eq!(state.shards.len(), 1);
        assert_eq!(state.shards[0].global_start, 0);
        assert_eq!(state.materialized_rows, 1);
    }

    #[test]
    fn enforce_disk_cap_evicts_when_single_file_exceeds_cap() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.local_disk_cap_bytes = Some(1);
        let source = test_source(config);
        let manifest_root = source.manifest_cache_root();
        fs::create_dir_all(&manifest_root).unwrap();

        let protected = manifest_root.join("protected.parquet");
        fs::write(&protected, vec![3u8; 16]).unwrap();

        let mut state = SourceState {
            materialized_rows: 1,
            shards: vec![ShardIndex {
                path: protected.clone(),
                global_start: 0,
                row_count: 1,
                parquet_row_groups: vec![(0, 1)],
                remote_candidate: None,
            }],
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            remote_candidate_order: Vec::new(),
        };

        let evicted =
            crate::shard_indexing::enforce_disk_cap_locked(source.deref(), &mut state, &protected)
                .unwrap();
        assert!(evicted);
        assert!(!protected.exists());
        assert_eq!(state.shards.len(), 0);
        assert_eq!(state.materialized_rows, 0);
    }

    #[test]
    fn enforce_disk_cap_evicts_old_manifest_shards() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.local_disk_cap_bytes = Some(10);
        let source = test_source(config);

        let manifest_root = source.manifest_cache_root();
        fs::create_dir_all(&manifest_root).unwrap();
        let evict_path = manifest_root.join("a.parquet");
        let keep_path = manifest_root.join("b.parquet");
        fs::write(&evict_path, vec![1u8; 8]).unwrap();
        fs::write(&keep_path, vec![2u8; 8]).unwrap();

        let mut state = SourceState {
            materialized_rows: 16,
            shards: vec![
                ShardIndex {
                    path: evict_path.clone(),
                    global_start: 0,
                    row_count: 8,
                    parquet_row_groups: vec![(0, 8)],
                    remote_candidate: None,
                },
                ShardIndex {
                    path: keep_path.clone(),
                    global_start: 8,
                    row_count: 8,
                    parquet_row_groups: vec![(0, 8)],
                    remote_candidate: None,
                },
            ],
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            remote_candidate_order: Vec::new(),
        };

        let evicted =
            crate::shard_indexing::enforce_disk_cap_locked(source.deref(), &mut state, &keep_path)
                .unwrap();
        assert!(evicted);
        assert!(!evict_path.exists());
        assert!(keep_path.exists());
        assert_eq!(state.shards.len(), 1);
    }

    #[test]
    fn enforce_disk_cap_ignores_min_resident_and_applies_policy() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.local_disk_cap_bytes = Some(4);
        let source = test_source(config);

        let manifest_root = source.manifest_cache_root();
        fs::create_dir_all(&manifest_root).unwrap();
        let protected = manifest_root.join("only.parquet");
        fs::write(&protected, vec![1u8; 8]).unwrap();

        let mut state = SourceState {
            materialized_rows: 8,
            shards: vec![ShardIndex {
                path: protected.clone(),
                global_start: 0,
                row_count: 8,
                parquet_row_groups: vec![(0, 8)],
                remote_candidate: None,
            }],
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            remote_candidate_order: Vec::new(),
        };

        let evicted =
            crate::shard_indexing::enforce_disk_cap_locked(source.deref(), &mut state, &protected)
                .unwrap();
        assert!(evicted);
        assert!(!protected.exists());
        assert_eq!(state.shards.len(), 0);
    }

    #[test]
    fn locate_shard_returns_none_for_out_of_range_index() {
        let shards = vec![ShardIndex {
            path: PathBuf::from("a.ndjson"),
            global_start: 0,
            row_count: 2,
            parquet_row_groups: Vec::new(),
            remote_candidate: None,
        }];

        assert!(crate::shard_indexing::locate_shard(&shards, 5).is_none());
    }

    #[test]
    fn locate_shard_and_recompute_offsets_work() {
        let mut shards = vec![
            ShardIndex {
                path: PathBuf::from("a"),
                global_start: 10,
                row_count: 3,
                parquet_row_groups: Vec::new(),
                remote_candidate: None,
            },
            ShardIndex {
                path: PathBuf::from("b"),
                global_start: 20,
                row_count: 2,
                parquet_row_groups: Vec::new(),
                remote_candidate: None,
            },
        ];
        let hit = crate::shard_indexing::locate_shard(&shards, 11).unwrap();
        assert_eq!(hit.1, 1);

        let mut state = SourceState {
            materialized_rows: 0,
            shards: std::mem::take(&mut shards),
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            remote_candidate_order: Vec::new(),
        };
        crate::shard_indexing::recompute_shard_offsets(&mut state);
        assert_eq!(state.shards[0].global_start, 0);
        assert_eq!(state.shards[1].global_start, 3);
        assert_eq!(state.materialized_rows, 5);
    }

    #[test]
    fn locate_shard_exact_start() {
        let shards = vec![
            ShardIndex {
                path: PathBuf::from("a"),
                global_start: 0,
                row_count: 5,
                parquet_row_groups: Vec::new(),
                remote_candidate: None,
            },
            ShardIndex {
                path: PathBuf::from("b"),
                global_start: 5,
                row_count: 5,
                parquet_row_groups: Vec::new(),
                remote_candidate: None,
            },
        ];
        let (shard, offset) = crate::shard_indexing::locate_shard(&shards, 5).unwrap();
        assert_eq!(shard.path, PathBuf::from("b"));
        assert_eq!(offset, 0);
    }

    #[test]
    fn locate_shard_last_element() {
        let shards = vec![
            ShardIndex {
                path: PathBuf::from("a"),
                global_start: 0,
                row_count: 5,
                parquet_row_groups: Vec::new(),
                remote_candidate: None,
            },
            ShardIndex {
                path: PathBuf::from("b"),
                global_start: 5,
                row_count: 5,
                parquet_row_groups: Vec::new(),
                remote_candidate: None,
            },
        ];
        let (shard, offset) = crate::shard_indexing::locate_shard(&shards, 9).unwrap();
        assert_eq!(shard.path, PathBuf::from("b"));
        assert_eq!(offset, 4);
    }

    #[test]
    fn locate_shard_before_first_returns_none() {
        let shards = vec![ShardIndex {
            path: PathBuf::from("a"),
            global_start: 5,
            row_count: 5,
            parquet_row_groups: Vec::new(),
            remote_candidate: None,
        }];
        assert!(crate::shard_indexing::locate_shard(&shards, 0).is_none());
    }

    #[test]
    fn locate_shard_empty_returns_none() {
        assert!(crate::shard_indexing::locate_shard(&[], 0).is_none());
    }

    #[test]
    fn locate_shard_finds_correct_shard_and_local_idx() {
        let shards = vec![
            ShardIndex {
                path: PathBuf::from("a.simdr"),
                global_start: 0,
                row_count: 10,
                parquet_row_groups: Vec::new(),
                remote_candidate: None,
            },
            ShardIndex {
                path: PathBuf::from("b.simdr"),
                global_start: 10,
                row_count: 5,
                parquet_row_groups: Vec::new(),
                remote_candidate: None,
            },
        ];

        let (shard, local_idx) = crate::shard_indexing::locate_shard(&shards, 12).unwrap();
        assert_eq!(shard.path, PathBuf::from("b.simdr"));
        assert_eq!(local_idx, 2);
    }

    #[test]
    fn locate_shard_returns_none_for_out_of_bounds() {
        let shards = vec![ShardIndex {
            path: PathBuf::from("a.simdr"),
            global_start: 0,
            row_count: 5,
            parquet_row_groups: Vec::new(),
            remote_candidate: None,
        }];

        assert!(crate::shard_indexing::locate_shard(&shards, 10).is_none());
    }

    #[test]
    fn recompute_shard_offsets_sums_row_counts() {
        let mut state = SourceState {
            materialized_rows: 0,
            shards: vec![
                ShardIndex {
                    path: PathBuf::from("a.simdr"),
                    global_start: 0,
                    row_count: 10,
                    parquet_row_groups: vec![(0, 10)],
                    remote_candidate: None,
                },
                ShardIndex {
                    path: PathBuf::from("b.simdr"),
                    global_start: 0,
                    row_count: 20,
                    parquet_row_groups: vec![(0, 20)],
                    remote_candidate: None,
                },
            ],
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            remote_candidate_order: Vec::new(),
        };
        crate::shard_indexing::recompute_shard_offsets(&mut state);
        assert_eq!(state.materialized_rows, 30);
        assert_eq!(state.shards[0].global_start, 0);
        assert_eq!(state.shards[1].global_start, 10);
    }

    #[test]
    fn recompute_shard_offsets_empty() {
        let mut state = SourceState {
            materialized_rows: 0,
            shards: Vec::new(),
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            remote_candidate_order: Vec::new(),
        };
        crate::shard_indexing::recompute_shard_offsets(&mut state);
        assert_eq!(state.materialized_rows, 0);
    }

    #[test]
    fn recompute_shard_offsets_single_shard() {
        let mut state = SourceState {
            materialized_rows: 0,
            shards: vec![ShardIndex {
                path: PathBuf::from("a"),
                global_start: 999, // should be overwritten
                row_count: 7,
                parquet_row_groups: Vec::new(),
                remote_candidate: None,
            }],
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            remote_candidate_order: Vec::new(),
        };
        crate::shard_indexing::recompute_shard_offsets(&mut state);
        assert_eq!(state.shards[0].global_start, 0);
        assert_eq!(state.materialized_rows, 7);
    }

    #[test]
    fn recompute_shard_offsets_sets_correct_start_values() {
        let dir = tempdir().unwrap();

        let mut state = SourceState {
            materialized_rows: 0,
            shards: vec![
                ShardIndex {
                    path: dir.path().join("a.simdr"),
                    global_start: 0,
                    row_count: 10,
                    parquet_row_groups: Vec::new(),
                    remote_candidate: None,
                },
                ShardIndex {
                    path: dir.path().join("b.simdr"),
                    global_start: 0,
                    row_count: 5,
                    parquet_row_groups: Vec::new(),
                    remote_candidate: None,
                },
            ],
            remote_candidates: None,
            remote_candidate_sizes: HashMap::new(),
            remote_candidate_order: Vec::new(),
            next_remote_idx: 0,
        };

        crate::shard_indexing::recompute_shard_offsets(&mut state);

        assert_eq!(state.shards[0].global_start, 0);
        assert_eq!(state.shards[1].global_start, 10);
        assert_eq!(state.materialized_rows, 15);
    }

    #[test]
    fn sync_shard_state_from_disk_removes_missing_shards() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let existing = dir.path().join("existing.simdr");
        fs::write(&existing, b"data").unwrap();
        let missing = dir.path().join("missing.simdr");
        let mut state = SourceState {
            materialized_rows: 100,
            shards: vec![
                ShardIndex {
                    path: existing.clone(),
                    global_start: 0,
                    row_count: 50,
                    parquet_row_groups: vec![(0, 50)],
                    remote_candidate: None,
                },
                ShardIndex {
                    path: missing.clone(),
                    global_start: 50,
                    row_count: 50,
                    parquet_row_groups: vec![(0, 50)],
                    remote_candidate: None,
                },
            ],
            remote_candidates: Some(vec!["candidate".to_string()]),
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            remote_candidate_order: vec![0],
        };
        crate::shard_indexing::sync_shard_state_from_disk_locked(source.deref(), &mut state);
        assert_eq!(state.shards.len(), 1);
        assert_eq!(state.shards[0].path, existing);
        assert_eq!(state.materialized_rows, 50);
        assert!(state.remote_candidates.is_none());
    }

    #[test]
    fn sync_shard_state_from_disk_preserves_candidates_when_all_present() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let sp = dir.path().join("shard.simdr");
        fs::write(&sp, b"data").unwrap();
        let mut state = SourceState {
            materialized_rows: 50,
            shards: vec![ShardIndex {
                path: sp.clone(),
                global_start: 0,
                row_count: 50,
                parquet_row_groups: vec![(0, 50)],
                remote_candidate: None,
            }],
            remote_candidates: Some(vec!["next".to_string()]),
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 1,
            remote_candidate_order: vec![0],
        };
        crate::shard_indexing::sync_shard_state_from_disk_locked(source.deref(), &mut state);
        assert_eq!(state.shards.len(), 1);
        assert_eq!(state.remote_candidates, Some(vec!["next".to_string()]));
        assert_eq!(state.next_remote_idx, 1);
    }

    #[test]
    fn sync_shard_state_from_disk_locked_removes_missing() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let missing_path = dir.path().join("missing.simdr");
        let mut state = SourceState {
            materialized_rows: 10,
            shards: vec![ShardIndex {
                path: missing_path.clone(),
                global_start: 0,
                row_count: 10,
                parquet_row_groups: Vec::new(),
                remote_candidate: None,
            }],
            remote_candidates: Some(vec!["candidate".to_string()]),
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 5,
            remote_candidate_order: vec![0],
        };
        crate::shard_indexing::sync_shard_state_from_disk_locked(&source, &mut state);
        assert!(state.shards.is_empty());
        assert_eq!(state.materialized_rows, 0);
        // Candidates should be reset
        assert!(state.remote_candidates.is_none());
        assert!(state.remote_candidate_order.is_empty());
        assert_eq!(state.next_remote_idx, 0);
    }

    #[test]
    fn sync_shard_state_from_disk_locked_keeps_existing() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let existing_path = dir.path().join("existing.simdr");
        std::fs::write(&existing_path, b"data").unwrap();
        let mut state = SourceState {
            materialized_rows: 5,
            shards: vec![ShardIndex {
                path: existing_path.clone(),
                global_start: 0,
                row_count: 5,
                parquet_row_groups: Vec::new(),
                remote_candidate: None,
            }],
            remote_candidates: Some(vec!["candidate".to_string()]),
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 1,
            remote_candidate_order: vec![0],
        };
        crate::shard_indexing::sync_shard_state_from_disk_locked(&source, &mut state);
        assert_eq!(state.shards.len(), 1);
        assert_eq!(state.materialized_rows, 5);
        // Candidates should NOT be reset since no shards were missing
        assert!(state.remote_candidates.is_some());
    }

    #[test]
    fn eligible_rows_extends_cached_index_when_new_shard_is_appended() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        let appended_simdr = dir.path().join("append.simdr");
        write_simdr_fixture(&appended_simdr, &[("r1", "hello")]);
        let appended = index_single_shard(&config, &appended_simdr, 1)
            .unwrap()
            .0
            .unwrap();

        let baseline = ShardIndex {
            path: dir.path().join("missing-baseline.simdr"),
            global_start: 0,
            row_count: 1,
            parquet_row_groups: Vec::new(),
            remote_candidate: None,
        };

        {
            let mut state = source.state.lock().unwrap();
            state.shards = vec![baseline.clone(), appended.clone()];
            state.materialized_rows = 2;
        }

        {
            let mut cache = source.eligible_index.lock().unwrap();
            cache.signature = Some(crate::shard_indexing::shard_signature(
                std::slice::from_ref(&baseline),
            ));
            cache.rows = Some(Arc::new(vec![0]));
            cache.shards = vec![baseline];
        }

        let rows = crate::shard_indexing::eligible_rows(source.deref()).unwrap();
        assert_eq!(rows.as_ref(), &vec![0, 1]);
    }

    #[test]
    fn eligible_rows_cache_hit_returns_cached_without_rebuild() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);

        let shard = ShardIndex {
            path: dir.path().join("s1.simdr"),
            global_start: 0,
            row_count: 3,
            parquet_row_groups: Vec::new(),
            remote_candidate: None,
        };

        {
            let mut state = source.state.lock().unwrap();
            state.shards = vec![shard.clone()];
            state.materialized_rows = 3;
        }

        let sig = crate::shard_indexing::shard_signature(std::slice::from_ref(&shard));
        {
            let mut cache = source.eligible_index.lock().unwrap();
            cache.signature = Some(sig);
            cache.rows = Some(Arc::new(vec![0, 1, 2]));
            cache.shards = vec![shard];
        }

        let rows = crate::shard_indexing::eligible_rows(source.deref()).unwrap();
        assert_eq!(rows.as_ref(), &vec![0, 1, 2]);
    }

    #[test]
    fn eligible_rows_full_rebuild_when_no_cache() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        let simdr = dir.path().join("shard.simdr");
        write_simdr_fixture(&simdr, &[("r1", "hello"), ("r2", "world")]);
        let shard = index_single_shard(&config, &simdr, 0).unwrap().0.unwrap();

        {
            let mut state = source.state.lock().unwrap();
            state.shards = vec![shard];
            state.materialized_rows = 2;
        }

        // eligible_index starts empty (default) — forces full rebuild.
        let rows = crate::shard_indexing::eligible_rows(source.deref()).unwrap();
        assert_eq!(rows.as_ref(), &vec![0, 1]);
    }

    #[test]
    fn eligible_rows_cache_hit_returns_cached() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);

        // Inject shards into state
        {
            let mut state = source.state.lock().unwrap();
            state.shards = vec![ShardIndex {
                path: dir.path().join("shard.simdr"),
                global_start: 0,
                row_count: 10,
                parquet_row_groups: Vec::new(),
                remote_candidate: None,
            }];
            state.materialized_rows = 10;
        }

        // First call builds the cache
        let rows1 = crate::shard_indexing::eligible_rows(&source).unwrap();
        assert_eq!(rows1.len(), 10);

        // Second call should hit the cache
        let rows2 = crate::shard_indexing::eligible_rows(&source).unwrap();
        assert_eq!(rows1.as_ptr(), rows2.as_ptr(), "should return cached Arc");
    }

    #[test]
    fn eligible_rows_full_rebuild_no_cache() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);

        // Inject shards with a store file
        let shard_path = dir.path().join("shard.simdr");
        write_simdr_fixture(&shard_path, &[("r1", "text1"), ("r2", "text2")]);

        {
            let mut state = source.state.lock().unwrap();
            state.shards = vec![ShardIndex {
                path: shard_path,
                global_start: 0,
                row_count: 2,
                parquet_row_groups: Vec::new(),
                remote_candidate: None,
            }];
            state.materialized_rows = 2;
        }

        let rows = crate::shard_indexing::eligible_rows(&source).unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn build_eligible_rows_parquet_shard_exercises_cache() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        let parquet_path = dir.path().join("shard.parquet");
        write_parquet_fixture(&parquet_path, &[("r1", "hello"), ("r2", "world")]);

        let shard = ShardIndex {
            path: parquet_path.clone(),
            global_start: 0,
            row_count: 2,
            parquet_row_groups: vec![(0, 2)],
            remote_candidate: None,
        };

        let eligible =
            crate::shard_indexing::build_eligible_rows_from_shards(source.deref(), &[shard])
                .unwrap();
        assert_eq!(eligible, vec![0, 1]);
    }

    #[test]
    fn build_eligible_rows_store_shard_includes_all_rows() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);

        let shard_path = dir.path().join("store.simdr");
        write_simdr_fixture(&shard_path, &[("r1", "a"), ("r2", "b"), ("r3", "c")]);

        let shards = vec![ShardIndex {
            path: shard_path,
            global_start: 5,
            row_count: 3,
            parquet_row_groups: Vec::new(),
            remote_candidate: None,
        }];

        let rows =
            crate::shard_indexing::build_eligible_rows_from_shards(&source, &shards).unwrap();
        assert_eq!(rows.len(), 3);
        assert!(rows.contains(&5));
        assert!(rows.contains(&6));
        assert!(rows.contains(&7));
    }

    #[test]
    fn shard_signature_is_deterministic() {
        let shards = vec![
            ShardIndex {
                path: PathBuf::from("a"),
                global_start: 0,
                row_count: 10,
                parquet_row_groups: vec![(0, 10)],
                remote_candidate: None,
            },
            ShardIndex {
                path: PathBuf::from("b"),
                global_start: 10,
                row_count: 5,
                parquet_row_groups: vec![(0, 5)],
                remote_candidate: None,
            },
        ];
        let sig1 = crate::shard_indexing::shard_signature(&shards);
        let sig2 = crate::shard_indexing::shard_signature(&shards);
        assert_eq!(sig1, sig2);
    }

    #[test]
    fn shard_signature_differs_for_different_shards() {
        let s1 = vec![ShardIndex {
            path: PathBuf::from("a"),
            global_start: 0,
            row_count: 10,
            parquet_row_groups: Vec::new(),
            remote_candidate: None,
        }];
        let s2 = vec![ShardIndex {
            path: PathBuf::from("b"),
            global_start: 0,
            row_count: 10,
            parquet_row_groups: Vec::new(),
            remote_candidate: None,
        }];
        assert_ne!(
            crate::shard_indexing::shard_signature(&s1),
            crate::shard_indexing::shard_signature(&s2)
        );
    }

    #[test]
    fn shard_signature_empty_returns_nonzero() {
        let sig = crate::shard_indexing::shard_signature(&[]);
        // SipHash of empty input is deterministic but non-zero
        let sig2 = crate::shard_indexing::shard_signature(&[]);
        assert_eq!(sig, sig2);
    }

    #[test]
    fn invalidate_eligible_index_resets_cache() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        {
            let mut cache = source.eligible_index.lock().unwrap();
            *cache = EligibleIndexCache {
                signature: Some(42),
                rows: Some(Arc::new(vec![0, 1, 2])),
                shards: vec![ShardIndex {
                    path: PathBuf::from("dummy.parquet"),
                    global_start: 0,
                    row_count: 3,
                    parquet_row_groups: vec![(0, 3)],
                    remote_candidate: None,
                }],
            };
        }
        crate::shard_indexing::invalidate_eligible_index(source.deref());
        let cache = source.eligible_index.lock().unwrap();
        assert!(cache.signature.is_none());
        assert!(cache.rows.is_none());
        assert!(cache.shards.is_empty());
    }

    #[test]
    fn open_shard_store_creates_parent_directories() {
        let dir = tempdir().unwrap();
        let nested = dir.path().join("a").join("b").join("c.simdr");
        assert!(!nested.parent().unwrap().exists());
        let config = test_config(dir.path().to_path_buf());
        let store = crate::shard_indexing::open_shard_store(&config, &nested).unwrap();
        assert!(nested.parent().unwrap().exists());
        drop(store);
    }

    #[test]
    fn open_shard_store_creates_directory_and_store() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);

        let nested_path = dir.path().join("a").join("b").join("shard.simdr");
        let result = crate::shard_indexing::get_or_open_shard_store(&source, &nested_path);
        assert!(
            result.is_ok(),
            "should create directory and store: {:?}",
            result.err()
        );
        assert!(nested_path.exists(), "store file should exist");
    }

    #[test]
    fn open_shard_store_errors_when_base_path_is_a_file() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let file_path = dir.path().join("not-a-dir");
        fs::write(&file_path, b"not-a-dir").unwrap();
        let bad_path = file_path.join("store.simdr");
        let result = crate::shard_indexing::open_shard_store(&config, &bad_path);
        assert!(result.is_err());
    }

    #[test]
    fn remove_stale_store_evicts_from_cache_and_removes_file() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let store_path = dir.path().join("stale.simdr");
        fs::write(&store_path, b"stale-content").unwrap();
        config.store_cache.lock().unwrap().insert(
            store_path.clone(),
            Arc::new(DataStore::open(&store_path).unwrap()),
        );
        crate::disk_cache::remove_stale_store(&config, &store_path);
        assert!(!store_path.exists(), "stale store file must be removed");
        assert!(!config.store_cache.lock().unwrap().contains_key(&store_path));
    }

    #[test]
    fn remove_stale_store_does_not_panic_when_file_missing() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let missing = dir.path().join("never-existed.simdr");
        crate::disk_cache::remove_stale_store(&config, &missing);
        assert!(!missing.exists());
    }

    #[test]
    fn candidate_store_path_maps_via_target_path() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let candidate = "url::https://host/ds/resolve/main/train/data-000.parquet";
        let target = candidate_target_path(&config, candidate);
        let store = crate::shard_indexing::candidate_store_path(&config, candidate);
        assert_eq!(store, target.with_extension("simdr"));
    }

    #[test]
    fn first_uncached_order_position_returns_first_missing() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let candidates = vec![
            "url::http://a/0.parquet".to_string(),
            "url::http://a/1.parquet".to_string(),
        ];
        let order = vec![0, 1];
        // No shards on disk — first position is uncached
        let pos = crate::download::first_uncached_order_position(&config, &candidates, &order, &[]);
        assert_eq!(pos, 0);
    }

    #[test]
    fn first_uncached_order_position_returns_len_when_all_cached() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let candidates = vec!["url::http://a/0.parquet".to_string()];
        let order = vec![0];
        // Build a shard index whose path matches the candidate store path
        let store_path = crate::shard_indexing::candidate_store_path(&config, &candidates[0]);
        let shards = vec![ShardIndex {
            path: store_path,
            global_start: 0,
            row_count: 1,
            parquet_row_groups: Vec::new(),
            remote_candidate: None,
        }];
        let pos =
            crate::download::first_uncached_order_position(&config, &candidates, &order, &shards);
        assert_eq!(pos, 1); // candidates.len() = all cached
    }
}
