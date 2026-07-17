use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::{BufRead, BufReader};

use chrono::{DateTime, Utc};
use flate2::read::GzDecoder;
use rayon::prelude::*;
use serde_json::{Value, json};
use simd_r_drive::storage_engine::DataStore;
use simd_r_drive::storage_engine::traits::{DataStoreReader, DataStoreWriter};
use tracing::warn;

use crate::constants::{HF_SHARD_STORE_META_ROWS_KEY, HF_TEMP_DOWNLOAD_PREFIX};
use crate::file_utils::{is_gzip_path, is_transient_text, resolve_inner_extension};
use crate::huggingface_source::{ParquetGroupKey, ParquetGroupRequest, RowTextField, RowView};
use crate::shard_index::{is_store_shard_path, row_store_row_key, shard_store_path_for};
use crate::source_core::HuggingFaceRowSource;
use crate::types::ShardIndex;
use triplets_core::SamplerError;
use triplets_core::data::{DataRecord, QualityScore, SectionRole};
use triplets_core::utils::make_section;

/// Parse a raw row payload into normalized `RowView` fields.
pub(crate) fn parse_row(
    source: &HuggingFaceRowSource,
    absolute_idx: usize,
    row_value: &Value,
) -> Result<Option<RowView>, SamplerError> {
    if !source.config.has_explicit_mapping() {
        return Err(SamplerError::SourceInconsistent {
            source_id: source.config.source_id.clone(),
            details:
                "huggingface row parsing requires explicit field mapping; no columns configured"
                    .to_string(),
        });
    }

    let row_payload = row_value.get("row").unwrap_or(row_value);
    let row_obj = row_payload
        .as_object()
        .ok_or_else(|| SamplerError::SourceUnavailable {
            source_id: source.config.source_id.clone(),
            reason: "snapshot row entry missing JSON object payload".to_string(),
        })?;

    let row_id = source
        .config
        .id_column
        .as_ref()
        .and_then(|col| resolve_json_path(row_obj, col))
        .and_then(|v| value_to_text(&v))
        .unwrap_or_else(|| {
            format!(
                "{}:{}:{}",
                source.config.dataset_name, source.config.split_name, absolute_idx
            )
        });

    let mut text_fields = Vec::new();
    let use_role_columns = !source.config.anchor_columns.is_empty()
        || !source.config.positive_columns.is_empty()
        || !source.config.negative_columns.is_empty()
        || !source.config.context_columns.is_empty();

    if use_role_columns {
        // Anchor: try each candidate column in order; use the first
        // whose value is present and non-empty.  Skip the row when the
        // list is non-empty but no candidate yields content.
        if !source.config.anchor_columns.is_empty() {
            match coalesce_field(&source.config.anchor_columns, row_obj) {
                Some(field) => text_fields.push(field),
                None => return Ok(None),
            }
        }

        // Positive: try each candidate column in order; use the first
        // whose value is present and non-empty.  Skip the row when the
        // list is non-empty but no candidate yields content.
        if !source.config.positive_columns.is_empty() {
            match coalesce_field(&source.config.positive_columns, row_obj) {
                Some(field) => text_fields.push(field),
                None => return Ok(None),
            }
        }

        for name in &source.config.context_columns {
            let Some(value) = resolve_json_path(row_obj, name) else {
                return Ok(None);
            };
            let Some(text) = value_to_text(&value) else {
                return Ok(None);
            };
            text_fields.push(RowTextField {
                name: name.clone(),
                text,
            });
        }

        // Negative columns: expand list values into multiple Context sections.
        for name in &source.config.negative_columns {
            let Some(value) = resolve_json_path(row_obj, name) else {
                return Ok(None);
            };
            let Some(fields) = coalesce_list_field(name, &value) else {
                return Ok(None);
            };
            text_fields.extend(fields);
        }
    } else {
        // Text-columns mode: try each candidate column in order; use the
        // first whose value is present and non-empty.  The row is skipped
        // when no candidate yields content (handled by the is_empty guard
        // below).
        if let Some(field) = coalesce_field(&source.config.text_columns, row_obj) {
            text_fields.push(field);
        }
    }

    if text_fields.is_empty() {
        return Ok(None);
    }

    Ok(Some(RowView {
        row_id: Some(row_id),
        timestamp: None,
        text_fields,
    }))
}

/// Convert a `RowView` into a sampler `DataRecord`.
pub(crate) fn row_to_record(
    source: &HuggingFaceRowSource,
    row: &RowView,
    row_index: u64,
) -> Result<Option<DataRecord>, SamplerError> {
    if row.text_fields.is_empty() {
        return Ok(None);
    }

    let record_id = row
        .row_id
        .as_ref()
        .cloned()
        .unwrap_or_else(|| format!("row_{row_index}"));
    let id = format!("{}::{}", source.config.source_id, record_id);

    let mut sections = Vec::new();
    let anchor = &row.text_fields[0];
    sections.push(make_section(
        SectionRole::Anchor,
        Some(anchor.name.as_str()),
        anchor.text.as_str(),
    ));

    let positive = row.text_fields.get(1).unwrap_or(anchor);
    sections.push(make_section(
        SectionRole::Context,
        Some(positive.name.as_str()),
        positive.text.as_str(),
    ));

    for field in row.text_fields.iter().skip(2) {
        sections.push(make_section(
            SectionRole::Context,
            Some(field.name.as_str()),
            field.text.as_str(),
        ));
    }

    let timestamp = row.timestamp.unwrap_or(DateTime::<Utc>::UNIX_EPOCH);
    Ok(Some(DataRecord {
        id,
        source: source.config.source_id.clone(),
        created_at: timestamp,
        updated_at: timestamp,
        quality: source
            .config
            .trust_override
            .map_or_else(QualityScore::default, |t| QualityScore { trust: t }),
        taxonomy: vec![
            format!("dataset={}", source.config.dataset_name),
            format!("config={}", source.config.config_name),
            format!("split={}", source.config.split_name),
        ],
        sections,
        meta_prefix: None,
        label: None,
    }))
}

/// Materialize records for requested indices into output buffer.
pub(crate) fn read_row_batch(
    source: &HuggingFaceRowSource,
    indices: &[usize],
    out: &mut Vec<DataRecord>,
    limit: Option<usize>,
) -> Result<(), SamplerError> {
    let mut sorted = indices.to_vec();
    sorted.sort_unstable();

    let mut fetched = HashMap::with_capacity(sorted.len());
    let mut pending = Vec::new();
    for idx in &sorted {
        if !source.ensure_row_available(*idx)? {
            fetched.insert(*idx, None);
            continue;
        }

        if let Some(row) = source
            .cache
            .lock()
            .map_err(|_| SamplerError::SourceUnavailable {
                source_id: source.config.source_id.clone(),
                reason: "huggingface row cache lock poisoned".to_string(),
            })?
            .get(*idx)
        {
            let record = row_to_record(source, &row, *idx as u64)?;
            fetched.insert(*idx, record);
            continue;
        }

        pending.push(*idx);
    }

    if !pending.is_empty() {
        let resolutions = {
            let state = source
                .state
                .lock()
                .map_err(|_| SamplerError::SourceUnavailable {
                    source_id: source.config.source_id.clone(),
                    reason: "huggingface source state lock poisoned".to_string(),
                })?;
            let mut resolved = Vec::with_capacity(pending.len());
            for idx in &pending {
                let (shard, local_idx) = crate::shard_indexing::locate_shard(&state.shards, *idx)
                    .ok_or_else(|| SamplerError::SourceUnavailable {
                    source_id: source.config.source_id.clone(),
                    reason: format!("row index out of range: {idx}"),
                })?;
                resolved.push((*idx, shard.clone(), local_idx));
            }
            resolved
        };

        let mut parquet_groups: HashMap<ParquetGroupKey, Vec<ParquetGroupRequest>> = HashMap::new();
        for (idx, shard, local_idx) in resolutions {
            let (group_pos, local_in_group) =
                crate::shard_indexing::locate_parquet_group(source, &shard, local_idx)?;
            parquet_groups
                .entry((shard.path.clone(), group_pos))
                .or_default()
                .push((idx, local_in_group, shard));
        }

        for ((shard_path, group_pos), mut requested) in parquet_groups {
            requested.sort_by_key(|(_, local_in_group, _)| *local_in_group);
            let shard = requested
                .first()
                .map(|(_, _, shard)| shard.clone())
                .ok_or_else(|| SamplerError::SourceUnavailable {
                    source_id: source.config.source_id.clone(),
                    reason: format!(
                        "missing parquet request metadata for shard {} row_group {}",
                        shard_path.display(),
                        group_pos
                    ),
                })?;

            let mut targets: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
            for (idx, local_in_group, _) in requested {
                targets.entry(local_in_group).or_default().push(idx);
            }

            let (group_start, _) = shard.parquet_row_groups[group_pos];
            let mut unresolved_targets: BTreeMap<usize, Vec<usize>> = targets.clone();

            if is_store_shard_path(&shard.path) {
                let store = crate::shard_indexing::get_or_open_shard_store(source, &shard.path)?;
                let requested_positions = targets.keys().copied().collect::<Vec<_>>();
                let store_keys = requested_positions
                    .iter()
                    .map(|position| {
                        let local_idx = group_start.saturating_add(*position);
                        row_store_row_key(local_idx)
                    })
                    .collect::<Vec<_>>();
                let store_key_refs = store_keys
                    .iter()
                    .map(|key| key.as_slice())
                    .collect::<Vec<_>>();
                let store_entries = store.batch_read(&store_key_refs).map_err(|err| {
                    SamplerError::SourceUnavailable {
                        source_id: source.config.source_id.clone(),
                        reason: format!("row-store batch read failed: {err}"),
                    }
                })?;

                unresolved_targets.clear();
                for (position, entry) in requested_positions.into_iter().zip(store_entries) {
                    let Some(indices_for_position) = targets.get(&position).cloned() else {
                        continue;
                    };
                    let Some(entry) = entry else {
                        unresolved_targets.insert(position, indices_for_position);
                        continue;
                    };

                    let row = decode_row_view(source, entry.as_ref())?;
                    for idx in indices_for_position {
                        let record = row_to_record(source, &row, idx as u64)?;
                        if let Some(record) = record {
                            source
                                .cache
                                .lock()
                                .map_err(|_| SamplerError::SourceUnavailable {
                                    source_id: source.config.source_id.clone(),
                                    reason: "huggingface row cache lock poisoned".to_string(),
                                })?
                                .insert(idx, row.clone(), source.config.cache_capacity);
                            fetched.insert(idx, Some(record));
                        } else {
                            fetched.insert(idx, None);
                        }
                    }
                }

                if unresolved_targets.is_empty() {
                    continue;
                }

                let missing = unresolved_targets
                    .keys()
                    .copied()
                    .map(|value| value.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                return Err(SamplerError::SourceUnavailable {
                    source_id: source.config.source_id.clone(),
                    reason: format!(
                        "row-store rows missing in shard {} row_group {} at local offsets [{}]",
                        shard.path.display(),
                        group_pos,
                        missing
                    ),
                });
            }

            let row_group_rows = source
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

            let mut missing_offsets = Vec::new();
            for (position, indices_for_position) in unresolved_targets {
                let Some(row_value) = row_group_rows.get(position) else {
                    missing_offsets.push(position);
                    continue;
                };

                let parsed = indices_for_position
                    .into_par_iter()
                    .map(|idx| {
                        let row = parse_row(source, idx, row_value)?;
                        if let Some(row) = row {
                            let record = row_to_record(source, &row, idx as u64)?;
                            Ok((idx, Some(row), record))
                        } else {
                            Ok((idx, None, None))
                        }
                    })
                    .collect::<Result<Vec<_>, SamplerError>>()?;

                for (idx, row, record) in parsed {
                    if let Some(row) = row {
                        source
                            .cache
                            .lock()
                            .map_err(|_| SamplerError::SourceUnavailable {
                                source_id: source.config.source_id.clone(),
                                reason: "huggingface row cache lock poisoned".to_string(),
                            })?
                            .insert(idx, row, source.config.cache_capacity);
                        fetched.insert(idx, record);
                    } else {
                        fetched.insert(idx, None);
                    }
                }
            }

            if !missing_offsets.is_empty() {
                let missing = missing_offsets
                    .into_iter()
                    .map(|value| value.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                return Err(SamplerError::SourceUnavailable {
                    source_id: source.config.source_id.clone(),
                    reason: format!(
                        "parquet rows missing in shard {} row_group {} at local offsets [{}]",
                        shard.path.display(),
                        group_pos,
                        missing
                    ),
                });
            }
        }
    }

    for idx in indices {
        if limit.is_some_and(|max| out.len() >= max) {
            break;
        }
        if let Some(record) = fetched.remove(idx).flatten() {
            out.push(record);
        }
    }
    Ok(())
}

pub(crate) fn transcode_transient_shard_to_store(
    source: &HuggingFaceRowSource,
    shard: &ShardIndex,
) -> Result<Option<ShardIndex>, SamplerError> {
    if is_store_shard_path(&shard.path) {
        return Ok(Some(shard.clone()));
    }

    let store_path = shard_store_path_for(&shard.path);
    let in_manifest = shard.path.starts_with(source.manifest_cache_root());
    let is_temp_download = shard
        .path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .starts_with(HF_TEMP_DOWNLOAD_PREFIX);
    let can_delete_transient = in_manifest || is_temp_download;

    let store = crate::shard_indexing::get_or_open_shard_store(source, &store_path)?;
    if store_path.exists() {
        let existing_rows = read_store_row_count(source, &store)?;
        if existing_rows > 0 {
            // Simdr store is already fully populated.  Clean up the
            // transient source file if it is still present.
            if shard.path != store_path
                && shard.path.exists()
                && let Err(err) = fs::remove_file(&shard.path)
            {
                warn!(
                    "[triplets:hf] failed removing stale transient shard after store hit {}: {}",
                    shard.path.display(),
                    err
                );
            }
            return Ok(Some(ShardIndex {
                path: store_path,
                global_start: shard.global_start,
                row_count: existing_rows,
                parquet_row_groups: vec![(0, existing_rows)],
                remote_candidate: shard.remote_candidate.clone(),
            }));
        }
    }

    let mut served_rows = 0usize;

    if is_gzip_path(&shard.path) || is_transient_text(&shard.path) {
        // Transcode .jsonl.gz / .jsonl / .ndjson → .simdr store
        let file =
            std::fs::File::open(&shard.path).map_err(|err| SamplerError::SourceUnavailable {
                source_id: source.config.source_id.clone(),
                reason: format!(
                    "failed opening text shard for transcode {}: {err}",
                    shard.path.display()
                ),
            })?;
        let reader: Box<dyn BufRead> = if is_gzip_path(&shard.path) {
            Box::new(BufReader::new(GzDecoder::new(file)))
        } else {
            Box::new(BufReader::new(file))
        };
        let mut batch: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(1024);

        for (local_idx, line_result) in reader.lines().enumerate() {
            let line = line_result.map_err(|err| SamplerError::SourceUnavailable {
                source_id: source.config.source_id.clone(),
                reason: format!("failed reading text shard {}: {err}", shard.path.display()),
            })?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            // Use served_rows for absolute_idx to maintain bounded, non-overlapping IDs.
            // local_idx is only used for error reporting in parse_non_parquet_line.
            let absolute_idx = shard.global_start.saturating_add(served_rows);
            let line_value = parse_non_parquet_line(source, shard, local_idx, trimmed)?;

            let Some(row) = parse_row(source, absolute_idx, &line_value)? else {
                continue;
            };

            let key = row_store_row_key(served_rows);
            let payload = encode_row_view(source, &row)?;
            batch.push((key, payload));
            served_rows = served_rows.saturating_add(1);

            // Flush batch periodically
            if batch.len() >= 1024 {
                let refs: Vec<(&[u8], &[u8])> = batch
                    .iter()
                    .map(|(key, payload)| (key.as_slice(), payload.as_slice()))
                    .collect();
                store
                    .batch_write(&refs)
                    .map_err(|err| SamplerError::SourceUnavailable {
                        source_id: source.config.source_id.clone(),
                        reason: format!("row-store batch write failed: {err}"),
                    })?;
                batch.clear();
            }
        }

        // Flush remaining batch
        if !batch.is_empty() {
            let refs: Vec<(&[u8], &[u8])> = batch
                .iter()
                .map(|(key, payload)| (key.as_slice(), payload.as_slice()))
                .collect();
            store
                .batch_write(&refs)
                .map_err(|err| SamplerError::SourceUnavailable {
                    source_id: source.config.source_id.clone(),
                    reason: format!("row-store batch write failed: {err}"),
                })?;
        }
    } else {
        // Parquet binary decoding
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

            let mut batch: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(group_count);

            for local_in_group in 0..group_count {
                let local_idx = group_start.saturating_add(local_in_group);
                if local_idx >= shard.row_count {
                    break;
                }
                let Some(row_value) = rows.get(local_in_group) else {
                    break;
                };
                let absolute_idx = shard.global_start.saturating_add(served_rows);
                let Some(row) = parse_row(source, absolute_idx, row_value)? else {
                    continue;
                };

                let key = row_store_row_key(served_rows);
                let payload = encode_row_view(source, &row)?;
                batch.push((key, payload));
                served_rows = served_rows.saturating_add(1);
            }

            if !batch.is_empty() {
                let refs: Vec<(&[u8], &[u8])> = batch
                    .iter()
                    .map(|(key, payload)| (key.as_slice(), payload.as_slice()))
                    .collect();
                store
                    .batch_write(&refs)
                    .map_err(|err| SamplerError::SourceUnavailable {
                        source_id: source.config.source_id.clone(),
                        reason: format!("row-store batch write failed: {err}"),
                    })?;
            }
        }
    }

    write_store_row_count(source, &store, served_rows)?;

    // Only delete transient files inside the managed manifest root or
    // system temp downloads, never delete user-provided local files.
    if shard.path != store_path && shard.path.exists() && can_delete_transient {
        fs::remove_file(&shard.path).map_err(|err| SamplerError::SourceUnavailable {
            source_id: source.config.source_id.clone(),
            reason: format!(
                "failed removing transient shard after store transcode {}: {err}",
                shard.path.display()
            ),
        })?;
    }

    if served_rows == 0 {
        return Ok(None);
    }

    Ok(Some(ShardIndex {
        path: store_path,
        global_start: shard.global_start,
        row_count: served_rows,
        parquet_row_groups: vec![(0, served_rows)],
        remote_candidate: shard.remote_candidate.clone(),
    }))
}

fn encode_row_view(source: &HuggingFaceRowSource, row: &RowView) -> Result<Vec<u8>, SamplerError> {
    serde_json::to_vec(row).map_err(|err| SamplerError::SourceUnavailable {
        source_id: source.config.source_id.clone(),
        reason: format!("failed encoding row-view payload: {err}"),
    })
}

fn decode_row_view(source: &HuggingFaceRowSource, bytes: &[u8]) -> Result<RowView, SamplerError> {
    serde_json::from_slice(bytes).map_err(|err| SamplerError::SourceUnavailable {
        source_id: source.config.source_id.clone(),
        reason: format!("failed decoding row-view payload: {err}"),
    })
}

/// Convert a serde JSON value into non-empty text when possible.
///
/// `label_names` optionally provides an ordered list of label strings for
/// ClassLabel-style integer columns.  When the value is an integer `n` and
/// `label_names[n]` exists, that label string is returned instead of the
/// raw numeric string.
pub(crate) fn value_to_text(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(s) => {
            if s.trim().is_empty() {
                None
            } else {
                Some(s.clone())
            }
        }
        Value::Bool(b) => Some(b.to_string()),
        Value::Number(n) => Some(n.to_string()),
        Value::Array(arr) => {
            for element in arr {
                if let Some(text) = value_to_text(element) {
                    return Some(text);
                }
            }
            None
        }
        Value::Object(obj) => serde_json::to_string(obj).ok().filter(|s| !s.is_empty()),
    }
}

/// Resolve a dot-separated column path against a JSON object.
///
/// Splits `name` on `.` and walks the JSON hierarchy.  For example
/// `"set.query"` resolves to `row["set"]["query"]`.  A bare name
/// without dots (e.g. `"query"`) performs a simple top-level lookup.
///
/// Returns `None` when any segment of the path is missing or when an
/// intermediate value is not a JSON object.
pub(crate) fn resolve_json_path(
    row_obj: &serde_json::Map<String, Value>,
    name: &str,
) -> Option<Value> {
    let mut current = Value::Object(row_obj.clone());
    for segment in name.split('.') {
        current = match current {
            Value::Object(map) => map.get(segment)?.clone(),
            _ => return None,
        };
    }
    Some(current)
}

/// Try each candidate column name in order and return the first one that
/// yields a non-empty text value.  Returns `None` when no candidate
/// matches, which the caller uses to decide whether to skip the row.
fn coalesce_field(
    candidates: &[String],
    row_obj: &serde_json::Map<String, Value>,
) -> Option<RowTextField> {
    for name in candidates {
        if let Some(ref value) = resolve_json_path(row_obj, name)
            && let Some(text) = value_to_text(value)
        {
            return Some(RowTextField {
                name: name.clone(),
                text,
            });
        }
    }
    None
}

/// Extract all elements from a list-valued column, returning one
/// `RowTextField` per element.
///
/// For non-list values, returns a single-element vector (same behavior
/// as `coalesce_field`).  For lists, each non-empty element becomes a
/// separate field.  Empty/blank elements are skipped.  Returns `None`
/// when no non-empty elements are found.
fn coalesce_list_field(name: &str, value: &Value) -> Option<Vec<RowTextField>> {
    match value {
        Value::Array(arr) => {
            let fields: Vec<RowTextField> = arr
                .iter()
                .enumerate()
                .filter_map(|(i, element)| {
                    value_to_text(element).map(|text| RowTextField {
                        name: format!("{name}[{i}]"),
                        text,
                    })
                })
                .collect();
            if fields.is_empty() {
                None
            } else {
                Some(fields)
            }
        }
        other => value_to_text(other).map(|text| {
            vec![RowTextField {
                name: name.to_string(),
                text,
            }]
        }),
    }
}

/// Decode one line from a non-parquet shard into an object-like row payload.
fn parse_non_parquet_line(
    source: &HuggingFaceRowSource,
    shard: &ShardIndex,
    local_idx: usize,
    line: &str,
) -> Result<Value, SamplerError> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Err(SamplerError::SourceInconsistent {
            source_id: source.config.source_id.clone(),
            details: format!(
                "empty row in shard {} at local index {}",
                shard.path.display(),
                local_idx
            ),
        });
    }

    let is_strict_json_lines =
        resolve_inner_extension(&shard.path).is_some_and(|ext| ext == "jsonl" || ext == "ndjson");

    match serde_json::from_str::<Value>(trimmed) {
        Ok(value) => {
            let payload = value.get("row").unwrap_or(&value);
            if payload.is_object() {
                Ok(value)
            } else if let Some(text) = value_to_text(payload) {
                Ok(json!({ "text": text }))
            } else {
                Err(SamplerError::SourceInconsistent {
                    source_id: source.config.source_id.clone(),
                    details: format!(
                        "non-object JSON row in shard {} at local index {} could not be converted to text",
                        shard.path.display(),
                        local_idx
                    ),
                })
            }
        }
        Err(err) => {
            if is_strict_json_lines {
                Err(SamplerError::SourceInconsistent {
                    source_id: source.config.source_id.clone(),
                    details: format!(
                        "failed decoding JSON row from shard {} at local index {}: {err}",
                        shard.path.display(),
                        local_idx
                    ),
                })
            } else {
                Ok(json!({ "text": trimmed }))
            }
        }
    }
}

pub(crate) fn read_store_row_count(
    source: &HuggingFaceRowSource,
    store: &DataStore,
) -> Result<usize, SamplerError> {
    let Some(entry) = store.read(HF_SHARD_STORE_META_ROWS_KEY).map_err(|err| {
        SamplerError::SourceUnavailable {
            source_id: source.config.source_id.clone(),
            reason: format!("row-store meta read failed: {err}"),
        }
    })?
    else {
        return Ok(0);
    };

    let bytes = entry.as_ref();
    if bytes.len() != std::mem::size_of::<u64>() {
        return Err(SamplerError::SourceUnavailable {
            source_id: source.config.source_id.clone(),
            reason: "row-store meta payload size mismatch".to_string(),
        });
    }
    let mut raw = [0u8; 8];
    raw.copy_from_slice(bytes);
    Ok(u64::from_le_bytes(raw) as usize)
}

pub(crate) fn write_store_row_count(
    source: &HuggingFaceRowSource,
    store: &DataStore,
    rows: usize,
) -> Result<(), SamplerError> {
    let payload = (rows as u64).to_le_bytes();
    store
        .write(HF_SHARD_STORE_META_ROWS_KEY, payload.as_slice())
        .map_err(|err| SamplerError::SourceUnavailable {
            source_id: source.config.source_id.clone(),
            reason: format!("row-store meta write failed: {err}"),
        })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shard_index::index_single_shard;
    use crate::test_utils::{
        test_config, test_source, write_gzip_fixture, write_parquet_fixture, write_simdr_fixture,
    };
    use crate::types::ShardIndex;
    use serde_json::json;
    use tempfile::tempdir;

    // ── Phase 1a: coalesce_list_field (pure function) ──────────────────────

    #[test]
    fn coalesce_list_field_array_with_nonempty_elements() {
        let result = coalesce_list_field("neg", &json!(["a", "b", "c"]));
        let fields = result.expect("should return Some");
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].name, "neg[0]");
        assert_eq!(fields[0].text, "a");
        assert_eq!(fields[1].name, "neg[1]");
        assert_eq!(fields[1].text, "b");
        assert_eq!(fields[2].name, "neg[2]");
        assert_eq!(fields[2].text, "c");
    }

    #[test]
    fn coalesce_list_field_array_all_empty_returns_none() {
        assert!(coalesce_list_field("neg", &json!(["", "  ", null])).is_none());
    }

    #[test]
    fn coalesce_list_field_array_mixed_empty_and_nonempty() {
        let result = coalesce_list_field("neg", &json!(["", "keep", null]));
        let fields = result.expect("should return Some");
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].name, "neg[1]");
        assert_eq!(fields[0].text, "keep");
    }

    #[test]
    fn coalesce_list_field_scalar_string() {
        let result = coalesce_list_field("neg", &json!("single"));
        let fields = result.expect("should return Some");
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].name, "neg");
        assert_eq!(fields[0].text, "single");
    }

    #[test]
    fn coalesce_list_field_scalar_number() {
        let result = coalesce_list_field("score", &json!(42));
        let fields = result.expect("should return Some");
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].text, "42");
    }

    #[test]
    fn coalesce_list_field_null_returns_none() {
        assert!(coalesce_list_field("neg", &json!(null)).is_none());
    }

    #[test]
    fn coalesce_list_field_bool_returns_some() {
        let result = coalesce_list_field("flag", &json!(true));
        let fields = result.expect("should return Some");
        assert_eq!(fields[0].text, "true");
    }

    // ── Phase 1b: parse_non_parquet_line ───────────────────────────────────

    fn make_source_and_shard(
        dir: &tempfile::TempDir,
        ext: &str,
    ) -> (crate::test_utils::SafeTestSource, ShardIndex) {
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let path = dir.path().join(format!("shard.{ext}"));
        let shard = ShardIndex {
            path,
            global_start: 0,
            row_count: 0,
            parquet_row_groups: Vec::new(),
            remote_candidate: None,
        };
        (source, shard)
    }

    #[test]
    fn parse_non_parquet_json_object_ndjson() {
        let dir = tempdir().unwrap();
        let (source, shard) = make_source_and_shard(&dir, "ndjson");
        let result = parse_non_parquet_line(&source, &shard, 0, r#"{"id":"r1","text":"hello"}"#);
        let value = result.expect("should parse");
        assert!(value.is_object());
        assert_eq!(value["id"], "r1");
    }

    #[test]
    fn parse_non_parquet_json_scalar_ndjson() {
        let dir = tempdir().unwrap();
        let (source, shard) = make_source_and_shard(&dir, "ndjson");
        let result = parse_non_parquet_line(&source, &shard, 0, r#""hello""#);
        let value = result.expect("should parse");
        assert_eq!(value["text"], "hello");
    }

    #[test]
    fn parse_non_parquet_plain_text_txt() {
        let dir = tempdir().unwrap();
        let (source, shard) = make_source_and_shard(&dir, "txt");
        let result = parse_non_parquet_line(&source, &shard, 0, "hello world");
        let value = result.expect("should parse");
        assert_eq!(value["text"], "hello world");
    }

    #[test]
    fn parse_non_parquet_invalid_json_strict_jsonl() {
        let dir = tempdir().unwrap();
        let (source, shard) = make_source_and_shard(&dir, "jsonl");
        let result = parse_non_parquet_line(&source, &shard, 0, "not json");
        assert!(result.is_err(), "strict JSONL should reject invalid JSON");
    }

    #[test]
    fn parse_non_parquet_invalid_json_lenient_txt() {
        let dir = tempdir().unwrap();
        let (source, shard) = make_source_and_shard(&dir, "txt");
        let result = parse_non_parquet_line(&source, &shard, 0, "not json");
        let value = result.expect("lenient txt should accept plain text");
        assert_eq!(value["text"], "not json");
    }

    #[test]
    fn parse_non_parquet_empty_line_returns_error() {
        let dir = tempdir().unwrap();
        let (source, shard) = make_source_and_shard(&dir, "ndjson");
        let result = parse_non_parquet_line(&source, &shard, 0, "");
        assert!(result.is_err(), "empty line should error");
    }

    #[test]
    fn parse_non_parquet_json_with_row_wrapper() {
        let dir = tempdir().unwrap();
        let (source, shard) = make_source_and_shard(&dir, "ndjson");
        let result =
            parse_non_parquet_line(&source, &shard, 0, r#"{"row":{"id":"r1","text":"hi"}}"#);
        let value = result.expect("should unwrap row wrapper");
        assert_eq!(value["row"]["id"], "r1");
    }

    #[test]
    fn parse_non_parquet_json_null_scalar_returns_error() {
        let dir = tempdir().unwrap();
        let (source, shard) = make_source_and_shard(&dir, "ndjson");
        let result = parse_non_parquet_line(&source, &shard, 0, "null");
        assert!(
            result.is_err(),
            "null scalar should error (cannot convert to text)"
        );
    }

    // ── Phase 1d: encode_row_view / decode_row_view round-trip ─────────────

    #[test]
    fn encode_decode_row_view_round_trip() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);

        let row = RowView {
            row_id: Some("r1".to_string()),
            timestamp: None,
            text_fields: vec![
                RowTextField {
                    name: "anchor".to_string(),
                    text: "hello".to_string(),
                },
                RowTextField {
                    name: "positive".to_string(),
                    text: "world".to_string(),
                },
            ],
        };

        let encoded = encode_row_view(&source, &row).expect("encode should succeed");
        let decoded = decode_row_view(&source, &encoded).expect("decode should succeed");

        assert_eq!(decoded.row_id, Some("r1".to_string()));
        assert_eq!(decoded.text_fields.len(), 2);
        assert_eq!(decoded.text_fields[0].name, "anchor");
        assert_eq!(decoded.text_fields[0].text, "hello");
        assert_eq!(decoded.text_fields[1].name, "positive");
        assert_eq!(decoded.text_fields[1].text, "world");
    }

    #[test]
    fn encode_decode_row_view_empty_fields() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);

        let row = RowView {
            row_id: None,
            timestamp: None,
            text_fields: Vec::new(),
        };

        let encoded = encode_row_view(&source, &row).expect("encode should succeed");
        let decoded = decode_row_view(&source, &encoded).expect("decode should succeed");

        assert!(decoded.row_id.is_none());
        assert!(decoded.text_fields.is_empty());
    }

    #[test]
    fn decode_row_view_truncated_input_returns_error() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);

        let result = decode_row_view(&source, b"{\"row_id\":");
        assert!(result.is_err(), "truncated JSON should fail");
    }

    #[test]
    fn decode_row_view_invalid_json_returns_error() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);

        let result = decode_row_view(&source, b"not valid json at all");
        assert!(result.is_err(), "invalid JSON should fail");
    }

    #[test]
    fn coalesce_list_field_mixed_types_array() {
        let result = coalesce_list_field("data", &json!(["text", 42, true]));
        let fields = result.expect("should return Some");
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].text, "text");
        assert_eq!(fields[1].text, "42");
        assert_eq!(fields[2].text, "true");
    }

    #[test]
    fn coalesce_list_field_nested_array() {
        let result = coalesce_list_field("data", &json!([["a", "b"], ["c"]]));
        let fields = result.expect("should return Some");
        assert_eq!(fields.len(), 2);
        // Nested arrays are flattened by value_to_text - returns first non-empty text
        assert_eq!(fields[0].text, "a");
        assert_eq!(fields[1].text, "c");
    }

    #[test]
    fn parse_row_role_columns_mode_builds_expected_fields() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.anchor_columns = vec!["anchor".into()];
        config.positive_columns = vec!["positive".into()];
        config.context_columns = vec!["ctx1".into(), "ctx2".into()];
        let source = test_source(config);

        let row = parse_row(
            &source,
            2,
            &json!({"id":"r","anchor":"a","positive":"p","ctx1":"c1","ctx2":2}),
        )
        .unwrap()
        .unwrap();
        assert_eq!(row.text_fields.len(), 4);
        assert_eq!(row.text_fields[0].name, "anchor");
        assert_eq!(row.text_fields[1].name, "positive");
    }

    #[test]
    fn parse_row_role_columns_mode_skips_missing_or_empty_values() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.anchor_columns = vec!["anchor".into()];
        config.context_columns = vec!["ctx".into()];
        let source = test_source(config);

        let missing = parse_row(&source, 0, &json!({"anchor":"a"}));
        assert!(missing.unwrap().is_none());

        let empty_anchor = parse_row(&source, 1, &json!({"anchor":"   ", "ctx":"ok"}));
        assert!(empty_anchor.unwrap().is_none());
    }

    #[test]
    fn parse_row_falls_back_to_synthetic_id_when_missing_id_column() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.id_column = Some("id".into());
        let source = test_source(config);

        let row = parse_row(&source, 42, &json!({"text": "hello"}))
            .unwrap()
            .unwrap();
        assert_eq!(row.row_id, Some("org/dataset:train:42".to_string()));
    }

    #[test]
    fn parse_row_text_columns_accept_numeric_values() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.text_columns = vec!["score".into()];
        let source = test_source(config);

        let row = parse_row(&source, 0, &json!({"score": 123}))
            .unwrap()
            .unwrap();
        assert_eq!(row.text_fields.len(), 1);
        assert_eq!(row.text_fields[0].text, "123");
    }

    #[test]
    fn parse_row_uses_explicit_text_columns() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.id_column = Some("id".into());
        config.text_columns = vec!["title".into(), "body".into()];
        let source = test_source(config);

        let row = parse_row(
            &source,
            5,
            &json!({
                "id": "row-5",
                "title": "Anchor text",
                "body": "Context text",
                "flag": true
            }),
        )
        .unwrap()
        .unwrap();

        // Candidate coalescing: the first non-empty column (title) is selected;
        // body is never tried because title already yielded a value.
        assert_eq!(row.row_id.as_deref(), Some("row-5"));
        assert_eq!(row.text_fields.len(), 1);
        assert_eq!(row.text_fields[0].name, "title");
        assert_eq!(row.text_fields[0].text, "Anchor text");
        assert!(row.text_fields.iter().all(|field| field.name != "id"));
    }

    #[test]
    fn parse_row_with_required_columns_skips_when_missing() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.anchor_columns = vec!["anchor".into()];
        config.positive_columns = vec!["positive".into()];
        config.context_columns = vec!["context".into()];
        let source = test_source(config);

        let parsed = parse_row(&source, 0, &json!({"anchor": "x", "context": "z"}));
        assert!(parsed.unwrap().is_none());
    }

    #[test]
    fn parse_row_errors_when_payload_is_not_object() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);

        let err = parse_row(&source, 0, &json!("not-an-object"));
        assert!(err.is_err());
    }

    #[test]
    fn parse_row_supports_row_wrapped_payload_and_text_columns() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.text_columns = vec!["headline".into(), "body".into()];
        config.id_column = Some("rid".into());
        let source = test_source(config);

        let parsed = parse_row(
            &source,
            0,
            &json!({"row": {"rid": "r-1", "headline": "h", "body": "b"}}),
        )
        .unwrap()
        .unwrap();

        // Candidate coalescing: headline is non-empty so it is selected;
        // body is not tried.
        assert_eq!(parsed.row_id.as_deref(), Some("r-1"));
        assert_eq!(parsed.text_fields.len(), 1);
        assert_eq!(parsed.text_fields[0].name, "headline");
    }

    #[test]
    fn parse_row_returns_none_when_all_positive_or_text_candidates_are_missing() {
        let dir = tempdir().unwrap();

        // Role mode: all positive_columns candidates absent → row skipped.
        let mut role_config = test_config(dir.path().to_path_buf());
        role_config.anchor_columns = vec!["anchor".into()];
        role_config.positive_columns = vec!["positive".into()];
        let role_source = test_source(role_config);

        let role_missing = parse_row(&role_source, 0, &json!({"anchor":"a"})).unwrap();
        assert!(role_missing.is_none());

        // Text-columns mode: a row that lacks all listed candidates → row skipped.
        let mut text_config = test_config(dir.path().to_path_buf());
        text_config.text_columns = vec!["title".into(), "body".into()];
        let text_source = test_source(text_config);
        // Row has neither "title" nor "body" → no candidate matches → skipped.
        let text_missing =
            parse_row(&text_source, 1, &json!({"other_field": "irrelevant"})).unwrap();
        assert!(text_missing.is_none());
    }

    #[test]
    fn parse_row_text_columns_coalesces_to_first_nonempty_candidate() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.text_columns = vec!["title".into(), "body".into()];
        let source = test_source(config);

        // "title" is empty string → coalesces to "body".
        let row = parse_row(
            &source,
            0,
            &json!({"title": "", "body": "fallback content"}),
        )
        .unwrap()
        .unwrap();
        assert_eq!(row.text_fields.len(), 1);
        assert_eq!(row.text_fields[0].name, "body");
        assert_eq!(row.text_fields[0].text, "fallback content");

        // "title" is present and non-empty → it is used; "body" is never tried.
        let row2 = parse_row(
            &source,
            1,
            &json!({"title": "primary content", "body": "ignored"}),
        )
        .unwrap()
        .unwrap();
        assert_eq!(row2.text_fields.len(), 1);
        assert_eq!(row2.text_fields[0].name, "title");
        assert_eq!(row2.text_fields[0].text, "primary content");
    }

    #[test]
    fn parse_row_positive_columns_coalesces_to_first_nonempty_candidate() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.anchor_columns = vec!["anchor".into()];
        config.positive_columns = vec!["summary".into(), "body".into()];
        let source = test_source(config);

        // "summary" is absent → coalesces to "body".
        let row = parse_row(
            &source,
            0,
            &json!({"anchor": "a", "body": "fallback positive"}),
        )
        .unwrap()
        .unwrap();
        assert_eq!(row.text_fields.len(), 2);
        assert_eq!(row.text_fields[0].name, "anchor");
        assert_eq!(row.text_fields[1].name, "body");

        // "summary" is present and non-empty → it is used; "body" is ignored.
        let row2 = parse_row(
            &source,
            1,
            &json!({"anchor": "a", "summary": "chosen", "body": "ignored"}),
        )
        .unwrap()
        .unwrap();
        assert_eq!(row2.text_fields.len(), 2);
        assert_eq!(row2.text_fields[1].name, "summary");
        assert_eq!(row2.text_fields[1].text, "chosen");

        // Both positive candidates absent → row skipped.
        let none = parse_row(&source, 2, &json!({"anchor": "a"})).unwrap();
        assert!(none.is_none());
    }

    #[test]
    fn parse_row_anchor_columns_coalesces_to_first_nonempty_candidate() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.anchor_columns = vec!["headline".into(), "title".into()];
        config.positive_columns = vec!["body".into()];
        let source = test_source(config);

        // "headline" is absent → coalesces to "title".
        let row = parse_row(
            &source,
            0,
            &json!({"title": "fallback anchor", "body": "positive text"}),
        )
        .unwrap()
        .unwrap();
        assert_eq!(row.text_fields.len(), 2);
        assert_eq!(row.text_fields[0].name, "title");
        assert_eq!(row.text_fields[0].text, "fallback anchor");

        // "headline" is present and non-empty → it is used; "title" is ignored.
        let row2 = parse_row(
            &source,
            1,
            &json!({"headline": "chosen anchor", "title": "ignored", "body": "positive"}),
        )
        .unwrap()
        .unwrap();
        assert_eq!(row2.text_fields[0].name, "headline");
        assert_eq!(row2.text_fields[0].text, "chosen anchor");

        // Both anchor candidates absent → row skipped.
        let none = parse_row(&source, 2, &json!({"body": "positive only"})).unwrap();
        assert!(none.is_none());
    }

    #[test]
    fn parse_row_errors_when_no_mapping_is_configured() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.id_column = Some("id".into());
        config.text_columns.clear();
        let source = test_source(config);

        let parsed = parse_row(&source, 7, &json!({"id":"only-id"}));
        assert!(matches!(
            parsed,
            Err(SamplerError::SourceInconsistent { .. })
        ));
    }

    #[test]
    fn parse_row_dict_dataset() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.anchor_columns = vec!["set.query".into()];
        config.positive_columns = vec!["set.pos".into()];
        config.negative_columns = vec!["set.neg".into()];
        config.id_column = None;
        let source = test_source(config);

        let row = parse_row(
            &source,
            0,
            &json!({
                "set": {
                    "query": "What is Rust?",
                    "pos": ["How does Rust work?"],
                    "neg": ["Is Python better?", "What is Java?", "Tell me a joke"]
                }
            }),
        )
        .unwrap()
        .unwrap();

        // anchor = set.query, positive = first set.pos element, negatives = expanded set.neg elements
        assert_eq!(row.text_fields.len(), 5);
        assert_eq!(row.text_fields[0].name, "set.query");
        assert_eq!(row.text_fields[0].text, "What is Rust?");
        assert_eq!(row.text_fields[1].name, "set.pos");
        assert_eq!(row.text_fields[1].text, "How does Rust work?");
        assert_eq!(row.text_fields[2].name, "set.neg[0]");
        assert_eq!(row.text_fields[2].text, "Is Python better?");
        assert_eq!(row.text_fields[3].name, "set.neg[1]");
        assert_eq!(row.text_fields[3].text, "What is Java?");
        assert_eq!(row.text_fields[4].name, "set.neg[2]");
        assert_eq!(row.text_fields[4].text, "Tell me a joke");
    }

    #[test]
    fn parse_row_dict_flat_columns_unaffected() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.anchor_columns = vec!["anchor".into()];
        config.positive_columns = vec!["positive".into()];
        config.id_column = Some("id".into());
        let source = test_source(config);

        // Flat (non-dict) row — should work exactly as before.
        let row = parse_row(
            &source,
            0,
            &json!({"id": "r1", "anchor": "a", "positive": "p"}),
        )
        .unwrap()
        .unwrap();
        assert_eq!(row.text_fields.len(), 2);
        assert_eq!(row.text_fields[0].text, "a");
        assert_eq!(row.text_fields[1].text, "p");
    }

    #[test]
    fn row_to_record_uses_anchor_for_positive_when_single_field() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let row = RowView {
            row_id: Some("r1".into()),
            timestamp: None,
            text_fields: vec![RowTextField {
                name: "text".into(),
                text: "alpha".into(),
            }],
        };

        let record = row_to_record(&source, &row, 0).unwrap().unwrap();
        assert_eq!(record.sections.len(), 2);
        assert_eq!(record.sections[0].text, record.sections[1].text);
    }

    #[test]
    fn row_to_record_falls_back_to_row_index_when_row_id_missing() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let row = RowView {
            row_id: None,
            timestamp: None,
            text_fields: vec![RowTextField {
                name: "text".into(),
                text: "body".into(),
            }],
        };

        let record = row_to_record(&source, &row, 7).unwrap().unwrap();
        assert!(record.id.ends_with("::row_7"));
    }

    #[test]
    fn row_to_record_preserves_explicit_timestamp() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let ts = Utc::now();
        let row = RowView {
            row_id: Some("r1".into()),
            timestamp: Some(ts),
            text_fields: vec![RowTextField {
                name: "text".into(),
                text: "alpha".into(),
            }],
        };

        let record = row_to_record(&source, &row, 0).unwrap().unwrap();
        assert_eq!(record.created_at, ts);
        assert_eq!(record.updated_at, ts);
    }

    #[test]
    fn row_to_record_builds_expected_sections() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let row = RowView {
            row_id: Some("abc".into()),
            timestamp: Some(Utc::now()),
            text_fields: vec![
                RowTextField {
                    name: "title".into(),
                    text: "anchor".into(),
                },
                RowTextField {
                    name: "pos".into(),
                    text: "positive".into(),
                },
                RowTextField {
                    name: "ctx".into(),
                    text: "extra".into(),
                },
            ],
        };

        let record = row_to_record(&source, &row, 1).unwrap().unwrap();
        assert_eq!(record.sections.len(), 3);
        assert_eq!(record.sections[0].role, SectionRole::Anchor);
        assert_eq!(record.sections[1].role, SectionRole::Context);
        assert_eq!(record.id, "hf_test::abc");
    }

    #[test]
    fn row_to_record_returns_none_for_empty_fields() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let row = RowView {
            row_id: Some("x".into()),
            timestamp: None,
            text_fields: Vec::new(),
        };
        assert!(row_to_record(&source, &row, 0).unwrap().is_none());
    }

    #[test]
    fn row_to_record_dict_dataset_sections() {
        let dir = tempdir().unwrap();
        let mut config = test_config(dir.path().to_path_buf());
        config.anchor_columns = vec!["query".into()];
        config.positive_columns = vec!["pos".into()];
        config.negative_columns = vec!["neg".into()];
        let source = test_source(config);

        let row = RowView {
            row_id: Some("dict:0".into()),
            timestamp: None,
            text_fields: vec![
                RowTextField {
                    name: "query".into(),
                    text: "anchor text".into(),
                },
                RowTextField {
                    name: "pos".into(),
                    text: "positive text".into(),
                },
                RowTextField {
                    name: "neg[0]".into(),
                    text: "negative one".into(),
                },
                RowTextField {
                    name: "neg[1]".into(),
                    text: "negative two".into(),
                },
            ],
        };

        let record = row_to_record(&source, &row, 0).unwrap().unwrap();
        // First field → Anchor, second → Context (positive), rest → Context (negatives)
        assert_eq!(record.sections.len(), 4);
        assert_eq!(record.sections[0].role, SectionRole::Anchor);
        assert_eq!(record.sections[0].text, "anchor text");
        assert_eq!(record.sections[1].role, SectionRole::Context);
        assert_eq!(record.sections[1].text, "positive text");
        assert_eq!(record.sections[2].role, SectionRole::Context);
        assert_eq!(record.sections[2].text, "negative one");
        assert_eq!(record.sections[3].role, SectionRole::Context);
        assert_eq!(record.sections[3].text, "negative two");
    }

    #[test]
    fn read_row_batch_errors_when_row_not_mappable_to_shard() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 1;
            state.shards.clear();
        }

        let mut out = Vec::new();
        let err = read_row_batch(&source, &[0], &mut out, Some(1));
        assert!(err.is_err());
    }

    #[test]
    fn read_row_batch_errors_when_parquet_local_offsets_are_missing() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("rows.parquet");
        write_parquet_fixture(&path, &[("id-1", "text-1")]);
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);

        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 3;
            state.shards = vec![ShardIndex {
                path,
                global_start: 0,
                row_count: 3,
                parquet_row_groups: vec![(0, 3)],
                remote_candidate: None,
            }];
        }

        let mut out = Vec::new();
        let err = read_row_batch(&source, &[2], &mut out, Some(1)).unwrap_err();
        assert!(matches!(
            err,
            SamplerError::SourceUnavailable { ref reason, .. } if reason.contains("parquet rows missing")
        ));
    }

    #[test]
    fn read_row_batch_errors_when_parquet_reader_cannot_open_file() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 1;
            state.shards = vec![ShardIndex {
                path: dir.path().join("missing.parquet"),
                global_start: 0,
                row_count: 1,
                parquet_row_groups: vec![(0, 1)],
                remote_candidate: None,
            }];
        }

        let mut out = Vec::new();
        let err = read_row_batch(&source, &[0], &mut out, Some(1));
        assert!(err.is_err());
    }

    #[test]
    fn read_row_batch_skips_unavailable_indices_without_error() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 0;
            state.remote_candidates = Some(Vec::new());
        }

        let mut out = Vec::new();
        read_row_batch(&source, &[0, 1], &mut out, Some(2)).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn read_row_batch_reads_parquet_rows_and_uses_cache_on_repeat() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("rows.parquet");
        write_parquet_fixture(&path, &[("r10", "ten"), ("r11", "eleven")]);

        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());
        let shard = index_single_shard(&config, &path, 0).unwrap().0.unwrap();
        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 2;
            state.shards = vec![shard];
        }

        let mut first = Vec::new();
        read_row_batch(&source, &[0, 1], &mut first, None).unwrap();
        assert_eq!(first.len(), 2);
        assert!(first.iter().any(|record| record.id.ends_with("::r10")));

        let mut second = Vec::new();
        read_row_batch(&source, &[0, 1], &mut second, None).unwrap();
        assert_eq!(second.len(), 2);
    }

    #[test]
    fn read_row_batch_uses_cached_rows_and_respects_limit() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 2;
        }

        let row0 = RowView {
            row_id: Some("r0".into()),
            timestamp: Some(Utc::now()),
            text_fields: vec![RowTextField {
                name: "text".into(),
                text: "alpha".into(),
            }],
        };
        let row1 = RowView {
            row_id: Some("r1".into()),
            timestamp: Some(Utc::now()),
            text_fields: vec![RowTextField {
                name: "text".into(),
                text: "beta".into(),
            }],
        };
        {
            let mut cache = source.cache.lock().unwrap();
            cache.insert(0, row0, config.cache_capacity);
            cache.insert(1, row1, config.cache_capacity);
        }

        let mut out = Vec::new();
        read_row_batch(&source, &[0, 1], &mut out, Some(1)).unwrap();
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn read_row_batch_errors_on_invalid_json_line() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("broken.jsonl");
        fs::write(&path, b"not-json\n").unwrap();

        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());
        let shard = index_single_shard(&config, &path, 0).unwrap().0.unwrap();
        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 1;
            state.shards = vec![shard];
        }

        let mut out = Vec::new();
        let result = read_row_batch(&source, &[0], &mut out, Some(1));
        assert!(result.is_err());
    }

    #[test]
    fn value_to_text_handles_scalar_and_structured_values() {
        assert_eq!(value_to_text(&json!(null)), None);
        assert_eq!(value_to_text(&json!("   ")), None);
        assert_eq!(value_to_text(&json!("hello")), Some("hello".into()));
        assert_eq!(value_to_text(&json!(true)), Some("true".into()));
        assert_eq!(value_to_text(&json!(3.5)), Some("3.5".into()));
        assert_eq!(value_to_text(&json!([1, 2])), Some("1".into()));
    }

    #[test]
    fn value_to_text_object_returns_serialized() {
        let obj = json!({"key": "value"});
        let result = value_to_text(&obj);
        assert!(result.is_some());
        let text = result.unwrap();
        assert!(text.contains("key"));
    }

    #[test]
    fn value_to_text_empty_object_returns_none() {
        let obj = json!({});
        let result = value_to_text(&obj);
        // Empty object serializes to "{}" which is non-empty, so it returns Some
        assert!(result.is_some());
    }

    #[test]
    fn value_to_text_array_first_element() {
        assert_eq!(value_to_text(&json!(["single"])), Some("single".into()));
        assert_eq!(value_to_text(&json!(["a", "b", "c"])), Some("a".into()));
        assert_eq!(value_to_text(&json!([])), None);
        assert_eq!(value_to_text(&json!([null, "valid"])), Some("valid".into()));
    }

    #[test]
    fn resolve_json_path_top_level() {
        let row = json!({"query": "hello", "pos": ["p"], "neg": ["n"]});
        let obj = row.as_object().unwrap();
        assert_eq!(resolve_json_path(obj, "query"), Some(json!("hello")));
    }

    #[test]
    fn resolve_json_path_nested_dict() {
        let row = json!({"set": {"query": "hello", "pos": ["p"], "neg": ["n"]}});
        let obj = row.as_object().unwrap();
        assert_eq!(resolve_json_path(obj, "set.query"), Some(json!("hello")));
        assert_eq!(resolve_json_path(obj, "set.pos"), Some(json!(["p"])));
    }

    #[test]
    fn resolve_json_path_missing_returns_none() {
        let row = json!({"set": {"query": "hello"}});
        let obj = row.as_object().unwrap();
        assert_eq!(resolve_json_path(obj, "missing"), None);
    }

    #[test]
    fn resolve_json_path_non_object_intermediate_returns_none() {
        let row = json!({"set": "not-an-object"});
        let obj = row.as_object().unwrap();
        // "set" exists but is a string, not an object — inner.get should fail.
        assert_eq!(resolve_json_path(obj, "set.query"), None);
    }

    #[test]
    fn write_store_row_count_and_read_store_row_count_roundtrip() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let store_path = dir.path().join("roundtrip.simdr");
        let store = DataStore::open(&store_path).unwrap();
        write_store_row_count(&source, &store, 42).unwrap();
        assert_eq!(read_store_row_count(&source, &store).unwrap(), 42);
        write_store_row_count(&source, &store, 99).unwrap();
        assert_eq!(read_store_row_count(&source, &store).unwrap(), 99);
    }

    #[test]
    fn read_store_row_count_validates_payload_size_and_roundtrips_written_value() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let store_path = dir.path().join("rows.simdr");
        let store = DataStore::open(&store_path).unwrap();

        store
            .write(HF_SHARD_STORE_META_ROWS_KEY, &[1u8, 2, 3])
            .unwrap();
        let err = read_store_row_count(&source, &store).unwrap_err();
        let message = format!("{err}");
        assert!(message.contains("payload size mismatch"));

        write_store_row_count(&source, &store, 7).unwrap();
        assert_eq!(read_store_row_count(&source, &store).unwrap(), 7);
    }

    #[test]
    fn read_store_row_count_returns_zero_when_meta_key_is_absent() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let store_path = dir.path().join("empty.simdr");
        let store = DataStore::open(&store_path).unwrap();

        assert_eq!(read_store_row_count(&source, &store).unwrap(), 0);
    }

    #[test]
    fn read_store_row_count_errors_on_payload_size_mismatch() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let store_path = dir.path().join("bad-meta.simdr");
        let store = DataStore::open(&store_path).unwrap();
        store.write(HF_SHARD_STORE_META_ROWS_KEY, b"abc").unwrap();
        match read_store_row_count(&source, &store) {
            Err(SamplerError::SourceUnavailable { reason, .. }) => {
                assert!(reason.contains("payload size"));
            }
            other => panic!("expected SourceUnavailable error, got {other:?}"),
        }
    }

    #[test]
    fn shard_store_row_count_errors_on_payload_size_mismatch() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let store_path = dir.path().join("test.simdr");
        let store = simd_r_drive::storage_engine::DataStore::open(&store_path).unwrap();
        // Write a payload that's not 8 bytes (wrong size)
        store.write(HF_SHARD_STORE_META_ROWS_KEY, b"short").unwrap();
        let err = read_store_row_count(&source, &store).unwrap_err();
        assert!(matches!(err, SamplerError::SourceUnavailable { .. }));
    }

    #[test]
    fn read_store_row_count_returns_zero_when_no_meta_key() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        let store_path = dir.path().join("test.simdr");
        let store = simd_r_drive::storage_engine::DataStore::open(&store_path).unwrap();
        // No meta key written — should return 0
        assert_eq!(read_store_row_count(&source, &store).unwrap(), 0);
    }

    // Temp-staged files (with triplet_hf_ prefix) must be removed after
    // transcoding completes — no leaked temp files in the OS temp directory.
    #[test]
    fn transcode_transient_shard_to_store_cleans_up_temp_downloads() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        // Create a temp file with the triplet_hf_ prefix (simulates a staged download).
        let temp_parquet = dir.path().join("triplets_hf_aabbccdd00112233_temp.parquet");
        write_parquet_fixture(&temp_parquet, &[("r1", "hello")]);
        assert!(temp_parquet.exists(), "temp parquet must exist before test");

        let shard = ShardIndex {
            path: temp_parquet.clone(),
            global_start: 0,
            row_count: 1,
            parquet_row_groups: vec![(0, 1)],
            remote_candidate: None,
        };

        let result =
            transcode_transient_shard_to_store(&source, &shard).expect("transcode must succeed");

        // Temp file must be gone after transcoding.
        assert!(
            !temp_parquet.exists(),
            "temp download file must be cleaned up after transcode"
        );

        // Store must still be present.
        let store_path = result.expect("must yield Some(ShardIndex)").path;
        assert!(
            store_path.exists(),
            "simdr store must exist after transcode"
        );
    }

    // When transcode_transient_shard_to_store takes the early-return path
    // (simdr store already fully populated), it must still delete the input
    // transient file and return a ShardIndex
    #[test]
    fn transcode_transient_shard_to_store_early_return_cleans_up_transient() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        // Write a parquet fixture (simulates a stale/unconsumed parquet from a
        // previous run that crashed before the delete step fired).
        let parquet_path = dir.path().join("stale.parquet");
        write_parquet_fixture(&parquet_path, &[("r1", "hello"), ("r2", "world")]);
        assert!(
            parquet_path.exists(),
            "parquet fixture must exist before test"
        );

        // Pre-populate the corresponding simdr store so the function short-circuits.
        let store_path = shard_store_path_for(&parquet_path);
        write_simdr_fixture(&store_path, &[("r1", "hello"), ("r2", "world")]);
        assert!(store_path.exists(), "simdr store must exist before test");

        let shard = ShardIndex {
            path: parquet_path.clone(),
            global_start: 0,
            row_count: 2,
            parquet_row_groups: vec![(0, 2)],
            remote_candidate: None,
        };

        let result =
            transcode_transient_shard_to_store(&source, &shard).expect("transcode must succeed");

        // Stale parquet must be gone.
        assert!(
            !parquet_path.exists(),
            "stale parquet must be removed on early return"
        );

        // Simdr store must still be present.
        assert!(store_path.exists(), "simdr store must survive early return");

        // Returned shard must point to the store.
        let returned = result.expect("early return must yield Some(ShardIndex)");
        assert_eq!(returned.path, store_path);
        assert_eq!(returned.row_count, 2);
    }

    #[test]
    fn transcode_ndjson_multiple_rows_batch_flush() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        // Create an .ndjson file with 2000+ rows to cross the 1024 batch flush boundary.
        let ndjson_path = dir.path().join("big.ndjson");
        let mut content = String::new();
        for i in 0..2000 {
            content.push_str(&format!(r#"{{"id":"r{i}","text":"txt_{i}"}}"#));
            content.push('\n');
        }
        std::fs::write(&ndjson_path, content).unwrap();

        let shard = ShardIndex {
            path: ndjson_path.clone(),
            global_start: 0,
            row_count: 2000,
            parquet_row_groups: vec![(0, 2000)],
            remote_candidate: None,
        };

        let result =
            transcode_transient_shard_to_store(&source, &shard).expect("transcode must succeed");

        let store_shard = result.expect("must yield Some(ShardIndex)");
        assert_eq!(store_shard.row_count, 2000);
        assert!(shard_store_path_for(&ndjson_path).exists());
    }

    #[test]
    fn transcode_jsonl_gzip_decompression() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        let gz_path = dir.path().join("shard.jsonl.gz");
        let payload = br#"{"id":"r1","text":"hello"}"#;
        write_gzip_fixture(&gz_path, payload);

        let shard = ShardIndex {
            path: gz_path.clone(),
            global_start: 0,
            row_count: 1,
            parquet_row_groups: vec![(0, 1)],
            remote_candidate: None,
        };

        let result =
            transcode_transient_shard_to_store(&source, &shard).expect("transcode must succeed");

        let store_shard = result.expect("must yield Some(ShardIndex)");
        assert_eq!(store_shard.row_count, 1);
    }

    #[test]
    fn transcode_empty_shard_returns_none() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        // Empty .ndjson file (no rows).
        let ndjson_path = dir.path().join("empty.ndjson");
        std::fs::write(&ndjson_path, "").unwrap();

        let shard = ShardIndex {
            path: ndjson_path.clone(),
            global_start: 0,
            row_count: 0,
            parquet_row_groups: Vec::new(),
            remote_candidate: None,
        };

        let result =
            transcode_transient_shard_to_store(&source, &shard).expect("transcode must succeed");

        assert!(result.is_none(), "empty shard should return None");
    }

    #[test]
    fn transcode_manifest_dir_file_cleans_up() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config.clone());

        // Create a .ndjson file inside the _parquet_manifest directory.
        let manifest_root = source.manifest_cache_root();
        fs::create_dir_all(&manifest_root).unwrap();
        let transient_path = manifest_root.join("remote_shard.ndjson");
        fs::write(&transient_path, r#"{"id":"r1","text":"data"}"#).unwrap();
        assert!(transient_path.exists());

        let shard = ShardIndex {
            path: transient_path.clone(),
            global_start: 0,
            row_count: 1,
            parquet_row_groups: vec![(0, 1)],
            remote_candidate: None,
        };

        let _result =
            transcode_transient_shard_to_store(&source, &shard).expect("transcode must succeed");

        // The transient file in manifest dir should be cleaned up after transcode.
        assert!(
            !transient_path.exists(),
            "transient file in manifest dir must be cleaned up"
        );
    }
}
