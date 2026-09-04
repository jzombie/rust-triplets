use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::{BufRead, BufReader};

use chrono::{DateTime, Utc};
use flate2::read::GzDecoder;
use rayon::prelude::*;
use serde::de::{Deserializer as _, SeqAccess, Visitor};
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
                let (shard, local_idx) = crate::shard_indexer::locate_shard(&state.shards, *idx)
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
                crate::shard_indexer::locate_parquet_group(source, &shard, local_idx)?;
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
                let store = crate::shard_indexer::get_or_open_shard_store(source, &shard.path)?;
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

/// Flush a batch of key-value pairs to the row store.
fn flush_batch(
    source: &HuggingFaceRowSource,
    store: &DataStore,
    batch: &[(Vec<u8>, Vec<u8>)],
) -> Result<(), SamplerError> {
    if batch.is_empty() {
        return Ok(());
    }
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
    Ok(())
}

/// Check if a file path is a `.json` file (not `.jsonl` or `.ndjson`).
fn is_json_file(path: &std::path::Path) -> bool {
    resolve_inner_extension(path).is_some_and(|ext| ext == "json")
}

/// Peek at the first non-whitespace byte of a buffered reader without consuming it.
fn peek_first_non_whitespace(reader: &mut (impl BufRead + ?Sized)) -> Option<u8> {
    loop {
        let buf = reader.fill_buf().ok()?;
        let &byte = buf.first()?;
        if byte.is_ascii_whitespace() {
            reader.consume(1);
            continue;
        }
        return Some(byte);
    }
}

/// Visitor that streams JSON array elements one at a time for O(1) memory usage.
struct StreamingArrayVisitor<'a> {
    source: &'a HuggingFaceRowSource,
    shard: &'a ShardIndex,
    served_rows: &'a mut usize,
    batch: &'a mut Vec<(Vec<u8>, Vec<u8>)>,
    store: &'a DataStore,
}

impl<'de> Visitor<'de> for StreamingArrayVisitor<'_> {
    type Value = ();
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a JSON array")
    }
    fn visit_seq<A>(self, mut seq: A) -> Result<(), A::Error>
    where
        A: SeqAccess<'de>,
    {
        while let Some(item) = seq
            .next_element::<Value>()
            .map_err(serde::de::Error::custom)?
        {
            let absolute_idx = self.shard.global_start.saturating_add(*self.served_rows);
            // Normalize: object → pass through; scalar → wrap as {"text": "..."}
            let line_value = if item.is_object() {
                item
            } else if let Some(text) = value_to_text(&item) {
                json!({ "text": text })
            } else {
                continue; // skip null/empty values
            };
            // Parse row — explicit error mapping
            let Some(row) = parse_row(self.source, absolute_idx, &line_value)
                .map_err(serde::de::Error::custom)?
            else {
                continue;
            };
            let key = row_store_row_key(*self.served_rows);
            let payload =
                encode_row_view(self.source, &row).map_err(serde::de::Error::custom)?;
            self.batch.push((key, payload));
            *self.served_rows = self.served_rows.saturating_add(1);
            if self.batch.len() >= 1024 {
                flush_batch(self.source, self.store, self.batch)
                    .map_err(serde::de::Error::custom)?;
                self.batch.clear();
            }
        }
        Ok(())
    }
}

/// Transcode a JSON array shard via streaming deserialization for O(1) memory.
fn transcode_json_array_streaming(
    source: &HuggingFaceRowSource,
    shard: &ShardIndex,
    reader: &mut dyn BufRead,
    store: &DataStore,
) -> Result<usize, SamplerError> {
    let mut batch: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(1024);
    let mut served_rows = 0usize;
    {
        let mut deserializer = serde_json::Deserializer::from_reader(reader);
        let visitor = StreamingArrayVisitor {
            source,
            shard,
            served_rows: &mut served_rows,
            batch: &mut batch,
            store,
        };
        deserializer
            .deserialize_seq(visitor)
            .map_err(|e| SamplerError::SourceInconsistent {
                source_id: source.config.source_id.clone(),
                details: format!(
                    "failed streaming JSON array from shard {}: {e}",
                    shard.path.display()
                ),
            })?;
    }
    // CRITICAL: flush remaining rows after stream ends
    flush_batch(source, store, &batch)?;
    Ok(served_rows)
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

    let store = crate::shard_indexer::get_or_open_shard_store(source, &store_path)?;
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
        // Transcode .jsonl.gz / .jsonl / .ndjson / .json → .simdr store
        let file =
            std::fs::File::open(&shard.path).map_err(|err| SamplerError::SourceUnavailable {
                source_id: source.config.source_id.clone(),
                reason: format!(
                    "failed opening text shard for transcode {}: {err}",
                    shard.path.display()
                ),
            })?;

        // BufReader::new(GzDecoder::new(BufReader::new(file)))
        // Outer BufReader buffers decompressed text for line-by-line reading.
        // Inner BufReader buffers raw disk reads before decompression.
        let mut reader: Box<dyn BufRead> = if is_gzip_path(&shard.path) {
            Box::new(BufReader::new(GzDecoder::new(BufReader::new(file))))
        } else {
            Box::new(BufReader::new(file))
        };

        // Non-destructive peek: check first non-whitespace byte without consuming.
        let is_json_array = is_json_file(&shard.path)
            && peek_first_non_whitespace(&mut reader).is_some_and(|b| b == b'[');

        if is_json_array {
            // Streaming path — pass &mut *reader to deref Box and avoid ownership issues
            served_rows = transcode_json_array_streaming(source, shard, &mut *reader, &store)?;
        } else {
            // Existing line-by-line path for JSONL/NDJSON/text/gzip.
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
                    flush_batch(source, &store, &batch)?;
                    batch.clear();
                }
            }

            // Flush remaining batch
            flush_batch(source, &store, &batch)?;
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
                flush_batch(source, &store, &batch)?;
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
