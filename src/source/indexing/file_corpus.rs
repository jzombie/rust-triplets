use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::constants::file_corpus::{
    FILE_INDEX_META_KEY, FILE_INDEX_PATH_KEY_PREFIX, FILE_INDEX_READ_BATCH, FILE_INDEX_STORE_DIR,
    SKIP_UNREADABLE_MSG,
};
use crate::errors::SamplerError;
use crate::hash::{stable_hash_path, stable_hash_with};
use crate::source::{SourceCursor, SourceSnapshot};
use crate::transport::fs::{FileStream, is_text_file};
use crate::types::{GroupKey, PathString, SourceId};
use crate::utils::normalize_inline_whitespace;
use simd_r_drive::storage_engine::DataStore;
use simd_r_drive::storage_engine::traits::{DataStoreReader, DataStoreWriter};
use std::fs;
use tempfile::TempDir;
use tracing::{debug, warn};
use walkdir::WalkDir;

#[derive(bitcode::Encode, bitcode::Decode)]
/// Persisted metadata for the temporary file index store.
struct FileIndexMeta {
    root: String,
    follow_links: bool,
    text_files_only: bool,
    count: u64,
}

/// Helper for filesystem-backed sources and persisted file indexing.
///
/// It builds a deterministic path list, persists it per process in a temp-store,
/// and pages through the list with `IndexablePager` using batched index reads.
pub struct FileCorpusIndex {
    root: PathBuf,
    source_id: SourceId,
    sampler_seed: Option<u64>,
    follow_links: bool,
    text_files_only: bool,
    group_by_directory: bool,
    group_window_divisor: usize,
}

impl FileCorpusIndex {
    /// Create an index helper for a filesystem root and logical source id.
    pub fn new(root: impl Into<PathBuf>, source_id: impl Into<SourceId>) -> Self {
        Self {
            root: root.into(),
            source_id: source_id.into(),
            sampler_seed: None,
            follow_links: true,
            text_files_only: true,
            group_by_directory: false,
            group_window_divisor: 8,
        }
    }

    /// Control whether symlinks are followed while walking the root.
    pub fn with_follow_links(mut self, follow_links: bool) -> Self {
        self.follow_links = follow_links;
        self
    }

    /// Filter to text files only (based on extension) during indexing.
    pub fn with_text_files_only(mut self, text_files_only: bool) -> Self {
        self.text_files_only = text_files_only;
        self
    }

    /// Enable deterministic directory grouping for refresh paging.
    pub fn with_directory_grouping(mut self, group_by_directory: bool) -> Self {
        self.group_by_directory = group_by_directory;
        self
    }

    /// Control the per-window grouping size divisor used for directory grouping.
    pub fn with_directory_grouping_window_divisor(mut self, divisor: usize) -> Self {
        self.group_window_divisor = divisor.max(1);
        self
    }

    /// Sampler seed used to derive deterministic paging order.
    pub fn with_sampler_seed(mut self, sampler_seed: u64) -> Self {
        self.sampler_seed = Some(sampler_seed);
        self
    }

    fn required_sampler_seed(&self) -> Result<u64, SamplerError> {
        self.sampler_seed
            .ok_or_else(|| SamplerError::SourceInconsistent {
                source_id: self.source_id.clone(),
                details: "file corpus sampler seed not provided".to_string(),
            })
    }

    /// Refresh a filesystem source using streaming walk order.
    ///
    /// This avoids building a persistent index and is suitable for purely streaming
    /// sources that do not require deterministic shuffling.
    pub fn refresh_streaming<F>(
        &self,
        cursor: Option<&SourceCursor>,
        limit: Option<usize>,
        mut build_record: F,
    ) -> Result<SourceSnapshot, SamplerError>
    where
        F: FnMut(&Path) -> Result<Option<crate::data::DataRecord>, SamplerError>,
    {
        FileStream::new(&self.root)
            .with_follow_symlinks(self.follow_links)
            .stream_incremental(cursor, limit, |path| {
                if self.text_files_only && !is_text_file(path) {
                    return Ok(None);
                }
                match build_record(path) {
                    Ok(record) => Ok(record),
                    Err(err) if Self::should_skip_record_error(&err) => {
                        warn!(
                            source_id = %self.source_id,
                            path = %path.display(),
                            error = %err,
                            SKIP_UNREADABLE_MSG
                        );
                        Ok(None)
                    }
                    Err(err) => Err(err),
                }
            })
    }

    /// Return the indexed file count for this source.
    ///
    /// This does not call source-level `refresh` and is suitable for
    /// metadata-only estimators.
    pub fn indexed_record_count(&self) -> Result<u64, SamplerError> {
        let (_, meta) = self.load_or_build_index_meta()?;
        Ok(meta.count)
    }

    /// Refresh a filesystem source using indexable paging over a deterministic path list.
    /// Batch reads preserve the `IndexPermutation` order, so chunked IO does not change shuffling.
    pub fn refresh_indexable<F>(
        &self,
        cursor: Option<&SourceCursor>,
        limit: Option<usize>,
        mut build_record: F,
    ) -> Result<SourceSnapshot, SamplerError>
    where
        F: FnMut(&Path) -> Result<Option<crate::data::DataRecord>, SamplerError>,
    {
        let (store, meta) = self.load_or_build_index_meta()?;
        let total = meta.count as usize;
        if total == 0 {
            return Ok(SourceSnapshot {
                records: Vec::new(),
                cursor: SourceCursor {
                    last_seen: chrono::Utc::now(),
                    revision: 0,
                },
            });
        }
        if self.group_by_directory {
            return self.refresh_grouped_indexable(cursor, limit, store, build_record);
        }
        let mut start = cursor.map(|cursor| cursor.revision as usize).unwrap_or(0);
        if start >= total {
            start = 0;
        }
        let max = limit.unwrap_or(total);
        let seed = stable_group_seed(&self.source_id, total, self.required_sampler_seed()?);
        let mut permutation = crate::source::IndexPermutation::new(total, seed, start as u64);
        let mut records = Vec::new();
        let mut pending_indices = Vec::with_capacity(FILE_INDEX_READ_BATCH);
        for _ in 0..total {
            if records.len() >= max {
                break;
            }
            pending_indices.push(permutation.next());
            if pending_indices.len() == FILE_INDEX_READ_BATCH {
                self.read_index_batch(
                    &store,
                    &mut build_record,
                    &pending_indices,
                    &mut records,
                    Some(max),
                )?;
                pending_indices.clear();
            }
        }
        if !pending_indices.is_empty() && records.len() < max {
            self.read_index_batch(
                &store,
                &mut build_record,
                &pending_indices,
                &mut records,
                Some(max),
            )?;
        }
        let last_seen = records
            .iter()
            .map(|record| record.updated_at)
            .max()
            .unwrap_or_else(chrono::Utc::now);
        let next_start = permutation.cursor();
        Ok(SourceSnapshot {
            records,
            cursor: SourceCursor {
                last_seen,
                revision: next_start as u64,
            },
        })
    }

    /// Build a normalized title from a file stem (optionally replacing underscores).
    pub fn normalized_title_from_stem(
        path: &Path,
        source_id: &SourceId,
        replace_underscores: bool,
    ) -> Result<String, SamplerError> {
        let mut stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| SamplerError::SourceInconsistent {
                source_id: source_id.to_string(),
                details: format!("path '{}' has no valid file stem", path.display()),
            })?
            .to_string();
        if replace_underscores {
            stem = stem.replace('_', " ");
        }
        let normalized = normalize_inline_whitespace(stem);
        if normalized.is_empty() {
            return Err(SamplerError::SourceInconsistent {
                source_id: source_id.to_string(),
                details: format!("path '{}' has an empty normalized title", path.display()),
            });
        }
        Ok(normalized)
    }

    /// Produce a stable record id as `<source_id>::<rel_path>`.
    pub fn source_scoped_record_id(source_id: &SourceId, root: &Path, path: &Path) -> String {
        let rel: PathBuf = path
            .strip_prefix(root)
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|_| path.to_path_buf());
        format!("{}::{}", source_id, rel.to_string_lossy())
    }

    /// Compute the on-disk path used for the per-process file index store.
    pub fn file_index_store_path(root: &Path, source_id: &SourceId) -> PathBuf {
        let root_hash = stable_hash_path(0, root);
        file_index_root_dir().join(format!("{source_id}-{root_hash:x}.bin"))
    }

    /// Open the index store and return the meta if it matches this source.
    /// Otherwise, rebuild the index entries and return the new meta.
    fn load_or_build_index_meta(&self) -> Result<(DataStore, FileIndexMeta), SamplerError> {
        let store = self.open_index_store()?;
        if let Some(meta) = self.read_index_meta(&store)? {
            if let Ok(metadata) =
                fs::metadata(Self::file_index_store_path(&self.root, &self.source_id))
            {
                debug!(
                    source_id = %self.source_id,
                    bytes = metadata.len(),
                    count = meta.count,
                    "reusing file index"
                );
            }
            return Ok((store, meta));
        }
        let index_path = Self::file_index_store_path(&self.root, &self.source_id);
        let _ = fs::remove_file(&index_path);
        let store = self.open_index_store()?;
        let mut candidates: Vec<PathBuf> = Vec::new();
        let mut walker = WalkDir::new(&self.root);
        if self.follow_links {
            walker = walker.follow_links(true);
        }
        for entry in walker
            .into_iter()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_file())
        {
            let path = entry.path().to_path_buf();
            if self.text_files_only && !is_text_file(&path) {
                continue;
            }
            candidates.push(path);
        }
        candidates.sort();
        let paths = candidates;
        let meta = FileIndexMeta {
            root: self.root.to_string_lossy().to_string(),
            follow_links: self.follow_links,
            text_files_only: self.text_files_only,
            count: paths.len() as u64,
        };
        self.write_index_entries(&store, &meta, &paths)?;
        Ok((store, meta))
    }

    /// Open the simd-r-drive store for this source.
    fn open_index_store(&self) -> Result<DataStore, SamplerError> {
        let path = Self::file_index_store_path(&self.root, &self.source_id);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        DataStore::open(path.as_path()).map_err(|err| SamplerError::SourceUnavailable {
            source_id: self.source_id.clone(),
            reason: format!("file index store open failed: {err}"),
        })
    }

    /// Read and validate the index meta entry.
    fn read_index_meta(&self, store: &DataStore) -> Result<Option<FileIndexMeta>, SamplerError> {
        let meta = match store.read(FILE_INDEX_META_KEY) {
            Ok(Some(entry)) => match self.decode_index_meta(entry.as_ref())? {
                Some(meta) => meta,
                None => return Ok(None),
            },
            Ok(None) => return Ok(None),
            Err(err) => {
                return Err(SamplerError::SourceUnavailable {
                    source_id: self.source_id.clone(),
                    reason: format!("file index store read failed: {err}"),
                });
            }
        };
        if meta.follow_links != self.follow_links || meta.text_files_only != self.text_files_only {
            return Ok(None);
        }
        if meta.root != self.root.to_string_lossy().as_ref() {
            return Ok(None);
        }
        Ok(Some(meta))
    }

    /// Persist the meta entry plus one path entry per index using batch writes.
    fn write_index_entries(
        &self,
        store: &DataStore,
        meta: &FileIndexMeta,
        paths: &[PathBuf],
    ) -> Result<(), SamplerError> {
        let meta_bytes = bitcode::encode(meta);
        let mut entries: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(paths.len() + 1);
        entries.push((FILE_INDEX_META_KEY.to_vec(), meta_bytes.clone()));
        let mut raw_path_bytes: u64 = 0;
        for (idx, path) in paths.iter().enumerate() {
            let value = path.to_string_lossy().as_bytes().to_vec();
            raw_path_bytes += value.len() as u64;
            entries.push((Self::index_key(idx), value));
        }
        let entry_refs: Vec<(&[u8], &[u8])> = entries
            .iter()
            .map(|(key, value)| (key.as_slice(), value.as_slice()))
            .collect();
        store
            .batch_write(&entry_refs)
            .map_err(|err| SamplerError::SourceUnavailable {
                source_id: self.source_id.clone(),
                reason: format!("file index store write failed: {err}"),
            })?;
        let encoded_bytes: u64 = raw_path_bytes + meta_bytes.len() as u64;
        let path_count = paths.len() as u64;
        let bytes_per_path = if path_count == 0 {
            0
        } else {
            raw_path_bytes / path_count
        };
        if let Ok(metadata) = fs::metadata(Self::file_index_store_path(&self.root, &self.source_id))
        {
            let overhead_bytes = metadata.len().saturating_sub(encoded_bytes);
            debug!(
                source_id = %self.source_id,
                raw_path_bytes,
                encoded_bytes,
                file_bytes = metadata.len(),
                path_count,
                avg_path_bytes = bytes_per_path,
                overhead_bytes,
                "wrote file index"
            );
        }
        Ok(())
    }

    /// Decode the meta entry from the index store.
    fn decode_index_meta(&self, bytes: &[u8]) -> Result<Option<FileIndexMeta>, SamplerError> {
        let meta: FileIndexMeta =
            bitcode::decode(bytes).map_err(|err| SamplerError::SourceInconsistent {
                source_id: self.source_id.clone(),
                details: format!("file index meta decode failed: {err}"),
            })?;
        Ok(Some(meta))
    }

    /// Generate a stable key for a path entry at `idx`.
    fn index_key(idx: usize) -> Vec<u8> {
        let mut key = Vec::with_capacity(FILE_INDEX_PATH_KEY_PREFIX.len() + 8);
        key.extend_from_slice(FILE_INDEX_PATH_KEY_PREFIX);
        key.extend_from_slice(&(idx as u64).to_le_bytes());
        key
    }

    /// Read a batch of path entries and build records from them.
    fn read_index_batch<F>(
        &self,
        store: &DataStore,
        build_record: &mut F,
        indices: &[usize],
        records: &mut Vec<crate::data::DataRecord>,
        max: Option<usize>,
    ) -> Result<(), SamplerError>
    where
        F: FnMut(&Path) -> Result<Option<crate::data::DataRecord>, SamplerError>,
    {
        let keys: Vec<Vec<u8>> = indices.iter().map(|idx| Self::index_key(*idx)).collect();
        let key_refs: Vec<&[u8]> = keys.iter().map(|key| key.as_slice()).collect();
        let entries =
            store
                .batch_read(&key_refs)
                .map_err(|err| SamplerError::SourceUnavailable {
                    source_id: self.source_id.clone(),
                    reason: format!("file index store read failed: {err}"),
                })?;
        for (idx, entry) in indices.iter().zip(entries.into_iter()) {
            if let Some(max) = max
                && records.len() >= max
            {
                break;
            }
            let entry = entry.ok_or_else(|| SamplerError::SourceInconsistent {
                source_id: self.source_id.clone(),
                details: format!("file index missing entry {idx}"),
            })?;
            let path_str = String::from_utf8_lossy(entry.as_ref());
            let path = PathBuf::from(path_str.as_ref());
            match build_record(&path) {
                Ok(Some(record)) => records.push(record),
                Ok(None) => {}
                Err(err) if Self::should_skip_record_error(&err) => {
                    warn!(
                        source_id = %self.source_id,
                        path = %path.display(),
                        error = %err,
                        SKIP_UNREADABLE_MSG
                    );
                }
                Err(err) => return Err(err),
            }
        }
        Ok(())
    }

    fn refresh_grouped_indexable<F>(
        &self,
        cursor: Option<&SourceCursor>,
        limit: Option<usize>,
        store: DataStore,
        mut build_record: F,
    ) -> Result<SourceSnapshot, SamplerError>
    where
        F: FnMut(&Path) -> Result<Option<crate::data::DataRecord>, SamplerError>,
    {
        let total = self
            .read_index_meta(&store)?
            .map(|meta| meta.count as usize)
            .unwrap_or(0);
        if total == 0 {
            return Ok(SourceSnapshot {
                records: Vec::new(),
                cursor: SourceCursor {
                    last_seen: chrono::Utc::now(),
                    revision: 0,
                },
            });
        }
        let grouped_order = self.build_grouped_order(&store, total)?;
        if grouped_order.is_empty() {
            return Ok(SourceSnapshot {
                records: Vec::new(),
                cursor: SourceCursor {
                    last_seen: chrono::Utc::now(),
                    revision: 0,
                },
            });
        }
        let mut start = cursor.map(|cursor| cursor.revision as usize).unwrap_or(0);
        if start >= grouped_order.len() {
            start = 0;
        }
        let max = limit.unwrap_or(grouped_order.len());
        // Cap per-group contribution for each grouped refresh so extremely large
        // directories (for example, a single taxonomy bucket) cannot dominate a
        // limited batch. The cap is deterministic and derived from the current
        // request size: ceil(max / number_of_groups).
        let unique_groups = grouped_order
            .iter()
            .map(|path| self.group_key_for_path(path))
            .collect::<HashSet<GroupKey>>();
        let per_group_cap =
            crate::source::indexing::grouping::per_group_refresh_cap(max, unique_groups.len());
        let mut selected_per_group: HashMap<GroupKey, usize> = HashMap::new();
        let mut records = Vec::new();
        let mut steps = 0usize;
        for _ in 0..grouped_order.len() {
            if records.len() >= max {
                break;
            }
            let idx = (start + steps) % grouped_order.len();
            steps += 1;
            let path = &grouped_order[idx];
            let group_key = self.group_key_for_path(path);
            let current = selected_per_group.get(&group_key).copied().unwrap_or(0);
            if current >= per_group_cap {
                continue;
            }
            match build_record(path) {
                Ok(Some(record)) => {
                    records.push(record);
                    selected_per_group.insert(group_key, current + 1);
                }
                Ok(None) => {}
                Err(err) if Self::should_skip_record_error(&err) => {
                    warn!(
                        source_id = %self.source_id,
                        path = %path.display(),
                        error = %err,
                        SKIP_UNREADABLE_MSG
                    );
                }
                Err(err) => return Err(err),
            }
        }
        let last_seen = records
            .iter()
            .map(|record| record.updated_at)
            .max()
            .unwrap_or_else(chrono::Utc::now);
        let next_start = (start + steps) % grouped_order.len();
        Ok(SourceSnapshot {
            records,
            cursor: SourceCursor {
                last_seen,
                revision: next_start as u64,
            },
        })
    }

    fn build_grouped_order(
        &self,
        store: &DataStore,
        total: usize,
    ) -> Result<Vec<PathBuf>, SamplerError> {
        // Build a deterministic order that keeps same-group files closer together:
        // 1) deterministically permute all paths with a stable seed,
        // 2) process the permutation in fixed-size windows,
        // 3) within each window, group by key and interleave groups,
        // 4) deterministically shuffle the interleaved pool, then append to output.
        let paths = self.read_index_paths(store, total)?;
        if paths.is_empty() {
            return Ok(Vec::new());
        }
        let seed = stable_group_seed(&self.source_id, total, self.required_sampler_seed()?);
        let mut permutation = crate::source::IndexPermutation::new(total, seed, 0);
        let mut permuted_paths = Vec::with_capacity(total);
        for _ in 0..total {
            let idx = permutation.next();
            permuted_paths.push(paths[idx].clone());
        }

        Ok(
            crate::source::indexing::grouping::deterministic_grouped_order(
                &permuted_paths,
                seed,
                self.group_window_divisor,
                |path| path_relative_key(&self.root, path),
                |path| self.group_key_for_path(path),
            ),
        )
    }

    fn read_index_paths(
        &self,
        store: &DataStore,
        total: usize,
    ) -> Result<Vec<PathBuf>, SamplerError> {
        let mut paths = Vec::with_capacity(total);
        let mut batch = Vec::with_capacity(FILE_INDEX_READ_BATCH);
        for idx in 0..total {
            batch.push(idx);
            if batch.len() == FILE_INDEX_READ_BATCH {
                self.read_index_paths_batch(store, &batch, &mut paths)?;
                batch.clear();
            }
        }
        if !batch.is_empty() {
            self.read_index_paths_batch(store, &batch, &mut paths)?;
        }
        Ok(paths)
    }

    fn read_index_paths_batch(
        &self,
        store: &DataStore,
        indices: &[usize],
        paths: &mut Vec<PathBuf>,
    ) -> Result<(), SamplerError> {
        let keys: Vec<Vec<u8>> = indices.iter().map(|idx| Self::index_key(*idx)).collect();
        let key_refs: Vec<&[u8]> = keys.iter().map(|key| key.as_slice()).collect();
        let entries =
            store
                .batch_read(&key_refs)
                .map_err(|err| SamplerError::SourceUnavailable {
                    source_id: self.source_id.clone(),
                    reason: format!("file index store read failed: {err}"),
                })?;
        for (idx, entry) in indices.iter().zip(entries.into_iter()) {
            let entry = entry.ok_or_else(|| SamplerError::SourceInconsistent {
                source_id: self.source_id.clone(),
                details: format!("file index missing entry {idx}"),
            })?;
            let path_str = String::from_utf8_lossy(entry.as_ref());
            paths.push(PathBuf::from(path_str.as_ref()));
        }
        Ok(())
    }

    fn group_key_for_path(&self, path: &Path) -> GroupKey {
        let rel = path.strip_prefix(&self.root).unwrap_or(path);
        let mut components: Vec<GroupKey> = rel
            .components()
            .filter_map(|component| component.as_os_str().to_str().map(|s| s.to_string()))
            .collect();
        if components.is_empty() {
            return GroupKey::new();
        }
        components.pop();
        if components.is_empty() {
            return GroupKey::new();
        }
        components.join("/")
    }

    fn should_skip_record_error(err: &SamplerError) -> bool {
        matches!(
            err,
            SamplerError::Io(_) | SamplerError::SourceInconsistent { .. }
        )
    }
}
/// Resolve the per-process temp directory used for file index stores.
fn file_index_root_dir() -> PathBuf {
    static FILE_INDEX_ROOT: OnceLock<TempDir> = OnceLock::new();
    FILE_INDEX_ROOT
        .get_or_init(|| TempDir::new().expect("failed to create temp file index dir"))
        .path()
        .join(FILE_INDEX_STORE_DIR)
}

fn stable_group_seed(source_id: &SourceId, total: usize, sampler_seed: u64) -> u64 {
    let base = stable_hash_with(|hasher| {
        source_id.hash(hasher);
        total.hash(hasher);
    });
    base ^ stable_hash_with(|hasher| {
        "triplets_sampler_seed".hash(hasher);
        source_id.hash(hasher);
        total.hash(hasher);
        sampler_seed.hash(hasher);
    })
}

fn path_relative_key(root: &Path, path: &Path) -> PathString {
    let rel = path.strip_prefix(root).unwrap_or(path);
    rel.components()
        .filter_map(|component| component.as_os_str().to_str())
        .collect::<Vec<_>>()
        .join("/")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{DataRecord, QualityScore, RecordSection, SectionRole};
    use crate::errors::SamplerError;
    use crate::source::IndexablePager;
    use crate::types::{GroupKey, PathString, RecordId};
    use chrono::Utc;
    use std::collections::HashMap;
    use std::fs;
    use std::path::{Path, PathBuf};
    use tempfile::tempdir;

    fn build_stub_record(
        path: &Path,
        source_id: &SourceId,
        root: &Path,
        bad_path: &Path,
    ) -> Result<Option<DataRecord>, SamplerError> {
        if path == bad_path {
            return Err(SamplerError::SourceInconsistent {
                source_id: source_id.clone(),
                details: "empty normalized body".into(),
            });
        }
        let now = Utc::now();
        Ok(Some(DataRecord {
            id: FileCorpusIndex::source_scoped_record_id(source_id, root, path),
            source: source_id.clone(),
            created_at: now,
            updated_at: now,
            quality: QualityScore { trust: 1.0 },
            taxonomy: Vec::new(),
            sections: vec![RecordSection {
                role: SectionRole::Anchor,
                heading: None,
                text: "stub".into(),
                sentences: vec!["stub".into()],
            }],
            meta_prefix: None,
        }))
    }

    #[test]
    fn filesystem_indexable_spans_multiple_regimes() {
        // Ensure the filesystem refresh path uses indexable paging rather than
        // a sequential walk by checking spread across the index space.
        let total = 256usize;
        let mask = (1u64 << (64 - (total as u64 - 1).leading_zeros())) - 1;
        let source_id = (0..512)
            .map(|idx| format!("fs_regime_{idx}"))
            .find(|id| {
                let seed = IndexablePager::seed_for(id, total);
                let a = (seed | 1) & mask;
                a != 1 && a != mask
            })
            .unwrap();

        // Create a deterministic corpus of files to index.
        let temp = tempdir().unwrap();
        for idx in 0..total {
            let path = temp.path().join(format!("file_{idx:03}.txt"));
            fs::write(path, "stub").unwrap();
        }

        let index = FileCorpusIndex::new(temp.path(), &source_id)
            .with_sampler_seed(1)
            .with_follow_links(false)
            .with_text_files_only(true);
        let snapshot = index
            .refresh_indexable(None, Some(64), |path| {
                let now = Utc::now();
                Ok(Some(DataRecord {
                    id: FileCorpusIndex::source_scoped_record_id(&source_id, temp.path(), path),
                    source: source_id.clone(),
                    created_at: now,
                    updated_at: now,
                    quality: QualityScore { trust: 1.0 },
                    taxonomy: Vec::new(),
                    sections: vec![RecordSection {
                        role: SectionRole::Anchor,
                        heading: None,
                        text: "stub".into(),
                        sentences: vec!["stub".into()],
                    }],
                    meta_prefix: None,
                }))
            })
            .unwrap();

        // Map snapshot record ids back to their position in the sorted path list.
        let mut paths: Vec<PathBuf> = Vec::new();
        let mut walker = WalkDir::new(temp.path());
        if index.follow_links {
            walker = walker.follow_links(true);
        }
        for entry in walker
            .into_iter()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_file())
        {
            let path = entry.path().to_path_buf();
            if index.text_files_only && !is_text_file(&path) {
                continue;
            }
            paths.push(path);
        }
        paths.sort();
        let id_to_index: HashMap<RecordId, usize> = paths
            .iter()
            .enumerate()
            .map(|(idx, path)| {
                (
                    FileCorpusIndex::source_scoped_record_id(&source_id, temp.path(), path),
                    idx,
                )
            })
            .collect();
        let indices: Vec<usize> = snapshot
            .records
            .iter()
            .filter_map(|record| id_to_index.get(&record.id).copied())
            .collect();

        let min = indices.iter().copied().min().unwrap_or(0);
        let max = indices.iter().copied().max().unwrap_or(0);
        let span = max.saturating_sub(min);
        assert!(span > total / 2, "index span too narrow: {span}");
    }

    #[test]
    fn file_index_store_path_isolated_per_source() {
        let temp = tempdir().unwrap();
        let root = temp.path();
        let source_a: SourceId = "source_a".into();
        let source_b: SourceId = "source_b".into();
        let path_a = FileCorpusIndex::file_index_store_path(root, &source_a);
        let path_b = FileCorpusIndex::file_index_store_path(root, &source_b);
        assert_ne!(path_a, path_b);
    }

    #[test]
    fn file_index_store_size_within_path_budget() {
        let temp = tempdir().unwrap();
        let root = temp.path();
        let source_id: SourceId = "fs_size_budget".into();

        let paths = [
            root.join("alpha.txt"),
            root.join("beta.txt"),
            root.join("gamma.txt"),
        ];
        let mut payload_bytes = 0u64;
        for (idx, path) in paths.iter().enumerate() {
            // Write non-trivial content so we can assert it is not reflected in the index size.
            let content = "x".repeat(1024 * (idx + 1));
            payload_bytes += content.len() as u64;
            fs::write(path, content).unwrap();
        }

        let index = FileCorpusIndex::new(root, &source_id)
            .with_sampler_seed(1)
            .with_follow_links(false)
            .with_text_files_only(true);
        let _ = index
            .refresh_indexable(None, Some(3), |path| {
                let now = Utc::now();
                Ok(Some(DataRecord {
                    id: FileCorpusIndex::source_scoped_record_id(&source_id, root, path),
                    source: source_id.to_string(),
                    created_at: now,
                    updated_at: now,
                    quality: QualityScore { trust: 1.0 },
                    taxonomy: Vec::new(),
                    sections: vec![RecordSection {
                        role: SectionRole::Anchor,
                        heading: None,
                        text: "stub".into(),
                        sentences: vec!["stub".into()],
                    }],
                    meta_prefix: None,
                }))
            })
            .unwrap();

        let index_path = FileCorpusIndex::file_index_store_path(root, &source_id);
        let size = fs::metadata(&index_path).unwrap().len();
        let raw_bytes: u64 = paths
            .iter()
            .map(|path| path.to_string_lossy().len() as u64)
            .sum();
        // Per-index entries add key and store overhead, so allow a bounded cushion.
        let per_entry_overhead = 128u64;
        let max_size = raw_bytes
            .saturating_add((paths.len() as u64).saturating_mul(per_entry_overhead))
            .saturating_add(1024);
        assert!(
            size <= max_size,
            "index file too large: size={} max={}",
            size,
            max_size
        );
        assert!(
            size < payload_bytes,
            "index file should not include file contents: size={} payload={}",
            size,
            payload_bytes
        );
    }

    #[test]
    fn file_corpus_skips_unreadable_records() {
        let temp = tempdir().unwrap();
        let root = temp.path();
        let source_id: SourceId = "fs_skip_unreadable".into();
        let good_path = root.join("good.txt");
        let bad_path = root.join("bad.txt");
        fs::write(&good_path, "ok").unwrap();
        fs::write(&bad_path, "").unwrap();

        let index = FileCorpusIndex::new(root, &source_id)
            .with_sampler_seed(1)
            .with_follow_links(false)
            .with_text_files_only(true);
        let indexable = index
            .refresh_indexable(None, Some(10), |path| {
                build_stub_record(path, &source_id, root, &bad_path)
            })
            .unwrap();
        assert_eq!(indexable.records.len(), 1);
        assert!(indexable.records[0].id.contains("good.txt"));

        let streaming = index
            .refresh_streaming(None, Some(10), |path| {
                build_stub_record(path, &source_id, root, &bad_path)
            })
            .unwrap();
        assert_eq!(streaming.records.len(), 1);
        assert!(streaming.records[0].id.contains("good.txt"));
    }

    #[test]
    fn file_corpus_groups_by_date_directory() {
        let temp = tempdir().unwrap();
        let root = temp.path();
        let source_id: SourceId = "fs_grouped".into();
        let dir_count = 12;
        for idx in 0..dir_count {
            let month = (idx % 12) + 1;
            let day = (idx % 27) + 1;
            let year = 2020 + (idx % 3);
            let date = format!("{:02}-{:02}-{}", month, day, year);
            let dir = root.join(
                Path::new(&date)
                    .join(format!("{idx:02}"))
                    .join(format!("{:02}", (idx + 10) % 97))
                    .join(format!("{:02}", (idx + 20) % 97)),
            );
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join(format!("file_{idx:02}_a.txt")), "stub").unwrap();
            fs::write(dir.join(format!("file_{idx:02}_b.txt")), "stub").unwrap();
        }

        let index = FileCorpusIndex::new(root, &source_id)
            .with_sampler_seed(1)
            .with_follow_links(false)
            .with_text_files_only(true)
            .with_directory_grouping(true)
            .with_directory_grouping_window_divisor(3);
        let snapshot = index
            .refresh_indexable(None, Some(dir_count * 2), |path| {
                let now = Utc::now();
                Ok(Some(DataRecord {
                    id: FileCorpusIndex::source_scoped_record_id(&source_id, root, path),
                    source: source_id.to_string(),
                    created_at: now,
                    updated_at: now,
                    quality: QualityScore { trust: 1.0 },
                    taxonomy: Vec::new(),
                    sections: vec![RecordSection {
                        role: SectionRole::Anchor,
                        heading: None,
                        text: "stub".into(),
                        sentences: vec!["stub".into()],
                    }],
                    meta_prefix: None,
                }))
            })
            .unwrap();
        let grouped_snapshot_again = index
            .refresh_indexable(None, Some(dir_count * 2), |path| {
                let now = Utc::now();
                Ok(Some(DataRecord {
                    id: FileCorpusIndex::source_scoped_record_id(&source_id, root, path),
                    source: source_id.to_string(),
                    created_at: now,
                    updated_at: now,
                    quality: QualityScore { trust: 1.0 },
                    taxonomy: Vec::new(),
                    sections: vec![RecordSection {
                        role: SectionRole::Anchor,
                        heading: None,
                        text: "stub".into(),
                        sentences: vec!["stub".into()],
                    }],
                    meta_prefix: None,
                }))
            })
            .unwrap();

        let rel_path_from_record_id =
            |record_id: &str| PathBuf::from(record_id.split("::").nth(1).unwrap_or(""));
        let grouped_rel_paths: Vec<PathString> = snapshot
            .records
            .iter()
            .map(|record| {
                rel_path_from_record_id(&record.id)
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        let ungrouped_index = FileCorpusIndex::new(root, &source_id)
            .with_sampler_seed(1)
            .with_follow_links(false)
            .with_text_files_only(true);
        let ungrouped_snapshot = ungrouped_index
            .refresh_indexable(None, Some(dir_count * 2), |path| {
                let now = Utc::now();
                Ok(Some(DataRecord {
                    id: FileCorpusIndex::source_scoped_record_id(&source_id, root, path),
                    source: source_id.to_string(),
                    created_at: now,
                    updated_at: now,
                    quality: QualityScore { trust: 1.0 },
                    taxonomy: Vec::new(),
                    sections: vec![RecordSection {
                        role: SectionRole::Anchor,
                        heading: None,
                        text: "stub".into(),
                        sentences: vec!["stub".into()],
                    }],
                    meta_prefix: None,
                }))
            })
            .unwrap();
        let ungrouped_rel_paths: Vec<PathString> = ungrouped_snapshot
            .records
            .iter()
            .map(|record| {
                rel_path_from_record_id(&record.id)
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();

        // Average span between the first and last occurrence for each date key.
        // Grouping works by first collecting same-date files into the same window,
        // then shuffling only within that window. Because of that, two files from
        // the same date cannot end up farther apart than the window size, while
        // ungrouped shuffling can spread them across the whole list.
        let avg_pair_distance = |paths: &[PathString]| {
            let mut positions: HashMap<GroupKey, Vec<usize>> = HashMap::new();
            for (idx, path) in paths.iter().enumerate() {
                let date_key = path
                    .split(std::path::MAIN_SEPARATOR)
                    .next()
                    .unwrap_or("")
                    .to_string();
                positions.entry(date_key).or_default().push(idx);
            }
            let mut total = 0usize;
            let mut count = 0usize;
            for pos in positions.values() {
                if pos.len() >= 2 {
                    total += pos[pos.len() - 1] - pos[0];
                    count += 1;
                }
            }
            total as f64 / count.max(1) as f64
        };

        let grouped_rel_paths_again: Vec<PathString> = grouped_snapshot_again
            .records
            .iter()
            .map(|record| {
                rel_path_from_record_id(&record.id)
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();

        let sep = std::path::MAIN_SEPARATOR;
        let path_str = |value: &str| value.replace('/', &sep.to_string());
        let expected_grouped_rel_paths = vec![
            path_str("09-09-2022/08/18/28/file_08_b.txt"),
            path_str("03-03-2022/02/12/22/file_02_b.txt"),
            path_str("08-08-2021/07/17/27/file_07_a.txt"),
            path_str("05-05-2021/04/14/24/file_04_a.txt"),
            path_str("12-12-2022/11/21/31/file_11_a.txt"),
            path_str("10-10-2020/09/19/29/file_09_a.txt"),
            path_str("04-04-2020/03/13/23/file_03_b.txt"),
            path_str("07-07-2020/06/16/26/file_06_a.txt"),
            path_str("09-09-2022/08/18/28/file_08_a.txt"),
            path_str("03-03-2022/02/12/22/file_02_a.txt"),
            path_str("08-08-2021/07/17/27/file_07_b.txt"),
            path_str("05-05-2021/04/14/24/file_04_b.txt"),
            path_str("12-12-2022/11/21/31/file_11_b.txt"),
            path_str("10-10-2020/09/19/29/file_09_b.txt"),
            path_str("04-04-2020/03/13/23/file_03_a.txt"),
            path_str("07-07-2020/06/16/26/file_06_b.txt"),
            path_str("01-01-2020/00/10/20/file_00_b.txt"),
            path_str("01-01-2020/00/10/20/file_00_a.txt"),
            path_str("06-06-2022/05/15/25/file_05_a.txt"),
            path_str("06-06-2022/05/15/25/file_05_b.txt"),
            path_str("02-02-2021/01/11/21/file_01_b.txt"),
            path_str("02-02-2021/01/11/21/file_01_a.txt"),
            path_str("11-11-2021/10/20/30/file_10_a.txt"),
            path_str("11-11-2021/10/20/30/file_10_b.txt"),
        ];
        let expected_ungrouped_rel_paths = vec![
            path_str("09-09-2022/08/18/28/file_08_b.txt"),
            path_str("11-11-2021/10/20/30/file_10_a.txt"),
            path_str("12-12-2022/11/21/31/file_11_b.txt"),
            path_str("01-01-2020/00/10/20/file_00_a.txt"),
            path_str("02-02-2021/01/11/21/file_01_b.txt"),
            path_str("04-04-2020/03/13/23/file_03_a.txt"),
            path_str("05-05-2021/04/14/24/file_04_b.txt"),
            path_str("07-07-2020/06/16/26/file_06_a.txt"),
            path_str("08-08-2021/07/17/27/file_07_b.txt"),
            path_str("10-10-2020/09/19/29/file_09_a.txt"),
            path_str("11-11-2021/10/20/30/file_10_b.txt"),
            path_str("01-01-2020/00/10/20/file_00_b.txt"),
            path_str("03-03-2022/02/12/22/file_02_a.txt"),
            path_str("04-04-2020/03/13/23/file_03_b.txt"),
            path_str("06-06-2022/05/15/25/file_05_a.txt"),
            path_str("07-07-2020/06/16/26/file_06_b.txt"),
            path_str("09-09-2022/08/18/28/file_08_a.txt"),
            path_str("10-10-2020/09/19/29/file_09_b.txt"),
            path_str("12-12-2022/11/21/31/file_11_a.txt"),
            path_str("02-02-2021/01/11/21/file_01_a.txt"),
            path_str("03-03-2022/02/12/22/file_02_b.txt"),
            path_str("05-05-2021/04/14/24/file_04_a.txt"),
            path_str("06-06-2022/05/15/25/file_05_b.txt"),
            path_str("08-08-2021/07/17/27/file_07_a.txt"),
        ];
        assert_eq!(grouped_rel_paths, expected_grouped_rel_paths);
        assert_eq!(ungrouped_rel_paths, expected_ungrouped_rel_paths);
        assert_eq!(grouped_rel_paths, grouped_rel_paths_again);
        assert!(
            avg_pair_distance(&grouped_rel_paths) < avg_pair_distance(&ungrouped_rel_paths),
            "grouping should keep same-date pairs closer together"
        );
    }

    #[test]
    fn file_corpus_grouping_changes_pool_when_limited() {
        let temp = tempdir().unwrap();
        let root = temp.path();
        let source_id: SourceId = "fs_grouped_limited".into();
        let groups = ["alpha", "beta", "gamma", "delta"];
        for (gidx, group) in groups.iter().enumerate() {
            let dir = root.join(group).join(format!("section_{gidx}"));
            fs::create_dir_all(&dir).unwrap();
            for idx in 0..4 {
                fs::write(dir.join(format!("file_{gidx}_{idx}.txt")), "stub").unwrap();
            }
        }

        let limit = 6;
        let grouped_index = FileCorpusIndex::new(root, &source_id)
            .with_sampler_seed(1)
            .with_follow_links(false)
            .with_text_files_only(true)
            .with_directory_grouping(true)
            .with_directory_grouping_window_divisor(2);
        let grouped_snapshot = grouped_index
            .refresh_indexable(None, Some(limit), |path| {
                let now = Utc::now();
                Ok(Some(DataRecord {
                    id: FileCorpusIndex::source_scoped_record_id(&source_id, root, path),
                    source: source_id.to_string(),
                    created_at: now,
                    updated_at: now,
                    quality: QualityScore { trust: 1.0 },
                    taxonomy: Vec::new(),
                    sections: vec![RecordSection {
                        role: SectionRole::Anchor,
                        heading: None,
                        text: "stub".into(),
                        sentences: vec!["stub".into()],
                    }],
                    meta_prefix: None,
                }))
            })
            .unwrap();

        let ungrouped_index = FileCorpusIndex::new(root, &source_id)
            .with_sampler_seed(1)
            .with_follow_links(false)
            .with_text_files_only(true);
        let ungrouped_snapshot = ungrouped_index
            .refresh_indexable(None, Some(limit), |path| {
                let now = Utc::now();
                Ok(Some(DataRecord {
                    id: FileCorpusIndex::source_scoped_record_id(&source_id, root, path),
                    source: source_id.to_string(),
                    created_at: now,
                    updated_at: now,
                    quality: QualityScore { trust: 1.0 },
                    taxonomy: Vec::new(),
                    sections: vec![RecordSection {
                        role: SectionRole::Anchor,
                        heading: None,
                        text: "stub".into(),
                        sentences: vec!["stub".into()],
                    }],
                    meta_prefix: None,
                }))
            })
            .unwrap();

        let rel_path_from_record_id =
            |record_id: &str| PathBuf::from(record_id.split("::").nth(1).unwrap_or(""));
        let grouped_rel_paths: Vec<PathString> = grouped_snapshot
            .records
            .iter()
            .map(|record| {
                rel_path_from_record_id(&record.id)
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        let ungrouped_rel_paths: Vec<PathString> = ungrouped_snapshot
            .records
            .iter()
            .map(|record| {
                rel_path_from_record_id(&record.id)
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();

        let sep = std::path::MAIN_SEPARATOR;
        let path_str = |value: &str| value.replace('/', &sep.to_string());
        let expected_grouped_rel_paths = vec![
            path_str("beta/section_1/file_1_1.txt"),
            path_str("beta/section_1/file_1_0.txt"),
            path_str("gamma/section_2/file_2_3.txt"),
            path_str("gamma/section_2/file_2_1.txt"),
            path_str("delta/section_3/file_3_1.txt"),
            path_str("delta/section_3/file_3_3.txt"),
        ];
        let expected_ungrouped_rel_paths = vec![
            path_str("beta/section_1/file_1_1.txt"),
            path_str("alpha/section_0/file_0_0.txt"),
            path_str("delta/section_3/file_3_3.txt"),
            path_str("beta/section_1/file_1_2.txt"),
            path_str("alpha/section_0/file_0_1.txt"),
            path_str("gamma/section_2/file_2_0.txt"),
        ];

        assert_eq!(grouped_rel_paths, expected_grouped_rel_paths);
        assert_eq!(ungrouped_rel_paths, expected_ungrouped_rel_paths);
    }

    #[test]
    fn file_corpus_grouped_refresh_caps_dominant_directory() {
        let temp = tempdir().unwrap();
        let root = temp.path();
        let source_id: SourceId = "fs_grouped_cap".into();

        let dominant = root.join("dominant").join("bucket");
        fs::create_dir_all(&dominant).unwrap();
        for idx in 0..20 {
            fs::write(dominant.join(format!("dominant_{idx:02}.txt")), "stub").unwrap();
        }

        let minor_a = root.join("minor_a").join("bucket");
        fs::create_dir_all(&minor_a).unwrap();
        for idx in 0..4 {
            fs::write(minor_a.join(format!("minor_a_{idx:02}.txt")), "stub").unwrap();
        }

        let minor_b = root.join("minor_b").join("bucket");
        fs::create_dir_all(&minor_b).unwrap();
        for idx in 0..4 {
            fs::write(minor_b.join(format!("minor_b_{idx:02}.txt")), "stub").unwrap();
        }

        let limit = 9usize;
        let grouped_index = FileCorpusIndex::new(root, &source_id)
            .with_sampler_seed(1)
            .with_follow_links(false)
            .with_text_files_only(true)
            .with_directory_grouping(true)
            .with_directory_grouping_window_divisor(2);

        let snapshot = grouped_index
            .refresh_indexable(None, Some(limit), |path| {
                let now = Utc::now();
                Ok(Some(DataRecord {
                    id: FileCorpusIndex::source_scoped_record_id(&source_id, root, path),
                    source: source_id.to_string(),
                    created_at: now,
                    updated_at: now,
                    quality: QualityScore { trust: 1.0 },
                    taxonomy: Vec::new(),
                    sections: vec![RecordSection {
                        role: SectionRole::Anchor,
                        heading: None,
                        text: "stub".into(),
                        sentences: vec!["stub".into()],
                    }],
                    meta_prefix: None,
                }))
            })
            .unwrap();

        let mut counts: HashMap<GroupKey, usize> = HashMap::new();
        for record in &snapshot.records {
            let rel = record.id.split("::").nth(1).unwrap_or("");
            let top = Path::new(rel)
                .components()
                .next()
                .and_then(|part| part.as_os_str().to_str())
                .unwrap_or("")
                .to_string();
            *counts.entry(top).or_insert(0) += 1;
        }

        let expected_cap = 3usize; // ceil(9 / 3 groups)
        assert_eq!(snapshot.records.len(), limit);
        assert!(counts.contains_key("dominant"));
        assert!(counts.contains_key("minor_a"));
        assert!(counts.contains_key("minor_b"));
        assert!(
            counts.values().all(|count| *count <= expected_cap),
            "per-group cap must bound all groups in grouped refresh"
        );
    }

    #[test]
    fn file_corpus_grouped_refresh_cap_is_deterministic() {
        let temp = tempdir().unwrap();
        let root = temp.path();
        let source_id: SourceId = "fs_grouped_cap_deterministic".into();

        for (group, count) in [
            ("dominant", 20usize),
            ("minor_a", 4usize),
            ("minor_b", 4usize),
        ] {
            let dir = root.join(group).join("bucket");
            fs::create_dir_all(&dir).unwrap();
            for idx in 0..count {
                fs::write(dir.join(format!("{group}_{idx:02}.txt")), "stub").unwrap();
            }
        }

        let limit = 9usize;
        let index = FileCorpusIndex::new(root, &source_id)
            .with_sampler_seed(1)
            .with_follow_links(false)
            .with_text_files_only(true)
            .with_directory_grouping(true)
            .with_directory_grouping_window_divisor(2);

        let snapshot_a = index
            .refresh_indexable(None, Some(limit), |path| {
                let now = Utc::now();
                Ok(Some(DataRecord {
                    id: FileCorpusIndex::source_scoped_record_id(&source_id, root, path),
                    source: source_id.to_string(),
                    created_at: now,
                    updated_at: now,
                    quality: QualityScore { trust: 1.0 },
                    taxonomy: Vec::new(),
                    sections: vec![RecordSection {
                        role: SectionRole::Anchor,
                        heading: None,
                        text: "stub".into(),
                        sentences: vec!["stub".into()],
                    }],
                    meta_prefix: None,
                }))
            })
            .unwrap();

        let snapshot_b = index
            .refresh_indexable(None, Some(limit), |path| {
                let now = Utc::now();
                Ok(Some(DataRecord {
                    id: FileCorpusIndex::source_scoped_record_id(&source_id, root, path),
                    source: source_id.to_string(),
                    created_at: now,
                    updated_at: now,
                    quality: QualityScore { trust: 1.0 },
                    taxonomy: Vec::new(),
                    sections: vec![RecordSection {
                        role: SectionRole::Anchor,
                        heading: None,
                        text: "stub".into(),
                        sentences: vec!["stub".into()],
                    }],
                    meta_prefix: None,
                }))
            })
            .unwrap();

        let ids_a: Vec<RecordId> = snapshot_a
            .records
            .iter()
            .map(|record| record.id.clone())
            .collect();
        let ids_b: Vec<RecordId> = snapshot_b
            .records
            .iter()
            .map(|record| record.id.clone())
            .collect();
        assert_eq!(ids_a, ids_b);
    }

    #[test]
    fn builder_options_and_seed_requirements_are_enforced() {
        let dir = tempdir().unwrap();
        let index = FileCorpusIndex::new(dir.path(), "seeded")
            .with_follow_links(false)
            .with_text_files_only(false)
            .with_directory_grouping(true)
            .with_directory_grouping_window_divisor(0);

        assert!(!index.follow_links);
        assert!(!index.text_files_only);
        assert!(index.group_by_directory);
        assert_eq!(index.group_window_divisor, 1);

        let err = index.required_sampler_seed().unwrap_err();
        assert!(matches!(
            err,
            SamplerError::SourceInconsistent { ref details, .. } if details.contains("sampler seed")
        ));

        let seeded = index.with_sampler_seed(123);
        assert_eq!(seeded.required_sampler_seed().unwrap(), 123);
    }

    #[test]
    fn title_and_record_id_helpers_cover_error_paths() {
        let source_id = "helper_source".to_string();
        let root = PathBuf::from("/tmp/root");

        let with_underscores = PathBuf::from("/tmp/root/my_title.txt");
        let preserved =
            FileCorpusIndex::normalized_title_from_stem(&with_underscores, &source_id, false)
                .unwrap();
        assert_eq!(preserved, "my_title");
        let replaced =
            FileCorpusIndex::normalized_title_from_stem(&with_underscores, &source_id, true)
                .unwrap();
        assert_eq!(replaced, "my title");

        let err = FileCorpusIndex::normalized_title_from_stem(Path::new("/"), &source_id, true)
            .unwrap_err();
        assert!(matches!(
            err,
            SamplerError::SourceInconsistent { ref details, .. } if details.contains("file stem")
        ));

        let empty_title = FileCorpusIndex::normalized_title_from_stem(
            Path::new("/tmp/root/___ .txt"),
            &source_id,
            true,
        )
        .unwrap_err();
        assert!(matches!(
            empty_title,
            SamplerError::SourceInconsistent { ref details, .. } if details.contains("empty normalized title")
        ));

        let scoped = FileCorpusIndex::source_scoped_record_id(
            &source_id,
            &root,
            Path::new("/outside/path/doc.txt"),
        );
        assert!(scoped.starts_with("helper_source::"));
    }

    #[test]
    fn read_index_meta_mismatch_and_decode_errors_return_none_or_error() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("root");
        fs::create_dir_all(&root).unwrap();
        let index = FileCorpusIndex::new(&root, "meta_mismatch");
        let store = index.open_index_store().unwrap();

        let mismatch = FileIndexMeta {
            root: root.to_string_lossy().to_string(),
            follow_links: false,
            text_files_only: true,
            count: 0,
        };
        let mismatch_bytes = bitcode::encode(&mismatch);
        store.write(FILE_INDEX_META_KEY, &mismatch_bytes).unwrap();
        assert!(index.read_index_meta(&store).unwrap().is_none());

        let bad_bytes = vec![0xff, 0x00];
        store.write(FILE_INDEX_META_KEY, &bad_bytes).unwrap();
        match index.read_index_meta(&store) {
            Err(SamplerError::SourceInconsistent { details, .. }) => {
                assert!(details.contains("meta decode failed"));
            }
            _ => panic!("unexpected result variant"),
        }
    }

    #[test]
    fn read_index_batch_handles_missing_entries_and_skippable_errors() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("root");
        fs::create_dir_all(&root).unwrap();
        let index = FileCorpusIndex::new(&root, "batch_read").with_sampler_seed(5);
        let store = index.open_index_store().unwrap();

        let path = root.join("one.txt");
        fs::write(&path, "hello").unwrap();
        store
            .write(
                &FileCorpusIndex::index_key(0),
                path.to_string_lossy().as_bytes(),
            )
            .unwrap();

        let mut records = Vec::new();
        let missing =
            index.read_index_batch(&store, &mut |_path| Ok(None), &[1], &mut records, Some(10));
        assert!(matches!(
            missing,
            Err(SamplerError::SourceInconsistent { ref details, .. }) if details.contains("missing entry")
        ));

        let bad = root.join("bad.txt");
        fs::write(&bad, "bad").unwrap();
        store
            .write(
                &FileCorpusIndex::index_key(1),
                bad.to_string_lossy().as_bytes(),
            )
            .unwrap();

        let mut records = Vec::new();
        index
            .read_index_batch(
                &store,
                &mut |p| {
                    if p == bad.as_path() {
                        return Err(SamplerError::SourceInconsistent {
                            source_id: "batch_read".to_string(),
                            details: "skip me".to_string(),
                        });
                    }
                    Ok(Some(DataRecord {
                        id: "ok".to_string(),
                        source: "batch_read".to_string(),
                        created_at: Utc::now(),
                        updated_at: Utc::now(),
                        quality: QualityScore { trust: 1.0 },
                        taxonomy: Vec::new(),
                        sections: vec![RecordSection {
                            role: SectionRole::Anchor,
                            heading: None,
                            text: "ok".to_string(),
                            sentences: vec!["ok".to_string()],
                        }],
                        meta_prefix: None,
                    }))
                },
                &[0, 1],
                &mut records,
                Some(10),
            )
            .unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].id, "ok");
    }
}
