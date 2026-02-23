use chrono::{DateTime, Utc};
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::data::DataRecord;
use crate::errors::SamplerError;
use crate::hash::stable_hash_path;
use crate::source::{SourceCursor, SourceSnapshot};

/// Filesystem transport that can incrementally scan files under a root.
pub struct FileStream {
    root: PathBuf,
    follow_links: bool,
}

impl FileStream {
    /// Create a stream rooted at `root`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            follow_links: false,
        }
    }

    /// Configure symlink traversal.
    pub fn with_follow_symlinks(mut self, follow_links: bool) -> Self {
        self.follow_links = follow_links;
        self
    }

    /// Build a paged snapshot that advances through files using the cursor.
    pub fn stream_incremental<F>(
        &self,
        cursor: Option<&SourceCursor>,
        limit: Option<usize>,
        mut build_record: F,
    ) -> Result<SourceSnapshot, SamplerError>
    where
        F: FnMut(&Path) -> Result<Option<DataRecord>, SamplerError>,
    {
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
            candidates.push(entry.path().to_path_buf());
        }
        candidates.sort_by(|a, b| {
            let left = stable_path_shuffle_key(a);
            let right = stable_path_shuffle_key(b);
            left.cmp(&right).then_with(|| a.cmp(b))
        });
        let total = candidates.len();
        let mut start = cursor.map(|cursor| cursor.revision as usize).unwrap_or(0);
        if total > 0 && start >= total {
            start = 0;
        }
        let max = limit.unwrap_or(total);
        let mut records = Vec::new();
        for idx in 0..total {
            let pos = (start + idx) % total;
            let path = &candidates[pos];
            match build_record(path)? {
                Some(record) => {
                    records.push(record);
                    if records.len() >= max {
                        break;
                    }
                }
                None => continue,
            }
        }
        let last_seen = records
            .iter()
            .map(|record| record.updated_at)
            .max()
            .unwrap_or_else(Utc::now);
        let next_start = if total == 0 {
            0
        } else {
            (start + records.len()) % total
        };
        Ok(SourceSnapshot {
            records,
            cursor: SourceCursor {
                last_seen,
                revision: next_start as u64,
            },
        })
    }
}

/// True if the path has a `.txt` extension (case-insensitive).
pub fn is_text_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("txt"))
        .unwrap_or(false)
}

/// Best-effort file modified time.
pub fn file_mtime(path: &Path) -> Option<DateTime<Utc>> {
    let metadata = fs::metadata(path).ok()?;
    let modified = metadata.modified().ok()?;
    Some(system_time_to_utc(modified))
}

/// Best-effort (created_at, updated_at) pair for a file.
pub fn file_times(path: &Path) -> (DateTime<Utc>, DateTime<Utc>) {
    let metadata = fs::metadata(path).ok();
    let updated_at = metadata
        .as_ref()
        .and_then(|meta| meta.modified().ok())
        .map(system_time_to_utc)
        .unwrap_or_else(Utc::now);
    let created_at = metadata
        .and_then(|meta| meta.created().ok())
        .map(system_time_to_utc)
        .unwrap_or(updated_at);
    (created_at, updated_at)
}

fn system_time_to_utc(time: std::time::SystemTime) -> DateTime<Utc> {
    DateTime::<Utc>::from(time)
}

/// Stable hash used to pseudo-randomize file iteration order.
///
/// Paths are hashed to avoid lexicographic ordering biases (e.g., time-based folders).
/// This is used by `FileStream::stream_incremental` when a direct hash-ordered
/// traversal is desired.
pub fn stable_path_shuffle_key(path: &Path) -> u64 {
    stable_hash_path(0, path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};
    use std::collections::HashSet;
    use std::fs;
    use tempfile::tempdir;

    fn build_record(path: &Path) -> DataRecord {
        let id = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown")
            .to_string();
        let created_at = Utc.with_ymd_and_hms(2025, 1, 1, 12, 0, 0).unwrap();
        DataRecord {
            id,
            source: "file_stream_test".to_string(),
            created_at,
            updated_at: created_at,
            quality: crate::QualityScore { trust: 1.0 },
            taxonomy: Vec::new(),
            sections: Vec::new(),
            meta_prefix: None,
        }
    }

    #[test]
    fn stream_pages_follow_pseudorandom_order() {
        let temp = tempdir().unwrap();
        let root = temp.path();

        let files = [
            "alpha.txt",
            "bravo.txt",
            "charlie.txt",
            "delta.txt",
            "echo.txt",
            "foxtrot.txt",
            "golf.txt",
            "hotel.txt",
        ];
        for name in &files {
            fs::write(root.join(name), name.as_bytes()).unwrap();
        }

        let full_snapshot = FileStream::new(root)
            .stream_incremental(None, None, |path| Ok(Some(build_record(path))))
            .unwrap();
        let full_order: Vec<crate::types::PathString> = full_snapshot
            .records
            .into_iter()
            .map(|record| record.id)
            .collect();
        let expected_set: HashSet<crate::types::PathString> =
            files.iter().map(|name| (*name).to_string()).collect();
        let expected: Vec<crate::types::PathString> = full_order
            .iter()
            .filter(|id| expected_set.contains(*id))
            .cloned()
            .collect();
        let total = expected.len();
        assert_eq!(total, files.len());

        let mut seen = HashSet::new();
        let mut cursor = None;
        let mut collected = Vec::new();
        let page_size = 3;

        let max_pages = full_order.len().div_ceil(page_size) + 2;
        for _ in 0..max_pages {
            let snapshot = FileStream::new(root)
                .stream_incremental(cursor.as_ref(), Some(page_size), |path| {
                    Ok(Some(build_record(path)))
                })
                .unwrap();
            cursor = Some(snapshot.cursor);
            for record in snapshot.records {
                if seen.insert(record.id.clone()) {
                    collected.push(record.id);
                }
            }
            if collected
                .iter()
                .filter(|id| expected_set.contains(*id))
                .count()
                >= total
            {
                break;
            }
        }

        let collected_expected: Vec<crate::types::PathString> = collected
            .into_iter()
            .filter(|id| expected_set.contains(id))
            .collect();
        assert_eq!(collected_expected.len(), total);
        assert_eq!(collected_expected, expected);
    }
}
