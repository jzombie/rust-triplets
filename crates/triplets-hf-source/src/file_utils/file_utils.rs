use std::path::Path;

/// Extract the inner format extension from a file path, handling compound
/// extensions like `.jsonl.gz`. Returns the innermost recognized format.
///
/// Examples:
/// - `file.jsonl.gz` → `Some("jsonl")`
/// - `file.parquet` → `Some("parquet")`
/// - `file.gz` → `None` (no inner format)
/// - `file.simdr` → `Some("simdr")`
pub(crate) fn resolve_inner_extension(path: &Path) -> Option<String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())?
        .to_ascii_lowercase();

    // Handle compression extensions
    if ext == "gz" {
        let stem_ext = path
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|stem| Path::new(stem).extension())
            .and_then(|e| e.to_str())?
            .to_ascii_lowercase();
        Some(stem_ext)
    } else {
        Some(ext)
    }
}

/// Check if a file path has a gzip compression extension.
pub(crate) fn is_gzip_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gz"))
}

// TODO: Obtain list from centralized place
/// Check if a file path is a text-based format that should be transcoded to .simdr.
pub(crate) fn is_transient_text(path: &Path) -> bool {
    resolve_inner_extension(path)
        .is_some_and(|ext| ext == "jsonl" || ext == "ndjson" || ext == "json" || ext == "txt")
}
