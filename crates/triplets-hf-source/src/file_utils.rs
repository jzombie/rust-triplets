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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn resolve_inner_extension_handles_compound_extensions() {
        // Compound extensions
        assert_eq!(
            resolve_inner_extension(Path::new("file.jsonl.gz")),
            Some("jsonl".to_string())
        );
        assert_eq!(
            resolve_inner_extension(Path::new("file.ndjson.gz")),
            Some("ndjson".to_string())
        );
        assert_eq!(
            resolve_inner_extension(Path::new("file.txt.gz")),
            Some("txt".to_string())
        );

        // Simple extensions
        assert_eq!(
            resolve_inner_extension(Path::new("file.parquet")),
            Some("parquet".to_string())
        );
        assert_eq!(
            resolve_inner_extension(Path::new("file.simdr")),
            Some("simdr".to_string())
        );
        assert_eq!(
            resolve_inner_extension(Path::new("file.jsonl")),
            Some("jsonl".to_string())
        );
        assert_eq!(
            resolve_inner_extension(Path::new("file.ndjson")),
            Some("ndjson".to_string())
        );

        // No inner format (just .gz)
        assert_eq!(resolve_inner_extension(Path::new("file.gz")), None);

        // No extension
        assert_eq!(resolve_inner_extension(Path::new("no-extension")), None);

        // Case insensitive
        assert_eq!(
            resolve_inner_extension(Path::new("file.JSONL.GZ")),
            Some("jsonl".to_string())
        );
        assert_eq!(
            resolve_inner_extension(Path::new("file.PARQUET")),
            Some("parquet".to_string())
        );

        // Boundary cases
        assert_eq!(
            resolve_inner_extension(Path::new("file.tar.gz")),
            Some("tar".to_string())
        );
        assert_eq!(
            resolve_inner_extension(Path::new(".hidden.jsonl.gz")),
            Some("jsonl".to_string())
        );
        assert_eq!(
            resolve_inner_extension(Path::new("file.gz.bak")),
            Some("bak".to_string())
        );
        assert_eq!(resolve_inner_extension(Path::new("")), None);
    }
}
