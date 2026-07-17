use super::*;
use std::path::Path;

#[test]
fn is_gzip_path_detects_gz_extension() {
    assert!(is_gzip_path(Path::new("file.jsonl.gz")));
    assert!(is_gzip_path(Path::new("file.GZ")));
    assert!(is_gzip_path(Path::new("file.Gz")));
    assert!(is_gzip_path(Path::new("file.tar.gz")));
    assert!(!is_gzip_path(Path::new("file.parquet")));
    assert!(!is_gzip_path(Path::new("file.jsonl")));
    assert!(!is_gzip_path(Path::new("file.simdr")));
    assert!(!is_gzip_path(Path::new("no-extension")));
    assert!(!is_gzip_path(Path::new("file.gz.bak")));
    assert!(!is_gzip_path(Path::new("")));
}

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
