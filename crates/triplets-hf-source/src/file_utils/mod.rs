pub mod file_utils;
pub(crate) use file_utils::{is_gzip_path, is_transient_text, resolve_inner_extension};

#[cfg(test)]
mod file_utils_tests;
