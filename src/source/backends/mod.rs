/// Column-mapped CSV file source implementation.
pub mod csv_source;
/// Generic configurable filesystem-backed source implementation.
pub mod file_source;
/// In-memory source backed by a `Vec<DataRecord>`.
pub mod in_memory_source;

#[cfg(feature = "huggingface")]
/// Hugging Face snapshot-backed row source implementation.
pub mod huggingface_source;
