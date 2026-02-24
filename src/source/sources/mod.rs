/// Generic configurable filesystem-backed source implementation.
pub mod file_source;

#[cfg(feature = "huggingface")]
/// Hugging Face snapshot-backed row source implementation.
pub mod huggingface;
