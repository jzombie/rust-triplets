#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

//! Read and write embedding triplets stored in simd-r-drive format.

/// Error types for SRD operations.
pub mod error;
/// Low-level source backed by a simd-r-drive data store.
pub mod srd_source;
/// Entry encoding/decoding and batch I/O for pair and triplet modes.
pub mod srd_triplet;

pub use error::SrdError;
pub use srd_source::SrdSource;
pub use srd_triplet::{
    FLAG_LABEL_NEGATIVE, SrdMode, SrdPairRecord, SrdPairWriteEntry, SrdRecord, SrdTripletRecord,
    SrdTripletWriteEntry, batch_read_entries, decode_entry, encode_entry, validate_write_batch,
    write_pair_entries, write_triplet_entries,
};
pub use triplets_core::data::PairLabel;
