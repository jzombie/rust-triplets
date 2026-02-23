use chrono::NaiveDate;
use std::fmt::Display;

pub use crate::constants::metadata::{META_FIELD_DATE, METADATA_DELIMITER};

/// Canonical identifier for metadata fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetadataKey {
    name: &'static str,
}

impl MetadataKey {
    pub const fn new(name: &'static str) -> Self {
        Self { name }
    }

    pub const fn as_str(&self) -> &'static str {
        self.name
    }

    /// Encode a value using the shared delimiter (e.g., "date=2025-01-01").
    pub fn encode(&self, value: impl Display) -> String {
        format!("{}{}{}", self.name, METADATA_DELIMITER, value)
    }

    /// Strip the field prefix from a serialized metadata entry.
    pub fn strip<'a>(&self, entry: &'a str) -> Option<&'a str> {
        entry
            .strip_prefix(self.name)
            .and_then(|rest| rest.strip_prefix(METADATA_DELIMITER))
    }
}

pub fn build_date_meta_values(date: &NaiveDate) -> Vec<crate::types::MetaValue> {
    let mut vals = vec![
        date.format("%Y-%m-%d").to_string(),    // 2024-10-15 (ISO)
        date.format("%b. %-d, %Y").to_string(), // Oct. 15, 2024
        date.format("%B %-d, %Y").to_string(),  // October 15, 2024
        date.format("%d.%m.%Y").to_string(),    // 15.10.2024
        date.format("%m/%d/%Y").to_string(),    // 02/25/2025
        date.format("%b %-d, %Y").to_string(),  // Oct 15, 2024 (no dot)
    ];
    vals.sort();
    vals.dedup();
    vals
}
