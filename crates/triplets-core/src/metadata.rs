use jiff::civil::Date;
use std::fmt::Display;

pub use crate::constants::metadata::{META_FIELD_DATE, METADATA_DELIMITER};

/// Canonical identifier for metadata fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetadataKey {
    name: &'static str,
}

impl MetadataKey {
    /// Create a metadata key with a canonical static name.
    pub const fn new(name: &'static str) -> Self {
        Self { name }
    }

    /// Return the raw key name.
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

// TODO: This could be more extensible for non-U.S formats; it's a bit too hardcoded
/// Build common date string variants for metadata prefixes.
pub fn build_date_meta_values(date: &Date) -> Vec<crate::types::MetaValue> {
    // Month names are fixed English tables: civil dates expose only numeric
    // components, and these display variants are stable by contract.
    const MONTH_SHORT: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    const MONTH_FULL: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    let year = date.year();
    let month = date.month() as usize; // 1..=12, guaranteed by construction
    let day = date.day();
    let short = MONTH_SHORT[month - 1];
    let full = MONTH_FULL[month - 1];
    let mut vals = vec![
        date.to_string(),                      // 2024-10-15 (ISO)
        format!("{short}. {day}, {year}"),     // Oct. 15, 2024
        format!("{full} {day}, {year}"),       // October 15, 2024
        format!("{day:02}.{month:02}.{year}"), // 15.10.2024
        format!("{month:02}/{day:02}/{year}"), // 02/25/2025
        format!("{short} {day}, {year}"),      // Oct 15, 2024 (no dot)
    ];
    vals.sort();
    vals.dedup();
    vals
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_key_encodes_and_strips_values() {
        let encoded = META_FIELD_DATE.encode("2025-02-23");
        assert_eq!(encoded, "date=2025-02-23");
        assert_eq!(META_FIELD_DATE.strip(&encoded), Some("2025-02-23"));
        assert_eq!(META_FIELD_DATE.strip("other=2025-02-23"), None);
    }

    #[test]
    fn date_meta_values_are_deduped_and_include_expected_formats() {
        let date = Date::new(2025, 2, 25).unwrap();
        let values = build_date_meta_values(&date);

        assert!(values.contains(&"2025-02-25".to_string()));
        assert!(values.contains(&"02/25/2025".to_string()));
        assert!(values.contains(&"25.02.2025".to_string()));

        let mut uniq = values.clone();
        uniq.sort();
        uniq.dedup();
        assert_eq!(uniq.len(), values.len());
    }

    #[test]
    fn metadata_key_new_and_as_str_work() {
        const CUSTOM: MetadataKey = MetadataKey::new("custom");
        assert_eq!(CUSTOM.as_str(), "custom");
        assert_eq!(CUSTOM.encode(42), "custom=42");
        assert_eq!(CUSTOM.strip("custom=42"), Some("42"));
        assert_eq!(CUSTOM.strip("custom42"), None);

        let runtime_key = MetadataKey::new("runtime");
        assert_eq!(runtime_key.as_str(), "runtime");
    }
}
