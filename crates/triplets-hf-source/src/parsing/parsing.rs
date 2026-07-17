use std::fs;

use crate::constants::{
    HF_SOURCE_KEY_ANCHOR, HF_SOURCE_KEY_CONTEXT, HF_SOURCE_KEY_NEGATIVE, HF_SOURCE_KEY_POSITIVE,
    HF_SOURCE_KEY_SOURCE_ID, HF_SOURCE_KEY_TEXT, HF_SOURCE_KEY_TEXT_COLUMNS, HF_SOURCE_KEY_TRUST,
    HF_SOURCE_KEY_WEIGHT,
};

/// Parsed Hugging Face source-list entry with explicit field mappings.
#[derive(Clone, Debug)]
pub struct HfSourceEntry {
    /// Full hf:// URI for dataset/config/split.
    pub uri: String,
    /// Anchor candidate columns (ordered).
    ///
    /// Each candidate is tried in order; the first whose value is present and
    /// non-empty is used as the anchor role for the row.  When the list is
    /// non-empty and no candidate yields content, the row is skipped.
    pub anchor_columns: Vec<String>,
    /// Positive candidate columns (ordered).
    ///
    /// Each candidate is tried in order; the first whose value is present and
    /// non-empty is used as the positive role for the row.  When the list is
    /// non-empty and no candidate yields content, the row is skipped.
    pub positive_columns: Vec<String>,
    /// Negative candidate columns (ordered).
    ///
    /// Used only in **role-based mode**.  Every listed column is required: if
    /// any is missing or blank the row is skipped.
    ///
    /// When a column value is a list (e.g. `["neg1", "neg2"]`), each element
    /// is expanded into a separate `SectionRole::Context` section.  This
    /// supports HuggingFace "dict" datasets where negatives are provided as
    /// a list within the same row.
    pub negative_columns: Vec<String>,
    /// Optional context columns (ordered).
    ///
    /// Used only in **role-based mode** (i.e. when `anchor_columns` and/or
    /// `positive_columns` are set).  Every listed column is required: if any
    /// is missing or blank the row is skipped.
    ///
    /// Each column becomes an additional `SectionRole::Context` section in the
    /// emitted record, appended after the positive section.  In contrast to
    /// `anchor_columns`/`positive_columns`, there is no coalescing — all
    /// columns contribute independently as separate sections.
    ///
    /// Not used in **text-columns mode** (`text_columns` non-empty,
    /// `anchor_columns` empty): in that mode only `text_columns` is consulted.
    pub context_columns: Vec<String>,
    /// Text candidate columns (ordered) for text-columns mode.
    ///
    /// Each candidate is tried in order; the first whose value is present and
    /// non-empty is used as the single text content for the row.  When the
    /// list is non-empty and no candidate yields content, the row is skipped.
    pub text_columns: Vec<String>,
    /// Optional trust/quality override for all records produced by this source.
    ///
    /// When set, overrides the default `QualityScore::default().trust` (0.5)
    /// for every record emitted by this source.  Must be in `[0.0, 1.0]`.
    pub trust: Option<f32>,
    /// Optional weight for weighted source scheduling.
    ///
    /// When set, used by [`crate::builder::build_hf_sources_with_weights`] to populate a
    /// per-source weight map that callers pass to
    /// `Sampler::next_triplet_batch_with_weights` for weighted scheduling.
    /// Must be `> 0.0`.
    pub weight: Option<f32>,
    /// Optional source ID override.
    ///
    /// When set, this string is used as the source identifier instead of the
    /// auto-derived slug from the dataset URI.  Useful for giving a stable,
    /// human-readable name to a source independently of its dataset/config/split
    /// path.  Deduplication suffixes are **not** applied to explicit source IDs.
    pub source_id: Option<String>,
}

impl PartialEq for HfSourceEntry {
    fn eq(&self, other: &Self) -> bool {
        self.uri == other.uri
            && self.anchor_columns == other.anchor_columns
            && self.positive_columns == other.positive_columns
            && self.negative_columns == other.negative_columns
            && self.context_columns == other.context_columns
            && self.text_columns == other.text_columns
            && self.source_id == other.source_id
            // Compare f32 bits so that identical bit patterns are considered equal.
            // Valid trust values are never NaN, so bit-level comparison is correct.
            && self.trust.map(f32::to_bits) == other.trust.map(f32::to_bits)
            && self.weight.map(f32::to_bits) == other.weight.map(f32::to_bits)
    }
}

impl Eq for HfSourceEntry {}

/// Parsed Hugging Face source list with explicit mappings.
#[derive(Debug, Clone)]
pub struct HfListRoots {
    /// The source list file path used for loading.
    pub source_list: String,
    /// Parsed sources with explicit field mappings.
    pub sources: Vec<HfSourceEntry>,
}

/// Split a comma-delimited field list into trimmed column names.
pub fn parse_csv_fields(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(ToString::to_string)
        .collect()
}

/// Parse a single source-list line of the form:
/// `hf://org/dataset/config/split anchor=... positive=... context=a,b text=x,y`.
pub fn parse_hf_source_line(line: &str) -> Result<HfSourceEntry, String> {
    let mut parts = line.split_whitespace();
    let Some(uri) = parts.next() else {
        return Err("empty source line".to_string());
    };
    if !uri.starts_with("hf://") {
        return Err(format!("unsupported source URI (expected hf://...): {uri}"));
    }

    let mut entry = HfSourceEntry {
        uri: uri.to_string(),
        anchor_columns: Vec::new(),
        positive_columns: Vec::new(),
        negative_columns: Vec::new(),
        context_columns: Vec::new(),
        text_columns: Vec::new(),
        trust: None,
        weight: None,
        source_id: None,
    };

    for token in parts {
        let Some((raw_key, raw_value)) = token.split_once('=') else {
            return Err(format!(
                "invalid mapping token '{token}' (expected key=value)"
            ));
        };
        let key = raw_key.trim().to_ascii_lowercase();
        let value = raw_value.trim();
        match key.as_str() {
            HF_SOURCE_KEY_ANCHOR => {
                entry.anchor_columns = parse_csv_fields(value);
            }
            HF_SOURCE_KEY_POSITIVE => {
                entry.positive_columns = parse_csv_fields(value);
            }
            HF_SOURCE_KEY_NEGATIVE => {
                entry.negative_columns = parse_csv_fields(value);
            }
            HF_SOURCE_KEY_CONTEXT => {
                entry.context_columns = parse_csv_fields(value);
            }
            HF_SOURCE_KEY_TEXT | HF_SOURCE_KEY_TEXT_COLUMNS => {
                entry.text_columns = parse_csv_fields(value);
            }
            HF_SOURCE_KEY_TRUST => {
                let t: f32 = value.parse().map_err(|_| {
                    format!("invalid trust value '{value}': expected a float in [0.0, 1.0]")
                })?;
                if !(0.0..=1.0).contains(&t) {
                    return Err(format!("trust value {t} is out of range [0.0, 1.0]"));
                }
                entry.trust = Some(t);
            }
            HF_SOURCE_KEY_SOURCE_ID => {
                if value.is_empty() {
                    return Err("source_id must not be empty".to_string());
                }
                entry.source_id = Some(value.to_string());
            }
            HF_SOURCE_KEY_WEIGHT => {
                let w: f32 = value.parse().map_err(|_| {
                    format!("invalid weight value '{value}': expected a positive float")
                })?;
                if w <= 0.0 {
                    return Err(format!("weight value {w} must be > 0.0"));
                }
                entry.weight = Some(w);
            }
            _ => {
                return Err(format!("unsupported mapping key '{raw_key}'"));
            }
        }
    }

    let has_explicit_mapping = !entry.anchor_columns.is_empty()
        || !entry.positive_columns.is_empty()
        || !entry.negative_columns.is_empty()
        || !entry.context_columns.is_empty()
        || !entry.text_columns.is_empty();
    if !has_explicit_mapping {
        return Err(format!(
            "source '{}' has no field mapping; expected at least one of anchor=, positive=, negative=, context=, text=",
            entry.uri
        ));
    }

    Ok(entry)
}

/// Parse an hf:// URI into dataset/config/split components.
pub fn parse_hf_uri(uri: &str) -> Result<(String, String, String), String> {
    let trimmed = uri.trim();
    let Some(rest) = trimmed.strip_prefix("hf://") else {
        return Err(format!(
            "unsupported source URI (expected hf://...): {trimmed}"
        ));
    };

    let parts = rest
        .split('/')
        .filter(|part| !part.trim().is_empty())
        .collect::<Vec<_>>();

    if parts.len() < 2 {
        return Err(format!("invalid hf URI (need hf://org/dataset): {trimmed}"));
    }

    let dataset = format!("{}/{}", parts[0], parts[1]);
    let config = parts.get(2).copied().unwrap_or("default").to_string();
    // No trailing split component → empty string, which disables split-filtering
    // so all HF splits are discovered and triplets' own split logic handles partitioning.
    let split = parts.get(3).copied().unwrap_or("").to_string();

    Ok((dataset, config, split))
}

/// Load a Hugging Face source list file containing explicit field mappings.
pub fn load_hf_sources_from_list(path: &str) -> Result<Vec<HfSourceEntry>, String> {
    let body = fs::read_to_string(path).map_err(|err| format!("{err}"))?;
    let mut out = Vec::new();
    for (line_no, raw) in body.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parsed = parse_hf_source_line(line).map_err(|err| {
            format!(
                "invalid source-list entry at {}:{} -> {}",
                path,
                line_no + 1,
                err
            )
        })?;
        out.push(parsed);
    }
    Ok(out)
}

/// Resolve parsed Hugging Face source list entries into a structured root.
pub fn resolve_hf_list_roots(source_list: String) -> Result<HfListRoots, String> {
    let sources = load_hf_sources_from_list(&source_list)?;
    if sources.is_empty() {
        return Err(format!("no hf:// entries found in {}", source_list));
    }
    Ok(HfListRoots {
        source_list,
        sources,
    })
}

/// Sanitize a single component string for use in a source ID.
///
/// Replaces any character that is not alphanumeric, `-`, or `_` with `-`.
pub(crate) fn sanitize_source_id_component(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

/// Derive a human-readable source ID slug from parsed HF URI components.
///
/// Uses the short dataset name (the portion after the last `/`), then appends
/// `.{config}` when config is not `"default"` and `.{split}` when split is
/// not empty and not `"train"`.  Any character that is not alphanumeric,
/// `-`, or `_` is replaced with `-`.
pub fn hf_source_id_slug(dataset: &str, config: &str, split: &str) -> String {
    let short_name = dataset.rfind('/').map_or(dataset, |i| &dataset[i + 1..]);
    let mut slug = sanitize_source_id_component(short_name);
    if !config.is_empty() && config != "default" {
        slug.push('.');
        slug.push_str(&sanitize_source_id_component(config));
    }
    if !split.is_empty() && split != "train" {
        slug.push('.');
        slug.push_str(&sanitize_source_id_component(split));
    }
    if slug.is_empty() {
        slug = "hf".to_string();
    }
    slug
}
