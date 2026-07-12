use simd_r_drive::storage_engine::DataStore;
use simd_r_drive::storage_engine::traits::{DataStoreReader, DataStoreWriter};

use crate::error::SrdError;

type Result<T> = std::result::Result<T, SrdError>;

/// Mode byte stored as the first byte of each simd-r-drive entry.
pub const MODE_PAIR: u8 = 0;
/// Mode byte for triplet entries (anchor + positive + negative).
pub const MODE_TRIPLET: u8 = 1;

/// Flag bits stored as the second byte of each entry.
pub const FLAG_POS_SAME: u8 = 1 << 0; // positive same as anchor
/// Negative-embedding flag: bit 1 indicates negative is the same as anchor (triplet only).
pub const FLAG_NEG_SAME: u8 = 1 << 1; // negative same as anchor (triplet only)

/// Mode of an simd-r-drive embedding store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SrdMode {
    /// Pair mode: anchor + positive per entry.
    Pair,
    /// Triplet mode: anchor + positive + negative per entry.
    Triplet,
}

impl SrdMode {
    /// Decode a mode byte into an `SrdMode`.
    pub fn from_byte(b: u8) -> Result<Self> {
        match b {
            MODE_PAIR => Ok(SrdMode::Pair),
            MODE_TRIPLET => Ok(SrdMode::Triplet),
            other => Err(SrdError::UnknownMode(other)),
        }
    }

    /// Encode this mode as a single byte.
    pub fn to_byte(self) -> u8 {
        match self {
            SrdMode::Pair => MODE_PAIR,
            SrdMode::Triplet => MODE_TRIPLET,
        }
    }

    /// Number of sections (roles) per entry in this mode.
    pub fn section_count(self) -> usize {
        match self {
            SrdMode::Pair => 2,
            SrdMode::Triplet => 3,
        }
    }
}

/// A decoded simd-r-drive entry containing embeddings and text.
#[derive(Clone, Debug)]
pub struct SrdEntry {
    /// Whether this entry is pair or triplet mode.
    pub mode: SrdMode,
    /// True when the positive embedding is the same pointer as the anchor.
    pub same_as_anchor: bool,
    /// Anchor embedding vector.
    pub anchor_emb: Vec<f32>,
    /// Anchor text string.
    pub anchor_text: String,
    /// Positive embedding vector.
    pub pos_emb: Vec<f32>,
    /// Positive text string.
    pub pos_text: String,
    /// Negative embedding vector (triplet mode only).
    pub neg_emb: Option<Vec<f32>>,
    /// Negative text string (triplet mode only).
    pub neg_text: Option<String>,
}

// ---------------------------------------------------------------------------
// Unified encoder
// ---------------------------------------------------------------------------

/// Encode an entry into bytes using the unified format.
///
/// Format: `[mode:1] [flags:1] [text_lens...] [embeddings...] [texts...]`
///
/// Flags: bit 0 = pos same as anchor, bit 1 = neg same as anchor.
/// Only unique embeddings and texts are stored.
pub fn encode_entry(entry: &SrdEntry) -> Vec<u8> {
    let emb_dim = entry.anchor_emb.len();
    let emb_bytes = emb_dim * 4;

    // Compute flags
    let mut flags: u8 = 0;
    if entry.same_as_anchor {
        flags |= FLAG_POS_SAME;
    }
    if entry.mode == SrdMode::Triplet
        && let Some((neg_emb, neg_text)) = entry.neg_emb.as_ref().zip(entry.neg_text.as_ref())
        && neg_emb == &entry.anchor_emb
        && neg_text == &entry.anchor_text
    {
        flags |= FLAG_NEG_SAME;
    }

    // Count unique texts and embeddings
    let unique_texts = match entry.mode {
        SrdMode::Pair => {
            if flags & FLAG_POS_SAME != 0 {
                1
            } else {
                2
            }
        }
        SrdMode::Triplet => {
            let mut n = 1; // anchor always
            if flags & FLAG_POS_SAME == 0 {
                n += 1;
            }
            if flags & FLAG_NEG_SAME == 0 {
                n += 1;
            }
            n
        }
    };
    let unique_embs = unique_texts; // 1 emb per unique text

    // Compute buffer size
    let text_lens_size = unique_texts * 4;
    let embs_size = unique_embs * emb_bytes;
    let texts_size: usize = match entry.mode {
        SrdMode::Pair => {
            if flags & FLAG_POS_SAME != 0 {
                entry.anchor_text.len()
            } else {
                entry.anchor_text.len() + entry.pos_text.len()
            }
        }
        SrdMode::Triplet => {
            let mut s = entry.anchor_text.len();
            if flags & FLAG_POS_SAME == 0 {
                s += entry.pos_text.len();
            }
            if let (Some(t), 0) = (&entry.neg_text, flags & FLAG_NEG_SAME) {
                s += t.len();
            }
            s
        }
    };

    let mut buf = Vec::with_capacity(2 + text_lens_size + embs_size + texts_size);

    // Header
    buf.push(entry.mode.to_byte());
    buf.push(flags);

    // Text lengths
    match entry.mode {
        SrdMode::Pair => {
            buf.extend_from_slice(&(entry.anchor_text.len() as u32).to_le_bytes());
            if flags & FLAG_POS_SAME == 0 {
                buf.extend_from_slice(&(entry.pos_text.len() as u32).to_le_bytes());
            }
        }
        SrdMode::Triplet => {
            buf.extend_from_slice(&(entry.anchor_text.len() as u32).to_le_bytes());
            if flags & FLAG_POS_SAME == 0 {
                buf.extend_from_slice(&(entry.pos_text.len() as u32).to_le_bytes());
            }
            if flags & FLAG_NEG_SAME == 0 {
                let n = entry.neg_text.as_ref().map_or(0, |t| t.len());
                buf.extend_from_slice(&(n as u32).to_le_bytes());
            }
        }
    }

    // Embeddings
    write_emb_slice(&mut buf, &entry.anchor_emb);
    if flags & FLAG_POS_SAME == 0 {
        write_emb_slice(&mut buf, &entry.pos_emb);
    }
    if entry.mode == SrdMode::Triplet
        && flags & FLAG_NEG_SAME == 0
        && let Some(neg_emb) = &entry.neg_emb
    {
        write_emb_slice(&mut buf, neg_emb);
    }

    // Texts
    buf.extend_from_slice(entry.anchor_text.as_bytes());
    if flags & FLAG_POS_SAME == 0 {
        buf.extend_from_slice(entry.pos_text.as_bytes());
    }
    if entry.mode == SrdMode::Triplet
        && flags & FLAG_NEG_SAME == 0
        && let Some(neg_text) = &entry.neg_text
    {
        buf.extend_from_slice(neg_text.as_bytes());
    }

    buf
}

fn write_emb_slice(buf: &mut Vec<u8>, emb: &[f32]) {
    for &x in emb {
        buf.extend_from_slice(&x.to_le_bytes());
    }
}

// ---------------------------------------------------------------------------
// Unified decoder
// ---------------------------------------------------------------------------

/// Decode a raw simd-r-drive entry value into an [`SrdEntry`].
///
/// Format: `[mode:1] [flags:1] [text_lens...] [embeddings...] [texts...]`
/// Reader reconstructs full entry by duplicating anchor where flags indicate sameness.
pub fn decode_entry(data: &[u8], emb_dim: usize) -> Result<SrdEntry> {
    if data.len() < 2 {
        return Err(SrdError::EntryTooShort);
    }
    let mode = SrdMode::from_byte(data[0])?;
    let flags = data[1];
    let emb_bytes = emb_dim * 4;

    // Count unique texts/embeddings from flags
    let (unique_texts, pos_same, neg_same) = match mode {
        SrdMode::Pair => {
            let pos_same = flags & FLAG_POS_SAME != 0;
            (if pos_same { 1 } else { 2 }, pos_same, false)
        }
        SrdMode::Triplet => {
            let pos_same = flags & FLAG_POS_SAME != 0;
            let neg_same = flags & FLAG_NEG_SAME != 0;
            let n = 1 + if pos_same { 0 } else { 1 } + if neg_same { 0 } else { 1 };
            (n, pos_same, neg_same)
        }
    };

    // Read text lengths
    let mut offset = 2;
    let mut text_lens = Vec::with_capacity(unique_texts);
    for _ in 0..unique_texts {
        if offset + 4 > data.len() {
            return Err(SrdError::TruncatedTextLength);
        }
        let len = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        text_lens.push(len);
        offset += 4;
    }

    // Validate enough data for embeddings + texts
    let embs_size = unique_texts * emb_bytes;
    let texts_size: usize = text_lens.iter().sum();
    let expected = offset + embs_size + texts_size;
    if data.len() < expected {
        return Err(SrdError::TruncatedEntry {
            actual: data.len(),
            expected,
        });
    }

    // Read embeddings
    let mut embs = Vec::with_capacity(unique_texts);
    for _ in 0..unique_texts {
        embs.push(decode_emb_slice(&data[offset..offset + emb_bytes], emb_dim));
        offset += emb_bytes;
    }

    // Read texts
    let mut texts = Vec::with_capacity(unique_texts);
    for &len in &text_lens {
        let text = std::str::from_utf8(&data[offset..offset + len])?.to_owned();
        texts.push(text);
        offset += len;
    }

    // Reconstruct full entry
    let anchor_emb = embs[0].clone();
    let anchor_text = texts[0].clone();

    let (pos_emb, pos_text) = if pos_same {
        (anchor_emb.clone(), anchor_text.clone())
    } else {
        (embs[1].clone(), texts[1].clone())
    };

    let (neg_emb, neg_text) = match mode {
        SrdMode::Pair => (None, None),
        SrdMode::Triplet => {
            if neg_same {
                (Some(anchor_emb.clone()), Some(anchor_text.clone()))
            } else {
                let idx = 1 + if pos_same { 0 } else { 1 };
                (Some(embs[idx].clone()), Some(texts[idx].clone()))
            }
        }
    };

    Ok(SrdEntry {
        mode,
        same_as_anchor: pos_same,
        anchor_emb,
        anchor_text,
        pos_emb,
        pos_text,
        neg_emb,
        neg_text,
    })
}

fn decode_emb_slice(data: &[u8], emb_dim: usize) -> Vec<f32> {
    data[..emb_dim * 4]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

// ---------------------------------------------------------------------------
// Batch validation
// ---------------------------------------------------------------------------

/// Validate that a batch of vectors and texts are consistent before writing.
pub fn validate_write_batch(vecs: &[Vec<f32>], texts: &[&str]) -> Result<()> {
    if vecs.len() != texts.len() {
        return Err(SrdError::BatchLengthMismatch {
            vec_count: vecs.len(),
            text_count: texts.len(),
        });
    }
    if vecs.is_empty() {
        return Ok(());
    }
    let dim = vecs[0].len();
    if dim == 0 {
        return Err(SrdError::ZeroDimension);
    }
    for (i, vec) in vecs.iter().enumerate() {
        if vec.len() != dim {
            return Err(SrdError::InconsistentDimension {
                index: i,
                got: vec.len(),
                expected: dim,
            });
        }
        if let Some(j) = vec.iter().position(|x| !x.is_finite()) {
            return Err(SrdError::NonFiniteEmbedding {
                index: i,
                component: j,
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Batch write
// ---------------------------------------------------------------------------

/// Batch-write pair entries (anchor + positive) into `store` starting at `start_idx`.
///
/// Automatically sets `same_as_anchor` when anchor and positive are identical.
pub fn write_pair_entries(
    store: &DataStore,
    start_idx: u64,
    anchor_vecs: &[Vec<f32>],
    anchor_texts: &[&str],
    pos_vecs: &[Vec<f32>],
    pos_texts: &[&str],
) -> Result<()> {
    validate_write_batch(anchor_vecs, anchor_texts)?;
    validate_write_batch(pos_vecs, pos_texts)?;
    if anchor_vecs.len() != pos_vecs.len() {
        return Err(SrdError::PairLengthMismatch);
    }
    let mut key_bufs: Vec<[u8; 8]> = Vec::with_capacity(anchor_vecs.len());
    let mut value_bufs: Vec<Vec<u8>> = Vec::with_capacity(anchor_vecs.len());
    for (i, ((a_vec, a_text), (p_vec, p_text))) in anchor_vecs
        .iter()
        .zip(anchor_texts.iter())
        .zip(pos_vecs.iter().zip(pos_texts.iter()))
        .enumerate()
    {
        let same = a_vec == p_vec && a_text == p_text;
        key_bufs.push((start_idx + i as u64).to_le_bytes());
        value_bufs.push(encode_entry(&SrdEntry {
            mode: SrdMode::Pair,
            same_as_anchor: same,
            anchor_emb: a_vec.clone(),
            anchor_text: a_text.to_string(),
            pos_emb: p_vec.clone(),
            pos_text: p_text.to_string(),
            neg_emb: None,
            neg_text: None,
        }));
    }
    let entries: Vec<(&[u8], &[u8])> = key_bufs
        .iter()
        .zip(value_bufs.iter())
        .map(|(k, v)| (k.as_slice(), v.as_slice()))
        .collect();
    store.batch_write(&entries)?;
    Ok(())
}

/// Batch-write triplet entries (anchor + positive + negative) into `store` starting at `start_idx`.
#[allow(clippy::too_many_arguments)]
pub fn write_triplet_entries(
    store: &DataStore,
    start_idx: u64,
    anchor_vecs: &[Vec<f32>],
    anchor_texts: &[&str],
    pos_vecs: &[Vec<f32>],
    pos_texts: &[&str],
    neg_vecs: &[Vec<f32>],
    neg_texts: &[&str],
) -> Result<()> {
    validate_write_batch(anchor_vecs, anchor_texts)?;
    validate_write_batch(pos_vecs, pos_texts)?;
    validate_write_batch(neg_vecs, neg_texts)?;
    if anchor_vecs.len() != pos_vecs.len() || pos_vecs.len() != neg_vecs.len() {
        return Err(SrdError::TripletLengthMismatch);
    }
    let mut key_bufs: Vec<[u8; 8]> = Vec::with_capacity(anchor_vecs.len());
    let mut value_bufs: Vec<Vec<u8>> = Vec::with_capacity(anchor_vecs.len());
    for (i, (((a_vec, a_text), (p_vec, p_text)), (n_vec, n_text))) in anchor_vecs
        .iter()
        .zip(anchor_texts.iter())
        .zip(pos_vecs.iter().zip(pos_texts.iter()))
        .zip(neg_vecs.iter().zip(neg_texts.iter()))
        .enumerate()
    {
        let same = a_vec == p_vec && a_text == p_text;
        key_bufs.push((start_idx + i as u64).to_le_bytes());
        value_bufs.push(encode_entry(&SrdEntry {
            mode: SrdMode::Triplet,
            same_as_anchor: same,
            anchor_emb: a_vec.clone(),
            anchor_text: a_text.to_string(),
            pos_emb: p_vec.clone(),
            pos_text: p_text.to_string(),
            neg_emb: Some(n_vec.clone()),
            neg_text: Some(n_text.to_string()),
        }));
    }
    let entries: Vec<(&[u8], &[u8])> = key_bufs
        .iter()
        .zip(value_bufs.iter())
        .map(|(k, v)| (k.as_slice(), v.as_slice()))
        .collect();
    store.batch_write(&entries)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Batch read
// ---------------------------------------------------------------------------

/// Batch-read entries from `store` at the given indices and decode each one.
///
/// Missing entries (index beyond store length) are silently skipped.
pub fn batch_read_entries(
    store: &DataStore,
    indices: &[usize],
    emb_dim: usize,
) -> Result<Vec<SrdEntry>> {
    let key_bufs: Vec<[u8; 8]> = indices.iter().map(|&i| (i as u64).to_le_bytes()).collect();
    let keys: Vec<&[u8]> = key_bufs.iter().map(|k| k.as_slice()).collect();
    let raw = store.batch_read(&keys)?;
    let mut entries = Vec::with_capacity(indices.len());
    for entry in raw.into_iter().flatten() {
        entries.push(decode_entry(entry.as_slice(), emb_dim)?);
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // -----------------------------------------------------------------------
    // encode/decode roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn roundtrip_pair() {
        let a_emb = vec![0.1, 0.2, 0.3];
        let p_emb = vec![0.4, 0.5, 0.6];
        let a_text = "anchor text";
        let p_text = "positive text";

        let entry = SrdEntry {
            mode: SrdMode::Pair,
            same_as_anchor: false,
            anchor_emb: a_emb.clone(),
            anchor_text: a_text.to_string(),
            pos_emb: p_emb.clone(),
            pos_text: p_text.to_string(),
            neg_emb: None,
            neg_text: None,
        };
        let encoded = encode_entry(&entry);
        let decoded = decode_entry(&encoded, 3).unwrap();

        assert_eq!(decoded.mode, SrdMode::Pair);
        assert!(!decoded.same_as_anchor);
        assert_eq!(decoded.anchor_emb, a_emb);
        assert_eq!(decoded.anchor_text, a_text);
        assert_eq!(decoded.pos_emb, p_emb);
        assert_eq!(decoded.pos_text, p_text);
        assert!(decoded.neg_emb.is_none());
        assert!(decoded.neg_text.is_none());
    }

    #[test]
    fn roundtrip_pair_same_as_anchor() {
        let a_emb = vec![0.1, 0.2, 0.3];
        let a_text = "shared text";

        let entry = SrdEntry {
            mode: SrdMode::Pair,
            same_as_anchor: true,
            anchor_emb: a_emb.clone(),
            anchor_text: a_text.to_string(),
            pos_emb: a_emb.clone(),
            pos_text: a_text.to_string(),
            neg_emb: None,
            neg_text: None,
        };
        let encoded = encode_entry(&entry);
        let decoded = decode_entry(&encoded, 3).unwrap();

        assert_eq!(decoded.mode, SrdMode::Pair);
        assert!(decoded.same_as_anchor);
        assert_eq!(decoded.anchor_emb, a_emb);
        assert_eq!(decoded.anchor_text, a_text);
        assert_eq!(decoded.pos_emb, a_emb);
        assert_eq!(decoded.pos_text, a_text);
    }

    #[test]
    fn pair_same_is_smaller() {
        let a_emb = vec![0.1, 0.2, 0.3];
        let a_text = "shared text";

        let full = encode_entry(&SrdEntry {
            mode: SrdMode::Pair,
            same_as_anchor: false,
            anchor_emb: a_emb.clone(),
            anchor_text: a_text.to_string(),
            pos_emb: a_emb.clone(),
            pos_text: a_text.to_string(),
            neg_emb: None,
            neg_text: None,
        });
        let compact = encode_entry(&SrdEntry {
            mode: SrdMode::Pair,
            same_as_anchor: true,
            anchor_emb: a_emb.clone(),
            anchor_text: a_text.to_string(),
            pos_emb: a_emb,
            pos_text: a_text.to_string(),
            neg_emb: None,
            neg_text: None,
        });
        assert!(
            compact.len() < full.len(),
            "compact ({}) should be smaller than full ({})",
            compact.len(),
            full.len()
        );
    }

    #[test]
    fn roundtrip_triplet() {
        let a_emb = vec![0.1, 0.2, 0.3];
        let p_emb = vec![0.4, 0.5, 0.6];
        let n_emb = vec![0.7, 0.8, 0.9];
        let a_text = "anchor";
        let p_text = "positive";
        let n_text = "negative";

        let entry = SrdEntry {
            mode: SrdMode::Triplet,
            same_as_anchor: false,
            anchor_emb: a_emb.clone(),
            anchor_text: a_text.to_string(),
            pos_emb: p_emb.clone(),
            pos_text: p_text.to_string(),
            neg_emb: Some(n_emb.clone()),
            neg_text: Some(n_text.to_string()),
        };
        let encoded = encode_entry(&entry);
        let decoded = decode_entry(&encoded, 3).unwrap();

        assert_eq!(decoded.mode, SrdMode::Triplet);
        assert_eq!(decoded.anchor_emb, a_emb);
        assert_eq!(decoded.anchor_text, a_text);
        assert_eq!(decoded.pos_emb, p_emb);
        assert_eq!(decoded.pos_text, p_text);
        assert_eq!(decoded.neg_emb, Some(n_emb));
        assert_eq!(decoded.neg_text, Some(n_text.to_owned()));
    }

    #[test]
    fn roundtrip_triplet_pos_same_as_anchor() {
        let a_emb = vec![0.1, 0.2, 0.3];
        let n_emb = vec![0.7, 0.8, 0.9];
        let a_text = "anchor";
        let n_text = "negative";

        let entry = SrdEntry {
            mode: SrdMode::Triplet,
            same_as_anchor: true,
            anchor_emb: a_emb.clone(),
            anchor_text: a_text.to_string(),
            pos_emb: a_emb.clone(),
            pos_text: a_text.to_string(),
            neg_emb: Some(n_emb.clone()),
            neg_text: Some(n_text.to_string()),
        };
        let encoded = encode_entry(&entry);
        let decoded = decode_entry(&encoded, 3).unwrap();

        assert_eq!(decoded.mode, SrdMode::Triplet);
        assert!(decoded.same_as_anchor);
        assert_eq!(decoded.anchor_emb, a_emb);
        assert_eq!(decoded.anchor_text, a_text);
        assert_eq!(decoded.pos_emb, a_emb);
        assert_eq!(decoded.pos_text, a_text);
        assert_eq!(decoded.neg_emb, Some(n_emb));
        assert_eq!(decoded.neg_text, Some(n_text.to_owned()));
    }

    #[test]
    fn roundtrip_triplet_all_same() {
        let a_emb = vec![0.1, 0.2, 0.3];
        let a_text = "all same";

        let entry = SrdEntry {
            mode: SrdMode::Triplet,
            same_as_anchor: true,
            anchor_emb: a_emb.clone(),
            anchor_text: a_text.to_string(),
            pos_emb: a_emb.clone(),
            pos_text: a_text.to_string(),
            neg_emb: Some(a_emb.clone()),
            neg_text: Some(a_text.to_string()),
        };
        let encoded = encode_entry(&entry);
        let decoded = decode_entry(&encoded, 3).unwrap();

        assert_eq!(decoded.mode, SrdMode::Triplet);
        assert_eq!(decoded.anchor_emb, a_emb);
        assert_eq!(decoded.pos_emb, a_emb);
        assert_eq!(decoded.neg_emb, Some(a_emb));
    }

    #[test]
    fn roundtrip_triplet_neg_same_pos_different() {
        let a_emb = vec![0.1, 0.2, 0.3];
        let p_emb = vec![0.4, 0.5, 0.6];
        let n_emb = vec![0.1, 0.2, 0.3]; // same as anchor
        let a_text = "anchor";
        let p_text = "positive";
        let n_text = "anchor"; // same as anchor

        let entry = SrdEntry {
            mode: SrdMode::Triplet,
            same_as_anchor: false, // pos is different
            anchor_emb: a_emb.clone(),
            anchor_text: a_text.to_string(),
            pos_emb: p_emb.clone(),
            pos_text: p_text.to_string(),
            neg_emb: Some(n_emb.clone()),
            neg_text: Some(n_text.to_string()),
        };
        let encoded = encode_entry(&entry);
        let decoded = decode_entry(&encoded, 3).unwrap();

        assert_eq!(decoded.mode, SrdMode::Triplet);
        assert!(!decoded.same_as_anchor);
        assert_eq!(decoded.anchor_emb, a_emb);
        assert_eq!(decoded.anchor_text, a_text);
        assert_eq!(decoded.pos_emb, p_emb);
        assert_eq!(decoded.pos_text, p_text);
        assert_eq!(decoded.neg_emb, Some(a_emb)); // neg == anchor
        assert_eq!(decoded.neg_text, Some(a_text.to_owned())); // neg == anchor
    }

    #[test]
    fn pair_empty_text() {
        let entry = SrdEntry {
            mode: SrdMode::Pair,
            same_as_anchor: false,
            anchor_emb: vec![1.0],
            anchor_text: String::new(),
            pos_emb: vec![2.0],
            pos_text: String::new(),
            neg_emb: None,
            neg_text: None,
        };
        let encoded = encode_entry(&entry);
        let decoded = decode_entry(&encoded, 1).unwrap();
        assert_eq!(decoded.mode, SrdMode::Pair);
        assert_eq!(decoded.anchor_text, "");
        assert_eq!(decoded.pos_text, "");
    }

    #[test]
    fn triplet_unicode_text() {
        let entry = SrdEntry {
            mode: SrdMode::Triplet,
            same_as_anchor: false,
            anchor_emb: vec![1.0],
            anchor_text: "hello \u{1F600}".to_string(),
            pos_emb: vec![2.0],
            pos_text: "\u{00E9}\u{00E8}\u{00EA}".to_string(),
            neg_emb: Some(vec![3.0]),
            neg_text: Some("\u{4E16}\u{754C}".to_string()),
        };
        let encoded = encode_entry(&entry);
        let decoded = decode_entry(&encoded, 1).unwrap();
        assert_eq!(decoded.anchor_text, "hello \u{1F600}");
        assert_eq!(decoded.pos_text, "\u{00E9}\u{00E8}\u{00EA}");
        assert_eq!(decoded.neg_text.as_deref(), Some("\u{4E16}\u{754C}"));
    }

    #[test]
    fn decode_empty_returns_error() {
        assert!(decode_entry(&[], 3).is_err());
    }

    #[test]
    fn decode_bad_mode_returns_error() {
        assert!(decode_entry(&[99, 0], 3).is_err());
    }

    #[test]
    fn decode_truncated_returns_error() {
        assert!(decode_entry(&[0, 0, 0], 3).is_err());
    }

    // -----------------------------------------------------------------------
    // mode byte consistency
    // -----------------------------------------------------------------------

    #[test]
    fn pair_mode_byte_is_zero() {
        let entry = SrdEntry {
            mode: SrdMode::Pair,
            same_as_anchor: false,
            anchor_emb: vec![1.0],
            anchor_text: "a".into(),
            pos_emb: vec![2.0],
            pos_text: "b".into(),
            neg_emb: None,
            neg_text: None,
        };
        let encoded = encode_entry(&entry);
        assert_eq!(encoded[0], MODE_PAIR);
    }

    #[test]
    fn triplet_mode_byte_is_one() {
        let entry = SrdEntry {
            mode: SrdMode::Triplet,
            same_as_anchor: false,
            anchor_emb: vec![1.0],
            anchor_text: "a".into(),
            pos_emb: vec![2.0],
            pos_text: "b".into(),
            neg_emb: Some(vec![3.0]),
            neg_text: Some("c".into()),
        };
        let encoded = encode_entry(&entry);
        assert_eq!(encoded[0], MODE_TRIPLET);
    }

    #[test]
    fn mode_byte_consistency_pair() {
        let entry = SrdEntry {
            mode: SrdMode::Pair,
            same_as_anchor: false,
            anchor_emb: vec![1.0, 2.0],
            anchor_text: "hello".into(),
            pos_emb: vec![3.0, 4.0],
            pos_text: "world".into(),
            neg_emb: None,
            neg_text: None,
        };
        let encoded = encode_entry(&entry);
        let decoded = decode_entry(&encoded, 2).unwrap();
        assert_eq!(decoded.mode, SrdMode::Pair);
    }

    #[test]
    fn mode_byte_consistency_triplet() {
        let entry = SrdEntry {
            mode: SrdMode::Triplet,
            same_as_anchor: false,
            anchor_emb: vec![1.0],
            anchor_text: "a".into(),
            pos_emb: vec![2.0],
            pos_text: "b".into(),
            neg_emb: Some(vec![3.0]),
            neg_text: Some("c".into()),
        };
        let encoded = encode_entry(&entry);
        let decoded = decode_entry(&encoded, 1).unwrap();
        assert_eq!(decoded.mode, SrdMode::Triplet);
    }

    // -----------------------------------------------------------------------
    // large embedding dimension
    // -----------------------------------------------------------------------

    #[test]
    fn large_emb_dim_roundtrip() {
        let dim = 2048;
        let a_emb: Vec<f32> = (0..dim).map(|i| i as f32 * 0.001).collect();
        let p_emb: Vec<f32> = (0..dim).map(|i| i as f32 * -0.001).collect();
        let entry = SrdEntry {
            mode: SrdMode::Pair,
            same_as_anchor: false,
            anchor_emb: a_emb.clone(),
            anchor_text: "anchor".into(),
            pos_emb: p_emb.clone(),
            pos_text: "positive".into(),
            neg_emb: None,
            neg_text: None,
        };
        let encoded = encode_entry(&entry);
        let decoded = decode_entry(&encoded, dim).unwrap();
        assert_eq!(decoded.anchor_emb, a_emb);
        assert_eq!(decoded.pos_emb, p_emb);
    }

    // -----------------------------------------------------------------------
    // error: truncated text
    // -----------------------------------------------------------------------

    #[test]
    fn truncated_text_returns_error() {
        let entry = SrdEntry {
            mode: SrdMode::Pair,
            same_as_anchor: false,
            anchor_emb: vec![1.0],
            anchor_text: "hello".into(),
            pos_emb: vec![2.0],
            pos_text: "world".into(),
            neg_emb: None,
            neg_text: None,
        };
        let buf = encode_entry(&entry);
        let truncated = &buf[..buf.len() - 2];
        assert!(decode_entry(truncated, 1).is_err());
    }

    // -----------------------------------------------------------------------
    // batch write + batch_read roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn batch_write_read_pair_roundtrip() {
        let dir = TempDir::new().unwrap();
        let store = DataStore::open(&dir.path().join("test.srd")).unwrap();

        let n = 100;
        let all_a_emb: Vec<Vec<f32>> = (0..n).map(|i| vec![i as f32, i as f32 + 0.5]).collect();
        let all_a_text: Vec<String> = (0..n).map(|i| format!("anchor_{i}")).collect();
        let all_p_emb: Vec<Vec<f32>> = (0..n)
            .map(|i| vec![i as f32 + 100.0, i as f32 + 100.5])
            .collect();
        let all_p_text: Vec<String> = (0..n).map(|i| format!("positive_{i}")).collect();

        let a_refs: Vec<&str> = all_a_text.iter().map(|s| s.as_str()).collect();
        let p_refs: Vec<&str> = all_p_text.iter().map(|s| s.as_str()).collect();

        write_pair_entries(&store, 0, &all_a_emb, &a_refs, &all_p_emb, &p_refs).unwrap();
        assert_eq!(store.len().unwrap(), n);

        let indices: Vec<usize> = (0..n).collect();
        let entries = batch_read_entries(&store, &indices, 2).unwrap();
        assert_eq!(entries.len(), n);

        for (i, entry) in entries.iter().enumerate() {
            assert_eq!(entry.mode, SrdMode::Pair);
            assert_eq!(entry.anchor_emb, all_a_emb[i]);
            assert_eq!(entry.anchor_text, all_a_text[i]);
            assert_eq!(entry.pos_emb, all_p_emb[i]);
            assert_eq!(entry.pos_text, all_p_text[i]);
        }
    }

    #[test]
    fn batch_write_read_triplet_roundtrip() {
        let dir = TempDir::new().unwrap();
        let store = DataStore::open(&dir.path().join("test.srd")).unwrap();

        let n = 50;
        let dim = 4;
        let a_emb: Vec<Vec<f32>> = (0..n).map(|i| vec![i as f32; dim]).collect();
        let a_text: Vec<String> = (0..n).map(|i| format!("a{i}")).collect();
        let p_emb: Vec<Vec<f32>> = (0..n).map(|i| vec![i as f32 + 100.0; dim]).collect();
        let p_text: Vec<String> = (0..n).map(|i| format!("p{i}")).collect();
        let n_emb: Vec<Vec<f32>> = (0..n).map(|i| vec![i as f32 + 200.0; dim]).collect();
        let n_text: Vec<String> = (0..n).map(|i| format!("n{i}")).collect();

        let a_refs: Vec<&str> = a_text.iter().map(|s| s.as_str()).collect();
        let p_refs: Vec<&str> = p_text.iter().map(|s| s.as_str()).collect();
        let n_refs: Vec<&str> = n_text.iter().map(|s| s.as_str()).collect();

        write_triplet_entries(&store, 0, &a_emb, &a_refs, &p_emb, &p_refs, &n_emb, &n_refs)
            .unwrap();
        assert_eq!(store.len().unwrap(), n);

        let indices: Vec<usize> = (0..n).collect();
        let entries = batch_read_entries(&store, &indices, dim).unwrap();
        assert_eq!(entries.len(), n);

        for (i, entry) in entries.iter().enumerate() {
            assert_eq!(entry.mode, SrdMode::Triplet);
            assert_eq!(entry.anchor_emb, a_emb[i]);
            assert_eq!(entry.anchor_text, a_text[i]);
            assert_eq!(entry.pos_emb, p_emb[i]);
            assert_eq!(entry.pos_text, p_text[i]);
            assert_eq!(entry.neg_emb.as_ref().unwrap(), &n_emb[i]);
            assert_eq!(entry.neg_text.as_deref(), Some(n_text[i].as_str()));
        }
    }

    // -----------------------------------------------------------------------
    // batch read subset / edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn batch_read_subset() {
        let dir = TempDir::new().unwrap();
        let store = DataStore::open(&dir.path().join("test.srd")).unwrap();

        let vecs: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32]).collect();
        let texts: Vec<String> = (0..10).map(|i| format!("t{i}")).collect();
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        write_pair_entries(&store, 0, &vecs, &refs, &vecs, &refs).unwrap();

        let entries = batch_read_entries(&store, &[0, 5, 9], 1).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].anchor_text, "t0");
        assert_eq!(entries[1].anchor_text, "t5");
        assert_eq!(entries[2].anchor_text, "t9");
    }

    #[test]
    fn batch_read_empty_indices() {
        let dir = TempDir::new().unwrap();
        let store = DataStore::open(&dir.path().join("test.srd")).unwrap();
        let entries = batch_read_entries(&store, &[], 1).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn batch_read_missing_index_skipped() {
        let dir = TempDir::new().unwrap();
        let store = DataStore::open(&dir.path().join("test.srd")).unwrap();

        let vecs = vec![vec![1.0], vec![2.0]];
        let texts = vec!["a", "b"];
        write_pair_entries(&store, 0, &vecs, &texts, &vecs, &texts).unwrap();

        // Index 99 doesn't exist — should be skipped
        let entries = batch_read_entries(&store, &[0, 99], 1).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].anchor_text, "a");
    }

    // -----------------------------------------------------------------------
    // validation
    // -----------------------------------------------------------------------

    #[test]
    fn validate_write_batch_ok() {
        let vecs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let texts = vec!["hello", "world"];
        assert!(validate_write_batch(&vecs, &texts).is_ok());
    }

    #[test]
    fn validate_write_batch_empty() {
        let vecs: Vec<Vec<f32>> = vec![];
        let texts: Vec<&str> = vec![];
        assert!(validate_write_batch(&vecs, &texts).is_ok());
    }

    #[test]
    fn validate_write_batch_len_mismatch() {
        let vecs = vec![vec![1.0], vec![2.0]];
        let texts = vec!["hello"];
        assert!(validate_write_batch(&vecs, &texts).is_err());
    }

    #[test]
    fn validate_write_batch_non_finite() {
        let vecs = vec![vec![1.0, f32::NAN]];
        let texts = vec!["hello"];
        assert!(validate_write_batch(&vecs, &texts).is_err());
    }

    #[test]
    fn validate_write_batch_inconsistent_dim() {
        let vecs = vec![vec![1.0, 2.0], vec![3.0]];
        let texts = vec!["hello", "world"];
        assert!(validate_write_batch(&vecs, &texts).is_err());
    }

    #[test]
    fn validate_write_batch_zero_dim() {
        let vecs = vec![vec![]];
        let texts = vec!["hello"];
        assert!(validate_write_batch(&vecs, &texts).is_err());
    }

    // -----------------------------------------------------------------------
    // write_pair_entries / write_triplet_entries validation
    // -----------------------------------------------------------------------

    #[test]
    fn write_pair_entries_rejects_len_mismatch() {
        let dir = TempDir::new().unwrap();
        let store = DataStore::open(&dir.path().join("test.srd")).unwrap();
        let a_vecs = vec![vec![1.0], vec![2.0]];
        let a_texts = vec!["a"];
        let p_vecs = vec![vec![3.0]];
        let p_texts = vec!["p"];
        assert!(write_pair_entries(&store, 0, &a_vecs, &a_texts, &p_vecs, &p_texts).is_err());
    }

    #[test]
    fn write_triplet_entries_rejects_len_mismatch() {
        let dir = TempDir::new().unwrap();
        let store = DataStore::open(&dir.path().join("test.srd")).unwrap();
        let a_vecs = vec![vec![1.0]];
        let a_texts = vec!["a"];
        let p_vecs = vec![vec![2.0], vec![3.0]];
        let p_texts = vec!["p", "q"];
        let n_vecs = vec![vec![4.0]];
        let n_texts = vec!["n"];
        assert!(
            write_triplet_entries(
                &store, 0, &a_vecs, &a_texts, &p_vecs, &p_texts, &n_vecs, &n_texts
            )
            .is_err()
        );
    }

    #[test]
    fn write_pair_entries_correct_start_idx() {
        let dir = TempDir::new().unwrap();
        let store = DataStore::open(&dir.path().join("test.srd")).unwrap();

        // Write first batch at idx 0
        let vecs1 = vec![vec![1.0], vec![2.0]];
        let texts1 = vec!["a", "b"];
        write_pair_entries(&store, 0, &vecs1, &texts1, &vecs1, &texts1).unwrap();

        // Write second batch at idx 2
        let vecs2 = vec![vec![3.0]];
        let texts2 = vec!["c"];
        write_pair_entries(&store, 2, &vecs2, &texts2, &vecs2, &texts2).unwrap();

        assert_eq!(store.len().unwrap(), 3);

        let entries = batch_read_entries(&store, &[0, 1, 2], 1).unwrap();
        assert_eq!(entries[0].anchor_text, "a");
        assert_eq!(entries[1].anchor_text, "b");
        assert_eq!(entries[2].anchor_text, "c");
    }

    // -----------------------------------------------------------------------
    // start_idx with non-zero offset
    // -----------------------------------------------------------------------

    #[test]
    fn batch_write_nonzero_start_idx() {
        let dir = TempDir::new().unwrap();
        let store = DataStore::open(&dir.path().join("test.srd")).unwrap();

        let vecs = vec![vec![1.0, 2.0]];
        let texts = vec!["first"];
        write_pair_entries(&store, 5, &vecs, &texts, &vecs, &texts).unwrap();

        let entries = batch_read_entries(&store, &[5], 2).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].anchor_text, "first");
    }

    // -----------------------------------------------------------------------
    // cross-batch read safety
    // -----------------------------------------------------------------------

    #[test]
    fn cross_batch_read_safety_pair() {
        let dir = TempDir::new().unwrap();
        let store = DataStore::open(&dir.path().join("test.srd")).unwrap();

        // Batch A: entry 0 same_as_anchor=true, entry 1 different
        let a_vecs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let p_vecs = vec![vec![1.0, 2.0], vec![30.0, 40.0]]; // different emb for diff entry
        let a_texts = vec!["same", "diff_a"];
        let p_texts = vec!["same", "diff_p"];
        write_pair_entries(&store, 0, &a_vecs, &a_texts, &p_vecs, &p_texts).unwrap();

        // Batch B: entry 2 same_as_anchor=false, entry 3 same_as_anchor=true
        let b_vecs = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let b_texts = vec!["batch_b_0", "batch_b_1"];
        write_pair_entries(&store, 2, &b_vecs, &b_texts, &b_vecs, &b_texts).unwrap();

        assert_eq!(store.len().unwrap(), 4);

        // Read all entries from both batches
        let entries = batch_read_entries(&store, &[0, 1, 2, 3], 2).unwrap();
        assert_eq!(entries.len(), 4);

        // Entry 0: same_as_anchor=true
        assert!(entries[0].same_as_anchor);
        assert_eq!(entries[0].anchor_text, "same");
        assert_eq!(entries[0].pos_text, "same");
        assert_eq!(entries[0].anchor_emb, entries[0].pos_emb);

        // Entry 1: different anchor/pos — verify embedding attribution is correct
        assert!(!entries[1].same_as_anchor);
        assert_eq!(entries[1].anchor_text, "diff_a");
        assert_eq!(entries[1].pos_text, "diff_p");
        assert_eq!(
            entries[1].anchor_emb,
            vec![3.0, 4.0],
            "anchor embedding should match anchor vec"
        );
        assert_eq!(
            entries[1].pos_emb,
            vec![30.0, 40.0],
            "positive embedding should match positive vec"
        );

        // Entry 2: same anchor/pos (write_pair_entries auto-detects)
        assert!(entries[2].same_as_anchor);
        assert_eq!(entries[2].anchor_text, "batch_b_0");

        // Entry 3: same_as_anchor=true
        assert!(entries[3].same_as_anchor);
        assert_eq!(entries[3].anchor_text, "batch_b_1");
        assert_eq!(entries[3].pos_text, "batch_b_1");
    }

    #[test]
    fn cross_batch_read_safety_triplet() {
        let dir = TempDir::new().unwrap();
        let store = DataStore::open(&dir.path().join("test.srd")).unwrap();

        // Batch A: entry 0 all different (each role gets its own embedding),
        // entry 1 pos same as anchor (same emb + text as anchor).
        let a_a_vecs = vec![vec![1.0], vec![2.0]];
        let a_p_vecs = vec![vec![10.0], vec![2.0]]; // entry 0: distinct pos emb
        let a_n_vecs = vec![vec![100.0], vec![200.0]]; // entry 0/1: distinct neg emb
        let a_a_texts = vec!["a0", "a1"];
        let a_p_texts = vec!["p0", "a1"]; // entry 1 pos same as anchor
        let a_n_texts = vec!["n0", "n1"];
        write_triplet_entries(
            &store, 0, &a_a_vecs, &a_a_texts, &a_p_vecs, &a_p_texts, &a_n_vecs, &a_n_texts,
        )
        .unwrap();

        // Batch B: entry 2 all same
        let b_a_vecs = vec![vec![3.0]];
        let b_a_texts = vec!["all_same"];
        let b_p_texts = vec!["all_same"];
        let b_n_texts = vec!["all_same"];
        write_triplet_entries(
            &store, 2, &b_a_vecs, &b_a_texts, &b_a_vecs, &b_p_texts, &b_a_vecs, &b_n_texts,
        )
        .unwrap();

        assert_eq!(store.len().unwrap(), 3);

        let entries = batch_read_entries(&store, &[0, 1, 2], 1).unwrap();
        assert_eq!(entries.len(), 3);

        // Entry 0: all different — verify each embedding matches its role
        assert!(!entries[0].same_as_anchor);
        assert_eq!(entries[0].anchor_text, "a0");
        assert_eq!(entries[0].anchor_emb, vec![1.0]);
        assert_eq!(entries[0].pos_text, "p0");
        assert_eq!(
            entries[0].pos_emb,
            vec![10.0],
            "pos emb should be distinct from anchor"
        );
        assert_eq!(entries[0].neg_text.as_deref(), Some("n0"));
        assert_eq!(
            entries[0].neg_emb.as_ref(),
            Some(&vec![100.0]),
            "neg emb should be distinct from anchor and pos"
        );

        // Entry 1: pos same as anchor
        assert!(entries[1].same_as_anchor);
        assert_eq!(entries[1].anchor_text, "a1");
        assert_eq!(entries[1].pos_text, "a1");
        assert_eq!(entries[1].neg_text.as_deref(), Some("n1"));

        // Entry 2: all same
        assert!(entries[2].same_as_anchor);
        assert_eq!(entries[2].anchor_text, "all_same");
        assert_eq!(entries[2].pos_text, "all_same");
        assert_eq!(entries[2].neg_text.as_deref(), Some("all_same"));
        assert_eq!(entries[2].anchor_emb, entries[2].pos_emb);
        assert_eq!(
            entries[2].anchor_emb,
            entries[2].neg_emb.as_ref().unwrap().clone()
        );
    }
}
