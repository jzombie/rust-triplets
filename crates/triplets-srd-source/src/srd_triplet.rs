use simd_r_drive::storage_engine::DataStore;
use simd_r_drive::storage_engine::traits::{DataStoreReader, DataStoreWriter};

use triplets_core::data::PairLabel;

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
/// Label-negative flag: bit 2 indicates this pair is negative (pair mode only).
pub const FLAG_LABEL_NEGATIVE: u8 = 1 << 2;

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

/// A decoded pair-mode entry from SRD storage.
#[derive(Clone, Debug)]
pub struct SrdPairRecord {
    /// Anchor embedding vector.
    pub anchor_emb: Vec<f32>,
    /// Anchor text string.
    pub anchor_text: String,
    /// Candidate embedding vector (positive or negative).
    pub candidate_emb: Vec<f32>,
    /// Candidate text string (positive or negative).
    pub candidate_text: String,
    /// Pair label (Positive or Negative).
    pub label: PairLabel,
}

/// A decoded triplet-mode entry from SRD storage.
#[derive(Clone, Debug)]
pub struct SrdTripletRecord {
    /// Anchor embedding vector.
    pub anchor_emb: Vec<f32>,
    /// Anchor text string.
    pub anchor_text: String,
    /// Positive embedding vector.
    pub pos_emb: Vec<f32>,
    /// Positive text string.
    pub pos_text: String,
    /// Negative embedding vector.
    pub neg_emb: Vec<f32>,
    /// Negative text string.
    pub neg_text: String,
}

/// A decoded entry from SRD storage.
#[derive(Clone, Debug)]
pub enum SrdRecord {
    /// A pair-mode entry.
    Pair(SrdPairRecord),
    /// A triplet-mode entry.
    Triplet(SrdTripletRecord),
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
pub fn encode_entry(record: &SrdRecord) -> Vec<u8> {
    match record {
        SrdRecord::Pair(pair) => {
            let mut buf = vec![SrdMode::Pair.to_byte()];
            let emb_dim = pair.anchor_emb.len();
            let emb_bytes = emb_dim * 4;

            let mut flags: u8 = 0;
            if pair.anchor_emb == pair.candidate_emb && pair.anchor_text == pair.candidate_text {
                flags |= FLAG_POS_SAME;
            }
            if pair.label == PairLabel::Negative {
                flags |= FLAG_LABEL_NEGATIVE;
            }
            buf.push(flags);

            let unique_texts = if flags & FLAG_POS_SAME != 0 { 1 } else { 2 };
            let text_lens_size = unique_texts * 4;
            let embs_size = unique_texts * emb_bytes;
            let texts_size = if flags & FLAG_POS_SAME != 0 {
                pair.anchor_text.len()
            } else {
                pair.anchor_text.len() + pair.candidate_text.len()
            };

            let mut inner = Vec::with_capacity(text_lens_size + embs_size + texts_size);
            inner.extend_from_slice(&(pair.anchor_text.len() as u32).to_le_bytes());
            if flags & FLAG_POS_SAME == 0 {
                inner.extend_from_slice(&(pair.candidate_text.len() as u32).to_le_bytes());
            }
            write_emb_slice(&mut inner, &pair.anchor_emb);
            if flags & FLAG_POS_SAME == 0 {
                write_emb_slice(&mut inner, &pair.candidate_emb);
            }
            inner.extend_from_slice(pair.anchor_text.as_bytes());
            if flags & FLAG_POS_SAME == 0 {
                inner.extend_from_slice(pair.candidate_text.as_bytes());
            }

            buf.extend(inner);
            buf
        }
        SrdRecord::Triplet(triplet) => {
            let mut buf = vec![SrdMode::Triplet.to_byte()];
            let emb_dim = triplet.anchor_emb.len();
            let emb_bytes = emb_dim * 4;

            let mut flags: u8 = 0;
            if triplet.pos_emb == triplet.anchor_emb && triplet.pos_text == triplet.anchor_text {
                flags |= FLAG_POS_SAME;
            }
            if triplet.neg_emb == triplet.anchor_emb && triplet.neg_text == triplet.anchor_text {
                flags |= FLAG_NEG_SAME;
            }
            buf.push(flags);

            let mut unique_texts = 1;
            if flags & FLAG_POS_SAME == 0 {
                unique_texts += 1;
            }
            if flags & FLAG_NEG_SAME == 0 {
                unique_texts += 1;
            }
            let text_lens_size = unique_texts * 4;
            let embs_size = unique_texts * emb_bytes;
            let mut texts_size = triplet.anchor_text.len();
            if flags & FLAG_POS_SAME == 0 {
                texts_size += triplet.pos_text.len();
            }
            if flags & FLAG_NEG_SAME == 0 {
                texts_size += triplet.neg_text.len();
            }

            let mut inner = Vec::with_capacity(text_lens_size + embs_size + texts_size);
            inner.extend_from_slice(&(triplet.anchor_text.len() as u32).to_le_bytes());
            if flags & FLAG_POS_SAME == 0 {
                inner.extend_from_slice(&(triplet.pos_text.len() as u32).to_le_bytes());
            }
            if flags & FLAG_NEG_SAME == 0 {
                inner.extend_from_slice(&(triplet.neg_text.len() as u32).to_le_bytes());
            }
            write_emb_slice(&mut inner, &triplet.anchor_emb);
            if flags & FLAG_POS_SAME == 0 {
                write_emb_slice(&mut inner, &triplet.pos_emb);
            }
            if flags & FLAG_NEG_SAME == 0 {
                write_emb_slice(&mut inner, &triplet.neg_emb);
            }
            inner.extend_from_slice(triplet.anchor_text.as_bytes());
            if flags & FLAG_POS_SAME == 0 {
                inner.extend_from_slice(triplet.pos_text.as_bytes());
            }
            if flags & FLAG_NEG_SAME == 0 {
                inner.extend_from_slice(triplet.neg_text.as_bytes());
            }

            buf.extend(inner);
            buf
        }
    }
}

fn write_emb_slice(buf: &mut Vec<u8>, emb: &[f32]) {
    for &x in emb {
        buf.extend_from_slice(&x.to_le_bytes());
    }
}

// ---------------------------------------------------------------------------
// Unified decoder
// ---------------------------------------------------------------------------

/// Decode a raw simd-r-drive entry value into an [`SrdRecord`].
///
/// Format: `[mode:1] [flags:1] [text_lens...] [embeddings...] [texts...]`
/// Reader reconstructs full entry by duplicating anchor where flags indicate sameness.
pub fn decode_entry(data: &[u8], emb_dim: usize) -> Result<SrdRecord> {
    if data.is_empty() {
        return Err(SrdError::EntryTooShort);
    }
    let mode = SrdMode::from_byte(data[0])?;
    if data.len() < 2 {
        return Err(SrdError::EntryTooShort);
    }
    let flags = data[1];
    let emb_bytes = emb_dim * 4;

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

    let embs_size = unique_texts * emb_bytes;
    let texts_size: usize = text_lens.iter().sum();
    let expected = offset + embs_size + texts_size;
    if data.len() < expected {
        return Err(SrdError::TruncatedEntry {
            actual: data.len(),
            expected,
        });
    }

    let mut embs = Vec::with_capacity(unique_texts);
    for _ in 0..unique_texts {
        embs.push(decode_emb_slice(&data[offset..offset + emb_bytes], emb_dim));
        offset += emb_bytes;
    }

    let mut texts = Vec::with_capacity(unique_texts);
    for &len in &text_lens {
        let text = std::str::from_utf8(&data[offset..offset + len])?.to_owned();
        texts.push(text);
        offset += len;
    }

    let anchor_emb = embs[0].clone();
    let anchor_text = texts[0].clone();

    match mode {
        SrdMode::Pair => {
            let (candidate_emb, candidate_text) = if pos_same {
                (anchor_emb.clone(), anchor_text.clone())
            } else {
                (embs[1].clone(), texts[1].clone())
            };
            let label = if flags & FLAG_LABEL_NEGATIVE != 0 {
                PairLabel::Negative
            } else {
                PairLabel::Positive
            };
            Ok(SrdRecord::Pair(SrdPairRecord {
                anchor_emb,
                anchor_text,
                candidate_emb,
                candidate_text,
                label,
            }))
        }
        SrdMode::Triplet => {
            let (pos_emb, pos_text) = if pos_same {
                (anchor_emb.clone(), anchor_text.clone())
            } else {
                (embs[1].clone(), texts[1].clone())
            };
            let (neg_emb, neg_text) = if neg_same {
                (anchor_emb.clone(), anchor_text.clone())
            } else {
                let idx = 1 + if pos_same { 0 } else { 1 };
                (embs[idx].clone(), texts[idx].clone())
            };
            Ok(SrdRecord::Triplet(SrdTripletRecord {
                anchor_emb,
                anchor_text,
                pos_emb,
                pos_text,
                neg_emb,
                neg_text,
            }))
        }
    }
}

fn decode_emb_slice(data: &[u8], emb_dim: usize) -> Vec<f32> {
    data[..emb_dim * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

// ---------------------------------------------------------------------------
// Batch validation
// ---------------------------------------------------------------------------

/// Validate a batch of `(&[f32], &str)` pairs are consistent before writing.
/// Accepts an iterator so it works with any memory layout (SoA or AoS).
pub fn validate_write_batch<'a, I>(iter: I) -> Result<()>
where
    I: Iterator<Item = (&'a [f32], &'a str)> + Clone,
{
    let mut peek_iter = iter.clone();
    let dim = match peek_iter.next() {
        Some((vec, _)) => vec.len(),
        None => return Ok(()),
    };
    if dim == 0 {
        return Err(SrdError::ZeroDimension);
    }
    for (i, (vec, _text)) in iter.enumerate() {
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

/// A single pair entry for batch writing.
#[derive(Debug, Clone, Copy)]
pub struct SrdPairWriteEntry<'a> {
    /// Anchor embedding vector.
    pub anchor_vec: &'a [f32],
    /// Anchor text string.
    pub anchor_text: &'a str,
    /// Candidate embedding vector (may be positive or negative).
    pub candidate_vec: &'a [f32],
    /// Candidate text string (may be positive or negative).
    pub candidate_text: &'a str,
    /// Label for this pair entry.
    pub label: &'a PairLabel,
}

/// Batch-write pair entries into `store` starting at `start_idx`.
///
/// Each entry carries its anchor, candidate, and label in a single struct.
/// Automatically sets `same_as_anchor` when anchor and candidate are identical.
pub fn write_pair_entries(
    store: &DataStore,
    start_idx: u64,
    entries: &[SrdPairWriteEntry],
) -> Result<()> {
    if entries.is_empty() {
        return Ok(());
    }
    validate_write_batch(entries.iter().map(|e| (e.anchor_vec, e.anchor_text)))?;
    validate_write_batch(entries.iter().map(|e| (e.candidate_vec, e.candidate_text)))?;

    let mut key_bufs: Vec<[u8; 8]> = Vec::with_capacity(entries.len());
    let mut value_bufs: Vec<Vec<u8>> = Vec::with_capacity(entries.len());
    for (i, entry) in entries.iter().enumerate() {
        key_bufs.push((start_idx + i as u64).to_le_bytes());
        value_bufs.push(encode_entry(&SrdRecord::Pair(SrdPairRecord {
            anchor_emb: entry.anchor_vec.to_vec(),
            anchor_text: entry.anchor_text.to_string(),
            candidate_emb: entry.candidate_vec.to_vec(),
            candidate_text: entry.candidate_text.to_string(),
            label: entry.label.clone(),
        })));
    }
    let entries: Vec<(&[u8], &[u8])> = key_bufs
        .iter()
        .zip(value_bufs.iter())
        .map(|(k, v)| (k.as_slice(), v.as_slice()))
        .collect();
    store.batch_write(&entries)?;
    Ok(())
}

/// A single triplet entry for batch writing.
#[derive(Debug, Clone, Copy)]
pub struct SrdTripletWriteEntry<'a> {
    /// Anchor embedding vector.
    pub anchor_vec: &'a [f32],
    /// Anchor text string.
    pub anchor_text: &'a str,
    /// Positive embedding vector.
    pub pos_vec: &'a [f32],
    /// Positive text string.
    pub pos_text: &'a str,
    /// Negative embedding vector.
    pub neg_vec: &'a [f32],
    /// Negative text string.
    pub neg_text: &'a str,
}

/// Batch-write triplet entries into `store` starting at `start_idx`.
///
/// Each entry carries anchor, positive, and negative data in a single struct.
pub fn write_triplet_entries(
    store: &DataStore,
    start_idx: u64,
    entries: &[SrdTripletWriteEntry],
) -> Result<()> {
    if entries.is_empty() {
        return Ok(());
    }
    validate_write_batch(entries.iter().map(|e| (e.anchor_vec, e.anchor_text)))?;
    validate_write_batch(entries.iter().map(|e| (e.pos_vec, e.pos_text)))?;
    validate_write_batch(entries.iter().map(|e| (e.neg_vec, e.neg_text)))?;

    let mut key_bufs: Vec<[u8; 8]> = Vec::with_capacity(entries.len());
    let mut value_bufs: Vec<Vec<u8>> = Vec::with_capacity(entries.len());
    for (i, entry) in entries.iter().enumerate() {
        key_bufs.push((start_idx + i as u64).to_le_bytes());
        value_bufs.push(encode_entry(&SrdRecord::Triplet(SrdTripletRecord {
            anchor_emb: entry.anchor_vec.to_vec(),
            anchor_text: entry.anchor_text.to_string(),
            pos_emb: entry.pos_vec.to_vec(),
            pos_text: entry.pos_text.to_string(),
            neg_emb: entry.neg_vec.to_vec(),
            neg_text: entry.neg_text.to_string(),
        })));
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
) -> Result<Vec<SrdRecord>> {
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

    /// Helper to create an SrdPairWriteEntry slice for tests.
    fn make_pair_entries<'a>(
        a_vecs: &'a [Vec<f32>],
        a_texts: &'a [&'a str],
        p_vecs: &'a [Vec<f32>],
        p_texts: &'a [&'a str],
    ) -> Vec<SrdPairWriteEntry<'a>> {
        a_vecs
            .iter()
            .zip(a_texts.iter())
            .zip(p_vecs.iter().zip(p_texts.iter()))
            .map(|((av, at), (pv, pt))| SrdPairWriteEntry {
                anchor_vec: av,
                anchor_text: at,
                candidate_vec: pv,
                candidate_text: pt,
                label: &PairLabel::Positive,
            })
            .collect()
    }

    /// Helper to create an SrdTripletWriteEntry slice for tests.
    fn make_triplet_entries<'a>(
        a_vecs: &'a [Vec<f32>],
        a_texts: &'a [&'a str],
        p_vecs: &'a [Vec<f32>],
        p_texts: &'a [&'a str],
        n_vecs: &'a [Vec<f32>],
        n_texts: &'a [&'a str],
    ) -> Vec<SrdTripletWriteEntry<'a>> {
        a_vecs
            .iter()
            .zip(a_texts.iter())
            .zip(p_vecs.iter().zip(p_texts.iter()))
            .zip(n_vecs.iter().zip(n_texts.iter()))
            .map(|(((av, at), (pv, pt)), (nv, nt))| SrdTripletWriteEntry {
                anchor_vec: av,
                anchor_text: at,
                pos_vec: pv,
                pos_text: pt,
                neg_vec: nv,
                neg_text: nt,
            })
            .collect()
    }
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

        let record = SrdRecord::Pair(SrdPairRecord {
            anchor_emb: a_emb.clone(),
            anchor_text: a_text.to_string(),
            candidate_emb: p_emb.clone(),
            candidate_text: p_text.to_string(),
            label: PairLabel::Positive,
        });
        let encoded = encode_entry(&record);
        let decoded = decode_entry(&encoded, 3).unwrap();

        assert!(matches!(decoded, SrdRecord::Pair(_)));
        if let SrdRecord::Pair(p) = decoded {
            assert_eq!(p.anchor_emb, a_emb);
            assert_eq!(p.anchor_text, a_text);
            assert_eq!(p.candidate_emb, p_emb);
            assert_eq!(p.candidate_text, p_text);
            assert_eq!(p.label, PairLabel::Positive);
        }
    }

    #[test]
    fn roundtrip_pair_same_as_anchor() {
        let a_emb = vec![0.1, 0.2, 0.3];
        let a_text = "shared text";

        let record = SrdRecord::Pair(SrdPairRecord {
            anchor_emb: a_emb.clone(),
            anchor_text: a_text.to_string(),
            candidate_emb: a_emb.clone(),
            candidate_text: a_text.to_string(),
            label: PairLabel::Positive,
        });
        let encoded = encode_entry(&record);
        let decoded = decode_entry(&encoded, 3).unwrap();

        assert!(matches!(decoded, SrdRecord::Pair(_)));
        if let SrdRecord::Pair(p) = decoded {
            assert_eq!(p.anchor_emb, a_emb);
            assert_eq!(p.anchor_text, a_text);
            assert_eq!(p.candidate_emb, a_emb);
            assert_eq!(p.candidate_text, a_text);
        }
    }

    #[test]
    fn pair_same_is_smaller() {
        let a_emb = vec![0.1, 0.2, 0.3];
        let a_text = "some shared text"; // Purposely longer than `different text`
        let c_emb = vec![0.4, 0.5, 0.6];
        let c_text = "different text";

        let distinct = encode_entry(&SrdRecord::Pair(SrdPairRecord {
            anchor_emb: a_emb.clone(),
            anchor_text: a_text.to_string(),
            candidate_emb: c_emb,
            candidate_text: c_text.to_string(),
            label: PairLabel::Positive,
        }));
        let same = encode_entry(&SrdRecord::Pair(SrdPairRecord {
            anchor_emb: a_emb.clone(),
            anchor_text: a_text.to_string(),
            candidate_emb: a_emb,
            candidate_text: a_text.to_string(),
            label: PairLabel::Positive,
        }));
        // When anchor == candidate, FLAG_POS_SAME is set and less data is stored.
        assert!(same.len() < distinct.len());
    }

    #[test]
    fn pair_same_sets_flag_byte() {
        let a_emb = vec![0.1, 0.2, 0.3];
        let a_text = "shared text";

        let encoded = encode_entry(&SrdRecord::Pair(SrdPairRecord {
            anchor_emb: a_emb.clone(),
            anchor_text: a_text.to_string(),
            candidate_emb: a_emb,
            candidate_text: a_text.to_string(),
            label: PairLabel::Positive,
        }));
        // Byte 0 = mode (0), byte 1 = flags. FLAG_POS_SAME = 0x01.
        assert_eq!(encoded[0], MODE_PAIR);
        assert_eq!(encoded[1], FLAG_POS_SAME);
    }

    #[test]
    fn pair_distinct_does_not_set_pos_same_flag() {
        let a_emb = vec![0.1, 0.2, 0.3];
        let c_emb = vec![0.4, 0.5, 0.6];

        let encoded = encode_entry(&SrdRecord::Pair(SrdPairRecord {
            anchor_emb: a_emb,
            anchor_text: "anchor".to_string(),
            candidate_emb: c_emb,
            candidate_text: "candidate".to_string(),
            label: PairLabel::Positive,
        }));
        assert_eq!(encoded[0], MODE_PAIR);
        assert_eq!(encoded[1] & FLAG_POS_SAME, 0);
    }

    #[test]
    fn roundtrip_triplet() {
        let a_emb = vec![0.1, 0.2, 0.3];
        let p_emb = vec![0.4, 0.5, 0.6];
        let n_emb = vec![0.7, 0.8, 0.9];
        let a_text = "anchor";
        let p_text = "positive";
        let n_text = "negative";

        let record = SrdRecord::Triplet(SrdTripletRecord {
            anchor_emb: a_emb.clone(),
            anchor_text: a_text.to_string(),
            pos_emb: p_emb.clone(),
            pos_text: p_text.to_string(),
            neg_emb: n_emb.clone(),
            neg_text: n_text.to_string(),
        });
        let encoded = encode_entry(&record);
        let decoded = decode_entry(&encoded, 3).unwrap();

        assert!(matches!(decoded, SrdRecord::Triplet(_)));
        if let SrdRecord::Triplet(t) = decoded {
            assert_eq!(t.anchor_emb, a_emb);
            assert_eq!(t.anchor_text, a_text);
            assert_eq!(t.pos_emb, p_emb);
            assert_eq!(t.pos_text, p_text);
            assert_eq!(t.neg_emb, n_emb);
            assert_eq!(t.neg_text, n_text);
        }
    }

    #[test]
    fn roundtrip_triplet_pos_same_as_anchor() {
        let a_emb = vec![0.1, 0.2, 0.3];
        let n_emb = vec![0.7, 0.8, 0.9];
        let a_text = "anchor";
        let n_text = "negative";

        let record = SrdRecord::Triplet(SrdTripletRecord {
            anchor_emb: a_emb.clone(),
            anchor_text: a_text.to_string(),
            pos_emb: a_emb.clone(),
            pos_text: a_text.to_string(),
            neg_emb: n_emb.clone(),
            neg_text: n_text.to_string(),
        });
        let encoded = encode_entry(&record);
        let decoded = decode_entry(&encoded, 3).unwrap();

        assert!(matches!(decoded, SrdRecord::Triplet(_)));
        if let SrdRecord::Triplet(t) = decoded {
            assert_eq!(t.anchor_emb, a_emb);
            assert_eq!(t.anchor_text, a_text);
            assert_eq!(t.pos_emb, a_emb);
            assert_eq!(t.pos_text, a_text);
            assert_eq!(t.neg_emb, n_emb);
            assert_eq!(t.neg_text, n_text);
        }
    }

    #[test]
    fn triplet_pos_same_is_smaller() {
        let a_emb = vec![0.1, 0.2, 0.3];
        let a_text = "anchor";
        let p_emb = vec![0.4, 0.5, 0.6];
        let p_text = "positive";
        let n_emb = vec![0.7, 0.8, 0.9];
        let n_text = "negative";

        let distinct = encode_entry(&SrdRecord::Triplet(SrdTripletRecord {
            anchor_emb: a_emb.clone(),
            anchor_text: a_text.to_string(),
            pos_emb: p_emb.clone(),
            pos_text: p_text.to_string(),
            neg_emb: n_emb.clone(),
            neg_text: n_text.to_string(),
        }));
        let pos_same = encode_entry(&SrdRecord::Triplet(SrdTripletRecord {
            anchor_emb: a_emb.clone(),
            anchor_text: a_text.to_string(),
            pos_emb: a_emb.clone(),
            pos_text: a_text.to_string(),
            neg_emb: n_emb,
            neg_text: n_text.to_string(),
        }));
        assert!(pos_same.len() < distinct.len());
    }

    #[test]
    fn triplet_neg_same_is_smaller() {
        let a_emb = vec![0.1, 0.2, 0.3];
        let a_text = "anchor";
        let p_emb = vec![0.4, 0.5, 0.6];
        let p_text = "positive";
        let n_emb = vec![0.7, 0.8, 0.9];
        let n_text = "negative";

        let distinct = encode_entry(&SrdRecord::Triplet(SrdTripletRecord {
            anchor_emb: a_emb.clone(),
            anchor_text: a_text.to_string(),
            pos_emb: p_emb.clone(),
            pos_text: p_text.to_string(),
            neg_emb: n_emb,
            neg_text: n_text.to_string(),
        }));
        let neg_same = encode_entry(&SrdRecord::Triplet(SrdTripletRecord {
            anchor_emb: a_emb.clone(),
            anchor_text: a_text.to_string(),
            pos_emb: p_emb,
            pos_text: p_text.to_string(),
            neg_emb: a_emb.clone(),
            neg_text: a_text.to_string(),
        }));
        assert!(neg_same.len() < distinct.len());
    }

    #[test]
    fn triplet_all_same_is_smallest() {
        let a_emb = vec![0.1, 0.2, 0.3];
        let a_text = "same";

        let distinct = encode_entry(&SrdRecord::Triplet(SrdTripletRecord {
            anchor_emb: a_emb.clone(),
            anchor_text: a_text.to_string(),
            pos_emb: vec![0.4, 0.5, 0.6],
            pos_text: "positive".to_string(),
            neg_emb: vec![0.7, 0.8, 0.9],
            neg_text: "negative".to_string(),
        }));
        let pos_same = encode_entry(&SrdRecord::Triplet(SrdTripletRecord {
            anchor_emb: a_emb.clone(),
            anchor_text: a_text.to_string(),
            pos_emb: a_emb.clone(),
            pos_text: a_text.to_string(),
            neg_emb: vec![0.7, 0.8, 0.9],
            neg_text: "negative".to_string(),
        }));
        let all_same = encode_entry(&SrdRecord::Triplet(SrdTripletRecord {
            anchor_emb: a_emb.clone(),
            anchor_text: a_text.to_string(),
            pos_emb: a_emb.clone(),
            pos_text: a_text.to_string(),
            neg_emb: a_emb,
            neg_text: a_text.to_string(),
        }));
        assert!(pos_same.len() < distinct.len());
        assert!(all_same.len() < pos_same.len());
    }

    #[test]
    fn triplet_pos_same_sets_flag() {
        let a_emb = vec![0.1, 0.2, 0.3];
        let n_emb = vec![0.7, 0.8, 0.9];

        let encoded = encode_entry(&SrdRecord::Triplet(SrdTripletRecord {
            anchor_emb: a_emb.clone(),
            anchor_text: "anchor".to_string(),
            pos_emb: a_emb,
            pos_text: "anchor".to_string(),
            neg_emb: n_emb,
            neg_text: "negative".to_string(),
        }));
        assert_eq!(encoded[0], MODE_TRIPLET);
        assert_eq!(encoded[1] & FLAG_POS_SAME, FLAG_POS_SAME);
        assert_eq!(encoded[1] & FLAG_NEG_SAME, 0);
    }

    #[test]
    fn triplet_neg_same_sets_flag() {
        let a_emb = vec![0.1, 0.2, 0.3];
        let p_emb = vec![0.4, 0.5, 0.6];

        let encoded = encode_entry(&SrdRecord::Triplet(SrdTripletRecord {
            anchor_emb: a_emb.clone(),
            anchor_text: "anchor".to_string(),
            pos_emb: p_emb,
            pos_text: "positive".to_string(),
            neg_emb: a_emb,
            neg_text: "anchor".to_string(),
        }));
        assert_eq!(encoded[0], MODE_TRIPLET);
        assert_eq!(encoded[1] & FLAG_NEG_SAME, FLAG_NEG_SAME);
        assert_eq!(encoded[1] & FLAG_POS_SAME, 0);
    }

    #[test]
    fn triplet_all_same_sets_both_flags() {
        let a_emb = vec![0.1, 0.2, 0.3];

        let encoded = encode_entry(&SrdRecord::Triplet(SrdTripletRecord {
            anchor_emb: a_emb.clone(),
            anchor_text: "same".to_string(),
            pos_emb: a_emb.clone(),
            pos_text: "same".to_string(),
            neg_emb: a_emb,
            neg_text: "same".to_string(),
        }));
        assert_eq!(encoded[0], MODE_TRIPLET);
        assert_eq!(encoded[1] & FLAG_POS_SAME, FLAG_POS_SAME);
        assert_eq!(encoded[1] & FLAG_NEG_SAME, FLAG_NEG_SAME);
    }

    #[test]
    fn roundtrip_triplet_all_same() {
        let a_emb = vec![0.1, 0.2, 0.3];
        let a_text = "all same";

        let record = SrdRecord::Triplet(SrdTripletRecord {
            anchor_emb: a_emb.clone(),
            anchor_text: a_text.to_string(),
            pos_emb: a_emb.clone(),
            pos_text: a_text.to_string(),
            neg_emb: a_emb.clone(),
            neg_text: a_text.to_string(),
        });
        let encoded = encode_entry(&record);
        let decoded = decode_entry(&encoded, 3).unwrap();

        assert!(matches!(decoded, SrdRecord::Triplet(_)));
        if let SrdRecord::Triplet(t) = decoded {
            assert_eq!(t.anchor_emb, a_emb);
            assert_eq!(t.pos_emb, a_emb);
            assert_eq!(t.neg_emb, a_emb);
        }
    }

    #[test]
    fn roundtrip_triplet_neg_same_pos_different() {
        let a_emb = vec![0.1, 0.2, 0.3];
        let p_emb = vec![0.4, 0.5, 0.6];
        let n_emb = vec![0.1, 0.2, 0.3]; // same as anchor
        let a_text = "anchor";
        let p_text = "positive";
        let n_text = "anchor"; // same as anchor

        let record = SrdRecord::Triplet(SrdTripletRecord {
            anchor_emb: a_emb.clone(),
            anchor_text: a_text.to_string(),
            pos_emb: p_emb.clone(),
            pos_text: p_text.to_string(),
            neg_emb: n_emb.clone(),
            neg_text: n_text.to_string(),
        });
        let encoded = encode_entry(&record);
        let decoded = decode_entry(&encoded, 3).unwrap();

        assert!(matches!(decoded, SrdRecord::Triplet(_)));
        if let SrdRecord::Triplet(t) = decoded {
            assert_eq!(t.anchor_emb, a_emb);
            assert_eq!(t.anchor_text, a_text);
            assert_eq!(t.pos_emb, p_emb);
            assert_eq!(t.pos_text, p_text);
            assert_eq!(t.neg_emb, a_emb); // neg == anchor
            assert_eq!(t.neg_text, a_text); // neg == anchor
        }
    }

    #[test]
    fn pair_empty_text() {
        let record = SrdRecord::Pair(SrdPairRecord {
            anchor_emb: vec![1.0],
            anchor_text: String::new(),
            candidate_emb: vec![2.0],
            candidate_text: String::new(),
            label: PairLabel::Positive,
        });
        let encoded = encode_entry(&record);
        let decoded = decode_entry(&encoded, 1).unwrap();
        assert!(matches!(decoded, SrdRecord::Pair(_)));
        if let SrdRecord::Pair(p) = decoded {
            assert_eq!(p.anchor_text, "");
            assert_eq!(p.candidate_text, "");
        }
    }

    #[test]
    fn triplet_unicode_text() {
        let record = SrdRecord::Triplet(SrdTripletRecord {
            anchor_emb: vec![1.0],
            anchor_text: "hello \u{1F600}".to_string(),
            pos_emb: vec![2.0],
            pos_text: "\u{00E9}\u{00E8}\u{00EA}".to_string(),
            neg_emb: vec![3.0],
            neg_text: "\u{4E16}\u{754C}".to_string(),
        });
        let encoded = encode_entry(&record);
        let decoded = decode_entry(&encoded, 1).unwrap();
        if let SrdRecord::Triplet(t) = decoded {
            assert_eq!(t.anchor_text, "hello \u{1F600}");
            assert_eq!(t.pos_text, "\u{00E9}\u{00E8}\u{00EA}");
            assert_eq!(t.neg_text, "\u{4E16}\u{754C}");
        }
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
        let record = SrdRecord::Pair(SrdPairRecord {
            anchor_emb: vec![1.0],
            anchor_text: "a".into(),
            candidate_emb: vec![2.0],
            candidate_text: "b".into(),
            label: PairLabel::Positive,
        });
        let encoded = encode_entry(&record);
        assert_eq!(encoded[0], MODE_PAIR);
    }

    #[test]
    fn triplet_mode_byte_is_one() {
        let record = SrdRecord::Triplet(SrdTripletRecord {
            anchor_emb: vec![1.0],
            anchor_text: "a".into(),
            pos_emb: vec![2.0],
            pos_text: "b".into(),
            neg_emb: vec![3.0],
            neg_text: "c".into(),
        });
        let encoded = encode_entry(&record);
        assert_eq!(encoded[0], MODE_TRIPLET);
    }

    #[test]
    fn mode_byte_consistency_pair() {
        let record = SrdRecord::Pair(SrdPairRecord {
            anchor_emb: vec![1.0, 2.0],
            anchor_text: "hello".into(),
            candidate_emb: vec![3.0, 4.0],
            candidate_text: "world".into(),
            label: PairLabel::Positive,
        });
        let encoded = encode_entry(&record);
        let decoded = decode_entry(&encoded, 2).unwrap();
        assert!(matches!(decoded, SrdRecord::Pair(_)));
    }

    #[test]
    fn mode_byte_consistency_triplet() {
        let record = SrdRecord::Triplet(SrdTripletRecord {
            anchor_emb: vec![1.0],
            anchor_text: "a".into(),
            pos_emb: vec![2.0],
            pos_text: "b".into(),
            neg_emb: vec![3.0],
            neg_text: "c".into(),
        });
        let encoded = encode_entry(&record);
        let decoded = decode_entry(&encoded, 1).unwrap();
        assert!(matches!(decoded, SrdRecord::Triplet(_)));
    }

    // -----------------------------------------------------------------------
    // large embedding dimension
    // -----------------------------------------------------------------------

    #[test]
    fn large_emb_dim_roundtrip() {
        let dim = 2048;
        let a_emb: Vec<f32> = (0..dim).map(|i| i as f32 * 0.001).collect();
        let p_emb: Vec<f32> = (0..dim).map(|i| i as f32 * -0.001).collect();
        let record = SrdRecord::Pair(SrdPairRecord {
            anchor_emb: a_emb.clone(),
            anchor_text: "anchor".into(),
            candidate_emb: p_emb.clone(),
            candidate_text: "positive".into(),
            label: PairLabel::Positive,
        });
        let encoded = encode_entry(&record);
        let decoded = decode_entry(&encoded, dim).unwrap();
        if let SrdRecord::Pair(p) = decoded {
            assert_eq!(p.anchor_emb, a_emb);
            assert_eq!(p.candidate_emb, p_emb);
        }
    }

    // -----------------------------------------------------------------------
    // error: truncated text
    // -----------------------------------------------------------------------

    #[test]
    fn truncated_text_returns_error() {
        let record = SrdRecord::Pair(SrdPairRecord {
            anchor_emb: vec![1.0],
            anchor_text: "hello".into(),
            candidate_emb: vec![2.0],
            candidate_text: "world".into(),
            label: PairLabel::Positive,
        });
        let buf = encode_entry(&record);
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

        let pair_entries = make_pair_entries(&all_a_emb, &a_refs, &all_p_emb, &p_refs);
        write_pair_entries(&store, 0, pair_entries.as_slice()).unwrap();
        assert_eq!(store.len().unwrap(), n);

        let indices: Vec<usize> = (0..n).collect();
        let entries = batch_read_entries(&store, &indices, 2).unwrap();
        assert_eq!(entries.len(), n);

        for (i, entry) in entries.iter().enumerate() {
            if let SrdRecord::Pair(p) = entry {
                assert_eq!(p.anchor_emb, all_a_emb[i]);
                assert_eq!(p.anchor_text, all_a_text[i]);
                assert_eq!(p.candidate_emb, all_p_emb[i]);
                assert_eq!(p.candidate_text, all_p_text[i]);
            } else {
                panic!("expected Pair");
            }
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

        let trip_entries = make_triplet_entries(&a_emb, &a_refs, &p_emb, &p_refs, &n_emb, &n_refs);
        write_triplet_entries(&store, 0, trip_entries.as_slice()).unwrap();
        assert_eq!(store.len().unwrap(), n);

        let indices: Vec<usize> = (0..n).collect();
        let entries = batch_read_entries(&store, &indices, dim).unwrap();
        assert_eq!(entries.len(), n);

        for (i, entry) in entries.iter().enumerate() {
            if let SrdRecord::Triplet(t) = entry {
                assert_eq!(t.anchor_emb, a_emb[i]);
                assert_eq!(t.anchor_text, a_text[i]);
                assert_eq!(t.pos_emb, p_emb[i]);
                assert_eq!(t.pos_text, p_text[i]);
                assert_eq!(t.neg_emb, n_emb[i]);
                assert_eq!(t.neg_text, n_text[i]);
            } else {
                panic!("expected Triplet");
            }
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
        let pair_entries = make_pair_entries(&vecs, &refs, &vecs, &refs);
        write_pair_entries(&store, 0, pair_entries.as_slice()).unwrap();

        let entries = batch_read_entries(&store, &[0, 5, 9], 1).unwrap();
        assert_eq!(entries.len(), 3);
        if let SrdRecord::Pair(p) = &entries[0] {
            assert_eq!(p.anchor_text, "t0");
        }
        if let SrdRecord::Pair(p) = &entries[1] {
            assert_eq!(p.anchor_text, "t5");
        }
        if let SrdRecord::Pair(p) = &entries[2] {
            assert_eq!(p.anchor_text, "t9");
        }
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
        let pair_entries = make_pair_entries(&vecs, &texts, &vecs, &texts);
        write_pair_entries(&store, 0, pair_entries.as_slice()).unwrap();

        // Index 99 doesn't exist — should be skipped
        let entries = batch_read_entries(&store, &[0, 99], 1).unwrap();
        assert_eq!(entries.len(), 1);
        if let SrdRecord::Pair(p) = &entries[0] {
            assert_eq!(p.anchor_text, "a");
        }
    }

    // -----------------------------------------------------------------------
    // validation
    // -----------------------------------------------------------------------

    #[test]
    fn validate_write_batch_ok() {
        let vecs = [[1.0, 2.0], [3.0, 4.0]];
        let texts = ["hello", "world"];
        assert!(
            validate_write_batch(
                vecs.iter()
                    .zip(texts.iter())
                    .map(|(v, t)| (v.as_slice(), *t))
            )
            .is_ok()
        );
    }

    #[test]
    fn validate_write_batch_empty() {
        let vecs: Vec<Vec<f32>> = vec![];
        let texts: Vec<&str> = vec![];
        assert!(
            validate_write_batch(
                vecs.iter()
                    .zip(texts.iter())
                    .map(|(v, t)| (v.as_slice(), *t))
            )
            .is_ok()
        );
    }

    #[test]
    fn validate_write_batch_non_finite() {
        let vecs = [[1.0, f32::NAN]];
        let texts = ["hello"];
        assert!(
            validate_write_batch(
                vecs.iter()
                    .zip(texts.iter())
                    .map(|(v, t)| (v.as_slice(), *t))
            )
            .is_err()
        );
    }

    #[test]
    #[allow(clippy::useless_vec)]
    fn validate_write_batch_inconsistent_dim() {
        let vecs = vec![vec![1.0, 2.0], vec![3.0]];
        let texts = ["hello", "world"];
        assert!(
            validate_write_batch(
                vecs.iter()
                    .zip(texts.iter())
                    .map(|(v, t)| (v.as_slice(), *t))
            )
            .is_err()
        );
    }

    #[test]
    fn validate_write_batch_zero_dim() {
        let vecs: Vec<Vec<f32>> = vec![vec![]];
        let texts = ["hello"];
        assert!(
            validate_write_batch(
                vecs.iter()
                    .zip(texts.iter())
                    .map(|(v, t)| (v.as_slice(), *t))
            )
            .is_err()
        );
    }

    // -----------------------------------------------------------------------
    // write_pair_entries / write_triplet_entries validation
    // -----------------------------------------------------------------------

    #[test]
    fn write_pair_entries_correct_start_idx() {
        let dir = TempDir::new().unwrap();
        let store = DataStore::open(&dir.path().join("test.srd")).unwrap();

        // Write first batch at idx 0
        let vecs1 = vec![vec![1.0], vec![2.0]];
        let texts1 = vec!["a", "b"];
        let pair_entries = make_pair_entries(&vecs1, &texts1, &vecs1, &texts1);
        write_pair_entries(&store, 0, pair_entries.as_slice()).unwrap();

        // Write second batch at idx 2
        let vecs2 = vec![vec![3.0]];
        let texts2 = vec!["c"];
        let pair_entries = make_pair_entries(&vecs2, &texts2, &vecs2, &texts2);
        write_pair_entries(&store, 2, pair_entries.as_slice()).unwrap();

        assert_eq!(store.len().unwrap(), 3);

        let entries = batch_read_entries(&store, &[0, 1, 2], 1).unwrap();
        if let SrdRecord::Pair(p) = &entries[0] {
            assert_eq!(p.anchor_text, "a");
        }
        if let SrdRecord::Pair(p) = &entries[1] {
            assert_eq!(p.anchor_text, "b");
        }
        if let SrdRecord::Pair(p) = &entries[2] {
            assert_eq!(p.anchor_text, "c");
        }
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
        let pair_entries = make_pair_entries(&vecs, &texts, &vecs, &texts);
        write_pair_entries(&store, 5, pair_entries.as_slice()).unwrap();

        let entries = batch_read_entries(&store, &[5], 2).unwrap();
        assert_eq!(entries.len(), 1);
        if let SrdRecord::Pair(p) = &entries[0] {
            assert_eq!(p.anchor_text, "first");
        }
    }

    // -----------------------------------------------------------------------
    // cross-batch read safety
    // -----------------------------------------------------------------------

    #[test]
    fn cross_batch_read_safety_pair() {
        let dir = TempDir::new().unwrap();
        let store = DataStore::open(&dir.path().join("test.srd")).unwrap();

        // Batch A: entry 0 same as anchor, entry 1 different
        let a_vecs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let p_vecs = vec![vec![1.0, 2.0], vec![30.0, 40.0]]; // different emb for diff entry
        let a_texts = vec!["same", "diff_a"];
        let p_texts = vec!["same", "diff_p"];
        let pair_entries = make_pair_entries(&a_vecs, &a_texts, &p_vecs, &p_texts);
        write_pair_entries(&store, 0, pair_entries.as_slice()).unwrap();

        // Batch B: entry 2 same, entry 3 same
        let b_vecs = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let b_texts = vec!["batch_b_0", "batch_b_1"];
        let pair_entries = make_pair_entries(&b_vecs, &b_texts, &b_vecs, &b_texts);
        write_pair_entries(&store, 2, pair_entries.as_slice()).unwrap();

        assert_eq!(store.len().unwrap(), 4);

        // Read all entries from both batches
        let entries = batch_read_entries(&store, &[0, 1, 2, 3], 2).unwrap();
        assert_eq!(entries.len(), 4);

        // Entry 0: same anchor/candidate
        if let SrdRecord::Pair(p) = &entries[0] {
            assert_eq!(p.anchor_text, "same");
            assert_eq!(p.candidate_text, "same");
            assert_eq!(p.anchor_emb, p.candidate_emb);
        }

        // Entry 1: different anchor/candidate
        if let SrdRecord::Pair(p) = &entries[1] {
            assert_eq!(p.anchor_text, "diff_a");
            assert_eq!(p.candidate_text, "diff_p");
            assert_eq!(p.anchor_emb, vec![3.0, 4.0]);
            assert_eq!(p.candidate_emb, vec![30.0, 40.0]);
        }

        // Entry 2: same anchor/candidate
        if let SrdRecord::Pair(p) = &entries[2] {
            assert_eq!(p.anchor_text, "batch_b_0");
            assert_eq!(p.candidate_text, "batch_b_0");
        }

        // Entry 3: same anchor/candidate
        if let SrdRecord::Pair(p) = &entries[3] {
            assert_eq!(p.anchor_text, "batch_b_1");
            assert_eq!(p.candidate_text, "batch_b_1");
        }
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
            &store,
            0,
            &make_triplet_entries(
                &a_a_vecs, &a_a_texts, &a_p_vecs, &a_p_texts, &a_n_vecs, &a_n_texts,
            ),
        )
        .unwrap();

        // Batch B: entry 2 all same
        let b_a_vecs = vec![vec![3.0]];
        let b_a_texts = vec!["all_same"];
        let b_p_texts = vec!["all_same"];
        let b_n_texts = vec!["all_same"];
        write_triplet_entries(
            &store,
            2,
            &make_triplet_entries(
                &b_a_vecs, &b_a_texts, &b_a_vecs, &b_p_texts, &b_a_vecs, &b_n_texts,
            ),
        )
        .unwrap();

        assert_eq!(store.len().unwrap(), 3);

        let entries = batch_read_entries(&store, &[0, 1, 2], 1).unwrap();
        assert_eq!(entries.len(), 3);

        // Entry 0: all different — verify each embedding matches its role
        if let SrdRecord::Triplet(t) = &entries[0] {
            assert_eq!(t.anchor_text, "a0");
            assert_eq!(t.anchor_emb, vec![1.0]);
            assert_eq!(t.pos_text, "p0");
            assert_eq!(t.pos_emb, vec![10.0]);
            assert_eq!(t.neg_text, "n0");
            assert_eq!(t.neg_emb, vec![100.0]);
        }

        // Entry 1: pos same as anchor
        if let SrdRecord::Triplet(t) = &entries[1] {
            assert_eq!(t.anchor_text, "a1");
            assert_eq!(t.pos_text, "a1");
            assert_eq!(t.neg_text, "n1");
        }

        // Entry 2: all same
        if let SrdRecord::Triplet(t) = &entries[2] {
            assert_eq!(t.anchor_text, "all_same");
            assert_eq!(t.pos_text, "all_same");
            assert_eq!(t.neg_text, "all_same");
            assert_eq!(t.anchor_emb, t.pos_emb);
            assert_eq!(t.anchor_emb, t.neg_emb);
        }
    }

    #[test]
    fn roundtrip_pair_negative_label() {
        let a_emb = vec![0.1, 0.2, 0.3];
        let c_emb = vec![0.4, 0.5, 0.6];

        let record = SrdRecord::Pair(SrdPairRecord {
            anchor_emb: a_emb.clone(),
            anchor_text: "anchor".to_string(),
            candidate_emb: c_emb.clone(),
            candidate_text: "negative_candidate".to_string(),
            label: PairLabel::Negative,
        });
        let encoded = encode_entry(&record);
        let decoded = decode_entry(&encoded, 3).unwrap();

        assert!(matches!(decoded, SrdRecord::Pair(_)));
        if let SrdRecord::Pair(p) = decoded {
            assert_eq!(p.label, PairLabel::Negative);
            assert_eq!(p.anchor_emb, a_emb);
            assert_eq!(p.anchor_text, "anchor");
            assert_eq!(p.candidate_emb, c_emb);
            assert_eq!(p.candidate_text, "negative_candidate");
        }
    }

    #[test]
    fn roundtrip_pair_negative_label_same_as_anchor() {
        let shared_emb = vec![0.1, 0.2, 0.3];
        let shared_text = "shared";

        let record = SrdRecord::Pair(SrdPairRecord {
            anchor_emb: shared_emb.clone(),
            anchor_text: shared_text.to_string(),
            candidate_emb: shared_emb.clone(),
            candidate_text: shared_text.to_string(),
            label: PairLabel::Negative,
        });
        let encoded = encode_entry(&record);
        let decoded = decode_entry(&encoded, 3).unwrap();

        assert!(matches!(decoded, SrdRecord::Pair(_)));
        if let SrdRecord::Pair(p) = decoded {
            assert_eq!(p.label, PairLabel::Negative);
            assert_eq!(p.anchor_emb, shared_emb);
            assert_eq!(p.anchor_text, shared_text);
        }
    }

    #[test]
    fn srd_mode_section_count_returns_correct_values() {
        assert_eq!(SrdMode::Pair.section_count(), 2);
        assert_eq!(SrdMode::Triplet.section_count(), 3);
    }

    #[test]
    fn decode_entry_single_byte_returns_too_short() {
        let result = decode_entry(&[MODE_PAIR], 2);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), SrdError::EntryTooShort));
    }

    #[test]
    fn write_pair_entries_empty_batch_returns_ok() {
        let tmp = TempDir::new().unwrap();
        let store = DataStore::open(&tmp.path().join("test.bin")).unwrap();
        write_pair_entries(&store, 0, &[]).unwrap();
    }

    #[test]
    fn write_triplet_entries_empty_batch_returns_ok() {
        let tmp = TempDir::new().unwrap();
        let store = DataStore::open(&tmp.path().join("test.bin")).unwrap();
        write_triplet_entries(&store, 0, &[]).unwrap();
    }
}
