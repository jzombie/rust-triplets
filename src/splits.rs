use serde::{Deserialize, Serialize};
use simd_r_drive::storage_engine::DataStore;
use simd_r_drive::storage_engine::traits::{DataStoreReader, DataStoreWriter};
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use crate::constants::splits::{
    ALL_SPLITS, BITCODE_PREFIX, DEFAULT_STORE_FILENAME, EPOCH_HASH_RECORD_VERSION,
    EPOCH_HASHES_PREFIX, EPOCH_META_PREFIX, EPOCH_META_RECORD_VERSION, EPOCH_RECORD_TOMBSTONE,
    META_KEY, SAMPLER_STATE_KEY, SAMPLER_STATE_RECORD_VERSION, SPLIT_PREFIX, STORE_VERSION,
};
use crate::data::RecordId;
use crate::errors::SamplerError;
use crate::types::SourceId;

/// Train/validation/test split labels.
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    bitcode::Encode,
    bitcode::Decode,
)]
pub enum SplitLabel {
    /// Training split.
    Train,
    /// Validation split.
    Validation,
    /// Test split.
    Test,
}

/// Ratios for train/validation/test splits (must sum to 1.0).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, bitcode::Encode, bitcode::Decode)]
pub struct SplitRatios {
    /// Fraction assigned to train.
    pub train: f32,
    /// Fraction assigned to validation.
    pub validation: f32,
    /// Fraction assigned to test.
    pub test: f32,
}

impl Default for SplitRatios {
    fn default() -> Self {
        Self {
            train: 0.8,
            validation: 0.1,
            test: 0.1,
        }
    }
}

impl SplitRatios {
    /// Validate ratios sum to `1.0` (within epsilon) and return normalized config.
    pub fn normalized(self) -> Result<Self, SamplerError> {
        let sum = self.train + self.validation + self.test;
        if (sum - 1.0).abs() > 1e-6 {
            return Err(SamplerError::Configuration(
                "split ratios must sum to 1.0".to_string(),
            ));
        }
        Ok(self)
    }
}

pub use crate::constants::splits::{DEFAULT_STORE_DIR, EPOCH_STATE_VERSION};

/// Persisted epoch metadata for a split.
#[derive(Clone, Debug, bitcode::Encode, bitcode::Decode)]
pub struct PersistedSplitMeta {
    /// Current epoch for this split.
    pub epoch: u64,
    /// Cursor offset within the epoch hash list.
    pub offset: u64,
    /// Checksum of the persisted hash list.
    pub hashes_checksum: u64,
}

/// Persisted epoch hash list for a split.
#[derive(Clone, Debug, bitcode::Encode, bitcode::Decode)]
pub struct PersistedSplitHashes {
    /// Checksum of `hashes`.
    pub checksum: u64,
    /// Deterministic per-epoch hash ordering.
    pub hashes: Vec<u64>,
}

/// Persisted sampler state (round-robin indices, cursors, RNG).
#[derive(Clone, Debug, bitcode::Encode, bitcode::Decode)]
pub struct PersistedSamplerState {
    /// Source-cycle round-robin index.
    pub source_cycle_idx: u64,
    /// Per-source record cursors.
    pub source_record_cursors: Vec<(SourceId, u64)>,
    /// Current source epoch used for deterministic reshuffle.
    pub source_epoch: u64,
    /// Deterministic RNG internal state.
    pub rng_state: u64,
    /// Round-robin index for triplet recipes.
    pub triplet_recipe_rr_idx: u64,
    /// Round-robin index for text recipes.
    pub text_recipe_rr_idx: u64,
    /// Persisted source stream refresh cursors.
    pub source_stream_cursors: Vec<(SourceId, u64)>,
}

/// Split assignment store (deterministic via `record_id + seed + ratios`).
pub trait SplitStore: Send + Sync {
    /// Return stored/derived split label for `id`.
    fn label_for(&self, id: &RecordId) -> Option<SplitLabel>;
    /// Persist an explicit split assignment.
    fn upsert(&self, id: RecordId, label: SplitLabel) -> Result<(), SamplerError>;
    /// Return configured split ratios.
    fn ratios(&self) -> SplitRatios;
    /// Returns the split label for `id`.
    fn ensure(&self, id: RecordId) -> Result<SplitLabel, SamplerError>;
}

/// Storage backend for epoch cursor state.
pub trait EpochStateStore: Send + Sync {
    /// Load split→epoch metadata map.
    fn load_epoch_meta(&self) -> Result<HashMap<SplitLabel, PersistedSplitMeta>, SamplerError>;
    /// Load persisted epoch hashes for one split, if available.
    fn load_epoch_hashes(
        &self,
        label: SplitLabel,
    ) -> Result<Option<PersistedSplitHashes>, SamplerError>;
    /// Persist split→epoch metadata map.
    fn store_epoch_meta(
        &self,
        meta: &HashMap<SplitLabel, PersistedSplitMeta>,
    ) -> Result<(), SamplerError>;
    /// Persist epoch hash list for one split.
    fn store_epoch_hashes(
        &self,
        label: SplitLabel,
        hashes: &PersistedSplitHashes,
    ) -> Result<(), SamplerError>;
}

/// Storage backend for sampler state (round-robin indices, RNG).
pub trait SamplerStateStore: Send + Sync {
    /// Load persisted sampler runtime state, if present.
    fn load_sampler_state(&self) -> Result<Option<PersistedSamplerState>, SamplerError>;
    /// Persist sampler runtime state.
    fn store_sampler_state(&self, state: &PersistedSamplerState) -> Result<(), SamplerError>;
}

/// In-memory split store with deterministic assignments.
pub struct DeterministicSplitStore {
    ratios: SplitRatios,
    assignments: RwLock<HashMap<RecordId, SplitLabel>>,
    seed: u64,
    epoch_meta: RwLock<HashMap<SplitLabel, PersistedSplitMeta>>,
    epoch_hashes: RwLock<HashMap<SplitLabel, PersistedSplitHashes>>,
    sampler_state: RwLock<Option<PersistedSamplerState>>,
}

impl DeterministicSplitStore {
    /// Construct an in-memory deterministic split store.
    pub fn new(ratios: SplitRatios, seed: u64) -> Result<Self, SamplerError> {
        ratios.normalized()?;
        Ok(Self {
            ratios,
            assignments: RwLock::new(HashMap::new()),
            seed,
            epoch_meta: RwLock::new(HashMap::new()),
            epoch_hashes: RwLock::new(HashMap::new()),
            sampler_state: RwLock::new(None),
        })
    }

    fn derive_label(&self, id: &RecordId) -> SplitLabel {
        derive_label_for_id(id, self.seed, self.ratios)
    }
}

impl SplitStore for DeterministicSplitStore {
    fn label_for(&self, id: &RecordId) -> Option<SplitLabel> {
        if let Some(label) = self.assignments.read().ok()?.get(id).copied() {
            return Some(label);
        }
        Some(self.derive_label(id))
    }

    fn upsert(&self, id: RecordId, label: SplitLabel) -> Result<(), SamplerError> {
        let mut guard = self
            .assignments
            .write()
            .map_err(|_| SamplerError::SplitStore("lock poisoned".into()))?;
        guard.insert(id, label);
        Ok(())
    }

    fn ratios(&self) -> SplitRatios {
        self.ratios
    }

    fn ensure(&self, id: RecordId) -> Result<SplitLabel, SamplerError> {
        Ok(self.derive_label(&id))
    }
}

impl EpochStateStore for DeterministicSplitStore {
    fn load_epoch_meta(&self) -> Result<HashMap<SplitLabel, PersistedSplitMeta>, SamplerError> {
        self.epoch_meta
            .read()
            .map_err(|_| SamplerError::SplitStore("epoch meta lock poisoned".into()))
            .map(|guard| guard.clone())
    }

    fn load_epoch_hashes(
        &self,
        label: SplitLabel,
    ) -> Result<Option<PersistedSplitHashes>, SamplerError> {
        Ok(self
            .epoch_hashes
            .read()
            .map_err(|_| SamplerError::SplitStore("epoch hashes lock poisoned".into()))?
            .get(&label)
            .cloned())
    }

    fn store_epoch_meta(
        &self,
        meta: &HashMap<SplitLabel, PersistedSplitMeta>,
    ) -> Result<(), SamplerError> {
        *self
            .epoch_meta
            .write()
            .map_err(|_| SamplerError::SplitStore("epoch meta lock poisoned".into()))? =
            meta.clone();
        Ok(())
    }

    fn store_epoch_hashes(
        &self,
        label: SplitLabel,
        hashes: &PersistedSplitHashes,
    ) -> Result<(), SamplerError> {
        self.epoch_hashes
            .write()
            .map_err(|_| SamplerError::SplitStore("epoch hashes lock poisoned".into()))?
            .insert(label, hashes.clone());
        Ok(())
    }
}

impl SamplerStateStore for DeterministicSplitStore {
    fn load_sampler_state(&self) -> Result<Option<PersistedSamplerState>, SamplerError> {
        self.sampler_state
            .read()
            .map_err(|_| SamplerError::SplitStore("sampler state lock poisoned".into()))
            .map(|guard| guard.clone())
    }

    fn store_sampler_state(&self, state: &PersistedSamplerState) -> Result<(), SamplerError> {
        *self
            .sampler_state
            .write()
            .map_err(|_| SamplerError::SplitStore("sampler state lock poisoned".into()))? =
            Some(state.clone());
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, bitcode::Encode, bitcode::Decode)]
/// Versioned metadata header persisted in file-backed split stores.
struct StoreMeta {
    version: u8,
    seed: u64,
    ratios: SplitRatios,
}

fn encode_store_meta(meta: &StoreMeta) -> Vec<u8> {
    encode_bitcode_payload(&bitcode::encode(meta))
}

fn decode_store_meta(bytes: &[u8]) -> Result<StoreMeta, SamplerError> {
    let raw = decode_bitcode_payload(bytes)?;
    bitcode::decode(&raw).map_err(|err| {
        SamplerError::SplitStore(format!("failed to decode split store metadata: {err}"))
    })
}

/// File-backed split store for persistent runs (metadata + epoch/sampler state).
pub struct FileSplitStore {
    store: DataStore,
    ratios: SplitRatios,
    seed: u64,
}

impl fmt::Debug for FileSplitStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FileSplitStore")
            .field("ratios", &self.ratios)
            .field("seed", &self.seed)
            .finish()
    }
}

impl FileSplitStore {
    /// Open (or create) a file-backed split store at `path`.
    pub fn open<P: Into<PathBuf>>(
        path: P,
        ratios: SplitRatios,
        seed: u64,
    ) -> Result<Self, SamplerError> {
        let ratios = ratios.normalized()?;
        let path = coerce_store_path(path.into());
        ensure_parent_dir(&path)?;
        let store = DataStore::open(path.as_path()).map_err(map_store_err)?;
        let store = Self {
            store,
            ratios,
            seed,
        };
        store.verify_metadata()?;
        Ok(store)
    }

    /// Default split-store file path under the crate's default store directory.
    pub fn default_path() -> PathBuf {
        Self::default_path_in_dir(DEFAULT_STORE_DIR)
    }

    /// Default split-store file path inside a custom directory.
    pub fn default_path_in_dir<P: AsRef<Path>>(dir: P) -> PathBuf {
        dir.as_ref().join(DEFAULT_STORE_FILENAME)
    }

    fn verify_metadata(&self) -> Result<(), SamplerError> {
        match read_bytes(&self.store, META_KEY)? {
            Some(bytes) => {
                let meta = decode_store_meta(&bytes)?;
                if meta.version != STORE_VERSION {
                    return Err(SamplerError::SplitStore(format!(
                        "split store version mismatch (expected {}, found {})",
                        STORE_VERSION, meta.version
                    )));
                }
                if meta.seed != self.seed {
                    return Err(SamplerError::SplitStore(format!(
                        "split store seed mismatch (expected {}, found {})",
                        self.seed, meta.seed
                    )));
                }
                if !ratios_close(meta.ratios, self.ratios) {
                    return Err(SamplerError::SplitStore(
                        "split store ratios mismatch".into(),
                    ));
                }
            }
            None => {
                let blob = StoreMeta {
                    version: STORE_VERSION,
                    seed: self.seed,
                    ratios: self.ratios,
                };
                let payload = encode_store_meta(&blob);
                write_bytes(&self.store, META_KEY, &payload)?;
            }
        }
        Ok(())
    }

    fn read_epoch_meta_entry(
        &self,
        label: SplitLabel,
    ) -> Result<Option<PersistedSplitMeta>, SamplerError> {
        let key = epoch_meta_key(label);
        let entry = self.store.read(&key).map_err(map_store_err)?;
        match entry {
            None => Ok(None),
            Some(bytes) => decode_epoch_meta(bytes.as_ref()),
        }
    }

    fn write_epoch_meta_entry(
        &self,
        label: SplitLabel,
        meta: Option<&PersistedSplitMeta>,
    ) -> Result<(), SamplerError> {
        let key = epoch_meta_key(label);
        let payload = encode_epoch_meta(meta);
        self.store
            .write(&key, payload.as_slice())
            .map_err(map_store_err)?;
        Ok(())
    }

    fn read_epoch_hashes_entry(
        &self,
        label: SplitLabel,
    ) -> Result<Option<PersistedSplitHashes>, SamplerError> {
        let key = epoch_hashes_key(label);
        let entry = self.store.read(&key).map_err(map_store_err)?;
        match entry {
            None => Ok(None),
            Some(bytes) => decode_epoch_hashes(bytes.as_ref()),
        }
    }

    fn write_epoch_hashes_entry(
        &self,
        label: SplitLabel,
        hashes: &PersistedSplitHashes,
    ) -> Result<(), SamplerError> {
        let key = epoch_hashes_key(label);
        let payload = encode_epoch_hashes(hashes);
        self.store
            .write(&key, payload.as_slice())
            .map_err(map_store_err)?;
        Ok(())
    }
}

impl SplitStore for FileSplitStore {
    fn label_for(&self, id: &RecordId) -> Option<SplitLabel> {
        let key = split_key(id);
        if let Ok(Some(value)) = self.store.read(&key)
            && let Ok(label) = decode_label(value.as_ref())
        {
            return Some(label);
        }
        Some(derive_label_for_id(id, self.seed, self.ratios))
    }

    fn upsert(&self, id: RecordId, label: SplitLabel) -> Result<(), SamplerError> {
        let _ = (id, label);
        Ok(())
    }

    fn ratios(&self) -> SplitRatios {
        self.ratios
    }

    fn ensure(&self, id: RecordId) -> Result<SplitLabel, SamplerError> {
        Ok(derive_label_for_id(&id, self.seed, self.ratios))
    }
}

impl EpochStateStore for FileSplitStore {
    fn load_epoch_meta(&self) -> Result<HashMap<SplitLabel, PersistedSplitMeta>, SamplerError> {
        let mut meta = HashMap::new();
        for label in ALL_SPLITS {
            if let Some(entry) = self.read_epoch_meta_entry(label)? {
                meta.insert(label, entry);
            }
        }
        Ok(meta)
    }

    fn load_epoch_hashes(
        &self,
        label: SplitLabel,
    ) -> Result<Option<PersistedSplitHashes>, SamplerError> {
        self.read_epoch_hashes_entry(label)
    }

    fn store_epoch_meta(
        &self,
        meta: &HashMap<SplitLabel, PersistedSplitMeta>,
    ) -> Result<(), SamplerError> {
        for label in ALL_SPLITS {
            self.write_epoch_meta_entry(label, meta.get(&label))?;
        }
        Ok(())
    }

    fn store_epoch_hashes(
        &self,
        label: SplitLabel,
        hashes: &PersistedSplitHashes,
    ) -> Result<(), SamplerError> {
        self.write_epoch_hashes_entry(label, hashes)
    }
}

impl SamplerStateStore for FileSplitStore {
    fn load_sampler_state(&self) -> Result<Option<PersistedSamplerState>, SamplerError> {
        match read_bytes(&self.store, SAMPLER_STATE_KEY)? {
            Some(bytes) => decode_sampler_state(bytes.as_ref()),
            None => Ok(None),
        }
    }

    fn store_sampler_state(&self, state: &PersistedSamplerState) -> Result<(), SamplerError> {
        let payload = encode_sampler_state(state);
        write_bytes(&self.store, SAMPLER_STATE_KEY, &payload)
    }
}

fn decode_label(bytes: &[u8]) -> Result<SplitLabel, SamplerError> {
    match bytes.first() {
        Some(b'0') => Ok(SplitLabel::Train),
        Some(b'1') => Ok(SplitLabel::Validation),
        Some(b'2') => Ok(SplitLabel::Test),
        _ => Err(SamplerError::SplitStore("invalid split label".into())),
    }
}

fn derive_label_for_id(id: &RecordId, seed: u64, ratios: SplitRatios) -> SplitLabel {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    id.hash(&mut hasher);
    seed.hash(&mut hasher);
    let value = hasher.finish() as f64 / u64::MAX as f64;
    let train_cut = ratios.train as f64;
    let val_cut = train_cut + ratios.validation as f64;
    if value < train_cut {
        SplitLabel::Train
    } else if value < val_cut {
        SplitLabel::Validation
    } else {
        SplitLabel::Test
    }
}

fn ratios_close(a: SplitRatios, b: SplitRatios) -> bool {
    ((a.train - b.train).abs() + (a.validation - b.validation).abs() + (a.test - b.test).abs())
        < 1e-5
}

fn split_key(id: &RecordId) -> Vec<u8> {
    let mut key = Vec::with_capacity(SPLIT_PREFIX.len() + id.len());
    key.extend_from_slice(SPLIT_PREFIX);
    key.extend_from_slice(id.as_bytes());
    key
}

fn read_bytes(store: &DataStore, key: &[u8]) -> Result<Option<Vec<u8>>, SamplerError> {
    store
        .read(key)
        .map_err(map_store_err)?
        .map(|entry| Ok(entry.as_ref().to_vec()))
        .transpose()
}

fn write_bytes(store: &DataStore, key: &[u8], payload: &[u8]) -> Result<(), SamplerError> {
    store.write(key, payload).map_err(map_store_err)?;
    Ok(())
}

fn epoch_meta_key(label: SplitLabel) -> Vec<u8> {
    let mut key = Vec::with_capacity(EPOCH_META_PREFIX.len() + 12);
    key.extend_from_slice(EPOCH_META_PREFIX);
    let suffix = match label {
        SplitLabel::Train => b"train".as_ref(),
        SplitLabel::Validation => b"validation".as_ref(),
        SplitLabel::Test => b"test".as_ref(),
    };
    key.extend_from_slice(suffix);
    key
}

fn epoch_hashes_key(label: SplitLabel) -> Vec<u8> {
    let mut key = Vec::with_capacity(EPOCH_HASHES_PREFIX.len() + 12);
    key.extend_from_slice(EPOCH_HASHES_PREFIX);
    let suffix = match label {
        SplitLabel::Train => b"train".as_ref(),
        SplitLabel::Validation => b"validation".as_ref(),
        SplitLabel::Test => b"test".as_ref(),
    };
    key.extend_from_slice(suffix);
    key
}

fn encode_epoch_meta(meta: Option<&PersistedSplitMeta>) -> Vec<u8> {
    match meta {
        None => vec![EPOCH_RECORD_TOMBSTONE],
        Some(meta) => {
            let payload = encode_bitcode_payload(&bitcode::encode(meta));
            let mut buf = Vec::with_capacity(1 + payload.len());
            buf.push(EPOCH_META_RECORD_VERSION);
            buf.extend_from_slice(&payload);
            buf
        }
    }
}

fn decode_epoch_meta(bytes: &[u8]) -> Result<Option<PersistedSplitMeta>, SamplerError> {
    if bytes.is_empty() || bytes[0] == EPOCH_RECORD_TOMBSTONE {
        return Ok(None);
    }
    if bytes[0] != EPOCH_META_RECORD_VERSION {
        return Err(SamplerError::SplitStore(
            "epoch meta record version mismatch".into(),
        ));
    }
    let raw = decode_bitcode_payload(&bytes[1..])?;
    bitcode::decode(&raw)
        .map(Some)
        .map_err(|err| SamplerError::SplitStore(format!("corrupt epoch meta record: {err}")))
}

fn encode_epoch_hashes(hashes: &PersistedSplitHashes) -> Vec<u8> {
    let payload = encode_bitcode_payload(&bitcode::encode(hashes));
    let mut buf = Vec::with_capacity(1 + payload.len());
    buf.push(EPOCH_HASH_RECORD_VERSION);
    buf.extend_from_slice(&payload);
    buf
}

fn decode_epoch_hashes(bytes: &[u8]) -> Result<Option<PersistedSplitHashes>, SamplerError> {
    if bytes.is_empty() || bytes[0] == EPOCH_RECORD_TOMBSTONE {
        return Ok(None);
    }
    if bytes[0] != EPOCH_HASH_RECORD_VERSION {
        return Err(SamplerError::SplitStore(
            "epoch hashes record version mismatch".into(),
        ));
    }
    let raw = decode_bitcode_payload(&bytes[1..])?;
    bitcode::decode(&raw)
        .map(Some)
        .map_err(|err| SamplerError::SplitStore(format!("corrupt epoch hashes record: {err}")))
}

fn encode_sampler_state(state: &PersistedSamplerState) -> Vec<u8> {
    let payload = encode_bitcode_payload(&bitcode::encode(state));
    let mut buf = Vec::with_capacity(1 + payload.len());
    buf.push(SAMPLER_STATE_RECORD_VERSION);
    buf.extend_from_slice(&payload);
    buf
}

fn decode_sampler_state(bytes: &[u8]) -> Result<Option<PersistedSamplerState>, SamplerError> {
    if bytes.is_empty() {
        return Ok(None);
    }
    if bytes[0] != SAMPLER_STATE_RECORD_VERSION {
        return Err(SamplerError::SplitStore(
            "sampler state record version mismatch".into(),
        ));
    }
    let raw = decode_bitcode_payload(&bytes[1..])?;
    bitcode::decode(&raw)
        .map(Some)
        .map_err(|err| SamplerError::SplitStore(format!("corrupt sampler state record: {err}")))
}

fn encode_bitcode_payload(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + bytes.len());
    out.push(BITCODE_PREFIX);
    out.extend_from_slice(bytes);
    out
}

fn decode_bitcode_payload(bytes: &[u8]) -> Result<Vec<u8>, SamplerError> {
    if bytes.first().copied() != Some(BITCODE_PREFIX) {
        return Err(SamplerError::SplitStore(
            "bitcode payload missing expected prefix".into(),
        ));
    }
    Ok(bytes[1..].to_vec())
}

fn coerce_store_path(path: PathBuf) -> PathBuf {
    if path.is_dir() {
        return path.join(DEFAULT_STORE_FILENAME);
    }
    path
}

fn ensure_parent_dir(path: &Path) -> Result<(), SamplerError> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn map_store_err(err: io::Error) -> SamplerError {
    SamplerError::SplitStore(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn split_ratios_reject_non_unit_sum() {
        let invalid = SplitRatios {
            train: 0.6,
            validation: 0.3,
            test: 0.3,
        };

        let err = match DeterministicSplitStore::new(invalid, 1) {
            Ok(_) => panic!("expected non-unit split ratios to fail"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            SamplerError::Configuration(ref msg) if msg.contains("split ratios must sum to 1.0")
        ));

        let dir = tempdir().unwrap();
        let path = dir.path().join("split_store.bin");
        let err = FileSplitStore::open(&path, invalid, 1).unwrap_err();
        assert!(matches!(
            err,
            SamplerError::Configuration(ref msg) if msg.contains("split ratios must sum to 1.0")
        ));
    }

    #[test]
    fn zero_test_ratio_never_assigns_test_labels() {
        let ratios = SplitRatios {
            train: 0.5,
            validation: 0.5,
            test: 0.0,
        };
        let store = DeterministicSplitStore::new(ratios, 7).unwrap();

        let mut saw_train = false;
        let mut saw_validation = false;
        for idx in 0..20_000 {
            let id = format!("record_{idx}");
            let label = store.ensure(id).unwrap();
            assert_ne!(label, SplitLabel::Test);
            saw_train |= label == SplitLabel::Train;
            saw_validation |= label == SplitLabel::Validation;
            if saw_train && saw_validation {
                break;
            }
        }

        assert!(saw_train);
        assert!(saw_validation);
    }

    #[test]
    fn file_store_persists_assignments() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("splits.json");
        let ratios = SplitRatios::default();
        let store = FileSplitStore::open(&path, ratios, 123).unwrap();
        let label = store.ensure("abc".to_string()).unwrap();
        assert!(matches!(
            label,
            SplitLabel::Train | SplitLabel::Validation | SplitLabel::Test
        ));
        drop(store);

        let store_again = FileSplitStore::open(&path, ratios, 123).unwrap();
        assert_eq!(store_again.label_for(&"abc".to_string()).unwrap(), label);
    }

    #[test]
    fn file_store_rejects_seed_mismatch() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("splits.json");
        let ratios = SplitRatios::default();
        let store = FileSplitStore::open(&path, ratios, 123).unwrap();
        store.ensure("abc".to_string()).unwrap();
        drop(store);

        let err = FileSplitStore::open(&path, ratios, 999).unwrap_err();
        match err {
            SamplerError::SplitStore(msg) => {
                assert!(msg.contains("seed"))
            }
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn file_store_accepts_directory_path() {
        let dir = tempdir().unwrap();
        let ratios = SplitRatios::default();
        let store = FileSplitStore::open(dir.path(), ratios, 777).unwrap();
        store.ensure("abc".to_string()).unwrap();
        let expected_file = dir.path().join(DEFAULT_STORE_FILENAME);
        assert!(expected_file.is_file());
    }
}
