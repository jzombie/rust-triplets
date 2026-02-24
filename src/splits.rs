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

/// Logical dataset partitions used during sampling.
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

/// Ratio configuration for train/validation/test assignment.
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
    /// Validate that ratios sum to `1.0` (within epsilon).
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

/// Persisted epoch cursor metadata for one split.
#[derive(Clone, Debug, bitcode::Encode, bitcode::Decode)]
pub struct PersistedSplitMeta {
    /// Current epoch for this split.
    pub epoch: u64,
    /// Cursor offset within the epoch hash list.
    pub offset: u64,
    /// Checksum of the persisted hash list.
    pub hashes_checksum: u64,
}

/// Persisted deterministic epoch hash ordering for one split.
#[derive(Clone, Debug, bitcode::Encode, bitcode::Decode)]
pub struct PersistedSplitHashes {
    /// Checksum of `hashes`.
    pub checksum: u64,
    /// Deterministic per-epoch hash ordering.
    pub hashes: Vec<u64>,
}

/// Persisted sampler runtime state (cursors, recipe indices, RNG).
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

/// Split assignment backend.
///
/// Implementations map `RecordId` values to split labels deterministically.
pub trait SplitStore: Send + Sync {
    /// Return split label for `id` if known/derivable.
    fn label_for(&self, id: &RecordId) -> Option<SplitLabel>;
    /// Persist an explicit split assignment for `id`.
    fn upsert(&self, id: RecordId, label: SplitLabel) -> Result<(), SamplerError>;
    /// Return configured split ratios.
    fn ratios(&self) -> SplitRatios;
    /// Return the split label for `id`, creating/deriving one when needed.
    fn ensure(&self, id: RecordId) -> Result<SplitLabel, SamplerError>;
}

/// Persistence backend for epoch metadata and epoch hash orderings.
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

/// Persistence backend for sampler runtime state.
pub trait SamplerStateStore: Send + Sync {
    /// Load persisted sampler runtime state, if present.
    fn load_sampler_state(&self) -> Result<Option<PersistedSamplerState>, SamplerError>;
    /// Persist sampler runtime state.
    fn store_sampler_state(&self, state: &PersistedSamplerState) -> Result<(), SamplerError>;
}

/// In-memory split store with deterministic assignment derivation.
pub struct DeterministicSplitStore {
    ratios: SplitRatios,
    assignments: RwLock<HashMap<RecordId, SplitLabel>>,
    seed: u64,
    epoch_meta: RwLock<HashMap<SplitLabel, PersistedSplitMeta>>,
    epoch_hashes: RwLock<HashMap<SplitLabel, PersistedSplitHashes>>,
    sampler_state: RwLock<Option<PersistedSamplerState>>,
}

impl DeterministicSplitStore {
    /// Create an in-memory split store configured with `ratios` and `seed`.
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
/// Versioned metadata header stored in file-backed split stores.
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

/// File-backed split store for persistent runs.
///
/// Persists assignment metadata, epoch state, and sampler runtime state.
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
    use std::collections::HashMap;
    use tempfile::tempdir;

    #[test]
    fn split_ratios_reject_non_unit_sum() {
        let invalid = SplitRatios {
            train: 0.6,
            validation: 0.3,
            test: 0.3,
        };

        let err = DeterministicSplitStore::new(invalid, 1)
            .err()
            .expect("expected non-unit split ratios to fail");
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
        assert!(matches!(
            err,
            SamplerError::SplitStore(msg) if msg.contains("seed")
        ));
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

    #[test]
    fn bitcode_payload_requires_prefix() {
        let err = decode_bitcode_payload(&[0x00, 0x01]).unwrap_err();
        assert!(
            matches!(err, SamplerError::SplitStore(msg) if msg.contains("missing expected prefix"))
        );
    }

    #[test]
    fn file_store_round_trips_epoch_and_sampler_state() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("epoch_sampler_state.bin");
        let store = FileSplitStore::open(&path, SplitRatios::default(), 222).unwrap();

        assert!(store.load_epoch_hashes(SplitLabel::Test).unwrap().is_none());

        let mut epoch_meta = HashMap::new();
        epoch_meta.insert(
            SplitLabel::Train,
            PersistedSplitMeta {
                epoch: 3,
                offset: 7,
                hashes_checksum: 42,
            },
        );
        store.store_epoch_meta(&epoch_meta).unwrap();

        let loaded_meta = store.load_epoch_meta().unwrap();
        let loaded_train = loaded_meta.get(&SplitLabel::Train).unwrap();
        assert_eq!(loaded_train.epoch, 3);
        assert_eq!(loaded_train.offset, 7);
        assert_eq!(loaded_train.hashes_checksum, 42);

        let hashes = PersistedSplitHashes {
            checksum: 99,
            hashes: vec![10, 20, 30],
        };
        store
            .store_epoch_hashes(SplitLabel::Validation, &hashes)
            .unwrap();
        let loaded_hashes = store
            .load_epoch_hashes(SplitLabel::Validation)
            .unwrap()
            .unwrap();
        assert_eq!(loaded_hashes.checksum, 99);
        assert_eq!(loaded_hashes.hashes, vec![10, 20, 30]);

        let state = PersistedSamplerState {
            source_cycle_idx: 11,
            source_record_cursors: vec![("source_a".to_string(), 1)],
            source_epoch: 8,
            rng_state: 1234,
            triplet_recipe_rr_idx: 2,
            text_recipe_rr_idx: 5,
            source_stream_cursors: vec![("source_a".to_string(), 9)],
        };
        store.store_sampler_state(&state).unwrap();
        let loaded_state = store.load_sampler_state().unwrap().unwrap();
        assert_eq!(loaded_state.source_cycle_idx, 11);
        assert_eq!(loaded_state.source_epoch, 8);
        assert_eq!(loaded_state.rng_state, 1234);
        assert_eq!(loaded_state.triplet_recipe_rr_idx, 2);
        assert_eq!(loaded_state.text_recipe_rr_idx, 5);
        assert_eq!(
            loaded_state.source_record_cursors,
            vec![("source_a".to_string(), 1)]
        );
        assert_eq!(
            loaded_state.source_stream_cursors,
            vec![("source_a".to_string(), 9)]
        );
    }

    #[test]
    fn split_keys_and_labels_cover_helper_paths() {
        let key = split_key(&"abc".to_string());
        assert!(key.starts_with(SPLIT_PREFIX));

        assert!(matches!(decode_label(b"0"), Ok(SplitLabel::Train)));
        assert!(matches!(decode_label(b"1"), Ok(SplitLabel::Validation)));
        assert!(matches!(decode_label(b"2"), Ok(SplitLabel::Test)));
        assert!(decode_label(b"x").is_err());

        let epoch_meta_train = epoch_meta_key(SplitLabel::Train);
        let epoch_hashes_train = epoch_hashes_key(SplitLabel::Train);
        let epoch_hashes_test = epoch_hashes_key(SplitLabel::Test);
        assert!(epoch_meta_train.starts_with(EPOCH_META_PREFIX));
        assert!(epoch_hashes_train.starts_with(EPOCH_HASHES_PREFIX));
        assert!(epoch_hashes_test.starts_with(EPOCH_HASHES_PREFIX));
    }

    #[test]
    fn encode_decode_store_meta_roundtrip_and_corrupt_prefix_error() {
        let meta = StoreMeta {
            version: STORE_VERSION,
            seed: 55,
            ratios: SplitRatios::default(),
        };
        let encoded = encode_store_meta(&meta);
        let decoded = decode_store_meta(&encoded).unwrap();
        assert_eq!(decoded.version, STORE_VERSION);
        assert_eq!(decoded.seed, 55);

        let err = decode_store_meta(&[0x00, 0x01]).unwrap_err();
        assert!(matches!(
            err,
            SamplerError::SplitStore(msg) if msg.contains("missing expected prefix")
        ));
    }

    #[test]
    fn epoch_and_sampler_decoders_cover_tombstone_and_version_mismatch() {
        assert!(decode_epoch_meta(&[]).unwrap().is_none());
        assert!(
            decode_epoch_meta(&[EPOCH_RECORD_TOMBSTONE])
                .unwrap()
                .is_none()
        );
        assert!(decode_epoch_hashes(&[]).unwrap().is_none());
        assert!(
            decode_epoch_hashes(&[EPOCH_RECORD_TOMBSTONE])
                .unwrap()
                .is_none()
        );
        assert!(decode_sampler_state(&[]).unwrap().is_none());

        let meta_mismatch = decode_epoch_meta(&[EPOCH_META_RECORD_VERSION.wrapping_add(1), 1]);
        assert!(matches!(
            meta_mismatch,
            Err(SamplerError::SplitStore(msg)) if msg.contains("version mismatch")
        ));
        let hashes_mismatch = decode_epoch_hashes(&[EPOCH_HASH_RECORD_VERSION.wrapping_add(1), 1]);
        assert!(matches!(
            hashes_mismatch,
            Err(SamplerError::SplitStore(msg)) if msg.contains("version mismatch")
        ));
        let state_mismatch =
            decode_sampler_state(&[SAMPLER_STATE_RECORD_VERSION.wrapping_add(1), 1]);
        assert!(matches!(
            state_mismatch,
            Err(SamplerError::SplitStore(msg)) if msg.contains("version mismatch")
        ));
    }

    #[test]
    fn split_store_trait_methods_and_path_helpers_are_exercised() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("nested").join("store.bin");
        ensure_parent_dir(&file_path).unwrap();
        assert!(file_path.parent().unwrap().exists());

        let existing_dir_path = coerce_store_path(dir.path().to_path_buf());
        assert_eq!(existing_dir_path, dir.path().join(DEFAULT_STORE_FILENAME));

        let ratios = SplitRatios::default();
        let store = FileSplitStore::open(&file_path, ratios, 444).unwrap();
        assert!((store.ratios().train - ratios.train).abs() < 1e-6);
        store
            .upsert("record_1".to_string(), SplitLabel::Validation)
            .unwrap();
        let ensured = store.ensure("record_1".to_string()).unwrap();
        assert!(matches!(
            ensured,
            SplitLabel::Train | SplitLabel::Validation | SplitLabel::Test
        ));

        let mapped = map_store_err(io::Error::other("boom"));
        assert!(matches!(mapped, SamplerError::SplitStore(msg) if msg.contains("boom")));
    }

    #[test]
    fn epoch_and_sampler_encode_decode_roundtrips() {
        let meta = PersistedSplitMeta {
            epoch: 4,
            offset: 9,
            hashes_checksum: 21,
        };
        let encoded_meta = encode_epoch_meta(Some(&meta));
        let decoded_meta = decode_epoch_meta(&encoded_meta).unwrap().unwrap();
        assert_eq!(decoded_meta.epoch, 4);
        assert_eq!(decoded_meta.offset, 9);

        let hashes = PersistedSplitHashes {
            checksum: 7,
            hashes: vec![1, 2, 3],
        };
        let encoded_hashes = encode_epoch_hashes(&hashes);
        let decoded_hashes = decode_epoch_hashes(&encoded_hashes).unwrap().unwrap();
        assert_eq!(decoded_hashes.checksum, 7);
        assert_eq!(decoded_hashes.hashes, vec![1, 2, 3]);

        let state = PersistedSamplerState {
            source_cycle_idx: 1,
            source_record_cursors: vec![("s".to_string(), 2)],
            source_epoch: 3,
            rng_state: 4,
            triplet_recipe_rr_idx: 5,
            text_recipe_rr_idx: 6,
            source_stream_cursors: vec![("s".to_string(), 7)],
        };
        let encoded_state = encode_sampler_state(&state);
        let decoded_state = decode_sampler_state(&encoded_state).unwrap().unwrap();
        assert_eq!(decoded_state.source_cycle_idx, 1);
        assert_eq!(decoded_state.source_epoch, 3);
        assert_eq!(decoded_state.rng_state, 4);
    }

    #[test]
    fn deterministic_store_trait_methods_and_default_paths_work() {
        let ratios = SplitRatios::default();
        let store = DeterministicSplitStore::new(ratios, 999).unwrap();

        assert_eq!(store.ratios().train, ratios.train);

        let id = "source::record".to_string();
        let derived = store.label_for(&id).unwrap();
        store.upsert(id.clone(), SplitLabel::Validation).unwrap();
        assert_eq!(store.label_for(&id), Some(SplitLabel::Validation));
        assert!(matches!(
            derived,
            SplitLabel::Train | SplitLabel::Validation | SplitLabel::Test
        ));

        let mut meta = HashMap::new();
        meta.insert(
            SplitLabel::Test,
            PersistedSplitMeta {
                epoch: 1,
                offset: 2,
                hashes_checksum: 3,
            },
        );
        store.store_epoch_meta(&meta).unwrap();
        let loaded_meta = store.load_epoch_meta().unwrap();
        assert_eq!(loaded_meta.get(&SplitLabel::Test).unwrap().offset, 2);

        assert!(
            store
                .load_epoch_hashes(SplitLabel::Train)
                .unwrap()
                .is_none()
        );
        store
            .store_epoch_hashes(
                SplitLabel::Train,
                &PersistedSplitHashes {
                    checksum: 11,
                    hashes: vec![4, 5],
                },
            )
            .unwrap();
        assert_eq!(
            store
                .load_epoch_hashes(SplitLabel::Train)
                .unwrap()
                .unwrap()
                .checksum,
            11
        );

        assert!(store.load_sampler_state().unwrap().is_none());
        let sampler_state = PersistedSamplerState {
            source_cycle_idx: 1,
            source_record_cursors: vec![("s1".to_string(), 2)],
            source_epoch: 3,
            rng_state: 4,
            triplet_recipe_rr_idx: 5,
            text_recipe_rr_idx: 6,
            source_stream_cursors: vec![("s1".to_string(), 7)],
        };
        store.store_sampler_state(&sampler_state).unwrap();
        assert_eq!(
            store.load_sampler_state().unwrap().unwrap().source_epoch,
            sampler_state.source_epoch
        );

        let in_dir = FileSplitStore::default_path_in_dir("tmp-test-store");
        assert_eq!(
            in_dir.file_name().and_then(|name| name.to_str()),
            Some(DEFAULT_STORE_FILENAME)
        );
        let default_path = FileSplitStore::default_path();
        assert_eq!(
            default_path.file_name().and_then(|name| name.to_str()),
            Some(DEFAULT_STORE_FILENAME)
        );
    }

    #[test]
    fn file_store_metadata_mismatch_and_debug_paths_are_covered() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("meta_mismatch.bin");
        let ratios = SplitRatios::default();
        let store = FileSplitStore::open(&path, ratios, 123).unwrap();

        let debug_repr = format!("{store:?}");
        assert!(debug_repr.contains("FileSplitStore"));

        let wrong_version = StoreMeta {
            version: STORE_VERSION.wrapping_add(1),
            seed: 123,
            ratios,
        };
        let payload = encode_store_meta(&wrong_version);
        store.store.write(META_KEY, &payload).unwrap();
        drop(store);

        let err = FileSplitStore::open(&path, ratios, 123).unwrap_err();
        assert!(matches!(err, SamplerError::SplitStore(msg) if msg.contains("version mismatch")));

        let ratio_path = dir.path().join("ratio_mismatch.bin");
        let _baseline = FileSplitStore::open(&ratio_path, ratios, 777).unwrap();

        let different_ratios = SplitRatios {
            train: 0.7,
            validation: 0.2,
            test: 0.1,
        };
        let err = FileSplitStore::open(&ratio_path, different_ratios, 777).unwrap_err();
        assert!(matches!(err, SamplerError::SplitStore(msg) if msg.contains("ratios mismatch")));
    }

    #[test]
    fn split_decode_helpers_reject_corrupt_bitcode_payloads() {
        let store_meta_err = decode_store_meta(&[BITCODE_PREFIX, 0xFF, 0xEE]).unwrap_err();
        assert!(matches!(
            store_meta_err,
            SamplerError::SplitStore(msg) if msg.contains("failed to decode split store metadata")
        ));

        let epoch_meta_err =
            decode_epoch_meta(&[EPOCH_META_RECORD_VERSION, BITCODE_PREFIX, 0xFF]).unwrap_err();
        assert!(
            matches!(epoch_meta_err, SamplerError::SplitStore(msg) if msg.contains("corrupt epoch meta record"))
        );

        let epoch_hashes_err =
            decode_epoch_hashes(&[EPOCH_HASH_RECORD_VERSION, BITCODE_PREFIX, 0xFF]).unwrap_err();
        assert!(matches!(
            epoch_hashes_err,
            SamplerError::SplitStore(msg) if msg.contains("corrupt epoch hashes record")
        ));

        let sampler_state_err =
            decode_sampler_state(&[SAMPLER_STATE_RECORD_VERSION, BITCODE_PREFIX, 0xFF])
                .unwrap_err();
        assert!(matches!(
            sampler_state_err,
            SamplerError::SplitStore(msg) if msg.contains("corrupt sampler state record")
        ));
    }

    #[test]
    fn file_store_label_fallback_and_validation_keys_are_covered() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("labels.bin");
        let store = FileSplitStore::open(&path, SplitRatios::default(), 42).unwrap();

        let id = "bad_label_record".to_string();
        let expected = derive_label_for_id(&id, 42, SplitRatios::default());
        let key = split_key(&id);

        store.store.write(&key, b"x").unwrap();
        assert_eq!(store.label_for(&id), Some(expected));

        store.store.write(&key, b"1").unwrap();
        assert_eq!(store.label_for(&id), Some(SplitLabel::Validation));

        let meta_validation = epoch_meta_key(SplitLabel::Validation);
        let hashes_validation = epoch_hashes_key(SplitLabel::Validation);
        assert!(meta_validation.starts_with(EPOCH_META_PREFIX));
        assert!(hashes_validation.starts_with(EPOCH_HASHES_PREFIX));
        assert!(meta_validation.ends_with(b"validation"));
        assert!(hashes_validation.ends_with(b"validation"));
    }

    #[test]
    fn ensure_parent_dir_allows_plain_file_names() {
        ensure_parent_dir(Path::new("split_store_local.bin")).unwrap();
        let coerced = coerce_store_path(PathBuf::from("explicit_store.bin"));
        assert_eq!(coerced, PathBuf::from("explicit_store.bin"));
    }
}
