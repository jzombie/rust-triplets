use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;

pub fn stable_hash_with(f: impl FnOnce(&mut DefaultHasher)) -> u64 {
    let mut hasher = DefaultHasher::new();
    f(&mut hasher);
    hasher.finish()
}

pub fn stable_hash_str(seed: u64, value: &str) -> u64 {
    stable_hash_with(|hasher| {
        seed.hash(hasher);
        value.hash(hasher);
    })
}

pub fn stable_hash_path(seed: u64, path: &Path) -> u64 {
    stable_hash_with(|hasher| {
        seed.hash(hasher);
        path.to_string_lossy().hash(hasher);
    })
}

/// Derive a per-epoch seed by mixing an epoch counter (or constant offset) into a base seed.
///
/// All seed derivations that incorporate an epoch value must go through this function so the
/// derivation strategy can be changed in one place.  Both the source-shuffling path
/// (`base_seed ^ source_epoch`) and the epoch-tracker initialisation path
/// (`base_seed ^ EPOCH_SEED_OFFSET`) are expressed as `derive_epoch_seed(base_seed, epoch)`.
pub fn derive_epoch_seed(base_seed: u64, epoch: u64) -> u64 {
    base_seed ^ epoch
}
