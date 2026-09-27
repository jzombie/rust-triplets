use siphasher::sip::SipHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;

/// Run `f` against a fresh [`SipHasher`] and return the resulting digest.
///
/// Uses the pinned `siphasher` crate (SipHash-2-4) instead of
/// `std::collections::hash_map::DefaultHasher`, guaranteeing stable output
/// across Rust compiler versions and platforms.
pub fn stable_hash_with(f: impl FnOnce(&mut SipHasher)) -> u64 {
    let mut hasher = SipHasher::new();
    f(&mut hasher);
    hasher.finish()
}

/// Return a deterministic u64 hash of `value` mixed with `seed`.
///
/// Suitable for deriving per-record RNG seeds where `seed` is a global base
/// (e.g. a shuffle seed constant) and `value` is a unique identifier such as
/// a ticker symbol or record id.
pub fn stable_hash_str(seed: u64, value: &str) -> u64 {
    stable_hash_with(|hasher| {
        seed.hash(hasher);
        value.hash(hasher);
    })
}

/// Return a deterministic u64 hash of a composite record identity mixed with
/// `seed`.
///
/// Only the id part feeds the hasher — deliberately, not by omission: this
/// keeps deterministic sequences (split assignment, orderings, cursor
/// offsets) bit-identical for existing single-source corpora, so the
/// composite-key migration changes coexistence, not behavior. Labels still
/// conform to ratios per dataset; same-id records across sources simply share
/// an assignment, which is harmless (pools and cursors stay per-source).
pub fn stable_hash_key(seed: u64, key: &crate::types::RecordKey) -> u64 {
    stable_hash_str(seed, &key.id)
}

/// Return a deterministic u64 hash selecting a chunk-window cursor offset for
/// a (record, section) pair.
///
/// Replicates the historical `{id}::{section}` preimage exactly, so cursor
/// sequences are unchanged for existing corpora.
pub fn stable_hash_key_section(
    seed: u64,
    key: &crate::types::RecordKey,
    section_idx: usize,
) -> u64 {
    stable_hash_str(seed, &format!("{}::{section_idx}", key.id))
}

/// Return a deterministic u64 hash of the string form of `path` mixed with `seed`.
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
/// (`base_seed ^ epoch`) and the epoch-tracker initialisation path
/// (`base_seed ^ EPOCH_SEED_OFFSET`) are expressed as `derive_epoch_seed(base_seed, epoch)`.
pub fn derive_epoch_seed(base_seed: u64, epoch: u64) -> u64 {
    base_seed ^ epoch
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ensures the pinned SipHash-2-4 produces a known digest for a fixed input.
    /// If this test fails, the hasher has changed — update the expected value.
    #[test]
    fn stable_hash_str_is_deterministic() {
        let result = stable_hash_str(42, "hello");
        assert_eq!(result, 16678829552985060110);
    }

    #[test]
    fn stable_hash_with_produces_consistent_results() {
        let a = stable_hash_str(0, "test");
        let b = stable_hash_str(0, "test");
        assert_eq!(a, b);
    }
}
