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
