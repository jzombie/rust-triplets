use crate::errors::SamplerError;
use crate::hash::stable_hash_with;
use crate::splits::{EpochStateStore, PersistedSplitMeta, SplitLabel};
use crate::types::{RecordId, SourceId};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

#[derive(Debug, Default)]
struct SplitEpochState {
    epoch: u64,
    /// Vector of (RecordId, SourceId) tuples. SourceId is used for interleaved sorting.
    population: Vec<(RecordId, SourceId)>,
    order: Vec<usize>,
    offset: usize,
    dirty_order: bool,
    hashes_checksum: u64,
}

impl SplitEpochState {
    fn reset_epoch(&mut self) {
        if self.population.is_empty() {
            self.order.clear();
            self.offset = 0;
            self.dirty_order = false;
            return;
        }
        self.epoch = self.epoch.saturating_add(1);
        self.offset = 0;
        self.dirty_order = true;
    }

    fn ensure_order(&mut self, label: SplitLabel, seed: u64) {
        if !self.dirty_order && !self.order.is_empty() {
            return;
        }
        self.order.clear();
        self.order.reserve(self.population.len());

        // Group indices by source
        let mut by_source: HashMap<&str, Vec<usize>> = HashMap::new();
        for (idx, (_, source)) in self.population.iter().enumerate() {
            by_source.entry(source).or_default().push(idx);
        }

        // Sort each source bucket deterministically based on seed + epoch + label
        let mut sorted_sources: Vec<(&str, Vec<usize>)> = by_source.into_iter().collect();
        // Deterministically shuffle sources per epoch without extra state.
        sorted_sources.sort_by_key(|(source, _)| shuffle_key(source, self.epoch, label, seed));

        for (_, indices) in sorted_sources.iter_mut() {
            // Sort indices based on the shuffle key of their record ID
            indices.sort_by_cached_key(|&idx| {
                let (id, _) = &self.population[idx];
                shuffle_key(id, self.epoch, label, seed)
            });
        }

        // Calculate max length to oversample smaller sources
        let max_len = sorted_sources
            .iter()
            .map(|(_, indices)| indices.len())
            .max()
            .unwrap_or(0);

        // Interleave with oversampling for Round-Robin
        for i in 0..max_len {
            for (_, indices) in &sorted_sources {
                if indices.is_empty() {
                    continue;
                }
                // Use modulo to cycle through smaller sources
                let idx = indices[i % indices.len()];
                self.order.push(idx);
            }
        }

        self.dirty_order = false;
        if self.offset > self.order.len() {
            self.offset = self.order.len();
        }
    }
}

/// Tracks per-split epoch order and offsets for deterministic sampling.
pub struct EpochTracker {
    enabled: bool,
    backend: Option<Arc<dyn EpochStateStore>>,
    seed: u64,
    loaded: bool,
    dirty: bool,
    splits: HashMap<SplitLabel, SplitEpochState>,
}

impl EpochTracker {
    pub fn new(enabled: bool, backend: Option<Arc<dyn EpochStateStore>>, seed: u64) -> Self {
        Self {
            enabled,
            backend: if enabled { backend } else { None },
            seed,
            loaded: false,
            dirty: false,
            splits: HashMap::new(),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub(crate) fn force_epoch(&mut self, epoch: u64) {
        if !self.enabled {
            return;
        }
        for state in self.splits.values_mut() {
            state.epoch = epoch;
            state.offset = 0;
            state.dirty_order = true;
        }
        self.dirty = self.backend.is_some();
    }

    pub fn ensure_loaded(&mut self) -> Result<(), SamplerError> {
        if !self.enabled || self.loaded {
            return Ok(());
        }
        let Some(store) = self.backend.as_ref() else {
            self.loaded = true;
            return Ok(());
        };
        let meta = store.load_epoch_meta()?;
        let mut splits = HashMap::new();
        for (label, entry) in meta {
            let state = SplitEpochState {
                epoch: entry.epoch,
                offset: entry.offset as usize,
                hashes_checksum: entry.hashes_checksum,
                dirty_order: true,
                ..SplitEpochState::default()
            };
            splits.insert(label, state);
        }
        self.splits = splits;
        self.loaded = true;
        self.dirty = false;
        Ok(())
    }

    pub fn reconcile(
        &mut self,
        target_split: SplitLabel,
        records: &HashMap<SplitLabel, Vec<(RecordId, SourceId)>>,
    ) {
        self.splits.retain(|label, _| *label == target_split);
        let mut changed = false;
        let ids = records.get(&target_split).cloned().unwrap_or_default();
        let split_changed = self.reconcile_split(target_split, ids);
        changed |= split_changed;
        if changed {
            self.dirty = self.backend.is_some();
        }
    }

    fn reconcile_split(&mut self, label: SplitLabel, mut ids: Vec<(RecordId, SourceId)>) -> bool {
        let state = self.splits.entry(label).or_default();
        ids.sort_unstable();
        ids.dedup();
        if ids == state.population {
            return false;
        }

        let had_population = !state.population.is_empty();
        let new_ids_checksum = population_checksum(&ids);

        // If this is a fresh load (empty population) and checksum matches,
        // we can trust the persisted offset/state completely.
        if !had_population && state.hashes_checksum == new_ids_checksum {
            state.population = ids;
            state.dirty_order = true;
            // No need to adjust offset or checking consumed_ids,
            // as the population is identical to what was persisted.
            // ensure_order will happen lazily on next_record.
            return true;
        }

        let mut consumed_ids: HashSet<RecordId> = HashSet::new();
        if had_population {
            state.ensure_order(label, self.seed);
            let consumed = state.offset.min(state.order.len());
            consumed_ids.extend(
                state.order[..consumed]
                    .iter()
                    .filter_map(|&idx| state.population.get(idx).map(|(id, _)| id.clone())),
            );
        }

        state.population = ids;
        state.dirty_order = true;

        if state.population.is_empty() {
            state.order.clear();
            state.offset = 0;
            state.dirty_order = false;
            return true;
        }

        state.ensure_order(label, self.seed);
        if !consumed_ids.is_empty() {
            let mut reordered = Vec::with_capacity(state.order.len());
            for idx in &state.order {
                if let Some((id, _)) = state.population.get(*idx)
                    && !consumed_ids.contains(id)
                {
                    reordered.push(*idx);
                }
            }
            for idx in &state.order {
                if let Some((id, _)) = state.population.get(*idx)
                    && consumed_ids.contains(id)
                {
                    reordered.push(*idx);
                }
            }
            state.order = reordered;
            state.offset = 0;
        } else {
            state.offset = state.offset.min(state.order.len());
        }

        let new_checksum = population_checksum(&state.population);

        if new_checksum != state.hashes_checksum {
            state.hashes_checksum = new_checksum;
        }
        true
    }

    pub fn next_record(&mut self, target_split: SplitLabel) -> Option<String> {
        let id = self.next_from_split(target_split)?;
        self.dirty |= self.backend.is_some();
        Some(id)
    }

    fn next_from_split(&mut self, label: SplitLabel) -> Option<String> {
        let state = self.splits.get_mut(&label)?;
        if state.population.is_empty() {
            return None;
        }
        state.ensure_order(label, self.seed);
        if state.offset >= state.order.len() {
            state.reset_epoch();
            state.ensure_order(label, self.seed);
        }
        if state.offset >= state.order.len() {
            return None;
        }
        let idx = state.order[state.offset];
        state.offset += 1;
        state.population.get(idx).map(|(id, _)| id.clone())
    }

    pub fn persist(&mut self) -> Result<(), SamplerError> {
        if !self.enabled || !self.dirty {
            return Ok(());
        }
        let Some(store) = self.backend.as_ref() else {
            self.dirty = false;
            return Ok(());
        };
        let mut meta = HashMap::new();
        for (label, state) in &self.splits {
            meta.insert(
                *label,
                PersistedSplitMeta {
                    epoch: state.epoch,
                    offset: state.offset as u64,
                    hashes_checksum: if state.population.is_empty() {
                        0
                    } else {
                        state.hashes_checksum
                    },
                },
            );
        }
        store.store_epoch_meta(&meta)?;
        self.dirty = false;
        Ok(())
    }
}

fn shuffle_key(id: &str, epoch: u64, label: SplitLabel, seed: u64) -> u64 {
    stable_hash_with(|hasher| {
        id.hash(hasher);
        epoch.hash(hasher);
        seed.hash(hasher);
        label_discriminant(label).hash(hasher);
    })
}

fn id_fingerprint(item: &(RecordId, SourceId)) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    item.0.hash(&mut hasher);
    item.1.hash(&mut hasher);
    hasher.finish()
}

fn population_checksum(ids: &[(RecordId, SourceId)]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let mut fingerprints: Vec<u64> = ids.iter().map(id_fingerprint).collect();
    fingerprints.sort_unstable();
    for fp in fingerprints {
        fp.hash(&mut hasher);
    }
    hasher.finish()
}

fn label_discriminant(label: SplitLabel) -> u8 {
    match label {
        SplitLabel::Train => 0,
        SplitLabel::Validation => 1,
        SplitLabel::Test => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::splits::{DeterministicSplitStore, SplitRatios};
    use std::collections::HashMap;

    #[test]
    fn persists_offsets_across_restarts() {
        let backend = Arc::new(DeterministicSplitStore::new(SplitRatios::default(), 7).unwrap());
        let mut tracker = EpochTracker::new(true, Some(backend.clone()), 99);
        tracker.ensure_loaded().unwrap();
        let records = single_split_records(vec!["a", "b", "c", "d"]);
        tracker.reconcile(SplitLabel::Train, &records);
        let _first = tracker.next_record(SplitLabel::Train).unwrap();
        let _second = tracker.next_record(SplitLabel::Train).unwrap();
        tracker.persist().unwrap();

        let mut resumed = EpochTracker::new(true, Some(backend.clone()), 99);
        resumed.ensure_loaded().unwrap();
        resumed.reconcile(SplitLabel::Train, &records);
        let resumed_id = resumed.next_record(SplitLabel::Train).unwrap();
        let expected_id = tracker.next_record(SplitLabel::Train).unwrap();
        assert_eq!(resumed_id, expected_id);
    }

    #[test]
    fn advances_epoch_after_full_pass() {
        let backend = Arc::new(DeterministicSplitStore::new(SplitRatios::default(), 11).unwrap());
        let mut tracker = EpochTracker::new(true, Some(backend.clone()), 1234);
        tracker.ensure_loaded().unwrap();
        let records = single_split_records(vec!["r1", "r2", "r3"]);
        tracker.reconcile(SplitLabel::Train, &records);
        for _ in 0..records.get(&SplitLabel::Train).unwrap().len() {
            assert!(tracker.next_record(SplitLabel::Train).is_some());
        }
        tracker.persist().unwrap();
        // next call should trigger a new epoch
        assert!(tracker.next_record(SplitLabel::Train).is_some());
        tracker.persist().unwrap();
        let meta = backend.load_epoch_meta().unwrap();
        let split_meta = meta.get(&SplitLabel::Train).unwrap();
        assert!(split_meta.epoch >= 1);
    }

    #[test]
    fn new_records_inserted_after_restart_are_not_skipped() {
        let backend = Arc::new(DeterministicSplitStore::new(SplitRatios::default(), 17).unwrap());
        let mut tracker = EpochTracker::new(true, Some(backend.clone()), 5150);
        tracker.ensure_loaded().unwrap();

        let mut records = single_split_records(vec!["alpha", "beta"]);
        tracker.reconcile(SplitLabel::Train, &records);
        let first = tracker.next_record(SplitLabel::Train).unwrap();
        tracker.persist().unwrap();

        // Restart and ensure persisted offset is honored.
        let mut tracker = EpochTracker::new(true, Some(backend.clone()), 5150);
        tracker.ensure_loaded().unwrap();
        tracker.reconcile(SplitLabel::Train, &records);
        let state = tracker.splits.get(&SplitLabel::Train).unwrap();
        assert_eq!(
            state.offset, 1,
            "persisted offset should be restored before sampling"
        );
        let second = tracker.next_record(SplitLabel::Train).unwrap();
        assert_ne!(second, first);

        // Add a new record mid-epoch and ensure it appears before already consumed IDs repeat.
        records
            .get_mut(&SplitLabel::Train)
            .unwrap()
            .push(("gamma".to_string(), "unit".to_string()));
        tracker.reconcile(SplitLabel::Train, &records);

        let mut remainder = Vec::new();
        let mut saw_first_repeat = false;
        let max_checks = records.get(&SplitLabel::Train).unwrap().len() * 2;
        for _ in 0..max_checks {
            if let Some(id) = tracker.next_record(SplitLabel::Train) {
                if id == first {
                    saw_first_repeat = true;
                    break;
                }
                remainder.push(id);
            } else {
                break;
            }
        }
        assert!(
            saw_first_repeat,
            "expected the previously consumed record to reappear"
        );
        assert!(
            remainder.contains(&"gamma".to_string()),
            "new record should be scheduled before epoch reset"
        );
        assert!(
            !remainder.contains(&first),
            "previously consumed record resurfaced before completing the epoch"
        );
    }

    fn single_split_records(ids: Vec<&str>) -> HashMap<SplitLabel, Vec<(RecordId, SourceId)>> {
        let mut map = HashMap::new();
        map.insert(
            SplitLabel::Train,
            ids.into_iter()
                .map(|id| (id.to_string(), "unit".to_string()))
                .collect(),
        );
        map
    }

    #[test]
    fn interleaved_oversampling_works() {
        let backend = Arc::new(DeterministicSplitStore::new(SplitRatios::default(), 42).unwrap());
        // Seed 42 for deterministic shuffle
        let mut tracker = EpochTracker::new(true, Some(backend.clone()), 42);
        tracker.ensure_loaded().unwrap();

        let mut ids = Vec::new();
        // create 10 items for source "large" (0..10)
        for i in 0..10 {
            ids.push((format!("large_{}", i), "large".to_string()));
        }
        // create 2 items for source "small" (0..2)
        for i in 0..2 {
            ids.push((format!("small_{}", i), "small".to_string()));
        }

        let mut map = HashMap::new();
        map.insert(SplitLabel::Train, ids);

        tracker.reconcile(SplitLabel::Train, &map);

        let mut output_sequence = Vec::new();
        // With oversampling enabled, we iterate up to max_len (10) for ALL sources.
        // Since we have 2 sources, total items = 10 * 2 = 20.
        for _ in 0..20 {
            // Check exactly one epoch
            if let Some(id) = tracker.next_record(SplitLabel::Train) {
                output_sequence.push(id);
            } else {
                break;
            }
        }

        assert_eq!(
            output_sequence.len(),
            20,
            "Epoch should contain max_source_len * num_sources items"
        );

        let mut small_counts = HashMap::new();
        let mut large_counts = HashMap::new();

        for (i, id) in output_sequence.iter().enumerate() {
            if id.starts_with("small") {
                *small_counts.entry(id.clone()).or_insert(0) += 1;
                // Sources are sorted by name: "large" (even index) then "small" (odd index).
                assert_eq!(i % 2, 1, "Small source should be at odd indices");
            } else {
                *large_counts.entry(id.clone()).or_insert(0) += 1;
                assert_eq!(i % 2, 0, "Large source should be at even indices");
            }
        }

        // Verify large source items appear exactly once
        for i in 0..10 {
            let key = format!("large_{}", i);
            assert_eq!(
                large_counts.get(&key),
                Some(&1),
                "Large items should be seen exactly once"
            );
        }

        // Verify small source items appear multiple times (5 times each, since 10/2 = 5)
        for i in 0..2 {
            let key = format!("small_{}", i);
            assert_eq!(
                small_counts.get(&key),
                Some(&5),
                "Small items should be upsampled"
            );
        }
    }

    #[test]
    fn interleaves_three_sources_with_oversampling() {
        let backend = Arc::new(DeterministicSplitStore::new(SplitRatios::default(), 9).unwrap());
        let mut tracker = EpochTracker::new(true, Some(backend.clone()), 9);
        tracker.ensure_loaded().unwrap();

        let mut ids = Vec::new();
        // 4 items for alpha (max source)
        for i in 0..4 {
            ids.push((format!("alpha_{}", i), "alpha".to_string()));
        }
        // 2 items for beta
        for i in 0..2 {
            ids.push((format!("beta_{}", i), "beta".to_string()));
        }
        // 1 item for gamma
        ids.push(("gamma_0".to_string(), "gamma".to_string()));

        let mut map = HashMap::new();
        map.insert(SplitLabel::Train, ids);
        tracker.reconcile(SplitLabel::Train, &map);

        let mut output_sequence = Vec::new();
        // max_len = 4, sources = 3 => 12 total items per epoch
        for _ in 0..12 {
            if let Some(id) = tracker.next_record(SplitLabel::Train) {
                output_sequence.push(id);
            } else {
                break;
            }
        }

        assert_eq!(
            output_sequence.len(),
            12,
            "Epoch should contain max_source_len * num_sources items"
        );

        let mut alpha_counts = HashMap::new();
        let mut beta_counts = HashMap::new();
        let mut gamma_counts = HashMap::new();

        for (i, id) in output_sequence.iter().enumerate() {
            if id.starts_with("alpha") {
                *alpha_counts.entry(id.clone()).or_insert(0) += 1;
                assert_eq!(i % 3, 0, "Alpha source should be at index % 3 == 0");
            } else if id.starts_with("beta") {
                *beta_counts.entry(id.clone()).or_insert(0) += 1;
                assert_eq!(i % 3, 1, "Beta source should be at index % 3 == 1");
            } else {
                *gamma_counts.entry(id.clone()).or_insert(0) += 1;
                assert_eq!(i % 3, 2, "Gamma source should be at index % 3 == 2");
            }
        }

        // Alpha items appear exactly once
        for i in 0..4 {
            let key = format!("alpha_{}", i);
            assert_eq!(
                alpha_counts.get(&key),
                Some(&1),
                "Alpha items should be seen exactly once"
            );
        }

        // Beta items appear twice each (4/2 = 2)
        for i in 0..2 {
            let key = format!("beta_{}", i);
            assert_eq!(
                beta_counts.get(&key),
                Some(&2),
                "Beta items should be upsampled evenly"
            );
        }

        // Gamma item appears 4 times (4/1 = 4)
        assert_eq!(
            gamma_counts.get("gamma_0"),
            Some(&4),
            "Gamma item should be upsampled to match max source"
        );
    }

    #[test]
    fn reconcile_prunes_non_target_split_state() {
        let backend = Arc::new(DeterministicSplitStore::new(SplitRatios::default(), 3).unwrap());
        let mut tracker = EpochTracker::new(true, Some(backend.clone()), 3);
        tracker.ensure_loaded().unwrap();

        let mut records: HashMap<SplitLabel, Vec<(RecordId, SourceId)>> = HashMap::new();
        // Train split: two sources
        records.insert(
            SplitLabel::Train,
            vec![
                ("train_a0".to_string(), "alpha".to_string()),
                ("train_b0".to_string(), "beta".to_string()),
            ],
        );
        // Validation split: two sources
        records.insert(
            SplitLabel::Validation,
            vec![
                ("val_a0".to_string(), "alpha".to_string()),
                ("val_b0".to_string(), "beta".to_string()),
            ],
        );

        tracker.reconcile(SplitLabel::Validation, &records);
        assert!(tracker.next_record(SplitLabel::Validation).is_some());

        tracker.reconcile(SplitLabel::Train, &records);
        assert!(tracker.splits.contains_key(&SplitLabel::Train));
        assert!(!tracker.splits.contains_key(&SplitLabel::Validation));

        let next = tracker.next_record(SplitLabel::Train).unwrap();
        assert!(next.starts_with("train_"));
    }
}
