use crate::config::SamplerConfig;
use crate::data::DataRecord;
use crate::errors::SamplerError;
use crate::hash::derive_epoch_seed;
use crate::source::{DataSource, SourceCursor, SourceSnapshot};
use crate::types::{RecordId, SourceId};
use chrono::Utc;
use indexmap::IndexMap;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread;
use std::time::Duration;
use tracing::debug;

/// Thread-safe in-memory cache of ingested records keyed by record id.
#[derive(Clone)]
pub struct RecordCache {
    inner: Arc<RwLock<RecordCacheInner>>,
    notifier: Arc<(Mutex<CacheStats>, Condvar)>,
}

/// Internal mutable cache storage behind `RecordCache` locks.
struct RecordCacheInner {
    records: IndexMap<RecordId, CachedRecord>,
    order: VecDeque<RecordId>,
    max_records: usize,
    next_version: u64,
}

/// Internal cache entry plus monotonic version marker.
struct CachedRecord {
    record: DataRecord,
    version: u64,
}

/// Internal ingest notification counters.
#[derive(Default)]
struct CacheStats {
    ingests: u64,
}

impl RecordCache {
    /// Create a cache capped to at most `max_records` live records.
    pub fn new(max_records: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(RecordCacheInner {
                records: IndexMap::new(),
                order: VecDeque::new(),
                max_records,
                next_version: 0,
            })),
            notifier: Arc::new((Mutex::new(CacheStats::default()), Condvar::new())),
        }
    }

    /// Ingest a batch of records, replacing existing entries by id.
    pub fn ingest<I>(&self, records: I)
    where
        I: IntoIterator<Item = DataRecord>,
    {
        let mut batch: Vec<DataRecord> = records.into_iter().collect();
        if batch.is_empty() {
            return;
        }
        let mut inner = self.inner.write().expect("record cache poisoned");
        inner.ingest_batch(&mut batch);
        drop(inner);
        let (lock, cvar) = &*self.notifier;
        let mut stats = lock.lock().expect("record cache stats poisoned");
        stats.ingests = stats.ingests.saturating_add(1);
        cvar.notify_all();
    }

    /// Remove all cached records.
    pub fn clear(&self) {
        let mut inner = self.inner.write().expect("record cache poisoned");
        inner.records.clear();
        inner.order.clear();
    }

    /// Return a cloned snapshot of current cached records.
    pub fn snapshot(&self) -> Vec<DataRecord> {
        let inner = self.inner.read().expect("record cache poisoned");
        inner
            .records
            .values()
            .map(|entry| entry.record.clone())
            .collect()
    }

    /// Return the number of completed ingest operations.
    pub fn ingest_count(&self) -> u64 {
        let (lock, _) = &*self.notifier;
        lock.lock().expect("record cache stats poisoned").ingests
    }

    /// Wait until ingest count exceeds `last_seen`, or until timeout elapses.
    pub fn wait_for_ingest(&self, last_seen: u64, timeout: Duration) -> u64 {
        let (lock, cvar) = &*self.notifier;
        let mut stats = lock.lock().expect("record cache stats poisoned");
        while stats.ingests <= last_seen {
            let result = cvar
                .wait_timeout(stats, timeout)
                .expect("record cache stats poisoned");
            stats = result.0;
            if result.1.timed_out() {
                break;
            }
        }
        stats.ingests
    }

    /// Wait indefinitely until ingest count exceeds `last_seen`.
    pub fn wait_for_ingest_blocking(&self, last_seen: u64) -> u64 {
        let (lock, cvar) = &*self.notifier;
        let mut stats = lock.lock().expect("record cache stats poisoned");
        while stats.ingests <= last_seen {
            stats = cvar.wait(stats).expect("record cache stats poisoned");
        }
        stats.ingests
    }

    /// Returns `true` when the cache has no records.
    pub fn is_empty(&self) -> bool {
        let inner = self.inner.read().expect("record cache poisoned");
        inner.records.is_empty()
    }

    /// Return the number of records currently cached.
    pub fn len(&self) -> usize {
        let inner = self.inner.read().expect("record cache poisoned");
        inner.records.len()
    }
}

impl RecordCacheInner {
    fn ingest_batch(&mut self, records: &mut Vec<DataRecord>) {
        for record in records.drain(..) {
            self.next_version = self.next_version.saturating_add(1);
            let record_id = record.id.clone();
            if self.records.contains_key(&record_id) {
                if let Some(entry) = self.records.get_mut(&record_id) {
                    entry.record = record;
                    entry.version = self.next_version;
                }
                Self::refresh_order(&mut self.order, &record_id);
                self.order.push_back(record_id);
            } else {
                self.order.push_back(record_id.clone());
                self.records.insert(
                    record_id,
                    CachedRecord {
                        record,
                        version: self.next_version,
                    },
                );
            }
            self.enforce_limit();
        }
    }

    fn enforce_limit(&mut self) {
        if self.max_records == 0 {
            self.records.clear();
            self.order.clear();
            return;
        }
        while self.records.len() > self.max_records {
            if let Some(oldest) = self.order.pop_front() {
                self.records.swap_remove(&oldest);
            } else {
                break;
            }
        }
    }

    fn refresh_order(order: &mut VecDeque<RecordId>, id: &RecordId) {
        if order.is_empty() {
            return;
        }
        if let Some(pos) = order.iter().position(|existing| existing == id) {
            order.remove(pos);
        }
    }
}

/// Coordinates on-demand source refresh and per-source-cache population.
pub struct IngestionManager {
    sources: Vec<SourceState>,
    max_records: usize,
    sampler_config: SamplerConfig,
    /// Current source epoch used to vary per-source permutation seeds across epochs.
    source_epoch: u64,
    /// Monotonic generation incremented whenever at least one source is refreshed.
    source_refresh_generation: u64,
    /// Source ids refreshed during the most recent `refresh_all_internal` call.
    ///
    /// This is updated even when cache ingest does not change, and is cleared when
    /// no source refresh occurs in that cycle.
    last_refreshed_sources: Vec<SourceId>,
    /// Rotating start index for the round-robin buffer drain.  Instead of always
    /// draining from source 0 (which starves high-index sources of refresh
    /// opportunities), each cycle begins at this position and advances by one.
    /// Over N cycles every source is drained first exactly once.
    drain_start: usize,
}

#[derive(Clone, Debug, Default)]
/// Last-refresh telemetry captured per source.
pub struct SourceRefreshStats {
    /// Duration of the most recent refresh in milliseconds.
    pub last_refresh_ms: u128,
    /// Number of records returned by the most recent refresh.
    pub last_record_count: usize,
    /// Throughput estimate from the most recent refresh.
    pub last_records_per_sec: f64,
    /// Last refresh error message, if any.
    pub last_error: Option<String>,
    /// Total refresh failures seen for this source.
    pub error_count: u64,
}

impl IngestionManager {
    /// Create a new ingestion manager that ingests on demand.
    pub fn new(max_records: usize, sampler_config: SamplerConfig) -> Self {
        Self {
            sources: Vec::new(),
            max_records,
            sampler_config,
            source_epoch: 0,
            source_refresh_generation: 0,
            last_refreshed_sources: Vec::new(),
            drain_start: 0,
        }
    }

    /// Return a monotonic generation for source refresh cycles.
    pub fn source_refresh_generation(&self) -> u64 {
        self.source_refresh_generation
    }

    /// Return source ids refreshed by the most recent refresh cycle.
    pub fn last_refreshed_sources(&self) -> &[SourceId] {
        &self.last_refreshed_sources
    }

    /// Update the current source epoch.
    ///
    /// Only changes the epoch value so subsequent `refresh` calls pass
    /// `seed ^ epoch` to sources, producing a different permutation.
    /// Stream cursors are intentionally NOT reset here — the cursor is a raw
    /// I/O offset into the source's stream and must continue advancing so
    /// every record is eventually fetched (resetting it would repeat the
    /// leading slice of the source on every epoch boundary).
    pub(crate) fn set_source_epoch(&mut self, epoch: u64) {
        self.source_epoch = epoch;
    }

    /// Return the current source epoch.
    #[cfg(test)]
    pub fn source_epoch(&self) -> u64 {
        self.source_epoch
    }

    /// Reset all raw source stream cursors and drain per-source buffers.
    ///
    /// Use this only when starting a deterministic replay from a specific
    /// epoch (e.g. explicit `set_epoch` calls). A clean-start reset ensures
    /// the new permutation begins at position 0 of the permuted index space.
    pub(crate) fn reset_stream_cursors(&mut self) {
        for state in &mut self.sources {
            state.cursor = None;
            state.buffer.clear();
            state.cache.clear();
        }
    }

    /// Register a source for on-demand ingestion.
    pub fn register_source(&mut self, source: Box<dyn DataSource + 'static>) {
        let cache = RecordCache::new(self.max_records);
        self.sources.push(SourceState {
            source,
            cursor: None,
            buffer: VecDeque::new(),
            cache,
            stats: SourceRefreshStats::default(),
        });
    }

    /// Load persisted per-source stream cursors.
    pub fn load_cursors(&mut self, cursors: &[(SourceId, u64)]) {
        if cursors.is_empty() {
            return;
        }
        let mut map = std::collections::HashMap::with_capacity(cursors.len());
        for (id, revision) in cursors {
            map.insert(id.as_str(), *revision);
        }
        for state in &mut self.sources {
            if let Some(revision) = map.get(state.source.id()) {
                state.cursor = Some(SourceCursor {
                    last_seen: Utc::now(),
                    revision: *revision,
                });
            }
        }
    }

    /// Snapshot current per-source stream cursors.
    pub fn snapshot_cursors(&self) -> Vec<(SourceId, u64)> {
        let mut out = Vec::new();
        for state in &self.sources {
            if let Some(cursor) = state.cursor.as_ref() {
                out.push((state.source.id().to_string(), cursor.revision));
            }
        }
        out
    }

    /// Return latest refresh telemetry for each registered source.
    pub fn source_refresh_stats(&self) -> Vec<(SourceId, SourceRefreshStats)> {
        self.sources
            .iter()
            .map(|state| (state.source.id().to_string(), state.stats.clone()))
            .collect()
    }

    /// Return a flat snapshot of every record currently in all per-source caches.
    ///
    /// Records are cloned in source order; the `source` field is guaranteed
    /// to be set (it is normalised in `refresh_all_internal`).
    pub fn all_records_snapshot(&self) -> Vec<DataRecord> {
        self.sources
            .iter()
            .flat_map(|s| s.cache.snapshot())
            .collect()
    }

    /// Returns `true` when ALL per-source caches are empty.
    pub fn all_caches_empty(&self) -> bool {
        self.sources.iter().all(|s| s.cache.is_empty())
    }

    /// Returns the total number of records across all per-source caches.
    pub fn all_records_len(&self) -> usize {
        self.sources.iter().map(|s| s.cache.len()).sum()
    }

    /// Returns the sum of ingest counts across all per-source caches.
    ///
    /// Used as a monotonic proxy to detect whether any cache has been updated
    /// since the last sync.
    pub fn total_ingest_count(&self) -> u64 {
        self.sources.iter().map(|s| s.cache.ingest_count()).sum()
    }

    /// Refresh all registered sources once.
    pub fn refresh_all(&mut self) {
        self.refresh_all_internal(false, None, None);
    }

    /// Advance the ingestion window by ingesting `step` new records.
    pub fn advance(&mut self, step: usize) {
        self.refresh_all_internal(false, Some(step), None);
    }

    /// Advance the ingestion window by ingesting `step` new records with weights.
    ///
    /// Returns `Err(SamplerError::InvalidWeight)` if `weights` contains an unregistered
    /// source ID or a negative value.
    pub fn advance_with_weights(
        &mut self,
        step: usize,
        weights: &HashMap<SourceId, f32>,
    ) -> Result<(), SamplerError> {
        self.validate_weights(weights)?;
        self.refresh_all_internal(false, Some(step), Some(weights));
        Ok(())
    }

    /// Force refresh all registered sources, discarding buffered records.
    pub fn force_refresh_all(&mut self) {
        self.refresh_all_internal(true, None, None);
    }

    /// Refresh all registered sources once with per-call source weights.
    ///
    /// Returns `Err(SamplerError::InvalidWeight)` if `weights` contains an unregistered
    /// source ID or a negative value.
    pub fn refresh_all_with_weights(
        &mut self,
        weights: &HashMap<SourceId, f32>,
    ) -> Result<(), SamplerError> {
        self.validate_weights(weights)?;
        self.refresh_all_internal(false, None, Some(weights));
        Ok(())
    }

    /// Force refresh all registered sources with per-call source weights.
    ///
    /// Returns `Err(SamplerError::InvalidWeight)` if `weights` contains an unregistered
    /// source ID or a negative value.
    pub fn force_refresh_all_with_weights(
        &mut self,
        weights: &HashMap<SourceId, f32>,
    ) -> Result<(), SamplerError> {
        self.validate_weights(weights)?;
        self.refresh_all_internal(true, None, Some(weights));
        Ok(())
    }

    fn validate_weights(&self, weights: &HashMap<SourceId, f32>) -> Result<(), SamplerError> {
        let known_ids: std::collections::HashSet<&str> =
            self.sources.iter().map(|s| s.source.id()).collect();
        for (id, &w) in weights {
            if !known_ids.contains(id.as_str()) {
                return Err(SamplerError::InvalidWeight {
                    source_id: id.clone(),
                    reason: "source is not registered".to_string(),
                });
            }
            if w < 0.0 {
                return Err(SamplerError::InvalidWeight {
                    source_id: id.clone(),
                    reason: format!("weight {w} is negative"),
                });
            }
        }
        Ok(())
    }

    /// Rebuild the shared cache by round-robin draining per-source buffers.
    ///
    /// When `force_refresh` is false, each source only refreshes when its buffer
    /// is empty; when true, all buffers are cleared and all sources refresh.
    /// If `step` is provided, performs a rolling update of `step` records (no clear).
    /// If `step` is None, clears the cache and fills up to max capacity.
    fn refresh_all_internal(
        &mut self,
        force_refresh: bool,
        step: Option<usize>,
        weights: Option<&HashMap<SourceId, f32>>,
    ) {
        self.last_refreshed_sources.clear();
        let mut refresh_plan = Vec::new();
        for (idx, state) in self.sources.iter_mut().enumerate() {
            if force_refresh {
                state.buffer.clear();
            }
            if force_refresh || state.buffer.is_empty() {
                refresh_plan.push((idx, state.cursor.clone()));
            }
        }

        if !refresh_plan.is_empty() {
            self.source_refresh_generation = self.source_refresh_generation.saturating_add(1);
            self.last_refreshed_sources = refresh_plan
                .iter()
                .map(|(idx, _)| self.sources[*idx].source.id().to_string())
                .collect();
            let mut results: Vec<
                Option<(Result<SourceSnapshot, SamplerError>, std::time::Duration)>,
            > = Vec::with_capacity(self.sources.len());
            results.resize_with(self.sources.len(), || None);
            let fetch_limit = self.max_records;
            let sampler_config = self.sampler_config.clone();
            thread::scope(|scope| {
                let mut handles = Vec::with_capacity(refresh_plan.len());
                for (idx, cursor) in &refresh_plan {
                    let source = &self.sources[*idx].source;
                    let cursor = cursor.clone();
                    let sampler_config = sampler_config.clone();
                    let source_epoch = self.source_epoch;
                    handles.push((
                        *idx,
                        scope.spawn(move || {
                            let start = std::time::Instant::now();
                            // XOR the source epoch into the seed so each epoch
                            // produces a distinct permutation within the source.
                            let epoch_config = SamplerConfig {
                                seed: derive_epoch_seed(sampler_config.seed, source_epoch),
                                ..sampler_config
                            };
                            let result =
                                source.refresh(&epoch_config, cursor.as_ref(), Some(fetch_limit));
                            let elapsed = start.elapsed();
                            (result, elapsed)
                        }),
                    ));
                }
                for (idx, handle) in handles {
                    let result = match handle.join() {
                        Ok((result, elapsed)) => {
                            debug!(
                                source_id = %self.sources[idx].source.id(),
                                refresh_ms = elapsed.as_millis(),
                                "source refresh completed"
                            );
                            (result, elapsed)
                        }
                        Err(_) => (
                            Err(SamplerError::SourceUnavailable {
                                source_id: self.sources[idx].source.id().to_string(),
                                reason: "source refresh thread panicked".into(),
                            }),
                            std::time::Duration::from_secs(0),
                        ),
                    };
                    results[idx] = Some(result);
                }
            });

            for (idx, _) in refresh_plan {
                let Some((result, elapsed)) = results[idx].take() else {
                    continue;
                };
                match result {
                    Ok(snapshot) => {
                        let SourceSnapshot {
                            records,
                            cursor: next_cursor,
                        } = snapshot;
                        let record_count = records.len();
                        let seconds = elapsed.as_secs_f64();
                        let per_sec = if seconds > 0.0 {
                            (record_count as f64) / seconds
                        } else {
                            0.0
                        };
                        let stats = &mut self.sources[idx].stats;
                        stats.last_refresh_ms = elapsed.as_millis();
                        stats.last_record_count = record_count;
                        stats.last_records_per_sec = per_sec;
                        stats.last_error = None;
                        debug!(
                            source_id = %self.sources[idx].source.id(),
                            record_count,
                            refresh_ms = elapsed.as_millis(),
                            records_per_sec = per_sec,
                            "source refresh ingested records"
                        );
                        let source_id = self.sources[idx].source.id().to_string();
                        let normalized = records
                            .into_iter()
                            .map(|mut record| {
                                record.source = source_id.clone();
                                record
                            })
                            .collect::<Vec<_>>();
                        self.sources[idx].buffer.extend(normalized);
                        self.sources[idx].cursor = Some(next_cursor);
                    }
                    Err(err) => {
                        let stats = &mut self.sources[idx].stats;
                        stats.last_refresh_ms = elapsed.as_millis();
                        stats.last_record_count = 0;
                        stats.last_records_per_sec = 0.0;
                        stats.last_error = Some(err.to_string());
                        stats.error_count = stats.error_count.saturating_add(1);
                        eprintln!(
                            "[data_sampler] source '{}' refresh failed: {}",
                            self.sources[idx].source.id(),
                            err
                        );
                    }
                }
            }
        }

        // On a full refresh (step=None) clear every per-source cache so that the
        // snapshot always reflects the newest window, matching the previous
        // shared-cache clear semantics.
        if step.is_none() {
            for state in self.sources.iter_mut() {
                state.cache.clear();
            }
        }
        if self.max_records == 0 {
            return;
        }
        let target_limit = step.unwrap_or(self.max_records);
        if let Some(weights) = weights {
            self.weighted_drain_into_caches(target_limit, weights);
        } else {
            // Fair round-robin drain: start from `drain_start` instead of 0 so
            // that the drain cursor rotates across cycles.  This prevents head
            // sources (low indices) from always draining faster than tail sources,
            // which was starving tail sources of refresh opportunities.
            let n = self.sources.len();
            if n > 0 {
                let mut per_source: Vec<Vec<DataRecord>> = vec![Vec::new(); n];
                let mut total_drained = 0;
                let mut any_remaining = true;
                while total_drained < target_limit && any_remaining {
                    any_remaining = false;
                    for offset in 0..n {
                        if total_drained >= target_limit {
                            break;
                        }
                        let idx = (self.drain_start + offset) % n;
                        if let Some(record) = self.sources[idx].buffer.pop_front() {
                            per_source[idx].push(record);
                            total_drained += 1;
                            any_remaining = true;
                        }
                    }
                }
                // Advance the drain cursor so the next cycle starts from a different
                // position.  Only advance when at least one record was drained, so a
                // burst of drain-noop cycles on an empty source list doesn't rotate.
                if total_drained > 0 {
                    self.drain_start = (self.drain_start + 1) % n;
                }
                for (idx, batch) in per_source.into_iter().enumerate() {
                    if !batch.is_empty() {
                        self.sources[idx].cache.ingest(batch);
                    }
                }
            }
        }
    }

    fn weighted_drain_into_caches(&mut self, limit: usize, weights: &HashMap<SourceId, f32>) {
        let len = self.sources.len();
        if len == 0 {
            return;
        }
        let mut weight_values = Vec::with_capacity(len);
        let mut any_positive = false;
        for state in &self.sources {
            let weight = weights.get(state.source.id()).copied().unwrap_or(1.0);
            if weight > 0.0 {
                any_positive = true;
            }
            weight_values.push(weight);
        }
        if !any_positive {
            weight_values.fill(1.0);
        }

        let mut current = vec![0.0f32; len];
        let mut per_source: Vec<Vec<DataRecord>> = vec![Vec::new(); len];
        let mut total = 0;
        while total < limit {
            let mut total_weight = 0.0f32;
            for (idx, weight) in weight_values.iter().copied().enumerate().take(len) {
                if weight <= 0.0 {
                    continue;
                }
                if self.sources[idx].buffer.is_empty() {
                    continue;
                }
                total_weight += weight;
            }
            if total_weight == 0.0 {
                break;
            }

            let mut best_idx = None;
            let mut best_score = f32::MIN;
            for idx in 0..len {
                if weight_values[idx] <= 0.0 {
                    continue;
                }
                if self.sources[idx].buffer.is_empty() {
                    continue;
                }
                current[idx] += weight_values[idx];
                if current[idx] > best_score {
                    best_score = current[idx];
                    best_idx = Some(idx);
                }
            }

            let idx = match best_idx {
                Some(idx) => idx,
                None => break,
            };
            current[idx] -= total_weight;
            if let Some(record) = self.sources[idx].buffer.pop_front() {
                per_source[idx].push(record);
                total += 1;
            }
        }

        for (idx, batch) in per_source.into_iter().enumerate() {
            if !batch.is_empty() {
                self.sources[idx].cache.ingest(batch);
            }
        }
    }

    /// Returns `true` when at least one source is registered.
    pub fn has_sources(&self) -> bool {
        !self.sources.is_empty()
    }
}

/// Per-source ingestion runtime state.
struct SourceState {
    source: Box<dyn DataSource + 'static>,
    cursor: Option<SourceCursor>,
    buffer: VecDeque<DataRecord>,
    /// Per-source LRU record cache capped at `max_records`.
    cache: RecordCache,
    stats: SourceRefreshStats,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TripletRecipe;
    use crate::data::{QualityScore, RecordSection, SectionRole};
    use chrono::Utc;
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    fn make_record(id: &str, source: &str) -> DataRecord {
        let now = Utc::now();
        DataRecord {
            id: id.to_string(),
            source: source.to_string(),
            created_at: now,
            updated_at: now,
            quality: QualityScore { trust: 1.0 },
            taxonomy: Vec::new(),
            sections: vec![RecordSection {
                role: SectionRole::Anchor,
                heading: None,
                text: id.to_string(),
                sentences: vec![id.to_string()],
            }],
            meta_prefix: None,
        }
    }

    struct ScriptedSource {
        id: String,
        refreshes: Arc<AtomicUsize>,
        script: Arc<Mutex<VecDeque<Result<SourceSnapshot, SamplerError>>>>,
    }

    impl ScriptedSource {
        fn new(
            id: &str,
            refreshes: Arc<AtomicUsize>,
            script: Vec<Result<SourceSnapshot, SamplerError>>,
        ) -> Self {
            Self {
                id: id.to_string(),
                refreshes,
                script: Arc::new(Mutex::new(script.into_iter().collect())),
            }
        }
    }

    impl DataSource for ScriptedSource {
        fn id(&self) -> &str {
            &self.id
        }

        fn refresh(
            &self,
            _config: &SamplerConfig,
            _cursor: Option<&SourceCursor>,
            _limit: Option<usize>,
        ) -> Result<SourceSnapshot, SamplerError> {
            self.refreshes.fetch_add(1, Ordering::SeqCst);
            let mut guard = self.script.lock().expect("script lock poisoned");
            guard.pop_front().unwrap_or_else(|| {
                Ok(SourceSnapshot {
                    records: Vec::new(),
                    cursor: SourceCursor {
                        last_seen: Utc::now(),
                        revision: 0,
                    },
                })
            })
        }

        fn reported_record_count(&self, _config: &SamplerConfig) -> Result<u128, SamplerError> {
            Ok(0)
        }

        fn default_triplet_recipes(&self) -> Vec<TripletRecipe> {
            Vec::new()
        }
    }

    struct PanicSource {
        id: String,
    }

    impl DataSource for PanicSource {
        fn id(&self) -> &str {
            &self.id
        }

        fn refresh(
            &self,
            _config: &SamplerConfig,
            _cursor: Option<&SourceCursor>,
            _limit: Option<usize>,
        ) -> Result<SourceSnapshot, SamplerError> {
            panic!("panic source refresh")
        }

        fn reported_record_count(&self, _config: &SamplerConfig) -> Result<u128, SamplerError> {
            Ok(0)
        }
    }

    #[test]
    fn record_cache_waits_len_and_clear_paths_are_covered() {
        let cache = RecordCache::new(2);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.ingest_count(), 0);

        cache.ingest(Vec::<DataRecord>::new());
        assert_eq!(cache.wait_for_ingest(0, Duration::from_millis(1)), 0);

        let cache_for_waiter = cache.clone();
        let handle = std::thread::spawn(move || cache_for_waiter.wait_for_ingest_blocking(0));
        std::thread::sleep(Duration::from_millis(5));
        cache.ingest(vec![make_record("r1", "s")]);
        assert_eq!(handle.join().unwrap(), 1);
        assert_eq!(cache.ingest_count(), 1);

        cache.ingest(vec![make_record("r2", "s"), make_record("r3", "s")]);
        assert_eq!(cache.len(), 2);
        let ids: Vec<String> = cache
            .snapshot()
            .into_iter()
            .map(|record| record.id)
            .collect();
        assert!(ids.contains(&"r2".to_string()));
        assert!(ids.contains(&"r3".to_string()));

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn record_cache_zero_limit_discards_everything() {
        let cache = RecordCache::new(0);
        cache.ingest(vec![make_record("r1", "s")]);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn manager_loads_and_snapshots_cursors_and_reports_has_sources() {
        let mut manager = IngestionManager::new(4, SamplerConfig::default());
        assert!(!manager.has_sources());
        manager.load_cursors(&[]);

        let refreshes = Arc::new(AtomicUsize::new(0));
        manager.register_source(Box::new(ScriptedSource::new(
            "cursor_source",
            refreshes,
            vec![Ok(SourceSnapshot {
                records: vec![make_record("id_1", "original_source")],
                cursor: SourceCursor {
                    last_seen: Utc::now(),
                    revision: 33,
                },
            })],
        )));
        assert!(manager.has_sources());

        manager.load_cursors(&[("cursor_source".to_string(), 7)]);
        let cursors = manager.snapshot_cursors();
        assert_eq!(cursors, vec![("cursor_source".to_string(), 7)]);

        manager.refresh_all();
        let updated = manager.snapshot_cursors();
        assert_eq!(updated, vec![("cursor_source".to_string(), 33)]);
        let records = manager.all_records_snapshot();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].source, "cursor_source");
    }

    #[test]
    fn advance_uses_buffer_before_refreshing_again() {
        let refreshes = Arc::new(AtomicUsize::new(0));
        let mut manager = IngestionManager::new(5, SamplerConfig::default());
        manager.register_source(Box::new(ScriptedSource::new(
            "buffered",
            refreshes.clone(),
            vec![Ok(SourceSnapshot {
                records: vec![
                    make_record("a", "x"),
                    make_record("b", "x"),
                    make_record("c", "x"),
                ],
                cursor: SourceCursor {
                    last_seen: Utc::now(),
                    revision: 1,
                },
            })],
        )));

        manager.advance(1);
        assert_eq!(refreshes.load(Ordering::SeqCst), 1);
        assert_eq!(manager.all_records_len(), 1);

        manager.advance(1);
        assert_eq!(refreshes.load(Ordering::SeqCst), 1);
        assert_eq!(manager.all_records_len(), 2);
    }

    #[test]
    fn force_refresh_clears_buffer_and_fetches_again() {
        let refreshes = Arc::new(AtomicUsize::new(0));
        let mut manager = IngestionManager::new(4, SamplerConfig::default());
        manager.register_source(Box::new(ScriptedSource::new(
            "force",
            refreshes.clone(),
            vec![
                Ok(SourceSnapshot {
                    records: vec![
                        make_record("r1", "x"),
                        make_record("r2", "x"),
                        make_record("r3", "x"),
                    ],
                    cursor: SourceCursor {
                        last_seen: Utc::now(),
                        revision: 10,
                    },
                }),
                Ok(SourceSnapshot {
                    records: vec![make_record("r4", "x")],
                    cursor: SourceCursor {
                        last_seen: Utc::now(),
                        revision: 11,
                    },
                }),
            ],
        )));

        manager.advance(1);
        assert_eq!(manager.all_records_len(), 1);
        assert_eq!(refreshes.load(Ordering::SeqCst), 1);

        manager.force_refresh_all();
        assert_eq!(refreshes.load(Ordering::SeqCst), 2);
        let records = manager.all_records_snapshot();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].id, "r4");
    }

    #[test]
    fn weighted_drain_respects_zero_and_fallback_weights() {
        let mut manager = IngestionManager::new(6, SamplerConfig::default());
        manager.register_source(Box::new(ScriptedSource::new(
            "a",
            Arc::new(AtomicUsize::new(0)),
            vec![Ok(SourceSnapshot {
                records: vec![make_record("a1", "a"), make_record("a2", "a")],
                cursor: SourceCursor {
                    last_seen: Utc::now(),
                    revision: 1,
                },
            })],
        )));
        manager.register_source(Box::new(ScriptedSource::new(
            "b",
            Arc::new(AtomicUsize::new(0)),
            vec![Ok(SourceSnapshot {
                records: vec![make_record("b1", "b"), make_record("b2", "b")],
                cursor: SourceCursor {
                    last_seen: Utc::now(),
                    revision: 1,
                },
            })],
        )));

        let mut only_b = HashMap::new();
        only_b.insert("a".to_string(), 0.0);
        only_b.insert("b".to_string(), 1.0);
        manager.refresh_all_with_weights(&only_b).unwrap();
        let ids: Vec<String> = manager
            .all_records_snapshot()
            .into_iter()
            .map(|record| record.id)
            .collect();
        assert!(ids.iter().all(|id| id.starts_with('b')));

        let mut manager_fallback = IngestionManager::new(6, SamplerConfig::default());
        manager_fallback.register_source(Box::new(ScriptedSource::new(
            "a",
            Arc::new(AtomicUsize::new(0)),
            vec![Ok(SourceSnapshot {
                records: vec![make_record("a1", "a")],
                cursor: SourceCursor {
                    last_seen: Utc::now(),
                    revision: 2,
                },
            })],
        )));
        manager_fallback.register_source(Box::new(ScriptedSource::new(
            "b",
            Arc::new(AtomicUsize::new(0)),
            vec![Ok(SourceSnapshot {
                records: vec![make_record("b1", "b")],
                cursor: SourceCursor {
                    last_seen: Utc::now(),
                    revision: 2,
                },
            })],
        )));

        let mut all_zero = HashMap::new();
        all_zero.insert("a".to_string(), 0.0);
        all_zero.insert("b".to_string(), 0.0);
        manager_fallback
            .refresh_all_with_weights(&all_zero)
            .unwrap();
        let ids: Vec<String> = manager_fallback
            .all_records_snapshot()
            .into_iter()
            .map(|record| record.id)
            .collect();
        assert!(ids.contains(&"a1".to_string()));
        assert!(ids.contains(&"b1".to_string()));
    }

    #[test]
    fn refresh_errors_and_panics_update_source_stats() {
        let mut manager = IngestionManager::new(4, SamplerConfig::default());
        manager.register_source(Box::new(ScriptedSource::new(
            "err_source",
            Arc::new(AtomicUsize::new(0)),
            vec![Err(SamplerError::SourceUnavailable {
                source_id: "err_source".to_string(),
                reason: "boom".to_string(),
            })],
        )));
        manager.register_source(Box::new(PanicSource {
            id: "panic_source".to_string(),
        }));

        manager.refresh_all();
        let stats = manager.source_refresh_stats();
        let err_stats = stats
            .iter()
            .find(|(source, _)| source == "err_source")
            .map(|(_, stats)| stats)
            .unwrap();
        assert_eq!(err_stats.error_count, 1);
        assert!(
            err_stats
                .last_error
                .as_ref()
                .is_some_and(|msg| msg.contains("boom"))
        );

        let panic_stats = stats
            .iter()
            .find(|(source, _)| source == "panic_source")
            .map(|(_, stats)| stats)
            .unwrap();
        assert_eq!(panic_stats.error_count, 1);
        assert!(
            panic_stats
                .last_error
                .as_ref()
                .is_some_and(|msg| msg.contains("panicked"))
        );
    }

    #[test]
    fn force_refresh_with_weights_path_is_exercised() {
        let mut manager = IngestionManager::new(3, SamplerConfig::default());
        manager.register_source(Box::new(ScriptedSource::new(
            "w",
            Arc::new(AtomicUsize::new(0)),
            vec![Ok(SourceSnapshot {
                records: vec![make_record("w1", "w")],
                cursor: SourceCursor {
                    last_seen: Utc::now(),
                    revision: 3,
                },
            })],
        )));

        let mut weights = HashMap::new();
        weights.insert("w".to_string(), 1.0);
        manager.force_refresh_all_with_weights(&weights).unwrap();
        assert_eq!(manager.all_records_len(), 1);
    }

    #[test]
    fn advance_with_weights_rejects_unknown_source() {
        let mut manager = IngestionManager::new(4, SamplerConfig::default());
        manager.register_source(Box::new(ScriptedSource::new(
            "known",
            Arc::new(AtomicUsize::new(0)),
            vec![],
        )));

        let mut weights = HashMap::new();
        weights.insert("known".to_string(), 1.0);
        weights.insert("unknown".to_string(), 0.5);

        let err = manager.advance_with_weights(1, &weights).unwrap_err();
        assert!(
            matches!(err, SamplerError::InvalidWeight { ref source_id, .. } if source_id == "unknown"),
            "expected InvalidWeight for unknown source, got {err:?}"
        );
    }

    #[test]
    fn refresh_all_with_weights_rejects_negative_weight() {
        let mut manager = IngestionManager::new(4, SamplerConfig::default());
        manager.register_source(Box::new(ScriptedSource::new(
            "src",
            Arc::new(AtomicUsize::new(0)),
            vec![],
        )));

        let mut weights = HashMap::new();
        weights.insert("src".to_string(), -1.0);

        let err = manager.refresh_all_with_weights(&weights).unwrap_err();
        assert!(
            matches!(err, SamplerError::InvalidWeight { ref source_id, .. } if source_id == "src"),
            "expected InvalidWeight for negative weight, got {err:?}"
        );
    }

    #[test]
    fn force_refresh_all_with_weights_rejects_unknown_source() {
        let mut manager = IngestionManager::new(4, SamplerConfig::default());
        manager.register_source(Box::new(ScriptedSource::new(
            "real",
            Arc::new(AtomicUsize::new(0)),
            vec![],
        )));

        let mut weights = HashMap::new();
        weights.insert("ghost".to_string(), 1.0);

        let err = manager
            .force_refresh_all_with_weights(&weights)
            .unwrap_err();
        assert!(
            matches!(err, SamplerError::InvalidWeight { ref source_id, .. } if source_id == "ghost"),
            "expected InvalidWeight for unknown source, got {err:?}"
        );
    }

    /// A source that records the `config.seed` value it receives on each `refresh()` call.
    struct SeedCapturingSource {
        id: String,
        received_seeds: Arc<Mutex<Vec<u64>>>,
    }

    impl SeedCapturingSource {
        fn new(id: &str, received_seeds: Arc<Mutex<Vec<u64>>>) -> Self {
            Self {
                id: id.to_string(),
                received_seeds,
            }
        }
    }

    impl DataSource for SeedCapturingSource {
        fn id(&self) -> &str {
            &self.id
        }

        fn refresh(
            &self,
            config: &SamplerConfig,
            _cursor: Option<&SourceCursor>,
            _limit: Option<usize>,
        ) -> Result<SourceSnapshot, SamplerError> {
            self.received_seeds
                .lock()
                .expect("seed lock poisoned")
                .push(config.seed);
            Ok(SourceSnapshot {
                records: Vec::new(),
                cursor: SourceCursor {
                    last_seen: Utc::now(),
                    revision: 0,
                },
            })
        }

        fn reported_record_count(&self, _config: &SamplerConfig) -> Result<u128, SamplerError> {
            Ok(0)
        }

        fn default_triplet_recipes(&self) -> Vec<crate::config::TripletRecipe> {
            Vec::new()
        }
    }

    #[test]
    fn source_epoch_xor_changes_seed_received_by_source() {
        // Verify that derive_epoch_seed(base, epoch) is actually threaded through to
        // the source's refresh() call, and that different epochs produce different seeds.
        let base_seed = 0xDEAD_BEEF_u64;
        let config = SamplerConfig {
            seed: base_seed,
            ..SamplerConfig::default()
        };

        let seeds_epoch0 = Arc::new(Mutex::new(Vec::<u64>::new()));
        let seeds_epoch1 = Arc::new(Mutex::new(Vec::<u64>::new()));

        // --- epoch 0 ---
        let mut manager = IngestionManager::new(4, config.clone());
        manager.register_source(Box::new(SeedCapturingSource::new(
            "src",
            Arc::clone(&seeds_epoch0),
        )));
        // source_epoch defaults to 0; refresh_all passes derive_epoch_seed(base, 0)
        manager.refresh_all();

        // --- epoch 1 ---
        let mut manager2 = IngestionManager::new(4, config.clone());
        manager2.register_source(Box::new(SeedCapturingSource::new(
            "src",
            Arc::clone(&seeds_epoch1),
        )));
        manager2.set_source_epoch(1);
        manager2.refresh_all();

        let received0 = seeds_epoch0.lock().unwrap();
        let received1 = seeds_epoch1.lock().unwrap();

        assert!(!received0.is_empty(), "epoch-0 source was never refreshed");
        assert!(!received1.is_empty(), "epoch-1 source was never refreshed");

        let seed_at_epoch0 = received0[0];
        let seed_at_epoch1 = received1[0];

        // The seeds must differ — epoch XOR has a real effect.
        assert_ne!(
            seed_at_epoch0, seed_at_epoch1,
            "epoch 0 and epoch 1 both produced seed {seed_at_epoch0:#x}; \
             derive_epoch_seed is not reaching the source"
        );

        // They must match the expected derive_epoch_seed values.
        assert_eq!(
            seed_at_epoch0,
            derive_epoch_seed(base_seed, 0),
            "epoch-0 seed mismatch"
        );
        assert_eq!(
            seed_at_epoch1,
            derive_epoch_seed(base_seed, 1),
            "epoch-1 seed mismatch"
        );
    }

    #[test]
    fn scripted_and_panic_sources_cover_default_trait_paths() {
        let refreshes = Arc::new(AtomicUsize::new(0));
        let scripted = ScriptedSource::new("scripted", refreshes, vec![]);

        // Empty script falls back to an empty snapshot.
        let snapshot = scripted
            .refresh(&SamplerConfig::default(), None, None)
            .expect("fallback snapshot");
        assert!(snapshot.records.is_empty());
        assert_eq!(snapshot.cursor.revision, 0);

        assert_eq!(
            scripted
                .reported_record_count(&SamplerConfig::default())
                .expect("record count"),
            0
        );
        assert!(scripted.default_triplet_recipes().is_empty());

        let panic_source = PanicSource {
            id: "panic_count".to_string(),
        };
        assert_eq!(
            panic_source
                .reported_record_count(&SamplerConfig::default())
                .expect("record count"),
            0
        );
    }

    #[test]
    fn seed_capturing_source_trait_defaults_are_exercised() {
        let source = SeedCapturingSource::new("seed_defaults", Arc::new(Mutex::new(Vec::new())));
        assert_eq!(
            source
                .reported_record_count(&SamplerConfig::default())
                .expect("record count"),
            0
        );
        assert!(source.default_triplet_recipes().is_empty());
    }

    #[test]
    fn refresh_paths_handle_zero_capacity_and_no_sources() {
        let mut manager = IngestionManager::new(0, SamplerConfig::default());
        manager.register_source(Box::new(ScriptedSource::new(
            "zero_capacity",
            Arc::new(AtomicUsize::new(0)),
            vec![Ok(SourceSnapshot {
                records: vec![make_record("r1", "zero_capacity")],
                cursor: SourceCursor {
                    last_seen: Utc::now(),
                    revision: 1,
                },
            })],
        )));
        manager.refresh_all();
        assert!(manager.all_caches_empty());

        // Weighted refresh with no sources should be a no-op.
        let mut empty_manager = IngestionManager::new(4, SamplerConfig::default());
        let empty_weights = HashMap::new();
        empty_manager
            .refresh_all_with_weights(&empty_weights)
            .expect("no sources should not error");
        assert!(empty_manager.all_caches_empty());
    }

    #[test]
    fn drain_start_rotates_fairly_across_sources() {
        // Create 3 sources, each with 10 records in their buffer after refresh.
        // The fair round-robin should ensure all 3 drain at the same rate
        // over multiple advance cycles.
        struct FairSource {
            id: String,
            refresh_count: Arc<AtomicUsize>,
        }

        impl DataSource for FairSource {
            fn id(&self) -> &str {
                &self.id
            }
            fn refresh(
                &self,
                _config: &SamplerConfig,
                _cursor: Option<&SourceCursor>,
                _limit: Option<usize>,
            ) -> Result<SourceSnapshot, SamplerError> {
                self.refresh_count.fetch_add(1, Ordering::SeqCst);
                Ok(SourceSnapshot {
                    records: (0..10)
                        .map(|i| make_record(&format!("r{i}"), &self.id))
                        .collect(),
                    cursor: SourceCursor {
                        last_seen: Utc::now(),
                        revision: 1,
                    },
                })
            }
            fn reported_record_count(&self, _: &SamplerConfig) -> Result<u128, SamplerError> {
                Ok(10)
            }
        }

        let counts = (
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
        );

        let mut manager = IngestionManager::new(30, SamplerConfig::default());
        manager.register_source(Box::new(FairSource {
            id: "src_0".to_string(),
            refresh_count: Arc::clone(&counts.0),
        }));
        manager.register_source(Box::new(FairSource {
            id: "src_1".to_string(),
            refresh_count: Arc::clone(&counts.1),
        }));
        manager.register_source(Box::new(FairSource {
            id: "src_2".to_string(),
            refresh_count: Arc::clone(&counts.2),
        }));

        // First refresh_all fills all buffers.
        manager.refresh_all();
        // All 3 refreshed once.
        assert_eq!(counts.0.load(Ordering::SeqCst), 1);
        assert_eq!(counts.1.load(Ordering::SeqCst), 1);
        assert_eq!(counts.2.load(Ordering::SeqCst), 1);

        // Each advance(1) drains 1 record from 1 source, rotating via
        // drain_start.  With 3 sources, after 3 advances each source
        // loses 1 record.  After 30 advances each source loses 10 records
        // and triggers a refresh.  Run 33 advances and check all 3 refreshed
        // roughly the same number of times.
        for _ in 0..33 {
            manager.advance(1);
        }

        let r0 = counts.0.load(Ordering::SeqCst);
        let r1 = counts.1.load(Ordering::SeqCst);
        let r2 = counts.2.load(Ordering::SeqCst);

        // Each had 10 records after initial refresh_all.
        // 10 records / (1 drained per 3 cycles) = 30 cycles to drain each buffer.
        // After 33 more advances each buffer emptied ~1 time and re-filled, so
        // each source should have refreshed ~2 times total (initial + 1 drain).
        // The exact count can vary by 1 due to timing, but all 3 must be within
        // 1 of each other — no source can be starved.
        let min = r0.min(r1).min(r2);
        let max = r0.max(r1).max(r2);
        assert!(
            max <= min + 1,
            "sources should refresh at roughly the same rate: got {r0}/{r1}/{r2}"
        );
    }
}
