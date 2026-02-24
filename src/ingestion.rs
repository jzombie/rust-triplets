use crate::config::SamplerConfig;
use crate::data::DataRecord;
use crate::errors::SamplerError;
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

/// Coordinates on-demand source refresh and shared-cache population.
pub struct IngestionManager {
    cache: RecordCache,
    sources: Vec<SourceState>,
    max_records: usize,
    sampler_config: SamplerConfig,
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
            cache: RecordCache::new(max_records),
            sources: Vec::new(),
            max_records,
            sampler_config,
        }
    }

    /// Register a source for on-demand ingestion.
    pub fn register_source(&mut self, source: Box<dyn DataSource + 'static>) {
        self.sources.push(SourceState {
            source,
            cursor: None,
            buffer: VecDeque::new(),
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

    /// Access the shared record cache.
    pub fn cache(&self) -> RecordCache {
        self.cache.clone()
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
    pub fn advance_with_weights(&mut self, step: usize, weights: &HashMap<SourceId, f32>) {
        self.refresh_all_internal(false, Some(step), Some(weights));
    }

    /// Force refresh all registered sources, discarding buffered records.
    pub fn force_refresh_all(&mut self) {
        self.refresh_all_internal(true, None, None);
    }

    /// Refresh all registered sources once with per-call source weights.
    pub fn refresh_all_with_weights(&mut self, weights: &HashMap<SourceId, f32>) {
        self.refresh_all_internal(false, None, Some(weights));
    }

    /// Force refresh all registered sources with per-call source weights.
    pub fn force_refresh_all_with_weights(&mut self, weights: &HashMap<SourceId, f32>) {
        self.refresh_all_internal(true, None, Some(weights));
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
            let mut results: Vec<
                Option<(Result<SourceSnapshot, SamplerError>, std::time::Duration)>,
            > = Vec::with_capacity(self.sources.len());
            results.resize_with(self.sources.len(), || None);
            let fetch_limit = step.unwrap_or(self.max_records);
            let sampler_config = self.sampler_config.clone();
            thread::scope(|scope| {
                let mut handles = Vec::with_capacity(refresh_plan.len());
                for (idx, cursor) in &refresh_plan {
                    let source = &self.sources[*idx].source;
                    let cursor = cursor.clone();
                    let sampler_config = sampler_config.clone();
                    handles.push((
                        *idx,
                        scope.spawn(move || {
                            let start = std::time::Instant::now();
                            let result =
                                source.refresh(&sampler_config, cursor.as_ref(), Some(fetch_limit));
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

        if step.is_none() {
            self.cache.clear();
        }
        let mut batch = Vec::new();
        if self.max_records == 0 {
            return;
        }
        let target_limit = step.unwrap_or(self.max_records);
        if let Some(weights) = weights {
            self.weighted_drain_into_cache(&mut batch, target_limit, weights);
        } else {
            let mut any_remaining = true;
            while batch.len() < target_limit && any_remaining {
                any_remaining = false;
                for state in self.sources.iter_mut() {
                    if batch.len() >= target_limit {
                        break;
                    }
                    if let Some(record) = state.buffer.pop_front() {
                        batch.push(record);
                        any_remaining = true;
                    }
                }
            }
        }
        if !batch.is_empty() {
            self.cache.ingest(batch);
        }
    }

    fn weighted_drain_into_cache(
        &mut self,
        batch: &mut Vec<DataRecord>,
        limit: usize,
        weights: &HashMap<SourceId, f32>,
    ) {
        let len = self.sources.len();
        if len == 0 {
            return;
        }
        let mut weight_values = Vec::with_capacity(len);
        let mut any_positive = false;
        for state in &self.sources {
            let weight = weights.get(state.source.id()).copied().unwrap_or(1.0);
            let weight = if weight.is_sign_negative() {
                0.0
            } else {
                weight
            };
            if weight > 0.0 {
                any_positive = true;
            }
            weight_values.push(weight);
        }
        if !any_positive {
            weight_values.fill(1.0);
        }

        let mut current = vec![0.0f32; len];
        while batch.len() < limit {
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
                batch.push(record);
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
        let records = manager.cache().snapshot();
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
        assert_eq!(manager.cache().len(), 1);

        manager.advance(1);
        assert_eq!(refreshes.load(Ordering::SeqCst), 1);
        assert_eq!(manager.cache().len(), 2);
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
        assert_eq!(manager.cache().len(), 1);
        assert_eq!(refreshes.load(Ordering::SeqCst), 1);

        manager.force_refresh_all();
        assert_eq!(refreshes.load(Ordering::SeqCst), 2);
        let records = manager.cache().snapshot();
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
        manager.refresh_all_with_weights(&only_b);
        let ids: Vec<String> = manager
            .cache()
            .snapshot()
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

        let mut non_positive = HashMap::new();
        non_positive.insert("a".to_string(), -3.0);
        non_positive.insert("b".to_string(), 0.0);
        manager_fallback.refresh_all_with_weights(&non_positive);
        let ids: Vec<String> = manager_fallback
            .cache()
            .snapshot()
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
        manager.force_refresh_all_with_weights(&weights);
        assert_eq!(manager.cache().len(), 1);
    }
}
