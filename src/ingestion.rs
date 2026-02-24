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
