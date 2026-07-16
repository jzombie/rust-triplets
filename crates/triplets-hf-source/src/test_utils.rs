//! Shared mock HTTP servers and test utilities for Hugging Face source tests.
//!
//! Used by both unit tests inside the `triplets-hf` crate and integration
//! tests in `tests/huggingface_integration.rs` so that mock-server logic is
//! defined once rather than duplicated in every test module.

/// URL that is guaranteed to be unreachable (port 1 is never bound on any
/// system).  Used in tests to simulate network failures without running a
/// server.
pub const TEST_UNREACHABLE_URL: &str = "http://127.0.0.1:1";

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use crate::config::HuggingFaceRowsConfig;
use crate::constants::{HF_SHARD_STORE_META_ROWS_KEY, HF_SHARD_STORE_ROW_PREFIX};
use crate::download::build_http_runtime;
use crate::huggingface_source::{
    EligibleIndexCache, ParquetCache, RowCache, RowTextField, RowView,
};
use crate::source_core::HuggingFaceRowSource;
use crate::types::SourceState;
use parquet::data_type::{ByteArray, ByteArrayType};
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;
use reqwest_drive::ClientWithMiddleware;
use simd_r_drive::DataStore;
use simd_r_drive::storage_engine::traits::DataStoreWriter;
use triplets_core::config::SamplerConfig;

// ---------------------------------------------------------------------------
// HfMockServer — full-featured mock HF datasets-server
// ---------------------------------------------------------------------------

/// A mock HF Hub API server that returns parquet manifests and shard payloads.
///
/// The server:
/// - Responds to any path with a manifest listing `n_shards` shards in the
///   hierarchical Hub API format: `{"config": {"split": ["url1", "url2"]}}`.
/// - Responds to `/resolve/main/train/{idx:03}.ndjson` with that shard's NDJSON.
/// - Counts each manifest fetch in [`manifest_fetch_count`](Self::manifest_fetch_count).
/// - Shuts down gracefully on drop or via [`shut_down`](Self::shut_down).
///
/// # Payload format
///
/// Each shard contains unique rows: `{"id":"s{shard}_r{row}","text":"txt_{shard}_{row}"}`
pub struct HfMockServer {
    base_url: String,
    manifest_counter: Arc<AtomicUsize>,
    shutdown: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl HfMockServer {
    /// Create a mock server with `n_shards` shards of `n_rows_per_shard` rows each.
    pub fn new(n_shards: usize, n_rows_per_shard: usize) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{addr}");
        let manifest_counter: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
        let mc = Arc::clone(&manifest_counter);
        let shutdown = Arc::new(AtomicBool::new(false));
        let sd = Arc::clone(&shutdown);

        // Build shard payloads.
        let shard_payloads: Vec<Vec<u8>> = (0..n_shards)
            .map(|s| {
                let mut buf = String::new();
                for r in 0..n_rows_per_shard {
                    buf.push_str(&format!(r#"{{"id":"s{s}_r{r}","text":"txt_{s}_{r}"}}"#));
                    buf.push('\n');
                }
                buf.into_bytes()
            })
            .collect();

        // Build the manifest JSON in Hub API tree endpoint format.
        // The tree endpoint returns an array of {"path": "...", "size": N} objects.
        // Use full URLs so downloads are served by this mock server.
        let manifest_entries: Vec<String> = (0..n_shards)
            .map(|s| {
                format!(
                    r#"{{"type":"file","path":"{base_url}/datasets/org/dataset/resolve/main/train/{s:03}.ndjson","size":{}}}"#,
                    shard_payloads[s].len()
                )
            })
            .collect();
        let manifest_body = format!("[{}]", manifest_entries.join(","));

        let handle = std::thread::spawn(move || {
            loop {
                if sd.load(Ordering::SeqCst) {
                    break;
                }
                if let Ok((mut stream, _)) = listener.accept() {
                    let mut buf = [0u8; 4096];
                    let _ = stream.read(&mut buf);
                    let request = String::from_utf8_lossy(&buf);
                    let first_line = request.lines().next().unwrap_or_default();

                    let body: Vec<u8> = if first_line.contains("/tree") {
                        mc.fetch_add(1, Ordering::SeqCst);
                        manifest_body.as_bytes().to_vec()
                    } else {
                        let idx: usize = first_line
                            .split_whitespace()
                            .nth(1)
                            .and_then(|path| {
                                path.split('/')
                                    .filter_map(|s| {
                                        s.trim_end_matches(".ndjson").parse::<usize>().ok()
                                    })
                                    .next()
                            })
                            .unwrap_or(0);
                        shard_payloads[idx.min(n_shards.saturating_sub(1))].clone()
                    };

                    let headers = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(headers.as_bytes());
                    let _ = stream.write_all(&body);
                    let _ = stream.flush();
                }
            }
        });

        HfMockServer {
            base_url,
            manifest_counter,
            shutdown,
            handle: Some(handle),
        }
    }

    /// The base URL (e.g. `http://127.0.0.1:56789`).
    pub fn url(&self) -> &str {
        &self.base_url
    }

    /// Number of times the `/parquet` manifest endpoint was queried.
    pub fn manifest_fetch_count(&self) -> usize {
        self.manifest_counter.load(Ordering::SeqCst)
    }

    /// Signal the server thread to shut down.  Blocking until the thread exits.
    pub fn shut_down(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(addr) = self.base_url.strip_prefix("http://") {
            let _ = TcpStream::connect(addr);
        }
    }
}

impl Drop for HfMockServer {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            if let Some(addr) = self.base_url.strip_prefix("http://") {
                let _ = TcpStream::connect(addr);
            }
            let _ = handle.join();
        }
    }
}

// ---------------------------------------------------------------------------
// TestHttpServer — simple mock HTTP server (fixed status + body)
// ---------------------------------------------------------------------------

/// A test HTTP server that responds with a fixed status and body to every
/// request.  Accepts connections in a loop until dropped; resilient to extra
/// connections (keep-alive probes, retries) that `reqwest` may make.
pub struct TestHttpServer {
    url: String,
    shutdown: Arc<AtomicBool>,
    accept_count: Arc<AtomicUsize>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl TestHttpServer {
    /// Create a server returning `status` and `body` for every request.
    pub fn new(status: u16, body: Vec<u8>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let url = format!("http://{addr}");
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = Arc::clone(&shutdown);
        let accept_count = Arc::new(AtomicUsize::new(0));
        let accept_count_clone = Arc::clone(&accept_count);

        let handle = std::thread::spawn(move || {
            while !shutdown_clone.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        accept_count_clone.fetch_add(1, Ordering::SeqCst);
                        drain_http_request(&mut stream);
                        let reason = match status {
                            200 => "OK",
                            400 => "Bad Request",
                            401 => "Unauthorized",
                            404 => "Not Found",
                            500 => "Internal Server Error",
                            501 => "Not Implemented",
                            _ => "Unknown",
                        };
                        let headers = format!(
                            "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        );
                        let _ = stream.write_all(headers.as_bytes());
                        let _ = stream.write_all(&body);
                        let _ = stream.flush();
                    }
                    Err(_) => break,
                }
            }
        });

        TestHttpServer {
            url,
            shutdown,
            accept_count,
            handle: Some(handle),
        }
    }

    /// The base URL (e.g. `http://127.0.0.1:56789`).
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Number of accepted connections since creation.
    pub fn accept_count(&self) -> usize {
        self.accept_count.load(Ordering::SeqCst)
    }
}

impl Drop for TestHttpServer {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            if let Some(addr) = self.url.strip_prefix("http://") {
                let _ = TcpStream::connect(addr);
            }
            let _ = handle.join();
        }
    }
}

/// Convenience: create a [`TestHttpServer`] returning HTTP 200.
pub fn spawn_one_shot_http(payload: Vec<u8>) -> TestHttpServer {
    TestHttpServer::new(200, payload)
}

// ---------------------------------------------------------------------------
// spawn_manifest_and_shard_http — convenience for HF manifest + shard server
// ---------------------------------------------------------------------------

/// Spawn a thread that acts as a minimal HF Hub API server.
///
/// Accepts up to `max_accepts` connections, returning the parquet manifest
/// in hierarchical Hub API format on any path, and `shard_payload` on
/// `/resolve/` paths.
///
/// Returns `(base_url, manifest_counter, join_handle)`.
pub fn spawn_manifest_and_shard_http(
    max_accepts: usize,
    shard_payload: Vec<u8>,
) -> (String, Arc<AtomicUsize>, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let base_url = format!("http://{addr}");
    let manifest_counter = Arc::new(AtomicUsize::new(0));
    let manifest_counter_arc = Arc::clone(&manifest_counter);
    // Hub API tree endpoint format: array of {"path": "...", "size": N} objects
    // The path must be a full URL pointing back to the mock server so that
    // remote_url_for_candidate returns the mock URL (not the real HF CDN).
    let manifest_body = serde_json::json!([
        {
            "type": "file",
            "path": format!("{base_url}/datasets/org/dataset/resolve/main/train/bootstrap.ndjson"),
            "size": shard_payload.len()
        }
    ])
    .to_string();
    let handle = std::thread::spawn(move || {
        for _ in 0..max_accepts {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    let mut request_buf = [0u8; 4096];
                    let read = stream.read(&mut request_buf).unwrap_or(0);
                    let request = String::from_utf8_lossy(&request_buf[..read]);
                    let first_line = request.lines().next().unwrap_or_default();
                    let body = if first_line.contains("/tree") {
                        manifest_counter_arc.fetch_add(1, Ordering::SeqCst);
                        manifest_body.as_bytes().to_vec()
                    } else {
                        shard_payload.clone()
                    };
                    let headers = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(headers.as_bytes());
                    let _ = stream.write_all(&body);
                    let _ = stream.flush();
                }
                Err(_) => break,
            }
        }
    });
    (base_url, manifest_counter, handle)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Drain incoming HTTP request headers so the connection closes cleanly.
fn drain_http_request(stream: &mut TcpStream) {
    let mut buf = Vec::with_capacity(2048);
    let mut tmp = [0u8; 512];
    loop {
        match stream.read(&mut tmp) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                buf.extend_from_slice(&tmp[..n]);
                if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
        }
    }
}

#[allow(dead_code)]
pub(crate) fn test_config(snapshot_dir: PathBuf) -> HuggingFaceRowsConfig {
    let mut config =
        HuggingFaceRowsConfig::new("hf_test", "org/dataset", "default", "train", snapshot_dir);
    // Unit tests should be deterministic and fully mock-driven; ignore any
    // process-level HF_TOKEN that CI might inject.
    config.hf_token = None;
    config.cache_capacity = 10;
    config.remote_expansion_headroom_multiplier = 3;
    // Point endpoints to connection-refused so tests never wait on
    // real HF servers.  Tests that exercise HTTP against mock servers
    // override these in their own body.
    config.parquet_endpoint = TEST_UNREACHABLE_URL.to_string();
    config
}

#[allow(dead_code)]
pub(crate) fn write_simdr_fixture(path: &Path, rows: &[(&str, &str)]) {
    // Create/open the simd-r-drive DataStore and write row-view entries
    let store = DataStore::open(path).expect("open simdr store");
    if rows.is_empty() {
        store
            .write(HF_SHARD_STORE_META_ROWS_KEY, &(0u64).to_le_bytes())
            .expect("write meta");
        return;
    }

    let mut batch: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    for (i, (id, text)) in rows.iter().enumerate() {
        let row = RowView {
            row_id: Some(id.to_string()),
            timestamp: None,
            text_fields: vec![RowTextField {
                name: "text".to_string(),
                text: text.to_string(),
            }],
        };
        let payload = serde_json::to_vec(&row).expect("encode row");
        let mut key = HF_SHARD_STORE_ROW_PREFIX.to_vec();
        key.extend_from_slice(&(i as u64).to_le_bytes());
        batch.push((key, payload));
    }

    let refs: Vec<(&[u8], &[u8])> = batch
        .iter()
        .map(|(k, v)| (k.as_slice(), v.as_slice()))
        .collect();
    store.batch_write(&refs).expect("batch write");
    store
        .write(
            HF_SHARD_STORE_META_ROWS_KEY,
            &(rows.len() as u64).to_le_bytes(),
        )
        .expect("write meta");
}

#[allow(dead_code)]
pub(crate) fn test_http_client() -> ClientWithMiddleware {
    use reqwest_drive::ClientBuilder;

    let inner = reqwest::Client::builder()
        .connect_timeout(Duration::from_millis(500))
        .timeout(Duration::from_secs(1))
        .build()
        .expect("fast test reqwest client should build");
    ClientBuilder::new(inner).build()
}

#[allow(dead_code)]
pub(crate) fn test_source(config: HuggingFaceRowsConfig) -> SafeTestSource {
    let http_runtime = Arc::new(build_http_runtime(&config).unwrap());
    // Use a non-throttled client in tests — mock servers serve a single
    // request then shut down, so retry backoff would add unnecessary delay.
    let http_client = test_http_client();
    let source = HuggingFaceRowSource {
        config,
        http_runtime,
        http_client,
        sampler_config: Arc::new(Mutex::new(None)),
        state: Arc::new(Mutex::new(SourceState {
            materialized_rows: 0,
            shards: Vec::new(),
            // Use Some(vec![]) rather than None so that trigger_expansion_if_needed
            // treats this source as "no remote candidates" and never spawns a
            // background thread that would make live network calls during tests.
            // Tests that explicitly exercise the remote-fetch path reset this field
            // to None before the call under test.
            remote_candidates: Some(vec![]),
            remote_candidate_sizes: HashMap::new(),
            next_remote_idx: 0,
            remote_candidate_order: Vec::new(),
        })),
        cache: Arc::new(Mutex::new(RowCache::default())),
        parquet_cache: Arc::new(Mutex::new(ParquetCache::default())),
        eligible_index: Arc::new(Mutex::new(EligibleIndexCache::default())),
        expansion_thread: Arc::new(Mutex::new(None)),
    };
    source.set_active_sampler_config(&SamplerConfig {
        seed: 1,
        ingestion_max_records: source.config.cache_capacity,
        ..SamplerConfig::default()
    });
    SafeTestSource { source }
}

/// RAII wrapper that joins any leaked expansion thread on drop.
/// Prevents zombie threads from holding `EXPANSION_GATE` and
/// deadlocking subsequent tests in the CI runner.
#[allow(dead_code)]
pub(crate) struct SafeTestSource {
    source: HuggingFaceRowSource,
}

impl std::ops::Deref for SafeTestSource {
    type Target = HuggingFaceRowSource;
    fn deref(&self) -> &Self::Target {
        &self.source
    }
}

impl std::ops::DerefMut for SafeTestSource {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.source
    }
}

impl Drop for SafeTestSource {
    fn drop(&mut self) {
        let mut lock = match self.source.expansion_thread.lock() {
            Ok(l) => l,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some(handle) = lock.take() {
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let _ = tx.send(handle.join());
            });
            let _ = rx
                .recv_timeout(std::time::Duration::from_secs(5))
                .expect("Test teardown leaked a deadlocked expansion thread");
        }
    }
}

#[allow(dead_code)]
pub(crate) fn with_env_var<R>(key: &str, value: &str, run: impl FnOnce() -> R) -> R {
    let previous = env::var(key).ok();
    struct EnvRestore {
        key: String,
        previous: Option<String>,
    }
    impl Drop for EnvRestore {
        fn drop(&mut self) {
            if let Some(old) = self.previous.clone() {
                unsafe { env::set_var(&self.key, old) };
            } else {
                unsafe { env::remove_var(&self.key) };
            }
        }
    }
    let _restore = EnvRestore {
        key: key.to_string(),
        previous,
    };
    unsafe { env::set_var(key, value) };
    run()
}

/// Sets multiple `(key, value)` pairs atomically, restoring originals on drop.
/// Use this instead of nesting `with_env_var` calls.
#[allow(dead_code)]
pub(crate) fn with_env_vars<R>(pairs: &[(&str, &str)], run: impl FnOnce() -> R) -> R {
    let previous: Vec<(String, Option<String>)> = pairs
        .iter()
        .map(|(key, _)| (key.to_string(), env::var(key).ok()))
        .collect();
    struct EnvRestore(Vec<(String, Option<String>)>);
    impl Drop for EnvRestore {
        fn drop(&mut self) {
            for (key, prev) in &self.0 {
                if let Some(old) = prev {
                    unsafe { env::set_var(key, old) };
                } else {
                    unsafe { env::remove_var(key) };
                }
            }
        }
    }
    let _restore = EnvRestore(previous);
    for (key, value) in pairs {
        unsafe { env::set_var(key, value) };
    }
    run()
}

#[allow(dead_code)]
pub(crate) fn with_current_dir<R>(dir: &Path, run: impl FnOnce() -> R) -> R {
    let previous = env::current_dir().expect("get cwd");
    struct CwdRestore {
        previous: PathBuf,
    }
    impl Drop for CwdRestore {
        fn drop(&mut self) {
            let _ = env::set_current_dir(&self.previous);
        }
    }
    let _restore = CwdRestore { previous };
    env::set_current_dir(dir).expect("set cwd");
    run()
}

#[allow(dead_code)]
pub(crate) fn write_parquet_fixture(path: &Path, rows: &[(&str, &str)]) {
    let schema = Arc::new(
        parse_message_type(
            "message test_schema {
                    REQUIRED BINARY id (UTF8);
                    REQUIRED BINARY text (UTF8);
                }",
        )
        .unwrap(),
    );
    let props = Arc::new(WriterProperties::builder().build());
    let file = File::create(path).unwrap();
    let mut writer = SerializedFileWriter::new(file, schema, props).unwrap();
    let mut row_group = writer.next_row_group().unwrap();

    if let Some(mut col_writer) = row_group.next_column().unwrap() {
        let values = rows
            .iter()
            .map(|(id, _)| ByteArray::from(*id))
            .collect::<Vec<_>>();
        col_writer
            .typed::<ByteArrayType>()
            .write_batch(&values, None, None)
            .unwrap();
        col_writer.close().unwrap();
    }

    if let Some(mut col_writer) = row_group.next_column().unwrap() {
        let values = rows
            .iter()
            .map(|(_, text)| ByteArray::from(*text))
            .collect::<Vec<_>>();
        col_writer
            .typed::<ByteArrayType>()
            .write_batch(&values, None, None)
            .unwrap();
        col_writer.close().unwrap();
    }

    assert!(row_group.next_column().unwrap().is_none());
    row_group.close().unwrap();
    writer.close().unwrap();
}
