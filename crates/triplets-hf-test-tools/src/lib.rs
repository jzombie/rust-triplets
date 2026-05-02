//! Shared mock HTTP servers and test utilities for HuggingFace source tests.
//!
//! Used by both unit tests inside the `triplets` crate and integration tests
//! in `tests/huggingface_integration.rs` so that mock-server logic is defined
//! once rather than duplicated in every test module.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

// ---------------------------------------------------------------------------
// HfMockServer — full-featured mock HF datasets-server
// ---------------------------------------------------------------------------

/// A mock HF datasets-server that returns parquet manifests and shard payloads.
///
/// The server:
/// - Responds to `/parquet` paths with a manifest listing `n_shards` shards.
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

        // Build the manifest JSON.
        // URLs must have a recognised extension AND contain `/resolve/`.
        let manifest_entries: Vec<String> = (0..n_shards)
            .map(|s| {
                format!(
                    r#"{{"url":"{base_url}/resolve/main/train/{s:03}.ndjson","size":{}}}"#,
                    shard_payloads[s].len()
                )
            })
            .collect();
        let manifest_body = format!(r#"{{"parquet_files":[{}]}}"#, manifest_entries.join(","));

        let handle = std::thread::spawn(move || {
            loop {
                if sd.load(Ordering::SeqCst) {
                    break;
                }
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let mut buf = [0u8; 4096];
                        let _ = stream.read(&mut buf);
                        let request = String::from_utf8_lossy(&buf);
                        let first_line = request.lines().next().unwrap_or_default();

                        let body: Vec<u8> = if first_line.contains("/parquet") {
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
                    Err(_) => {}
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

/// Spawn a thread that acts as a minimal HF datasets-server.
///
/// Accepts up to `max_accepts` connections, returning the parquet manifest
/// on `/parquet` paths and `shard_payload` on everything else.
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
    let manifest_body = serde_json::json!({
        "parquet_files": [
            {
                "url": format!("{base_url}/resolve/main/train/bootstrap.ndjson"),
                "size": shard_payload.len()
            }
        ]
    })
    .to_string();
    let handle = std::thread::spawn(move || {
        for _ in 0..max_accepts {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    let mut request_buf = [0u8; 4096];
                    let read = stream.read(&mut request_buf).unwrap_or(0);
                    let request = String::from_utf8_lossy(&request_buf[..read]);
                    let first_line = request.lines().next().unwrap_or_default();
                    let body = if first_line.contains("/parquet") {
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
// EnvGuard — env-var lifecycle for async-thread tests
// ---------------------------------------------------------------------------

/// Set an environment variable and restore it on drop.
///
/// Unlike `with_env_var` closures, this keeps the env var set across async
/// thread boundaries (the expansion thread spawned by
/// `trigger_expansion_if_needed` reads env vars at HTTP-request time, which
/// may be after the caller returns).
pub struct EnvGuard {
    key: String,
    previous: Option<String>,
}

impl EnvGuard {
    /// Save the current value of `key` (or `None`), then set it to `value`.
    pub fn set(key: &str, value: &str) -> Self {
        let previous = std::env::var(key).ok();
        unsafe { std::env::set_var(key, value) };
        EnvGuard {
            key: key.to_string(),
            previous,
        }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        if let Some(ref old) = self.previous {
            unsafe { std::env::set_var(&self.key, old) };
        } else {
            unsafe { std::env::remove_var(&self.key) };
        }
    }
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
