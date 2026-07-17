use std::thread;

use crate::source_core::HuggingFaceRowSource;
use tracing::warn;

/// Global gate that serializes expansion downloads across ALL HuggingFace
/// sources.  Only one source downloads a shard at any given time, preventing
/// bursts when multiple sources trigger expansion on the same cycle.
pub(crate) static EXPANSION_GATE: std::sync::OnceLock<std::sync::Mutex<()>> =
    std::sync::OnceLock::new();

/// Spawn the background shard-expansion thread if expansion is needed and
/// no download is already in progress.  This is separate from `refresh()`
/// so the ingestion manager can call it on every scheduling cycle even
/// when the per-source buffer has not yet drained to empty, preventing
/// expansion from stalling for long epochs.
pub(crate) fn trigger_expansion_if_needed(source: &HuggingFaceRowSource) {
    let needs_expansion = source
        .state
        .lock()
        .map(|state| {
            let all_consumed = state
                .remote_candidates
                .as_ref()
                .is_some_and(|c| state.next_remote_idx >= c.len());
            !all_consumed
        })
        .unwrap_or(false);

    if !needs_expansion {
        return;
    }

    let already_running = source
        .expansion_thread
        .lock()
        .map(|t| t.as_ref().is_some_and(|h| !h.is_finished()))
        .unwrap_or(false);

    if already_running {
        return;
    }

    let handle = {
        let source = source.clone();
        thread::spawn(move || {
            // Acquire the global expansion gate so only one source downloads
            // a shard at a time across all Hugging Face sources.  The gate is
            // released when the thread exits (guard dropped).
            let _gate = EXPANSION_GATE
                .get_or_init(|| std::sync::Mutex::new(()))
                .lock()
                .expect("expansion gate not poisoned");

            // If candidates not yet fetched, discover them first.
            let needs_candidates = source
                .state
                .lock()
                .map(|s| s.remote_candidates.is_none())
                .unwrap_or(false);
            if needs_candidates {
                let target = source
                    .state
                    .lock()
                    .map(|s| s.materialized_rows)
                    .unwrap_or(0);
                if let Err(err) = source.ensure_row_available(target) {
                    warn!(
                        "[triplets:hf] background expansion (candidate fetch) error \
                             (source '{}'): {}",
                        source.config.source_id, err
                    );
                }
                return;
            }
            if let Err(err) = source.download_next_remote_shard() {
                warn!(
                    "[triplets:hf] background expansion error (source '{}'): {}",
                    source.config.source_id, err
                );
            }
        })
    };
    if let Ok(mut slot) = source.expansion_thread.lock() {
        *slot = Some(handle);
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::{test_config, test_source};
    use serial_test::serial;
    use tempfile::tempdir;

    // FIXME: [WINDOWS] This test passes in isolation, but times out when running with all of the tests.
    //
    // Additional context (may be inaccurate):
    //
    // Testing live thread spawning combined with a deliberate fallback to an
    // unreachable dead port (127.0.0.1:1) binds your test suite's determinism
    // directly to OS-level TCP/IP implementation details. While Unix environments
    // typically reject connections to unbound low ports instantaneously (ECONNREFUSED),
    // the Windows Winsock layer behaves non-deterministically under parallel test
    // execution profiles, frequently caching socket state or delaying connection drops
    // to match synthetic connect timeouts.
    #[test]
    #[serial(global_state)]
    #[cfg(not(target_os = "windows"))]
    fn trigger_expansion_if_needed_starts_background_thread() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let mut source = test_source(config);

        // Override with ultra-short timeouts to force immediate connection failure on Windows
        let inner_client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_millis(100))
            .timeout(std::time::Duration::from_millis(200))
            .build()
            .expect("failed to build ultra-short timeout client");
        source.http_client = reqwest_drive::ClientBuilder::new(inner_client).build();

        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 5;
            state.remote_candidates = Some(vec![
                "url::http://127.0.0.1:1/ds/resolve/main/train/000.ndjson".to_string(),
            ]);
            state.next_remote_idx = 0;
            state.remote_candidate_order = vec![0];
        }
        assert!(source.expansion_thread.lock().unwrap().is_none());
        crate::expansion::trigger_expansion_if_needed(&source);
        let handle = source.expansion_thread.lock().unwrap().take();
        assert!(handle.is_some());
        if let Some(h) = handle {
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let _ = tx.send(h.join());
            });
            let _ = rx
                .recv_timeout(std::time::Duration::from_secs(5))
                .expect("expansion thread hung or deadlocked: Timeout");
        }
    }

    #[test]
    fn trigger_expansion_if_needed_skips_when_all_remote_candidates_consumed() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 100;
            state.remote_candidates = Some(vec!["done".to_string()]);
            state.next_remote_idx = 1;
        }
        crate::expansion::trigger_expansion_if_needed(&source);
        assert!(source.expansion_thread.lock().unwrap().is_none());
    }

    #[test]
    fn trigger_expansion_if_needed_skips_when_total_rows_is_zero() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);
        {
            let mut state = source.state.lock().unwrap();
            state.materialized_rows = 0;
        }
        crate::expansion::trigger_expansion_if_needed(&source);
        assert!(source.expansion_thread.lock().unwrap().is_none());
    }

    #[test]
    fn trigger_expansion_if_needed_skips_when_already_running() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path().to_path_buf());
        let source = test_source(config);

        // Inject a dummy thread that blocks until explicitly released.
        // No network I/O, no sleep, no global mutex contention.
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let dummy = std::thread::spawn(move || {
            let _ = rx.recv();
        });
        *source.expansion_thread.lock().unwrap() = Some(dummy);

        // Must skip: slot is already occupied.
        crate::expansion::trigger_expansion_if_needed(&source);
        assert!(source.expansion_thread.lock().unwrap().as_ref().is_some());

        // Release: signal the dummy to exit, join cleanly.
        drop(tx);
        let handle = source.expansion_thread.lock().unwrap().take().unwrap();
        handle.join().unwrap();
    }
}
