use crate::test_utils::{test_config, test_source};
use tempfile::tempdir;

#[cfg(not(target_os = "windows"))]
use serial_test::serial;

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
