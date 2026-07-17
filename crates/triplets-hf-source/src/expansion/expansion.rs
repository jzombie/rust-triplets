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
