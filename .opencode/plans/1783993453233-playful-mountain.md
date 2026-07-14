# Migration: Datasets Server Parquet Endpoint → Hub API

## Context

The Hugging Face Datasets Server `/parquet` endpoint is deprecated. Only the `/parquet` endpoint migrates to the Hub API; `/size` and `/info` remain on `datasets-server.huggingface.co`. The new Hub API returns a hierarchical dictionary of URL strings (not objects), requires no query parameters, and returns no file sizes.

---

## Phase 1: Update Parquet Endpoint Constant

**File:** `crates/triplets-hf-source/src/constants.rs` (line 130)

Only the parquet endpoint changes. `/size` and `/info` are untouched.

```rust
// BEFORE:
pub const HF_PARQUET_DEFAULT_ENDPOINT: &str = "https://datasets-server.huggingface.co/parquet";

// AFTER:
pub const HF_PARQUET_DEFAULT_ENDPOINT: &str = "https://huggingface.co/api/datasets";
```

`HF_SIZE_DEFAULT_ENDPOINT` and `HF_INFO_DEFAULT_ENDPOINT` stay as-is on `datasets-server.huggingface.co`.

---

## Phase 2: Rewrite `/parquet` URL Construction

**File:** `crates/triplets-hf-source/src/huggingface_source.rs`, function `list_remote_candidates_from_parquet_manifest_with_runtime` (lines 2567-2599)

The Hub API embeds the dataset name in the URL path. No query parameters — the entire config/split hierarchy is returned and filtered client-side.

```rust
fn list_remote_candidates_from_parquet_manifest_with_runtime(
    http_client: &ClientWithMiddleware,
    config: &HuggingFaceRowsConfig,
    runtime: Option<&tokio::runtime::Runtime>,
) -> Result<ParquetManifestCandidates, SamplerError> {
    let base = &config.parquet_endpoint;
    // Hub API: {base}/{dataset}/parquet — dataset in path, no query params
    let url = format!("{base}/{}/parquet", config.dataset_name);
    let body = Self::block_on_http_with_runtime(
        runtime,
        config,
        Self::fetch_http_body_text(
            http_client,
            &config.source_id,
            &url,
            &[],  // No query params — Hub API returns full hierarchy
            "Hub parquet endpoint",
        ),
    )?;
    Self::parse_parquet_manifest_response(config, &body)
}
```

---

## Phase 3: Rewrite `/parquet` JSON Parsing

**File:** `crates/triplets-hf-source/src/huggingface_source.rs`, function `all_candidates_from_parquet_manifest` (lines 2035-2102)

**Old schema** (flat array of objects):
```json
{"parquet_files": [{"url": "https://...", "size": 12345}]}
```

**New Hub API schema** (hierarchical: config → split → array of URL strings):
```json
{
  "config_name": {
    "split_name": ["https://host/.../000.parquet", "https://host/.../001.parquet"]
  }
}
```

Key differences:
- Top-level keys are config names (not `"parquet_files"`)
- Second-level keys are split names
- Terminal values are **plain URL strings**, not objects
- No `"size"` field — sizes are unavailable from the manifest

### All-Splits Mode

When `config.split_name` is empty, the old code omitted the split query param so the server returned all splits. Under the Hub API, we must iterate **all keys** in the config object to gather URLs from every split. Similarly, when `config.config_name` is empty, iterate all top-level config keys.

```rust
fn all_candidates_from_parquet_manifest(
    config: &HuggingFaceRowsConfig,
    json: &Value,
) -> Result<ParquetManifestCandidates, SamplerError> {
    let accepted = Self::normalized_shard_extensions(config);
    let mut candidates = Vec::new();
    let mut candidate_sizes = HashMap::new(); // Empty — Hub API has no sizes
    let mut matched_manifest_entries = 0usize;

    // ── Helper: collect URL strings from a JSON array ──────────────────
    let mut collect_urls = |urls: &Value| {
        if let Some(arr) = urls.as_array() {
            for url_val in arr {
                let Some(url) = url_val.as_str() else { continue; };
                let ext = Path::new(url)
                    .extension()
                    .and_then(|v| v.to_str())
                    .map(|v| v.to_ascii_lowercase());
                if !ext.as_deref().is_some_and(|v| accepted.iter().any(|a| a == v)) {
                    continue;
                }
                matched_manifest_entries += 1;
                candidates.push(format!("{HF_REMOTE_URL_PREFIX}{url}"));
            }
        }
    };

    // ── Try new Hub API format: config → split → [urls] ───────────────
    let mut hub_matched = false;
    if let Some(root_obj) = json.as_object() {
        // Guard: if root has "parquet_files" key, this is NOT Hub API format.
        if !root_obj.contains_key(HF_JSON_KEY_PARQUET_FILES) {
            hub_matched = true;
            let config_names: Vec<&str> = if config.config_name.is_empty() {
                // All-splits mode: iterate every config in root
                root_obj.keys().map(|k| k.as_str()).collect()
            } else {
                // Specific config
                vec![config.config_name.as_str()]
            };

            for cname in config_names {
                let Some(config_val) = root_obj.get(cname) else { continue; };
                let Some(config_obj) = config_val.as_object() else { continue; };

                if config.split_name.is_empty() {
                    // All-splits mode: iterate every split in config
                    for split_val in config_obj.values() {
                        collect_urls(split_val);
                    }
                } else {
                    // Specific split
                    if let Some(split_val) = config_obj.get(&config.split_name) {
                        collect_urls(split_val);
                    }
                }
            }
        }
    }

    // ── Fallback to old datasets-server format ─────────────────────────
    if !hub_matched {
        if let Some(entries) = json.get(HF_JSON_KEY_PARQUET_FILES).and_then(Value::as_array) {
            for entry in entries {
                let Some(url) = entry.get(HF_JSON_KEY_URL).and_then(Value::as_str) else {
                    continue;
                };
                let ext = Path::new(url)
                    .extension()
                    .and_then(|v| v.to_str())
                    .map(|v| v.to_ascii_lowercase());
                if !ext.as_deref().is_some_and(|v| accepted.iter().any(|a| a == v)) {
                    continue;
                }
                matched_manifest_entries += 1;
                let candidate = format!("{HF_REMOTE_URL_PREFIX}{url}");
                let expected_size = entry.get(HF_JSON_KEY_SIZE).and_then(Value::as_u64);

                let target = Self::candidate_target_path(config, &candidate);
                if target.exists() && !Self::target_matches_expected_size(&target, expected_size) {
                    warn!(
                        "[triplets:hf] incomplete cached shard detected (will redownload): {}",
                        target.display()
                    );
                    if let Err(err) = fs::remove_file(&target)
                        && err.kind() != std::io::ErrorKind::NotFound
                    {
                        return Err(SamplerError::SourceUnavailable {
                            source_id: config.source_id.clone(),
                            reason: format!(
                                "failed removing incomplete shard {}: {err}",
                                target.display()
                            ),
                        });
                    }
                }
                if let Some(size) = expected_size {
                    candidate_sizes.insert(candidate.clone(), size);
                }
                candidates.push(candidate);
            }
        }
    }

    candidates.sort();
    candidates.dedup();
    candidate_sizes.retain(|c, _| candidates.binary_search(c).is_ok());
    Ok((candidates, candidate_sizes, matched_manifest_entries))
}
```

---

## Phase 4: Fix Live E2E Test Panic

**File:** `crates/triplets-hf-source/tests/huggingface_integration.rs`, line 1523-1525

When the parquet manifest returns 0 entries (e.g. new format not parsed yet), the hf-hub fallback returns bare `rfilename` paths without the `url::` prefix. The test panics on `.expect()`.

Replace:
```rust
let shard_url = first_candidate
    .strip_prefix(triplets_hf_source::HF_REMOTE_URL_PREFIX)
    .expect("candidate should have url:: prefix")
    .to_string();
```

With:
```rust
let shard_url = first_candidate
    .strip_prefix(triplets_hf_source::HF_REMOTE_URL_PREFIX)
    .unwrap_or(first_candidate)
    .to_string();
```

---

## Phase 5: Update Mock Servers

**File:** `crates/triplets-hf-source/src/test_utils.rs`

### HfMockServer (lines 32-120)
Update manifest body (line 72) to hierarchical format with URL strings:

```rust
// NEW format: config → split → array of URL strings
let manifest_body = serde_json::json!({
    "default": {  // config_name
        "train": [  // split_name
            // Each shard is a plain URL string
            format!("{base_url}/resolve/main/train/{s:03}.ndjson")
        ]
    }
}).to_string();
```

Note: sizes are no longer in the manifest. The `manifest_counter` and shard payload serving remain unchanged.

### spawn_manifest_and_shard_http (lines 252-297)
Same update to hierarchical format with plain URL strings.

### Unit tests in huggingface_source.rs
Update all tests that build `{"parquet_files": [...]}` payloads to use the new hierarchical format with URL string arrays:
- `all_candidates_from_parquet_manifest_returns_all_with_sizes` (line 6374) — remove size assertions
- `all_candidates_from_parquet_manifest_includes_cached_and_replaces_stale` (line 6404) — adapt stale cache test for no-size scenario
- `candidates_from_parquet_manifest_errors_when_removing_incomplete_target_fails` (line 6453)
- `parse_parquet_manifest_response_returns_candidates` (line 7875)
- `list_remote_candidates_from_parquet_manifest_uses_test_endpoint_override` (line 7894)

Add new test cases for all-splits and multi-config scenarios:
- **All-splits empty split_name**: payload with multiple splits under one config, `split_name = ""`, assert URLs from ALL splits are collected
- **All-splits empty config_name**: payload with multiple configs, `config_name = ""`, assert URLs from ALL configs are collected
- **Old-format fallback**: payload with `"parquet_files"` key, assert legacy parsing still works
- **Multi-config with specific split**: payload with two configs, assert only the requested config's split URLs are returned

---

## Phase 6: CI — Enable Auto-Skip for Live Tests

**File:** `.github/workflows/rust-tests.yml` (line 86)

Uncomment the auto-skip line so PRs from forks (no secrets) don't fail on rate-limited anonymous requests:

```yaml
TRIPLETS_SKIP_LIVE_TESTS: ${{ secrets.HF_TOKEN == '' && '1' || '' }}
```

The existing `reqwest_drive::DriveThrottleBackoff` middleware handles 429 retry/backoff for authenticated requests — no manual retry logic.

---

## Files Modified

| File | Change |
|------|--------|
| `crates/triplets-hf-source/src/constants.rs` | Update `HF_PARQUET_DEFAULT_ENDPOINT` only |
| `crates/triplets-hf-source/src/huggingface_source.rs` | Rewrite parquet URL construction (no query params), rewrite JSON parsing (URL strings, hierarchical dict, old-format fallback) |
| `crates/triplets-hf-source/src/test_utils.rs` | Update mock manifest format to hierarchical URL strings |
| `crates/triplets-hf-source/tests/huggingface_integration.rs` | Fix E2E test panic on bare candidate paths |
| `.github/workflows/rust-tests.yml` | Enable auto-skip for live tests without credentials |

**Not modified:** `HF_SIZE_DEFAULT_ENDPOINT`, `HF_INFO_DEFAULT_ENDPOINT`, `fetch_global_row_count_with_runtime`, `fetch_classlabel_maps_with_runtime`, `extract_split_row_count_from_size_response`, `extract_classlabel_maps`.

---

## Verification

1. `cargo test -p triplets-hf-source` — unit tests pass with new mock format
2. `cargo test -p triplets-hf-source -- --ignored` (with `HF_TOKEN`) — live E2E test passes
3. `cargo test --workspace --all-features` — no regressions
