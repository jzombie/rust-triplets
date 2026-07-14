# Migration: Complete Datasets Server Removal + Hub API Parquet

## Context

All interaction with `datasets-server.huggingface.co` must be eliminated. The `/parquet` endpoint migrates to the Hub API. The `/size` and `/info` endpoints are removed entirely (no replacement). ClassLabel resolution degrades gracefully — integer columns pass through as raw numbers.

---

## Phase 1: Remove `/size` Subsystem

### 1.1 Constants (`constants.rs`)

Delete `HF_SIZE_DEFAULT_ENDPOINT` (line 133).

### 1.2 Config struct (`huggingface_source.rs`)

Remove `size_endpoint` field from `HuggingFaceRowsConfig` (line 831) and its default init (line 886).

### 1.3 SourceState struct

Remove `total_rows: Option<usize>` field from `SourceState` (line 1173).

### 1.4 Production functions to delete

- `fetch_global_row_count` (line 2784, `#[cfg(test)]` wrapper)
- `fetch_global_row_count_with_runtime` (line 2791)
- `parse_global_row_count_response` (line 2826)
- `extract_split_row_count_from_size_response` (line 3050)
- `known_total_rows()` (line 1336)

### 1.5 Constructor changes (`new()`)

In `HuggingFaceRowSource::new()`, delete lines 1277-1292 that call `fetch_global_row_count_with_runtime`. Set `total_rows: None` unconditionally in the `SourceState` struct literal (line 1316).

### 1.6 Replace `len_hint()` (line 4931)

Delete the entire existing implementation and replace with:

```rust
fn len_hint(&self) -> Option<usize> {
    let state = self.state.lock().ok()?;
    let known = state.materialized_rows;
    if known > 0 {
        let mut upper = known;
        // Derive ongoing expansion from the remote candidate vector lengths
        if let Some(ref candidates) = state.remote_candidates {
            if state.next_remote_idx < candidates.len() {
                let headroom = self.effective_expansion_headroom_rows();
                upper = known.saturating_add(headroom);
            }
        }
        return Some(upper.max(known));
    }
    if state.remote_candidates.as_ref().is_some_and(|c| c.is_empty()) {
        return Some(0);
    }
    Some(1)
}
```

This derives expansion boundaries from the remote manifest file list when `state.total_rows` is absent. No `total_rows` references remain.

### 1.7 `trigger_expansion_if_needed()` (line 4986)

Remove `let known_empty = state.total_rows == Some(0);` (line 4991). Since `total_rows` no longer exists, `known_empty` is always `false`.

### 1.8 `refresh()` (line 5085)

Remove `let known_empty = state.total_rows == Some(0);` (line 5093). Since `total_rows` no longer exists, `known_empty` is always `false`.

### 1.9 `lib.rs` exports

Remove `HF_PARQUET_DEFAULT_ENDPOINT` from the public re-export if `size_endpoint` removal makes it unnecessary — but keep it since parquet is still used. No other export changes needed.

### 1.10 Tests to delete

All unit tests referencing size endpoint functions:
- `fetch_global_row_count_uses_test_endpoint_override` (7919)
- `fetch_global_row_count_with_runtime_uses_test_endpoint_override` (7941)
- `fetch_global_row_count_returns_none_when_split_not_present` (8002)
- `fetch_global_row_count_errors_when_endpoint_unreachable` (8136)
- `fetch_global_row_count_returns_ok_none_on_501` (10805)
- `extract_split_row_count_*` tests (7012, 7024, 7469, 8895, 8918, 8923)
- `parse_global_row_count_response_*` tests (7033, 7859, 10113, 10145, 10160)
- `known_total_rows_returns_state_value` (11216)
- `len_hint_*` tests that set `total_rows` (9242, 9265, 9275)
- `trigger_expansion_if_needed_skips_when_total_rows_is_zero` (11150)

Integration test to delete:
- `huggingface_live_size_endpoint_reports_dataset_row_count` (1327-1377)

### 1.11 Tests to update

All test helpers that set `total_rows: Some(...)` or `total_rows: None` — remove the field from struct literals. There are ~30 such locations.

Integration tests that set `config.size_endpoint` — remove those assignments.

---

## Phase 2: Remove `/info` Subsystem

### 2.1 Constants (`constants.rs`)

Delete:
- `HF_INFO_DEFAULT_ENDPOINT` (line 136)
- `HF_JSON_KEY_DATASET_INFO` (line 71)
- `HF_JSON_KEY_FEATURES` (line 73)
- `HF_JSON_KEY_FEATURE_TYPE` (line 75)
- `HF_JSON_KEY_LABEL_NAMES` (line 77)
- `HF_CLASSLABEL_TYPE` (line 79)

Update doc comment on `HF_PUBLIC_TEST_DATASET` (lines 138-144) — remove ClassLabel reference.

### 2.2 Config struct (`huggingface_source.rs`)

Remove:
- `label_maps: HashMap<String, Vec<String>>` field (lines 803-816)
- `info_endpoint: String` field (lines 832-835)
- Default init of both in `new()` (lines 881, 887)

### 2.3 Production functions to delete

- `fetch_classlabel_maps` (line 2458, `#[cfg(test)]` wrapper)
- `fetch_classlabel_maps_with_runtime` (line 2465)
- `extract_classlabel_maps` (line 2525)

### 2.4 Constructor changes (`new()`)

Delete lines 1244-1250 that call `fetch_classlabel_maps_with_runtime` and assign to `config.label_maps`.

### 2.5 Parquet transcode changes

`value_to_text()` (line 4315): Remove the `label_names` parameter and the integer-to-label resolution block (lines 4327-4333). The function simplifies to always return `Some(n.to_string())` for numbers.

`coalesce_field()` (line 4373): Remove `label_maps` parameter. Call `value_to_text(value, None)`.

`coalesce_list_field()` (line 4399): Remove `label_maps` parameter. Call `value_to_text(value, None)`.

`parse_row()` call sites (lines 4479, 4493, 4504, 4519, 4531): Remove `&self.config.label_maps` arguments.

### 2.6 Imports

Remove imports of deleted constants: `HF_CLASSLABEL_TYPE`, `HF_JSON_KEY_DATASET_INFO`, `HF_JSON_KEY_FEATURES`, `HF_JSON_KEY_FEATURE_TYPE`, `HF_JSON_KEY_LABEL_NAMES` (lines 35, 37-39).

### 2.7 Tests to delete

All unit tests for ClassLabel/info functionality:
- `extract_classlabel_maps_*` tests (10442, 10460, 10476, 10490, 10506, 10519)
- `fetch_classlabel_maps_*` tests (10534, 10549, 10566, 10583, 10917)
- `info_endpoint_called_exactly_once_per_source_construction` (10647)
- `value_to_text_resolves_integer_to_label_name` (10611)
- `value_to_text_falls_back_to_raw_integer_when_*` (10628, 10638)
- Tests referencing `info_endpoint` in cursor/sampler tests (9555, 9629)

Integration test to delete:
- `huggingface_live_classlabel_resolution_maps_integers_to_label_strings` (1382-1447)

### 2.8 Tests to update

Integration tests that set `config.info_endpoint = TEST_UNREACHABLE_URL` — remove those assignments (lines 2042, 2159, 2209, 2274, 2393, 2450).

Update `test_config()` helper — remove `info_endpoint` and `label_maps` fields.

### 2.9 Documentation

Update `examples/common/hf_sources.txt` — remove ClassLabel documentation comments (lines 12-14, 22-23).

---

## Phase 3: Migrate `/parquet` to Hub API

### 3.1 Constant (`constants.rs`)

```rust
// BEFORE:
pub const HF_PARQUET_DEFAULT_ENDPOINT: &str = "https://datasets-server.huggingface.co/parquet";

// AFTER:
pub const HF_PARQUET_DEFAULT_ENDPOINT: &str = "https://huggingface.co/api/datasets";
```

### 3.2 URL construction (`list_remote_candidates_from_parquet_manifest_with_runtime`)

Build `{base}/{dataset}/parquet` with no query params:

```rust
let base = &config.parquet_endpoint;
let url = format!("{base}/{}/parquet", config.dataset_name);
let body = Self::block_on_http_with_runtime(
    runtime, config,
    Self::fetch_http_body_text(http_client, &config.source_id, &url, &[], "Hub parquet endpoint"),
)?;
```

### 3.3 JSON parsing (`all_candidates_from_parquet_manifest`)

New Hub API schema — hierarchical dict of URL strings:
```json
{"config_name": {"split_name": ["https://...000.parquet", "https://...001.parquet"]}}
```

Key rules:
- Top-level keys = config names (not `"parquet_files"`)
- Second-level keys = split names
- Terminal values = plain URL strings (not objects)
- No `"size"` field
- When `split_name` is empty → iterate ALL splits in the config
- When `config_name` is empty → iterate ALL configs in root

Detection: if root JSON object does NOT contain `"parquet_files"` key, treat as Hub API format.

**Strongly-typed validation**: Every element pulled from the terminal array must be verified via `.as_str()` before allocation. No implicit type coercion or guessing at JSON structure.

```rust
// Inside the Hub API branch — collect URLs with explicit type validation
if config.split_name.is_empty() {
    for split_val in config_obj.values() {
        if let Some(arr) = split_val.as_array() {
            for element in arr {
                // Block malformed payloads: every element MUST be a string
                let Some(url) = element.as_str() else { continue; };
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
    }
} else {
    if let Some(split_val) = config_obj.get(&config.split_name) {
        if let Some(arr) = split_val.as_array() {
            for element in arr {
                let Some(url) = element.as_str() else { continue; };
                // ... same extension filter and candidate push ...
            }
        }
    }
}
```

The old-format fallback path retains the existing `entry.get(HF_JSON_KEY_URL).and_then(Value::as_str)` pattern.

### 3.4 Stale cache cleanup

Since the Hub API provides no file sizes, the stale shard detection (`target_matches_expected_size`) is skipped for Hub API responses. The old-format fallback path retains this logic.

---

## Phase 4: Fix E2E Test Panic

**File:** `tests/huggingface_integration.rs`, line 1523-1525

Replace `.expect("candidate should have url:: prefix")` with `.unwrap_or(first_candidate)` to handle bare hf-hub fallback paths.

---

## Phase 5: Update Mock Servers

**File:** `test_utils.rs`

Update `HfMockServer::new()` and `spawn_manifest_and_shard_http()` to produce hierarchical format with URL strings instead of `{"parquet_files": [...]}`.

Add `config_name` and `split_name` parameters to `HfMockServer::new()` for flexible test scenarios.

---

## Phase 6: CI

**File:** `.github/workflows/rust-tests.yml` (line 86)

Uncomment auto-skip: `TRIPLETS_SKIP_LIVE_TESTS: ${{ secrets.HF_TOKEN == '' && '1' || '' }}`

---

## Files Modified

| File | Change |
|------|--------|
| `crates/triplets-hf-source/src/constants.rs` | Delete `/size` and `/info` constants, update `/parquet` endpoint |
| `crates/triplets-hf-source/src/huggingface_source.rs` | Remove `size_endpoint`, `info_endpoint`, `total_rows`, `label_maps` fields; delete `/size` and `/info` functions; simplify `len_hint`, `trigger_expansion_if_needed`, `refresh`; simplify `value_to_text`; remove label_maps params from `coalesce_field`, `coalesce_list_field`, `parse_row`; rewrite parquet URL construction and JSON parsing |
| `crates/triplets-hf-source/src/test_utils.rs` | Update mock manifest format |
| `crates/triplets-hf-source/tests/huggingface_integration.rs` | Delete `/size` and `/info` live tests; fix E2E panic; remove `size_endpoint`/`info_endpoint` assignments |
| `.github/workflows/rust-tests.yml` | Enable auto-skip |
| `examples/common/hf_sources.txt` | Remove ClassLabel docs |

---

## Verification

1. `cargo test -p triplets-hf-source` — unit tests pass
2. `cargo test -p triplets-hf-source -- --ignored` (with `HF_TOKEN`) — live E2E test passes
3. `cargo test --workspace --all-features` — no regressions
4. `cargo clippy --workspace --all-features` — no warnings
