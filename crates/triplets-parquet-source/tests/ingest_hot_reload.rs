//! Acceptance tests for the generic parquet reader (issue #59).
//!
//! - A local `*.parquet` directory is readable with zero Hugging Face
//!   involvement (this crate has no network dependencies — no `reqwest`).
//! - Concurrent `ingest_shard` + `read_row_batch` never collides on
//!   `global_start` and newly ingested rows become readable (SEV-1
//!   ingest-vs-read path).
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tempfile::tempdir;
use triplets_core::source::DataSource;
use triplets_parquet_source::fixtures::{test_config, test_source, write_parquet_fixture};
use triplets_parquet_source::shard_index::index_single_shard;

#[test]
fn local_parquet_dir_reads_without_any_hf_layer() {
    let dir = tempdir().unwrap();
    write_parquet_fixture(
        &dir.path().join("a.parquet"),
        &[("r0", "alpha"), ("r1", "beta")],
    );
    write_parquet_fixture(&dir.path().join("b.parquet"), &[("r2", "gamma")]);

    let config = test_config(dir.path().to_path_buf());
    let source = triplets_parquet_source::ParquetRowSource::new(config).unwrap();
    assert_eq!(source.materialized_row_count(), 3);

    let snapshot = source
        .refresh(
            &triplets_core::config::SamplerConfig::default(),
            None,
            Some(10),
        )
        .unwrap();
    assert_eq!(snapshot.records.len(), 3);
    let ids: HashSet<_> = snapshot.records.iter().map(|r| r.id.clone()).collect();
    assert_eq!(ids.len(), 3);
    assert!(ids.iter().all(|id| id.starts_with("parquet_test::")));
}

#[test]
fn concurrent_ingest_and_read_never_collides() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path().to_path_buf());
    let source = Arc::new(test_source(config.clone()));

    // Index 4 fixtures up front (provisional offsets); ingestion assigns
    // the real global_start values concurrently.
    let mut shards = Vec::new();
    for (i, rows) in [
        vec![("a0", "a0t"), ("a1", "a1t")],
        vec![("b0", "b0t"), ("b1", "b1t")],
        vec![("c0", "c0t"), ("c1", "c1t")],
        vec![("d0", "d0t"), ("d1", "d1t")],
    ]
    .iter()
    .enumerate()
    {
        let path = dir.path().join(format!("s{i}.parquet"));
        write_parquet_fixture(&path, rows);
        let shard = index_single_shard(&config, &path, 0).unwrap().0.unwrap();
        shards.push(shard);
    }

    // One ingester thread per shard; readers run concurrently on main.
    let mut handles = Vec::new();
    for shard in shards {
        let source = Arc::clone(&source);
        handles.push(std::thread::spawn(move || {
            // Small stagger to maximize overlap with concurrent readers.
            std::thread::sleep(Duration::from_millis(5));
            source.ingest_shard(shard).unwrap();
            let mut out = Vec::new();
            let _ = source.read_row_batch(&[0, 1, 2, 3, 4, 5, 6, 7], &mut out, None);
        }));
    }

    // Poll until all rows are visible (or time out).
    let deadline = Instant::now() + Duration::from_secs(15);
    let seen = loop {
        let mut out = Vec::new();
        source
            .read_row_batch(&[0, 1, 2, 3, 4, 5, 6, 7], &mut out, None)
            .unwrap();
        if out.len() == 8 {
            break out;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for ingested rows"
        );
        std::thread::sleep(Duration::from_millis(5));
    };
    for handle in handles {
        handle.join().unwrap();
    }

    assert_eq!(source.materialized_row_count(), 8);

    // No two shards share a global_start: exactly {0, 2, 4, 6}.
    let mut starts: Vec<usize> = source
        .shards_snapshot()
        .iter()
        .map(|shard| shard.global_start)
        .collect();
    starts.sort_unstable();
    assert_eq!(starts, vec![0, 2, 4, 6]);

    // All rows readable with distinct ids.
    let ids: HashSet<_> = seen.iter().map(|record| record.id.clone()).collect();
    assert_eq!(ids.len(), 8);
}
