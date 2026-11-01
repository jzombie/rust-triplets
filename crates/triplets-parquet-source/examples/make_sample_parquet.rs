//! Generate the sample parquet dataset used by `local_ingest`.
//!
//! Usage:
//! ```sh
//! cargo run -p triplets-parquet-source --example make_sample_parquet -- <out-dir>
//! ```
//!
//! Writes two small files with `(id, title, body, date)` columns:
//! - `tech_articles.parquet` (5 rows)
//! - `space_articles.parquet` (4 rows)
//!
//! All configuration is positional CLI args; no environment variables.

use std::error::Error;
use std::path::Path;
use std::sync::Arc;

use parquet::data_type::{ByteArray, ByteArrayType};
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;

const SCHEMA: &str = "message article_schema {
    REQUIRED BINARY id (UTF8);
    REQUIRED BINARY title (UTF8);
    REQUIRED BINARY body (UTF8);
    REQUIRED BINARY date (UTF8);
}";

fn column_values(rows: &[(&str, &str, &str, &str)], col: usize) -> Vec<ByteArray> {
    rows.iter()
        .map(|row| {
            ByteArray::from(match col {
                0 => row.0,
                1 => row.1,
                2 => row.2,
                _ => row.3,
            })
        })
        .collect()
}

fn write_articles(path: &Path, rows: &[(&str, &str, &str, &str)]) -> Result<(), Box<dyn Error>> {
    let schema = Arc::new(parse_message_type(SCHEMA)?);
    let props = Arc::new(WriterProperties::builder().build());
    let file = std::fs::File::create(path)?;
    let mut writer = SerializedFileWriter::new(file, schema, props)?;
    let mut row_group = writer.next_row_group()?;

    for col_idx in 0..4 {
        if let Some(mut col_writer) = row_group.next_column()? {
            let values = column_values(rows, col_idx);
            col_writer
                .typed::<ByteArrayType>()
                .write_batch(&values, None, None)?;
            col_writer.close()?;
        } else {
            return Err(format!("missing column {col_idx}").into());
        }
    }
    assert!(row_group.next_column()?.is_none());
    row_group.close()?;
    writer.close()?;
    println!("wrote {} ({} rows)", path.display(), rows.len());
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let out_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "crates/triplets-parquet-source/example-dataset".to_string());
    let out_dir = Path::new(&out_dir);
    std::fs::create_dir_all(out_dir)?;

    write_articles(
        &out_dir.join("tech_articles.parquet"),
        &[
            (
                "tech-001",
                "Rustaceans ship lock-free sampler",
                "A new deterministic sampler pages sharded corpora without coordination.",
                "2026-03-02",
            ),
            (
                "tech-002",
                "Parquet row groups speed random reads",
                "Columnar row groups let readers fetch only the groups they need.",
                "2026-04-11",
            ),
            (
                "tech-003",
                "Simd stores back hot paths",
                "Memory-mapped row stores serve point lookups in microseconds.",
                "2026-05-19",
            ),
            (
                "tech-004",
                "Deterministic seeds stabilize evals",
                "Seeded permutations make refresh output reproducible across runs.",
                "2026-06-07",
            ),
            (
                "tech-005",
                "Small caches win on embedded",
                "Bounded LRU caches keep hot rows resident on tiny devices.",
                "2026-07-23",
            ),
        ],
    )?;

    write_articles(
        &out_dir.join("space_articles.parquet"),
        &[
            (
                "space-001",
                "Probe phones home from orbit",
                "Telemetry downlinked overnight shows nominal power and thermals.",
                "2026-02-14",
            ),
            (
                "space-002",
                "Lunar gateway adds docking port",
                "The new port supports crew transfer and cargo logistics.",
                "2026-03-30",
            ),
            (
                "space-003",
                "Solar sail demo unfurls",
                "Photon pressure now provides the mission's cruise thrust.",
                "2026-08-01",
            ),
            (
                "space-004",
                "Ground stations sync clocks",
                "Picosecond timing keeps the ranging array coherent.",
                "2026-09-12",
            ),
        ],
    )?;

    Ok(())
}
