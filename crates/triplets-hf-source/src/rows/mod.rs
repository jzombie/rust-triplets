pub mod rows;
pub(crate) use rows::{parse_row, read_row_batch, transcode_transient_shard_to_store};
#[cfg(test)]
pub(crate) use rows::{
    coalesce_list_field, parse_non_parquet_line, encode_row_view, decode_row_view,
    row_to_record, value_to_text, resolve_json_path,
    write_store_row_count, read_store_row_count,
    transcode_json_array_streaming, peek_first_non_whitespace,
};

#[cfg(test)]
mod rows_tests;
