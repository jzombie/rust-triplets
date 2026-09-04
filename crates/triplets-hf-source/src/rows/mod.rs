pub mod rows;
#[cfg(test)]
pub(crate) use rows::{
    coalesce_list_field, decode_row_view, encode_row_view, parse_non_parquet_line,
    peek_first_non_whitespace, read_store_row_count, resolve_json_path, row_to_record,
    transcode_json_array_streaming, value_to_text, write_store_row_count,
};
pub(crate) use rows::{parse_row, read_row_batch, transcode_transient_shard_to_store};

#[cfg(test)]
mod rows_tests;
