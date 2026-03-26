#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/docs/diagrams"
OUT_DIR="$ROOT_DIR/assets/diagrams"

mkdir -p "$OUT_DIR"

if ! command -v mmdr >/dev/null 2>&1; then
  echo "error: mmdr is required to render diagrams." >&2
  echo "install: cargo install --locked mermaid-rs-renderer" >&2
  exit 1
fi

for input in "$SRC_DIR"/*.mmd; do
  [[ -e "$input" ]] || continue
  base_name="$(basename "$input" .mmd)"
  output="$OUT_DIR/${base_name}.svg"
  echo "rendering $input -> $output"
  mmdr -i "$input" -o "$output" -e svg
  # Normalize line endings and keep deterministic output ordering for diffs.
  perl -i -pe 's/\r\n?/\n/g' "$output"
done

echo "done"
