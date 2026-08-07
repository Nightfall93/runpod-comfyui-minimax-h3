#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

MINIMAX_H3_SETUP_LIB_ONLY=1 source "$repo_root/minimax_h3_download_setup.sh"

"${PYTHON_TEST_BIN:-python3}" - "$tmp/valid.safetensors" <<'PY'
import json
import struct
import sys

header = json.dumps({"tensor": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
with open(sys.argv[1], "wb") as handle:
    handle.write(struct.pack("<Q", len(header)))
    handle.write(header)
    handle.write(b"\0\0\0\0")
PY

size="$(stat -c '%s' "$tmp/valid.safetensors")"
validate_safetensors_file "$tmp/valid.safetensors" "$size" >/dev/null

if validate_safetensors_file "$tmp/valid.safetensors" "$((size + 1))" >/dev/null 2>&1; then
  echo "Expected exact-size validation to fail." >&2
  exit 1
fi

printf 'not-a-safetensors-file' > "$tmp/invalid.safetensors"
if validate_safetensors_file "$tmp/invalid.safetensors" 22 >/dev/null 2>&1; then
  echo "Expected invalid safetensors header to fail." >&2
  exit 1
fi

echo "Safetensors validation tests passed."
