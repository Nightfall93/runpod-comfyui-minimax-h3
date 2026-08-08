#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

export MINIMAX_H3_SETUP_LIB_ONLY=1
export MINIMAX_H3_STATUS_FILE="$tmp/minimax-h3-download.status"
source "$repo_root/minimax_h3_download_setup.sh"

MINIMAX_H3_FIRST_MODEL=ref2va select_model_priority >/dev/null
[ "$FIRST_MODEL_INDEX" = "1" ]
[ "$FIRST_MODEL_NAME" = "REF2VA" ]
[ "$DEFERRED_MODEL_INDEX" = "0" ]
[ "$DEFERRED_MODEL_NAME" = "FL2VA" ]

MINIMAX_H3_FIRST_MODEL=fl2va select_model_priority >/dev/null
[ "$FIRST_MODEL_INDEX" = "0" ]
[ "$FIRST_MODEL_NAME" = "FL2VA" ]
[ "$DEFERRED_MODEL_INDEX" = "1" ]
[ "$DEFERRED_MODEL_NAME" = "REF2VA" ]

MINIMAX_H3_FIRST_MODEL=invalid select_model_priority >/dev/null
[ "$FIRST_MODEL_INDEX" = "1" ]
[ "$FIRST_MODEL_NAME" = "REF2VA" ]
[ "$DEFERRED_MODEL_INDEX" = "0" ]
[ "$DEFERRED_MODEL_NAME" = "FL2VA" ]

"${PYTHON_TEST_BIN:-python3}" - "$tmp/source.safetensors" <<'PY'
import json
import struct
import sys

header = json.dumps({"tensor": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}}).encode()
with open(sys.argv[1], "wb") as handle:
    handle.write(struct.pack("<Q", len(header)))
    handle.write(header)
    handle.write(b"test")
PY

mkdir -p "$tmp/models"
DOWNLOAD_KINDS=("MODEL")
DOWNLOAD_URLS=("file://$tmp/source.safetensors")
DOWNLOAD_OUTPUTS=("$tmp/models/deferred.safetensors")
DOWNLOAD_SIZES=("$(stat -c '%s' "$tmp/source.safetensors")")
DOWNLOAD_SHA256=("$(sha256_file "$tmp/source.safetensors")")
FIRST_MODEL_NAME="REF2VA"

download_group() {
  cp "$tmp/source.safetensors" "${DOWNLOAD_OUTPUTS[0]}"
}
notify_ntfy() {
  return 0
}

start_deferred_model_download 0 "FL2VA" >/dev/null
background_pid=$!
wait "$background_pid"
grep -q '^state=ready$' "$MINIMAX_H3_STATUS_FILE"
grep -q 'Both MiniMax H3 diffusion models' "$MINIMAX_H3_STATUS_FILE"
validate_model_file "${DOWNLOAD_OUTPUTS[0]}" "${DOWNLOAD_SIZES[0]}" "${DOWNLOAD_SHA256[0]}"

echo "Download-priority tests passed."
