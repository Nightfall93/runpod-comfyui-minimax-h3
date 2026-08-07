#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

export MINIMAX_H3_SETUP_LIB_ONLY=1
source "$repo_root/minimax_h3_download_setup.sh"

"${PYTHON_TEST_BIN:-python3}" - "$tmp/source.safetensors" <<'PY'
import json
import struct
import sys

header = json.dumps({"tensor": {"dtype": "U8", "shape": [4096], "data_offsets": [0, 4096]}}).encode()
with open(sys.argv[1], "wb") as handle:
    handle.write(struct.pack("<Q", len(header)))
    handle.write(header)
    handle.write(bytes(range(256)) * 16)
PY

printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  'out=""' \
  'url=""' \
  'while [ "$#" -gt 0 ]; do' \
  '  case "$1" in' \
  '    -o) out="$2"; shift 2 ;;' \
  '    -H|--retry|--retry-delay|--connect-timeout|--speed-limit|--speed-time|--continue-at) shift 2 ;;' \
  '    -*) shift ;;' \
  '    *) url="$1"; shift ;;' \
  '  esac' \
  'done' \
  'source_path="${url#file://}"' \
  'existing=$(stat -c "%s" "$out" 2>/dev/null || echo 0)' \
  'tail -c "+$((existing + 1))" "$source_path" >> "$out"' \
  > "$tmp/mock-curl"
chmod +x "$tmp/mock-curl"

CURL_BIN="$tmp/mock-curl"
out="$tmp/models/test.safetensors"
mkdir -p "$(dirname "$out")"
head -c 1000 "$tmp/source.safetensors" > "${out}.part"
expected_size="$(stat -c '%s' "$tmp/source.safetensors")"
expected_sha="$(sha256_file "$tmp/source.safetensors")"

download_file "MODEL" "file://$tmp/source.safetensors" "$out" "$expected_size" "$expected_sha" >/dev/null
cmp "$tmp/source.safetensors" "$out"
[ ! -e "${out}.part" ]
validate_model_file "$out" "$expected_size" "$expected_sha"

echo "Resumable download test passed."
