#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if python3 -c "pass" >/dev/null 2>&1; then
  python_bin=python3
elif python -c "pass" >/dev/null 2>&1; then
  python_bin=python
else
  echo "Python 3 is required for the template tests." >&2
  exit 1
fi
export MINIMAX_H3_PYTHON_BIN="$python_bin"
export PYTHON_TEST_BIN="$python_bin"

bash -n "$repo_root"/*.sh "$repo_root"/tests/*.sh
bash "$repo_root/tests/driver_preflight_test.sh"
bash "$repo_root/tests/sage_matrix_test.sh"
bash "$repo_root/tests/safetensors_validator_test.sh"
bash "$repo_root/tests/resumable_download_test.sh"
bash "$repo_root/tests/download_priority_test.sh"
bash "$repo_root/tests/bundle_install_test.sh"
"$python_bin" "$repo_root/tests/validate_bundle.py" --root "$repo_root"

echo "All MiniMax H3 template tests passed."
