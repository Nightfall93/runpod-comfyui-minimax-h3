#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

printf '%s\n' \
  '#!/usr/bin/env bash' \
  'printf "%s\\n" "${MOCK_DRIVER:?}"' \
  > "$tmp/nvidia-smi"

printf '%s\n' \
  '#!/usr/bin/env bash' \
  'cat >/dev/null' \
  'printf "Mock GPU|%s|13.0\\n" "${MOCK_CC:?}"' \
  > "$tmp/python3"

chmod +x "$tmp/nvidia-smi" "$tmp/python3"

for expected_cc in 8.6 8.9 12.0; do
  case "$expected_cc" in
    8.6) gpu_family=ampere; wrong_cc=8.9 ;;
    8.9) gpu_family=ada; wrong_cc=8.6 ;;
    12.0) gpu_family=blackwell; wrong_cc=8.9 ;;
  esac

  run_preflight() {
    NVIDIA_SMI_BIN="$tmp/nvidia-smi" \
    CUDA_PROBE_PYTHON="$tmp/python3" \
    MINIMAX_H3_REQUIRED_DRIVER_MAJOR=580 \
    SAGE_SUPPORTED_CC="$expected_cc" \
    MINIMAX_H3_GPU_FAMILY="$gpu_family" \
    bash "$repo_root/driver_preflight.sh"
  }

  MOCK_DRIVER=580.95.05 MOCK_CC="$expected_cc" run_preflight \
    | grep -q "CUDA 13 preflight passed"

  if MOCK_DRIVER=579.99 MOCK_CC="$expected_cc" run_preflight >"$tmp/old-driver.log" 2>&1; then
    echo "Expected driver 579 to fail for $gpu_family." >&2
    exit 1
  fi
  grep -q "below the CUDA 13 requirement" "$tmp/old-driver.log"

  if MOCK_DRIVER=580.95.05 MOCK_CC="$wrong_cc" run_preflight >"$tmp/wrong-cc.log" 2>&1; then
    echo "Expected compute capability $wrong_cc to fail for $gpu_family." >&2
    exit 1
  fi
  grep -q "expects compute capability $expected_cc" "$tmp/wrong-cc.log"
done

echo "Driver preflight tests passed for Ampere, Ada, and Blackwell."
