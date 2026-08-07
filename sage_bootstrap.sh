#!/usr/bin/env bash
set -euo pipefail

sage_bootstrap_main() {
  echo "=== SageAttention CUDA 13 bootstrap starting ==="

  local comfy="${COMFY:-/workspace/runpod-slim/ComfyUI}"
  local python_bin="${SAGE_PYTHON_BIN:-${COMFY_VENV:-$comfy/.venv-cu130}/bin/python}"
  local args_file="${COMFY_ARGS_FILE:-/workspace/runpod-slim/comfyui_args.txt}"
  local gpu_family="${MINIMAX_H3_GPU_FAMILY:-}"
  local configured_cc="${SAGE_SUPPORTED_CC:-}"
  local configured_major="${SAGE_ATTENTION_MAJOR:-}"
  local configured_version="${SAGE_ATTENTION_VERSION:-}"
  local require_sage="${REQUIRE_SAGE_ATTENTION:-1}"

  disable_sage_attention() {
    mkdir -p "$(dirname "$args_file")"
    if [ -f "$args_file" ]; then
      sed -i -E \
        's/(^|[[:space:]])--use-sage-attention([[:space:]]|$)/ /g' \
        "$args_file"
    fi
    export COMFY_SAGE_ATTENTION3=0
  }

  enable_sage_attention() {
    local major="$1"
    mkdir -p "$(dirname "$args_file")"
    touch "$args_file"
    sed -i -E \
      's/(^|[[:space:]])--use-(flash|sage)-attention([[:space:]]|$)/ /g' \
      "$args_file"

    if [ "$major" = "2" ]; then
      echo " --use-sage-attention" >> "$args_file"
      export COMFY_SAGE_ATTENTION3=0
    else
      export COMFY_SAGE_ATTENTION3=1
    fi

    if ! grep -q -- "--preview-method none" "$args_file"; then
      echo " --preview-method none" >> "$args_file"
    fi
  }

  sage_failure() {
    local message="$1"
    disable_sage_attention
    if [ "$require_sage" = "1" ]; then
      echo "ERROR: $message" >&2
      return 1
    fi
    echo "WARNING: $message Continuing without SageAttention." >&2
    return 0
  }

  if [ "${ENABLE_SAGE_ATTENTION:-1}" != "1" ]; then
    echo "ENABLE_SAGE_ATTENTION is disabled; continuing without SageAttention."
    disable_sage_attention
    return 0
  fi

  local expected_cc expected_major expected_version expected_sage3
  case "$gpu_family" in
    ampere)
      expected_cc="8.6"
      expected_major="2"
      expected_version="2.2.0"
      expected_sage3="0"
      ;;
    ada)
      expected_cc="8.9"
      expected_major="2"
      expected_version="2.2.0"
      expected_sage3="0"
      ;;
    blackwell)
      expected_cc="12.0"
      expected_major="3"
      expected_version="1.0.0"
      expected_sage3="1"
      ;;
    *)
      sage_failure "Unknown MiniMax H3 GPU family: ${gpu_family:-unset}."
      return $?
      ;;
  esac

  if [ "$configured_cc" != "$expected_cc" ]; then
    sage_failure "$gpu_family image expects Sage compute capability $expected_cc, not ${configured_cc:-unset}."
    return $?
  fi
  if [ "$configured_major" != "$expected_major" ]; then
    sage_failure "$gpu_family requires SageAttention $expected_major, not ${configured_major:-unset}."
    return $?
  fi
  if [ "$configured_version" != "$expected_version" ]; then
    sage_failure "$gpu_family requires Sage package version $expected_version, not ${configured_version:-unset}."
    return $?
  fi
  if [ "${COMFY_SAGE_ATTENTION3:-0}" != "$expected_sage3" ]; then
    sage_failure "$gpu_family has an invalid COMFY_SAGE_ATTENTION3 setting."
    return $?
  fi

  if [ ! -x "$python_bin" ]; then
    sage_failure "ComfyUI CUDA 13 venv not found at $python_bin."
    return $?
  fi

  local gpu_cc
  if ! gpu_cc=$("$python_bin" - <<'PY_CAPABILITY' 2>/dev/null
import torch
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
major, minor = torch.cuda.get_device_capability(0)
print(f"{major}.{minor}")
PY_CAPABILITY
  ); then
    sage_failure "No usable CUDA GPU detected for SageAttention."
    return $?
  fi

  echo "Detected GPU compute capability: $gpu_cc"
  if [ "$gpu_cc" != "$expected_cc" ]; then
    sage_failure "$gpu_family SageAttention $expected_major wheel supports $expected_cc, not $gpu_cc."
    return $?
  fi

  local sage_probe
  if ! sage_probe=$("$python_bin" - "$expected_major" "$expected_version" "${SAGE_ATTENTION_RUNTIME_SMOKE:-1}" <<'PY_SAGE' 2>&1
import importlib.metadata
import sys

import torch

major, expected_version, run_smoke = sys.argv[1:]
if major == "2":
    from sageattention import sageattn
    package = "sageattention"
    attention = sageattn
elif major == "3":
    from sageattn3 import sageattn3_blackwell
    package = "sageattn3"
    attention = sageattn3_blackwell
else:
    raise RuntimeError(f"unsupported SageAttention major: {major}")

actual_version = importlib.metadata.version(package)
if actual_version != expected_version:
    raise RuntimeError(
        f"{package} version mismatch: expected {expected_version}, got {actual_version}"
    )

smoke_status = "disabled"
if run_smoke == "1":
    q = torch.randn((1, 4, 2048, 64), device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    output = attention(q, k, v, tensor_layout="HND", is_causal=False) if major == "2" else attention(q, k, v, is_causal=False)
    torch.cuda.synchronize()
    if output.shape != q.shape or not torch.isfinite(output).all().item():
        raise RuntimeError(
            f"SageAttention {major} CUDA smoke test returned an invalid tensor"
        )
    smoke_status = "passed"

print(f"SageAttention {major} package {package} {actual_version}; CUDA smoke={smoke_status}")
PY_SAGE
  ); then
    sage_failure "SageAttention $expected_major validation failed: $sage_probe"
    return $?
  fi

  enable_sage_attention "$expected_major"
  echo "$sage_probe"
  if [ "$expected_major" = "2" ]; then
    echo "SageAttention 2 is enabled globally through ComfyUI's Sage flag."
  else
    echo "SageAttention 3 is enabled globally through ComfyUI's native sage3 backend."
  fi
  echo "=== SageAttention CUDA 13 bootstrap finished ==="
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  sage_bootstrap_main "$@"
fi
