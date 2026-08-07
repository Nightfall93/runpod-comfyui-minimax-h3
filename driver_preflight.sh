#!/usr/bin/env bash
set -euo pipefail

required_driver_major="${MINIMAX_H3_REQUIRED_DRIVER_MAJOR:-580}"
expected_cc="${SAGE_SUPPORTED_CC:-}"
nvidia_smi_bin="${NVIDIA_SMI_BIN:-nvidia-smi}"
cuda_probe_python="${CUDA_PROBE_PYTHON:-python3}"

fatal() {
  echo "FATAL: $*" >&2
  echo "No MiniMax H3 setup script or model download was started." >&2
  exit 78
}

command -v "$nvidia_smi_bin" >/dev/null 2>&1 || \
  fatal "nvidia-smi is unavailable; the RunPod host GPU driver was not mounted."

driver_version="$("$nvidia_smi_bin" --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d '[:space:]')"
[[ "$driver_version" =~ ^[0-9]+([.][0-9]+)*$ ]] || \
  fatal "could not parse RunPod host driver version: ${driver_version:-empty}"

driver_major="${driver_version%%.*}"
if (( driver_major < required_driver_major )); then
  fatal "RunPod host driver $driver_version is below the CUDA 13 requirement ($required_driver_major or newer). Replace this host."
fi

cuda_probe="$(timeout 30s "$cuda_probe_python" - <<'PY_CUDA'
import torch

if torch.version.cuda != "13.0":
    raise RuntimeError(f"expected PyTorch CUDA 13.0, found {torch.version.cuda}")
if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() returned False")

device = torch.cuda.current_device()
probe = torch.zeros(1, device=device)
torch.cuda.synchronize(device)
major, minor = torch.cuda.get_device_capability(device)
print(f"{torch.cuda.get_device_name(device)}|{major}.{minor}|{torch.version.cuda}")
del probe
PY_CUDA
)" || fatal "CUDA 13 tensor preflight failed."

IFS='|' read -r gpu_name gpu_cc torch_cuda <<< "$cuda_probe"
if [ -n "$expected_cc" ] && [ "$gpu_cc" != "$expected_cc" ]; then
  fatal "this ${MINIMAX_H3_GPU_FAMILY:-GPU}-specific image expects compute capability $expected_cc, but $gpu_name reports $gpu_cc. Select the correct MiniMax H3 CUDA 13 image."
fi

echo "CUDA 13 preflight passed: $gpu_name, compute capability $gpu_cc, host driver $driver_version, PyTorch CUDA $torch_cuda."
