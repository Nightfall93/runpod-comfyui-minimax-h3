# RunPod ComfyUI MiniMax H3 workflow template

Pinned RunPod image and cold-start setup for the ten MiniMax H3 workflows in
the Pixaroma EP29 workflow folder. Both H3 diffusion families are ready before
ComfyUI starts:

- `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- `minimax_h3_ref2va_pruned_int8_convrot.safetensors`

The shared Qwen3-VL text encoder and MiniMax audio/video VAEs are downloaded in
the same blocking cold start. All five files are pinned to Hugging Face revision
`eb8a16107c595128b3a578f82d2ce2f75920c355` and total 63,440,965,087 bytes
(about 59.1 GiB).

## RunPod template settings

Publish this repository, let GitHub Actions build the images, and select the tag
matching the RunPod GPU:

- Ampere compute capability 8.6 (A40, RTX A6000, RTX 30):
  `ghcr.io/nightfall93/runpod-comfyui-minimax-h3:cuda13-ampere` — SageAttention 2.2.0
- Ada compute capability 8.9 (L40/L40S, RTX 40):
  `ghcr.io/nightfall93/runpod-comfyui-minimax-h3:cuda13-ada` — SageAttention 2.2.0
- Blackwell compute capability 12.0 (RTX 50 and RTX PRO Blackwell):
  `ghcr.io/nightfall93/runpod-comfyui-minimax-h3:cuda13-blackwell` — SageAttention 3

Keep the container start command empty. Expose HTTP ports `8188`, `8080`, and
`8888`. Use at least 100 GB of Container Disk; 120 GB is recommended for useful
output and cache headroom. No Network Volume is required, but a reset or delete
also removes completed and partial model downloads.

The setup script, workflows, and sample inputs are baked into the image, so
`SETUP_SCRIPT_URL` is optional. After publishing the repository, set it when you
want new pods to consume the current setup logic without rebuilding the image:

```text
SETUP_SCRIPT_URL=https://raw.githubusercontent.com/Nightfall93/runpod-comfyui-minimax-h3/main/minimax_h3_download_setup.sh
```

Supported environment variables:

- `FILEBROWSER_USERNAME`
- `FILEBROWSER_PASSWORD` (use at least 12 characters)
- `JUPYTER_PASSWORD`
- `HF_TOKEN` (optional, but recommended for better Hugging Face rate limits)
- `NTFY_TOPIC`, `NTFY_SERVER_URL`, and `NTFY_TOKEN` (optional notifications)
- `MINIMAX_H3_DOWNLOAD_JOBS` (default `2`, allowed `1`-`4`)
- `MINIMAX_H3_DISK_RESERVE_GB` (default `10`)
- `MINIMAX_H3_MIN_DOWNLOAD_MIBPS` (default `5` before slow-link reconnects)
- `MINIMAX_H3_RECONNECT_LIMIT` (default `3` per file)
- `MINIMAX_H3_VERIFY_SHA256=1` (optional full 59.1 GiB hash pass; exact size and
  safetensors headers are always validated)
- `MINIMAX_H3_REFRESH_WORKFLOWS=1` (replace installed workflows from the managed
  bundle for one startup)
- `MINIMAX_H3_REFRESH_MEDIA=1` (replace managed sample inputs for one startup)
- `ENABLE_SAGE_ATTENTION=0` (emergency opt-out; Sage is required by default)
- `SAGE_ATTENTION_RUNTIME_SMOKE=0` (skip the startup CUDA-kernel smoke test;
  leave enabled unless diagnosing a driver problem)

## What is pinned in the image

- RunPod `cuda13.0` base image by immutable digest
- PyTorch CUDA 13.0 and a host-driver gate requiring major version 580+
- ComfyUI `dec5d9450a5290bcf63430409ea41018e67f41c3` (v0.30.2), which contains
  native MiniMax H3 support
- ComfyUI-Pixaroma `433bbedc7f43d717fcb9e8e9aa9cbd26b0439226`, including the H3 audio-sync node
- Official SageAttention source `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`:
  SageAttention 2.2.0 is compiled separately for Ampere 8.6 and Ada 8.9;
  SageAttention 3 is compiled only for Blackwell 12.0 using pinned CUTLASS
  `dcf215af68a2d08d305076c152a06f201728cd53`
- The ten EP29 workflow JSON files and every non-empty sample input they reference

The original EP29 prompt-formula notes are retained under `resources/` for the
repository owner; they are reference material and are not copied into ComfyUI.
The workflow copies use Linux-portable `h3/...` model paths for RunPod.

Models are deliberately not included in the image. They download to Container
Disk on first boot and valid completed files are skipped on later boots.

## Cold-start safety and recovery

Before model bandwidth is used, the wrapper and setup perform independent CUDA
tensor checks, verify the exact GPU architecture, validate the pinned H3 core and
Pixaroma files, checksum the workflow bundle, probe all remote model sizes, and
confirm enough disk remains plus the configured reserve.

Before ComfyUI starts, the Sage bootstrap also checks the image's required Sage
major and package version, imports the architecture-specific extension, and runs
a small CUDA attention kernel. Ampere/Ada start with ComfyUI's Sage 2 flag;
Blackwell starts through ComfyUI's native `sage3` backend. A wrong package,
compute capability, failed import, or failed kernel smoke test stops startup.

Downloads use `.part` files, resume across restarts, run two at a time by default,
retry transient errors, reconnect persistently slow transfers, and move into the
final model path only after exact-size and safetensors-header validation. A full
SHA256 pass is available through `MINIMAX_H3_VERIFY_SHA256=1`. Invalid completed
or oversized partial files are preserved with an `.invalid-TIMESTAMP` suffix.

Startup state is written atomically to:

```text
/workspace/runpod-slim/minimax-h3-download.status
```

ComfyUI starts only when both diffusion models, the text encoder, both VAEs, and
all ten workflows have passed validation.

## Local validation

From Git Bash or Linux:

```bash
bash ./tests/run_all.sh
```

The test suite covers all three driver/architecture gates, the Sage 2/Sage 3
build matrix and fatal mismatch paths, valid and invalid
safetensors headers, a mocked partial-download resume, manifest installation and
customization preservation, manifest hashes, JSON parsing, Linux model paths,
exact workflow count, model coverage, required H3 node types, and referenced
sample inputs. The same suite runs before every image build in GitHub Actions.
