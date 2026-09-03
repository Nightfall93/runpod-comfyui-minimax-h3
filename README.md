# RunPod ComfyUI MiniMax H3 workflow template

Pinned RunPod image and cold-start setup for twelve MiniMax H3 workflows on Ada
and Blackwell. The bundle contains the ten workflows from the Pixaroma EP29
folder, a five-seed REF2VA seed hunter, and the MiniMax Seed Hunter v1.2.1
workflow with latent upscaling and seamless continuation. REF2VA is ready when
ComfyUI starts and FL2VA downloads in the background by default:

- `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- `minimax_h3_ref2va_pruned_int8_convrot.safetensors`

The shared Qwen3-VL text encoder and MiniMax audio/video VAEs are downloaded in
the blocking phase with the selected first diffusion model. These five base
files are pinned to Hugging Face revision
`eb8a16107c595128b3a578f82d2ce2f75920c355` and total 63,440,965,087 bytes
(about 59.1 GiB). Three additional Seed Hunter model files
totaling 3,872,055,292 bytes download in the background alongside FL2VA. The
24,636,301-byte RIFE interpolation model is verified and baked into the image.

## RunPod template settings

Publish this repository, let GitHub Actions build the images, and select the tag
matching the RunPod GPU:

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
- `MINIMAX_H3_FIRST_MODEL` (`ref2va` by default; set `fl2va` to reverse the
  foreground/background diffusion-model order)
- `MINIMAX_H3_DOWNLOAD_JOBS` (default `2`, allowed `1`-`4`)
- `MINIMAX_H3_DISK_RESERVE_GB` (default `10`)
- `MINIMAX_H3_MIN_DOWNLOAD_MIBPS` (default `5` before slow-link reconnects)
- `MINIMAX_H3_RECONNECT_LIMIT` (default `3` per file)
- `MINIMAX_H3_VERIFY_SHA256=1` (optional full hash pass; exact size and
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
- ComfyUI `12d5279438bfefc058a269eae805ceab6047777f` (v0.34.0), which contains
  native MiniMax H3 support
- ComfyUI-Pixaroma `433bbedc7f43d717fcb9e8e9aa9cbd26b0439226`, including the H3 audio-sync node
- Official SageAttention source `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`:
  SageAttention 2.2.0 is compiled for Ada 8.9; SageAttention 3 is compiled only
  for Blackwell 12.0 using pinned CUTLASS
  `dcf215af68a2d08d305076c152a06f201728cd53`
- The ten EP29 workflow JSON files, five-seed REF2VA seed hunter, and normalized
  v1.2.1 Seed Hunter workflow
- Eleven pinned Seed Hunter custom-node repositories; see
  `seed-hunter-node-lock.tsv`
- RIFE v4.26 `flownet.pkl`, verified during the image build

The original EP29 prompt-formula notes are retained under `resources/` for the
repository owner; they are reference material and are not copied into ComfyUI.
The workflow copies use Linux-portable `h3/...` model paths for RunPod.

Large models are deliberately not included in the image. They download to
Container Disk on first boot and valid completed files are skipped on later
boots. The small RIFE interpolation model is the only baked model asset.

## Cold-start safety and recovery

Before model bandwidth is used, the wrapper and setup perform independent CUDA
tensor checks, verify the exact GPU architecture, validate the pinned H3 core and
Pixaroma files, checksum the workflow bundle, probe all remote model sizes, and
confirm enough disk remains plus the configured reserve.

Before ComfyUI starts, the Sage bootstrap also checks the image's required Sage
major and package version, imports the architecture-specific extension, and runs
a small CUDA attention kernel. Ada starts with ComfyUI's Sage 2 flag; Blackwell
starts through ComfyUI's native `sage3` backend. A wrong package,
compute capability, failed import, or failed kernel smoke test stops startup.

Downloads use `.part` files, resume across restarts, run two at a time by default,
retry transient errors, reconnect persistently slow transfers, and move into the
final model path only after exact-size and safetensors-header validation. The
selected diffusion model and all shared assets block startup; the other diffusion
model continues in the background while ComfyUI is usable. A full SHA256 pass is
available through `MINIMAX_H3_VERIFY_SHA256=1`. Invalid completed or oversized
partial files are preserved with an `.invalid-TIMESTAMP` suffix.

Startup state is written atomically to:

```text
/workspace/runpod-slim/minimax-h3-download.status
```

ComfyUI starts when REF2VA, the text encoder, both base VAEs, and the applicable
workflow bundle have passed validation. FL2VA and the three additional Seed Hunter models then download
in the background. Completion or failure is reported in the container log,
optional ntfy notifications, and the status file.

## REF2VA seed hunter

`Minimax H3 - Seed Hunter - Five Seeds` shares one prompt, reference-conditioning
node, diffusion loader, text encoder, and both VAE loaders across five KSamplers.
All five samplers use identical generation settings and independently randomize
their seeds after each queued run. An execution barrier completes every diffusion
pass before any VAE decode begins, avoiding repeated UNET/VAE swapping. The five
MP4 branches use the prefixes `SeedHunter_S01` through `SeedHunter_S05` so each
result maps directly to its numbered sampler.

## Seed Hunter v1.2.1

The Civitai Seed Hunter workflow is included in both supported images. Its
creator-local audio/video selections are cleared, diffusion-model references
use Linux-portable `h3/...` paths, and the pinned INT8 video VAE, latent upscaler,
and TaeH3 preview model download in the background with FL2VA. SolAttn remains
bypassed in the managed workflow until the custom kernel is smoke-tested on a
real Ada and Blackwell pod. The existing global architecture-specific
SageAttention backend remains enabled.

## Local validation

From Git Bash or Linux:

```bash
bash ./tests/run_all.sh
```

The test suite covers all three driver/architecture gates, the Sage 2/Sage 3
build matrix and fatal mismatch paths, valid and invalid
safetensors headers, foreground/background model-priority selection, a mocked
partial-download resume, manifest installation and
customization preservation, manifest hashes, JSON parsing, Linux model paths,
workflow-family scoping, recursive subgraph inspection, model coverage,
custom-node lock coverage, required H3 node types, and referenced sample inputs.
The same suite runs before every image build in GitHub Actions.
