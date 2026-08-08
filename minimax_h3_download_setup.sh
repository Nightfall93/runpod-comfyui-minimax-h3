#!/usr/bin/env bash
set -euo pipefail

COMFY="${COMFY:-/workspace/runpod-slim/ComfyUI}"
VENV="${COMFY_VENV:-$COMFY/.venv-cu130}"
DOWNLOAD_JOBS="${MINIMAX_H3_DOWNLOAD_JOBS:-2}"
STATUS_FILE="${MINIMAX_H3_STATUS_FILE:-/workspace/runpod-slim/minimax-h3-download.status}"
BUNDLE_ROOT="${MINIMAX_H3_BUNDLE_ROOT:-/opt/minimax-h3-bundle}"
MODEL_REVISION="eb8a16107c595128b3a578f82d2ce2f75920c355"
EXPECTED_CORE_COMMIT="${COMFYUI_H3_COMMIT:-dec5d9450a5290bcf63430409ea41018e67f41c3}"
PIXAROMA_COMMIT="${PIXAROMA_H3_COMMIT:-433bbedc7f43d717fcb9e8e9aa9cbd26b0439226}"
CURL_BIN="${CURL_BIN:-curl}"
BOOTSTRAP_PYTHON="${MINIMAX_H3_PYTHON_BIN:-python3}"

if ! [[ "$DOWNLOAD_JOBS" =~ ^[1-4]$ ]]; then
  echo "WARNING: MINIMAX_H3_DOWNLOAD_JOBS must be from 1 to 4; using 2."
  DOWNLOAD_JOBS=2
fi

notify_ntfy() {
  local title="$1"
  local priority="$2"
  local tags="$3"
  local message="$4"
  local server="${NTFY_SERVER_URL:-https://ntfy.sh}"
  local -a auth=()

  [ -n "${NTFY_TOPIC:-}" ] || return 0
  if [ -n "${NTFY_TOKEN:-}" ]; then
    auth=(-H "Authorization: Bearer ${NTFY_TOKEN}")
  fi

  if ! "$CURL_BIN" -fsS --max-time 10 --retry 2 \
    "${auth[@]}" \
    -H "Title: $title" \
    -H "Priority: $priority" \
    -H "Tags: $tags" \
    --data-binary "$message" \
    "${server%/}/${NTFY_TOPIC#/}" >/dev/null; then
    echo "WARNING: Could not send ntfy notification: $title"
  fi
}

write_status() {
  local state="$1"
  local detail="$2"
  local temp="${STATUS_FILE}.tmp"
  mkdir -p "$(dirname "$STATUS_FILE")"
  {
    printf 'state=%s\n' "$state"
    printf 'updated_at=%s\n' "$(date -Is)"
    printf 'detail=%s\n' "$detail"
    printf 'model_revision=%s\n' "$MODEL_REVISION"
  } > "$temp"
  mv "$temp" "$STATUS_FILE"
}

format_eta() {
  local seconds="${1:-0}"
  printf '%02d:%02d:%02d' \
    "$((seconds / 3600))" "$(((seconds / 60) % 60))" "$((seconds % 60))"
}

file_size() {
  stat -c '%s' "$1" 2>/dev/null || echo 0
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

validate_safetensors_file() {
  local path="$1"
  local expected_size="${2:-0}"
  "$BOOTSTRAP_PYTHON" - "$path" "$expected_size" <<'PY_SAFE'
import json
import os
import struct
import sys

path = sys.argv[1]
expected = int(sys.argv[2])
size = os.path.getsize(path)
if expected and size != expected:
    raise SystemExit(f"size mismatch: expected {expected}, found {size}")
with open(path, "rb") as handle:
    raw = handle.read(8)
    if len(raw) != 8:
        raise SystemExit("missing safetensors header length")
    header_size = struct.unpack("<Q", raw)[0]
    if header_size < 2 or header_size > min(128 * 1024 * 1024, size - 8):
        raise SystemExit(f"invalid safetensors header length: {header_size}")
    header = json.loads(handle.read(header_size))
if not isinstance(header, dict) or not header:
    raise SystemExit("safetensors header is empty or not an object")
print(f"valid safetensors header ({len(header)} entries)")
PY_SAFE
}

validate_model_file() {
  local path="$1"
  local expected_size="$2"
  local expected_sha256="$3"
  [ -f "$path" ] || return 1
  [ "$(file_size "$path")" = "$expected_size" ] || return 1
  validate_safetensors_file "$path" "$expected_size" >/dev/null || return 1
}

quarantine_invalid_file() {
  local path="$1"
  local reason="$2"
  local backup="${path}.invalid-$(date +%Y%m%dT%H%M%S)"
  [ -e "$path" ] || return 0
  mv "$path" "$backup"
  echo "WARNING: Moved invalid $reason to $backup"
}

remote_size() {
  local url="$1"
  local -a headers=()
  if [ -n "${HF_TOKEN:-}" ]; then
    headers=(-H "Authorization: Bearer ${HF_TOKEN}")
  fi
  "$CURL_BIN" -fsSIL --retry 3 --retry-all-errors --connect-timeout 30 \
    "${headers[@]}" "$url" | awk '
    BEGIN { IGNORECASE=1 }
    /^content-length:/ { size=$2 }
    END { gsub(/\r/, "", size); print size }
  ' || true
}

preflight_cuda() {
  local cuda_ready=0 attempt cuda_details
  echo "Running independent CUDA tensor preflight before model setup..."
  for attempt in 1 2 3; do
    if cuda_details=$(timeout 20s "$BOOTSTRAP_PYTHON" - <<'PY_CUDA' 2>&1
import torch

if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() returned False")
device = torch.cuda.current_device()
probe = torch.zeros(1, device=device)
torch.cuda.synchronize(device)
major, minor = torch.cuda.get_device_capability(device)
print(
    f"{torch.cuda.get_device_name(device)} "
    f"(compute capability {major}.{minor}, PyTorch CUDA {torch.version.cuda})"
)
del probe
PY_CUDA
    ); then
      echo "CUDA tensor preflight passed: $cuda_details"
      cuda_ready=1
      break
    fi

    echo "CUDA tensor preflight attempt $attempt/3 failed:"
    printf '%s\n' "$cuda_details"
    if [ "$attempt" -lt 3 ]; then
      echo "Retrying CUDA tensor preflight in 5 seconds..."
      sleep 5
    fi
  done

  if [ "$cuda_ready" -ne 1 ]; then
    notify_ntfy \
      "RunPod CUDA failure" "urgent" "warning" \
      "MiniMax H3 CUDA compute failed before any models were downloaded. Replace or reset this host."
    echo "FATAL: CUDA compute is unavailable after 3 attempts." >&2
    return 1
  fi
}

ensure_comfyui() {
  # The RunPod base normally copies ComfyUI on first boot. Creating the target
  # first breaks that path, so copy the image-baked source only when main.py is absent.
  if [ ! -f "$COMFY/main.py" ]; then
    echo "ComfyUI not found in /workspace yet. Copying the pinned H3 build..."
    test -f /opt/comfyui-baked/main.py
    mkdir -p /workspace/runpod-slim
    cp -r /opt/comfyui-baked "$COMFY"
  fi

  if [ ! -f "$COMFY/comfy_extras/nodes_minimax_h3.py" ] \
    || [ ! -f "$COMFY/comfy/text_encoders/minimax.py" ] \
    || [ ! -f "$COMFY/comfy/ldm/minimax/model.py" ]; then
    echo "FATAL: This ComfyUI image predates native MiniMax H3 support." >&2
    echo "Use a MiniMax H3 image from this repository; model downloads were not started." >&2
    return 1
  fi

  if [ -f "$COMFY/.minimax-h3-core-commit" ]; then
    installed_commit="$(tr -d '[:space:]' < "$COMFY/.minimax-h3-core-commit")"
    if [ "$installed_commit" != "$EXPECTED_CORE_COMMIT" ]; then
      echo "FATAL: Baked ComfyUI commit is $installed_commit, expected $EXPECTED_CORE_COMMIT." >&2
      return 1
    fi
    echo "Pinned ComfyUI H3 core verified: $installed_commit"
  else
    echo "WARNING: H3 core files exist without the managed commit marker; continuing with compatibility checks."
  fi

  if [ ! -d "$VENV" ]; then
    echo "Creating isolated ComfyUI CUDA 13 venv..."
    cd "$COMFY"
    python3.12 -m venv --system-site-packages "$VENV"
    source "$VENV/bin/activate"
    python -m ensurepip
  else
    source "$VENV/bin/activate"
  fi

  mkdir -p \
    "$COMFY/custom_nodes" \
    "$COMFY/models/diffusion_models/h3" \
    "$COMFY/models/text_encoders" \
    "$COMFY/models/vae" \
    "$COMFY/user/default/workflows" \
    "$COMFY/input"
}

ensure_pixaroma_h3() {
  local target="$COMFY/custom_nodes/ComfyUI-Pixaroma"
  local staging="${target}.minimax-h3-part"
  local backup

  if [ -f "$target/nodes/node_h3_audio_sync.py" ] \
    && grep -q 'PixaromaH3AudioSync' "$target/nodes/node_h3_audio_sync.py"; then
    echo "Ready NODE      ComfyUI-Pixaroma MiniMax H3 support"
    return 0
  fi

  if [ -e "$target" ]; then
    backup="${target}.pre-minimax-h3-$(date +%Y%m%dT%H%M%S)"
    mv "$target" "$backup"
    echo "Preserved incompatible Pixaroma installation at: $backup"
  fi

  echo "Installing pinned ComfyUI-Pixaroma H3 nodes: $PIXAROMA_COMMIT"
  rm -rf "$staging"
  git init "$staging"
  git -C "$staging" remote add origin https://gitlab.com/pixaroma/ComfyUI-Pixaroma.git
  git -C "$staging" fetch --depth 1 origin "$PIXAROMA_COMMIT"
  git -C "$staging" checkout --detach FETCH_HEAD
  test -f "$staging/nodes/node_h3_audio_sync.py"
  grep -q 'PixaromaH3AudioSync' "$staging/nodes/node_h3_audio_sync.py"
  rm -rf "$staging/.git"
  printf '%s\n' "$PIXAROMA_COMMIT" > "$staging/.minimax-h3-managed-commit"
  mv "$staging" "$target"
  echo "Installed NODE  ComfyUI-Pixaroma MiniMax H3 support"
}

resolve_script_base_url() {
  if [ -n "${MINIMAX_H3_ASSET_BASE_URL:-}" ]; then
    SCRIPT_BASE_URL="${MINIMAX_H3_ASSET_BASE_URL%/}"
  elif [ -n "${SETUP_SCRIPT_URL:-}" ]; then
    SCRIPT_BASE_URL="${SETUP_SCRIPT_URL%/*}"
  elif [ -n "${MINIMAX_H3_SETUP_SCRIPT_URL:-}" ]; then
    SCRIPT_BASE_URL="${MINIMAX_H3_SETUP_SCRIPT_URL%/*}"
  else
    SCRIPT_BASE_URL=""
  fi
}

install_bundle() {
  local manifest="$BUNDLE_ROOT/asset-manifest.tsv"
  local temp_manifest="/tmp/minimax-h3-asset-manifest.tsv"
  local kind repo_path install_path expected_sha source dest part actual_sha refresh

  if [ ! -s "$manifest" ]; then
    if [ -z "$SCRIPT_BASE_URL" ]; then
      echo "FATAL: The baked workflow bundle is missing and no asset base URL is available." >&2
      return 1
    fi
    echo "Downloading workflow bundle manifest fallback..."
    "$CURL_BIN" -fsSL --retry 5 --retry-all-errors --retry-delay 2 \
      "$SCRIPT_BASE_URL/asset-manifest.tsv" -o "$temp_manifest"
    manifest="$temp_manifest"
  fi

  while IFS=$'\t' read -r kind repo_path install_path expected_sha; do
    [ -n "$kind" ] || continue
    [[ "$kind" = \#* ]] && continue
    [ -n "$repo_path" ] && [ -n "$install_path" ] && [ -n "$expected_sha" ]
    dest="$COMFY/$install_path"
    part="${dest}.part"
    refresh=0
    [ "$kind" = "workflow" ] && refresh="${MINIMAX_H3_REFRESH_WORKFLOWS:-0}"
    [ "$kind" = "input" ] && refresh="${MINIMAX_H3_REFRESH_MEDIA:-0}"

    if [ -f "$dest" ] && [ "$refresh" != "1" ]; then
      actual_sha="$(sha256_file "$dest")"
      if [ "$actual_sha" = "$expected_sha" ]; then
        printf 'Ready %-8s %s\n' "$kind" "$install_path"
      else
        printf 'Preserved customized %-8s %s\n' "$kind" "$install_path"
      fi
      continue
    fi

    mkdir -p "$(dirname "$dest")"
    source="$BUNDLE_ROOT/$repo_path"
    if [ -f "$source" ]; then
      cp "$source" "$part"
    else
      if [ -z "$SCRIPT_BASE_URL" ]; then
        echo "FATAL: Missing baked bundle asset and no fallback URL: $repo_path" >&2
        return 1
      fi
      "$CURL_BIN" -fsSL --retry 5 --retry-all-errors --retry-delay 2 \
        "$SCRIPT_BASE_URL/$repo_path" -o "$part"
    fi

    actual_sha="$(sha256_file "$part")"
    if [ "$actual_sha" != "$expected_sha" ]; then
      echo "FATAL: Bundle SHA256 mismatch for $repo_path" >&2
      return 1
    fi
    if [ "$kind" = "workflow" ]; then
      python -m json.tool "$part" >/dev/null
    fi
    mv "$part" "$dest"
    printf 'Installed %-8s %s\n' "$kind" "$install_path"
  done < "$manifest"
}

validate_installed_workflows() {
  local workflow_root="$COMFY/user/default/workflows/MiniMax H3"
  python - "$workflow_root" <<'PY_WORKFLOWS'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
files = sorted(root.rglob("*.json"))
if len(files) != 10:
    raise SystemExit(f"expected 10 MiniMax H3 workflows, found {len(files)}")

required_models = {
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "minimax_h3_audio_vae_fp32.safetensors",
    "minimax_h3_video_vae_fp16.safetensors",
}
seen_models = set()
seen_nodes = set()

def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)

for path in files:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data.get("nodes"), list):
        raise SystemExit(f"workflow has no node list: {path}")
    for node in data["nodes"]:
        seen_nodes.add(node.get("type"))
        for value in strings(node.get("widgets_values")):
            name = value.replace("\\", "/").rsplit("/", 1)[-1]
            if name in required_models:
                seen_models.add(name)

missing_models = required_models - seen_models
missing_nodes = {"MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo", "PixaromaH3AudioSync"} - seen_nodes
if missing_models:
    print(f"WARNING: customized installed workflows do not reference: {sorted(missing_models)}")
if missing_nodes:
    print(f"WARNING: customized installed workflows do not exercise: {sorted(missing_nodes)}")
print(f"Validated JSON for {len(files)} installed MiniMax H3 workflows.")
PY_WORKFLOWS
}

init_model_manifest() {
  local base="https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/$MODEL_REVISION"
  DOWNLOAD_KINDS=("MODEL" "MODEL" "TE" "VAE" "VAE")
  DOWNLOAD_URLS=(
    "$base/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    "$base/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    "$base/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    "$base/vae/minimax_h3_audio_vae_fp32.safetensors"
    "$base/vae/minimax_h3_video_vae_fp16.safetensors"
  )
  DOWNLOAD_OUTPUTS=(
    "$COMFY/models/diffusion_models/h3/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    "$COMFY/models/diffusion_models/h3/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    "$COMFY/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    "$COMFY/models/vae/minimax_h3_audio_vae_fp32.safetensors"
    "$COMFY/models/vae/minimax_h3_video_vae_fp16.safetensors"
  )
  DOWNLOAD_SIZES=(
    "20970379616"
    "20970379616"
    "15687142551"
    "605254808"
    "5207808496"
  )
  DOWNLOAD_SHA256=(
    "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
    "9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779"
    "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6"
    "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48"
    "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522"
  )
}

select_model_priority() {
  local configured="${MINIMAX_H3_FIRST_MODEL:-ref2va}"
  configured="${configured,,}"

  case "$configured" in
    ref2va)
      FIRST_MODEL_INDEX=1
      FIRST_MODEL_NAME="REF2VA"
      DEFERRED_MODEL_INDEX=0
      DEFERRED_MODEL_NAME="FL2VA"
      ;;
    fl2va)
      FIRST_MODEL_INDEX=0
      FIRST_MODEL_NAME="FL2VA"
      DEFERRED_MODEL_INDEX=1
      DEFERRED_MODEL_NAME="REF2VA"
      ;;
    *)
      echo "WARNING: MINIMAX_H3_FIRST_MODEL must be 'ref2va' or 'fl2va'; using ref2va."
      FIRST_MODEL_INDEX=1
      FIRST_MODEL_NAME="REF2VA"
      DEFERRED_MODEL_INDEX=0
      DEFERRED_MODEL_NAME="FL2VA"
      ;;
  esac

  echo "Startup priority: $FIRST_MODEL_NAME is required before ComfyUI starts; $DEFERRED_MODEL_NAME will download in the background."
}

prepare_local_models() {
  local index out part expected actual
  for index in "${!DOWNLOAD_OUTPUTS[@]}"; do
    out="${DOWNLOAD_OUTPUTS[$index]}"
    part="${out}.part"
    expected="${DOWNLOAD_SIZES[$index]}"
    mkdir -p "$(dirname "$out")"

    if [ -e "$out" ] && ! validate_model_file "$out" "$expected" "${DOWNLOAD_SHA256[$index]}"; then
      quarantine_invalid_file "$out" "completed model"
    fi

    if [ -e "$part" ]; then
      actual="$(file_size "$part")"
      if [ "$actual" -gt "$expected" ]; then
        quarantine_invalid_file "$part" "oversized partial model"
      elif [ "$actual" -eq "$expected" ]; then
        if validate_model_file "$part" "$expected" "${DOWNLOAD_SHA256[$index]}"; then
          mv "$part" "$out"
          echo "Recovered complete model from partial file: $(basename "$out")"
        else
          quarantine_invalid_file "$part" "invalid complete partial model"
        fi
      fi
    fi
  done
}

preflight_remote_models() {
  local index observed expected
  echo "Probing pinned Hugging Face assets before downloading..."
  for index in "${!DOWNLOAD_URLS[@]}"; do
    if validate_model_file "${DOWNLOAD_OUTPUTS[$index]}" \
      "${DOWNLOAD_SIZES[$index]}" "${DOWNLOAD_SHA256[$index]}"; then
      continue
    fi
    observed="$(remote_size "${DOWNLOAD_URLS[$index]}")"
    expected="${DOWNLOAD_SIZES[$index]}"
    if [ "$observed" != "$expected" ]; then
      echo "FATAL: Remote-size preflight failed for $(basename "${DOWNLOAD_OUTPUTS[$index]}")." >&2
      echo "Expected $expected bytes from revision $MODEL_REVISION; observed ${observed:-unknown}." >&2
      return 1
    fi
    echo "Remote MODEL   $(basename "${DOWNLOAD_OUTPUTS[$index]}") ($observed bytes)"
  done
}

preflight_disk_space() {
  local index out part expected have remaining=0 available reserve_gib reserve required
  reserve_gib="${MINIMAX_H3_DISK_RESERVE_GB:-10}"
  [[ "$reserve_gib" =~ ^[0-9]+$ ]] || reserve_gib=10
  reserve=$((reserve_gib * 1024 * 1024 * 1024))

  for index in "${!DOWNLOAD_OUTPUTS[@]}"; do
    out="${DOWNLOAD_OUTPUTS[$index]}"
    part="${out}.part"
    expected="${DOWNLOAD_SIZES[$index]}"
    if validate_model_file "$out" "$expected" "${DOWNLOAD_SHA256[$index]}"; then
      continue
    fi
    have="$(file_size "$part")"
    [ "$have" -gt "$expected" ] && have=0
    remaining=$((remaining + expected - have))
  done

  available="$(df -PB1 "$COMFY" | awk 'NR==2 {print $4}')"
  [[ "$available" =~ ^[0-9]+$ ]] || {
    echo "FATAL: Could not determine available container-disk space." >&2
    return 1
  }
  required=$((remaining + reserve))
  echo "Disk preflight: $(numfmt --to=iec-i --suffix=B "$remaining") remains to download; preserving $(numfmt --to=iec-i --suffix=B "$reserve") free."
  if [ "$available" -lt "$required" ]; then
    echo "FATAL: Insufficient container disk for MiniMax H3 cold start." >&2
    echo "Available: $(numfmt --to=iec-i --suffix=B "$available"); required now: $(numfmt --to=iec-i --suffix=B "$required")." >&2
    return 1
  fi
}

download_file() {
  local kind="$1"
  local url="$2"
  local out="$3"
  local expected_size="$4"
  local expected_sha256="$5"
  local part="${out}.part"
  local curl_log="${part}.curl.log"
  local name started initial_bytes now bytes elapsed transferred average_speed
  local sample_started sample_elapsed sample_bytes current_speed file_remaining file_eta
  local status_printed=0 last_status_elapsed=0 last_observed_bytes stall_notified=0
  local slow_samples=0 reconnect_count=0 reconnect_requested=0 reconnect_limit_reported=0 curl_pid
  local min_speed_mib="${MINIMAX_H3_MIN_DOWNLOAD_MIBPS:-5}"
  local reconnect_limit="${MINIMAX_H3_RECONNECT_LIMIT:-3}"
  local min_speed
  local -a curl_headers=()

  [[ "$min_speed_mib" =~ ^[0-9]+$ ]] || min_speed_mib=5
  [[ "$reconnect_limit" =~ ^[0-9]+$ ]] || reconnect_limit=3
  min_speed=$((min_speed_mib * 1024 * 1024))
  mkdir -p "$(dirname "$out")"
  name="$(basename "$out")"

  if validate_model_file "$out" "$expected_size" "$expected_sha256"; then
    printf 'Ready %-5s  %s (validated)\n' "$kind" "$name"
    return 0
  fi
  if validate_model_file "$part" "$expected_size" "$expected_sha256"; then
    mv "$part" "$out"
    printf 'Ready %-5s  %s (recovered complete partial)\n' "$kind" "$name"
    return 0
  fi

  if [ -n "${HF_TOKEN:-}" ]; then
    curl_headers=(-H "Authorization: Bearer ${HF_TOKEN}")
  fi
  started=$(date +%s)
  initial_bytes=$(file_size "$part")

  while true; do
    reconnect_requested=0
    slow_samples=0
    last_observed_bytes=$(file_size "$part")
    sample_started=$(date +%s)

    "$CURL_BIN" -L --fail --silent --show-error --retry 5 --retry-all-errors --retry-delay 5 \
      --connect-timeout 30 --speed-limit 1024 --speed-time 15 \
      "${curl_headers[@]}" \
      --continue-at - -o "$part" "$url" >"$curl_log" 2>&1 &
    curl_pid=$!

    while kill -0 "$curl_pid" 2>/dev/null; do
      sleep 10
      kill -0 "$curl_pid" 2>/dev/null || break
      now=$(date +%s)
      bytes=$(file_size "$part")
      sample_elapsed=$((now - sample_started))
      [ "$sample_elapsed" -lt 1 ] && sample_elapsed=1
      sample_bytes=$((bytes - last_observed_bytes))
      [ "$sample_bytes" -lt 0 ] && sample_bytes=0
      current_speed=$((sample_bytes / sample_elapsed))

      if [ "$bytes" -le "$last_observed_bytes" ]; then
        if [ "$stall_notified" -eq 0 ]; then
          echo "WARNING: $kind $name has not grown in the last 10 seconds."
          notify_ntfy \
            "RunPod download stalled" "urgent" "warning" \
            "$kind $name has not changed size for 10 seconds. Curl is still retrying."
          stall_notified=1
        fi
      else
        stall_notified=0
      fi

      elapsed=$((now - started))
      [ "$elapsed" -lt 1 ] && elapsed=1
      transferred=$((bytes - initial_bytes))
      [ "$transferred" -lt 0 ] && transferred=0
      average_speed=$((transferred / elapsed))

      if [ "$transferred" -gt 0 ] \
        && { [ "$status_printed" -eq 0 ] || [ $((elapsed - last_status_elapsed)) -ge 10 ]; }; then
        if [ "$expected_size" -gt "$bytes" ] && [ "$average_speed" -gt 0 ]; then
          file_remaining=$((expected_size - bytes))
          file_eta=$(((file_remaining + average_speed - 1) / average_speed))
          printf 'Downloading %-5s  %s | %s/%s | current %s/s | average %s/s | ETA %s\n' \
            "$kind" "$name" \
            "$(numfmt --to=iec-i --suffix=B "$bytes")" \
            "$(numfmt --to=iec-i --suffix=B "$expected_size")" \
            "$(numfmt --to=iec-i --suffix=B "$current_speed")" \
            "$(numfmt --to=iec-i --suffix=B "$average_speed")" \
            "$(format_eta "$file_eta")"
        fi
        status_printed=1
        last_status_elapsed=$elapsed
      fi

      if [ "$current_speed" -lt "$min_speed" ]; then
        slow_samples=$((slow_samples + 1))
      else
        slow_samples=0
      fi

      if [ "$slow_samples" -ge 3 ]; then
        if [ "$reconnect_count" -lt "$reconnect_limit" ]; then
          reconnect_count=$((reconnect_count + 1))
          echo "WARNING: $kind $name stayed below ${min_speed_mib}MiB/s for 30 seconds; reconnecting ($reconnect_count/$reconnect_limit)."
          reconnect_requested=1
          kill "$curl_pid" 2>/dev/null || true
        elif [ "$reconnect_limit_reported" -eq 0 ]; then
          echo "WARNING: $kind $name is still slow, but the reconnect limit was reached; continuing."
          reconnect_limit_reported=1
        fi
        slow_samples=0
      fi

      last_observed_bytes=$bytes
      sample_started=$now
      [ "$reconnect_requested" -eq 1 ] && break
    done

    if [ "$reconnect_requested" -eq 1 ]; then
      if wait "$curl_pid" 2>/dev/null; then
        break
      fi
      sleep 2
      continue
    fi

    if ! wait "$curl_pid"; then
      echo "Download failed after retries; showing the last curl errors:"
      tail -n 20 "$curl_log" || true
      return 1
    fi
    break
  done

  rm -f "$curl_log"
  if ! validate_model_file "$part" "$expected_size" "$expected_sha256"; then
    echo "ERROR: Downloaded file failed exact-size or safetensors validation: $name" >&2
    return 1
  fi
  mv "$part" "$out"

  now=$(date +%s)
  elapsed=$((now - started))
  bytes=$(file_size "$out")
  transferred=$((bytes - initial_bytes))
  [ "$elapsed" -lt 1 ] && elapsed=1
  [ "$transferred" -lt 0 ] && transferred=0
  average_speed=$((transferred / elapsed))
  printf 'Downloaded  %-5s  %s | %s | average %s/s\n' \
    "$kind" "$name" "$(numfmt --to=iec-i --suffix=B "$bytes")" \
    "$(numfmt --to=iec-i --suffix=B "$average_speed")"
  notify_ntfy \
    "RunPod download complete" "default" "white_check_mark" \
    "$kind $name finished downloading and passed validation."
}

download_group() {
  local label="$1"
  shift
  local -a indices=("$@")
  local index next=0 active=0 failures=0 pid
  local started finished elapsed initial_bytes=0 final_bytes=0 transferred average_speed
  local path part_size name

  [ "${#indices[@]}" -gt 0 ] || return 0
  echo "Starting $label downloads with up to $DOWNLOAD_JOBS concurrent transfers."
  started=$(date +%s)

  for index in "${indices[@]}"; do
    path="${DOWNLOAD_OUTPUTS[$index]}"
    if validate_model_file "$path" "${DOWNLOAD_SIZES[$index]}" "${DOWNLOAD_SHA256[$index]}"; then
      initial_bytes=$((initial_bytes + $(file_size "$path")))
    else
      part_size=$(file_size "${path}.part")
      initial_bytes=$((initial_bytes + part_size))
    fi
  done

  while [ "$next" -lt "${#indices[@]}" ] || [ "$active" -gt 0 ]; do
    while [ "$next" -lt "${#indices[@]}" ] && [ "$active" -lt "$DOWNLOAD_JOBS" ]; do
      index="${indices[$next]}"
      name="$(basename "${DOWNLOAD_OUTPUTS[$index]}")"
      (
        download_file "${DOWNLOAD_KINDS[$index]}" "${DOWNLOAD_URLS[$index]}" \
          "${DOWNLOAD_OUTPUTS[$index]}" "${DOWNLOAD_SIZES[$index]}" \
          "${DOWNLOAD_SHA256[$index]}"
      ) 2>&1 | sed -u "s/^/[$label][$name] /" &
      pid=$!
      echo "[$label] Started $name (worker PID $pid)."
      next=$((next + 1))
      active=$((active + 1))
    done

    if [ "$active" -gt 0 ]; then
      if wait -n; then
        :
      else
        failures=$((failures + 1))
      fi
      active=$((active - 1))
    fi
  done

  finished=$(date +%s)
  elapsed=$((finished - started))
  [ "$elapsed" -lt 1 ] && elapsed=1
  for index in "${indices[@]}"; do
    final_bytes=$((final_bytes + $(file_size "${DOWNLOAD_OUTPUTS[$index]}")))
  done
  transferred=$((final_bytes - initial_bytes))
  [ "$transferred" -lt 0 ] && transferred=0
  average_speed=$((transferred / elapsed))

  if [ "$failures" -gt 0 ]; then
    echo "ERROR: $label finished with $failures failed download worker(s)."
    return 1
  fi
  echo "Completed $label downloads in $(format_eta "$elapsed") at aggregate average $(numfmt --to=iec-i --suffix=B "$average_speed")/s."
}

validate_model_group() {
  local label="$1"
  shift
  local index actual_sha256 out
  for index in "$@"; do
    out="${DOWNLOAD_OUTPUTS[$index]}"
    if ! validate_model_file "$out" \
      "${DOWNLOAD_SIZES[$index]}" "${DOWNLOAD_SHA256[$index]}"; then
      echo "FATAL: Final validation failed for $out" >&2
      return 1
    fi
    if [ "${MINIMAX_H3_VERIFY_SHA256:-0}" = "1" ]; then
      echo "Hashing $(basename "$out")..."
      actual_sha256="$(sha256_file "$out")"
      if [ "$actual_sha256" != "${DOWNLOAD_SHA256[$index]}" ]; then
        echo "FATAL: SHA256 validation failed for $out" >&2
        quarantine_invalid_file "$out" "model with a SHA256 mismatch"
        return 1
      fi
    fi
  done
  echo "Validated $label."
}

validate_all_models() {
  local -a all_downloads=(0 1 2 3 4)
  validate_model_group "both H3 diffusion models and all three shared model assets" \
    "${all_downloads[@]}"
}

patch_start_credentials() {
  echo "Patching original /start.sh for custom FileBrowser credentials..."
  python3 - <<'PY_PATCH_START'
from pathlib import Path

p = Path('/start.sh')
s = p.read_text()
old = 'filebrowser users add admin adminadmin12 --perm.admin'
new = 'filebrowser users add "${FILEBROWSER_USERNAME:-admin}" "${FILEBROWSER_PASSWORD:-adminadmin12}" --perm.admin'
if old in s:
    s = s.replace(old, new)
    print('Patched FileBrowser credentials line in /start.sh.')
elif new in s:
    print('/start.sh already has custom FileBrowser credential support.')
else:
    print('WARNING: Could not find FileBrowser credentials line in /start.sh; leaving it unchanged.')
p.write_text(s)
PY_PATCH_START
}

start_ready_notification() {
  local readiness_detail="$1"
  [ -n "${NTFY_TOPIC:-}" ] || return 0
  echo "Watching for ComfyUI readiness before sending ntfy notification..."
  (
    trap - EXIT
    trap '' HUP
    for _ in $(seq 1 180); do
      if "$CURL_BIN" -fsS --max-time 2 http://127.0.0.1:8188/ >/dev/null 2>&1; then
        notify_ntfy \
          "MiniMax H3 ComfyUI is ready" "high" "tada" \
          "$readiness_detail"
        exit 0
      fi
      sleep 5
    done
    echo "WARNING: ComfyUI did not become ready within 15 minutes; no ready notification sent."
  ) &
}

start_deferred_model_download() {
  local index="$1"
  local deferred_name="$2"
  local output="${DOWNLOAD_OUTPUTS[$index]}"
  local filename="$(basename "$output")"

  write_status "background-downloading" \
    "ComfyUI is starting with $FIRST_MODEL_NAME ready; downloading $deferred_name ($filename) in the background."
  (
    # The foreground setup EXIT trap must not mark a later background failure as
    # a failed cold start. Ignore HUP so replacing the wrapper with /start.sh does
    # not interrupt the resumable transfer.
    trap - EXIT
    trap '' HUP
    if download_group "MINIMAX-H3-BACKGROUND" "$index" \
      && validate_model_group "$deferred_name diffusion model" "$index"; then
      write_status "ready" \
        "Both MiniMax H3 diffusion models and all shared assets passed validation."
      notify_ntfy \
        "MiniMax H3 background model ready" "high" "white_check_mark" \
        "$deferred_name finished downloading and is ready in ComfyUI. Refresh the browser model list if it is already open."
      echo "=== Background $deferred_name download is ready: $filename ==="
    else
      write_status "background-failed" \
        "$deferred_name background download failed; its partial file was preserved for the next restart."
      notify_ntfy \
        "MiniMax H3 background download failed" "urgent" "warning" \
        "$deferred_name failed to download. The ready $FIRST_MODEL_NAME workflow remains usable and the partial download will resume on restart."
      echo "ERROR: Background $deferred_name download failed; partial data was preserved." >&2
      exit 1
    fi
  ) &
  echo "Background $deferred_name downloader started as PID $!."
}

main() {
  local -a foreground_downloads
  local setup_complete=0
  local readiness_detail

  echo "=== MiniMax H3 cold-start setup starting ==="
  write_status "preflight" "Validating CUDA, ComfyUI, bundle, remote assets, and disk space."
  trap 'rc=$?; if [ "$setup_complete" -ne 1 ]; then write_status "failed" "Setup exited with code $rc; completed and partial files were preserved."; fi; exit "$rc"' EXIT

  preflight_cuda
  ensure_comfyui
  ensure_pixaroma_h3
  python -m py_compile \
    "$COMFY/comfy_extras/nodes_minimax_h3.py" \
    "$COMFY/comfy/text_encoders/minimax.py" \
    "$COMFY/custom_nodes/ComfyUI-Pixaroma/nodes/node_h3_audio_sync.py"

  resolve_script_base_url
  install_bundle
  validate_installed_workflows

  init_model_manifest
  select_model_priority
  foreground_downloads=("$FIRST_MODEL_INDEX" 2 3 4)
  prepare_local_models
  preflight_remote_models
  preflight_disk_space

  write_status "foreground-downloading" \
    "Downloading $FIRST_MODEL_NAME and all three shared assets before ComfyUI starts."
  download_group "MINIMAX-H3-FOREGROUND" "${foreground_downloads[@]}"
  validate_model_group "$FIRST_MODEL_NAME and all three shared model assets" \
    "${foreground_downloads[@]}"

  patch_start_credentials
  readiness_detail="ComfyUI started with $FIRST_MODEL_NAME ready. $DEFERRED_MODEL_NAME is downloading in the background; refresh the model list after its completion notification."
  start_ready_notification "$readiness_detail"
  write_status "foreground-ready" \
    "$FIRST_MODEL_NAME and all shared assets are ready; ComfyUI is starting and $DEFERRED_MODEL_NAME will download in the background."
  setup_complete=1
  trap - EXIT

  notify_ntfy \
    "MiniMax H3 foreground models ready" "high" "white_check_mark" \
    "$FIRST_MODEL_NAME, the Qwen text encoder, and audio/video VAEs are ready. ComfyUI is starting while $DEFERRED_MODEL_NAME downloads in the background."
  start_deferred_model_download "$DEFERRED_MODEL_INDEX" "$DEFERRED_MODEL_NAME"
  echo "=== MiniMax H3 $FIRST_MODEL_NAME workflows are ready ==="
  echo "Returning to wrapper. SageAttention bootstrap will run next."
}

if [ "${MINIMAX_H3_SETUP_LIB_ONLY:-0}" != "1" ]; then
  main "$@"
fi
