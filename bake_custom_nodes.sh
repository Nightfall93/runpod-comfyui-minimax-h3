#!/usr/bin/env bash
set -euo pipefail

COMFY_BAKED="/opt/comfyui-baked"
CUSTOM_NODES="$COMFY_BAKED/custom_nodes"
CONSTRAINTS="/opt/comfyui-runtime-constraints.txt"
PIXAROMA_COMMIT="${PIXAROMA_COMMIT:-433bbedc7f43d717fcb9e8e9aa9cbd26b0439226}"
GPU_FAMILY="${MINIMAX_H3_GPU_FAMILY:-ada}"
NODE_LOCK="${SEED_HUNTER_NODE_LOCK:-/opt/minimax-h3/seed-hunter-node-lock.tsv}"
REQS="/tmp/minimax-h3-custom-node-requirements.txt"
RIFE_REVISION="01fdc7e97404120c243c3ea7b427046e5dc7643e"
RIFE_ZIP_SHA256="1fa9b9cda3d9b8c3e301359e2595960902f97bf926c08598b0e9957a3f3f760e"
RIFE_MODEL_SHA256="45c7f74156704769dc9f85cfcaf8552e1e926f9399dcfa3a553dee88fac6f53f"

clone_pinned_node() {
  local folder="$1"
  local repository="$2"
  local commit="$3"
  local target="$CUSTOM_NODES/$folder"
  local staging="/tmp/${folder}.h3-bake"

  echo "Baking $folder at $commit"
  rm -rf "$staging"
  git init "$staging"
  git -C "$staging" remote add origin "$repository"
  git -C "$staging" fetch --depth 1 origin "$commit"
  git -C "$staging" checkout --detach FETCH_HEAD
  test "$(git -C "$staging" rev-parse HEAD)" = "$commit"

  if [ -s "$staging/requirements.txt" ]; then
    # SAM2 is optional for the one Impact switch used here. VHS's GUI OpenCV
    # build conflicts with the headless build required by the other packs.
    grep -Ev '^(git\+https://github.com/facebookresearch/sam2|sam2[[:space:]]*@|opencv-python([[:space:]]|$))' \
      "$staging/requirements.txt" >> "$REQS" || true
  fi

  rm -rf "$staging/.git" "$target"
  printf '%s\n' "$commit" > "$staging/.minimax-h3-managed-commit"
  mv "$staging" "$target"
}

install_rife_model() {
  local target="$CUSTOM_NODES/ComfyUI-VFI/rife/train_log/flownet.pkl"
  local archive="/tmp/RIFEv4.26_0921.zip"
  local extract_dir="/tmp/RIFEv4.26_0921"
  local extracted

  echo "Baking the pinned RIFE interpolation model."
  curl -fsSL --retry 5 --retry-all-errors --retry-delay 2 \
    "https://huggingface.co/hzwer/RIFE/resolve/$RIFE_REVISION/RIFEv4.26_0921.zip" \
    -o "$archive"
  echo "$RIFE_ZIP_SHA256  $archive" | sha256sum -c -
  rm -rf "$extract_dir"
  mkdir -p "$extract_dir"
  python3 -m zipfile -e "$archive" "$extract_dir"
  extracted="$(find "$extract_dir" -type f -name flownet.pkl -print -quit)"
  test -n "$extracted"
  test "$(find "$extract_dir" -type f -name flownet.pkl | wc -l)" -eq 1
  install -D -m 0644 "$extracted" "$target"
  test "$(stat -c '%s' "$target")" -eq 24636301
  echo "$RIFE_MODEL_SHA256  $target" | sha256sum -c -
  rm -rf "$archive" "$extract_dir"
}

mkdir -p "$CUSTOM_NODES"
: > "$REQS"

clone_pinned_node \
  "ComfyUI-Pixaroma" \
  "https://gitlab.com/pixaroma/ComfyUI-Pixaroma.git" \
  "$PIXAROMA_COMMIT"
test -f "$CUSTOM_NODES/ComfyUI-Pixaroma/nodes/node_h3_audio_sync.py"
grep -q 'PixaromaH3AudioSync' \
  "$CUSTOM_NODES/ComfyUI-Pixaroma/nodes/node_h3_audio_sync.py"

case "$GPU_FAMILY" in
  ampere)
    echo "Skipping Seed Hunter custom nodes on Ampere."
    ;;
  ada|blackwell)
    test -s "$NODE_LOCK"
    while IFS=$'\t' read -r folder repository commit; do
      [ -n "$folder" ] || continue
      [[ "$folder" = \#* ]] && continue
      [ -n "$repository" ] && [ -n "$commit" ]
      clone_pinned_node "$folder" "$repository" "$commit"
    done < "$NODE_LOCK"

    if [ -s "$REQS" ]; then
      sort -u -o "$REQS" "$REQS"
      if [ -f "$CONSTRAINTS" ]; then
        python3 -m pip install --no-cache-dir -c "$CONSTRAINTS" -r "$REQS"
      else
        python3 -m pip install --no-cache-dir -r "$REQS"
      fi
    fi
    install_rife_model
    ;;
  *)
    echo "Unsupported MiniMax H3 GPU family: $GPU_FAMILY" >&2
    exit 1
    ;;
esac

rm -f "$REQS"
echo "Pinned MiniMax H3 custom-node set baked successfully for $GPU_FAMILY."
