#!/usr/bin/env bash
set -euo pipefail

COMFY_BAKED="/opt/comfyui-baked"
CUSTOM_NODES="$COMFY_BAKED/custom_nodes"
PIXAROMA_COMMIT="${PIXAROMA_COMMIT:-433bbedc7f43d717fcb9e8e9aa9cbd26b0439226}"
TARGET="$CUSTOM_NODES/ComfyUI-Pixaroma"
STAGING="/tmp/ComfyUI-Pixaroma.h3-bake"

mkdir -p "$CUSTOM_NODES"
echo "Baking ComfyUI-Pixaroma with H3 workflow nodes: $PIXAROMA_COMMIT"
rm -rf "$STAGING"
git init "$STAGING"
git -C "$STAGING" remote add origin https://gitlab.com/pixaroma/ComfyUI-Pixaroma.git
git -C "$STAGING" fetch --depth 1 origin "$PIXAROMA_COMMIT"
git -C "$STAGING" checkout --detach FETCH_HEAD

test -f "$STAGING/nodes/node_h3_audio_sync.py"
grep -q 'PixaromaH3AudioSync' "$STAGING/nodes/node_h3_audio_sync.py"

rm -rf "$STAGING/.git" "$TARGET"
printf '%s\n' "$PIXAROMA_COMMIT" > "$STAGING/.minimax-h3-managed-commit"
mv "$STAGING" "$TARGET"

echo "Pinned MiniMax H3 custom-node set baked successfully."
