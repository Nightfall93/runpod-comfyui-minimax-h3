#!/usr/bin/env bash
set -euo pipefail

COMFYUI_COMMIT="${COMFYUI_COMMIT:-dec5d9450a5290bcf63430409ea41018e67f41c3}"
COMFY_BAKED="/opt/comfyui-baked"
STAGING="/tmp/ComfyUI-h3.bake"
CONSTRAINTS="/opt/comfyui-runtime-constraints.txt"
COMFYUI_SAGE3_PATCH="${COMFYUI_SAGE3_PATCH:-}"

echo "Baking ComfyUI core with MiniMax H3 support: $COMFYUI_COMMIT"
rm -rf "$STAGING"
git init "$STAGING"
git -C "$STAGING" remote add origin https://github.com/Comfy-Org/ComfyUI.git
git -C "$STAGING" fetch --depth 1 origin "$COMFYUI_COMMIT"
git -C "$STAGING" checkout --detach FETCH_HEAD

test -f "$STAGING/comfy_extras/nodes_minimax_h3.py"
test -f "$STAGING/comfy/text_encoders/minimax.py"
test -f "$STAGING/comfy/ldm/minimax/model.py"

if [ -n "$COMFYUI_SAGE3_PATCH" ]; then
  test -f "$COMFYUI_SAGE3_PATCH"
  git -C "$STAGING" apply --check "$COMFYUI_SAGE3_PATCH"
  git -C "$STAGING" apply "$COMFYUI_SAGE3_PATCH"
  grep -Fq 'COMFY_SAGE_ATTENTION3' "$STAGING/comfy/ldm/modules/attention.py"
fi

if [ -f "$CONSTRAINTS" ]; then
  python3 -m pip install --no-cache-dir -c "$CONSTRAINTS" \
    -r "$STAGING/requirements.txt"
else
  python3 -m pip install --no-cache-dir -r "$STAGING/requirements.txt"
fi

rm -rf "$STAGING/.git" "$COMFY_BAKED"
printf '%s\n' "$COMFYUI_COMMIT" > "$STAGING/.minimax-h3-core-commit"
mv "$STAGING" "$COMFY_BAKED"

echo "Pinned MiniMax H3 ComfyUI core baked successfully."
