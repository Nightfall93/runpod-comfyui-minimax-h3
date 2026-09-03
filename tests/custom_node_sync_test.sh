#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

export MINIMAX_H3_SETUP_LIB_ONLY=1
export MINIMAX_H3_COMFY_BAKED="$tmp/baked"
export MINIMAX_H3_NODE_LOCK="$tmp/nodes.tsv"
export COMFY="$tmp/ComfyUI"
source "$repo_root/minimax_h3_download_setup.sh"

printf '%s\t%s\t%s\n' \
  "ExampleNode" "https://example.invalid/node.git" "1111111111111111111111111111111111111111" \
  "ComfyUI-VFI" "https://example.invalid/vfi.git" "2222222222222222222222222222222222222222" \
  > "$NODE_LOCK"

for folder in ExampleNode ComfyUI-VFI; do
  mkdir -p "$COMFY_BAKED/custom_nodes/$folder"
done
printf '%s\n' "1111111111111111111111111111111111111111" \
  > "$COMFY_BAKED/custom_nodes/ExampleNode/.minimax-h3-managed-commit"
printf '%s\n' "2222222222222222222222222222222222222222" \
  > "$COMFY_BAKED/custom_nodes/ComfyUI-VFI/.minimax-h3-managed-commit"
mkdir -p "$COMFY_BAKED/custom_nodes/ComfyUI-VFI/rife/train_log"
truncate -s 24636301 "$COMFY_BAKED/custom_nodes/ComfyUI-VFI/rife/train_log/flownet.pkl"

mkdir -p "$COMFY/custom_nodes/ExampleNode"
printf 'user copy\n' > "$COMFY/custom_nodes/ExampleNode/custom.txt"
GPU_FAMILY=ada
ensure_seed_hunter_nodes >/dev/null
grep -q '^1111111111111111111111111111111111111111$' \
  "$COMFY/custom_nodes/ExampleNode/.minimax-h3-managed-commit"
[ -f "$COMFY/custom_nodes/ComfyUI-VFI/rife/train_log/flownet.pkl" ]
find "$COMFY/custom_nodes" -maxdepth 1 -type d \
  -name 'ExampleNode.pre-minimax-h3-*' | grep -q .

COMFY="$tmp/ComfyUI-Ampere"
mkdir -p "$COMFY/custom_nodes"
GPU_FAMILY=ampere
ensure_seed_hunter_nodes >/dev/null
[ ! -e "$COMFY/custom_nodes/ExampleNode" ]

echo "Custom-node synchronization tests passed."
