#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

export COMFY="$tmp/ComfyUI"
export MINIMAX_H3_BUNDLE_ROOT="$repo_root"
export MINIMAX_H3_SETUP_LIB_ONLY=1
source "$repo_root/minimax_h3_download_setup.sh"

mkdir -p "$COMFY"
SCRIPT_BASE_URL=""
install_bundle
validate_installed_workflows

workflow_count="$(find "$COMFY/user/default/workflows/MiniMax H3" -type f -name '*.json' | wc -l | tr -d '[:space:]')"
[ "$workflow_count" = "10" ]
[ -f "$COMFY/input/.h1 Collage.png" ]
[ -f "$COMFY/input/Going to be completely honest here_ it was not pleasant poor things___Yall know the feeling wh.mp4" ]

custom="$COMFY/user/default/workflows/MiniMax H3/4. Generate Image H3 (fl2va)/Minimax H3 - Text to image.json"
printf '{"customized":true}\n' > "$custom"
second_install_output="$(install_bundle)"
grep -q "Preserved customized workflow" <<< "$second_install_output"
grep -q '"customized":true' "$custom"

echo "Bundle installation and customization-preservation tests passed."
