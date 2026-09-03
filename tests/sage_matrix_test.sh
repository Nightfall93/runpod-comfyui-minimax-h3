#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

cat > "$tmp/python" <<'MOCK_PYTHON'
#!/usr/bin/env bash
set -euo pipefail
cat >/dev/null
if [ "$#" -eq 1 ]; then
  printf '%s\n' "${MOCK_CC:?}"
  exit 0
fi
if [ "${MOCK_SAGE_FAIL:-0}" = "1" ]; then
  echo "mock Sage import failure" >&2
  exit 1
fi
printf 'SageAttention %s package mock %s; CUDA smoke=passed\n' "$2" "$3"
MOCK_PYTHON
chmod +x "$tmp/python"

run_bootstrap() {
  local family="$1"
  local cc="$2"
  local major="$3"
  local version="$4"
  local sage3="$5"
  local args_file="$tmp/$family.args"

  MOCK_CC="$cc" \
  MINIMAX_H3_GPU_FAMILY="$family" \
  SAGE_SUPPORTED_CC="$cc" \
  SAGE_ATTENTION_MAJOR="$major" \
  SAGE_ATTENTION_VERSION="$version" \
  COMFY_SAGE_ATTENTION3="$sage3" \
  REQUIRE_SAGE_ATTENTION=1 \
  SAGE_PYTHON_BIN="$tmp/python" \
  COMFY_ARGS_FILE="$args_file" \
  bash "$repo_root/sage_bootstrap.sh"
}

for family in ampere ada; do
  if [ "$family" = "ampere" ]; then
    cc="8.6"
  else
    cc="8.9"
  fi
  output="$(run_bootstrap "$family" "$cc" 2 2.2.0 0)"
  grep -q "SageAttention 2 is enabled globally" <<< "$output"
  [ "$(grep -o -- '--use-sage-attention' "$tmp/$family.args" | wc -l | tr -d '[:space:]')" = "1" ]
done

blackwell_output="$(run_bootstrap blackwell 12.0 3 1.0.0 1)"
grep -q "SageAttention 3 is enabled globally" <<< "$blackwell_output"
if grep -q -- '--use-sage-attention' "$tmp/blackwell.args"; then
  echo "Blackwell must use ComfyUI's sage3 backend, not the Sage 2 CLI flag." >&2
  exit 1
fi

if run_bootstrap blackwell 12.0 2 2.2.0 0 >"$tmp/wrong-major.log" 2>&1; then
  echo "Expected Blackwell with SageAttention 2 to fail." >&2
  exit 1
fi
grep -q "blackwell requires SageAttention 3" "$tmp/wrong-major.log"

if MOCK_SAGE_FAIL=1 run_bootstrap ada 8.9 2 2.2.0 0 >"$tmp/import-failure.log" 2>&1; then
  echo "Expected a required Sage package import failure to stop startup." >&2
  exit 1
fi
grep -q "SageAttention 2 validation failed" "$tmp/import-failure.log"

MOCK_CC=8.9 \
MINIMAX_H3_GPU_FAMILY=blackwell \
SAGE_SUPPORTED_CC=12.0 \
SAGE_ATTENTION_MAJOR=3 \
SAGE_ATTENTION_VERSION=1.0.0 \
COMFY_SAGE_ATTENTION3=1 \
REQUIRE_SAGE_ATTENTION=1 \
SAGE_PYTHON_BIN="$tmp/python" \
COMFY_ARGS_FILE="$tmp/wrong-cc.args" \
bash "$repo_root/sage_bootstrap.sh" >"$tmp/wrong-cc.log" 2>&1 && {
  echo "Expected a Blackwell image on an Ada GPU to fail." >&2
  exit 1
}
grep -q "supports 12.0, not 8.9" "$tmp/wrong-cc.log"

grep -Fq 'SAGE_ATTENTION_MAJOR=${{ matrix.sage_major }}' \
  "$repo_root/.github/workflows/docker-publish.yml"
grep -Fq 'COMFY_SAGE_ATTENTION3=${{ matrix.comfy_sage3 }}' \
  "$repo_root/.github/workflows/docker-publish.yml"
grep -Fq 'sageattention3_blackwell' "$repo_root/Dockerfile"
grep -Fq 'COPY --from=cuda-devel /usr/local/cuda/targets/x86_64-linux/lib/stubs/' \
  "$repo_root/Dockerfile"
grep -Fq 'LIBRARY_PATH=/usr/local/cuda/lib64/stubs' "$repo_root/Dockerfile"
grep -Fq 'COMFY_SAGE_ATTENTION3' "$repo_root/patches/comfyui-sage3-global.patch"
grep -Fq 'ARG COMFYUI_COMMIT=12d5279438bfefc058a269eae805ceab6047777f' \
  "$repo_root/Dockerfile"
grep -Fq 'COPY seed-hunter-node-lock.tsv /opt/minimax-h3/seed-hunter-node-lock.tsv' \
  "$repo_root/Dockerfile"
grep -Fq 'COPY seed-hunter-node-lock.tsv /opt/minimax-h3-bundle/seed-hunter-node-lock.tsv' \
  "$repo_root/Dockerfile"
grep -Fq 'python3 main.py --cpu --quick-test-for-ci' "$repo_root/Dockerfile"
grep -Fq 'Skipping Seed Hunter custom nodes on Ampere.' "$repo_root/bake_custom_nodes.sh"
if grep -Fq -- '- family: ampere' "$repo_root/.github/workflows/docker-publish.yml"; then
  echo "Ampere must not be present in the publish matrix." >&2
  exit 1
fi
grep -Fq -- '- family: ada' "$repo_root/.github/workflows/docker-publish.yml"
grep -Fq -- '- family: blackwell' "$repo_root/.github/workflows/docker-publish.yml"
grep -Fq 'cancel-in-progress: true' "$repo_root/.github/workflows/docker-publish.yml"

echo "SageAttention 2 Ampere/Ada and SageAttention 3 Blackwell matrix tests passed."
