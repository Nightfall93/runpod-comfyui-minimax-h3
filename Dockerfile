FROM nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c AS cuda-devel

FROM runpod/comfyui:cuda13.0@sha256:949b0688db0692b97b9aab9efd1c8f5afe94cfaa32c32008f31bcafcff63baf1 AS sage-builder

ARG SAGEATTENTION_COMMIT=d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5
ARG CUTLASS_COMMIT=dcf215af68a2d08d305076c152a06f201728cd53
ARG SAGE_CUDA_ARCH_LIST=8.9
ARG SAGE_ATTENTION_MAJOR=2
ARG SAGE_ATTENTION_VERSION=2.2.0
ARG MINIMAX_H3_GPU_FAMILY=ada

ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=${SAGE_CUDA_ARCH_LIST} \
    SAGE_CUDA_ARCH_LIST=${SAGE_CUDA_ARCH_LIST} \
    MAX_JOBS=1

COPY --from=cuda-devel /usr/local/cuda/include/ /usr/local/cuda/include/
COPY --from=cuda-devel /usr/local/cuda/targets/x86_64-linux/lib/stubs/ \
  /usr/local/cuda/lib64/stubs/

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && command -v nvcc \
    && nvcc --version \
    && python3 -c "import torch; assert torch.version.cuda == '13.0', torch.version.cuda"

RUN git init /tmp/SageAttention \
    && git -C /tmp/SageAttention remote add origin https://github.com/thu-ml/SageAttention.git \
    && git -C /tmp/SageAttention fetch --depth 1 origin "$SAGEATTENTION_COMMIT" \
    && git -C /tmp/SageAttention checkout --detach FETCH_HEAD \
    && case "$SAGE_ATTENTION_MAJOR" in \
      2) \
        test "$MINIMAX_H3_GPU_FAMILY" = "ampere" -o "$MINIMAX_H3_GPU_FAMILY" = "ada"; \
        test "$SAGE_ATTENTION_VERSION" = "2.2.0"; \
        python3 -m pip wheel --no-deps --no-build-isolation \
          --wheel-dir /tmp/sage-wheels /tmp/SageAttention \
        ;; \
      3) \
        test "$MINIMAX_H3_GPU_FAMILY" = "blackwell"; \
        test "$SAGE_CUDA_ARCH_LIST" = "12.0"; \
        test "$SAGE_ATTENTION_VERSION" = "1.0.0"; \
        git init /tmp/SageAttention/sageattention3_blackwell/csrc/cutlass; \
        git -C /tmp/SageAttention/sageattention3_blackwell/csrc/cutlass \
          remote add origin https://github.com/NVIDIA/cutlass.git; \
        git -C /tmp/SageAttention/sageattention3_blackwell/csrc/cutlass \
          fetch --depth 1 origin "$CUTLASS_COMMIT"; \
        git -C /tmp/SageAttention/sageattention3_blackwell/csrc/cutlass \
          checkout --detach FETCH_HEAD; \
        grep -Fq 'cc_major, cc_minor = torch.cuda.get_device_capability()' \
          /tmp/SageAttention/sageattention3_blackwell/setup.py; \
        sed -i \
          's/cc_major, cc_minor = torch.cuda.get_device_capability()/cc_major, cc_minor = map(int, os.environ["SAGE_CUDA_ARCH_LIST"].split("."))/' \
          /tmp/SageAttention/sageattention3_blackwell/setup.py; \
        LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
          python3 -m pip wheel --no-deps --no-build-isolation \
          --wheel-dir /tmp/sage-wheels \
          /tmp/SageAttention/sageattention3_blackwell \
        ;; \
      *) \
        echo "Unsupported SageAttention major: $SAGE_ATTENTION_MAJOR" >&2; \
        exit 1 \
        ;; \
    esac \
    && test "$(find /tmp/sage-wheels -maxdepth 1 -type f -name '*.whl' | wc -l)" -eq 1

FROM runpod/comfyui:cuda13.0@sha256:949b0688db0692b97b9aab9efd1c8f5afe94cfaa32c32008f31bcafcff63baf1

ARG SAGE_CUDA_ARCH_LIST=8.9
ARG SAGE_ATTENTION_MAJOR=2
ARG SAGE_ATTENTION_VERSION=2.2.0
ARG COMFY_SAGE_ATTENTION3=0
ARG MINIMAX_H3_GPU_FAMILY=ada
ARG COMFYUI_COMMIT=12d5279438bfefc058a269eae805ceab6047777f
ARG PIXAROMA_COMMIT=433bbedc7f43d717fcb9e8e9aa9cbd26b0439226

ENV COMFY_VENV=/workspace/runpod-slim/ComfyUI/.venv-cu130 \
    SAGE_SUPPORTED_CC=${SAGE_CUDA_ARCH_LIST} \
    SAGE_ATTENTION_MAJOR=${SAGE_ATTENTION_MAJOR} \
    SAGE_ATTENTION_VERSION=${SAGE_ATTENTION_VERSION} \
    COMFY_SAGE_ATTENTION3=${COMFY_SAGE_ATTENTION3} \
    REQUIRE_SAGE_ATTENTION=1 \
    MINIMAX_H3_GPU_FAMILY=${MINIMAX_H3_GPU_FAMILY} \
    MINIMAX_H3_REQUIRED_DRIVER_MAJOR=580 \
    COMFYUI_H3_COMMIT=${COMFYUI_COMMIT} \
    PIXAROMA_H3_COMMIT=${PIXAROMA_COMMIT}

COPY --from=sage-builder /tmp/sage-wheels /tmp/sage-wheels
RUN python3 -m pip uninstall -y sageattention sageattn3 >/dev/null 2>&1 || true
RUN python3 -m pip install --no-deps /tmp/sage-wheels/*.whl \
    && SAGE_ATTENTION_MAJOR="$SAGE_ATTENTION_MAJOR" \
      SAGE_ATTENTION_VERSION="$SAGE_ATTENTION_VERSION" \
      python3 -c 'import os; from importlib.metadata import version; package = "sageattention" if os.environ["SAGE_ATTENTION_MAJOR"] == "2" else "sageattn3"; assert version(package) == os.environ["SAGE_ATTENTION_VERSION"], (package, version(package))' \
    && rm -rf /tmp/sage-wheels

COPY bake_comfyui_core.sh /tmp/bake_comfyui_core.sh
COPY bake_custom_nodes.sh /tmp/bake_custom_nodes.sh
COPY seed-hunter-node-lock.tsv /opt/minimax-h3/seed-hunter-node-lock.tsv
COPY patches/comfyui-sage3-global.patch /tmp/comfyui-sage3-global.patch
RUN COMFYUI_COMMIT="$COMFYUI_H3_COMMIT" \
      COMFYUI_SAGE3_PATCH=/tmp/comfyui-sage3-global.patch \
      bash /tmp/bake_comfyui_core.sh \
    && PIXAROMA_COMMIT="$PIXAROMA_H3_COMMIT" \
      SEED_HUNTER_NODE_LOCK=/opt/minimax-h3/seed-hunter-node-lock.tsv \
      bash /tmp/bake_custom_nodes.sh \
    && cd /opt/comfyui-baked \
    && python3 main.py --cpu --quick-test-for-ci \
    && rm -f /tmp/bake_comfyui_core.sh /tmp/bake_custom_nodes.sh \
      /tmp/comfyui-sage3-global.patch

COPY asset-manifest.tsv /opt/minimax-h3-bundle/asset-manifest.tsv
COPY workflows/ /opt/minimax-h3-bundle/workflows/
COPY media/ /opt/minimax-h3-bundle/media/
COPY tests/validate_bundle.py /tmp/validate_bundle.py
RUN python3 /tmp/validate_bundle.py --root /opt/minimax-h3-bundle \
    && rm -f /tmp/validate_bundle.py

COPY minimax_h3_download_setup.sh /opt/minimax-h3/minimax_h3_download_setup.sh
COPY driver_preflight.sh /driver_preflight.sh
COPY entrypoint.sh /entrypoint.sh
COPY sage_bootstrap.sh /sage_bootstrap.sh

RUN bash -n /opt/minimax-h3/minimax_h3_download_setup.sh \
    /driver_preflight.sh /entrypoint.sh /sage_bootstrap.sh \
    && chmod +x /opt/minimax-h3/minimax_h3_download_setup.sh \
      /driver_preflight.sh /entrypoint.sh /sage_bootstrap.sh

ENTRYPOINT ["/entrypoint.sh"]
