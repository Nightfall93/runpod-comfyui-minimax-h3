# Relationship to the WAN template

This repository intentionally mirrors the proven WAN startup behavior without
changing the published WAN images or setup repository.

Shared design:

- immutable RunPod CUDA base digests;
- architecture-specific attention builds (SageAttention 2 for Ada and
  SageAttention 3 for Blackwell), with a required runtime CUDA-kernel smoke test;
- driver, CUDA tensor, and compute-capability gates before model bandwidth;
- pinned custom nodes and a runtime fallback installer;
- concurrent resumable `.part` downloads with retry, stall reporting, and
  controlled slow-transfer reconnects;
- optional ntfy notifications and FileBrowser credential patching;
- atomic workflow installation that preserves customized copies;
- syntax and mocked preflight tests in CI.

H3-specific additions:

- a pinned ComfyUI core newer than the WAN image because native MiniMax H3 nodes
  first appear in the August 2026 ComfyUI release;
- exact expected sizes and safetensors-header validation for every model;
- disk-capacity preflight and optional full SHA256 verification;
- baked workflow/media bundle with a checksum-verified remote fallback;
- one repository and build matrix for the two supported GPU families instead of
  duplicated wrapper repositories. Ampere images are intentionally not built.

Do not point an existing WAN template at this image: its baked core and workflow
bundle are versioned independently so WAN deployments remain unchanged.
