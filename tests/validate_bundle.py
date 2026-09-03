#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath


MODEL_FILES = {
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "minimax_h3_audio_vae_fp32.safetensors",
    "minimax_h3_video_vae_fp16.safetensors",
    "minimax_h3_video_vae_int8_convrot.safetensors",
    "minimax_h3_latent_upscaler_3d_bf16.safetensors",
    "taeh3.safetensors",
}
DIFFUSION_MODEL_FILES = {
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
}
REQUIRED_NODE_TYPES = {
    "MiniMaxH3ImageToVideo",
    "MiniMaxH3ReferenceToVideo",
    "PixaromaH3AudioSync",
}
MEDIA_NODE_TYPES = {
    "PixaromaLoadImageMini",
    "PixaromaLoadVideo",
}
SEED_HUNTER_NODE_FOLDERS = {
    "cg-use-everywhere",
    "ComfyUI-Impact-Pack",
    "ComfyUI-VFI",
    "WhatDreamsCost-ComfyUI",
    "comfyui-obvpm",
    "Comfyui_Minimax_h3_latent_Upscaler",
    "ComfyUI-SolAttn_triton",
    "comfyui-kjnodes",
    "rgthree-comfy",
    "ComfyUI-Easy-Use",
    "comfyui-videohelpersuite",
}
SEED_HUNTER_REQUIRED_NODE_TYPES = {
    "Anything Everywhere",
    "ImpactSwitch",
    "LoadAudioUI",
    "LoadVideoUI",
    "LoadImageCrop",
    "RIFEInterpolation",
    "MinimaxH3LatentUpscaler3D",
    "SolAttnPatch",
    "ImageResizeKJv2",
    "ModelPreviewOverrideKJ",
    "Any Switch (rgthree)",
    "Fast Groups Bypasser (rgthree)",
    "Power Lora Loader (rgthree)",
    "easy seed",
    "VHS_VideoCombine",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)


def workflow_nodes(data):
    yield from data.get("nodes", [])
    for subgraph in data.get("definitions", {}).get("subgraphs", []):
        yield from subgraph.get("nodes", [])


def validate_ref2va_seed_hunter(data, source: Path) -> None:
    nodes = data["nodes"]
    counts = Counter(node.get("type") for node in nodes)
    expected = {
        "UNETLoader": 1,
        "CLIPLoader": 1,
        "VAELoader": 2,
        "MiniMaxH3ReferenceToVideo": 1,
        "KSampler": 5,
        "ComfyAndNode": 1,
        "ComfySwitchNode": 5,
        "VAEDecode": 5,
        "VAEDecodeAudio": 5,
        "PixaromaSaveMp4": 5,
    }
    wrong = {
        node_type: (counts[node_type], wanted)
        for node_type, wanted in expected.items()
        if counts[node_type] != wanted
    }
    if wrong:
        raise SystemExit(f"seed hunter node-count mismatch in {source.name}: {wrong}")

    samplers = sorted(
        (node for node in nodes if node.get("type") == "KSampler"),
        key=lambda node: node["id"],
    )
    sampler_values = [node.get("widgets_values", []) for node in samplers]
    if any(len(values) < 7 or values[1] != "randomize" for values in sampler_values):
        raise SystemExit(f"seed hunter samplers must independently randomize: {source.name}")
    if len({values[0] for values in sampler_values}) != 5:
        raise SystemExit(f"seed hunter initial seeds are not unique: {source.name}")
    if len({tuple(values[2:]) for values in sampler_values}) != 1:
        raise SystemExit(f"seed hunter sampler parameters differ: {source.name}")

    links = data.get("links", [])
    incoming = {(link[3], link[4]): (link[1], link[2]) for link in links}
    by_id = {node["id"]: node for node in nodes}
    sampler_ids = {node["id"] for node in samplers}
    barrier = next(node for node in nodes if node.get("type") == "ComfyAndNode")
    barrier_names = [item.get("name") for item in barrier.get("inputs", [])]
    expected_barrier_names = [f"values.value{slot}" for slot in range(5)]
    if barrier_names != expected_barrier_names:
        raise SystemExit(
            f"seed hunter barrier input names are invalid: {barrier_names}; "
            f"expected {expected_barrier_names} in {source.name}"
        )
    barrier_origins = {incoming[(barrier["id"], slot)][0] for slot in range(5)}
    if barrier_origins != sampler_ids:
        raise SystemExit(f"seed hunter barrier does not depend on all samplers: {source.name}")

    shared_input_types = {
        0: "UNETLoader",
        1: "MiniMaxH3ReferenceToVideo",
        2: "ConditioningZeroOut",
        3: "MiniMaxH3ReferenceToVideo",
    }
    for slot, expected_type in shared_input_types.items():
        origins = {incoming[(sampler["id"], slot)][0] for sampler in samplers}
        if len(origins) != 1 or by_id[next(iter(origins))].get("type") != expected_type:
            raise SystemExit(
                f"seed hunter sampler input {slot} is not shared from {expected_type}: {source.name}"
            )

    switches = [node for node in nodes if node.get("type") == "ComfySwitchNode"]
    gated_sampler_ids = set()
    for switch in switches:
        if incoming[(switch["id"], 0)][0] != barrier["id"]:
            raise SystemExit(f"seed hunter decode gate bypasses the barrier: {source.name}")
        false_origin = incoming[(switch["id"], 1)][0]
        true_origin = incoming[(switch["id"], 2)][0]
        if false_origin != true_origin or false_origin not in sampler_ids:
            raise SystemExit(f"seed hunter decode gate does not preserve one sampler: {source.name}")
        gated_sampler_ids.add(true_origin)
    if gated_sampler_ids != sampler_ids:
        raise SystemExit(f"seed hunter does not gate every sampler exactly once: {source.name}")

    prefixes = {
        node.get("widgets_values", [None, None])[1]
        for node in nodes
        if node.get("type") == "PixaromaSaveMp4"
    }
    if prefixes != {f"SeedHunter_S{index:02d}" for index in range(1, 6)}:
        raise SystemExit(f"seed hunter output prefixes are not stable and unique: {source.name}")


def validate_civitai_seed_hunter(data, source: Path, families: str) -> None:
    nodes = list(workflow_nodes(data))
    node_types = {node.get("type") for node in nodes}
    missing = SEED_HUNTER_REQUIRED_NODE_TYPES - node_types
    if missing:
        raise SystemExit(
            f"Civitai Seed Hunter is missing required node types: {sorted(missing)}"
        )
    subgraphs = data.get("definitions", {}).get("subgraphs", [])
    if len(subgraphs) != 6:
        raise SystemExit(
            f"expected six Civitai Seed Hunter subgraphs, found {len(subgraphs)}"
        )
    if set(families.split(",")) != {"ada", "blackwell"}:
        raise SystemExit("Civitai Seed Hunter must be scoped to Ada and Blackwell")

    sol_nodes = [node for node in nodes if node.get("type") == "SolAttnPatch"]
    if len(sol_nodes) != 1 or sol_nodes[0].get("mode", 0) != 4:
        raise SystemExit("SolAttnPatch must remain bypassed until GPU smoke-tested")

    media_suffixes = (".mp4", ".mov", ".webm", ".wav", ".mp3", ".flac")
    for node in nodes:
        if node.get("type") not in {"LoadAudioUI", "LoadVideoUI", "LoadImageCrop"}:
            continue
        values = node.get("widgets_values") or []
        if values and isinstance(values[0], str) and values[0].lower().endswith(media_suffixes):
            raise SystemExit(
                f"creator-local media is still selected in {source.name}: {values[0]}"
            )
        named_values = node.get("widgets_values_named") or {}
        if not isinstance(named_values, dict):
            raise SystemExit(f"invalid named widget values in {source.name}")
        unet_name = named_values.get("unet_name")
        if unet_name in DIFFUSION_MODEL_FILES or (
            isinstance(unet_name, str) and "\\" in unet_name
        ):
            raise SystemExit(
                f"named diffusion-model path is not Linux-portable in {source.name}: {unet_name}"
            )

    values = [value for node in nodes for value in strings(node.get("widgets_values"))]
    expected = {
        "h3/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "minimax_h3_video_vae_int8_convrot.safetensors",
        "minimax_h3_latent_upscaler_3d_bf16.safetensors",
        "taeh3.safetensors",
        "flownet.pkl",
    }
    if not expected.issubset(set(values)):
        raise SystemExit(
            f"Civitai Seed Hunter model references are incomplete: {sorted(expected - set(values))}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = root / "asset-manifest.tsv"
    if not manifest.is_file():
        raise SystemExit(f"missing manifest: {manifest}")

    node_lock = root / "seed-hunter-node-lock.tsv"
    if not node_lock.is_file():
        raise SystemExit(f"missing Seed Hunter node lock: {node_lock}")
    locked_folders = set()
    for line_number, raw in enumerate(node_lock.read_text(encoding="utf-8").splitlines(), 1):
        if not raw or raw.startswith("#"):
            continue
        fields = raw.split("\t")
        if len(fields) != 3:
            raise SystemExit(f"node lock line {line_number} must have three fields")
        folder, repository, commit = fields
        if "/" in folder or "\\" in folder or not repository.startswith("https://"):
            raise SystemExit(f"unsafe node lock entry on line {line_number}")
        if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
            raise SystemExit(f"invalid node commit on line {line_number}: {commit}")
        locked_folders.add(folder)
    if locked_folders != SEED_HUNTER_NODE_FOLDERS:
        raise SystemExit(
            f"Seed Hunter node-lock coverage mismatch: missing "
            f"{sorted(SEED_HUNTER_NODE_FOLDERS - locked_folders)}"
        )

    rows = []
    for line_number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not raw or raw.startswith("#"):
            continue
        fields = raw.split("\t")
        if len(fields) not in {4, 5}:
            raise SystemExit(
                f"manifest line {line_number} must have four or five tab-separated fields"
            )
        kind, repo_name, install_name, expected_sha = fields[:4]
        families = fields[4] if len(fields) == 5 else "all"
        if families != "all" and not set(families.split(",")) <= {
            "ampere",
            "ada",
            "blackwell",
        }:
            raise SystemExit(f"manifest line {line_number} has invalid families: {families}")
        if kind not in {"workflow", "input"}:
            raise SystemExit(f"manifest line {line_number} has unknown kind: {kind}")
        for name in (repo_name, install_name):
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts:
                raise SystemExit(f"manifest line {line_number} has unsafe path: {name}")
        source = root / Path(*PurePosixPath(repo_name).parts)
        if not source.is_file():
            raise SystemExit(f"manifest source is missing: {source}")
        actual_sha = digest(source)
        if actual_sha != expected_sha:
            raise SystemExit(
                f"manifest SHA256 mismatch for {repo_name}: expected {expected_sha}, found {actual_sha}"
            )
        rows.append((kind, repo_name, install_name, source, families))

    workflow_rows = [row for row in rows if row[0] == "workflow"]
    input_rows = [row for row in rows if row[0] == "input"]
    if len(workflow_rows) != 12:
        raise SystemExit(f"expected 12 workflow rows, found {len(workflow_rows)}")
    if len(input_rows) < 12:
        raise SystemExit(f"expected at least 12 input rows, found {len(input_rows)}")

    install_paths = [row[2] for row in rows]
    if len(install_paths) != len(set(install_paths)):
        raise SystemExit("manifest contains duplicate install paths")

    family_workflow_counts = {
        family: sum(
            families == "all" or family in families.split(",")
            for _, _, _, _, families in workflow_rows
        )
        for family in ("ampere", "ada", "blackwell")
    }
    if family_workflow_counts != {"ampere": 11, "ada": 12, "blackwell": 12}:
        raise SystemExit(f"workflow family coverage mismatch: {family_workflow_counts}")

    available_inputs = {
        PurePosixPath(row[2]).name
        for row in input_rows
        if PurePosixPath(row[2]).parts[0] == "input"
    }
    seen_models = set()
    seen_nodes = set()
    missing_inputs = set()
    for _, _, _, source, families in workflow_rows:
        data = json.loads(source.read_text(encoding="utf-8-sig"))
        nodes = data.get("nodes")
        if not isinstance(nodes, list):
            raise SystemExit(f"workflow has no node list: {source}")
        if source.name == "11-ref2va-seed-hunter-five-seeds.json":
            validate_ref2va_seed_hunter(data, source)
        if source.name == "12-fl2va-seed-hunter-upscale-continuation.json":
            validate_civitai_seed_hunter(data, source, families)
        for node in workflow_nodes(data):
            node_type = node.get("type")
            seen_nodes.add(node_type)
            values = list(strings(node.get("widgets_values")))
            if node_type == "UNETLoader" and values:
                if "\\" in values[0] or not values[0].startswith("h3/"):
                    raise SystemExit(
                        f"UNETLoader is not Linux-portable in {source.name}: {values[0]}"
                    )
            for value in values:
                normalized = value.replace("\\", "/")
                if value in DIFFUSION_MODEL_FILES or "\\" in value and value.rsplit("\\", 1)[-1] in DIFFUSION_MODEL_FILES:
                    raise SystemExit(
                        f"diffusion model path is not Linux-portable in {source.name}: {value}"
                    )
                filename = normalized.rsplit("/", 1)[-1]
                if filename.endswith(".safetensors"):
                    if filename not in MODEL_FILES:
                        raise SystemExit(f"unexpected model reference in {source.name}: {filename}")
                    seen_models.add(filename)
            if node_type in MEDIA_NODE_TYPES and values and values[0]:
                filename = values[0].replace("\\", "/").rsplit("/", 1)[-1]
                if filename not in available_inputs:
                    missing_inputs.add(filename)

    if seen_models != MODEL_FILES:
        raise SystemExit(f"model coverage mismatch: missing {sorted(MODEL_FILES - seen_models)}")
    missing_nodes = REQUIRED_NODE_TYPES - seen_nodes
    if missing_nodes:
        raise SystemExit(f"required workflow node types are missing: {sorted(missing_nodes)}")
    if missing_inputs:
        raise SystemExit(f"referenced sample inputs are not bundled: {sorted(missing_inputs)}")

    print(
        f"Validated {len(workflow_rows)} workflows, {len(input_rows)} inputs, "
        f"{len(MODEL_FILES)} model references, and all required H3 node families."
    )


if __name__ == "__main__":
    main()
