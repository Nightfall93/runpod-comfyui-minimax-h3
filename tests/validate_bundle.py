#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


MODEL_FILES = {
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "minimax_h3_audio_vae_fp32.safetensors",
    "minimax_h3_video_vae_fp16.safetensors",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = root / "asset-manifest.tsv"
    if not manifest.is_file():
        raise SystemExit(f"missing manifest: {manifest}")

    rows = []
    for line_number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not raw or raw.startswith("#"):
            continue
        fields = raw.split("\t")
        if len(fields) != 4:
            raise SystemExit(f"manifest line {line_number} does not have four tab-separated fields")
        kind, repo_name, install_name, expected_sha = fields
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
        rows.append((kind, repo_name, install_name, source))

    workflow_rows = [row for row in rows if row[0] == "workflow"]
    input_rows = [row for row in rows if row[0] == "input"]
    if len(workflow_rows) != 10:
        raise SystemExit(f"expected 10 workflow rows, found {len(workflow_rows)}")
    if len(input_rows) < 12:
        raise SystemExit(f"expected at least 12 input rows, found {len(input_rows)}")

    install_paths = [row[2] for row in rows]
    if len(install_paths) != len(set(install_paths)):
        raise SystemExit("manifest contains duplicate install paths")

    available_inputs = {
        PurePosixPath(row[2]).name
        for row in input_rows
        if PurePosixPath(row[2]).parts[0] == "input"
    }
    seen_models = set()
    seen_nodes = set()
    missing_inputs = set()
    for _, _, _, source in workflow_rows:
        data = json.loads(source.read_text(encoding="utf-8-sig"))
        nodes = data.get("nodes")
        if not isinstance(nodes, list):
            raise SystemExit(f"workflow has no node list: {source}")
        for node in nodes:
            node_type = node.get("type")
            seen_nodes.add(node_type)
            values = list(strings(node.get("widgets_values")))
            if node_type == "UNETLoader" and values:
                if "\\" in values[0] or not values[0].startswith("h3/"):
                    raise SystemExit(
                        f"UNETLoader is not Linux-portable in {source.name}: {values[0]}"
                    )
            for value in values:
                filename = value.replace("\\", "/").rsplit("/", 1)[-1]
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
        "five model references, and all required H3 node families."
    )


if __name__ == "__main__":
    main()
