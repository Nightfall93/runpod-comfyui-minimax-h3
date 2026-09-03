#!/usr/bin/env python3
"""Build the managed REF2VA seed hunter and normalize the Civitai workflow."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"{path}: {hashlib.sha256(path.read_bytes()).hexdigest()}")


def iter_nodes(data: dict):
    yield from data.get("nodes", [])
    for subgraph in data.get("definitions", {}).get("subgraphs", []):
        yield from subgraph.get("nodes", [])


def build_ref2va_seed_hunter(source: Path, output: Path) -> None:
    data = json.loads(source.read_text(encoding="utf-8-sig"))
    by_type = {node["type"]: node for node in data["nodes"]}
    sampler_template = copy.deepcopy(by_type["KSampler"])
    video_decode_template = copy.deepcopy(by_type["VAEDecode"])
    audio_decode_template = copy.deepcopy(by_type["VAEDecodeAudio"])
    save_template = copy.deepcopy(by_type["PixaromaSaveMp4"])
    removed_ids = {
        sampler_template["id"],
        video_decode_template["id"],
        audio_decode_template["id"],
        save_template["id"],
    }
    removed_links = {
        link[0]
        for link in data["links"]
        if link[1] in removed_ids or link[3] in removed_ids
    }
    data["nodes"] = [node for node in data["nodes"] if node["id"] not in removed_ids]
    data["links"] = [link for link in data["links"] if link[0] not in removed_links]
    for node in data["nodes"]:
        for item in node.get("inputs", []):
            if item.get("link") in removed_links:
                item["link"] = None
        for item in node.get("outputs", []):
            item["links"] = [
                link for link in (item.get("links") or []) if link not in removed_links
            ]

    nodes = {node["id"]: node for node in data["nodes"]}
    next_node_id = max(nodes) + 1
    next_link_id = max(link[0] for link in data["links"]) + 1

    def new_node_id() -> int:
        nonlocal next_node_id
        value = next_node_id
        next_node_id += 1
        return value

    def connect(
        source_id: int,
        source_slot: int,
        target_id: int,
        target_slot: int,
        value_type: str,
    ) -> int:
        nonlocal next_link_id
        value = next_link_id
        next_link_id += 1
        data["links"].append(
            [value, source_id, source_slot, target_id, target_slot, value_type]
        )
        nodes[source_id]["outputs"][source_slot].setdefault("links", []).append(value)
        nodes[target_id]["inputs"][target_slot]["link"] = value
        return value

    barrier = {
        "id": new_node_id(),
        "type": "ComfyAndNode",
        "pos": [246.0, -70.0],
        "size": [260.0, 150.0],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {
                "label": f"value{index}",
                "name": f"values.value{index}",
                "type": "*",
                "link": None,
            }
            for index in range(5)
        ],
        "outputs": [{"name": "BOOLEAN", "type": "BOOLEAN", "links": []}],
        "properties": {
            "cnr_id": "comfy-core",
            "ver": "0.34.0",
            "Node name for S&R": "ComfyAndNode",
        },
        "widgets_values": [],
        "color": "#1d1d1d",
        "bgcolor": "#2a2a2a",
    }
    data["nodes"].append(barrier)
    nodes[barrier["id"]] = barrier

    shared = {
        "model": by_type["UNETLoader"]["id"],
        "positive": by_type["MiniMaxH3ReferenceToVideo"]["id"],
        "negative": by_type["ConditioningZeroOut"]["id"],
        "latent": by_type["MiniMaxH3ReferenceToVideo"]["id"],
        "video_vae": by_type["VAELoader"]["id"],
    }
    vaes = [node for node in data["nodes"] if node["type"] == "VAELoader"]
    shared["video_vae"] = next(
        node["id"] for node in vaes if "video_vae" in node["widgets_values"][0]
    )
    shared["audio_vae"] = next(
        node["id"] for node in vaes if "audio_vae" in node["widgets_values"][0]
    )

    for branch in range(5):
        x = 240.0 + branch * 520.0
        sampler = copy.deepcopy(sampler_template)
        sampler["id"] = new_node_id()
        sampler["pos"] = [x, 220.0]
        sampler["outputs"][0]["links"] = []
        for item in sampler["inputs"]:
            item["link"] = None
        sampler["widgets_values"][0] = 428003307890273 + branch * 104729
        sampler["widgets_values"][1] = "randomize"

        switch = {
            "id": new_node_id(),
            "type": "ComfySwitchNode",
            "pos": [x, 515.0],
            "size": [270.0, 110.0],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": [
                {"name": "switch", "type": "BOOLEAN", "link": None},
                {"name": "on_false", "type": "*", "link": None},
                {"name": "on_true", "type": "*", "link": None},
            ],
            "outputs": [{"name": "output", "type": "*", "links": []}],
            "properties": {
                "cnr_id": "comfy-core",
                "ver": "0.34.0",
                "Node name for S&R": "ComfySwitchNode",
            },
            "widgets_values": [],
            "color": "#1d1d1d",
            "bgcolor": "#2a2a2a",
        }

        video_decode = copy.deepcopy(video_decode_template)
        video_decode["id"] = new_node_id()
        video_decode["pos"] = [x, 665.0]
        for item in video_decode["inputs"]:
            item["link"] = None
        video_decode["outputs"][0]["links"] = []

        audio_decode = copy.deepcopy(audio_decode_template)
        audio_decode["id"] = new_node_id()
        audio_decode["pos"] = [x, 755.0]
        for item in audio_decode["inputs"]:
            item["link"] = None
        audio_decode["outputs"][0]["links"] = []

        save = copy.deepcopy(save_template)
        save["id"] = new_node_id()
        save["pos"] = [x, 875.0]
        for item in save["inputs"]:
            item["link"] = None
        save["widgets_values"][1] = f"SeedHunter_S{branch + 1:02d}"
        save.get("properties", {}).pop("pixMp4Video", None)

        for node in (sampler, switch, video_decode, audio_decode, save):
            data["nodes"].append(node)
            nodes[node["id"]] = node

        connect(shared["model"], 0, sampler["id"], 0, "MODEL")
        connect(shared["positive"], 0, sampler["id"], 1, "CONDITIONING")
        connect(shared["negative"], 0, sampler["id"], 2, "CONDITIONING")
        connect(shared["latent"], 1, sampler["id"], 3, "LATENT")
        connect(sampler["id"], 0, barrier["id"], branch, "LATENT")
        connect(barrier["id"], 0, switch["id"], 0, "BOOLEAN")
        connect(sampler["id"], 0, switch["id"], 1, "LATENT")
        connect(sampler["id"], 0, switch["id"], 2, "LATENT")
        connect(switch["id"], 0, video_decode["id"], 0, "LATENT")
        connect(shared["video_vae"], 0, video_decode["id"], 1, "VAE")
        connect(switch["id"], 0, audio_decode["id"], 0, "LATENT")
        connect(shared["audio_vae"], 0, audio_decode["id"], 1, "VAE")
        connect(video_decode["id"], 0, save["id"], 0, "IMAGE")
        connect(audio_decode["id"], 0, save["id"], 1, "AUDIO")

    for order, node in enumerate(data["nodes"]):
        node["order"] = order
    data["last_node_id"] = max(nodes)
    data["last_link_id"] = max(link[0] for link in data["links"])
    data["id"] = "68ed2b6f-8527-50cf-aa86-e1ccb5591354"
    write_json(output, data)


def normalize_civitai_seed_hunter(source: Path, output: Path) -> None:
    data = json.loads(source.read_text(encoding="utf-8-sig"))
    diffusion_name = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    portable_name = f"h3/{diffusion_name}"
    media_suffixes = (".mp4", ".mov", ".webm", ".wav", ".mp3", ".flac")

    for node in iter_nodes(data):
        values = node.get("widgets_values")
        if isinstance(values, list):
            for index, value in enumerate(values):
                if value == diffusion_name:
                    values[index] = portable_name
            if (
                node.get("type") in {"LoadAudioUI", "LoadVideoUI", "LoadImageCrop"}
                and values
                and isinstance(values[0], str)
                and values[0].lower().endswith(media_suffixes)
            ):
                values[0] = ""
        properties = node.get("properties")
        if isinstance(properties, dict) and properties.get("unet_name") == diffusion_name:
            properties["unet_name"] = portable_name
        named_values = node.get("widgets_values_named")
        if isinstance(named_values, dict) and named_values.get("unet_name") == diffusion_name:
            named_values["unet_name"] = portable_name
        if node.get("type") == "SolAttnPatch":
            # Enable only after a real Ada/Blackwell kernel smoke test.
            node["mode"] = 4

    write_json(output, data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--civitai-source", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    build_ref2va_seed_hunter(
        repo / "workflows/06-ref2va-reference-two-images.json",
        repo / "workflows/11-ref2va-seed-hunter-five-seeds.json",
    )
    normalize_civitai_seed_hunter(
        args.civitai_source,
        repo / "workflows/12-fl2va-seed-hunter-upscale-continuation.json",
    )


if __name__ == "__main__":
    main()
