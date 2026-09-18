"""Pinned official QwenPaw-Flash-9B frontend resources used by artifact conversion.

This module owns the checkpoint-invariant resource profile for the
QwenPaw-Flash-9B target.  The converter synthesizes generation_config.json
and video_preprocessor_config.json deterministically from registered
special-token ids and an empty stub respectively.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from tools.convert.qwen3_6.common.conversion import ResourcePayload, load_resources
from tools.convert.qwen3_6.common.inventory import ResourceSpec

PAD_TOKEN_ID = 248044
EOS_TOKEN_ID = 248046

GENERATION_CONFIG_JSON = {
    "bos_token_id": PAD_TOKEN_ID,
    "do_sample": True,
    "eos_token_id": [EOS_TOKEN_ID, PAD_TOKEN_ID],
    "pad_token_id": PAD_TOKEN_ID,
    "temperature": 1.0,
    "top_k": 20,
    "top_p": 0.95,
    "transformers_version": "5.13.0",
}

OFFICIAL_RESOURCE_SHA256 = {
    "frontend/tokenizer.json": (
        "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4"
    ),
    "frontend/tokenizer_config.json": (
        "ff19cad4dcf439c2bb0d97ddeb41a51d60de511f6c0a7a74054cdaaca9c1878b"
    ),
    "frontend/chat_template.jinja": (
        "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
    ),
    "frontend/generation_config.json": (
        "d0dbf670c6a372817b2ff92d5d47e3130d35de9c3a7164ba455fd7a88255b362"
    ),
    "frontend/preprocessor_config.json": (
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
    ),
    "frontend/video_preprocessor_config.json": (
        "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"
    ),
}

GENERATION_CONFIG_SHA256 = OFFICIAL_RESOURCE_SHA256["frontend/generation_config.json"]

DISK_RESOURCES = {
    "frontend/tokenizer.json": "tokenizer.json",
    "frontend/tokenizer_config.json": "tokenizer_config.json",
    "frontend/chat_template.jinja": "chat_template.jinja",
    "frontend/preprocessor_config.json": "preprocessor_config.json",
    "frontend/video_preprocessor_config.json": "video_preprocessor_config.json",
}


def synthesize_generation_config() -> bytes:
    return json.dumps(GENERATION_CONFIG_JSON, indent=2).encode("utf-8") + b"\n"


def synthesize_video_preprocessor_config() -> bytes:
    return b"{}\n"


def patch_tokenizer_config(raw_bytes: bytes, chat_template: str) -> bytes:
    config = json.loads(raw_bytes)
    config["add_bos_token"] = False
    config["chat_template"] = chat_template
    return json.dumps(config, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"


def validate_official_resource_hashes(
    actual_hashes: Mapping[str, str],
) -> None:
    expected_names = tuple(OFFICIAL_RESOURCE_SHA256)
    actual_names = tuple(actual_hashes)
    if actual_names != expected_names:
        raise ValueError(
            "QwenPaw-Flash-9B frontend resource set mismatch: "
            f"expected {expected_names!r}, got {actual_names!r}"
        )
    for name, expected in OFFICIAL_RESOURCE_SHA256.items():
        actual = actual_hashes[name]
        if actual != expected:
            filename = name.removeprefix("frontend/")
            raise ValueError(
                f"official QwenPaw-Flash-9B resource hash mismatch for {filename}: "
                f"expected {expected}, got {actual}"
            )


def validate_official_resources(resources: Sequence[ResourcePayload]) -> None:
    hashes = {
        resource.name: hashlib.sha256(resource.data).hexdigest()
        for resource in resources
    }
    if len(hashes) != len(resources):
        raise ValueError("QwenPaw-Flash-9B frontend resource set contains duplicate names")
    validate_official_resource_hashes(hashes)


def load_official_resources(
    model_dir: str | Path,
    resource_specs: Sequence[ResourceSpec],
) -> tuple[ResourcePayload, ...]:
    root = Path(model_dir)
    spec_names = tuple(spec.name for spec in resource_specs)
    expected_names = tuple(OFFICIAL_RESOURCE_SHA256)
    if spec_names != expected_names:
        raise ValueError(
            "converter resource inventory does not match the official "
            f"QwenPaw-Flash-9B profile: expected {expected_names!r}, got {spec_names!r}"
        )

    chat_template_data = (root / "chat_template.jinja").read_bytes()
    if not chat_template_data:
        raise ValueError("frontend resource chat_template.jinja is empty")
    chat_template_str = chat_template_data.decode("utf-8")

    resources: list[ResourcePayload] = []
    for spec in resource_specs:
        if spec.name == "frontend/generation_config.json":
            data = synthesize_generation_config()
        elif spec.name == "frontend/tokenizer_config.json":
            raw = (root / "tokenizer_config.json").read_bytes()
            if not raw:
                raise ValueError("frontend resource tokenizer_config.json is empty")
            data = patch_tokenizer_config(raw, chat_template_str)
        elif spec.name == "frontend/chat_template.jinja":
            data = chat_template_data
        else:
            disk_name = DISK_RESOURCES[spec.name]
            data = (root / disk_name).read_bytes()
            if not data:
                raise ValueError(f"frontend resource {disk_name} is empty")
        resources.append(ResourcePayload(spec.name, data))

    validate_official_resources(resources)
    return tuple(resources)


__all__ = [
    "GENERATION_CONFIG_SHA256",
    "OFFICIAL_RESOURCE_SHA256",
    "load_official_resources",
    "patch_tokenizer_config",
    "synthesize_generation_config",
    "synthesize_video_preprocessor_config",
    "validate_official_resource_hashes",
    "validate_official_resources",
]
