"""Pinned official Qwen3.5-4B frontend resources used by artifact conversion.

This module owns only the checkpoint-invariant resource profile for the
Qwen3.5-2B target.  The Qwen3.5-2B release ships no `generation_config.json`;
the registered converter synthesizes that resource deterministically from the
registered special-token ids so the artifact keeps the complete six-resource
profile that the shared runtime requires.
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

OFFICIAL_RESOURCE_NAMES = (
    "frontend/tokenizer.json",
    "frontend/tokenizer_config.json",
    "frontend/chat_template.jinja",
    "frontend/generation_config.json",
    "frontend/preprocessor_config.json",
    "frontend/video_preprocessor_config.json",
)


def synthesize_generation_config() -> bytes:
    """Return the registered deterministic generation_config.json payload."""
    return json.dumps(GENERATION_CONFIG_JSON, indent=2).encode("utf-8") + b"\n"


def load_official_resources(
    model_dir: str | Path,
    resource_specs: Sequence[ResourceSpec],
) -> tuple[ResourcePayload, ...]:
    """Load the pinned resource set, synthesizing the missing generation config."""

    spec_names = tuple(spec.name for spec in resource_specs)
    expected_names = OFFICIAL_RESOURCE_NAMES
    if spec_names != expected_names:
        raise ValueError(
            "converter resource inventory does not match the official "
            f"Qwen3.5-2B profile: expected {expected_names!r}, got {spec_names!r}"
        )
    disk_specs = [
        spec for spec in resource_specs if spec.name != "frontend/generation_config.json"
    ]
    resources = list(load_resources(model_dir, disk_specs))
    resources.append(
        ResourcePayload(
            "frontend/generation_config.json", synthesize_generation_config()
        )
    )
    by_name = {payload.name: payload for payload in resources}
    resources = [by_name[spec.name] for spec in resource_specs]
    return tuple(resources)


__all__ = [
    "load_official_resources",
    "synthesize_generation_config",
]
