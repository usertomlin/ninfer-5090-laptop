"""Persistent-object contract for the Qwen3.5-4B target.

This module contains only target storage roles. Source-checkpoint mapping and
materialization live in the sibling conversion recipe.
"""

from __future__ import annotations

from tools.convert.qwen3_6.common.inventory import (
    BF16,
    CONTIGUOUS_LAYOUT,
    DIRECT_FORMATS,
    FORMAT_NAMES,
    FP32,
    I32,
    LAYOUT_NAMES,
    LogicalAliasSpec,
    LogicalRowViewSpec,
    Q4,
    Q5,
    Q6,
    RESOURCE_ENCODING,
    RESOURCE_SPECS,
    ROW_SPLIT_LAYOUT,
    ResourceSpec,
    StoredObjectSpec,
    TensorSpec,
    W8,
    tensor_spec,
)


MODEL_ID = "qwen3.5-4b"
WEIGHTS_ID = "groupwise-int"
TARGET_KEY = "qwen3_5_4b"

HIDDEN = 2560
FULL_ATTENTION_LAYERS = tuple(range(3, 32, 4))
GDN_LAYERS = tuple(layer for layer in range(32) if layer not in FULL_ATTENTION_LAYERS)

KEY_DIM = 2048
VALUE_DIM = 4096
QUERY_SIZE = 4096

VISION_DEPTH = 24
VISION_HIDDEN = 1024
VISION_INTERMEDIATE = 4096
VISION_MERGER_HIDDEN = 4096

_tensor = tensor_spec


def _build_text_core_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        _tensor("text/token_embedding", (248320, HIDDEN), Q6),
    ]

    for layer in range(32):
        prefix = f"text/layers/{layer}/"
        specs.append(_tensor(prefix + "input_norm", (HIDDEN,), BF16))

        if layer in FULL_ATTENTION_LAYERS:
            specs.extend(
                (
                    _tensor(prefix + "attention/query_key", (5120, HIDDEN), Q4),
                    _tensor(prefix + "attention/gate_value", (5120, HIDDEN), Q5),
                    _tensor(prefix + "attention/query_norm", (256,), BF16),
                    _tensor(prefix + "attention/key_norm", (256,), BF16),
                    _tensor(prefix + "attention/output", (HIDDEN, QUERY_SIZE), Q5),
                )
            )
        else:
            specs.extend(
                (
                    _tensor(prefix + "gdn/a_log", (32,), FP32),
                    _tensor(prefix + "gdn/dt_bias", (32,), FP32),
                    _tensor(prefix + "gdn/convolution", (4, 8192), BF16),
                    _tensor(prefix + "gdn/a_projection", (32, HIDDEN), BF16),
                    _tensor(prefix + "gdn/b_projection", (32, HIDDEN), BF16),
                    _tensor(prefix + "gdn/query_key", (2 * KEY_DIM, HIDDEN), Q4),
                    _tensor(prefix + "gdn/value_z", (2 * VALUE_DIM, HIDDEN), Q5),
                    _tensor(prefix + "gdn/norm", (128,), BF16),
                    _tensor(prefix + "gdn/output", (HIDDEN, VALUE_DIM), Q5),
                )
            )

        specs.extend(
            (
                _tensor(prefix + "post_attention_norm", (HIDDEN,), BF16),
                _tensor(prefix + "mlp/gate_up", (18432, HIDDEN), Q4),
                _tensor(prefix + "mlp/down", (HIDDEN, 9216), Q5),
            )
        )

    specs.extend(
        (
            _tensor("text/final_norm", (HIDDEN,), BF16),
            _tensor("text/output_head", (248320, HIDDEN), Q6),
        )
    )
    return tuple(specs)


def _build_draft_head_specs() -> tuple[TensorSpec, ...]:
    return (
        _tensor("text/draft_head", (131072, HIDDEN), Q4),
        _tensor("text/draft_head_token_ids", (131072,), I32),
    )


def _build_mtp_specs() -> tuple[TensorSpec, ...]:
    return (
        _tensor("mtp/input_projection", (HIDDEN, 5120), W8),
        _tensor("mtp/embedding_norm", (HIDDEN,), BF16),
        _tensor("mtp/hidden_norm", (HIDDEN,), BF16),
        _tensor("mtp/layer/input_norm", (HIDDEN,), BF16),
        _tensor("mtp/layer/attention/query_key_gate_value", (10240, HIDDEN), W8),
        _tensor("mtp/layer/attention/query_norm", (256,), BF16),
        _tensor("mtp/layer/attention/key_norm", (256,), BF16),
        _tensor("mtp/layer/attention/output", (HIDDEN, QUERY_SIZE), W8),
        _tensor("mtp/layer/post_attention_norm", (HIDDEN,), BF16),
        _tensor("mtp/layer/mlp/gate_up", (18432, HIDDEN), W8),
        _tensor("mtp/layer/mlp/down", (HIDDEN, 9216), W8),
        _tensor("mtp/final_norm", (HIDDEN,), BF16),
    )


def _build_vision_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        _tensor("vision/patch_embedding", (VISION_HIDDEN, 1536), Q6),
        _tensor("vision/patch_embedding_bias", (VISION_HIDDEN,), BF16),
        _tensor("vision/position_embedding", (2304, VISION_HIDDEN), BF16),
    ]

    for layer in range(VISION_DEPTH):
        prefix = f"vision/layers/{layer}/"
        specs.extend(
            (
                _tensor(prefix + "attention/qkv", (3 * VISION_HIDDEN, VISION_HIDDEN), Q4),
                _tensor(prefix + "attention/qkv_bias", (3 * VISION_HIDDEN,), BF16),
                _tensor(prefix + "attention/output", (VISION_HIDDEN, VISION_HIDDEN), Q5),
                _tensor(prefix + "attention/output_bias", (VISION_HIDDEN,), BF16),
                _tensor(prefix + "mlp/fc1", (VISION_INTERMEDIATE, VISION_HIDDEN), Q4),
                _tensor(prefix + "mlp/fc1_bias", (VISION_INTERMEDIATE,), BF16),
                _tensor(prefix + "mlp/fc2", (VISION_HIDDEN, VISION_INTERMEDIATE), Q5),
                _tensor(prefix + "mlp/fc2_bias", (VISION_HIDDEN,), BF16),
                _tensor(prefix + "norm1/weight", (VISION_HIDDEN,), BF16),
                _tensor(prefix + "norm1/bias", (VISION_HIDDEN,), BF16),
                _tensor(prefix + "norm2/weight", (VISION_HIDDEN,), BF16),
                _tensor(prefix + "norm2/bias", (VISION_HIDDEN,), BF16),
            )
        )

    specs.extend(
        (
            _tensor("vision/merger/fc1", (VISION_MERGER_HIDDEN, VISION_MERGER_HIDDEN), W8),
            _tensor("vision/merger/fc1_bias", (VISION_MERGER_HIDDEN,), BF16),
            _tensor("vision/merger/fc2", (HIDDEN, VISION_MERGER_HIDDEN), W8),
            _tensor("vision/merger/fc2_bias", (HIDDEN,), BF16),
            _tensor("vision/merger/norm/weight", (VISION_HIDDEN,), BF16),
            _tensor("vision/merger/norm/bias", (VISION_HIDDEN,), BF16),
        )
    )
    return tuple(specs)


TEXT_CORE_TENSOR_SPECS = _build_text_core_specs()
DRAFT_HEAD_TENSOR_SPECS = _build_draft_head_specs()
MTP_TENSOR_SPECS = _build_mtp_specs()
VISION_TENSOR_SPECS = _build_vision_specs()

TENSOR_SPECS = (
    TEXT_CORE_TENSOR_SPECS
    + DRAFT_HEAD_TENSOR_SPECS
    + MTP_TENSOR_SPECS
    + VISION_TENSOR_SPECS
)
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS

FORMAT_COUNTS = {
    numeric_format: sum(spec.format == numeric_format for spec in TENSOR_SPECS)
    for numeric_format in FORMAT_NAMES
}
LAYOUT_COUNTS = {
    layout: sum(spec.layout == layout for spec in TENSOR_SPECS)
    for layout in LAYOUT_NAMES
}


LOGICAL_ROW_VIEW_SPECS = (
    LogicalRowViewSpec(
        "text/layers/{l}/attention/query",
        "text/layers/{l}/attention/query_key",
        0,
        4096,
        (4096, HIDDEN),
        FULL_ATTENTION_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/attention/key",
        "text/layers/{l}/attention/query_key",
        4096,
        5120,
        (1024, HIDDEN),
        FULL_ATTENTION_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/attention/output_gate",
        "text/layers/{l}/attention/gate_value",
        0,
        4096,
        (4096, HIDDEN),
        FULL_ATTENTION_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/attention/value",
        "text/layers/{l}/attention/gate_value",
        4096,
        5120,
        (1024, HIDDEN),
        FULL_ATTENTION_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/gdn/query",
        "text/layers/{l}/gdn/query_key",
        0,
        2048,
        (2048, HIDDEN),
        GDN_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/gdn/key",
        "text/layers/{l}/gdn/query_key",
        2048,
        4096,
        (2048, HIDDEN),
        GDN_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/gdn/value",
        "text/layers/{l}/gdn/value_z",
        0,
        4096,
        (4096, HIDDEN),
        GDN_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/gdn/z",
        "text/layers/{l}/gdn/value_z",
        4096,
        8192,
        (4096, HIDDEN),
        GDN_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/mlp/gate",
        "text/layers/{l}/mlp/gate_up",
        0,
        9216,
        (9216, HIDDEN),
        tuple(range(32)),
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/mlp/up",
        "text/layers/{l}/mlp/gate_up",
        9216,
        18432,
        (9216, HIDDEN),
        tuple(range(32)),
    ),
    LogicalRowViewSpec(
        "mtp/layer/attention/query",
        "mtp/layer/attention/query_key_gate_value",
        0,
        4096,
        (4096, HIDDEN),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/attention/key",
        "mtp/layer/attention/query_key_gate_value",
        4096,
        5120,
        (1024, HIDDEN),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/attention/output_gate",
        "mtp/layer/attention/query_key_gate_value",
        5120,
        9216,
        (4096, HIDDEN),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/attention/value",
        "mtp/layer/attention/query_key_gate_value",
        9216,
        10240,
        (1024, HIDDEN),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/mlp/gate",
        "mtp/layer/mlp/gate_up",
        0,
        9216,
        (9216, HIDDEN),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/mlp/up",
        "mtp/layer/mlp/gate_up",
        9216,
        18432,
        (9216, HIDDEN),
        None,
    ),
)


ALIAS_SPECS = (
    LogicalAliasSpec("mtp/token_embedding", ("text/token_embedding",)),
    LogicalAliasSpec("mtp/full_output_head", ("text/output_head",)),
    LogicalAliasSpec(
        "mtp/optimized_proposal_head",
        ("text/draft_head", "text/draft_head_token_ids"),
    ),
    LogicalAliasSpec(
        "text/layers/{l}/gdn/channel_major_convolution",
        ("text/layers/{l}/gdn/convolution",),
        layers=GDN_LAYERS,
        axis_order=(1, 0),
    ),
)
