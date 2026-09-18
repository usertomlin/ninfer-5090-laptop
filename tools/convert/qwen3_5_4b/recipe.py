"""Hugging Face source recipe for the complete Qwen3.5-4B inventory.

The registered checkpoint ships no ``lm_head.weight`` because the output head is
tied to the language-model embedding matrix; both the full output head and the
draft head therefore source ``model.language_model.embed_tokens.weight``.  The
Qwen3.5-4B vision tower is also checkpoint-specific (24 layers, width 1024), so
this package builds its own Vision recipes instead of the shared Qwen3.6 ones.
"""

from __future__ import annotations

from pathlib import Path

from tools.convert.qwen3_6.common.recipe import (
    SOURCE_DTYPE,
    Cast,
    Concat,
    DraftHeadTokenIds,
    Expression,
    GatherRows,
    Reshape,
    ShardReader,
    Slice,
    SourcePreflight,
    SourceTensor,
    TensorRecipe,
    Transpose,
    attention_qproj_part,
    expression_shape,
    expression_sources,
    materialize_expression,
    materialize_recipe,
    preflight_sources as _preflight_recipe_sources,
    source,
    source_requirements as _recipe_source_requirements,
    validate_recipe_coverage as _validate_recipe_coverage,
)

from . import inventory


DRAFT_ROWS = 131072

HIDDEN = 2560
INTERMEDIATE = 9216
KEY_DIM = 2048
VALUE_DIM = 4096
CONVOLUTION_DIM = 8192
QUERY_SIZE = 4096
KV_SIZE = 1024

VISION_DEPTH = 24
VISION_HIDDEN = 1024
VISION_INTERMEDIATE = 4096
VISION_MERGER_HIDDEN = 4096

_sources = expression_sources
_source = source


def _attention_qproj_part(source_name: str, gate: bool) -> Expression:
    return attention_qproj_part(
        source_name,
        gate,
        num_heads=16,
        hidden_size=HIDDEN,
    )


def _build_text_recipes() -> tuple[TensorRecipe, ...]:
    recipes: list[TensorRecipe] = [
        TensorRecipe(
            "text/token_embedding",
            _source("model.language_model.embed_tokens.weight", (248320, HIDDEN)),
        )
    ]

    for layer in range(32):
        source_prefix = f"model.language_model.layers.{layer}."
        object_prefix = f"text/layers/{layer}/"
        recipes.append(
            TensorRecipe(
                object_prefix + "input_norm",
                _source(source_prefix + "input_layernorm.weight", (HIDDEN,)),
            )
        )

        if layer in inventory.FULL_ATTENTION_LAYERS:
            q_proj = source_prefix + "self_attn.q_proj.weight"
            query = _attention_qproj_part(q_proj, gate=False)
            gate = _attention_qproj_part(q_proj, gate=True)
            recipes.extend(
                (
                    TensorRecipe(
                        object_prefix + "attention/query_key",
                        Concat(
                            (
                                query,
                                _source(source_prefix + "self_attn.k_proj.weight", (KV_SIZE, HIDDEN)),
                            ),
                            0,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "attention/gate_value",
                        Concat(
                            (
                                gate,
                                _source(source_prefix + "self_attn.v_proj.weight", (KV_SIZE, HIDDEN)),
                            ),
                            0,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "attention/query_norm",
                        _source(source_prefix + "self_attn.q_norm.weight", (256,)),
                    ),
                    TensorRecipe(
                        object_prefix + "attention/key_norm",
                        _source(source_prefix + "self_attn.k_norm.weight", (256,)),
                    ),
                    TensorRecipe(
                        object_prefix + "attention/output",
                        _source(source_prefix + "self_attn.o_proj.weight", (HIDDEN, QUERY_SIZE)),
                    ),
                )
            )
        else:
            qkv_source = _source(
                source_prefix + "linear_attn.in_proj_qkv.weight",
                (CONVOLUTION_DIM, HIDDEN),
            )
            convolution = _source(
                source_prefix + "linear_attn.conv1d.weight",
                (CONVOLUTION_DIM, 1, 4),
            )
            recipes.extend(
                (
                    TensorRecipe(
                        object_prefix + "gdn/a_log",
                        Cast(
                            _source(
                                source_prefix + "linear_attn.A_log",
                                (32,),
                                dtype="F32",
                            ),
                            inventory.FP32,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/dt_bias",
                        Cast(_source(source_prefix + "linear_attn.dt_bias", (32,)), inventory.FP32),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/convolution",
                        Transpose(
                            Reshape(Slice(convolution, 1, 0, 1), (CONVOLUTION_DIM, 4)),
                            (1, 0),
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/a_projection",
                        _source(source_prefix + "linear_attn.in_proj_a.weight", (32, HIDDEN)),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/b_projection",
                        _source(source_prefix + "linear_attn.in_proj_b.weight", (32, HIDDEN)),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/query_key",
                        Slice(qkv_source, 0, 0, 2 * KEY_DIM),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/value_z",
                        Concat(
                            (
                                Slice(qkv_source, 0, 2 * KEY_DIM, 2 * KEY_DIM + VALUE_DIM),
                                _source(
                                    source_prefix + "linear_attn.in_proj_z.weight",
                                    (VALUE_DIM, HIDDEN),
                                ),
                            ),
                            0,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/norm",
                        Cast(
                            _source(
                                source_prefix + "linear_attn.norm.weight",
                                (128,),
                                dtype="F32",
                            ),
                            inventory.BF16,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/output",
                        _source(source_prefix + "linear_attn.out_proj.weight", (HIDDEN, VALUE_DIM)),
                    ),
                )
            )

        recipes.extend(
            (
                TensorRecipe(
                    object_prefix + "post_attention_norm",
                    _source(source_prefix + "post_attention_layernorm.weight", (HIDDEN,)),
                ),
                TensorRecipe(
                    object_prefix + "mlp/gate_up",
                    Concat(
                        (
                            _source(source_prefix + "mlp.gate_proj.weight", (INTERMEDIATE, HIDDEN)),
                            _source(source_prefix + "mlp.up_proj.weight", (INTERMEDIATE, HIDDEN)),
                        ),
                        0,
                    ),
                ),
                TensorRecipe(
                    object_prefix + "mlp/down",
                    _source(source_prefix + "mlp.down_proj.weight", (HIDDEN, INTERMEDIATE)),
                ),
            )
        )

    recipes.extend(
        (
            TensorRecipe(
                "text/final_norm",
                _source("model.language_model.norm.weight", (HIDDEN,)),
            ),
            TensorRecipe(
                "text/output_head",
                _source("model.language_model.embed_tokens.weight", (248320, HIDDEN)),
            ),
        )
    )
    return tuple(recipes)


def _build_draft_head_recipes() -> tuple[TensorRecipe, ...]:
    return (
        TensorRecipe(
            "text/draft_head",
            GatherRows(
                _source("model.language_model.embed_tokens.weight", (248320, HIDDEN)),
                token_ids_object="text/draft_head_token_ids",
                rows=DRAFT_ROWS,
            ),
        ),
        TensorRecipe(
            "text/draft_head_token_ids",
            DraftHeadTokenIds(
                ranking_path="tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64",
                tokenizer_resource="frontend/tokenizer_config.json",
                vocab_rows=248320,
                tokenizer_id_count=248077,
                rows=DRAFT_ROWS,
            ),
        ),
    )


def _build_mtp_recipes() -> tuple[TensorRecipe, ...]:
    source_prefix = "mtp.layers.0."
    q_proj = source_prefix + "self_attn.q_proj.weight"
    return (
        TensorRecipe("mtp/input_projection", _source("mtp.fc.weight", (HIDDEN, 5120))),
        TensorRecipe(
            "mtp/embedding_norm",
            _source("mtp.pre_fc_norm_embedding.weight", (HIDDEN,)),
        ),
        TensorRecipe(
            "mtp/hidden_norm",
            _source("mtp.pre_fc_norm_hidden.weight", (HIDDEN,)),
        ),
        TensorRecipe(
            "mtp/layer/input_norm",
            _source(source_prefix + "input_layernorm.weight", (HIDDEN,)),
        ),
        TensorRecipe(
            "mtp/layer/attention/query_key_gate_value",
            Concat(
                (
                    _attention_qproj_part(q_proj, gate=False),
                    _source(source_prefix + "self_attn.k_proj.weight", (KV_SIZE, HIDDEN)),
                    _attention_qproj_part(q_proj, gate=True),
                    _source(source_prefix + "self_attn.v_proj.weight", (KV_SIZE, HIDDEN)),
                ),
                0,
            ),
        ),
        TensorRecipe(
            "mtp/layer/attention/query_norm",
            _source(source_prefix + "self_attn.q_norm.weight", (256,)),
        ),
        TensorRecipe(
            "mtp/layer/attention/key_norm",
            _source(source_prefix + "self_attn.k_norm.weight", (256,)),
        ),
        TensorRecipe(
            "mtp/layer/attention/output",
            _source(source_prefix + "self_attn.o_proj.weight", (HIDDEN, QUERY_SIZE)),
        ),
        TensorRecipe(
            "mtp/layer/post_attention_norm",
            _source(source_prefix + "post_attention_layernorm.weight", (HIDDEN,)),
        ),
        TensorRecipe(
            "mtp/layer/mlp/gate_up",
            Concat(
                (
                    _source(source_prefix + "mlp.gate_proj.weight", (INTERMEDIATE, HIDDEN)),
                    _source(source_prefix + "mlp.up_proj.weight", (INTERMEDIATE, HIDDEN)),
                ),
                0,
            ),
        ),
        TensorRecipe(
            "mtp/layer/mlp/down",
            _source(source_prefix + "mlp.down_proj.weight", (HIDDEN, INTERMEDIATE)),
        ),
        TensorRecipe("mtp/final_norm", _source("mtp.norm.weight", (HIDDEN,))),
    )


def _build_vision_recipes() -> tuple[TensorRecipe, ...]:
    source_prefix = "model.visual."
    recipes: list[TensorRecipe] = [
        TensorRecipe(
            "vision/patch_embedding",
            Reshape(
                _source(
                    source_prefix + "patch_embed.proj.weight",
                    (VISION_HIDDEN, 3, 2, 16, 16),
                ),
                (VISION_HIDDEN, 1536),
            ),
        ),
        TensorRecipe(
            "vision/patch_embedding_bias",
            _source(source_prefix + "patch_embed.proj.bias", (VISION_HIDDEN,)),
        ),
        TensorRecipe(
            "vision/position_embedding",
            _source(source_prefix + "pos_embed.weight", (2304, VISION_HIDDEN)),
        ),
    ]

    for layer in range(VISION_DEPTH):
        source_layer = source_prefix + f"blocks.{layer}."
        object_layer = f"vision/layers/{layer}/"
        for object_suffix, source_suffix, shape in (
            ("attention/qkv", "attn.qkv.weight", (3 * VISION_HIDDEN, VISION_HIDDEN)),
            ("attention/qkv_bias", "attn.qkv.bias", (3 * VISION_HIDDEN,)),
            ("attention/output", "attn.proj.weight", (VISION_HIDDEN, VISION_HIDDEN)),
            ("attention/output_bias", "attn.proj.bias", (VISION_HIDDEN,)),
            ("mlp/fc1", "mlp.linear_fc1.weight", (VISION_INTERMEDIATE, VISION_HIDDEN)),
            ("mlp/fc1_bias", "mlp.linear_fc1.bias", (VISION_INTERMEDIATE,)),
            ("mlp/fc2", "mlp.linear_fc2.weight", (VISION_HIDDEN, VISION_INTERMEDIATE)),
            ("mlp/fc2_bias", "mlp.linear_fc2.bias", (VISION_HIDDEN,)),
            ("norm1/weight", "norm1.weight", (VISION_HIDDEN,)),
            ("norm1/bias", "norm1.bias", (VISION_HIDDEN,)),
            ("norm2/weight", "norm2.weight", (VISION_HIDDEN,)),
            ("norm2/bias", "norm2.bias", (VISION_HIDDEN,)),
        ):
            recipes.append(
                TensorRecipe(
                    object_layer + object_suffix,
                    _source(source_layer + source_suffix, shape),
                )
            )

    for object_suffix, source_suffix, shape in (
        ("fc1", "linear_fc1.weight", (VISION_MERGER_HIDDEN, VISION_MERGER_HIDDEN)),
        ("fc1_bias", "linear_fc1.bias", (VISION_MERGER_HIDDEN,)),
        ("fc2", "linear_fc2.weight", (HIDDEN, VISION_MERGER_HIDDEN)),
        ("fc2_bias", "linear_fc2.bias", (HIDDEN,)),
        ("norm/weight", "norm.weight", (VISION_HIDDEN,)),
        ("norm/bias", "norm.bias", (VISION_HIDDEN,)),
    ):
        recipes.append(
            TensorRecipe(
                "vision/merger/" + object_suffix,
                _source(source_prefix + "merger." + source_suffix, shape),
            )
        )
    return tuple(recipes)


RECIPE_SPECS = (
    _build_text_recipes()
    + _build_draft_head_recipes()
    + _build_mtp_recipes()
    + _build_vision_recipes()
)
RECIPES_BY_NAME = {recipe.object_name: recipe for recipe in RECIPE_SPECS}


def validate_recipe_coverage() -> None:
    _validate_recipe_coverage(RECIPE_SPECS, inventory.TENSOR_SPECS)


def source_requirements() -> dict[str, SourceTensor]:
    return _recipe_source_requirements(RECIPE_SPECS)


def preflight_sources(model_dir: str | Path) -> SourcePreflight:
    return _preflight_recipe_sources(model_dir, RECIPE_SPECS)


validate_recipe_coverage()
