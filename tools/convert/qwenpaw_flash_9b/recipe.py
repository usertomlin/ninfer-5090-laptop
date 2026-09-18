"""Hugging Face source recipe for the QwenPaw-Flash-9B inventory.

Text and vision recipes are identical to Qwen3.5-9B.  MTP recipes are absent
because the checkpoint ships no MTP weights; the converter fills zero tensors
for those objects instead.
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
    build_vision_recipes,
    expression_shape,
    expression_sources,
    materialize_expression,
    materialize_recipe,
    preflight_source_reader,
    source,
    source_requirements as _recipe_source_requirements,
    validate_recipe_coverage as _validate_recipe_coverage,
)

from . import inventory


DRAFT_ROWS = 131072


_sources = expression_sources
_source = source


def _attention_qproj_part(source_name: str, gate: bool) -> Expression:
    return attention_qproj_part(
        source_name,
        gate,
        num_heads=16,
        hidden_size=4096,
    )


def _build_text_recipes() -> tuple[TensorRecipe, ...]:
    recipes: list[TensorRecipe] = [
        TensorRecipe(
            "text/token_embedding",
            _source("model.language_model.embed_tokens.weight", (248320, 4096)),
        )
    ]

    for layer in range(32):
        source_prefix = f"model.language_model.layers.{layer}."
        object_prefix = f"text/layers/{layer}/"
        recipes.append(
            TensorRecipe(
                object_prefix + "input_norm",
                _source(source_prefix + "input_layernorm.weight", (4096,)),
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
                                _source(source_prefix + "self_attn.k_proj.weight", (1024, 4096)),
                            ),
                            0,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "attention/gate_value",
                        Concat(
                            (
                                gate,
                                _source(source_prefix + "self_attn.v_proj.weight", (1024, 4096)),
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
                        _source(source_prefix + "self_attn.o_proj.weight", (4096, 4096)),
                    ),
                )
            )
        else:
            qkv_source = _source(
                source_prefix + "linear_attn.in_proj_qkv.weight",
                (8192, 4096),
            )
            convolution = _source(
                source_prefix + "linear_attn.conv1d.weight",
                (8192, 1, 4),
            )
            recipes.extend(
                (
                    TensorRecipe(
                        object_prefix + "gdn/a_log",
                        Cast(
                            _source(
                                source_prefix + "linear_attn.A_log",
                                (32,),
                            ),
                            inventory.FP32,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/dt_bias",
                        Cast(
                            _source(source_prefix + "linear_attn.dt_bias", (32,)),
                            inventory.FP32,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/convolution",
                        Transpose(
                            Reshape(Slice(convolution, 1, 0, 1), (8192, 4)),
                            (1, 0),
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/a_projection",
                        _source(source_prefix + "linear_attn.in_proj_a.weight", (32, 4096)),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/b_projection",
                        _source(source_prefix + "linear_attn.in_proj_b.weight", (32, 4096)),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/query_key",
                        Slice(qkv_source, 0, 0, 4096),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/value_z",
                        Concat(
                            (
                                Slice(qkv_source, 0, 4096, 8192),
                                _source(
                                    source_prefix + "linear_attn.in_proj_z.weight",
                                    (4096, 4096),
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
                            ),
                            inventory.BF16,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/output",
                        _source(source_prefix + "linear_attn.out_proj.weight", (4096, 4096)),
                    ),
                )
            )

        recipes.extend(
            (
                TensorRecipe(
                    object_prefix + "post_attention_norm",
                    _source(source_prefix + "post_attention_layernorm.weight", (4096,)),
                ),
                TensorRecipe(
                    object_prefix + "mlp/gate_up",
                    Concat(
                        (
                            _source(source_prefix + "mlp.gate_proj.weight", (12288, 4096)),
                            _source(source_prefix + "mlp.up_proj.weight", (12288, 4096)),
                        ),
                        0,
                    ),
                ),
                TensorRecipe(
                    object_prefix + "mlp/down",
                    _source(source_prefix + "mlp.down_proj.weight", (4096, 12288)),
                ),
            )
        )

    recipes.extend(
        (
            TensorRecipe(
                "text/final_norm",
                _source("model.language_model.norm.weight", (4096,)),
            ),
            TensorRecipe(
                "text/output_head",
                _source("lm_head.weight", (248320, 4096)),
            ),
        )
    )
    return tuple(recipes)


def _build_draft_head_recipes() -> tuple[TensorRecipe, ...]:
    return (
        TensorRecipe(
            "text/draft_head",
            GatherRows(
                _source("lm_head.weight", (248320, 4096)),
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


def _build_vision_recipes() -> tuple[TensorRecipe, ...]:
    return build_vision_recipes(4096)


SOURCE_RECIPES = _build_text_recipes() + _build_draft_head_recipes() + _build_vision_recipes()
SOURCE_RECIPES_BY_NAME = {recipe.object_name: recipe for recipe in SOURCE_RECIPES}


def _build_mtp_dummy_specs() -> tuple[tuple[str, tuple[int, ...]], ...]:
    return (
        ("mtp/input_projection", (4096, 8192)),
        ("mtp/embedding_norm", (4096,)),
        ("mtp/hidden_norm", (4096,)),
        ("mtp/layer/input_norm", (4096,)),
        ("mtp/layer/attention/query_key_gate_value", (10240, 4096)),
        ("mtp/layer/attention/query_norm", (256,)),
        ("mtp/layer/attention/key_norm", (256,)),
        ("mtp/layer/attention/output", (4096, 4096)),
        ("mtp/layer/post_attention_norm", (4096,)),
        ("mtp/layer/mlp/gate_up", (24576, 4096)),
        ("mtp/layer/mlp/down", (4096, 12288)),
        ("mtp/final_norm", (4096,)),
    )


MTP_DUMMY_SPECS = _build_mtp_dummy_specs()


def validate_recipe_coverage() -> None:
    source_names = tuple(recipe.object_name for recipe in SOURCE_RECIPES)
    source_specs = tuple(
        spec
        for spec in inventory.TENSOR_SPECS
        if not spec.name.startswith("mtp/")
    )
    _validate_recipe_coverage(SOURCE_RECIPES, source_specs)


def source_requirements() -> dict[str, SourceTensor]:
    return _recipe_source_requirements(SOURCE_RECIPES)


def preflight_sources(model_dir: str | Path) -> SourcePreflight:
    with ShardReader(model_dir) as reader:
        return preflight_source_reader(reader, SOURCE_RECIPES)


validate_recipe_coverage()
