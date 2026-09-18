#pragma once

// Shared vision-backbone binding and materialization source. Include this from an instantiation
// point (the reference configuration in bindings.cpp, or a Variant that re-parameterizes the
// tower) after the selected BackboneConfig is visible.

#include <ninfer/targets/qwen3_6/vision.h>

#include "artifact/materializer.h"
#include "artifact/typed_binding.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>

namespace ninfer::targets::qwen3_6 {

template <class BackboneConfig>
VisionBackbonePlanT<BackboneConfig> bind_vision_backbone(artifact::Binder& binder,
                                                         artifact::TensorPlacement placement) {
    using artifact::NumericFormat;
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };

    VisionBackbonePlanT<BackboneConfig> out;
    out.patch_embedding = bind("vision/patch_embedding", NumericFormat::Q6G64_F16S,
                               {BackboneConfig::hidden, BackboneConfig::patch_dim});
    out.patch_embedding_bias =
        bind("vision/patch_embedding_bias", NumericFormat::BF16, {BackboneConfig::hidden});
    out.position_embedding =
        bind("vision/position_embedding", NumericFormat::BF16,
             {BackboneConfig::position_embeddings, BackboneConfig::hidden});

    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        VisionLayerPlanT<BackboneConfig>& target = out.layers[layer];
        const std::string prefix                 = "vision/layers/" + std::to_string(layer) + "/";
        target.qkv = bind(prefix + "attention/qkv", NumericFormat::Q4G64_F16S,
                          {3 * BackboneConfig::hidden, BackboneConfig::hidden});
        target.qkv_bias = bind(prefix + "attention/qkv_bias", NumericFormat::BF16,
                               {3 * BackboneConfig::hidden});
        target.output = bind(prefix + "attention/output", NumericFormat::Q5G64_F16S,
                             {BackboneConfig::hidden, BackboneConfig::hidden});
        target.output_bias =
            bind(prefix + "attention/output_bias", NumericFormat::BF16, {BackboneConfig::hidden});
        target.fc1 = bind(prefix + "mlp/fc1", NumericFormat::Q4G64_F16S,
                          {BackboneConfig::intermediate, BackboneConfig::hidden});
        target.fc1_bias =
            bind(prefix + "mlp/fc1_bias", NumericFormat::BF16, {BackboneConfig::intermediate});
        target.fc2 = bind(prefix + "mlp/fc2", NumericFormat::Q5G64_F16S,
                          {BackboneConfig::hidden, BackboneConfig::intermediate});
        target.fc2_bias =
            bind(prefix + "mlp/fc2_bias", NumericFormat::BF16, {BackboneConfig::hidden});
        target.norm1_weight =
            bind(prefix + "norm1/weight", NumericFormat::BF16, {BackboneConfig::hidden});
        target.norm1_bias =
            bind(prefix + "norm1/bias", NumericFormat::BF16, {BackboneConfig::hidden});
        target.norm2_weight =
            bind(prefix + "norm2/weight", NumericFormat::BF16, {BackboneConfig::hidden});
        target.norm2_bias =
            bind(prefix + "norm2/bias", NumericFormat::BF16, {BackboneConfig::hidden});
    }
    return out;
}

template <class BackboneConfig>
VisionMergerInputPlanT<BackboneConfig>
bind_vision_merger_input(artifact::Binder& binder, artifact::TensorPlacement placement) {
    using artifact::NumericFormat;
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };
    return VisionMergerInputPlanT<BackboneConfig>{
        .fc1 = bind("vision/merger/fc1", NumericFormat::W8G32_F16S,
                    {BackboneConfig::merger_hidden, BackboneConfig::merger_hidden}),
        .fc1_bias = bind("vision/merger/fc1_bias", NumericFormat::BF16,
                         {BackboneConfig::merger_hidden}),
    };
}

template <class BackboneConfig>
VisionMergerNormPlanT<BackboneConfig>
bind_vision_merger_norm(artifact::Binder& binder, artifact::TensorPlacement placement) {
    using artifact::NumericFormat;
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };
    return VisionMergerNormPlanT<BackboneConfig>{
        .weight =
            bind("vision/merger/norm/weight", NumericFormat::BF16, {BackboneConfig::hidden}),
        .bias = bind("vision/merger/norm/bias", NumericFormat::BF16, {BackboneConfig::hidden}),
    };
}

template <class BackboneConfig>
VisionCommonWeightsT<BackboneConfig>
materialize_vision_common(const artifact::MaterializedArtifact& materialized,
                          const VisionBackbonePlanT<BackboneConfig>& backbone,
                          const VisionMergerInputPlanT<BackboneConfig>& merger_input,
                          const VisionMergerNormPlanT<BackboneConfig>& merger_norm) {
    using artifact::NumericFormat;

    VisionCommonWeightsT<BackboneConfig> out;
    out.patch_embedding = artifact::materialized_weight(
        materialized, backbone.patch_embedding, NumericFormat::Q6G64_F16S, BackboneConfig::hidden,
        BackboneConfig::patch_dim);
    out.patch_embedding_bias =
        artifact::materialized_tensor(materialized, backbone.patch_embedding_bias,
                                      NumericFormat::BF16, {BackboneConfig::hidden});
    out.position_embedding = artifact::materialized_tensor(
        materialized, backbone.position_embedding, NumericFormat::BF16,
        {BackboneConfig::hidden, BackboneConfig::position_embeddings});

    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        const VisionLayerPlanT<BackboneConfig>& source = backbone.layers[layer];
        VisionLayerWeightsT<BackboneConfig>& target    = out.layers[layer];
        target.qkv                                     = artifact::materialized_weight(
            materialized, source.qkv, NumericFormat::Q4G64_F16S, 3 * BackboneConfig::hidden,
            BackboneConfig::hidden);
        target.qkv_bias = artifact::materialized_tensor(
            materialized, source.qkv_bias, NumericFormat::BF16, {3 * BackboneConfig::hidden});
        target.output = artifact::materialized_weight(
            materialized, source.output, NumericFormat::Q5G64_F16S, BackboneConfig::hidden,
            BackboneConfig::hidden);
        target.output_bias = artifact::materialized_tensor(
            materialized, source.output_bias, NumericFormat::BF16, {BackboneConfig::hidden});
        target.fc1 = artifact::materialized_weight(
            materialized, source.fc1, NumericFormat::Q4G64_F16S, BackboneConfig::intermediate,
            BackboneConfig::hidden);
        target.fc1_bias =
            artifact::materialized_tensor(materialized, source.fc1_bias, NumericFormat::BF16,
                                          {BackboneConfig::intermediate});
        target.fc2 = artifact::materialized_weight(
            materialized, source.fc2, NumericFormat::Q5G64_F16S, BackboneConfig::hidden,
            BackboneConfig::intermediate);
        target.fc2_bias = artifact::materialized_tensor(
            materialized, source.fc2_bias, NumericFormat::BF16, {BackboneConfig::hidden});
        target.norm1_weight = artifact::materialized_tensor(
            materialized, source.norm1_weight, NumericFormat::BF16, {BackboneConfig::hidden});
        target.norm1_bias = artifact::materialized_tensor(
            materialized, source.norm1_bias, NumericFormat::BF16, {BackboneConfig::hidden});
        target.norm2_weight = artifact::materialized_tensor(
            materialized, source.norm2_weight, NumericFormat::BF16, {BackboneConfig::hidden});
        target.norm2_bias = artifact::materialized_tensor(
            materialized, source.norm2_bias, NumericFormat::BF16, {BackboneConfig::hidden});
    }

    out.merger_fc1 = artifact::materialized_weight(
        materialized, merger_input.fc1, NumericFormat::W8G32_F16S, BackboneConfig::merger_hidden,
        BackboneConfig::merger_hidden);
    out.merger_fc1_bias =
        artifact::materialized_tensor(materialized, merger_input.fc1_bias, NumericFormat::BF16,
                                      {BackboneConfig::merger_hidden});
    out.merger_norm_weight = artifact::materialized_tensor(
        materialized, merger_norm.weight, NumericFormat::BF16, {BackboneConfig::hidden});
    out.merger_norm_bias = artifact::materialized_tensor(
        materialized, merger_norm.bias, NumericFormat::BF16, {BackboneConfig::hidden});
    return out;
}

} // namespace ninfer::targets::qwen3_6
