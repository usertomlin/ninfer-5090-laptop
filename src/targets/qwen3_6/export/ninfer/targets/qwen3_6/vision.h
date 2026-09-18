#pragma once

#include "artifact/binder.h"
#include "core/tensor.h"

#include <array>
#include <cstddef>

namespace ninfer::artifact {
class MaterializedArtifact;
}

namespace ninfer::targets::qwen3_6 {

// Reference family vision-backbone constants. An exact Variant whose vision tower differs
// (for example a narrower tower with fewer layers) overrides the affected constants in its own
// VisionConfig and instantiates the structures and bindings below with that configuration.
struct VisionBackboneConfig {
    static constexpr int layers              = 27;
    static constexpr int hidden              = 1152;
    static constexpr int intermediate        = 4304;
    static constexpr int heads               = 16;
    static constexpr int head_dim            = hidden / heads;
    static constexpr int patch_dim           = 3 * 2 * 16 * 16;
    static constexpr int merge               = 2;
    static constexpr int merge_unit          = merge * merge;
    static constexpr int merger_hidden       = hidden * merge_unit;
    static constexpr int position_embeddings = 48 * 48;
    static constexpr int rotary_dim          = head_dim;
    static constexpr float rope_theta        = 10'000.0F;
    static constexpr float norm_epsilon      = 1.0e-6F;
    static constexpr float attention_scale   = 0.11785113019775792F;
};

template <class BackboneConfig>
struct VisionLayerPlanT {
    artifact::ObjectHandle qkv;
    artifact::ObjectHandle qkv_bias;
    artifact::ObjectHandle output;
    artifact::ObjectHandle output_bias;
    artifact::ObjectHandle fc1;
    artifact::ObjectHandle fc1_bias;
    artifact::ObjectHandle fc2;
    artifact::ObjectHandle fc2_bias;
    artifact::ObjectHandle norm1_weight;
    artifact::ObjectHandle norm1_bias;
    artifact::ObjectHandle norm2_weight;
    artifact::ObjectHandle norm2_bias;
};

template <class BackboneConfig>
struct VisionBackbonePlanT {
    artifact::ObjectHandle patch_embedding;
    artifact::ObjectHandle patch_embedding_bias;
    artifact::ObjectHandle position_embedding;
    std::array<VisionLayerPlanT<BackboneConfig>, BackboneConfig::layers> layers;
};

template <class BackboneConfig>
struct VisionMergerInputPlanT {
    artifact::ObjectHandle fc1;
    artifact::ObjectHandle fc1_bias;
};

template <class BackboneConfig>
struct VisionMergerNormPlanT {
    artifact::ObjectHandle weight;
    artifact::ObjectHandle bias;
};

template <class BackboneConfig>
struct VisionLayerWeightsT {
    Weight qkv;
    Tensor qkv_bias;
    Weight output;
    Tensor output_bias;
    Weight fc1;
    Tensor fc1_bias;
    Weight fc2;
    Tensor fc2_bias;
    Tensor norm1_weight;
    Tensor norm1_bias;
    Tensor norm2_weight;
    Tensor norm2_bias;
};

template <class BackboneConfig>
struct VisionCommonWeightsT {
    Weight patch_embedding;
    Tensor patch_embedding_bias;
    Tensor position_embedding;
    std::array<VisionLayerWeightsT<BackboneConfig>, BackboneConfig::layers> layers;
    Weight merger_fc1;
    Tensor merger_fc1_bias;
    Tensor merger_norm_weight;
    Tensor merger_norm_bias;
};

template <class BackboneConfig>
struct VisionWeightsT {
    VisionCommonWeightsT<BackboneConfig> common;
    Weight merger_fc2;
    Tensor merger_fc2_bias;
};

using VisionLayerPlan       = VisionLayerPlanT<VisionBackboneConfig>;
using VisionBackbonePlan    = VisionBackbonePlanT<VisionBackboneConfig>;
using VisionMergerInputPlan = VisionMergerInputPlanT<VisionBackboneConfig>;
using VisionMergerNormPlan  = VisionMergerNormPlanT<VisionBackboneConfig>;
using VisionLayerWeights    = VisionLayerWeightsT<VisionBackboneConfig>;
using VisionCommonWeights   = VisionCommonWeightsT<VisionBackboneConfig>;
using VisionWeights         = VisionWeightsT<VisionBackboneConfig>;

// Definitions live in "targets/qwen3_6/impl/vision/bindings_impl.h". The reference configuration
// symbols are published by "targets/qwen3_6/impl/vision/bindings.cpp"; a Variant whose vision
// tower differs includes the implementation header and instantiates its own configuration.
template <class BackboneConfig = VisionBackboneConfig>
[[nodiscard]] VisionBackbonePlanT<BackboneConfig>
bind_vision_backbone(artifact::Binder& binder, artifact::TensorPlacement placement);

template <class BackboneConfig = VisionBackboneConfig>
[[nodiscard]] VisionMergerInputPlanT<BackboneConfig>
bind_vision_merger_input(artifact::Binder& binder, artifact::TensorPlacement placement);

template <class BackboneConfig = VisionBackboneConfig>
[[nodiscard]] VisionMergerNormPlanT<BackboneConfig>
bind_vision_merger_norm(artifact::Binder& binder, artifact::TensorPlacement placement);

template <class BackboneConfig = VisionBackboneConfig>
[[nodiscard]] VisionCommonWeightsT<BackboneConfig>
materialize_vision_common(const artifact::MaterializedArtifact& materialized,
                          const VisionBackbonePlanT<BackboneConfig>& backbone,
                          const VisionMergerInputPlanT<BackboneConfig>& merger_input,
                          const VisionMergerNormPlanT<BackboneConfig>& merger_norm);

} // namespace ninfer::targets::qwen3_6
