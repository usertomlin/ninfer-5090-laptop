#include <ninfer/targets/qwen3_6/vision.h>

#include "targets/qwen3_6/impl/vision/bindings_impl.h"

namespace ninfer::targets::qwen3_6 {

// Publishes the reference configuration symbols consumed by the 27-layer packages. A Variant
// whose vision tower differs instantiates its own configuration from bindings_impl.h and does
// not depend on these definitions.
template VisionBackbonePlan
bind_vision_backbone<VisionBackboneConfig>(artifact::Binder& binder,
                                           artifact::TensorPlacement placement);

template VisionMergerInputPlan
bind_vision_merger_input<VisionBackboneConfig>(artifact::Binder& binder,
                                               artifact::TensorPlacement placement);

template VisionMergerNormPlan
bind_vision_merger_norm<VisionBackboneConfig>(artifact::Binder& binder,
                                              artifact::TensorPlacement placement);

template VisionCommonWeights materialize_vision_common<VisionBackboneConfig>(
    const artifact::MaterializedArtifact& materialized, const VisionBackbonePlan& backbone,
    const VisionMergerInputPlan& merger_input, const VisionMergerNormPlan& merger_norm);

} // namespace ninfer::targets::qwen3_6
