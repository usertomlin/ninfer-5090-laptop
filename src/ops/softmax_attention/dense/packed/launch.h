#pragma once

#include "core/tensor.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Vision towers differ in packed head count and head width per registered model: the 0.8B tower
// packs 12 heads of width 64, and the wider variants pack 16 heads of width 64 (2B/4B) or 72 (27B).
// Every accepted pair is instantiated in launch.cu; this predicate is the single authority for the
// geometries the op admits.
[[nodiscard]] constexpr bool packed_attention_supported_geometry(std::int32_t head_dim,
                                                                 std::int32_t heads) {
    return (head_dim == 64 && (heads == 12 || heads == 16)) || (head_dim == 72 && heads == 16);
}

void packed_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Tensor& cu_seqlens, Tensor* tiles, float scale, Tensor& out,
                             cudaStream_t stream);

std::int32_t packed_attention_uniform_tile(std::int32_t segment_length);

void packed_attention_uniform_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                     std::int32_t segment_length, float scale, Tensor& out,
                                     cudaStream_t stream);

void packed_attention_uniform_launch_with_tile(const Tensor& q, const Tensor& k, const Tensor& v,
                                               std::int32_t segment_length, std::int32_t tile_size,
                                               float scale, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
