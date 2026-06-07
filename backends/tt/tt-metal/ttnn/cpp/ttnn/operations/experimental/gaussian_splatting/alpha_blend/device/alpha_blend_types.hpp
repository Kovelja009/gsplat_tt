// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend {

struct AlphaBlendParams {
    uint32_t image_height;
    uint32_t image_width;
    uint32_t num_tiles;
    // Per-frame LPT (longest-processing-time) tile->core schedule. These are
    // EXCLUDED from the program-cache hash (see attribute_names /
    // attribute_values below): they change every frame but never change the
    // compiled program, only its runtime args. override_runtime_arguments
    // applies them on a cache hit.
    std::vector<uint32_t> per_core_offset;
    std::vector<uint32_t> per_core_count;

    // Only the structural fields participate in the program-cache hash. The
    // per-frame schedule above is intentionally omitted so warm frames at a
    // fixed resolution reuse the cached program (mirrors the seed-exclusion
    // pattern in BernoulliDeviceOperation).
    static constexpr auto attribute_names = std::forward_as_tuple("image_height", "image_width", "num_tiles");
    auto attribute_values() const { return std::forward_as_tuple(image_height, image_width, num_tiles); }
};

struct AlphaBlendInputs {
    ttnn::Tensor packs;     // (max_entries, 16) fp32 packed scalars (9 used + pad)
    ttnn::Tensor offsets;   // (num_tiles + 1,) u32 prefix offsets
    ttnn::Tensor px;        // (num_tiles, 32, 32) bf16 resident x-grid
    ttnn::Tensor py;        // (num_tiles, 32, 32) bf16 resident y-grid
    ttnn::Tensor tile_ids;  // (>= num_tiles,) u32 LPT-concatenated tile lists
};

}  // namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend
