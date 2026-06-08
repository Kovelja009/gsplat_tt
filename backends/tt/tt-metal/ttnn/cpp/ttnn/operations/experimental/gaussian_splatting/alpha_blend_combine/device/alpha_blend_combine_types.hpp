// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_combine {

// Phase-2 of intra-tile parallelism: merge per-segment partials produced by
// gaussian_alpha_blend_partial back into the final image via the associative
// Porter-Duff `over` operator (validated host-side in backends/tt/segments.py).
struct CombineParams {
    uint32_t num_tiles;  // output screen tiles (output is (num_tiles*3, 1024) bf16)
    // Per-core slice of the combine plan rows (hash-excluded, like
    // AlphaBlendParams' per-core schedule): core c owns plan rows
    // [per_core_offset[c], per_core_offset[c] + per_core_count[c]).
    std::vector<uint32_t> per_core_offset;
    std::vector<uint32_t> per_core_count;

    static constexpr auto attribute_names = std::forward_as_tuple("num_tiles");
    auto attribute_values() const { return std::forward_as_tuple(num_tiles); }
};

struct CombineInputs {
    ttnn::Tensor partials;  // (num_jobs*4, 1024) bf16 : per-job R,G,B,T tiles
    ttnn::Tensor plan;      // (num_nonempty_tiles, 4) u32 rows [out_tile, first_slot, K, pad]
};

}  // namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_combine
