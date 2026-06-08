// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <vector>

#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_partial {

// Phase-1 partial of intra-tile parallelism. Composites each segment-job (a
// contiguous depth-range of a tile's Gaussians) into a partial (R,G,B,T) tile,
// emitting a (num_jobs*4, 1024) bf16 partials buffer that the combine op merges.
ttnn::Tensor gaussian_alpha_blend_partial(
    const ttnn::Tensor& packs,
    const ttnn::Tensor& px,
    const ttnn::Tensor& py,
    const ttnn::Tensor& job_table,
    uint32_t image_height,
    uint32_t image_width,
    uint32_t num_tiles,
    uint32_t num_jobs,
    const std::vector<uint32_t>& per_core_offset,
    const std::vector<uint32_t>& per_core_count);

}  // namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_partial
