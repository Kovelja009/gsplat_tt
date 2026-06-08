// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <vector>

#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_combine {

// Phase-2 combine: merge per-segment partials (R,G,B,T) produced by
// gaussian_alpha_blend_partial into the final (num_tiles, 3, 32, 32) bf16 image
// via the associative Porter-Duff `over` operator.
ttnn::Tensor gaussian_alpha_blend_combine(
    const ttnn::Tensor& partials,
    const ttnn::Tensor& plan,
    uint32_t num_tiles,
    const std::vector<uint32_t>& per_core_offset,
    const std::vector<uint32_t>& per_core_count);

}  // namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_combine
