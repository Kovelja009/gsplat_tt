// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <vector>

#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend {

// Forward-pass 3DGS alpha-blend. Consumes device-resident SoA inputs and
// returns a (num_tiles, 3, 32, 32) bf16 image tensor. The reader/compute/writer
// kernels are reused from the original tt-metal programming example; this op
// re-homes the host orchestration (CB setup, work-split, runtime args) into a
// ttnn program factory so the device stays open and the kernels stay warm.
ttnn::Tensor gaussian_alpha_blend(
    const ttnn::Tensor& packs,
    const ttnn::Tensor& offsets,
    const ttnn::Tensor& px,
    const ttnn::Tensor& py,
    const ttnn::Tensor& tile_ids,
    uint32_t image_height,
    uint32_t image_width,
    uint32_t num_tiles,
    const std::vector<uint32_t>& per_core_offset,
    const std::vector<uint32_t>& per_core_count);

}  // namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend
