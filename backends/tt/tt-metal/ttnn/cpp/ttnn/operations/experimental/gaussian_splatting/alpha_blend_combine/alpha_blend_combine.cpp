// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "alpha_blend_combine.hpp"

#include "device/alpha_blend_combine_device_operation.hpp"

namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_combine {

ttnn::Tensor gaussian_alpha_blend_combine(
    const ttnn::Tensor& partials,
    const ttnn::Tensor& plan,
    uint32_t num_tiles,
    const std::vector<uint32_t>& per_core_offset,
    const std::vector<uint32_t>& per_core_count) {
    return ttnn::prim::gaussian_alpha_blend_combine(
        partials, plan, num_tiles, per_core_offset, per_core_count);
}

}  // namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_combine
