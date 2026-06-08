// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "alpha_blend_partial.hpp"

#include "device/alpha_blend_partial_device_operation.hpp"

namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_partial {

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
    const std::vector<uint32_t>& per_core_count) {
    return ttnn::prim::gaussian_alpha_blend_partial(
        packs, px, py, job_table,
        image_height, image_width, num_tiles, num_jobs, per_core_offset, per_core_count);
}

}  // namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_partial
