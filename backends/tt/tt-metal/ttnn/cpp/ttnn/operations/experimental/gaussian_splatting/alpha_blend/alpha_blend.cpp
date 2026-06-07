// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "alpha_blend.hpp"

#include "device/alpha_blend_device_operation.hpp"

namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend {

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
    const std::vector<uint32_t>& per_core_count) {
    return ttnn::prim::gaussian_alpha_blend(
        packs, offsets, px, py, tile_ids,
        image_height, image_width, num_tiles, per_core_offset, per_core_count);
}

}  // namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend
