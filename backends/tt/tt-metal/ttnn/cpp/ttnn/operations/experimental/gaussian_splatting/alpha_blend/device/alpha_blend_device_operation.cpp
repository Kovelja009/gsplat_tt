// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "alpha_blend_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend {

void AlphaBlendDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t&, const tensor_args_t& t) {
    TT_FATAL(t.packs.storage_type() == StorageType::DEVICE, "packs must be on device");
    TT_FATAL(t.tile_ids.storage_type() == StorageType::DEVICE, "tile_ids must be on device");
}

void AlphaBlendDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& a, const tensor_args_t& t) {
    validate_on_program_cache_hit(a, t);

    TT_FATAL(t.offsets.storage_type() == StorageType::DEVICE, "offsets must be on device");
    TT_FATAL(t.px.storage_type() == StorageType::DEVICE, "px must be on device");
    TT_FATAL(t.py.storage_type() == StorageType::DEVICE, "py must be on device");

    TT_FATAL(t.packs.dtype() == DataType::FLOAT32, "packs must be float32");
    TT_FATAL(t.offsets.dtype() == DataType::UINT32, "offsets must be uint32");
    TT_FATAL(t.px.dtype() == DataType::BFLOAT16, "px must be bfloat16");
    TT_FATAL(t.py.dtype() == DataType::BFLOAT16, "py must be bfloat16");
    TT_FATAL(t.tile_ids.dtype() == DataType::UINT32, "tile_ids must be uint32");

    TT_FATAL(
        a.per_core_offset.size() == a.per_core_count.size(),
        "per_core_offset ({}) and per_core_count ({}) must have equal length",
        a.per_core_offset.size(),
        a.per_core_count.size());
}

ttnn::TensorSpec AlphaBlendDeviceOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t& t) {
    // Output is tile-major: (num_tiles, 3 channels, 32, 32) bf16.
    const ttnn::Shape out_shape({a.num_tiles, 3u, 32u, 32u});
    return TensorSpec(
        out_shape,
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::TILE), t.packs.memory_config()));
}

ttnn::Tensor AlphaBlendDeviceOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& t) {
    // NOTE: LPT filtering skips empty tiles, so the writer never touches their
    // output slots — they must read as background (zero). Phase 2 / Task 2.5
    // confirms create_device_tensor zero-fills, or replaces this with an
    // explicit zero-init if it does not.
    const auto spec = compute_output_specs(a, t);
    return tt::tt_metal::create_device_tensor(spec, t.packs.device());
}

}  // namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend

namespace ttnn::prim {

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
    namespace ab = ttnn::operations::experimental::gaussian_splatting::alpha_blend;
    return ttnn::device_operation::launch<ab::AlphaBlendDeviceOperation>(
        ab::AlphaBlendParams{image_height, image_width, num_tiles, per_core_offset, per_core_count},
        ab::AlphaBlendInputs{packs, offsets, px, py, tile_ids});
}

}  // namespace ttnn::prim
