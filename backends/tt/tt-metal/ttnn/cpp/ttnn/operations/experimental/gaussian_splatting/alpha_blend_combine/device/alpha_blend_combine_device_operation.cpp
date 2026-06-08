// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "alpha_blend_combine_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"                          // ttnn::DRAM_MEMORY_CONFIG
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/creation/creation.hpp"  // ttnn::zeros

namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_combine {

void CombineDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t&, const tensor_args_t& t) {
    TT_FATAL(t.partials.storage_type() == StorageType::DEVICE, "partials must be on device");
    TT_FATAL(t.plan.storage_type() == StorageType::DEVICE, "plan must be on device");
}

void CombineDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& a, const tensor_args_t& t) {
    validate_on_program_cache_hit(a, t);
    TT_FATAL(t.partials.dtype() == DataType::BFLOAT16, "partials must be bfloat16");
    TT_FATAL(t.plan.dtype() == DataType::UINT32, "plan must be uint32");
    TT_FATAL(
        a.per_core_offset.size() == a.per_core_count.size(),
        "per_core_offset ({}) and per_core_count ({}) must have equal length",
        a.per_core_offset.size(),
        a.per_core_count.size());
}

ttnn::TensorSpec CombineDeviceOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t& /*t*/) {
    // Same output contract as the single-phase op: 3 raw bf16 32x32 tiles
    // (R,G,B) per screen tile, one 2048-byte interleaved-DRAM page per row.
    const ttnn::Shape out_shape({a.num_tiles * 3u, 1024u});
    return TensorSpec(
        out_shape,
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), ttnn::DRAM_MEMORY_CONFIG));
}

ttnn::Tensor CombineDeviceOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& t) {
    // Empty tiles have no plan row and are never written; zero so they read as
    // background, matching the single-phase op.
    const ttnn::Shape out_shape({a.num_tiles * 3u, 1024u});
    return ttnn::zeros(
        out_shape, DataType::BFLOAT16, Layout::ROW_MAJOR, *t.partials.device(), ttnn::DRAM_MEMORY_CONFIG);
}

}  // namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_combine

namespace ttnn::prim {

ttnn::Tensor gaussian_alpha_blend_combine(
    const ttnn::Tensor& partials,
    const ttnn::Tensor& plan,
    uint32_t num_tiles,
    const std::vector<uint32_t>& per_core_offset,
    const std::vector<uint32_t>& per_core_count) {
    namespace abc = ttnn::operations::experimental::gaussian_splatting::alpha_blend_combine;
    return ttnn::device_operation::launch<abc::CombineDeviceOperation>(
        abc::CombineParams{num_tiles, per_core_offset, per_core_count},
        abc::CombineInputs{partials, plan});
}

}  // namespace ttnn::prim
