// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "alpha_blend_partial_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"                          // ttnn::DRAM_MEMORY_CONFIG
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/creation/creation.hpp"  // ttnn::zeros

namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_partial {

void AlphaBlendPartialDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t&, const tensor_args_t& t) {
    // Every input whose buffer address is patched in override_runtime_arguments
    // (the warm-frame path) must be on device, or .buffer()->address() null-derefs.
    TT_FATAL(t.packs.storage_type() == StorageType::DEVICE, "packs must be on device");
    TT_FATAL(t.px.storage_type() == StorageType::DEVICE, "px must be on device");
    TT_FATAL(t.py.storage_type() == StorageType::DEVICE, "py must be on device");
    TT_FATAL(t.job_table.storage_type() == StorageType::DEVICE, "job_table must be on device");
}

void AlphaBlendPartialDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& a, const tensor_args_t& t) {
    validate_on_program_cache_hit(a, t);

    TT_FATAL(t.packs.dtype() == DataType::FLOAT32, "packs must be float32");
    TT_FATAL(t.px.dtype() == DataType::BFLOAT16, "px must be bfloat16");
    TT_FATAL(t.py.dtype() == DataType::BFLOAT16, "py must be bfloat16");
    TT_FATAL(t.job_table.dtype() == DataType::UINT32, "job_table must be uint32");

    TT_FATAL(
        a.per_core_offset.size() == a.per_core_count.size(),
        "per_core_offset ({}) and per_core_count ({}) must have equal length",
        a.per_core_offset.size(),
        a.per_core_count.size());
}

ttnn::TensorSpec AlphaBlendPartialDeviceOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t& /*t*/) {
    // The partial writer emits 4 raw bf16 32x32 tiles (R,G,B,T) per segment-job
    // to DRAM pages 4*slot+{0,1,2,3} with page size 2048 B. The output is shaped
    // (num_jobs*4, 1024): one 2048-byte interleaved-DRAM page per row. The combine
    // op reads these back; the host never reshapes partials directly.
    const ttnn::Shape out_shape({a.num_jobs * 4u, 1024u});
    return TensorSpec(
        out_shape,
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), ttnn::DRAM_MEMORY_CONFIG));
}

ttnn::Tensor AlphaBlendPartialDeviceOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& t) {
    // Every job's 4 slots are written by the partial writer; zero-init is belt-
    // and-suspenders for any slot a degenerate schedule leaves untouched.
    const ttnn::Shape out_shape({a.num_jobs * 4u, 1024u});
    return ttnn::zeros(
        out_shape, DataType::BFLOAT16, Layout::ROW_MAJOR, *t.packs.device(), ttnn::DRAM_MEMORY_CONFIG);
}

}  // namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_partial

namespace ttnn::prim {

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
    namespace ab = ttnn::operations::experimental::gaussian_splatting::alpha_blend_partial;
    return ttnn::device_operation::launch<ab::AlphaBlendPartialDeviceOperation>(
        ab::AlphaBlendPartialParams{image_height, image_width, num_tiles, num_jobs, per_core_offset, per_core_count},
        ab::AlphaBlendPartialInputs{packs, px, py, job_table});
}

}  // namespace ttnn::prim
