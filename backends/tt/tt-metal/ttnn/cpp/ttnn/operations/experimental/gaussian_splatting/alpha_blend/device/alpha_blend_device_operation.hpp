// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <variant>
#include <vector>

#include "alpha_blend_types.hpp"
#include "alpha_blend_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend {

struct AlphaBlendDeviceOperation {
    using operation_attributes_t = AlphaBlendParams;
    using tensor_args_t = AlphaBlendInputs;
    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;
    using program_factory_t = std::variant<AlphaBlendProgramFactory>;

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

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
    const std::vector<uint32_t>& per_core_count);

}  // namespace ttnn::prim
