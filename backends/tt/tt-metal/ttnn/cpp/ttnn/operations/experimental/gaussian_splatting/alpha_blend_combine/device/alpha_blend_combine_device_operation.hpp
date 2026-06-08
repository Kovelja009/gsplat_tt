// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <variant>
#include <vector>

#include "alpha_blend_combine_types.hpp"
#include "alpha_blend_combine_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_combine {

struct CombineDeviceOperation {
    using operation_attributes_t = CombineParams;
    using tensor_args_t = CombineInputs;
    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;
    using program_factory_t = std::variant<CombineProgramFactory>;

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_combine

namespace ttnn::prim {

ttnn::Tensor gaussian_alpha_blend_combine(
    const ttnn::Tensor& partials,
    const ttnn::Tensor& plan,
    uint32_t num_tiles,
    const std::vector<uint32_t>& per_core_offset,
    const std::vector<uint32_t>& per_core_count);

}  // namespace ttnn::prim
