// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector>

#include "alpha_blend_combine_types.hpp"

#include "ttnn/device_operation.hpp"
#include <tt-metalium/mesh_coord.hpp>

namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_combine {

struct CombineProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id;
        tt::tt_metal::KernelHandle compute_kernel_id;
        tt::tt_metal::KernelHandle writer_kernel_id;
        std::vector<tt::tt_metal::CoreCoord> cores;
    };

    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const CombineParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const CombineInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const CombineParams& operation_attributes,
        const CombineInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_combine
