// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector>

#include "alpha_blend_partial_types.hpp"

#include "ttnn/device_operation.hpp"
#include <tt-metalium/mesh_coord.hpp>

namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_partial {

struct AlphaBlendPartialProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id;
        tt::tt_metal::KernelHandle compute_kernel_id;
        tt::tt_metal::KernelHandle writer_kernel_id;
        // Ordered exactly as runtime args were set, so override_runtime_arguments
        // can patch per-core args by the same index.
        std::vector<tt::tt_metal::CoreCoord> cores;
    };

    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const AlphaBlendPartialParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const AlphaBlendPartialInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const AlphaBlendPartialParams& operation_attributes,
        const AlphaBlendPartialInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_partial
