// SPDX-License-Identifier: Apache-2.0
//
// STUB implementation — returns a zero-filled output. The full
// alpha-blend kernel lands in Task 5; this exists so the host wrapper,
// pybind11 glue, and JIT-load path can be exercised end-to-end first.

#include "alpha_blend.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

torch::Tensor alpha_blend(
    torch::Tensor means_2d,
    torch::Tensor conics,
    torch::Tensor rgba,
    torch::Tensor sorted_gaussian_ids,
    torch::Tensor tile_ranges,
    int64_t image_height,
    int64_t image_width
) {
    TORCH_CHECK(means_2d.is_cuda(), "means_2d must be CUDA");
    TORCH_CHECK(conics.is_cuda(),   "conics must be CUDA");
    TORCH_CHECK(rgba.is_cuda(),     "rgba must be CUDA");
    TORCH_CHECK(sorted_gaussian_ids.is_cuda(), "sorted_gaussian_ids must be CUDA");
    TORCH_CHECK(tile_ranges.is_cuda(), "tile_ranges must be CUDA");
    TORCH_CHECK(means_2d.dtype() == torch::kFloat32, "means_2d dtype must be float32");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(means_2d));

    auto opts = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(means_2d.device());
    return torch::zeros({image_height, image_width, 3}, opts);
}
