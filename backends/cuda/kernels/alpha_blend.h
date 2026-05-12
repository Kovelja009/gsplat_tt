// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/extension.h>

// Forward-pass alpha-blend rasterization.
//
// Inputs (all CUDA tensors, contiguous, float32 / int32 as noted):
//   means_2d            (M, 2)         float32
//   conics              (M, 3)         float32  [cov_inv_a, 2*cov_inv_b, cov_inv_c]
//   rgba                (M, 4)         float32  [r, g, b, opacity]
//   sorted_gaussian_ids (P,)           int32
//   tile_ranges         (num_tiles, 2) int32    [start, end)
//   image_height, image_width          int
//
// Returns: image (H, W, 3) float32, CUDA tensor.
torch::Tensor alpha_blend(
    torch::Tensor means_2d,
    torch::Tensor conics,
    torch::Tensor rgba,
    torch::Tensor sorted_gaussian_ids,
    torch::Tensor tile_ranges,
    int64_t image_height,
    int64_t image_width
);
