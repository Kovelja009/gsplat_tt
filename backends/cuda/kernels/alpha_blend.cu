// SPDX-License-Identifier: Apache-2.0
//
// Forward-pass alpha-blend rasterization for 3D Gaussian Splatting.
//
// Launch grid: (tiles_x, tiles_y, 1) blocks, (32, 32, 1) threads.
// One block per 32x32 screen tile, one thread per pixel. Each block
// loads its tile's Gaussians from the sorted_gaussian_ids slice in
// batches of BATCH=256 into shared memory, then every thread iterates
// the batch composing its own pixel front-to-back.
//
// Numerics match the CPU reference in gsplat.rasterization.alpha_blend
// and the TT compute kernel: alpha clamped at 0.99, power clamped at 0,
// per-pixel transmittance early termination at T < 1e-4.

#include "alpha_blend.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

constexpr int TILE_SIZE = 32;
constexpr int BATCH     = 256;
constexpr float T_EPS   = 1e-4f;
constexpr float ALPHA_MAX = 0.99f;

__global__ void alpha_blend_kernel(
    const float2* __restrict__ means_2d,         // (M, 2)
    const float3* __restrict__ conics,           // (M, 3) [a, 2b, c]
    const float4* __restrict__ rgba,             // (M, 4) [r, g, b, opacity]
    const int*    __restrict__ sorted_ids,       // (P,)
    const int2*   __restrict__ tile_ranges,      // (num_tiles, 2) [start, end)
    float*        __restrict__ out_image,        // (H, W, 3)
    int           image_height,
    int           image_width,
    int           tiles_x
) {
    __shared__ float2 s_means [BATCH];
    __shared__ float3 s_conics[BATCH];
    __shared__ float4 s_rgba  [BATCH];

    const int tile_id = blockIdx.y * tiles_x + blockIdx.x;
    const int2 range = tile_ranges[tile_id];
    const int  start = range.x;
    const int  end   = range.y;

    const int px_int = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int py_int = blockIdx.y * TILE_SIZE + threadIdx.y;
    const bool inside_image = (px_int < image_width) && (py_int < image_height);
    const float px = float(px_int) + 0.5f;
    const float py = float(py_int) + 0.5f;

    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    float  T     = 1.0f;
    bool   done  = false;  // per-thread early-exit flag

    const int tid_in_block = threadIdx.y * TILE_SIZE + threadIdx.x;  // 0..1023

    for (int g_base = start; g_base < end; g_base += BATCH) {
        const int n = min(BATCH, end - g_base);

        // Cooperative load: first `n` threads each load one full Gaussian.
        if (tid_in_block < n) {
            const int gid = sorted_ids[g_base + tid_in_block];
            s_means [tid_in_block] = means_2d[gid];
            s_conics[tid_in_block] = conics  [gid];
            s_rgba  [tid_in_block] = rgba    [gid];
        }
        __syncthreads();

        // Per-pixel inner loop: composite this batch front-to-back.
        if (!done) {
            for (int j = 0; j < n; j++) {
                const float2 m = s_means [j];
                const float3 q = s_conics[j];
                const float4 cw = s_rgba [j];

                const float dx = px - m.x;
                const float dy = py - m.y;
                float power = -0.5f * (q.x * dx * dx + q.y * dx * dy + q.z * dy * dy);
                if (power > 0.0f) power = 0.0f;     // defensive: PSD covariance => Q >= 0

                const float weight = __expf(power);
                float alpha = cw.w * weight;
                if (alpha > ALPHA_MAX) alpha = ALPHA_MAX;

                const float contrib = alpha * T;
                color.x += contrib * cw.x;
                color.y += contrib * cw.y;
                color.z += contrib * cw.z;
                T *= (1.0f - alpha);

                if (T < T_EPS) { done = true; break; }
            }
        }
        // Whole-block early-out: cooperatively vote — if every thread
        // in the block has saturated, no warp needs to load further
        // batches. __syncthreads_and acts as both the post-loop barrier
        // and the AND-reduction across all threads.
        if (__syncthreads_and(done)) break;
    }

    if (inside_image) {
        const int out_idx = (py_int * image_width + px_int) * 3;
        out_image[out_idx + 0] = color.x;
        out_image[out_idx + 1] = color.y;
        out_image[out_idx + 2] = color.z;
    }
}

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
    TORCH_CHECK(means_2d.dtype() == torch::kFloat32, "means_2d must be float32");
    TORCH_CHECK(conics.dtype()   == torch::kFloat32, "conics must be float32");
    TORCH_CHECK(rgba.dtype()     == torch::kFloat32, "rgba must be float32");
    TORCH_CHECK(sorted_gaussian_ids.dtype() == torch::kInt32,
                "sorted_gaussian_ids must be int32");
    TORCH_CHECK(tile_ranges.dtype() == torch::kInt32, "tile_ranges must be int32");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(means_2d));

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(means_2d.device());
    auto image = torch::zeros({image_height, image_width, 3}, opts);

    const int tiles_x = static_cast<int>((image_width  + TILE_SIZE - 1) / TILE_SIZE);
    const int tiles_y = static_cast<int>((image_height + TILE_SIZE - 1) / TILE_SIZE);

    dim3 grid(tiles_x, tiles_y, 1);
    dim3 block(TILE_SIZE, TILE_SIZE, 1);

    auto stream = at::cuda::getCurrentCUDAStream();
    alpha_blend_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float2*>(means_2d.data_ptr<float>()),
        reinterpret_cast<const float3*>(conics.data_ptr<float>()),
        reinterpret_cast<const float4*>(rgba.data_ptr<float>()),
        sorted_gaussian_ids.data_ptr<int>(),
        reinterpret_cast<const int2*>(tile_ranges.data_ptr<int>()),
        image.data_ptr<float>(),
        static_cast<int>(image_height),
        static_cast<int>(image_width),
        tiles_x
    );
    AT_CUDA_CHECK(cudaGetLastError());
    return image;
}
