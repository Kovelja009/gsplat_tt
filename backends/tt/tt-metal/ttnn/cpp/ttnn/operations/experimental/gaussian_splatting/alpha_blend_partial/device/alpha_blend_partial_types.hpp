// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_partial {

struct AlphaBlendPartialParams {
    uint32_t image_height;
    uint32_t image_width;
    uint32_t num_tiles;   // for px/py resident-grid indexing (per-job tile_id)
    uint32_t num_jobs;    // segment-jobs; output is (num_jobs*4, 1024) partials
    // Per-frame LPT job->core schedule (EXCLUDED from the program-cache hash, as
    // in the single-phase op): per_core_offset/count index into the job table.
    std::vector<uint32_t> per_core_offset;
    std::vector<uint32_t> per_core_count;

    // Structural fields only in the hash; the per-frame schedule is excluded so
    // warm frames at a fixed resolution reuse the cached program.
    static constexpr auto attribute_names =
        std::forward_as_tuple("image_height", "image_width", "num_tiles", "num_jobs");
    auto attribute_values() const {
        return std::forward_as_tuple(image_height, image_width, num_tiles, num_jobs);
    }
};

struct AlphaBlendPartialInputs {
    ttnn::Tensor packs;      // (max_entries, 16) fp32 packed scalars (9 used + pad)
    ttnn::Tensor px;         // (num_tiles, 32, 32) bf16 resident x-grid
    ttnn::Tensor py;         // (num_tiles, 32, 32) bf16 resident y-grid
    ttnn::Tensor job_table;  // (num_jobs, 4) u32 rows [tile_id, gseg_start, gseg_count, partial_slot]
};

}  // namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_partial
