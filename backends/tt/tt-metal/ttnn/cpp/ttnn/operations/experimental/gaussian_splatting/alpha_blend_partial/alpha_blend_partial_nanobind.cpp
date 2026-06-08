// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "alpha_blend_partial_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "alpha_blend_partial.hpp"

namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_partial::detail {

void bind_gaussian_alpha_blend_partial(nb::module_& mod) {
    ttnn::bind_function<"gaussian_alpha_blend_partial", "ttnn.experimental.">(
        mod,
        R"doc(
            Phase-1 partial for intra-tile parallel 3DGS alpha-blend.

            Composites each segment-job (a contiguous depth-range of a tile's
            Gaussians, from the job_table) into a partial (R,G,B,T) tile, emitting
            a (num_jobs*4, 1024) bf16 partials buffer for the combine op. The
            per-core job schedule (per_core_offset/per_core_count) is excluded
            from the program-cache hash.

            Args:
                packs (ttnn.Tensor): (max_entries, 16) fp32 packed per-Gaussian scalars.
                px (ttnn.Tensor): (num_tiles, 32, 32) bf16 resident x-grid.
                py (ttnn.Tensor): (num_tiles, 32, 32) bf16 resident y-grid.
                job_table (ttnn.Tensor): (num_jobs, 4) u32 [tile_id, gseg_start, gseg_count, partial_slot].
                image_height (int): output height in pixels.
                image_width (int): output width in pixels.
                num_tiles (int): number of 32x32 screen tiles.
                num_jobs (int): number of segment-jobs (partials buffer length / 4).
                per_core_offset (List[int]): per-core start index into the job table.
                per_core_count (List[int]): per-core job count.

            Returns:
                ttnn.Tensor: (num_jobs*4, 1024) bf16 partial (R,G,B,T) tiles.
        )doc",
        &ttnn::operations::experimental::gaussian_splatting::alpha_blend_partial::gaussian_alpha_blend_partial,
        nb::arg("packs").noconvert(),
        nb::arg("px").noconvert(),
        nb::arg("py").noconvert(),
        nb::arg("job_table").noconvert(),
        nb::kw_only(),
        nb::arg("image_height"),
        nb::arg("image_width"),
        nb::arg("num_tiles"),
        nb::arg("num_jobs"),
        nb::arg("per_core_offset"),
        nb::arg("per_core_count"));
}

}  // namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_partial::detail
