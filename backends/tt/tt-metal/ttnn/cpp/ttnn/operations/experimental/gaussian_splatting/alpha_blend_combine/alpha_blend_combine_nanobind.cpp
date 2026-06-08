// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "alpha_blend_combine_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "alpha_blend_combine.hpp"

namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_combine::detail {

void bind_gaussian_alpha_blend_combine(nb::module_& mod) {
    ttnn::bind_function<"gaussian_alpha_blend_combine", "ttnn.experimental.">(
        mod,
        R"doc(
            Phase-2 combine for intra-tile parallel 3DGS alpha-blend.

            Merges per-segment partials (R,G,B,T) produced by
            gaussian_alpha_blend_partial into the final image via the
            associative Porter-Duff `over` operator. The per-core combine
            schedule (per_core_offset/per_core_count, indexing the plan rows)
            is excluded from the program-cache hash.

            Args:
                partials (ttnn.Tensor): (num_jobs*4, 1024) bf16, per-job R,G,B,T tiles.
                plan (ttnn.Tensor): (num_nonempty_tiles, 4) u32 [out_tile, first_slot, K, pad].
                num_tiles (int): number of 32x32 output screen tiles.
                per_core_offset (List[int]): per-core start index into plan rows.
                per_core_count (List[int]): per-core plan-row count.

            Returns:
                ttnn.Tensor: (num_tiles, 3, 32, 32) bf16 rendered image tiles.
        )doc",
        &ttnn::operations::experimental::gaussian_splatting::alpha_blend_combine::gaussian_alpha_blend_combine,
        nb::arg("partials").noconvert(),
        nb::arg("plan").noconvert(),
        nb::kw_only(),
        nb::arg("num_tiles"),
        nb::arg("per_core_offset"),
        nb::arg("per_core_count"));
}

}  // namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_combine::detail
