// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "alpha_blend_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "alpha_blend.hpp"

namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend::detail {

void bind_gaussian_alpha_blend(nb::module_& mod) {
    ttnn::bind_function<"gaussian_alpha_blend", "ttnn.experimental.">(
        mod,
        R"doc(
            Forward-pass 3D Gaussian Splatting alpha-blend.

            Reads device-resident SoA inputs (packs/offsets/px/py/tile_ids) and
            returns the rendered image as a (num_tiles, 3, 32, 32) bf16 tensor.
            The per-core tile->core schedule (per_core_offset/per_core_count) is
            an LPT load balance computed host-side; it is excluded from the
            program-cache hash so frames at a fixed resolution reuse the warm
            compiled program.

            Args:
                packs (ttnn.Tensor): (max_entries, 16) fp32 packed per-Gaussian scalars.
                offsets (ttnn.Tensor): (num_tiles + 1,) u32 prefix offsets.
                px (ttnn.Tensor): (num_tiles, 32, 32) bf16 resident x-grid.
                py (ttnn.Tensor): (num_tiles, 32, 32) bf16 resident y-grid.
                tile_ids (ttnn.Tensor): u32 LPT-concatenated per-core tile-id lists.
                image_height (int): output height in pixels.
                image_width (int): output width in pixels.
                num_tiles (int): number of 32x32 screen tiles.
                per_core_offset (List[int]): per-core start index into tile_ids.
                per_core_count (List[int]): per-core tile count.

            Returns:
                ttnn.Tensor: (num_tiles, 3, 32, 32) bf16 rendered image tiles.
        )doc",
        &ttnn::operations::experimental::gaussian_splatting::alpha_blend::gaussian_alpha_blend,
        nb::arg("packs").noconvert(),
        nb::arg("offsets").noconvert(),
        nb::arg("px").noconvert(),
        nb::arg("py").noconvert(),
        nb::arg("tile_ids").noconvert(),
        nb::kw_only(),
        nb::arg("image_height"),
        nb::arg("image_width"),
        nb::arg("num_tiles"),
        nb::arg("per_core_offset"),
        nb::arg("per_core_count"));
}

}  // namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend::detail
