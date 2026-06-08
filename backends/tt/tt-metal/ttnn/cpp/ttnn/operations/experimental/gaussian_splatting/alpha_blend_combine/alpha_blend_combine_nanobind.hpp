// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_combine::detail {

void bind_gaussian_alpha_blend_combine(::nanobind::module_& mod);

}  // namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_combine::detail
