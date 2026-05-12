// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include "alpha_blend.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("alpha_blend",      &alpha_blend,
          "Alpha-blend forward rasterization (CUDA, fp32 storage)");
    m.def("alpha_blend_bf16", &alpha_blend_bf16,
          "Alpha-blend forward rasterization (CUDA, bf16 storage + fp32 compute)");
}
