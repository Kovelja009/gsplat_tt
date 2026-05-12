// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include "alpha_blend.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("alpha_blend", &alpha_blend, "Alpha-blend forward rasterization (CUDA)");
}
