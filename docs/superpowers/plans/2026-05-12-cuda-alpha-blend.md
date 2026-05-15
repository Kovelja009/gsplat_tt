# CUDA Alpha-Blend Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CUDA backend (`backends/cuda/`) implementing `Backend.blend(...)` via a JIT-compiled custom CUDA kernel, so the per-stage benchmark pipeline can target NVIDIA GPUs alongside the existing CPU and Tenstorrent backends.

**Architecture:** One CUDA block per 32×32 screen tile, one thread per pixel, cooperative batched load of sorted Gaussians into shared memory, per-pixel front-to-back compositing with per-thread and warp-wide early termination. Kernel is JIT-compiled via `torch.utils.cpp_extension.load` on first `CudaBackend()` instantiation — no `setup.sh` changes, no required `nvcc` on CPU-only boxes. Backend is registered in `backends.REGISTRY` only when the import succeeds.

**Tech Stack:** Python 3.10+, PyTorch 2.10 (CUDA wheel for hosts running the backend), CUDA 12.x, pybind11 (vendored with torch), ninja, pytest.

**Execution-host note.** Tasks 1, 2, 7, 8, 9 are fully testable on a CPU-only box. Tasks 3–6 require a CUDA-capable host (build + run). On a CUDA host, install requirements-cuda.txt (Task 7) FIRST so torch is CUDA-enabled before Tasks 3–6 execute. The plan groups CPU-testable work upfront so progress is verifiable even when the CUDA host isn't available.

---

## File map

| Path | Action | Responsibility |
|---|---|---|
| `backends/__init__.py` | Modify | Guarded registry entry for `"cuda"` |
| `backends/cuda/__init__.py` | Modify | Update placeholder docstring; no executable imports |
| `backends/cuda/backend.py` | Create | `CudaBackend` class + `_pack_conics` / `_pack_rgba` helpers |
| `backends/cuda/kernels/build.py` | Create | `_load_extension()` — `torch.utils.cpp_extension.load` wrapper, memoised |
| `backends/cuda/kernels/alpha_blend.cu` | Create | `__global__` kernel + host-side entry function |
| `backends/cuda/kernels/alpha_blend.h` | Create | Entry-function declaration |
| `backends/cuda/kernels/pybind.cpp` | Create | `PYBIND11_MODULE` exposing the entry |
| `backends/cuda/README.md` | Modify | Replace planned-status wording |
| `backends/README.md` | Modify | Flip `cuda/` row in "Existing backends" |
| `README.md` | Modify | Flip `cuda` row in the backend table |
| `requirements-cuda.txt` | Create | CUDA torch wheel + ninja, with extra index URL |
| `tests/test_cuda_backend.py` | Create | Registry guard + helper unit tests (CPU-runnable) |
| `tests/test_cuda_kernel.py` | Create | PSNR + perf integration tests (skipped without CUDA) |

---

## Task 1: Registry guard and CudaBackend stub (CPU-testable)

**Files:**
- Modify: `backends/__init__.py:14-24`
- Modify: `backends/cuda/__init__.py`
- Create: `backends/cuda/backend.py`
- Create: `tests/test_cuda_backend.py`

Goal: when the CUDA backend can't be constructed (no torch.cuda, no compiled kernel, etc.), `REGISTRY` simply omits `"cuda"` — never raises at `import backends` time. Falsifiable on the CPU-only dev box.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cuda_backend.py`:

```python
"""Tests for the CUDA backend that do not require a CUDA device.

Covers the registry guard (import safety on CPU-only boxes) and the
pure-torch SoA packing helpers. Kernel-level tests live in
test_cuda_kernel.py and are gated on torch.cuda.is_available().
"""
import importlib

import pytest
import torch


def test_registry_import_is_safe_without_cuda():
    """Importing `backends` must succeed on CPU-only boxes."""
    import backends  # must not raise
    assert "cpu" in backends.REGISTRY
    assert "tt" in backends.REGISTRY
    # "cuda" may or may not be present depending on host; either is fine
    # so long as the import did not blow up.


def test_get_backend_cuda_errors_helpfully_when_unavailable():
    """`get_backend('cuda')` on a CPU-only box raises a clear KeyError."""
    import backends
    if "cuda" in backends.REGISTRY:
        pytest.skip("CUDA backend is registered on this host")
    with pytest.raises(KeyError, match="cuda"):
        backends.get_backend("cuda")


def test_cuda_backend_init_rejects_no_cuda():
    """CudaBackend() raises if no CUDA device is available."""
    if torch.cuda.is_available():
        pytest.skip("CUDA available; this check only fires on CPU-only hosts")
    from backends.cuda.backend import CudaBackend
    with pytest.raises(RuntimeError, match="CUDA"):
        CudaBackend()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source venv/bin/activate
pytest tests/test_cuda_backend.py -v
```

Expected: `ImportError` on `from backends.cuda.backend import CudaBackend` (module does not exist yet). Both `test_get_backend_cuda_errors_helpfully_when_unavailable` and `test_cuda_backend_init_rejects_no_cuda` fail at collection.

- [ ] **Step 3: Create `backends/cuda/backend.py` with the stub**

```python
"""CUDA backend — JIT-compiled alpha-blend kernel.

The custom CUDA kernel is compiled on first `CudaBackend()` instantiation
via torch.utils.cpp_extension.load; importing this module on a CPU-only
box is safe (the kernel is only loaded when the backend is constructed,
not at import time).
"""
from __future__ import annotations

import time

import numpy as np
import torch

from gsplat.backend import Backend


class CudaBackend(Backend):
    """Alpha-blend backend running on an NVIDIA GPU via a custom kernel."""

    def __init__(self, verbose: bool = False):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CudaBackend requires a CUDA-capable device; "
                "torch.cuda.is_available() returned False"
            )
        self.verbose = verbose
        self._device = torch.device("cuda")
        self._ext = None  # populated in Task 4

    def blend(
        self,
        means_2d: torch.Tensor,
        covs_2d: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        sorted_gaussian_ids: torch.Tensor,
        tile_ranges: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> tuple[np.ndarray, dict[str, float]]:
        raise NotImplementedError("blend implemented in Task 4")
```

- [ ] **Step 4: Update `backends/cuda/__init__.py`**

Replace the current placeholder docstring with a description that reflects the new status. The package must still avoid executable imports at top level so `import backends.cuda` is cheap on CPU boxes.

```python
"""CUDA backend — alpha-blend on NVIDIA GPUs.

Implements `gsplat.backend.Backend.blend(...)` via a custom CUDA kernel
that is JIT-compiled on first `CudaBackend()` construction. The kernel
source lives in `kernels/`; the build wrapper in `kernels/build.py`
memoises the loaded extension across `blend(...)` calls.

This package intentionally does NOT eagerly import `backend` — that's
done lazily by `backends.__init__` inside a try/except, so importing
`backends` on a CPU-only box does not require torch.cuda.
"""
```

- [ ] **Step 5: Add the guarded import in `backends/__init__.py`**

Replace lines 20-24 (the REGISTRY definition with the commented placeholder) with:

```python
REGISTRY: dict[str, type[Backend]] = {
    "cpu": CpuBackend,
    "tt":  KernelBackend,
}

# CudaBackend is optional — register only if its module imports cleanly.
# On a CPU-only box, the import itself is harmless (no top-level torch.cuda
# access), but we keep the try/except so a future kernel-import failure
# inside backends/cuda/* doesn't prevent the CPU/TT backends from loading.
try:
    from backends.cuda.backend import CudaBackend
    REGISTRY["cuda"] = CudaBackend
except ImportError:
    pass
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
source venv/bin/activate
pytest tests/test_cuda_backend.py -v
```

Expected: all three tests PASS on a CPU-only box. `test_get_backend_cuda_errors_helpfully_when_unavailable` confirms `"cuda"` is registered (a host with the kernel built) or absent gracefully (CPU box). On a CPU box, `test_cuda_backend_init_rejects_no_cuda` exercises the explicit `RuntimeError`.

Also verify the rest of the suite still passes:

```bash
pytest tests/ -v -k "not perf and not 640"
```

Expected: existing tests in `test_numeric_sanity.py` and the small TT smoke tests are unchanged.

- [ ] **Step 7: Commit**

```bash
git add backends/__init__.py backends/cuda/__init__.py backends/cuda/backend.py tests/test_cuda_backend.py
git commit -m "$(cat <<'EOF'
feat(cuda): registry guard and CudaBackend stub

CudaBackend is registered in backends.REGISTRY only when its module
imports cleanly. The class itself raises a clear RuntimeError on hosts
without torch.cuda, and blend() is a NotImplementedError placeholder
filled in later in the kernel-build phase.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: SoA packing helpers (CPU-testable)

**Files:**
- Modify: `backends/cuda/backend.py`
- Modify: `tests/test_cuda_backend.py`

Goal: implement and unit-test `_pack_conics(covs_2d)` and `_pack_rgba(colors, opacities)` against the math used in `gsplat.rasterization.prepare_kernel_inputs`. These run on the host (input tensors may be on CPU), so the tests don't need CUDA.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cuda_backend.py`:

```python
# ---------------------------------------------------------------------------
# SoA packing helpers — pure host-side, CPU-testable.
# ---------------------------------------------------------------------------


def test_pack_conics_matches_reference():
    """_pack_conics produces (cov_inv_a, 2*cov_inv_b, cov_inv_c) per row."""
    from backends.cuda.backend import _pack_conics

    # Build a known-invertible covariance matrix.
    covs = torch.tensor([
        [[2.0, 0.5], [0.5, 3.0]],
        [[1.0, 0.0], [0.0, 1.0]],
    ], dtype=torch.float32)

    out = _pack_conics(covs)

    assert out.shape == (2, 3)
    assert out.dtype == torch.float32

    # Manual inversion for the first matrix:
    # det = 2*3 - 0.5*0.5 = 6 - 0.25 = 5.75
    # cov_inv = [[3/5.75, -0.5/5.75], [-0.5/5.75, 2/5.75]]
    # Pack stores [a_inv, 2*b_inv, c_inv]
    assert out[0, 0].item() == pytest.approx(3.0 / 5.75, rel=1e-5)
    assert out[0, 1].item() == pytest.approx(2.0 * (-0.5 / 5.75), rel=1e-5)
    assert out[0, 2].item() == pytest.approx(2.0 / 5.75, rel=1e-5)

    # Identity matrix: inverse is itself; b = 0 so 2*b = 0.
    assert out[1, 0].item() == pytest.approx(1.0, rel=1e-5)
    assert out[1, 1].item() == pytest.approx(0.0, abs=1e-6)
    assert out[1, 2].item() == pytest.approx(1.0, rel=1e-5)


def test_pack_conics_clamps_tiny_determinant():
    """Near-singular covariances do not blow up — det floored at 1e-6."""
    from backends.cuda.backend import _pack_conics

    # Very nearly singular: rows are almost linearly dependent.
    covs = torch.tensor([
        [[1.0, 1.0], [1.0, 1.0 + 1e-9]],
    ], dtype=torch.float32)
    out = _pack_conics(covs)
    # No NaN/Inf in the packed result.
    assert torch.isfinite(out).all()


def test_pack_rgba_layout():
    """_pack_rgba(colors, opacities) -> (N, 4) with opacity in last column."""
    from backends.cuda.backend import _pack_rgba

    colors = torch.tensor([
        [0.1, 0.2, 0.3],
        [0.5, 0.6, 0.7],
    ], dtype=torch.float32)
    opacities = torch.tensor([0.9, 0.4], dtype=torch.float32)

    out = _pack_rgba(colors, opacities)
    assert out.shape == (2, 4)
    assert out.dtype == torch.float32
    assert torch.allclose(out[:, :3], colors)
    assert torch.allclose(out[:, 3], opacities)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cuda_backend.py::test_pack_conics_matches_reference tests/test_cuda_backend.py::test_pack_conics_clamps_tiny_determinant tests/test_cuda_backend.py::test_pack_rgba_layout -v
```

Expected: `ImportError: cannot import name '_pack_conics'`.

- [ ] **Step 3: Implement the helpers**

Add to `backends/cuda/backend.py`, between the module docstring/imports and the `CudaBackend` class:

```python
def _pack_conics(covs_2d: torch.Tensor) -> torch.Tensor:
    """Invert each 2x2 covariance and pack as (cov_inv_a, 2*cov_inv_b, cov_inv_c).

    The off-diagonal is pre-multiplied by 2 so the kernel can compute the
    Mahalanobis quadratic form as `a*dx^2 + (2b)*dx*dy + c*dy^2` with a
    single mul per term (matches the TT kernel's CB_SCALARS layout).
    """
    a = covs_2d[:, 0, 0]
    b = covs_2d[:, 0, 1]
    c = covs_2d[:, 1, 1]
    det = torch.clamp(a * c - b * b, min=1e-6)
    out = torch.empty((covs_2d.shape[0], 3), dtype=torch.float32, device=covs_2d.device)
    out[:, 0] = c / det
    out[:, 1] = 2.0 * (-b / det)
    out[:, 2] = a / det
    return out


def _pack_rgba(colors: torch.Tensor, opacities: torch.Tensor) -> torch.Tensor:
    """Concatenate per-Gaussian colors and opacity into an (N, 4) float32 tensor."""
    out = torch.empty((colors.shape[0], 4), dtype=torch.float32, device=colors.device)
    out[:, :3] = colors
    out[:, 3] = opacities
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_cuda_backend.py -v
```

Expected: all six tests in `test_cuda_backend.py` PASS.

- [ ] **Step 5: Commit**

```bash
git add backends/cuda/backend.py tests/test_cuda_backend.py
git commit -m "$(cat <<'EOF'
feat(cuda): SoA packing helpers _pack_conics and _pack_rgba

Mirrors the inverse-covariance and color/opacity packing used by
prepare_kernel_inputs (TT backend's CB_SCALARS layout). Host-side
pure-torch, runnable on CPU-only boxes; cuda-host tests in Task 4 will
upload these tensors to the device.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Kernel build skeleton (stub kernel, CUDA host required)

**Files:**
- Create: `backends/cuda/kernels/alpha_blend.h`
- Create: `backends/cuda/kernels/alpha_blend.cu`
- Create: `backends/cuda/kernels/pybind.cpp`
- Create: `backends/cuda/kernels/build.py`

Goal: a torch extension that compiles cleanly and exposes a callable `alpha_blend(...)` that returns a zero-filled output tensor of the right shape. Lets the host wrapper land before the real kernel exists.

**This task must run on a CUDA-capable host** (Task 7's `requirements-cuda.txt` should already be installed). Verify with `nvcc --version && python -c "import torch; assert torch.cuda.is_available()"`.

- [ ] **Step 1: Create `backends/cuda/kernels/alpha_blend.h`**

```cpp
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
```

- [ ] **Step 2: Create `backends/cuda/kernels/alpha_blend.cu` (stub kernel)**

```cpp
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
```

- [ ] **Step 3: Create `backends/cuda/kernels/pybind.cpp`**

```cpp
// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include "alpha_blend.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("alpha_blend", &alpha_blend, "Alpha-blend forward rasterization (CUDA)");
}
```

- [ ] **Step 4: Create `backends/cuda/kernels/build.py`**

```python
"""JIT loader for the CUDA alpha-blend extension.

Wraps `torch.utils.cpp_extension.load` and memoises the resulting module
so a viewer session with many `blend(...)` calls pays the compile cost
only on the first invocation.
"""
from __future__ import annotations

import os
from typing import Any

_CACHED_EXT: Any = None


def _load_extension() -> Any:
    """Compile (if needed) and return the alpha_blend torch extension.

    Returns the loaded module exposing `alpha_blend(...)`. Cached across
    calls so re-import / repeated `CudaBackend()` doesn't re-pay the
    ~30-60s JIT compile.
    """
    global _CACHED_EXT
    if _CACHED_EXT is not None:
        return _CACHED_EXT

    # Local import keeps `import backends.cuda.kernels.build` cheap on
    # CPU-only boxes — the torch.utils.cpp_extension subpackage probes
    # the compiler on import in some torch versions.
    from torch.utils.cpp_extension import load

    here = os.path.dirname(os.path.abspath(__file__))
    sources = [
        os.path.join(here, "alpha_blend.cu"),
        os.path.join(here, "pybind.cpp"),
    ]
    _CACHED_EXT = load(
        name="gsplat_cuda_alpha_blend",
        sources=sources,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )
    return _CACHED_EXT
```

- [ ] **Step 5: Smoke-compile on a CUDA host**

```bash
source venv/bin/activate          # ensure torch CUDA wheel is installed (Task 7)
python -c "from backends.cuda.kernels.build import _load_extension; ext = _load_extension(); print('loaded:', ext.alpha_blend)"
```

Expected: prints `loaded: <built-in method alpha_blend of PyCapsule object ...>`. First run takes 30–60s; subsequent runs hit the `~/.cache/torch_extensions/` cache and complete in <1s.

If this fails with "Could not find nvcc", install CUDA toolkit / set `CUDA_HOME`. If it fails with "torch not built with CUDA", reinstall via `pip install -r requirements-cuda.txt` (Task 7).

- [ ] **Step 6: Commit**

```bash
git add backends/cuda/kernels/
git commit -m "$(cat <<'EOF'
feat(cuda): JIT build wrapper and stub kernel skeleton

Adds the torch.utils.cpp_extension.load scaffolding (memoised in
_CACHED_EXT) and a stub alpha_blend(...) that returns a zero tensor of
the right shape. Lets the host-side CudaBackend.blend(...) be wired up
and exercised end-to-end before the real kernel body lands in Task 5.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Wire CudaBackend.blend through the stub kernel (CUDA host required)

**Files:**
- Modify: `backends/cuda/backend.py`
- Create: `tests/test_cuda_kernel.py`

Goal: `CudaBackend().blend(...)` calls the (stub) kernel and returns a correctly-shaped, all-zero image with populated sub-timings. Catches integration bugs (dtype mismatch, shape mismatch, sync-before-return) before the real kernel lands.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cuda_kernel.py`:

```python
"""Integration tests for the CUDA alpha-blend backend.

All tests are gated on `torch.cuda.is_available()` — they are skipped on
CPU-only hosts. The CUDA kernel JIT-compiles on the first instantiation
of CudaBackend (~30-60s on a clean cache); a session-scoped fixture
amortises that cost across the whole test module.
"""
import time

import numpy as np
import pytest
import torch

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires a CUDA-capable device",
)


@pytest.fixture(scope="module")
def cuda_backend():
    """Construct a CudaBackend once per test module (amortises JIT compile)."""
    from backends.cuda.backend import CudaBackend
    return CudaBackend()


@cuda_only
def test_cuda_backend_smoke_shape(cuda_backend):
    """blend(...) returns an (H, W, 3) float32 numpy array and the right keys."""
    H, W = 64, 64
    M = 4
    means_2d = torch.zeros((M, 2), dtype=torch.float32)
    covs_2d = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]] * M, dtype=torch.float32
    )
    colors = torch.zeros((M, 3), dtype=torch.float32)
    opacities = torch.zeros((M,), dtype=torch.float32)
    sorted_ids = torch.zeros((0,), dtype=torch.int64)         # no entries
    tile_ranges = torch.zeros(((H // 32) * (W // 32), 2), dtype=torch.int64)

    image, sub = cuda_backend.blend(
        means_2d, covs_2d, colors, opacities,
        sorted_ids, tile_ranges, H, W,
    )

    assert image.shape == (H, W, 3)
    assert image.dtype == np.float32
    # Empty scene → output is all zeros.
    assert np.all(image == 0.0)

    # Sub-timing keys conform to the dotted-key convention.
    assert "upload" in sub
    assert "kernel" in sub
    assert "kernel.device" in sub
    # All sub-timings are non-negative.
    for k, v in sub.items():
        assert v >= 0.0, f"sub timing {k} negative: {v}"
```

- [ ] **Step 2: Run the test on a CUDA host to verify it fails**

```bash
source venv/bin/activate
pytest tests/test_cuda_kernel.py::test_cuda_backend_smoke_shape -v
```

Expected: `NotImplementedError: blend implemented in Task 4`.

- [ ] **Step 3: Implement `CudaBackend.blend`**

Replace the `blend` method in `backends/cuda/backend.py` with:

```python
    def blend(
        self,
        means_2d: torch.Tensor,
        covs_2d: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        sorted_gaussian_ids: torch.Tensor,
        tile_ranges: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> tuple[np.ndarray, dict[str, float]]:
        if self._ext is None:
            from backends.cuda.kernels.build import _load_extension
            self._ext = _load_extension()

        sub: dict[str, float] = {}

        # ---- Stage A: H2D upload + SoA repack ----
        t = time.perf_counter()
        d_means  = means_2d.to(self._device, dtype=torch.float32, non_blocking=True)
        d_conics = _pack_conics(covs_2d).to(self._device, non_blocking=True)
        d_rgba   = _pack_rgba(colors, opacities).to(self._device, non_blocking=True)
        d_ids    = sorted_gaussian_ids.to(self._device, dtype=torch.int32, non_blocking=True)
        d_ranges = tile_ranges.to(self._device, dtype=torch.int32, non_blocking=True)
        sub["upload"] = (time.perf_counter() - t) * 1000.0

        # ---- Stage B: kernel launch + D2H readback ----
        ev_s = torch.cuda.Event(enable_timing=True)
        ev_e = torch.cuda.Event(enable_timing=True)
        t = time.perf_counter()
        ev_s.record()
        d_out = self._ext.alpha_blend(
            d_means, d_conics, d_rgba, d_ids, d_ranges,
            image_height, image_width,
        )
        ev_e.record()
        image = d_out.cpu().numpy()  # implicit sync via D2H memcpy
        sub["kernel"]        = (time.perf_counter() - t) * 1000.0
        sub["kernel.device"] = ev_s.elapsed_time(ev_e)

        return image, sub
```

Also add the missing `close` (optional — base class default is a no-op, but be explicit):

```python
    def close(self) -> None:
        # Nothing to release: the JIT-loaded extension lives in the
        # torch_extensions cache and tensors are freed by GC.
        pass
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_cuda_kernel.py::test_cuda_backend_smoke_shape -v -s
```

Expected: PASS. First run takes ~30-60s for JIT compile; subsequent runs <2s.

Also confirm the CPU-only registry tests still pass:

```bash
pytest tests/test_cuda_backend.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add backends/cuda/backend.py tests/test_cuda_kernel.py
git commit -m "$(cat <<'EOF'
feat(cuda): wire CudaBackend.blend through the JIT-compiled extension

Implements the host-side upload / kernel / readback path with the
dotted-key sub-timing convention (kernel.device nests under kernel in
the benchmark output). Currently routes to the stub kernel from Task 3;
the real alpha-blend body lands in Task 5.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Real alpha-blend kernel (CUDA host required)

**Files:**
- Modify: `backends/cuda/kernels/alpha_blend.cu`
- Modify: `tests/test_cuda_kernel.py`

Goal: replace the stub with the full kernel and prove PSNR ≥ 35 dB against the CPU reference.

- [ ] **Step 1: Write the failing PSNR test**

Append to `tests/test_cuda_kernel.py`:

```python
def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0:
        return 100.0
    return -10.0 * np.log10(mse)


@cuda_only
def test_cuda_psnr_64(cuda_backend):
    """PSNR vs CPU reference >= 35 dB on a 64x64 / 50-Gaussian scene.

    Uses the same seed/scene shape as test_full_scene_psnr in
    test_kernel_integration.py so failures can be compared 1:1 with the
    TT backend's correctness numbers.
    """
    from gsplat.rasterization import (
        project_gaussians, get_tile_assignments, sort_and_bin, alpha_blend,
    )

    torch.manual_seed(42)
    H, W = 64, 64
    N = 50
    means = torch.rand(N, 3) * torch.tensor([2.0, 2.0, 0.0]) + torch.tensor([-1.0, -1.0, 1.0])
    scales = torch.rand(N, 3) * 0.1 + 0.02
    q = torch.randn(N, 4); q = q / q.norm(dim=-1, keepdim=True)
    colors = torch.rand(N, 3)
    opacities = torch.rand(N) * 0.5 + 0.2

    extrinsics = torch.eye(4)
    fx = fy = 40.0
    intrinsics = torch.tensor(
        [[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=torch.float32
    )

    means_2d, covs_2d, depths, radii, valid = project_gaussians(
        means, scales, q, extrinsics, intrinsics, H, W,
    )
    if valid.sum().item() == 0:
        pytest.skip("no visible Gaussians — reroll seed")

    colors_v = colors[valid]
    opacities_v = opacities[valid]

    gids, tids, _ = get_tile_assignments(means_2d, radii, H, W, tile_size=32)
    tiles_x = (W + 31) // 32
    tiles_y = (H + 31) // 32
    sorted_gids, tile_ranges = sort_and_bin(gids, tids, depths, tiles_x, tiles_y)

    cpu_img = alpha_blend(
        means_2d, covs_2d, colors_v, opacities_v,
        sorted_gids, tile_ranges, H, W, tile_size=32,
    ).numpy()

    cuda_img, sub = cuda_backend.blend(
        means_2d, covs_2d, colors_v, opacities_v,
        sorted_gids, tile_ranges, H, W,
    )

    psnr = _psnr(cpu_img, cuda_img)
    print(f"\nPSNR (CUDA vs CPU): {psnr:.2f} dB")
    print(f"sub-timings: {sub}")
    assert psnr >= 35.0, f"PSNR too low: {psnr:.2f} dB (want >= 35)"

    try:
        from skimage.metrics import structural_similarity as ssim
        ssim_val = ssim(cpu_img, cuda_img, channel_axis=2, data_range=1.0)
        print(f"SSIM: {ssim_val:.4f}")
        assert ssim_val >= 0.98, f"SSIM too low: {ssim_val:.4f}"
    except ImportError:
        print("scikit-image not installed; skipping SSIM check")
```

- [ ] **Step 2: Run the PSNR test to verify it fails**

```bash
pytest tests/test_cuda_kernel.py::test_cuda_psnr_64 -v -s
```

Expected: FAIL with `PSNR too low: <very low number> dB` because the stub kernel returns zeros.

- [ ] **Step 3: Replace the stub with the real kernel**

Overwrite `backends/cuda/kernels/alpha_blend.cu` with:

```cpp
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
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

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
        __syncthreads();

        // Warp-wide early-out: if the whole warp has saturated, skip
        // remaining batches. (Whole-block early-out would need
        // __syncthreads_and; warp granularity is the cheap win.)
        if (__all_sync(0xffffffffu, done)) break;
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
```

- [ ] **Step 4: Force a rebuild and run the PSNR test**

`torch.utils.cpp_extension.load` detects source changes via mtime; an explicit cache clear is rarely needed but is safe:

```bash
rm -rf ~/.cache/torch_extensions/*/gsplat_cuda_alpha_blend
pytest tests/test_cuda_kernel.py::test_cuda_psnr_64 -v -s
```

Expected: PASS with PSNR in the 40-90 dB range (CPU vs CUDA, both float32). If PSNR is below 35 dB, investigate (most likely cause: wrong indexing, e.g. forgot the 0.5 pixel-center offset, or `2*b` packed twice).

If `__expf` precision is implicated, swap `--use_fast_math` off in `build.py:extra_cuda_cflags` and replace `__expf` with `expf` in the kernel.

- [ ] **Step 5: Run the rest of the suite to confirm no regression**

```bash
pytest tests/ -v -k "not 640_perf"
```

Expected: all PASS (the 640 perf tests still need the TT kernel built; they are unchanged).

- [ ] **Step 6: Commit**

```bash
git add backends/cuda/kernels/alpha_blend.cu tests/test_cuda_kernel.py
git commit -m "$(cat <<'EOF'
feat(cuda): full alpha-blend kernel + PSNR test

One-block-per-tile / one-thread-per-pixel CUDA kernel matching the CPU
reference and TT kernel numerics (alpha <= 0.99, power <= 0, T < 1e-4
early-out). Cooperative batched load of 256 Gaussians per pass into
shared memory; warp-wide __all_sync early-out at batch boundaries.

Verified against the CPU reference at 64x64 / 50 Gaussians with PSNR
>= 35 dB (matching the TT correctness gate).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: 640×640 perf test + BATCH tuning note (CUDA host required)

**Files:**
- Modify: `tests/test_cuda_kernel.py`

Goal: catch gross perf regressions and print a comparable wall-clock-vs-device-ms breakdown alongside the existing TT 640 baseline.

- [ ] **Step 1: Append the perf test**

Add to `tests/test_cuda_kernel.py`:

```python
@cuda_only
def test_cuda_640_perf(cuda_backend):
    """640x640 / 10K-Gaussian perf — prints wall-vs-device ms, loose ceiling.

    Counterpart to test_640_perf_baseline in test_kernel_integration.py.
    Soft assertion only — fail if kernel.device exceeds 200 ms (any
    consumer NVIDIA GPU from the last 5 years should clear this easily).
    """
    from gsplat.rasterization import (
        project_gaussians, get_tile_assignments, sort_and_bin, alpha_blend,
    )

    torch.manual_seed(42)
    H, W = 640, 640
    N = 10_000
    means = torch.rand(N, 3) * torch.tensor([2.0, 2.0, 2.0]) + torch.tensor([-1.0, -1.0, 1.0])
    scales = torch.rand(N, 3) * 0.1 + 0.02
    q = torch.randn(N, 4); q = q / q.norm(dim=-1, keepdim=True)
    colors = torch.rand(N, 3)
    opacities = torch.rand(N) * 0.5 + 0.2

    extrinsics = torch.eye(4)
    fx = fy = 400.0
    intrinsics = torch.tensor(
        [[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=torch.float32
    )

    means_2d, covs_2d, depths, radii, valid = project_gaussians(
        means, scales, q, extrinsics, intrinsics, H, W,
    )
    colors_v = colors[valid]
    opacities_v = opacities[valid]
    gids, tids, _ = get_tile_assignments(means_2d, radii, H, W, tile_size=32)
    tiles_x = (W + 31) // 32
    tiles_y = (H + 31) // 32
    sorted_gids, tile_ranges = sort_and_bin(gids, tids, depths, tiles_x, tiles_y)

    # Warm up to amortise per-call cuBLAS / context init.
    _ = cuda_backend.blend(
        means_2d, covs_2d, colors_v, opacities_v,
        sorted_gids, tile_ranges, H, W,
    )

    t0 = time.perf_counter()
    cuda_img, sub = cuda_backend.blend(
        means_2d, covs_2d, colors_v, opacities_v,
        sorted_gids, tile_ranges, H, W,
    )
    wall = (time.perf_counter() - t0) * 1000.0

    # Diagnostic: also run the CPU reference for a sanity PSNR.
    cpu_img = alpha_blend(
        means_2d, covs_2d, colors_v, opacities_v,
        sorted_gids, tile_ranges, H, W, tile_size=32,
    ).numpy()
    psnr = _psnr(cpu_img, cuda_img)

    print()
    print("===== CUDA 640x640 perf =====")
    print(f"Scene: H={H} W={W}, N={N} input, {int(valid.sum()):,} visible")
    print(f"Sorted entries: {sorted_gids.numel():,}  Total tiles: {tiles_x * tiles_y}")
    print(f"Wall  (Python perf_counter):        {wall:>7.2f} ms")
    print(f"Upload:                             {sub['upload']:>7.2f} ms")
    print(f"Kernel (host wall + sync via D2H):  {sub['kernel']:>7.2f} ms")
    print(f"Kernel.device (CUDA event):         {sub['kernel.device']:>7.2f} ms")
    print(f"PSNR vs CPU: {psnr:.2f} dB")

    # Loose ceiling to catch regressions, not a fixed target.
    assert sub["kernel.device"] < 200.0, (
        f"kernel.device too slow: {sub['kernel.device']:.1f} ms"
    )
    assert psnr >= 35.0, f"PSNR regressed: {psnr:.2f} dB"
```

- [ ] **Step 2: Run the perf test**

```bash
pytest tests/test_cuda_kernel.py::test_cuda_640_perf -v -s
```

Expected: PASS, printing the per-stage breakdown. Typical numbers on a desktop GPU (RTX 3060+): `kernel.device` in the 5–30 ms range, `upload` 1–3 ms, total wall under 50 ms.

If `kernel.device` is far above 30 ms, consider tuning. Most common knobs (do this in `alpha_blend.cu`, then re-run):
- `BATCH` (currently 256). Try 128 (less shared mem pressure) or 512 (more amortisation of global loads). Higher values may spill shared memory.
- Replace per-thread `done` flag with a per-warp short-circuit only — saves a register, may help occupancy.

Note any deviation from the BATCH=256 default in the commit message.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cuda_kernel.py
git commit -m "$(cat <<'EOF'
test(cuda): 640x640 / 10K-Gaussian perf test

Prints upload / kernel-wall / kernel.device breakdown plus a CPU-reference
PSNR. Loose 200ms kernel.device ceiling catches gross regressions; the
real benchmark numbers are the printed values, captured by the per-session
benchmark logger.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: requirements-cuda.txt (any host)

**Files:**
- Create: `requirements-cuda.txt`

Goal: a one-line `pip install -r requirements-cuda.txt` makes any host CUDA-ready (assuming the box has the matching CUDA runtime libraries installed).

- [ ] **Step 1: Create the file**

```
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.10.0+cu121
ninja>=1.11
```

- [ ] **Step 2: Verify the install on a CUDA host**

```bash
# Optional: in a clean venv on a CUDA host:
pip install -r requirements.txt
pip install -r requirements-cuda.txt
python -c "import torch; assert torch.cuda.is_available(); print(torch.version.cuda)"
```

Expected: prints `12.x`. No verification needed on the CPU-only dev box — the file is just a tracked manifest.

- [ ] **Step 3: Commit**

```bash
git add requirements-cuda.txt
git commit -m "$(cat <<'EOF'
build(cuda): add requirements-cuda.txt with cu121 wheel + ninja

Opt-in install for CUDA hosts: pip install -r requirements-cuda.txt
upgrades torch in place to the cu121 build and pulls ninja (needed by
torch.utils.cpp_extension.load). Base requirements.txt continues to
install the CPU wheel so the TT dev box is unaffected.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Documentation updates (any host)

**Files:**
- Modify: `README.md:18`, `README.md:109-115`
- Modify: `backends/README.md:111`
- Modify: `backends/cuda/README.md`

Goal: the public docs reflect that CUDA is no longer a placeholder.

- [ ] **Step 1: Update the top-level README backend table (`README.md:18`)**

Find the row:

```
| `cuda` | planned  | Placeholder; same wrapper interface, CUDA implementation TBD. |
```

Replace with:

```
| `cuda` | experimental | JIT-compiles on first use via `torch.utils.cpp_extension`. Requires a CUDA-capable GPU and `requirements-cuda.txt` installed. |
```

- [ ] **Step 2: Update the `cuda` directory comment in `README.md` (around line 109)**

Find:

```
    └── cuda/                  # placeholder for the future
```

Replace with:

```
    └── cuda/                  # CUDA backend (kernels JIT-compiled on first use)
```

- [ ] **Step 3: Add an install hint to `README.md`'s setup section**

After the existing `./setup.sh` step, add a note. Find the "Setup" section in `README.md` and append (immediately after the setup.sh paragraph):

```markdown
For the CUDA backend on a CUDA-capable host, after `./setup.sh`:

```bash
source venv/bin/activate
pip install -r requirements-cuda.txt
```

This swaps the CPU torch wheel for the cu121 build in place and installs
`ninja` (used by the JIT compiler). The TT dev box can skip this — `cuda`
will simply not appear in the backend registry.
```

(If the README has no obvious "after setup.sh" hook, place this note in the same section that explains the venv.)

- [ ] **Step 4: Update `backends/README.md:111`**

Find:

```
- **`cuda/`** — placeholder for an upcoming CUDA implementation.
```

Replace with:

```
- **`cuda/`** — NVIDIA GPU via a custom CUDA kernel JIT-compiled by
  `torch.utils.cpp_extension`. Block-per-tile / thread-per-pixel
  alpha-blend, sources in `cuda/kernels/`. Reports
  `blend.{upload, kernel, kernel.device}` sub-timings — the dotted key
  nests `kernel.device` (CUDA event elapsed time) under `kernel` (host
  wall including dispatch + D2H sync).
```

- [ ] **Step 5: Update `backends/cuda/README.md`**

Replace the opening line:

```
# CUDA backend (planned)

Placeholder — implementation pending. See `../README.md` for the wrapper
interface this backend needs to expose.
```

with:

```
# CUDA backend

NVIDIA GPU alpha-blend via a custom CUDA kernel. Compiled JIT on first
`CudaBackend()` instantiation through `torch.utils.cpp_extension.load`
(~30-60s on a clean cache; cached under `~/.cache/torch_extensions/`).

Sources live in `kernels/`:
- `alpha_blend.cu` — `__global__` kernel + host-side entry function
- `alpha_blend.h` — entry-function declaration
- `pybind.cpp` — `PYBIND11_MODULE` wrapper
- `build.py` — memoised `torch.utils.cpp_extension.load` loader
```

Keep the "Timing" section that follows — it documents the exact sub-timing convention the implemented backend uses.

- [ ] **Step 6: Commit**

```bash
git add README.md backends/README.md backends/cuda/README.md
git commit -m "$(cat <<'EOF'
docs(cuda): flip placeholder wording to reflect implemented backend

README backend table, backends/README.md "existing backends" list, and
the CUDA package README updated to describe the JIT-compiled kernel,
sub-timing convention, and requirements-cuda.txt install step.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: End-to-end smoke through the viewer entry point (CUDA host required)

**Files:**
- No new files. Verifies the existing `gsplat` CLI works with `--backend cuda` end-to-end.

Goal: catch any wiring issue between the backend, Pipeline, and viewer that unit tests would miss.

- [ ] **Step 1: Run the CLI**

On a CUDA host with `scenes/luigi.ply` present:

```bash
source venv/bin/activate
pip install -r requirements-cuda.txt   # if not done already
gsplat scenes/luigi.ply --backend cuda --max-resolution 480
```

Expected: viewer comes up; the per-frame benchmark log (in `bench/`, per
`d6caff8 feat(viewer): per-session benchmark logger`) shows non-zero
`blend.upload`, `blend.kernel`, `blend.kernel.device` rows. No crash on
exit.

If the CLI doesn't accept `--backend cuda`, check that the `argparse`
choices in `gsplat/__main__.py` use `backends.REGISTRY.keys()` dynamically
(not a hardcoded `["cpu", "tt"]`). Fix that if needed and add to Task 1's
file map.

- [ ] **Step 2: Capture one frame's timing breakdown to confirm parity with TT**

Run the same scene through TT (`--backend tt`) and compare the recorded
per-stage CSV/markdown. The `project`, `tile_assign`, `sort` rows should
be identical (both backends use the CPU defaults); only the `blend` row
differs.

- [ ] **Step 3: If anything regressed, file a fix in a separate task**

No commit at this step unless code changed.

---

## Self-review

### Spec coverage

- Algorithm details (`a·dx² + 2b·dx·dy + c·dy²`, α≤0.99, T<1e-4, power≤0): Task 5 kernel.
- Launch geometry (32×32 block, BATCH=256, shared memory): Task 5 kernel.
- Per-pixel and warp-wide early termination: Task 5 kernel (`done` flag + `__all_sync`).
- Host wrapper with `upload / kernel / kernel.device` sub-timings: Task 4.
- SoA packing helpers: Task 2.
- File layout (`backends/cuda/{__init__.py, backend.py, kernels/*}`): Tasks 1, 3.
- Guarded registry import: Task 1.
- `requirements-cuda.txt`: Task 7.
- README + `backends/README.md` + `backends/cuda/README.md` updates: Task 8.
- PSNR ≥ 35 dB test, 640×640 perf test: Tasks 5, 6.
- Risks (JIT compile cost, empty tiles, numeric divergence, large M): handled in code (memoised loader; `start==end` short-circuits; fast-math escape hatch in Task 5 step 4; SoA buffers measured in spec).
- Out-of-scope items (no `project` override, no fp16, no autograd, no daemon mode): plan implements none of these — clean.

### Placeholder scan

No `TBD`, `TODO`, `implement later`, or "similar to Task N" hand-waves. Every code-bearing step contains the actual source. Every command has expected output. No undefined symbols across tasks (`_pack_conics`, `_pack_rgba`, `_load_extension`, `alpha_blend` extension function, `CudaBackend`, `_CACHED_EXT` are all defined in the task that first uses them).

### Type / symbol consistency

- `CudaBackend.__init__` raises `RuntimeError` (Task 1) — tested in `test_cuda_backend_init_rejects_no_cuda` (Task 1).
- `_load_extension()` returns the loaded module; `CudaBackend._ext.alpha_blend(...)` signature matches the `PYBIND11_MODULE` definition and the `torch.Tensor alpha_blend(...)` C++ signature (7 args, all match).
- `_pack_conics` returns `(N, 3)` float32 — kernel reads as `float3*` (3 contiguous fp32) — consistent.
- `_pack_rgba` returns `(N, 4)` float32 — kernel reads as `float4*` — consistent.
- `sorted_gaussian_ids` cast to `int32` in `blend(...)` (Task 4) — kernel reads as `int*` — consistent.
- `tile_ranges` cast to `int32`, read as `int2*` — consistent (each row is two int32s in adjacent memory).
- Sub-timing keys (`upload`, `kernel`, `kernel.device`) match across spec, Task 4 wrapper, Task 4 test assertions, Task 6 test, Task 8 docs.

---

**Plan complete.** Everything is ready for execution. Plan saved to `docs/superpowers/plans/2026-05-12-cuda-alpha-blend.md`.
