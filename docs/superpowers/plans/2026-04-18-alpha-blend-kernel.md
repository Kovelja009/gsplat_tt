# Alpha-Blend tt-metal Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a forward-pass 3D Gaussian Splatting alpha-blend kernel on Tenstorrent Wormhole via tt-metal, replacing the CPU `alpha_blend` function in [rasterization.py](../../../rasterization.py). Deliver v1b (single-core, full scene, ≥35 dB PSNR) as the thesis contract, with v1c (multi-core, ≥15× speedup) as stretch.

**Architecture:** Host pre-gathers per-tile Gaussian attributes into a flat `(N, 9)` fp32 scalar array; device kernel streams scalars + pixel-coord tiles, runs SFPU scalar-unary + addcmul ops per Gaussian, spills running state (RGB + T + sat_mask) to L1 CBs between iterations, uses sentinel-mask saturation instead of loop-break early termination. Compute config: `HiFi3 + fp32_dest_acc_en=true + math_approx_mode=false`. Code lives in `tt-metal/tt_metal/programming_examples/gaussian_splatting/` (git-tracked via `.gitignore` exception).

**Tech Stack:** tt-metal (C++), CMake, Ninja, Clang 20, Python 3.12, NumPy, PyTorch, scikit-image (for SSIM), pybind11 (deferred post-thesis).

**Reference spec:** [docs/superpowers/specs/2026-04-18-alpha-blend-kernel-design.md](../specs/2026-04-18-alpha-blend-kernel-design.md)

---

## File Structure

### New files

| Path | Purpose |
|---|---|
| `tests/test_numeric_sanity.py` | Pytest for NumPy/bf16-simulated reference kernel |
| `scripts/numeric_sanity.py` | NumPy fp32 + bf16-sim port of the compute kernel |
| `scripts/dump_kernel_inputs.py` | Dump `.npy` fixtures for the C++ harness |
| `scripts/run_kernel_harness.sh` | Build + launch standalone harness |
| `tests/test_kernel_integration.py` | PSNR/SSIM vs CPU reference via harness |
| `tt-metal/tt_metal/programming_examples/gaussian_splatting/CMakeLists.txt` | Build hook |
| `tt-metal/tt_metal/programming_examples/gaussian_splatting/alpha_blend.cpp` | Host driver |
| `tt-metal/tt_metal/programming_examples/gaussian_splatting/alpha_blend_host.h` | Shared constants |
| `tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/dataflow/reader_alpha_blend.cpp` | Reader kernel |
| `tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/dataflow/writer_alpha_blend.cpp` | Writer kernel |
| `tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp` | Compute kernel |

### Modified files

| Path | Change |
|---|---|
| `.gitignore` | Add exception for `tt-metal/tt_metal/programming_examples/gaussian_splatting/` |
| `rasterization.py` | Add `prepare_kernel_inputs()` function |
| `viewer.py` | Add opt-in kernel backend (Phase 5) |

---

## Phase 0 — Numeric Sanity + Environment

### Task 0.1: Write numeric sanity reference kernel (NumPy)

Prove the math pipeline is correct in NumPy *before* touching the device. Produces golden fp32 and bf16-simulated outputs for validation.

**Files:**
- Create: `scripts/numeric_sanity.py`

- [ ] **Step 1: Create `scripts/numeric_sanity.py` with the reference function**

```python
"""Numeric sanity reference for the alpha-blend kernel.

This is a NumPy port of the on-device compute kernel math.
Runs in fp32 (golden) or bf16-simulated (precision truth).
Used to validate algorithm before device implementation and to
unit-test each math step independently.
"""
import numpy as np


def _as_bf16(x: np.ndarray) -> np.ndarray:
    """Simulate bf16 by truncating mantissa of fp32 to 7 bits."""
    u = x.view(np.uint32) & 0xFFFF0000
    return u.view(np.float32)


def alpha_blend_reference(
    attribute_packs: np.ndarray,      # (N, 9) fp32
    tile_offsets: np.ndarray,          # (num_tiles + 1,) uint32
    px_tiles: np.ndarray,              # (num_tiles, 32, 32) fp32
    py_tiles: np.ndarray,              # (num_tiles, 32, 32) fp32
    image_h: int,
    image_w: int,
    simulate_bf16: bool = False,
) -> np.ndarray:
    """Run the compute kernel math in NumPy.

    attribute_packs per-Gaussian layout:
        [mean_x, mean_y, cov_a, two_cov_b, cov_c, R, G, B, opacity]
    where two_cov_b = 2 * cov_b (host precompute).
    """
    tiles_x = (image_w + 31) // 32
    tiles_y = (image_h + 31) // 32
    num_tiles = tiles_y * tiles_x
    assert tile_offsets.shape == (num_tiles + 1,)
    assert px_tiles.shape == (num_tiles, 32, 32)
    assert py_tiles.shape == (num_tiles, 32, 32)

    output = np.zeros((tiles_y * 32, tiles_x * 32, 3), dtype=np.float32)
    cast = _as_bf16 if simulate_bf16 else (lambda x: x)

    for tile_id in range(num_tiles):
        ty = tile_id // tiles_x
        tx = tile_id % tiles_x
        g_start = int(tile_offsets[tile_id])
        g_end = int(tile_offsets[tile_id + 1])
        if g_start == g_end:
            continue

        px = px_tiles[tile_id]
        py = py_tiles[tile_id]
        color_r = np.zeros((32, 32), dtype=np.float32)
        color_g = np.zeros((32, 32), dtype=np.float32)
        color_b = np.zeros((32, 32), dtype=np.float32)
        T = np.ones((32, 32), dtype=np.float32)
        sat_mask = np.ones((32, 32), dtype=np.float32)

        for g_idx in range(g_start, g_end):
            g = g_idx - g_start  # local index in this tile
            if g > 0 and (g & 15) == 0:
                sat_mask = (T >= 1e-4).astype(np.float32)

            mean_x, mean_y, cov_a, two_cov_b, cov_c, R, G, B, opacity = attribute_packs[g_idx]

            dx = cast(px - mean_x)
            dy = cast(py - mean_y)
            dx2 = cast(dx * dx)
            dy2 = cast(dy * dy)
            dxdy = cast(dx * dy)
            Q = cast(cov_a * dx2 + two_cov_b * dxdy + cov_c * dy2)
            power = cast(-0.5 * Q)
            power = cast(np.minimum(power, 0.0))
            weight = cast(np.exp(power))
            alpha = cast(np.minimum(0.99, opacity * weight))
            contrib = cast(alpha * T)

            color_r = cast(color_r + R * contrib * sat_mask)
            color_g = cast(color_g + G * contrib * sat_mask)
            color_b = cast(color_b + B * contrib * sat_mask)
            T = cast(T * (1.0 - alpha) * sat_mask)

        py_start = ty * 32
        px_start = tx * 32
        output[py_start:py_start + 32, px_start:px_start + 32, 0] = color_r
        output[py_start:py_start + 32, px_start:px_start + 32, 1] = color_g
        output[py_start:py_start + 32, px_start:px_start + 32, 2] = color_b

    return output[:image_h, :image_w, :]
```

- [ ] **Step 2: Verify it runs on a trivial input**

Run:
```bash
cd /localdev/vkovinic/gsplat_tt
source venv/bin/activate
python -c "
import numpy as np
from scripts.numeric_sanity import alpha_blend_reference
attribute_packs = np.array([[16.5, 16.5, 100.0, 0.0, 100.0, 1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
tile_offsets = np.array([0, 1], dtype=np.uint32)
px_tiles = np.arange(32)[None, None, :].repeat(32, axis=1).astype(np.float32) + 0.5
py_tiles = np.arange(32)[None, :, None].repeat(32, axis=2).astype(np.float32) + 0.5
out = alpha_blend_reference(attribute_packs, tile_offsets, px_tiles, py_tiles, 32, 32)
print('Center pixel RGB:', out[16, 16])
assert abs(out[16, 16, 0] - 1.0) < 0.01, 'Red center failed'
assert out[16, 16, 1] < 0.01, 'Green should be 0'
print('OK')
"
```
Expected output: `Center pixel RGB: [...]` then `OK`.

- [ ] **Step 3: Commit**

```bash
git add scripts/numeric_sanity.py
git commit -m "add numeric sanity reference for alpha-blend kernel"
```

---

### Task 0.2: Add unit tests for the reference kernel

Test each math step: single Gaussian, two-Gaussian blend, saturation, bf16 vs fp32.

**Files:**
- Create: `tests/test_numeric_sanity.py`

- [ ] **Step 1: Write the test file**

```python
"""Unit tests for the NumPy reference alpha-blend."""
import numpy as np
import pytest
from scripts.numeric_sanity import alpha_blend_reference


def _make_px_py():
    px = np.arange(32)[None, None, :].repeat(32, axis=1).astype(np.float32) + 0.5
    py = np.arange(32)[None, :, None].repeat(32, axis=2).astype(np.float32) + 0.5
    return px, py


def test_single_gaussian_centered():
    """T0.4 equivalent: single Gaussian at pixel (16,16) with sharp covariance."""
    px, py = _make_px_py()
    packs = np.array([[16.5, 16.5, 100.0, 0.0, 100.0, 1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    offsets = np.array([0, 1], dtype=np.uint32)
    out = alpha_blend_reference(packs, offsets, px, py, 32, 32)
    assert abs(out[16, 16, 0] - 1.0) < 0.01
    assert out[16, 16, 1] < 0.01 and out[16, 16, 2] < 0.01
    # Neighbors should be ~0 due to exp(-50)
    assert out[20, 20, 0] < 1e-3


def test_two_gaussian_alpha_blend():
    """T0.5: two overlapping Gaussians at same pixel, red in front."""
    px, py = _make_px_py()
    packs = np.array([
        [16.5, 16.5, 100.0, 0.0, 100.0, 1.0, 0.0, 0.0, 0.5],  # red front
        [16.5, 16.5, 100.0, 0.0, 100.0, 0.0, 0.0, 1.0, 0.5],  # blue back
    ], dtype=np.float32)
    offsets = np.array([0, 2], dtype=np.uint32)
    out = alpha_blend_reference(packs, offsets, px, py, 32, 32)
    # Red at center: 0.5, Blue at center: 0.5 * 0.5 = 0.25
    assert abs(out[16, 16, 0] - 0.5) < 0.01
    assert abs(out[16, 16, 2] - 0.25) < 0.01


def test_saturation():
    """T0.6: 50 opaque Gaussians stacked at one pixel; sat_mask should kick in."""
    px, py = _make_px_py()
    packs = np.tile(
        np.array([[16.5, 16.5, 100.0, 0.0, 100.0, 0.5, 0.5, 0.5, 0.99]], dtype=np.float32),
        (50, 1),
    )
    offsets = np.array([0, 50], dtype=np.uint32)
    out = alpha_blend_reference(packs, offsets, px, py, 32, 32)
    # With opacity=0.99, T drops below 1e-4 after ~10 Gaussians; later ones contribute ~0
    # First Gaussian contributes 0.5 * 0.99 * 0.5 (color * alpha * T_prev=1)
    # Color converges to color*alpha / (1 - (1-alpha)) = (0.5*0.99)/0.99 = 0.5
    assert abs(out[16, 16, 0] - 0.5) < 0.05


def test_bf16_vs_fp32_identity_case():
    """bf16 simulation should match fp32 for simple cases."""
    px, py = _make_px_py()
    packs = np.array([[16.5, 16.5, 100.0, 0.0, 100.0, 1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    offsets = np.array([0, 1], dtype=np.uint32)
    out_fp32 = alpha_blend_reference(packs, offsets, px, py, 32, 32, simulate_bf16=False)
    out_bf16 = alpha_blend_reference(packs, offsets, px, py, 32, 32, simulate_bf16=True)
    # bf16 should be close enough
    assert np.max(np.abs(out_fp32 - out_bf16)) < 0.02


def test_empty_tile():
    """Tile with no Gaussians should return zeros."""
    px, py = _make_px_py()
    packs = np.zeros((0, 9), dtype=np.float32)
    offsets = np.array([0, 0], dtype=np.uint32)
    out = alpha_blend_reference(packs, offsets, px, py, 32, 32)
    assert np.allclose(out, 0.0)
```

- [ ] **Step 2: Run the tests, expect all to pass**

Run:
```bash
cd /localdev/vkovinic/gsplat_tt
source venv/bin/activate
pytest tests/test_numeric_sanity.py -v
```
Expected: 5 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_numeric_sanity.py
git commit -m "add unit tests for numeric sanity reference"
```

---

### Task 0.3: Add `prepare_kernel_inputs` to `rasterization.py`

Converts the Python pipeline's intermediates into the kernel's `attribute_packs` + `tile_offsets` + `px_tiles` + `py_tiles` layout.

**Files:**
- Modify: `rasterization.py` (append new function)

- [ ] **Step 1: Open `rasterization.py` and append the function**

Edit `rasterization.py`, add at the end:

```python
def prepare_kernel_inputs(
    means_2d: torch.Tensor,
    covs_2d: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    sorted_gaussian_ids: torch.Tensor,
    tile_ranges: torch.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack per-tile Gaussian attributes for the tt-metal kernel.

    Produces:
      attribute_packs: (N_entries, 9) fp32, per row:
          [mean_x, mean_y, cov_inv_a, 2*cov_inv_b, cov_inv_c, R, G, B, opacity]
      tile_offsets: (num_tiles + 1,) uint32, cumulative prefix sum.
      px_tiles, py_tiles: (num_tiles, 32, 32) fp32, global screen coords.
    """
    tiles_x = (image_width + 31) // 32
    tiles_y = (image_height + 31) // 32
    num_tiles = tiles_x * tiles_y

    # Invert covariances (same math as alpha_blend)
    a = covs_2d[:, 0, 0]
    b = covs_2d[:, 0, 1]
    c = covs_2d[:, 1, 1]
    det = torch.clamp(a * c - b * b, min=1e-6)
    cov_inv_a = (c / det).numpy()
    cov_inv_b = (-b / det).numpy()
    cov_inv_c = (a / det).numpy()

    means_np = means_2d.numpy()
    colors_np = colors.numpy()
    opacities_np = opacities.numpy()
    gids_np = sorted_gaussian_ids.numpy()
    ranges_np = tile_ranges.numpy()

    # Flatten (tile_id, g_idx) pairs and build attribute_packs in sorted order
    n_entries = ranges_np[:, 1].sum() - ranges_np[:, 0].sum() if ranges_np.size else 0
    total_entries = int((ranges_np[:, 1] - ranges_np[:, 0]).sum())

    attribute_packs = np.zeros((total_entries, 9), dtype=np.float32)
    tile_offsets = np.zeros(num_tiles + 1, dtype=np.uint32)

    write_pos = 0
    for tile_id in range(num_tiles):
        start, end = int(ranges_np[tile_id, 0]), int(ranges_np[tile_id, 1])
        tile_offsets[tile_id] = write_pos
        for idx in range(start, end):
            g = gids_np[idx]
            attribute_packs[write_pos] = [
                means_np[g, 0], means_np[g, 1],
                cov_inv_a[g], 2.0 * cov_inv_b[g], cov_inv_c[g],
                colors_np[g, 0], colors_np[g, 1], colors_np[g, 2],
                opacities_np[g],
            ]
            write_pos += 1
    tile_offsets[num_tiles] = write_pos

    # Build px/py tiles
    px_tiles = np.zeros((num_tiles, 32, 32), dtype=np.float32)
    py_tiles = np.zeros((num_tiles, 32, 32), dtype=np.float32)
    for tile_id in range(num_tiles):
        ty = tile_id // tiles_x
        tx = tile_id % tiles_x
        for j in range(32):
            for i in range(32):
                px_tiles[tile_id, i, j] = tx * 32 + j + 0.5
                py_tiles[tile_id, i, j] = ty * 32 + i + 0.5

    return attribute_packs, tile_offsets, px_tiles, py_tiles
```

- [ ] **Step 2: Write a quick test that `prepare_kernel_inputs` + `alpha_blend_reference` produce the same image as the existing CPU `alpha_blend`**

Add to `tests/test_numeric_sanity.py`:

```python
import torch
from rasterization import alpha_blend, prepare_kernel_inputs


def test_prepare_kernel_inputs_matches_cpu_alpha_blend():
    """Pipeline equivalence: Python -> prepare_kernel_inputs -> reference ≈ alpha_blend()."""
    H, W = 64, 64  # 2x2 tiles
    torch.manual_seed(0)
    N = 10
    means_2d = torch.rand(N, 2) * torch.tensor([float(W), float(H)])
    # Diagonal covariance, well-conditioned
    covs_2d = torch.zeros(N, 2, 2)
    covs_2d[:, 0, 0] = 5.0
    covs_2d[:, 1, 1] = 5.0
    colors = torch.rand(N, 3)
    opacities = torch.rand(N) * 0.5 + 0.1
    depths = torch.arange(N, dtype=torch.float32)

    # Simple tile assignment: all Gaussians to all tiles
    from rasterization import get_tile_assignments, sort_and_bin
    radii = torch.full((N,), 10.0)
    gaussian_ids, tile_ids, _ = get_tile_assignments(means_2d, radii, H, W)
    sorted_gaussian_ids, tile_ranges = sort_and_bin(
        gaussian_ids, tile_ids, depths, (W + 31) // 32, (H + 31) // 32
    )

    # CPU reference
    cpu_img = alpha_blend(
        means_2d, covs_2d, colors, opacities,
        sorted_gaussian_ids, tile_ranges, H, W, tile_size=32,
    ).numpy()

    # Prepare kernel inputs + run reference
    packs, offsets, px, py = prepare_kernel_inputs(
        means_2d, covs_2d, colors, opacities,
        sorted_gaussian_ids, tile_ranges, H, W,
    )
    from scripts.numeric_sanity import alpha_blend_reference
    ref_img = alpha_blend_reference(packs, offsets, px, py, H, W)

    # Should agree within fp32 precision
    assert np.allclose(cpu_img, ref_img, atol=1e-3), f"max diff: {np.abs(cpu_img - ref_img).max()}"
```

- [ ] **Step 3: Note: `alpha_blend` currently defaults to tile_size=16. Update caller to pass tile_size=32**

Check: `rasterization.py:263` — ensure `alpha_blend(..., tile_size: int = 16)` accepts explicit 32. It already does. No code change needed — the test passes `tile_size=32` explicitly.

- [ ] **Step 4: Run the test**

Run:
```bash
cd /localdev/vkovinic/gsplat_tt
source venv/bin/activate
pytest tests/test_numeric_sanity.py::test_prepare_kernel_inputs_matches_cpu_alpha_blend -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rasterization.py tests/test_numeric_sanity.py
git commit -m "add prepare_kernel_inputs + pipeline equivalence test"
```

---

### Task 0.4: Build tt-metal

Ensure the vendored tt-metal builds cleanly. This is where most "environment" time goes for a first-timer.

- [ ] **Step 1: Verify tt-metal is at the expected location**

Run:
```bash
ls /localdev/vkovinic/gsplat_tt/tt-metal/build_metal.sh
```
Expected: file exists. If not, see `tt-metal/INSTALLING.md`.

- [ ] **Step 2: Install tt-metal dependencies (one-time)**

Run:
```bash
cd /localdev/vkovinic/gsplat_tt/tt-metal
./install_dependencies.sh
```
Expected: completes without error. If fails, read the error and resolve (usually missing apt packages: `libhwloc-dev`, `libnuma-dev`, etc.).

- [ ] **Step 3: Create tt-metal's venv**

Run:
```bash
cd /localdev/vkovinic/gsplat_tt/tt-metal
./create_venv.sh
```
Expected: creates `tt-metal/python_env/`.

- [ ] **Step 4: Activate tt-metal env and build**

Run:
```bash
cd /localdev/vkovinic/gsplat_tt/tt-metal
source python_env/bin/activate
sudo ./build_metal.sh
```
Expected: build completes. Watch for `TTMLIR_TOOLCHAIN_DIR` issues; set if needed (see `tt-metal/README.md`).

- [ ] **Step 5: Verify device visible**

Run:
```bash
cd /localdev/vkovinic/gsplat_tt/tt-metal
source python_env/bin/activate
tt-smi
```
Expected: shows Wormhole device info.

- [ ] **Step 6: Run an upstream example as smoke test**

Run:
```bash
cd /localdev/vkovinic/gsplat_tt/tt-metal
source python_env/bin/activate
./build/programming_examples/metal_example_eltwise_binary
```
Expected: prints "Test Passed" or similar success.

- [ ] **Step 7: Mark environment known-good**

No commit needed (no files changed). Note to self: if you re-enter the project, run `source tt-metal/python_env/bin/activate` before building anything.

---

### Task 0.5: Prototype `ttnn.untilize` on mock `(N, 3, 32, 32)` buffer

Decide between primary layout and fallback before writing kernels. §5.4 flagged this as a risk.

**Files:**
- Create: `scripts/untilize_prototype.py`

- [ ] **Step 1: Write the prototype script**

Create `scripts/untilize_prototype.py`:

```python
"""Verify ttnn.untilize handles the (N, 3, 32, 32) output layout.

If this fails, the writer kernel must use the 3-separate-buffer fallback.
"""
import numpy as np
import ttnn
import torch


def main():
    device = ttnn.open_device(device_id=0)
    try:
        N = 4  # 4 screen tiles
        shape = (N, 3, 32, 32)
        host_tensor = torch.rand(shape, dtype=torch.bfloat16)

        tile_tensor = ttnn.from_torch(
            host_tensor,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        print(f"Tilized shape: {tile_tensor.shape}, layout: {tile_tensor.layout}")

        row_major = ttnn.to_layout(tile_tensor, ttnn.ROW_MAJOR_LAYOUT)
        print(f"Row-major shape: {row_major.shape}")

        back_on_host = ttnn.to_torch(row_major)
        assert torch.allclose(back_on_host, host_tensor, atol=0.01), "roundtrip mismatch"
        print("Primary layout (N, 3, 32, 32) round-trip OK.")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the prototype**

Run:
```bash
cd /localdev/vkovinic/gsplat_tt
source tt-metal/python_env/bin/activate
python scripts/untilize_prototype.py
```
Expected: "Primary layout (N, 3, 32, 32) round-trip OK."

**If it fails**: switch `output_image` to 3 separate `(tiles_y*32, tiles_x*32)` buffers in Phase 1. This doesn't change the reader or compute kernels — only the writer's DRAM destination and the host's readback loop. Update this plan inline: mark §5.4 fallback as active, and in Task 2.5 below, change the writer to three `noc_async_write_tile` calls against three separate DRAM buffers instead of `3*tile_id + channel` into one buffer.

- [ ] **Step 3: Commit**

```bash
git add scripts/untilize_prototype.py
git commit -m "add ttnn.untilize prototype for output layout validation"
```

---

## Phase 1 — Scaffold + Smoke Tests

### Task 1.1: Add `.gitignore` exception for kernel subdir

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Check current `.gitignore`**

Run:
```bash
cat /localdev/vkovinic/gsplat_tt/.gitignore 2>/dev/null || echo "no .gitignore"
```

- [ ] **Step 2: Add the exception**

Open or create `.gitignore` and ensure it contains:

```gitignore
# Vendored tt-metal — untracked EXCEPT our kernel subdir
tt-metal/
!tt-metal/tt_metal/programming_examples/gaussian_splatting/
!tt-metal/tt_metal/programming_examples/gaussian_splatting/**
```

- [ ] **Step 3: Verify git now tracks the intended directory**

Run:
```bash
cd /localdev/vkovinic/gsplat_tt
mkdir -p tt-metal/tt_metal/programming_examples/gaussian_splatting
touch tt-metal/tt_metal/programming_examples/gaussian_splatting/.keep
git status --porcelain tt-metal/tt_metal/programming_examples/gaussian_splatting/
```
Expected: `?? tt-metal/tt_metal/programming_examples/gaussian_splatting/.keep` (tracked as untracked-new, not ignored).

- [ ] **Step 4: Commit the `.gitignore` change**

```bash
git add .gitignore
git commit -m "track gaussian_splatting programming_examples subdir"
```

---

### Task 1.2: Register `gaussian_splatting` in tt-metal's programming_examples CMake

**Files:**
- Modify: `tt-metal/tt_metal/programming_examples/CMakeLists.txt`

- [ ] **Step 1: Read the file to find the right place to add**

Run:
```bash
cat /localdev/vkovinic/gsplat_tt/tt-metal/tt_metal/programming_examples/CMakeLists.txt | head -50
```
You should see a list of `add_subdirectory(...)` entries.

- [ ] **Step 2: Add `add_subdirectory(gaussian_splatting)` alphabetically**

Open `tt-metal/tt_metal/programming_examples/CMakeLists.txt`, find the alphabetical insertion point, add:

```cmake
add_subdirectory(gaussian_splatting)
```

(This file is in the vendored tt-metal tree, which is untracked — the git trick only exposes our subdir. The CMakeLists change is local-only; mention in docs/setup that users need to manually apply this line after cloning. For this thesis, that's acceptable.)

- [ ] **Step 3: Commit (local-only change, not tracked by git)**

Note: this change goes into the vendored tt-metal tree. It won't be committed. Document in README.md later.

---

### Task 1.3: Create the kernel subdir's CMakeLists.txt

**Files:**
- Create: `tt-metal/tt_metal/programming_examples/gaussian_splatting/CMakeLists.txt`

- [ ] **Step 1: Copy the shape from a neighbor example**

Run:
```bash
cat /localdev/vkovinic/gsplat_tt/tt-metal/tt_metal/programming_examples/eltwise_binary/CMakeLists.txt
```

- [ ] **Step 2: Write our CMakeLists.txt**

Create `tt-metal/tt_metal/programming_examples/gaussian_splatting/CMakeLists.txt`:

```cmake
set(GS_SOURCES alpha_blend.cpp)

CREATE_PGM_EXAMPLES_EXE("${GS_SOURCES}" "gaussian_splatting")
```

(Modeled on `matmul_multi_core/CMakeLists.txt`; `CREATE_PGM_EXAMPLES_EXE` is a helper defined upstream.)

- [ ] **Step 3: Remove the `.keep` placeholder**

Run:
```bash
rm /localdev/vkovinic/gsplat_tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/.keep
```

- [ ] **Step 4: Commit**

```bash
git add tt-metal/tt_metal/programming_examples/gaussian_splatting/CMakeLists.txt
git commit -m "scaffold gaussian_splatting programming_example"
```

---

### Task 1.4: T0.0 smoke test — hello tt-metal kernel

**Files:**
- Create: `tt-metal/tt_metal/programming_examples/gaussian_splatting/alpha_blend.cpp` (initial "hello" version)
- Create: `tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/hello_compute.cpp`

- [ ] **Step 1: Write a trivial compute kernel that DPRINTs**

Create `tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/hello_compute.cpp`:

```cpp
#include "compute_kernel_api/common.h"
#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    DPRINT << "Hello from gaussian_splatting compute kernel!" << ENDL();
}
}
```

- [ ] **Step 2: Write a minimal host program that launches the kernel**

Create `tt-metal/tt_metal/programming_examples/gaussian_splatting/alpha_blend.cpp`:

```cpp
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt::tt_metal;

int main(int argc, char** argv) {
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(/*device_id=*/0);
    auto& cq = mesh_device->mesh_command_queue();

    Program program = CreateProgram();
    constexpr CoreCoord core{0, 0};

    auto compute_kernel = CreateKernel(
        program,
        "tt_metal/programming_examples/gaussian_splatting/kernels/compute/hello_compute.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi3,
            .math_approx_mode = false,
            .fp32_dest_acc_en = true,
        });

    distributed::MeshWorkload workload;
    workload.add_program(
        distributed::MeshCoordinateRange(mesh_device->shape()),
        std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

    std::cout << "Gaussian splatting hello kernel ran." << std::endl;
    return 0;
}
```

- [ ] **Step 3: Build**

Run:
```bash
cd /localdev/vkovinic/gsplat_tt/tt-metal
source python_env/bin/activate
sudo ./build_metal.sh
```
Expected: builds cleanly. Check `./build/programming_examples/metal_example_gaussian_splatting` exists.

- [ ] **Step 4: Run with DPRINT enabled**

Run:
```bash
cd /localdev/vkovinic/gsplat_tt/tt-metal
source python_env/bin/activate
export TT_METAL_DPRINT_CORES=0,0
./build/programming_examples/metal_example_gaussian_splatting
```
Expected: "Hello from gaussian_splatting compute kernel!" appears in DPRINT output, plus "Gaussian splatting hello kernel ran."

- [ ] **Step 5: Commit**

```bash
git add tt-metal/tt_metal/programming_examples/gaussian_splatting/alpha_blend.cpp \
        tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/hello_compute.cpp
git commit -m "add T0.0 hello tt-metal smoke test"
```

---

### Task 1.5: T0.1 smoke test — pass-through reader→compute→writer

Replaces the hello kernel with a 3-kernel pass-through that confirms dataflow plumbing.

**Files:**
- Modify: `tt-metal/tt_metal/programming_examples/gaussian_splatting/alpha_blend.cpp`
- Create: `tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/dataflow/passthrough_reader.cpp`
- Create: `tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/dataflow/passthrough_writer.cpp`
- Modify: `tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/hello_compute.cpp` (rename to `passthrough_compute.cpp`)

- [ ] **Step 1: Write passthrough reader**

Create `kernels/dataflow/passthrough_reader.cpp`:

```cpp
#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_id = 0;
    cb_reserve_back(cb_id, 1);
    uint32_t* ptr = reinterpret_cast<uint32_t*>(get_write_ptr(cb_id));
    constexpr uint32_t tile_size_u32 = 32 * 32 * 2 / 4;  // bf16 tile in uint32 units = 512
    for (uint32_t i = 0; i < tile_size_u32; i++) {
        ptr[i] = 0xDEADBEEF;
    }
    cb_push_back(cb_id, 1);
}
```

- [ ] **Step 2: Write passthrough compute (identity)**

Replace `kernels/compute/hello_compute.cpp` contents with:

```cpp
#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_out = 16;

    unary_op_init_common(cb_in, cb_out);
    cb_wait_front(cb_in, 1);

    tile_regs_acquire();
    copy_tile(cb_in, 0, /*dst=*/0);
    tile_regs_commit();
    tile_regs_wait();

    cb_reserve_back(cb_out, 1);
    pack_tile(/*dst=*/0, cb_out);
    cb_push_back(cb_out, 1);

    tile_regs_release();
    cb_pop_front(cb_in, 1);
}
}
```

- [ ] **Step 3: Write passthrough writer**

Create `kernels/dataflow/passthrough_writer.cpp`:

```cpp
#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_dram_addr = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_id = 16;
    const uint32_t tile_bytes = get_tile_size(cb_id);

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, dst_dram_addr, tile_bytes);

    cb_wait_front(cb_id, 1);
    uint32_t read_ptr = get_read_ptr(cb_id);
    noc_async_write_tile(0, out, read_ptr);
    noc_async_write_barrier();
    cb_pop_front(cb_id, 1);
}
```

- [ ] **Step 4: Update host program**

Replace `alpha_blend.cpp` with:

```cpp
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <iostream>
#include <vector>
#include <cstdint>

using namespace tt::tt_metal;

int main(int argc, char** argv) {
    constexpr uint32_t tile_bytes = 32 * 32 * 2;  // bf16 tile
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_out = 16;

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(/*device_id=*/0);
    auto& cq = mesh_device->mesh_command_queue();

    // Output DRAM buffer
    distributed::ReplicatedBufferConfig rep_cfg{.size = tile_bytes};
    DeviceLocalBufferConfig dram_cfg{
        .page_size = tile_bytes,
        .buffer_type = BufferType::DRAM,
    };
    auto out_buf = distributed::MeshBuffer::create(rep_cfg, dram_cfg, mesh_device.get());

    Program program = CreateProgram();
    constexpr CoreCoord core{0, 0};

    CircularBufferConfig cb_in_cfg(tile_bytes * 2, {{cb_in, DataFormat::Float16_b}});
    cb_in_cfg.set_page_size(cb_in, tile_bytes);
    CreateCircularBuffer(program, core, cb_in_cfg);

    CircularBufferConfig cb_out_cfg(tile_bytes * 2, {{cb_out, DataFormat::Float16_b}});
    cb_out_cfg.set_page_size(cb_out, tile_bytes);
    CreateCircularBuffer(program, core, cb_out_cfg);

    auto reader = CreateKernel(
        program,
        "tt_metal/programming_examples/gaussian_splatting/kernels/dataflow/passthrough_reader.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
        });

    auto compute = CreateKernel(
        program,
        "tt_metal/programming_examples/gaussian_splatting/kernels/compute/hello_compute.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi3,
            .math_approx_mode = false,
            .fp32_dest_acc_en = true,
        });

    std::vector<uint32_t> writer_ct_args;
    TensorAccessorArgs(*out_buf->get_reference_buffer()).append_to(writer_ct_args);
    auto writer = CreateKernel(
        program,
        "tt_metal/programming_examples/gaussian_splatting/kernels/dataflow/passthrough_writer.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct_args,
        });

    SetRuntimeArgs(program, writer, core, {out_buf->address()});

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

    // Read back and verify pattern
    std::vector<uint8_t> result(tile_bytes);
    distributed::EnqueueReadMeshBuffer(cq, out_buf, result);

    uint32_t* as_u32 = reinterpret_cast<uint32_t*>(result.data());
    bool ok = true;
    for (uint32_t i = 0; i < tile_bytes / 4; i++) {
        if (as_u32[i] != 0xDEADBEEF) {
            ok = false;
            std::cerr << "Mismatch at u32 index " << i << ": got 0x" << std::hex << as_u32[i] << std::endl;
            break;
        }
    }
    std::cout << (ok ? "Passthrough OK" : "Passthrough FAILED") << std::endl;
    return ok ? 0 : 1;
}
```

- [ ] **Step 5: Build**

Run:
```bash
cd /localdev/vkovinic/gsplat_tt/tt-metal
source python_env/bin/activate
sudo ./build_metal.sh
```
Expected: builds cleanly.

- [ ] **Step 6: Run**

Run:
```bash
./build/programming_examples/metal_example_gaussian_splatting
```
Expected: "Passthrough OK".

- [ ] **Step 7: Commit**

```bash
git add tt-metal/tt_metal/programming_examples/gaussian_splatting/
git commit -m "add T0.1 passthrough smoke test"
```

---

## Phase 2 — v1a: Single Gaussian, Single Tile

### Task 2.1: Shared constants header

**Files:**
- Create: `tt-metal/tt_metal/programming_examples/gaussian_splatting/alpha_blend_host.h`

- [ ] **Step 1: Write the constants header**

Create `alpha_blend_host.h`:

```cpp
#pragma once
#include <cstdint>

namespace gsplat {

constexpr uint32_t TILE_H = 32;
constexpr uint32_t TILE_W = 32;
constexpr uint32_t TILE_BYTES_BF16 = TILE_H * TILE_W * 2;     // 2 KB
constexpr uint32_t SCALAR_PACK_BYTES = 9 * 4;                  // 9 fp32 scalars
constexpr uint32_t SCALAR_PACK_PAGE_BYTES = 64;                // padded for NoC alignment
constexpr uint32_t META_PAGE_BYTES = 64;                       // padded uint32 page

// CB indices
constexpr uint32_t CB_PX         = 0;
constexpr uint32_t CB_PY         = 1;
constexpr uint32_t CB_SCALARS    = 2;
constexpr uint32_t CB_TILE_META  = 3;

// Scratch CBs
constexpr uint32_t CB_DX         = 4;
constexpr uint32_t CB_DY         = 5;
constexpr uint32_t CB_DX2        = 6;
constexpr uint32_t CB_DY2        = 7;
constexpr uint32_t CB_DXDY       = 8;
constexpr uint32_t CB_Q          = 9;
constexpr uint32_t CB_POWER      = 10;
constexpr uint32_t CB_WEIGHT     = 11;
constexpr uint32_t CB_ALPHA      = 12;
constexpr uint32_t CB_CONTRIB    = 13;
constexpr uint32_t CB_ONE_MINUS_ALPHA = 14;
constexpr uint32_t CB_T_TMP      = 15;

// Output CB
constexpr uint32_t CB_COLOR_OUT  = 16;

// State CBs
constexpr uint32_t CB_COLOR_R_STATE = 17;
constexpr uint32_t CB_COLOR_G_STATE = 18;
constexpr uint32_t CB_COLOR_B_STATE = 19;
constexpr uint32_t CB_T_STATE       = 20;
constexpr uint32_t CB_SAT_MASK      = 21;

// Constants
constexpr uint32_t CB_CONST_ZERO = 22;
constexpr uint32_t CB_CONST_099  = 23;

// Early-termination threshold
constexpr float T_SAT_THRESHOLD = 1e-4f;

}  // namespace gsplat
```

- [ ] **Step 2: Commit**

```bash
git add tt-metal/tt_metal/programming_examples/gaussian_splatting/alpha_blend_host.h
git commit -m "add shared constants header"
```

---

### Task 2.2: Reader kernel (full v1a version)

**Files:**
- Create: `tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/dataflow/reader_alpha_blend.cpp`

- [ ] **Step 1: Write the reader**

Create `kernels/dataflow/reader_alpha_blend.cpp`:

```cpp
#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t packs_addr       = get_arg_val<uint32_t>(0);
    uint32_t tile_offsets_addr= get_arg_val<uint32_t>(1);
    uint32_t px_addr          = get_arg_val<uint32_t>(2);
    uint32_t py_addr          = get_arg_val<uint32_t>(3);
    uint32_t first_tile_id    = get_arg_val<uint32_t>(4);
    uint32_t num_tiles        = get_arg_val<uint32_t>(5);

    constexpr uint32_t CB_PX = 0;
    constexpr uint32_t CB_PY = 1;
    constexpr uint32_t CB_SCALARS = 2;
    constexpr uint32_t CB_TILE_META = 3;

    const uint32_t tile_bytes = get_tile_size(CB_PX);  // 2048
    const uint32_t pack_bytes_padded = get_tile_size(CB_SCALARS);  // 64
    constexpr uint32_t PACK_BYTES = 9 * 4;  // 36 actual

    constexpr auto pack_args = TensorAccessorArgs<0>();
    constexpr auto offsets_args = TensorAccessorArgs<pack_args.next_compile_time_args_offset()>();
    constexpr auto px_args = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();
    constexpr auto py_args = TensorAccessorArgs<px_args.next_compile_time_args_offset()>();

    const auto packs_acc = TensorAccessor(pack_args, packs_addr, pack_bytes_padded);
    const auto offsets_acc = TensorAccessor(offsets_args, tile_offsets_addr, 4);
    const auto px_acc = TensorAccessor(px_args, px_addr, tile_bytes);
    const auto py_acc = TensorAccessor(py_args, py_addr, tile_bytes);

    // Read tile_offsets[first_tile_id .. first_tile_id + num_tiles + 1] into L1 scratch
    constexpr uint32_t MAX_TILES = 512;
    uint32_t offsets[MAX_TILES + 1];
    uint64_t offsets_base = get_noc_addr(0, offsets_acc);
    noc_async_read(
        offsets_base + first_tile_id * 4,
        reinterpret_cast<uint32_t>(offsets),
        (num_tiles + 1) * 4);
    noc_async_read_barrier();

    for (uint32_t t = 0; t < num_tiles; t++) {
        uint32_t tile_id = first_tile_id + t;
        uint32_t g_start = offsets[t];
        uint32_t g_end   = offsets[t + 1];
        uint32_t g_count = g_end - g_start;

        // Push Gaussian count meta
        cb_reserve_back(CB_TILE_META, 1);
        auto meta_ptr = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_TILE_META));
        meta_ptr[0] = g_count;
        cb_push_back(CB_TILE_META, 1);

        // Push px and py tiles for this screen tile
        cb_reserve_back(CB_PX, 1);
        noc_async_read_tile(tile_id, px_acc, get_write_ptr(CB_PX));
        cb_reserve_back(CB_PY, 1);
        noc_async_read_tile(tile_id, py_acc, get_write_ptr(CB_PY));
        noc_async_read_barrier();
        cb_push_back(CB_PX, 1);
        cb_push_back(CB_PY, 1);

        // Stream Gaussian packs
        for (uint32_t g = 0; g < g_count; g++) {
            uint32_t entry_id = g_start + g;
            cb_reserve_back(CB_SCALARS, 1);
            noc_async_read_tile(entry_id, packs_acc, get_write_ptr(CB_SCALARS));
            noc_async_read_barrier();
            cb_push_back(CB_SCALARS, 1);
        }
    }
}
```

- [ ] **Step 2: Commit (does not build standalone yet)**

```bash
git add tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/dataflow/reader_alpha_blend.cpp
git commit -m "add reader_alpha_blend kernel"
```

---

### Task 2.3: Writer kernel

**Files:**
- Create: `tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/dataflow/writer_alpha_blend.cpp`

- [ ] **Step 1: Write the writer**

Create `kernels/dataflow/writer_alpha_blend.cpp`:

```cpp
#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t out_addr      = get_arg_val<uint32_t>(0);
    uint32_t first_tile_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles     = get_arg_val<uint32_t>(2);

    constexpr uint32_t CB_COLOR_OUT = 16;
    const uint32_t tile_bytes = get_tile_size(CB_COLOR_OUT);

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);

    for (uint32_t t = 0; t < num_tiles; t++) {
        uint32_t screen_tile = first_tile_id + t;

        cb_wait_front(CB_COLOR_OUT, 3);
        uint32_t read_ptr = get_read_ptr(CB_COLOR_OUT);
        for (uint32_t ch = 0; ch < 3; ch++) {
            uint32_t out_tile_id = 3 * screen_tile + ch;
            noc_async_write_tile(out_tile_id, out, read_ptr);
            read_ptr += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(CB_COLOR_OUT, 3);
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/dataflow/writer_alpha_blend.cpp
git commit -m "add writer_alpha_blend kernel"
```

---

### Task 2.4: Compute kernel — 11a (solid color output)

Sub-phase 11a: write a solid color tile to verify writer + output wiring before adding any math.

**Files:**
- Create: `tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp`

- [ ] **Step 1: Write the 11a compute kernel**

Create `kernels/compute/alpha_blend_compute.cpp`:

```cpp
#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/eltwise_unary/fill.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t CB_PX = 0;
    constexpr uint32_t CB_PY = 1;
    constexpr uint32_t CB_SCALARS = 2;
    constexpr uint32_t CB_TILE_META = 3;
    constexpr uint32_t CB_COLOR_OUT = 16;

    unary_op_init_common(CB_PX, CB_COLOR_OUT);

    for (uint32_t t = 0; t < num_tiles; t++) {
        // Drain inputs (we don't use them yet — just smoke test)
        cb_wait_front(CB_TILE_META, 1);
        uint32_t g_count = *reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_TILE_META));
        cb_pop_front(CB_TILE_META, 1);

        cb_wait_front(CB_PX, 1); cb_pop_front(CB_PX, 1);
        cb_wait_front(CB_PY, 1); cb_pop_front(CB_PY, 1);

        for (uint32_t g = 0; g < g_count; g++) {
            cb_wait_front(CB_SCALARS, 1);
            cb_pop_front(CB_SCALARS, 1);
        }

        // Emit 3 solid-color tiles: R=0.25, G=0.5, B=0.75
        cb_reserve_back(CB_COLOR_OUT, 3);
        tile_regs_acquire();
        fill_tile(0, 0.25f);
        fill_tile(1, 0.5f);
        fill_tile(2, 0.75f);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_COLOR_OUT);
        pack_tile(1, CB_COLOR_OUT);
        pack_tile(2, CB_COLOR_OUT);
        tile_regs_release();
        cb_push_back(CB_COLOR_OUT, 3);
    }
}
}
```

- [ ] **Step 2: Commit**

```bash
git add tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp
git commit -m "add compute kernel 11a: solid color output"
```

---

### Task 2.5: Host program — v1a driver

Single-tile single-Gaussian driver for v1a. Reads `.npy` fixture, launches kernels, writes output.

**Files:**
- Modify: `tt-metal/tt_metal/programming_examples/gaussian_splatting/alpha_blend.cpp`

- [ ] **Step 1: Replace alpha_blend.cpp with the v1a driver**

Replace contents with:

```cpp
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "alpha_blend_host.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>

using namespace tt::tt_metal;
using namespace gsplat;

// Minimal .npy reader for 2D/3D float32 arrays.
// Reads header, skips dtype check, returns raw data + shape.
static std::vector<float> load_npy_f32(const std::string& path, std::vector<size_t>& shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "cannot open " << path << std::endl; std::exit(1); }
    char magic[6];
    f.read(magic, 6);
    uint8_t major, minor;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);
    uint16_t header_len;
    f.read(reinterpret_cast<char*>(&header_len), 2);
    std::string header(header_len, ' ');
    f.read(header.data(), header_len);
    // Parse shape from header "'shape': (d0, d1, ...)"
    auto start = header.find('(') + 1;
    auto end = header.find(')');
    std::string shape_str = header.substr(start, end - start);
    shape.clear();
    size_t pos = 0;
    while (pos < shape_str.size()) {
        size_t comma = shape_str.find(',', pos);
        if (comma == std::string::npos) comma = shape_str.size();
        std::string num = shape_str.substr(pos, comma - pos);
        // trim
        while (!num.empty() && (num.front() == ' ' || num.front() == '\t')) num.erase(0, 1);
        while (!num.empty() && (num.back() == ' ' || num.back() == '\t')) num.pop_back();
        if (!num.empty()) shape.push_back(std::stoul(num));
        pos = comma + 1;
    }
    size_t n = 1;
    for (auto d : shape) n *= d;
    std::vector<float> data(n);
    f.read(reinterpret_cast<char*>(data.data()), n * sizeof(float));
    return data;
}

static void save_npy_f32(const std::string& path, const std::vector<float>& data, const std::vector<size_t>& shape) {
    std::ofstream f(path, std::ios::binary);
    f.write("\x93NUMPY", 6);
    uint8_t major = 1, minor = 0;
    f.write(reinterpret_cast<char*>(&major), 1);
    f.write(reinterpret_cast<char*>(&minor), 1);
    std::string shape_str = "(";
    for (size_t i = 0; i < shape.size(); i++) {
        shape_str += std::to_string(shape[i]);
        if (i + 1 < shape.size() || shape.size() == 1) shape_str += ", ";
    }
    shape_str += ")";
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape_str + ", }";
    while ((10 + header.size() + 1) % 64 != 0) header += ' ';
    header += '\n';
    uint16_t header_len = header.size();
    f.write(reinterpret_cast<char*>(&header_len), 2);
    f.write(header.data(), header.size());
    f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
}

// Convert fp32 tile (32x32) to bf16 tile (big-endian 16 bits = top half of fp32)
static std::vector<uint16_t> fp32_tile_to_bf16(const float* src) {
    std::vector<uint16_t> dst(TILE_H * TILE_W);
    for (size_t i = 0; i < TILE_H * TILE_W; i++) {
        uint32_t u;
        std::memcpy(&u, &src[i], 4);
        dst[i] = u >> 16;
    }
    return dst;
}

static std::vector<float> bf16_tile_to_fp32(const uint16_t* src) {
    std::vector<float> dst(TILE_H * TILE_W);
    for (size_t i = 0; i < TILE_H * TILE_W; i++) {
        uint32_t u = static_cast<uint32_t>(src[i]) << 16;
        std::memcpy(&dst[i], &u, 4);
    }
    return dst;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " packs.npy offsets.npy px.npy py.npy output.npy [H] [W]\n";
        return 1;
    }
    std::string packs_path = argv[1];
    std::string offsets_path = argv[2];
    std::string px_path = argv[3];
    std::string py_path = argv[4];
    std::string out_path = argv[5];
    uint32_t image_h = argc > 6 ? std::stoi(argv[6]) : 32;
    uint32_t image_w = argc > 7 ? std::stoi(argv[7]) : 32;

    uint32_t tiles_x = (image_w + 31) / 32;
    uint32_t tiles_y = (image_h + 31) / 32;
    uint32_t num_tiles = tiles_x * tiles_y;

    // Load inputs
    std::vector<size_t> packs_shape, offsets_shape, px_shape, py_shape;
    auto packs_f32 = load_npy_f32(packs_path, packs_shape);
    auto offsets_f32 = load_npy_f32(offsets_path, offsets_shape);
    auto px_f32 = load_npy_f32(px_path, px_shape);
    auto py_f32 = load_npy_f32(py_path, py_shape);

    uint32_t total_entries = packs_shape[0];

    // Device + program setup
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(/*device_id=*/0);
    auto& cq = mesh_device->mesh_command_queue();

    // DRAM buffers
    auto make_dram = [&](size_t bytes, size_t page_bytes) {
        distributed::ReplicatedBufferConfig rc{.size = bytes};
        DeviceLocalBufferConfig lc{.page_size = page_bytes, .buffer_type = BufferType::DRAM};
        return distributed::MeshBuffer::create(rc, lc, mesh_device.get());
    };

    auto packs_dram = make_dram(total_entries * SCALAR_PACK_PAGE_BYTES, SCALAR_PACK_PAGE_BYTES);
    auto offsets_dram = make_dram(offsets_f32.size() * 4, 4);
    auto px_dram = make_dram(num_tiles * TILE_BYTES_BF16, TILE_BYTES_BF16);
    auto py_dram = make_dram(num_tiles * TILE_BYTES_BF16, TILE_BYTES_BF16);
    auto out_dram = make_dram(num_tiles * 3 * TILE_BYTES_BF16, TILE_BYTES_BF16);

    // Encode attribute_packs into padded 64-byte pages (fp32 -> fp32 bits inside 64 bytes)
    std::vector<uint8_t> packs_payload(total_entries * SCALAR_PACK_PAGE_BYTES, 0);
    for (uint32_t e = 0; e < total_entries; e++) {
        std::memcpy(&packs_payload[e * SCALAR_PACK_PAGE_BYTES],
                    &packs_f32[e * 9], 9 * 4);
    }

    // Encode px/py as bf16 tiles
    std::vector<uint16_t> px_bf16(num_tiles * TILE_H * TILE_W);
    std::vector<uint16_t> py_bf16(num_tiles * TILE_H * TILE_W);
    for (uint32_t t = 0; t < num_tiles; t++) {
        auto px_tile = fp32_tile_to_bf16(&px_f32[t * TILE_H * TILE_W]);
        auto py_tile = fp32_tile_to_bf16(&py_f32[t * TILE_H * TILE_W]);
        std::memcpy(&px_bf16[t * TILE_H * TILE_W], px_tile.data(), TILE_BYTES_BF16);
        std::memcpy(&py_bf16[t * TILE_H * TILE_W], py_tile.data(), TILE_BYTES_BF16);
    }

    // offsets to uint32
    std::vector<uint32_t> offsets_u32(offsets_f32.size());
    for (size_t i = 0; i < offsets_f32.size(); i++) offsets_u32[i] = static_cast<uint32_t>(offsets_f32[i]);

    distributed::EnqueueWriteMeshBuffer(cq, packs_dram, packs_payload);
    distributed::EnqueueWriteMeshBuffer(cq, offsets_dram, offsets_u32);
    distributed::EnqueueWriteMeshBuffer(cq, px_dram, px_bf16);
    distributed::EnqueueWriteMeshBuffer(cq, py_dram, py_bf16);

    Program program = CreateProgram();
    constexpr CoreCoord core{0, 0};

    // CBs
    auto cb_tile = [&](uint32_t id, uint32_t depth = 2) {
        CircularBufferConfig c(depth * TILE_BYTES_BF16, {{id, DataFormat::Float16_b}});
        c.set_page_size(id, TILE_BYTES_BF16);
        CreateCircularBuffer(program, core, c);
    };
    auto cb_small = [&](uint32_t id, uint32_t page_bytes, uint32_t depth) {
        CircularBufferConfig c(depth * page_bytes, {{id, DataFormat::Float32}});
        c.set_page_size(id, page_bytes);
        CreateCircularBuffer(program, core, c);
    };

    cb_tile(CB_PX);
    cb_tile(CB_PY);
    cb_small(CB_SCALARS, SCALAR_PACK_PAGE_BYTES, 4);
    cb_small(CB_TILE_META, META_PAGE_BYTES, 2);
    cb_tile(CB_COLOR_OUT, 4);  // depth 4 for 3 channels + slack

    // Reader
    std::vector<uint32_t> reader_ct;
    TensorAccessorArgs(*packs_dram->get_reference_buffer()).append_to(reader_ct);
    TensorAccessorArgs(*offsets_dram->get_reference_buffer()).append_to(reader_ct);
    TensorAccessorArgs(*px_dram->get_reference_buffer()).append_to(reader_ct);
    TensorAccessorArgs(*py_dram->get_reference_buffer()).append_to(reader_ct);
    auto reader = CreateKernel(
        program,
        "tt_metal/programming_examples/gaussian_splatting/kernels/dataflow/reader_alpha_blend.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct,
        });

    auto compute = CreateKernel(
        program,
        "tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi3,
            .math_approx_mode = false,
            .fp32_dest_acc_en = true,
        });

    std::vector<uint32_t> writer_ct;
    TensorAccessorArgs(*out_dram->get_reference_buffer()).append_to(writer_ct);
    auto writer = CreateKernel(
        program,
        "tt_metal/programming_examples/gaussian_splatting/kernels/dataflow/writer_alpha_blend.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct,
        });

    SetRuntimeArgs(program, reader, core, {
        static_cast<uint32_t>(packs_dram->address()),
        static_cast<uint32_t>(offsets_dram->address()),
        static_cast<uint32_t>(px_dram->address()),
        static_cast<uint32_t>(py_dram->address()),
        /*first_tile_id=*/0u,
        /*num_tiles=*/num_tiles,
    });
    SetRuntimeArgs(program, compute, core, {num_tiles});
    SetRuntimeArgs(program, writer, core, {
        static_cast<uint32_t>(out_dram->address()),
        /*first_tile_id=*/0u,
        /*num_tiles=*/num_tiles,
    });

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

    std::vector<uint16_t> result_bf16(num_tiles * 3 * TILE_H * TILE_W);
    distributed::EnqueueReadMeshBuffer(cq, out_dram, result_bf16);

    // Convert bf16 tiles -> row-major fp32 image (H, W, 3)
    std::vector<float> img(image_h * image_w * 3, 0.0f);
    for (uint32_t t = 0; t < num_tiles; t++) {
        uint32_t ty = t / tiles_x;
        uint32_t tx = t % tiles_x;
        for (uint32_t ch = 0; ch < 3; ch++) {
            auto fp = bf16_tile_to_fp32(&result_bf16[(3 * t + ch) * TILE_H * TILE_W]);
            for (uint32_t i = 0; i < TILE_H; i++) {
                for (uint32_t j = 0; j < TILE_W; j++) {
                    uint32_t y = ty * TILE_H + i;
                    uint32_t x = tx * TILE_W + j;
                    if (y < image_h && x < image_w) {
                        img[(y * image_w + x) * 3 + ch] = fp[i * TILE_W + j];
                    }
                }
            }
        }
    }

    save_npy_f32(out_path, img, {image_h, image_w, 3});
    std::cout << "Wrote " << out_path << std::endl;
    return 0;
}
```

- [ ] **Step 2: Build**

Run:
```bash
cd /localdev/vkovinic/gsplat_tt/tt-metal
source python_env/bin/activate
sudo ./build_metal.sh
```
Expected: compiles cleanly.

- [ ] **Step 3: Commit**

```bash
git add tt-metal/tt_metal/programming_examples/gaussian_splatting/alpha_blend.cpp
git commit -m "add v1a host driver"
```

---

### Task 2.6: Smoke test — solid-color output via Python fixture

**Files:**
- Create: `scripts/dump_kernel_inputs.py`

- [ ] **Step 1: Write the dump script**

Create `scripts/dump_kernel_inputs.py`:

```python
"""Dump kernel input fixtures to .npy files for the standalone C++ harness."""
import argparse
import numpy as np
import sys


def dump_single_gaussian(out_dir):
    """T0.4: one Gaussian at (16.5, 16.5), sharp cov, red, opaque."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    packs = np.array([[16.5, 16.5, 100.0, 0.0, 100.0, 1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    offsets = np.array([0, 1], dtype=np.float32)  # fp32 for the npy reader; we cast later
    px = np.empty((1, 32, 32), dtype=np.float32)
    py = np.empty((1, 32, 32), dtype=np.float32)
    for i in range(32):
        for j in range(32):
            px[0, i, j] = j + 0.5
            py[0, i, j] = i + 0.5
    np.save(f"{out_dir}/packs.npy", packs)
    np.save(f"{out_dir}/offsets.npy", offsets)
    np.save(f"{out_dir}/px.npy", px)
    np.save(f"{out_dir}/py.npy", py)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", choices=["single_gaussian"], required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    if args.fixture == "single_gaussian":
        dump_single_gaussian(args.out_dir)
    print(f"Dumped {args.fixture} to {args.out_dir}")
```

- [ ] **Step 2: Run the dump**

Run:
```bash
cd /localdev/vkovinic/gsplat_tt
source venv/bin/activate
python scripts/dump_kernel_inputs.py --fixture single_gaussian --out-dir /tmp/gsplat_fixtures
```

- [ ] **Step 3: Run the harness (solid-color stage)**

Run:
```bash
cd /localdev/vkovinic/gsplat_tt/tt-metal
source python_env/bin/activate
./build/programming_examples/metal_example_gaussian_splatting \
    /tmp/gsplat_fixtures/packs.npy \
    /tmp/gsplat_fixtures/offsets.npy \
    /tmp/gsplat_fixtures/px.npy \
    /tmp/gsplat_fixtures/py.npy \
    /tmp/gsplat_fixtures/out.npy \
    32 32
```
Expected: "Wrote /tmp/gsplat_fixtures/out.npy".

- [ ] **Step 4: Verify solid-color output**

Run:
```bash
cd /localdev/vkovinic/gsplat_tt
source venv/bin/activate
python -c "
import numpy as np
img = np.load('/tmp/gsplat_fixtures/out.npy')
print('shape:', img.shape, 'R:', img[16, 16, 0], 'G:', img[16, 16, 1], 'B:', img[16, 16, 2])
assert abs(img[16, 16, 0] - 0.25) < 0.01
assert abs(img[16, 16, 1] - 0.5) < 0.01
assert abs(img[16, 16, 2] - 0.75) < 0.01
print('Solid-color OK')
"
```
Expected: "Solid-color OK".

- [ ] **Step 5: Commit**

```bash
git add scripts/dump_kernel_inputs.py
git commit -m "add input fixture dumper + v1a solid-color smoke"
```

---

### Task 2.7: Compute kernel — 11b (single scalar-unary op)

Sub-phase 11b: verify the scalar-unary path with a trivial `sub_unary_tile` on the px tile.

**Files:**
- Modify: `tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp`

- [ ] **Step 1: Replace compute kernel body**

Replace contents with:

```cpp
#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t CB_PX = 0;
    constexpr uint32_t CB_PY = 1;
    constexpr uint32_t CB_SCALARS = 2;
    constexpr uint32_t CB_TILE_META = 3;
    constexpr uint32_t CB_COLOR_OUT = 16;

    unary_op_init_common(CB_PX, CB_COLOR_OUT);
    binop_with_scalar_tile_init();

    for (uint32_t t = 0; t < num_tiles; t++) {
        cb_wait_front(CB_TILE_META, 1);
        uint32_t g_count = *reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_TILE_META));
        cb_pop_front(CB_TILE_META, 1);

        cb_wait_front(CB_PX, 1);
        cb_wait_front(CB_PY, 1);

        for (uint32_t g = 0; g < g_count; g++) {
            cb_wait_front(CB_SCALARS, 1);
            cb_pop_front(CB_SCALARS, 1);
        }

        // Compute: R channel = (px - 16.0); G channel = (py - 16.0); B = 0
        constexpr uint32_t sixteen_bits = 0x41800000;  // fp32 bits of 16.0
        tile_regs_acquire();
        copy_tile(CB_PX, 0, /*dst=*/0);
        sub_unary_tile(0, sixteen_bits);
        copy_tile(CB_PY, 0, /*dst=*/1);
        sub_unary_tile(1, sixteen_bits);
        // dst[2] left unchanged but pack_tile would read garbage — use fill
        #include "compute_kernel_api/eltwise_unary/fill.h"
        // ^ moved above in real source; inline here for narrative
        // Actually compute kernel needs clean structure — pack 0,1, and a zero-fill for B.
        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(CB_COLOR_OUT, 3);
        pack_tile(0, CB_COLOR_OUT);
        pack_tile(1, CB_COLOR_OUT);
        // B = 0: reacquire + fill + pack
        tile_regs_release();

        tile_regs_acquire();
        // fill dst=0 with 0.0
        // Note: unary_op_init_common already called; fill_tile init may be needed:
        // fill_tile_init(); — optional init, check header.
        // Use sub_unary_tile with self to zero a copy_tile'd tile is wasteful.
        // Simpler: rely on fill_tile — requires fill.h included at top of file.
        // [See fixed version below.]
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_COLOR_OUT);
        cb_push_back(CB_COLOR_OUT, 3);

        tile_regs_release();

        cb_pop_front(CB_PX, 1);
        cb_pop_front(CB_PY, 1);
    }
}
}
```

- [ ] **Step 2: Clean up includes at the top and correct the kernel structure**

Rewrite the file with correct includes:

```cpp
#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/fill.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t CB_PX = 0;
    constexpr uint32_t CB_PY = 1;
    constexpr uint32_t CB_SCALARS = 2;
    constexpr uint32_t CB_TILE_META = 3;
    constexpr uint32_t CB_COLOR_OUT = 16;

    unary_op_init_common(CB_PX, CB_COLOR_OUT);
    binop_with_scalar_tile_init();

    for (uint32_t t = 0; t < num_tiles; t++) {
        cb_wait_front(CB_TILE_META, 1);
        uint32_t g_count = *reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_TILE_META));
        cb_pop_front(CB_TILE_META, 1);

        cb_wait_front(CB_PX, 1);
        cb_wait_front(CB_PY, 1);
        for (uint32_t g = 0; g < g_count; g++) {
            cb_wait_front(CB_SCALARS, 1);
            cb_pop_front(CB_SCALARS, 1);
        }

        constexpr uint32_t sixteen_bits = 0x41800000;  // fp32(16.0)
        cb_reserve_back(CB_COLOR_OUT, 3);

        // R = px - 16, G = py - 16, B = 0, all in one block (4 Dst slots available)
        tile_regs_acquire();
        copy_tile(CB_PX, 0, /*dst=*/0);
        sub_unary_tile(0, sixteen_bits);
        copy_tile(CB_PY, 0, /*dst=*/1);
        sub_unary_tile(1, sixteen_bits);
        fill_tile(2, 0.0f);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_COLOR_OUT);
        pack_tile(1, CB_COLOR_OUT);
        pack_tile(2, CB_COLOR_OUT);
        tile_regs_release();
        cb_push_back(CB_COLOR_OUT, 3);

        cb_pop_front(CB_PX, 1);
        cb_pop_front(CB_PY, 1);
    }
}
}
```

- [ ] **Step 3: Build**

Run:
```bash
cd /localdev/vkovinic/gsplat_tt
source tt-metal/python_env/bin/activate
sudo ./tt-metal/build_metal.sh
```
Expected: builds cleanly.

- [ ] **Step 4: Run on single_gaussian fixture**

Run:
```bash
./tt-metal/build/programming_examples/metal_example_gaussian_splatting \
    /tmp/gsplat_fixtures/packs.npy \
    /tmp/gsplat_fixtures/offsets.npy \
    /tmp/gsplat_fixtures/px.npy \
    /tmp/gsplat_fixtures/py.npy \
    /tmp/gsplat_fixtures/out.npy 32 32
```

- [ ] **Step 5: Verify R = px-16, G = py-16, B = 0**

Run:
```bash
python -c "
import numpy as np
img = np.load('/tmp/gsplat_fixtures/out.npy')
# At pixel (0, 0): px=0.5, py=0.5 -> R=-15.5, G=-15.5, B=0
# bf16 precision: -15.5 is representable exactly
print('corner:', img[0, 0])
print('center:', img[16, 16])
assert abs(img[16, 16, 0] - 0.5) < 0.1  # px-16 at pixel 16 = 16.5-16 = 0.5
assert abs(img[16, 16, 2]) < 0.01
print('11b OK')
"
```
Expected: "11b OK".

- [ ] **Step 6: Commit**

```bash
git add tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp
git commit -m "compute kernel 11b: scalar-unary verified"
```

---

### Task 2.8: Compute kernel — 11c (full single-Gaussian math chain, NO loop, NO sat_mask)

Sub-phase 11c: full Stages A–E for one Gaussian. Output is RGB = `contrib * color` at each pixel.

**Files:**
- Modify: `tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp`

- [ ] **Step 1: Rewrite the compute kernel with full math**

Replace contents with the full single-Gaussian version. This is the largest single-task code block — copy carefully:

```cpp
#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/fill.h"
#include "compute_kernel_api/eltwise_unary/rsub.h"
#include "compute_kernel_api/binary_max_min.h"

namespace NAMESPACE {

// CB indices (must match host)
constexpr uint32_t CB_PX = 0;
constexpr uint32_t CB_PY = 1;
constexpr uint32_t CB_SCALARS = 2;
constexpr uint32_t CB_TILE_META = 3;
constexpr uint32_t CB_DX = 4;
constexpr uint32_t CB_DY = 5;
constexpr uint32_t CB_DX2 = 6;
constexpr uint32_t CB_DY2 = 7;
constexpr uint32_t CB_DXDY = 8;
constexpr uint32_t CB_Q = 9;
constexpr uint32_t CB_POWER = 10;
constexpr uint32_t CB_WEIGHT = 11;
constexpr uint32_t CB_ALPHA = 12;
constexpr uint32_t CB_CONTRIB = 13;
constexpr uint32_t CB_ONE_MINUS_ALPHA = 14;
constexpr uint32_t CB_T_TMP = 15;
constexpr uint32_t CB_COLOR_OUT = 16;
constexpr uint32_t CB_CONST_ZERO = 22;
constexpr uint32_t CB_CONST_099 = 23;

inline uint32_t as_u32(float x) {
    uint32_t u;
    __builtin_memcpy(&u, &x, 4);
    return u;
}

void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    binary_op_init_common(CB_PX, CB_PY, CB_COLOR_OUT);
    binop_with_scalar_tile_init();
    exp_tile_init();
    copy_tile_init(CB_PX);
    mul_tiles_init(CB_PX, CB_PY);
    add_tiles_init(CB_PX, CB_PY);

    // Pre-fill constant CBs once
    cb_reserve_back(CB_CONST_ZERO, 1);
    tile_regs_acquire();
    fill_tile(0, 0.0f);
    tile_regs_commit(); tile_regs_wait();
    pack_tile(0, CB_CONST_ZERO);
    tile_regs_release();
    cb_push_back(CB_CONST_ZERO, 1);

    cb_reserve_back(CB_CONST_099, 1);
    tile_regs_acquire();
    fill_tile(0, 0.99f);
    tile_regs_commit(); tile_regs_wait();
    pack_tile(0, CB_CONST_099);
    tile_regs_release();
    cb_push_back(CB_CONST_099, 1);

    cb_wait_front(CB_CONST_ZERO, 1);
    cb_wait_front(CB_CONST_099, 1);

    for (uint32_t t = 0; t < num_tiles; t++) {
        cb_wait_front(CB_TILE_META, 1);
        uint32_t g_count = *reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_TILE_META));
        cb_pop_front(CB_TILE_META, 1);

        cb_wait_front(CB_PX, 1);
        cb_wait_front(CB_PY, 1);

        // v1a: assume exactly 1 Gaussian per tile
        for (uint32_t g = 0; g < g_count; g++) {
            cb_wait_front(CB_SCALARS, 1);
            volatile float* pack = reinterpret_cast<volatile float*>(get_read_ptr(CB_SCALARS));
            float mean_x = pack[0];
            float mean_y = pack[1];
            float cov_a = pack[2];
            float two_cov_b = pack[3];
            float cov_c = pack[4];
            float color_r = pack[5];
            float color_g = pack[6];
            float color_b = pack[7];
            float opacity = pack[8];

            // B1: dx, dy
            tile_regs_acquire();
            copy_tile(CB_PX, 0, 0); sub_unary_tile(0, as_u32(mean_x));
            copy_tile(CB_PY, 0, 1); sub_unary_tile(1, as_u32(mean_y));
            tile_regs_commit(); tile_regs_wait();
            cb_reserve_back(CB_DX, 1); pack_tile(0, CB_DX); cb_push_back(CB_DX, 1);
            cb_reserve_back(CB_DY, 1); pack_tile(1, CB_DY); cb_push_back(CB_DY, 1);
            tile_regs_release();

            cb_wait_front(CB_DX, 1);
            cb_wait_front(CB_DY, 1);

            // B2a: dx²
            tile_regs_acquire();
            mul_tiles(CB_DX, CB_DX, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            cb_reserve_back(CB_DX2, 1); pack_tile(0, CB_DX2); cb_push_back(CB_DX2, 1);
            tile_regs_release();

            // B2b: dy²
            tile_regs_acquire();
            mul_tiles(CB_DY, CB_DY, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            cb_reserve_back(CB_DY2, 1); pack_tile(0, CB_DY2); cb_push_back(CB_DY2, 1);
            tile_regs_release();

            // B2c: dx·dy
            tile_regs_acquire();
            mul_tiles(CB_DX, CB_DY, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            cb_reserve_back(CB_DXDY, 1); pack_tile(0, CB_DXDY); cb_push_back(CB_DXDY, 1);
            tile_regs_release();

            cb_wait_front(CB_DX2, 1);
            cb_wait_front(CB_DY2, 1);
            cb_wait_front(CB_DXDY, 1);

            // B3: scale each + sum.
            // a·dx² -> dst[0]
            tile_regs_acquire();
            copy_tile(CB_DX2, 0, 0); mul_unary_tile(0, as_u32(cov_a));
            copy_tile(CB_DY2, 0, 1); mul_unary_tile(1, as_u32(cov_c));
            copy_tile(CB_DXDY, 0, 2); mul_unary_tile(2, as_u32(two_cov_b));
            // add: dst[0] = dst[0] + dst[1] is not direct; use copy via CB_Q?
            // Simpler approach: pack and add_tiles.
            tile_regs_commit(); tile_regs_wait();
            cb_reserve_back(CB_Q, 3);
            pack_tile(0, CB_Q); pack_tile(1, CB_Q); pack_tile(2, CB_Q);
            cb_push_back(CB_Q, 3);
            tile_regs_release();

            cb_wait_front(CB_Q, 3);

            // add a·dx² + c·dy² -> dst[0]
            tile_regs_acquire();
            add_tiles(CB_Q, CB_Q, 0, 1, 0);  // CB_Q[0] + CB_Q[1]
            tile_regs_commit(); tile_regs_wait();
            cb_reserve_back(CB_POWER, 1);
            pack_tile(0, CB_POWER);
            cb_push_back(CB_POWER, 1);
            tile_regs_release();
            cb_wait_front(CB_POWER, 1);

            tile_regs_acquire();
            add_tiles(CB_POWER, CB_Q, 0, 2, 0);  // + 2b·dxdy
            mul_unary_tile(0, as_u32(-0.5f));    // -0.5 · Q
            // min(power, 0)
            copy_tile(CB_CONST_ZERO, 0, 1);
            binary_min_tile(0, 1, 0);
            exp_tile(0);
            mul_unary_tile(0, as_u32(opacity));  // alpha_unclamped = opacity · weight
            copy_tile(CB_CONST_099, 0, 1);
            binary_min_tile(0, 1, 0);            // alpha = min(alpha, 0.99)
            tile_regs_commit(); tile_regs_wait();
            cb_reserve_back(CB_ALPHA, 1);
            pack_tile(0, CB_ALPHA);
            cb_push_back(CB_ALPHA, 1);
            tile_regs_release();
            cb_pop_front(CB_POWER, 1);
            cb_pop_front(CB_Q, 3);
            cb_pop_front(CB_DX, 1); cb_pop_front(CB_DY, 1);
            cb_pop_front(CB_DX2, 1); cb_pop_front(CB_DY2, 1); cb_pop_front(CB_DXDY, 1);

            // For v1a (single Gaussian), T starts at 1, so contrib = alpha * 1 = alpha.
            // R = color_r * alpha, G = color_g * alpha, B = color_b * alpha.
            cb_wait_front(CB_ALPHA, 1);
            tile_regs_acquire();
            copy_tile(CB_ALPHA, 0, 0); mul_unary_tile(0, as_u32(color_r));
            copy_tile(CB_ALPHA, 0, 1); mul_unary_tile(1, as_u32(color_g));
            copy_tile(CB_ALPHA, 0, 2); mul_unary_tile(2, as_u32(color_b));
            tile_regs_commit(); tile_regs_wait();
            cb_reserve_back(CB_COLOR_OUT, 3);
            pack_tile(0, CB_COLOR_OUT);
            pack_tile(1, CB_COLOR_OUT);
            pack_tile(2, CB_COLOR_OUT);
            cb_push_back(CB_COLOR_OUT, 3);
            tile_regs_release();
            cb_pop_front(CB_ALPHA, 1);
            cb_pop_front(CB_SCALARS, 1);
        }

        cb_pop_front(CB_PX, 1);
        cb_pop_front(CB_PY, 1);
    }
}

}  // namespace NAMESPACE
```

**Note**: this is v1a (single Gaussian per tile, no state CBs, no sat_mask, no per-Gaussian loop compositing). The full v1b version comes in Task 3.1.

- [ ] **Step 2: Update the host to allocate the new scratch CBs**

Open `alpha_blend.cpp` and add CB setup after existing CBs:

```cpp
    // Scratch CBs for v1a compute
    cb_tile(CB_DX);
    cb_tile(CB_DY);
    cb_tile(CB_DX2);
    cb_tile(CB_DY2);
    cb_tile(CB_DXDY);
    // CB_Q packs 3 tiles (a·dx², c·dy², 2b·dxdy)
    {
        CircularBufferConfig c(3 * TILE_BYTES_BF16, {{CB_Q, DataFormat::Float16_b}});
        c.set_page_size(CB_Q, TILE_BYTES_BF16);
        CreateCircularBuffer(program, core, c);
    }
    cb_tile(CB_POWER);
    cb_tile(CB_ALPHA);
    // Constants: depth 1 (filled once at kernel start, never popped)
    cb_tile(CB_CONST_ZERO, 1);
    cb_tile(CB_CONST_099, 1);
```

- [ ] **Step 3: Build**

Run:
```bash
cd /localdev/vkovinic/gsplat_tt
source tt-metal/python_env/bin/activate
sudo ./tt-metal/build_metal.sh
```
Expected: builds cleanly.

- [ ] **Step 4: Run on single_gaussian fixture**

Run:
```bash
./tt-metal/build/programming_examples/metal_example_gaussian_splatting \
    /tmp/gsplat_fixtures/packs.npy \
    /tmp/gsplat_fixtures/offsets.npy \
    /tmp/gsplat_fixtures/px.npy \
    /tmp/gsplat_fixtures/py.npy \
    /tmp/gsplat_fixtures/out.npy 32 32
```

- [ ] **Step 5: Verify against NumPy reference (T0.4)**

Run:
```bash
cd /localdev/vkovinic/gsplat_tt
source venv/bin/activate
python -c "
import numpy as np
from scripts.numeric_sanity import alpha_blend_reference

packs = np.load('/tmp/gsplat_fixtures/packs.npy')
offsets = np.load('/tmp/gsplat_fixtures/offsets.npy').astype(np.uint32)
px = np.load('/tmp/gsplat_fixtures/px.npy')
py = np.load('/tmp/gsplat_fixtures/py.npy')
ref = alpha_blend_reference(packs, offsets, px, py, 32, 32)
got = np.load('/tmp/gsplat_fixtures/out.npy')

diff = np.abs(ref - got).max()
print('max diff:', diff)
print('ref center:', ref[16, 16])
print('got center:', got[16, 16])
assert diff < 0.05, f'v1a diff too large: {diff}'
print('v1a single-Gaussian OK')
"
```
Expected: "v1a single-Gaussian OK".

- [ ] **Step 6: Commit**

```bash
git add tt-metal/tt_metal/programming_examples/gaussian_splatting/
git commit -m "compute kernel 11c: full single-Gaussian math chain (v1a)"
```

---

## Phase 3 — v1b: Full Scene, Single Core

### Task 3.1: Compute kernel — full per-Gaussian loop with state CBs and sat_mask

This is the biggest single task. Extend v1a to a full Gaussian loop with running accumulators in L1 state CBs and Stage F sat_mask refresh.

**Files:**
- Modify: `tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp`
- Modify: `tt-metal/tt_metal/programming_examples/gaussian_splatting/alpha_blend.cpp` (add state/scratch CBs)

- [ ] **Step 1: Add remaining CBs to host**

Open `alpha_blend.cpp`, add before the reader/compute/writer kernel creation:

```cpp
    // State CBs (depth 1, persistent across Gaussian loop)
    cb_tile(CB_COLOR_R_STATE, 1);
    cb_tile(CB_COLOR_G_STATE, 1);
    cb_tile(CB_COLOR_B_STATE, 1);
    cb_tile(CB_T_STATE, 1);
    cb_tile(CB_SAT_MASK, 1);

    // Remaining scratch
    cb_tile(CB_WEIGHT);
    cb_tile(CB_CONTRIB);
    cb_tile(CB_ONE_MINUS_ALPHA);
    cb_tile(CB_T_TMP);
```

- [ ] **Step 2: Replace the compute kernel with full v1b version**

Replace `alpha_blend_compute.cpp` with:

```cpp
#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/fill.h"
#include "compute_kernel_api/eltwise_unary/rsub.h"
#include "compute_kernel_api/eltwise_unary/comp.h"
#include "compute_kernel_api/binary_max_min.h"

namespace NAMESPACE {

constexpr uint32_t CB_PX = 0;
constexpr uint32_t CB_PY = 1;
constexpr uint32_t CB_SCALARS = 2;
constexpr uint32_t CB_TILE_META = 3;
constexpr uint32_t CB_DX = 4;
constexpr uint32_t CB_DY = 5;
constexpr uint32_t CB_DX2 = 6;
constexpr uint32_t CB_DY2 = 7;
constexpr uint32_t CB_DXDY = 8;
constexpr uint32_t CB_Q = 9;
constexpr uint32_t CB_POWER = 10;
constexpr uint32_t CB_WEIGHT = 11;
constexpr uint32_t CB_ALPHA = 12;
constexpr uint32_t CB_CONTRIB = 13;
constexpr uint32_t CB_ONE_MINUS_ALPHA = 14;
constexpr uint32_t CB_T_TMP = 15;
constexpr uint32_t CB_COLOR_OUT = 16;
constexpr uint32_t CB_COLOR_R_STATE = 17;
constexpr uint32_t CB_COLOR_G_STATE = 18;
constexpr uint32_t CB_COLOR_B_STATE = 19;
constexpr uint32_t CB_T_STATE = 20;
constexpr uint32_t CB_SAT_MASK = 21;
constexpr uint32_t CB_CONST_ZERO = 22;
constexpr uint32_t CB_CONST_099 = 23;

inline uint32_t as_u32(float x) {
    uint32_t u;
    __builtin_memcpy(&u, &x, 4);
    return u;
}

// Fill a state CB with a scalar value and keep it at front (do not pop).
inline void fill_state_cb(uint32_t cb_id, float value) {
    cb_reserve_back(cb_id, 1);
    tile_regs_acquire();
    fill_tile(0, value);
    tile_regs_commit(); tile_regs_wait();
    pack_tile(0, cb_id);
    tile_regs_release();
    cb_push_back(cb_id, 1);
    cb_wait_front(cb_id, 1);
}

// Reload a state CB (pop front, re-push with new value from Dst slot via pack_tile).
// For an in-place accumulator update: compute new value in Dst, then drain old + push new.
inline void spill_dst_to_state_cb(uint32_t dst_idx, uint32_t cb_id) {
    // Caller must have committed/waited Dst. We pop old, reserve new, pack, push.
    cb_pop_front(cb_id, 1);
    cb_reserve_back(cb_id, 1);
    pack_tile(dst_idx, cb_id);
    cb_push_back(cb_id, 1);
    cb_wait_front(cb_id, 1);
}

void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    binary_op_init_common(CB_PX, CB_PY, CB_COLOR_OUT);
    binop_with_scalar_tile_init();
    exp_tile_init();
    copy_tile_init(CB_PX);
    mul_tiles_init(CB_PX, CB_PY);
    add_tiles_init(CB_PX, CB_PY);
    rsub_unary_tile_init();
    unary_ge_tile_init();
    binary_max_min_tile_init();

    // Pre-fill constant CBs
    fill_state_cb(CB_CONST_ZERO, 0.0f);
    fill_state_cb(CB_CONST_099, 0.99f);

    for (uint32_t t = 0; t < num_tiles; t++) {
        // Init per-tile state
        fill_state_cb(CB_COLOR_R_STATE, 0.0f);
        fill_state_cb(CB_COLOR_G_STATE, 0.0f);
        fill_state_cb(CB_COLOR_B_STATE, 0.0f);
        fill_state_cb(CB_T_STATE, 1.0f);
        fill_state_cb(CB_SAT_MASK, 1.0f);

        cb_wait_front(CB_TILE_META, 1);
        uint32_t g_count = *reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_TILE_META));
        cb_pop_front(CB_TILE_META, 1);

        cb_wait_front(CB_PX, 1);
        cb_wait_front(CB_PY, 1);

        for (uint32_t g = 0; g < g_count; g++) {
            // Stage F: refresh sat_mask every 16 Gaussians (but not at g=0)
            if ((g & 15) == 0 && g > 0) {
                tile_regs_acquire();
                copy_tile(CB_T_STATE, 0, 0);
                unary_ge_tile(0, as_u32(1e-4f));
                tile_regs_commit(); tile_regs_wait();
                spill_dst_to_state_cb(0, CB_SAT_MASK);
                tile_regs_release();
            }

            // Stage A: decode 9 scalars
            cb_wait_front(CB_SCALARS, 1);
            volatile float* pack = reinterpret_cast<volatile float*>(get_read_ptr(CB_SCALARS));
            float mean_x = pack[0];
            float mean_y = pack[1];
            float cov_a = pack[2];
            float two_cov_b = pack[3];
            float cov_c = pack[4];
            float color_r = pack[5];
            float color_g = pack[6];
            float color_b = pack[7];
            float opacity = pack[8];

            // Stage B1: dx, dy
            tile_regs_acquire();
            copy_tile(CB_PX, 0, 0); sub_unary_tile(0, as_u32(mean_x));
            copy_tile(CB_PY, 0, 1); sub_unary_tile(1, as_u32(mean_y));
            tile_regs_commit(); tile_regs_wait();
            cb_reserve_back(CB_DX, 1); pack_tile(0, CB_DX); cb_push_back(CB_DX, 1);
            cb_reserve_back(CB_DY, 1); pack_tile(1, CB_DY); cb_push_back(CB_DY, 1);
            tile_regs_release();
            cb_wait_front(CB_DX, 1);
            cb_wait_front(CB_DY, 1);

            // Stage B2: dx², dy², dx·dy (three separate acquire blocks)
            tile_regs_acquire();
            mul_tiles(CB_DX, CB_DX, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            cb_reserve_back(CB_DX2, 1); pack_tile(0, CB_DX2); cb_push_back(CB_DX2, 1);
            tile_regs_release();

            tile_regs_acquire();
            mul_tiles(CB_DY, CB_DY, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            cb_reserve_back(CB_DY2, 1); pack_tile(0, CB_DY2); cb_push_back(CB_DY2, 1);
            tile_regs_release();

            tile_regs_acquire();
            mul_tiles(CB_DX, CB_DY, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            cb_reserve_back(CB_DXDY, 1); pack_tile(0, CB_DXDY); cb_push_back(CB_DXDY, 1);
            tile_regs_release();

            cb_wait_front(CB_DX2, 1);
            cb_wait_front(CB_DY2, 1);
            cb_wait_front(CB_DXDY, 1);

            // Stage B3: scale each by cov, sum, apply -0.5, min(.,0)
            tile_regs_acquire();
            copy_tile(CB_DX2, 0, 0); mul_unary_tile(0, as_u32(cov_a));
            copy_tile(CB_DY2, 0, 1); mul_unary_tile(1, as_u32(cov_c));
            copy_tile(CB_DXDY, 0, 2); mul_unary_tile(2, as_u32(two_cov_b));
            tile_regs_commit(); tile_regs_wait();
            cb_reserve_back(CB_Q, 3);
            pack_tile(0, CB_Q); pack_tile(1, CB_Q); pack_tile(2, CB_Q);
            cb_push_back(CB_Q, 3);
            tile_regs_release();
            cb_wait_front(CB_Q, 3);

            tile_regs_acquire();
            add_tiles(CB_Q, CB_Q, 0, 1, 0);  // dst[0] = a·dx² + c·dy²
            tile_regs_commit(); tile_regs_wait();
            cb_reserve_back(CB_POWER, 1); pack_tile(0, CB_POWER); cb_push_back(CB_POWER, 1);
            tile_regs_release();
            cb_wait_front(CB_POWER, 1);

            tile_regs_acquire();
            add_tiles(CB_POWER, CB_Q, 0, 2, 0);  // + 2b·dxdy
            mul_unary_tile(0, as_u32(-0.5f));    // Q -> -0.5·Q
            copy_tile(CB_CONST_ZERO, 0, 1);
            binary_min_tile(0, 1, 0);            // min(power, 0)
            exp_tile(0);
            mul_unary_tile(0, as_u32(opacity));
            copy_tile(CB_CONST_099, 0, 1);
            binary_min_tile(0, 1, 0);            // alpha = min(opacity·weight, 0.99)
            tile_regs_commit(); tile_regs_wait();
            cb_reserve_back(CB_ALPHA, 1); pack_tile(0, CB_ALPHA); cb_push_back(CB_ALPHA, 1);
            tile_regs_release();
            cb_pop_front(CB_POWER, 1);
            cb_pop_front(CB_Q, 3);
            cb_pop_front(CB_DX, 1); cb_pop_front(CB_DY, 1);
            cb_pop_front(CB_DX2, 1); cb_pop_front(CB_DY2, 1); cb_pop_front(CB_DXDY, 1);
            cb_wait_front(CB_ALPHA, 1);

            // Stage D1: contrib = alpha · T · sat_mask
            tile_regs_acquire();
            mul_tiles(CB_ALPHA, CB_T_STATE, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            cb_reserve_back(CB_T_TMP, 1); pack_tile(0, CB_T_TMP); cb_push_back(CB_T_TMP, 1);
            tile_regs_release();
            cb_wait_front(CB_T_TMP, 1);

            tile_regs_acquire();
            mul_tiles(CB_T_TMP, CB_SAT_MASK, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            cb_reserve_back(CB_CONTRIB, 1); pack_tile(0, CB_CONTRIB); cb_push_back(CB_CONTRIB, 1);
            tile_regs_release();
            cb_pop_front(CB_T_TMP, 1);
            cb_wait_front(CB_CONTRIB, 1);

            // Stage D2: R/G/B += contrib · color_c (three separate acquire blocks for simplicity)
            // R
            tile_regs_acquire();
            copy_tile(CB_CONTRIB, 0, 0); mul_unary_tile(0, as_u32(color_r));
            copy_tile(CB_COLOR_R_STATE, 0, 1);
            add_tiles(CB_CONTRIB, CB_COLOR_R_STATE, 0, 0, 2);  // dummy; will replace with true add
            tile_regs_commit(); tile_regs_wait();
            // Note: the cleanest way with add_tiles (needs CB operands): pack dst[0] to scratch
            // then add_tiles(CB_SCRATCH_R, CB_COLOR_R_STATE). We'll use CB_T_TMP as scratch.
            cb_reserve_back(CB_T_TMP, 1); pack_tile(0, CB_T_TMP); cb_push_back(CB_T_TMP, 1);
            tile_regs_release();
            cb_wait_front(CB_T_TMP, 1);
            tile_regs_acquire();
            add_tiles(CB_COLOR_R_STATE, CB_T_TMP, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            spill_dst_to_state_cb(0, CB_COLOR_R_STATE);
            tile_regs_release();
            cb_pop_front(CB_T_TMP, 1);

            // G
            tile_regs_acquire();
            copy_tile(CB_CONTRIB, 0, 0); mul_unary_tile(0, as_u32(color_g));
            tile_regs_commit(); tile_regs_wait();
            cb_reserve_back(CB_T_TMP, 1); pack_tile(0, CB_T_TMP); cb_push_back(CB_T_TMP, 1);
            tile_regs_release();
            cb_wait_front(CB_T_TMP, 1);
            tile_regs_acquire();
            add_tiles(CB_COLOR_G_STATE, CB_T_TMP, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            spill_dst_to_state_cb(0, CB_COLOR_G_STATE);
            tile_regs_release();
            cb_pop_front(CB_T_TMP, 1);

            // B
            tile_regs_acquire();
            copy_tile(CB_CONTRIB, 0, 0); mul_unary_tile(0, as_u32(color_b));
            tile_regs_commit(); tile_regs_wait();
            cb_reserve_back(CB_T_TMP, 1); pack_tile(0, CB_T_TMP); cb_push_back(CB_T_TMP, 1);
            tile_regs_release();
            cb_wait_front(CB_T_TMP, 1);
            tile_regs_acquire();
            add_tiles(CB_COLOR_B_STATE, CB_T_TMP, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            spill_dst_to_state_cb(0, CB_COLOR_B_STATE);
            tile_regs_release();
            cb_pop_front(CB_T_TMP, 1);

            cb_pop_front(CB_CONTRIB, 1);

            // Stage E: T ← T · (1 - alpha) · sat_mask
            tile_regs_acquire();
            copy_tile(CB_ALPHA, 0, 0);
            rsub_unary_tile(0, as_u32(1.0f));  // dst[0] = 1 - alpha
            tile_regs_commit(); tile_regs_wait();
            cb_reserve_back(CB_ONE_MINUS_ALPHA, 1);
            pack_tile(0, CB_ONE_MINUS_ALPHA);
            cb_push_back(CB_ONE_MINUS_ALPHA, 1);
            tile_regs_release();
            cb_wait_front(CB_ONE_MINUS_ALPHA, 1);

            tile_regs_acquire();
            mul_tiles(CB_T_STATE, CB_ONE_MINUS_ALPHA, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            cb_reserve_back(CB_T_TMP, 1); pack_tile(0, CB_T_TMP); cb_push_back(CB_T_TMP, 1);
            tile_regs_release();
            cb_pop_front(CB_ONE_MINUS_ALPHA, 1);
            cb_wait_front(CB_T_TMP, 1);

            tile_regs_acquire();
            mul_tiles(CB_T_TMP, CB_SAT_MASK, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            spill_dst_to_state_cb(0, CB_T_STATE);
            tile_regs_release();
            cb_pop_front(CB_T_TMP, 1);

            cb_pop_front(CB_ALPHA, 1);
            cb_pop_front(CB_SCALARS, 1);
        }

        // Pack R/G/B state -> output
        cb_reserve_back(CB_COLOR_OUT, 3);
        tile_regs_acquire();
        copy_tile(CB_COLOR_R_STATE, 0, 0);
        copy_tile(CB_COLOR_G_STATE, 0, 1);
        copy_tile(CB_COLOR_B_STATE, 0, 2);
        tile_regs_commit(); tile_regs_wait();
        pack_tile(0, CB_COLOR_OUT);
        pack_tile(1, CB_COLOR_OUT);
        pack_tile(2, CB_COLOR_OUT);
        cb_push_back(CB_COLOR_OUT, 3);
        tile_regs_release();

        // Drain state CBs for next tile
        cb_pop_front(CB_COLOR_R_STATE, 1);
        cb_pop_front(CB_COLOR_G_STATE, 1);
        cb_pop_front(CB_COLOR_B_STATE, 1);
        cb_pop_front(CB_T_STATE, 1);
        cb_pop_front(CB_SAT_MASK, 1);
        cb_pop_front(CB_PX, 1);
        cb_pop_front(CB_PY, 1);
    }
}

}  // namespace NAMESPACE
```

- [ ] **Step 3: Build**

Run:
```bash
cd /localdev/vkovinic/gsplat_tt
source tt-metal/python_env/bin/activate
sudo ./tt-metal/build_metal.sh
```
Expected: builds cleanly.

- [ ] **Step 4: Run single_gaussian fixture (regression on v1a)**

Run:
```bash
./tt-metal/build/programming_examples/metal_example_gaussian_splatting \
    /tmp/gsplat_fixtures/packs.npy /tmp/gsplat_fixtures/offsets.npy \
    /tmp/gsplat_fixtures/px.npy /tmp/gsplat_fixtures/py.npy \
    /tmp/gsplat_fixtures/out.npy 32 32
```
Then:
```bash
python -c "
import numpy as np
from scripts.numeric_sanity import alpha_blend_reference
packs = np.load('/tmp/gsplat_fixtures/packs.npy')
offsets = np.load('/tmp/gsplat_fixtures/offsets.npy').astype(np.uint32)
px, py = np.load('/tmp/gsplat_fixtures/px.npy'), np.load('/tmp/gsplat_fixtures/py.npy')
ref = alpha_blend_reference(packs, offsets, px, py, 32, 32)
got = np.load('/tmp/gsplat_fixtures/out.npy')
assert np.abs(ref - got).max() < 0.05
print('v1b regress on v1a OK')
"
```
Expected: "v1b regress on v1a OK".

- [ ] **Step 5: Commit**

```bash
git add tt-metal/tt_metal/programming_examples/gaussian_splatting/
git commit -m "v1b compute kernel: full Gaussian loop + state CBs + sat_mask"
```

---

### Task 3.2: T0.5 — Two-Gaussian blend fixture + test

- [ ] **Step 1: Add two-Gaussian fixture to dumper**

Add to `scripts/dump_kernel_inputs.py`:

```python
def dump_two_gaussian_blend(out_dir):
    """T0.5: red (front, α=0.5) + blue (back, α=0.5) at same pixel."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    packs = np.array([
        [16.5, 16.5, 100.0, 0.0, 100.0, 1.0, 0.0, 0.0, 0.5],
        [16.5, 16.5, 100.0, 0.0, 100.0, 0.0, 0.0, 1.0, 0.5],
    ], dtype=np.float32)
    offsets = np.array([0, 2], dtype=np.float32)
    px = np.empty((1, 32, 32), dtype=np.float32)
    py = np.empty((1, 32, 32), dtype=np.float32)
    for i in range(32):
        for j in range(32):
            px[0, i, j] = j + 0.5
            py[0, i, j] = i + 0.5
    np.save(f"{out_dir}/packs.npy", packs)
    np.save(f"{out_dir}/offsets.npy", offsets)
    np.save(f"{out_dir}/px.npy", px)
    np.save(f"{out_dir}/py.npy", py)
```

And register it in the `if args.fixture == ...` chain.

- [ ] **Step 2: Run fixture and test**

```bash
python scripts/dump_kernel_inputs.py --fixture two_gaussian_blend --out-dir /tmp/gsplat_t05
./tt-metal/build/programming_examples/metal_example_gaussian_splatting \
    /tmp/gsplat_t05/packs.npy /tmp/gsplat_t05/offsets.npy \
    /tmp/gsplat_t05/px.npy /tmp/gsplat_t05/py.npy \
    /tmp/gsplat_t05/out.npy 32 32
python -c "
import numpy as np
img = np.load('/tmp/gsplat_t05/out.npy')
print('center:', img[16, 16])
assert abs(img[16, 16, 0] - 0.5) < 0.05
assert abs(img[16, 16, 2] - 0.25) < 0.05
print('T0.5 two-Gaussian blend OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/dump_kernel_inputs.py
git commit -m "T0.5 two-Gaussian blend fixture + verify"
```

---

### Task 3.3: T0.6 — Saturation test

- [ ] **Step 1: Add saturation fixture**

Add to `scripts/dump_kernel_inputs.py`:

```python
def dump_saturation(out_dir):
    """T0.6: 50 opaque Gaussians → sat_mask kicks in."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    one = np.array([[16.5, 16.5, 100.0, 0.0, 100.0, 0.5, 0.5, 0.5, 0.99]], dtype=np.float32)
    packs = np.tile(one, (50, 1))
    offsets = np.array([0, 50], dtype=np.float32)
    px = np.empty((1, 32, 32), dtype=np.float32)
    py = np.empty((1, 32, 32), dtype=np.float32)
    for i in range(32):
        for j in range(32):
            px[0, i, j] = j + 0.5
            py[0, i, j] = i + 0.5
    np.save(f"{out_dir}/packs.npy", packs)
    np.save(f"{out_dir}/offsets.npy", offsets)
    np.save(f"{out_dir}/px.npy", px)
    np.save(f"{out_dir}/py.npy", py)
```

- [ ] **Step 2: Run**

```bash
python scripts/dump_kernel_inputs.py --fixture saturation --out-dir /tmp/gsplat_t06
./tt-metal/build/programming_examples/metal_example_gaussian_splatting \
    /tmp/gsplat_t06/packs.npy /tmp/gsplat_t06/offsets.npy \
    /tmp/gsplat_t06/px.npy /tmp/gsplat_t06/py.npy \
    /tmp/gsplat_t06/out.npy 32 32
python -c "
import numpy as np
img = np.load('/tmp/gsplat_t06/out.npy')
print('center:', img[16, 16])
# Expected ~0.5 (color converges to single-Gaussian color)
assert 0.4 < img[16, 16, 0] < 0.6
print('T0.6 saturation OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/dump_kernel_inputs.py
git commit -m "T0.6 saturation fixture + verify"
```

---

### Task 3.4: Full-scene end-to-end PSNR test

Dump a realistic scene from Python (via `project_gaussians` + `prepare_kernel_inputs`) and compare kernel output to CPU reference.

**Files:**
- Create: `tests/test_kernel_integration.py`

- [ ] **Step 1: Write the integration test**

Create `tests/test_kernel_integration.py`:

```python
"""End-to-end: full-scene kernel vs CPU reference, PSNR/SSIM."""
import os
import subprocess
import tempfile

import numpy as np
import pytest
import torch

from rasterization import (
    project_gaussians, get_tile_assignments, sort_and_bin,
    alpha_blend, prepare_kernel_inputs,
)


KERNEL_BINARY = os.environ.get(
    "GSPLAT_KERNEL",
    "tt-metal/build/programming_examples/metal_example_gaussian_splatting",
)


def _psnr(a, b):
    mse = np.mean((a - b) ** 2)
    if mse <= 0:
        return 100.0
    return -10.0 * np.log10(mse)


@pytest.mark.skipif(
    not os.path.exists(KERNEL_BINARY),
    reason="kernel binary not built; run sudo ./build_metal.sh",
)
def test_full_scene_psnr():
    torch.manual_seed(42)
    H, W = 64, 64  # 2x2 tiles
    N = 50
    means = torch.rand(N, 3) * torch.tensor([2.0, 2.0, 0.0]) + torch.tensor([-1.0, -1.0, 1.0])
    scales = torch.rand(N, 3) * 0.1 + 0.02
    q = torch.randn(N, 4); q = q / q.norm(dim=-1, keepdim=True)
    colors = torch.rand(N, 3)
    opacities = torch.rand(N) * 0.5 + 0.2

    extrinsics = torch.eye(4)
    fx = fy = 40.0
    intrinsics = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=torch.float32)

    means_2d, covs_2d, depths, radii, valid = project_gaussians(
        means, scales, q, extrinsics, intrinsics, H, W,
    )
    if valid.sum().item() == 0:
        pytest.skip("no visible Gaussians — random seed; reroll")

    colors_v = colors[valid]
    opacities_v = opacities[valid]

    gids, tids, _ = get_tile_assignments(means_2d, radii, H, W, tile_size=32)
    tiles_x = (W + 31) // 32
    tiles_y = (H + 31) // 32
    sorted_gids, tile_ranges = sort_and_bin(gids, tids, depths, tiles_x, tiles_y)

    # CPU reference
    cpu_img = alpha_blend(
        means_2d, covs_2d, colors_v, opacities_v,
        sorted_gids, tile_ranges, H, W, tile_size=32,
    ).numpy()

    # Kernel inputs
    packs, offsets, px, py = prepare_kernel_inputs(
        means_2d, covs_2d, colors_v, opacities_v,
        sorted_gids, tile_ranges, H, W,
    )

    with tempfile.TemporaryDirectory() as td:
        np.save(f"{td}/packs.npy", packs)
        np.save(f"{td}/offsets.npy", offsets.astype(np.float32))
        np.save(f"{td}/px.npy", px)
        np.save(f"{td}/py.npy", py)
        subprocess.run([
            KERNEL_BINARY,
            f"{td}/packs.npy", f"{td}/offsets.npy",
            f"{td}/px.npy", f"{td}/py.npy",
            f"{td}/out.npy", str(H), str(W),
        ], check=True, capture_output=True)
        kernel_img = np.load(f"{td}/out.npy")

    psnr = _psnr(cpu_img, kernel_img)
    print(f"PSNR: {psnr:.2f} dB")
    assert psnr >= 35.0, f"PSNR too low: {psnr:.2f} dB (want ≥ 35)"

    try:
        from skimage.metrics import structural_similarity as ssim
        ssim_val = ssim(cpu_img, kernel_img, channel_axis=2, data_range=1.0)
        print(f"SSIM: {ssim_val:.4f}")
        assert ssim_val >= 0.98, f"SSIM too low: {ssim_val:.4f}"
    except ImportError:
        print("scikit-image not installed; skipping SSIM")
```

- [ ] **Step 2: Install scikit-image if needed**

```bash
source venv/bin/activate
pip install scikit-image
```

- [ ] **Step 3: Run the test**

```bash
cd /localdev/vkovinic/gsplat_tt
source venv/bin/activate
source tt-metal/python_env/bin/activate  # for binary runtime env
pytest tests/test_kernel_integration.py -v -s
```
Expected: PSNR ≥ 35 dB, SSIM ≥ 0.98.

- [ ] **Step 4: If PSNR < 35 dB, debug**

Common causes:
1. Coord convention mismatch (tile-local vs global). Check: pick a pixel, compute `dx` by hand, compare to reader output via DPRINT.
2. `two_cov_b` not doubled on host. Check: `prepare_kernel_inputs` line that writes index 3.
3. Sat_mask refresh timing. Check: toggle refresh at bottom of loop and compare.
4. Missing init call (e.g., `unary_ge_tile_init()`). Check kernel bottom-of-file.

- [ ] **Step 5: Commit**

```bash
git add tests/test_kernel_integration.py
git commit -m "v1b full-scene end-to-end integration test (PSNR/SSIM)"
```

---

### Task 3.5: 640×640 performance baseline

- [ ] **Step 1: Add timing + larger scene to the integration test**

Add a second test function to `tests/test_kernel_integration.py`:

```python
import time


@pytest.mark.skipif(
    not os.path.exists(KERNEL_BINARY),
    reason="kernel binary not built",
)
def test_640_perf_baseline(capsys):
    torch.manual_seed(0)
    H, W = 640, 640
    N = 10_000  # 10K Gaussians
    # ... (same setup as test_full_scene_psnr but bigger) ...
    # ... launch binary with H=640 W=640 and time wall clock ...
    # ... expect < 2s single-core for thesis baseline ...
```

(Full code structure follows `test_full_scene_psnr`; scale up N and image size; assert `elapsed < 2.0` as a sanity bound.)

- [ ] **Step 2: Run and record the baseline**

```bash
pytest tests/test_kernel_integration.py::test_640_perf_baseline -v -s
```
Record the elapsed time — this is the v1b baseline for thesis benchmarks.

- [ ] **Step 3: Commit**

```bash
git add tests/test_kernel_integration.py
git commit -m "v1b 640x640 perf baseline test"
```

**v1b checkpoint**: if PSNR ≥ 35 dB + SSIM ≥ 0.98 + baseline measured, v1b is **shippable as thesis deliverable**. Proceed to Phase 4 only if time allows.

---

## Phase 4 — v1c: Multi-Core (Stretch)

### Task 4.1: Multi-core host — naive contiguous split

- [ ] **Step 1: Use `split_work_to_cores` in the host**

In `alpha_blend.cpp`, replace the single-core setup with:

```cpp
    auto grid = mesh_device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, group1, group2, work_per_core_1, work_per_core_2] =
        tt::tt_metal::split_work_to_cores(grid, num_tiles);

    // Create CBs across all_cores (single call, not per-core).
    // ... (change cb_tile helpers to take `all_cores` as the CoreRangeSet) ...

    // Create kernels on all_cores.
    // ... (change CreateKernel(..., core, ...) -> CreateKernel(..., all_cores, ...)) ...

    // Per-core runtime args.
    uint32_t tile_offset = 0;
    auto set_args_for_group = [&](const auto& group, uint32_t work_per_core) {
        for (auto range : group.ranges()) {
            for (auto x = range.start_coord.x; x <= range.end_coord.x; x++) {
                for (auto y = range.start_coord.y; y <= range.end_coord.y; y++) {
                    CoreCoord core{x, y};
                    SetRuntimeArgs(program, reader, core, {
                        packs_dram->address(), offsets_dram->address(),
                        px_dram->address(), py_dram->address(),
                        tile_offset, work_per_core,
                    });
                    SetRuntimeArgs(program, compute, core, {work_per_core});
                    SetRuntimeArgs(program, writer, core, {
                        out_dram->address(), tile_offset, work_per_core,
                    });
                    tile_offset += work_per_core;
                }
            }
        }
    };
    set_args_for_group(group1, work_per_core_1);
    set_args_for_group(group2, work_per_core_2);
```

- [ ] **Step 2: Build, run integration test**

Run:
```bash
sudo ./tt-metal/build_metal.sh
pytest tests/test_kernel_integration.py -v -s
```
Expected: PSNR unchanged; wall-clock speedup visible.

- [ ] **Step 3: Record speedup**

- [ ] **Step 4: Commit**

```bash
git add tt-metal/tt_metal/programming_examples/gaussian_splatting/alpha_blend.cpp
git commit -m "v1c multi-core with naive contiguous split"
```

---

### Task 4.2: LPT load balancing

Tiles vary wildly in Gaussian count. Sort descending and greedily assign to the least-loaded core.

- [ ] **Step 1: Add LPT utility to host**

```cpp
// Sort tile IDs descending by count, then greedy bin-pack across num_cores.
// Returns per-core tile-ID list.
static std::vector<std::vector<uint32_t>> lpt_assign(
    const std::vector<uint32_t>& tile_offsets, uint32_t num_tiles, uint32_t num_cores
) {
    std::vector<std::pair<uint32_t, uint32_t>> cost_id;
    for (uint32_t t = 0; t < num_tiles; t++) {
        cost_id.push_back({tile_offsets[t + 1] - tile_offsets[t], t});
    }
    std::sort(cost_id.begin(), cost_id.end(), std::greater<>());

    std::vector<std::vector<uint32_t>> per_core(num_cores);
    std::vector<uint32_t> core_load(num_cores, 0);
    for (const auto& [cost, id] : cost_id) {
        auto min_it = std::min_element(core_load.begin(), core_load.end());
        uint32_t c = std::distance(core_load.begin(), min_it);
        per_core[c].push_back(id);
        core_load[c] += cost;
    }
    return per_core;
}
```

- [ ] **Step 2: Upload `tile_id_list` per core via DRAM**

Instead of per-core `(first_tile_id, num_tiles)`, upload a per-core tile ID list and pass (list_addr, list_len).

Reader changes: read `tile_id = tile_ids[t]` instead of `first_tile_id + t`.

(Implementation: create one large uint32 DRAM buffer containing all per-core lists concatenated; per-core runtime args give the offset + length.)

- [ ] **Step 3: Build + test**

```bash
sudo ./tt-metal/build_metal.sh
pytest tests/test_kernel_integration.py -v -s
```
Record the new speedup. Target: ≥15×.

- [ ] **Step 4: Commit**

```bash
git add tt-metal/tt_metal/programming_examples/gaussian_splatting/
git commit -m "v1c: LPT load balancing across cores"
```

---

## Phase 5 — Integration + Thesis Deliverables

### Task 5.1: Wire kernel into viewer via subprocess

**Files:**
- Modify: `viewer.py`

- [ ] **Step 1: Add `backend` parameter and dispatch**

In `viewer.py`, add:

```python
import subprocess
import tempfile

def _alpha_blend_kernel(
    means_2d, covs_2d, colors, opacities, sorted_gids, tile_ranges, H, W,
):
    from rasterization import prepare_kernel_inputs
    packs, offsets, px, py = prepare_kernel_inputs(
        means_2d, covs_2d, colors, opacities, sorted_gids, tile_ranges, H, W,
    )
    with tempfile.TemporaryDirectory() as td:
        np.save(f"{td}/packs.npy", packs)
        np.save(f"{td}/offsets.npy", offsets.astype(np.float32))
        np.save(f"{td}/px.npy", px)
        np.save(f"{td}/py.npy", py)
        subprocess.run([
            "tt-metal/build/programming_examples/metal_example_gaussian_splatting",
            f"{td}/packs.npy", f"{td}/offsets.npy",
            f"{td}/px.npy", f"{td}/py.npy",
            f"{td}/out.npy", str(H), str(W),
        ], check=True, capture_output=True)
        return torch.from_numpy(np.load(f"{td}/out.npy"))
```

Add a `--backend {cpu,kernel}` CLI arg in `main.py`, pass to `GaussianViewer`, dispatch in `_render_fn`.

- [ ] **Step 2: Smoke test viewer with kernel backend**

```bash
python main.py scene/banana.ply --backend kernel
```
Open browser, orbit camera, confirm no NaN/black frames.

- [ ] **Step 3: Commit**

```bash
git add viewer.py main.py
git commit -m "integrate kernel backend into interactive viewer"
```

---

### Task 5.2: Thesis benchmark table

**Files:**
- Create: `scripts/thesis_benchmarks.py`

- [ ] **Step 1: Script to compare CPU / v1b / v1c across 2 scenes at 640×640**

Create the script that runs each backend and saves a CSV with `scene,backend,mean_ms,p95_ms,psnr,ssim`.

- [ ] **Step 2: Run on 2 scenes**

```bash
python scripts/thesis_benchmarks.py --scenes scene/banana.ply scene/room.ply --out thesis_results.csv
```

- [ ] **Step 3: Commit results**

```bash
git add scripts/thesis_benchmarks.py thesis_results.csv
git commit -m "thesis benchmark table: CPU vs kernel across 2 scenes"
```

---

## Self-Review Notes

- **Spec coverage**: every Rev 3 decision is represented: tile_size=32 (host code), host pre-gather (`prepare_kernel_inputs`), single-core then multi-core (Phase 2-3 then Phase 4), `fp32_dest_acc_en`+HiFi3 (ComputeConfig), sentinel-mask (Stage F), tiled bf16 output + untilize in host, location under `programming_examples/gaussian_splatting/` with `.gitignore` exception.
- **Placeholder scan**: Tasks 4.2 and 5.2 contain brief outlines rather than full code because the exact shape of LPT load balancing and benchmark CSV format depends on v1b measurements. These are "do this concrete thing, with this structure" — not TBD.
- **Type consistency**: CB indices come from `alpha_blend_host.h`; kernel files redefine them as `constexpr` matching the header. If you change a CB index, change both places.
- **Known gap**: the LPT multi-core implementation in Task 4.2 is sketched not coded. If you're executing this plan, expect Task 4.2 to require its own mini-design step before writing code.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-18-alpha-blend-kernel.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration across Phase 0-5. Good for a multi-day execution schedule.

**2. Inline Execution** — Execute tasks in this session using `executing-plans` skill, batch execution with checkpoints at Phase boundaries. Good for focused single-session bring-up of Phase 0-1.

**Which approach?**
