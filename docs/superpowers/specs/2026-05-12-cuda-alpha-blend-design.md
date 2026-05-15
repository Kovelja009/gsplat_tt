# CUDA alpha-blend backend — design

**Date:** 2026-05-12
**Status:** Approved (pending user sign-off on this spec)
**Author:** Claude + mstojkovic

## Goal

Add a CUDA backend (`backends/cuda/`) that implements the `Backend.blend(...)`
contract using a custom CUDA kernel, so the same forward-pass rasterization
pipeline that runs on Tenstorrent today can be benchmarked on an NVIDIA GPU
with directly comparable per-stage timings.

Scope is intentionally narrow:

- **In:** GPU alpha-blend (`blend` stage) at parity with the CPU reference
  (PSNR ≥ 35 dB).
- **Out:** training / backward / differentiable rasterization, GPU
  acceleration of `project / tile_assign / sort` (defer until those stages
  measurably dominate), spherical-harmonics degree > 0, fp16/bf16 paths,
  PyTorch autograd integration.

## Non-goals

- This is not a fork of `gsplat` (nerfstudio) or
  `diff-gaussian-rasterization`. We borrow the algorithmic shape (one block
  per 32×32 tile, one thread per pixel, batched cooperative load of
  sorted Gaussians into shared memory), nothing else.
- No attempt to beat the production CUDA rasterizer. Goal is correctness +
  a benchmark-comparable number, not state-of-the-art throughput.

## Constraints / context

1. **CPU-only dev box.** The current development machine has no CUDA
   toolchain and uses `torch==2.10.0+cpu`. Importing `backends.cuda` must
   not raise on this box, and `pip install -e .` must continue to work
   unchanged. The CUDA kernel is compiled JIT only when `CudaBackend()`
   is instantiated, on a host that has CUDA available.
2. **Stage-by-stage benchmark parity with TT.** `KernelBackend` overrides
   only `blend`; CUDA does the same so the per-stage rows in the
   benchmark markdown differ only on the `blend` line.
3. **Sub-timings follow the dotted-key convention** documented in
   `backends/README.md` and `backends/cuda/README.md`. The CUDA backend
   reports `{"upload", "kernel", "kernel.device"}` so that `kernel.device`
   renders nested under `kernel` in the benchmark output.

## Algorithm

Standard 3DGS forward rasterization, mirroring the CPU reference in
`gsplat.rasterization.alpha_blend` and the TT compute kernel in
`alpha_blend_compute.cpp`:

For each 32×32 screen tile, for each pixel, for each Gaussian assigned to
the tile (sorted front-to-back by depth):

```
dx        = px - mean_x
dy        = py - mean_y
power     = -0.5 · (a · dx² + 2b · dx · dy + c · dy²)
weight    = exp(min(power, 0))
alpha     = min(opacity · weight, 0.99)
color    += alpha · T · gaussian_color
T        *= (1 − alpha)
if T < 1e-4: break    # per-pixel early termination
```

Same constants as the TT kernel: alpha clamped at 0.99 (keeps `1-alpha > 0`),
power clamped at 0 (defensive against fp rounding), early-termination
threshold `T < 1e-4`.

## Kernel architecture

### Launch configuration

- Grid: `(tiles_x, tiles_y, 1)` blocks.
- Block: `(32, 32, 1)` threads. One thread = one pixel of one tile.
- Each block resolves which tile it owns from `blockIdx`, reads its
  `(start, end)` slice from `tile_ranges`, and iterates Gaussians in
  batches.

### Inputs (all on device, fp32, SoA)

| Buffer | Shape | dtype | Notes |
|---|---|---|---|
| `means_2d` | (M, 2) | float32 | screen-space centers in pixels |
| `conics` | (M, 3) | float32 | `[cov_inv_a, 2·cov_inv_b, cov_inv_c]` — pre-multiplied off-diagonal as in `prepare_kernel_inputs` |
| `rgba` | (M, 4) | float32 | `[r, g, b, opacity]` — packed for coalesced loads |
| `sorted_gaussian_ids` | (P,) | int32 | output of `sort_and_bin` |
| `tile_ranges` | (num_tiles, 2) | int32 | `[start, end)` indices into `sorted_gaussian_ids` |
| `image` (output) | (H, W, 3) | float32 | written by kernel |

### Per-block flow

```
__shared__ float2  s_means [BATCH];
__shared__ float3  s_conics[BATCH];
__shared__ float4  s_rgba  [BATCH];

tile_id     = blockIdx.y * tiles_x + blockIdx.x;
start, end  = tile_ranges[tile_id];

px = blockIdx.x * 32 + threadIdx.x + 0.5f;
py = blockIdx.y * 32 + threadIdx.y + 0.5f;
T  = 1.0f;  C = (0, 0, 0);

for (g = start; g < end; g += BATCH) {
    // Cooperative load: 1024 threads collectively load BATCH Gaussians
    // from global → shared. BATCH = 256 (one Gaussian per 4 threads).
    if (threadIdx.y * 32 + threadIdx.x < min(BATCH, end - g)) {
        idx = sorted_gaussian_ids[g + tid];
        s_means [tid] = means_2d[idx];
        s_conics[tid] = conics  [idx];
        s_rgba  [tid] = rgba    [idx];
    }
    __syncthreads();

    int n = min(BATCH, end - g);
    for (int j = 0; j < n && T > 1e-4f; j++) {
        // dx, dy, power, weight, alpha, composite, T-update as above
    }
    __syncthreads();

    if (__all_sync(0xffffffff, T < 1e-4f)) break;   // warp-wide early-out
}

image[(py - 0.5) * W + (px - 0.5)] = C;
```

`BATCH = 256` is the starting choice — fits comfortably in shared memory
(<10 KB) and amortizes the global-memory hit over many composite ops.
Tunable later if profiling shows it's wrong.

### Per-pixel early termination

Two mechanisms, the cheaper one as an outer warp-vote and the per-thread
one as an inner short-circuit:

1. **Per-thread `T > 1e-4` guard** in the inner `j` loop — pixels that
   have saturated stop accumulating but still pay the loop-iteration cost.
2. **Warp-wide `__all_sync` vote** at the batch boundary — when every
   thread in the warp has saturated, skip remaining batches entirely.

This mirrors the TT kernel's `sat_mask` discipline, scaled to CUDA's
warp-level primitives.

## Host wrapper

File: `backends/cuda/backend.py`

```python
class CudaBackend(Backend):
    def __init__(self, verbose: bool = False):
        if not torch.cuda.is_available():
            raise RuntimeError("CudaBackend requires a CUDA-capable device")
        self.verbose = verbose
        self._ext = _load_extension()   # JIT compile via cpp_extension.load
        self._device = torch.device("cuda")

    def blend(
        self,
        means_2d, covs_2d, colors, opacities,
        sorted_gaussian_ids, tile_ranges,
        image_height, image_width,
    ) -> tuple[np.ndarray, dict[str, float]]:
        sub: dict[str, float] = {}

        # ---- Stage A: upload + pack to SoA ----
        t = time.perf_counter()
        d_means   = means_2d.to(self._device, dtype=torch.float32, non_blocking=True)
        d_conics  = _pack_conics(covs_2d).to(self._device, non_blocking=True)
        d_rgba    = _pack_rgba(colors, opacities).to(self._device, non_blocking=True)
        d_ids     = sorted_gaussian_ids.to(self._device, dtype=torch.int32, non_blocking=True)
        d_ranges  = tile_ranges.to(self._device, dtype=torch.int32, non_blocking=True)
        sub["upload"] = (time.perf_counter() - t) * 1000.0

        # ---- Stage B: kernel launch + readback ----
        ev_s = torch.cuda.Event(enable_timing=True)
        ev_e = torch.cuda.Event(enable_timing=True)
        t = time.perf_counter()
        ev_s.record()
        d_out = self._ext.alpha_blend(
            d_means, d_conics, d_rgba, d_ids, d_ranges,
            image_height, image_width,
        )
        ev_e.record()
        image = d_out.cpu().numpy()          # forces sync via D2H memcpy
        sub["kernel"]        = (time.perf_counter() - t) * 1000.0  # incl dispatch + exec + D2H
        sub["kernel.device"] = ev_s.elapsed_time(ev_e)              # pure on-device

        return image, sub
```

`_pack_conics` and `_pack_rgba` are small helpers that mirror the math in
`prepare_kernel_inputs` but produce torch tensors directly (no numpy
round-trip).

## File layout

```
backends/cuda/
├── __init__.py              # short package docstring (no top-level imports of .cu)
├── backend.py               # CudaBackend class
├── README.md                # (already exists — minor wording update)
└── kernels/
    ├── alpha_blend.cu       # __global__ kernel + entry function
    ├── alpha_blend.h        # entry-function declaration
    ├── pybind.cpp           # PYBIND11_MODULE wrapper exposing `alpha_blend`
    └── build.py             # torch.utils.cpp_extension.load wrapper, cached
```

`build.py` exposes a single `_load_extension()` that caches the loaded
module on a module-level global so a viewer session with many `blend(...)`
calls pays the JIT compile cost once.

## Registry / config changes

### `backends/__init__.py:23`

Replace the commented placeholder with a guarded import so the registry
gains a `"cuda"` entry only on CUDA-capable boxes:

```python
REGISTRY: dict[str, type[Backend]] = {"cpu": CpuBackend, "tt": KernelBackend}
try:
    from backends.cuda.backend import CudaBackend
    REGISTRY["cuda"] = CudaBackend
except ImportError:
    pass
```

This means `gsplat --backend cuda ...` on a CPU box raises a clear
"unknown backend" error from `get_backend`, which already lists what's
available.

### `backends/cuda/__init__.py`

Replace the "not yet implemented" docstring with a short description of
the new backend. No executable imports at package import time (keeps the
import safe on CPU-only boxes — the `.cu` only loads when `CudaBackend()`
is constructed).

### `requirements-cuda.txt` (new file at repo root)

```
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.10.0+cu121
ninja>=1.11
```

Documented in `README.md` setup section: on a CUDA host, after the
standard `pip install -e .`, run `pip install -r requirements-cuda.txt`
to upgrade torch in-place to the CUDA wheel and pull in `ninja` (used by
`torch.utils.cpp_extension.load`).

### `README.md` table (line 18)

Flip `cuda` row from "planned" to "experimental — JIT compiles on first
use; requires a CUDA-capable GPU".

### `backends/README.md` (line 111)

Same update — replace the "placeholder for an upcoming CUDA
implementation" wording with a paragraph mirroring the TT entry: build
mechanism (JIT via `cpp_extension`), reported sub-timings (`blend.upload
/ blend.kernel / blend.kernel.device`), where source lives.

## Tests

File: `tests/test_cuda_kernel.py`

Two tests, both gated on `torch.cuda.is_available()` via
`pytest.mark.skipif` (mirroring the structure of
`test_kernel_integration.py`):

1. **`test_cuda_psnr_64`** — 64×64, 50 random Gaussians, identical seed
   and scene setup to `test_full_scene_psnr`. Runs the CPU reference and
   the CUDA backend, asserts PSNR ≥ 35 dB. SSIM ≥ 0.98 if scikit-image
   is available (diagnostic-only, like the TT test).

2. **`test_cuda_640_perf`** — 640×640, 10K Gaussians, mirrors
   `test_640_perf_baseline`. Prints CPU reference ms vs CUDA host-wall
   ms vs CUDA on-device ms. Loose ceiling assertion (e.g.
   `kernel.device < 200 ms`) to catch gross regressions; not a fixed
   target.

No daemon-mode equivalent test is needed — there's no equivalent
long-init cost for the CUDA backend (JIT compile is a one-time hit
that's amortized across a viewer session).

## Sub-timing keys (recap)

The backend's `blend(...)` returns this exact dict shape:

```python
{
    "upload":         <H2D + repack ms>,
    "kernel":         <host wall-clock around dispatch+exec+D2H ms>,
    "kernel.device":  <CUDA event elapsed_time ms>,
}
```

`kernel.device` uses the dotted-key convention so the benchmark markdown
nests it under `kernel`. No separate `download` key — the D2H memcpy is
folded into `kernel` via the implicit `.cpu()` sync.

## Risks / edge cases

1. **JIT compile cost.** First `CudaBackend()` construction may take
   30–60 s on a clean cache. Acceptable for interactive use (one-time
   per viewer session). The cached `.so` lives under
   `~/.cache/torch_extensions/` and persists across runs.
2. **Empty tiles.** `tile_ranges[t] = (0, 0)` for tiles with no
   Gaussians. The kernel handles this naturally — the inner Gaussian
   loop runs zero iterations, `C` stays `(0, 0, 0)`, `T` stays `1`, and
   every thread writes black to its pixel at the end. The output buffer
   is allocated by the extension entry function (no pre-zeroing required).
3. **Numeric divergence from CPU reference.** The CPU reference uses
   `np.exp`; the CUDA kernel will use `__expf` (fast-math). Expect a
   few-dB drop from CPU-to-CPU PSNR (which is ~100 dB by construction),
   but should still clear 35 dB easily. If it doesn't, swap to `expf`
   (non-fast).
4. **Large M (visible Gaussian count).** Conics + rgba + means at
   M = 500K is ~12 MB — comfortable on any modern card.
5. **Maximum tile occupancy.** Sparse-tile scenes still launch one block
   per tile (most do nothing). Acceptable at 640×640 (400 tiles) and
   even 1920×1080 (~2K tiles). No need for the LPT-style filtering
   that the TT backend does — CUDA has cheap block-launch overhead.

## Out of scope (deferred)

- GPU `project / tile_assign / sort` overrides.
- fp16/bf16 paths or persistent-CTA kernel variants.
- A fused project-and-blend kernel.
- Autograd / backward pass.
- A "fair-comparison" build mode that strips fast-math.

## Implementation order (for the plan)

1. Stand up the kernel + minimal pybind11 entry; verify it compiles via
   `cpp_extension.load`.
2. Implement `CudaBackend.blend(...)` with no real launch — just a
   `torch.zeros` output — and wire up the registry + tests skeleton.
3. Land the kernel body; iterate until `test_cuda_psnr_64` passes.
4. Add `test_cuda_640_perf`; tune `BATCH` if obvious.
5. Update README + requirements-cuda.txt + backends/cuda/README.md
   wording.
6. Commit and open a PR (or just commit on `main` per project convention).
