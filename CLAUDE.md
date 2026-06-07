# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project overview

MSc thesis project: forward-pass rasterization of 3D Gaussian Splatting (3DGS)
on Tenstorrent Wormhole/Blackhole hardware via custom tt-metal kernels.

**Scope:** inference / rendering only — load a pre-trained `.ply` and render
it. No training, backward pass, or differentiable rasterization.

## Repository layout

```
gsplat_tt/
├── pyproject.toml                       # `pip install -e .` makes `gsplat` importable
├── setup.sh                             # one-shot bootstrap (venv, vendor tt-metal, build)
├── README.md, CLAUDE.md, requirements.txt, conftest.py, .gitignore
├── docs/
│   ├── thesis_plan.md                   # private working doc (gitignored)
│   └── plan_progress.md                 # design decisions + KI-1/KI-2 history
├── scenes/                              # .ply scenes (only luigi.ply tracked)
├── tests/                               # pytest suite
├── scripts/                             # one-off dev helpers
├── gsplat/                              # importable Python package (CPU pipeline + viewer)
│   ├── __main__.py                      # CLI entry — `python -m gsplat ...` or `gsplat ...`
│   ├── rasterization.py                 # project, tile, sort, alpha_blend, prepare_kernel_inputs
│   ├── viewer.py                        # interactive viewer (viser + nerfview)
│   ├── data_structures.py               # Gaussians dataclass
│   ├── loading_gaussians.py             # .ply loader
│   └── utils.py                         # camera helpers (c2w↔w2c, build_covariance_3d)
└── backends/                            # one subdir per hardware target
    ├── README.md                        # how to add a backend
    ├── tt/                              # Tenstorrent (tt-metal) backend
    │   ├── backend.py                   # daemon-subprocess wrapper
    │   └── tt-metal/                    # vendored SDK + our C++ kernels under
    │       └── tt_metal/programming_examples/gaussian_splatting/
    └── cuda/                            # placeholder for future CUDA backend
```

`gsplat/` is the host-side Python; `backends/<arch>/` holds everything that
talks to a specific accelerator. Adding CUDA later means dropping a
`backends/cuda/backend.py` exposing the same `render(...) / close()`
interface as `backends/tt/backend.py`.

## Pipeline stages

```
load_ply → project_gaussians → get_tile_assignments → sort_and_bin
                                                         │
                                                         ↓
                              ┌──────────────────────────┴──────────────────────────┐
                       (CPU)  alpha_blend                  prepare_kernel_inputs (TT)
                                                                  │
                                                                  ↓
                                                      backends.tt.backend.KernelBackend
                                                       (in-process: ttnn.open_device once,
                                                        ttnn.experimental.gaussian_alpha_blend
                                                        per frame — no daemon/IPC)
                                                                  ↓
                                                          tt-metal kernels:
                                                              reader (NCRISC) →
                                                              compute (TRISCs) →
                                                              writer (BRISC)
```

The kernels live in the ttnn op at
`backends/tt/tt-metal/ttnn/cpp/ttnn/operations/experimental/gaussian_splatting/alpha_blend/`
(device/kernels/). The op's program factory ports the old daemon host
orchestration; the per-frame LPT tile→core schedule (`backends/tt/lpt.py`) is
passed as hash-excluded op attributes so ttnn's program cache keeps kernels warm
(one compiled program per resolution).

The kernel side splits into 3 RISC kernels per Tensix core:
- **Reader** (NCRISC, NoC1): DRAM → L1 via circular buffers.
- **Compute** (TRISC0/1/2): SFPU/FPU work, alpha-blend math.
- **Writer** (BRISC, NoC0): L1 → DRAM, packs RGB tiles to the output buffer.

Key tt-metal constants we use:
- `bfloat16` for on-device storage; `fp32_dest_acc_en=true` keeps Dst register accumulation in fp32.
- `HiFi3` (HiFi4 has WH B0 bug #38306).
- 32×32 native tile, 4 SFPU passes per tile (32 lanes wide).
- Wormhole N150 logical grid: 8×8 cores after 1 row harvested.

## Conventions

- 32×32 screen tiles (matches the hardware tile).
- Structure-of-Arrays for device DRAM (separate `packs`, `px`, `py`, `offsets`, `tile_ids` buffers).
- SH degree 0 (3 color floats per Gaussian); higher degrees not implemented.
- Validate the kernel against the CPU reference via PSNR/SSIM (≥35 dB target).
- Render at 480-960px range for interactive use; 4K is the design ceiling.

## Project venv

Everything (viewer, CPU pipeline, tests, the TT backend, AND `ttnn`) runs in
one environment, `./venv`:
- Activate: `source venv/bin/activate`
- Used by: `gsplat ...`, `pytest`, anything in `gsplat/`, `tests/`, `backends/`.

The TT backend drives the device **in-process** via a custom ttnn op
(`ttnn.experimental.gaussian_alpha_blend`), so `ttnn` IS built and installed
into `./venv` (`setup.sh` runs `build_metal.sh` WITHOUT
`--without-python-bindings`, then `pip install -e backends/tt/tt-metal`).
tt-metal's `create_venv.sh` / `python_env` is intentionally NOT used — one
interpreter runs everything. Note `numpy` is pinned `<2` because ttnn requires
it. There is no longer a daemon subprocess. See
`docs/superpowers/specs/2026-06-07-in-process-ttnn-op-backend-design.md`.

## Setup

`./setup.sh` is the single canonical bootstrap. It is idempotent. Steps:

1. Creates `./venv`, installs `requirements.txt`, runs `pip install -e .`.
2. Vendors `tenstorrent/tt-metal` into `backends/tt/tt-metal/` (~5 GB; the
   target dir already contains our tracked kernel subdir, so the script
   clones to a temp location and merges with `cp -rn`).
3. Injects build wiring for our ttnn op into untracked upstream files
   (`ttnn/CMakeLists.txt` add_subdirectory + link lib, `ttnn/sources.cmake`
   nanobind source, `experimental_nanobind.cpp` include + bind call) —
   idempotent, re-applied on every (re)vendor. Only our op subtree is tracked.
4. `sudo ./build_metal.sh --build-programming-examples` (Python bindings ON)
   to compile the C++ libs + the ttnn op into `_ttnn.so`, then
   `pip install -e backends/tt/tt-metal` into `./venv`. `sudo` needed for
   tt-metal's root-owned SFPI / CPM caches.

Pin a tt-metal version with `TT_METAL_REF=v1.2.3 ./setup.sh`.

To rebuild after editing the op host code or kernels
(`gaussian_splatting/alpha_blend/...`):

```bash
sudo ./backends/tt/tt-metal/build_metal.sh --build-programming-examples
```

This recompiles the op into `_ttnn.so` (editable install picks it up, no
reinstall needed). The kernel `.cpp` sources under `device/kernels/` are
JIT-compiled at runtime; the host `.cpp`/`.hpp` need the rebuild above.

## Running

```bash
# CPU viewer
source venv/bin/activate
gsplat scenes/luigi.ply

# TT viewer (Tenstorrent)
source venv/bin/activate
export TT_METAL_HOME=$PWD/backends/tt/tt-metal
export TT_METAL_RUNTIME_ROOT=$PWD/backends/tt/tt-metal
gsplat scenes/bonsai_room.ply --backend tt --max-resolution 960
```

`gsplat` is registered as a console_script in `pyproject.toml`. Equivalent:
`python -m gsplat scenes/luigi.ply ...`.

## Tests

```bash
source venv/bin/activate
pytest tests/
```

Device tests (in-process, skip cleanly with no TT device) in
`tests/test_kernel_integration.py`:
- `test_full_scene_psnr` (64×64, 50 Gaussians) — PSNR ≥35 dB (~52 dB)
- `test_sparse_scene_empty_tiles` (128×128, sparse) — empty-tile zeroing
- `test_640_perf` (640×640, 10K Gaussians) — warm-frame median latency

Plus `tests/test_ttnn_op_cache.py` (program-cache warm-frame invariant),
`tests/test_lpt.py` (LPT scheduler), and `tests/test_numeric_sanity.py`
(pure-Python alpha-blend reference checks).

## Resolved known issues (kept here for context)

- **KI-1: T0.6 saturation** at 50 stacked α=0.99 Gaussians. Edges saturate
  to bf16 0x4790 (= 73728). Synthetic worst-case; real scenes hit
  41-52 dB PSNR. Deferred to v2 if ever needed. See `docs/plan_progress.md`.
- **KI-2: multi-tile dispatch deadlock** at ≥16 active cores with sparse
  tile occupancy. Was a CB deadlock from empty-tile churn. Fixed by
  filtering empty tiles in `compute_lpt_assignment` + pre-zeroing the
  output buffer (`process_frame`). See `docs/plan_progress.md::KI-2`.

## Reference implementations (for cross-checking)

- `hbb1/torch-splatting` — pure PyTorch reference for the CPU path.
- `gsplat` (nerfstudio) — production CUDA + PyTorch fallback.
- `antimatter15/splat` — ~300-line WebGL.
- `graphdeco-inria/diff-gaussian-rasterization` — original CUDA rasterizer.
