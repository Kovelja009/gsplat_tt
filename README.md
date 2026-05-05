# 3D Gaussian Splatting on Tenstorrent Wormhole

MSc thesis project: a forward-pass renderer for [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
running on Tenstorrent's [tt-metal](https://github.com/tenstorrent/tt-metal)
hardware. Loads a pre-trained `.ply` and renders it interactively in the
browser via a viser/nerfview viewer.

The repo ships two backends behind one viewer:

- **CPU**: pure PyTorch reference rasterizer. Slow (1-2 s/frame at 256×256)
  but mathematically clear; serves as the golden reference for kernel correctness.
- **Kernel**: 3-kernel tt-metal pipeline (reader / compute / writer) running
  on 64 Tensix cores of a Wormhole device. ~22× faster than the CPU
  reference on equivalent scenes, ~80 ms/frame at 640×640 on the synthetic
  test, and sub-second/frame on real Mip-NeRF 360 captures.

## Pipeline

```
Load PLY → Project 3D→2D → Tile assignment → Sort by depth → Alpha blend → Display
            └────────── CPU ──────────┘  └────────── kernel or CPU ───────┘
```

| Stage | Where | Description |
|---|---|---|
| Load PLY | CPU | Parse Gaussian attributes (position, scale, rotation, SH colors, opacity); apply activations |
| Project | CPU (PyTorch) | 3D → 2D via Jacobian linearization of perspective transform; frustum + opacity + radii cull |
| Tile assignment | CPU | Find which 32×32 screen tiles each Gaussian's bounding circle overlaps |
| Sort | CPU | Sort (gaussian, tile) pairs by composite key `(tile_id, depth)` |
| Alpha blend | **CPU or kernel** | Per-tile front-to-back compositing; sat-mask early-termination |

## File structure

### Python (host pipeline + viewer)

```
main.py                 — CLI entry (argparse, scene loading, viewer launch)
viewer.py               — GaussianViewer: nerfview/viser wrapper, dispatches to backend
kernel_backend.py       — KernelBackend: long-lived daemon-mode IPC wrapper
rasterization.py        — pipeline stages: project_gaussians, get_tile_assignments,
                          sort_and_bin, alpha_blend (CPU), prepare_kernel_inputs
loading_gaussians.py    — PLY parser; applies exp/sigmoid/SH activations at load time
data_structures.py      — Gaussians dataclass
utils.py                — math: quaternion → rotation, 3D covariance, c2w → w2c
tests/                  — numeric_sanity (NumPy reference) + kernel_integration (PSNR)
scripts/                — fixture dumpers, untilize prototype
```

### tt-metal kernels (Tenstorrent device-side)

```
tt-metal/tt_metal/programming_examples/gaussian_splatting/
├── alpha_blend.cpp                       — host driver (single-shot CLI + daemon mode)
├── alpha_blend_host.h                    — shared CB indices + tile constants
├── kernels/
│   ├── compute/alpha_blend_compute.cpp   — per-tile alpha-blend math (BRISC)
│   └── dataflow/
│       ├── reader_alpha_blend.cpp        — DRAM → CB streaming (NoC0)
│       └── writer_alpha_blend.cpp        — CB → DRAM output (NoC1)
└── CMakeLists.txt
```

## Setup

One-shot bootstrap (idempotent — safe to re-run):

```bash
./setup.sh
```

This:

1. Creates the project venv at `./venv` and installs `requirements.txt`
2. Clones `tenstorrent/tt-metal` into `./tt-metal` (~5 GB)
3. Wires our kernel subdir into tt-metal's CMake build
4. Sets up tt-metal's separate `python_env` (for ttnn / build tools)
5. Builds tt-metal (`sudo ./build_metal.sh`, ~10-20 min on first run)

To pin a specific tt-metal version: `TT_METAL_REF=v1.2.3 ./setup.sh`.

## Running

### CPU viewer (no Tenstorrent device required)

```bash
source venv/bin/activate
python main.py scene/luigi.ply
```

### Kernel-accelerated viewer (Wormhole required)

```bash
source venv/bin/activate
export TT_METAL_HOME=$PWD/tt-metal
export TT_METAL_RUNTIME_ROOT=$PWD/tt-metal
python main.py scene/luigi.ply --backend kernel
```

Then open <http://localhost:8080> in a browser. Orbit the camera with the mouse.

### Useful flags

| Flag | Default | What |
|---|---|---|
| `--backend {cpu,kernel}` | `cpu` | Pick rasterizer |
| `--max-resolution N` | `640` | Cap longest dim of the render (preserves aspect, snaps to multiples of 32). Increase for sharper screenshots, decrease for more interactive drag. |
| `--adaptive-resolution` | off | Enable nerfview's downsampled drag-frames (smoother camera movement at the cost of pixelated previews while moving) |
| `-v` / `--verbose` | off | Per-frame stage timing in the terminal (`[render-enter]` / `[render-mid]` / `[kernel-pre]` / `[kernel-post]` / `[render]`) |
| `--port N` | `8080` | Viewer port |

## Performance

Scene: 10K random Gaussians at 640×640 (synthetic integration test):

| Backend | Wall time / frame | Speedup vs CPU |
|---|---|---|
| CPU PyTorch (16 x86 cores) | ~1740 ms | 1.0× |
| Tenstorrent **single Tensix core** | ~2515 ms | 0.69× |
| Tenstorrent 64-core (contiguous split) | ~120 ms | 14.5× |
| Tenstorrent 64-core (LPT load balancing) | **~80 ms** | **21.7×** |

Scene: bonsai.ply (1.16M Gaussians, Mip-NeRF 360 capture) at 640×384:

| Pipeline state | Frame time | Notes |
|---|---|---|
| Original (accurate exp, no host culls) | ~6.0 s | baseline |
| + Approx exp | ~5.1 s | -11 dB PSNR (still 16 dB above 35 dB floor) |
| + Opacity cull (drop < 1/255 contribution) | ~2.5 s | no PSNR loss |
| + Radii cap (drop projection-breakdown blobs) | **~0.6-0.9 s** | **~10× cumulative** + visual artifact fix |

## Testing

```bash
source venv/bin/activate
python -m pytest tests/ -v -s
```

| Test | What | Hardware needed |
|---|---|---|
| `test_numeric_sanity.py` | NumPy reference vs CPU rasterizer | none |
| `test_kernel_integration.py::test_full_scene_psnr` | PSNR ≥ 35 dB / SSIM ≥ 0.98 vs CPU on a 64×64 / 50-Gaussian scene | Wormhole |
| `test_kernel_integration.py::test_640_perf_baseline` | 640×640 / 10K-Gaussian one-shot perf measurement | Wormhole |
| `test_kernel_integration.py::test_640_perf_daemon` | Same scene through the persistent daemon (init amortized) | Wormhole |

## Known issues

See `plan_progress.md` for full details:

- **KI-1 — T0.6 saturation edge artifacts**: a 32×32 tile with 50+ stacked
  α=0.99 Gaussians causes edge pixels to saturate to a bf16-representable
  garbage value (0x4790 = 73728). Realistic scenes don't trigger it; T0.6
  is shipped as a known-failing reproducer.

- **KI-2 — Multi-tile dispatch deadlock**: at certain "sparse populated tile"
  configurations (~18+ populated tiles each with ~6+ entries on a 1M+
  Gaussian scene), the kernel daemon hangs deterministically. Workaround:
  stick to `--max-resolution ≤ 640` for the interactive viewer.

## References

- [3D Gaussian Splatting (INRIA)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — the paper this implements
- [tenstorrent/tt-metal](https://github.com/tenstorrent/tt-metal) — the runtime and SDK
- [hbb1/torch-splatting](https://github.com/hbb1/torch-splatting) — pure-PyTorch reference
- [graphdeco-inria/diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) — original CUDA implementation
