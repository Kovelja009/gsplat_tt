# 3D Gaussian Splatting on Tenstorrent

Forward-pass renderer for [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
running on Tenstorrent's [tt-metal](https://github.com/tenstorrent/tt-metal)
hardware. Loads a pre-trained `.ply` and renders it interactively in the
browser.

MSc thesis project — inference only, no training or backward pass.

## Backends

The same viewer can dispatch to multiple rasterizers:

| `backend` | Status | What |
|---|---|---|
| `cpu`  | shipping | Pure PyTorch reference. Slow (~1–2 s/frame at 256×256), used as the correctness baseline. |
| `tt`   | shipping | tt-metal kernels on a Tenstorrent Wormhole device. ~80 ms/frame at 640×640 (21.7× CPU). |
| `cuda` | planned  | Placeholder; same wrapper interface, CUDA implementation TBD. |

Pipeline (CPU does setup, the chosen backend does the alpha-blend):

```
load_ply → project → tile-assign → sort  ──►  alpha_blend (cpu | tt | cuda)
            └────────────── CPU ──────────┘   └── per-backend rasterizer ──┘
```

## Quick start

```bash
# 1) one-shot bootstrap (creates ./venv, vendors tt-metal, builds the kernel)
./setup.sh

# 2) CPU viewer (no Tenstorrent device required)
source venv/bin/activate
gsplat scenes/luigi.ply

# 3) TT viewer (Wormhole device required)
source venv/bin/activate
export TT_METAL_HOME=$PWD/backends/tt/tt-metal
export TT_METAL_RUNTIME_ROOT=$PWD/backends/tt/tt-metal
gsplat scenes/luigi.ply --backend tt
```

Then open <http://localhost:8080>. Drag to orbit, **WASD / QE / arrows** to fly.

## CLI flags

| Flag | Default | What |
|---|---|---|
| `--backend {cpu,tt}` | `cpu` | Rasterizer (see table above). |
| `--max-resolution N` | `640`  | Cap on the longest render dim. Snapped to a multiple of 32. |
| `--adaptive-resolution` | off | Drop to lower res while moving the camera (smoother drag). |
| `--port N` | `8080` | Viewer port. |
| `-v` / `--verbose` | off | Per-frame stage timing. |

## Setup details

`./setup.sh` is idempotent and does:

1. Create `./venv`, install `requirements.txt`, and `pip install -e .`
   (this puts the `gsplat` command on PATH).
2. Clone `tenstorrent/tt-metal` into `backends/tt/tt-metal/` (~5 GB).
3. Register our kernel subdir in tt-metal's CMake (`add_subdirectory`).
4. `sudo ./build_metal.sh --build-programming-examples --without-python-bindings`
   to compile the C++ libs + our kernel host binary. `sudo` is needed for
   tt-metal's root-owned SFPI / CPM caches; we skip the `ttnn` Python wheel
   because the runtime only invokes the binary as a subprocess.

Pin a specific tt-metal version with `TT_METAL_REF=v1.2.3 ./setup.sh`.

After editing kernel C++ sources, rebuild just the binary:

```bash
sudo ninja -C backends/tt/tt-metal/build metal_example_gaussian_splatting
```

## Repository layout

```
gsplat_tt/
├── pyproject.toml, setup.sh, README.md, CLAUDE.md, .gitignore
├── docs/                      # design notes (plan_progress.md, …)
├── scenes/                    # *.ply (only luigi tracked; others gitignored)
├── tests/                     # pytest suite
├── scripts/                   # one-off dev helpers
├── gsplat/                    # host-side Python package
│   ├── __main__.py            # CLI entry — installed as `gsplat` console-script
│   ├── viewer.py              # interactive viewer (viser + nerfview)
│   ├── rasterization.py       # project, tile, sort, CPU alpha_blend, prep
│   └── …                      # data_structures, loading_gaussians, utils
└── backends/                  # one subpackage per accelerator
    ├── README.md              # how to add a new backend
    ├── tt/                    # Tenstorrent (tt-metal)
    │   ├── backend.py         # daemon-subprocess wrapper
    │   └── tt-metal/          # vendored SDK + our kernels under
    │       └── tt_metal/programming_examples/gaussian_splatting/
    └── cuda/                  # placeholder for the future
```

## Performance

640×640, 10K random Gaussians:

| Backend | Frame time | Speedup |
|---|---|---|
| CPU PyTorch (16 x86 cores) | ~1740 ms | 1.0× |
| TT, single Tensix core | ~2515 ms | 0.69× |
| TT, 64 cores (contiguous split) | ~120 ms | 14.5× |
| **TT, 64 cores (LPT load balancing)** | **~80 ms** | **21.7×** |

bonsai.ply (1.16M Gaussians, Mip-NeRF 360) at 640×384:

| State | Frame time |
|---|---|
| Naive (accurate exp, no culls) | ~6.0 s |
| + approx exp | ~5.1 s |
| + opacity cull (< 1/255) | ~2.5 s |
| **+ radius cap** | **~0.6–0.9 s** |

## Testing

```bash
source venv/bin/activate
pytest tests/
```

`test_numeric_sanity.py` runs anywhere; the three `test_kernel_integration.py`
tests need a Tenstorrent device.

## References

- [3D Gaussian Splatting (INRIA)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — the paper
- [tenstorrent/tt-metal](https://github.com/tenstorrent/tt-metal) — runtime + SDK
- [hbb1/torch-splatting](https://github.com/hbb1/torch-splatting) — PyTorch reference
- [graphdeco-inria/diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) — original CUDA rasterizer
