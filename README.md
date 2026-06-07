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
| `tt`   | shipping | tt-metal kernels on a Tenstorrent Wormhole/Blackhole device. ~30 ms/frame for a small scene at 640×640; scales with Gaussian count (see [Performance](#performance)). |
| `cuda` | experimental | JIT-compiles on first use via `torch.utils.cpp_extension`. Requires a CUDA-capable GPU and `requirements-cuda.txt` installed. |

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

# 4) CUDA viewer (NVIDIA GPU required; first run JIT-compiles the kernel)
source venv/bin/activate
pip install -r requirements-cuda.txt   # one-time: swap cpu torch wheel for cu121, install ninja
gsplat scenes/luigi.ply --backend cuda
```

Then open <http://localhost:8080>. Drag to orbit, **WASD / QE / arrows** to fly.

For the CUDA backend on a CUDA-capable host, after `./setup.sh`:

```bash
source venv/bin/activate
pip install -r requirements-cuda.txt
```

This swaps the CPU torch wheel for the cu121 build in place and installs
`ninja` (used by the JIT compiler). The TT dev box can skip this — `cuda`
will simply not appear in the backend registry.

## CLI flags

| Flag | Default | What |
|---|---|---|
| `--backend {cpu,tt,cuda}` | `cpu` | Rasterizer (see table above). |
| `--max-resolution N` | `640`  | Shorter render dim (480p/720p/1080p convention). Longer dim follows from browser aspect; both snap to multiples of 32. |
| `--port N` | `8080` | Viewer port. |
| `-v` / `--verbose` | off | Per-frame stage timing. |

## Benchmarks

Each viewer session writes a markdown summary to `benchmarks/` on shutdown
(Ctrl+C). Filename is `{scene}_{backend}_{max-resolution}_{timestamp}.md`;
the directory is created on demand and gitignored. Empty frames (no
visible Gaussians) are excluded from the aggregate.

The report records date, backend, scene, Gaussian count, the actual
modal `W×H`, and the per-stage **median** across all sampled frames —
plus the median total and the FPS implied by it. Each backend's
`blend(...)` may also report sub-timings (e.g. `blend.device_kernel`),
which are nested under the parent stage in the table.

For headless, repeatable benchmarking there are scripts under `scripts/`:
`bench_scene.py` runs the full pipeline on a real `.ply` and prints the
per-stage breakdown; `bench_mmap.py` A/Bs the zero-copy vs `.npy` hand-off and
checks the outputs are bit-identical.

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
    └── cuda/                  # CUDA backend (kernels JIT-compiled on first use)
```

## Performance

Per-frame cost splits between the on-device alpha-blend **kernel** and the
**host↔device data path** that feeds it (the CPU project/tile/sort stages plus
moving per-frame buffers to the daemon). Both have been optimized; all changes
below preserve the image bit-for-bit (or to ~41 dB PSNR vs the CPU reference).

### End-to-end (real scenes, 640×640, full pipeline, TT backend)

| Scene | Gaussians | Frame time | |
|---|--:|--:|--:|
| luigi.ply | 14.5K | ~30 ms | ~33 fps |
| point_cloud.ply (Mip-NeRF 360) | 742K | ~315 ms | ~3 fps |

The CPU reference renders the same scenes at seconds-to-minutes per frame
(its alpha-blend alone is ~3.5 s on luigi). On the 742K scene the device
kernel is ~130 ms; the rest is host CPU stages.

### Host ↔ device data path

The daemon receives each frame's inputs and returns the rendered image. The
hand-off and the entry sort were the bulk of a real frame; the wins, measured
on the synthetic 640×640 / 10K-Gaussian blend round-trip unless noted:

| Change | round-trip | |
|---|--:|---|
| Baseline — `.npy` files on disk-backed `/tmp` | 77 ms | 1.0× |
| Stage `.npy` on tmpfs (`/dev/shm`) | 54 ms | 1.4× |
| Zero-copy shared-memory hand-off (`MFRAME`, mmap, no `.npy`) | 27 ms | 2.9× |
| Keep static px/py grids resident on-device (`SETGRID`) | 26 ms | 3.0× |

- **Zero-copy `MFRAME`** maps one `/dev/shm` region into both host and daemon.
  Each frame the host writes the per-frame buffers — the 64-byte scalar packs and
  per-tile offsets — straight into it in device-ready form; the daemon uploads
  them to DRAM and reads the result back via pointer, with no `.npy`
  serialize/parse and no fp32↔bf16 conversion. The `.npy` `FRAME` path is kept as
  a fallback (and for the test suite).
- **Resident px/py grids (`SETGRID`)** — the per-pixel coordinate grids don't
  change at a fixed resolution, so they aren't shipped per frame: they're
  uploaded once (first frame / on resolution change) into DRAM buffers the daemon
  keeps resident and reuses for every subsequent `MFRAME`.
- **4 KB scalar-packs DRAM page** — paging the packs buffer at 4 KB (64 packs/
  page) instead of 64 B lifts host→DRAM upload from ~1.6 GB/s to ~40 GB/s; it
  was page-count bound (~235K tiny pages).
- **Integer-key sort** — the entry sort (largest CPU stage on big scenes) moved
  from a float64 `argsort` to a `torch.sort` over an integer `(tile_id<<32 |
  depth_bits)` key: **104 → 18 ms** on the 742K scene (~6×).

### Kernel side

The on-device blend uses LPT (longest-processing-time) load balancing across
Tensix cores (~21× over a single core), approximate `exp`, an opacity cull
(peak contribution < 1/255), and a bounding-radius cap — the last two together
cut a 1M+-Gaussian scene from ~6 s to under 1 s.

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
