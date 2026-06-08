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
# 1) one-shot bootstrap (creates ./venv, vendors tt-metal, builds the ttnn op,
#    and installs ttnn into ./venv)
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

Environment: `GSPLAT_TT_SPLIT=0` forces the TT backend's legacy single-op kernel
(disables the two-phase intra-tile-parallel path; default is on). See
[Performance → Intra-tile parallelism](#intra-tile-parallelism-two-phase-dense-scenes).

## Benchmarks

Each viewer session writes a markdown summary to `benchmarks/` on shutdown
(Ctrl+C). Filename is `{scene}_{backend}_{max-resolution}_{timestamp}.md`;
the directory is created on demand and gitignored. Empty frames (no
visible Gaussians) are excluded from the aggregate.

The report records date, backend, scene, Gaussian count, the actual
modal `W×H`, and the per-stage **median** across all sampled frames —
plus the median total and the FPS implied by it. The TT backend's
`blend(...)` reports sub-timings (`prep`, `upload`, `kernel`, `download` — plus
`partial_kernel`/`combine_kernel` on the two-phase split path), nested under the
parent stage in the table.

For headless, repeatable benchmarking, `scripts/bench_scene.py` runs the full
pipeline on a real `.ply` and prints the per-stage breakdown (pass `--cpu` to
also compare against the CPU reference and print the PSNR).

### Sweep harness (`benchmark/`)

Sweep one backend across scenes × resolutions, then plot:

```bash
source venv/bin/activate
# one backend per run (TT device / CUDA ext opened once, held across the sweep):
python -m benchmark.run tt   --res 256 480 640 960
python -m benchmark.run cuda --res 256 480 640 960
python -m benchmark.run cpu  --res 256 480 --skip-cpu-above 480   # CPU is slow on train.ply

# per-backend graphs (one CSV) or cross-backend comparison (multiple):
python -m benchmark.plot benchmark/results/tt.csv
python -m benchmark.plot benchmark/results/cpu.csv benchmark/results/cuda.csv benchmark/results/tt.csv
```

Results land in `benchmark/results/` (gitignored): `<backend>.csv`, `.json`,
and PNGs. Each blend timing is split into **load** (stage inputs) /
**compute** (device kernel) / **return** (result to host) / **transfer**
(residual host↔device movement); the four reconcile exactly to the blend
wall time. The shared host pre-stages (project / tile_assign / sort) are
reported separately.

## Setup details

`./setup.sh` is idempotent and does:

1. Create `./venv`, install `requirements.txt`, and `pip install -e .`
   (this puts the `gsplat` command on PATH). `numpy` is pinned `<2` because
   ttnn requires it.
2. Clone `tenstorrent/tt-metal` into `backends/tt/tt-metal/` (~5 GB).
3. Inject the build wiring for our three ttnn ops (`alpha_blend`,
   `alpha_blend_partial`, `alpha_blend_combine`) into tt-metal's CMake / nanobind
   (idempotent; only our op subtrees are tracked by git).
4. `sudo ./build_metal.sh --build-programming-examples` to compile the C++ libs
   **and** the ttnn Python extension (including our op), then
   `pip install -e backends/tt/tt-metal` to install `ttnn` into `./venv`. `sudo`
   is needed for tt-metal's root-owned SFPI / CPM caches.

The TT backend drives the device **in-process** through custom ttnn ops
(`ttnn.experimental.gaussian_alpha_blend`, plus `…_partial` / `…_combine` for the
intra-tile-parallel split path) — there is no daemon subprocess, and everything
(viewer, tests, `ttnn`) runs in the single `./venv`.

Pin a specific tt-metal version with `TT_METAL_REF=v1.2.3 ./setup.sh`.

After editing the op host code or kernels, rebuild the ttnn extension (the
editable install picks it up, no reinstall):

```bash
sudo ./backends/tt/tt-metal/build_metal.sh --build-programming-examples
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
    │   ├── backend.py         # in-process ttnn-op caller; single-op or two-phase split path
    │   ├── lpt.py             # host-side LPT tile→core load balancer (single-op path)
    │   ├── segments.py        # depth-segment scheduler + over-merge (intra-tile parallelism)
    │   └── tt-metal/          # vendored SDK + our 3 ttnn ops under
    │       └── …/experimental/gaussian_splatting/{alpha_blend, alpha_blend_partial, alpha_blend_combine}/
    └── cuda/                  # CUDA backend (kernels JIT-compiled on first use)
```

## Performance

The TT backend opens the device once and runs the kernels as an **in-process
ttnn op**; ttnn's program cache compiles the program once per resolution and
keeps it warm, so per-frame cost is the on-device **kernel** plus host work —
the CPU project/tile/sort stages and packing inputs to / unpacking the image
from the device. (The previous design shuttled each frame to a daemon
subprocess over shared memory; that whole IPC layer is gone.) All optimizations
below preserve the image to ~41 dB PSNR vs the CPU reference.

### End-to-end (real scenes, 640×640, full pipeline, TT backend)

| Scene | Gaussians (visible) | Frame time | |
|---|--:|--:|--:|
| luigi.ply | 14.5K  | ~24 ms  | ~42 fps |
| train.ply | 608K   | ~224 ms | ~4.5 fps |

The CPU reference renders the same scenes at seconds-to-minutes per frame (its
alpha-blend alone is ~3.5 s on luigi). With **intra-tile parallelism** (below)
the device compute on `train.ply` dropped to ~46 ms at 640 (it was ~130 ms), and
the frame is now **host-bound**: the per-frame CPU prep + upload (~74 ms) exceeds
device compute (~46 ms). Where a single frame's ~24 ms goes on luigi 640: device
compute ~7.5 ms, host pack/upload ~5 ms, readback + dispatch ~3 ms, CPU
project/tile/sort ~8 ms.

### Host data path

With no daemon, "host↔device" is now just `ttnn.from_torch` uploads and one
`ttnn.to_torch` readback per frame. The optimizations that matter:

- **Resident px/py grids** — the per-pixel coordinate grids depend only on
  resolution, so they're uploaded once per `(H, W)` and kept as device tensors,
  not re-shipped each frame (the in-process equivalent of the old `SETGRID`).
- **4 KB scalar-packs DRAM page** — the packs tensor is shaped so its ttnn
  interleaved-DRAM page is 4 KB (64 packs/page) rather than 64 B, lifting
  host→DRAM upload from ~1.6 GB/s to ~40 GB/s (it was page-count bound). The op
  validates these page sizes match the kernels at dispatch.
- **Reusable packs scratch** — the page-padded packs buffer is built into a
  persistent host scratch (only the `<64`-row tail re-zeroed per frame) instead
  of a fresh zero-allocation: **~43 → ~10 ms** of per-frame host work on
  `train.ply`.
- **bf16 readback + transpose** — the tile-major→`(H, W, 3)` conversion runs in
  torch on bf16 (then a single widen to fp32 on the cropped result), **~7 → ~0.5
  ms** vs the old numpy fp32 transpose.
- **Integer-key sort** — the entry sort (largest CPU stage on big scenes) uses a
  `torch.sort` over an integer `(tile_id<<32 | depth_bits)` key instead of a
  float64 `argsort`: **104 → 18 ms** on a ~740K scene (~6×).

### Kernel side

The on-device blend uses LPT (longest-processing-time) load balancing across
Tensix cores (~21× over a single core), approximate `exp`, an opacity cull
(peak contribution < 1/255), and a bounding-radius cap — the last two together
cut a 1M+-Gaussian scene from ~6 s to under 1 s.

### Intra-tile parallelism (two-phase, dense scenes)

A 32×32 tile is the *atomic* unit of the single-op kernel: one core composites
all of a tile's Gaussians front-to-back, so a frame can't finish until the
**densest tile** does. On `train.ply` the densest tile holds ~63K low-opacity
Gaussians, which made `train@256` compute ~345 ms (and, counterintuitively,
*faster* at higher resolution, as the dense screen region subdivides into more
tiles — see `docs/plan_progress.md` for the investigation).

When a tile's Gaussian count exceeds the per-core ideal
(`target = total_entries / num_cores`), the backend splits it into K contiguous
**depth-segments** and renders in two device passes:

1. **`ttnn.experimental.gaussian_alpha_blend_partial`** — each segment is
   composited on its own core into a partial `(R, G, B, T)`; the 4th plane is the
   segment's leftover per-pixel transmittance.
2. **`ttnn.experimental.gaussian_alpha_blend_combine`** — the partials are merged
   per tile, in depth order, via the associative Porter-Duff *over* operator:
   `C = C₀ + T₀·C₁ + T₀T₁·C₂ + …`.

The host scheduler `backends/tt/segments.py::build_segmented_assignment` does the
split and LPT-balances the resulting segment-jobs across cores. The backend takes
this path automatically whenever a tile splits (`num_jobs > non-empty tiles`) and
leaves scenes with no heavy tiles (e.g. luigi) on the untouched single op; set
`GSPLAT_TT_SPLIT=0` to force the legacy single-op path. The merge is exact in real
arithmetic (only bf16 rounding differs) — validated at 45–47 dB vs the CPU
reference on dense scenes.

Effect on `train.ply` device compute (single-op → two-phase, Blackhole 130 cores):

| res | single-op | two-phase | speedup |
|----:|----------:|----------:|--------:|
| 256 | 345 ms | 41 ms | **8.5×** |
| 480 | 189 ms | 44 ms | 4.3× |
| 640 | 130 ms | 46 ms | 2.8× |
| 960 |  73 ms | 55 ms | 1.3× |

Compute is now ~flat across resolution (the inversion is gone); `train` becomes
host-bound. luigi (no heavy tiles) is unaffected.

### Scheduling: choosing the kernel, splitting, and balancing

**Choosing the kernel.** The host builds the segmented schedule every frame and
picks the path with one test (`backends/tt/backend.py`):

```python
sched = build_segmented_assignment(offsets, num_tiles, self.num_cores)
if sched.num_jobs > sched.combine_plan.shape[0]:   # at least one tile was split
    # two-phase: partial → combine
else:
    # single op
```

`combine_plan` has exactly one row per non-empty tile and `num_jobs` is the total
segment count, so they're equal **iff** nothing split. luigi → equal → single op
(zero overhead); train → some tile exceeds `target` → two-phase. `GSPLAT_TT_SPLIT=0`
forces the single op regardless.

**Two problems, two tools.** Cores run in parallel, so a frame's compute time is
the **busiest core's** time, and a core's time is proportional to the total
Gaussian load of the jobs assigned to it. Minimizing the busiest core faces two
independent obstacles:

1. *Balancing across cores* — spreading many variable-sized work units so no core
   is overloaded. **LPT** (greedy longest-processing-time-first: sort jobs
   heaviest-first, drop each onto the currently least-loaded core) solves this. It
   runs on **both** paths — `lpt.py::build_tile_assignment` balances whole tiles
   (single op), `segments.py::build_segmented_assignment` balances segment-jobs
   (two-phase) — and both balance by **Gaussian load**, not job count. Without it,
   load variance or spatial clustering of dense tiles could pile several heavy
   units on one core. (Real frames land in the `jobs > cores` regime where LPT
   genuinely bin-packs: `train@256` produces 174 segment-jobs on 130 cores — up to
   2 per core — and `train@960` produces 918, up to 8 per core. Only when
   `num_jobs ≤ num_cores` would LPT degenerate to one-job-per-core.)

2. *Units too big to balance* — LPT can't help when a single unit already exceeds
   a core's fair share, because a tile is **atomic**: it just parks the monster
   tile on one core, and that core becomes the floor (on `train@256`, one tile =
   75K Gaussians vs an ideal of ~6.7K). **Segmentation** solves this by splitting
   any tile over `target` into contiguous depth-pieces ≤ `target`, bounding the
   largest unit LPT has to place.

They compose: **segmentation caps the biggest job; LPT then packs the jobs to
balance the per-core sums.**

**Why `target = total_entries / num_cores`.** Two hard lower bounds on the busiest
core — no schedule can beat either:

- `total / cores` — the average; you can't do better than perfectly even spreading.
- `max single job` — the largest indivisible piece.

Choosing `target = total/cores` makes every job ≤ the average, so the second bound
never dominates the first and the schedule can approach the `total/cores` floor.
Splitting finer can't beat that floor (it's the parallelism limit) — it only adds
per-job overhead (each job re-streams its tile's `px/py` + CB setup) and risks the
per-core job cap (`MAX_TILE_IDS_PER_CORE = 256`, enforced with a loud error). So
`total/cores` is the sweet spot: small enough to be balanceable, not so small that
overhead dominates. If device compute at that floor is still too slow, intra-tile
parallelism is exhausted (as on `train`, now host-bound) — the remaining levers are
fewer Gaussians, more cores, or a faster per-Gaussian kernel.

## Testing

```bash
source venv/bin/activate
pytest tests/
```

`test_numeric_sanity.py`, `test_lpt.py`, and `test_tt_segments.py` (the
intra-tile depth-segment scheduler + associative-`over` merge math) run
anywhere; the device tests (`test_kernel_integration.py` — PSNR, empty-tile,
perf — and `test_ttnn_op_cache.py` — program-cache warm-frame invariant) skip
cleanly without a Tenstorrent device. Note: the two-phase `partial`/`combine`
ops are validated against the CPU reference manually (see `docs/plan_progress.md`)
but do not yet have a committed device test.

## References

- [3D Gaussian Splatting (INRIA)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — the paper
- [tenstorrent/tt-metal](https://github.com/tenstorrent/tt-metal) — runtime + SDK
- [hbb1/torch-splatting](https://github.com/hbb1/torch-splatting) — PyTorch reference
- [graphdeco-inria/diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) — original CUDA rasterizer
