# Alpha-blend benchmark harness — design

**Date:** 2026-06-07
**Status:** approved (pending spec review)

## Goal

A single runnable tool that measures alpha-blend rendering performance for
**one backend at a time** across multiple scenes and resolutions, emitting
machine-readable results (CSV + JSON). A separate plotting script turns one
or more result files into graphs, including the cross-backend
CPU-vs-CUDA-vs-TT comparison.

The tool defines a consistent **load / compute / return** timing model that
maps each backend's native sub-timings into the same buckets, so the same
graph axes mean the same thing across CPU, CUDA, and TT.

Scope: forward-pass rendering only (consistent with the project). No
training, no backward pass. Reuses the existing `Pipeline`, `get_backend`,
and per-stage timing infrastructure — this tool adds the sweep, the bucket
model, the result files, and the plots; it does not change the backends.

## Non-goals

- Plotting bundled into the bench tool (decoupled by design — bench emits
  data, plot reads data).
- A reusable `gsplat/bench/` package or `BenchRunner` abstraction (YAGNI for
  a thesis measurement tool).
- Editing or importing from `scripts/` (the existing `bench_*.py` scripts
  stay as-is; this harness is independent).
- Measuring backend one-shot init / JIT-compile cost. The TT daemon and the
  CUDA extension are warmed once per sweep and held, matching real
  interactive use where init is paid once at startup.

## Layout

One new top-level folder, self-contained:

```
benchmark/
├── __init__.py        # makes the package importable for tests
├── phases.py          # pure: native timings → load/compute/return buckets
├── camera.py          # look_at_c2w + make_camera (re-homed here)
├── run.py             # CLI: sweep ONE backend × scenes × resolutions → CSV+JSON
├── plot.py            # CLI: read CSV(s) → PNG graphs
└── results/           # runtime output (gitignored): <backend>.csv/.json, *.png
```

Invocation:

```bash
source venv/bin/activate
# Sweep one backend (TT daemon stays warm across the whole sweep):
python -m benchmark.run tt   --res 256 480 640 960
python -m benchmark.run cpu  --res 256 480 640 960 --skip-cpu-above 480
python -m benchmark.run cuda --res 256 480 640 960

# Plot — one CSV = per-backend deep dive:
python -m benchmark.plot benchmark/results/tt.csv
# Multiple CSVs = cross-backend comparison:
python -m benchmark.plot benchmark/results/{cpu,cuda,tt}.csv
```

`benchmark/results/` is added to `.gitignore` (the folder is a runtime
artifact dir; the `benchmark/` code is tracked). Note the existing
`.gitignore` already ignores `/benchmarks/` (plural, viewer-written
markdowns) — this is a different path and must be added separately as
`/benchmark/results/`.

## `camera.py` — scene framing

Re-homed copies of `look_at_c2w(center, eye, up)` and
`make_camera(means, H, W, fov_deg=60.0)` (median-center / p60-radius
look-at framing). Logic is the same as the helpers currently in
`scripts/bench_scene.py`, but copied here so `benchmark/` has no dependency
on `scripts/`. Produces `(extrinsics, intrinsics)` for a `.ply`'s point
cloud.

## `run.py` — the sweep

**Args:**

| arg | default | meaning |
|---|---|---|
| `backend` (positional) | — | one of `cpu` / `tt` / `cuda` (exactly one) |
| `--scenes` | `scenes/luigi.ply scenes/train.ply` | one or more `.ply` paths |
| `--res` | `256 480 640 960` | resolutions; each snapped to a /32 multiple |
| `--warmup` | `3` | warmup renders (not measured) |
| `--measure` | `20` | measured renders; median is primary, min also recorded |
| `--out` | `benchmark/results` | output dir for `<backend>.csv` / `.json` |
| `--skip-cpu-above` | `None` | skip (scene,res) rows where res > this value (guards CPU on huge scenes) |

**Per-scene / per-resolution loop:**

1. Load each `.ply` once per scene (not per resolution).
2. `extr, intr = make_camera(gaussians.means, H, W)`.
3. Construct the backend **once per sweep** via `get_backend(backend)`;
   reuse across all scene×res cells; `close()` in a `finally`.
4. For each (scene, res): `warmup` renders, then `measure` renders through
   `Pipeline(backend).render(...)`. Collect **median** (primary) and **min**
   of every key in `timings` + `sub_timings`, plus `num_visible`,
   `num_entries`, and derived `fps = 1000 / median(total_ms)`.
5. Map the per-cell median timings into load/compute/return/transfer via
   `phases.to_buckets(...)`.

**Output:** one CSV (schema below) + one JSON. The JSON mirrors the CSV rows
and additionally carries the full raw median sub-timing dict per cell (for
anything the flat CSV drops).

**Failure handling:**

- Backend unavailable (`get_backend` raises — no CUDA device, TT device
  busy, kernel binary missing): report a clear message and exit non-zero
  *before* starting the sweep, rather than crashing mid-loop.
- Empty frame (`num_visible == 0`): the look-at framing is designed to avoid
  it, but if a cell still produces zero visible Gaussians, skip that row and
  emit a warning (do not write a misleading all-zero row).
- `--skip-cpu-above`: rows exceeding the cap are skipped with a logged note
  so the gap in the data is explicit, not silent.

## `phases.py` — the load / compute / return model

A single pure function, unit-tested:

```python
def to_buckets(timings: dict, sub_timings: dict, backend: str) -> dict:
    """Map a cell's native timings into {load, compute, return, transfer} ms.

    Buckets reconcile exactly to the blend-stage wall time:
        load + compute + return + transfer == blend
    'transfer' is the explicit residual — host<->device movement that the
    backend does not separately instrument — so no time is hidden.
    Returns the four buckets plus 'device_kernel' (pure-kernel sub-metric)
    and the shared host pre-stages (project/tile_assign/sort) unchanged.
    """
```

Bucket definitions (all in ms, sourced from `Pipeline` `timings` /
`sub_timings`, where blend sub-timings are prefixed `blend.`):

| bucket | TT | CUDA | CPU |
|---|---|---|---|
| **load** — stage device-ready inputs (host) | `blend.prep + blend.write_shm` | `blend.upload` | 0 |
| **compute** — pure device kernel | `blend.mframe_rt.device_kernel` | `blend.kernel.device` | `blend` |
| **return** — result back to host | `blend.read_shm` | `blend.kernel − blend.kernel.device` | 0 |
| **transfer** — residual = `blend − load − compute − return` | DRAM up/down + IPC inside `mframe_rt` | ~0 | ~0 |

Notes:

- **Shared host pre-stages** (`project`, `tile_assign`, `sort`) are reported
  as their own columns, *not* folded into the four buckets. They are
  identical CPU work for every backend; keeping them separate keeps the
  cross-backend `compute` comparison honest.
- **`transfer` is an explicit, named residual** (approved approach). For TT
  it surfaces the real cost of moving data host↔device through the daemon's
  shared-memory → DRAM path plus IPC latency, which `mframe_rt` bundles and
  does not break out. For CUDA and CPU it is ≈0.
- **`device_kernel`** is reported as an informational sub-metric (TT:
  `mframe_rt.device_kernel`; CUDA: `kernel.device`; CPU: `blend`) so the
  pure-kernel number is always available even though TT's `compute` bucket
  equals it by definition and CUDA's does too.
- If a backend lacks an expected sub-timing key (e.g. TT fell back to the
  `.npy` FRAME path instead of MFRAME, emitting `daemon_rt.*`/`load_npy`
  instead of `mframe_rt.*`/`read_shm`), `to_buckets` maps the equivalent
  keys; if none are present it puts the whole `blend` wall into `compute`
  and zeroes the rest (degrades gracefully, never throws).

## CSV schema

One row per (scene, resolution). All timing columns are **medians** in ms
unless suffixed `_min`:

```
backend, scene, n_gaussians, height, width, num_visible, num_entries,
project_ms, tile_assign_ms, sort_ms, blend_ms, total_ms, fps,
load_ms, compute_ms, return_ms, transfer_ms, device_kernel_ms,
blend_ms_min, total_ms_min
```

`scene` is the basename (e.g. `luigi`, `train`). `fps` is derived from
`total_ms` median.

## `plot.py` — graphs from saved CSV(s)

matplotlib only. Reads one or more CSV paths; saves PNGs into `--out`
(default: the directory of the first CSV).

- **One CSV (per-backend deep dive):**
  1. **Frame-time vs resolution** — line plot, one line per scene, y =
     `total_ms` (and a second figure for `blend_ms`).
  2. **Phase breakdown** — stacked bar per resolution
     (load / compute / return / transfer), one chart per scene.
- **Multiple CSVs (cross-backend comparison):**
  3. **CPU vs CUDA vs TT at fixed scene/resolution** — grouped bars (or
     lines) of `total_ms` per resolution, one series per backend, one
     chart (facet) per scene. A companion figure uses `blend_ms` to compare
     just the accelerated stage.

Missing cells (skipped rows, e.g. CPU above the guard) are simply absent
from their series — the gap is visible rather than zero-filled.

## Testing

- `tests/test_benchmark_phases.py` (TDD, written first): feed
  `to_buckets` synthetic `timings`/`sub_timings` dicts for each backend and
  assert:
  - correct bucket values per the mapping table,
  - `load + compute + return + transfer == blend` (reconciliation) for TT
    (non-trivial transfer), CUDA (~0 transfer), CPU (all in compute),
  - graceful fallback when expected sub-timing keys are absent.
- `run.py` / `plot.py` are thin CLI glue over tested pieces (`Pipeline`,
  `to_buckets`); no device-dependent unit tests are added (the existing
  `tests/test_kernel_integration.py` already covers device correctness).
  A smoke check: `python -m benchmark.run cpu --scenes scenes/luigi.ply
  --res 256 --warmup 0 --measure 2` should produce a valid one-row CSV on a
  CPU-only box.

## Open risks

- `train.ply` on the **CPU** backend at higher resolutions is very slow;
  `--skip-cpu-above` is the mitigation. CPU rows for the huge scene at high
  res are expected to be omitted by default usage.
- TT MFRAME vs `.npy` fallback changes which sub-timing keys appear;
  `to_buckets` handles both, but the `transfer` residual semantics differ
  slightly between paths (documented in the function).
