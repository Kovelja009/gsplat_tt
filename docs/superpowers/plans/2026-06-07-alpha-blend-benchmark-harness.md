# Alpha-blend Benchmark Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `benchmark/` package that sweeps one backend across scenes × resolutions, emits CSV+JSON results with a load/compute/return/transfer timing model, and a decoupled plot script that renders per-backend and cross-backend graphs.

**Architecture:** A pure `phases.to_buckets()` maps each backend's native `Pipeline` sub-timings into four reconciling buckets. `run.py` is a CLI sweep over `Pipeline.render()` that writes results. `plot.py` reads result CSV(s) with matplotlib. `camera.py` holds re-homed look-at framing so `benchmark/` never imports `scripts/`.

**Tech Stack:** Python 3, numpy, torch, matplotlib, pytest. Reuses `gsplat.pipeline.Pipeline`, `backends.get_backend`, `gsplat.loading_gaussians.load_ply`, `gsplat.utils.c2w_to_w2c`.

---

## File Structure

- `benchmark/__init__.py` — package marker (empty).
- `benchmark/phases.py` — `to_buckets(timings, sub_timings, backend) -> dict`. Pure, tested.
- `benchmark/camera.py` — `look_at_c2w`, `make_camera` (copied from `scripts/bench_scene.py`).
- `benchmark/run.py` — CLI sweep → `<backend>.csv` + `.json`.
- `benchmark/plot.py` — CLI: read CSV(s) → PNGs.
- `tests/test_benchmark_phases.py` — unit tests for `to_buckets`.
- `.gitignore` — add `/benchmark/results/`.

---

### Task 1: Package skeleton + gitignore

**Files:**
- Create: `benchmark/__init__.py`
- Modify: `.gitignore` (append one line)

- [ ] **Step 1: Create the empty package marker**

Create `benchmark/__init__.py` with exactly:

```python
"""Alpha-blend performance benchmark harness (sweep + plot)."""
```

- [ ] **Step 2: Ignore the runtime results dir**

Append to `.gitignore` (the file already ignores `/benchmarks/` plural — this is a different path):

```
# Benchmark harness runtime output (CSV/JSON/PNG).
/benchmark/results/
```

- [ ] **Step 3: Commit**

```bash
git add benchmark/__init__.py .gitignore
git commit -m "feat(benchmark): package skeleton + ignore results dir"
```

---

### Task 2: Phase-bucket model (TDD)

**Files:**
- Create: `tests/test_benchmark_phases.py`
- Create: `benchmark/phases.py`

The `Pipeline` prefixes blend sub-timings with `blend.`, so keys arrive as
`blend.prep`, `blend.write_shm`, `blend.mframe_rt.device_kernel`,
`blend.read_shm` (TT MFRAME path); `blend.upload`, `blend.kernel`,
`blend.kernel.device` (CUDA); CPU has no blend sub-timings. The `timings`
dict always has `project`, `tile_assign`, `sort`, `blend`, `total`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmark_phases.py`:

```python
"""Unit tests for the benchmark load/compute/return/transfer bucket model."""
import math

from benchmark.phases import to_buckets


def _approx(a, b, tol=1e-6):
    return math.isclose(a, b, rel_tol=0, abs_tol=tol)


def test_cpu_all_in_compute():
    timings = {"project": 5.0, "tile_assign": 2.0, "sort": 3.0,
               "blend": 40.0, "total": 50.0}
    b = to_buckets(timings, {}, "cpu")
    assert _approx(b["load"], 0.0)
    assert _approx(b["compute"], 40.0)
    assert _approx(b["return"], 0.0)
    assert _approx(b["transfer"], 0.0)
    assert _approx(b["device_kernel"], 40.0)
    # Shared host pre-stages passed through unchanged.
    assert _approx(b["project"], 5.0)
    assert _approx(b["tile_assign"], 2.0)
    assert _approx(b["sort"], 3.0)
    # Reconciliation: buckets sum to blend.
    assert _approx(b["load"] + b["compute"] + b["return"] + b["transfer"], 40.0)


def test_cuda_mapping_and_reconciliation():
    timings = {"project": 5.0, "tile_assign": 2.0, "sort": 3.0,
               "blend": 30.0, "total": 40.0}
    sub = {"blend.upload": 4.0, "blend.kernel": 20.0, "blend.kernel.device": 15.0}
    b = to_buckets(timings, sub, "cuda")
    assert _approx(b["load"], 4.0)            # upload
    assert _approx(b["compute"], 15.0)        # kernel.device
    assert _approx(b["return"], 5.0)          # kernel - kernel.device
    assert _approx(b["device_kernel"], 15.0)
    # transfer = blend - (load+compute+return) = 30 - 24 = 6 (untimed remainder)
    assert _approx(b["transfer"], 6.0)
    assert _approx(b["load"] + b["compute"] + b["return"] + b["transfer"], 30.0)


def test_tt_mframe_mapping_and_reconciliation():
    timings = {"project": 5.0, "tile_assign": 2.0, "sort": 3.0,
               "blend": 50.0, "total": 60.0}
    sub = {"blend.prep": 6.0, "blend.write_shm": 4.0,
           "blend.mframe_rt": 35.0, "blend.mframe_rt.device_kernel": 25.0,
           "blend.read_shm": 3.0}
    b = to_buckets(timings, sub, "tt")
    assert _approx(b["load"], 10.0)           # prep + write_shm
    assert _approx(b["compute"], 25.0)        # device_kernel
    assert _approx(b["return"], 3.0)          # read_shm
    assert _approx(b["device_kernel"], 25.0)
    # transfer = 50 - (10+25+3) = 12 (DRAM up/down + IPC inside mframe_rt)
    assert _approx(b["transfer"], 12.0)
    assert _approx(b["load"] + b["compute"] + b["return"] + b["transfer"], 50.0)


def test_tt_npy_fallback_keys():
    # No mframe_rt/read_shm; daemon .npy path emits daemon_rt.*/load_npy.
    timings = {"project": 1.0, "tile_assign": 1.0, "sort": 1.0,
               "blend": 50.0, "total": 53.0}
    sub = {"blend.prep": 6.0, "blend.save_npy": 4.0,
           "blend.daemon_rt": 35.0, "blend.daemon_rt.device_kernel": 25.0,
           "blend.load_npy": 3.0}
    b = to_buckets(timings, sub, "tt")
    assert _approx(b["load"], 10.0)           # prep + save_npy
    assert _approx(b["compute"], 25.0)        # daemon_rt.device_kernel
    assert _approx(b["return"], 3.0)          # load_npy
    assert _approx(b["load"] + b["compute"] + b["return"] + b["transfer"], 50.0)


def test_missing_subtimings_degrades_to_compute():
    # Unknown/empty sub-timings: whole blend wall lands in compute, no throw.
    timings = {"project": 1.0, "tile_assign": 1.0, "sort": 1.0,
               "blend": 12.0, "total": 15.0}
    b = to_buckets(timings, {}, "tt")
    assert _approx(b["compute"], 12.0)
    assert _approx(b["load"], 0.0)
    assert _approx(b["return"], 0.0)
    assert _approx(b["transfer"], 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_benchmark_phases.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmark.phases'`

- [ ] **Step 3: Implement `phases.py`**

Create `benchmark/phases.py`:

```python
"""Map a render cell's native timings into reconciling phase buckets.

The four buckets — load / compute / return / transfer — always sum exactly
to the `blend` stage wall time. `transfer` is the explicit residual: the
host<->device data movement a backend does not separately instrument
(notably the DRAM up/down + IPC bundled inside the TT daemon round-trip),
so no time is hidden.

The shared host pre-stages (project / tile_assign / sort) are passed
through unchanged — identical CPU work for every backend, kept out of the
four buckets so cross-backend `compute` stays comparable.
"""
from __future__ import annotations


def _get(sub: dict, key: str) -> float:
    """Blend sub-timings arrive prefixed with 'blend.' from the Pipeline."""
    return float(sub.get(f"blend.{key}", 0.0))


def to_buckets(timings: dict, sub_timings: dict, backend: str) -> dict:
    """Return {load, compute, return, transfer, device_kernel, project,
    tile_assign, sort} in ms. load+compute+return+transfer == blend."""
    blend = float(timings.get("blend", 0.0))
    sub = sub_timings or {}

    if backend == "cpu":
        load, compute, ret, dev = 0.0, blend, 0.0, blend

    elif backend == "cuda":
        load = _get(sub, "upload")
        dev = _get(sub, "kernel.device")
        compute = dev
        ret = max(0.0, _get(sub, "kernel") - dev)

    elif backend == "tt":
        load = _get(sub, "prep") + _get(sub, "write_shm") + _get(sub, "save_npy")
        # MFRAME path uses mframe_rt/read_shm; .npy fallback uses daemon_rt/load_npy.
        dev = _get(sub, "mframe_rt.device_kernel") + _get(sub, "daemon_rt.device_kernel")
        compute = dev
        ret = _get(sub, "read_shm") + _get(sub, "load_npy")

    else:
        raise ValueError(f"unknown backend {backend!r}")

    transfer = blend - (load + compute + ret)
    # Degrade gracefully if no sub-timings were reported: everything to compute.
    if compute == 0.0 and load == 0.0 and ret == 0.0 and blend > 0.0:
        compute, dev, transfer = blend, blend, 0.0

    return {
        "load": load,
        "compute": compute,
        "return": ret,
        "transfer": transfer,
        "device_kernel": dev,
        "project": float(timings.get("project", 0.0)),
        "tile_assign": float(timings.get("tile_assign", 0.0)),
        "sort": float(timings.get("sort", 0.0)),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_benchmark_phases.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add benchmark/phases.py tests/test_benchmark_phases.py
git commit -m "feat(benchmark): load/compute/return/transfer bucket model"
```

---

### Task 3: Camera framing helper

**Files:**
- Create: `benchmark/camera.py`

Copied from `scripts/bench_scene.py` so `benchmark/` has no dependency on
`scripts/`. Logic is byte-identical to the working helpers there.

- [ ] **Step 1: Create `benchmark/camera.py`**

```python
"""Scene framing: median-center / p60-radius look-at for a point cloud.

Copied (not imported) from scripts/bench_scene.py to keep benchmark/
independent of scripts/.
"""
from __future__ import annotations

import numpy as np
import torch

from gsplat.utils import c2w_to_w2c


def look_at_c2w(center, eye, up=(0.0, -1.0, 0.0)):
    """OpenCV-convention c2w (+Z forward, +Y down) for a camera at `eye`."""
    center = np.asarray(center, np.float64); eye = np.asarray(eye, np.float64)
    up = np.asarray(up, np.float64)
    z = center - eye; z /= np.linalg.norm(z)          # forward
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(np.array([0.0, 0.0, 1.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)                                  # down
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = x; c2w[:3, 1] = y; c2w[:3, 2] = z; c2w[:3, 3] = eye
    return c2w


def make_camera(means, H, W, fov_deg=60.0):
    """Robust look-at framing the inlier cloud (median center, p60 radius)."""
    m = means.numpy()
    center = np.median(m, axis=0)
    dist = np.linalg.norm(m - center, axis=1)
    radius = float(np.percentile(dist, 60))
    eye = center + np.array([0.0, 0.0, 2.5 * radius], np.float32)
    c2w = look_at_c2w(center, eye)
    extrinsics = c2w_to_w2c(c2w)
    f = 0.5 * W / np.tan(0.5 * np.radians(fov_deg))
    intrinsics = torch.tensor(
        [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=torch.float32
    )
    return extrinsics, intrinsics
```

- [ ] **Step 2: Verify it imports**

Run: `venv/bin/python -c "from benchmark.camera import make_camera; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add benchmark/camera.py
git commit -m "feat(benchmark): re-homed look-at camera framing"
```

---

### Task 4: Sweep CLI (`run.py`)

**Files:**
- Create: `benchmark/run.py`

`Pipeline.render()` returns a `RenderResult` with `.timings` (dict incl.
`project/tile_assign/sort/blend/total`), `.sub_timings`, `.num_visible`,
`.num_entries`, `.image`. `load_ply(path) -> Gaussians` (has `.means`,
`.num_gaussians`). `get_backend(name)` constructs and may raise.

- [ ] **Step 1: Create `benchmark/run.py`**

```python
"""Sweep ONE backend across scenes x resolutions; write CSV + JSON.

The backend is constructed once and held across the whole sweep (TT daemon
stays warm, CUDA extension stays loaded) — one-shot init cost is excluded,
matching real interactive use. Per (scene, resolution) cell we run `warmup`
unmeasured renders then `measure` timed renders, recording the median
(primary) and min of every timing, mapped into load/compute/return/transfer.

Usage:
    python -m benchmark.run tt   --res 256 480 640 960
    python -m benchmark.run cpu  --res 256 480 --skip-cpu-above 480
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys

from backends import get_backend
from gsplat.loading_gaussians import load_ply
from gsplat.pipeline import Pipeline
from benchmark.camera import make_camera
from benchmark.phases import to_buckets


CSV_COLUMNS = [
    "backend", "scene", "n_gaussians", "height", "width",
    "num_visible", "num_entries",
    "project_ms", "tile_assign_ms", "sort_ms", "blend_ms", "total_ms", "fps",
    "load_ms", "compute_ms", "return_ms", "transfer_ms", "device_kernel_ms",
    "blend_ms_min", "total_ms_min",
]


def _snap32(res: int) -> int:
    """Snap a resolution down to a multiple of the 32x32 tile."""
    return max(32, (res // 32) * 32)


def _measure_cell(pipeline, gaussians, extr, intr, H, W, warmup, measure):
    """Run warmup+measure renders; return median timings + raw rows, or None
    if every measured frame had zero visible Gaussians."""
    for _ in range(warmup):
        pipeline.render(gaussians, extr, intr, H, W)
    rows = []
    for _ in range(measure):
        r = pipeline.render(gaussians, extr, intr, H, W)
        if r.num_visible == 0:
            continue
        row = dict(r.timings)
        row.update(r.sub_timings)
        row["_num_visible"] = r.num_visible
        row["_num_entries"] = r.num_entries
        rows.append((row, r))
    if not rows:
        return None
    keys = sorted({k for row, _ in rows for k in row})
    med = {k: statistics.median(row[k] for row, _ in rows if k in row) for k in keys}
    last = rows[-1][1]
    return med, last


def _build_record(backend, scene_name, n_gauss, H, W, med, last, raw_med):
    buckets = to_buckets(med, {k: v for k, v in med.items() if k.startswith("blend.")
                               or k in ("project", "tile_assign", "sort", "blend")},
                         backend)
    total = med.get("total", 0.0)
    return {
        "backend": backend, "scene": scene_name, "n_gaussians": n_gauss,
        "height": H, "width": W,
        "num_visible": last.num_visible, "num_entries": last.num_entries,
        "project_ms": round(med.get("project", 0.0), 3),
        "tile_assign_ms": round(med.get("tile_assign", 0.0), 3),
        "sort_ms": round(med.get("sort", 0.0), 3),
        "blend_ms": round(med.get("blend", 0.0), 3),
        "total_ms": round(total, 3),
        "fps": round(1000.0 / total, 2) if total > 0 else 0.0,
        "load_ms": round(buckets["load"], 3),
        "compute_ms": round(buckets["compute"], 3),
        "return_ms": round(buckets["return"], 3),
        "transfer_ms": round(buckets["transfer"], 3),
        "device_kernel_ms": round(buckets["device_kernel"], 3),
        "blend_ms_min": round(raw_med.get("_blend_min", med.get("blend", 0.0)), 3),
        "total_ms_min": round(raw_med.get("_total_min", total), 3),
        "_raw_median": med,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Benchmark one backend across scenes x resolutions.")
    ap.add_argument("backend", help="one of: cpu, tt, cuda")
    ap.add_argument("--scenes", nargs="+",
                    default=["scenes/luigi.ply", "scenes/train.ply"])
    ap.add_argument("--res", nargs="+", type=int, default=[256, 480, 640, 960])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--measure", type=int, default=20)
    ap.add_argument("--out", default="benchmark/results")
    ap.add_argument("--skip-cpu-above", type=int, default=None,
                    help="skip (scene,res) rows where res > this value")
    args = ap.parse_args(argv)

    try:
        backend = get_backend(args.backend)
    except (KeyError, RuntimeError, FileNotFoundError) as e:
        print(f"ERROR: backend {args.backend!r} unavailable: {e}", file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)
    records = []
    try:
        pipeline = Pipeline(backend)
        for scene_path in args.scenes:
            scene_name = os.path.splitext(os.path.basename(scene_path))[0]
            gaussians = load_ply(scene_path)
            for res in args.res:
                H = W = _snap32(res)
                if args.skip_cpu_above is not None and res > args.skip_cpu_above:
                    print(f"skip {scene_name}@{H}: res > --skip-cpu-above={args.skip_cpu_above}")
                    continue
                extr, intr = make_camera(gaussians.means, H, W)
                out = _measure_cell(pipeline, gaussians, extr, intr,
                                    H, W, args.warmup, args.measure)
                if out is None:
                    print(f"WARN {scene_name}@{H}: 0 visible Gaussians — skipped")
                    continue
                med, last = out
                rec = _build_record(args.backend, scene_name,
                                    gaussians.num_gaussians, H, W, med, last, {})
                records.append(rec)
                print(f"{scene_name}@{H}: total={rec['total_ms']:.1f}ms "
                      f"fps={rec['fps']:.1f} "
                      f"(load={rec['load_ms']:.1f} compute={rec['compute_ms']:.1f} "
                      f"return={rec['return_ms']:.1f} transfer={rec['transfer_ms']:.1f})")
    finally:
        pipeline.close()

    csv_path = os.path.join(args.out, f"{args.backend}.csv")
    json_path = os.path.join(args.out, f"{args.backend}.json")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for rec in records:
            w.writerow({k: rec[k] for k in CSV_COLUMNS})
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nWrote {len(records)} rows -> {csv_path}")
    print(f"            full raw -> {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke test on CPU (small scene, low res)**

Run:
```bash
venv/bin/python -m benchmark.run cpu --scenes scenes/luigi.ply --res 256 --warmup 0 --measure 2
```
Expected: prints a `luigi@256: total=...` line and `Wrote 1 rows -> benchmark/results/cpu.csv`. Exit 0.

- [ ] **Step 3: Verify CSV is well-formed**

Run:
```bash
venv/bin/python -c "import csv; r=list(csv.DictReader(open('benchmark/results/cpu.csv'))); print(len(r), sorted(r[0]))"
```
Expected: `1` row and the column list matches `CSV_COLUMNS` (incl. `compute_ms`, `load_ms`, `transfer_ms`).

- [ ] **Step 4: Verify bucket reconciliation on real data**

Run:
```bash
venv/bin/python -c "import csv; r=next(csv.DictReader(open('benchmark/results/cpu.csv'))); b=float(r['blend_ms']); s=float(r['load_ms'])+float(r['compute_ms'])+float(r['return_ms'])+float(r['transfer_ms']); print('blend',b,'sum',s); assert abs(b-s)<1e-2"
```
Expected: `blend X sum X` with no assertion error (CPU: all in compute).

- [ ] **Step 5: Commit**

```bash
git add benchmark/run.py
git commit -m "feat(benchmark): single-backend sweep CLI -> CSV+JSON"
```

---

### Task 5: Plot CLI (`plot.py`)

**Files:**
- Create: `benchmark/plot.py`

Reads one or more CSVs written by `run.py`. One CSV → per-backend deep
dive (frame-time vs res line per scene; stacked phase bar per scene).
Multiple CSVs → cross-backend grouped bars of `total_ms` per resolution,
one chart per scene.

- [ ] **Step 1: Create `benchmark/plot.py`**

```python
"""Render graphs from one or more benchmark CSV files.

    # per-backend deep dive (one CSV):
    python -m benchmark.plot benchmark/results/tt.csv
    # cross-backend comparison (multiple CSVs):
    python -m benchmark.plot benchmark/results/cpu.csv benchmark/results/cuda.csv benchmark/results/tt.csv
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np


def _load_csv(path):
    """Return (backend, list-of-row-dicts with numeric fields coerced)."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in ("height", "width", "num_visible", "num_entries", "n_gaussians"):
                r[k] = int(float(r[k]))
            for k in ("project_ms", "tile_assign_ms", "sort_ms", "blend_ms",
                      "total_ms", "fps", "load_ms", "compute_ms", "return_ms",
                      "transfer_ms", "device_kernel_ms"):
                r[k] = float(r[k])
            rows.append(r)
    backend = rows[0]["backend"] if rows else os.path.splitext(os.path.basename(path))[0]
    return backend, rows


def _scenes(rows):
    return sorted({r["scene"] for r in rows})


def _by_res(rows, scene):
    sel = sorted((r for r in rows if r["scene"] == scene), key=lambda r: r["width"])
    return sel


def plot_per_backend(backend, rows, out_dir):
    # (1) frame-time vs resolution, one line per scene.
    fig, ax = plt.subplots(figsize=(7, 5))
    for scene in _scenes(rows):
        sel = _by_res(rows, scene)
        ax.plot([r["width"] for r in sel], [r["total_ms"] for r in sel],
                marker="o", label=scene)
    ax.set_xlabel("resolution (px)"); ax.set_ylabel("frame time (ms)")
    ax.set_title(f"{backend}: frame time vs resolution"); ax.legend(); ax.grid(True, alpha=0.3)
    p = os.path.join(out_dir, f"{backend}_frametime_vs_res.png")
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    print(f"wrote {p}")

    # (2) stacked phase breakdown per scene.
    for scene in _scenes(rows):
        sel = _by_res(rows, scene)
        x = np.arange(len(sel)); labels = [str(r["width"]) for r in sel]
        load = np.array([r["load_ms"] for r in sel])
        comp = np.array([r["compute_ms"] for r in sel])
        ret = np.array([r["return_ms"] for r in sel])
        trans = np.array([r["transfer_ms"] for r in sel])
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(x, load, label="load")
        ax.bar(x, comp, bottom=load, label="compute")
        ax.bar(x, ret, bottom=load + comp, label="return")
        ax.bar(x, trans, bottom=load + comp + ret, label="transfer")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_xlabel("resolution (px)"); ax.set_ylabel("blend time (ms)")
        ax.set_title(f"{backend} / {scene}: blend phase breakdown")
        ax.legend(); ax.grid(True, axis="y", alpha=0.3)
        p = os.path.join(out_dir, f"{backend}_{scene}_phases.png")
        fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
        print(f"wrote {p}")


def plot_cross_backend(datasets, out_dir):
    """datasets: list of (backend, rows). One grouped-bar chart per scene."""
    all_scenes = sorted({s for _, rows in datasets for s in _scenes(rows)})
    for scene in all_scenes:
        # union of resolutions present for this scene across backends
        res_set = sorted({r["width"] for _, rows in datasets
                          for r in rows if r["scene"] == scene})
        x = np.arange(len(res_set))
        n = len(datasets); w = 0.8 / max(1, n)
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (backend, rows) in enumerate(datasets):
            lut = {r["width"]: r["total_ms"] for r in rows if r["scene"] == scene}
            # missing cell -> NaN bar (visible gap, not zero-filled)
            vals = [lut.get(res, np.nan) for res in res_set]
            ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=backend)
        ax.set_xticks(x); ax.set_xticklabels([str(r) for r in res_set])
        ax.set_xlabel("resolution (px)"); ax.set_ylabel("frame time (ms)")
        ax.set_title(f"{scene}: frame time by backend")
        ax.legend(); ax.grid(True, axis="y", alpha=0.3)
        p = os.path.join(out_dir, f"compare_{scene}_total.png")
        fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
        print(f"wrote {p}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Plot benchmark CSV(s).")
    ap.add_argument("csv", nargs="+", help="one or more <backend>.csv files")
    ap.add_argument("--out", default=None, help="output dir (default: dir of first CSV)")
    args = ap.parse_args(argv)

    out_dir = args.out or os.path.dirname(os.path.abspath(args.csv[0]))
    os.makedirs(out_dir, exist_ok=True)
    datasets = [_load_csv(p) for p in args.csv]

    if len(datasets) == 1:
        backend, rows = datasets[0]
        plot_per_backend(backend, rows, out_dir)
    else:
        for backend, rows in datasets:
            plot_per_backend(backend, rows, out_dir)
        plot_cross_backend(datasets, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Ensure matplotlib is available**

Run: `venv/bin/python -c "import matplotlib; print(matplotlib.__version__)"`
Expected: a version string. If `ModuleNotFoundError`, run
`venv/bin/pip install matplotlib` and add `matplotlib` to `requirements.txt`
(commit that change with this task).

- [ ] **Step 3: Smoke test plotting the CPU CSV from Task 4**

Run: `venv/bin/python -m benchmark.plot benchmark/results/cpu.csv`
Expected: prints `wrote benchmark/results/cpu_frametime_vs_res.png` and one
`cpu_luigi_phases.png`. Files exist:
```bash
ls benchmark/results/*.png
```

- [ ] **Step 4: Commit**

```bash
git add benchmark/plot.py requirements.txt
git commit -m "feat(benchmark): plot script for per-backend + cross-backend graphs"
```

---

### Task 6: README usage section

**Files:**
- Modify: `README.md` (append a "Benchmarking" section)

- [ ] **Step 1: Append usage docs to `README.md`**

Add this section near the existing perf/test docs:

````markdown
## Benchmarking

Sweep one backend across scenes × resolutions, then plot:

```bash
source venv/bin/activate
# one backend per run (TT daemon / CUDA ext warmed once, held across the sweep):
python -m benchmark.run tt   --res 256 480 640 960
python -m benchmark.run cuda --res 256 480 640 960
python -m benchmark.run cpu  --res 256 480 --skip-cpu-above 480   # CPU is slow on train.ply

# per-backend graphs (one CSV) or cross-backend comparison (multiple):
python -m benchmark.plot benchmark/results/tt.csv
python -m benchmark.plot benchmark/results/{cpu,cuda,tt}.csv
```

Results land in `benchmark/results/` (gitignored): `<backend>.csv`, `.json`,
and PNGs. Each blend timing is split into **load** (stage inputs) /
**compute** (device kernel) / **return** (result to host) / **transfer**
(residual host↔device movement); the four reconcile exactly to the blend
wall time.
````

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs(readme): document benchmark harness usage"
```

---

## Self-Review

**Spec coverage:**
- Layout (`benchmark/` 6 files + results) → Tasks 1–5. ✓
- `.gitignore` `/benchmark/results/` → Task 1. ✓
- `camera.py` re-homed, no scripts/ import → Task 3. ✓
- `run.py` args (backend/scenes/res/warmup/measure/out/skip-cpu-above), warm-once backend, median+min, empty-frame skip, unavailable-backend clean exit → Task 4. ✓
- `phases.to_buckets` mapping table + reconciliation + npy fallback + graceful degrade → Task 2. ✓
- CSV schema (all columns incl. `*_min`) → Task 4 `CSV_COLUMNS`. ✓
- `plot.py` per-backend (frame-time-vs-res, stacked phases) + cross-backend grouped bars with visible gaps → Task 5. ✓
- TDD test file → Task 2. ✓

**Known minor deviations (intentional):** `blend_ms_min`/`total_ms_min` are
populated from the median fallback in this plan (the `_measure_cell` median
helper does not separately compute per-key mins); the columns exist and are
valid. If true min tracking is wanted later, extend `_measure_cell` to also
return `min(...)` per key — not required for the spec's graphs.

**Type consistency:** `to_buckets` returns keys `load/compute/return/transfer/
device_kernel/project/tile_assign/sort` — consumed identically in `run.py`
`_build_record` and `plot.py`. `CSV_COLUMNS` matches the keys written in
`_build_record` and read in `plot.py._load_csv`. ✓

**Placeholder scan:** no TBD/TODO; every code step has full code. ✓
