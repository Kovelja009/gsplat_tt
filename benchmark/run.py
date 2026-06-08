"""Sweep ONE backend across scenes x resolutions; write CSV + JSON.

The backend is constructed once and held across the whole sweep (TT device
opened once + ttnn program cache stays warm, CUDA extension stays loaded) —
one-shot init cost is excluded, matching real interactive use. Per (scene,
resolution) cell we run `warmup`
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


def _measure_cell(pipeline, gaussians, extr, intr, H, W, warmup, measure, backend):
    """Run warmup+measure renders; return (median, mins, bucket_median, last)
    or None if every measured frame had zero visible Gaussians.

    Buckets are computed per frame then medianed (not bucketed from the median
    timings) so each bucket stays >= 0 — `transfer` is a per-frame residual and
    medianing the raw timings independently could otherwise drive it negative."""
    for _ in range(warmup):
        pipeline.render(gaussians, extr, intr, H, W)
    rows, bucket_rows = [], []
    for _ in range(measure):
        r = pipeline.render(gaussians, extr, intr, H, W)
        if r.num_visible == 0:
            continue
        row = dict(r.timings)
        row.update(r.sub_timings)
        rows.append((row, r))
        # to_buckets reads "blend"/"project" from timings and "blend.*" from
        # sub_timings; `row` carries both, so pass it as both arguments.
        bucket_rows.append(to_buckets(row, row, backend))
    if not rows:
        return None
    keys = sorted({k for row, _ in rows for k in row})
    med = {k: statistics.median(row[k] for row, _ in rows if k in row) for k in keys}
    mins = {k: min(row[k] for row, _ in rows if k in row) for k in keys}
    bucket_med = {k: statistics.median(b[k] for b in bucket_rows) for k in bucket_rows[0]}
    last = rows[-1][1]
    return med, mins, bucket_med, last


def _build_record(backend, scene_name, n_gauss, H, W, med, mins, buckets, last):
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
        "blend_ms_min": round(mins.get("blend", med.get("blend", 0.0)), 3),
        "total_ms_min": round(mins.get("total", total), 3),
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
                                    H, W, args.warmup, args.measure, args.backend)
                if out is None:
                    print(f"WARN {scene_name}@{H}: 0 visible Gaussians — skipped")
                    continue
                med, mins, bucket_med, last = out
                rec = _build_record(args.backend, scene_name,
                                    gaussians.num_gaussians, H, W, med, mins, bucket_med, last)
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
