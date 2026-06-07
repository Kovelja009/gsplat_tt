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
