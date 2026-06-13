"""Compare TT scheduling strategies: plot compute_ms per scene, grouped by sched.

    python -m benchmark.compare_sched \
        benchmark/results/sched_round_robin/tt.csv \
        benchmark/results/sched_lpt/tt.csv \
        benchmark/results/sched_segmented/tt.csv \
        --out benchmark/results/sched_compare

Reads each CSV, tags rows by their `sched` column (falling back to the parent
directory name with any "sched_" prefix stripped), then emits one grouped-bar
plot of compute_ms per scene (x = resolution, one bar per sched) plus a combined
CSV of all rows.
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np

_INT_FIELDS = ("height", "width", "num_visible", "num_entries", "n_gaussians")
_FLOAT_FIELDS = ("project_ms", "tile_assign_ms", "sort_ms", "blend_ms",
                 "total_ms", "fps", "load_ms", "compute_ms", "return_ms",
                 "transfer_ms", "device_kernel_ms", "blend_ms_min", "total_ms_min")


def load_tagged(path):
    """Read a benchmark CSV; return row-dicts with numerics coerced and a `sched`
    field guaranteed present (from the column, else the parent dir name with any
    'sched_' prefix stripped)."""
    fallback = os.path.basename(os.path.dirname(os.path.abspath(path)))
    if fallback.startswith("sched_"):
        fallback = fallback[len("sched_"):]
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in _INT_FIELDS:
                if k in r and r[k] != "":
                    r[k] = int(float(r[k]))
            for k in _FLOAT_FIELDS:
                if k in r and r[k] != "":
                    r[k] = float(r[k])
            if not r.get("sched"):
                r["sched"] = fallback
            rows.append(r)
    return rows


def group_by_sched(rows):
    """Partition rows into {sched: [rows]} preserving input order."""
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r["sched"], []).append(r)
    return groups


def _scenes(rows):
    return sorted({r["scene"] for r in rows})


def plot_compute_by_sched(groups, out_dir):
    """One grouped-bar chart per scene: x = resolution, one bar per sched,
    y = compute_ms. Missing cells render as NaN (visible gap)."""
    scheds = list(groups)
    all_scenes = sorted({r["scene"] for rows in groups.values() for r in rows})
    for scene in all_scenes:
        res_set = sorted({r["width"] for rows in groups.values()
                          for r in rows if r["scene"] == scene})
        x = np.arange(len(res_set))
        n = len(scheds); w = 0.8 / max(1, n)
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, sched in enumerate(scheds):
            lut = {r["width"]: r["compute_ms"]
                   for r in groups[sched] if r["scene"] == scene}
            vals = [lut.get(res, np.nan) for res in res_set]
            ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=sched)
        ax.set_xticks(x); ax.set_xticklabels([str(r) for r in res_set])
        ax.set_xlabel("resolution (px)"); ax.set_ylabel("compute (ms)")
        ax.set_title(f"{scene}: kernel compute by scheduling strategy")
        ax.legend(); ax.grid(True, axis="y", alpha=0.3)
        p = os.path.join(out_dir, f"sched_compare_{scene}_compute.png")
        fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
        print(f"wrote {p}")


def write_combined_csv(groups, out_dir):
    """Concatenate all rows (with sched) into one CSV for downstream analysis."""
    all_rows = [r for rows in groups.values() for r in rows]
    if not all_rows:
        return
    cols = list(all_rows[0].keys())
    p = os.path.join(out_dir, "sched_compare.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"wrote {p}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare TT scheduling strategies from CSVs.")
    ap.add_argument("csv", nargs="+", help="per-config <backend>.csv files")
    ap.add_argument("--out", default="benchmark/results/sched_compare",
                    help="output dir for plots + combined CSV")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    rows = [r for p in args.csv for r in load_tagged(p)]
    groups = group_by_sched(rows)
    plot_compute_by_sched(groups, args.out)
    write_combined_csv(groups, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
