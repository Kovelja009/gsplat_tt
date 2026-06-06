"""Diagnose compute load: per-tile entry distribution and simulated LPT core
balance. The device blend makespan is set by the heaviest core, so if LPT is
imbalanced (max core load >> mean) that's wasted device time."""
import os
import sys

import numpy as np
import torch

from gsplat.loading_gaussians import load_ply
from gsplat.rasterization import project_gaussians, get_tile_assignments, sort_and_bin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_scene import make_camera  # noqa: E402


def lpt_max_load(counts, num_cores):
    """Replicate compute_lpt_assignment: drop empty tiles, sort desc, greedily
    assign each to the least-loaded core. Return (max_load, mean_load, n_nonempty)."""
    nz = counts[counts > 0]
    nz = np.sort(nz)[::-1]
    load = np.zeros(num_cores, dtype=np.int64)
    for c in nz:
        i = int(np.argmin(load))
        load[i] += int(c)
    return load.max(), load.mean(), len(nz)


def main(ply="scenes/point_cloud.ply", H=640, W=640):
    g = load_ply(ply)
    extr, intr = make_camera(g.means, H, W)
    m2d, c2d, depths, radii, valid = project_gaussians(
        g.means, g.scales, g.rotations, extr, intr, H, W, opacities=g.opacities)
    gids, tids, _ = get_tile_assignments(m2d, radii, H, W, tile_size=32)
    tx, ty = (W + 31) // 32, (H + 31) // 32
    num_tiles = tx * ty
    counts = np.bincount(tids.numpy(), minlength=num_tiles)
    P = int(counts.sum())

    print(f"{ply}  {H}x{W}  tiles={num_tiles} ({tx}x{ty})  entries={P:,}")
    nz = counts[counts > 0]
    print(f"per-tile entries: nonempty={len(nz)}/{num_tiles}  "
          f"mean={counts.mean():.0f}  median={np.median(nz):.0f}  "
          f"max={counts.max()}  p99={np.percentile(nz,99):.0f}")
    print(f"heaviest tile = {counts.max()} entries = {counts.max()/P*100:.1f}% of all work\n")

    for nc in (64, 130):
        mx, mn, ne = lpt_max_load(counts, nc)
        # makespan is set by max core; ideal is total/cores. imbalance = max/ideal.
        # but a single tile can't be split, so max >= heaviest tile.
        ideal = P / nc
        print(f"cores={nc:3d}:  LPT max_core={mx:>8,}  mean={mn:>8,.0f}  "
              f"ideal={ideal:>8,.0f}  imbalance(max/ideal)={mx/ideal:.2f}x  "
              f"(heaviest single tile caps it at {counts.max():,})")


if __name__ == "__main__":
    main(*sys.argv[1:])
