"""Bound the whole-tile early-out win: cap each tile to its front-N depth-sorted
Gaussians and measure device-kernel time + PSNR vs the uncapped render.

A perfect early-out would skip a tile's remaining Gaussians once all its pixels
saturate. Capping front-N is an upper bound on that: if PSNR stays high at small
N, the back Gaussians are occluded (early-out is a big, safe win); if PSNR drops,
they still contribute (early-out won't help much on this scene).

    venv/bin/python scripts/cap_experiment.py scenes/point_cloud.ply --res 640
"""
import argparse
import os
import statistics
import sys
import time

import numpy as np
import torch

from gsplat.loading_gaussians import load_ply
from gsplat.rasterization import project_gaussians, get_tile_assignments, sort_and_bin
from backends import get_backend

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_scene import make_camera  # noqa: E402


def cap_tiles(sorted_gids, tile_ranges, N):
    """Keep only the front-N entries of each tile; return compacted (gids, ranges)."""
    sg = sorted_gids.numpy()
    rg = tile_ranges.numpy()
    parts, starts, ends = [], [], []
    pos = 0
    for s, e in rg:
        e2 = min(int(e), int(s) + N)
        parts.append(sg[int(s):e2])
        cnt = e2 - int(s)
        starts.append(pos); ends.append(pos + cnt); pos += cnt
    new_g = np.concatenate(parts) if parts else sg[:0]
    new_r = np.stack([np.array(starts), np.array(ends)], axis=1).astype(np.int64)
    return torch.from_numpy(new_g), torch.from_numpy(new_r)


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 100.0 if mse <= 0 else -10.0 * np.log10(mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("--res", type=int, default=640)
    args = ap.parse_args()
    H = W = (args.res // 32) * 32

    g = load_ply(args.ply)
    extr, intr = make_camera(g.means, H, W)
    m2d, c2d, depths, radii, valid = project_gaussians(
        g.means, g.scales, g.rotations, extr, intr, H, W, opacities=g.opacities)
    cols, ops = g.colors[valid], g.opacities[valid]
    gids, tids, _ = get_tile_assignments(m2d, radii, H, W, tile_size=32)
    tx, ty = (W + 31) // 32, (H + 31) // 32
    sg, rg = sort_and_bin(gids, tids, depths, tx, ty)
    print(f"{args.ply}  {H}x{W}  entries={sg.numel():,}")

    be = get_backend("tt")
    def render(sgi, rgi, reps=8):
        be.blend(m2d, c2d, cols, ops, sgi, rgi, H, W)  # warm
        ks, img = [], None
        for _ in range(reps):
            img, sub = be.blend(m2d, c2d, cols, ops, sgi, rgi, H, W)
            ks.append(sub.get("mframe_rt.device_kernel", float("nan")))
        return img, statistics.median(ks)
    try:
        ref_img, ref_k = render(sg, rg)
        full_entries = int(sg.numel())
        print(f"\n{'cap N':>8}{'entries':>12}{'device_kernel ms':>18}{'PSNR vs full':>14}")
        print(f"{'full':>8}{full_entries:>12,}{ref_k:>18.2f}{'-':>14}")
        for N in [2000, 1000, 500, 250, 100]:
            cg, cr = cap_tiles(sg, rg, N)
            img, k = render(cg, cr)
            print(f"{N:>8}{int(cg.numel()):>12,}{k:>18.2f}{psnr(img, ref_img):>13.2f} dB")
    finally:
        be.close()


if __name__ == "__main__":
    main()
