"""Verify + time the radix sort_and_bin against the previous float-key argsort.

Correctness: identical tile_ranges, and (on the small scene) a CPU-blended image
identical to the old ordering. Perf: median sort time old vs new on real inputs.

    venv/bin/python scripts/verify_sort.py
"""
import os
import statistics
import sys
import time

import numpy as np
import torch

from gsplat.loading_gaussians import load_ply
from gsplat.rasterization import (
    project_gaussians, get_tile_assignments, sort_and_bin, alpha_blend,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_scene import make_camera  # noqa: E402


def old_sort_and_bin(gaussian_ids, tile_ids, depths, tiles_x, tiles_y):
    """Previous implementation: float composite key + torch.argsort."""
    num_tiles = tiles_x * tiles_y
    max_depth = depths.max().item() + 1.0
    sort_keys = tile_ids.float() * max_depth + depths[gaussian_ids]
    sorted_indices = torch.argsort(sort_keys)
    sorted_gaussian_ids = gaussian_ids[sorted_indices]
    sorted_tile_ids = tile_ids[sorted_indices]
    tile_ranges = torch.zeros(num_tiles, 2, dtype=torch.int64)
    if sorted_tile_ids.numel() > 0:
        changes = sorted_tile_ids[1:] != sorted_tile_ids[:-1]
        change_indices = torch.where(changes)[0] + 1
        starts = torch.cat([torch.zeros(1, dtype=torch.int64), change_indices])
        ends = torch.cat([change_indices, torch.tensor([len(sorted_tile_ids)])])
        segment_tiles = sorted_tile_ids[starts]
        tile_ranges[segment_tiles, 0] = starts
        tile_ranges[segment_tiles, 1] = ends
    return sorted_gaussian_ids, tile_ranges


def build_inputs(ply, H=640, W=640):
    g = load_ply(ply)
    extr, intr = make_camera(g.means, H, W)
    m2d, c2d, depths, radii, valid = project_gaussians(
        g.means, g.scales, g.rotations, extr, intr, H, W, opacities=g.opacities)
    cols, ops = g.colors[valid], g.opacities[valid]
    gids, tids, _ = get_tile_assignments(m2d, radii, H, W, tile_size=32)
    tx, ty = (W + 31) // 32, (H + 31) // 32
    return dict(gids=gids, tids=tids, depths=depths, tx=tx, ty=ty,
                m2d=m2d, c2d=c2d, cols=cols, ops=ops, H=H, W=W)


def timeit(fn, n=7):
    fn()  # warm
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000.0)
    return statistics.median(ts)


def main():
    for ply, do_image in [("scenes/luigi.ply", True), ("scenes/point_cloud.ply", False)]:
        inp = build_inputs(ply)
        old_g, old_r = old_sort_and_bin(inp["gids"], inp["tids"], inp["depths"], inp["tx"], inp["ty"])
        new_g, new_r = sort_and_bin(inp["gids"], inp["tids"], inp["depths"], inp["tx"], inp["ty"])

        ranges_match = torch.equal(old_r, new_r)
        old_ms = timeit(lambda: old_sort_and_bin(inp["gids"], inp["tids"], inp["depths"], inp["tx"], inp["ty"]))
        new_ms = timeit(lambda: sort_and_bin(inp["gids"], inp["tids"], inp["depths"], inp["tx"], inp["ty"]))

        print(f"\n=== {ply}  entries={inp['gids'].shape[0]:,} ===")
        print(f"tile_ranges identical: {ranges_match}")
        print(f"sort time   old={old_ms:7.2f} ms   new={new_ms:7.2f} ms   speedup={old_ms/new_ms:.2f}x")

        if do_image:
            old_img = alpha_blend(inp["m2d"], inp["c2d"], inp["cols"], inp["ops"],
                                  old_g, old_r, inp["H"], inp["W"]).numpy()
            new_img = alpha_blend(inp["m2d"], inp["c2d"], inp["cols"], inp["ops"],
                                  new_g, new_r, inp["H"], inp["W"]).numpy()
            mse = float(np.mean((old_img - new_img) ** 2))
            psnr = 100.0 if mse <= 0 else -10.0 * np.log10(mse)
            print(f"image old-vs-new: max|diff|={float(np.max(np.abs(old_img-new_img))):.2e}  PSNR={psnr:.2f} dB")


if __name__ == "__main__":
    main()
