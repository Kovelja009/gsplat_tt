"""Quantify entry reduction from tile-assigning with a per-axis AABB
(extent_x=3*sqrt(cov00), extent_y=3*sqrt(cov11)) vs the current circular bound
(radius=3*sqrt(lambda_max), same in x and y). Fewer entries -> less compute,
sort, and prep. Near-lossless: it only drops tiles a Gaussian's circular bound
reaches but its ellipse does not."""
import os
import sys

import numpy as np
import torch

from gsplat.loading_gaussians import load_ply
from gsplat.rasterization import project_gaussians

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_scene import make_camera  # noqa: E402


def entry_count(mx, my, rx, ry, tiles_x, tiles_y, tile=32):
    tminx = np.clip(((mx - rx) / tile).astype(int), 0, tiles_x - 1)
    tmaxx = np.clip(((mx + rx) / tile).astype(int), 0, tiles_x - 1)
    tminy = np.clip(((my - ry) / tile).astype(int), 0, tiles_y - 1)
    tmaxy = np.clip(((my + ry) / tile).astype(int), 0, tiles_y - 1)
    w = (tmaxx - tminx + 1); h = (tmaxy - tminy + 1)
    return int((w * h).sum()), w * h


def main(ply="scenes/point_cloud.ply", H=640, W=640):
    g = load_ply(ply)
    extr, intr = make_camera(g.means, H, W)
    m2d, c2d, depths, radii, valid = project_gaussians(
        g.means, g.scales, g.rotations, extr, intr, H, W, opacities=g.opacities)
    tx, ty = (W + 31) // 32, (H + 31) // 32
    mx = m2d[:, 0].numpy(); my = m2d[:, 1].numpy()
    r = radii.numpy()
    cov = c2d.numpy()
    rx = np.ceil(3.0 * np.sqrt(np.maximum(cov[:, 0, 0], 0)))
    ry = np.ceil(3.0 * np.sqrt(np.maximum(cov[:, 1, 1], 0)))

    print(f"{ply}  {H}x{W}  visible={mx.shape[0]:,}")
    p_circ, _ = entry_count(mx, my, r, r, tx, ty)
    p_aabb, _ = entry_count(mx, my, rx, ry, tx, ty)
    print(f"circular (current): {p_circ:,} entries")
    print(f"per-axis AABB:      {p_aabb:,} entries   ({(1 - p_aabb / p_circ) * 100:.1f}% fewer)")
    # how anisotropic are the Gaussians?
    ratio = np.maximum(rx, ry) / np.maximum(np.minimum(rx, ry), 1e-6)
    print(f"axis ratio (max/min extent): median={np.median(ratio):.2f}  p90={np.percentile(ratio,90):.2f}")


if __name__ == "__main__":
    main(*( [sys.argv[1]] if len(sys.argv) > 1 else [] ))
