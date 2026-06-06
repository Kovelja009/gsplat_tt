"""Confirm the per-axis AABB tile bound is near-lossless vs the old circular
bound: render both (CPU) and compare PSNR + entry counts."""
import os
import sys

import numpy as np

from gsplat.loading_gaussians import load_ply
from gsplat.rasterization import (
    project_gaussians, get_tile_assignments, sort_and_bin, alpha_blend,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_scene import make_camera  # noqa: E402


def render(m2d, c2d, cols, ops, depths, radii, H, W, covs):
    gids, tids, _ = get_tile_assignments(m2d, radii, H, W, tile_size=32, covs_2d=covs)
    sg, rg = sort_and_bin(gids, tids, depths, (W + 31) // 32, (H + 31) // 32)
    return alpha_blend(m2d, c2d, cols, ops, sg, rg, H, W).numpy(), gids.shape[0]


def main(ply="scenes/luigi.ply", H=640, W=640):
    g = load_ply(ply)
    extr, intr = make_camera(g.means, H, W)
    m2d, c2d, depths, radii, valid = project_gaussians(
        g.means, g.scales, g.rotations, extr, intr, H, W, opacities=g.opacities)
    cols, ops = g.colors[valid], g.opacities[valid]

    circ_img, p_circ = render(m2d, c2d, cols, ops, depths, radii, H, W, None)
    aabb_img, p_aabb = render(m2d, c2d, cols, ops, depths, radii, H, W, c2d)
    mse = float(np.mean((circ_img - aabb_img) ** 2))
    psnr = 100.0 if mse <= 0 else -10.0 * np.log10(mse)
    print(f"{ply}  {H}x{W}")
    print(f"entries circular={p_circ:,}  aabb={p_aabb:,}  ({(1-p_aabb/p_circ)*100:.1f}% fewer)")
    print(f"AABB vs circular image: max|diff|={float(np.max(np.abs(circ_img-aabb_img))):.2e}  PSNR={psnr:.2f} dB")


if __name__ == "__main__":
    main(*( [sys.argv[1]] if len(sys.argv) > 1 else [] ))
