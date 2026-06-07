"""Dev harness: compare the in-process ttnn alpha_blend op vs the CPU reference.

Run on a TT box after building ttnn:
    source venv/bin/activate && python scripts/dev_ttnn_op_render.py
"""
import numpy as np
import torch

from gsplat.rasterization import (
    project_gaussians, get_tile_assignments, sort_and_bin, alpha_blend,
)
from backends.tt.backend import KernelBackend


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 100.0 if mse <= 0 else -10.0 * np.log10(mse)


def main():
    torch.manual_seed(42)
    H, W = 64, 64           # 2x2 tiles
    N = 50
    means = torch.rand(N, 3) * torch.tensor([2.0, 2.0, 0.0]) + torch.tensor([-1.0, -1.0, 1.0])
    scales = torch.rand(N, 3) * 0.1 + 0.02
    q = torch.randn(N, 4); q = q / q.norm(dim=-1, keepdim=True)
    colors = torch.rand(N, 3)
    opacities = torch.rand(N) * 0.5 + 0.2
    extrinsics = torch.eye(4)
    fx = fy = 40.0
    intrinsics = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=torch.float32)

    means_2d, covs_2d, depths, radii, valid = project_gaussians(
        means, scales, q, extrinsics, intrinsics, H, W)
    colors_v, opacities_v = colors[valid], opacities[valid]
    gids, tids, _ = get_tile_assignments(means_2d, radii, H, W, tile_size=32)
    tiles_x, tiles_y = (W + 31) // 32, (H + 31) // 32
    sorted_gids, tile_ranges = sort_and_bin(gids, tids, depths, tiles_x, tiles_y)

    cpu_img = alpha_blend(means_2d, covs_2d, colors_v, opacities_v,
                          sorted_gids, tile_ranges, H, W, tile_size=32).numpy()

    backend = KernelBackend()
    try:
        tt_img, timings = backend.blend(
            means_2d, covs_2d, colors_v, opacities_v, sorted_gids, tile_ranges, H, W)
    finally:
        backend.close()

    print("RESULT timings:", {k: round(v, 2) for k, v in timings.items()})
    print("RESULT cpu range", float(cpu_img.min()), float(cpu_img.max()))
    print("RESULT tt  range", float(tt_img.min()), float(tt_img.max()))
    print("RESULT shapes", cpu_img.shape, tt_img.shape)
    print(f"RESULT PSNR {psnr(cpu_img, tt_img):.2f} dB")


if __name__ == "__main__":
    main()
