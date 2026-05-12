"""Stable A/B bench harness for the CUDA alpha-blend kernel.

Renders the same 640x640 / 10K-Gaussian scene N times after warmup;
reports min and median kernel.device (CUDA event), kernel wall, and
upload. Use to compare kernel variants — much lower variance than
the one-shot perf test.
"""
import statistics as stats
import sys

import torch

from backends.cuda.backend import CudaBackend
from gsplat.rasterization import (
    project_gaussians, get_tile_assignments, sort_and_bin,
)


def main(iters: int = 50, warmup: int = 5) -> None:
    torch.manual_seed(42)
    H, W = 640, 640
    N = 10_000
    means = torch.rand(N, 3) * torch.tensor([2.0, 2.0, 2.0]) + torch.tensor([-1.0, -1.0, 1.0])
    scales = torch.rand(N, 3) * 0.1 + 0.02
    q = torch.randn(N, 4); q = q / q.norm(dim=-1, keepdim=True)
    colors = torch.rand(N, 3)
    opacities = torch.rand(N) * 0.5 + 0.2

    extrinsics = torch.eye(4)
    fx = fy = 400.0
    intrinsics = torch.tensor(
        [[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=torch.float32
    )

    means_2d, covs_2d, depths, radii, valid = project_gaussians(
        means, scales, q, extrinsics, intrinsics, H, W,
    )
    colors_v = colors[valid]
    opacities_v = opacities[valid]
    gids, tids, _ = get_tile_assignments(means_2d, radii, H, W, tile_size=32)
    tiles_x = (W + 31) // 32
    tiles_y = (H + 31) // 32
    sorted_gids, tile_ranges = sort_and_bin(gids, tids, depths, tiles_x, tiles_y)

    backend = CudaBackend()
    args = (means_2d, covs_2d, colors_v, opacities_v, sorted_gids, tile_ranges, H, W)

    # Warm up
    for _ in range(warmup):
        backend.blend(*args)
    torch.cuda.synchronize()

    dev_times: list[float] = []
    kern_times: list[float] = []
    up_times: list[float] = []
    for _ in range(iters):
        _, sub = backend.blend(*args)
        dev_times.append(sub["kernel.device"])
        kern_times.append(sub["kernel"])
        up_times.append(sub["upload"])

    def fmt(name: str, vals: list[float]) -> str:
        return (f"{name:>20s}: min={min(vals):>6.3f}  median={stats.median(vals):>6.3f}  "
                f"p90={sorted(vals)[int(len(vals)*0.9)]:>6.3f}  ms")

    print(f"Scene: H={H} W={W}, N={N} input, {int(valid.sum()):,} visible, "
          f"{sorted_gids.numel():,} sorted entries, {tiles_x * tiles_y} tiles")
    print(f"iters={iters}, warmup={warmup}")
    print(fmt("kernel.device", dev_times))
    print(fmt("kernel (wall+D2H)", kern_times))
    print(fmt("upload", up_times))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    main(iters=n)
