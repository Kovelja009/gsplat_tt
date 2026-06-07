"""A/B benchmark: staging the daemon hand-off .npy files on disk (/tmp) vs
tmpfs (/dev/shm).

Drives the real KernelBackend.blend(...) end-to-end on identical inputs, so
the only thing that varies between the two runs is where the four input
.npy files (and out.npy) are written/read. The per-frame device work is
byte-identical, so blend.daemon_rt.device_kernel should match across configs
and any delta in prep/save_npy/daemon_rt/load_npy is pure I/O.

Run from repo root:  venv/bin/python scripts/bench_stage_dir.py
"""
import os
import statistics
import time

import torch

from gsplat.rasterization import (
    project_gaussians, get_tile_assignments, sort_and_bin,
)
from backends.tt.backend import KernelBackend

H, W = 640, 640
N = 10_000
WARMUP = 3
MEASURE = 20


def build_inputs():
    torch.manual_seed(42)
    means = torch.rand(N, 3) * torch.tensor([2.0, 2.0, 2.0]) + torch.tensor([-1.0, -1.0, 1.0])
    scales = torch.rand(N, 3) * 0.1 + 0.02
    q = torch.randn(N, 4); q = q / q.norm(dim=-1, keepdim=True)
    colors = torch.rand(N, 3)
    opacities = torch.rand(N) * 0.5 + 0.2
    extrinsics = torch.eye(4)
    fx = fy = 400.0
    intrinsics = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=torch.float32)

    means_2d, covs_2d, depths, radii, valid = project_gaussians(
        means, scales, q, extrinsics, intrinsics, H, W,
    )
    colors_v = colors[valid]
    opacities_v = opacities[valid]
    gids, tids, _ = get_tile_assignments(means_2d, radii, H, W, tile_size=32)
    tiles_x, tiles_y = (W + 31) // 32, (H + 31) // 32
    sorted_gids, tile_ranges = sort_and_bin(gids, tids, depths, tiles_x, tiles_y)
    return dict(
        means_2d=means_2d, covs_2d=covs_2d, colors=colors_v, opacities=opacities_v,
        sorted_gaussian_ids=sorted_gids, tile_ranges=tile_ranges,
        image_height=H, image_width=W,
    )


def bench(stage_dir, inputs):
    backend = KernelBackend(stage_dir=stage_dir)
    try:
        for _ in range(WARMUP):
            backend.blend(**inputs)
        rows = []
        for _ in range(MEASURE):
            t0 = time.perf_counter()
            _, sub = backend.blend(**inputs)
            sub["wall"] = (time.perf_counter() - t0) * 1000.0
            rows.append(sub)
    finally:
        backend.close()
    keys = ["prep", "save_npy", "daemon_rt", "daemon_rt.device_kernel", "load_npy", "wall"]
    return {k: statistics.median(r[k] for r in rows if k in r) for k in keys
            if any(k in r for r in rows)}


def main():
    inputs = build_inputs()
    print(f"Scene: {H}x{W}, {N} Gaussians, {inputs['sorted_gaussian_ids'].numel()} sorted entries")
    print(f"Warmup {WARMUP}, measure {MEASURE} frames (median ms)\n")

    configs = [("/tmp (disk/overlay)", "/tmp"), ("/dev/shm (tmpfs)", "/dev/shm")]
    results = {}
    for label, sd in configs:
        print(f"--- {label} ---")
        results[label] = bench(sd, inputs)
        print()

    cols = list(results)
    keys = ["prep", "save_npy", "daemon_rt", "daemon_rt.device_kernel", "load_npy", "wall"]
    w = 26
    print(f"{'metric':<26}" + "".join(f"{c:>{w}}" for c in cols) + f"{'Δ (shm−disk)':>{w}}")
    for k in keys:
        if not all(k in results[c] for c in cols):
            continue
        a, b = results[cols[0]][k], results[cols[1]][k]
        print(f"{k:<26}" + f"{a:>{w}.2f}" + f"{b:>{w}.2f}" + f"{b - a:>{w}.2f}")


if __name__ == "__main__":
    main()
