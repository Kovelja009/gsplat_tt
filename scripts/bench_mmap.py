"""Verify + benchmark the zero-copy shared-memory (MFRAME) path against the
.npy (FRAME) path, both staged on /dev/shm.

Correctness: the mmap path uploads byte-identical device buffers to the .npy
path, so their outputs should match bit-for-bit; both are checked against the
CPU reference via PSNR.

Perf: median per-frame wall + sub-timings for each path on identical inputs.

Run from repo root:  venv/bin/python scripts/bench_mmap.py
"""
import statistics
import time

import numpy as np
import torch

from gsplat.rasterization import (
    project_gaussians, get_tile_assignments, sort_and_bin, alpha_blend,
)
from backends.tt.backend import KernelBackend

H, W = 640, 640
N = 10_000
WARMUP = 3
MEASURE = 20


def build():
    torch.manual_seed(42)
    means = torch.rand(N, 3) * torch.tensor([2.0, 2.0, 2.0]) + torch.tensor([-1.0, -1.0, 1.0])
    scales = torch.rand(N, 3) * 0.1 + 0.02
    q = torch.randn(N, 4); q = q / q.norm(dim=-1, keepdim=True)
    colors = torch.rand(N, 3)
    opacities = torch.rand(N) * 0.5 + 0.2
    extrinsics = torch.eye(4)
    fx = fy = 400.0
    intrinsics = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=torch.float32)
    m2d, c2d, depths, radii, valid = project_gaussians(means, scales, q, extrinsics, intrinsics, H, W)
    cols, ops = colors[valid], opacities[valid]
    gids, tids, _ = get_tile_assignments(m2d, radii, H, W, tile_size=32)
    tx, ty = (W + 31) // 32, (H + 31) // 32
    sgids, tr = sort_and_bin(gids, tids, depths, tx, ty)
    inputs = dict(means_2d=m2d, covs_2d=c2d, colors=cols, opacities=ops,
                  sorted_gaussian_ids=sgids, tile_ranges=tr,
                  image_height=H, image_width=W)
    cpu_img = alpha_blend(m2d, c2d, cols, ops, sgids, tr, H, W, tile_size=32).numpy()
    return inputs, cpu_img


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 100.0 if mse <= 0 else -10.0 * np.log10(mse)


def measure(backend, inputs):
    for _ in range(WARMUP):
        backend.blend(**inputs)
    rows, img = [], None
    for _ in range(MEASURE):
        t = time.perf_counter()
        img, sub = backend.blend(**inputs)
        sub["wall"] = (time.perf_counter() - t) * 1000.0
        rows.append(sub)
    keys = sorted({k for r in rows for k in r})
    return img, {k: statistics.median(r[k] for r in rows if k in r) for k in keys}


def main():
    inputs, cpu_img = build()
    print(f"Scene: {H}x{W}, {N} Gaussians, {inputs['sorted_gaussian_ids'].numel()} entries\n")

    backend = KernelBackend()
    try:
        assert backend._mm is not None, "shared-memory handoff did not initialize"

        # --- mmap path ---
        mmap_img, mmap_t = measure(backend, inputs)

        # --- .npy path (force fallback by detaching the mmap) ---
        saved = backend._mm
        backend._mm = None
        npy_img, npy_t = measure(backend, inputs)
        backend._mm = saved
    finally:
        backend.close()

    print("=== correctness ===")
    print(f"PSNR mmap vs CPU ref:  {psnr(mmap_img, cpu_img):6.2f} dB")
    print(f"PSNR npy  vs CPU ref:  {psnr(npy_img, cpu_img):6.2f} dB")
    print(f"max|mmap - npy|:       {float(np.max(np.abs(mmap_img - npy_img))):.6g}  "
          f"(expect 0 — identical device bytes)\n")

    print("=== perf (median ms) ===")
    allk = ["prep", "save_npy", "write_shm", "daemon_rt", "mframe_rt",
            "daemon_rt.device_kernel", "mframe_rt.device_kernel",
            "load_npy", "read_shm", "wall"]
    w = 14
    print(f"{'metric':<26}{'npy(/dev/shm)':>{w}}{'mmap':>{w}}")
    for k in allk:
        a, b = npy_t.get(k), mmap_t.get(k)
        if a is None and b is None:
            continue
        sa = f"{a:>{w}.2f}" if a is not None else f"{'-':>{w}}"
        sb = f"{b:>{w}.2f}" if b is not None else f"{'-':>{w}}"
        print(f"{k:<26}{sa}{sb}")
    print(f"\nwall speedup: {npy_t['wall'] / mmap_t['wall']:.2f}x")


if __name__ == "__main__":
    main()
