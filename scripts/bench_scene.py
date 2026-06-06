"""End-to-end real-scene benchmark for the full pipeline.

Unlike scripts/bench_mmap.py (which times only the blend stage on a synthetic
random scene), this drives Pipeline.render() on a real .ply so the CPU stages
that run every frame — project / tile_assign / sort — are measured alongside
blend. That's the honest per-frame cost an interactive viewer sees.

Usage:
    venv/bin/python scripts/bench_scene.py scenes/point_cloud.ply --res 640
    venv/bin/python scripts/bench_scene.py scenes/luigi.ply --res 640 --cpu
"""
import argparse
import statistics
import time

import numpy as np
import torch

from gsplat.loading_gaussians import load_ply
from gsplat.pipeline import Pipeline
from gsplat.utils import c2w_to_w2c
from backends import get_backend


def look_at_c2w(center, eye, up=(0.0, -1.0, 0.0)):
    """OpenCV-convention c2w (+Z forward, +Y down) for a camera at `eye`."""
    center = np.asarray(center, np.float64); eye = np.asarray(eye, np.float64)
    up = np.asarray(up, np.float64)
    z = center - eye; z /= np.linalg.norm(z)          # forward
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(np.array([0.0, 0.0, 1.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)                                  # down
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = x; c2w[:3, 1] = y; c2w[:3, 2] = z; c2w[:3, 3] = eye
    return c2w


def make_camera(means, H, W, fov_deg=60.0):
    """Robust look-at framing the inlier cloud (median center, p60 radius)."""
    m = means.numpy()
    center = np.median(m, axis=0)
    dist = np.linalg.norm(m - center, axis=1)
    radius = float(np.percentile(dist, 60))
    eye = center + np.array([0.0, 0.0, 2.5 * radius], np.float32)
    c2w = look_at_c2w(center, eye)
    extrinsics = c2w_to_w2c(c2w)
    f = 0.5 * W / np.tan(0.5 * np.radians(fov_deg))
    intrinsics = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=torch.float32)
    return extrinsics, intrinsics


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 100.0 if mse <= 0 else -10.0 * np.log10(mse)


def run(pipeline, gaussians, extr, intr, H, W, warmup, measure):
    for _ in range(warmup):
        pipeline.render(gaussians, extr, intr, H, W)
    rows, last = [], None
    for _ in range(measure):
        t = time.perf_counter()
        r = pipeline.render(gaussians, extr, intr, H, W)
        wall = (time.perf_counter() - t) * 1000.0
        d = dict(r.timings); d.update(r.sub_timings); d["wall"] = wall
        rows.append(d); last = r
    keys = sorted({k for r in rows for k in r})
    med = {k: statistics.median(r[k] for r in rows if k in r) for k in keys}
    return med, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("--res", type=int, default=640)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--measure", type=int, default=20)
    ap.add_argument("--cpu", action="store_true", help="also run CPU backend (slow on big scenes)")
    args = ap.parse_args()

    H = W = (args.res // 32) * 32
    gaussians = load_ply(args.ply)
    extr, intr = make_camera(gaussians.means, H, W)

    tt = Pipeline(get_backend("tt"))
    try:
        tt_med, tt_last = run(tt, gaussians, extr, intr, H, W, args.warmup, args.measure)
    finally:
        tt.close()

    print(f"\nScene: {args.ply}  N={gaussians.num_gaussians:,}  render={H}x{W}")
    print(f"visible={tt_last.num_visible:,}  sorted_entries={tt_last.num_entries:,}\n")

    order = ["project", "tile_assign", "sort", "blend",
             "blend.prep", "blend.write_shm", "blend.mframe_rt",
             "blend.mframe_rt.device_kernel", "blend.read_shm",
             "blend.save_npy", "blend.daemon_rt", "blend.daemon_rt.device_kernel",
             "blend.load_npy", "total", "wall"]
    print("=== TT pipeline (median ms) ===")
    for k in order:
        if k in tt_med:
            indent = "  " if k.startswith("blend.") else ""
            print(f"{indent}{k:<32}{tt_med[k]:7.2f}")
    fps = 1000.0 / tt_med["total"] if tt_med.get("total") else 0
    print(f"\n~{fps:.1f} fps (pipeline total)")

    if args.cpu:
        cpu = Pipeline(get_backend("cpu"))
        try:
            cpu_med, cpu_last = run(cpu, gaussians, extr, intr, H, W, 0, max(2, args.measure // 4))
        finally:
            cpu.close()
        print("\n=== CPU pipeline (median ms) ===")
        for k in ["project", "tile_assign", "sort", "blend", "total"]:
            if k in cpu_med:
                print(f"{k:<32}{cpu_med[k]:7.2f}")
        if tt_last.image is not None and cpu_last.image is not None:
            print(f"\nPSNR TT vs CPU: {psnr(tt_last.image, cpu_last.image):.2f} dB")


if __name__ == "__main__":
    main()
