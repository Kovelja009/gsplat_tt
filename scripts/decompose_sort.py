"""Decompose where sort_and_bin's time goes on the big scene, and compare sort
strategies (float argsort, uint64 stable, uint32 stable, lexsort)."""
import os
import statistics
import sys
import time

import numpy as np
import torch

from gsplat.loading_gaussians import load_ply
from gsplat.rasterization import project_gaussians, get_tile_assignments

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_scene import make_camera  # noqa: E402


def med(fn, n=7):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); r = fn(); ts.append((time.perf_counter() - t) * 1000.0)
    return statistics.median(ts), r


def main():
    H = W = 640
    g = load_ply("scenes/point_cloud.ply")
    extr, intr = make_camera(g.means, H, W)
    m2d, c2d, depths, radii, valid = project_gaussians(
        g.means, g.scales, g.rotations, extr, intr, H, W, opacities=g.opacities)
    gids_t, tids_t, _ = get_tile_assignments(m2d, radii, H, W, tile_size=32)
    P = gids_t.shape[0]
    print(f"entries P={P:,}\n")

    # one-time conversions
    gids = gids_t.numpy()
    tids = tids_t.numpy().astype(np.int64)
    dnp = depths.numpy()

    def step(name, fn):
        ms, r = med(fn)
        print(f"{name:<34}{ms:8.2f} ms")
        return r

    dpe = step("gather depths[gids]", lambda: dnp[gids])
    dpe = dnp[gids]
    step("  .astype(f32).view(u32)", lambda: np.ascontiguousarray(dpe, np.float32).view(np.uint32))
    db = np.ascontiguousarray(dpe, np.float32).view(np.uint32)
    step("build u64 key", lambda: (tids.astype(np.uint64) << np.uint64(32)) | db.astype(np.uint64))
    key64 = (tids.astype(np.uint64) << np.uint64(32)) | db.astype(np.uint64)

    print()
    step("argsort u64 stable (radix?)", lambda: np.argsort(key64, kind="stable"))
    step("argsort u64 quicksort", lambda: np.argsort(key64, kind="quicksort"))
    order = np.argsort(key64, kind="stable")
    step("gather gids[order]", lambda: gids[order])
    step("bincount+cumsum", lambda: np.cumsum(np.bincount(tids, minlength=400)))

    # uint32 packed key: tile_id (14b) << 18 | depth quantized to 18b
    print()
    dmin, dmax = float(dnp.min()), float(dnp.max())
    q = ((dpe - dmin) / (dmax - dmin) * ((1 << 18) - 1)).astype(np.uint32)
    step("build u32 key", lambda: (tids.astype(np.uint32) << np.uint32(18)) | q)
    key32 = (tids.astype(np.uint32) << np.uint32(18)) | q
    step("argsort u32 stable", lambda: np.argsort(key32, kind="stable"))
    step("argsort u32 quicksort", lambda: np.argsort(key32, kind="quicksort"))

    print()
    step("np.lexsort((depth, tile))", lambda: np.lexsort((dpe, tids)))

    print()
    # values-only sort (no permutation array) is cheaper than argsort
    step("np.sort u64 (values)", lambda: np.sort(key64))
    step("np.sort u32 (values)", lambda: np.sort(key32))

    # pack gid INTO the key: tile(14b)<<38 | depth(18b)<<20 | gid(20b). One
    # values-sort, then extract gid from the low bits — no argsort, no gather.
    print()
    M = int(valid.sum().item())
    print(f"  num_visible M={M:,} (gid needs {M.bit_length()} bits)")
    packed = ((tids.astype(np.uint64) << np.uint64(38))
              | (q.astype(np.uint64) << np.uint64(20))
              | gids.astype(np.uint64))
    step("build packed key+gid u64", lambda: ((tids.astype(np.uint64) << np.uint64(38))
              | (q.astype(np.uint64) << np.uint64(20)) | gids.astype(np.uint64)))
    step("np.sort packed (values)", lambda: np.sort(packed))
    step("  + extract gid (& mask)", lambda: (np.sort(packed) & np.uint64((1 << 20) - 1)).astype(np.int64))

    print()
    step("torch.sort int64 key", lambda: torch.sort(torch.from_numpy(key64.astype(np.int64))).values)

    print()
    step("OLD: torch float argsort", lambda: torch.argsort(tids_t.float() * (dnp.max() + 1.0) + depths[gids_t]))


if __name__ == "__main__":
    main()
