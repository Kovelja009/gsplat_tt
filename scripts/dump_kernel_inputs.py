"""Dump kernel input fixtures to .npy files for the standalone C++ harness."""
import argparse
import numpy as np
import os


def dump_single_gaussian(out_dir):
    """T0.4: one Gaussian at (16.5, 16.5), sharp cov, red, opaque."""
    os.makedirs(out_dir, exist_ok=True)
    packs = np.array([[16.5, 16.5, 100.0, 0.0, 100.0, 1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    offsets = np.array([0, 1], dtype=np.float32)  # fp32 for the npy reader; cast in C++
    px = np.empty((1, 32, 32), dtype=np.float32)
    py = np.empty((1, 32, 32), dtype=np.float32)
    for i in range(32):
        for j in range(32):
            px[0, i, j] = j + 0.5
            py[0, i, j] = i + 0.5
    np.save(f"{out_dir}/packs.npy", packs)
    np.save(f"{out_dir}/offsets.npy", offsets)
    np.save(f"{out_dir}/px.npy", px)
    np.save(f"{out_dir}/py.npy", py)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", choices=["single_gaussian"], required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    if args.fixture == "single_gaussian":
        dump_single_gaussian(args.out_dir)
    print(f"Dumped {args.fixture} to {args.out_dir}")
