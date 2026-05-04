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


def dump_two_gaussian_blend(out_dir):
    """T0.5: red (front, alpha=0.5) + blue (back, alpha=0.5) at same pixel."""
    os.makedirs(out_dir, exist_ok=True)
    packs = np.array([
        [16.5, 16.5, 100.0, 0.0, 100.0, 1.0, 0.0, 0.0, 0.5],
        [16.5, 16.5, 100.0, 0.0, 100.0, 0.0, 0.0, 1.0, 0.5],
    ], dtype=np.float32)
    offsets = np.array([0, 2], dtype=np.float32)
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


def dump_saturation(out_dir):
    """T0.6: 50 opaque Gaussians -> sat_mask kicks in."""
    os.makedirs(out_dir, exist_ok=True)
    one = np.array([[16.5, 16.5, 100.0, 0.0, 100.0, 0.5, 0.5, 0.5, 0.99]], dtype=np.float32)
    packs = np.tile(one, (50, 1))
    offsets = np.array([0, 50], dtype=np.float32)
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


def dump_two_tile(out_dir):
    """Multi-tile reproducer: 32x64 image (1 row x 2 cols of tiles).
    Tile 0 has one red Gaussian centered at (15.5, 15.5).
    Tile 1 has one blue Gaussian centered at (47.5, 15.5) (i.e. local (15.5, 15.5)).
    Same simple cov format as T0.5 (cov_a=cov_c=100, two_cov_b=0), opacity=0.5.
    """
    os.makedirs(out_dir, exist_ok=True)
    # Means in GLOBAL screen coordinates (matching px/py convention below).
    packs = np.array([
        [15.5, 15.5, 100.0, 0.0, 100.0, 1.0, 0.0, 0.0, 0.5],   # tile 0: red
        [47.5, 15.5, 100.0, 0.0, 100.0, 0.0, 0.0, 1.0, 0.5],   # tile 1: blue
    ], dtype=np.float32)
    offsets = np.array([0, 1, 2], dtype=np.float32)
    px = np.empty((2, 32, 32), dtype=np.float32)
    py = np.empty((2, 32, 32), dtype=np.float32)
    for t in range(2):
        for i in range(32):
            for j in range(32):
                # Global screen coords: tile 0 spans cols 0..31, tile 1 spans cols 32..63.
                px[t, i, j] = t * 32 + j + 0.5
                py[t, i, j] = i + 0.5
    np.save(f"{out_dir}/packs.npy", packs)
    np.save(f"{out_dir}/offsets.npy", offsets)
    np.save(f"{out_dir}/px.npy", px)
    np.save(f"{out_dir}/py.npy", py)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fixture",
        choices=["single_gaussian", "two_gaussian_blend", "saturation", "two_tile"],
        required=True,
    )
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    if args.fixture == "single_gaussian":
        dump_single_gaussian(args.out_dir)
    elif args.fixture == "two_gaussian_blend":
        dump_two_gaussian_blend(args.out_dir)
    elif args.fixture == "saturation":
        dump_saturation(args.out_dir)
    elif args.fixture == "two_tile":
        dump_two_tile(args.out_dir)
    print(f"Dumped {args.fixture} to {args.out_dir}")
