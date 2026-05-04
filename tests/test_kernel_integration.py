"""End-to-end: full-scene kernel vs CPU reference, PSNR/SSIM."""
import os
import subprocess
import tempfile
import time

import numpy as np
import pytest
import torch

from rasterization import (
    project_gaussians, get_tile_assignments, sort_and_bin,
    alpha_blend, prepare_kernel_inputs,
)


KERNEL_BINARY = os.environ.get(
    "GSPLAT_KERNEL",
    "tt-metal/build/programming_examples/metal_example_gaussian_splatting",
)


def _psnr(a, b):
    mse = np.mean((a - b) ** 2)
    if mse <= 0:
        return 100.0
    return -10.0 * np.log10(mse)


@pytest.mark.skipif(
    not os.path.exists(KERNEL_BINARY),
    reason="kernel binary not built; run sudo ./build_metal.sh",
)
def test_full_scene_psnr():
    torch.manual_seed(42)
    H, W = 64, 64  # 2x2 tiles
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
        means, scales, q, extrinsics, intrinsics, H, W,
    )
    if valid.sum().item() == 0:
        pytest.skip("no visible Gaussians — random seed; reroll")

    colors_v = colors[valid]
    opacities_v = opacities[valid]

    gids, tids, _ = get_tile_assignments(means_2d, radii, H, W, tile_size=32)
    tiles_x = (W + 31) // 32
    tiles_y = (H + 31) // 32
    sorted_gids, tile_ranges = sort_and_bin(gids, tids, depths, tiles_x, tiles_y)

    cpu_img = alpha_blend(
        means_2d, covs_2d, colors_v, opacities_v,
        sorted_gids, tile_ranges, H, W, tile_size=32,
    ).numpy()

    packs, offsets, px, py = prepare_kernel_inputs(
        means_2d, covs_2d, colors_v, opacities_v,
        sorted_gids, tile_ranges, H, W,
    )

    with tempfile.TemporaryDirectory() as td:
        np.save(f"{td}/packs.npy", packs)
        np.save(f"{td}/offsets.npy", offsets.astype(np.float32))
        np.save(f"{td}/px.npy", px)
        np.save(f"{td}/py.npy", py)
        env = os.environ.copy()
        env.setdefault("TT_METAL_HOME", os.path.abspath("tt-metal"))
        env.setdefault("TT_METAL_RUNTIME_ROOT", os.path.abspath("tt-metal"))
        subprocess.run([
            KERNEL_BINARY,
            f"{td}/packs.npy", f"{td}/offsets.npy",
            f"{td}/px.npy", f"{td}/py.npy",
            f"{td}/out.npy", str(H), str(W),
        ], check=True, capture_output=True, env=env)
        kernel_img = np.load(f"{td}/out.npy")

    psnr = _psnr(cpu_img, kernel_img)
    print(f"PSNR: {psnr:.2f} dB")
    assert psnr >= 35.0, f"PSNR too low: {psnr:.2f} dB (want >= 35)"

    try:
        from skimage.metrics import structural_similarity as ssim
        ssim_val = ssim(cpu_img, kernel_img, channel_axis=2, data_range=1.0)
        print(f"SSIM: {ssim_val:.4f}")
        assert ssim_val >= 0.98, f"SSIM too low: {ssim_val:.4f}"
    except ImportError:
        print("scikit-image not installed; skipping SSIM")


@pytest.mark.skipif(
    not os.path.exists(KERNEL_BINARY),
    reason="kernel binary not built; run sudo ./build_metal.sh",
)
def test_640_perf_baseline():
    """v1b 640x640 / 10K-Gaussian single-core wall-clock baseline.

    Reports CPU reference, kernel input prep, and kernel binary timings
    separately. The kernel binary timing includes one-shot device init +
    JIT compile (~6s) — that is part of the per-frame cost in the current
    one-shot launch model. Daemon-mode binary is a separate task.
    """
    torch.manual_seed(42)
    H, W = 640, 640  # 20x20 = 400 tiles
    N = 10_000

    # Spread means across visible frustum: x,y in [-1, 1], z in [1, 3]
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
    V = int(valid.sum().item())
    if V < 100:
        print(f"WARNING: only {V} visible Gaussians (scene very sparse), proceeding anyway")

    colors_v = colors[valid]
    opacities_v = opacities[valid]

    gids, tids, _ = get_tile_assignments(means_2d, radii, H, W, tile_size=32)
    tiles_x = (W + 31) // 32
    tiles_y = (H + 31) // 32
    sorted_gids, tile_ranges = sort_and_bin(gids, tids, depths, tiles_x, tiles_y)
    sorted_entries = int(sorted_gids.numel())

    # --- Time CPU reference ---
    t0 = time.perf_counter()
    cpu_img = alpha_blend(
        means_2d, covs_2d, colors_v, opacities_v,
        sorted_gids, tile_ranges, H, W, tile_size=32,
    ).numpy()
    cpu_elapsed = time.perf_counter() - t0

    # --- Time kernel input prep ---
    t0 = time.perf_counter()
    packs, offsets, px, py = prepare_kernel_inputs(
        means_2d, covs_2d, colors_v, opacities_v,
        sorted_gids, tile_ranges, H, W,
    )
    prep_elapsed = time.perf_counter() - t0

    # --- Time kernel binary subprocess (incl. device init + JIT) ---
    with tempfile.TemporaryDirectory() as td:
        np.save(f"{td}/packs.npy", packs)
        np.save(f"{td}/offsets.npy", offsets.astype(np.float32))
        np.save(f"{td}/px.npy", px)
        np.save(f"{td}/py.npy", py)
        env = os.environ.copy()
        env.setdefault("TT_METAL_HOME", os.path.abspath("tt-metal"))
        env.setdefault("TT_METAL_RUNTIME_ROOT", os.path.abspath("tt-metal"))
        t0 = time.perf_counter()
        subprocess.run([
            KERNEL_BINARY,
            f"{td}/packs.npy", f"{td}/offsets.npy",
            f"{td}/px.npy", f"{td}/py.npy",
            f"{td}/out.npy", str(H), str(W),
        ], check=True, capture_output=True, env=env, timeout=120)
        kernel_elapsed = time.perf_counter() - t0
        kernel_img = np.load(f"{td}/out.npy")

    speedup = cpu_elapsed / kernel_elapsed if kernel_elapsed > 0 else float("inf")

    print()
    print("===== v1b 640x640 perf baseline =====")
    print(f"Scene: H={H} W={W}, N={N} input Gaussians, {V} visible")
    print(f"Sorted entries: {sorted_entries}  Total tiles: {tiles_x * tiles_y}")
    print(f"CPU reference (PyTorch):           {cpu_elapsed:>6.2f} s")
    print(f"Kernel input prep:                 {prep_elapsed:>6.2f} s")
    print(f"Kernel binary (incl. device init): {kernel_elapsed:>6.2f} s")
    print("---")
    print(f"Speedup vs CPU (with init):        {speedup:.2f}x")

    # Diagnostic-only PSNR/SSIM
    psnr = _psnr(cpu_img, kernel_img)
    try:
        from skimage.metrics import structural_similarity as ssim
        ssim_val = ssim(cpu_img, kernel_img, channel_axis=2, data_range=1.0)
        print(f"PSNR: {psnr:.2f} dB  SSIM: {ssim_val:.4f}")
    except ImportError:
        print(f"PSNR: {psnr:.2f} dB  (scikit-image not installed; SSIM skipped)")

    # Loose sanity bounds
    assert kernel_elapsed < 60.0, f"kernel took too long: {kernel_elapsed:.2f}s"
    assert kernel_img.shape == (H, W, 3), f"unexpected output shape: {kernel_img.shape}"
