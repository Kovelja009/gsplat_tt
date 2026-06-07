"""End-to-end: in-process ttnn alpha-blend op vs CPU reference (PSNR) + perf."""
import statistics
import time

import numpy as np
import pytest
import torch

from gsplat.rasterization import (
    project_gaussians, get_tile_assignments, sort_and_bin, alpha_blend,
)


def _have_tt():
    """True if a Tenstorrent device + ttnn are usable (skip cleanly otherwise)."""
    try:
        import ttnn
        d = ttnn.open_device(device_id=0)
        ttnn.close_device(d)
        return True
    except Exception:
        return False


HAVE_TT = _have_tt()
skip_no_tt = pytest.mark.skipif(not HAVE_TT, reason="no Tenstorrent device / ttnn available")


def _psnr(a, b):
    mse = np.mean((a - b) ** 2)
    if mse <= 0:
        return 100.0
    return -10.0 * np.log10(mse)


def _make_scene(H, W, N, seed, fx=40.0, spread=2.0, zmax=0.0):
    """Project/tile/sort a random scene; return blend() args + CPU reference."""
    torch.manual_seed(seed)
    means = torch.rand(N, 3) * torch.tensor([spread, spread, zmax]) + torch.tensor([-spread / 2, -spread / 2, 1.0])
    scales = torch.rand(N, 3) * 0.1 + 0.02
    q = torch.randn(N, 4); q = q / q.norm(dim=-1, keepdim=True)
    colors = torch.rand(N, 3)
    opacities = torch.rand(N) * 0.5 + 0.2
    extrinsics = torch.eye(4)
    intrinsics = torch.tensor([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], dtype=torch.float32)

    means_2d, covs_2d, depths, radii, valid = project_gaussians(
        means, scales, q, extrinsics, intrinsics, H, W)
    colors_v, opacities_v = colors[valid], opacities[valid]
    gids, tids, _ = get_tile_assignments(means_2d, radii, H, W, tile_size=32)
    tiles_x, tiles_y = (W + 31) // 32, (H + 31) // 32
    sorted_gids, tile_ranges = sort_and_bin(gids, tids, depths, tiles_x, tiles_y)
    args = (means_2d, covs_2d, colors_v, opacities_v, sorted_gids, tile_ranges, H, W)
    return args, int(valid.sum().item())


@skip_no_tt
def test_full_scene_psnr():
    """64x64 / 50-Gaussian render vs CPU reference (PSNR >= 35 dB)."""
    from backends.tt.backend import KernelBackend
    args, nvis = _make_scene(64, 64, 50, seed=42)
    if nvis == 0:
        pytest.skip("no visible Gaussians")
    cpu_img = alpha_blend(*args[:6], args[6], args[7], tile_size=32).numpy()

    backend = KernelBackend()
    try:
        tt_img, _ = backend.blend(*args)
    finally:
        backend.close()

    psnr = _psnr(cpu_img, tt_img)
    print(f"PSNR: {psnr:.2f} dB")
    assert tt_img.shape == (64, 64, 3)
    assert psnr >= 35.0, f"PSNR too low: {psnr:.2f} dB (want >= 35)"


@skip_no_tt
def test_sparse_scene_empty_tiles():
    """Sparse scene (many empty tiles) must render background as zero."""
    from backends.tt.backend import KernelBackend
    args, nvis = _make_scene(128, 128, 8, seed=3, fx=60.0, spread=0.3)
    if nvis == 0:
        pytest.skip("no visible Gaussians")
    cpu_img = alpha_blend(*args[:6], args[6], args[7], tile_size=32).numpy()

    backend = KernelBackend()
    try:
        tt_img, _ = backend.blend(*args)
    finally:
        backend.close()

    psnr = _psnr(cpu_img, tt_img)
    print(f"sparse PSNR: {psnr:.2f} dB")
    assert tt_img.max() < 2.0, f"empty tiles leaked garbage (max={tt_img.max():.3f})"
    assert psnr >= 35.0, f"PSNR too low: {psnr:.2f} dB"


@skip_no_tt
def test_640_perf():
    """640x640 / 10K-Gaussian in-process perf: warm a few frames, measure median.

    The device is opened once and held; the first frame pays program-cache /
    JIT cost, subsequent frames run warm. We report the warm median.
    """
    from backends.tt.backend import KernelBackend
    H, W = 640, 640
    args, nvis = _make_scene(H, W, 10_000, seed=42, fx=400.0, spread=2.0, zmax=2.0)
    print(f"\n640x640 perf: {nvis} visible Gaussians")

    backend = KernelBackend()
    try:
        for _ in range(3):              # warmup (incl. cold first frame)
            backend.blend(*args)
        times = []
        for _ in range(10):             # measured
            t = time.perf_counter()
            img, _sub = backend.blend(*args)
            times.append((time.perf_counter() - t) * 1000.0)
    finally:
        backend.close()

    med = statistics.median(times)
    print(f"warm frame median: {med:.2f} ms (min {min(times):.2f})")
    assert img.shape == (H, W, 3)
    assert med < 500.0, f"warm frame too slow: {med:.2f} ms"
