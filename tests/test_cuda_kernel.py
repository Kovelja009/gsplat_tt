"""Integration tests for the CUDA alpha-blend backend.

All tests are gated on `torch.cuda.is_available()` — they are skipped on
CPU-only hosts. The CUDA kernel JIT-compiles on the first instantiation
of CudaBackend (~30-60s on a clean cache); a session-scoped fixture
amortises that cost across the whole test module.
"""
import time

import numpy as np
import pytest
import torch

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires a CUDA-capable device",
)


@pytest.fixture(scope="module")
def cuda_backend():
    """Construct a CudaBackend once per test module (amortises JIT compile)."""
    from backends.cuda.backend import CudaBackend
    return CudaBackend()


@cuda_only
def test_cuda_backend_smoke_shape(cuda_backend):
    """blend(...) returns an (H, W, 3) float32 numpy array and the right keys."""
    H, W = 64, 64
    M = 4
    means_2d = torch.zeros((M, 2), dtype=torch.float32)
    covs_2d = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]] * M, dtype=torch.float32
    )
    colors = torch.zeros((M, 3), dtype=torch.float32)
    opacities = torch.zeros((M,), dtype=torch.float32)
    sorted_ids = torch.zeros((0,), dtype=torch.int64)         # no entries
    tile_ranges = torch.zeros(((H // 32) * (W // 32), 2), dtype=torch.int64)

    image, sub = cuda_backend.blend(
        means_2d, covs_2d, colors, opacities,
        sorted_ids, tile_ranges, H, W,
    )

    assert image.shape == (H, W, 3)
    assert image.dtype == np.float32
    # Empty scene → output is all zeros.
    assert np.all(image == 0.0)

    # Sub-timing keys conform to the dotted-key convention.
    assert "upload" in sub
    assert "kernel" in sub
    assert "kernel.device" in sub
    # All sub-timings are non-negative.
    for k, v in sub.items():
        assert v >= 0.0, f"sub timing {k} negative: {v}"


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0:
        return 100.0
    return -10.0 * np.log10(mse)


@cuda_only
def test_cuda_psnr_64(cuda_backend):
    """PSNR vs CPU reference >= 35 dB on a 64x64 / 50-Gaussian scene.

    Uses the same seed/scene shape as test_full_scene_psnr in
    test_kernel_integration.py so failures can be compared 1:1 with the
    TT backend's correctness numbers.
    """
    from gsplat.rasterization import (
        project_gaussians, get_tile_assignments, sort_and_bin, alpha_blend,
    )

    torch.manual_seed(42)
    H, W = 64, 64
    N = 50
    means = torch.rand(N, 3) * torch.tensor([2.0, 2.0, 0.0]) + torch.tensor([-1.0, -1.0, 1.0])
    scales = torch.rand(N, 3) * 0.1 + 0.02
    q = torch.randn(N, 4); q = q / q.norm(dim=-1, keepdim=True)
    colors = torch.rand(N, 3)
    opacities = torch.rand(N) * 0.5 + 0.2

    extrinsics = torch.eye(4)
    fx = fy = 40.0
    intrinsics = torch.tensor(
        [[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=torch.float32
    )

    means_2d, covs_2d, depths, radii, valid = project_gaussians(
        means, scales, q, extrinsics, intrinsics, H, W,
    )
    if valid.sum().item() == 0:
        pytest.skip("no visible Gaussians — reroll seed")

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

    cuda_img, sub = cuda_backend.blend(
        means_2d, covs_2d, colors_v, opacities_v,
        sorted_gids, tile_ranges, H, W,
    )

    psnr = _psnr(cpu_img, cuda_img)
    print(f"\nPSNR (CUDA vs CPU): {psnr:.2f} dB")
    print(f"sub-timings: {sub}")
    assert psnr >= 35.0, f"PSNR too low: {psnr:.2f} dB (want >= 35)"

    try:
        from skimage.metrics import structural_similarity as ssim
        ssim_val = ssim(cpu_img, cuda_img, channel_axis=2, data_range=1.0)
        print(f"SSIM: {ssim_val:.4f}")
        assert ssim_val >= 0.98, f"SSIM too low: {ssim_val:.4f}"
    except ImportError:
        print("scikit-image not installed; skipping SSIM check")
