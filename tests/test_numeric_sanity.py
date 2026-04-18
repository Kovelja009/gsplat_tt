"""Unit tests for the NumPy reference alpha-blend."""
import numpy as np
import pytest
from scripts.numeric_sanity import alpha_blend_reference


def _make_px_py():
    px = np.arange(32)[None, None, :].repeat(32, axis=1).astype(np.float32) + 0.5
    py = np.arange(32)[None, :, None].repeat(32, axis=2).astype(np.float32) + 0.5
    return px, py


def test_single_gaussian_centered():
    """T0.4 equivalent: single Gaussian at pixel (16,16) with sharp covariance."""
    px, py = _make_px_py()
    packs = np.array([[16.5, 16.5, 100.0, 0.0, 100.0, 1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    offsets = np.array([0, 1], dtype=np.uint32)
    out = alpha_blend_reference(packs, offsets, px, py, 32, 32)
    assert abs(out[16, 16, 0] - 1.0) < 0.01
    assert out[16, 16, 1] < 0.01 and out[16, 16, 2] < 0.01
    # Neighbors should be ~0 due to exp(-50)
    assert out[20, 20, 0] < 1e-3


def test_two_gaussian_alpha_blend():
    """T0.5: two overlapping Gaussians at same pixel, red in front."""
    px, py = _make_px_py()
    packs = np.array([
        [16.5, 16.5, 100.0, 0.0, 100.0, 1.0, 0.0, 0.0, 0.5],  # red front
        [16.5, 16.5, 100.0, 0.0, 100.0, 0.0, 0.0, 1.0, 0.5],  # blue back
    ], dtype=np.float32)
    offsets = np.array([0, 2], dtype=np.uint32)
    out = alpha_blend_reference(packs, offsets, px, py, 32, 32)
    # Red at center: 0.5, Blue at center: 0.5 * 0.5 = 0.25
    assert abs(out[16, 16, 0] - 0.5) < 0.01
    assert abs(out[16, 16, 2] - 0.25) < 0.01


def test_saturation():
    """T0.6: 50 opaque Gaussians stacked at one pixel; sat_mask should kick in."""
    px, py = _make_px_py()
    packs = np.tile(
        np.array([[16.5, 16.5, 100.0, 0.0, 100.0, 0.5, 0.5, 0.5, 0.99]], dtype=np.float32),
        (50, 1),
    )
    offsets = np.array([0, 50], dtype=np.uint32)
    out = alpha_blend_reference(packs, offsets, px, py, 32, 32)
    # With opacity=0.99, T drops below 1e-4 after ~10 Gaussians; later ones contribute ~0
    # First Gaussian contributes 0.5 * 0.99 * 0.5 (color * alpha * T_prev=1)
    # Color converges to color*alpha / (1 - (1-alpha)) = (0.5*0.99)/0.99 = 0.5
    assert abs(out[16, 16, 0] - 0.5) < 0.05


def test_bf16_vs_fp32_identity_case():
    """bf16 simulation should match fp32 for simple cases."""
    px, py = _make_px_py()
    packs = np.array([[16.5, 16.5, 100.0, 0.0, 100.0, 1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    offsets = np.array([0, 1], dtype=np.uint32)
    out_fp32 = alpha_blend_reference(packs, offsets, px, py, 32, 32, simulate_bf16=False)
    out_bf16 = alpha_blend_reference(packs, offsets, px, py, 32, 32, simulate_bf16=True)
    # bf16 should be close enough
    assert np.max(np.abs(out_fp32 - out_bf16)) < 0.02


def test_empty_tile():
    """Tile with no Gaussians should return zeros."""
    px, py = _make_px_py()
    packs = np.zeros((0, 9), dtype=np.float32)
    offsets = np.array([0, 0], dtype=np.uint32)
    out = alpha_blend_reference(packs, offsets, px, py, 32, 32)
    assert np.allclose(out, 0.0)


import torch
from rasterization import alpha_blend, prepare_kernel_inputs


def test_prepare_kernel_inputs_matches_cpu_alpha_blend():
    """Pipeline equivalence: Python -> prepare_kernel_inputs -> reference ≈ alpha_blend()."""
    H, W = 64, 64  # 2x2 tiles
    torch.manual_seed(0)
    N = 10
    means_2d = torch.rand(N, 2) * torch.tensor([float(W), float(H)])
    # Diagonal covariance, well-conditioned
    covs_2d = torch.zeros(N, 2, 2)
    covs_2d[:, 0, 0] = 5.0
    covs_2d[:, 1, 1] = 5.0
    colors = torch.rand(N, 3)
    opacities = torch.rand(N) * 0.5 + 0.1
    depths = torch.arange(N, dtype=torch.float32)

    # Simple tile assignment: all Gaussians to all tiles
    from rasterization import get_tile_assignments, sort_and_bin
    radii = torch.full((N,), 10.0)
    gaussian_ids, tile_ids, _ = get_tile_assignments(means_2d, radii, H, W, tile_size=32)
    sorted_gaussian_ids, tile_ranges = sort_and_bin(
        gaussian_ids, tile_ids, depths, (W + 31) // 32, (H + 31) // 32
    )

    # CPU reference
    cpu_img = alpha_blend(
        means_2d, covs_2d, colors, opacities,
        sorted_gaussian_ids, tile_ranges, H, W, tile_size=32,
    ).numpy()

    # Prepare kernel inputs + run reference
    packs, offsets, px, py = prepare_kernel_inputs(
        means_2d, covs_2d, colors, opacities,
        sorted_gaussian_ids, tile_ranges, H, W,
    )
    from scripts.numeric_sanity import alpha_blend_reference
    ref_img = alpha_blend_reference(packs, offsets, px, py, H, W)

    # Should agree within fp32 precision
    assert np.allclose(cpu_img, ref_img, atol=1e-3), f"max diff: {np.abs(cpu_img - ref_img).max()}"
