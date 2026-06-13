"""Tests for the CUDA backend that do not require a CUDA device.

Covers the registry guard (import safety on CPU-only boxes) and the
pure-torch SoA packing helpers. Kernel-level tests live in
test_cuda_kernel.py and are gated on torch.cuda.is_available().
"""
import importlib

import pytest
import torch


def test_registry_import_is_safe_without_cuda():
    """Importing `backends` must succeed regardless of host accelerators."""
    import backends  # must not raise
    # `cpu` is the only mandatory backend. `tt` (needs ttnn) and `cuda` (needs
    # a CUDA device) register lazily — each is present only if its deps import,
    # so the absence of either must not break `import backends`.
    assert "cpu" in backends.REGISTRY


def test_get_backend_cuda_errors_helpfully_when_unavailable():
    """`get_backend('cuda')` on a CPU-only box raises a clear KeyError."""
    import backends
    if "cuda" in backends.REGISTRY:
        pytest.skip("CUDA backend is registered on this host")
    with pytest.raises(KeyError, match="cuda"):
        backends.get_backend("cuda")


def test_cuda_backend_init_rejects_no_cuda():
    """CudaBackend() raises if no CUDA device is available."""
    if torch.cuda.is_available():
        pytest.skip("CUDA available; this check only fires on CPU-only hosts")
    from backends.cuda.backend import CudaBackend
    with pytest.raises(RuntimeError, match="CUDA"):
        CudaBackend()


# ---------------------------------------------------------------------------
# SoA packing helpers — pure host-side, CPU-testable.
# ---------------------------------------------------------------------------


def test_pack_conics_matches_reference():
    """_pack_conics produces (cov_inv_a, 2*cov_inv_b, cov_inv_c) per row."""
    from backends.cuda.backend import _pack_conics

    # Build a known-invertible covariance matrix.
    covs = torch.tensor([
        [[2.0, 0.5], [0.5, 3.0]],
        [[1.0, 0.0], [0.0, 1.0]],
    ], dtype=torch.float32)

    out = _pack_conics(covs)

    assert out.shape == (2, 3)
    assert out.dtype == torch.float32

    # Manual inversion for the first matrix:
    # det = 2*3 - 0.5*0.5 = 6 - 0.25 = 5.75
    # cov_inv = [[3/5.75, -0.5/5.75], [-0.5/5.75, 2/5.75]]
    # Pack stores [a_inv, 2*b_inv, c_inv]
    assert out[0, 0].item() == pytest.approx(3.0 / 5.75, rel=1e-5)
    assert out[0, 1].item() == pytest.approx(2.0 * (-0.5 / 5.75), rel=1e-5)
    assert out[0, 2].item() == pytest.approx(2.0 / 5.75, rel=1e-5)

    # Identity matrix: inverse is itself; b = 0 so 2*b = 0.
    assert out[1, 0].item() == pytest.approx(1.0, rel=1e-5)
    assert out[1, 1].item() == pytest.approx(0.0, abs=1e-6)
    assert out[1, 2].item() == pytest.approx(1.0, rel=1e-5)


def test_pack_conics_clamps_tiny_determinant():
    """Near-singular covariances do not blow up — det floored at 1e-6."""
    from backends.cuda.backend import _pack_conics

    # Very nearly singular: rows are almost linearly dependent.
    covs = torch.tensor([
        [[1.0, 1.0], [1.0, 1.0 + 1e-9]],
    ], dtype=torch.float32)
    out = _pack_conics(covs)
    # No NaN/Inf in the packed result.
    assert torch.isfinite(out).all()


def test_pack_rgba_layout():
    """_pack_rgba(colors, opacities) -> (N, 4) with opacity in last column."""
    from backends.cuda.backend import _pack_rgba

    colors = torch.tensor([
        [0.1, 0.2, 0.3],
        [0.5, 0.6, 0.7],
    ], dtype=torch.float32)
    opacities = torch.tensor([0.9, 0.4], dtype=torch.float32)

    out = _pack_rgba(colors, opacities)
    assert out.shape == (2, 4)
    assert out.dtype == torch.float32
    assert torch.allclose(out[:, :3], colors)
    assert torch.allclose(out[:, 3], opacities)
