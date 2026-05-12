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
