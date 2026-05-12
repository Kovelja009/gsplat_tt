"""CUDA backend — JIT-compiled alpha-blend kernel.

The custom CUDA kernel is compiled on first `CudaBackend()` instantiation
via torch.utils.cpp_extension.load; importing this module on a CPU-only
box is safe (the kernel is only loaded when the backend is constructed,
not at import time).
"""
from __future__ import annotations

import time

import numpy as np
import torch

from gsplat.backend import Backend


def _pack_conics(covs_2d: torch.Tensor) -> torch.Tensor:
    """Invert each 2x2 covariance and pack as (cov_inv_a, 2*cov_inv_b, cov_inv_c).

    The off-diagonal is pre-multiplied by 2 so the kernel can compute the
    Mahalanobis quadratic form as `a*dx^2 + (2b)*dx*dy + c*dy^2` with a
    single mul per term (matches the TT kernel's CB_SCALARS layout).
    """
    a = covs_2d[:, 0, 0]
    b = covs_2d[:, 0, 1]
    c = covs_2d[:, 1, 1]
    det = torch.clamp(a * c - b * b, min=1e-6)
    out = torch.empty((covs_2d.shape[0], 3), dtype=torch.float32, device=covs_2d.device)
    out[:, 0] = c / det
    out[:, 1] = 2.0 * (-b / det)
    out[:, 2] = a / det
    return out


def _pack_rgba(colors: torch.Tensor, opacities: torch.Tensor) -> torch.Tensor:
    """Concatenate per-Gaussian colors and opacity into an (N, 4) float32 tensor."""
    out = torch.empty((colors.shape[0], 4), dtype=torch.float32, device=colors.device)
    out[:, :3] = colors
    out[:, 3] = opacities
    return out


class CudaBackend(Backend):
    """Alpha-blend backend running on an NVIDIA GPU via a custom kernel."""

    def __init__(self, verbose: bool = False):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CudaBackend requires a CUDA-capable device; "
                "torch.cuda.is_available() returned False"
            )
        self.verbose = verbose
        self._device = torch.device("cuda")
        self._ext = None  # populated in Task 4

    def blend(
        self,
        means_2d: torch.Tensor,
        covs_2d: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        sorted_gaussian_ids: torch.Tensor,
        tile_ranges: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> tuple[np.ndarray, dict[str, float]]:
        raise NotImplementedError("blend implemented in Task 4")
