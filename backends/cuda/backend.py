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
