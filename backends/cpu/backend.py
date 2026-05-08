"""CPU backend — pure PyTorch reference rasterizer.

All four pipeline stages fall back to the CPU defaults provided by
`gsplat.backend.Backend`, so this class is intentionally minimal: it
only needs to override `blend` (the abstract method) and route it to
`gsplat.rasterization.alpha_blend`.

Speed: ~1-2 s/frame at 256×256 on a 16-core x86 box. Used as the
correctness reference, not for interactive rendering.
"""
from __future__ import annotations

import numpy as np
import torch

from gsplat.backend import Backend
from gsplat import rasterization


class CpuBackend(Backend):
    def __init__(self, verbose: bool = False, tile_size: int = 32):
        self.verbose = verbose
        self.tile_size = tile_size

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
        # `alpha_blend` returns a torch.Tensor; convert to numpy for the
        # cross-backend contract. No internal sub-stages worth surfacing.
        image = rasterization.alpha_blend(
            means_2d, covs_2d, colors, opacities,
            sorted_gaussian_ids, tile_ranges,
            image_height, image_width, tile_size=self.tile_size,
        )
        return image.numpy(), {}
