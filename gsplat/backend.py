"""Backend interface for the alpha-blend forward pass.

A backend is anything that can take camera-projected Gaussians + tile
assignments and produce a rendered image. Backends differ in *where* the
work happens (CPU PyTorch, Tenstorrent kernel, future CUDA, ...) but share
this interface so the viewer / pipeline can swap them by name.

The interface separates the four pipeline stages:

    project    — 3D world → 2D screen (covariance + projection + culling)
    tile_assign — find which 32×32 screen tiles each Gaussian touches
    sort       — order (gaussian, tile) pairs by (tile_id, depth)
    blend      — per-tile front-to-back alpha compositing → RGB image

Default implementations for `project`, `tile_assign`, `sort` use the CPU
PyTorch reference in `gsplat.rasterization`. A backend overrides only the
stages it actually accelerates (TT today only accelerates `blend`; a
hypothetical CUDA backend might also do `project` on the GPU).

All stage methods are wrapped in timing by `gsplat.pipeline.Pipeline`, so
backend implementers do not write timing code for the outer wall-clock
duration. They MAY return finer-grained sub-timings from `blend(...)`'s
second return value (an empty dict is fine if not measured).

Async-backend caveat: if your backend dispatches work asynchronously (e.g.
a CUDA stream), synchronize before returning from `blend(...)` so the
outer wall-clock timer captures actual completion, not just kernel-launch
time. The TT backend is naturally synchronous because the daemon's
EnqueueReadMeshBuffer blocks until the device finishes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch

from gsplat import rasterization


class Backend(ABC):
    """Abstract backend interface.

    Subclasses MUST implement `blend`. Any stage they don't override falls
    back to the CPU-PyTorch default in this base class.
    """

    # ------------------------------------------------------------------
    # Stages with default CPU implementations.
    # Override in a subclass only if the backend accelerates that stage.
    # ------------------------------------------------------------------

    def project(
        self,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        image_height: int,
        image_width: int,
        opacities: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """3D → 2D projection + frustum/opacity/radius culling.

        Returns: (means_2d, covs_2d, depths, radii, valid_mask).
        """
        return rasterization.project_gaussians(
            means, scales, rotations, extrinsics, intrinsics,
            image_height, image_width, opacities=opacities,
        )

    def tile_assign(
        self,
        means_2d: torch.Tensor,
        radii: torch.Tensor,
        image_height: int,
        image_width: int,
        tile_size: int = 32,
        covs_2d: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tile-overlap assignment.

        When covs_2d is given, uses the tighter per-axis AABB instead of the
        circular radius. Returns: (gaussian_ids, tile_ids, tiles_per_gaussian).
        """
        return rasterization.get_tile_assignments(
            means_2d, radii, image_height, image_width, tile_size=tile_size,
            covs_2d=covs_2d,
        )

    def sort(
        self,
        gaussian_ids: torch.Tensor,
        tile_ids: torch.Tensor,
        depths: torch.Tensor,
        tiles_x: int,
        tiles_y: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sort (gaussian, tile) pairs by (tile_id, depth) and bin into tiles.

        Returns: (sorted_gaussian_ids, tile_ranges).
        """
        return rasterization.sort_and_bin(
            gaussian_ids, tile_ids, depths, tiles_x, tiles_y,
        )

    # ------------------------------------------------------------------
    # The required stage. Every backend MUST implement this.
    # ------------------------------------------------------------------

    @abstractmethod
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
        """Per-tile front-to-back alpha compositing.

        Returns: (image, sub_timings_ms).
        - `image`: (H, W, 3) float32 numpy array in [0, 1].
        - `sub_timings_ms`: optional dict of internal stage names to
          milliseconds. Pipeline already times the OUTER wall-clock of
          this call; sub-timings are only needed if you want to break
          down where time went inside (e.g. {"prep": ..., "kernel": ...,
          "readback": ...}). An empty dict is fine.

        Async backends MUST synchronize before returning so the outer
        wall-clock timer reflects actual completion.
        """

    # ------------------------------------------------------------------
    # Lifecycle.
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release any resources held by the backend (e.g. subprocesses)."""
