"""Pipeline orchestrator with built-in per-stage timing.

The Pipeline calls each stage on the backend, wrapping every call in a
`time.perf_counter()` timer. Backend implementers do not write any
outer-timing code — adding a new backend automatically gets benchmarked
out of the box. Sub-timings (the breakdown inside a stage, e.g. blend's
prep/kernel/readback split) are reported by the backend via the second
return value of `blend(...)`.

Usage:

    pipeline = Pipeline(backend)
    result = pipeline.render(gaussians, extrinsics, intrinsics, H, W)
    result.image          # (H, W, 3) float32 numpy in [0, 1]
    result.timings        # {"project": 12.5, "tile_assign": 3.1,
                          #  "sort": 4.0, "blend": 80.2, "total": 99.8}
    result.sub_timings    # {"blend.prep": 3.5, "blend.kernel": 70.0, ...}
                          # (only populated if backend reports them)
    result.num_visible    # int — Gaussians that passed culling
    result.num_entries    # int — (gaussian, tile) pairs after sort
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch

from gsplat.backend import Backend
from gsplat.data_structures import Gaussians


TILE_SIZE = 32


@dataclass
class RenderResult:
    """Result of a single Pipeline.render() call.

    `image` is None when the frame had zero visible Gaussians (early-exit).
    Caller should treat that as a black/empty frame; callers that need an
    actual array can do `result.image if result.image is not None else
    np.zeros((H, W, 3), dtype=np.float32)`.
    """
    image: np.ndarray | None
    timings: dict[str, float] = field(default_factory=dict)
    sub_timings: dict[str, float] = field(default_factory=dict)
    num_visible: int = 0
    num_entries: int = 0
    height: int = 0
    width: int = 0


class Pipeline:
    """Glue between a chosen backend and the per-frame stage sequence."""

    def __init__(self, backend: Backend, *, tile_size: int = TILE_SIZE):
        self.backend = backend
        self.tile_size = tile_size

    # ------------------------------------------------------------------
    # Per-stage timer
    # ------------------------------------------------------------------

    @staticmethod
    @contextmanager
    def _timer(timings: dict[str, float], name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            timings[name] = (time.perf_counter() - t0) * 1000.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        gaussians: Gaussians,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> RenderResult:
        """Run the full forward pass: project → tile_assign → sort → blend.

        Each stage is timed separately; their wall-clock durations land in
        `result.timings`. The backend's `blend(...)` may also return a
        dict of internal sub-timings, prefixed and copied into
        `result.sub_timings` (e.g. `{"blend.kernel": 70.0}`).
        """
        timings: dict[str, float] = {}
        sub_timings: dict[str, float] = {}
        t_total = time.perf_counter()

        # Stage 1: project
        with self._timer(timings, "project"):
            means_2d, covs_2d, depths, radii, valid_mask = self.backend.project(
                gaussians.means, gaussians.scales, gaussians.rotations,
                extrinsics, intrinsics, image_height, image_width,
                opacities=gaussians.opacities,
            )
        num_visible = int(valid_mask.sum().item())

        # Empty frame: no visible Gaussians → skip downstream stages.
        if num_visible == 0:
            timings["tile_assign"] = 0.0
            timings["sort"] = 0.0
            timings["blend"] = 0.0
            timings["total"] = (time.perf_counter() - t_total) * 1000.0
            return RenderResult(
                image=None,
                timings=timings,
                sub_timings=sub_timings,
                num_visible=0,
                num_entries=0,
                height=image_height,
                width=image_width,
            )

        colors = gaussians.colors[valid_mask]
        opacities = gaussians.opacities[valid_mask]

        # Stage 2: tile assignment
        with self._timer(timings, "tile_assign"):
            gaussian_ids, tile_ids, _ = self.backend.tile_assign(
                means_2d, radii, image_height, image_width,
                tile_size=self.tile_size, covs_2d=covs_2d,
            )
        tiles_x = (image_width + self.tile_size - 1) // self.tile_size
        tiles_y = (image_height + self.tile_size - 1) // self.tile_size

        # Stage 3: sort + bin
        with self._timer(timings, "sort"):
            sorted_gaussian_ids, tile_ranges = self.backend.sort(
                gaussian_ids, tile_ids, depths, tiles_x, tiles_y,
            )
        num_entries = int(sorted_gaussian_ids.numel())

        # Stage 4: blend (with optional sub-timings from backend)
        with self._timer(timings, "blend"):
            image, blend_sub = self.backend.blend(
                means_2d, covs_2d, colors, opacities,
                sorted_gaussian_ids, tile_ranges,
                image_height, image_width,
            )
        for k, v in (blend_sub or {}).items():
            sub_timings[f"blend.{k}"] = v

        timings["total"] = (time.perf_counter() - t_total) * 1000.0

        return RenderResult(
            image=image,
            timings=timings,
            sub_timings=sub_timings,
            num_visible=num_visible,
            num_entries=num_entries,
            height=image_height,
            width=image_width,
        )

    def close(self) -> None:
        """Release the backend's resources."""
        self.backend.close()


def format_timings(result: RenderResult) -> str:
    """Pretty-print the per-stage breakdown for verbose / debug output."""
    lines = [f"[stage] {name:<12} {result.timings.get(name, 0.0):6.1f} ms"
             for name in ("project", "tile_assign", "sort", "blend")]
    for k, v in result.sub_timings.items():
        # Indent sub-timings under their parent stage.
        lines.append(f"   └─ {k:<14} {v:6.1f} ms")
    total = result.timings.get("total", 0.0)
    fps = 1000.0 / total if total > 0 else 0.0
    lines.append(f"[total] {total:6.1f} ms  ({fps:.1f} fps)")
    return "\n".join(lines)
