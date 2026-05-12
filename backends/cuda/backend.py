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
    """Alpha-blend backend running on an NVIDIA GPU via a custom kernel.

    `dtype` selects the on-device storage precision for Gaussian
    attributes (means / conics / rgba):
      "fp32" — full fp32 everywhere (max precision; PSNR ~158 dB vs CPU).
      "bf16" — bfloat16 storage + fp32 in-register compute and accumulate.
               Halves shared-mem and global-mem footprint per Gaussian.
               Matches the TT kernel's mixed-precision design. PSNR drop
               is small (still well above the 35 dB gate).
    """

    def __init__(self, verbose: bool = False, dtype: str = "fp32"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CudaBackend requires a CUDA-capable device; "
                "torch.cuda.is_available() returned False"
            )
        if dtype not in ("fp32", "bf16"):
            raise ValueError(f"CudaBackend dtype must be 'fp32' or 'bf16'; got {dtype!r}")
        self.verbose = verbose
        self.dtype = dtype
        self._torch_dtype = torch.float32 if dtype == "fp32" else torch.bfloat16
        self._device = torch.device("cuda")
        self._ext = None

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
        if self._ext is None:
            from backends.cuda.kernels.build import _load_extension
            self._ext = _load_extension()

        sub: dict[str, float] = {}
        dt = self._torch_dtype

        # ---- Stage A: H2D upload + SoA repack ----
        t = time.perf_counter()
        d_means  = means_2d.to(self._device, dtype=dt, non_blocking=True)
        d_conics = _pack_conics(covs_2d).to(self._device, dtype=dt, non_blocking=True)
        d_rgba   = _pack_rgba(colors, opacities).to(self._device, dtype=dt, non_blocking=True)
        d_ids    = sorted_gaussian_ids.to(self._device, dtype=torch.int32, non_blocking=True)
        d_ranges = tile_ranges.to(self._device, dtype=torch.int32, non_blocking=True)
        sub["upload"] = (time.perf_counter() - t) * 1000.0

        # ---- Stage B: kernel launch + D2H readback ----
        kernel_fn = (self._ext.alpha_blend if self.dtype == "fp32"
                     else self._ext.alpha_blend_bf16)
        ev_s = torch.cuda.Event(enable_timing=True)
        ev_e = torch.cuda.Event(enable_timing=True)
        t = time.perf_counter()
        ev_s.record()
        d_out = kernel_fn(
            d_means, d_conics, d_rgba, d_ids, d_ranges,
            image_height, image_width,
        )
        ev_e.record()
        image = d_out.cpu().numpy()  # implicit sync via D2H memcpy
        sub["kernel"]        = (time.perf_counter() - t) * 1000.0
        sub["kernel.device"] = ev_s.elapsed_time(ev_e)

        return image, sub

    def close(self) -> None:
        # Nothing to release: the JIT-loaded extension lives in the
        # torch_extensions cache and tensors are freed by GC.
        pass
