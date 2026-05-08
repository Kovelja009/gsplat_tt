"""CPU (PyTorch reference) backend.

Pure-Python rasterizer; serves as the correctness baseline against which
all accelerator backends are checked. Implementation lives in
`gsplat.rasterization`; this module just exposes a thin Backend wrapper.
"""

from backends.cpu.backend import CpuBackend

__all__ = ["CpuBackend"]
