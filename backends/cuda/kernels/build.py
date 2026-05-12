"""JIT loader for the CUDA alpha-blend extension.

Wraps `torch.utils.cpp_extension.load` and memoises the resulting module
so a viewer session with many `blend(...)` calls pays the compile cost
only on the first invocation.
"""
from __future__ import annotations

import os
import shutil
from typing import Any

_CACHED_EXT: Any = None


def _detect_arch_flags() -> list[str]:
    """Pick nvcc -arch flag from the visible GPU.

    Defaults to sm_89 (Ada Lovelace, RTX 4060) because that's the dev box,
    but probes torch for the actual compute capability when CUDA is up.
    """
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return [f"-arch=sm_{major}{minor}"]
    except Exception:
        pass
    return ["-arch=sm_89"]


def _detect_host_compiler_flag() -> list[str]:
    """Return `-ccbin <gcc>` only when a known-compatible g++ is on PATH.

    nvcc 13.x on Arch ships a default toolchain that may not match the
    system g++. Prefer g++-15 if present (matches the dev box note), else
    let nvcc pick.
    """
    for candidate in ("/usr/bin/g++-15", "/usr/bin/g++-14", "/usr/bin/g++-13"):
        if os.path.exists(candidate):
            return [f"-ccbin={candidate}"]
    gxx = shutil.which("g++")
    if gxx:
        return [f"-ccbin={gxx}"]
    return []


def _load_extension() -> Any:
    """Compile (if needed) and return the alpha_blend torch extension.

    Returns the loaded module exposing `alpha_blend(...)`. Cached across
    calls so re-import / repeated `CudaBackend()` doesn't re-pay the
    ~30-60s JIT compile.
    """
    global _CACHED_EXT
    if _CACHED_EXT is not None:
        return _CACHED_EXT

    # Local import keeps `import backends.cuda.kernels.build` cheap on
    # CPU-only boxes — the torch.utils.cpp_extension subpackage probes
    # the compiler on import in some torch versions.
    from torch.utils.cpp_extension import load

    here = os.path.dirname(os.path.abspath(__file__))
    sources = [
        os.path.join(here, "alpha_blend.cu"),
        os.path.join(here, "pybind.cpp"),
    ]
    cuda_flags = ["-O3", "--use_fast_math"]
    cuda_flags += _detect_host_compiler_flag()
    cuda_flags += _detect_arch_flags()
    _CACHED_EXT = load(
        name="gsplat_cuda_alpha_blend",
        sources=sources,
        extra_cuda_cflags=cuda_flags,
        extra_cflags=["-O3"],
        verbose=False,
    )
    return _CACHED_EXT
