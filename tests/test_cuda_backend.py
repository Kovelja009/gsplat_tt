"""Tests for the CUDA backend that do not require a CUDA device.

Covers the registry guard (import safety on CPU-only boxes) and the
pure-torch SoA packing helpers. Kernel-level tests live in
test_cuda_kernel.py and are gated on torch.cuda.is_available().
"""
import importlib

import pytest
import torch


def test_registry_import_is_safe_without_cuda():
    """Importing `backends` must succeed on CPU-only boxes."""
    import backends  # must not raise
    assert "cpu" in backends.REGISTRY
    assert "tt" in backends.REGISTRY
    # "cuda" may or may not be present depending on host; either is fine
    # so long as the import did not blow up.


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
