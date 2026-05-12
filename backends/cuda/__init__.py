"""CUDA backend — alpha-blend on NVIDIA GPUs.

Implements `gsplat.backend.Backend.blend(...)` via a custom CUDA kernel
that is JIT-compiled on first `CudaBackend()` construction. The kernel
source lives in `kernels/`; the build wrapper in `kernels/build.py`
memoises the loaded extension across `blend(...)` calls.

This package intentionally does NOT eagerly import `backend` — that's
done lazily by `backends.__init__` inside a try/except, so importing
`backends` on a CPU-only box does not require torch.cuda.
"""
