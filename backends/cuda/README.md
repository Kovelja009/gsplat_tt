# CUDA backend (planned)

Placeholder — implementation pending. See `../README.md` for the wrapper
interface this backend needs to expose.

## Timing

The Pipeline times each stage from the host side with `time.perf_counter()`,
which only reflects real work if the call is synchronous from Python's
perspective. CUDA launches are async, so two things matter when filling in
the second return value of `blend(...)`:

1. **Synchronize before returning.** Either issue the final
   `result.cpu().numpy()` (the implicit memcpy syncs) or call
   `torch.cuda.synchronize()` explicitly. Otherwise the outer Pipeline
   timer under-counts blend (it ends before the kernel finishes).

2. **Device-only time needs CUDA events**, not wall-clock. A
   `time.perf_counter()` bracket around a kernel launch measures the
   launch overhead, not the kernel runtime. Use `torch.cuda.Event`s with
   `enable_timing=True` and report their elapsed time as a sub-key.

Pattern:

```python
def blend(self, ...) -> tuple[np.ndarray, dict[str, float]]:
    sub: dict[str, float] = {}

    t = time.perf_counter()
    # H2D copies (cudaMemcpyAsync)
    sub["upload"] = (time.perf_counter() - t) * 1000.0

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end   = torch.cuda.Event(enable_timing=True)
    t = time.perf_counter()
    ev_start.record()
    my_cuda_kernel(...)
    ev_end.record()
    image = result.cpu().numpy()         # forces a sync via the memcpy
    sub["kernel"]        = (time.perf_counter() - t) * 1000.0   # incl. dispatch + exec + D2H
    sub["kernel.device"] = ev_start.elapsed_time(ev_end)        # pure on-device time

    return image, sub
```

`kernel.device` uses the dotted-key convention from `../README.md` so the
benchmark markdown nests it visually under `kernel` — same as
`daemon_rt.device_kernel` does for the TT backend.
