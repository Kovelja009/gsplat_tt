# Backends

One subpackage per hardware target. Each backend implements the
`gsplat.backend.Backend` ABC and registers itself in `backends/REGISTRY`,
so the viewer / CLI can swap them by name (`--backend cpu|tt|cuda|...`).
The backend name on the CLI matches the dictionary key in `REGISTRY`.

## Layout per backend

```
backends/<arch>/
├── __init__.py
├── backend.py                  # the Backend subclass
├── kernels/                    # native source (.cpp / .cu / ...) — optional
└── <vendored-sdk>/             # e.g. tt-metal/, cuda-toolkit/ — optional
```

## Adding a new backend

```python
# backends/cuda/backend.py
import numpy as np
import torch
from gsplat.backend import Backend

class CudaBackend(Backend):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        # ... CUDA context init, allocate persistent buffers, etc.

    def blend(
        self,
        means_2d, covs_2d, colors, opacities,
        sorted_gaussian_ids, tile_ranges,
        image_height, image_width,
    ) -> tuple[np.ndarray, dict[str, float]]:
        # ... your CUDA blend implementation ...
        # (synchronize the stream before returning so wall-clock timing
        # captures actual completion, not just kernel launch!)
        torch.cuda.synchronize()
        return image, {"upload": ..., "kernel": ..., "download": ...}

    def close(self):
        ...
```

Then add one line in `backends/__init__.py`:

```python
from backends.cuda.backend import CudaBackend
REGISTRY = {"cpu": CpuBackend, "tt": KernelBackend, "cuda": CudaBackend}
```

That's it. The viewer's `--backend` argument auto-discovers the new
choice; `gsplat.pipeline.Pipeline` times the new backend's stages
without the implementer writing any timing code.

## Stage ownership

The Backend ABC defines four stages:

| Stage | Default | When to override |
|---|---|---|
| `project`     | CPU PyTorch | If your hardware can do EWA splatting + culling faster |
| `tile_assign` | CPU         | If you have a parallel tile-overlap routine |
| `sort`        | CPU         | If you have GPU/device sort (often not worth it) |
| `blend`       | **(abstract)** | Always — this is where the rasterization happens |

Only `blend` is required. Override the others only if your backend
genuinely accelerates them; otherwise the CPU defaults run on the host
in lock-step.

## Per-stage benchmarking

`gsplat.pipeline.Pipeline` wraps each stage call in
`time.perf_counter()`. Backend implementers do not write outer-timing
code — every backend gets per-stage `project / tile_assign / sort /
blend` measurements automatically.

If you want a finer breakdown of what happens *inside* `blend` (e.g.
prep / kernel / readback split), return a dict of sub-stage names →
milliseconds as the second element of `blend(...)`'s return tuple. An
empty dict is fine for backends that don't measure internally.

```python
return image, {"prep": 3.5, "kernel": 70.0, "readback": 0.3}
```

These show up in `RenderResult.sub_timings` prefixed with `blend.`
(e.g. `blend.kernel`).

**Async-backend caveat:** if your backend dispatches asynchronously
(CUDA streams, etc.), synchronize before returning from `blend(...)` so
the outer wall-clock timer captures actual completion, not just
kernel-launch time. The TT backend is naturally synchronous because
`EnqueueReadMeshBuffer` blocks until the device finishes.

## Existing backends

- **`cpu/`** — pure-PyTorch reference (`gsplat.rasterization`). Slow but
  correct; used as the golden reference for kernel-correctness tests.
- **`tt/`** — Tenstorrent Wormhole / tt-metal. Spawns a long-lived
  daemon process; per-frame data goes through stdin/stdout + .npy files.
  Vendored tt-metal SDK lives in `tt/tt-metal/`. Reports
  `blend.{prep,save_npy,daemon_rt,load_npy,device_kernel}` sub-timings.
- **`cuda/`** — placeholder for an upcoming CUDA implementation.
