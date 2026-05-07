# Backends

One subpackage per hardware target. Each backend implements the same
interface so the viewer can swap them by name (`--backend cpu|tt|cuda|...`).
The backend name on the CLI matches the directory name under `backends/`.

## Layout per backend

```
backends/<arch>/
├── __init__.py
├── backend.py                  # Python wrapper / IPC / device init
├── kernels/                    # native source (.cpp / .cu / ...) — optional
└── <vendored-sdk>/             # e.g. tt-metal/, cuda-toolkit/ — optional
```

The Python wrapper is the only piece `gsplat.viewer` imports. Everything
else (vendored SDKs, kernel C++) is implementation detail of the wrapper.

## Wrapper interface

```python
class Backend:
    def __init__(self, verbose: bool = False): ...

    def render(
        self,
        means_2d, covs_2d, colors, opacities,
        sorted_gids, tile_ranges,
        H: int, W: int,
    ) -> np.ndarray:
        """Return an (H, W, 3) float32 RGB image in [0, 1]."""

    def close(self): ...
```

`gsplat.viewer.GaussianViewer` consumes this contract. As long as the
new backend exposes these three methods, the viewer needs no changes.

## Existing backends

- **`tt/`** — Tenstorrent Wormhole / tt-metal. Spawns a long-lived
  daemon process; per-frame data goes through stdin/stdout + .npy files.
  Vendored tt-metal SDK lives in `tt/tt-metal/`.

- **`cuda/`** — Placeholder for an upcoming CUDA implementation.

## Adding a new backend

1. Create `backends/<arch>/` with `__init__.py` and `backend.py`.
2. Implement the wrapper interface above.
3. (Optional) Add native kernels under `backends/<arch>/kernels/` and
   wire them up via whatever build system the target uses.
4. Register the new backend name in `gsplat/__main__.py`'s `--backend`
   choices and dispatch in `gsplat/viewer.py`.
