"""Per-architecture backends for the alpha-blend forward pass.

Each subpackage exposes a `backend.py` with a class implementing the
common backend interface used by `gsplat.viewer`:

    class Backend:
        def render(means_2d, covs_2d, colors, opacities,
                   sorted_gids, tile_ranges, H, W) -> np.ndarray
        def close()

See `backends/README.md` for instructions on adding a new backend.
"""
