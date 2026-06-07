"""Map a render cell's native timings into reconciling phase buckets.

The four buckets — load / compute / return / transfer — always sum exactly
to the `blend` stage wall time. `transfer` is the explicit residual: the
host<->device data movement a backend does not separately instrument
(notably the DRAM up/down + IPC bundled inside the TT daemon round-trip),
so no time is hidden.

The shared host pre-stages (project / tile_assign / sort) are passed
through unchanged — identical CPU work for every backend, kept out of the
four buckets so cross-backend `compute` stays comparable.
"""
from __future__ import annotations


def _get(sub: dict, key: str) -> float:
    """Blend sub-timings arrive prefixed with 'blend.' from the Pipeline."""
    return float(sub.get(f"blend.{key}", 0.0))


def to_buckets(timings: dict, sub_timings: dict, backend: str) -> dict:
    """Return {load, compute, return, transfer, device_kernel, project,
    tile_assign, sort} in ms. load+compute+return+transfer == blend."""
    blend = float(timings.get("blend", 0.0))
    sub = sub_timings or {}

    if backend == "cpu":
        load, compute, ret, dev = 0.0, blend, 0.0, blend

    elif backend == "cuda":
        load = _get(sub, "upload")
        dev = _get(sub, "kernel.device")
        compute = dev
        ret = max(0.0, _get(sub, "kernel") - dev)

    elif backend == "tt":
        load = _get(sub, "prep") + _get(sub, "write_shm") + _get(sub, "save_npy")
        # MFRAME path uses mframe_rt/read_shm; .npy fallback uses daemon_rt/load_npy.
        dev = _get(sub, "mframe_rt.device_kernel") + _get(sub, "daemon_rt.device_kernel")
        compute = dev
        ret = _get(sub, "read_shm") + _get(sub, "load_npy")

    else:
        raise ValueError(f"unknown backend {backend!r}")

    transfer = blend - (load + compute + ret)
    # Degrade gracefully if no sub-timings were reported: everything to compute.
    if compute == 0.0 and load == 0.0 and ret == 0.0 and blend > 0.0:
        compute, dev, transfer = blend, blend, 0.0

    return {
        "load": load,
        "compute": compute,
        "return": ret,
        "transfer": transfer,
        "device_kernel": dev,
        "project": float(timings.get("project", 0.0)),
        "tile_assign": float(timings.get("tile_assign", 0.0)),
        "sort": float(timings.get("sort", 0.0)),
    }
