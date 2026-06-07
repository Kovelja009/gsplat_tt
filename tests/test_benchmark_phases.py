"""Unit tests for the benchmark load/compute/return/transfer bucket model."""
import math

from benchmark.phases import to_buckets


def _approx(a, b, tol=1e-6):
    return math.isclose(a, b, rel_tol=0, abs_tol=tol)


def test_cpu_all_in_compute():
    timings = {"project": 5.0, "tile_assign": 2.0, "sort": 3.0,
               "blend": 40.0, "total": 50.0}
    b = to_buckets(timings, {}, "cpu")
    assert _approx(b["load"], 0.0)
    assert _approx(b["compute"], 40.0)
    assert _approx(b["return"], 0.0)
    assert _approx(b["transfer"], 0.0)
    assert _approx(b["device_kernel"], 40.0)
    # Shared host pre-stages passed through unchanged.
    assert _approx(b["project"], 5.0)
    assert _approx(b["tile_assign"], 2.0)
    assert _approx(b["sort"], 3.0)
    # Reconciliation: buckets sum to blend.
    assert _approx(b["load"] + b["compute"] + b["return"] + b["transfer"], 40.0)


def test_cuda_mapping_and_reconciliation():
    timings = {"project": 5.0, "tile_assign": 2.0, "sort": 3.0,
               "blend": 30.0, "total": 40.0}
    sub = {"blend.upload": 4.0, "blend.kernel": 20.0, "blend.kernel.device": 15.0}
    b = to_buckets(timings, sub, "cuda")
    assert _approx(b["load"], 4.0)            # upload
    assert _approx(b["compute"], 15.0)        # kernel.device
    assert _approx(b["return"], 5.0)          # kernel - kernel.device
    assert _approx(b["device_kernel"], 15.0)
    # transfer = blend - (load+compute+return) = 30 - 24 = 6 (untimed remainder)
    assert _approx(b["transfer"], 6.0)
    assert _approx(b["load"] + b["compute"] + b["return"] + b["transfer"], 30.0)


def test_tt_mframe_mapping_and_reconciliation():
    timings = {"project": 5.0, "tile_assign": 2.0, "sort": 3.0,
               "blend": 50.0, "total": 60.0}
    sub = {"blend.prep": 6.0, "blend.write_shm": 4.0,
           "blend.mframe_rt": 35.0, "blend.mframe_rt.device_kernel": 25.0,
           "blend.read_shm": 3.0}
    b = to_buckets(timings, sub, "tt")
    assert _approx(b["load"], 10.0)           # prep + write_shm
    assert _approx(b["compute"], 25.0)        # device_kernel
    assert _approx(b["return"], 3.0)          # read_shm
    assert _approx(b["device_kernel"], 25.0)
    # transfer = 50 - (10+25+3) = 12 (DRAM up/down + IPC inside mframe_rt)
    assert _approx(b["transfer"], 12.0)
    assert _approx(b["load"] + b["compute"] + b["return"] + b["transfer"], 50.0)


def test_tt_npy_fallback_keys():
    # No mframe_rt/read_shm; daemon .npy path emits daemon_rt.*/load_npy.
    timings = {"project": 1.0, "tile_assign": 1.0, "sort": 1.0,
               "blend": 50.0, "total": 53.0}
    sub = {"blend.prep": 6.0, "blend.save_npy": 4.0,
           "blend.daemon_rt": 35.0, "blend.daemon_rt.device_kernel": 25.0,
           "blend.load_npy": 3.0}
    b = to_buckets(timings, sub, "tt")
    assert _approx(b["load"], 10.0)           # prep + save_npy
    assert _approx(b["compute"], 25.0)        # daemon_rt.device_kernel
    assert _approx(b["return"], 3.0)          # load_npy
    assert _approx(b["load"] + b["compute"] + b["return"] + b["transfer"], 50.0)


def test_missing_subtimings_degrades_to_compute():
    # Unknown/empty sub-timings: whole blend wall lands in compute, no throw.
    timings = {"project": 1.0, "tile_assign": 1.0, "sort": 1.0,
               "blend": 12.0, "total": 15.0}
    b = to_buckets(timings, {}, "tt")
    assert _approx(b["compute"], 12.0)
    assert _approx(b["load"], 0.0)
    assert _approx(b["return"], 0.0)
    assert _approx(b["transfer"], 0.0)
