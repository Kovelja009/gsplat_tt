"""Unit tests for host-side intra-tile segmentation + combine."""
import math

import numpy as np

from backends.tt.segments import (
    composite_tile, combine_over, split_segments,
    SegmentedSchedule, build_segmented_assignment,
)


def _synthetic_tile(n, seed=0, H=32, W=32):
    """A dense, low-opacity 32x32 tile (train-like) with diagonal covariances."""
    rng = np.random.default_rng(seed)
    ys = np.arange(H, dtype=np.float32) + 0.5
    xs = np.arange(W, dtype=np.float32) + 0.5
    gy, gx = np.meshgrid(ys, xs, indexing="ij")          # both (H, W)
    means = rng.uniform(0.0, 32.0, (n, 2)).astype(np.float32)
    cov_inv = np.zeros((n, 2, 2), dtype=np.float32)
    diag = rng.uniform(0.05, 0.5, (n, 2)).astype(np.float32)
    cov_inv[:, 0, 0] = diag[:, 0]
    cov_inv[:, 1, 1] = diag[:, 1]                        # diagonal SPD => valid inverse-cov
    colors = rng.uniform(0.0, 1.0, (n, 3)).astype(np.float32)
    opac = rng.uniform(0.01, 0.3, n).astype(np.float32)  # low opacity, like the dense tile
    return gx, gy, means, cov_inv, colors, opac


def test_combine_over_two_segments_identity_math():
    # Hand-checked: one segment color C0 with T0, then C1 with T1 -> C0 + T0*C1.
    C0 = np.full((2, 2, 3), 0.4, dtype=np.float32)
    T0 = np.full((2, 2), 0.5, dtype=np.float32)
    C1 = np.full((2, 2, 3), 0.2, dtype=np.float32)
    T1 = np.full((2, 2), 0.7, dtype=np.float32)
    out = combine_over([C0, C1], [T0, T1])
    assert np.allclose(out, 0.4 + 0.5 * 0.2)            # 0.5


def test_segmented_combine_matches_monolithic():
    n = 300
    gx, gy, means, cov_inv, colors, opac = _synthetic_tile(n, seed=1)
    gidx = list(range(n))

    c_mono, _ = composite_tile(gx, gy, means, cov_inv, colors, opac, gidx)

    target = math.ceil(n / 7)                            # ~7 segments
    segs = split_segments(0, n, target)
    cs, ts = [], []
    for s, c in segs:
        ci, ti = composite_tile(gx, gy, means, cov_inv, colors, opac, gidx[s:s + c])
        cs.append(ci); ts.append(ti)
    c_seg = combine_over(cs, ts)

    assert np.max(np.abs(c_mono - c_seg)) < 1e-4         # exact in real arithmetic; fp32 noise only


def test_single_segment_equals_monolithic():
    n = 50
    gx, gy, means, cov_inv, colors, opac = _synthetic_tile(n, seed=2)
    gidx = list(range(n))
    c_mono, t_mono = composite_tile(gx, gy, means, cov_inv, colors, opac, gidx)
    c_one = combine_over([c_mono], [t_mono])
    assert np.array_equal(c_mono, c_one)                 # K=1 combine is identity


def _offsets(loads):
    """Per-tile counts -> (num_tiles+1,) prefix sums, like build_tile_assignment."""
    return np.concatenate([[0], np.cumsum(np.asarray(loads, dtype=np.int64))])


def test_no_split_one_job_per_tile():
    loads = [10, 0, 25, 7]                                # tile 1 empty
    offs = _offsets(loads)
    sch = build_segmented_assignment(offs, num_tiles=4, num_cores=8, target=10_000)
    assert isinstance(sch, SegmentedSchedule)
    assert sch.num_jobs == 3                              # empties dropped, no splits
    # Every non-empty tile -> exactly one combine entry with K=1.
    ks = {int(r[0]): int(r[2]) for r in sch.combine_plan}
    assert ks == {0: 1, 2: 1, 3: 1}
    # job_table rows sum in count back to the loads of non-empty tiles.
    counts = {int(r[0]): int(r[2]) for r in sch.job_table}
    assert counts == {0: 10, 2: 25, 3: 7}


def test_heavy_tile_splits_into_contiguous_ordered_segments():
    loads = [1000]                                        # one heavy tile
    offs = _offsets(loads)
    sch = build_segmented_assignment(offs, num_tiles=1, num_cores=16, target=100)
    assert sch.num_jobs == 10                             # ceil(1000/100)
    # combine plan: tile 0, K=10, slots first..first+9
    assert sch.combine_plan.shape == (1, 3)
    out_tile, first_slot, k = (int(x) for x in sch.combine_plan[0])
    assert out_tile == 0 and k == 10
    # Reconstruct the tile's segments by partial_slot order; must be contiguous
    # and cover [0, 1000) with no gaps/overlaps, in increasing gseg_start.
    rows = sorted((int(r[3]), int(r[1]), int(r[2])) for r in sch.job_table)  # by slot
    slots = [s for s, _, _ in rows]
    assert slots == list(range(first_slot, first_slot + 10))
    cursor = 0
    for _, gstart, gcount in rows:
        assert gstart == cursor                           # contiguous, in order
        cursor += gcount
    assert cursor == 1000                                 # full coverage
    assert max(gc for _, _, gc in rows) <= 100            # each segment <= target


def test_jobs_lpt_balanced_across_cores():
    # One fat tile + many small ones; LPT should keep max-core load near target.
    loads = [1000] + [50] * 20
    offs = _offsets(loads)
    num_cores = 16
    target = 100
    sch = build_segmented_assignment(offs, num_tiles=len(loads), num_cores=num_cores, target=target)
    # Per-core load = sum of its jobs' gseg_count.
    core_load = np.zeros(num_cores, dtype=np.int64)
    for c in range(num_cores):
        cnt = int(sch.per_core_count[c])
        off = int(sch.per_core_offset[c])
        for j in range(off, off + cnt):
            core_load[c] += int(sch.job_table[j, 2])
    total = sum(loads)
    assert core_load.sum() == total                       # all work assigned
    # Greedy-LPT bound: max core <= ideal + largest single job.
    assert core_load.max() <= math.ceil(total / num_cores) + target


def test_combine_slots_partition_all_jobs():
    loads = [300, 0, 450, 120]
    offs = _offsets(loads)
    sch = build_segmented_assignment(offs, num_tiles=4, num_cores=8, target=100)
    # Sum of K over combine plan == num_jobs, and slots 0..num_jobs-1 each once.
    assert int(sch.combine_plan[:, 2].sum()) == sch.num_jobs
    all_slots = sorted(int(r[3]) for r in sch.job_table)
    assert all_slots == list(range(sch.num_jobs))
    # Each tile's K slots are the contiguous block [first_slot, first_slot+K).
    for out_tile, first_slot, k in sch.combine_plan:
        tile_slots = sorted(int(r[3]) for r in sch.job_table if int(r[0]) == int(out_tile))
        assert tile_slots == list(range(int(first_slot), int(first_slot) + int(k)))
