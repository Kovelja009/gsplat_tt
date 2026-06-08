# Intra-tile Parallelism — Phase 1 (host math + scheduler) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and prove the host-side foundations of intra-tile parallelism — the associative-`over` combine math and the depth-segment scheduler — so the kernel/op work (Phase 2) implements an already-validated contract.

**Architecture:** Two pure-Python, fully-tested units in `backends/tt/segments.py`: (1) a reference compositor + `over` combine that proves splitting a tile's depth-sorted Gaussians into K segments and merging the partials reproduces the monolithic front-to-back result; (2) `build_segmented_assignment`, which splits heavy tiles into contiguous depth-segments, LPT-balances the resulting jobs across cores, and emits the combine plan the Phase-2 ops consume.

**Tech Stack:** Python 3, numpy, pytest. No device, no C++ — this phase is host-only and runs on any box.

**Scope:** This is **Phase 1 of 2** for the spec `docs/superpowers/specs/2026-06-08-intra-tile-parallelism-design.md` (spec milestones 1–2). Phase 2 (milestones 3–6: phase-1 partial op, phase-2 combine op, backend wiring, integration PSNR + perf) is a separate plan, written after Phase 1 lands so the kernel implements this phase's tested interfaces.

---

## File Structure

- `backends/tt/segments.py` — new. Host-side segmentation + combine:
  - `composite_tile(...)` — reference front-to-back compositor returning `(color, T)`.
  - `combine_over(...)` — merge per-segment partials via Porter-Duff `over`.
  - `split_segments(...)` — split a Gaussian range into contiguous depth-segments.
  - `SegmentedSchedule` dataclass + `build_segmented_assignment(...)` — the scheduler.
- `tests/test_tt_segments.py` — new. Unit tests for all of the above.

`backends/tt/lpt.py` is left untouched (it stays the no-split fast path). `segments.py` is the new, focused home for the split path so neither file does too much.

---

### Task 1: Reference compositor + `over` combine (the math)

**Files:**
- Create: `backends/tt/segments.py`
- Test: `tests/test_tt_segments.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_tt_segments.py`:

```python
"""Unit tests for host-side intra-tile segmentation + combine."""
import math

import numpy as np

from backends.tt.segments import (
    composite_tile, combine_over, split_segments,
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_tt_segments.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backends.tt.segments'`

- [ ] **Step 3: Implement `composite_tile`, `combine_over`, `split_segments`**

Create `backends/tt/segments.py`:

```python
"""Host-side intra-tile depth-segmentation + associative-over combine.

Splits a heavy tile's depth-sorted Gaussian list into contiguous segments so
its alpha-blend work can spread across cores, and provides the reference combine
(Porter-Duff `over`) that merges per-segment partials back into the exact
front-to-back result. The `over` operator is associative, which is what makes
the split valid:  (Ca,Ta) over (Cb,Tb) = (Ca + Ta*Cb, Ta*Tb).

See docs/superpowers/specs/2026-06-08-intra-tile-parallelism-design.md
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np


def composite_tile(gx, gy, means, cov_inv, colors, opac, gidx):
    """Front-to-back composite Gaussians `gidx` over a tile's pixel grid.

    gx, gy: (H, W) per-pixel coords. means: (M,2). cov_inv: (M,2,2).
    colors: (M,3). opac: (M,). gidx: iterable of Gaussian indices, depth order.
    Returns (color (H,W,3) float32, T (H,W) float32) with T starting at 1.
    """
    H, W = gx.shape
    C = np.zeros((H, W, 3), dtype=np.float32)
    T = np.ones((H, W), dtype=np.float32)
    for g in gidx:
        dx = gx - means[g, 0]
        dy = gy - means[g, 1]
        ci = cov_inv[g]
        power = -0.5 * (ci[0, 0] * dx * dx + 2.0 * ci[0, 1] * dx * dy + ci[1, 1] * dy * dy)
        alpha = np.clip(opac[g] * np.exp(np.minimum(power, 0.0)), None, 0.99)
        C += (alpha * T)[:, :, np.newaxis] * colors[g]
        T *= (1.0 - alpha)
    return C, T


def combine_over(seg_colors, seg_Ts):
    """Merge per-segment partials in depth order via the `over` operator.

    seg_colors: list of (H,W,3) accumulated colors (each composited from T=1).
    seg_Ts: list of (H,W) final transmittances per segment.
    Returns (H,W,3): C = sum_i (prod_{j<i} T_j) * C_i.
    """
    C = np.zeros_like(seg_colors[0])
    T = np.ones_like(seg_Ts[0])
    for ci, ti in zip(seg_colors, seg_Ts):
        C = C + T[:, :, np.newaxis] * ci
        T = T * ti
    return C


def split_segments(g_start, g_end, target):
    """Split [g_start, g_end) into ceil(n/target) contiguous (start, count)
    segments whose sizes differ by at most 1.

    target <= 0 or n <= target => a single segment. Sizes are <= target.
    """
    n = g_end - g_start
    if n <= 0:
        return []
    if target <= 0 or n <= target:
        return [(g_start, n)]
    k = math.ceil(n / target)
    base, rem = divmod(n, k)
    segs = []
    s = g_start
    for i in range(k):
        cnt = base + (1 if i < rem else 0)
        segs.append((s, cnt))
        s += cnt
    return segs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/test_tt_segments.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add backends/tt/segments.py tests/test_tt_segments.py
git commit -m "feat(tt): reference compositor + associative-over combine (intra-tile)"
```

---

### Task 2: Depth-segment scheduler `build_segmented_assignment`

**Files:**
- Modify: `backends/tt/segments.py`
- Test: `tests/test_tt_segments.py`

This mirrors `backends/tt/lpt.py::build_tile_assignment` (which takes
`offsets` = per-tile prefix sums and LPT-balances tiles) but splits heavy tiles
into segment-jobs first. A "job" is `(tile_id, gseg_start, gseg_count,
partial_slot)`. Partial slots are assigned so all segments of a tile are
**contiguous and depth-ordered**, letting Phase-2 combine read
`partials[first_slot : first_slot+K]` directly.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tt_segments.py`:

```python
from backends.tt.segments import SegmentedSchedule, build_segmented_assignment


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
    row = 0
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_tt_segments.py -k "split or lpt or combine_slots or no_split" -v`
Expected: FAIL — `ImportError: cannot import name 'SegmentedSchedule'`

- [ ] **Step 3: Implement the scheduler**

Append to `backends/tt/segments.py`:

```python
@dataclass
class SegmentedSchedule:
    """Output of build_segmented_assignment.

    job_table: (num_jobs, 4) uint32 rows [tile_id, gseg_start, gseg_count,
        partial_slot], ordered by owning core then within-core order.
    per_core_offset / per_core_count: (num_cores,) uint32 — core c owns
        job_table[per_core_offset[c] : per_core_offset[c] + per_core_count[c]].
    combine_plan: (num_nonempty_tiles, 3) uint32 rows
        [out_tile_id, first_slot, K] — tile's K partials live in contiguous
        slots [first_slot, first_slot+K) in depth order.
    num_jobs: total segment-jobs (== partials buffer length).
    """
    job_table: np.ndarray
    per_core_offset: np.ndarray
    per_core_count: np.ndarray
    combine_plan: np.ndarray
    num_jobs: int


def build_segmented_assignment(offsets, num_tiles, num_cores, target=None):
    """Split heavy tiles into contiguous depth-segments and LPT-balance the
    resulting jobs across cores.

    offsets: (num_tiles+1,) prefix sums of per-tile Gaussian counts.
    target: max Gaussians per segment; default ceil(total / num_cores) (the
        load-balance ideal). A tile with load <= target is not split.
    """
    offsets = np.asarray(offsets, dtype=np.int64)
    loads = offsets[1:num_tiles + 1] - offsets[:num_tiles]
    total = int(loads.sum())
    if target is None:
        target = max(1, math.ceil(total / max(1, num_cores)))

    # Build jobs (tile order) + combine plan; slots contiguous per tile.
    jobs = []          # (tile_id, gseg_start, gseg_count)
    combine = []       # (tile_id, first_slot, K)
    slot = 0
    for t in range(num_tiles):
        if loads[t] <= 0:
            continue
        segs = split_segments(int(offsets[t]), int(offsets[t + 1]), target)
        combine.append((t, slot, len(segs)))
        for gs, gc in segs:
            jobs.append((t, gs, gc, slot))
            slot += 1
    num_jobs = len(jobs)

    # Greedy-LPT: heaviest segment first onto the least-loaded core.
    order = sorted(range(num_jobs), key=lambda j: jobs[j][2], reverse=True)
    buckets = [[] for _ in range(num_cores)]
    heap = [(0, c) for c in range(num_cores)]
    heapq.heapify(heap)
    for j in order:
        cur, c = heapq.heappop(heap)
        buckets[c].append(j)
        heapq.heappush(heap, (cur + jobs[j][2], c))

    per_core_offset = np.zeros(num_cores, dtype=np.uint32)
    per_core_count = np.zeros(num_cores, dtype=np.uint32)
    rows = []
    for c in range(num_cores):
        per_core_offset[c] = len(rows)
        per_core_count[c] = len(buckets[c])
        for j in buckets[c]:
            rows.append(jobs[j])
    job_table = (np.asarray(rows, dtype=np.uint32).reshape(-1, 4)
                 if rows else np.zeros((0, 4), dtype=np.uint32))
    combine_plan = (np.asarray(combine, dtype=np.uint32).reshape(-1, 3)
                    if combine else np.zeros((0, 3), dtype=np.uint32))
    return SegmentedSchedule(job_table, per_core_offset, per_core_count, combine_plan, num_jobs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/test_tt_segments.py -v`
Expected: PASS (7 tests total)

- [ ] **Step 5: Commit**

```bash
git add backends/tt/segments.py tests/test_tt_segments.py
git commit -m "feat(tt): depth-segment scheduler build_segmented_assignment"
```

---

## Follow-on: Phase 2 (separate plan, written after Phase 1 lands)

Phase 2 covers spec milestones 3–6 and is **not** detailed here because its exact
interfaces are pinned by Phase 1's tested output (the `job_table` / `combine_plan`
shapes) and its kernel code is developed iteratively against the device. It will
be its own plan once Phase 1 is merged:

1. **Phase-1 partial op** (`gaussian_alpha_blend_partial`): reader feeds each
   job's `(gseg_start, gseg_count)` sub-range; compute emits `CB_T_STATE` as a
   4th output tile; writer writes `(R,G,B,T)` to `partials[partial_slot*4]`.
2. **Phase-2 combine op** (`gaussian_alpha_blend_combine`): per output tile, run
   the `combine_over` scan (validated in Task 1) over its `[first_slot, first_slot+K)`
   partials in fp32 Dst; write RGB to `out`.
3. **Backend wiring**: build the segmented schedule; no-split → existing op;
   split → partial then combine; new `partial_kernel`/`combine_kernel` sub-timings.
4. **Validation**: integration PSNR (train@256 two-phase vs CPU ≥ 35 dB, ≈ current
   ~41 dB) + perf via `benchmark/run.py tt` (train@256 compute 345ms → toward the
   entries/cores ideal; luigi unchanged).

---

## Self-Review

**Spec coverage (Phase 1 = milestones 1–2):**
- Milestone 1 (Python combine reference proving segment+combine == monolithic) → Task 1. ✓
- Milestone 2 (scheduler: contiguous depth-segments, LPT balance, combine plan) → Task 2. ✓
- Milestones 3–6 → explicitly deferred to the Phase-2 plan (scope note + follow-on section). ✓

**Placeholder scan:** no TBD/TODO; every code step has complete code and exact run/commit commands. ✓

**Type consistency:** `composite_tile` returns `(color, T)`, consumed by `combine_over(seg_colors, seg_Ts)` in Task 1 and the Task-2 tests. `split_segments` returns `(start, count)` tuples, consumed identically in Task 1 test, `build_segmented_assignment`, and Task-2 tests. `SegmentedSchedule` field names (`job_table`, `per_core_offset`, `per_core_count`, `combine_plan`, `num_jobs`) match between the dataclass, the constructor, and every Task-2 assertion. `job_table` columns `[tile_id, gseg_start, gseg_count, partial_slot]` are indexed consistently (`r[0..3]`) across tests and impl. ✓
