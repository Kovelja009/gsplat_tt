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
