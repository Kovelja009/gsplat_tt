"""Host-side LPT tile->core load balancing for the alpha-blend kernel.

Ported from the C++ `compute_lpt_assignment` / `build_tile_assignment` in the
original tt-metal programming example (alpha_blend.cpp). Each 32x32 screen tile
costs roughly its Gaussian count; we distribute tiles across the compute cores
with a greedy longest-processing-time (LPT) heuristic to minimise the maximum
per-core load. Empty tiles are filtered out — the kernel never writes their
output slots, which stay zero (background).

This lives on the Python side now that the device is driven in-process via the
ttnn op: the schedule is passed to the op as hash-excluded attributes
(per_core_offset / per_core_count) plus the concatenated tile_ids buffer.
"""
from __future__ import annotations

import heapq

import numpy as np


def greedy_lpt(item_weights, num_cores):
    """Greedy longest-processing-time bin-packing.

    Assigns each `(item_id, weight)` to the currently least-loaded of `num_cores`
    bins, heaviest item first, to minimise the maximum per-bin total. Returns
    `num_cores` buckets, each a list of `item_id`s. Tie-breaks are deterministic:
    equal weights keep input order (stable sort); equal bin loads pick the lowest
    bin index (heap order). Shared by the single-op (`build_tile_assignment`) and
    two-phase (`segments.build_segmented_assignment`) schedulers.
    """
    order = sorted(item_weights, key=lambda iw: iw[1], reverse=True)
    buckets: list[list[int]] = [[] for _ in range(num_cores)]
    heap = [(0, c) for c in range(num_cores)]  # (current_load, core)
    heapq.heapify(heap)
    for item_id, w in order:
        cur, c = heapq.heappop(heap)
        buckets[c].append(item_id)
        heapq.heappush(heap, (cur + int(w), c))
    return buckets


def build_tile_assignment(offsets: np.ndarray, num_tiles: int, num_cores: int):
    """Greedy-LPT tile->core assignment.

    Args:
        offsets: (num_tiles + 1,) prefix sums of per-tile Gaussian counts.
        num_tiles: number of 32x32 screen tiles.
        num_cores: number of compute cores to distribute across.

    Returns:
        (per_core_offset, per_core_count, tile_ids) where:
          - tile_ids (u32) is the concatenation of each core's tile-id list,
          - core c owns the contiguous slice
            tile_ids[per_core_offset[c] : per_core_offset[c] + per_core_count[c]].
    """
    offsets = np.asarray(offsets, dtype=np.int64)
    loads = offsets[1:num_tiles + 1] - offsets[:num_tiles]

    # LPT over non-empty tiles (empty tiles dropped — the kernel never writes
    # their output slots, which stay zero/background).
    items = [(t, int(loads[t])) for t in range(num_tiles) if loads[t] > 0]
    buckets = greedy_lpt(items, num_cores)

    per_core_offset = np.zeros(num_cores, dtype=np.uint32)
    per_core_count = np.zeros(num_cores, dtype=np.uint32)
    tile_ids_list: list[int] = []
    for c in range(num_cores):
        per_core_offset[c] = len(tile_ids_list)
        per_core_count[c] = len(buckets[c])
        tile_ids_list.extend(buckets[c])

    tile_ids = np.asarray(tile_ids_list, dtype=np.uint32)
    return per_core_offset, per_core_count, tile_ids


def build_round_robin_assignment(offsets: np.ndarray, num_tiles: int, num_cores: int):
    """Load-blind tile->core assignment: non-empty tile t -> core (t % num_cores).

    Baseline against build_tile_assignment's greedy-LPT. Empty tiles are dropped
    (their output slots stay zero via the op's pre-zero, exactly as in the LPT
    builder); each surviving tile t is appended to core (t % num_cores). Tiles
    are walked in ascending tile-id order, so a core's tile_ids slice is
    ascending. Returns the same (per_core_offset, per_core_count, tile_ids)
    tuple as build_tile_assignment, so the single-op path consumes it unchanged.
    """
    offsets = np.asarray(offsets, dtype=np.int64)
    loads = offsets[1:num_tiles + 1] - offsets[:num_tiles]

    buckets: list[list[int]] = [[] for _ in range(num_cores)]
    for t in range(num_tiles):
        if loads[t] > 0:
            buckets[t % num_cores].append(t)

    per_core_offset = np.zeros(num_cores, dtype=np.uint32)
    per_core_count = np.zeros(num_cores, dtype=np.uint32)
    tile_ids_list: list[int] = []
    for c in range(num_cores):
        per_core_offset[c] = len(tile_ids_list)
        per_core_count[c] = len(buckets[c])
        tile_ids_list.extend(buckets[c])

    tile_ids = np.asarray(tile_ids_list, dtype=np.uint32)
    return per_core_offset, per_core_count, tile_ids
