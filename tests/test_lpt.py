"""Unit tests for the host-side LPT tile->core load balancer."""
import numpy as np

from backends.tt.lpt import build_tile_assignment


def test_lpt_balances_and_covers_nonempty_tiles():
    # per-tile counts [10, 0, 5, 20] -> prefix offsets [0,10,10,15,35]
    offsets = np.array([0, 10, 10, 15, 35], dtype=np.uint32)
    num_tiles = 4
    num_cores = 2
    per_core_offset, per_core_count, tile_ids = build_tile_assignment(
        offsets, num_tiles, num_cores)

    # Every NON-empty tile appears exactly once across all cores' slices.
    seen = []
    for c in range(num_cores):
        s = int(per_core_offset[c])
        n = int(per_core_count[c])
        seen.extend(tile_ids[s:s + n].tolist())
    assert sorted(seen) == [0, 2, 3]  # tile 1 is empty -> filtered out

    # Greedy LPT keeps the max per-core load tight: heaviest tile (3, load 20)
    # goes to one core; tiles 0 (10) + 2 (5) = 15 to the other. Max bin <= 25.
    loads = []
    for c in range(num_cores):
        s, n = int(per_core_offset[c]), int(per_core_count[c])
        loads.append(sum(int(offsets[t + 1] - offsets[t]) for t in tile_ids[s:s + n]))
    assert max(loads) <= 25


def test_lpt_slices_are_contiguous_and_sum_to_total():
    offsets = np.array([0, 3, 7, 8, 8, 15], dtype=np.uint32)
    num_tiles = 5
    num_cores = 3
    per_core_offset, per_core_count, tile_ids = build_tile_assignment(
        offsets, num_tiles, num_cores)
    # Slices tile the tile_ids array with no gaps/overlaps.
    cursor = 0
    for c in range(num_cores):
        assert int(per_core_offset[c]) == cursor
        cursor += int(per_core_count[c])
    assert cursor == tile_ids.shape[0]
    # 4 non-empty tiles (3 is empty) across <= num_cores buckets.
    assert tile_ids.shape[0] == 4
    assert per_core_offset.shape == (num_cores,)
    assert per_core_count.shape == (num_cores,)


def test_lpt_all_empty_tiles():
    offsets = np.array([0, 0, 0], dtype=np.uint32)
    per_core_offset, per_core_count, tile_ids = build_tile_assignment(offsets, 2, 4)
    assert tile_ids.shape[0] == 0
    assert per_core_count.sum() == 0
    assert per_core_offset.shape == (4,)
