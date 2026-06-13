"""Unit tests for the host-side LPT tile->core load balancer."""
import numpy as np

from backends.tt.lpt import build_tile_assignment, build_round_robin_assignment


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


def test_round_robin_assigns_by_tile_index_modulo_cores():
    # per-tile counts [10, 0, 5, 20, 7] -> offsets [0,10,10,15,35,42].
    # Non-empty tiles in screen order: 0, 2, 3, 4.
    # Round-robin by ORIGINAL index t % num_cores (num_cores=2):
    #   tile 0 -> core 0, tile 2 -> core 0, tile 3 -> core 1, tile 4 -> core 0.
    offsets = np.array([0, 10, 10, 15, 35, 42], dtype=np.uint32)
    per_core_offset, per_core_count, tile_ids = build_round_robin_assignment(
        offsets, num_tiles=5, num_cores=2)

    def core_tiles(c):
        s, n = int(per_core_offset[c]), int(per_core_count[c])
        return tile_ids[s:s + n].tolist()

    assert core_tiles(0) == [0, 2, 4]   # ascending, t % 2 == 0
    assert core_tiles(1) == [3]         # t % 2 == 1
    # Empty tile 1 dropped.
    assert sorted(core_tiles(0) + core_tiles(1)) == [0, 2, 3, 4]


def test_round_robin_slices_contiguous_and_shapes_match_lpt():
    offsets = np.array([0, 3, 7, 8, 8, 15], dtype=np.uint32)
    num_tiles, num_cores = 5, 3
    rr = build_round_robin_assignment(offsets, num_tiles, num_cores)
    lpt = build_tile_assignment(offsets, num_tiles, num_cores)
    # Same return contract: 3 arrays, per-core arrays sized num_cores,
    # same set of non-empty tiles covered.
    for got, exp in zip(rr, lpt):
        assert got.dtype == exp.dtype
    rr_off, rr_cnt, rr_ids = rr
    assert rr_off.shape == (num_cores,)
    assert rr_cnt.shape == (num_cores,)
    # Slices tile the array with no gaps/overlaps.
    cursor = 0
    for c in range(num_cores):
        assert int(rr_off[c]) == cursor
        cursor += int(rr_cnt[c])
    assert cursor == rr_ids.shape[0]
    # 4 non-empty tiles (tile 3 empty), same coverage as LPT.
    assert sorted(rr_ids.tolist()) == sorted(lpt[2].tolist())


def test_round_robin_all_empty_tiles():
    offsets = np.array([0, 0, 0], dtype=np.uint32)
    per_core_offset, per_core_count, tile_ids = build_round_robin_assignment(
        offsets, num_tiles=2, num_cores=4)
    assert tile_ids.shape[0] == 0
    assert per_core_count.sum() == 0
    assert per_core_offset.shape == (4,)
