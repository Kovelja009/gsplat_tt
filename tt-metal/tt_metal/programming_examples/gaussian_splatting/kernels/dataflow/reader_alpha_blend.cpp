// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// T2.2 / T4.2 alpha-blend reader: streams 4 input data sources from DRAM into 4
// distinct circular buffers per screen tile.
//
//   CB 0 (cb_px)        <- one bf16 32x32 tile per screen tile (px coords)
//   CB 1 (cb_py)        <- one bf16 32x32 tile per screen tile (py coords)
//   CB 2 (cb_scalars)   <- one 64-byte fp32 pack per Gaussian
//   CB 3 (cb_tile_meta) <- one uint32 (g_count) per screen tile
//
// T4.2 LPT load balancing: instead of processing a contiguous range
// [first_tile_id, first_tile_id + num_tiles) this core processes a list of
// (potentially non-contiguous) tile IDs read from a small per-core slice of
// a concatenated tile-ID DRAM buffer. The slice is loaded into L1 once at
// startup; the per-tile loop then indexes by element rather than computing
// `first_tile_id + t`.
//
// Per screen tile id (drawn from the per-core tile-ID list):
//   1. fetch tile_offsets[id] and tile_offsets[id+1] -> g_start, g_end
//   2. push g_count = g_end - g_start to cb_tile_meta
//   3. push px and py tiles for this screen tile
//   4. for each Gaussian g in [g_start, g_end), push one scalar pack
//
// Runtime args layout:
//   0: packs_addr        (DRAM base of scalar packs buffer; one 64B page per Gaussian)
//   1: tile_offsets_addr (DRAM base of tile_offsets[]; one uint32 page per tile boundary)
//   2: px_addr           (DRAM base of px tiles; one 32x32 bf16 tile per screen tile)
//   3: py_addr           (DRAM base of py tiles; one 32x32 bf16 tile per screen tile)
//   4: tile_ids_addr     (DRAM base of concatenated per-core tile-ID list, uint32 elems,
//                         page size 64B)
//   5: tile_ids_start    (uint32 element offset of this core's slice)
//   6: tile_ids_count    (number of tile IDs this core handles)
//
// Compile-time args: 5 TensorAccessorArgs in order
//   packs, tile_offsets, px, py, tile_ids.

// Max per-core tile IDs we cache in L1. The 64-core / 400-tile case puts
// ~6-7 tiles on each core after LPT; even with severe imbalance from
// pathological cost distributions we stay well under 64. 256 bytes total.
constexpr uint32_t MAX_TILE_IDS_PER_CORE = 64;

void kernel_main() {
    uint32_t packs_addr        = get_arg_val<uint32_t>(0);
    uint32_t tile_offsets_addr = get_arg_val<uint32_t>(1);
    uint32_t px_addr           = get_arg_val<uint32_t>(2);
    uint32_t py_addr           = get_arg_val<uint32_t>(3);
    uint32_t tile_ids_addr     = get_arg_val<uint32_t>(4);
    uint32_t tile_ids_start    = get_arg_val<uint32_t>(5);
    uint32_t tile_ids_count    = get_arg_val<uint32_t>(6);

    constexpr uint32_t CB_PX        = 0;
    constexpr uint32_t CB_PY        = 1;
    constexpr uint32_t CB_SCALARS   = 2;
    constexpr uint32_t CB_TILE_META = 3;

    const uint32_t tile_bytes        = get_tile_size(CB_PX);       // 2048 (32x32 bf16)
    // NOTE: do NOT use get_tile_size(CB_SCALARS) here. CB_SCALARS is configured
    // as DataFormat::Float32 (so the SFPU sees fp32), but with a sub-tile page
    // size of 64 bytes (9 fp32 scalar pack). get_tile_size returns the format
    // tile size (4096 for fp32 32x32), not the configured page size, so using
    // it as the NoC transaction size would overflow the CB by 3840 bytes per
    // Gaussian and trample adjacent L1 (manifests as a multi-tile hang once
    // the corrupted L1 overlaps a CB descriptor).
    constexpr uint32_t pack_bytes_padded = 64;  // matches host SCALAR_PACK_PAGE_BYTES
    constexpr uint32_t tile_ids_page_bytes = 64;  // matches host TILE_IDS_PAGE_BYTES

    constexpr auto packs_args     = TensorAccessorArgs<0>();
    constexpr auto offsets_args   = TensorAccessorArgs<packs_args.next_compile_time_args_offset()>();
    constexpr auto px_args        = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();
    constexpr auto py_args        = TensorAccessorArgs<px_args.next_compile_time_args_offset()>();
    constexpr auto tile_ids_args  = TensorAccessorArgs<py_args.next_compile_time_args_offset()>();

    const auto packs_acc    = TensorAccessor(packs_args,    packs_addr,        pack_bytes_padded);
    const auto offsets_acc  = TensorAccessor(offsets_args,  tile_offsets_addr, /*page_size=*/4);
    const auto px_acc       = TensorAccessor(px_args,       px_addr,           tile_bytes);
    const auto py_acc       = TensorAccessor(py_args,       py_addr,           tile_bytes);
    const auto tile_ids_acc = TensorAccessor(tile_ids_args, tile_ids_addr,     tile_ids_page_bytes);

    // No-work cores (LPT may leave some cores empty when num_tiles < num_cores).
    if (tile_ids_count == 0) {
        return;
    }

    // Per-tile offset scratch + per-core tile-ID cache. Both live in a small
    // L1 region grabbed from the cb_tile_meta CB write pointer. Layout:
    //   bytes [0, 4)               -> offset scratch slot (overwritten per push)
    //   bytes [4, 4 + 4*count)     -> cached per-core tile-ID list
    //
    // We grab the L1 region from cb_tile_meta because we won't push to it
    // until after all the prefetches are complete. The first per-tile iter
    // refreshes scratch_addr (cb_tile_meta wraps on push_back), so we keep
    // the cache valid by reading it into a local stack array before the loop.
    uint32_t scratch_addr = get_write_ptr(CB_TILE_META);
    auto scratch_ptr = reinterpret_cast<volatile uint32_t*>(scratch_addr);

    // Read this core's tile-ID slice into L1 with one NoC transaction per
    // page. The slice may straddle multiple 64-byte pages; we loop over
    // pages and copy into a local buffer.
    uint32_t tile_ids[MAX_TILE_IDS_PER_CORE];
    {
        // First page that contains tile_ids_start; offset within that page.
        const uint32_t ids_per_page = tile_ids_page_bytes / 4;  // 16
        uint32_t page_idx = tile_ids_start / ids_per_page;
        uint32_t in_page  = tile_ids_start % ids_per_page;
        uint32_t remaining = tile_ids_count;
        uint32_t out_idx = 0;
        // L1 scratch slot for one page (64B), reuse slot at scratch_addr.
        while (remaining > 0) {
            uint64_t page_noc = get_noc_addr(page_idx, tile_ids_acc);
            noc_async_read(page_noc, scratch_addr, tile_ids_page_bytes);
            noc_async_read_barrier();
            uint32_t take = ids_per_page - in_page;
            if (take > remaining) take = remaining;
            for (uint32_t i = 0; i < take; i++) {
                tile_ids[out_idx + i] = scratch_ptr[in_page + i];
            }
            out_idx   += take;
            remaining -= take;
            page_idx  += 1;
            in_page    = 0;
        }
    }

    for (uint32_t t = 0; t < tile_ids_count; t++) {
        uint32_t tile_id = tile_ids[t];

        // Fetch offsets[tile_id] and offsets[tile_id+1] -> g_start, g_end.
        // Note: with non-contiguous tile IDs we can't carry-forward g_end
        // from the previous iteration, so each iteration does two reads.
        // Cost is small relative to the per-tile Gaussian stream.
        {
            uint64_t off_noc = get_noc_addr(tile_id, offsets_acc);
            noc_async_read(off_noc, scratch_addr, 4);
            noc_async_read_barrier();
        }
        uint32_t g_start = scratch_ptr[0];

        {
            uint64_t off_noc = get_noc_addr(tile_id + 1, offsets_acc);
            noc_async_read(off_noc, scratch_addr, 4);
            noc_async_read_barrier();
        }
        uint32_t g_end   = scratch_ptr[0];
        uint32_t g_count = g_end - g_start;

        // Push g_count to meta CB. We write directly to the meta CB write pointer;
        // this overwrites the scratch slot we just used (safe -- we no longer need it).
        cb_reserve_back(CB_TILE_META, 1);
        auto meta_ptr = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_TILE_META));
        meta_ptr[0] = g_count;
        cb_push_back(CB_TILE_META, 1);

        // Push px and py tiles for this screen tile.
        cb_reserve_back(CB_PX, 1);
        noc_async_read_tile(tile_id, px_acc, get_write_ptr(CB_PX));
        cb_reserve_back(CB_PY, 1);
        noc_async_read_tile(tile_id, py_acc, get_write_ptr(CB_PY));
        noc_async_read_barrier();
        cb_push_back(CB_PX, 1);
        cb_push_back(CB_PY, 1);

        // Stream Gaussian scalar packs for this tile.
        for (uint32_t g = 0; g < g_count; g++) {
            uint32_t entry_id = g_start + g;
            cb_reserve_back(CB_SCALARS, 1);
            noc_async_read_tile(entry_id, packs_acc, get_write_ptr(CB_SCALARS));
            noc_async_read_barrier();
            cb_push_back(CB_SCALARS, 1);
        }

        // Refresh the scratch_addr in case cb_tile_meta wrapped (write_ptr moves on push_back).
        scratch_addr = get_write_ptr(CB_TILE_META);
        scratch_ptr  = reinterpret_cast<volatile uint32_t*>(scratch_addr);
    }
}
