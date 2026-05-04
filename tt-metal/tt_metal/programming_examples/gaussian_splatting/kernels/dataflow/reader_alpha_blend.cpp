// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// T2.2 alpha-blend reader: streams 4 input data sources from DRAM into 4
// distinct circular buffers per screen tile.
//
//   CB 0 (cb_px)        <- one bf16 32x32 tile per screen tile (px coords)
//   CB 1 (cb_py)        <- one bf16 32x32 tile per screen tile (py coords)
//   CB 2 (cb_scalars)   <- one 64-byte fp32 pack per Gaussian
//   CB 3 (cb_tile_meta) <- one uint32 (g_count) per screen tile
//
// Per screen tile t (in [first_tile_id, first_tile_id + num_tiles)):
//   1. fetch tile_offsets[t] and tile_offsets[t+1] -> g_start, g_end
//   2. push g_count = g_end - g_start to cb_tile_meta
//   3. push px and py tiles for this screen tile
//   4. for each Gaussian g in [g_start, g_end), push one scalar pack
//
// Runtime args layout:
//   0: packs_addr       (DRAM base of scalar packs buffer; one 64B page per Gaussian)
//   1: tile_offsets_addr(DRAM base of tile_offsets[]; one uint32 page per tile boundary)
//   2: px_addr          (DRAM base of px tiles; one 32x32 bf16 tile per screen tile)
//   3: py_addr          (DRAM base of py tiles; one 32x32 bf16 tile per screen tile)
//   4: first_tile_id    (first screen tile this core handles)
//   5: num_tiles        (number of screen tiles this core handles)
//
// Compile-time args: 4 TensorAccessorArgs in order
//   packs, tile_offsets, px, py.
void kernel_main() {
    uint32_t packs_addr        = get_arg_val<uint32_t>(0);
    uint32_t tile_offsets_addr = get_arg_val<uint32_t>(1);
    uint32_t px_addr           = get_arg_val<uint32_t>(2);
    uint32_t py_addr           = get_arg_val<uint32_t>(3);
    uint32_t first_tile_id     = get_arg_val<uint32_t>(4);
    uint32_t num_tiles         = get_arg_val<uint32_t>(5);

    constexpr uint32_t CB_PX        = 0;
    constexpr uint32_t CB_PY        = 1;
    constexpr uint32_t CB_SCALARS   = 2;
    constexpr uint32_t CB_TILE_META = 3;

    const uint32_t tile_bytes        = get_tile_size(CB_PX);       // 2048 (32x32 bf16)
    const uint32_t pack_bytes_padded = get_tile_size(CB_SCALARS);  // 64 (9 fp32 -> padded)

    constexpr auto packs_args   = TensorAccessorArgs<0>();
    constexpr auto offsets_args = TensorAccessorArgs<packs_args.next_compile_time_args_offset()>();
    constexpr auto px_args      = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();
    constexpr auto py_args      = TensorAccessorArgs<px_args.next_compile_time_args_offset()>();

    const auto packs_acc   = TensorAccessor(packs_args,   packs_addr,        pack_bytes_padded);
    const auto offsets_acc = TensorAccessor(offsets_args, tile_offsets_addr, /*page_size=*/4);
    const auto px_acc      = TensorAccessor(px_args,      px_addr,           tile_bytes);
    const auto py_acc      = TensorAccessor(py_args,      py_addr,           tile_bytes);

    // Per-tile offset scratch: a 4-byte slot in L1 we read offsets into one at a time.
    // We cache the previous iteration's g_end and reuse it as the next g_start, so we
    // only need one offset fetch per iteration (plus one priming fetch up-front).
    //
    // We grab a 16-byte L1 scratch region from the cb_tile_meta CB write pointer.
    // cb_tile_meta has page size >= 4 bytes; we won't push to it until after the
    // scratch fetch is complete, so the scratch region is safe to reuse.
    uint32_t scratch_addr = get_write_ptr(CB_TILE_META);
    auto scratch_ptr = reinterpret_cast<volatile uint32_t*>(scratch_addr);

    // Prime g_start by reading offsets[first_tile_id].
    {
        uint64_t off_noc = get_noc_addr(first_tile_id, offsets_acc);
        noc_async_read(off_noc, scratch_addr, 4);
        noc_async_read_barrier();
    }
    uint32_t g_start = scratch_ptr[0];

    for (uint32_t t = 0; t < num_tiles; t++) {
        uint32_t tile_id = first_tile_id + t;

        // Fetch offsets[tile_id + 1] -> g_end.
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

        // Carry g_end into next iteration's g_start.
        g_start = g_end;

        // Refresh the scratch_addr in case cb_tile_meta wrapped (write_ptr moves on push_back).
        scratch_addr = get_write_ptr(CB_TILE_META);
        scratch_ptr  = reinterpret_cast<volatile uint32_t*>(scratch_addr);
    }
}
