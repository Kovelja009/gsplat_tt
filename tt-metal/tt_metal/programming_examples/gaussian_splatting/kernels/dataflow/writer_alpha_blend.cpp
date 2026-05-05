// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// T2.3 / T4.2 alpha-blend writer: drains 3 tiles per screen tile from
// cb_color_out (R, G, B in that order) and async-writes them to consecutive
// DRAM tile slots {3*tile_id + 0, 3*tile_id + 1, 3*tile_id + 2}.
//
// T4.2 LPT load balancing: this core processes a list of (potentially
// non-contiguous) tile IDs read from a small per-core slice of a
// concatenated tile-ID DRAM buffer. The slice is loaded into L1 once at
// startup; the per-tile loop indexes by element rather than computing
// `first_tile_id + t`.
//
// Per screen tile id (drawn from the per-core tile-ID list):
//   1. wait for 3 tiles in cb_color_out (compute pushes R, G, B)
//   2. async-write tile 0 (R) -> DRAM page 3*screen_tile + 0
//      async-write tile 1 (G) -> DRAM page 3*screen_tile + 1
//      async-write tile 2 (B) -> DRAM page 3*screen_tile + 2
//   3. barrier + pop 3 tiles
//
// Runtime args layout:
//   0: out_addr        (DRAM base of the rgb output buffer; one 32x32 bf16
//                       tile per channel, interleaved as R,G,B,R,G,B,...)
//   1: tile_ids_addr   (DRAM base of concatenated per-core tile-ID list,
//                       uint32 elems, page size 64B)
//   2: tile_ids_start  (uint32 element offset of this core's slice)
//   3: tile_ids_count  (number of tile IDs this core handles)
//
// Compile-time args: 2 TensorAccessorArgs in order
//   out, tile_ids.

constexpr uint32_t MAX_TILE_IDS_PER_CORE = 64;

void kernel_main() {
    uint32_t out_addr        = get_arg_val<uint32_t>(0);
    uint32_t tile_ids_addr   = get_arg_val<uint32_t>(1);
    uint32_t tile_ids_start  = get_arg_val<uint32_t>(2);
    uint32_t tile_ids_count  = get_arg_val<uint32_t>(3);

    constexpr uint32_t CB_COLOR_OUT = 16;
    const uint32_t tile_bytes = get_tile_size(CB_COLOR_OUT);
    constexpr uint32_t tile_ids_page_bytes = 64;

    constexpr auto out_args      = TensorAccessorArgs<0>();
    constexpr auto tile_ids_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    const auto out          = TensorAccessor(out_args,      out_addr,      tile_bytes);
    const auto tile_ids_acc = TensorAccessor(tile_ids_args, tile_ids_addr, tile_ids_page_bytes);

    if (tile_ids_count == 0) {
        return;
    }

    // Read per-core tile-ID slice into L1. Use a one-page L1 scratch slot
    // (64 bytes) that we read each page into; no other CB activity happens
    // before we start consuming CB_COLOR_OUT, so we can safely use the
    // CB_COLOR_OUT write pointer region as scratch (it isn't in use yet).
    uint32_t scratch_addr = get_write_ptr(CB_COLOR_OUT);
    auto scratch_ptr = reinterpret_cast<volatile uint32_t*>(scratch_addr);

    uint32_t tile_ids[MAX_TILE_IDS_PER_CORE];
    {
        const uint32_t ids_per_page = tile_ids_page_bytes / 4;  // 16
        uint32_t page_idx = tile_ids_start / ids_per_page;
        uint32_t in_page  = tile_ids_start % ids_per_page;
        uint32_t remaining = tile_ids_count;
        uint32_t out_idx = 0;
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
        uint32_t screen_tile = tile_ids[t];

        cb_wait_front(CB_COLOR_OUT, 3);
        uint32_t read_ptr = get_read_ptr(CB_COLOR_OUT);
        for (uint32_t ch = 0; ch < 3; ch++) {
            uint32_t out_tile_id = 3 * screen_tile + ch;
            noc_async_write_tile(out_tile_id, out, read_ptr);
            read_ptr += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(CB_COLOR_OUT, 3);
    }
}
