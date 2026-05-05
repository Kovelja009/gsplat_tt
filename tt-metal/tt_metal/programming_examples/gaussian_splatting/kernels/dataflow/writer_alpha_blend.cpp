// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// Alpha-blend WRITER kernel (NCRISC, NoC1).
//
// ROLE
// ----
// The mirror of the reader on the output side. For each screen tile this
// core processed, the compute kernel pushes 3 bf16 32x32 tiles (R, G, B in
// that order) to CB_COLOR_OUT. We async-write them to consecutive DRAM
// pages at offsets `3*screen_tile + {0, 1, 2}` of the output buffer. The
// writer uses NoC1 while the reader uses NoC0 — bidirectional dual-NoC
// torus lets I/O overlap.
//
// LPT TILE ASSIGNMENT
// -------------------
// Same shape as the reader: the host writes a per-core slice of (possibly
// non-contiguous) tile IDs into a shared DRAM buffer; we cache our slice
// in an L1 stack array at startup and index by element. Sizing is the same
// (MAX_TILE_IDS_PER_CORE = 256, supporting up to 4K renders).
//
// PER-TILE WORK
// -------------
//   1. cb_wait_front(CB_COLOR_OUT, 3)   wait for compute's R/G/B push
//   2. async-write 3 channels to DRAM pages 3*screen_tile + {0, 1, 2}
//   3. cb_pop_front(CB_COLOR_OUT, 3)
//
// RUNTIME ARGS
//   0: out_addr           DRAM base of the (num_tiles, 3, 32, 32) bf16 output buffer
//   1: tile_ids_addr      DRAM base of concatenated tile-id list
//   2: tile_ids_start     this core's element offset into that list
//   3: tile_ids_count     number of tile IDs this core handles
//
// COMPILE-TIME ARGS: 2 TensorAccessorArgs in order: out, tile_ids.

// Match the reader's cap (sized for 4K renders); see reader_alpha_blend.cpp
// for the reasoning.
constexpr uint32_t MAX_TILE_IDS_PER_CORE = 256;

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

    // Main per-tile loop: drain 3 R/G/B tiles compute pushed for this screen
    // tile and async-write them to their global slots in the output buffer.
    for (uint32_t t = 0; t < tile_ids_count; t++) {
        uint32_t screen_tile = tile_ids[t];

        // Wait for compute's batch of 3 tiles (R, then G, then B) in order.
        // Note: CB_COLOR_OUT has depth 6 (multiple of 3) on the host side so
        // this 3-tile batch never straddles a CB wrap, which would break the
        // `read_ptr += tile_bytes` arithmetic below.
        cb_wait_front(CB_COLOR_OUT, 3);
        uint32_t read_ptr = get_read_ptr(CB_COLOR_OUT);
        for (uint32_t ch = 0; ch < 3; ch++) {
            // Output buffer layout: (num_tiles, 3, 32, 32) bf16. Tile-major
            // order with R/G/B interleaved per screen tile, so the global
            // page index for channel `ch` of `screen_tile` is `3*screen_tile + ch`.
            uint32_t out_tile_id = 3 * screen_tile + ch;
            noc_async_write_tile(out_tile_id, out, read_ptr);
            read_ptr += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(CB_COLOR_OUT, 3);
    }
}
