// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// Combine WRITER (BRISC, NoC0). Phase-2 of intra-tile parallelism.
//
// For each output tile this core owns (a combine-plan row), the compute kernel
// pushes 3 merged bf16 tiles (R,G,B) to CB_COLOR_OUT. We async-write them to
// out[3*out_tile + {0,1,2}]. out_tile comes from the plan row's first field; we
// cache all of this core's out_tiles up front (reusing the CB_COLOR_OUT write
// ptr as scratch, safe before any compute push).
//
// RUNTIME ARGS
//   0: out_addr  1: plan_addr  2: plan_start  3: plan_count
// COMPILE-TIME ARGS: 2 TensorAccessorArgs: out, plan.

constexpr uint32_t MAX_TILES_PER_CORE = 256;  // mirrors the partial op's per-core cap

void kernel_main() {
    uint32_t out_addr    = get_arg_val<uint32_t>(0);
    uint32_t plan_addr   = get_arg_val<uint32_t>(1);
    uint32_t plan_start  = get_arg_val<uint32_t>(2);
    uint32_t plan_count  = get_arg_val<uint32_t>(3);

    constexpr uint32_t CB_COLOR_OUT = 16;
    const uint32_t tile_bytes = get_tile_size(CB_COLOR_OUT);  // 2048
    constexpr uint32_t plan_row_bytes = 16;  // 4 u32 per plan row

    constexpr auto out_args  = TensorAccessorArgs<0>();
    constexpr auto plan_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto out      = TensorAccessor(out_args,  out_addr,   tile_bytes);
    const auto plan_acc = TensorAccessor(plan_args, plan_addr,  plan_row_bytes);

    if (plan_count == 0) {
        return;
    }

    // Cache this core's out_tiles (plan row field 0) up front.
    uint32_t scratch_addr = get_write_ptr(CB_COLOR_OUT);
    auto scratch_ptr = reinterpret_cast<volatile uint32_t*>(scratch_addr);
    uint32_t out_tiles[MAX_TILES_PER_CORE];
    for (uint32_t p = 0; p < plan_count; p++) {
        uint64_t row_noc = get_noc_addr(plan_start + p, plan_acc);
        noc_async_read(row_noc, scratch_addr, plan_row_bytes);
        noc_async_read_barrier();
        out_tiles[p] = scratch_ptr[0];  // [out_tile, first_slot, K, pad]
    }

    // Drain compute's 3-tile (R,G,B) batch per output tile and write to DRAM.
    for (uint32_t p = 0; p < plan_count; p++) {
        uint32_t out_tile = out_tiles[p];
        cb_wait_front(CB_COLOR_OUT, 3);
        uint32_t read_ptr = get_read_ptr(CB_COLOR_OUT);
        for (uint32_t ch = 0; ch < 3; ch++) {
            noc_async_write_tile(3 * out_tile + ch, out, read_ptr);
            read_ptr += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(CB_COLOR_OUT, 3);
    }
}
