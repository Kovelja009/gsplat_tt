// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// Combine READER (NCRISC, NoC1). Phase-2 of intra-tile parallelism.
//
// Driven by the combine PLAN: each row is (out_tile, first_slot, K). For each
// output tile this core owns, we push K to CB_META and stream its K segments'
// 4 partial tiles (R,G,B,T) from the partials buffer into CB_PARTIAL, in slot
// order (depth order). The compute kernel folds them via the `over` operator.
//
// RUNTIME ARGS
//   0: partials_addr  1: plan_addr  2: plan_start  3: plan_count
// COMPILE-TIME ARGS: 2 TensorAccessorArgs: partials, plan.

void kernel_main() {
    uint32_t partials_addr = get_arg_val<uint32_t>(0);
    uint32_t plan_addr     = get_arg_val<uint32_t>(1);
    uint32_t plan_start    = get_arg_val<uint32_t>(2);
    uint32_t plan_count    = get_arg_val<uint32_t>(3);

    constexpr uint32_t CB_PARTIAL = 0;
    constexpr uint32_t CB_META    = 1;

    const uint32_t tile_bytes = get_tile_size(CB_PARTIAL);  // 2048 (32x32 bf16)
    constexpr uint32_t plan_row_bytes = 16;  // 4 u32 per plan row

    constexpr auto part_args = TensorAccessorArgs<0>();
    constexpr auto plan_args = TensorAccessorArgs<part_args.next_compile_time_args_offset()>();
    const auto part_acc = TensorAccessor(part_args, partials_addr, tile_bytes);
    const auto plan_acc = TensorAccessor(plan_args, plan_addr,     plan_row_bytes);

    if (plan_count == 0) {
        return;
    }

    uint32_t scratch_addr = get_write_ptr(CB_META);
    auto scratch_ptr = reinterpret_cast<volatile uint32_t*>(scratch_addr);

    for (uint32_t p = 0; p < plan_count; p++) {
        // Read plan row (out_tile, first_slot, K, pad) into scratch.
        uint64_t row_noc = get_noc_addr(plan_start + p, plan_acc);
        noc_async_read(row_noc, scratch_addr, plan_row_bytes);
        noc_async_read_barrier();
        uint32_t first_slot = scratch_ptr[1];
        uint32_t K          = scratch_ptr[2];
        // scratch_ptr[0] = out_tile (writer's concern)

        // Push K to CB_META (overwrites the scratch slot — fields extracted).
        cb_reserve_back(CB_META, 1);
        auto meta_ptr = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_META));
        meta_ptr[0] = K;
        cb_push_back(CB_META, 1);

        // Stream the K segments' 4 partial tiles each (R,G,B,T), in slot order.
        for (uint32_t i = 0; i < K; i++) {
            cb_reserve_back(CB_PARTIAL, 4);
            uint32_t wptr = get_write_ptr(CB_PARTIAL);
            for (uint32_t ch = 0; ch < 4; ch++) {
                uint32_t page = (first_slot + i) * 4 + ch;
                noc_async_read_tile(page, part_acc, wptr + ch * tile_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(CB_PARTIAL, 4);
        }

        scratch_addr = get_write_ptr(CB_META);
        scratch_ptr  = reinterpret_cast<volatile uint32_t*>(scratch_addr);
    }
}
