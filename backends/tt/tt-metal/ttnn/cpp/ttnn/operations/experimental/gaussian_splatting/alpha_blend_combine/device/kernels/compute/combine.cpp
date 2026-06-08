// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"

// Combine COMPUTE kernel. Phase-2 of intra-tile parallelism.
//
// For each output tile, fold its K depth-segments' partials (R,G,B,T) via the
// associative Porter-Duff `over` operator (validated host-side in
// backends/tt/segments.py::combine_over):
//
//   C_acc = 0 ; T_acc = 1
//   for i in 0..K-1:                 # depth order
//       C_acc += T_acc * C_i         # per channel R,G,B
//       T_acc *= T_i
//   out = C_acc                      # T_acc discarded
//
// Accumulators C_acc (CB_C_R/G/B) and T_acc (CB_T_ACC) are fp32 and spilled to
// L1 CBs between segments (Dst is too small to hold them across the loop, same
// pattern as the single-phase compute kernel). CB_PROD is fp32 scratch for the
// T_acc*C_i product. The reader pushes 4 partial tiles (R,G,B,T) per segment.
//
// RUNTIME ARGS  0: plan_count (output tiles this core owns)

void kernel_main() {
    uint32_t plan_count = get_arg_val<uint32_t>(0);

    constexpr uint32_t CB_PARTIAL   = 0;
    constexpr uint32_t CB_META      = 1;
    constexpr uint32_t CB_PROD      = 15;
    constexpr uint32_t CB_COLOR_OUT = 16;
    constexpr uint32_t CB_C_R       = 17;
    constexpr uint32_t CB_C_G       = 18;
    constexpr uint32_t CB_C_B       = 19;
    constexpr uint32_t CB_T_ACC     = 20;

    binary_op_init_common(CB_T_ACC, CB_PARTIAL, CB_COLOR_OUT);
    fill_tile_init();

    // Accumulator-init helper: set a CB's single tile to `val`.
    auto init_acc = [&](uint32_t cb, float val) {
        cb_reserve_back(cb, 1);
        tile_regs_acquire();
        fill_tile(0, val);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb);
        tile_regs_release();
        cb_push_back(cb, 1);
        cb_wait_front(cb, 1);
    };

    // C_acc_ch += T_acc * C_i_ch  (C_i_ch = CB_PARTIAL front tile `ch`).
    auto accumulate_channel = [&](uint32_t cb_acc, uint32_t ch) {
        // prod = T_acc * C_i_ch
        tile_regs_acquire();
        mul_tiles_init(CB_T_ACC, CB_PARTIAL);
        mul_tiles(CB_T_ACC, CB_PARTIAL, 0, ch, 0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(CB_PROD, 1);
        pack_tile(0, CB_PROD);
        cb_push_back(CB_PROD, 1);
        tile_regs_release();
        cb_wait_front(CB_PROD, 1);
        // C_acc_ch = C_acc_ch + prod  (spill back to cb_acc)
        tile_regs_acquire();
        add_tiles_init(cb_acc, CB_PROD);
        add_tiles(cb_acc, CB_PROD, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        cb_pop_front(cb_acc, 1);
        cb_reserve_back(cb_acc, 1);
        pack_tile(0, cb_acc);
        cb_push_back(cb_acc, 1);
        tile_regs_release();
        cb_pop_front(CB_PROD, 1);
        cb_wait_front(cb_acc, 1);
    };

    for (uint32_t t = 0; t < plan_count; t++) {
        cb_wait_front(CB_META, 1);
        uint32_t K = ckernel::read_tile_value(CB_META, /*tile_index=*/0, /*element_offset=*/0);
        cb_pop_front(CB_META, 1);

        init_acc(CB_C_R, 0.0f);
        init_acc(CB_C_G, 0.0f);
        init_acc(CB_C_B, 0.0f);
        init_acc(CB_T_ACC, 1.0f);

        for (uint32_t i = 0; i < K; i++) {
            cb_wait_front(CB_PARTIAL, 4);  // R,G,B,T at front indices 0,1,2,3

            // Channels first, using the CURRENT T_acc.
            accumulate_channel(CB_C_R, 0);
            accumulate_channel(CB_C_G, 1);
            accumulate_channel(CB_C_B, 2);

            // Then advance transmittance: T_acc *= T_i (CB_PARTIAL tile 3).
            tile_regs_acquire();
            mul_tiles_init(CB_T_ACC, CB_PARTIAL);
            mul_tiles(CB_T_ACC, CB_PARTIAL, 0, 3, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_pop_front(CB_T_ACC, 1);
            cb_reserve_back(CB_T_ACC, 1);
            pack_tile(0, CB_T_ACC);
            cb_push_back(CB_T_ACC, 1);
            tile_regs_release();
            cb_wait_front(CB_T_ACC, 1);

            cb_pop_front(CB_PARTIAL, 4);
        }

        // Pack the merged R,G,B accumulators to the output (T_acc discarded).
        tile_regs_acquire();
        copy_tile_to_dst_init_short(CB_C_R);
        copy_tile(CB_C_R, 0, 0);
        copy_tile_to_dst_init_short(CB_C_G);
        copy_tile(CB_C_G, 0, 1);
        copy_tile_to_dst_init_short(CB_C_B);
        copy_tile(CB_C_B, 0, 2);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(CB_COLOR_OUT, 3);
        pack_tile(0, CB_COLOR_OUT);
        pack_tile(1, CB_COLOR_OUT);
        pack_tile(2, CB_COLOR_OUT);
        cb_push_back(CB_COLOR_OUT, 3);
        tile_regs_release();

        // Drain accumulators for the next output tile.
        cb_pop_front(CB_C_R, 1);
        cb_pop_front(CB_C_G, 1);
        cb_pop_front(CB_C_B, 1);
        cb_pop_front(CB_T_ACC, 1);
    }
}
