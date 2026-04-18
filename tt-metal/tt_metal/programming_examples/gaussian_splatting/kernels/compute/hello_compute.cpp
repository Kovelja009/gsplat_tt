// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"

// T0.1 passthrough compute: identity copy from cb_in (c_0) to cb_out (c_16).
// Exercises the compute engine tile register dance (acquire / commit / wait /
// pack / release) without any math. Processes a single tile.
void kernel_main() {
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr uint32_t dst_reg = 0;

    unary_op_init_common(cb_in, cb_out);

    cb_wait_front(cb_in, 1);

    tile_regs_acquire();
    copy_tile(cb_in, /*itile=*/0, dst_reg);
    tile_regs_commit();
    tile_regs_wait();

    cb_reserve_back(cb_out, 1);
    pack_tile(dst_reg, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();

    cb_pop_front(cb_in, 1);
}
