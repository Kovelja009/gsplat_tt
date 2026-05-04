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
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/binary_max_min.h"

// T2.8 alpha-blend compute (sub-phase 11c, full single-Gaussian math chain).
//
// Per screen tile, for each Gaussian (only 1 in v1a single-Gaussian fixture):
//   B1: dx = px - mean_x,  dy = py - mean_y
//   B2: dx2 = dx*dx,  dy2 = dy*dy,  dxdy = dx*dy
//   B3: Q = a*dx2 + 2b*dxdy + c*dy2,  power = -0.5*Q
//   C : weight = exp(min(power, 0)),  alpha = min(0.99, opacity * weight)
//   D : R = color_r * alpha,  G = color_g * alpha,  B = color_b * alpha
//
// CB allocation (matches reader_alpha_blend.cpp / writer_alpha_blend.cpp,
// alpha_blend_host.h):
//   CB 0  cb_px         input  -- one bf16 32x32 tile per screen tile
//   CB 1  cb_py         input  -- one bf16 32x32 tile per screen tile
//   CB 2  cb_scalars    input  -- one fp32 pack per Gaussian (9 scalars)
//   CB 3  cb_tile_meta  input  -- one uint32 (g_count) per screen tile
//   CB 4..14 scratch    intermediate
//   CB 16 cb_color_out  output -- three bf16 32x32 tiles (R,G,B) per screen tile
//   CB 22, 23 const     constants ZERO, 0.99 (preloaded once, never popped)
//
// Runtime args layout:
//   0: num_tiles   (number of screen tiles this core handles)
void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t CB_PX         = 0;
    constexpr uint32_t CB_PY         = 1;
    constexpr uint32_t CB_SCALARS    = 2;
    constexpr uint32_t CB_TILE_META  = 3;
    constexpr uint32_t CB_DX         = 4;
    constexpr uint32_t CB_DY         = 5;
    constexpr uint32_t CB_DX2        = 6;
    constexpr uint32_t CB_DY2        = 7;
    constexpr uint32_t CB_DXDY       = 8;
    constexpr uint32_t CB_Q          = 9;
    constexpr uint32_t CB_POWER      = 10;
    constexpr uint32_t CB_ALPHA      = 12;
    constexpr uint32_t CB_COLOR_OUT  = 16;
    constexpr uint32_t CB_CONST_ZERO = 22;
    constexpr uint32_t CB_CONST_099  = 23;

    // Foundational init: configures unpack/pack for binary FPU ops and SFPU.
    binary_op_init_common(CB_PX, CB_PY, CB_COLOR_OUT);

    // Pre-fill const CBs once (depth 1, never popped). Use SFPU fill_tile.
    fill_tile_init();
    cb_reserve_back(CB_CONST_ZERO, 1);
    tile_regs_acquire();
    fill_tile(0, 0.0f);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, CB_CONST_ZERO);
    tile_regs_release();
    cb_push_back(CB_CONST_ZERO, 1);

    cb_reserve_back(CB_CONST_099, 1);
    tile_regs_acquire();
    fill_tile(0, 0.99f);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, CB_CONST_099);
    tile_regs_release();
    cb_push_back(CB_CONST_099, 1);

    cb_wait_front(CB_CONST_ZERO, 1);
    cb_wait_front(CB_CONST_099, 1);

    for (uint32_t t = 0; t < num_tiles; t++) {
        // Drain g_count from meta CB.
        cb_wait_front(CB_TILE_META, 1);
        uint32_t g_count = ckernel::read_tile_value(CB_TILE_META, /*tile_index=*/0, /*element_offset=*/0);
        cb_pop_front(CB_TILE_META, 1);

        cb_wait_front(CB_PX, 1);
        cb_wait_front(CB_PY, 1);

        for (uint32_t g = 0; g < g_count; g++) {
            cb_wait_front(CB_SCALARS, 1);

            // Read 9 fp32 scalars (already in bit-pattern form). The unary scalar ops
            // take fp32 bit patterns directly, so we keep them as uint32.
            uint32_t mean_x_bits    = ckernel::read_tile_value(CB_SCALARS, 0, 0);
            uint32_t mean_y_bits    = ckernel::read_tile_value(CB_SCALARS, 0, 1);
            uint32_t cov_a_bits     = ckernel::read_tile_value(CB_SCALARS, 0, 2);
            uint32_t two_cov_b_bits = ckernel::read_tile_value(CB_SCALARS, 0, 3);
            uint32_t cov_c_bits     = ckernel::read_tile_value(CB_SCALARS, 0, 4);
            uint32_t color_r_bits   = ckernel::read_tile_value(CB_SCALARS, 0, 5);
            uint32_t color_g_bits   = ckernel::read_tile_value(CB_SCALARS, 0, 6);
            uint32_t color_b_bits   = ckernel::read_tile_value(CB_SCALARS, 0, 7);
            uint32_t opacity_bits   = ckernel::read_tile_value(CB_SCALARS, 0, 8);

            // ===== Stage B1: dx = px - mean_x; dy = py - mean_y =====
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_PX);
            copy_tile(CB_PX, 0, 0);
            sub_unary_tile(0, mean_x_bits);
            copy_tile_to_dst_init_short(CB_PY);
            copy_tile(CB_PY, 0, 1);
            sub_unary_tile(1, mean_y_bits);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_DX, 1);
            pack_tile(0, CB_DX);
            cb_push_back(CB_DX, 1);
            cb_reserve_back(CB_DY, 1);
            pack_tile(1, CB_DY);
            cb_push_back(CB_DY, 1);
            tile_regs_release();
            cb_wait_front(CB_DX, 1);
            cb_wait_front(CB_DY, 1);

            // ===== Stage B2a: dx2 = dx * dx =====
            tile_regs_acquire();
            mul_tiles_init(CB_DX, CB_DX);
            mul_tiles(CB_DX, CB_DX, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_DX2, 1);
            pack_tile(0, CB_DX2);
            cb_push_back(CB_DX2, 1);
            tile_regs_release();

            // ===== Stage B2b: dy2 = dy * dy =====
            tile_regs_acquire();
            mul_tiles_init(CB_DY, CB_DY);
            mul_tiles(CB_DY, CB_DY, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_DY2, 1);
            pack_tile(0, CB_DY2);
            cb_push_back(CB_DY2, 1);
            tile_regs_release();

            // ===== Stage B2c: dxdy = dx * dy =====
            tile_regs_acquire();
            mul_tiles_init(CB_DX, CB_DY);
            mul_tiles(CB_DX, CB_DY, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_DXDY, 1);
            pack_tile(0, CB_DXDY);
            cb_push_back(CB_DXDY, 1);
            tile_regs_release();

            cb_wait_front(CB_DX2, 1);
            cb_wait_front(CB_DY2, 1);
            cb_wait_front(CB_DXDY, 1);

            // ===== Stage B3a: scale dx2,dy2,dxdy by cov coefficients into CB_Q (3 tiles) =====
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_DX2);
            copy_tile(CB_DX2, 0, 0);
            mul_unary_tile(0, cov_a_bits);
            copy_tile_to_dst_init_short(CB_DY2);
            copy_tile(CB_DY2, 0, 1);
            mul_unary_tile(1, cov_c_bits);
            copy_tile_to_dst_init_short(CB_DXDY);
            copy_tile(CB_DXDY, 0, 2);
            mul_unary_tile(2, two_cov_b_bits);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_Q, 3);
            pack_tile(0, CB_Q);
            pack_tile(1, CB_Q);
            pack_tile(2, CB_Q);
            cb_push_back(CB_Q, 3);
            tile_regs_release();
            cb_wait_front(CB_Q, 3);

            // ===== Stage B3b1: power_partial = a*dx2 + c*dy2  (CB_Q[0] + CB_Q[1]) =====
            tile_regs_acquire();
            add_tiles_init(CB_Q, CB_Q);
            add_tiles(CB_Q, CB_Q, 0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_POWER, 1);
            pack_tile(0, CB_POWER);
            cb_push_back(CB_POWER, 1);
            tile_regs_release();
            cb_wait_front(CB_POWER, 1);

            // ===== Stage B3b2 + C: alpha = min(0.99, opacity * exp(min(-0.5*Q, 0))) =====
            //   add 2b*dxdy -> Q
            //   mul by -0.5 -> power
            //   min(power, 0)
            //   exp -> weight
            //   mul by opacity
            //   min(_, 0.99) -> alpha
            tile_regs_acquire();
            add_tiles_init(CB_POWER, CB_Q);
            add_tiles(CB_POWER, CB_Q, 0, 2, 0);  // dst[0] = (a*dx2 + c*dy2) + 2b*dxdy = Q

            // -0.5 * Q -> dst[0]
            constexpr uint32_t NEG_HALF_BITS = 0xBF000000u;  // fp32(-0.5)
            mul_unary_tile(0, NEG_HALF_BITS);

            // dst[1] = 0.0 (constant)
            copy_tile_to_dst_init_short(CB_CONST_ZERO);
            copy_tile(CB_CONST_ZERO, 0, 1);

            // dst[0] = min(dst[0], dst[1]) = min(power, 0)
            binary_min_tile_init();
            binary_min_tile(0, 1, 0);

            // dst[0] = exp(dst[0])
            exp_tile_init();
            exp_tile(0);

            // dst[0] *= opacity
            mul_unary_tile(0, opacity_bits);

            // dst[1] = 0.99 (constant)
            copy_tile_to_dst_init_short(CB_CONST_099);
            copy_tile(CB_CONST_099, 0, 1);

            // dst[0] = min(dst[0], dst[1]) = min(opacity*weight, 0.99)
            binary_min_tile(0, 1, 0);

            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_ALPHA, 1);
            pack_tile(0, CB_ALPHA);
            cb_push_back(CB_ALPHA, 1);
            tile_regs_release();

            // Cleanup: pop intermediates we no longer need.
            cb_pop_front(CB_POWER, 1);
            cb_pop_front(CB_Q, 3);
            cb_pop_front(CB_DX, 1);
            cb_pop_front(CB_DY, 1);
            cb_pop_front(CB_DX2, 1);
            cb_pop_front(CB_DY2, 1);
            cb_pop_front(CB_DXDY, 1);

            // ===== Stage D: R = color_r * alpha,  G = color_g * alpha,  B = color_b * alpha =====
            // (v1a simplification: assumes T=1 prior, no sat_mask, single Gaussian)
            cb_wait_front(CB_ALPHA, 1);
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_ALPHA);
            copy_tile(CB_ALPHA, 0, 0);
            mul_unary_tile(0, color_r_bits);
            copy_tile(CB_ALPHA, 0, 1);
            mul_unary_tile(1, color_g_bits);
            copy_tile(CB_ALPHA, 0, 2);
            mul_unary_tile(2, color_b_bits);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_COLOR_OUT, 3);
            pack_tile(0, CB_COLOR_OUT);
            pack_tile(1, CB_COLOR_OUT);
            pack_tile(2, CB_COLOR_OUT);
            cb_push_back(CB_COLOR_OUT, 3);
            tile_regs_release();

            cb_pop_front(CB_ALPHA, 1);
            cb_pop_front(CB_SCALARS, 1);
        }

        cb_pop_front(CB_PX, 1);
        cb_pop_front(CB_PY, 1);
    }
}
