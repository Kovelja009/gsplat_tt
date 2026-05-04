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
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/rsub.h"
#include "api/compute/binary_max_min.h"

// Task 3.1 (v1b) alpha-blend compute: full per-Gaussian loop with state CBs +
// sat_mask early-termination scheme.
//
// Per screen tile, with running color/transmittance state:
//   Init  : R_state = G_state = B_state = 0,  T_state = 1,  sat_mask = 1
//   Loop g in [0, g_count):
//     F : if g>0 && (g%16)==0, sat_mask = (T_state >= 1e-4)
//     A : read 9 fp32 scalars
//     B1: dx = px - mean_x,  dy = py - mean_y
//     B2: dx2 = dx*dx,  dy2 = dy*dy,  dxdy = dx*dy
//     B3: Q = a*dx2 + 2b*dxdy + c*dy2,  power = -0.5*Q
//     C : alpha = min(0.99, opacity * exp(min(power, 0)))
//     D1: contrib = alpha * T_state * sat_mask
//     D2: R_state += color_r * contrib (and same for G, B)
//     E : T_state = T_state * (1 - alpha) * sat_mask
//   Pack R_state, G_state, B_state to CB_COLOR_OUT (3 tiles).
//   Drain state CBs.
//
// CB allocation (matches alpha_blend_host.h):
//   CB 0..3   inputs (px, py, scalars, tile_meta)
//   CB 4..15  scratch
//   CB 16     output (color_out)
//   CB 17..21 state (R_state, G_state, B_state, T_state, sat_mask)
//   CB 22, 23 const (ZERO, 0.99) preloaded once, never popped
//
// Runtime args:
//   0: num_tiles
void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    // IMPORTANT: keep these in sync with alpha_blend_host.h CB indices.
    // (Compute kernels can't include the host-side namespace.)
    constexpr uint32_t CB_PX            = 0;
    constexpr uint32_t CB_PY            = 1;
    constexpr uint32_t CB_SCALARS       = 2;
    constexpr uint32_t CB_TILE_META     = 3;
    constexpr uint32_t CB_DX            = 4;
    constexpr uint32_t CB_DY            = 5;
    constexpr uint32_t CB_DX2           = 6;
    constexpr uint32_t CB_DY2           = 7;
    constexpr uint32_t CB_DXDY          = 8;
    constexpr uint32_t CB_Q             = 9;
    constexpr uint32_t CB_POWER         = 10;
    constexpr uint32_t CB_ALPHA         = 12;
    constexpr uint32_t CB_CONTRIB       = 13;
    constexpr uint32_t CB_ONE_MINUS_ALPHA = 14;
    constexpr uint32_t CB_T_TMP         = 15;
    constexpr uint32_t CB_COLOR_OUT     = 16;
    constexpr uint32_t CB_COLOR_R_STATE = 17;
    constexpr uint32_t CB_COLOR_G_STATE = 18;
    constexpr uint32_t CB_COLOR_B_STATE = 19;
    constexpr uint32_t CB_T_STATE       = 20;
    constexpr uint32_t CB_SAT_MASK      = 21;
    constexpr uint32_t CB_CONST_ZERO    = 22;
    constexpr uint32_t CB_CONST_099     = 23;
    constexpr uint32_t CB_CONST_NEG88   = 11;  // -88.0f, lower clamp for exp_tile input

    constexpr uint32_t NEG_HALF_BITS  = 0xBF000000u;  // fp32(-0.5)
    constexpr uint32_t ONE_F_BITS     = 0x3F800000u;  // fp32( 1.0)
    constexpr uint32_t T_THRESH_BITS  = 0x38D1B717u;  // fp32(1e-4)

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

    cb_reserve_back(CB_CONST_NEG88, 1);
    tile_regs_acquire();
    // Use -89 (slightly below the -88.04 underflow threshold of _sfpu_exp_fp32_accurate_,
    // where z=INV_LN2*x <= -127 → result=0). Clamping to -88 falls just outside the
    // underflow path and lets the polynomial overshoot at large-magnitude inputs.
    fill_tile(0, -89.0f);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, CB_CONST_NEG88);
    tile_regs_release();
    cb_push_back(CB_CONST_NEG88, 1);

    cb_wait_front(CB_CONST_ZERO, 1);
    cb_wait_front(CB_CONST_099, 1);
    cb_wait_front(CB_CONST_NEG88, 1);

    for (uint32_t t = 0; t < num_tiles; t++) {
        // ===== Initialize per-tile state CBs =====
        // R_state, G_state, B_state = 0.0
        cb_reserve_back(CB_COLOR_R_STATE, 1);
        tile_regs_acquire();
        fill_tile(0, 0.0f);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_COLOR_R_STATE);
        tile_regs_release();
        cb_push_back(CB_COLOR_R_STATE, 1);

        cb_reserve_back(CB_COLOR_G_STATE, 1);
        tile_regs_acquire();
        fill_tile(0, 0.0f);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_COLOR_G_STATE);
        tile_regs_release();
        cb_push_back(CB_COLOR_G_STATE, 1);

        cb_reserve_back(CB_COLOR_B_STATE, 1);
        tile_regs_acquire();
        fill_tile(0, 0.0f);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_COLOR_B_STATE);
        tile_regs_release();
        cb_push_back(CB_COLOR_B_STATE, 1);

        // T_state = 1.0
        cb_reserve_back(CB_T_STATE, 1);
        tile_regs_acquire();
        fill_tile(0, 1.0f);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_T_STATE);
        tile_regs_release();
        cb_push_back(CB_T_STATE, 1);

        // sat_mask = 1.0 (all pixels active)
        cb_reserve_back(CB_SAT_MASK, 1);
        tile_regs_acquire();
        fill_tile(0, 1.0f);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_SAT_MASK);
        tile_regs_release();
        cb_push_back(CB_SAT_MASK, 1);

        cb_wait_front(CB_COLOR_R_STATE, 1);
        cb_wait_front(CB_COLOR_G_STATE, 1);
        cb_wait_front(CB_COLOR_B_STATE, 1);
        cb_wait_front(CB_T_STATE, 1);
        cb_wait_front(CB_SAT_MASK, 1);

        // Drain g_count from meta CB.
        cb_wait_front(CB_TILE_META, 1);
        uint32_t g_count = ckernel::read_tile_value(CB_TILE_META, /*tile_index=*/0, /*element_offset=*/0);
        cb_pop_front(CB_TILE_META, 1);

        cb_wait_front(CB_PX, 1);
        cb_wait_front(CB_PY, 1);

        for (uint32_t g = 0; g < g_count; g++) {
            // ===== Stage F: sat_mask refresh (every 16 Gaussians, skip g=0) =====
            if (g > 0 && (g & 0xFu) == 0u) {
                tile_regs_acquire();
                copy_tile_to_dst_init_short(CB_T_STATE);
                copy_tile(CB_T_STATE, 0, 0);
                unary_ge_tile_init();
                unary_ge_tile(0, T_THRESH_BITS);
                tile_regs_commit();
                tile_regs_wait();
                // Spill: replace existing sat_mask tile.
                cb_pop_front(CB_SAT_MASK, 1);
                cb_reserve_back(CB_SAT_MASK, 1);
                pack_tile(0, CB_SAT_MASK);
                cb_push_back(CB_SAT_MASK, 1);
                tile_regs_release();
                cb_wait_front(CB_SAT_MASK, 1);
            }

            cb_wait_front(CB_SCALARS, 1);

            // ===== Stage A: Read 9 fp32 scalars (uint32 bit patterns) =====
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
            tile_regs_acquire();
            add_tiles_init(CB_POWER, CB_Q);
            add_tiles(CB_POWER, CB_Q, 0, 2, 0);  // dst[0] = (a*dx2 + c*dy2) + 2b*dxdy = Q

            // -0.5 * Q -> dst[0]
            mul_unary_tile(0, NEG_HALF_BITS);

            // dst[1] = 0.0 (constant)
            copy_tile_to_dst_init_short(CB_CONST_ZERO);
            copy_tile(CB_CONST_ZERO, 0, 1);

            // dst[0] = min(dst[0], dst[1]) = min(power, 0)
            binary_min_tile_init();
            binary_min_tile(0, 1, 0);

            // Clamp power to >= -88 to prevent exp_tile polynomial overshoot.
            // (In approx=false mode, exp produces garbage for inputs below ~-88.5.)
            copy_tile_to_dst_init_short(CB_CONST_NEG88);
            copy_tile(CB_CONST_NEG88, 0, 1);
            binary_max_tile_init();
            binary_max_tile(0, 1, 0);  // dst[0] = max(power, -88)

            // dst[0] = exp(dst[0])
            exp_tile_init();
            exp_tile(0);

            // dst[0] *= opacity
            mul_unary_tile(0, opacity_bits);

            // dst[1] = 0.99 (constant)
            copy_tile_to_dst_init_short(CB_CONST_099);
            copy_tile(CB_CONST_099, 0, 1);

            // dst[0] = min(dst[0], dst[1]) = min(opacity*weight, 0.99)
            // (Re-init after binary_max_tile_init above switched SFPU op type.)
            binary_min_tile_init();
            binary_min_tile(0, 1, 0);

            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_ALPHA, 1);
            pack_tile(0, CB_ALPHA);
            cb_push_back(CB_ALPHA, 1);
            tile_regs_release();

            // Cleanup intermediates from B/C stages.
            cb_pop_front(CB_POWER, 1);
            cb_pop_front(CB_Q, 3);
            cb_pop_front(CB_DX, 1);
            cb_pop_front(CB_DY, 1);
            cb_pop_front(CB_DX2, 1);
            cb_pop_front(CB_DY2, 1);
            cb_pop_front(CB_DXDY, 1);

            cb_wait_front(CB_ALPHA, 1);

            // ===== Stage D1: contrib = alpha * T_state * sat_mask =====
            // step 1: t_tmp = alpha * T_state
            tile_regs_acquire();
            mul_tiles_init(CB_ALPHA, CB_T_STATE);
            mul_tiles(CB_ALPHA, CB_T_STATE, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_T_TMP, 1);
            pack_tile(0, CB_T_TMP);
            cb_push_back(CB_T_TMP, 1);
            tile_regs_release();
            cb_wait_front(CB_T_TMP, 1);

            // step 2: contrib = t_tmp * sat_mask
            tile_regs_acquire();
            mul_tiles_init(CB_T_TMP, CB_SAT_MASK);
            mul_tiles(CB_T_TMP, CB_SAT_MASK, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_CONTRIB, 1);
            pack_tile(0, CB_CONTRIB);
            cb_push_back(CB_CONTRIB, 1);
            tile_regs_release();

            cb_pop_front(CB_T_TMP, 1);
            cb_wait_front(CB_CONTRIB, 1);

            // ===== Stage D2: per-channel R_state += color_c * contrib =====
            // ----- R channel -----
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_CONTRIB);
            copy_tile(CB_CONTRIB, 0, 0);
            mul_unary_tile(0, color_r_bits);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_T_TMP, 1);
            pack_tile(0, CB_T_TMP);
            cb_push_back(CB_T_TMP, 1);
            tile_regs_release();
            cb_wait_front(CB_T_TMP, 1);

            tile_regs_acquire();
            add_tiles_init(CB_COLOR_R_STATE, CB_T_TMP);
            add_tiles(CB_COLOR_R_STATE, CB_T_TMP, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_pop_front(CB_COLOR_R_STATE, 1);
            cb_reserve_back(CB_COLOR_R_STATE, 1);
            pack_tile(0, CB_COLOR_R_STATE);
            cb_push_back(CB_COLOR_R_STATE, 1);
            tile_regs_release();
            cb_wait_front(CB_COLOR_R_STATE, 1);
            cb_pop_front(CB_T_TMP, 1);

            // ----- G channel -----
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_CONTRIB);
            copy_tile(CB_CONTRIB, 0, 0);
            mul_unary_tile(0, color_g_bits);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_T_TMP, 1);
            pack_tile(0, CB_T_TMP);
            cb_push_back(CB_T_TMP, 1);
            tile_regs_release();
            cb_wait_front(CB_T_TMP, 1);

            tile_regs_acquire();
            add_tiles_init(CB_COLOR_G_STATE, CB_T_TMP);
            add_tiles(CB_COLOR_G_STATE, CB_T_TMP, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_pop_front(CB_COLOR_G_STATE, 1);
            cb_reserve_back(CB_COLOR_G_STATE, 1);
            pack_tile(0, CB_COLOR_G_STATE);
            cb_push_back(CB_COLOR_G_STATE, 1);
            tile_regs_release();
            cb_wait_front(CB_COLOR_G_STATE, 1);
            cb_pop_front(CB_T_TMP, 1);

            // ----- B channel -----
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_CONTRIB);
            copy_tile(CB_CONTRIB, 0, 0);
            mul_unary_tile(0, color_b_bits);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_T_TMP, 1);
            pack_tile(0, CB_T_TMP);
            cb_push_back(CB_T_TMP, 1);
            tile_regs_release();
            cb_wait_front(CB_T_TMP, 1);

            tile_regs_acquire();
            add_tiles_init(CB_COLOR_B_STATE, CB_T_TMP);
            add_tiles(CB_COLOR_B_STATE, CB_T_TMP, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_pop_front(CB_COLOR_B_STATE, 1);
            cb_reserve_back(CB_COLOR_B_STATE, 1);
            pack_tile(0, CB_COLOR_B_STATE);
            cb_push_back(CB_COLOR_B_STATE, 1);
            tile_regs_release();
            cb_wait_front(CB_COLOR_B_STATE, 1);
            cb_pop_front(CB_T_TMP, 1);

            cb_pop_front(CB_CONTRIB, 1);

            // ===== Stage E: T_state = T_state * (1 - alpha) * sat_mask =====
            // step 1: one_minus_alpha = 1.0 - alpha (rsub_unary_tile: param - x)
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_ALPHA);
            copy_tile(CB_ALPHA, 0, 0);
            rsub_unary_tile(0, ONE_F_BITS);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_ONE_MINUS_ALPHA, 1);
            pack_tile(0, CB_ONE_MINUS_ALPHA);
            cb_push_back(CB_ONE_MINUS_ALPHA, 1);
            tile_regs_release();
            cb_wait_front(CB_ONE_MINUS_ALPHA, 1);

            // step 2: t_tmp = T_state * (1 - alpha)
            tile_regs_acquire();
            mul_tiles_init(CB_T_STATE, CB_ONE_MINUS_ALPHA);
            mul_tiles(CB_T_STATE, CB_ONE_MINUS_ALPHA, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_T_TMP, 1);
            pack_tile(0, CB_T_TMP);
            cb_push_back(CB_T_TMP, 1);
            tile_regs_release();
            cb_pop_front(CB_ONE_MINUS_ALPHA, 1);
            cb_wait_front(CB_T_TMP, 1);

            // step 3: T_state = t_tmp * sat_mask  (spill into CB_T_STATE)
            tile_regs_acquire();
            mul_tiles_init(CB_T_TMP, CB_SAT_MASK);
            mul_tiles(CB_T_TMP, CB_SAT_MASK, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_pop_front(CB_T_STATE, 1);
            cb_reserve_back(CB_T_STATE, 1);
            pack_tile(0, CB_T_STATE);
            cb_push_back(CB_T_STATE, 1);
            tile_regs_release();
            cb_wait_front(CB_T_STATE, 1);

            cb_pop_front(CB_T_TMP, 1);
            cb_pop_front(CB_ALPHA, 1);
            cb_pop_front(CB_SCALARS, 1);
        }

        // ===== After Gaussian loop: pack R/G/B state to CB_COLOR_OUT =====
        tile_regs_acquire();
        copy_tile_to_dst_init_short(CB_COLOR_R_STATE);
        copy_tile(CB_COLOR_R_STATE, 0, 0);
        copy_tile_to_dst_init_short(CB_COLOR_G_STATE);
        copy_tile(CB_COLOR_G_STATE, 0, 1);
        copy_tile_to_dst_init_short(CB_COLOR_B_STATE);
        copy_tile(CB_COLOR_B_STATE, 0, 2);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(CB_COLOR_OUT, 3);
        pack_tile(0, CB_COLOR_OUT);
        pack_tile(1, CB_COLOR_OUT);
        pack_tile(2, CB_COLOR_OUT);
        cb_push_back(CB_COLOR_OUT, 3);
        tile_regs_release();

        // ===== Drain state CBs and per-tile inputs =====
        cb_pop_front(CB_COLOR_R_STATE, 1);
        cb_pop_front(CB_COLOR_G_STATE, 1);
        cb_pop_front(CB_COLOR_B_STATE, 1);
        cb_pop_front(CB_T_STATE, 1);
        cb_pop_front(CB_SAT_MASK, 1);
        cb_pop_front(CB_PX, 1);
        cb_pop_front(CB_PY, 1);
    }
}
