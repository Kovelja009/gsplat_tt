#pragma once
#include <cstdint>

namespace gsplat {

constexpr uint32_t TILE_H = 32;
constexpr uint32_t TILE_W = 32;
constexpr uint32_t TILE_BYTES_BF16 = TILE_H * TILE_W * 2;     // 2 KB
constexpr uint32_t SCALAR_PACK_BYTES = 9 * 4;                  // 9 fp32 scalars
constexpr uint32_t SCALAR_PACK_PAGE_BYTES = 64;                // padded for NoC alignment
constexpr uint32_t META_PAGE_BYTES = 64;                       // padded uint32 page

// CB indices
constexpr uint32_t CB_PX         = 0;
constexpr uint32_t CB_PY         = 1;
constexpr uint32_t CB_SCALARS    = 2;
constexpr uint32_t CB_TILE_META  = 3;

// Scratch CBs
constexpr uint32_t CB_DX         = 4;
constexpr uint32_t CB_DY         = 5;
constexpr uint32_t CB_DX2        = 6;
constexpr uint32_t CB_DY2        = 7;
constexpr uint32_t CB_DXDY       = 8;
constexpr uint32_t CB_Q          = 9;
constexpr uint32_t CB_POWER      = 10;
// CB 11: constant tile filled with -88.0f, used to clamp `power` from below
// before exp_tile (in approx=false mode, exp produces garbage for inputs <-88.5).
// (Slot was previously reserved as CB_WEIGHT in an earlier draft; reused here.)
constexpr uint32_t CB_CONST_NEG88 = 11;
constexpr uint32_t CB_ALPHA      = 12;
constexpr uint32_t CB_CONTRIB    = 13;
constexpr uint32_t CB_ONE_MINUS_ALPHA = 14;
constexpr uint32_t CB_T_TMP      = 15;

// Output CB
constexpr uint32_t CB_COLOR_OUT  = 16;

// State CBs
constexpr uint32_t CB_COLOR_R_STATE = 17;
constexpr uint32_t CB_COLOR_G_STATE = 18;
constexpr uint32_t CB_COLOR_B_STATE = 19;
constexpr uint32_t CB_T_STATE       = 20;
constexpr uint32_t CB_SAT_MASK      = 21;

// Constants
constexpr uint32_t CB_CONST_ZERO = 22;
constexpr uint32_t CB_CONST_099  = 23;

// Early-termination threshold
constexpr float T_SAT_THRESHOLD = 1e-4f;

}  // namespace gsplat
