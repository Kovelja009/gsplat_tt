// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// T2.3 alpha-blend writer: drains 3 tiles per screen tile from cb_color_out
// (R, G, B in that order) and async-writes them to consecutive DRAM tile
// slots {3*tile_id + 0, 3*tile_id + 1, 3*tile_id + 2}.
//
// Per screen tile t (in [first_tile_id, first_tile_id + num_tiles)):
//   1. wait for 3 tiles in cb_color_out (compute pushes R, G, B)
//   2. async-write tile 0 (R) -> DRAM page 3*screen_tile + 0
//      async-write tile 1 (G) -> DRAM page 3*screen_tile + 1
//      async-write tile 2 (B) -> DRAM page 3*screen_tile + 2
//   3. barrier + pop 3 tiles
//
// Runtime args layout:
//   0: out_addr        (DRAM base of the rgb output buffer; one 32x32 bf16
//                       tile per channel, interleaved as R,G,B,R,G,B,...)
//   1: first_tile_id   (first screen tile this core handles)
//   2: num_tiles       (number of screen tiles this core handles)
//
// Compile-time args: 1 TensorAccessorArgs for the rgb output buffer.
//
// Mirrors the C-style dataflow API used by reader_alpha_blend.cpp (T2.2),
// rather than the experimental::Noc / experimental::CircularBuffer wrappers
// from passthrough_writer.cpp.
void kernel_main() {
    uint32_t out_addr      = get_arg_val<uint32_t>(0);
    uint32_t first_tile_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles     = get_arg_val<uint32_t>(2);

    constexpr uint32_t CB_COLOR_OUT = 16;
    const uint32_t tile_bytes = get_tile_size(CB_COLOR_OUT);

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);

    for (uint32_t t = 0; t < num_tiles; t++) {
        uint32_t screen_tile = first_tile_id + t;

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
