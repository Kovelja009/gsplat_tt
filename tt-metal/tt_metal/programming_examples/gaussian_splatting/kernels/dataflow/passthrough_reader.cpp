// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "experimental/circular_buffer.h"

// T0.1 passthrough reader: fills one 32x32 bf16 tile in the input circular
// buffer with the 4-byte pattern 0xDEADBEEF, then pushes it to the compute
// kernel. No DRAM reads -- the point is to exercise the reader -> compute ->
// writer dataflow path end-to-end.
void kernel_main() {
    constexpr uint32_t cb_id = tt::CBIndex::c_0;

    experimental::CircularBuffer cb_in(cb_id);

    // bf16 tile = 32 * 32 * 2 bytes = 2048 bytes = 512 uint32 words.
    constexpr uint32_t tile_size_u32 = 32 * 32 * 2 / 4;

    cb_in.reserve_back(1);
    volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(cb_in.get_write_ptr());
    for (uint32_t i = 0; i < tile_size_u32; i++) {
        ptr[i] = 0xDEADBEEF;
    }
    cb_in.push_back(1);
}
