// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

// T0.1 passthrough writer: waits for one tile in the output CB and writes it
// to page 0 of the destination DRAM buffer.
void kernel_main() {
    uint32_t dst_dram_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_id = tt::CBIndex::c_16;
    const uint32_t tile_size_bytes = get_tile_size(cb_id);

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, dst_dram_addr);

    experimental::Noc noc;
    experimental::CircularBuffer cb_out(cb_id);

    cb_out.wait_front(1);
    noc.async_write(cb_out, out, tile_size_bytes, {}, {.page_id = 0});
    noc.async_write_barrier();
    cb_out.pop_front(1);
}
