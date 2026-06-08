// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// Partial alpha-blend WRITER (BRISC, NoC0). Phase-1 of intra-tile parallelism.
//
// For each segment-job this core processed, the compute kernel pushes 4 bf16
// 32x32 tiles (R, G, B, T) to CB_COLOR_OUT. We async-write them to the partials
// buffer at pages `4*partial_slot + {0,1,2,3}`. The slot comes from the job
// row's 4th field; we cache all of this core's slots up front (reusing the
// CB_COLOR_OUT write-ptr region as scratch, safe before any compute push).
//
// RUNTIME ARGS
//   0: out_addr (partials base)  1: job_table_addr  2: job_start  3: job_count
// COMPILE-TIME ARGS: 2 TensorAccessorArgs: out (partials), job_table.

constexpr uint32_t MAX_JOBS_PER_CORE = 256;  // mirrors the reader's cap

void kernel_main() {
    uint32_t out_addr       = get_arg_val<uint32_t>(0);
    uint32_t job_table_addr = get_arg_val<uint32_t>(1);
    uint32_t job_start      = get_arg_val<uint32_t>(2);
    uint32_t job_count      = get_arg_val<uint32_t>(3);

    constexpr uint32_t CB_COLOR_OUT = 16;
    const uint32_t tile_bytes = get_tile_size(CB_COLOR_OUT);
    constexpr uint32_t job_row_bytes = 16;  // 4 u32 per job row

    constexpr auto out_args = TensorAccessorArgs<0>();
    constexpr auto job_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto out     = TensorAccessor(out_args, out_addr,        tile_bytes);
    const auto job_acc = TensorAccessor(job_args, job_table_addr,  job_row_bytes);

    if (job_count == 0) {
        return;
    }

    // Cache this core's partial_slots (job row field 3) into L1 up front, using
    // the CB_COLOR_OUT write-ptr region as scratch (no compute push has landed
    // yet, so it's free).
    uint32_t scratch_addr = get_write_ptr(CB_COLOR_OUT);
    auto scratch_ptr = reinterpret_cast<volatile uint32_t*>(scratch_addr);
    uint32_t partial_slots[MAX_JOBS_PER_CORE];
    for (uint32_t j = 0; j < job_count; j++) {
        uint64_t row_noc = get_noc_addr(job_start + j, job_acc);
        noc_async_read(row_noc, scratch_addr, job_row_bytes);
        noc_async_read_barrier();
        partial_slots[j] = scratch_ptr[3];  // [tile_id, gseg_start, gseg_count, partial_slot]
    }

    // Drain compute's 4-tile (R,G,B,T) batch per job and write to the partials
    // buffer. CB_COLOR_OUT depth is a multiple of 4 (host side) so a 4-tile
    // batch never straddles a CB wrap (keeps the read_ptr += tile_bytes valid).
    for (uint32_t j = 0; j < job_count; j++) {
        uint32_t slot = partial_slots[j];
        cb_wait_front(CB_COLOR_OUT, 4);
        uint32_t read_ptr = get_read_ptr(CB_COLOR_OUT);
        for (uint32_t ch = 0; ch < 4; ch++) {
            // partials layout: (num_jobs*4, 1024) bf16; page for channel ch of
            // this job's partial is 4*slot + ch.
            uint32_t out_tile_id = 4 * slot + ch;
            noc_async_write_tile(out_tile_id, out, read_ptr);
            read_ptr += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(CB_COLOR_OUT, 4);
    }
}
