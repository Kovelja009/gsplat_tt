// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// Partial alpha-blend READER (NCRISC, NoC1). Phase-1 of intra-tile parallelism.
//
// Unlike the single-phase reader (which reads a per-core tile-id list and looks
// up each tile's [g_start,g_end) in offsets[]), this reader is driven by a JOB
// TABLE: each row is a contiguous depth-SEGMENT of a tile's Gaussians, so the
// gaussian range is explicit and no offsets[] lookup is needed.
//
//   job row (4 u32): [tile_id, gseg_start, gseg_count, partial_slot]
//
// PER-JOB WORK
//   1. read this core's job row j from the job table (one 16-byte page)
//   2. push gseg_count to CB_TILE_META
//   3. push px/py tiles for the job's tile_id
//   4. push gseg_count scalar packs from packs[gseg_start..) to CB_SCALARS
// (partial_slot is consumed by the writer, not here.)
//
// RUNTIME ARGS
//   0: packs_addr       1: px_addr   2: py_addr
//   3: job_table_addr   4: job_start (row offset)   5: job_count
// COMPILE-TIME ARGS: 4 TensorAccessorArgs: packs, px, py, job_table.

void kernel_main() {
    uint32_t packs_addr     = get_arg_val<uint32_t>(0);
    uint32_t px_addr        = get_arg_val<uint32_t>(1);
    uint32_t py_addr        = get_arg_val<uint32_t>(2);
    uint32_t job_table_addr = get_arg_val<uint32_t>(3);
    uint32_t job_start      = get_arg_val<uint32_t>(4);
    uint32_t job_count      = get_arg_val<uint32_t>(5);

    constexpr uint32_t CB_PX        = 0;
    constexpr uint32_t CB_PY        = 1;
    constexpr uint32_t CB_SCALARS   = 2;
    constexpr uint32_t CB_TILE_META = 3;

    const uint32_t tile_bytes = get_tile_size(CB_PX);  // 2048 (32x32 bf16)
    // CB_SCALARS is fp32 format with a 64-byte sub-tile page; do NOT use
    // get_tile_size here (it returns 4096, overflowing the CB — see single-phase reader).
    constexpr uint32_t pack_bytes_padded = 64;
    constexpr uint32_t packs_dram_page_bytes = 4096;
    constexpr uint32_t packs_per_page = packs_dram_page_bytes / pack_bytes_padded;  // 64
    constexpr uint32_t job_row_bytes = 16;  // 4 u32 per job row

    constexpr auto packs_args = TensorAccessorArgs<0>();
    constexpr auto px_args    = TensorAccessorArgs<packs_args.next_compile_time_args_offset()>();
    constexpr auto py_args    = TensorAccessorArgs<px_args.next_compile_time_args_offset()>();
    constexpr auto job_args   = TensorAccessorArgs<py_args.next_compile_time_args_offset()>();

    const auto packs_acc = TensorAccessor(packs_args, packs_addr,     packs_dram_page_bytes);
    const auto px_acc    = TensorAccessor(px_args,    px_addr,        tile_bytes);
    const auto py_acc    = TensorAccessor(py_args,    py_addr,        tile_bytes);
    const auto job_acc   = TensorAccessor(job_args,   job_table_addr, job_row_bytes);

    // No-work cores (LPT may leave some cores empty when num_jobs < num_cores).
    if (job_count == 0) {
        return;
    }

    // Scratch for the per-job row read, grabbed from the meta CB write pointer
    // (not pushed until after we extract the row's fields). META_PAGE_BYTES=64
    // holds the 16-byte row comfortably.
    uint32_t scratch_addr = get_write_ptr(CB_TILE_META);
    auto scratch_ptr = reinterpret_cast<volatile uint32_t*>(scratch_addr);

    for (uint32_t j = 0; j < job_count; j++) {
        // (1) read job row (4 u32) into scratch.
        uint64_t row_noc = get_noc_addr(job_start + j, job_acc);
        noc_async_read(row_noc, scratch_addr, job_row_bytes);
        noc_async_read_barrier();
        uint32_t tile_id = scratch_ptr[0];
        uint32_t g_start = scratch_ptr[1];
        uint32_t g_count = scratch_ptr[2];
        // scratch_ptr[3] = partial_slot (writer's concern)

        // (2) push gseg_count into CB_TILE_META (overwrites the scratch slot we
        // just read — fields already extracted).
        cb_reserve_back(CB_TILE_META, 1);
        auto meta_ptr = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_TILE_META));
        meta_ptr[0] = g_count;
        cb_push_back(CB_TILE_META, 1);

        // (3) push px/py tiles for this job's tile_id.
        cb_reserve_back(CB_PX, 1);
        noc_async_read_tile(tile_id, px_acc, get_write_ptr(CB_PX));
        cb_reserve_back(CB_PY, 1);
        noc_async_read_tile(tile_id, py_acc, get_write_ptr(CB_PY));
        noc_async_read_barrier();
        cb_push_back(CB_PX, 1);
        cb_push_back(CB_PY, 1);

        // (4) stream this segment's Gaussian packs [g_start, g_start+g_count).
        for (uint32_t g = 0; g < g_count; g++) {
            uint32_t entry_id = g_start + g;
            uint32_t pk_page = entry_id / packs_per_page;
            uint32_t pk_off  = (entry_id % packs_per_page) * pack_bytes_padded;
            cb_reserve_back(CB_SCALARS, 1);
            uint64_t pk_noc = get_noc_addr(pk_page, packs_acc) + pk_off;
            noc_async_read(pk_noc, get_write_ptr(CB_SCALARS), pack_bytes_padded);
            noc_async_read_barrier();
            cb_push_back(CB_SCALARS, 1);
        }

        // CB_TILE_META wraps on push_back; refresh scratch for the next row read.
        scratch_addr = get_write_ptr(CB_TILE_META);
        scratch_ptr  = reinterpret_cast<volatile uint32_t*>(scratch_addr);
    }
}
