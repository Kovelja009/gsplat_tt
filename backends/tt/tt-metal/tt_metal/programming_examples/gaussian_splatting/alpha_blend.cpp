// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host driver for the gaussian_splatting alpha-blend kernel.
//
// CLI signatures:
//   metal_example_gaussian_splatting packs.npy offsets.npy px.npy py.npy output.npy [H] [W]
//   metal_example_gaussian_splatting --daemon
//
// Single-shot mode: loads four .npy fixtures, opens the device, JIT-compiles
// kernels, runs once, writes the output, exits. Used by tests and benchmarks.
//
// Daemon mode: opens the device + JIT-compiles kernels once, then reads
// "FRAME H W packs.npy offsets.npy px.npy py.npy out.npy" lines from stdin
// in a loop, processes each frame (per-frame DRAM realloc + new runtime
// args), prints "OK <ms>" with kernel-only elapsed time, and exits on
// EOF or "QUIT". Used by the interactive viewer to keep the ~3s device
// init + JIT cost off the per-frame path.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "tt-metalium/base_types.hpp"
#include "tt-metalium/kernel_types.hpp"

#include "alpha_blend_host.h"

using namespace tt;
using namespace tt::tt_metal;
using namespace gsplat;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

// ---------------------------------------------------------------------------
// .npy I/O helpers
// ---------------------------------------------------------------------------

static std::vector<float> load_npy_f32(const std::string& path, std::vector<size_t>& shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }
    char magic[6];
    f.read(magic, 6);
    uint8_t major, minor;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);
    uint16_t header_len;
    f.read(reinterpret_cast<char*>(&header_len), 2);
    std::string header(header_len, ' ');
    f.read(header.data(), header_len);
    auto start = header.find('(') + 1;
    auto end = header.find(')');
    std::string shape_str = header.substr(start, end - start);
    shape.clear();
    size_t pos = 0;
    while (pos < shape_str.size()) {
        size_t comma = shape_str.find(',', pos);
        if (comma == std::string::npos) {
            comma = shape_str.size();
        }
        std::string num = shape_str.substr(pos, comma - pos);
        while (!num.empty() && (num.front() == ' ' || num.front() == '\t')) {
            num.erase(0, 1);
        }
        while (!num.empty() && (num.back() == ' ' || num.back() == '\t')) {
            num.pop_back();
        }
        if (!num.empty()) {
            shape.push_back(std::stoul(num));
        }
        pos = comma + 1;
    }
    size_t n = 1;
    for (auto d : shape) {
        n *= d;
    }
    std::vector<float> data(n);
    f.read(reinterpret_cast<char*>(data.data()), n * sizeof(float));
    return data;
}

static void save_npy_f32(const std::string& path, const std::vector<float>& data, const std::vector<size_t>& shape) {
    std::ofstream f(path, std::ios::binary);
    f.write("\x93NUMPY", 6);
    uint8_t major = 1, minor = 0;
    f.write(reinterpret_cast<char*>(&major), 1);
    f.write(reinterpret_cast<char*>(&minor), 1);
    std::string shape_str = "(";
    for (size_t i = 0; i < shape.size(); i++) {
        shape_str += std::to_string(shape[i]);
        if (i + 1 < shape.size() || shape.size() == 1) {
            shape_str += ", ";
        }
    }
    shape_str += ")";
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape_str + ", }";
    while ((10 + header.size() + 1) % 64 != 0) {
        header += ' ';
    }
    header += '\n';
    uint16_t header_len = header.size();
    f.write(reinterpret_cast<char*>(&header_len), 2);
    f.write(header.data(), header.size());
    f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
}

static std::vector<uint16_t> fp32_tile_to_bf16(const float* src) {
    std::vector<uint16_t> dst(TILE_H * TILE_W);
    for (size_t i = 0; i < TILE_H * TILE_W; i++) {
        uint32_t u;
        std::memcpy(&u, &src[i], 4);
        dst[i] = static_cast<uint16_t>(u >> 16);
    }
    return dst;
}

static std::vector<float> bf16_tile_to_fp32(const uint16_t* src) {
    std::vector<float> dst(TILE_H * TILE_W);
    for (size_t i = 0; i < TILE_H * TILE_W; i++) {
        uint32_t u = static_cast<uint32_t>(src[i]) << 16;
        std::memcpy(&dst[i], &u, 4);
    }
    return dst;
}

// ---------------------------------------------------------------------------
// Device + program reusable context
// ---------------------------------------------------------------------------

struct DeviceContext {
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    distributed::MeshCommandQueue* cq = nullptr;
    distributed::MeshWorkload workload;
    KernelHandle reader{};
    KernelHandle compute{};
    KernelHandle writer{};
    // Full compute grid, queried from the device (8x8=64 on Wormhole N150,
    // 13x10=130 on Blackhole P150 unharvested).
    // CBs and kernels are allocated on every core in this range at init;
    // per-frame, split_work_to_cores carves [0, num_tiles) into a contiguous
    // slice per core via SetRuntimeArgs.
    CoreCoord grid{0, 0};
    CoreRangeSet all_cores;
    // Shared-memory zero-copy handoff (option B). Mapped once via the MMAP
    // command; MFRAME frames read inputs from / write output to this region
    // directly, bypassing the .npy serialize/parse + fp32<->bf16 round-trips
    // of the FRAME path. nullptr until MMAP succeeds.
    void* shm_base = nullptr;
    size_t shm_cap = 0;
    // Resident px/py grids (option C). At a fixed resolution the per-pixel
    // coordinate tiles never change, so they are uploaded once via SETGRID
    // and reused by every MFRAME instead of being re-encoded/re-uploaded each
    // frame. resident_num_tiles is the grid they are valid for (0 = unset).
    std::shared_ptr<distributed::MeshBuffer> px_resident;
    std::shared_ptr<distributed::MeshBuffer> py_resident;
    uint32_t resident_num_tiles = 0;
};

// Build a Program with all CBs allocated and the 3 kernels compiled.
// TensorAccessorArgs for DRAM-interleaved buffers only encode (IsDram,
// aligned_page_size) at compile time; both are buffer-instance-independent
// for our use, so we can reuse this Program across frames whose inputs have
// different sizes — only DRAM addresses + num_tiles change per frame and are
// passed via SetRuntimeArgs.
//
// CBs and kernels are created on the full compute grid (`ctx.all_cores`).
// Each core gets its own independent copy of every CB (state, scratch, etc.).
// Per-frame work distribution happens in process_frame() via
// split_work_to_cores + SetRuntimeArgs.
static void build_program_and_workload(DeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;

    auto cb_tile = [&](uint32_t id, uint32_t depth) {
        CircularBufferConfig c(depth * TILE_BYTES_BF16, {{id, DataFormat::Float16_b}});
        c.set_page_size(id, TILE_BYTES_BF16);
        CreateCircularBuffer(program, cores, c);
    };
    auto cb_small = [&](uint32_t id, uint32_t page_bytes, uint32_t depth, DataFormat fmt) {
        CircularBufferConfig c(depth * page_bytes, {{id, fmt}});
        c.set_page_size(id, page_bytes);
        CreateCircularBuffer(program, cores, c);
    };

    cb_tile(CB_PX, 2);
    cb_tile(CB_PY, 2);
    cb_small(CB_SCALARS, SCALAR_PACK_PAGE_BYTES, 4, DataFormat::Float32);
    cb_small(CB_TILE_META, META_PAGE_BYTES, 2, DataFormat::UInt32);
    // Depth must be a multiple of 3 (the per-tile batch size) so no
    // single push-of-3 ever straddles the CB wrap. Picking 6 keeps two
    // batches in flight (parity with the previous double-buffering depth).
    cb_tile(CB_COLOR_OUT, 6);

    cb_tile(CB_DX, 2);
    cb_tile(CB_DY, 2);
    cb_tile(CB_DX2, 2);
    cb_tile(CB_DY2, 2);
    cb_tile(CB_DXDY, 2);
    {
        CircularBufferConfig c(3 * TILE_BYTES_BF16, {{CB_Q, DataFormat::Float16_b}});
        c.set_page_size(CB_Q, TILE_BYTES_BF16);
        CreateCircularBuffer(program, cores, c);
    }
    cb_tile(CB_POWER, 2);
    cb_tile(CB_ALPHA, 2);

    cb_tile(CB_CONTRIB, 1);
    cb_tile(CB_ONE_MINUS_ALPHA, 1);
    cb_tile(CB_T_TMP, 1);

    cb_tile(CB_COLOR_R_STATE, 1);
    cb_tile(CB_COLOR_G_STATE, 1);
    cb_tile(CB_COLOR_B_STATE, 1);
    cb_tile(CB_T_STATE, 1);
    cb_tile(CB_SAT_MASK, 1);

    cb_tile(CB_CONST_ZERO, 1);
    cb_tile(CB_CONST_099, 1);
    // CB_CONST_NEG88 (index 11) is reserved but unused now that the kernel
    // uses exp_tile<approx=true>, which clamps negative inputs internally.
    // Slot kept reserved to avoid renumbering downstream CBs.

    // Reader: 5 DRAM-interleaved TensorAccessorArgs for
    // packs/offsets/px/py/tile_ids. For non-sharded interleaved buffers, the
    // compile-time args reduce to (IsDram flag, aligned_page_size). Page sizes
    // are compile-time constants, so this is independent of any specific
    // buffer instance.
    std::vector<uint32_t> reader_ct;
    TensorAccessorArgs::create_dram_interleaved().append_to(reader_ct);
    TensorAccessorArgs::create_dram_interleaved().append_to(reader_ct);
    TensorAccessorArgs::create_dram_interleaved().append_to(reader_ct);
    TensorAccessorArgs::create_dram_interleaved().append_to(reader_ct);
    TensorAccessorArgs::create_dram_interleaved().append_to(reader_ct);
    ctx.reader = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "gaussian_splatting/kernels/dataflow/reader_alpha_blend.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct,
        });

    ctx.compute = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "gaussian_splatting/kernels/compute/alpha_blend_compute.cpp",
        cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi3,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false,
        });

    // Writer: 2 TensorAccessorArgs for out + tile_ids.
    std::vector<uint32_t> writer_ct;
    TensorAccessorArgs::create_dram_interleaved().append_to(writer_ct);
    TensorAccessorArgs::create_dram_interleaved().append_to(writer_ct);
    ctx.writer = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "gaussian_splatting/kernels/dataflow/writer_alpha_blend.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct,
        });

    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(ctx.mesh_device->shape());
    ctx.workload.add_program(device_range, std::move(program));
}

static DeviceContext init_device_context() {
    DeviceContext ctx;
    constexpr int device_id = 0;
    ctx.mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    ctx.cq = &ctx.mesh_device->mesh_command_queue();
    ctx.grid = ctx.mesh_device->compute_with_storage_grid_size();
    ctx.all_cores = CoreRangeSet(CoreRange({0, 0}, {ctx.grid.x - 1, ctx.grid.y - 1}));
    std::cout << "compute_with_storage_grid_size = " << ctx.grid.x << "x" << ctx.grid.y
              << " (" << (ctx.grid.x * ctx.grid.y) << " cores)" << std::endl;
    build_program_and_workload(ctx);
    return ctx;
}

// ---------------------------------------------------------------------------
// Per-frame work
// ---------------------------------------------------------------------------

struct FrameInputs {
    std::string packs_path;
    std::string offsets_path;
    std::string px_path;
    std::string py_path;
    std::string out_path;
    uint32_t image_h;
    uint32_t image_w;
};

// Result of LPT load balancing for one frame.
struct TileAssignment {
    // Concatenated per-core tile ID slices, padded up to a multiple of
    // TILE_IDS_PAGE_BYTES so the DRAM write matches the buffer page size.
    std::vector<uint32_t> tile_id_buffer_padded;
    std::vector<uint32_t> per_core_offset;  // size num_cores
    std::vector<uint32_t> per_core_count;   // size num_cores
    size_t tile_id_buffer_bytes_padded = 0;
};

constexpr size_t TILE_IDS_PAGE_BYTES = 64;

// LPT (Longest Processing Time first): sort tiles descending by Gaussian
// count, then greedily assign each tile to the currently least-loaded core.
// 4/3-approximation of the optimal makespan; usually within a few percent of
// perfect on our workloads. Returns a per-core list of tile IDs.
//
// Empty tiles (cost == 0) are filtered out and never reach the kernel. This
// is correctness-critical, not just a perf optimization: dispatching the full
// CB pump (TILE_META + PX + PY + state CBs + COLOR_OUT) for thousands of
// empty tiles per frame triggers a producer/consumer CB deadlock at >= 16
// active cores. The host pre-zeros the output buffer in process_frame() so
// skipped tile slots show as black instead of stale DRAM content.
static std::vector<std::vector<uint32_t>> compute_lpt_assignment(
    const std::vector<float>& offsets_f32, uint32_t num_tiles, uint32_t num_cores) {
    std::vector<std::pair<uint32_t, uint32_t>> cost_id;
    cost_id.reserve(num_tiles);
    for (uint32_t t = 0; t < num_tiles; t++) {
        const uint32_t cost = static_cast<uint32_t>(offsets_f32[t + 1] - offsets_f32[t]);
        if (cost > 0) {
            cost_id.emplace_back(cost, t);
        }
    }
    std::sort(cost_id.begin(), cost_id.end(), std::greater<>());

    std::vector<std::vector<uint32_t>> per_core_tile_ids(num_cores);
    std::vector<uint64_t> core_load(num_cores, 0);
    for (const auto& [cost, id] : cost_id) {
        const auto min_it = std::min_element(core_load.begin(), core_load.end());
        const uint32_t c = static_cast<uint32_t>(std::distance(core_load.begin(), min_it));
        per_core_tile_ids[c].push_back(id);
        core_load[c] += cost;
    }
    return per_core_tile_ids;
}

// Concatenate per-core tile-ID lists into one buffer with per-core (offset,
// count) bookkeeping, padded so DRAM write covers a whole multiple of pages.
// (tt-metal asserts size % page_size == 0; std::max alone only guarantees
// >= one page, not multiple-of-page.)
static TileAssignment build_tile_assignment(
    const std::vector<float>& offsets_f32, uint32_t num_tiles, uint32_t num_cores) {
    auto per_core = compute_lpt_assignment(offsets_f32, num_tiles, num_cores);

    TileAssignment a;
    a.per_core_offset.assign(num_cores, 0);
    a.per_core_count.assign(num_cores, 0);

    std::vector<uint32_t> flat;
    flat.reserve(num_tiles);
    for (uint32_t c = 0; c < num_cores; c++) {
        a.per_core_offset[c] = static_cast<uint32_t>(flat.size());
        a.per_core_count[c]  = static_cast<uint32_t>(per_core[c].size());
        flat.insert(flat.end(), per_core[c].begin(), per_core[c].end());
    }

    const size_t bytes_payload = flat.size() * sizeof(uint32_t);
    const size_t bytes_min = std::max<size_t>(bytes_payload, TILE_IDS_PAGE_BYTES);
    a.tile_id_buffer_bytes_padded =
        ((bytes_min + TILE_IDS_PAGE_BYTES - 1) / TILE_IDS_PAGE_BYTES) * TILE_IDS_PAGE_BYTES;
    a.tile_id_buffer_padded.assign(a.tile_id_buffer_bytes_padded / sizeof(uint32_t), 0);
    std::copy(flat.begin(), flat.end(), a.tile_id_buffer_padded.begin());
    return a;
}

// Pack N x 9 fp32 attribute rows into 64-byte pages
// (9 fp32 = 36 bytes payload, 28 bytes zero-padded per row).
static std::vector<uint32_t> encode_attribute_packs(
    const std::vector<float>& packs_f32, uint32_t total_entries) {
    std::vector<uint32_t> packs_payload(
        (static_cast<size_t>(total_entries) * SCALAR_PACK_PAGE_BYTES) / 4, 0);
    constexpr size_t row_payload_bytes = 9 * sizeof(float);
    for (uint32_t e = 0; e < total_entries; e++) {
        std::memcpy(
            reinterpret_cast<uint8_t*>(packs_payload.data())
                + static_cast<size_t>(e) * SCALAR_PACK_PAGE_BYTES,
            &packs_f32[e * 9],
            row_payload_bytes);
    }
    return packs_payload;
}

// Encode (num_tiles, 32, 32) fp32 input as bf16 tile-major bytes.
static std::vector<uint16_t> encode_tiles_to_bf16(
    const std::vector<float>& f32, uint32_t num_tiles) {
    std::vector<uint16_t> bf16(static_cast<size_t>(num_tiles) * TILE_H * TILE_W);
    for (uint32_t t = 0; t < num_tiles; t++) {
        auto tile = fp32_tile_to_bf16(&f32[t * TILE_H * TILE_W]);
        std::memcpy(&bf16[t * TILE_H * TILE_W], tile.data(), TILE_BYTES_BF16);
    }
    return bf16;
}

// Convert tile-major (num_tiles, 3, 32, 32) bf16 result into a row-major
// (image_h, image_w, 3) fp32 image, cropping to the requested image dims.
static std::vector<float> tiles_to_image(
    const std::vector<uint16_t>& result_bf16,
    uint32_t num_tiles,
    uint32_t tiles_x,
    uint32_t image_h,
    uint32_t image_w) {
    std::vector<float> img(static_cast<size_t>(image_h) * image_w * 3, 0.0f);
    for (uint32_t t = 0; t < num_tiles; t++) {
        const uint32_t ty = t / tiles_x;
        const uint32_t tx = t % tiles_x;
        for (uint32_t ch = 0; ch < 3; ch++) {
            const auto fp = bf16_tile_to_fp32(&result_bf16[(3 * t + ch) * TILE_H * TILE_W]);
            for (uint32_t i = 0; i < TILE_H; i++) {
                for (uint32_t j = 0; j < TILE_W; j++) {
                    const uint32_t y = ty * TILE_H + i;
                    const uint32_t x = tx * TILE_W + j;
                    if (y < image_h && x < image_w) {
                        img[(static_cast<size_t>(y) * image_w + x) * 3 + ch] = fp[i * TILE_W + j];
                    }
                }
            }
        }
    }
    return img;
}

struct FrameDramBuffers {
    std::shared_ptr<distributed::MeshBuffer> packs;
    std::shared_ptr<distributed::MeshBuffer> offsets;
    std::shared_ptr<distributed::MeshBuffer> px;
    std::shared_ptr<distributed::MeshBuffer> py;
    std::shared_ptr<distributed::MeshBuffer> output;
    std::shared_ptr<distributed::MeshBuffer> tile_ids;
};

// Allocate the 6 DRAM buffers a frame needs. Sizes are derived from the
// scene's total_entries + tile count + the LPT-balanced tile-id list.
// All buffers are RAII via shared_ptr; they free on scope exit.
static std::shared_ptr<distributed::MeshBuffer> make_dram_buffer(
    DeviceContext& ctx, size_t bytes, size_t page_bytes) {
    distributed::ReplicatedBufferConfig rc{.size = bytes};
    distributed::DeviceLocalBufferConfig lc{
        .page_size = page_bytes, .buffer_type = BufferType::DRAM};
    return distributed::MeshBuffer::create(rc, lc, ctx.mesh_device.get());
}

// Allocate the DRAM buffers a frame needs. When alloc_pxpy is false the px/py
// buffers are left null (the MFRAME path fills them from resident buffers that
// persist across frames — see the SETGRID command), so they are neither
// reallocated nor re-uploaded per frame.
static FrameDramBuffers allocate_frame_buffers(
    DeviceContext& ctx,
    uint32_t total_entries,
    uint32_t num_tiles,
    size_t offsets_count,
    size_t tile_ids_bytes,
    bool alloc_pxpy = true) {
    FrameDramBuffers b;
    b.packs    = make_dram_buffer(ctx, static_cast<size_t>(total_entries) * SCALAR_PACK_PAGE_BYTES, SCALAR_PACK_PAGE_BYTES);
    b.offsets  = make_dram_buffer(ctx, offsets_count * sizeof(uint32_t), sizeof(uint32_t));
    if (alloc_pxpy) {
        b.px   = make_dram_buffer(ctx, static_cast<size_t>(num_tiles) * TILE_BYTES_BF16, TILE_BYTES_BF16);
        b.py   = make_dram_buffer(ctx, static_cast<size_t>(num_tiles) * TILE_BYTES_BF16, TILE_BYTES_BF16);
    }
    b.output   = make_dram_buffer(ctx, static_cast<size_t>(num_tiles) * 3 * TILE_BYTES_BF16, TILE_BYTES_BF16);
    b.tile_ids = make_dram_buffer(ctx, tile_ids_bytes, TILE_IDS_PAGE_BYTES);
    return b;
}

// Set per-core runtime args for reader/compute/writer. Each core's slice of
// the concatenated tile_id_buffer is identified by (per_core_offset[c],
// per_core_count[c]); reader/writer kernels look up their tile IDs at runtime
// via this slice.
static void set_per_core_runtime_args(
    Program& program,
    const DeviceContext& ctx,
    const FrameDramBuffers& bufs,
    const TileAssignment& assign) {
    const uint32_t packs_addr    = static_cast<uint32_t>(bufs.packs->address());
    const uint32_t offsets_addr  = static_cast<uint32_t>(bufs.offsets->address());
    const uint32_t px_addr       = static_cast<uint32_t>(bufs.px->address());
    const uint32_t py_addr       = static_cast<uint32_t>(bufs.py->address());
    const uint32_t out_addr      = static_cast<uint32_t>(bufs.output->address());
    const uint32_t tile_ids_addr = static_cast<uint32_t>(bufs.tile_ids->address());

    uint32_t core_index = 0;
    for (const auto& range : ctx.all_cores.ranges()) {
        for (auto x = range.start_coord.x; x <= range.end_coord.x; x++) {
            for (auto y = range.start_coord.y; y <= range.end_coord.y; y++) {
                CoreCoord core{x, y};
                const uint32_t start = assign.per_core_offset[core_index];
                const uint32_t count = assign.per_core_count[core_index];
                SetRuntimeArgs(program, ctx.reader, core, {
                    packs_addr, offsets_addr, px_addr, py_addr,
                    tile_ids_addr, start, count,
                });
                SetRuntimeArgs(program, ctx.compute, core, {count});
                SetRuntimeArgs(program, ctx.writer, core, {
                    out_addr, tile_ids_addr, start, count,
                });
                core_index++;
            }
        }
    }
}

// Pull the program out of the workload (it was moved in at init time) so we
// can refresh its runtime args before each frame.
static Program& get_program_for_workload(DeviceContext& ctx) {
    auto& programs = ctx.workload.get_programs();
    auto it = programs.find(distributed::MeshCoordinateRange(ctx.mesh_device->shape()));
    if (it == programs.end()) {
        throw std::runtime_error("workload missing program for device range");
    }
    return it->second;
}

// Returns the kernel-only elapsed time (EnqueueWriteBuffer start ->
// EnqueueReadBuffer end) in milliseconds.
static double process_frame(DeviceContext& ctx, const FrameInputs& f) {
    // Phase profiling (emitted to stderr as GSPLAT_TIMING; the stdout "OK"
    // protocol the host parses is unaffected). Splits the non-kernel portion
    // of the daemon round-trip into load / assign / alloc / encode / save.
    using prof_clk = std::chrono::steady_clock;
    auto prof_ms = [](prof_clk::time_point a, prof_clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const auto t_prof0 = prof_clk::now();
    const uint32_t image_h = f.image_h;
    const uint32_t image_w = f.image_w;
    const uint32_t tiles_x = (image_w + TILE_W - 1) / TILE_W;
    const uint32_t tiles_y = (image_h + TILE_H - 1) / TILE_H;
    const uint32_t num_tiles = tiles_x * tiles_y;
    const uint32_t num_cores = ctx.grid.x * ctx.grid.y;

    // 1. Load .npy fixtures.
    std::vector<size_t> packs_shape, offsets_shape, px_shape, py_shape;
    auto packs_f32   = load_npy_f32(f.packs_path,   packs_shape);
    auto offsets_f32 = load_npy_f32(f.offsets_path, offsets_shape);
    auto px_f32      = load_npy_f32(f.px_path,      px_shape);
    auto py_f32      = load_npy_f32(f.py_path,      py_shape);
    const uint32_t total_entries = static_cast<uint32_t>(packs_shape[0]);
    const auto t_load = prof_clk::now();

    // 2. LPT-balanced tile-to-core assignment.
    const TileAssignment assign = build_tile_assignment(offsets_f32, num_tiles, num_cores);
    const auto t_assign = prof_clk::now();

    // 3. Allocate per-frame DRAM buffers and prepare upload payloads.
    // (Non-const because EnqueueWrite/ReadMeshBuffer takes non-const lvalue
    // refs to shared_ptr<MeshBuffer>.)
    FrameDramBuffers bufs = allocate_frame_buffers(
        ctx, total_entries, num_tiles, offsets_f32.size(),
        assign.tile_id_buffer_bytes_padded);
    const auto t_alloc = prof_clk::now();
    auto packs_payload = encode_attribute_packs(packs_f32, total_entries);
    auto px_bf16 = encode_tiles_to_bf16(px_f32, num_tiles);
    auto py_bf16 = encode_tiles_to_bf16(py_f32, num_tiles);
    std::vector<uint32_t> offsets_u32(offsets_f32.size());
    for (size_t i = 0; i < offsets_f32.size(); i++) {
        offsets_u32[i] = static_cast<uint32_t>(offsets_f32[i]);
    }
    const auto t_encode = prof_clk::now();

    // 4. Refresh runtime args for this frame.
    Program& program = get_program_for_workload(ctx);
    set_per_core_runtime_args(program, ctx, bufs, assign);
    const auto t_args = prof_clk::now();

    // 5. Kernel timing window: DRAM upload start -> output readback end.
    const auto t_start = std::chrono::steady_clock::now();
    // Zero-fill the output buffer. compute_lpt_assignment() filters out
    // empty tiles, so their slots are never written by the writer kernel.
    // The DRAM allocator may reuse addresses across frames in daemon mode,
    // so without this fill, empty regions would show stale pixels.
    std::vector<uint16_t> output_zero(
        static_cast<size_t>(num_tiles) * 3 * TILE_H * TILE_W, 0);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, bufs.output,   output_zero);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, bufs.packs,    packs_payload);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, bufs.offsets,  offsets_u32);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, bufs.px,       px_bf16);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, bufs.py,       py_bf16);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, bufs.tile_ids, assign.tile_id_buffer_padded);
    distributed::EnqueueMeshWorkload(*ctx.cq, ctx.workload, /*blocking=*/false);
    std::vector<uint16_t> result_bf16(
        static_cast<size_t>(num_tiles) * 3 * TILE_H * TILE_W);
    distributed::EnqueueReadMeshBuffer(*ctx.cq, result_bf16, bufs.output, /*blocking=*/true);
    const auto t_end = std::chrono::steady_clock::now();
    const double kernel_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // 6. Tile-major bf16 output -> row-major fp32 image; save .npy.
    const auto img = tiles_to_image(result_bf16, num_tiles, tiles_x, image_h, image_w);
    save_npy_f32(f.out_path, img, {image_h, image_w, 3});
    const auto t_save = prof_clk::now();

    std::cerr << "GSPLAT_TIMING"
              << " load="   << prof_ms(t_prof0,  t_load)
              << " assign=" << prof_ms(t_load,   t_assign)
              << " alloc="  << prof_ms(t_assign, t_alloc)
              << " encode=" << prof_ms(t_alloc,  t_encode)
              << " args="   << prof_ms(t_encode, t_args)
              << " kernel=" << kernel_ms
              << " save="   << prof_ms(t_end,    t_save)
              << " total="  << prof_ms(t_prof0,  t_save) << std::endl;
    return kernel_ms;
}

// ---------------------------------------------------------------------------
// Shared-memory (MFRAME) frame: byte offsets into the mapped region for each
// device-ready buffer. The host lays the buffers out in shm; the daemon reads
// inputs from / writes the bf16 output to the region in place.
// ---------------------------------------------------------------------------
struct MFrameInputs {
    uint32_t image_h, image_w;
    uint32_t total_entries;
    uint32_t num_tiles;
    uint32_t offsets_count;
    size_t packs_off, offsets_off, out_off;
};

// One-shot diagnostic: upload ~approx_bytes of host data into throwaway DRAM
// buffers at a range of page sizes, reporting effective GB/s for each. The
// real packs buffer uses a 64-byte page (one pack per Gaussian) -> hundreds of
// thousands of pages; this isolates whether the upload is bound by per-page
// overhead (GB/s rises with page size) or raw bandwidth (GB/s flat). Gated by
// GSPLAT_PROBE_UPLOAD; runs once, decoupled from the kernel.
static void probe_upload_bandwidth(DeviceContext& ctx, size_t approx_bytes) {
    using clk = std::chrono::steady_clock;
    const int iters = 10;
    std::cerr << "GSPLAT_PROBE upload-bandwidth approx_bytes=" << approx_bytes
              << " iters=" << iters << std::endl;
    for (size_t page : {size_t(64), size_t(256), size_t(1024), size_t(4096), size_t(16384)}) {
        const size_t bytes = ((approx_bytes + page - 1) / page) * page;
        const size_t npages = bytes / page;
        auto buf = make_dram_buffer(ctx, bytes, page);
        std::vector<uint8_t> host(bytes, 0);
        ctx.cq->enqueue_write_mesh_buffer(buf, host.data(), /*blocking=*/true);  // warmup
        const auto t0 = clk::now();
        for (int i = 0; i < iters; i++) {
            ctx.cq->enqueue_write_mesh_buffer(buf, host.data(), /*blocking=*/false);
        }
        distributed::Finish(*ctx.cq);
        const auto t1 = clk::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        const double gbps = (static_cast<double>(bytes) / 1e9) / (ms / 1e3);
        std::cerr << "GSPLAT_PROBE page=" << page << " pages=" << npages
                  << " bytes=" << bytes << " ms=" << ms << " GBps=" << gbps << std::endl;
    }
}

// Zero-copy counterpart of process_frame(): inputs are already device-ready in
// shared memory (packs in 64-byte pages, offsets u32, px/py truncated bf16), so
// there is no .npy load and no fp32->bf16 encode. Uploads straight from the
// mapped region and reads the bf16 result back into it; the host does the
// tile-major -> image rearrange. Returns kernel-only ms (same window as
// process_frame, so the "OK <ms>" the host parses keeps its meaning).
static double process_mframe(DeviceContext& ctx, const MFrameInputs& f) {
    using prof_clk = std::chrono::steady_clock;
    auto prof_ms = [](prof_clk::time_point a, prof_clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const auto t_prof0 = prof_clk::now();

    if (ctx.shm_base == nullptr) {
        throw std::runtime_error("MFRAME received before a successful MMAP");
    }
    uint8_t* base = static_cast<uint8_t*>(ctx.shm_base);
    const uint32_t num_tiles = f.num_tiles;
    const uint32_t num_cores = ctx.grid.x * ctx.grid.y;
    const size_t out_bytes = static_cast<size_t>(num_tiles) * 3 * TILE_BYTES_BF16;
    if (f.out_off + out_bytes > ctx.shm_cap) {
        throw std::runtime_error("MFRAME buffers exceed mapped shm capacity");
    }
    if (ctx.resident_num_tiles != num_tiles || !ctx.px_resident) {
        throw std::runtime_error("MFRAME before SETGRID for this resolution");
    }

    static bool probe_done = false;
    if (!probe_done && std::getenv("GSPLAT_PROBE_UPLOAD") != nullptr) {
        probe_done = true;
        probe_upload_bandwidth(ctx, static_cast<size_t>(f.total_entries) * SCALAR_PACK_PAGE_BYTES);
    }

    // 1. LPT assignment needs per-tile Gaussian counts; offsets live as u32.
    const uint32_t* offs_u32 = reinterpret_cast<const uint32_t*>(base + f.offsets_off);
    std::vector<float> offsets_f32(f.offsets_count);
    for (uint32_t i = 0; i < f.offsets_count; i++) {
        offsets_f32[i] = static_cast<float>(offs_u32[i]);
    }
    const auto t_load = prof_clk::now();

    const TileAssignment assign = build_tile_assignment(offsets_f32, num_tiles, num_cores);
    const auto t_assign = prof_clk::now();

    FrameDramBuffers bufs = allocate_frame_buffers(
        ctx, f.total_entries, num_tiles, f.offsets_count,
        assign.tile_id_buffer_bytes_padded, /*alloc_pxpy=*/false);
    // px/py are resident (uploaded once via SETGRID); reuse their addresses.
    bufs.px = ctx.px_resident;
    bufs.py = ctx.py_resident;
    const auto t_alloc = prof_clk::now();
    const auto t_encode = t_alloc;  // no encode: shm is already device-ready

    Program& program = get_program_for_workload(ctx);
    set_per_core_runtime_args(program, ctx, bufs, assign);
    const auto t_args = prof_clk::now();

    // Kernel timing window: upload start -> readback end (matches process_frame).
    // When GSPLAT_PROFILE_KERNEL is set, Finish() barriers split the window into
    // upload / compute / readback. Barriers serialize the phases (so the summed
    // total is slightly higher than the fused fast path), but they attribute
    // where the on-device time actually goes.
    static const bool kprofile = std::getenv("GSPLAT_PROFILE_KERNEL") != nullptr;
    const auto t_start = std::chrono::steady_clock::now();
    std::vector<uint16_t> output_zero(out_bytes / sizeof(uint16_t), 0);
    // px/py are already resident on-device — not re-uploaded here (option C).
    ctx.cq->enqueue_write_mesh_buffer(bufs.output,   output_zero.data(),                   false);
    ctx.cq->enqueue_write_mesh_buffer(bufs.packs,    base + f.packs_off,                   false);
    ctx.cq->enqueue_write_mesh_buffer(bufs.offsets,  base + f.offsets_off,                 false);
    ctx.cq->enqueue_write_mesh_buffer(bufs.tile_ids, assign.tile_id_buffer_padded.data(),  false);
    if (kprofile) {
        distributed::Finish(*ctx.cq);
        const auto t_up = std::chrono::steady_clock::now();
        distributed::EnqueueMeshWorkload(*ctx.cq, ctx.workload, /*blocking=*/false);
        distributed::Finish(*ctx.cq);
        const auto t_cp = std::chrono::steady_clock::now();
        ctx.cq->enqueue_read_mesh_buffer(base + f.out_off, bufs.output, /*blocking=*/true);
        const auto t_rb = std::chrono::steady_clock::now();
        std::cerr << "GSPLAT_KPROFILE"
                  << " upload="    << prof_ms(t_start, t_up)
                  << " compute="   << prof_ms(t_up,    t_cp)
                  << " readback="  << prof_ms(t_cp,    t_rb)
                  << " entries="   << f.total_entries
                  << " num_tiles=" << num_tiles << std::endl;
    } else {
        distributed::EnqueueMeshWorkload(*ctx.cq, ctx.workload, /*blocking=*/false);
        ctx.cq->enqueue_read_mesh_buffer(base + f.out_off, bufs.output, /*blocking=*/true);
    }
    const auto t_end = std::chrono::steady_clock::now();
    const double kernel_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();
    const auto t_save = prof_clk::now();  // no daemon-side save: host reads shm

    std::cerr << "GSPLAT_TIMING(mframe)"
              << " load="   << prof_ms(t_prof0,  t_load)
              << " assign=" << prof_ms(t_load,   t_assign)
              << " alloc="  << prof_ms(t_assign, t_alloc)
              << " encode=" << prof_ms(t_alloc,  t_encode)
              << " args="   << prof_ms(t_encode, t_args)
              << " kernel=" << kernel_ms
              << " save="   << prof_ms(t_end,    t_save)
              << " total="  << prof_ms(t_prof0,  t_save) << std::endl;
    return kernel_ms;
}

// ---------------------------------------------------------------------------
// Daemon mode
// ---------------------------------------------------------------------------

static int run_daemon() {
    DeviceContext ctx;
    try {
        ctx = init_device_context();
    } catch (const std::exception& e) {
        std::cerr << "daemon init failed: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "READY" << std::endl;
    std::cout.flush();

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "QUIT" || line == "quit") {
            break;
        }
        if (line.empty()) {
            continue;
        }
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;

        // MMAP <path> <capacity>: map the host's shared region once, up front.
        if (cmd == "MMAP") {
            std::string path;
            size_t cap = 0;
            if (!(iss >> path >> cap)) {
                std::cout << "ERR malformed MMAP line" << std::endl;
                std::cout.flush();
                continue;
            }
            if (ctx.shm_base != nullptr) {
                munmap(ctx.shm_base, ctx.shm_cap);
                ctx.shm_base = nullptr;
                ctx.shm_cap = 0;
            }
            int fd = open(path.c_str(), O_RDWR);
            if (fd < 0) {
                std::cout << "ERR cannot open shm " << path << std::endl;
                std::cout.flush();
                continue;
            }
            void* p = mmap(nullptr, cap, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            close(fd);
            if (p == MAP_FAILED) {
                std::cout << "ERR mmap failed" << std::endl;
                std::cout.flush();
                continue;
            }
            ctx.shm_base = p;
            ctx.shm_cap = cap;
            std::cout << "MMAP_OK" << std::endl;
            std::cout.flush();
            continue;
        }

        // SETGRID num_tiles px_off py_off: upload the static px/py grids once
        // into resident DRAM buffers reused by subsequent MFRAMEs (option C).
        if (cmd == "SETGRID") {
            size_t num_tiles = 0, px_off = 0, py_off = 0;
            if (!(iss >> num_tiles >> px_off >> py_off)) {
                std::cout << "ERR malformed SETGRID line" << std::endl;
                std::cout.flush();
                continue;
            }
            if (ctx.shm_base == nullptr) {
                std::cout << "ERR SETGRID before MMAP" << std::endl;
                std::cout.flush();
                continue;
            }
            try {
                if (ctx.resident_num_tiles != num_tiles) {
                    ctx.px_resident = make_dram_buffer(ctx, num_tiles * TILE_BYTES_BF16, TILE_BYTES_BF16);
                    ctx.py_resident = make_dram_buffer(ctx, num_tiles * TILE_BYTES_BF16, TILE_BYTES_BF16);
                    ctx.resident_num_tiles = static_cast<uint32_t>(num_tiles);
                }
                uint8_t* base = static_cast<uint8_t*>(ctx.shm_base);
                ctx.cq->enqueue_write_mesh_buffer(ctx.px_resident, base + px_off, false);
                ctx.cq->enqueue_write_mesh_buffer(ctx.py_resident, base + py_off, /*blocking=*/true);
                std::cout << "SETGRID_OK" << std::endl;
                std::cout.flush();
            } catch (const std::exception& e) {
                std::cout << "ERR " << e.what() << std::endl;
                std::cout.flush();
            }
            continue;
        }

        // MFRAME H W total_entries num_tiles offsets_count packs_off offsets_off out_off
        if (cmd == "MFRAME") {
            MFrameInputs f;
            if (!(iss >> f.image_h >> f.image_w >> f.total_entries >> f.num_tiles
                      >> f.offsets_count >> f.packs_off >> f.offsets_off
                      >> f.out_off)) {
                std::cout << "ERR malformed MFRAME line" << std::endl;
                std::cout.flush();
                continue;
            }
            try {
                double ms = process_mframe(ctx, f);
                std::cout << "OK " << std::fixed << std::setprecision(2) << ms << std::endl;
                std::cout.flush();
            } catch (const std::exception& e) {
                std::cout << "ERR " << e.what() << std::endl;
                std::cout.flush();
            }
            continue;
        }

        // FRAME H W packs offsets px py out  (.npy path; kept for tests)
        if (cmd != "FRAME") {
            std::cout << "ERR unknown command: " << cmd << std::endl;
            std::cout.flush();
            continue;
        }
        FrameInputs f;
        if (!(iss >> f.image_h >> f.image_w >> f.packs_path >> f.offsets_path >> f.px_path >> f.py_path >> f.out_path)) {
            std::cout << "ERR malformed FRAME line" << std::endl;
            std::cout.flush();
            continue;
        }
        try {
            double ms = process_frame(ctx, f);
            std::cout << "OK " << std::fixed << std::setprecision(2) << ms << std::endl;
            std::cout.flush();
        } catch (const std::exception& e) {
            std::cout << "ERR " << e.what() << std::endl;
            std::cout.flush();
        }
    }

    if (ctx.shm_base != nullptr) {
        munmap(ctx.shm_base, ctx.shm_cap);
        ctx.shm_base = nullptr;
    }
    bool ok = true;
    if (ctx.mesh_device) {
        ok = ctx.mesh_device->close();
    }
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Single-shot mode (original CLI)
// ---------------------------------------------------------------------------

static int run_single_shot(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " packs.npy offsets.npy px.npy py.npy output.npy [H] [W]\n"
                  << "       " << argv[0] << " --daemon\n";
        return 1;
    }
    FrameInputs f;
    f.packs_path = argv[1];
    f.offsets_path = argv[2];
    f.px_path = argv[3];
    f.py_path = argv[4];
    f.out_path = argv[5];
    f.image_h = argc > 6 ? static_cast<uint32_t>(std::stoi(argv[6])) : 32;
    f.image_w = argc > 7 ? static_cast<uint32_t>(std::stoi(argv[7])) : 32;

    bool pass = true;
    DeviceContext ctx;
    try {
        ctx = init_device_context();
        process_frame(ctx, f);
        std::cout << "Wrote " << f.out_path << std::endl;
        if (ctx.mesh_device) {
            pass &= ctx.mesh_device->close();
        }
    } catch (const std::exception& e) {
        std::cerr << "Run failed with exception: " << e.what() << std::endl;
        throw;
    }
    return pass ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--daemon") {
        return run_daemon();
    }
    return run_single_shot(argc, argv);
}
