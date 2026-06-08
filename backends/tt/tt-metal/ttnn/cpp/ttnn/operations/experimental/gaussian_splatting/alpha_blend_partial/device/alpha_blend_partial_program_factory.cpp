// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Host orchestration for the alpha-blend op, ported from the original tt-metal
// programming example (alpha_blend_partial.cpp: build_program_and_workload +
// set_per_core_runtime_args). CB layout and kernel configs are reproduced
// verbatim so the reused reader/compute/writer kernels behave identically; the
// difference is that the program is now cached by ttnn and its per-core runtime
// args (buffer addresses + LPT (start,count) slices) are patched on cache hits
// via override_runtime_arguments.
#include "alpha_blend_partial_program_factory.hpp"

#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "alpha_blend_partial_host.h"

namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_partial {

using namespace tt;
using namespace tt::tt_metal;

namespace {

// Kernel source paths, relative to the tt-metal root (TT_METAL_HOME).
constexpr const char* kReaderKernel =
    "ttnn/cpp/ttnn/operations/experimental/gaussian_splatting/alpha_blend_partial/device/kernels/dataflow/"
    "reader_alpha_blend_partial.cpp";
constexpr const char* kComputeKernel =
    "ttnn/cpp/ttnn/operations/experimental/gaussian_splatting/alpha_blend_partial/device/kernels/compute/"
    "alpha_blend_partial_compute.cpp";
constexpr const char* kWriterKernel =
    "ttnn/cpp/ttnn/operations/experimental/gaussian_splatting/alpha_blend_partial/device/kernels/dataflow/"
    "writer_alpha_blend_partial.cpp";

// The reader/writer kernels hardcode the DRAM page size they pass to
// TensorAccessor. The host (backends/tt/backend.py) must shape the ttnn
// input/output tensors so their interleaved-DRAM page size equals these EXACTLY,
// or the kernels read/write misaligned DRAM. This is the single C++ statement of
// that geometry contract; validate_page_sizes() enforces it at dispatch so a
// mismatch fails loud instead of silently corrupting pixels / L1.
constexpr uint32_t kPacksPageBytes   = 4096;  // 64 packs x 64-byte pack (reader: packs_dram_page_bytes)
constexpr uint32_t kTilePageBytes    = 2048;  // one 32x32 bf16 tile (px/py and partials)
constexpr uint32_t kJobRowPageBytes  = 16;    // one job row = 4 u32 (reader/writer: job_table page)

void validate_page_sizes(const AlphaBlendPartialInputs& t, const ttnn::Tensor& out) {
    auto pg = [](const ttnn::Tensor& x) { return static_cast<uint32_t>(x.buffer()->page_size()); };
    TT_FATAL(pg(t.packs) == kPacksPageBytes,
             "packs DRAM page size {} != kernel-expected {} (host tensor shaping drifted)", pg(t.packs), kPacksPageBytes);
    TT_FATAL(pg(t.px) == kTilePageBytes, "px DRAM page size {} != kernel-expected {}", pg(t.px), kTilePageBytes);
    TT_FATAL(pg(t.py) == kTilePageBytes, "py DRAM page size {} != kernel-expected {}", pg(t.py), kTilePageBytes);
    TT_FATAL(pg(t.job_table) == kJobRowPageBytes,
             "job_table DRAM page size {} != kernel-expected {}", pg(t.job_table), kJobRowPageBytes);
    TT_FATAL(pg(out) == kTilePageBytes, "partials DRAM page size {} != kernel-expected {}", pg(out), kTilePageBytes);
}

// Per-core runtime args, in the exact order the kernels read them. Single source
// of the layout so create_at (SetRuntimeArgs) and override_runtime_arguments
// (patch GetRuntimeArgs) cannot drift out of sync.
struct PerCoreArgs {
    std::vector<uint32_t> reader, compute, writer;
};
PerCoreArgs build_core_args(
    uint32_t packs, uint32_t px, uint32_t py, uint32_t out, uint32_t job_table,
    uint32_t start, uint32_t count) {
    return PerCoreArgs{
        {packs, px, py, job_table, start, count},
        {count},
        {out, job_table, start, count},
    };
}

struct CreatedProgram {
    Program program;
    AlphaBlendPartialProgramFactory::shared_variables_t shared_variables;
};

// Ordered core list over the full compute grid, matching the iteration order
// used when setting per-core runtime args (range -> x -> y).
std::vector<CoreCoord> ordered_cores(const CoreRangeSet& all_cores) {
    std::vector<CoreCoord> cores;
    for (const auto& range : all_cores.ranges()) {
        for (auto x = range.start_coord.x; x <= range.end_coord.x; x++) {
            for (auto y = range.start_coord.y; y <= range.end_coord.y; y++) {
                cores.push_back(CoreCoord{x, y});
            }
        }
    }
    return cores;
}

CreatedProgram create_at(
    const AlphaBlendPartialParams& attrs, const AlphaBlendPartialInputs& t, ttnn::Tensor& out) {
    using namespace gsplat;
    Program program = CreateProgram();

    auto* device = t.packs.device();
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const CoreRangeSet all_cores(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));

    // Enforce the host<->kernel DRAM page-size contract before building anything.
    validate_page_sizes(t, out);

    // --- Circular buffers (verbatim from build_program_and_workload) ---------
    auto cb_tile = [&](uint32_t id, uint32_t depth) {
        CircularBufferConfig c(depth * TILE_BYTES_BF16, {{id, DataFormat::Float16_b}});
        c.set_page_size(id, TILE_BYTES_BF16);
        CreateCircularBuffer(program, all_cores, c);
    };
    auto cb_small = [&](uint32_t id, uint32_t page_bytes, uint32_t depth, DataFormat fmt) {
        CircularBufferConfig c(depth * page_bytes, {{id, fmt}});
        c.set_page_size(id, page_bytes);
        CreateCircularBuffer(program, all_cores, c);
    };

    cb_tile(CB_PX, 2);
    cb_tile(CB_PY, 2);
    cb_small(CB_SCALARS, SCALAR_PACK_PAGE_BYTES, 4, DataFormat::Float32);
    cb_small(CB_TILE_META, META_PAGE_BYTES, 2, DataFormat::UInt32);
    cb_tile(CB_COLOR_OUT, 8);  // depth multiple of 4 so each (R,G,B,T) batch never wraps

    cb_tile(CB_DX, 2);
    cb_tile(CB_DY, 2);
    cb_tile(CB_DX2, 2);
    cb_tile(CB_DY2, 2);
    cb_tile(CB_DXDY, 2);
    cb_tile(CB_Q, 3);  // [a·dx², c·dy², 2b·dx·dy] — depth 3 (one batch in flight)
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

    // --- Compile-time args: TensorAccessorArgs from the actual input buffers --
    // Reader reads packs/px/py/job_table; writer writes partials + reads job_table.
    std::vector<uint32_t> reader_ct;
    TensorAccessorArgs(t.packs.buffer()).append_to(reader_ct);
    TensorAccessorArgs(t.px.buffer()).append_to(reader_ct);
    TensorAccessorArgs(t.py.buffer()).append_to(reader_ct);
    TensorAccessorArgs(t.job_table.buffer()).append_to(reader_ct);

    std::vector<uint32_t> writer_ct;
    TensorAccessorArgs(out.buffer()).append_to(writer_ct);
    TensorAccessorArgs(t.job_table.buffer()).append_to(writer_ct);

    // --- Kernels (configs verbatim) ------------------------------------------
    KernelHandle reader = CreateKernel(
        program, kReaderKernel, all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct,
        });

    KernelHandle compute = CreateKernel(
        program, kComputeKernel, all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi3,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false,
        });

    KernelHandle writer = CreateKernel(
        program, kWriterKernel, all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct,
        });

    // --- Per-core runtime args (port of set_per_core_runtime_args) ------------
    const auto cores = ordered_cores(all_cores);
    TT_FATAL(
        attrs.per_core_offset.size() == cores.size() && attrs.per_core_count.size() == cores.size(),
        "LPT schedule length ({}) must equal the compute-grid core count ({})",
        attrs.per_core_offset.size(), cores.size());
    const uint32_t packs_addr = static_cast<uint32_t>(t.packs.buffer()->address());
    const uint32_t px_addr = static_cast<uint32_t>(t.px.buffer()->address());
    const uint32_t py_addr = static_cast<uint32_t>(t.py.buffer()->address());
    const uint32_t out_addr = static_cast<uint32_t>(out.buffer()->address());
    const uint32_t job_table_addr = static_cast<uint32_t>(t.job_table.buffer()->address());

    for (size_t i = 0; i < cores.size(); i++) {
        const auto a = build_core_args(packs_addr, px_addr, py_addr, out_addr, job_table_addr,
                                       attrs.per_core_offset[i], attrs.per_core_count[i]);
        SetRuntimeArgs(program, reader, cores[i], a.reader);
        SetRuntimeArgs(program, compute, cores[i], a.compute);
        SetRuntimeArgs(program, writer, cores[i], a.writer);
    }

    return CreatedProgram{
        std::move(program),
        AlphaBlendPartialProgramFactory::shared_variables_t{reader, compute, writer, cores}};
}

}  // namespace

AlphaBlendPartialProgramFactory::cached_mesh_workload_t AlphaBlendPartialProgramFactory::create_mesh_workload(
    const AlphaBlendPartialParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const AlphaBlendPartialInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& coord : tensor_coords.coords()) {
        auto result = create_at(operation_attributes, tensor_args, tensor_return_value);
        auto coord_range = ttnn::MeshCoordinateRange(coord);
        workload.add_program(coord_range, std::move(result.program));
        shared_variables.emplace(coord_range, std::move(result.shared_variables));
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

void AlphaBlendPartialProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const AlphaBlendPartialParams& operation_attributes,
    const AlphaBlendPartialInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    const uint32_t packs_addr = static_cast<uint32_t>(tensor_args.packs.buffer()->address());
    const uint32_t px_addr = static_cast<uint32_t>(tensor_args.px.buffer()->address());
    const uint32_t py_addr = static_cast<uint32_t>(tensor_args.py.buffer()->address());
    const uint32_t out_addr = static_cast<uint32_t>(tensor_return_value.buffer()->address());
    const uint32_t job_table_addr = static_cast<uint32_t>(tensor_args.job_table.buffer()->address());

    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& svars = cached_workload.shared_variables.at(range);
        TT_FATAL(
            operation_attributes.per_core_offset.size() == svars.cores.size(),
            "LPT schedule length ({}) must equal core count ({}) on cache hit",
            operation_attributes.per_core_offset.size(), svars.cores.size());
        for (size_t i = 0; i < svars.cores.size(); i++) {
            const CoreCoord& core = svars.cores[i];
            const auto a = build_core_args(packs_addr, px_addr, py_addr, out_addr, job_table_addr,
                                           operation_attributes.per_core_offset[i], operation_attributes.per_core_count[i]);
            auto& reader_args = GetRuntimeArgs(program, svars.reader_kernel_id, core);
            for (size_t k = 0; k < a.reader.size(); k++) reader_args[k] = a.reader[k];
            auto& compute_args = GetRuntimeArgs(program, svars.compute_kernel_id, core);
            compute_args[0] = a.compute[0];
            auto& writer_args = GetRuntimeArgs(program, svars.writer_kernel_id, core);
            for (size_t k = 0; k < a.writer.size(); k++) writer_args[k] = a.writer[k];
        }
    }
}

}  // namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_partial
