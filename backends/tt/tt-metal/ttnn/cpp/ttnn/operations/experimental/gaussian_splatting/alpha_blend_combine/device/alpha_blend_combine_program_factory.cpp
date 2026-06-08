// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Host orchestration for the combine op (phase 2 of intra-tile parallelism).
// Mirrors alpha_blend_program_factory.cpp: builds CBs + reader/compute/writer
// kernels, sets per-core runtime args (buffer addresses + this core's slice of
// the combine plan), and patches them on cache hits via override_runtime_arguments.
#include "alpha_blend_combine_program_factory.hpp"

#include <unordered_map>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_combine {

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr const char* kReaderKernel =
    "ttnn/cpp/ttnn/operations/experimental/gaussian_splatting/alpha_blend_combine/device/kernels/dataflow/"
    "reader_combine.cpp";
constexpr const char* kComputeKernel =
    "ttnn/cpp/ttnn/operations/experimental/gaussian_splatting/alpha_blend_combine/device/kernels/compute/"
    "combine.cpp";
constexpr const char* kWriterKernel =
    "ttnn/cpp/ttnn/operations/experimental/gaussian_splatting/alpha_blend_combine/device/kernels/dataflow/"
    "writer_combine.cpp";

// CB indices — must match the combine kernels.
constexpr uint32_t CB_PARTIAL  = 0;   // bf16 partial tiles (R,G,B,T per job)
constexpr uint32_t CB_META     = 1;   // one uint32 (K) per output tile
constexpr uint32_t CB_COLOR_OUT = 16; // R,G,B output tiles
constexpr uint32_t CB_C_R      = 17;  // fp32 running color accumulators
constexpr uint32_t CB_C_G      = 18;
constexpr uint32_t CB_C_B      = 19;
constexpr uint32_t CB_T_ACC    = 20;  // fp32 running transmittance

constexpr uint32_t TILE_BYTES_BF16 = 32 * 32 * 2;  // 2048
constexpr uint32_t TILE_BYTES_FP32 = 32 * 32 * 4;  // 4096
constexpr uint32_t META_PAGE_BYTES = 64;           // padded uint32 page
constexpr uint32_t PLAN_PAGE_BYTES = 16;           // 4 u32 per plan row

void validate_page_sizes(const CombineInputs& t, const ttnn::Tensor& out) {
    auto pg = [](const ttnn::Tensor& x) { return static_cast<uint32_t>(x.buffer()->page_size()); };
    TT_FATAL(pg(t.partials) == TILE_BYTES_BF16,
             "partials DRAM page size {} != expected {}", pg(t.partials), TILE_BYTES_BF16);
    TT_FATAL(pg(t.plan) == PLAN_PAGE_BYTES,
             "plan DRAM page size {} != expected {}", pg(t.plan), PLAN_PAGE_BYTES);
    TT_FATAL(pg(out) == TILE_BYTES_BF16,
             "output DRAM page size {} != expected {}", pg(out), TILE_BYTES_BF16);
}

struct PerCoreArgs {
    std::vector<uint32_t> reader, compute, writer;
};
PerCoreArgs build_core_args(
    uint32_t partials, uint32_t plan, uint32_t out, uint32_t plan_start, uint32_t plan_count) {
    return PerCoreArgs{
        {partials, plan, plan_start, plan_count},
        {plan_count},
        {out, plan, plan_start, plan_count},
    };
}

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

struct CreatedProgram {
    Program program;
    CombineProgramFactory::shared_variables_t shared_variables;
};

CreatedProgram create_at(const CombineParams& attrs, const CombineInputs& t, ttnn::Tensor& out) {
    Program program = CreateProgram();
    auto* device = t.partials.device();
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const CoreRangeSet all_cores(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));

    validate_page_sizes(t, out);

    auto cb_tile = [&](uint32_t id, uint32_t depth, DataFormat fmt, uint32_t page) {
        CircularBufferConfig c(depth * page, {{id, fmt}});
        c.set_page_size(id, page);
        CreateCircularBuffer(program, all_cores, c);
    };
    cb_tile(CB_PARTIAL, 8, DataFormat::Float16_b, TILE_BYTES_BF16);
    cb_tile(CB_META, 2, DataFormat::UInt32, META_PAGE_BYTES);
    cb_tile(CB_COLOR_OUT, 3, DataFormat::Float16_b, TILE_BYTES_BF16);
    cb_tile(CB_C_R, 1, DataFormat::Float32, TILE_BYTES_FP32);
    cb_tile(CB_C_G, 1, DataFormat::Float32, TILE_BYTES_FP32);
    cb_tile(CB_C_B, 1, DataFormat::Float32, TILE_BYTES_FP32);
    cb_tile(CB_T_ACC, 1, DataFormat::Float32, TILE_BYTES_FP32);

    std::vector<uint32_t> reader_ct;
    TensorAccessorArgs(t.partials.buffer()).append_to(reader_ct);
    TensorAccessorArgs(t.plan.buffer()).append_to(reader_ct);

    std::vector<uint32_t> writer_ct;
    TensorAccessorArgs(out.buffer()).append_to(writer_ct);
    TensorAccessorArgs(t.plan.buffer()).append_to(writer_ct);

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

    const auto cores = ordered_cores(all_cores);
    TT_FATAL(
        attrs.per_core_offset.size() == cores.size() && attrs.per_core_count.size() == cores.size(),
        "combine schedule length ({}) must equal core count ({})",
        attrs.per_core_offset.size(), cores.size());
    const uint32_t partials_addr = static_cast<uint32_t>(t.partials.buffer()->address());
    const uint32_t plan_addr = static_cast<uint32_t>(t.plan.buffer()->address());
    const uint32_t out_addr = static_cast<uint32_t>(out.buffer()->address());

    for (size_t i = 0; i < cores.size(); i++) {
        const auto a = build_core_args(partials_addr, plan_addr, out_addr,
                                       attrs.per_core_offset[i], attrs.per_core_count[i]);
        SetRuntimeArgs(program, reader, cores[i], a.reader);
        SetRuntimeArgs(program, compute, cores[i], a.compute);
        SetRuntimeArgs(program, writer, cores[i], a.writer);
    }

    return CreatedProgram{
        std::move(program),
        CombineProgramFactory::shared_variables_t{reader, compute, writer, cores}};
}

}  // namespace

CombineProgramFactory::cached_mesh_workload_t CombineProgramFactory::create_mesh_workload(
    const CombineParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const CombineInputs& tensor_args,
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

void CombineProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const CombineParams& operation_attributes,
    const CombineInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    const uint32_t partials_addr = static_cast<uint32_t>(tensor_args.partials.buffer()->address());
    const uint32_t plan_addr = static_cast<uint32_t>(tensor_args.plan.buffer()->address());
    const uint32_t out_addr = static_cast<uint32_t>(tensor_return_value.buffer()->address());

    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& svars = cached_workload.shared_variables.at(range);
        TT_FATAL(
            operation_attributes.per_core_offset.size() == svars.cores.size(),
            "combine schedule length ({}) must equal core count ({}) on cache hit",
            operation_attributes.per_core_offset.size(), svars.cores.size());
        for (size_t i = 0; i < svars.cores.size(); i++) {
            const CoreCoord& core = svars.cores[i];
            const auto a = build_core_args(partials_addr, plan_addr, out_addr,
                                           operation_attributes.per_core_offset[i],
                                           operation_attributes.per_core_count[i]);
            auto& reader_args = GetRuntimeArgs(program, svars.reader_kernel_id, core);
            for (size_t k = 0; k < a.reader.size(); k++) reader_args[k] = a.reader[k];
            auto& compute_args = GetRuntimeArgs(program, svars.compute_kernel_id, core);
            compute_args[0] = a.compute[0];
            auto& writer_args = GetRuntimeArgs(program, svars.writer_kernel_id, core);
            for (size_t k = 0; k < a.writer.size(); k++) writer_args[k] = a.writer[k];
        }
    }
}

}  // namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_combine
