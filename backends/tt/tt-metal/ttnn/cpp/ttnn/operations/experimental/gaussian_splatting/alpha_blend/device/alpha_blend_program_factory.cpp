// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Host orchestration for the alpha-blend op, ported from the original tt-metal
// programming example (alpha_blend.cpp: build_program_and_workload +
// set_per_core_runtime_args). CB layout and kernel configs are reproduced
// verbatim so the reused reader/compute/writer kernels behave identically; the
// difference is that the program is now cached by ttnn and its per-core runtime
// args (buffer addresses + LPT (start,count) slices) are patched on cache hits
// via override_runtime_arguments.
#include "alpha_blend_program_factory.hpp"

#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "alpha_blend_host.h"

namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend {

using namespace tt;
using namespace tt::tt_metal;

namespace {

// Kernel source paths, relative to the tt-metal root (TT_METAL_HOME).
constexpr const char* kReaderKernel =
    "ttnn/cpp/ttnn/operations/experimental/gaussian_splatting/alpha_blend/device/kernels/dataflow/"
    "reader_alpha_blend.cpp";
constexpr const char* kComputeKernel =
    "ttnn/cpp/ttnn/operations/experimental/gaussian_splatting/alpha_blend/device/kernels/compute/"
    "alpha_blend_compute.cpp";
constexpr const char* kWriterKernel =
    "ttnn/cpp/ttnn/operations/experimental/gaussian_splatting/alpha_blend/device/kernels/dataflow/"
    "writer_alpha_blend.cpp";

struct CreatedProgram {
    Program program;
    AlphaBlendProgramFactory::shared_variables_t shared_variables;
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
    const AlphaBlendParams& attrs, const AlphaBlendInputs& t, ttnn::Tensor& out) {
    using namespace gsplat;
    Program program = CreateProgram();

    auto* device = t.packs.device();
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const CoreRangeSet all_cores(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));

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
    cb_tile(CB_COLOR_OUT, 6);

    cb_tile(CB_DX, 2);
    cb_tile(CB_DY, 2);
    cb_tile(CB_DX2, 2);
    cb_tile(CB_DY2, 2);
    cb_tile(CB_DXDY, 2);
    {
        CircularBufferConfig c(3 * TILE_BYTES_BF16, {{CB_Q, DataFormat::Float16_b}});
        c.set_page_size(CB_Q, TILE_BYTES_BF16);
        CreateCircularBuffer(program, all_cores, c);
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

    // --- Compile-time args: TensorAccessorArgs from the actual input buffers --
    // Reader reads packs/offsets/px/py/tile_ids; writer writes out + reads tile_ids.
    std::vector<uint32_t> reader_ct;
    TensorAccessorArgs(t.packs.buffer()).append_to(reader_ct);
    TensorAccessorArgs(t.offsets.buffer()).append_to(reader_ct);
    TensorAccessorArgs(t.px.buffer()).append_to(reader_ct);
    TensorAccessorArgs(t.py.buffer()).append_to(reader_ct);
    TensorAccessorArgs(t.tile_ids.buffer()).append_to(reader_ct);

    std::vector<uint32_t> writer_ct;
    TensorAccessorArgs(out.buffer()).append_to(writer_ct);
    TensorAccessorArgs(t.tile_ids.buffer()).append_to(writer_ct);

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
    const uint32_t packs_addr = static_cast<uint32_t>(t.packs.buffer()->address());
    const uint32_t offsets_addr = static_cast<uint32_t>(t.offsets.buffer()->address());
    const uint32_t px_addr = static_cast<uint32_t>(t.px.buffer()->address());
    const uint32_t py_addr = static_cast<uint32_t>(t.py.buffer()->address());
    const uint32_t out_addr = static_cast<uint32_t>(out.buffer()->address());
    const uint32_t tile_ids_addr = static_cast<uint32_t>(t.tile_ids.buffer()->address());

    for (size_t i = 0; i < cores.size(); i++) {
        const CoreCoord& core = cores[i];
        const uint32_t start = (i < attrs.per_core_offset.size()) ? attrs.per_core_offset[i] : 0;
        const uint32_t count = (i < attrs.per_core_count.size()) ? attrs.per_core_count[i] : 0;
        SetRuntimeArgs(program, reader, core,
                       {packs_addr, offsets_addr, px_addr, py_addr, tile_ids_addr, start, count});
        SetRuntimeArgs(program, compute, core, {count});
        SetRuntimeArgs(program, writer, core, {out_addr, tile_ids_addr, start, count});
    }

    return CreatedProgram{
        std::move(program),
        AlphaBlendProgramFactory::shared_variables_t{reader, compute, writer, cores}};
}

}  // namespace

AlphaBlendProgramFactory::cached_mesh_workload_t AlphaBlendProgramFactory::create_mesh_workload(
    const AlphaBlendParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const AlphaBlendInputs& tensor_args,
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

void AlphaBlendProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const AlphaBlendParams& operation_attributes,
    const AlphaBlendInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    const uint32_t packs_addr = static_cast<uint32_t>(tensor_args.packs.buffer()->address());
    const uint32_t offsets_addr = static_cast<uint32_t>(tensor_args.offsets.buffer()->address());
    const uint32_t px_addr = static_cast<uint32_t>(tensor_args.px.buffer()->address());
    const uint32_t py_addr = static_cast<uint32_t>(tensor_args.py.buffer()->address());
    const uint32_t out_addr = static_cast<uint32_t>(tensor_return_value.buffer()->address());
    const uint32_t tile_ids_addr = static_cast<uint32_t>(tensor_args.tile_ids.buffer()->address());

    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& svars = cached_workload.shared_variables.at(range);
        for (size_t i = 0; i < svars.cores.size(); i++) {
            const CoreCoord& core = svars.cores[i];
            const uint32_t start =
                (i < operation_attributes.per_core_offset.size()) ? operation_attributes.per_core_offset[i] : 0;
            const uint32_t count =
                (i < operation_attributes.per_core_count.size()) ? operation_attributes.per_core_count[i] : 0;

            auto& reader_args = GetRuntimeArgs(program, svars.reader_kernel_id, core);
            reader_args[0] = packs_addr;
            reader_args[1] = offsets_addr;
            reader_args[2] = px_addr;
            reader_args[3] = py_addr;
            reader_args[4] = tile_ids_addr;
            reader_args[5] = start;
            reader_args[6] = count;

            auto& compute_args = GetRuntimeArgs(program, svars.compute_kernel_id, core);
            compute_args[0] = count;

            auto& writer_args = GetRuntimeArgs(program, svars.writer_kernel_id, core);
            writer_args[0] = out_addr;
            writer_args[1] = tile_ids_addr;
            writer_args[2] = start;
            writer_args[3] = count;
        }
    }
}

}  // namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend
