// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt-metalium/base_types.hpp"
#include "tt-metalium/kernel_types.hpp"

using namespace tt;
using namespace tt::tt_metal;

// A bit of a hack to handle packaged examples but also work inside the Metalium git repo.
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main(int /*argc*/, char** /*argv*/) {
    // Task 1.5 (T0.1) passthrough smoke test for the gaussian_splatting pipeline.
    // Exercises the full reader -> compute -> writer 3-kernel dataflow pattern on
    // core (0, 0). The reader fills a single 32x32 bf16 tile in L1 with the 4-byte
    // pattern 0xDEADBEEF, the compute kernel copies the tile unchanged to the
    // output CB, and the writer DMAs it to DRAM. The host reads DRAM back and
    // verifies every uint32 word equals 0xDEADBEEF.
    bool pass = true;
    try {
        constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
        constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;  // 2048 bytes

        // 1x1 mesh on device 0.
        constexpr int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        Program program = CreateProgram();

        constexpr CoreCoord core = {0, 0};

        // Output DRAM buffer: a single tile replicated across the (unit) mesh.
        distributed::DeviceLocalBufferConfig dram_config{
            .page_size = tile_size_bytes,
            .buffer_type = BufferType::DRAM};
        distributed::ReplicatedBufferConfig buffer_config{.size = tile_size_bytes};
        auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

        // Circular buffers: c_0 (reader -> compute), c_16 (compute -> writer).
        // 2 tiles each so reader can stage the next tile while compute works on
        // the current one (not strictly needed here since we only process one
        // tile, but matches the canonical pipeline shape).
        constexpr uint32_t tiles_per_cb = 2;
        tt::CBIndex src_cb_index = tt::CBIndex::c_0;
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(
                /*total_size=*/tiles_per_cb * tile_size_bytes,
                /*data_format_spec=*/{{src_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src_cb_index, tile_size_bytes));
        tt::CBIndex dst_cb_index = tt::CBIndex::c_16;
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(
                /*total_size=*/tiles_per_cb * tile_size_bytes,
                /*data_format_spec=*/{{dst_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(dst_cb_index, tile_size_bytes));

        // Reader: no compile-time args needed (no DRAM read). Runs on RISCV_1.
        auto reader = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "gaussian_splatting/kernels/dataflow/passthrough_reader.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default});

        // Compute: identity copy tile. HiFi3 + fp32 accum per design spec section 3.
        auto compute = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "gaussian_splatting/kernels/compute/hello_compute.cpp",
            core,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi3,
                .fp32_dest_acc_en = true,
                .math_approx_mode = false});

        // Writer: needs TensorAccessor compile-time args for the DRAM buffer.
        std::vector<uint32_t> writer_compile_time_args;
        TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
        auto writer = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "gaussian_splatting/kernels/dataflow/passthrough_writer.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = writer_compile_time_args});

        SetRuntimeArgs(program, reader, core, {});
        SetRuntimeArgs(program, compute, core, {});
        SetRuntimeArgs(program, writer, core, {dst_dram_buffer->address()});

        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        distributed::Finish(cq);

        // Read back the output tile and verify every uint32 word == 0xDEADBEEF.
        std::vector<bfloat16> result_vec;
        distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, /*blocking=*/true);

        if (result_vec.size() * sizeof(bfloat16) != tile_size_bytes) {
            pass = false;
            std::cerr << "Result size mismatch: got " << result_vec.size() * sizeof(bfloat16) << " bytes, expected "
                      << tile_size_bytes << std::endl;
        } else {
            const uint32_t* as_u32 = reinterpret_cast<const uint32_t*>(result_vec.data());
            const uint32_t n_words = tile_size_bytes / sizeof(uint32_t);
            for (uint32_t i = 0; i < n_words; i++) {
                if (as_u32[i] != 0xDEADBEEF) {
                    pass = false;
                    std::cerr << "Mismatch at u32 index " << i << ": got 0x" << std::hex << as_u32[i] << std::dec
                              << std::endl;
                    break;
                }
            }
        }

        std::cout << (pass ? "Passthrough OK" : "Passthrough FAILED") << std::endl;
        pass &= mesh_device->close();
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        throw;
    }

    return pass ? 0 : 1;
}
