// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Task 2.5 v1a host driver for the gaussian_splatting alpha-blend pipeline.
//
// CLI signature:
//   metal_example_gaussian_splatting packs.npy offsets.npy px.npy py.npy output.npy [H] [W]
//
// Loads four .npy fixtures, sets up DRAM buffers + circular buffers, runs the
// reader/compute/writer kernels on a single core, reads back the output, and
// writes a (H, W, 3) fp32 .npy image. The 11a sub-phase compute kernel emits a
// constant solid color (R=0.25, G=0.5, B=0.75) so a successful run is trivially
// verifiable.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt-metalium/base_types.hpp"
#include "tt-metalium/kernel_types.hpp"

#include "alpha_blend_host.h"

using namespace tt;
using namespace tt::tt_metal;
using namespace gsplat;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

// Minimal .npy reader for fp32 arrays.
static std::vector<float> load_npy_f32(const std::string& path, std::vector<size_t>& shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "cannot open " << path << std::endl;
        std::exit(1);
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

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " packs.npy offsets.npy px.npy py.npy output.npy [H] [W]\n";
        return 1;
    }
    std::string packs_path = argv[1];
    std::string offsets_path = argv[2];
    std::string px_path = argv[3];
    std::string py_path = argv[4];
    std::string out_path = argv[5];
    uint32_t image_h = argc > 6 ? static_cast<uint32_t>(std::stoi(argv[6])) : 32;
    uint32_t image_w = argc > 7 ? static_cast<uint32_t>(std::stoi(argv[7])) : 32;

    uint32_t tiles_x = (image_w + 31) / 32;
    uint32_t tiles_y = (image_h + 31) / 32;
    uint32_t num_tiles = tiles_x * tiles_y;

    std::vector<size_t> packs_shape, offsets_shape, px_shape, py_shape;
    auto packs_f32 = load_npy_f32(packs_path, packs_shape);
    auto offsets_f32 = load_npy_f32(offsets_path, offsets_shape);
    auto px_f32 = load_npy_f32(px_path, px_shape);
    auto py_f32 = load_npy_f32(py_path, py_shape);

    uint32_t total_entries = static_cast<uint32_t>(packs_shape[0]);

    bool pass = true;
    try {
        constexpr int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

        auto make_dram = [&](size_t bytes, size_t page_bytes) {
            distributed::ReplicatedBufferConfig rc{.size = bytes};
            distributed::DeviceLocalBufferConfig lc{.page_size = page_bytes, .buffer_type = BufferType::DRAM};
            return distributed::MeshBuffer::create(rc, lc, mesh_device.get());
        };

        auto packs_dram = make_dram(static_cast<size_t>(total_entries) * SCALAR_PACK_PAGE_BYTES, SCALAR_PACK_PAGE_BYTES);
        auto offsets_dram = make_dram(offsets_f32.size() * 4, 4);
        auto px_dram = make_dram(static_cast<size_t>(num_tiles) * TILE_BYTES_BF16, TILE_BYTES_BF16);
        auto py_dram = make_dram(static_cast<size_t>(num_tiles) * TILE_BYTES_BF16, TILE_BYTES_BF16);
        auto out_dram = make_dram(static_cast<size_t>(num_tiles) * 3 * TILE_BYTES_BF16, TILE_BYTES_BF16);

        // Encode attribute packs into 64-byte pages (9 fp32 -> 36 bytes, 28 bytes zero-padded).
        std::vector<uint32_t> packs_payload((static_cast<size_t>(total_entries) * SCALAR_PACK_PAGE_BYTES) / 4, 0);
        for (uint32_t e = 0; e < total_entries; e++) {
            std::memcpy(
                reinterpret_cast<uint8_t*>(packs_payload.data()) + static_cast<size_t>(e) * SCALAR_PACK_PAGE_BYTES,
                &packs_f32[e * 9],
                9 * 4);
        }

        // Encode px/py as bf16 tiles.
        std::vector<uint16_t> px_bf16(static_cast<size_t>(num_tiles) * TILE_H * TILE_W);
        std::vector<uint16_t> py_bf16(static_cast<size_t>(num_tiles) * TILE_H * TILE_W);
        for (uint32_t t = 0; t < num_tiles; t++) {
            auto px_tile = fp32_tile_to_bf16(&px_f32[t * TILE_H * TILE_W]);
            auto py_tile = fp32_tile_to_bf16(&py_f32[t * TILE_H * TILE_W]);
            std::memcpy(&px_bf16[t * TILE_H * TILE_W], px_tile.data(), TILE_BYTES_BF16);
            std::memcpy(&py_bf16[t * TILE_H * TILE_W], py_tile.data(), TILE_BYTES_BF16);
        }

        // Cast offsets fp32 -> uint32.
        std::vector<uint32_t> offsets_u32(offsets_f32.size());
        for (size_t i = 0; i < offsets_f32.size(); i++) {
            offsets_u32[i] = static_cast<uint32_t>(offsets_f32[i]);
        }

        distributed::EnqueueWriteMeshBuffer(cq, packs_dram, packs_payload);
        distributed::EnqueueWriteMeshBuffer(cq, offsets_dram, offsets_u32);
        distributed::EnqueueWriteMeshBuffer(cq, px_dram, px_bf16);
        distributed::EnqueueWriteMeshBuffer(cq, py_dram, py_bf16);

        Program program = CreateProgram();
        constexpr CoreCoord core{0, 0};

        // Tile-sized circular buffers (Float16_b).
        auto cb_tile = [&](uint32_t id, uint32_t depth) {
            CircularBufferConfig c(depth * TILE_BYTES_BF16, {{id, DataFormat::Float16_b}});
            c.set_page_size(id, TILE_BYTES_BF16);
            CreateCircularBuffer(program, core, c);
        };
        // Small (sub-tile) circular buffers.
        auto cb_small = [&](uint32_t id, uint32_t page_bytes, uint32_t depth, DataFormat fmt) {
            CircularBufferConfig c(depth * page_bytes, {{id, fmt}});
            c.set_page_size(id, page_bytes);
            CreateCircularBuffer(program, core, c);
        };

        cb_tile(CB_PX, 2);
        cb_tile(CB_PY, 2);
        cb_small(CB_SCALARS, SCALAR_PACK_PAGE_BYTES, 4, DataFormat::Float32);
        cb_small(CB_TILE_META, META_PAGE_BYTES, 2, DataFormat::UInt32);
        // Depth must be a multiple of 3 (the per-tile batch size) so no
        // single push-of-3 ever straddles the CB wrap. Otherwise the writer's
        // `read_ptr += tile_bytes` arithmetic across the 3 channels is wrong:
        // after the wrap the second/third NoC write would source from
        // out-of-CB L1 instead of slots {0,1} or {0}. Picking 6 keeps two
        // batches in flight (parity with the previous double-buffering depth).
        cb_tile(CB_COLOR_OUT, 6);

        // Scratch CBs for v1a compute (depth 2 each, single-tile pages).
        cb_tile(CB_DX, 2);
        cb_tile(CB_DY, 2);
        cb_tile(CB_DX2, 2);
        cb_tile(CB_DY2, 2);
        cb_tile(CB_DXDY, 2);
        // CB_Q packs 3 tiles per push (a*dx2, c*dy2, 2b*dxdy).
        {
            CircularBufferConfig c(3 * TILE_BYTES_BF16, {{CB_Q, DataFormat::Float16_b}});
            c.set_page_size(CB_Q, TILE_BYTES_BF16);
            CreateCircularBuffer(program, core, c);
        }
        cb_tile(CB_POWER, 2);
        cb_tile(CB_ALPHA, 2);

        // v1b additional scratch CBs (depth 1).
        cb_tile(CB_CONTRIB, 1);
        cb_tile(CB_ONE_MINUS_ALPHA, 1);
        cb_tile(CB_T_TMP, 1);

        // v1b state CBs (depth 1, persistent across Gaussian loop).
        cb_tile(CB_COLOR_R_STATE, 1);
        cb_tile(CB_COLOR_G_STATE, 1);
        cb_tile(CB_COLOR_B_STATE, 1);
        cb_tile(CB_T_STATE, 1);
        cb_tile(CB_SAT_MASK, 1);

        // Constant CBs (depth 1; preloaded once by compute, never popped).
        cb_tile(CB_CONST_ZERO, 1);
        cb_tile(CB_CONST_099, 1);
        cb_tile(CB_CONST_NEG88, 1);

        // Reader: 4 TensorAccessorArgs in order packs, offsets, px, py.
        std::vector<uint32_t> reader_ct;
        TensorAccessorArgs(*packs_dram).append_to(reader_ct);
        TensorAccessorArgs(*offsets_dram).append_to(reader_ct);
        TensorAccessorArgs(*px_dram).append_to(reader_ct);
        TensorAccessorArgs(*py_dram).append_to(reader_ct);
        auto reader = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "gaussian_splatting/kernels/dataflow/reader_alpha_blend.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = reader_ct,
            });

        // Compute: HiFi3, !approx, fp32 dest accumulation.
        auto compute = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "gaussian_splatting/kernels/compute/alpha_blend_compute.cpp",
            core,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi3,
                .fp32_dest_acc_en = true,
                .math_approx_mode = false,
            });

        // Writer: 1 TensorAccessorArgs for the output buffer.
        std::vector<uint32_t> writer_ct;
        TensorAccessorArgs(*out_dram).append_to(writer_ct);
        auto writer = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "gaussian_splatting/kernels/dataflow/writer_alpha_blend.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = writer_ct,
            });

        SetRuntimeArgs(
            program,
            reader,
            core,
            {
                static_cast<uint32_t>(packs_dram->address()),
                static_cast<uint32_t>(offsets_dram->address()),
                static_cast<uint32_t>(px_dram->address()),
                static_cast<uint32_t>(py_dram->address()),
                /*first_tile_id=*/0u,
                /*num_tiles=*/num_tiles,
            });
        SetRuntimeArgs(program, compute, core, {num_tiles});
        SetRuntimeArgs(
            program,
            writer,
            core,
            {
                static_cast<uint32_t>(out_dram->address()),
                /*first_tile_id=*/0u,
                /*num_tiles=*/num_tiles,
            });

        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        distributed::Finish(cq);

        // Read back output buffer (bf16 tiles).
        std::vector<uint16_t> result_bf16(static_cast<size_t>(num_tiles) * 3 * TILE_H * TILE_W);
        distributed::EnqueueReadMeshBuffer(cq, result_bf16, out_dram, /*blocking=*/true);

        // Convert bf16 tiles -> row-major fp32 (H, W, 3).
        std::vector<float> img(static_cast<size_t>(image_h) * image_w * 3, 0.0f);
        for (uint32_t t = 0; t < num_tiles; t++) {
            uint32_t ty = t / tiles_x;
            uint32_t tx = t % tiles_x;
            for (uint32_t ch = 0; ch < 3; ch++) {
                auto fp = bf16_tile_to_fp32(&result_bf16[(3 * t + ch) * TILE_H * TILE_W]);
                for (uint32_t i = 0; i < TILE_H; i++) {
                    for (uint32_t j = 0; j < TILE_W; j++) {
                        uint32_t y = ty * TILE_H + i;
                        uint32_t x = tx * TILE_W + j;
                        if (y < image_h && x < image_w) {
                            img[(static_cast<size_t>(y) * image_w + x) * 3 + ch] = fp[i * TILE_W + j];
                        }
                    }
                }
            }
        }

        save_npy_f32(out_path, img, {image_h, image_w, 3});
        std::cout << "Wrote " << out_path << std::endl;
        pass &= mesh_device->close();
    } catch (const std::exception& e) {
        std::cerr << "Run failed with exception: " << e.what() << std::endl;
        throw;
    }

    return pass ? 0 : 1;
}
