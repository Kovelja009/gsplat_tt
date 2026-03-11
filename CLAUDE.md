# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MSc thesis project: implementing the **forward-pass rasterization pipeline** of 3D Gaussian Splatting (3DGS) as custom tt-metal kernels running on Tenstorrent Wormhole/Blackhole hardware.

**Scope:** Inference/rendering only (no training, backward pass, or differentiable rasterization). Load a pre-trained `.ply` file and render it to images.

## Architecture

The rendering pipeline has 6 stages, split across CPU and tt-metal device:

```
Load PLY → [tt-metal] Preprocess/Project → [CPU] Sort → [tt-metal] Rasterize → Display
```

### Pipeline Stages
1. **Load Gaussians** — Parse `.ply` files (position, scale, rotation quaternion, SH coefficients, opacity)
2. **Project 3D→2D** — Per-Gaussian: build 3D covariance, apply view/projection transforms, compute 2D covariance and screen position, frustum cull
3. **Tile Assignment** — Divide screen into 16x16 pixel tiles, find which tiles each Gaussian overlaps
4. **Sort** — Sort by (tile_id, depth) — kept on CPU host
5. **Tile Ranges** — Find start/end indices per tile in the sorted array
6. **Alpha Blending** — Per-pixel front-to-back compositing within each tile

### Target Code Structure
Each pipeline stage should be a separate function with clear tensor I/O, enabling individual stages to be swapped between CPU and tt-metal implementations:

```python
gaussians = load_ply("scene.ply")
means_2d, covs_2d, depths, colors, opacities = preprocess(gaussians, camera)
sorted_indices, tile_ranges = sort_and_bin(means_2d, depths, image_size, tile_size=16)
image = rasterize(means_2d, covs_2d, colors, opacities, sorted_indices, tile_ranges, image_size)
```

### tt-metal Kernel Pattern
Each tt-metal kernel follows the 3-kernel pattern:
- **Reader** — NoC → L1 (SRAM) via circular buffers
- **Compute** — SFPU/FPU processing
- **Writer** — L1 → NoC back to DRAM

Key hardware constraints:
- Use `bfloat16` for on-device data (hardware-optimized)
- SFPU operates on vectors of 32 values; a 16x16 tile (256 pixels) = 8 vector passes
- Wormhole has ~72 usable Tensix cores
- Pad Gaussian attributes to 16 floats for alignment

## Development Phases

1. **PyTorch CPU reference rasterizer** — Pure Python/PyTorch, runs on CPU, serves as golden reference
2. **Interactive viewer** — pygame-based at low resolution (256x256 or 512x512) for visual validation
3. **tt-metal kernels** — Preprocessing kernel first (embarrassingly parallel), then rasterization kernel
4. **Integration & evaluation** — Hybrid pipeline, benchmarking, PSNR/SSIM quality metrics

## Key Dependencies

- `plyfile` — parsing Gaussian `.ply` files
- `torch` — CPU reference implementation and tensor operations
- `pygame` — interactive viewer
- `tt-metal` / `ttnn` — Tenstorrent device programming

## Validation Strategy

Always compare tt-metal output against CPU reference:
```python
image_cpu = render_cpu(gaussians, camera)
image_tt = render_ttmetal(gaussians, camera)
psnr = compute_psnr(image_cpu, image_tt)
# Expect PSNR 30-40dB due to bfloat16 vs float32 precision differences
```

## Reference Implementations

- `hbb1/torch-splatting` — Pure PyTorch, primary reference for CPU implementation
- `gsplat` (nerfstudio) — Production Python+CUDA with pure PyTorch fallback path
- `antimatter15/splat` — Minimal ~300-line WebGL implementation
- Original CUDA rasterizer: `graphdeco-inria/diff-gaussian-rasterization`

## Conventions

- SH degree 0 first (3 color floats per Gaussian), higher degrees later
- Start with small scenes (<500K Gaussians) for development
- Start at 512x512 resolution, scale up later
- Structure-of-Arrays (SoA) layout preferred for device memory (separate buffers per attribute)

## Instructions

- first always activate venv - `source venv/bin/activate`