"""In-process ttnn alpha-blend backend.

Opens the Tenstorrent device once via ttnn, keeps the compiled kernels warm
through ttnn's program cache, and renders each frame by calling the
ttnn.experimental.gaussian_alpha_blend op. No daemon, no IPC.

Tensor page-size matching
-------------------------
The reader/writer kernels hardcode DRAM page sizes (packs=4096 B, px/py=2048 B,
offsets=4 B, tile_ids=64 B, output=2048 B). ttnn pages a ROW_MAJOR interleaved
buffer by its last-dim row, so we shape each input so last_dim * elemsize equals
the wanted page size:
  packs    -> (npages, 1024) f32   (1024*4 = 4096; one page = 64 packs of 16 f32)
  px / py  -> (num_tiles, 1024) bf16 (1024*2 = 2048; one page = one 32x32 tile)
  offsets  -> (M, 1) u32           (1*4 = 4)
  tile_ids -> (npages16, 16) u32   (16*4 = 64)
The op's output is (num_tiles*3, 1024) bf16 (page 2048); we reshape it back to
(num_tiles, 3, 32, 32) on readback.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import ttnn

from gsplat.backend import Backend
from gsplat.rasterization import prepare_kernel_inputs
from backends.tt.lpt import build_tile_assignment

PACK_FLOATS = 16          # 9 used + 7 zero-pad (matches 64-byte SCALAR_PACK_PAGE)
PACKS_PER_PAGE = 64       # 4096 / 64
PACK_PAGE_F32 = 1024      # 4096 / 4
TILE_ELEMS = 1024         # 32 * 32
IDS_PER_PAGE = 16         # 64 / 4


def _fp32_to_bf16_trunc(arr: np.ndarray) -> torch.Tensor:
    """Truncate fp32 -> bf16 (top 16 bits), matching the daemon's `u >> 16`."""
    u16 = (np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
    return torch.from_numpy(u16.view(np.int16)).view(torch.bfloat16)


def _u32_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.uint32).view(np.int32))


class KernelBackend(Backend):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.device = ttnn.open_device(device_id=0)
        self.device.enable_program_cache()
        grid = self.device.compute_with_storage_grid_size()
        self.num_cores = grid.x * grid.y

    def blend(self, means_2d, covs_2d, colors, opacities,
              sorted_gaussian_ids, tile_ranges, image_height, image_width):
        dev = self.device
        H, W = image_height, image_width
        tiles_x = (W + 31) // 32
        tiles_y = (H + 31) // 32
        num_tiles = tiles_x * tiles_y

        # --- host prep (reused) ---
        t = time.perf_counter()
        packs, offsets, px, py = prepare_kernel_inputs(
            means_2d, covs_2d, colors, opacities,
            sorted_gaussian_ids, tile_ranges, H, W)
        total_entries = int(packs.shape[0])
        offsets = np.ascontiguousarray(offsets).astype(np.uint32).reshape(-1)
        prep_ms = (time.perf_counter() - t) * 1000.0

        # --- LPT schedule (host) ---
        per_core_offset, per_core_count, tile_ids = build_tile_assignment(
            offsets, num_tiles, self.num_cores)

        # --- build page-matching host arrays ---
        t = time.perf_counter()
        npages = max(1, (total_entries + PACKS_PER_PAGE - 1) // PACKS_PER_PAGE)
        packs_padded = np.zeros((npages * PACKS_PER_PAGE, PACK_FLOATS), dtype=np.float32)
        packs_padded[:total_entries, :packs.shape[1]] = packs
        packs_page = packs_padded.reshape(npages, PACK_PAGE_F32)

        px_page = np.ascontiguousarray(px).reshape(num_tiles, TILE_ELEMS)
        py_page = np.ascontiguousarray(py).reshape(num_tiles, TILE_ELEMS)

        offsets_col = offsets.reshape(-1, 1)

        ntid_pages = max(1, (tile_ids.shape[0] + IDS_PER_PAGE - 1) // IDS_PER_PAGE)
        tile_ids_pad = np.zeros((ntid_pages * IDS_PER_PAGE,), dtype=np.uint32)
        tile_ids_pad[:tile_ids.shape[0]] = tile_ids
        tile_ids_page = tile_ids_pad.reshape(ntid_pages, IDS_PER_PAGE)

        # --- upload to device (ROW_MAJOR DRAM interleaved) ---
        mc = ttnn.DRAM_MEMORY_CONFIG
        rm = ttnn.ROW_MAJOR_LAYOUT
        packs_dev = ttnn.from_torch(torch.from_numpy(packs_page), dtype=ttnn.float32, layout=rm, device=dev, memory_config=mc)
        offsets_dev = ttnn.from_torch(_u32_tensor(offsets_col), dtype=ttnn.uint32, layout=rm, device=dev, memory_config=mc)
        px_dev = ttnn.from_torch(_fp32_to_bf16_trunc(px_page), dtype=ttnn.bfloat16, layout=rm, device=dev, memory_config=mc)
        py_dev = ttnn.from_torch(_fp32_to_bf16_trunc(py_page), dtype=ttnn.bfloat16, layout=rm, device=dev, memory_config=mc)
        tile_ids_dev = ttnn.from_torch(_u32_tensor(tile_ids_page), dtype=ttnn.uint32, layout=rm, device=dev, memory_config=mc)
        upload_ms = (time.perf_counter() - t) * 1000.0

        # --- op call (warm after first frame per resolution) ---
        t = time.perf_counter()
        out = ttnn.experimental.gaussian_alpha_blend(
            packs_dev, offsets_dev, px_dev, py_dev, tile_ids_dev,
            image_height=H, image_width=W, num_tiles=num_tiles,
            per_core_offset=[int(x) for x in per_core_offset],
            per_core_count=[int(x) for x in per_core_count])
        kernel_ms = (time.perf_counter() - t) * 1000.0

        # --- readback (num_tiles*3, 1024) bf16 -> (H, W, 3) ---
        t = time.perf_counter()
        out_t = ttnn.to_torch(out).float().numpy()          # (num_tiles*3, 1024)
        tiles = out_t.reshape(num_tiles, 3, 32, 32).copy()
        # LPT skips empty tiles, so the writer never touches their output slots
        # and the op-created tensor leaves them uninitialised. Zero them: an
        # empty tile is one whose per-tile Gaussian count (offsets diff) is 0.
        counts = offsets[1:num_tiles + 1].astype(np.int64) - offsets[:num_tiles].astype(np.int64)
        empty = counts == 0
        if empty.any():
            tiles[empty] = 0.0
        image = self._tiles_to_image(tiles, tiles_x, tiles_y, H, W)
        download_ms = (time.perf_counter() - t) * 1000.0

        return image, {
            "prep": prep_ms, "upload": upload_ms,
            "kernel": kernel_ms, "download": download_ms,
        }

    @staticmethod
    def _tiles_to_image(tiles, tiles_x, tiles_y, H, W):
        """(num_tiles, 3, 32, 32) tile-major -> (H, W, 3) row-major."""
        g = tiles.reshape(tiles_y, tiles_x, 3, 32, 32)
        g = g.transpose(0, 3, 1, 4, 2)            # (ty, 32, tx, 32, 3)
        img = g.reshape(tiles_y * 32, tiles_x * 32, 3)
        return np.ascontiguousarray(img[:H, :W, :])

    def close(self):
        if getattr(self, "device", None) is not None:
            ttnn.close_device(self.device)
            self.device = None
