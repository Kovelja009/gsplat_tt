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

import os
import time

import numpy as np
import torch
import ttnn

from gsplat.backend import Backend
from gsplat.rasterization import prepare_kernel_inputs
from backends.tt.lpt import build_tile_assignment
from backends.tt.segments import build_segmented_assignment

PACK_FLOATS = 16                      # 9 used + 7 zero-pad (64-byte SCALAR_PACK_PAGE)
PACKS_PER_PAGE = 64                   # 4096 / 64
PACK_PAGE_F32 = PACK_FLOATS * PACKS_PER_PAGE  # 1024 f32 = one 4096-byte DRAM page
TILE_ELEMS = 1024                     # 32 * 32
IDS_PER_PAGE = 16                     # 64 / 4
# The reader/writer kernels read a core's tile-id slice into a fixed
# uint32 tile_ids[256] L1 stack array (MAX_TILE_IDS_PER_CORE in the kernels)
# with no bounds check. LPT balances by Gaussian load, not tile count, so a
# core can be handed more tiles than this — which would overflow L1. Fail loud
# here rather than silently corrupt device memory.
MAX_TILE_IDS_PER_CORE = 256


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
        # Resident px/py device grids, keyed by (H, W). The per-pixel coordinate
        # grids depend only on resolution, so upload them once and reuse across
        # frames (the in-process equivalent of the old daemon's SETGRID).
        self._grids: dict[tuple[int, int], tuple] = {}
        # Reusable host scratch for the page-padded packs buffer; grows to the
        # max entry count seen. Avoids a per-frame np.zeros of the whole buffer
        # (the pad columns [9:16] stay zero once allocated).
        self._packs_scratch: np.ndarray | None = None
        # Intra-tile parallelism: when the segmented schedule splits a heavy tile
        # across cores, render via the two-phase partial+combine ops. On by
        # default; GSPLAT_TT_SPLIT=0 forces the legacy single-op path always.
        self._split_enabled = os.environ.get("GSPLAT_TT_SPLIT", "1") != "0"

    def _resident_grids(self, H, W, px, py, num_tiles):
        """Upload (once per resolution) and return the resident px/py device tensors."""
        key = (H, W)
        g = self._grids.get(key)
        if g is None:
            px_page = np.ascontiguousarray(px).reshape(num_tiles, TILE_ELEMS)
            py_page = np.ascontiguousarray(py).reshape(num_tiles, TILE_ELEMS)
            mc, rm = ttnn.DRAM_MEMORY_CONFIG, ttnn.ROW_MAJOR_LAYOUT
            px_dev = ttnn.from_torch(_fp32_to_bf16_trunc(px_page), dtype=ttnn.bfloat16,
                                     layout=rm, device=self.device, memory_config=mc)
            py_dev = ttnn.from_torch(_fp32_to_bf16_trunc(py_page), dtype=ttnn.bfloat16,
                                     layout=rm, device=self.device, memory_config=mc)
            g = (px_dev, py_dev)
            self._grids[key] = g
        return g

    def _packs_page(self, packs, total_entries):
        """Pad the packs SoA to the 4 KB DRAM page geometry the reader expects;
        returns an (npages, PACK_PAGE_F32) f32 view of the reusable scratch."""
        npages = max(1, (total_entries + PACKS_PER_PAGE - 1) // PACKS_PER_PAGE)
        rows = npages * PACKS_PER_PAGE
        sc = self._packs_scratch
        if sc is None or sc.shape[0] < rows:
            # New buffer is fully zeroed, so the pad columns [9:16] are 0.
            sc = np.zeros((rows, PACK_FLOATS), dtype=np.float32)
            self._packs_scratch = sc
        nused = packs.shape[1]
        sc[:total_entries, :nused] = packs          # overwrite used columns
        sc[total_entries:rows, :nused] = 0.0        # zero the <64-row tail padding
        return sc[:rows].reshape(npages, PACK_PAGE_F32)

    def _blend_two_phase(self, H, W, tiles_x, tiles_y, num_tiles, packs_page, px, py, sched, prep_ms):
        """Intra-tile parallel render: partial op composites depth-segments into
        (R,G,B,T) partials; combine op merges them per tile via the associative
        `over` operator. Used when the schedule splits a heavy tile across cores."""
        dev = self.device
        mc, rm = ttnn.DRAM_MEMORY_CONFIG, ttnn.ROW_MAJOR_LAYOUT

        # Per-core caps: the partial writer caches partial_slots and the combine
        # writer caches out_tiles in fixed L1 arrays (MAX_TILE_IDS_PER_CORE).
        if int(sched.per_core_count.max()) > MAX_TILE_IDS_PER_CORE:
            raise RuntimeError(
                f"per-core job count {int(sched.per_core_count.max())} exceeds the kernel "
                f"cap {MAX_TILE_IDS_PER_CORE} at {H}x{W}; lower resolution or raise the cap.")

        # --- combine plan: pad to 4 cols, distribute rows contiguously across cores ---
        plan = sched.combine_plan
        nplan = plan.shape[0]
        plan4 = np.zeros((nplan, 4), dtype=np.uint32)
        plan4[:, :3] = plan
        nc = self.num_cores
        base, rem = divmod(nplan, nc)
        cpo = np.zeros(nc, dtype=np.uint32)
        cpc = np.zeros(nc, dtype=np.uint32)
        off = 0
        for c in range(nc):
            cnt = base + (1 if c < rem else 0)
            cpo[c] = off
            cpc[c] = cnt
            off += cnt

        # --- upload ---
        t = time.perf_counter()
        px_dev, py_dev = self._resident_grids(H, W, px, py, num_tiles)
        packs_dev = ttnn.from_torch(torch.from_numpy(packs_page), dtype=ttnn.float32, layout=rm, device=dev, memory_config=mc)
        job_dev = ttnn.from_torch(_u32_tensor(sched.job_table.reshape(-1, 4)), dtype=ttnn.uint32, layout=rm, device=dev, memory_config=mc)
        plan_dev = ttnn.from_torch(_u32_tensor(plan4), dtype=ttnn.uint32, layout=rm, device=dev, memory_config=mc)
        ttnn.synchronize_device(dev)
        upload_ms = (time.perf_counter() - t) * 1000.0

        # --- phase 1: partial composite (each core does its segment-jobs) ---
        t = time.perf_counter()
        partials = ttnn.experimental.gaussian_alpha_blend_partial(
            packs_dev, px_dev, py_dev, job_dev,
            image_height=H, image_width=W, num_tiles=num_tiles, num_jobs=sched.num_jobs,
            per_core_offset=[int(x) for x in sched.per_core_offset],
            per_core_count=[int(x) for x in sched.per_core_count])
        ttnn.synchronize_device(dev)
        partial_ms = (time.perf_counter() - t) * 1000.0

        # --- phase 2: combine (associative over-merge per tile) ---
        t = time.perf_counter()
        out = ttnn.experimental.gaussian_alpha_blend_combine(
            partials, plan_dev, num_tiles=num_tiles,
            per_core_offset=[int(x) for x in cpo], per_core_count=[int(x) for x in cpc])
        ttnn.synchronize_device(dev)
        combine_ms = (time.perf_counter() - t) * 1000.0

        # --- readback ---
        t = time.perf_counter()
        out_t = ttnn.to_torch(out)
        readback_ms = (time.perf_counter() - t) * 1000.0
        t2 = time.perf_counter()
        image = self._tiles_to_image(out_t, tiles_x, tiles_y, H, W)
        unpack_ms = (time.perf_counter() - t2) * 1000.0

        return image, {
            "prep": prep_ms, "upload": upload_ms,
            "partial_kernel": partial_ms, "combine_kernel": combine_ms,
            "kernel": partial_ms + combine_ms,           # device compute (both phases)
            "download": readback_ms + unpack_ms,
            "readback": readback_ms, "unpack": unpack_ms,
        }

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

        # --- Intra-tile parallelism: if the segmented schedule splits a heavy
        # tile across cores, render via the two-phase partial+combine path.
        # combine_plan has one row per non-empty tile, so num_jobs > its length
        # iff at least one tile was split. ---
        if self._split_enabled:
            sched = build_segmented_assignment(offsets, num_tiles, self.num_cores)
            if sched.num_jobs > sched.combine_plan.shape[0]:
                packs_page = self._packs_page(packs, total_entries)
                return self._blend_two_phase(
                    H, W, tiles_x, tiles_y, num_tiles, packs_page, px, py, sched, prep_ms)

        # --- LPT schedule (host) ---
        per_core_offset, per_core_count, tile_ids = build_tile_assignment(
            offsets, num_tiles, self.num_cores)
        max_per_core = int(per_core_count.max())
        if max_per_core > MAX_TILE_IDS_PER_CORE:
            raise RuntimeError(
                f"per-core tile count {max_per_core} exceeds the kernel cap "
                f"{MAX_TILE_IDS_PER_CORE} at {H}x{W} ({num_tiles} tiles, {self.num_cores} "
                f"cores); the reader/writer L1 tile_ids buffer would overflow. "
                f"Lower the resolution or raise MAX_TILE_IDS_PER_CORE in the kernels.")

        try:
            # --- build page-matching host arrays (per-frame: packs/offsets/tile_ids) ---
            t = time.perf_counter()
            packs_page = self._packs_page(packs, total_entries)

            offsets_col = offsets.reshape(-1, 1)

            ntid_pages = max(1, (tile_ids.shape[0] + IDS_PER_PAGE - 1) // IDS_PER_PAGE)
            tile_ids_pad = np.zeros((ntid_pages * IDS_PER_PAGE,), dtype=np.uint32)
            tile_ids_pad[:tile_ids.shape[0]] = tile_ids
            tile_ids_page = tile_ids_pad.reshape(ntid_pages, IDS_PER_PAGE)

            # --- upload (ROW_MAJOR DRAM interleaved); px/py are resident per resolution ---
            mc, rm = ttnn.DRAM_MEMORY_CONFIG, ttnn.ROW_MAJOR_LAYOUT
            px_dev, py_dev = self._resident_grids(H, W, px, py, num_tiles)
            packs_dev = ttnn.from_torch(torch.from_numpy(packs_page), dtype=ttnn.float32, layout=rm, device=dev, memory_config=mc)
            offsets_dev = ttnn.from_torch(_u32_tensor(offsets_col), dtype=ttnn.uint32, layout=rm, device=dev, memory_config=mc)
            tile_ids_dev = ttnn.from_torch(_u32_tensor(tile_ids_page), dtype=ttnn.uint32, layout=rm, device=dev, memory_config=mc)
            # ttnn calls are async (enqueue + return). Sync so upload_ms reflects
            # real H2D completion, not just the enqueue cost.
            ttnn.synchronize_device(dev)
            upload_ms = (time.perf_counter() - t) * 1000.0

            # --- op call (warm after first frame per resolution); op zero-inits empty tiles ---
            t = time.perf_counter()
            out = ttnn.experimental.gaussian_alpha_blend(
                packs_dev, offsets_dev, px_dev, py_dev, tile_ids_dev,
                image_height=H, image_width=W, num_tiles=num_tiles,
                per_core_offset=[int(x) for x in per_core_offset],
                per_core_count=[int(x) for x in per_core_count])
            # Sync so kernel_ms is the real device compute. Without it, the op
            # only enqueues (~0.2 ms) and the blocking to_torch readback below
            # would absorb the entire compute wait into download_ms. Free in
            # wall-clock: the frame fully syncs at readback anyway and nothing
            # pipelines across the upload->op->readback dependency chain.
            ttnn.synchronize_device(dev)
            kernel_ms = (time.perf_counter() - t) * 1000.0

            # --- readback (num_tiles*3, 1024) bf16 -> (H, W, 3) ---
            t = time.perf_counter()
            out_t = ttnn.to_torch(out)                          # (num_tiles*3, 1024) bf16 torch
            readback_ms = (time.perf_counter() - t) * 1000.0
            t2 = time.perf_counter()
            image = self._tiles_to_image(out_t, tiles_x, tiles_y, H, W)
            unpack_ms = (time.perf_counter() - t2) * 1000.0
            download_ms = readback_ms + unpack_ms
        except Exception as e:
            # nerfview cancels an in-flight render by raising InterruptRenderException
            # from a trace hook at an arbitrary line — that's cooperative control
            # flow, not a device fault, so let it (and KeyboardInterrupt) propagate
            # untouched. Only wrap genuine errors with context.
            if type(e).__name__ == "InterruptRenderException":
                raise
            raise RuntimeError(
                f"TT alpha_blend failed at {H}x{W} ({total_entries} entries, "
                f"{num_tiles} tiles): {e}") from e

        return image, {
            "prep": prep_ms, "upload": upload_ms,
            "kernel": kernel_ms, "download": download_ms,
            "readback": readback_ms, "unpack": unpack_ms,
        }

    @staticmethod
    def _tiles_to_image(out_t, tiles_x, tiles_y, H, W):
        """(num_tiles*3, 1024) bf16 torch, tile-major -> (H, W, 3) fp32 numpy.

        The permute + contiguous run in torch on bf16 (half the bytes, so
        cache-friendlier and multithreaded), then a single widen to fp32 on the
        already-contiguous cropped image — ~10x faster than the numpy fp32
        transpose for interactive resolutions.
        """
        g = out_t.reshape(tiles_y, tiles_x, 3, 32, 32).permute(0, 3, 1, 4, 2)  # (ty,32,tx,32,3)
        img = g.reshape(tiles_y * 32, tiles_x * 32, 3)[:H, :W, :]
        return img.contiguous().float().numpy()

    def close(self):
        if getattr(self, "device", None) is not None:
            ttnn.close_device(self.device)
            self.device = None
