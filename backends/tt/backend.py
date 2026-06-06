"""Long-lived wrapper around the v1c alpha-blend kernel daemon.

The daemon (`metal_example_gaussian_splatting --daemon`) opens the Wormhole
device once, JIT-compiles the three kernels, and then loops reading FRAME
requests on stdin. This module spawns it, sends one FRAME per `blend(...)`
call, and reads back the OK/ERR response — keeping the ~3 s init cost off
the per-frame path so interactive use is feasible.

Two hand-off protocols are supported:

  MFRAME (zero-copy, preferred): a /dev/shm region is mmap'd by both
  processes once. Each frame the host writes device-ready buffers (packs in
  64-byte pages, offsets u32, px/py truncated bf16) directly into the region
  and the daemon uploads them straight to DRAM and reads the bf16 result
  back into the region — no .npy serialize/parse, no fp32<->bf16 round-trip.

  FRAME (.npy, fallback): four .npy files staged on disk/tmpfs. Used if the
  shared-memory handshake fails for any reason.

Implements the `gsplat.backend.Backend` contract: only `blend(...)` is
overridden — projection / tile-assignment / sort run on CPU via the
default implementations inherited from the base class.
"""
from __future__ import annotations

import mmap
import os
import subprocess
import tempfile
import time

import numpy as np
import torch

from gsplat.backend import Backend
from gsplat.rasterization import prepare_kernel_inputs

# Device-buffer layout constants (must match alpha_blend_host.h).
SCALAR_PACK_PAGE_BYTES = 64          # 9 fp32 payload + zero pad
TILE_BYTES_BF16 = 32 * 32 * 2        # one bf16 32x32 tile = 2 KB
_SHM_ALIGN = 4096                    # page-align each region in the shm
_SHM_CAPACITY = 512 * 1024 * 1024    # 512 MB; tmpfs-backed, lazily paged


def _align(x: int) -> int:
    return (x + _SHM_ALIGN - 1) & ~(_SHM_ALIGN - 1)


class KernelBackend(Backend):
    """Persistent IPC wrapper around the alpha-blend daemon subprocess.

    Spawn once at viewer/script start, call `blend(...)` per frame,
    `close()` on shutdown. The daemon's READY-then-FRAME-then-OK protocol
    is line-oriented over stdin/stdout with .npy files (or, preferred, a
    shared-memory region) for payload data; non-protocol log lines on
    stdout are skipped.
    """

    BINARY_PATH = "backends/tt/tt-metal/build/programming_examples/metal_example_gaussian_splatting"

    def __init__(self, verbose: bool = False, stage_dir: str | None = None):
        self.verbose = verbose
        env = os.environ.copy()
        env.setdefault("TT_METAL_HOME", os.path.abspath("backends/tt/tt-metal"))
        env.setdefault("TT_METAL_RUNTIME_ROOT", os.path.abspath("backends/tt/tt-metal"))
        self._proc = subprocess.Popen(
            [self.BINARY_PATH, "--daemon"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )
        # The daemon may emit tt-metal init log lines on stdout before the
        # READY sentinel; skip past them with a wall-clock deadline.
        ready = self._read_response(60.0, accept=("READY",))
        if ready != "READY":
            raise RuntimeError(f"daemon failed to start: last line {ready!r}")

        # .npy fallback staging dir (option A: tmpfs when available).
        if stage_dir is None:
            shm = "/dev/shm"
            stage_dir = shm if os.path.isdir(shm) and os.access(shm, os.W_OK) else None
        self._tmpdir = tempfile.mkdtemp(prefix="gsplat_viewer_", dir=stage_dir)

        # Shared-memory zero-copy handoff. If anything here fails we leave
        # self._mm = None and blend() transparently uses the .npy FRAME path.
        self._mm: mmap.mmap | None = None
        self._shm_path: str | None = None
        self._setup_shared_memory()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_shared_memory(self) -> None:
        """mmap a /dev/shm region and hand its path to the daemon (MMAP)."""
        if not (os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK)):
            return
        path = f"/dev/shm/gsplat_shm_{os.getpid()}.buf"
        try:
            fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
            os.ftruncate(fd, _SHM_CAPACITY)
            mm = mmap.mmap(fd, _SHM_CAPACITY)
            os.close(fd)
            self._proc.stdin.write(f"MMAP {path} {_SHM_CAPACITY}\n")
            self._proc.stdin.flush()
            resp = self._read_response(30.0)
            if resp != "MMAP_OK":
                mm.close()
                os.unlink(path)
                return
            self._mm = mm
            self._shm_path = path
        except OSError:
            # No shared memory: stay on the .npy path.
            try:
                os.unlink(path)
            except OSError:
                pass

    def _read_response(self, timeout_s: float, accept: tuple[str, ...] = ("OK ", "ERR", "MMAP_OK")) -> str:
        """Read stdout, skipping non-protocol log lines, until a response.

        Returns the first line that startswith any of `accept`, or raises on
        timeout / closed pipe.
        """
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("daemon closed stdout unexpectedly")
            line = line.strip()
            if any(line == a or line.startswith(a) for a in accept):
                return line
        raise RuntimeError("daemon timeout waiting for response")

    # ------------------------------------------------------------------
    # Backend API
    # ------------------------------------------------------------------

    def blend(
        self,
        means_2d: torch.Tensor,
        covs_2d: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        sorted_gaussian_ids: torch.Tensor,
        tile_ranges: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> tuple[np.ndarray, dict[str, float]]:
        if self._mm is not None:
            return self._blend_mmap(
                means_2d, covs_2d, colors, opacities,
                sorted_gaussian_ids, tile_ranges, image_height, image_width,
            )
        return self._blend_npy(
            means_2d, covs_2d, colors, opacities,
            sorted_gaussian_ids, tile_ranges, image_height, image_width,
        )

    # ------------------------------------------------------------------
    # Zero-copy shared-memory path (MFRAME)
    # ------------------------------------------------------------------

    def _blend_mmap(
        self, means_2d, covs_2d, colors, opacities,
        sorted_gaussian_ids, tile_ranges, image_height, image_width,
    ) -> tuple[np.ndarray, dict[str, float]]:
        H, W = image_height, image_width
        tiles_x = (W + 31) // 32
        tiles_y = (H + 31) // 32
        num_tiles = tiles_x * tiles_y

        # Sub-stage A: SoA repack (reuse the tested host prep).
        t = time.perf_counter()
        packs, offsets, px, py = prepare_kernel_inputs(
            means_2d, covs_2d, colors, opacities,
            sorted_gaussian_ids, tile_ranges, H, W,
        )
        total_entries = int(packs.shape[0])
        offsets_count = int(offsets.shape[0])
        prep_ms = (time.perf_counter() - t) * 1000.0

        # Compute the shm layout: page-aligned, contiguous regions.
        packs_off = 0
        offsets_off = _align(total_entries * SCALAR_PACK_PAGE_BYTES)
        px_off = _align(offsets_off + offsets_count * 4)
        py_off = _align(px_off + num_tiles * TILE_BYTES_BF16)
        out_off = _align(py_off + num_tiles * TILE_BYTES_BF16)
        total_needed = out_off + num_tiles * 3 * TILE_BYTES_BF16
        if total_needed > _SHM_CAPACITY:
            # Frame too large for the mapped region (e.g. 4K with huge entry
            # counts) — fall back to the .npy path for this one frame.
            return self._blend_npy(
                means_2d, covs_2d, colors, opacities,
                sorted_gaussian_ids, tile_ranges, H, W,
            )

        # Sub-stage B: write device-ready buffers directly into shm.
        t = time.perf_counter()
        # packs -> (N, 16) fp32 pages (9 used + 7 zero-pad).
        packs_view = np.ndarray((total_entries, 16), dtype=np.float32,
                                buffer=self._mm, offset=packs_off)
        packs_view[:, :9] = packs
        packs_view[:, 9:] = 0.0
        # offsets -> u32.
        off_view = np.ndarray((offsets_count,), dtype=np.uint32,
                              buffer=self._mm, offset=offsets_off)
        off_view[:] = offsets
        # px/py -> truncated bf16 (top 16 bits of fp32, matching the daemon's
        # fp32_tile_to_bf16 `u >> 16` — bit-identical, so PSNR is unchanged).
        px_view = np.ndarray((num_tiles * 1024,), dtype=np.uint16,
                             buffer=self._mm, offset=px_off)
        py_view = np.ndarray((num_tiles * 1024,), dtype=np.uint16,
                             buffer=self._mm, offset=py_off)
        px_view[:] = (np.ascontiguousarray(px).reshape(-1).view(np.uint32) >> 16).astype(np.uint16)
        py_view[:] = (np.ascontiguousarray(py).reshape(-1).view(np.uint32) >> 16).astype(np.uint16)
        write_ms = (time.perf_counter() - t) * 1000.0

        # Sub-stage C: daemon round-trip (upload + kernel + readback into shm).
        t = time.perf_counter()
        line = (
            f"MFRAME {H} {W} {total_entries} {num_tiles} {offsets_count} "
            f"{packs_off} {offsets_off} {px_off} {py_off} {out_off}\n"
        )
        self._proc.stdin.write(line)
        self._proc.stdin.flush()
        resp = self._read_response(30.0, accept=("OK ", "ERR"))
        if not resp.startswith("OK "):
            raise RuntimeError(f"daemon error: {resp!r}")
        rt_ms = (time.perf_counter() - t) * 1000.0
        kernel_ms = None
        try:
            kernel_ms = float(resp.split(maxsplit=1)[1])
        except (IndexError, ValueError):
            pass

        # Sub-stage D: read bf16 output from shm and rearrange to (H, W, 3).
        t = time.perf_counter()
        out_view = np.ndarray((num_tiles, 3, 32, 32), dtype=np.uint16,
                              buffer=self._mm, offset=out_off)
        # bf16 -> fp32: widen and shift left 16 (inverse of the daemon's pack).
        out_f32 = (out_view.astype(np.uint32) << 16).view(np.float32)
        image = self._tiles_to_image(out_f32, tiles_x, tiles_y, H, W)
        read_ms = (time.perf_counter() - t) * 1000.0

        sub_timings: dict[str, float] = {
            "prep": prep_ms,
            "write_shm": write_ms,
            "mframe_rt": rt_ms,
        }
        if kernel_ms is not None:
            sub_timings["mframe_rt.device_kernel"] = kernel_ms
        sub_timings["read_shm"] = read_ms
        return image, sub_timings

    @staticmethod
    def _tiles_to_image(tiles: np.ndarray, tiles_x: int, tiles_y: int,
                        H: int, W: int) -> np.ndarray:
        """(num_tiles, 3, 32, 32) tile-major bf16->fp32 -> (H, W, 3) row-major."""
        g = tiles.reshape(tiles_y, tiles_x, 3, 32, 32)
        g = g.transpose(0, 3, 1, 4, 2)            # (ty, 32, tx, 32, 3)
        img = g.reshape(tiles_y * 32, tiles_x * 32, 3)
        return np.ascontiguousarray(img[:H, :W, :])

    # ------------------------------------------------------------------
    # .npy fallback path (FRAME)
    # ------------------------------------------------------------------

    def _blend_npy(
        self, means_2d, covs_2d, colors, opacities,
        sorted_gaussian_ids, tile_ranges, image_height, image_width,
    ) -> tuple[np.ndarray, dict[str, float]]:
        H, W = image_height, image_width

        # Sub-stage A: SoA repack (kernel-friendly layout).
        t_prep = time.perf_counter()
        packs, offsets, px, py = prepare_kernel_inputs(
            means_2d, covs_2d, colors, opacities,
            sorted_gaussian_ids, tile_ranges, H, W,
        )
        prep_ms = (time.perf_counter() - t_prep) * 1000.0

        # Sub-stage B: serialize SoA buffers as .npy for the daemon.
        t_save = time.perf_counter()
        td = self._tmpdir
        np.save(f"{td}/packs.npy", packs)
        np.save(f"{td}/offsets.npy", offsets.astype(np.float32))
        np.save(f"{td}/px.npy", px)
        np.save(f"{td}/py.npy", py)
        save_ms = (time.perf_counter() - t_save) * 1000.0

        # Sub-stage C: daemon round-trip (DRAM upload + kernel + readback).
        t_rt = time.perf_counter()
        line = (
            f"FRAME {H} {W} "
            f"{td}/packs.npy {td}/offsets.npy {td}/px.npy {td}/py.npy {td}/out.npy\n"
        )
        self._proc.stdin.write(line)
        self._proc.stdin.flush()
        resp = self._read_response(30.0, accept=("OK ", "ERR"))
        if not resp.startswith("OK "):
            raise RuntimeError(f"daemon error: {resp!r}")
        rt_ms = (time.perf_counter() - t_rt) * 1000.0

        kernel_ms = None
        try:
            kernel_ms = float(resp.split(maxsplit=1)[1])
        except (IndexError, ValueError):
            pass

        # Sub-stage D: load the rendered image from the daemon's .npy.
        t_load = time.perf_counter()
        image = np.load(f"{td}/out.npy")
        load_ms = (time.perf_counter() - t_load) * 1000.0

        sub_timings: dict[str, float] = {
            "prep": prep_ms,
            "save_npy": save_ms,
            "daemon_rt": rt_ms,
        }
        if kernel_ms is not None:
            sub_timings["daemon_rt.device_kernel"] = kernel_ms
        sub_timings["load_npy"] = load_ms

        return image, sub_timings

    def close(self) -> None:
        # Don't rmtree tmpdir here — nerfview's render thread may still be
        # mid-flight in blend(), and pulling the directory out from under it
        # produces a confusing FileNotFoundError on Ctrl+C. /tmp is cleaned by
        # the OS on reboot; leaving the dir is harmless.
        #
        # Process cleanup: try graceful QUIT first; if the daemon doesn't
        # exit promptly, hard-kill so the Wormhole device is released.
        # Leaving a daemon orphaned holds the device and breaks the next
        # invocation (the tt-metal driver hangs trying to acquire it).
        if self._proc.poll() is None:
            try:
                self._proc.stdin.write("QUIT\n")
                self._proc.stdin.flush()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                try:
                    self._proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass

        # Release the shared-memory mapping and unlink its backing file.
        if self._mm is not None:
            try:
                self._mm.close()
            except Exception:
                pass
            self._mm = None
        if self._shm_path is not None:
            try:
                os.unlink(self._shm_path)
            except OSError:
                pass
            self._shm_path = None
