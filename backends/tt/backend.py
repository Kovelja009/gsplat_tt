"""Long-lived wrapper around the v1c alpha-blend kernel daemon.

The daemon (`metal_example_gaussian_splatting --daemon`) opens the Wormhole
device once, JIT-compiles the three kernels, and then loops reading FRAME
requests on stdin. This module spawns it, sends one FRAME per `blend(...)`
call, and reads back the OK/ERR response — keeping the ~3 s init cost off
the per-frame path so interactive use is feasible.

Implements the `gsplat.backend.Backend` contract: only `blend(...)` is
overridden — projection / tile-assignment / sort run on CPU via the
default implementations inherited from the base class.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time

import numpy as np
import torch

from gsplat.backend import Backend
from gsplat.rasterization import prepare_kernel_inputs


class KernelBackend(Backend):
    """Persistent IPC wrapper around the alpha-blend daemon subprocess.

    Spawn once at viewer/script start, call `blend(...)` per frame,
    `close()` on shutdown. The daemon's READY-then-FRAME-then-OK protocol
    is line-oriented over stdin/stdout with .npy files for payload data;
    non-protocol log lines on stdout are skipped.
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
        deadline = time.perf_counter() + 60.0
        ready = None
        line = ""
        while time.perf_counter() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line == "READY":
                ready = line
                break
        if ready != "READY":
            raise RuntimeError(f"daemon failed to start: last line {line!r}")
        # Stage the per-frame .npy hand-off files on tmpfs (/dev/shm) when
        # available so np.save / np.load are RAM memcpys rather than disk
        # round-trips. The daemon reads these four files back every frame and
        # writes out.npy, so on a disk-backed /tmp the
        # serialize→read→kernel→write→deserialize cycle lands directly on the
        # per-frame hot path. `stage_dir` overrides the location: None → auto
        # (/dev/shm if it exists and is writable, else the default temp dir).
        if stage_dir is None:
            shm = "/dev/shm"
            stage_dir = shm if os.path.isdir(shm) and os.access(shm, os.W_OK) else None
        self._tmpdir = tempfile.mkdtemp(prefix="gsplat_viewer_", dir=stage_dir)

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

        # The daemon may interleave non-protocol log lines on stdout; skip
        # past any line that isn't an OK/ERR response so the FRAME→OK pairing
        # stays robust. Bounded by a wall-clock deadline.
        deadline = time.perf_counter() + 30.0
        resp = ""
        while time.perf_counter() < deadline:
            resp = self._proc.stdout.readline()
            if not resp:
                raise RuntimeError("daemon closed stdout unexpectedly")
            resp = resp.strip()
            if resp.startswith("OK ") or resp.startswith("ERR"):
                break
        else:
            raise RuntimeError("daemon timeout waiting for OK/ERR")
        if not resp.startswith("OK "):
            raise RuntimeError(f"daemon error: {resp!r}")
        rt_ms = (time.perf_counter() - t_rt) * 1000.0

        # Parse the daemon's reported device-side kernel time from "OK <ms>".
        # Surface as a sub-timing so callers can separate dispatch+IO from
        # actual on-device kernel runtime.
        kernel_ms = None
        try:
            kernel_ms = float(resp.split(maxsplit=1)[1])
        except (IndexError, ValueError):
            pass

        # Sub-stage D: load the rendered image from the daemon's .npy.
        t_load = time.perf_counter()
        image = np.load(f"{td}/out.npy")
        load_ms = (time.perf_counter() - t_load) * 1000.0

        # Order matches the chronological flow; device_kernel uses a dotted
        # key so the renderer can nest it visually under its parent (it's a
        # sub-measurement of daemon_rt, not a sibling).
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
