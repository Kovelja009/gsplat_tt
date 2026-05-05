"""Long-lived wrapper around the v1c alpha-blend kernel daemon.

The daemon (`metal_example_gaussian_splatting --daemon`) opens the Wormhole
device once, JIT-compiles the three kernels, and then loops reading FRAME
requests on stdin. This module spawns it, sends one FRAME per render call,
and reads back the OK/ERR response — keeping the ~3 s init cost off the
per-frame path so interactive use is feasible.
"""
import os
import subprocess
import tempfile
import time

import numpy as np
import torch

from rasterization import prepare_kernel_inputs


class KernelBackend:
    """Persistent IPC wrapper around the alpha-blend daemon subprocess.

    Spawn once at viewer/script start, call `render(...)` per frame,
    and `close()` on shutdown. The daemon's READY-then-FRAME-then-OK
    protocol is line-oriented over stdin/stdout with .npy files for
    payload data; non-protocol log lines on stdout are skipped.
    """

    BINARY_PATH = "tt-metal/build/programming_examples/metal_example_gaussian_splatting"

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        env = os.environ.copy()
        env.setdefault("TT_METAL_HOME", os.path.abspath("tt-metal"))
        env.setdefault("TT_METAL_RUNTIME_ROOT", os.path.abspath("tt-metal"))
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
        self._tmpdir = tempfile.mkdtemp(prefix="gsplat_viewer_")

    def render(
        self,
        means_2d: torch.Tensor,
        covs_2d: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        sorted_gids: torch.Tensor,
        tile_ranges: torch.Tensor,
        H: int,
        W: int,
    ) -> np.ndarray:
        t0 = time.perf_counter()
        packs, offsets, px, py = prepare_kernel_inputs(
            means_2d, covs_2d, colors, opacities, sorted_gids, tile_ranges, H, W,
        )
        prep_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        td = self._tmpdir
        np.save(f"{td}/packs.npy", packs)
        np.save(f"{td}/offsets.npy", offsets.astype(np.float32))
        np.save(f"{td}/px.npy", px)
        np.save(f"{td}/py.npy", py)
        save_ms = (time.perf_counter() - t1) * 1000.0

        if self.verbose:
            print(
                f"[kernel-pre] prep={prep_ms:.0f}ms  save={save_ms:.0f}ms  "
                f"packs.shape={packs.shape}",
                flush=True,
            )

        t2 = time.perf_counter()
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

        if self.verbose:
            rt_ms = (time.perf_counter() - t2) * 1000.0
            print(f"[kernel-post] daemon-roundtrip={rt_ms:.0f}ms  resp={resp!r}", flush=True)

        # Binary already crops to (H, W, 3) on its side.
        return np.load(f"{td}/out.npy")

    def close(self):
        # Don't rmtree tmpdir here — nerfview's render thread may still be
        # mid-flight in render(), and pulling the directory out from under it
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
