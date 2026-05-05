import os
import shutil
import subprocess
import tempfile
import time

import numpy as np
import torch
import viser
import nerfview

from data_structures import Gaussians
from utils import c2w_to_w2c
from rasterization import (
    project_gaussians,
    get_tile_assignments,
    sort_and_bin,
    alpha_blend,
    prepare_kernel_inputs,
)


class KernelBackend:
    """Long-lived wrapper around the v1c kernel daemon subprocess.

    Spawn once on viewer start, send FRAME requests per render call,
    read OK/ERR per frame, cleanly terminate on close().
    """

    BINARY_PATH = "tt-metal/build/programming_examples/metal_example_gaussian_splatting"

    def __init__(self):
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
        t2 = time.perf_counter()
        print(
            f"[kernel-pre] prep={prep_ms:.0f}ms  save={save_ms:.0f}ms  "
            f"packs.shape={packs.shape}",
            flush=True,
        )
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
        rt_ms = (time.perf_counter() - t2) * 1000.0
        print(f"[kernel-post] daemon-roundtrip={rt_ms:.0f}ms  resp={resp!r}", flush=True)
        # Binary already crops to (H, W, 3) on its side.
        return np.load(f"{td}/out.npy")

    def close(self):
        # Don't rmtree tmpdir here — nerfview's render thread may still be
        # mid-flight in render(), and pulling the directory out from under it
        # produces a confusing FileNotFoundError on Ctrl+C. /tmp is cleaned by
        # the OS on reboot; leaving the dir is harmless.
        if self._proc.poll() is None:
            try:
                self._proc.stdin.write("QUIT\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()


class GaussianViewer:
    """Interactive viewer for 3D Gaussian Splatting scenes.

    Wraps nerfview/viser to display frames rendered by the CPU rasterizer
    or the tt-metal v1c kernel daemon, with orbit camera controls in a browser.
    """

    def __init__(
        self,
        gaussians: Gaussians,
        host: str = "0.0.0.0",
        port: int = 8080,
        backend: str = "cpu",
        max_resolution: int = 640,
    ):
        self.gaussians = gaussians
        self.backend = backend  # "cpu" or "kernel"
        # Cap longest output dim. Browser tabs ask for arbitrary sizes (often
        # 1920x1080); without a cap, prepare_kernel_inputs takes seconds and
        # the viewer feels frozen. Stretching the displayed image is fine for
        # interactive preview.
        self.max_resolution = max_resolution
        if backend == "kernel":
            self._kernel = KernelBackend()
        else:
            self._kernel = None

        # Compute scene bounds for initial camera placement
        self._scene_center = gaussians.means.mean(dim=0).numpy()
        scene_extent = (
            gaussians.means.max(dim=0).values - gaussians.means.min(dim=0).values
        ).numpy()
        self._camera_distance = float(np.linalg.norm(scene_extent)) * 1.5

        # Create viser server
        self.server = viser.ViserServer(host=host, port=port, verbose=False)
        self.server.scene.world_axes.visible = True

        # GUI: stats display
        self._stats_display = self.server.gui.add_markdown("**FPS:** --")

        # Create nerfview viewer (registers its own on_client_connect internally)
        self.viewer = nerfview.Viewer(
            server=self.server,
            render_fn=self._render_fn,
            mode="rendering",
        )

        # Set initial camera for new clients (registered after nerfview's handler)
        center = self._scene_center
        distance = self._camera_distance

        @self.server.on_client_connect
        def _set_initial_camera(client: viser.ClientHandle) -> None:
            client.camera.position = center + np.array([0.0, 0.0, distance])
            client.camera.look_at = center

        self._running = False
        print(
            f"Viewer running at http://localhost:{port} "
            f"(backend={backend}, max_resolution={max_resolution})",
            flush=True,
        )

    def _render_fn(
        self,
        camera_state: nerfview.CameraState,
        render_tab_state: nerfview.RenderTabState,
    ) -> np.ndarray:
        """Render callback invoked by nerfview for each frame.

        Bridges nerfview's camera state to our rasterization pipeline.
        """
        start = time.perf_counter()
        print(
            f"[render-enter] preview={render_tab_state.preview_render} "
            f"viewer={render_tab_state.viewer_width}x{render_tab_state.viewer_height} "
            f"render_tab={render_tab_state.render_width}x{render_tab_state.render_height}",
            flush=True,
        )

        # Resolve requested render resolution from viser/nerfview.
        if render_tab_state.preview_render:
            req_W = render_tab_state.render_width
            req_H = render_tab_state.render_height
        else:
            req_W = render_tab_state.viewer_width
            req_H = render_tab_state.viewer_height

        if req_W <= 0 or req_H <= 0:
            return np.zeros((max(req_H, 1), max(req_W, 1), 3), dtype=np.uint8)

        # nerfview adapts its render-size requests on the fly: small dims while
        # the camera is moving (low_move state), full viewer_res when static
        # (high state). The returned image MUST match what nerfview asked for
        # — viser/nerfview wires it directly into the client viewport.
        #
        # We only intervene as an upper cap: if a static-frame request would
        # blow past max_resolution, scale it down (preserving aspect) to keep
        # prepare_kernel_inputs from spending seconds at 1920x1080. Then snap
        # to multiples of 32 so the kernel gets whole tiles.
        W, H = req_W, req_H
        if max(W, H) > self.max_resolution:
            scale = self.max_resolution / max(W, H)
            W = int(W * scale)
            H = int(H * scale)
        W = max(32, (W // 32) * 32)
        H = max(32, (H // 32) * 32)

        # Invert c2w to get our W2C extrinsics
        extrinsics = c2w_to_w2c(camera_state.c2w)

        # Intrinsics calibrated to the dims we're actually rendering at.
        intrinsics = torch.tensor(
            camera_state.get_K((W, H)), dtype=torch.float32
        )

        # --- Rasterization pipeline ---
        g = self.gaussians

        t_proj = time.perf_counter()
        means_2d, covs_2d, depths, radii, valid_mask = project_gaussians(
            g.means, g.scales, g.rotations, extrinsics, intrinsics, H, W,
        )
        proj_ms = (time.perf_counter() - t_proj) * 1000.0

        num_visible = valid_mask.sum().item()

        # Early exit: no visible Gaussians
        if num_visible == 0:
            elapsed = time.perf_counter() - start
            print(
                f"[render] req={req_W}x{req_H} -> {W}x{H}  "
                f"sorted=0  visible=0 (early exit)  "
                f"total={elapsed*1000:.0f}ms  backend={self.backend}",
                flush=True,
            )
            self._update_stats(elapsed, W, H, 0)
            return np.zeros((H, W, 3), dtype=np.uint8)

        colors = g.colors[valid_mask]
        opacities = g.opacities[valid_mask]

        # Tile size matches the kernel (32x32) for both backends so CPU and
        # kernel renders are directly comparable.
        tile_size = 32

        t_assign = time.perf_counter()
        gaussian_ids, tile_ids, _ = get_tile_assignments(
            means_2d, radii, H, W, tile_size=tile_size,
        )

        tiles_x = (W + tile_size - 1) // tile_size
        tiles_y = (H + tile_size - 1) // tile_size
        sorted_gaussian_ids, tile_ranges = sort_and_bin(
            gaussian_ids, tile_ids, depths, tiles_x, tiles_y,
        )
        assign_ms = (time.perf_counter() - t_assign) * 1000.0
        print(
            f"[render-mid] visible={num_visible}  tiles={tiles_x*tiles_y}  "
            f"sorted={len(sorted_gaussian_ids)}  "
            f"proj={proj_ms:.0f}ms  assign={assign_ms:.0f}ms",
            flush=True,
        )

        prep_t = time.perf_counter()
        if self.backend == "kernel":
            image_np = self._kernel.render(
                means_2d, covs_2d, colors, opacities,
                sorted_gaussian_ids, tile_ranges, H, W,
            )
            image = torch.from_numpy(image_np)
        else:
            image = alpha_blend(
                means_2d, covs_2d, colors, opacities,
                sorted_gaussian_ids, tile_ranges, H, W, tile_size=tile_size,
            )
        blend_ms = (time.perf_counter() - prep_t) * 1000.0

        # Float [0,1] → uint8 [0,255]
        image_np = (image.clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)

        elapsed = time.perf_counter() - start
        print(
            f"[render] req={req_W}x{req_H} -> {W}x{H}  "
            f"sorted={len(sorted_gaussian_ids)}  "
            f"blend={blend_ms:.0f}ms  total={elapsed*1000:.0f}ms  "
            f"backend={self.backend}",
            flush=True,
        )
        self._update_stats(elapsed, W, H, num_visible)
        return image_np

    def _update_stats(
        self, elapsed: float, width: int, height: int, num_visible: int,
    ) -> None:
        fps = 1.0 / elapsed if elapsed > 0 else 0.0
        self._stats_display.content = (
            f"**FPS:** {fps:.1f} | "
            f"**Res:** {width}x{height} | "
            f"**Visible:** {num_visible:,}"
        )

    def run(self) -> None:
        """Block the main thread to keep the viewer alive."""
        self._running = True
        try:
            while self._running:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nViewer stopped.")
        finally:
            if self._kernel is not None:
                self._kernel.close()
            # viser's websocket thread + thread executor have a teardown race
            # during interpreter shutdown that produces a noisy traceback
            # ("cannot schedule new futures after shutdown"). Hard-exit after
            # our own cleanup so the user sees a clean shell prompt.
            os._exit(0)

    def stop(self) -> None:
        """Signal the viewer to stop."""
        self._running = False
