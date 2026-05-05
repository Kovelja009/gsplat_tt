"""Browser-based interactive viewer for 3D Gaussian Splatting scenes."""
import os
import time

import numpy as np
import torch
import viser
import nerfview

from data_structures import Gaussians
from kernel_backend import KernelBackend
from utils import c2w_to_w2c
from rasterization import (
    project_gaussians,
    get_tile_assignments,
    sort_and_bin,
    alpha_blend,
)


# Tile size matches the kernel (32x32) for both backends so CPU and
# kernel renders use the same tiling and are directly comparable.
TILE_SIZE = 32


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
        adaptive_resolution: bool = False,
        verbose: bool = False,
    ):
        self.gaussians = gaussians
        self.backend = backend  # "cpu" or "kernel"
        # Cap longest output dim. Browser tabs ask for arbitrary sizes (often
        # 1920x1080); without a cap, prepare_kernel_inputs takes seconds and
        # the viewer feels frozen. Stretching the displayed image is fine for
        # interactive preview.
        self.max_resolution = max_resolution
        # If False (default), always render at max_resolution on the longest
        # dim regardless of nerfview's adaptive-downsample requests. Looks
        # better, dragging is choppier. If True, honor nerfview's smaller
        # requests during camera movement (smoother drag, pixelated previews).
        self.adaptive_resolution = adaptive_resolution
        self.verbose = verbose
        if backend == "kernel":
            self._kernel = KernelBackend(verbose=verbose)
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
        flags = [f"backend={backend}", f"max_resolution={max_resolution}"]
        if adaptive_resolution:
            flags.append("adaptive_resolution")
        if verbose:
            flags.append("verbose")
        print(
            f"Viewer running at http://localhost:{port} ({', '.join(flags)})",
            flush=True,
        )

    def _resolve_render_size(
        self, render_tab_state: nerfview.RenderTabState
    ) -> tuple[int, int, int, int]:
        """Pick (W, H) for this frame.

        nerfview adapts its requests on the fly: small dims during camera
        movement (low_move state), full viewer_res when static. We use
        `req_W / req_H` only as the aspect ratio; the actual render size
        depends on `adaptive_resolution`:

          - False (default): always render at max_resolution on the longest
            dim (preserving aspect). Every frame is full-quality; dragging
            is choppier.
          - True: honor whatever nerfview asks for, capped at max_resolution
            on the longest dim. Smooth dragging via downsampled previews;
            full quality when static.

        Snaps both dims to multiples of 32 so the kernel gets whole tiles.

        Returns (req_W, req_H, W, H) — the original request alongside the
        resolved size, useful for logging.
        """
        if render_tab_state.preview_render:
            req_W = render_tab_state.render_width
            req_H = render_tab_state.render_height
        else:
            req_W = render_tab_state.viewer_width
            req_H = render_tab_state.viewer_height

        if req_W <= 0 or req_H <= 0:
            return req_W, req_H, max(req_W, 1), max(req_H, 1)

        if self.adaptive_resolution:
            # Honor nerfview's adaptive size, capped from above.
            W, H = req_W, req_H
            if max(W, H) > self.max_resolution:
                scale = self.max_resolution / max(W, H)
                W = int(W * scale)
                H = int(H * scale)
        else:
            # Always target max_resolution on the longest dim, keep aspect.
            aspect = req_W / req_H
            if aspect >= 1.0:
                W = self.max_resolution
                H = int(self.max_resolution / aspect)
            else:
                H = self.max_resolution
                W = int(self.max_resolution * aspect)

        W = max(TILE_SIZE, (W // TILE_SIZE) * TILE_SIZE)
        H = max(TILE_SIZE, (H // TILE_SIZE) * TILE_SIZE)
        return req_W, req_H, W, H

    def _render_fn(
        self,
        camera_state: nerfview.CameraState,
        render_tab_state: nerfview.RenderTabState,
    ) -> np.ndarray:
        """Render callback invoked by nerfview for each frame."""
        start = time.perf_counter()
        if self.verbose:
            print(
                f"[render-enter] preview={render_tab_state.preview_render} "
                f"viewer={render_tab_state.viewer_width}x{render_tab_state.viewer_height} "
                f"render_tab={render_tab_state.render_width}x{render_tab_state.render_height}",
                flush=True,
            )

        req_W, req_H, W, H = self._resolve_render_size(render_tab_state)
        if W <= 0 or H <= 0:
            return np.zeros((max(req_H, 1), max(req_W, 1), 3), dtype=np.uint8)

        extrinsics = c2w_to_w2c(camera_state.c2w)
        intrinsics = torch.tensor(camera_state.get_K((W, H)), dtype=torch.float32)

        g = self.gaussians

        t_proj = time.perf_counter()
        means_2d, covs_2d, depths, radii, valid_mask = project_gaussians(
            g.means, g.scales, g.rotations, extrinsics, intrinsics, H, W,
        )
        proj_ms = (time.perf_counter() - t_proj) * 1000.0

        num_visible = int(valid_mask.sum().item())

        if num_visible == 0:
            elapsed = time.perf_counter() - start
            if self.verbose:
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

        t_assign = time.perf_counter()
        gaussian_ids, tile_ids, _ = get_tile_assignments(
            means_2d, radii, H, W, tile_size=TILE_SIZE,
        )
        tiles_x = (W + TILE_SIZE - 1) // TILE_SIZE
        tiles_y = (H + TILE_SIZE - 1) // TILE_SIZE
        sorted_gaussian_ids, tile_ranges = sort_and_bin(
            gaussian_ids, tile_ids, depths, tiles_x, tiles_y,
        )
        assign_ms = (time.perf_counter() - t_assign) * 1000.0
        if self.verbose:
            print(
                f"[render-mid] visible={num_visible}  tiles={tiles_x*tiles_y}  "
                f"sorted={len(sorted_gaussian_ids)}  "
                f"proj={proj_ms:.0f}ms  assign={assign_ms:.0f}ms",
                flush=True,
            )

        t_blend = time.perf_counter()
        if self.backend == "kernel":
            image_np = self._kernel.render(
                means_2d, covs_2d, colors, opacities,
                sorted_gaussian_ids, tile_ranges, H, W,
            )
            image = torch.from_numpy(image_np)
        else:
            image = alpha_blend(
                means_2d, covs_2d, colors, opacities,
                sorted_gaussian_ids, tile_ranges, H, W, tile_size=TILE_SIZE,
            )
        blend_ms = (time.perf_counter() - t_blend) * 1000.0

        image_np = (image.clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)

        elapsed = time.perf_counter() - start
        if self.verbose:
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
