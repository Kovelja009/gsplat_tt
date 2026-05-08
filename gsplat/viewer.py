"""Browser-based interactive viewer for 3D Gaussian Splatting scenes."""
import os
import time

import numpy as np
import torch
import viser
import nerfview

from backends import get_backend
from gsplat.data_structures import Gaussians
from gsplat.pipeline import Pipeline, RenderResult, format_timings
from gsplat.utils import c2w_to_w2c


# Tile size matches the kernel (32x32) for both backends so CPU and
# kernel renders use the same tiling and are directly comparable.
TILE_SIZE = 32


class GaussianViewer:
    """Interactive viewer for 3D Gaussian Splatting scenes.

    Wraps nerfview/viser around a `gsplat.pipeline.Pipeline`. The pipeline
    owns the chosen backend and produces a `RenderResult` for every frame,
    including per-stage timings — so adding a new backend (CUDA, ...)
    requires no changes here as long as it's registered in
    `backends/REGISTRY`.
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
        self.backend_name = backend
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

        # Spin up the backend lazily through the registry, then wrap it in a
        # Pipeline so per-stage timing happens automatically.
        self.pipeline = Pipeline(get_backend(backend, verbose=verbose),
                                 tile_size=TILE_SIZE)

        # Compute scene bounds for initial camera placement.
        # Naive mean / min-max over ALL Gaussians is fragile: trained 3DGS
        # scenes routinely contain a small number of low-opacity outliers
        # at extreme positions, which inflate the bounding box and push the
        # camera so far back that the visible content collapses to a few
        # center tiles. We instead use percentile bounds restricted to
        # well-opaque Gaussians, which closely tracks the visible content.
        means = gaussians.means.numpy()
        opacities = gaussians.opacities.numpy()
        visible = means[opacities > 0.1]
        if visible.shape[0] < 100:
            visible = means  # fallback for synthetic / very sparse scenes
        lo = np.percentile(visible, 5, axis=0)
        hi = np.percentile(visible, 95, axis=0)
        self._scene_center = (lo + hi) * 0.5
        self._camera_distance = float(np.linalg.norm(hi - lo)) * 1.2

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
        wall_start = time.perf_counter()
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

        # One pipeline call covers project → tile_assign → sort → blend.
        # `result.timings` already has per-stage wall-clock; the backend may
        # also have populated `result.sub_timings` (e.g. blend.kernel_run).
        result = self.pipeline.render(self.gaussians, extrinsics, intrinsics, H, W)

        # Convert pipeline output → uint8 image for nerfview/viser.
        if result.image is None:
            image_np = np.zeros((H, W, 3), dtype=np.uint8)
        else:
            image_np = (np.clip(result.image, 0.0, 1.0) * 255).astype(np.uint8)

        wall_elapsed = time.perf_counter() - wall_start
        if self.verbose:
            self._log_verbose(req_W, req_H, W, H, result, wall_elapsed)
        self._update_stats(wall_elapsed, W, H, result.num_visible)
        return image_np

    def _log_verbose(
        self,
        req_W: int,
        req_H: int,
        W: int,
        H: int,
        result: RenderResult,
        wall_elapsed: float,
    ) -> None:
        print(
            f"[render] req={req_W}x{req_H} -> {W}x{H}  "
            f"visible={result.num_visible}  sorted={result.num_entries}  "
            f"backend={self.backend_name}",
            flush=True,
        )
        print(format_timings(result), flush=True)
        # Wall-clock is total inside _render_fn (incl. resize-resolve etc.);
        # log alongside the pipeline-internal total for sanity checking.
        print(f"[wall]  {wall_elapsed * 1000:6.1f} ms", flush=True)

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
            self.pipeline.close()
            # viser's websocket thread + thread executor have a teardown race
            # during interpreter shutdown that produces a noisy traceback
            # ("cannot schedule new futures after shutdown"). Hard-exit after
            # our own cleanup so the user sees a clean shell prompt.
            os._exit(0)

    def stop(self) -> None:
        """Signal the viewer to stop."""
        self._running = False
