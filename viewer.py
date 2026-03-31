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
)


class GaussianViewer:
    """Interactive viewer for 3D Gaussian Splatting scenes.

    Wraps nerfview/viser to display frames rendered by the CPU rasterizer
    with orbit camera controls in a browser.
    """

    def __init__(
        self,
        gaussians: Gaussians,
        host: str = "0.0.0.0",
        port: int = 8080,
    ):
        self.gaussians = gaussians

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
        print(f"Viewer running at http://localhost:{port}")

    def _render_fn(
        self,
        camera_state: nerfview.CameraState,
        render_tab_state: nerfview.RenderTabState,
    ) -> np.ndarray:
        """Render callback invoked by nerfview for each frame.

        Bridges nerfview's camera state to our rasterization pipeline.
        """
        start = time.perf_counter()

        # Resolve render resolution
        if render_tab_state.preview_render:
            W = render_tab_state.render_width
            H = render_tab_state.render_height
        else:
            W = render_tab_state.viewer_width
            H = render_tab_state.viewer_height

        if W <= 0 or H <= 0:
            return np.zeros((max(H, 1), max(W, 1), 3), dtype=np.uint8)

        # Invert c2w to get our W2C extrinsics
        extrinsics = c2w_to_w2c(camera_state.c2w)

        # Intrinsics from nerfview's camera FOV
        intrinsics = torch.tensor(
            camera_state.get_K((W, H)), dtype=torch.float32
        )

        # --- Rasterization pipeline ---
        g = self.gaussians

        means_2d, covs_2d, depths, radii, valid_mask = project_gaussians(
            g.means, g.scales, g.rotations, extrinsics, intrinsics, H, W,
        )

        num_visible = valid_mask.sum().item()

        # Early exit: no visible Gaussians
        if num_visible == 0:
            self._update_stats(time.perf_counter() - start, W, H, 0)
            return np.zeros((H, W, 3), dtype=np.uint8)

        colors = g.colors[valid_mask]
        opacities = g.opacities[valid_mask]

        gaussian_ids, tile_ids, _ = get_tile_assignments(
            means_2d, radii, H, W,
        )

        tiles_x = (W + 15) // 16
        tiles_y = (H + 15) // 16
        sorted_gaussian_ids, tile_ranges = sort_and_bin(
            gaussian_ids, tile_ids, depths, tiles_x, tiles_y,
        )

        image = alpha_blend(
            means_2d, covs_2d, colors, opacities,
            sorted_gaussian_ids, tile_ranges, H, W,
        )

        # Float [0,1] → uint8 [0,255]
        image_np = (image.clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)

        self._update_stats(time.perf_counter() - start, W, H, num_visible)
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

    def stop(self) -> None:
        """Signal the viewer to stop."""
        self._running = False
