"""Browser-based interactive viewer for 3D Gaussian Splatting scenes."""
import os
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

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

# Pipeline stages, in execution order — used to lay out the benchmark table.
_STAGE_KEYS = ("project", "tile_assign", "sort", "blend")


class _FrameSample(NamedTuple):
    """One frame's per-stage timings plus the (W, H) it was rendered at."""
    timings: dict[str, float]
    sub_timings: dict[str, float]
    width: int
    height: int


def _median_by_key(
    rows: list[dict[str, float]], keys: list[str] | tuple[str, ...]
) -> dict[str, float]:
    """Median of `row[key]` across rows, skipping rows where the key is absent."""
    out: dict[str, float] = {}
    for key in keys:
        values = [r[key] for r in rows if key in r]
        if values:
            out[key] = statistics.median(values)
    return out


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
        verbose: bool = False,
        scene_path: str | None = None,
    ):
        self.gaussians = gaussians
        self.backend_name = backend
        # Path to the .ply this viewer was launched with — used as the scene
        # name in the benchmark markdown filename (None for synthetic data).
        self.scene_path = scene_path
        # Shorter render dim, in pixels (e.g. 720 ≈ 720p). The longer dim
        # follows from the browser's aspect ratio; rendering at native browser
        # size makes prepare_kernel_inputs take seconds and the viewer feels
        # frozen. Stretching the displayed image is fine for interactive use.
        self.max_resolution = max_resolution
        self.verbose = verbose

        # Spin up the backend lazily through the registry, then wrap it in a
        # Pipeline so per-stage timing happens automatically.
        self.pipeline = Pipeline(get_backend(backend, verbose=verbose),
                                 tile_size=TILE_SIZE)

        # Per-frame timing log; aggregated into a markdown report on shutdown.
        # Empty frames (no visible Gaussians) are skipped so they don't pull
        # the median toward zero.
        self._frame_samples: list[_FrameSample] = []
        self._session_start = datetime.now()

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
        if verbose:
            flags.append("verbose")
        print(
            f"Viewer running at http://localhost:{port} ({', '.join(flags)})",
            flush=True,
        )

    def _resolve_render_size(
        self,
        camera_state: nerfview.CameraState,
        render_tab_state: nerfview.RenderTabState,
    ) -> tuple[int, int, int, int]:
        """Pick (W, H) for this frame.

        max_resolution sets the shorter dim (480p/720p/1080p convention);
        the longer dim follows from the camera's aspect ratio. Both dims
        are snapped down to multiples of TILE_SIZE so the kernel gets whole
        tiles.

        The aspect ratio comes from the *stable* camera viewport
        (`camera_state.aspect`, which only changes on browser-window resize),
        NOT from nerfview's `viewer_width/height`. nerfview adaptively
        downscales those every frame to hit a target FPS, and the W/H ratio it
        writes jitters frame-to-frame; deriving our render size from them made
        the output image resize on consecutive frames, which viser stretches to
        the viewport as visible flicker. Camera-path *preview* renders keep
        using the render panel's explicit dims (the user picked that aspect).

        Returns (req_W, req_H, W, H) — the original request alongside the
        resolved size, useful for logging.
        """
        if render_tab_state.preview_render:
            req_W = render_tab_state.render_width
            req_H = render_tab_state.render_height
            aspect = req_W / req_H if req_H > 0 else 0.0
        else:
            req_W = render_tab_state.viewer_width
            req_H = render_tab_state.viewer_height
            # Stable viewport aspect; fall back to requested dims if the client
            # hasn't reported its camera aspect yet.
            aspect = getattr(camera_state, "aspect", 0.0) or (
                req_W / req_H if req_H > 0 else 0.0
            )

        # The render size is derived purely from `aspect` + max_resolution, so
        # a valid aspect is sufficient even if nerfview hasn't reported
        # viewer_width/height yet (e.g. the first frame). Only bail when we
        # have no usable aspect at all.
        if aspect <= 0:
            return req_W, req_H, max(req_W, 1), max(req_H, 1)

        if aspect >= 1.0:  # landscape: H is the shorter dim
            H = self.max_resolution
            W = int(self.max_resolution * aspect)
        else:  # portrait: W is the shorter dim
            W = self.max_resolution
            H = int(self.max_resolution / aspect)

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

        req_W, req_H, W, H = self._resolve_render_size(camera_state, render_tab_state)
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
            # Empty frames (no visible Gaussians) short-circuit the pipeline
            # and would skew the median toward zero, so they're not recorded.
            self._frame_samples.append(_FrameSample(
                timings=dict(result.timings),
                sub_timings=dict(result.sub_timings),
                width=W,
                height=H,
            ))

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

    @property
    def _scene_name(self) -> str:
        return Path(self.scene_path).stem if self.scene_path else "synthetic"

    def _aggregate_session_medians(
        self,
    ) -> tuple[dict[str, float], dict[str, float], list[str], float, tuple[int, int]]:
        """Aggregate per-stage / sub-timing / total medians and modal (W, H)."""
        stage_rows = [s.timings for s in self._frame_samples]
        sub_rows = [s.sub_timings for s in self._frame_samples]
        sub_keys = list(dict.fromkeys(k for s in sub_rows for k in s))

        stage_medians = _median_by_key(stage_rows, _STAGE_KEYS)
        sub_medians = _median_by_key(sub_rows, sub_keys)
        median_total = _median_by_key(stage_rows, ("total",)).get("total", 0.0)
        modal_resolution = statistics.mode(
            (s.width, s.height) for s in self._frame_samples
        )
        return stage_medians, sub_medians, sub_keys, median_total, modal_resolution

    def _benchmark_filename(self) -> str:
        ts = self._session_start.strftime("%Y-%m-%d_%H-%M-%S")
        return f"{self._scene_name}_{self.backend_name}_{self.max_resolution}_{ts}.md"

    def _render_benchmark_md(
        self,
        stage_medians: dict[str, float],
        sub_medians: dict[str, float],
        sub_keys: list[str],
        median_total: float,
        modal_resolution: tuple[int, int],
    ) -> str:
        fps = 1000.0 / median_total if median_total > 0 else 0.0
        ts = self._session_start
        res_w, res_h = modal_resolution
        resolution = f"{res_w}x{res_h} (max-resolution={self.max_resolution})"

        rows = ["| Stage | ms |", "|---|---|"]
        for stage in _STAGE_KEYS:
            if stage in stage_medians:
                rows.append(f"| {stage} | {stage_medians[stage]:.2f} |")
            for sub in sub_keys:
                if not sub.startswith(f"{stage}.") or sub not in sub_medians:
                    continue
                # Dotted depth = nesting level under the stage; e.g.
                # "blend.daemon_rt.device_kernel" is depth 2 → indent twice.
                # Markdown table renderers collapse leading whitespace inside
                # cells, so we use &nbsp; to make the indent survive.
                depth = sub.count(".")
                leaf = sub.rsplit(".", 1)[-1]
                indent = "&nbsp;" * 4 * depth
                rows.append(f"| {indent}└─ {leaf} | {sub_medians[sub]:.2f} |")
        rows.append(f"| **Total** | **{median_total:.2f}** |")
        rows.append(f"| **FPS** | **{fps:.2f}** |")

        return "\n".join([
            f"# Benchmark: {self._scene_name}",
            "",
            f"- **Date:** {ts.strftime('%Y-%m-%d')}",
            f"- **Time:** {ts.strftime('%H:%M:%S')}",
            f"- **Backend:** {self.backend_name}",
            f"- **Scene:** {self.scene_path or '(synthetic)'}",
            f"- **Gaussians:** {self.gaussians.num_gaussians:,}",
            f"- **Resolution:** {resolution}",
            f"- **Frames sampled:** {len(self._frame_samples)}",
            "",
            "## Performance (median across frames)",
            "",
            *rows,
            "",
        ])

    def _write_benchmark(self) -> None:
        if not self._frame_samples:
            return
        out_dir = Path("benchmarks")
        out_dir.mkdir(exist_ok=True)
        path = out_dir / self._benchmark_filename()
        path.write_text(self._render_benchmark_md(*self._aggregate_session_medians()))
        print(f"Benchmark written to {path}", flush=True)

    def run(self) -> None:
        """Block the main thread to keep the viewer alive."""
        self._running = True
        try:
            while self._running:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nViewer stopped.")
        finally:
            self._write_benchmark()
            self.pipeline.close()
            # viser's websocket thread + thread executor have a teardown race
            # during interpreter shutdown that produces a noisy traceback
            # ("cannot schedule new futures after shutdown"). Hard-exit after
            # our own cleanup so the user sees a clean shell prompt.
            os._exit(0)

    def stop(self) -> None:
        """Signal the viewer to stop."""
        self._running = False
