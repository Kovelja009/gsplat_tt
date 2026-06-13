"""Regression test for viewer render-size stability (flicker/resize bug).

nerfview adaptively downscales `viewer_width/height` every frame while the
camera moves (to hit a target FPS), and the W/H ratio it writes jitters
frame-to-frame due to independent rounding. If our viewer derives the render
aspect from those jittering dims, our output image size changes every frame
and viser stretches each differently-sized frame to the viewport -> visible
flicker / resizing. The render size must instead follow the *stable* camera
viewport aspect, which only changes when the browser window is resized.
"""
from types import SimpleNamespace

from gsplat.viewer import GaussianViewer, TILE_SIZE


def _viewer(max_resolution=640):
    # Bypass __init__ (which spins up a viser server); we only exercise the
    # pure render-size resolver.
    v = GaussianViewer.__new__(GaussianViewer)
    v.max_resolution = max_resolution
    return v


def _nerfview_low_move_dims(true_aspect, heights):
    """Reproduce nerfview's _get_img_wh low_move output: H rounded to 10,
    W = int(H * aspect). Yields the (viewer_width, viewer_height) it writes."""
    for h in heights:
        yield int(h * true_aspect), h


def test_render_size_stable_under_nerfview_jitter():
    """Output (W, H) must not change frame-to-frame when only nerfview's
    adaptive downscale resolution jitters but the camera aspect is fixed."""
    true_aspect = 1280 / 960  # a non-round 4:3 aspect that rounds unevenly
    camera_state = SimpleNamespace(aspect=true_aspect)
    v = _viewer(max_resolution=640)

    sizes = set()
    for vw, vh in _nerfview_low_move_dims(true_aspect, [300, 310, 290, 320, 280, 470]):
        rts = SimpleNamespace(
            preview_render=False,
            viewer_width=vw, viewer_height=vh,
            render_width=0, render_height=0,
        )
        _, _, W, H = v._resolve_render_size(camera_state, rts)
        sizes.add((W, H))

    assert len(sizes) == 1, f"render size jittered across frames: {sizes}"
    (W, H), = sizes
    assert W % TILE_SIZE == 0 and H % TILE_SIZE == 0
    # Landscape: shorter dim (H) pinned to max_resolution, snapped to tiles.
    assert H == 640
    assert W == (int(640 * true_aspect) // TILE_SIZE) * TILE_SIZE


def test_preview_render_uses_panel_resolution():
    """Camera-path preview render keeps using the panel's explicit dims."""
    camera_state = SimpleNamespace(aspect=1.0)
    v = _viewer(max_resolution=640)
    rts = SimpleNamespace(
        preview_render=True,
        render_width=1920, render_height=1080,
        viewer_width=0, viewer_height=0,
    )
    _, _, W, H = v._resolve_render_size(camera_state, rts)
    assert H == 640
    assert W == (int(640 * (1920 / 1080)) // TILE_SIZE) * TILE_SIZE
