"""Regression test for viewer render-size stability (flicker/resize bug).

nerfview adaptively downscales `viewer_width/height` every frame while the
camera moves (to hit a target FPS), so the W/H it reports jitters frame to
frame. If our viewer derives the render aspect from those per-frame dims, the
output image size changes every frame and viser stretches each differently
sized frame to the viewport -> visible flicker / resizing. The render size
must instead follow the *stable* camera viewport aspect (`camera_state.aspect`),
which only changes when the browser window is resized.
"""
from types import SimpleNamespace

from gsplat.viewer import GaussianViewer, TILE_SIZE


def _viewer(max_resolution=640):
    # Bypass __init__ (which spins up a viser server); we only exercise the
    # pure render-size resolver.
    v = GaussianViewer.__new__(GaussianViewer)
    v.max_resolution = max_resolution
    return v


def test_render_size_ignores_jittering_viewer_dims():
    """Output (W, H) must follow camera_state.aspect and be invariant to the
    per-frame viewer_width/height nerfview reports.

    The dims below have *wildly different* aspect ratios (portrait, square,
    landscape). The old code derived aspect from them and would return a
    different size for each; the fix derives from the fixed camera aspect, so
    every frame resolves to the same size — feeding such divergent ratios is
    exactly what makes this a true regression guard (the previous version of
    this test fed dims that all matched the camera aspect, so it passed against
    the buggy code too).
    """
    camera_state = SimpleNamespace(aspect=1280 / 960)  # fixed 4:3 viewport
    v = _viewer(max_resolution=640)

    jittering_dims = [(200, 600), (600, 600), (1200, 600), (1800, 600), (853, 640)]
    sizes = set()
    for vw, vh in jittering_dims:
        rts = SimpleNamespace(
            preview_render=False,
            viewer_width=vw, viewer_height=vh,
            render_width=0, render_height=0,
        )
        _, _, W, H = v._resolve_render_size(camera_state, rts)
        sizes.add((W, H))

    # 4:3 -> shorter dim (H) pinned to 640, W = 640*4/3 snapped to a tile.
    assert sizes == {(832, 640)}, f"render size varied with viewer dims: {sizes}"
    (W, H), = sizes
    assert W % TILE_SIZE == 0 and H % TILE_SIZE == 0


def test_render_size_uses_aspect_when_viewer_dims_unset():
    """First-frame case: nerfview may not have reported viewer_width/height yet
    (0x0), but a valid camera aspect must still yield a proper render size, not
    a degenerate 1x1 (guards the aspect-only early-return condition)."""
    camera_state = SimpleNamespace(aspect=1280 / 960)
    v = _viewer(max_resolution=640)
    rts = SimpleNamespace(
        preview_render=False,
        viewer_width=0, viewer_height=0,
        render_width=0, render_height=0,
    )
    _, _, W, H = v._resolve_render_size(camera_state, rts)
    assert (W, H) == (832, 640)


def test_render_size_falls_back_to_viewer_dims_without_camera_aspect():
    """If the client hasn't reported a camera aspect, fall back to the viewer
    dims' ratio rather than bailing."""
    camera_state = SimpleNamespace()  # no .aspect attribute
    v = _viewer(max_resolution=640)
    rts = SimpleNamespace(
        preview_render=False,
        viewer_width=1920, viewer_height=1080,
        render_width=0, render_height=0,
    )
    _, _, W, H = v._resolve_render_size(camera_state, rts)
    assert (W, H) == (1120, 640)  # 16:9 derived from the viewer dims


def test_preview_render_uses_panel_resolution():
    """Camera-path preview render keeps using the panel's explicit dims, not
    the camera aspect."""
    camera_state = SimpleNamespace(aspect=1.0)  # deliberately != panel aspect
    v = _viewer(max_resolution=640)
    rts = SimpleNamespace(
        preview_render=True,
        render_width=1920, render_height=1080,
        viewer_width=0, viewer_height=0,
    )
    _, _, W, H = v._resolve_render_size(camera_state, rts)
    assert (W, H) == (1120, 640)  # 16:9 from the panel dims, ignoring aspect=1.0
