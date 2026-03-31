# 3D Gaussian Splatting - CPU Reference Rasterizer

MSc thesis project implementing the forward-pass rendering pipeline of
[3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
as a CPU reference, with the goal of porting it to
[Tenstorrent tt-metal](https://github.com/tenstorrent/tt-metal) hardware kernels.

The CPU rasterizer serves as the golden reference: it loads a pre-trained `.ply` file
and renders it interactively in the browser via a viser/nerfview viewer.

## Pipeline

```
Load PLY → Project 3D→2D → Tile Assignment → Sort by Depth → Alpha Blend → Display
```

| Stage | Description |
|---|---|
| Load PLY | Parse Gaussian attributes (position, scale, rotation, SH colors, opacity) and apply activations |
| Project | Build 3D covariance, transform to camera space, apply EWA Jacobian, compute 2D screen ellipses, frustum cull |
| Tile assignment | Divide screen into 16×16 tiles, find which tiles each Gaussian's bounding circle overlaps |
| Sort | Sort (gaussian, tile) pairs by composite key (tile_id, depth) |
| Alpha blend | Per-tile front-to-back compositing with transmittance early-exit |

## File Structure

```
main.py              — entry point (argparse → load → viewer)
viewer.py            — GaussianViewer: wraps nerfview/viser, calls rasterization pipeline per frame
rasterization.py     — the four pipeline stages: project_gaussians, get_tile_assignments, sort_and_bin, alpha_blend
loading_gaussians.py — PLY parser, applies exp/sigmoid/SH activations at load time
data_structures.py   — Gaussians dataclass
utils.py             — math: quaternion→rotation matrix, 3D covariance, 2D inverse covariance, Gaussian weight, c2w_to_w2c
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
source venv/bin/activate
python main.py path/to/scene.ply
```

Then open `http://localhost:8080` in a browser. Use the orbit controls to navigate the scene.

Optional arguments:

```bash
python main.py path/to/scene.ply --port 8080 --host 0.0.0.0
```

## Performance

The CPU rasterizer is intentionally simple - it is a reference implementation, not a
production renderer. At 256×256 resolution expect roughly 1–2 seconds per frame.
The bottleneck is the Python tile loop in `alpha_blend` (~70K iterations per frame).

The nerfview viewer automatically reduces resolution during camera movement and renders
at full viewport resolution when the camera is still.
