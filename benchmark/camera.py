"""Scene framing: median-center / p60-radius look-at for a point cloud.

Copied (not imported) from scripts/bench_scene.py to keep benchmark/
independent of scripts/.
"""
from __future__ import annotations

import numpy as np
import torch

from gsplat.utils import c2w_to_w2c


def look_at_c2w(center, eye, up=(0.0, -1.0, 0.0)):
    """OpenCV-convention c2w (+Z forward, +Y down) for a camera at `eye`."""
    center = np.asarray(center, np.float64); eye = np.asarray(eye, np.float64)
    up = np.asarray(up, np.float64)
    z = center - eye; z /= np.linalg.norm(z)          # forward
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(np.array([0.0, 0.0, 1.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)                                  # down
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = x; c2w[:3, 1] = y; c2w[:3, 2] = z; c2w[:3, 3] = eye
    return c2w


def make_camera(means, H, W, fov_deg=60.0):
    """Robust look-at framing the inlier cloud (median center, p60 radius)."""
    m = means.numpy()
    center = np.median(m, axis=0)
    dist = np.linalg.norm(m - center, axis=1)
    radius = float(np.percentile(dist, 60))
    eye = center + np.array([0.0, 0.0, 2.5 * radius], np.float32)
    c2w = look_at_c2w(center, eye)
    extrinsics = c2w_to_w2c(c2w)
    f = 0.5 * W / np.tan(0.5 * np.radians(fov_deg))
    intrinsics = torch.tensor(
        [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=torch.float32
    )
    return extrinsics, intrinsics
