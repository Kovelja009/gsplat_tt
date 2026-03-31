import torch
import numpy as np


def look_at(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray = np.array([0.0, 1.0, 0.0]),
) -> torch.Tensor:
    """Construct a world-to-camera (W2C) extrinsics matrix.

    Uses the project's camera convention: +X right, +Y down, +Z forward.

    Args:
        eye: (3,) camera position in world space.
        target: (3,) point the camera looks at.
        up: (3,) world up direction (default: +Y up).

    Returns:
        (4, 4) world-to-camera transformation matrix (float32).

    Raises:
        ValueError: if eye == target or up is parallel to the view direction.
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-10:
        raise ValueError("eye and target positions are identical")
    forward /= norm

    right = np.cross(up, forward)
    norm = np.linalg.norm(right)
    if norm < 1e-10:
        raise ValueError("up vector is parallel to the view direction")
    right /= norm

    # Recompute up to ensure orthogonality, then negate for +Y down convention
    down = -np.cross(forward, right)

    # W2C rotation: rows are camera axes (right, down, forward) in world coords
    R = np.stack([right, down, forward], axis=0)
    t = -R @ eye

    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3] = t

    return torch.tensor(w2c, dtype=torch.float32)


def fov_to_intrinsics(
    fov_y_rad: float,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    """Compute camera intrinsics from vertical field of view.

    Assumes square pixels (fx == fy) and principal point at image center.

    Args:
        fov_y_rad: vertical field of view in radians.
        image_width: image width in pixels.
        image_height: image height in pixels.

    Returns:
        (3, 3) intrinsic matrix (float32).
    """
    focal = image_height / (2.0 * np.tan(fov_y_rad / 2.0))
    return torch.tensor([
        [focal, 0.0, image_width / 2.0],
        [0.0, focal, image_height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)


def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    """Invert a camera-to-world matrix to get world-to-camera (our extrinsics).

    Viser/nerfview provides c2w in OpenCV convention (+Z forward, +Y down),
    which matches our rasterizer's convention. We just need to invert.

    Uses the closed-form inverse of a rigid-body transform (R^T, -R^T @ t)
    instead of np.linalg.inv for better numerical stability.

    Args:
        c2w: (4, 4) camera-to-world matrix.

    Returns:
        (4, 4) world-to-camera matrix (float64).
    """
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = -R.T @ t
    return w2c
