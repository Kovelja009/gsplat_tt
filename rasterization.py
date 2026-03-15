import torch
from dataclasses import dataclass

@dataclass
class Camera:
    extrinsics: torch.Tensor  # (4, 4) world-to-camera transform: [[R  | t],
                              #                                    [0  | 1]]
                              # R = 3x3 rotation, t = 3x1 translation
    intrinsics: torch.Tensor  # (3, 3) intrinsic matrix: [[fx, 0, cx],
                              #                           [0, fy, cy],
                              #                           [0, 0, 1]]
    image_height: int
    image_width: int


def quat_to_rotation_matrix(quats: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternions to 3x3 rotation matrices.

    A quaternion q = q0 + q1*i + q2*j + q3*k encodes a 3D rotation where:
    - q0 is the scalar (real) part, related to the rotation angle: q0 = cos(θ/2)
    - (q1, q2, q3) is the vector (imaginary) part, related to the rotation axis:
      (q1, q2, q3) = sin(θ/2) * axis

    For a unit quaternion (|q| = 1), this produces a proper rotation matrix
    (orthogonal, determinant = 1).

    Args:
        quats: (N, 4) tensor of unit quaternions in (q0, q1, q2, q3) order.

    Returns:
        (N, 3, 3) rotation matrices.
    """
    q0, q1, q2, q3 = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

    R = torch.stack([
        1 - 2*(q2*q2 + q3*q3),  2*(q1*q2 - q0*q3),      2*(q1*q3 + q0*q2),
        2*(q1*q2 + q0*q3),      1 - 2*(q1*q1 + q3*q3),  2*(q2*q3 - q0*q1),
        2*(q1*q3 - q0*q2),      2*(q2*q3 + q0*q1),      1 - 2*(q1*q1 + q2*q2),
    ], dim=-1).reshape(-1, 3, 3)

    return R


def build_covariance_3d(scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """Build 3D covariance matrices from scales and rotation quaternions.

    Each 3D Gaussian is parameterized by an axis-aligned ellipsoid (scales)
    rotated into world space (quaternion). The covariance encodes this shape:

        Σ = R · S · S^T · R^T

    where S = diag(scale_x, scale_y, scale_z) and R is the rotation matrix.

    Args:
        scales: (N, 3) actual scale values.
        rotations: (N, 4) unit quaternions (w, x, y, z).

    Returns:
        (N, 3, 3) covariance matrices.
    """
    R = quat_to_rotation_matrix(rotations)  # (N, 3, 3)

    # S = diag(scales), so R @ S is equivalent to element-wise multiplication of R by S.
    RS = R * scales.unsqueeze(1)  # (N, 3, 3) * (N, 1, 3) -> (N, 3, 3)

    # Σ = RS @ RS^T
    return RS @ RS.transpose(1, 2)  # (N, 3, 3)



def project_gaussians(
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project 3D Gaussians to 2D screen-space ellipses.

    This implements the EWA (Elliptical Weighted Average) splatting approach:
    1. Build 3D covariance from scale + rotation
    2. Transform Gaussian centers to camera space using the view matrix
    3. Cull Gaussians behind the camera or outside the frustum
    4. Approximate the projective transform with a first-order Taylor expansion (Jacobian)
    5. Project 3D covariance to 2D using: Σ_2D = J @ W @ Σ_3D @ W^T @ J^T
    6. Compute screen-space position using the intrinsic matrix

    Args:
        means: (N, 3) Gaussian centers in world space.
        scales: (N, 3) Gaussian scales.
        rotations: (N, 4) unit quaternions (w, x, y, z).
        extrinsics: (4, 4) world-to-camera transformation matrix.
        intrinsics: (3, 3) camera intrinsic matrix.
        image_height: output image height in pixels.
        image_width: output image width in pixels.

    Returns:
        means_2d: (M, 2) screen-space positions of visible Gaussians.
        covs_2d: (M, 2, 2) 2D covariance matrices.
        depths: (M,) camera-space depth values.
        radii: (M,) bounding circle radius in pixels (at 3σ).
        valid_mask: (N,) boolean mask indicating which input Gaussians are visible.
    """
    N = means.shape[0]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # --- Step 1: Build 3D covariance matrices ---
    cov3d = build_covariance_3d(scales, rotations)  # (N, 3, 3)

    # --- Step 2: Transform Gaussian centers to camera space ---
    # p_cam = R @ p_world + t, using extrinsics directly
    R = extrinsics[:3, :3]  # (3, 3)
    t = extrinsics[:3, 3]   # (3,)
    means_cam = means @ R.T + t  # (N, 3) @ (3, 3) + (3,) -> (N, 3)

    # --- Step 3: Frustum culling ---
    # Keep only Gaussians in front of the near plane
    near = 0.2
    valid_mask = means_cam[:, 2] > near  # z > near plane

    # --- Step 4: Compute screen-space positions (exact pinhole projection) ---
    tx, ty, tz = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]
    means_2d = torch.stack([fx * tx / tz + cx, fy * ty / tz + cy], dim=-1)  # (N, 2)

    # --- Step 5: Compute the Jacobian of the perspective projection ---
    # The perspective projection maps (x, y, z) -> (fx*x/z + cx, fy*y/z + cy).
    # Its Jacobian at point (x, y, z) is:
    #   J = [[fx/z,  0,   -fx*x/z²],
    #        [0,     fy/z, -fy*y/z²]]
    # This linearizes the nonlinear projection around each Gaussian's center, 
    # which is a good approximation if the size of the Gaussian is small 
    # compared to the distance to the camera.
    tz2 = tz * tz

    J = torch.zeros(N, 2, 3, dtype=means.dtype)
    J[:, 0, 0] = fx / tz
    J[:, 0, 2] = -fx * tx / tz2
    J[:, 1, 1] = fy / tz
    J[:, 1, 2] = -fy * ty / tz2

    # --- Step 6: Project 3D covariance to 2D ---
    # Σ_2D = J @ R @ Σ_3D @ R^T @ J^T
    # First transform to camera space: Σ_cam = R @ Σ_3D @ R^T
    cov_cam = R @ cov3d @ R.T  # (3, 3) @ (N, 3, 3) @ (3, 3) -> (N, 3, 3)

    # Then apply Jacobian: Σ_2D = J @ Σ_cam @ J^T
    covs_2d = J @ cov_cam @ J.transpose(1, 2)  # (N, 2, 2)

    # Add low-pass filter for anti-aliasing (0.3px variance, per original implementation)
    covs_2d[:, 0, 0] += 0.3
    covs_2d[:, 1, 1] += 0.3

    # --- Step 7: Compute bounding radii from 2D covariance eigenvalues ---
    # For a 2x2 symmetric matrix [[a, b], [b, c]], the max eigenvalue is:
    #   λ_max = ((a+c) + sqrt((a-c)² + 4b²)) / 2
    a = covs_2d[:, 0, 0]
    b = covs_2d[:, 0, 1]
    c = covs_2d[:, 1, 1]
    det = (a - c) ** 2 + 4 * b ** 2
    lambda_max = 0.5 * (a + c + torch.sqrt(torch.clamp(det, min=0.0)))
    radii = torch.ceil(3.0 * torch.sqrt(lambda_max))

    # Also cull Gaussians that project entirely outside the screen
    valid_mask = valid_mask & (means_2d[:, 0] + radii > 0)
    valid_mask = valid_mask & (means_2d[:, 0] - radii < image_width)
    valid_mask = valid_mask & (means_2d[:, 1] + radii > 0)
    valid_mask = valid_mask & (means_2d[:, 1] - radii < image_height)
    valid_mask = valid_mask & (radii > 0)

    depths = means_cam[valid_mask, 2]
    
    return means_2d[valid_mask], covs_2d[valid_mask], depths, radii[valid_mask], valid_mask
