import torch


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