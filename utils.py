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


def calc_det(covs_2d: torch.Tensor) -> torch.Tensor:
    """Compute determinant of 2x2 symmetric covariance matrices.

    For [[a, b], [b, c]], det = a*c - b².
    Clamped to 1e-6 to avoid division by zero when inverting.

    Args:
        covs_2d: (N, 2, 2) covariance matrices.

    Returns:
        (N,) determinants.
    """
    a = covs_2d[:, 0, 0]
    b = covs_2d[:, 0, 1]
    c = covs_2d[:, 1, 1]
    return torch.clamp(a * c - b * b, min=1e-6)


def inverse_cov_2d(covs_2d: torch.Tensor) -> torch.Tensor:
    """Compute inverse of 2x2 symmetric covariance matrices.

    For [[a, b], [b, c]], the inverse is (1/det) * [[c, -b], [-b, a]].

    Args:
        covs_2d: (N, 2, 2) covariance matrices.

    Returns:
        (N, 2, 2) inverse covariance matrices.
    """
    det = calc_det(covs_2d)

    a = covs_2d[:, 0, 0]
    b = covs_2d[:, 0, 1]
    c = covs_2d[:, 1, 1]

    cov_inv = torch.zeros_like(covs_2d)
    cov_inv[:, 0, 0] = c / det
    cov_inv[:, 0, 1] = -b / det
    cov_inv[:, 1, 0] = -b / det
    cov_inv[:, 1, 1] = a / det

    return cov_inv


def calc_gaussian_strength(
    dx: torch.Tensor,
    dy: torch.Tensor,
    cov_inv: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the Gaussian weight at pixel displacements from the center.

    Computes exp(-0.5 * d^T @ Σ⁻¹ @ d) for each pixel, which is the Gaussian's
    contribution strength (1.0 at center, falling off with distance).
    This is the Mahalanobis distance expanded into scalar form:
        -0.5 * (inv_00*dx² + 2*inv_01*dx*dy + inv_11*dy²)

    Args:
        dx: (...) x-displacement from Gaussian center per pixel.
        dy: (...) y-displacement from Gaussian center per pixel.
        cov_inv: (2, 2) inverse covariance matrix for this Gaussian.

    Returns:
        (...) Gaussian strength values in (0, 1].
    """
    power = -0.5 * (
        cov_inv[0, 0] * dx * dx
        + 2.0 * cov_inv[0, 1] * dx * dy
        + cov_inv[1, 1] * dy * dy
    )
    # Clamp to <= 0: positive values are numerical errors (Mahalanobis distance is always >= 0)
    power = torch.clamp(power, max=0.0)
    return torch.exp(power)