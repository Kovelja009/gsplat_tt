import torch

from utils import build_covariance_3d


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


def get_tile_assignments(
    means_2d: torch.Tensor,
    radii: torch.Tensor,
    image_height: int,
    image_width: int,
    tile_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign each visible Gaussian to the screen tiles it overlaps.

    The screen is divided into a grid of tile_size x tile_size pixel tiles.
    Each Gaussian's bounding circle (center + radius) is tested against the tile grid
    to find all tiles it touches. This produces a list of (gaussian_idx, tile_id) pairs
    that tells us which Gaussians contribute to which tiles.

    Args:
        means_2d: (M, 2) screen-space positions of visible Gaussians.
        radii: (M,) bounding circle radius in pixels (at 3σ).
        image_height: output image height in pixels.
        image_width: output image width in pixels.
        tile_size: tile dimension in pixels (default 16x16).

    Returns:
        gaussian_ids: (P,) index into means_2d for each Gaussian-tile pair.
        tile_ids: (P,) flat tile index for each Gaussian-tile pair.
        tiles_per_gaussian: (M,) how many tiles each Gaussian overlaps (useful for debugging).
    """
    tiles_x = (image_width + tile_size - 1) // tile_size
    tiles_y = (image_height + tile_size - 1) // tile_size

    """ NOTE: radii is derived from lambda_max (largest eigenvalue), so the bounding region
    is a circle, not an ellipse. For elongated Gaussians this overestimates — some tiles
    will be assigned even though the Gaussian barely reaches them. This is conservative
    (no visual errors), just slightly more work in alpha blending where those pixels will
    evaluate to near-zero alpha and get skipped. A tighter approach would use an AABB from
    the covariance diagonal: extent_x = 3*sqrt(cov[0,0]), extent_y = 3*sqrt(cov[1,1]),
    which is what the original CUDA implementation (diff-gaussian-rasterization) uses.
    """
    # Compute the tile range each Gaussian's bounding box covers.
    # Clamp to valid tile indices [0, tiles_x) and [0, tiles_y)
    tile_min_x = torch.clamp((means_2d[:, 0] - radii) / tile_size, min=0, max=tiles_x - 1).int()
    tile_max_x = torch.clamp((means_2d[:, 0] + radii) / tile_size, min=0, max=tiles_x - 1).int()
    tile_min_y = torch.clamp((means_2d[:, 1] - radii) / tile_size, min=0, max=tiles_y - 1).int()
    tile_max_y = torch.clamp((means_2d[:, 1] + radii) / tile_size, min=0, max=tiles_y - 1).int()

    # Width and height of each Gaussian's tile bounding box
    widths = tile_max_x - tile_min_x + 1
    heights = tile_max_y - tile_min_y + 1
    tiles_per_gaussian = widths * heights

    # Total number of (gaussian, tile) pairs
    P = tiles_per_gaussian.sum().item()

    # Repeat each Gaussian's index and tile-box params by its tile count
    gaussian_ids = torch.repeat_interleave(torch.arange(means_2d.shape[0]), tiles_per_gaussian)
    min_x_rep = torch.repeat_interleave(tile_min_x, tiles_per_gaussian)
    min_y_rep = torch.repeat_interleave(tile_min_y, tiles_per_gaussian)
    widths_rep = torch.repeat_interleave(widths, tiles_per_gaussian)

    # For each pair, compute the local offset within that Gaussian's tile grid
    # local_idx goes 0, 1, 2, ... tiles_per_gaussian[i]-1 for each Gaussian
    offsets = torch.arange(P) - torch.repeat_interleave(
        torch.cat([torch.zeros(1, dtype=torch.int32), tiles_per_gaussian.cumsum(0)[:-1]]),
        tiles_per_gaussian,
    )

    # Convert local offset to (dx, dy) within the Gaussian's tile bounding box
    dy = offsets // widths_rep
    dx = offsets % widths_rep

    # Compute flat tile index: (min_y + dy) * tiles_x + (min_x + dx)
    tile_ids = (min_y_rep + dy) * tiles_x + (min_x_rep + dx)

    return gaussian_ids, tile_ids, tiles_per_gaussian


def sort_and_bin(
    gaussian_ids: torch.Tensor,
    tile_ids: torch.Tensor,
    depths: torch.Tensor,
    tiles_x: int,
    tiles_y: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort Gaussians by (tile_id, depth) and compute per-tile start/end ranges.

    After tile assignment, we have a flat list of (gaussian_idx, tile_id) pairs.
    This function sorts them so that all Gaussians for the same tile are contiguous,
    and within each tile they are ordered front-to-back by depth. It then computes
    a range table so we can quickly look up which slice of the sorted array belongs
    to each tile.

    This stays on CPU — sorting is hard to parallelize on tt-metal and even the
    original CUDA implementation uses a GPU radix sort (a highly specialized algorithm).

    Args:
        gaussian_ids: (P,) Gaussian index for each pair.
        tile_ids: (P,) tile index for each pair.
        depths: (M,) depth of each visible Gaussian (indexed by gaussian_ids).
        tiles_x: number of tiles horizontally.
        tiles_y: number of tiles vertically.

    Returns:
        sorted_gaussian_ids: (P,) Gaussian indices sorted by (tile_id, depth).
        tile_ranges: (num_tiles, 2) start and end index in sorted array for each tile.
    """
    num_tiles = tiles_x * tiles_y

    # Create a composite sort key: tile_id first, then depth within each tile.
    # Sorting by (tile_id * large_number + depth) achieves lexicographic ordering.
    # We use a scale factor large enough that depth differences never cross tile boundaries.
    max_depth = depths.max().item() + 1.0
    sort_keys = tile_ids.float() * max_depth + depths[gaussian_ids]

    # Sort all pairs by the composite key
    sorted_indices = torch.argsort(sort_keys)
    sorted_gaussian_ids = gaussian_ids[sorted_indices]
    sorted_tile_ids = tile_ids[sorted_indices]

    # Build tile ranges: for each tile, find where its Gaussians start and end
    # in the sorted array. Tiles with no Gaussians get range (0, 0).
    tile_ranges = torch.zeros(num_tiles, 2, dtype=torch.int64)

    if sorted_tile_ids.numel() > 0:
        # Detect where tile_id changes in the sorted array
        changes = sorted_tile_ids[1:] != sorted_tile_ids[:-1]
        change_indices = torch.where(changes)[0] + 1

        # Start indices: position 0 + every change point
        starts = torch.cat([torch.zeros(1, dtype=torch.int64), change_indices])
        # End indices: every change point + final position
        ends = torch.cat([change_indices, torch.tensor([len(sorted_tile_ids)])])

        # The tile at each segment
        segment_tiles = sorted_tile_ids[starts]

        tile_ranges[segment_tiles, 0] = starts
        tile_ranges[segment_tiles, 1] = ends

    return sorted_gaussian_ids, tile_ranges
