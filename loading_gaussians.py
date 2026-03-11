import torch
import numpy as np
from plyfile import PlyData
from dataclasses import dataclass


@dataclass
class Gaussians:
    means: torch.Tensor       # (N, 3) - xyz positions
    scales: torch.Tensor      # (N, 3) - actual scales (already exp'd)
    rotations: torch.Tensor   # (N, 4) - unit quaternions (wxyz order)
    colors: torch.Tensor      # (N, 3) - RGB colors in [0, 1] (already converted from SH)
    opacities: torch.Tensor   # (N,)   - opacities in [0, 1] (already sigmoid'd)

    @property
    def num_gaussians(self) -> int:
        return self.means.shape[0]


def activate_scales(raw_scales: torch.Tensor) -> torch.Tensor:
    """Convert from log-space to actual scales: exp(log_scale) -> scale > 0."""
    return torch.exp(raw_scales)


def normalize_quaternions(quats: torch.Tensor) -> torch.Tensor:
    """Normalize quaternions to unit length for valid rotations."""
    return quats / quats.norm(dim=-1, keepdim=True)


def sh_to_rgb(sh_dc: torch.Tensor) -> torch.Tensor:
    """Convert degree-0 SH coefficients to RGB colors in [0, 1].

    color = 0.5 + C0 * sh_coeff, where C0 = Y_0^0 = 1 / (2 * sqrt(pi)).
    The 0.5 offset recenters the SH representation (centered at 0) into RGB space (centered at 0.5).
    """
    C0 = 0.28209479177387814
    return torch.clamp(0.5 + C0 * sh_dc, 0.0, 1.0)


def activate_opacities(raw_opacities: torch.Tensor) -> torch.Tensor:
    """Convert from logit-space to [0, 1] via sigmoid."""
    return torch.sigmoid(raw_opacities)


def load_ply(path: str) -> Gaussians:
    """Load pre-trained 3D Gaussians from a .ply file.

    The PLY file stores parameters in their raw optimizer space:
    - scales in log-space (exp maps to positive values)
    - opacities in logit-space (sigmoid maps to [0, 1])
    - colors as SH degree-0 coefficients (converted to RGB via 0.5 + C0 * sh)
    - quaternions unnormalized (normalized to unit length for valid rotations)

    All activations are applied here so the returned Gaussians are ready
    for direct use in the rasterization pipeline.
    """
    ply = PlyData.read(path)
    vertex = ply.elements[0]

    means = torch.tensor(np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1), dtype=torch.float32)
    scales = torch.tensor(np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=-1), dtype=torch.float32)
    rotations = torch.tensor(np.stack([vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=-1), dtype=torch.float32)
    sh_dc = torch.tensor(np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=-1), dtype=torch.float32)
    opacities = torch.tensor(np.array(vertex["opacity"]), dtype=torch.float32)

    # Activate from optimizer space to usable values
    scales = activate_scales(scales)
    rotations = normalize_quaternions(rotations)
    colors = sh_to_rgb(sh_dc)
    opacities = activate_opacities(opacities)

    return Gaussians(
        means=means,
        scales=scales,
        rotations=rotations,
        colors=colors,
        opacities=opacities,
    )


if __name__ == "__main__":
    gaussians = load_ply("scene/luigi.ply")
    print(f"Loaded {gaussians.num_gaussians} Gaussians")
    print(f"  means:     {gaussians.means.shape}  range [{gaussians.means.min():.3f}, {gaussians.means.max():.3f}]")
    print(f"  scales:    {gaussians.scales.shape}  range [{gaussians.scales.min():.4f}, {gaussians.scales.max():.4f}]")
    print(f"  rotations: {gaussians.rotations.shape}")
    print(f"  colors:    {gaussians.colors.shape}  range [{gaussians.colors.min():.3f}, {gaussians.colors.max():.3f}]")
    print(f"  opacities: {gaussians.opacities.shape}  range [{gaussians.opacities.min():.3f}, {gaussians.opacities.max():.3f}]")
