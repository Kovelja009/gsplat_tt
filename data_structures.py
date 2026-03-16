import torch
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