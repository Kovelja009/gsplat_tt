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