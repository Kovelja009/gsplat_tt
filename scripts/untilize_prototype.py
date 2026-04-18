"""Verify ttnn.untilize handles the (N, 3, 32, 32) output layout.

If this fails, the writer kernel must use the 3-separate-buffer fallback.
"""
import numpy as np
import ttnn
import torch


def main():
    device = ttnn.open_device(device_id=0)
    try:
        N = 4  # 4 screen tiles
        shape = (N, 3, 32, 32)
        host_tensor = torch.rand(shape, dtype=torch.bfloat16)

        tile_tensor = ttnn.from_torch(
            host_tensor,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        print(f"Tilized shape: {tile_tensor.shape}, layout: {tile_tensor.layout}")

        row_major = ttnn.to_layout(tile_tensor, ttnn.ROW_MAJOR_LAYOUT)
        print(f"Row-major shape: {row_major.shape}")

        back_on_host = ttnn.to_torch(row_major)
        assert torch.allclose(back_on_host, host_tensor, atol=0.01), "roundtrip mismatch"
        print("Primary layout (N, 3, 32, 32) round-trip OK.")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
