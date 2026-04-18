"""Numeric sanity reference for the alpha-blend kernel.

This is a NumPy port of the on-device compute kernel math.
Runs in fp32 (golden) or bf16-simulated (precision truth).
Used to validate algorithm before device implementation and to
unit-test each math step independently.
"""
import numpy as np


def _as_bf16(x: np.ndarray) -> np.ndarray:
    """Simulate bf16 by truncating mantissa of fp32 to 7 bits."""
    u = x.view(np.uint32) & 0xFFFF0000
    return u.view(np.float32)


def alpha_blend_reference(
    attribute_packs: np.ndarray,      # (N, 9) fp32
    tile_offsets: np.ndarray,          # (num_tiles + 1,) uint32
    px_tiles: np.ndarray,              # (num_tiles, 32, 32) fp32
    py_tiles: np.ndarray,              # (num_tiles, 32, 32) fp32
    image_h: int,
    image_w: int,
    simulate_bf16: bool = False,
) -> np.ndarray:
    """Run the compute kernel math in NumPy.

    attribute_packs per-Gaussian layout:
        [mean_x, mean_y, cov_a, two_cov_b, cov_c, R, G, B, opacity]
    where two_cov_b = 2 * cov_b (host precompute).
    """
    tiles_x = (image_w + 31) // 32
    tiles_y = (image_h + 31) // 32
    num_tiles = tiles_y * tiles_x
    assert tile_offsets.shape == (num_tiles + 1,)
    assert px_tiles.shape == (num_tiles, 32, 32)
    assert py_tiles.shape == (num_tiles, 32, 32)

    output = np.zeros((tiles_y * 32, tiles_x * 32, 3), dtype=np.float32)
    cast = _as_bf16 if simulate_bf16 else (lambda x: x)

    for tile_id in range(num_tiles):
        ty = tile_id // tiles_x
        tx = tile_id % tiles_x
        g_start = int(tile_offsets[tile_id])
        g_end = int(tile_offsets[tile_id + 1])
        if g_start == g_end:
            continue

        px = px_tiles[tile_id]
        py = py_tiles[tile_id]
        color_r = np.zeros((32, 32), dtype=np.float32)
        color_g = np.zeros((32, 32), dtype=np.float32)
        color_b = np.zeros((32, 32), dtype=np.float32)
        T = np.ones((32, 32), dtype=np.float32)
        sat_mask = np.ones((32, 32), dtype=np.float32)

        for g_idx in range(g_start, g_end):
            g = g_idx - g_start  # local index in this tile
            if g > 0 and (g & 15) == 0:
                sat_mask = (T >= 1e-4).astype(np.float32)

            mean_x, mean_y, cov_a, two_cov_b, cov_c, R, G, B, opacity = attribute_packs[g_idx]

            dx = cast(px - mean_x)
            dy = cast(py - mean_y)
            dx2 = cast(dx * dx)
            dy2 = cast(dy * dy)
            dxdy = cast(dx * dy)
            Q = cast(cov_a * dx2 + two_cov_b * dxdy + cov_c * dy2)
            power = cast(-0.5 * Q)
            power = cast(np.minimum(power, 0.0))
            weight = cast(np.exp(power))
            alpha = cast(np.minimum(0.99, opacity * weight))
            contrib = cast(alpha * T)

            color_r = cast(color_r + R * contrib * sat_mask)
            color_g = cast(color_g + G * contrib * sat_mask)
            color_b = cast(color_b + B * contrib * sat_mask)
            T = cast(T * (1.0 - alpha) * sat_mask)

        py_start = ty * 32
        px_start = tx * 32
        output[py_start:py_start + 32, px_start:px_start + 32, 0] = color_r
        output[py_start:py_start + 32, px_start:px_start + 32, 1] = color_g
        output[py_start:py_start + 32, px_start:px_start + 32, 2] = color_b

    return output[:image_h, :image_w, :]
