"""Program-cache invariant: warm frames reuse the compiled program; each new
resolution adds exactly one cache entry. Guards the central design decision
(per-frame LPT schedule is excluded from the program hash)."""
import pytest
import torch

from gsplat.rasterization import project_gaussians, get_tile_assignments, sort_and_bin


def _have_tt():
    try:
        import ttnn
        d = ttnn.open_device(device_id=0)
        ttnn.close_device(d)
        return True
    except Exception:
        return False


skip_no_tt = pytest.mark.skipif(not _have_tt(), reason="no Tenstorrent device / ttnn available")


def _scene(H, W, N, seed):
    torch.manual_seed(seed)
    means = torch.rand(N, 3) * torch.tensor([2.0, 2.0, 0.0]) + torch.tensor([-1.0, -1.0, 1.0])
    scales = torch.rand(N, 3) * 0.1 + 0.02
    q = torch.randn(N, 4); q = q / q.norm(dim=-1, keepdim=True)
    colors = torch.rand(N, 3)
    opacities = torch.rand(N) * 0.5 + 0.2
    extrinsics = torch.eye(4)
    intrinsics = torch.tensor([[40.0, 0, W / 2], [0, 40.0, H / 2], [0, 0, 1]], dtype=torch.float32)
    m2, c2, dep, rad, val = project_gaussians(means, scales, q, extrinsics, intrinsics, H, W)
    g, t, _ = get_tile_assignments(m2, rad, H, W, tile_size=32)
    sg, tr = sort_and_bin(g, t, dep, (W + 31) // 32, (H + 31) // 32)
    return (m2, c2, colors[val], opacities[val], sg, tr, H, W)


@skip_no_tt
def test_warm_frames_do_not_recompile():
    from backends.tt.backend import KernelBackend
    backend = KernelBackend()
    try:
        for seed in range(5):                       # 5 frames, fixed resolution
            backend.blend(*_scene(64, 64, 50, 42 + seed))
        n_fixed = backend.device.num_program_cache_entries()
        backend.blend(*_scene(96, 96, 80, 7))       # new resolution
        n_newres = backend.device.num_program_cache_entries()
    finally:
        backend.close()

    assert n_fixed == 1, f"expected 1 cached program after 5 same-res frames, got {n_fixed}"
    assert n_newres == 2, f"expected 2 cached programs after a new resolution, got {n_newres}"
