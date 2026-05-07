"""3D Gaussian Splatting forward-pass renderer (CPU pipeline + viewer).

Public modules:
    data_structures   — Gaussians dataclass
    loading_gaussians — .ply loader
    rasterization     — project / tile / sort / alpha_blend / prepare_kernel_inputs
    viewer            — interactive viewer (viser + nerfview)
    utils             — camera helpers (c2w↔w2c, build_covariance_3d)

Backends live in the top-level `backends/` package (e.g. backends.tt.backend).
"""
