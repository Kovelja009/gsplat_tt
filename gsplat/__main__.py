import argparse

from backends import REGISTRY as BACKEND_REGISTRY
from gsplat.loading_gaussians import load_ply
from gsplat.viewer import GaussianViewer


def main():
    parser = argparse.ArgumentParser(
        description="Interactive 3D Gaussian Splatting Viewer"
    )
    parser.add_argument("ply_path", help="Path to a pre-trained .ply file")
    parser.add_argument(
        "--port", type=int, default=8080, help="Viewer port (default: 8080)"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Viewer host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--backend",
        choices=sorted(BACKEND_REGISTRY),
        default="cpu",
        help=(
            "Rendering backend; choices come from the registry in "
            "backends/__init__.py. Default: cpu."
        ),
    )
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=640,
        help=(
            "Shorter render dim, in pixels (480p/720p/1080p convention). The "
            "longer dim follows from the browser's aspect ratio; both dims are "
            "snapped down to multiples of 32 so the kernel sees whole tiles. "
            "Default: 640."
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-frame stage timing (render-enter / render-mid / kernel-pre / kernel-post / render).",
    )
    args = parser.parse_args()

    print(f"Loading Gaussians from {args.ply_path}...")
    gaussians = load_ply(args.ply_path)
    print(f"Loaded {gaussians.num_gaussians:,} Gaussians")

    viewer = GaussianViewer(
        gaussians,
        host=args.host,
        port=args.port,
        backend=args.backend,
        max_resolution=args.max_resolution,
        verbose=args.verbose,
        scene_path=args.ply_path,
    )
    viewer.run()


if __name__ == "__main__":
    main()
