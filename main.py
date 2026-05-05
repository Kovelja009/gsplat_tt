import argparse

from loading_gaussians import load_ply
from viewer import GaussianViewer


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
        choices=["cpu", "kernel"],
        default="cpu",
        help="Rendering backend (default: cpu)",
    )
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=640,
        help=(
            "Cap longest dim of the render (preserving aspect ratio, rounded "
            "down to multiple of 32). Prevents the kernel backend from "
            "spending seconds in prepare_kernel_inputs at typical browser "
            "resolutions. Default: 640."
        ),
    )
    parser.add_argument(
        "--adaptive-resolution",
        action="store_true",
        help=(
            "Enable nerfview's adaptive low-res preview during camera movement "
            "(smoother drag at the cost of pixelated frames). Default: off, "
            "every frame is rendered at max-resolution."
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
        adaptive_resolution=args.adaptive_resolution,
        verbose=args.verbose,
    )
    viewer.run()


if __name__ == "__main__":
    main()
