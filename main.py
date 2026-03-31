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
    args = parser.parse_args()

    print(f"Loading Gaussians from {args.ply_path}...")
    gaussians = load_ply(args.ply_path)
    print(f"Loaded {gaussians.num_gaussians:,} Gaussians")

    viewer = GaussianViewer(gaussians, host=args.host, port=args.port)
    viewer.run()


if __name__ == "__main__":
    main()
