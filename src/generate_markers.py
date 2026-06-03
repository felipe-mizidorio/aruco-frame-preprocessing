import argparse
import logging
from pathlib import Path

import cv2
from cv2.typing import MatLike

logger = logging.getLogger(__name__)


def generate_marker(dictionary: int, marker_id: int, side_pixels: int) -> MatLike:
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary)
    image = cv2.aruco.generateImageMarker(aruco_dictionary, marker_id, side_pixels)
    return image


def save_markers(
    num_markers: int,
    side_pixels: int,
    output_dir: str | Path,
    dictionary: int = cv2.aruco.DICT_4X4_250,
) -> None:
    for i in range(num_markers):
        image = generate_marker(dictionary, i, side_pixels)
        cv2.imwrite(str(Path(output_dir) / f"marker_{i}.png"), image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ArUco marker images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-markers",
        type=int,
        default=10,
        help="Number of markers to generate (IDs 0 to N-1).",
    )
    parser.add_argument(
        "--side-pixels",
        type=int,
        default=200,
        help="Side length of each marker image in pixels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to write marker images. Created if it does not exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_markers(args.num_markers, args.side_pixels, args.output_dir)


if __name__ == "__main__":
    main()
