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


def main() -> None:
    save_markers(10, 200, "data/raw")


if __name__ == "__main__":
    main()
