import argparse
import json
import logging
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect aruco markers in images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Path to the metadata file",
    )
    return parser.parse_args()


def load_metadata(metadata_path: Path) -> dict:
    if not metadata_path.exists():
        logger.error("Metadata file not found: %s", metadata_path)
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with metadata_path.open() as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in metadata file: %s", metadata_path)
            raise ValueError(f"Invalid JSON in metadata file: {metadata_path}") from e


def detect_markers(metadata: dict, frames_dir: Path) -> list[dict]:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    detections: list[dict] = []
    frames = metadata["frames"]

    for i, frame in enumerate(frames):
        frame_path = frames_dir / frame["filename"]
        image = cv2.imread(str(frame_path))

        if image is None:
            logger.error("Cannot load image: %s", frame_path)
            detections.append(
                {
                    "filename": frame["filename"],
                    "frame_index": frame["frame_index"],
                    "markers_detected": 0,
                    "marker_ids": [],
                    "corners": [],
                }
            )
            continue

        corners, ids, _ = detector.detectMarkers(image)

        if ids is None:
            logger.warning("No markers detected in frame: %s", frame["filename"])
            detections.append(
                {
                    "filename": frame["filename"],
                    "frame_index": frame["frame_index"],
                    "markers_detected": 0,
                    "marker_ids": [],
                    "corners": [],
                }
            )
            continue

        detections.append(
            {
                "filename": frame["filename"],
                "frame_index": frame["frame_index"],
                "markers_detected": len(corners),
                "marker_ids": ids.flatten().tolist(),
                "corners": [c.tolist() for c in corners],
            }
        )

        if i % 50 == 0:
            logger.info(
                "Progress: %d / %d frames processed (video frame index: %d)",
                i,
                len(frames),
                frame["frame_index"],
            )

    return detections


def save_detections(detections: list[dict], metadata: dict, output_dir: Path) -> None:
    frames_with_detections = sum(1 for d in detections if d["markers_detected"] > 0)

    output = {
        "dictionary": "DICT_4X4_250",
        "total_frames": metadata["frames_extracted"],
        "frames_with_detections": frames_with_detections,
        "detections": detections,
    }

    detections_path = output_dir / "detections.json"
    with detections_path.open("w") as f:
        json.dump(output, f, indent=2)

    logger.info(
        "Detections saved to '%s'. %d / %d frames with markers.",
        detections_path,
        frames_with_detections,
        metadata["frames_extracted"],
    )


def main() -> None:
    args = parse_args()

    metadata = load_metadata(args.metadata)
    frames_dir = args.metadata.parent

    detections = detect_markers(metadata, frames_dir)
    save_detections(detections, metadata, frames_dir)


if __name__ == "__main__":
    main()
