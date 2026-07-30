import argparse
import logging
from pathlib import Path

import cv2

from ..config import load_config
from ..core import pipeline_io
from ..core.schemas import DetectionEntry, DetectionsFile, VideoMetadata

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
    parser.add_argument(
        "--dictionary",
        type=str,
        default=None,
        choices=sorted(pipeline_io.ARUCO_DICTIONARIES.keys()),
        help="ArUco dictionary to use for detection.",
    )
    return parser.parse_args()


def _empty_entry(filename: str, frame_index: int) -> DetectionEntry:
    """A DetectionEntry for a frame with no usable markers."""
    return DetectionEntry(
        filename=filename,
        frame_index=frame_index,
        markers_detected=0,
        marker_ids=[],
        corners=[],
    )


def detect_markers(
    metadata: VideoMetadata, frames_dir: Path, dictionary_name: str = "DICT_4X4_250"
) -> list[DetectionEntry]:
    dictionary = cv2.aruco.getPredefinedDictionary(
        pipeline_io.ARUCO_DICTIONARIES[dictionary_name]
    )
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    detections: list[DetectionEntry] = []
    frames = metadata.frames

    for i, frame in enumerate(frames):
        frame_path = frames_dir / frame.filename
        image = cv2.imread(str(frame_path))

        if image is None:
            logger.error("Cannot load image: %s", frame_path)
            detections.append(_empty_entry(frame.filename, frame.frame_index))
            continue

        corners, ids, _ = detector.detectMarkers(image)

        if ids is None:
            logger.warning("No markers detected in frame: %s", frame.filename)
            detections.append(_empty_entry(frame.filename, frame.frame_index))
            continue

        detections.append(
            DetectionEntry(
                filename=frame.filename,
                frame_index=frame.frame_index,
                markers_detected=len(corners),
                marker_ids=ids.flatten().tolist(),  # type: ignore[arg-type]  # numpy .tolist() is int at runtime
                corners=[c.tolist() for c in corners],
            )
        )

        pipeline_io.log_progress(i, len(frames))

    return detections


def save_detections(
    detections: list[DetectionEntry],
    metadata: VideoMetadata,
    output_dir: Path,
    dictionary_name: str,
) -> None:
    frames_with_detections = sum(1 for d in detections if d.markers_detected > 0)

    output = DetectionsFile(
        dictionary=dictionary_name,
        total_frames=metadata.frames_extracted,
        frames_with_detections=frames_with_detections,
        detections=detections,
    )

    detections_path = output_dir / "detections.json"
    pipeline_io.save_json(output.to_dict(), detections_path)

    logger.info(
        "Detections saved to '%s'. %d / %d frames with markers.",
        detections_path,
        frames_with_detections,
        metadata.frames_extracted,
    )


def main() -> None:
    pipeline_io.configure_logging()
    args = parse_args()

    metadata = VideoMetadata.from_dict(
        pipeline_io.load_json(args.metadata, "metadata"), source=str(args.metadata)
    )
    frames_dir = pipeline_io.session_dir(args.metadata)

    cfg = load_config()
    dictionary = args.dictionary if args.dictionary is not None else cfg.dictionary

    detections = detect_markers(metadata, frames_dir, dictionary)
    save_detections(detections, metadata, frames_dir, dictionary)


if __name__ == "__main__":
    main()
