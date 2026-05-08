import argparse
import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter frames by minimum number of detected ArUco markers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--detections",
        type=Path,
        required=True,
        help="Path to the detections.json file produced by aruco_detection.py.",
    )
    parser.add_argument(
        "--min-markers",
        type=int,
        default=1,
        help="Keep only frames with at least this many markers detected.",
    )
    return parser.parse_args()


def load_detections(detections_path: Path) -> dict:
    if not detections_path.exists():
        logger.error("Detections file not found: %s", detections_path)
        raise FileNotFoundError(f"Detections file not found: {detections_path}")

    with detections_path.open() as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in detections file: %s", detections_path)
            raise ValueError(
                f"Invalid JSON in detections file: {detections_path}"
            ) from e


def filter_frames(detections: dict, frames_dir: Path, min_markers: int) -> list[dict]:
    filtered_dir = frames_dir / "filtered"
    filtered_dir.mkdir(parents=True, exist_ok=True)

    passing: list[dict] = []

    for i, entry in enumerate(detections["detections"]):
        total = len(detections["detections"])
        if i % 50 == 0:
            logger.info("Progress: %d / %d frames processed", i, total)

        if entry["markers_detected"] < min_markers:
            continue

        src = frames_dir / entry["filename"]
        if not src.exists():
            logger.warning("Source frame not found, skipping: %s", src)
            continue

        shutil.copy2(src, filtered_dir / entry["filename"])
        passing.append(entry)

    logger.info(
        "Filtering complete. %d / %d frames passed (min_markers=%d).",
        len(passing),
        len(detections["detections"]),
        min_markers,
    )
    return passing


def save_filtered_detections(
    filtered: list[dict], detections: dict, output_dir: Path, min_markers: int
) -> None:
    original_total = detections["total_frames"]
    frames_with_detections = sum(1 for d in filtered if d["markers_detected"] > 0)

    output = {
        "dictionary": detections.get("dictionary", "DICT_4X4_250"),
        "min_markers": min_markers,
        "total_frames": len(filtered),
        "frames_with_detections": frames_with_detections,
        "frames_filtered_out": original_total - len(filtered),
        "detections": filtered,
    }

    out_path = output_dir / "filtered_detections.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)

    logger.info(
        "Filtered detections saved to '%s'. %d kept, %d filtered out.",
        out_path,
        len(filtered),
        original_total - len(filtered),
    )


def main() -> None:
    args = parse_args()

    detections = load_detections(args.detections)
    frames_dir = args.detections.parent

    filtered = filter_frames(detections, frames_dir, args.min_markers)
    save_filtered_detections(filtered, detections, frames_dir, args.min_markers)


if __name__ == "__main__":
    main()
