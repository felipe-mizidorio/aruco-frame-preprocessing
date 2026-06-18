import argparse
import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def strip_invalid_ids(entry: dict, valid_ids: set[int]) -> dict:
    pairs = [
        (mid, corner)
        for mid, corner in zip(entry["marker_ids"], entry["corners"])
        if mid in valid_ids
    ]
    if pairs:
        ids, corners = zip(*pairs)
        return {
            **entry,
            "marker_ids": list(ids),
            "corners": list(corners),
            "markers_detected": len(ids),
        }
    return {**entry, "marker_ids": [], "corners": [], "markers_detected": 0}


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
    parser.add_argument(
        "--valid-ids",
        type=int,
        nargs="+",
        default=None,
        metavar="ID",
        help=(
            "Whitelist of valid marker IDs. "
            "Detections with other IDs are stripped before filtering."
        ),
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


def filter_frames(
    detections: dict,
    frames_dir: Path,
    min_markers: int,
    valid_ids: set[int] | None = None,
) -> list[dict]:
    filtered_dir = frames_dir / "filtered"
    filtered_dir.mkdir(parents=True, exist_ok=True)

    passing: list[dict] = []
    all_entries = detections["detections"]
    total = len(all_entries)

    for i, entry in enumerate(all_entries):
        if i % 50 == 0:
            logger.info("Progress: %d / %d frames processed", i, total)

        if valid_ids is not None:
            entry = strip_invalid_ids(entry, valid_ids)

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
    frames_with_detections = sum(1 for d in filtered if d["markers_detected"] > 0)
    frames_filtered_out = sum(
        1 for d in detections["detections"] if d["markers_detected"] < min_markers
    )

    output = {
        "dictionary": detections.get("dictionary", "DICT_4X4_250"),
        "min_markers": min_markers,
        "total_frames": len(filtered),
        "frames_with_detections": frames_with_detections,
        "frames_filtered_out": frames_filtered_out,
        "detections": filtered,
    }

    out_path = output_dir / "filtered_detections.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)

    logger.info(
        "Filtered detections saved to '%s'. %d kept, %d filtered out.",
        out_path,
        len(filtered),
        frames_filtered_out,
    )


def main() -> None:
    args = parse_args()

    detections = load_detections(args.detections)
    frames_dir = args.detections.parent

    valid_ids = set(args.valid_ids) if args.valid_ids is not None else None
    filtered = filter_frames(
        detections, frames_dir, args.min_markers, valid_ids=valid_ids
    )
    save_filtered_detections(filtered, detections, frames_dir, args.min_markers)


if __name__ == "__main__":
    main()
