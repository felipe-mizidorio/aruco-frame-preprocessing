import argparse
import logging
import shutil
from dataclasses import replace
from pathlib import Path

import pipeline_io
from schemas import DetectionEntry, DetectionsFile, FilterManifest, MarkerDetection

logger = logging.getLogger(__name__)


def strip_invalid_ids(entry: DetectionEntry, valid_ids: set[int]) -> DetectionEntry:
    pairs = [
        (mid, corner)
        for mid, corner in zip(entry.marker_ids, entry.corners)
        if mid in valid_ids
    ]
    if pairs:
        ids, corners = zip(*pairs)
        return replace(
            entry,
            marker_ids=list(ids),
            corners=list(corners),
            markers_detected=len(ids),
        )
    return replace(entry, marker_ids=[], corners=[], markers_detected=0)


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


def filter_frames(
    detections: DetectionsFile,
    frames_dir: Path,
    min_markers: int,
    valid_ids: set[int] | None = None,
) -> list[DetectionEntry]:
    filtered_dir = frames_dir / "filtered"
    filtered_dir.mkdir(parents=True, exist_ok=True)

    passing: list[DetectionEntry] = []
    all_entries = detections.detections
    total = len(all_entries)

    for i, entry in enumerate(all_entries):
        pipeline_io.log_progress(i, total)

        if valid_ids is not None:
            entry = strip_invalid_ids(entry, valid_ids)

        if entry.markers_detected < min_markers:
            continue

        src = frames_dir / entry.filename
        if not src.exists():
            logger.warning("Source frame not found, skipping: %s", src)
            continue

        shutil.copy2(src, filtered_dir / entry.filename)
        passing.append(entry)

    logger.info(
        "Filtering complete. %d / %d frames passed (min_markers=%d).",
        len(passing),
        total,
        min_markers,
    )
    return passing


def save_manifest(
    filtered: list[DetectionEntry],
    detections: DetectionsFile,
    output_dir: Path,
    min_markers: int,
) -> None:
    frames = [entry.filename for entry in filtered]
    marker_detections = {
        entry.filename: [
            MarkerDetection(id=mid, corners=corners)
            for mid, corners in zip(entry.marker_ids, entry.corners)
        ]
        for entry in filtered
    }

    total_frames_input = len(detections.detections)
    frames_filtered_out = total_frames_input - len(filtered)

    manifest = FilterManifest(
        dictionary=detections.dictionary,
        min_markers=min_markers,
        total_frames_input=total_frames_input,
        frames_filtered_out=frames_filtered_out,
        frames=frames,
        marker_detections=marker_detections,
    )

    out_path = output_dir / "manifest.json"
    pipeline_io.save_json(manifest.to_dict(), out_path)

    logger.info(
        "Manifest saved to '%s'. %d kept, %d filtered out.",
        out_path,
        len(filtered),
        frames_filtered_out,
    )


def main() -> None:
    pipeline_io.configure_logging()
    args = parse_args()

    detections = DetectionsFile.from_dict(
        pipeline_io.load_json(args.detections, "detections"),
        source=str(args.detections),
    )
    frames_dir = pipeline_io.session_dir(args.detections)

    valid_ids = set(args.valid_ids) if args.valid_ids is not None else None
    filtered = filter_frames(
        detections, frames_dir, args.min_markers, valid_ids=valid_ids
    )
    save_manifest(filtered, detections, frames_dir, args.min_markers)


if __name__ == "__main__":
    main()
