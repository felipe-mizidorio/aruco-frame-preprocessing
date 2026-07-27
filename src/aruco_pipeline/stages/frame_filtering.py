import argparse
import logging
import platform
import shutil
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from ..core import pipeline_io
from ..core.schemas import (
    DetectionEntry,
    DetectionsFile,
    FilterManifest,
    MarkerDetection,
    VideoMetadata,
)

logger = logging.getLogger(__name__)

# Blur rejection is fully automatic (domestic-use constraint: no user tuning).
# A frame is rejected when its variance-of-Laplacian falls below the session's
# robust lower fence: median - K * 1.4826 * MAD. The MAD anchors on the sharp
# majority, so a uniformly sharp session rejects nothing and sessions where
# most frames are blurred degrade toward under-rejection, never over-rejection.
BLUR_MAD_K = 3.0
_MAD_TO_SIGMA = 1.4826
# Below this many scoreable frames the session distribution is meaningless.
_MIN_FRAMES_FOR_BLUR_STATS = 3


def sharpness_score(image_path: Path) -> float | None:
    """Variance of the Laplacian; None when the image cannot be read."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def compute_blur_rejects(
    detections: DetectionsFile, frames_dir: Path
) -> tuple[set[str], dict]:
    """Score every candidate frame and flag blurred outliers for rejection.

    Frames whose image cannot be read are never rejected here (they are
    handled by the existing missing-file path in filter_frames).

    Returns:
        (filenames_to_reject, sharpness_stats) — stats go into the manifest
        so the criterion and threshold are auditable per session.
    """
    scores: dict[str, float] = {}
    for entry in detections.detections:
        score = sharpness_score(frames_dir / entry.filename)
        if score is not None:
            scores[entry.filename] = score

    stats: dict = {
        "metric": "variance_of_laplacian",
        "mad_k": BLUR_MAD_K,
        "frames_scored": len(scores),
        "threshold": None,
        "median": None,
        "mad": None,
        "frames_rejected": 0,
    }

    if len(scores) < _MIN_FRAMES_FOR_BLUR_STATS:
        logger.info("Blur filtering skipped: only %d scoreable frame(s).", len(scores))
        return set(), stats

    values = np.array(list(scores.values()))
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = max(median - BLUR_MAD_K * _MAD_TO_SIGMA * mad, 0.0)

    rejects = {name for name, score in scores.items() if score < threshold}
    stats.update(
        threshold=threshold,
        median=median,
        mad=mad,
        frames_rejected=len(rejects),
    )
    logger.info(
        "Blur filtering: %d / %d frames below threshold %.1f (median=%.1f, MAD=%.1f).",
        len(rejects),
        len(scores),
        threshold,
        median,
        mad,
    )
    return rejects, stats


def tool_versions() -> dict:
    return {
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }


def camera_block(session_dir: Path) -> dict | None:
    """Camera metadata for the SfM side, from the session's metadata.json.

    Carries the container-probed 35mm-equivalent focal length (may be null)
    plus frame dimensions so the SfM pipeline can derive a focal prior in
    pixels. Returns None when metadata.json is absent (old sessions).
    """
    metadata_path = session_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    metadata = VideoMetadata.from_dict(
        pipeline_io.load_json(metadata_path, "metadata"), source=str(metadata_path)
    )
    return {
        "focal_length_35mm": metadata.focal_length_35mm,
        "width_px": metadata.resolution.get("width"),
        "height_px": metadata.resolution.get("height"),
    }


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
    blur_rejects: set[str] | None = None,
) -> list[DetectionEntry]:
    filtered_dir = frames_dir / "filtered"
    filtered_dir.mkdir(parents=True, exist_ok=True)

    passing: list[DetectionEntry] = []
    all_entries = detections.detections
    total = len(all_entries)

    for i, entry in enumerate(all_entries):
        pipeline_io.log_progress(i, total)

        if blur_rejects is not None and entry.filename in blur_rejects:
            continue

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
    sharpness: dict | None = None,
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
        sharpness=sharpness,
        tool_versions=tool_versions(),
        camera=camera_block(output_dir),
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
    blur_rejects, sharpness_stats = compute_blur_rejects(detections, frames_dir)
    filtered = filter_frames(
        detections,
        frames_dir,
        args.min_markers,
        valid_ids=valid_ids,
        blur_rejects=blur_rejects,
    )
    save_manifest(
        filtered, detections, frames_dir, args.min_markers, sharpness=sharpness_stats
    )


if __name__ == "__main__":
    main()
