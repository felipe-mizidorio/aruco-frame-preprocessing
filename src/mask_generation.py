"""Stage 3b: ArUco-hull foreground masks for COLMAP feature extraction.

With markers rigidly attached to the head (cap), head + markers form one rigid
body: masking out background features makes subject motion equivalent to
camera motion for SfM and suppresses the contiguous-background failure mode.
Deterministic and ML-free (MobileSAM was removed for degrading results).

Mask convention (COLMAP): `filtered/masks/<image filename>.png` — the mask for
`frame_0000.jpg` is `frame_0000.jpg.png`. White (255) keeps features, black
(0) discards them.

Frames with fewer than MIN_MARKERS_FOR_HULL distinct markers get a full-white
mask: a hull from 1–2 markers covers only a sliver of the head and would
starve SfM of features for that frame, and an explicit all-keep mask is
deterministic regardless of how pycolmap treats missing mask files.
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

import pipeline_io
from schemas import FilterManifest

logger = logging.getLogger(__name__)

# Margin around the corner hull, in units of the frame's median marker side.
# Markers sit on a cap on the crown; the head extends past the cap by roughly
# two marker sides (~100 mm for 50 mm markers). Derived from what the frame
# itself provides — no user tuning (domestic-use constraint).
HULL_MARGIN_MARKER_SIDES = 2.0
# Below this many distinct markers the hull is too small to bound the head.
MIN_MARKERS_FOR_HULL = 3

MASK_DIR_NAME = "masks"


def _marker_sides_px(corners: np.ndarray) -> np.ndarray:
    """Side lengths of one marker's 4-corner polygon, in pixels."""
    return np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)


def generate_mask(
    marker_corners: list, width: int, height: int
) -> np.ndarray | None:
    """Binary hull mask for one frame, or None when markers are insufficient.

    Args:
        marker_corners: One entry per marker: 4 [x, y] corner points.
        width/height: Frame dimensions in pixels.
    """
    if len(marker_corners) < MIN_MARKERS_FOR_HULL:
        return None

    corner_arrays = [
        np.asarray(c, dtype=np.float64).reshape(-1, 2) for c in marker_corners
    ]
    points = np.vstack(corner_arrays)
    all_sides = np.concatenate([_marker_sides_px(c) for c in corner_arrays])
    median_side = float(np.median(all_sides))
    margin_px = max(int(round(HULL_MARGIN_MARKER_SIDES * median_side)), 1)

    mask = np.zeros((height, width), dtype=np.uint8)
    hull = cv2.convexHull(points.astype(np.float32))
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1)
    )
    return cv2.dilate(mask, kernel)


def generate_masks(
    manifest: FilterManifest,
    session_dir: Path,
    manifest_path: Path | None = None,
) -> dict:
    """Write hull masks for every manifest frame; update the manifest on disk.

    Returns the mask_generation stats dict (also written into the manifest
    when manifest_path is given).
    """
    filtered_dir = session_dir / "filtered"
    masks_dir = filtered_dir / MASK_DIR_NAME
    masks_dir.mkdir(parents=True, exist_ok=True)

    frames_masked = 0
    frames_fallback = 0
    total = len(manifest.frames)

    for i, filename in enumerate(manifest.frames):
        pipeline_io.log_progress(i, total)

        frame_path = filtered_dir / filename
        img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning("Frame unreadable, skipping mask: %s", frame_path)
            continue
        # shape[:2]: ultralytics (when installed) patches cv2.imread globally
        # and returns (H, W, 1) for grayscale reads.
        height, width = img.shape[:2]

        detections = manifest.marker_detections.get(filename, [])
        mask = generate_mask([d.corners for d in detections], width, height)
        if mask is None:
            mask = np.full((height, width), 255, dtype=np.uint8)
            frames_fallback += 1
        else:
            frames_masked += 1

        cv2.imwrite(str(masks_dir / f"{filename}.png"), mask)

    stats = {
        "frames_masked": frames_masked,
        "frames_fallback_full": frames_fallback,
        "margin_marker_sides": HULL_MARGIN_MARKER_SIDES,
        "min_markers_for_hull": MIN_MARKERS_FOR_HULL,
    }
    logger.info(
        "Mask generation complete: %d hull mask(s), %d full-white fallback(s) → '%s'.",
        frames_masked,
        frames_fallback,
        masks_dir,
    )

    if manifest_path is not None:
        manifest.mask_dir = MASK_DIR_NAME
        manifest.mask_generation = stats
        pipeline_io.save_json(manifest.to_dict(), manifest_path)
        logger.info("Manifest updated with mask_dir: '%s'", manifest_path)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ArUco-hull foreground masks (COLMAP convention) from a "
            "filtering manifest."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to the manifest.json produced by frame_filtering.py.",
    )
    return parser.parse_args()


def main() -> None:
    pipeline_io.configure_logging()
    args = parse_args()

    manifest = FilterManifest.from_dict(
        pipeline_io.load_json(args.manifest, "manifest"), source=str(args.manifest)
    )
    session_dir = pipeline_io.session_dir(args.manifest)
    generate_masks(manifest, session_dir, manifest_path=args.manifest)


if __name__ == "__main__":
    main()
