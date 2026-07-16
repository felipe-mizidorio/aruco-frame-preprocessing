import json
from pathlib import Path

import cv2
import numpy as np

from mask_generation import (
    HULL_MARGIN_MARKER_SIDES,
    MIN_MARKERS_FOR_HULL,
    generate_mask,
    generate_masks,
)
from schemas import FilterManifest, MarkerDetection


def square(cx: float, cy: float, side: float) -> list:
    h = side / 2.0
    return [
        [cx - h, cy - h],
        [cx + h, cy - h],
        [cx + h, cy + h],
        [cx - h, cy + h],
    ]


def make_manifest(marker_detections: dict) -> FilterManifest:
    return FilterManifest(
        dictionary="DICT_4X4_250",
        min_markers=1,
        total_frames_input=len(marker_detections),
        frames_filtered_out=0,
        frames=list(marker_detections),
        marker_detections={
            name: [MarkerDetection(id=i, corners=c) for i, c in enumerate(corners)]
            for name, corners in marker_detections.items()
        },
    )


def write_frame(path: Path, w: int = 320, h: int = 240) -> None:
    cv2.imwrite(str(path), np.zeros((h, w, 3), dtype=np.uint8))


# --- generate_mask (single frame) ---


def test_hull_interior_white_background_black(tmp_path: Path) -> None:
    corners = [square(100, 100, 20), square(200, 100, 20), square(150, 180, 20)]
    mask = generate_mask(corners, width=320, height=240)

    assert mask is not None
    assert mask.shape == (240, 320)
    assert set(np.unique(mask)) <= {0, 255}
    assert mask[130, 150] == 255  # centroid of the three markers: inside hull
    assert mask[5, 5] == 0  # far corner: background
    assert mask[235, 315] == 0


def test_margin_extends_beyond_hull(tmp_path: Path) -> None:
    side = 20.0
    corners = [square(100, 100, side), square(200, 100, side), square(150, 180, side)]
    mask = generate_mask(corners, width=320, height=240)

    # A point just outside the raw hull but within margin (2 x side = 40 px)
    # above the top edge (y = 90 at the marker tops) must be kept.
    assert mask is not None
    assert mask[90 - int(HULL_MARGIN_MARKER_SIDES * side) + 5, 150] == 255


def test_too_few_markers_returns_none(tmp_path: Path) -> None:
    corners = [square(100, 100, 20)] * (MIN_MARKERS_FOR_HULL - 1)
    assert generate_mask(corners, width=320, height=240) is None


# --- generate_masks (session) ---


def test_generate_masks_writes_colmap_convention_files(tmp_path: Path) -> None:
    filtered = tmp_path / "filtered"
    filtered.mkdir()
    write_frame(filtered / "frame_0000.jpg")
    write_frame(filtered / "frame_0001.jpg")

    manifest = make_manifest(
        {
            "frame_0000.jpg": [square(100, 100, 20), square(200, 100, 20), square(150, 180, 20)],
            "frame_0001.jpg": [square(100, 100, 20)],  # too few -> full-white fallback
        }
    )

    stats = generate_masks(manifest, tmp_path)

    m0 = cv2.imread(str(filtered / "masks" / "frame_0000.jpg.png"), cv2.IMREAD_GRAYSCALE)
    m1 = cv2.imread(str(filtered / "masks" / "frame_0001.jpg.png"), cv2.IMREAD_GRAYSCALE)
    assert m0 is not None and m1 is not None
    assert m0[5, 5] == 0  # restrictive mask
    assert m1.min() == 255  # full-white fallback keeps everything
    assert stats["frames_masked"] == 1
    assert stats["frames_fallback_full"] == 1
    assert stats["margin_marker_sides"] == HULL_MARGIN_MARKER_SIDES


def test_generate_masks_updates_manifest_on_disk(tmp_path: Path) -> None:
    filtered = tmp_path / "filtered"
    filtered.mkdir()
    write_frame(filtered / "frame_0000.jpg")
    manifest = make_manifest(
        {"frame_0000.jpg": [square(100, 100, 20), square(200, 100, 20), square(150, 180, 20)]}
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest.to_dict()))

    generate_masks(manifest, tmp_path, manifest_path=tmp_path / "manifest.json")

    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data["mask_dir"] == "masks"
    assert data["mask_generation"]["frames_masked"] == 1
    # SfM handoff intact
    assert data["frames"] == ["frame_0000.jpg"]
