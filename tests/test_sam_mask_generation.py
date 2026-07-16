import json
from pathlib import Path

import cv2
import numpy as np

from sam_mask_generation import (
    MIN_MARGIN_PX,
    SAM_MARGIN_MARKER_SIDES,
    generate_sam_masks,
    postprocess_mask,
    prompt_points,
    sam_margin_px,
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


# --- prompt_points / sam_margin_px ---


def test_prompt_points_are_marker_centroids() -> None:
    pts = prompt_points([square(100, 100, 20), square(200, 120, 20)])
    np.testing.assert_allclose(pts, [[100, 100], [200, 120]])


def test_margin_scales_with_marker_side() -> None:
    assert sam_margin_px([square(100, 100, 40)]) == int(
        round(SAM_MARGIN_MARKER_SIDES * 40)
    )


def test_margin_has_floor() -> None:
    assert sam_margin_px([square(100, 100, 2)]) == MIN_MARGIN_PX


# --- postprocess_mask ---


def test_keeps_only_components_containing_prompts() -> None:
    raw = np.zeros((240, 320), dtype=np.uint8)
    cv2.rectangle(raw, (50, 50), (150, 150), 255, -1)  # head blob (has prompt)
    cv2.rectangle(raw, (250, 10), (300, 60), 255, -1)  # stray background blob
    out = postprocess_mask(raw, np.array([[100.0, 100.0]]), margin_px=1)

    assert out is not None
    assert out[100, 100] == 255
    assert out[35, 275] == 0  # stray blob dropped


def test_fills_enclosed_holes() -> None:
    raw = np.zeros((240, 320), dtype=np.uint8)
    cv2.rectangle(raw, (50, 50), (150, 150), 255, -1)
    cv2.rectangle(raw, (90, 90), (110, 110), 0, -1)  # hole (e.g. marker segmented out)
    out = postprocess_mask(raw, np.array([[60.0, 60.0]]), margin_px=1)

    assert out is not None
    assert out[100, 100] == 255  # hole filled


def test_dilation_extends_mask_by_margin() -> None:
    raw = np.zeros((240, 320), dtype=np.uint8)
    cv2.rectangle(raw, (100, 100), (200, 200), 255, -1)
    out = postprocess_mask(raw, np.array([[150.0, 150.0]]), margin_px=10)

    assert out is not None
    assert out[95, 150] == 255  # 5 px above the blob: inside the margin
    assert out[80, 150] == 0  # 20 px above: outside the margin


def test_no_component_contains_prompt_returns_none() -> None:
    raw = np.zeros((240, 320), dtype=np.uint8)
    cv2.rectangle(raw, (250, 10), (300, 60), 255, -1)
    assert postprocess_mask(raw, np.array([[100.0, 100.0]]), margin_px=1) is None


# --- generate_sam_masks (session driver, mocked segmentation) ---


def fake_segment_head(model, img, prompts):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (160, 120), 80, 255, -1)
    return mask


def fake_segment_fail(model, img, prompts):
    return None


def test_generate_masks_writes_colmap_convention_and_manifest_copy(
    tmp_path: Path,
) -> None:
    filtered = tmp_path / "filtered"
    filtered.mkdir()
    write_frame(filtered / "frame_0000.jpg")
    manifest = make_manifest({"frame_0000.jpg": [square(160, 120, 20)]})
    manifest_out = tmp_path / "manifest_sam.json"

    stats = generate_sam_masks(
        manifest,
        tmp_path,
        model=None,
        manifest_out=manifest_out,
        weights_provenance={"weights_sha256": "abc"},
        segment=fake_segment_head,
    )

    mask = cv2.imread(
        str(filtered / "masks_sam" / "frame_0000.jpg.png"), cv2.IMREAD_GRAYSCALE
    )
    assert mask is not None
    assert mask[120, 160] == 255  # head kept
    assert mask[5, 5] == 0  # background dropped
    assert stats["frames_masked"] == 1
    assert stats["frames_fallback_full"] == 0

    data = json.loads(manifest_out.read_text())
    assert data["mask_dir"] == "masks_sam"
    assert data["mask_generation"]["method"] == "mobilesam"
    assert data["mask_generation"]["weights"]["weights_sha256"] == "abc"
    assert data["frames"] == ["frame_0000.jpg"]


def test_segmentation_failure_falls_back_to_full_white(tmp_path: Path) -> None:
    filtered = tmp_path / "filtered"
    filtered.mkdir()
    write_frame(filtered / "frame_0000.jpg")
    manifest = make_manifest({"frame_0000.jpg": [square(160, 120, 20)]})

    stats = generate_sam_masks(
        manifest, tmp_path, model=None, segment=fake_segment_fail
    )

    mask = cv2.imread(
        str(filtered / "masks_sam" / "frame_0000.jpg.png"), cv2.IMREAD_GRAYSCALE
    )
    assert mask is not None
    assert mask.min() == 255  # full-white fallback
    assert stats["frames_fallback_full"] == 1


def test_tiny_mask_treated_as_failure(tmp_path: Path) -> None:
    def tiny_segment(model, img, prompts):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (160, 120), 3, 255, -1)  # far below MIN_AREA_FRACTION
        return mask

    filtered = tmp_path / "filtered"
    filtered.mkdir()
    write_frame(filtered / "frame_0000.jpg")
    manifest = make_manifest({"frame_0000.jpg": [square(160, 120, 20)]})

    stats = generate_sam_masks(manifest, tmp_path, model=None, segment=tiny_segment)

    mask = cv2.imread(
        str(filtered / "masks_sam" / "frame_0000.jpg.png"), cv2.IMREAD_GRAYSCALE
    )
    assert mask is not None
    assert mask.min() == 255
    assert stats["frames_fallback_full"] == 1
