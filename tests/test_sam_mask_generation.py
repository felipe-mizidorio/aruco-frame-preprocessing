import json
from pathlib import Path

import cv2
import numpy as np

from sam_mask_generation import (
    MIN_MARGIN_PX,
    SAM_MARGIN_MARKER_SIDES,
    dark_prompt_points,
    generate_sam_masks,
    postprocess_mask,
    prompt_points,
    sam_margin_px,
    select_head_mask,
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


# --- dark_prompt_points ---


def test_dark_prompts_keep_only_dark_pixels() -> None:
    gray = np.full((240, 320), 200, dtype=np.uint8)  # bright everywhere...
    gray[:, :160] = 30  # ...except the left half (dark head surface)
    centroids = np.array([[100.0, 100.0], [140.0, 100.0], [250.0, 100.0]])

    pts = dark_prompt_points(centroids, gray)

    # Candidates: mean(163, 100) bright -> dropped; midpoints (120,100) dark,
    # (120,100) dark (mutual nearest), (195,100) bright -> dropped.
    assert len(pts) > 0
    for x, y in pts:
        assert gray[int(round(y)), int(round(x))] < 100


def test_dark_prompts_exclude_marker_centroids_themselves() -> None:
    gray = np.zeros((240, 240), dtype=np.uint8)
    centroids = np.array([[60.0, 60.0], [180.0, 60.0]])

    pts = dark_prompt_points(centroids, gray)

    for c in centroids:
        assert not any(np.allclose(p, c) for p in pts)


def test_dark_prompts_empty_when_everything_bright() -> None:
    gray = np.full((240, 240), 220, dtype=np.uint8)
    centroids = np.array([[60.0, 60.0], [180.0, 60.0]])
    assert len(dark_prompt_points(centroids, gray)) == 0


# --- select_head_mask ---


def _head_scene(w: int = 320, h: int = 240):
    """Dark head blob carrying two bright 'markers' on a bright background."""
    gray = np.full((h, w), 220, dtype=np.uint8)  # bright desk/wall
    cv2.circle(gray, (160, 120), 70, 40, -1)  # dark head
    centroids = np.array([[140.0, 100.0], [180.0, 100.0]])
    for x, y in centroids:
        cv2.rectangle(gray, (int(x) - 8, int(y) - 8), (int(x) + 8, int(y) + 8), 255, -1)
    head = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(head, (160, 120), 70, 1, -1)
    marker = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(marker, (132, 92), (148, 108), 1, -1)  # one marker patch only
    scene = np.ones((h, w), dtype=np.uint8)  # everything
    return gray, centroids, head, marker, scene


def test_selects_head_from_hierarchy() -> None:
    gray, centroids, head, marker, scene = _head_scene()

    picked = select_head_mask(
        np.stack([marker, head, scene]), centroids, gray, median_side=16.0
    )

    assert picked is not None
    np.testing.assert_array_equal(picked, head.astype(bool))


def test_rejects_all_when_only_marker_patch_qualifies_containment() -> None:
    gray, centroids, head, marker, scene = _head_scene()
    # Marker patch: fails containment (covers 1 of 2 centroids).
    # Scene: fails the bright-fraction test. No head candidate offered.
    picked = select_head_mask(
        np.stack([marker, scene]), centroids, gray, median_side=16.0
    )
    assert picked is None


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


def fake_segment_head(model, img, prompts, centroids, median_side):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (160, 120), 80, 255, -1)
    return mask


def fake_segment_fail(model, img, prompts, centroids, median_side):
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


# --- anchored mode (Arm 3) ---


def test_anchored_mask_is_union_and_hull_wins(tmp_path: Path) -> None:
    # SAM mask deliberately EXCLUDES one marker (the historical failure);
    # the anchored mask must still keep that marker's hull region.
    def sam_missing_a_marker(model, img, prompts, centroids, median_side):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (120, 120), 60, 255, -1)  # covers markers at ~(100..150)
        return mask  # does NOT cover the marker at (250, 180)

    filtered = tmp_path / "filtered"
    filtered.mkdir()
    write_frame(filtered / "frame_0000.jpg")
    manifest = make_manifest(
        {
            "frame_0000.jpg": [
                square(100, 100, 20),
                square(150, 120, 20),
                square(250, 180, 20),  # hull must protect this one
            ]
        }
    )
    manifest_out = tmp_path / "manifest_sam_hull.json"

    stats = generate_sam_masks(
        manifest,
        tmp_path,
        model=None,
        manifest_out=manifest_out,
        segment=sam_missing_a_marker,
        anchor_hull=True,
    )

    mask = cv2.imread(
        str(filtered / "masks_sam_hull" / "frame_0000.jpg.png"), cv2.IMREAD_GRAYSCALE
    )
    assert mask is not None
    assert mask[120, 120] == 255  # SAM region kept
    assert mask[180, 250] == 255  # hull wins where SAM disagreed
    assert mask[5, 315] == 0  # background still dropped
    assert stats["method"] == "mobilesam_hull_anchored"
    assert stats["frames_masked"] == 1

    data = json.loads(manifest_out.read_text())
    assert data["mask_dir"] == "masks_sam_hull"


def test_anchored_mask_never_smaller_than_hull(tmp_path: Path) -> None:
    from mask_generation import generate_mask as generate_hull_mask

    def tiny_sam(model, img, prompts, centroids, median_side):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (100, 100), 40, 255, -1)
        return mask

    corners = [square(100, 100, 20), square(200, 100, 20), square(150, 180, 20)]
    filtered = tmp_path / "filtered"
    filtered.mkdir()
    write_frame(filtered / "frame_0000.jpg")
    manifest = make_manifest({"frame_0000.jpg": corners})

    generate_sam_masks(
        manifest, tmp_path, model=None, segment=tiny_sam, anchor_hull=True
    )

    mask = cv2.imread(
        str(filtered / "masks_sam_hull" / "frame_0000.jpg.png"), cv2.IMREAD_GRAYSCALE
    )
    hull = generate_hull_mask(corners, width=320, height=240)
    assert mask is not None and hull is not None
    # ultralytics (imported by other test modules) patches cv2.imread to
    # return (H, W, 1) for grayscale — normalize before array comparison.
    mask2d = mask[..., 0] if mask.ndim == 3 else mask
    assert ((hull > 0) & (mask2d == 0)).sum() == 0  # no hull pixel lost


def test_anchored_falls_back_to_hull_when_sam_fails(tmp_path: Path) -> None:
    filtered = tmp_path / "filtered"
    filtered.mkdir()
    write_frame(filtered / "frame_0000.jpg")
    manifest = make_manifest(
        {
            "frame_0000.jpg": [
                square(100, 100, 20),
                square(200, 100, 20),
                square(150, 180, 20),
            ]
        }
    )

    stats = generate_sam_masks(
        manifest, tmp_path, model=None, segment=fake_segment_fail, anchor_hull=True
    )

    mask = cv2.imread(
        str(filtered / "masks_sam_hull" / "frame_0000.jpg.png"), cv2.IMREAD_GRAYSCALE
    )
    assert mask is not None
    assert mask[130, 150] == 255  # hull interior kept
    assert mask[5, 5] == 0  # hull-only mask, not full-white
    assert stats["frames_hull_only"] == 1
    assert stats["frames_fallback_full"] == 0


def test_anchored_full_white_when_sam_and_hull_both_fail(tmp_path: Path) -> None:
    filtered = tmp_path / "filtered"
    filtered.mkdir()
    write_frame(filtered / "frame_0000.jpg")
    # Only one marker: hull needs >=3, and SAM fails -> full-white.
    manifest = make_manifest({"frame_0000.jpg": [square(160, 120, 20)]})

    stats = generate_sam_masks(
        manifest, tmp_path, model=None, segment=fake_segment_fail, anchor_hull=True
    )

    mask = cv2.imread(
        str(filtered / "masks_sam_hull" / "frame_0000.jpg.png"), cv2.IMREAD_GRAYSCALE
    )
    assert mask is not None
    assert mask.min() == 255
    assert stats["frames_fallback_full"] == 1


def test_tiny_mask_treated_as_failure(tmp_path: Path) -> None:
    def tiny_segment(model, img, prompts, centroids, median_side):
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
