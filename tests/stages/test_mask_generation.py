import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from aruco_pipeline.core.schemas import FilterManifest, MarkerDetection
from aruco_pipeline.stages import mask_generation
from aruco_pipeline.stages.mask_generation import (
    BOX_MARGIN_MARKER_SIDES,
    MIN_MARKERS_FOR_BOX,
    SOURCE_ARUCO_BOX,
    SOURCE_DINO,
    SOURCE_FALLBACK_FULL,
    aruco_prompt_box,
    generate_mask,
    generate_masks,
    select_box,
)


def square(cx: float, cy: float, side: float) -> list:
    h = side / 2.0
    return [
        [cx - h, cy - h],
        [cx + h, cy - h],
        [cx + h, cy + h],
        [cx - h, cy + h],
    ]


THREE_MARKERS = [square(100, 100, 20), square(200, 100, 20), square(150, 180, 20)]


class FakeModels:
    """Stand-in for GroundedSam2: fixed DINO boxes; SAM 2 fills the box."""

    def __init__(self, boxes=(), scores=(), empty_segment: bool = False) -> None:
        self.boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        self.scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        self.empty_segment = empty_segment
        self.segment_calls: list[np.ndarray] = []

    @property
    def info(self) -> dict:
        return {"device": "fake"}

    def detect(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.boxes, self.scores

    def segment(self, rgb: np.ndarray, box: np.ndarray) -> np.ndarray:
        self.segment_calls.append(np.asarray(box))
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        if not self.empty_segment:
            x0, y0, x1, y1 = (int(round(v)) for v in box)
            mask[y0 : y1 + 1, x0 : x1 + 1] = True
        return mask


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


def rgb(w: int = 320, h: int = 240) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# --- aruco_prompt_box ---


def test_aruco_box_grows_corner_bbox_by_margin() -> None:
    box = aruco_prompt_box(THREE_MARKERS, width=640, height=480)
    margin = BOX_MARGIN_MARKER_SIDES * 20
    assert box is not None
    np.testing.assert_allclose(
        box, [90 - margin, 90 - margin, 210 + margin, 190 + margin]
    )


def test_aruco_box_clipped_to_frame() -> None:
    box = aruco_prompt_box(THREE_MARKERS, width=200, height=150)
    assert box is not None
    assert box[0] >= 0 and box[1] >= 0
    assert box[2] == 199 and box[3] == 149


def test_aruco_box_none_below_min_markers() -> None:
    corners = [square(100, 100, 20)] * (MIN_MARKERS_FOR_BOX - 1)
    assert aruco_prompt_box(corners, width=320, height=240) is None


# --- select_box ---


def test_select_box_prefers_marker_box_over_higher_score() -> None:
    boxes = [[0, 0, 50, 50], [80, 80, 220, 200]]
    centers = np.array([[100, 100], [200, 100], [150, 180]])
    chosen = select_box(np.array(boxes), np.array([0.9, 0.5]), centers)
    assert chosen is not None
    np.testing.assert_array_equal(chosen, boxes[1])


def test_select_box_tie_broken_by_score() -> None:
    boxes = [[0, 0, 300, 300], [50, 50, 250, 250]]
    centers = np.array([[100, 100]])
    chosen = select_box(np.array(boxes), np.array([0.4, 0.7]), centers)
    assert chosen is not None
    np.testing.assert_array_equal(chosen, boxes[1])


def test_select_box_without_markers_takes_top_score() -> None:
    boxes = [[0, 0, 10, 10], [5, 5, 20, 20]]
    chosen = select_box(np.array(boxes), np.array([0.3, 0.8]), np.empty((0, 2)))
    assert chosen is not None
    np.testing.assert_array_equal(chosen, boxes[1])


def test_select_box_rejects_boxes_missing_all_markers() -> None:
    chosen = select_box(
        np.array([[0, 0, 10, 10]]), np.array([0.9]), np.array([[100, 100]])
    )
    assert chosen is None


def test_select_box_none_when_no_detections() -> None:
    assert select_box(np.empty((0, 4)), np.empty(0), np.array([[1, 1]])) is None


# --- generate_mask (single frame) ---


def test_dino_path_uses_dino_box() -> None:
    models = FakeModels(boxes=[[80, 70, 230, 200]], scores=[0.8])
    mask, source = generate_mask(rgb(), THREE_MARKERS, models)

    assert source == SOURCE_DINO
    np.testing.assert_array_equal(models.segment_calls[0], [80, 70, 230, 200])
    assert mask.shape == (240, 320) and mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 255}
    assert mask[130, 150] == 255
    assert mask[5, 5] == 0


def test_falls_back_to_aruco_box_when_dino_misses() -> None:
    models = FakeModels()
    mask, source = generate_mask(rgb(), THREE_MARKERS, models)

    assert source == SOURCE_ARUCO_BOX
    expected = aruco_prompt_box(THREE_MARKERS, width=320, height=240)
    np.testing.assert_allclose(models.segment_calls[0], expected)
    assert mask[5, 5] == 0


def test_full_white_when_no_prompt_available() -> None:
    models = FakeModels()
    mask, source = generate_mask(rgb(), [square(100, 100, 20)], models)

    assert source == SOURCE_FALLBACK_FULL
    assert mask.min() == 255
    assert models.segment_calls == []


def test_full_white_when_segment_empty_and_no_markers() -> None:
    models = FakeModels(boxes=[[10, 10, 50, 50]], scores=[0.9], empty_segment=True)
    mask, source = generate_mask(rgb(), [], models)
    assert source == SOURCE_FALLBACK_FULL
    assert mask.min() == 255


def test_marker_polygons_always_kept() -> None:
    # SAM 2 returns nothing; markers must still be white.
    models = FakeModels(boxes=[[80, 70, 230, 200]], scores=[0.8], empty_segment=True)
    mask, _ = generate_mask(rgb(), THREE_MARKERS, models)
    for cx, cy in [(100, 100), (200, 100), (150, 180)]:
        assert mask[cy, cx] == 255
    assert mask[5, 5] == 0


# --- generate_masks (session) ---


def test_generate_masks_writes_colmap_convention_files(tmp_path: Path) -> None:
    filtered = tmp_path / "filtered"
    filtered.mkdir()
    write_frame(filtered / "frame_0000.jpg")
    write_frame(filtered / "frame_0001.jpg")
    manifest = make_manifest(
        {
            "frame_0000.jpg": THREE_MARKERS,
            "frame_0001.jpg": [square(100, 100, 20)],  # too few, DINO misses
        }
    )

    stats = generate_masks(manifest, tmp_path, FakeModels())

    m0 = cv2.imread(
        str(filtered / "masks" / "frame_0000.jpg.png"), cv2.IMREAD_GRAYSCALE
    )
    m1 = cv2.imread(
        str(filtered / "masks" / "frame_0001.jpg.png"), cv2.IMREAD_GRAYSCALE
    )
    assert m0 is not None and m1 is not None
    assert m0[5, 5] == 0
    assert m1.min() == 255
    assert stats["frames_dino"] == 0
    assert stats["frames_aruco_box"] == 1
    assert stats["frames_fallback_full"] == 1
    assert stats["device"] == "fake"


def test_generate_masks_skips_unreadable_frame(tmp_path: Path) -> None:
    (tmp_path / "filtered").mkdir()
    manifest = make_manifest({"missing.jpg": THREE_MARKERS})

    stats = generate_masks(manifest, tmp_path, FakeModels())

    assert not (tmp_path / "filtered" / "masks" / "missing.jpg.png").exists()
    assert stats["frames_aruco_box"] == 0


def test_generate_masks_updates_manifest_on_disk(tmp_path: Path) -> None:
    filtered = tmp_path / "filtered"
    filtered.mkdir()
    write_frame(filtered / "frame_0000.jpg")
    manifest = make_manifest({"frame_0000.jpg": THREE_MARKERS})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest.to_dict()))

    models = FakeModels(boxes=[[80, 70, 230, 200]], scores=[0.8])
    generate_masks(manifest, tmp_path, models, manifest_path=tmp_path / "manifest.json")

    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data["mask_dir"] == "masks"
    assert data["mask_generation"]["frames_dino"] == 1
    # SfM handoff intact
    assert data["frames"] == ["frame_0000.jpg"]


# --- CLI / device ---


def test_cli_flags_default_to_none(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["aruco-mask", "--manifest", "m.json"])
    args = mask_generation.parse_args()
    assert args.text_prompt is None and args.device is None  # main() fills from config


def test_resolve_device_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        mask_generation.resolve_device("tpu")


# --- real models (opt-in) ---


@pytest.mark.slow
def test_grounded_sam2_smoke() -> None:
    """Loads the configured HF checkpoints; skipped unless already cached."""
    from huggingface_hub import try_to_load_from_cache

    from aruco_pipeline.config import load_config

    cfg = load_config().mask_generation
    for repo in (cfg.detector_model, cfg.segmenter_model):
        if not isinstance(try_to_load_from_cache(repo, "config.json"), str):
            pytest.skip(f"{repo} not in HF cache")

    models = mask_generation.GroundedSam2(
        cfg.text_prompt, cfg.detector_model, cfg.segmenter_model, cfg.device
    )
    mask, source = generate_mask(rgb(), THREE_MARKERS, models)
    assert mask.shape == (240, 320)
    assert source in {SOURCE_DINO, SOURCE_ARUCO_BOX, SOURCE_FALLBACK_FULL}
