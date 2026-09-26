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
    MASK_DILATE_MARKER_SIDES,
    MIN_MARKERS_FOR_BOX,
    SOURCE_ARUCO_BOX,
    SOURCE_DINO,
    aruco_prompt_box,
    finalize_mask,
    generate_masks,
    marker_coverage,
    normalized_area,
    prompt_candidates,
    select_box,
    track_healthy,
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
DINO_BOX = [80, 70, 230, 200]  # contains all three marker centers


def box_mask(box, w: int = 320, h: int = 240) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    mask[y0 : y1 + 1, x0 : x1 + 1] = True
    return mask


class FakeTracker:
    """Stand-in for GroundedSam2Tracker.

    Fixed DINO boxes; `start`/`correct` segment exactly their box; `step`
    repeats the last mask unless a scripted mask is queued in `steps`.
    """

    def __init__(
        self,
        boxes=(),
        scores=(),
        steps=(),
        fail_corrections: bool = False,
    ) -> None:
        self.boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        self.scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        self.steps = list(steps)
        self.fail_corrections = fail_corrections
        self.calls: list[str] = []
        self.prompts: list[np.ndarray] = []
        self.last = np.zeros((1, 1), dtype=bool)

    @property
    def info(self) -> dict:
        return {"device": "fake"}

    def detect(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append("detect")
        return self.boxes, self.scores

    def start(self, rgb: np.ndarray, box: np.ndarray) -> np.ndarray:
        self.calls.append("start")
        self.prompts.append(np.asarray(box))
        h, w = rgb.shape[:2]
        self.last = box_mask(box, w, h)
        return self.last.copy()

    def step(self, rgb: np.ndarray) -> np.ndarray:
        self.calls.append("step")
        if self.steps:
            self.last = self.steps.pop(0)
        return self.last.copy()

    def correct(self, box: np.ndarray) -> np.ndarray:
        self.calls.append("correct")
        self.prompts.append(np.asarray(box))
        h, w = self.last.shape
        self.last = (
            np.zeros((h, w), dtype=bool)
            if self.fail_corrections
            else box_mask(box, w, h)
        )
        return self.last.copy()


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


def run_session(
    tmp_path: Path, tracker: FakeTracker, n_frames: int = 3, corners=THREE_MARKERS
) -> tuple[dict, list[np.ndarray]]:
    filtered = tmp_path / "filtered"
    filtered.mkdir()
    names = [f"frame_{i:04d}.jpg" for i in range(n_frames)]
    for name in names:
        write_frame(filtered / name)
    stats = generate_masks(
        make_manifest({name: corners for name in names}), tmp_path, tracker
    )
    masks = [
        cv2.imread(str(filtered / "masks" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        for name in names
    ]
    return stats, masks


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


# --- prompt_candidates ---


def test_candidates_dino_box_first_then_aruco_box() -> None:
    tracker = FakeTracker(boxes=[DINO_BOX], scores=[0.8])
    candidates = prompt_candidates(rgb(), THREE_MARKERS, tracker)

    assert [source for _, source in candidates] == [SOURCE_DINO, SOURCE_ARUCO_BOX]
    np.testing.assert_array_equal(candidates[0][0], DINO_BOX)
    np.testing.assert_allclose(
        candidates[1][0], aruco_prompt_box(THREE_MARKERS, width=320, height=240)
    )


def test_candidates_aruco_box_only_when_dino_misses() -> None:
    candidates = prompt_candidates(rgb(), THREE_MARKERS, FakeTracker())
    assert [source for _, source in candidates] == [SOURCE_ARUCO_BOX]


def test_candidates_empty_without_any_prompt() -> None:
    assert prompt_candidates(rgb(), [square(100, 100, 20)], FakeTracker()) == []


# --- finalize_mask ---


def test_finalize_mask_binary_uint8() -> None:
    mask = finalize_mask(box_mask(DINO_BOX), THREE_MARKERS)
    assert mask.shape == (240, 320) and mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 255}
    assert mask[130, 150] == 255
    assert mask[5, 5] == 0


def test_finalize_mask_keeps_marker_polygons_dilated() -> None:
    mask = finalize_mask(np.zeros((240, 320), dtype=bool), THREE_MARKERS)
    margin = int(round(MASK_DILATE_MARKER_SIDES * 20))
    for cx, cy in [(100, 100), (200, 100), (150, 180)]:
        assert mask[cy, cx] == 255
    assert mask[100, 110 + margin - 1] == 255  # inside the dilation margin
    assert mask[100, 125] == 0  # beyond it
    assert mask[5, 5] == 0


def test_finalize_mask_without_markers_is_segmentation() -> None:
    seg = box_mask(DINO_BOX)
    np.testing.assert_array_equal(finalize_mask(seg, []), np.where(seg, 255, 0))


# --- health check ---


def test_marker_coverage_counts_centers_inside() -> None:
    assert marker_coverage(box_mask(DINO_BOX), THREE_MARKERS) == 1.0
    assert marker_coverage(box_mask([90, 90, 110, 110]), THREE_MARKERS) == 1 / 3


def test_normalized_area_is_scale_free() -> None:
    small = normalized_area(box_mask([0, 0, 9, 9]), [square(5, 5, 2)])
    big = normalized_area(box_mask([0, 0, 19, 19]), [square(10, 10, 4)])
    assert small == pytest.approx(big)


def test_unhealthy_when_empty() -> None:
    assert not track_healthy(np.zeros((240, 320), dtype=bool), THREE_MARKERS, None)


def test_healthy_without_markers_when_nonempty() -> None:
    assert track_healthy(box_mask(DINO_BOX), [], None)


def test_unhealthy_when_markers_left_outside() -> None:
    # Covers only one of three marker centers.
    assert not track_healthy(box_mask([90, 90, 110, 110]), THREE_MARKERS, None)
    # Covers two of three.
    assert track_healthy(box_mask([90, 90, 210, 110]), THREE_MARKERS, None)


def test_unhealthy_on_area_jump() -> None:
    ref = normalized_area(box_mask(DINO_BOX), THREE_MARKERS)
    assert track_healthy(box_mask(DINO_BOX), THREE_MARKERS, ref)
    full = np.ones((240, 320), dtype=bool)
    assert not track_healthy(full, THREE_MARKERS, ref)


# --- tracking state machine (generate_masks) ---


def test_first_frame_anchors_then_tracks(tmp_path: Path) -> None:
    tracker = FakeTracker(boxes=[DINO_BOX], scores=[0.8])
    stats, masks = run_session(tmp_path, tracker)

    assert tracker.calls == ["detect", "start", "step", "step"]
    assert stats["frames_dino"] == 1
    assert stats["frames_tracked"] == 2
    assert stats["reanchors"] == 0 and stats["track_resets"] == 0
    for mask in masks:
        assert mask is not None
        assert mask[130, 150] == 255 and mask[5, 5] == 0


def test_drift_reanchors_keeping_memory(tmp_path: Path) -> None:
    empty = np.zeros((240, 320), dtype=bool)
    tracker = FakeTracker(boxes=[DINO_BOX], scores=[0.8], steps=[empty])
    stats, masks = run_session(tmp_path, tracker)

    assert tracker.calls == ["detect", "start", "step", "detect", "correct", "step"]
    assert stats["frames_dino"] == 2
    assert stats["frames_tracked"] == 1
    assert stats["reanchors"] == 1 and stats["track_resets"] == 0
    assert masks[1] is not None and masks[1][5, 5] == 0


def test_leak_into_background_reanchors(tmp_path: Path) -> None:
    full = np.ones((240, 320), dtype=bool)
    tracker = FakeTracker(boxes=[DINO_BOX], scores=[0.8], steps=[full])
    stats, _ = run_session(tmp_path, tracker)

    assert "correct" in tracker.calls
    assert stats["reanchors"] == 1


def test_failed_reanchor_resets_to_search(tmp_path: Path) -> None:
    empty = np.zeros((240, 320), dtype=bool)
    tracker = FakeTracker(
        boxes=[DINO_BOX], scores=[0.8], steps=[empty], fail_corrections=True
    )
    stats, masks = run_session(tmp_path, tracker)

    # Frame 1: both candidates tried as corrections, then keep-all.
    assert tracker.calls.count("correct") == 2
    assert masks[1] is not None and masks[1].min() == 255
    # Frame 2: a fresh track is started.
    assert tracker.calls.count("start") == 2
    assert stats["frames_dino"] == 2
    assert stats["frames_fallback_full"] == 1
    assert stats["track_resets"] == 1 and stats["reanchors"] == 0


def test_aruco_box_anchors_when_dino_box_unhealthy(tmp_path: Path) -> None:
    # The DINO box holds one marker center: selected, but its mask misses two.
    tracker = FakeTracker(boxes=[[90, 90, 110, 110]], scores=[0.9])
    stats, _ = run_session(tmp_path, tracker, n_frames=1)

    assert tracker.calls == ["detect", "start", "start"]
    np.testing.assert_allclose(
        tracker.prompts[1], aruco_prompt_box(THREE_MARKERS, width=320, height=240)
    )
    assert stats["frames_aruco_box"] == 1 and stats["frames_dino"] == 0


def test_no_prompt_falls_back_full_and_keeps_searching(tmp_path: Path) -> None:
    tracker = FakeTracker()
    stats, masks = run_session(
        tmp_path, tracker, n_frames=2, corners=[square(100, 100, 20)]
    )

    assert tracker.calls == ["detect", "detect"]
    assert stats["frames_fallback_full"] == 2
    assert all(m is not None and m.min() == 255 for m in masks)


def test_generate_masks_writes_colmap_convention_files(tmp_path: Path) -> None:
    filtered = tmp_path / "filtered"
    filtered.mkdir()
    write_frame(filtered / "frame_0000.jpg")
    write_frame(filtered / "frame_0001.jpg")
    manifest = make_manifest(
        {
            "frame_0000.jpg": THREE_MARKERS,  # DINO misses → ArUco-box anchor
            "frame_0001.jpg": [square(100, 100, 20)],  # tracked
        }
    )

    stats = generate_masks(manifest, tmp_path, FakeTracker())

    m0 = cv2.imread(
        str(filtered / "masks" / "frame_0000.jpg.png"), cv2.IMREAD_GRAYSCALE
    )
    m1 = cv2.imread(
        str(filtered / "masks" / "frame_0001.jpg.png"), cv2.IMREAD_GRAYSCALE
    )
    assert m0 is not None and m1 is not None
    assert m0[5, 5] == 0 and m1[5, 5] == 0
    assert stats["frames_aruco_box"] == 1
    assert stats["frames_tracked"] == 1
    assert stats["frames_fallback_full"] == 0
    assert stats["device"] == "fake"


def test_generate_masks_skips_unreadable_frame(tmp_path: Path) -> None:
    (tmp_path / "filtered").mkdir()
    manifest = make_manifest({"missing.jpg": THREE_MARKERS})
    tracker = FakeTracker()

    stats = generate_masks(manifest, tmp_path, tracker)

    assert not (tmp_path / "filtered" / "masks" / "missing.jpg.png").exists()
    assert stats["frames_aruco_box"] == 0
    assert tracker.calls == []


def test_generate_masks_updates_manifest_on_disk(tmp_path: Path) -> None:
    filtered = tmp_path / "filtered"
    filtered.mkdir()
    write_frame(filtered / "frame_0000.jpg")
    manifest = make_manifest({"frame_0000.jpg": THREE_MARKERS})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest.to_dict()))

    tracker = FakeTracker(boxes=[DINO_BOX], scores=[0.8])
    generate_masks(
        manifest, tmp_path, tracker, manifest_path=tmp_path / "manifest.json"
    )

    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data["mask_dir"] == "masks"
    assert data["mask_generation"]["frames_dino"] == 1
    assert data["mask_generation"]["reanchors"] == 0
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
def test_grounded_sam2_tracker_smoke() -> None:
    """Loads the configured HF checkpoints; skipped unless already cached."""
    from huggingface_hub import try_to_load_from_cache

    from aruco_pipeline.config import load_config

    cfg = load_config().mask_generation
    for repo in (cfg.detector_model, cfg.segmenter_model):
        if not isinstance(try_to_load_from_cache(repo, "config.json"), str):
            pytest.skip(f"{repo} not in HF cache")

    tracker = mask_generation.GroundedSam2Tracker(
        cfg.text_prompt, cfg.detector_model, cfg.segmenter_model, cfg.device
    )
    box = np.array(DINO_BOX, dtype=np.float64)
    for mask in (
        tracker.start(rgb(), box),
        tracker.step(rgb()),
        tracker.correct(box),
        tracker.step(rgb()),
    ):
        assert mask.shape == (240, 320) and mask.dtype == bool
