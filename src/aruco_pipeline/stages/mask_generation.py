"""Stage 3b: Grounded-SAM-2 foreground masks for COLMAP feature extraction.

With markers rigidly attached to the head (cap), head + markers form one rigid
body: masking out background features makes subject motion equivalent to
camera motion for SfM and suppresses the contiguous-background failure mode.

The subject is anchored once with Grounded-SAM-2 (IDEA-Research) and then
tracked through the filtered frames, in manifest order, by the SAM 2 video
model. Prompting every frame independently let the mask's extent jump between
frames (head only, head + neck, head + background); the tracker's memory keeps
the object chosen at the anchor. Prompt boxes come from this chain:

1. the Grounding DINO box containing the most marker centers (ties → higher
   score);
2. else the marker-corner bbox + margin (>= MIN_MARKERS_FOR_BOX markers).

The ArUco detections also police the track. A tracked mask is trusted only
while it covers most of the frame's marker centers and its scale-free area does
not jump (`track_healthy`). On drift the current frame is re-prompted from the
chain above, keeping the track memory; if that fails too, the frame gets a
full-white (keep-all) mask and the next frame starts a fresh track.

The filled marker polygons are always unioned into the mask, then the mask is
dilated by a margin in marker-side units so edge features survive. Both models
run through HF `transformers` (no custom CUDA ops); weights download to the HF
cache (`HF_HOME`) on first run.

Mask convention (COLMAP): `filtered/masks/<image filename>.png` — the mask for
`frame_0000.jpg` is `frame_0000.jpg.png`. White (255) keeps features, black
(0) discards them.
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from ..config import load_config
from ..core import pipeline_io
from ..core.schemas import FilterManifest

logger = logging.getLogger(__name__)

# Grounding DINO thresholds, as in Grounded-SAM-2's HF demo.
BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.3
# Below this many distinct markers the corner bbox is too small to bound the
# head, so it is not used as a SAM 2 prompt.
MIN_MARKERS_FOR_BOX = 3
# Margin around the marker-corner bbox prompt, in units of the frame's median
# marker side. Markers sit on a cap on the crown; the head extends past the cap
# by roughly two marker sides (~100 mm for 50 mm markers).
BOX_MARGIN_MARKER_SIDES = 2.0
# Safety dilation of the final mask, in marker sides, so features right on the
# silhouette edge are not discarded.
MASK_DILATE_MARKER_SIDES = 0.25
# Track health: markers sit on the head, so a mask holding fewer than this
# fraction of the frame's marker centers has lost the head.
TRACK_MIN_MARKER_COVERAGE = 0.5
# Track health: the mask area in marker-side² units is scale-free; a jump by
# more than this factor against the last trusted frame means the mask spilled
# into the background (or collapsed). Gradual change (viewpoint) passes.
TRACK_MAX_AREA_JUMP = 2.0

MASK_DIR_NAME = "masks"

SOURCE_TRACKED = "tracked"
SOURCE_DINO = "dino"
SOURCE_ARUCO_BOX = "aruco_box"
SOURCE_FALLBACK_FULL = "fallback_full"

# The single object id tracked in the SAM 2 video session.
_OBJ_ID = 1


class MaskTracker(Protocol):
    """Text-prompted detector + box-prompted video segmenter.

    `start`, `step` and `correct` return a boolean (H, W) mask of the tracked
    object on the frame they act on.
    """

    @property
    def info(self) -> dict[str, Any]:
        """Model/device description recorded in the manifest stats."""
        ...

    def detect(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (N, 4) xyxy boxes in pixels and (N,) scores."""
        ...

    def start(self, rgb: np.ndarray, box: np.ndarray) -> np.ndarray:
        """Drop all tracking state; segment `rgb` from an xyxy box prompt."""
        ...

    def step(self, rgb: np.ndarray) -> np.ndarray:
        """Propagate the tracked object onto the next frame."""
        ...

    def correct(self, box: np.ndarray) -> np.ndarray:
        """Re-prompt the most recent frame with an xyxy box, keeping memory."""
        ...


def resolve_device(requested: str) -> str:
    """Map `auto|cuda|cpu` to a concrete torch device.

    Parameters
    ----------
    requested : str
        Requested device; `auto` picks CUDA when available.

    Returns
    -------
    str
        `"cuda"` or `"cpu"`.

    Raises
    ------
    RuntimeError
        If `cuda` is requested but unavailable.
    ValueError
        If `requested` is not one of `auto`, `cuda`, `cpu`.
    """
    import torch

    if requested not in ("auto", "cuda", "cpu"):
        raise ValueError(f"Unknown device '{requested}' (expected auto|cuda|cpu).")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Device 'cuda' requested but CUDA is not available.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        logger.warning("Running Grounded-SAM-2 on CPU: expect several s/frame.")
    return requested


class GroundedSam2Tracker:
    """Grounding DINO + SAM 2 video tracking via HF `transformers`.

    Models load once. SAM 2 runs a streaming session: frames are fed one at a
    time, so the video never has to be loaded up front.

    Parameters
    ----------
    text_prompt : str
        Grounding DINO prompt (lowercase, ending with '.').
    detector_model, segmenter_model : str
        HF hub ids of the Grounding DINO and SAM 2 checkpoints.
    device : str
        `auto`, `cuda` or `cpu`.
    """

    def __init__(
        self,
        text_prompt: str,
        detector_model: str,
        segmenter_model: str,
        device: str = "auto",
    ) -> None:
        import torch
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            Sam2VideoModel,
            Sam2VideoProcessor,
        )

        self._torch: Any = torch
        self.text_prompt = text_prompt
        self.detector_model = detector_model
        self.segmenter_model = segmenter_model
        self.device = resolve_device(device)
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        logger.info(
            "Loading %s and %s on %s ...", detector_model, segmenter_model, self.device
        )
        self._det_processor = AutoProcessor.from_pretrained(detector_model)
        self._det_model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(detector_model)
            .to(self.device)
            .eval()
        )
        self._seg_processor: Any = Sam2VideoProcessor.from_pretrained(segmenter_model)
        seg_model: Any = Sam2VideoModel.from_pretrained(segmenter_model)
        self._seg_model = seg_model.to(self.device).eval()
        # Past frames SAM 2 still reads: the last `num_maskmem` memories and
        # the last `max_object_pointers_in_encoder` object pointers.
        config = self._seg_model.config
        self._memory_window = max(
            config.num_maskmem, config.max_object_pointers_in_encoder
        )

        self._session: Any = None
        self._frame_idx = -1
        self._frame: Any = None  # processor inputs of the most recent frame

    @property
    def info(self) -> dict[str, Any]:
        return {
            "text_prompt": self.text_prompt,
            "detector_model": self.detector_model,
            "segmenter_model": self.segmenter_model,
            "device": self.device,
        }

    def _autocast(self):
        return self._torch.autocast(
            "cuda", dtype=self._torch.bfloat16, enabled=self.device == "cuda"
        )

    def detect(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self._torch
        height, width = rgb.shape[:2]
        inputs = self._det_processor(
            images=rgb, text=self.text_prompt, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode(), self._autocast():
            outputs = self._det_model(**inputs)
        result = self._det_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[(height, width)],
        )[0]
        boxes = result["boxes"].float().cpu().numpy().reshape(-1, 4)
        scores = result["scores"].float().cpu().numpy().reshape(-1)
        return boxes, scores

    def start(self, rgb: np.ndarray, box: np.ndarray) -> np.ndarray:
        self._session = self._seg_processor.init_video_session(
            inference_device=self.device
        )
        self._frame_idx = -1
        self._load_frame(rgb)
        self._add_box(box)
        return self._segment()

    def step(self, rgb: np.ndarray) -> np.ndarray:
        if self._session is None:
            raise RuntimeError("step() called before start().")
        self._load_frame(rgb)
        return self._segment()

    def correct(self, box: np.ndarray) -> np.ndarray:
        if self._session is None:
            raise RuntimeError("correct() called before start().")
        self._add_box(box)
        return self._segment()

    def _load_frame(self, rgb: np.ndarray) -> None:
        self._frame_idx += 1
        self._frame = self._seg_processor(
            images=rgb, device=self.device, return_tensors="pt"
        )

    def _add_box(self, box: np.ndarray) -> None:
        self._seg_processor.add_inputs_to_inference_session(
            inference_session=self._session,
            frame_idx=self._frame_idx,
            obj_ids=_OBJ_ID,
            input_boxes=[[[float(v) for v in box]]],
            original_size=self._frame.original_sizes[0],
        )

    def _segment(self) -> np.ndarray:
        torch = self._torch
        # Always pass `frame=`, also when correcting the current frame: that
        # keeps the model in streaming mode. The non-streaming path bounds its
        # memory lookups by `len(processed_frames)`, which pruning shrinks.
        with torch.inference_mode(), self._autocast():
            output = self._seg_model(
                inference_session=self._session,
                frame=self._frame.pixel_values[0],
                frame_idx=self._frame_idx,
            )
        masks = self._seg_processor.post_process_masks(
            [output.pred_masks.float().cpu()],
            original_sizes=self._frame.original_sizes.cpu(),
        )[0]  # (objects, 1, H, W)
        self._prune_session()
        return masks[0, 0].numpy().astype(bool)

    def _prune_session(self) -> None:
        """Drop session state SAM 2 no longer reads, so memory stays flat.

        A streaming session otherwise keeps every frame (1024² float) and every
        per-frame output. Relies on HF `Sam2VideoInferenceSession` internals
        (transformers 4.57): pixels are only read for the current frame;
        memory attention reads the conditioning (anchor) outputs plus the last
        `_memory_window` non-conditioning outputs. Re-prompts on a tracked
        frame are stored as non-conditioning outputs, so the conditioning set
        stays at the anchor frame.
        """
        session = self._session
        for idx in [i for i in session.processed_frames if i < self._frame_idx]:
            del session.processed_frames[idx]
        oldest = self._frame_idx - self._memory_window
        for outputs in session.output_dict_per_obj.values():
            non_cond = outputs["non_cond_frame_outputs"]
            for idx in [i for i in non_cond if i < oldest]:
                del non_cond[idx]


def _marker_sides_px(corners: np.ndarray) -> np.ndarray:
    """Side lengths of one marker's 4-corner polygon, in pixels."""
    return np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)


def _as_corner_arrays(marker_corners: list) -> list[np.ndarray]:
    return [np.asarray(c, dtype=np.float64).reshape(-1, 2) for c in marker_corners]


def _median_side(corner_arrays: list[np.ndarray]) -> float:
    return float(
        np.median(np.concatenate([_marker_sides_px(c) for c in corner_arrays]))
    )


def _marker_centers(corner_arrays: list[np.ndarray]) -> np.ndarray:
    if not corner_arrays:
        return np.empty((0, 2))
    return np.array([c.mean(axis=0) for c in corner_arrays])


def aruco_prompt_box(
    marker_corners: list, width: int, height: int
) -> np.ndarray | None:
    """SAM 2 box prompt from the marker corners, or None when too few markers.

    Parameters
    ----------
    marker_corners : list
        One entry per marker: 4 [x, y] corner points.
    width, height : int
        Frame dimensions in pixels.

    Returns
    -------
    np.ndarray or None
        xyxy box: corner bbox grown by BOX_MARGIN_MARKER_SIDES median marker
        sides, clipped to the frame.
    """
    if len(marker_corners) < MIN_MARKERS_FOR_BOX:
        return None
    corner_arrays = _as_corner_arrays(marker_corners)
    points = np.vstack(corner_arrays)
    margin = BOX_MARGIN_MARKER_SIDES * _median_side(corner_arrays)
    x0, y0 = points.min(axis=0) - margin
    x1, y1 = points.max(axis=0) + margin
    return np.array(
        [max(x0, 0.0), max(y0, 0.0), min(x1, width - 1.0), min(y1, height - 1.0)]
    )


def select_box(
    boxes: np.ndarray, scores: np.ndarray, marker_centers: np.ndarray
) -> np.ndarray | None:
    """Pick the DINO box that agrees with the ArUco markers.

    Parameters
    ----------
    boxes : np.ndarray
        (N, 4) xyxy candidate boxes.
    scores : np.ndarray
        (N,) detection scores.
    marker_centers : np.ndarray
        (M, 2) marker centers; may be empty.

    Returns
    -------
    np.ndarray or None
        The box containing the most marker centers (ties → higher score). With
        no markers, the top-scoring box. None when there are no boxes, or when
        markers exist but no box contains any (the detection is something else).
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(boxes) == 0:
        return None
    centers = np.asarray(marker_centers, dtype=np.float64).reshape(-1, 2)
    if len(centers) == 0:
        return boxes[int(np.argmax(scores))]

    inside = (
        (centers[None, :, 0] >= boxes[:, None, 0])
        & (centers[None, :, 0] <= boxes[:, None, 2])
        & (centers[None, :, 1] >= boxes[:, None, 1])
        & (centers[None, :, 1] <= boxes[:, None, 3])
    )
    counts = inside.sum(axis=1)
    if counts.max() == 0:
        return None
    best = max(range(len(boxes)), key=lambda i: (counts[i], scores[i]))
    return boxes[best]


def prompt_candidates(
    rgb: np.ndarray, marker_corners: list, detector: MaskTracker
) -> list[tuple[np.ndarray, str]]:
    """Box prompts for one frame, in the order they should be tried.

    Parameters
    ----------
    rgb : np.ndarray
        (H, W, 3) RGB frame.
    marker_corners : list
        One entry per marker: 4 [x, y] corner points.
    detector : MaskTracker
        Supplies the Grounding DINO boxes.

    Returns
    -------
    list[tuple[np.ndarray, str]]
        (xyxy box, source) pairs: the DINO box picked by `select_box`, then the
        ArUco box; either is omitted when unavailable.
    """
    height, width = rgb.shape[:2]
    centers = _marker_centers(_as_corner_arrays(marker_corners))
    candidates: list[tuple[np.ndarray, str]] = []
    boxes, scores = detector.detect(rgb)
    dino_box = select_box(boxes, scores, centers)
    if dino_box is not None:
        candidates.append((dino_box, SOURCE_DINO))
    aruco_box = aruco_prompt_box(marker_corners, width, height)
    if aruco_box is not None:
        candidates.append((aruco_box, SOURCE_ARUCO_BOX))
    return candidates


def finalize_mask(segmentation: np.ndarray, marker_corners: list) -> np.ndarray:
    """Union the marker polygons into a segmentation and dilate the result.

    Parameters
    ----------
    segmentation : np.ndarray
        Boolean (H, W) object mask.
    marker_corners : list
        One entry per marker: 4 [x, y] corner points.

    Returns
    -------
    np.ndarray
        uint8 (H, W) mask with values {0, 255}; dilated by
        MASK_DILATE_MARKER_SIDES median marker sides when markers exist.
    """
    mask = np.where(segmentation, 255, 0).astype(np.uint8)
    corner_arrays = _as_corner_arrays(marker_corners)
    if corner_arrays:
        polys = [np.round(c).astype(np.int32) for c in corner_arrays]
        cv2.fillPoly(mask, polys, 255)
        margin_px = int(round(MASK_DILATE_MARKER_SIDES * _median_side(corner_arrays)))
        if margin_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1)
            )
            mask = cv2.dilate(mask, kernel)
    return mask


def marker_coverage(segmentation: np.ndarray, marker_corners: list) -> float:
    """Fraction of marker centers inside the mask.

    Parameters
    ----------
    segmentation : np.ndarray
        Boolean (H, W) object mask.
    marker_corners : list
        One entry per marker (at least one): 4 [x, y] corner points.

    Returns
    -------
    float
        In [0, 1].
    """
    height, width = segmentation.shape
    centers = _marker_centers(_as_corner_arrays(marker_corners))
    cols = np.clip(np.round(centers[:, 0]).astype(int), 0, width - 1)
    rows = np.clip(np.round(centers[:, 1]).astype(int), 0, height - 1)
    return float(np.mean(segmentation[rows, cols]))


def normalized_area(segmentation: np.ndarray, marker_corners: list) -> float:
    """Mask area in units of median marker side², independent of image scale.

    Parameters
    ----------
    segmentation : np.ndarray
        Boolean (H, W) object mask.
    marker_corners : list
        One entry per marker (at least one): 4 [x, y] corner points.

    Returns
    -------
    float
        Pixel count divided by the squared median marker side.
    """
    side = _median_side(_as_corner_arrays(marker_corners))
    return float(np.count_nonzero(segmentation)) / side**2


def track_healthy(
    segmentation: np.ndarray, marker_corners: list, ref_area: float | None
) -> bool:
    """Whether a tracked or prompted mask agrees with the frame's markers.

    Parameters
    ----------
    segmentation : np.ndarray
        Boolean (H, W) object mask.
    marker_corners : list
        One entry per marker: 4 [x, y] corner points; may be empty.
    ref_area : float or None
        `normalized_area` of the last trusted frame; None skips the area check.

    Returns
    -------
    bool
        False for an empty mask, for a mask holding fewer than
        TRACK_MIN_MARKER_COVERAGE of the marker centers, or for an area jump
        beyond TRACK_MAX_AREA_JUMP. True when there are no markers to check.
    """
    if not segmentation.any():
        return False
    if not marker_corners:
        return True
    if marker_coverage(segmentation, marker_corners) < TRACK_MIN_MARKER_COVERAGE:
        return False
    if ref_area is not None:
        ratio = normalized_area(segmentation, marker_corners) / ref_area
        if not 1.0 / TRACK_MAX_AREA_JUMP <= ratio <= TRACK_MAX_AREA_JUMP:
            return False
    return True


class TrackingMasker:
    """Per-session state machine turning a `MaskTracker` into frame masks.

    Call once per frame, in video order. While searching, each frame's prompt
    candidates start a fresh track until one yields a healthy mask. While
    tracking, the object is propagated without detection; an unhealthy mask
    re-prompts the frame (keeping memory) and, if every re-prompt fails, the
    frame falls back to keep-all and the search restarts.

    Parameters
    ----------
    tracker : MaskTracker
        Detector + video segmenter.
    """

    def __init__(self, tracker: MaskTracker) -> None:
        self.tracker = tracker
        self.tracking = False
        self.ref_area: float | None = None
        self.reanchors = 0
        self.track_resets = 0

    def __call__(self, rgb: np.ndarray, marker_corners: list) -> tuple[np.ndarray, str]:
        """Mask for the next frame.

        Parameters
        ----------
        rgb : np.ndarray
            (H, W, 3) RGB frame.
        marker_corners : list
            One entry per marker: 4 [x, y] corner points.

        Returns
        -------
        tuple[np.ndarray, str]
            uint8 (H, W) mask with values {0, 255}, and its source (`tracked`,
            `dino`, `aruco_box` or `fallback_full`).
        """
        if self.tracking:
            segmentation = self.tracker.step(rgb)
            if self._accept(segmentation, marker_corners):
                return finalize_mask(segmentation, marker_corners), SOURCE_TRACKED
            for box, source in prompt_candidates(rgb, marker_corners, self.tracker):
                segmentation = self.tracker.correct(box)
                if self._accept(segmentation, marker_corners):
                    self.reanchors += 1
                    return finalize_mask(segmentation, marker_corners), source
            self.tracking = False
            self.ref_area = None
            self.track_resets += 1
        else:
            for box, source in prompt_candidates(rgb, marker_corners, self.tracker):
                segmentation = self.tracker.start(rgb, box)
                if self._accept(segmentation, marker_corners):
                    self.tracking = True
                    return finalize_mask(segmentation, marker_corners), source
        height, width = rgb.shape[:2]
        return np.full((height, width), 255, dtype=np.uint8), SOURCE_FALLBACK_FULL

    def _accept(self, segmentation: np.ndarray, marker_corners: list) -> bool:
        if not track_healthy(segmentation, marker_corners, self.ref_area):
            return False
        if marker_corners:
            self.ref_area = normalized_area(segmentation, marker_corners)
        return True


def generate_masks(
    manifest: FilterManifest,
    session_dir: Path,
    tracker: MaskTracker,
    manifest_path: Path | None = None,
) -> dict:
    """Track the subject through the manifest frames; write their masks.

    Parameters
    ----------
    manifest : FilterManifest
        Filtering manifest listing frames (in video order) and their marker
        detections.
    session_dir : Path
        Session directory holding `filtered/`.
    tracker : MaskTracker
        Detector + video segmenter.
    manifest_path : Path, optional
        When given, the manifest is updated with `mask_dir` and stats.

    Returns
    -------
    dict
        The mask_generation stats (also written into the manifest when
        `manifest_path` is given).
    """
    filtered_dir = session_dir / "filtered"
    masks_dir = filtered_dir / MASK_DIR_NAME
    masks_dir.mkdir(parents=True, exist_ok=True)

    masker = TrackingMasker(tracker)
    counts = {
        SOURCE_TRACKED: 0,
        SOURCE_DINO: 0,
        SOURCE_ARUCO_BOX: 0,
        SOURCE_FALLBACK_FULL: 0,
    }
    total = len(manifest.frames)

    for i, filename in enumerate(manifest.frames):
        pipeline_io.log_progress(i, total)

        frame_path = filtered_dir / filename
        bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if bgr is None:
            logger.warning("Frame unreadable, skipping mask: %s", frame_path)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        detections = manifest.marker_detections.get(filename, [])
        mask, source = masker(rgb, [d.corners for d in detections])
        counts[source] += 1

        cv2.imwrite(str(masks_dir / f"{filename}.png"), mask)

    stats = {
        "frames_tracked": counts[SOURCE_TRACKED],
        "frames_dino": counts[SOURCE_DINO],
        "frames_aruco_box": counts[SOURCE_ARUCO_BOX],
        "frames_fallback_full": counts[SOURCE_FALLBACK_FULL],
        "reanchors": masker.reanchors,
        "track_resets": masker.track_resets,
        **tracker.info,
    }
    logger.info(
        "Mask generation complete: %d tracked, %d DINO, %d ArUco-box, "
        "%d full-white (%d re-anchors, %d track resets) → '%s'.",
        counts[SOURCE_TRACKED],
        counts[SOURCE_DINO],
        counts[SOURCE_ARUCO_BOX],
        counts[SOURCE_FALLBACK_FULL],
        masker.reanchors,
        masker.track_resets,
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
            "Generate Grounded-SAM-2 foreground masks (COLMAP convention) from a "
            "filtering manifest, tracking the subject through the frames."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to the manifest.json produced by frame_filtering.py.",
    )
    parser.add_argument(
        "--text-prompt",
        default=None,
        help="Grounding DINO text prompt, lowercase ending with '.' "
        "(default: mask_generation.text_prompt in configs/pipeline.yaml).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default=None,
        help="Torch device (default: mask_generation.device in configs/pipeline.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    pipeline_io.configure_logging()
    args = parse_args()
    cfg = load_config().mask_generation

    manifest = FilterManifest.from_dict(
        pipeline_io.load_json(args.manifest, "manifest"), source=str(args.manifest)
    )
    session_dir = pipeline_io.session_dir(args.manifest)
    tracker = GroundedSam2Tracker(
        text_prompt=args.text_prompt or cfg.text_prompt,
        detector_model=cfg.detector_model,
        segmenter_model=cfg.segmenter_model,
        device=args.device or cfg.device,
    )
    generate_masks(manifest, session_dir, tracker, manifest_path=args.manifest)


if __name__ == "__main__":
    main()
