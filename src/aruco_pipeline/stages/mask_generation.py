"""Stage 3b: Grounded-SAM-2 foreground masks for COLMAP feature extraction.

With markers rigidly attached to the head (cap), head + markers form one rigid
body: masking out background features makes subject motion equivalent to
camera motion for SfM and suppresses the contiguous-background failure mode.

Per frame, Grounded-SAM-2 (IDEA-Research) segments the subject: Grounding DINO
proposes boxes for a text prompt, SAM 2 turns one box into a silhouette. The
ArUco detections pick and backstop the prompt box:

1. the DINO box containing the most marker centers (ties → higher score);
2. else the marker-corner bbox + margin (>= MIN_MARKERS_FOR_BOX markers);
3. else a full-white (keep-all) mask.

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
import os
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

MASK_DIR_NAME = "masks"

SOURCE_DINO = "dino"
SOURCE_ARUCO_BOX = "aruco_box"
SOURCE_FALLBACK_FULL = "fallback_full"


class MaskModels(Protocol):
    """Detector + promptable segmenter used by `generate_mask`."""

    @property
    def info(self) -> dict[str, Any]:
        """Model/device description recorded in the manifest stats."""
        ...

    def detect(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (N, 4) xyxy boxes in pixels and (N,) scores."""
        ...

    def segment(self, rgb: np.ndarray, box: np.ndarray) -> np.ndarray:
        """Return a boolean (H, W) mask for the xyxy box prompt."""
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


class GroundedSam2:
    """Grounding DINO + SAM 2 via HF `transformers`; models load once.

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
        # The optional deeparuco extra installs TensorFlow; keep transformers
        # from importing it.
        os.environ.setdefault("USE_TF", "0")
        import torch
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            Sam2Model,
            Sam2Processor,
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
        self._seg_processor = Sam2Processor.from_pretrained(segmenter_model)
        seg_model: Any = Sam2Model.from_pretrained(segmenter_model)
        self._seg_model = seg_model.to(self.device).eval()

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

    def segment(self, rgb: np.ndarray, box: np.ndarray) -> np.ndarray:
        torch = self._torch
        inputs = self._seg_processor(
            images=rgb,
            input_boxes=[[[float(v) for v in box]]],
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode(), self._autocast():
            outputs = self._seg_model(**inputs, multimask_output=True)
        original_sizes: Any = inputs["original_sizes"]
        masks = self._seg_processor.post_process_masks(
            outputs.pred_masks.float().cpu(), original_sizes.cpu()
        )[0]  # (objects, candidates, H, W)
        scores = outputs.iou_scores.float().cpu()[0, 0]  # (candidates,)
        return masks[0, int(scores.argmax())].numpy().astype(bool)


def _marker_sides_px(corners: np.ndarray) -> np.ndarray:
    """Side lengths of one marker's 4-corner polygon, in pixels."""
    return np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)


def _as_corner_arrays(marker_corners: list) -> list[np.ndarray]:
    return [np.asarray(c, dtype=np.float64).reshape(-1, 2) for c in marker_corners]


def _median_side(corner_arrays: list[np.ndarray]) -> float:
    return float(
        np.median(np.concatenate([_marker_sides_px(c) for c in corner_arrays]))
    )


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


def generate_mask(
    rgb: np.ndarray, marker_corners: list, models: MaskModels
) -> tuple[np.ndarray, str]:
    """Foreground mask for one frame.

    Parameters
    ----------
    rgb : np.ndarray
        (H, W, 3) RGB frame.
    marker_corners : list
        One entry per marker: 4 [x, y] corner points.
    models : MaskModels
        Detector + segmenter.

    Returns
    -------
    tuple[np.ndarray, str]
        uint8 (H, W) mask with values {0, 255}, and the prompt source
        (`dino`, `aruco_box` or `fallback_full`).
    """
    height, width = rgb.shape[:2]
    full = np.full((height, width), 255, dtype=np.uint8)
    corner_arrays = _as_corner_arrays(marker_corners)
    centers = (
        np.array([c.mean(axis=0) for c in corner_arrays])
        if corner_arrays
        else np.empty((0, 2))
    )

    boxes, scores = models.detect(rgb)
    box = select_box(boxes, scores, centers)
    source = SOURCE_DINO
    if box is None:
        box = aruco_prompt_box(marker_corners, width, height)
        source = SOURCE_ARUCO_BOX
    if box is None:
        return full, SOURCE_FALLBACK_FULL

    segmentation = models.segment(rgb, box)
    if not segmentation.any() and not corner_arrays:
        return full, SOURCE_FALLBACK_FULL

    mask = np.where(segmentation, 255, 0).astype(np.uint8)
    if corner_arrays:
        polys = [np.round(c).astype(np.int32) for c in corner_arrays]
        cv2.fillPoly(mask, polys, 255)
        margin_px = int(round(MASK_DILATE_MARKER_SIDES * _median_side(corner_arrays)))
        if margin_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1)
            )
            mask = cv2.dilate(mask, kernel)
    return mask, source


def generate_masks(
    manifest: FilterManifest,
    session_dir: Path,
    models: MaskModels,
    manifest_path: Path | None = None,
) -> dict:
    """Write masks for every manifest frame; update the manifest on disk.

    Parameters
    ----------
    manifest : FilterManifest
        Filtering manifest listing frames and their marker detections.
    session_dir : Path
        Session directory holding `filtered/`.
    models : MaskModels
        Detector + segmenter.
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

    counts = {SOURCE_DINO: 0, SOURCE_ARUCO_BOX: 0, SOURCE_FALLBACK_FULL: 0}
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
        mask, source = generate_mask(rgb, [d.corners for d in detections], models)
        counts[source] += 1

        cv2.imwrite(str(masks_dir / f"{filename}.png"), mask)

    stats = {
        "frames_dino": counts[SOURCE_DINO],
        "frames_aruco_box": counts[SOURCE_ARUCO_BOX],
        "frames_fallback_full": counts[SOURCE_FALLBACK_FULL],
        **models.info,
    }
    logger.info(
        "Mask generation complete: %d DINO, %d ArUco-box, %d full-white → '%s'.",
        counts[SOURCE_DINO],
        counts[SOURCE_ARUCO_BOX],
        counts[SOURCE_FALLBACK_FULL],
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
    models = GroundedSam2(
        text_prompt=args.text_prompt or cfg.text_prompt,
        detector_model=cfg.detector_model,
        segmenter_model=cfg.segmenter_model,
        device=args.device or cfg.device,
    )
    generate_masks(manifest, session_dir, models, manifest_path=args.manifest)


if __name__ == "__main__":
    main()
