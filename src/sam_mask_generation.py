"""Stage 3c: MobileSAM head-silhouette masks (Arm 2 of the masking comparison).

The ArUco-hull masks (mask_generation.py) protect only the marker-covered
crown; occlusion-boundary background bleeding happens in the marker-free
regions (cheek, forehead, base). This stage segments the FULL head silhouette
per frame with MobileSAM, prompted by the detected marker centroids (points
that are certainly on the head), so the mask also covers marker-free skin.

Anti-erosion: the previous MobileSAM attempt failed by being slightly tight
and cutting into the cranium. The segmentation is therefore dilated by
SAM_MARGIN_MARKER_SIDES x the frame's median marker side (~10 mm of physical
margin for 20 mm markers) before use: under-masking background is recoverable
downstream, eating the head is not. The margin is derived from what the frame
itself provides — no user tuning (domestic-use constraint).

Mask convention (COLMAP): `filtered/<mask dir>/<image filename>.png`, white
(255) keeps, black (0) discards. Frames where segmentation fails or returns an
implausibly small region get a full-white mask (keep everything rather than
starve SfM), mirroring the hull-mask fallback.

Reproducibility: the manifest records the weights file SHA-256, ultralytics /
torch versions, and the inference device. Inference runs on CPU by default;
MobileSAM inference with fixed weights on CPU is deterministic.

Outputs are written next to the existing masks without touching them:
masks -> `filtered/masks_sam/`, manifest -> `manifest_sam.json` (a copy of the
input manifest with mask_dir/mask_generation replaced), so the frozen Arm 1
artifacts stay intact.
"""

import argparse
import hashlib
import logging
from pathlib import Path

import cv2
import numpy as np

import pipeline_io
from schemas import FilterManifest

logger = logging.getLogger(__name__)

# Dilation margin around the SAM silhouette, in units of the frame's median
# marker side (20 mm physical). 0.5 sides ~ 10 mm: an order of magnitude above
# SAM's typical boundary error, while adding only a thin recoverable background
# ring. Bias: keep head > trim background.
SAM_MARGIN_MARKER_SIDES = 0.5
# Never dilate by less than this, whatever the marker scale.
MIN_MARGIN_PX = 5
# A "head" mask smaller than this fraction of the frame is treated as a
# segmentation failure (the head dominates these captures).
MIN_AREA_FRACTION = 0.02

MASK_DIR_NAME = "masks_sam"
DEFAULT_WEIGHTS = "mobile_sam.pt"


def prompt_points(marker_corners: list) -> np.ndarray:
    """One prompt point per marker: the centroid of its 4 corners. (N, 2)."""
    return np.array(
        [
            np.asarray(c, dtype=np.float64).reshape(-1, 2).mean(axis=0)
            for c in marker_corners
        ]
    )


def sam_margin_px(marker_corners: list) -> int:
    """Dilation margin in pixels from the frame's median marker side."""
    sides = []
    for c in marker_corners:
        corners = np.asarray(c, dtype=np.float64).reshape(-1, 2)
        sides.extend(np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1))
    median_side = float(np.median(sides))
    return max(int(round(SAM_MARGIN_MARKER_SIDES * median_side)), MIN_MARGIN_PX)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes: anything not reachable from the border background."""
    h, w = mask.shape
    flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flood[1:-1, 1:-1] = mask
    ff_mask = np.zeros((h + 4, w + 4), dtype=np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    outside = flood[1:-1, 1:-1] == 255
    holes = (mask == 0) & ~outside
    out = mask.copy()
    out[holes] = 255
    return out


def postprocess_mask(
    raw_mask: np.ndarray, prompts: np.ndarray, margin_px: int
) -> np.ndarray | None:
    """Clean a raw SAM mask into the final head mask.

    Keeps only connected components containing at least one prompt point
    (drops stray background blobs), fills enclosed holes (markers/specular
    patches segmented out of the head must stay foreground), then dilates by
    margin_px. Returns None when no component contains a prompt.
    """
    binary = np.where(raw_mask > 0, np.uint8(255), np.uint8(0))
    n_labels, labels = cv2.connectedComponents(binary)
    h, w = binary.shape

    keep_labels = set()
    for x, y in prompts:
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if 0 <= xi < w and 0 <= yi < h and labels[yi, xi] != 0:
            keep_labels.add(int(labels[yi, xi]))
    if not keep_labels:
        return None

    mask = np.where(np.isin(labels, list(keep_labels)), np.uint8(255), np.uint8(0))
    mask = _fill_holes(mask)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1)
    )
    return cv2.dilate(mask, kernel)


def load_sam_model(weights: str, device: str):
    """Load the MobileSAM predictor (requires the `sam` extra)."""
    try:
        from ultralytics.models.sam import SAM
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise SystemExit(
            "ultralytics is required for SAM mask generation: "
            "install with `uv sync --extra sam`."
        ) from exc
    model = SAM(weights)
    model.to(device)
    return model


def segment_frame(model, img_bgr: np.ndarray, prompts: np.ndarray) -> np.ndarray | None:
    """Run MobileSAM with all prompt points as foreground; union the masks.

    Returns a uint8 {0,255} mask at frame resolution, or None on failure.
    """
    results = model(
        img_bgr,
        points=prompts.tolist(),
        labels=[1] * len(prompts),
        verbose=False,
    )
    if not results or results[0].masks is None:
        return None
    data = results[0].masks.data
    union = data.any(dim=0).cpu().numpy().astype(np.uint8) * 255
    h, w = img_bgr.shape[:2]
    if union.shape != (h, w):
        union = cv2.resize(union, (w, h), interpolation=cv2.INTER_NEAREST)
    return union


def _weights_provenance(model, weights_arg: str, device: str) -> dict:
    """Pin the exact weights and inference stack used."""
    import torch
    import ultralytics

    ckpt = getattr(model, "ckpt_path", None) or weights_arg
    ckpt_path = Path(ckpt)
    sha256 = (
        hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
        if ckpt_path.exists()
        else None
    )
    return {
        "weights_file": str(ckpt_path),
        "weights_sha256": sha256,
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "device": device,
        "determinism": (
            "MobileSAM inference with fixed weights on CPU is deterministic."
        ),
    }


def generate_sam_masks(
    manifest: FilterManifest,
    session_dir: Path,
    model,
    manifest_out: Path | None = None,
    weights_provenance: dict | None = None,
    segment=segment_frame,
) -> dict:
    """Write SAM head masks for every manifest frame; write an updated manifest copy."""
    filtered_dir = session_dir / "filtered"
    masks_dir = filtered_dir / MASK_DIR_NAME
    masks_dir.mkdir(parents=True, exist_ok=True)

    frames_masked = 0
    frames_fallback = 0
    total = len(manifest.frames)

    for i, filename in enumerate(manifest.frames):
        pipeline_io.log_progress(i, total)

        frame_path = filtered_dir / filename
        img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Frame unreadable, skipping mask: %s", frame_path)
            continue
        height, width = img.shape[:2]

        detections = manifest.marker_detections.get(filename, [])
        mask = None
        if detections:
            corners = [d.corners for d in detections]
            prompts = prompt_points(corners)
            raw = segment(model, img, prompts)
            if raw is not None:
                mask = postprocess_mask(raw, prompts, sam_margin_px(corners))
            min_area = MIN_AREA_FRACTION * height * width
            if mask is not None and mask.sum() / 255 < min_area:
                logger.warning(
                    "SAM mask implausibly small on %s — using full-white fallback.",
                    filename,
                )
                mask = None

        if mask is None:
            mask = np.full((height, width), 255, dtype=np.uint8)
            frames_fallback += 1
        else:
            frames_masked += 1

        cv2.imwrite(str(masks_dir / f"{filename}.png"), mask)

    stats = {
        "method": "mobilesam",
        "frames_masked": frames_masked,
        "frames_fallback_full": frames_fallback,
        "margin_marker_sides": SAM_MARGIN_MARKER_SIDES,
        "min_margin_px": MIN_MARGIN_PX,
        "min_area_fraction": MIN_AREA_FRACTION,
    }
    if weights_provenance is not None:
        stats["weights"] = weights_provenance
    logger.info(
        "SAM mask generation complete: %d mask(s), %d full-white fallback(s) -> '%s'.",
        frames_masked,
        frames_fallback,
        masks_dir,
    )

    if manifest_out is not None:
        manifest.mask_dir = MASK_DIR_NAME
        manifest.mask_generation = stats
        pipeline_io.save_json(manifest.to_dict(), manifest_out)
        logger.info("Manifest with SAM masks written: '%s'", manifest_out)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate MobileSAM head-silhouette masks (COLMAP convention) from "
            "a filtering manifest, prompted by ArUco marker centroids."
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
        "--manifest-out",
        type=Path,
        default=None,
        help="Output manifest path (default: manifest_sam.json next to --manifest).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_WEIGHTS,
        help="MobileSAM weights file (auto-downloaded by ultralytics if absent).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device (cpu is deterministic; cuda if available).",
    )
    return parser.parse_args()


def main() -> None:
    pipeline_io.configure_logging()
    args = parse_args()

    manifest = FilterManifest.from_dict(
        pipeline_io.load_json(args.manifest, "manifest"), source=str(args.manifest)
    )
    session_dir = pipeline_io.session_dir(args.manifest)
    manifest_out = args.manifest_out or args.manifest.parent / "manifest_sam.json"

    model = load_sam_model(args.weights, args.device)
    generate_sam_masks(
        manifest,
        session_dir,
        model,
        manifest_out=manifest_out,
        weights_provenance=_weights_provenance(model, args.weights, args.device),
    )


if __name__ == "__main__":
    main()
