"""Stage 3c: MobileSAM head-silhouette masks (Arms 2 and 3 of the masking comparison).

The ArUco-hull masks (mask_generation.py) protect only the marker-covered
crown; occlusion-boundary background bleeding happens in the marker-free
regions (cheek, forehead, base). This stage segments the FULL head silhouette
per frame with MobileSAM so the mask also covers marker-free skin.

Prompting: marker CENTROIDS are deliberately NOT used as prompts — they sit on
the white marker patches, and SAM prompted there returns the marker patch, not
the head (measured: median mask = 32% of even the hull area; very likely the
mechanism behind the original "ate the head" failure). Instead, prompts are
points guaranteed to lie on the head SURFACE: the hull centroid plus the
midpoints between nearest-neighbour marker pairs, kept only where the image is
dark (the scalp/cap between markers). SAM runs with multimask_output=True and
the head is selected from the mask hierarchy: the SMALLEST mask that contains
>=CONTAINMENT_MIN of the marker centroids (the head carries the markers) and
whose non-marker interior is mostly dark (bright fraction <= BRIGHT_MAX —
rejects scene-level masks full of desk/wall). Assumes the head surface
(phantom / hair / cap) is darker than the white markers.

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

Anchored mode (--anchor-hull, Arm 3): the final mask is the UNION of the
dilated MobileSAM silhouette and the dilated ArUco-hull mask
(mask_generation.generate_mask). Resolution rule when they disagree: the hull
wins — the hull region is a protected core that segmentation can never remove
(it is *certainly* head, being spanned by the physical markers), so MobileSAM
can only ADD coverage (cheek/forehead/base), never erode it. Degradation:
SAM∪hull -> SAM alone (hull needs >=3 markers) -> hull alone (SAM failed) ->
full-white (both failed).

Outputs are written next to the existing masks without touching them:
masks -> `filtered/masks_sam/` (blind) or `filtered/masks_sam_hull/`
(anchored), manifest -> `manifest_sam.json` / `manifest_sam_hull.json` (a copy
of the input manifest with mask_dir/mask_generation replaced), so the frozen
Arm 1 artifacts stay intact.
"""

import argparse
import hashlib
import logging
from pathlib import Path

import cv2
import numpy as np

import pipeline_io
from mask_generation import generate_mask as generate_hull_mask
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
MASK_DIR_NAME_ANCHORED = "masks_sam_hull"
DEFAULT_WEIGHTS = "mobile_sam.pt"

# Prompt points must land on the head surface: pixels at least this dark.
DARK_PROMPT_THR = 100
# Head-mask selection: minimum fraction of marker centroids the mask must
# contain (the markers are ON the head) ...
CONTAINMENT_MIN = 0.8
# ... and maximum fraction of bright pixels in its non-marker interior
# (rejects scene-level masks that swallow desk/wall/mug).
BRIGHT_MAX = 0.25
BRIGHT_PX_THR = 150


def prompt_points(marker_corners: list) -> np.ndarray:
    """One point per marker: the centroid of its 4 corners. (N, 2)."""
    return np.array(
        [
            np.asarray(c, dtype=np.float64).reshape(-1, 2).mean(axis=0)
            for c in marker_corners
        ]
    )


def dark_prompt_points(
    centroids: np.ndarray, gray: np.ndarray, dark_thr: int = DARK_PROMPT_THR
) -> np.ndarray:
    """Prompt candidates on the head SURFACE, filtered to dark pixels.

    Candidates: the centroid of all marker centroids (crown centre) plus the
    midpoint between each marker and its nearest neighbour (scalp/cap fabric
    between adjacent markers). Marker centroids themselves are excluded — they
    sit on white patches and make SAM segment the marker. (N, 2), possibly
    empty.
    """
    candidates = [centroids.mean(axis=0)]
    if len(centroids) > 1:
        for i in range(len(centroids)):
            d = np.linalg.norm(centroids - centroids[i], axis=1)
            d[i] = np.inf
            j = int(np.argmin(d))
            candidates.append((centroids[i] + centroids[j]) / 2.0)

    h, w = gray.shape[:2]
    keep = []
    for x, y in candidates:
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if 0 <= xi < w and 0 <= yi < h and gray[yi, xi] < dark_thr:
            keep.append([float(x), float(y)])
    return np.array(keep) if keep else np.empty((0, 2))


def select_head_mask(
    masks: np.ndarray,
    centroids: np.ndarray,
    gray: np.ndarray,
    median_side: float,
) -> np.ndarray | None:
    """Pick the head from SAM's mask hierarchy.

    The head is the SMALLEST candidate that (a) contains at least
    CONTAINMENT_MIN of the marker centroids and (b) is mostly dark outside the
    marker patches (bright fraction <= BRIGHT_MAX). Larger qualifying masks are
    scene-level unions (head + desk + wall); smaller ones are marker patches or
    cap fragments that fail containment. Returns a bool (H, W) mask or None.
    """
    h, w = gray.shape[:2]
    marker_disks = np.zeros((h, w), dtype=np.uint8)
    for x, y in centroids:
        cv2.circle(
            marker_disks,
            (int(round(float(x))), int(round(float(y)))),
            max(int(round(median_side)), 1),
            255,
            -1,
        )

    best: np.ndarray | None = None
    for m in masks:
        mm = m.astype(bool)
        inside = 0
        for x, y in centroids:
            xi = int(round(float(x)))
            yi = int(round(float(y)))
            if 0 <= xi < w and 0 <= yi < h and mm[yi, xi]:
                inside += 1
        if inside / len(centroids) < CONTAINMENT_MIN:
            continue
        interior = mm & (marker_disks == 0)
        if not interior.any():
            continue
        if float((gray[interior] > BRIGHT_PX_THR).mean()) > BRIGHT_MAX:
            continue
        if best is None or mm.sum() < best.sum():
            best = mm
    return best


def median_marker_side(marker_corners: list) -> float:
    """Median marker side length in pixels across all markers in a frame."""
    sides = []
    for c in marker_corners:
        corners = np.asarray(c, dtype=np.float64).reshape(-1, 2)
        sides.extend(np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1))
    return float(np.median(sides))


def sam_margin_px(marker_corners: list) -> int:
    """Dilation margin in pixels from the frame's median marker side."""
    side = median_marker_side(marker_corners)
    return max(int(round(SAM_MARGIN_MARKER_SIDES * side)), MIN_MARGIN_PX)


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
    """Load the MobileSAM predictor (requires the `sam` extra).

    The Predictor is used directly (not the SAM model wrapper) because only
    the predictor exposes multimask_output, which the head selection needs.
    """
    try:
        from ultralytics.models.sam import Predictor as SAMPredictor
        from ultralytics.utils.downloads import attempt_download_asset
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise SystemExit(
            "ultralytics is required for SAM mask generation: "
            "install with `uv sync --extra sam`."
        ) from exc
    # Resolve/download the weights up front so their SHA-256 can be pinned in
    # the manifest before inference starts.
    weights_file = str(attempt_download_asset(weights))
    predictor = SAMPredictor(
        overrides=dict(
            task="segment",
            mode="predict",
            imgsz=1024,
            model=weights_file,
            device=device,
            verbose=False,
            save=False,
        )
    )
    return predictor, weights_file


def segment_frame(
    model,
    img_bgr: np.ndarray,
    prompts: np.ndarray,
    centroids: np.ndarray,
    median_side: float,
) -> np.ndarray | None:
    """Run MobileSAM (multimask) and select the head from the hierarchy.

    Returns a uint8 {0,255} mask at frame resolution, or None on failure.
    """
    model.set_image(img_bgr)
    results = model(
        points=prompts.tolist(),
        labels=[1] * len(prompts),
        multimask_output=True,
    )
    if not results or results[0].masks is None:
        return None
    data = results[0].masks.data.cpu().numpy()
    h, w = img_bgr.shape[:2]
    if data.shape[1:] != (h, w):
        data = np.stack(
            [
                cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                for m in data
            ]
        )
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    head = select_head_mask(data, centroids, gray, median_side)
    if head is None:
        return None
    return head.astype(np.uint8) * 255


def _weights_provenance(weights_file: str, device: str) -> dict:
    """Pin the exact weights and inference stack used."""
    import torch
    import ultralytics

    ckpt_path = Path(weights_file)
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
    anchor_hull: bool = False,
) -> dict:
    """Write SAM head masks for every manifest frame; write an updated manifest copy."""
    filtered_dir = session_dir / "filtered"
    mask_dir_name = MASK_DIR_NAME_ANCHORED if anchor_hull else MASK_DIR_NAME
    masks_dir = filtered_dir / mask_dir_name
    masks_dir.mkdir(parents=True, exist_ok=True)

    frames_masked = 0
    frames_fallback = 0
    frames_hull_only = 0
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
        sam_mask = None
        corners = [d.corners for d in detections]
        if detections:
            centroids = prompt_points(corners)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            prompts = dark_prompt_points(centroids, gray)
            if len(prompts) == 0:
                logger.warning("No dark on-head prompt on %s.", filename)
            else:
                raw = segment(
                    model, img, prompts, centroids, median_marker_side(corners)
                )
                if raw is not None:
                    sam_mask = postprocess_mask(raw, prompts, sam_margin_px(corners))
                min_area = MIN_AREA_FRACTION * height * width
                if sam_mask is not None and sam_mask.sum() / 255 < min_area:
                    logger.warning("SAM mask implausibly small on %s.", filename)
                    sam_mask = None

        # Anchored mode: the hull is a protected core the segmentation can
        # never remove — union it in (hull wins wherever SAM disagrees).
        hull_mask = None
        if anchor_hull and detections:
            hull_mask = generate_hull_mask(corners, width, height)

        if sam_mask is not None and hull_mask is not None:
            mask = cv2.bitwise_or(sam_mask, hull_mask)
            frames_masked += 1
        elif sam_mask is not None:
            mask = sam_mask
            frames_masked += 1
        elif hull_mask is not None:
            mask = hull_mask
            frames_hull_only += 1
            logger.warning("SAM failed on %s — hull-only mask.", filename)
        else:
            mask = np.full((height, width), 255, dtype=np.uint8)
            frames_fallback += 1
            logger.warning("No usable mask on %s — full-white fallback.", filename)

        cv2.imwrite(str(masks_dir / f"{filename}.png"), mask)

    stats = {
        "method": "mobilesam_hull_anchored" if anchor_hull else "mobilesam",
        "frames_masked": frames_masked,
        "frames_hull_only": frames_hull_only,
        "frames_fallback_full": frames_fallback,
        "margin_marker_sides": SAM_MARGIN_MARKER_SIDES,
        "min_margin_px": MIN_MARGIN_PX,
        "min_area_fraction": MIN_AREA_FRACTION,
        "dark_prompt_thr": DARK_PROMPT_THR,
        "containment_min": CONTAINMENT_MIN,
        "bright_max": BRIGHT_MAX,
        "bright_px_thr": BRIGHT_PX_THR,
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
        manifest.mask_dir = mask_dir_name
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
    parser.add_argument(
        "--anchor-hull",
        action="store_true",
        help=(
            "Arm 3: union the SAM silhouette with the dilated ArUco-hull mask "
            "(the hull is a protected core segmentation can never remove)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    pipeline_io.configure_logging()
    args = parse_args()

    manifest = FilterManifest.from_dict(
        pipeline_io.load_json(args.manifest, "manifest"), source=str(args.manifest)
    )
    session_dir = pipeline_io.session_dir(args.manifest)
    default_name = "manifest_sam_hull.json" if args.anchor_hull else "manifest_sam.json"
    manifest_out = args.manifest_out or args.manifest.parent / default_name

    model, weights_file = load_sam_model(args.weights, args.device)
    generate_sam_masks(
        manifest,
        session_dir,
        model,
        manifest_out=manifest_out,
        weights_provenance=_weights_provenance(weights_file, args.device),
        anchor_hull=args.anchor_hull,
    )


if __name__ == "__main__":
    main()
