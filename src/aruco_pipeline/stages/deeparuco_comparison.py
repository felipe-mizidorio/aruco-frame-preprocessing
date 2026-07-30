import argparse
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import load_config
from ..core import pipeline_io
from ..core.schemas import ComparisonFile, DetectionsFile
from ..deeparuco_vendor.aruco import find_id
from ..deeparuco_vendor.heatmaps import pos_from_heatmap
from ..deeparuco_vendor.losses import weighted_loss
from ..deeparuco_vendor.utils import marker_from_corners, ordered_corners

logger = logging.getLogger(__name__)


DEFAULT_WEIGHTS_DIR = Path.home() / ".cache" / "deeparuco"

_MODEL_FILENAMES = {
    "detector": "det_luma_bc_s.pt",
    "regressor": "reg_hmap_8.h5",
    "decoder": "dec_new.h5",
}
_BASE_URL = "https://raw.githubusercontent.com/AVAuco/deeparuco/master/models"

# DeepArUco model calibration constants (from the upstream AVAuco/deeparuco pipeline).
_DECODE_THRESHOLD = 9  # max hamming distance to accept a detection
_DETECTOR_IOU = 0.5
_DETECTOR_CONF = 0.03
_BBOX_PADDING_RATIO = 0.2  # expand each detector box by this fraction on each side
_CROP_SIZE = 64  # detector crop resolution fed to the corner regressor
_MARKER_CROP_SIZE = 32  # perspective-warped marker resolution fed to the decoder
_BLOB_AREA_PX = 75  # expected keypoint blob area in the regressor's heatmap output
_BLOB_AREA_TOLERANCE = 0.2  # +/-20% around _BLOB_AREA_PX for the blob detector's filter


@dataclass
class DeepArucoModels:
    detector: Any
    regressor: Any
    decoder: Any
    refine_corners: Any
    decode_markers: Any


def _download_weights(
    target_dir: Path,
    base_url: str = _BASE_URL,
    filenames: dict[str, str] = _MODEL_FILENAMES,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames.values():
        target = target_dir / filename
        if not target.exists():
            logger.info("Downloading %s...", filename)
            try:
                urllib.request.urlretrieve(f"{base_url}/{filename}", target)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download {filename}. Download manually from "
                    "https://github.com/AVAuco/deeparuco/tree/master/models"
                ) from e


def load_deeparuco_models(
    weights_dir: Path | None = None,
    *,
    base_url: str = _BASE_URL,
    filenames: dict[str, str] = _MODEL_FILENAMES,
    default_dir: Path = DEFAULT_WEIGHTS_DIR,
) -> DeepArucoModels:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from ultralytics import YOLO

    if weights_dir is None:
        _download_weights(default_dir, base_url=base_url, filenames=filenames)
        weights_dir = default_dir

    for filename in filenames.values():
        path = weights_dir / filename
        if not path.exists():
            raise RuntimeError(
                f"Model file not found: {path}. Download from "
                "https://github.com/AVAuco/deeparuco/tree/master/models"
            )

    detector = YOLO(str(weights_dir / filenames["detector"]))
    regressor = load_model(
        str(weights_dir / filenames["regressor"]),
        custom_objects={"weighted_loss": weighted_loss},
    )
    decoder = load_model(str(weights_dir / filenames["decoder"]))

    @tf.function(reduce_retracing=True)
    def refine_corners(crops: Any) -> Any:
        return regressor(crops)

    @tf.function(reduce_retracing=True)
    def decode_markers_fn(markers: Any) -> Any:
        return decoder(markers)

    return DeepArucoModels(
        detector=detector,
        regressor=regressor,
        decoder=decoder,
        refine_corners=refine_corners,
        decode_markers=decode_markers_fn,
    )


def _norm(x: np.ndarray) -> np.ndarray:
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-9)


def run_deeparuco_on_image(
    image: np.ndarray,
    models: DeepArucoModels,
    threshold: int = _DECODE_THRESHOLD,
) -> tuple[list[list[list[float]]], list[int]]:
    detections = (
        models.detector(image, verbose=False, iou=_DETECTOR_IOU, conf=_DETECTOR_CONF)[0]
        .cpu()
        .boxes
    )
    if not len(detections):
        return [], []

    xyxy = [
        [
            int(max(det[0] - (_BBOX_PADDING_RATIO * (det[2] - det[0]) + 0.5), 0)),
            int(max(det[1] - (_BBOX_PADDING_RATIO * (det[3] - det[1]) + 0.5), 0)),
            int(
                min(
                    det[2] + (_BBOX_PADDING_RATIO * (det[2] - det[0]) + 0.5),
                    image.shape[1] - 1,
                )
            ),
            int(
                min(
                    det[3] + (_BBOX_PADDING_RATIO * (det[3] - det[1]) + 0.5),
                    image.shape[0] - 1,
                )
            ),
        ]
        for det in [
            [int(val) for val in det.xyxy.cpu().numpy()[0]] for det in detections
        ]
    ]

    crops_ori = [
        cv2.resize(image[det[1] : det[3], det[0] : det[2]], (_CROP_SIZE, _CROP_SIZE))
        for det in xyxy
    ]
    crops = [_norm(crop) for crop in crops_ori]

    corners_raw = models.refine_corners(np.array(crops)).numpy()

    kp_params = cv2.SimpleBlobDetector_Params()
    kp_params.filterByArea = True
    kp_params.minArea = _BLOB_AREA_PX * (1 - _BLOB_AREA_TOLERANCE)
    kp_params.maxArea = _BLOB_AREA_PX * (1 + _BLOB_AREA_TOLERANCE)
    kp_detector = cv2.SimpleBlobDetector_create(kp_params)

    corners_normalized = [
        [(x, y) for x, y in zip(*pos_from_heatmap(pred, kp_detector))]
        for pred in corners_raw
    ]

    keep = [len(cs) == 4 for cs in corners_normalized]
    filtered = [
        (det, crop, cs)
        for det, crop, cs, k in zip(xyxy, crops_ori, corners_normalized, keep)
        if k
    ]
    if not filtered:
        return [], []

    xyxy_f, crops_f, corners_f = zip(*filtered)

    corners_ordered = [
        ordered_corners([c[0] for c in cs], [c[1] for c in cs]) for cs in corners_f
    ]

    markers = []
    for crop, cs in zip(crops_f, corners_ordered):
        marker = marker_from_corners(crop, cs, _MARKER_CROP_SIZE)
        markers.append(_norm(cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)))

    decoder_out = np.round(models.decode_markers(np.array(markers)).numpy())

    result_corners: list[list[list[float]]] = []
    result_ids: list[int] = []

    for raw_out, det, flat_cs in zip(decoder_out, xyxy_f, corners_ordered):
        id_, dist = find_id(raw_out)
        if dist > threshold:
            continue
        w = det[2] - det[0]
        h = det[3] - det[1]
        # flat_cs = [x1,y1,x2,y2,x3,y3,x4,y4] normalized [0,1] within crop
        pixel_corners = [
            [det[0] + flat_cs[i] * w, det[1] + flat_cs[i + 1] * h]
            for i in range(0, 8, 2)
        ]
        result_corners.append([pixel_corners])  # type: ignore[arg-type]  # OpenCV format: [[[x,y],...]]
        result_ids.append(id_)

    return result_corners, result_ids


def run_deeparuco(
    detections: dict,
    frames_dir: Path,
    models: DeepArucoModels,
) -> list[dict]:
    results: list[dict] = []
    entries = detections["detections"]

    for i, entry in enumerate(entries):
        pipeline_io.log_progress(i, len(entries))

        frame_path = frames_dir / entry["filename"]
        if not frame_path.exists():
            logger.warning("Frame not found, skipping: %s", frame_path)
            continue

        image = cv2.imread(str(frame_path))
        if image is None:
            logger.warning("Cannot load image: %s", frame_path)
            continue

        corners, ids = run_deeparuco_on_image(image, models)

        results.append(
            {
                "filename": entry["filename"],
                "frame_index": entry["frame_index"],
                "markers_detected": len(ids),
                "marker_ids": ids,
                "corners": corners,
            }
        )

    logger.info(
        "DeepArUco inference complete. %d / %d frames processed.",
        len(results),
        len(entries),
    )
    return results


def _mean_corner_distance(
    cv_corners: list,
    da_corners: list,
    cv_ids: list[int],
    da_ids: list[int],
) -> float:
    da_corner_by_id = dict(zip(da_ids, da_corners))
    distances: list[float] = []
    seen: set[int] = set()
    for id_, ca in zip(cv_ids, cv_corners):
        if id_ in seen or id_ not in da_corner_by_id:
            continue
        seen.add(id_)
        pts_a = np.array(ca[0], dtype=float)  # shape (4, 2)
        pts_b = np.array(da_corner_by_id[id_][0], dtype=float)
        distances.append(float(np.mean(np.linalg.norm(pts_a - pts_b, axis=1))))
    return float(np.mean(distances)) if distances else 0.0


def compare_frame(cv_entry: dict, da_entry: dict) -> dict:
    cv_ids: list[int] = cv_entry["marker_ids"]
    da_ids: list[int] = da_entry["marker_ids"]

    cv_id_set = set(cv_ids)
    da_id_set = set(da_ids)
    matched = sorted(cv_id_set & da_id_set)
    unmatched_cv = sorted(cv_id_set - da_id_set)
    unmatched_da = sorted(da_id_set - cv_id_set)

    id_agreement = (
        len(cv_ids) > 0
        and len(da_ids) > 0
        and len(unmatched_cv) == 0
        and len(unmatched_da) == 0
    )

    return {
        "matched_markers": len(matched),
        "unmatched_opencv": len(unmatched_cv),
        "unmatched_deeparuco": len(unmatched_da),
        "id_agreement": id_agreement,
        "mean_corner_distance_px": _mean_corner_distance(
            cv_entry["corners"], da_entry["corners"], cv_ids, da_ids
        ),
    }


def compare_frames(cv_results: list[dict], da_results: list[dict]) -> list[dict]:
    da_by_filename = {r["filename"]: r for r in da_results}
    output: list[dict] = []

    for cv_entry in cv_results:
        da_entry = da_by_filename.get(
            cv_entry["filename"],
            {
                "filename": cv_entry["filename"],
                "frame_index": cv_entry["frame_index"],
                "markers_detected": 0,
                "marker_ids": [],
                "corners": [],
            },
        )
        output.append(
            {
                "filename": cv_entry["filename"],
                "frame_index": cv_entry["frame_index"],
                "opencv": {
                    "markers_detected": cv_entry["markers_detected"],
                    "marker_ids": cv_entry["marker_ids"],
                    "corners": cv_entry["corners"],
                },
                "deeparuco": {
                    "markers_detected": da_entry["markers_detected"],
                    "marker_ids": da_entry["marker_ids"],
                    "corners": da_entry["corners"],
                },
                "comparison": compare_frame(cv_entry, da_entry),
            }
        )

    return output


def compute_metrics(compared_frames: list[dict]) -> dict:
    total = len(compared_frames)
    if total == 0:
        return {
            "opencv_detection_rate": 0.0,
            "deeparuco_detection_rate": 0.0,
            "id_agreement_rate": 0.0,
            "mean_corner_distance_px": 0.0,
        }

    cv_detected = sum(1 for f in compared_frames if f["opencv"]["markers_detected"] > 0)
    da_detected = sum(
        1 for f in compared_frames if f["deeparuco"]["markers_detected"] > 0
    )

    both_detected = [
        f
        for f in compared_frames
        if f["opencv"]["markers_detected"] > 0
        and f["deeparuco"]["markers_detected"] > 0
    ]
    agreed = sum(1 for f in both_detected if f["comparison"]["id_agreement"])
    id_agreement_rate = agreed / len(both_detected) if both_detected else 0.0

    matched_frames = [
        f for f in compared_frames if f["comparison"]["matched_markers"] > 0
    ]
    corner_dists = [f["comparison"]["mean_corner_distance_px"] for f in matched_frames]
    mean_corner_dist = float(np.mean(corner_dists)) if corner_dists else 0.0

    return {
        "opencv_detection_rate": cv_detected / total,
        "deeparuco_detection_rate": da_detected / total,
        "id_agreement_rate": id_agreement_rate,
        "mean_corner_distance_px": mean_corner_dist,
    }


def save_comparison(
    compared_frames: list[dict],
    summary: dict,
    output_dir: Path,
    weights_path: Path,
    dictionary: str,
) -> None:
    # `dictionary` is the OpenCV-side dictionary actually used to produce
    # detections.json. DeepArUco's own decoder always assumes DICT_4X4_250,
    # since that's what the vendored model was trained on — that assumption
    # is fixed and not reflected in this field.
    output = ComparisonFile(
        dictionary=dictionary,
        model="deeparuco",
        weights_path=str(weights_path),
        total_frames=len(compared_frames),
        summary=summary,
        frames=compared_frames,
    )

    out_path = output_dir / "comparison.json"
    pipeline_io.save_json(output.to_dict(), out_path)

    logger.info(
        "Comparison saved to '%s'. opencv=%.1f%% da=%.1f%%"
        " agreement=%.1f%% corner_dist=%.2fpx",
        out_path,
        summary.get("opencv_detection_rate", 0.0) * 100,
        summary.get("deeparuco_detection_rate", 0.0) * 100,
        summary.get("id_agreement_rate", 0.0) * 100,
        summary.get("mean_corner_distance_px", 0.0),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare OpenCV ArUco detection against DeepArUco on the same frames."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--detections",
        type=Path,
        required=True,
        help="Path to detections.json produced by aruco_detection.py.",
    )
    parser.add_argument(
        "--model-weights",
        type=Path,
        default=None,
        help=(
            "Directory containing DeepArUco model files. "
            "Auto-downloads to ~/.cache/deeparuco/ if omitted."
        ),
    )
    return parser.parse_args()


def main() -> None:
    pipeline_io.configure_logging()
    args = parse_args()

    detections_file = DetectionsFile.from_dict(
        pipeline_io.load_json(args.detections, "detections"),
        source=str(args.detections),
    )
    detections = detections_file.to_dict()
    frames_dir = pipeline_io.session_dir(args.detections)

    da = load_config().deeparuco
    default_dir = Path(da.weights_dir).expanduser()
    models = load_deeparuco_models(
        args.model_weights,
        base_url=da.base_url,
        filenames=da.weights,
        default_dir=default_dir,
    )
    da_results = run_deeparuco(detections, frames_dir, models)

    compared = compare_frames(detections["detections"], da_results)
    summary = compute_metrics(compared)
    save_comparison(
        compared,
        summary,
        frames_dir,
        weights_path=args.model_weights or default_dir,
        dictionary=detections_file.dictionary,
    )


if __name__ == "__main__":
    main()
