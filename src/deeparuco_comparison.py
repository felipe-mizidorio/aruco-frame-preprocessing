import json
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO

from deeparuco_vendor.aruco import find_id
from deeparuco_vendor.heatmaps import pos_from_heatmap
from deeparuco_vendor.losses import weighted_loss
from deeparuco_vendor.utils import marker_from_corners, ordered_corners

logger = logging.getLogger(__name__)


DEFAULT_WEIGHTS_DIR = Path.home() / ".cache" / "deeparuco"

_MODEL_FILENAMES = {
    "detector": "det_luma_bc_s.pt",
    "regressor": "reg_hmap_8.h5",
    "decoder": "dec_new.h5",
}
_BASE_URL = "https://raw.githubusercontent.com/AVAuco/deeparuco/master/models"


@dataclass
class DeepArucoModels:
    detector: Any
    regressor: Any
    decoder: Any
    refine_corners: Any
    decode_markers: Any


def _download_weights(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in _MODEL_FILENAMES.values():
        target = target_dir / filename
        if not target.exists():
            logger.info("Downloading %s...", filename)
            try:
                urllib.request.urlretrieve(f"{_BASE_URL}/{filename}", target)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download {filename}. Download manually from "
                    "https://github.com/AVAuco/deeparuco/tree/master/models"
                ) from e


def load_deeparuco_models(weights_dir: Path | None = None) -> DeepArucoModels:
    if weights_dir is None:
        _download_weights(DEFAULT_WEIGHTS_DIR)
        weights_dir = DEFAULT_WEIGHTS_DIR

    for filename in _MODEL_FILENAMES.values():
        path = weights_dir / filename
        if not path.exists():
            raise RuntimeError(
                f"Model file not found: {path}. Download from "
                "https://github.com/AVAuco/deeparuco/tree/master/models"
            )

    detector = YOLO(str(weights_dir / _MODEL_FILENAMES["detector"]))
    regressor = load_model(
        str(weights_dir / _MODEL_FILENAMES["regressor"]),
        custom_objects={"weighted_loss": weighted_loss},
    )
    decoder = load_model(str(weights_dir / _MODEL_FILENAMES["decoder"]))

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


def load_detections(detections_path: Path) -> dict:
    if not detections_path.exists():
        logger.error("Detections file not found: %s", detections_path)
        raise FileNotFoundError(f"Detections file not found: {detections_path}")

    with detections_path.open() as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in detections file: %s", detections_path)
            raise ValueError(
                f"Invalid JSON in detections file: {detections_path}"
            ) from e


_DECODE_THRESHOLD = 9  # max hamming distance to accept a detection


def _norm(x: np.ndarray) -> np.ndarray:
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-9)


def run_deeparuco_on_image(
    image: np.ndarray,
    models: DeepArucoModels,
    threshold: int = _DECODE_THRESHOLD,
) -> tuple[list[list[list[float]]], list[int]]:
    detections = (
        models.detector(image, verbose=False, iou=0.5, conf=0.03)[0].cpu().boxes
    )
    if not len(detections):
        return [], []

    xyxy = [
        [
            int(max(det[0] - (0.2 * (det[2] - det[0]) + 0.5), 0)),
            int(max(det[1] - (0.2 * (det[3] - det[1]) + 0.5), 0)),
            int(min(det[2] + (0.2 * (det[2] - det[0]) + 0.5), image.shape[1] - 1)),
            int(min(det[3] + (0.2 * (det[3] - det[1]) + 0.5), image.shape[0] - 1)),
        ]
        for det in [
            [int(val) for val in det.xyxy.cpu().numpy()[0]] for det in detections
        ]
    ]

    crops_ori = [
        cv2.resize(image[det[1] : det[3], det[0] : det[2]], (64, 64)) for det in xyxy
    ]
    crops = [_norm(crop) for crop in crops_ori]

    corners_raw = models.refine_corners(np.array(crops)).numpy()

    area = 75
    kp_params = cv2.SimpleBlobDetector_Params()
    kp_params.filterByArea = True
    kp_params.minArea = area * 0.8
    kp_params.maxArea = area * 1.2
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
        marker = marker_from_corners(crop, cs, 32)
        markers.append(_norm(cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)))

    decoder_out = np.round(models.decode_markers(np.array(markers)).numpy())

    result_corners: list[list[list[float]]] = []
    result_ids: list[int] = []

    for raw_out, det, flat_cs in zip(decoder_out, xyxy_f, corners_ordered):
        id_, dist = find_id(raw_out)
        if dist >= threshold:
            continue
        w = det[2] - det[0]
        h = det[3] - det[1]
        # flat_cs = [x1,y1,x2,y2,x3,y3,x4,y4] normalized [0,1] within crop
        pixel_corners = [
            [det[0] + flat_cs[i] * w, det[1] + flat_cs[i + 1] * h]
            for i in range(0, 8, 2)
        ]
        result_corners.append([pixel_corners])  # OpenCV format: [[[x,y],...]]
        result_ids.append(id_)

    return result_corners, result_ids
