import json
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO

from deeparuco_vendor.losses import weighted_loss

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
