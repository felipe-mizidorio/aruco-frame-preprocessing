import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


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
