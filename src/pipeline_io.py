"""Shared I/O, logging, and path-resolution helpers used by every pipeline script."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)

ARUCO_DICTIONARIES: dict[str, int] = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def load_json(path: Path, label: str) -> dict:
    if not path.exists():
        logger.error("%s file not found: %s", label, path)
        raise FileNotFoundError(f"{label} file not found: {path}")

    with path.open() as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in %s file: %s", label, path)
            raise ValueError(f"Invalid JSON in {label} file: {path}") from e


def save_json(data: dict, path: Path) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def log_progress(i: int, total: int, *, every: int = 50) -> None:
    if i % every == 0:
        logger.info("Progress: %d / %d", i, total)


def session_dir(primary_arg: Path) -> Path:
    """Sibling artifacts of a pipeline stage live in the same directory as its
    primary CLI input argument. This is the one place that convention is named."""
    return primary_arg.parent
