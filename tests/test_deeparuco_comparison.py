import json
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pytest

from deeparuco_comparison import (
    DeepArucoModels,
    load_deeparuco_models,
    load_detections,
    run_deeparuco,
    run_deeparuco_on_image,
)


def test_load_detections_raises_when_file_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_detections(tmp_path / "nonexistent.json")


def test_load_detections_raises_on_invalid_json(tmp_path: Path) -> None:
    bad = tmp_path / "detections.json"
    bad.write_text("not json")
    with pytest.raises(ValueError):
        load_detections(bad)


def test_load_detections_returns_dict(tmp_path: Path) -> None:
    data = {"dictionary": "DICT_4X4_250", "total_frames": 0, "detections": []}
    p = tmp_path / "detections.json"
    p.write_text(json.dumps(data))
    assert load_detections(p) == data


_MODEL_FILES = ["det_luma_bc_s.pt", "reg_hmap_8.h5", "dec_new.h5"]


def _make_fake_weights_dir(tmp_path: Path) -> Path:
    for name in _MODEL_FILES:
        (tmp_path / name).touch()
    return tmp_path


def test_load_deeparuco_models_returns_dataclass(tmp_path: Path) -> None:
    weights_dir = _make_fake_weights_dir(tmp_path)
    with (
        mock.patch("deeparuco_comparison.YOLO") as mock_yolo,
        mock.patch("deeparuco_comparison.load_model") as mock_keras,
    ):
        mock_yolo.return_value = mock.MagicMock()
        mock_keras.return_value = mock.MagicMock()
        result = load_deeparuco_models(weights_dir)

    assert isinstance(result, DeepArucoModels)
    assert result.detector is not None
    assert result.regressor is not None
    assert result.decoder is not None


def test_load_deeparuco_models_raises_when_model_file_missing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Model file not found"):
        load_deeparuco_models(tmp_path)


def test_run_deeparuco_on_image_no_detections_returns_empty() -> None:
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    boxes_mock = mock.MagicMock()
    boxes_mock.__len__ = mock.Mock(return_value=0)
    detector_result = mock.MagicMock()
    detector_result.cpu.return_value.boxes = boxes_mock

    models = mock.MagicMock(spec=DeepArucoModels(None, None, None, None, None))
    models.detector.return_value = [detector_result]

    corners, ids = run_deeparuco_on_image(image, models)

    assert corners == []
    assert ids == []


def test_run_deeparuco_returns_entry_per_frame(tmp_path: Path) -> None:
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "frame_0000.jpg"), blank)
    cv2.imwrite(str(tmp_path / "frame_0001.jpg"), blank)

    detections_data = {
        "detections": [
            {"filename": "frame_0000.jpg", "frame_index": 0},
            {"filename": "frame_0001.jpg", "frame_index": 1},
        ]
    }
    models = mock.MagicMock(spec=DeepArucoModels(None, None, None, None, None))

    with mock.patch(
        "deeparuco_comparison.run_deeparuco_on_image", return_value=([], [])
    ):
        result = run_deeparuco(detections_data, tmp_path, models)

    assert len(result) == 2
    assert result[0]["filename"] == "frame_0000.jpg"
    assert result[0]["frame_index"] == 0
    assert result[0]["markers_detected"] == 0
    assert result[0]["marker_ids"] == []
    assert result[0]["corners"] == []


def test_run_deeparuco_skips_missing_frame(tmp_path: Path) -> None:
    detections_data = {
        "detections": [
            {"filename": "missing.jpg", "frame_index": 0},
        ]
    }
    models = mock.MagicMock(spec=DeepArucoModels(None, None, None, None, None))
    result = run_deeparuco(detections_data, tmp_path, models)
    assert result == []
