import json
from pathlib import Path
from unittest import mock

import pytest

from deeparuco_comparison import DeepArucoModels, load_deeparuco_models, load_detections


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
