import json
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pytest

from deeparuco_comparison import (
    DeepArucoModels,
    compare_frame,
    compare_frames,
    compute_metrics,
    load_deeparuco_models,
    load_detections,
    run_deeparuco,
    run_deeparuco_on_image,
    save_comparison,
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


def _entry(filename: str, frame_index: int, ids: list[int], corners: list) -> dict:
    return {
        "filename": filename,
        "frame_index": frame_index,
        "markers_detected": len(ids),
        "marker_ids": ids,
        "corners": corners,
    }


def _corner(x: float, y: float) -> list:
    # Single marker in OpenCV format: corners[i] = [[[x,y],[x,y],[x,y],[x,y]]] (shape 1x4x2)
    return [[[[x, y], [x + 10, y], [x + 10, y + 10], [x, y + 10]]]]


def test_compare_frame_both_agree() -> None:
    cv_e = _entry("a.jpg", 0, [5], _corner(10, 10))
    da_e = _entry("a.jpg", 0, [5], _corner(12, 12))
    result = compare_frame(cv_e, da_e)
    assert result["matched_markers"] == 1
    assert result["unmatched_opencv"] == 0
    assert result["unmatched_deeparuco"] == 0
    assert result["id_agreement"] is True
    assert result["mean_corner_distance_px"] > 0


def test_compare_frame_no_id_overlap() -> None:
    cv_e = _entry("a.jpg", 0, [3], _corner(0, 0))
    da_e = _entry("a.jpg", 0, [7], _corner(0, 0))
    result = compare_frame(cv_e, da_e)
    assert result["matched_markers"] == 0
    assert result["unmatched_opencv"] == 1
    assert result["unmatched_deeparuco"] == 1
    assert result["id_agreement"] is False


def test_compare_frame_opencv_only() -> None:
    cv_e = _entry("a.jpg", 0, [1], _corner(0, 0))
    da_e = _entry("a.jpg", 0, [], [])
    result = compare_frame(cv_e, da_e)
    assert result["matched_markers"] == 0
    assert result["unmatched_opencv"] == 1
    assert result["unmatched_deeparuco"] == 0


def test_compare_frame_corner_distance_zero_for_identical() -> None:
    corners = _corner(50, 50)
    cv_e = _entry("a.jpg", 0, [2], corners)
    da_e = _entry("a.jpg", 0, [2], corners)
    result = compare_frame(cv_e, da_e)
    assert result["mean_corner_distance_px"] == pytest.approx(0.0)


def test_compare_frames_returns_one_entry_per_frame() -> None:
    cv_results = [_entry("a.jpg", 0, [1], _corner(0, 0))]
    da_results = [_entry("a.jpg", 0, [1], _corner(2, 2))]
    result = compare_frames(cv_results, da_results)
    assert len(result) == 1
    assert result[0]["filename"] == "a.jpg"
    assert "opencv" in result[0]
    assert "deeparuco" in result[0]
    assert "comparison" in result[0]


def _cframe(cv_det: bool, da_det: bool, matched: int = 0, dist: float = 0.0) -> dict:
    return {
        "opencv": {"markers_detected": 1 if cv_det else 0},
        "deeparuco": {"markers_detected": 1 if da_det else 0},
        "comparison": {
            "matched_markers": matched,
            "mean_corner_distance_px": dist,
            "id_agreement": matched > 0,
        },
    }


def test_compute_metrics_all_detected() -> None:
    frames = [
        _cframe(True, True, matched=1, dist=2.0),
        _cframe(True, True, matched=1, dist=4.0),
    ]
    m = compute_metrics(frames)
    assert m["opencv_detection_rate"] == pytest.approx(1.0)
    assert m["deeparuco_detection_rate"] == pytest.approx(1.0)
    assert m["id_agreement_rate"] == pytest.approx(1.0)
    assert m["mean_corner_distance_px"] == pytest.approx(3.0)


def test_compute_metrics_none_detected() -> None:
    frames = [_cframe(False, False), _cframe(False, False)]
    m = compute_metrics(frames)
    assert m["opencv_detection_rate"] == pytest.approx(0.0)
    assert m["deeparuco_detection_rate"] == pytest.approx(0.0)


def test_compute_metrics_partial() -> None:
    frames = [_cframe(True, True, matched=1), _cframe(True, False)]
    m = compute_metrics(frames)
    assert m["opencv_detection_rate"] == pytest.approx(1.0)
    assert m["deeparuco_detection_rate"] == pytest.approx(0.5)


def test_compute_metrics_empty() -> None:
    m = compute_metrics([])
    assert m["opencv_detection_rate"] == pytest.approx(0.0)
    assert m["deeparuco_detection_rate"] == pytest.approx(0.0)
    assert m["id_agreement_rate"] == pytest.approx(0.0)
    assert m["mean_corner_distance_px"] == pytest.approx(0.0)


def test_save_comparison_writes_file(tmp_path: Path) -> None:
    save_comparison([], {}, tmp_path, weights_path=tmp_path / "w.pt")
    assert (tmp_path / "comparison.json").exists()


def test_save_comparison_schema(tmp_path: Path) -> None:
    save_comparison([], {}, tmp_path, weights_path=tmp_path / "w.pt")
    data = json.loads((tmp_path / "comparison.json").read_text())
    assert "dictionary" in data
    assert "model" in data
    assert "weights_path" in data
    assert "total_frames" in data
    assert "summary" in data
    assert "frames" in data


def test_save_comparison_logs_output_path(tmp_path: Path, caplog) -> None:
    import logging

    summary = {
        "opencv_detection_rate": 1.0,
        "deeparuco_detection_rate": 0.9,
        "id_agreement_rate": 0.8,
        "mean_corner_distance_px": 2.5,
    }
    with caplog.at_level(logging.INFO, logger="deeparuco_comparison"):
        save_comparison([], summary, tmp_path, weights_path=tmp_path / "w.pt")
    assert any("comparison.json" in r.message for r in caplog.records)
