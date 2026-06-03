import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from aruco_detection import detect_markers, load_metadata, save_detections


def make_metadata(tmp_path: Path, frames: list[dict]) -> tuple[Path, dict]:
    data = {"frames_extracted": len(frames), "frames": frames}
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(data))
    return path, data


def write_blank_jpeg(path: Path) -> None:
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.imwrite(str(path), blank)


def write_marker_jpeg(
    path: Path, marker_id: int = 0, size: int = 200, border: int = 40
) -> None:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size)
    padded = np.full((size + 2 * border, size + 2 * border), 255, dtype=np.uint8)
    padded[border : border + size, border : border + size] = marker
    cv2.imwrite(str(path), padded)


# --- load_metadata ---


def test_load_metadata_raises_when_file_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_metadata(tmp_path / "nonexistent.json")


def test_load_metadata_raises_on_invalid_json(tmp_path: Path) -> None:
    bad = tmp_path / "metadata.json"
    bad.write_text("not json")
    with pytest.raises(ValueError):
        load_metadata(bad)


def test_load_metadata_returns_dict(tmp_path: Path) -> None:
    path, data = make_metadata(tmp_path, [])
    assert load_metadata(path) == data


# --- detect_markers ---


def test_detect_markers_blank_image_returns_zero_markers(tmp_path: Path) -> None:
    write_blank_jpeg(tmp_path / "frame_0000.jpg")
    _, metadata = make_metadata(
        tmp_path, [{"filename": "frame_0000.jpg", "frame_index": 0}]
    )

    result = detect_markers(metadata, tmp_path)

    assert len(result) == 1
    assert result[0]["markers_detected"] == 0
    assert result[0]["marker_ids"] == []


def test_detect_markers_finds_aruco_marker(tmp_path: Path) -> None:
    write_marker_jpeg(tmp_path / "frame_0000.jpg", marker_id=0)
    _, metadata = make_metadata(
        tmp_path, [{"filename": "frame_0000.jpg", "frame_index": 0}]
    )

    result = detect_markers(metadata, tmp_path)

    assert len(result) == 1
    assert result[0]["markers_detected"] == 1
    assert result[0]["marker_ids"] == [0]


def test_detect_markers_skips_missing_frame(tmp_path: Path) -> None:
    _, metadata = make_metadata(
        tmp_path, [{"filename": "missing.jpg", "frame_index": 0}]
    )

    result = detect_markers(metadata, tmp_path)

    assert len(result) == 1
    assert result[0]["markers_detected"] == 0
    assert result[0]["marker_ids"] == []
    assert result[0]["corners"] == []
    assert result[0]["filename"] == "missing.jpg"
    assert result[0]["frame_index"] == 0


def test_detect_markers_entry_schema(tmp_path: Path) -> None:
    write_blank_jpeg(tmp_path / "frame_0000.jpg")
    _, metadata = make_metadata(
        tmp_path, [{"filename": "frame_0000.jpg", "frame_index": 0}]
    )

    result = detect_markers(metadata, tmp_path)

    entry = result[0]
    assert "filename" in entry
    assert "frame_index" in entry
    assert "markers_detected" in entry
    assert "marker_ids" in entry
    assert "corners" in entry


# --- save_detections ---


def test_save_detections_writes_file(tmp_path: Path) -> None:
    _, metadata = make_metadata(tmp_path, [])
    save_detections([], metadata, tmp_path)
    assert (tmp_path / "detections.json").exists()


def test_save_detections_schema(tmp_path: Path) -> None:
    _, metadata = make_metadata(tmp_path, [])
    save_detections([], metadata, tmp_path)

    output = json.loads((tmp_path / "detections.json").read_text())
    assert "dictionary" in output
    assert "total_frames" in output
    assert "frames_with_detections" in output
    assert "detections" in output


def test_save_detections_counts_correctly(tmp_path: Path) -> None:
    detections = [
        {
            "filename": "a.jpg",
            "frame_index": 0,
            "markers_detected": 2,
            "marker_ids": [0, 1],
            "corners": [],
        },
        {
            "filename": "b.jpg",
            "frame_index": 1,
            "markers_detected": 0,
            "marker_ids": [],
            "corners": [],
        },
        {
            "filename": "c.jpg",
            "frame_index": 2,
            "markers_detected": 1,
            "marker_ids": [3],
            "corners": [],
        },
    ]
    _, metadata = make_metadata(tmp_path, [{} for _ in detections])
    save_detections(detections, metadata, tmp_path)

    output = json.loads((tmp_path / "detections.json").read_text())
    assert output["frames_with_detections"] == 2
    assert output["total_frames"] == 3
