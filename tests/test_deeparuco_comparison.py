import json
from pathlib import Path

import pytest

from deeparuco_comparison import load_detections


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
