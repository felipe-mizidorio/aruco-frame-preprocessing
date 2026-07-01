import json
import logging
from pathlib import Path

import pytest

from pipeline_io import (
    ARUCO_DICTIONARIES,
    configure_logging,
    load_json,
    log_progress,
    save_json,
    session_dir,
)


def test_load_json_raises_when_file_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_json(tmp_path / "nonexistent.json", "metadata")


def test_load_json_raises_on_invalid_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    with pytest.raises(ValueError):
        load_json(bad, "metadata")


def test_load_json_returns_dict(tmp_path: Path) -> None:
    data = {"a": 1}
    path = tmp_path / "data.json"
    path.write_text(json.dumps(data))
    assert load_json(path, "metadata") == data


def test_save_json_writes_file(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    save_json({"a": 1}, path)
    assert json.loads(path.read_text()) == {"a": 1}


def test_log_progress_logs_at_interval(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="pipeline_io"):
        log_progress(0, 100)
        log_progress(1, 100)
        log_progress(50, 100)
    assert len(caplog.records) == 2  # only i=0 and i=50 with default every=50


def test_log_progress_respects_custom_every(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="pipeline_io"):
        log_progress(0, 10, every=5)
        log_progress(3, 10, every=5)
        log_progress(5, 10, every=5)
    assert len(caplog.records) == 2


def test_session_dir_returns_parent(tmp_path: Path) -> None:
    primary_arg = tmp_path / "session" / "detections.json"
    assert session_dir(primary_arg) == tmp_path / "session"


def test_aruco_dictionaries_contains_default() -> None:
    assert "DICT_4X4_250" in ARUCO_DICTIONARIES


def test_configure_logging_does_not_raise() -> None:
    configure_logging()
