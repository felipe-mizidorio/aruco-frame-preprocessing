import json
from pathlib import Path

import pytest

from frame_filtering import filter_frames, load_detections, save_filtered_detections


def make_detections(tmp_path: Path, detections: list[dict]) -> tuple[Path, dict]:
    data = {
        "dictionary": "DICT_4X4_250",
        "total_frames": len(detections),
        "frames_with_detections": sum(
            1 for d in detections if d["markers_detected"] > 0
        ),
        "detections": detections,
    }
    path = tmp_path / "detections.json"
    path.write_text(json.dumps(data))
    return path, data


# --- load_detections ---


def test_load_detections_raises_when_file_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_detections(tmp_path / "nonexistent.json")


def test_load_detections_raises_on_invalid_json(tmp_path: Path) -> None:
    bad = tmp_path / "detections.json"
    bad.write_text("not json")
    with pytest.raises(ValueError):
        load_detections(bad)


def test_load_detections_returns_dict(tmp_path: Path) -> None:
    path, data = make_detections(tmp_path, [])
    result = load_detections(path)
    assert result == data


# --- filter_frames ---


def test_filter_frames_keeps_frames_at_or_above_threshold(tmp_path: Path) -> None:
    frame_a = tmp_path / "frame_0000.jpg"
    frame_b = tmp_path / "frame_0001.jpg"
    frame_a.write_bytes(b"img")
    frame_b.write_bytes(b"img")

    detections_data = {
        "detections": [
            {
                "filename": "frame_0000.jpg",
                "frame_index": 0,
                "markers_detected": 2,
                "marker_ids": [1, 2],
                "corners": [],
            },
            {
                "filename": "frame_0001.jpg",
                "frame_index": 1,
                "markers_detected": 0,
                "marker_ids": [],
                "corners": [],
            },
        ]
    }

    result = filter_frames(detections_data, tmp_path, min_markers=1)

    assert len(result) == 1
    assert result[0]["filename"] == "frame_0000.jpg"


def test_filter_frames_copies_passing_frames_to_filtered_subdir(tmp_path: Path) -> None:
    (tmp_path / "frame_0000.jpg").write_bytes(b"img")

    detections_data = {
        "detections": [
            {
                "filename": "frame_0000.jpg",
                "frame_index": 0,
                "markers_detected": 1,
                "marker_ids": [5],
                "corners": [],
            },
        ]
    }

    filter_frames(detections_data, tmp_path, min_markers=1)

    assert (tmp_path / "filtered" / "frame_0000.jpg").exists()


def test_filter_frames_does_not_copy_failing_frames(tmp_path: Path) -> None:
    (tmp_path / "frame_0000.jpg").write_bytes(b"img")

    detections_data = {
        "detections": [
            {
                "filename": "frame_0000.jpg",
                "frame_index": 0,
                "markers_detected": 0,
                "marker_ids": [],
                "corners": [],
            },
        ]
    }

    filter_frames(detections_data, tmp_path, min_markers=1)

    assert not (tmp_path / "filtered" / "frame_0000.jpg").exists()


def test_filter_frames_skips_missing_source_file(tmp_path: Path) -> None:
    detections_data = {
        "detections": [
            {
                "filename": "missing.jpg",
                "frame_index": 0,
                "markers_detected": 3,
                "marker_ids": [1, 2, 3],
                "corners": [],
            },
        ]
    }

    # Should not raise, just skip
    result = filter_frames(detections_data, tmp_path, min_markers=1)
    assert result == []


# --- save_filtered_detections ---


def test_save_filtered_detections_writes_file(tmp_path: Path) -> None:
    _, original = make_detections(tmp_path, [])
    filtered: list[dict] = []

    save_filtered_detections(filtered, original, tmp_path, min_markers=1)

    assert (tmp_path / "filtered_detections.json").exists()


def test_save_filtered_detections_schema(tmp_path: Path) -> None:
    passing = {
        "filename": "frame_0000.jpg",
        "frame_index": 0,
        "markers_detected": 2,
        "marker_ids": [1, 2],
        "corners": [],
    }
    failing = {
        "filename": "frame_0001.jpg",
        "frame_index": 1,
        "markers_detected": 0,
        "marker_ids": [],
        "corners": [],
    }
    _, original = make_detections(tmp_path, [passing, failing])

    save_filtered_detections([passing], original, tmp_path, min_markers=1)

    output = json.loads((tmp_path / "filtered_detections.json").read_text())
    assert output["dictionary"] == "DICT_4X4_250"
    assert output["min_markers"] == 1
    assert output["total_frames"] == 1
    assert output["frames_with_detections"] == 1
    assert output["frames_filtered_out"] == 1
    assert output["detections"] == [passing]


# --- valid_ids filtering ---


def test_filter_frames_valid_ids_none_leaves_data_unchanged(tmp_path: Path) -> None:
    (tmp_path / "frame_0000.jpg").write_bytes(b"img")

    detections_data = {
        "detections": [
            {
                "filename": "frame_0000.jpg",
                "frame_index": 0,
                "markers_detected": 3,
                "marker_ids": [0, 5, 99],
                "corners": [[[0, 0], [1, 0], [1, 1], [0, 1]]] * 3,
            },
        ]
    }

    result = filter_frames(detections_data, tmp_path, min_markers=1, valid_ids=None)

    assert result[0]["marker_ids"] == [0, 5, 99]
    assert result[0]["markers_detected"] == 3
    assert len(result[0]["corners"]) == 3


def test_filter_frames_valid_ids_strips_spurious_and_realigns_corners(
    tmp_path: Path,
) -> None:
    (tmp_path / "frame_0000.jpg").write_bytes(b"img")

    corner_valid_0 = [[[0, 0], [1, 0], [1, 1], [0, 1]]]
    corner_valid_5 = [[[2, 0], [3, 0], [3, 1], [2, 1]]]
    corner_spurious = [[[9, 0], [9, 1], [9, 2], [9, 3]]]

    detections_data = {
        "detections": [
            {
                "filename": "frame_0000.jpg",
                "frame_index": 0,
                "markers_detected": 3,
                "marker_ids": [0, 99, 5],
                "corners": corner_valid_0 + corner_spurious + corner_valid_5,
            },
        ]
    }

    result = filter_frames(
        detections_data, tmp_path, min_markers=1, valid_ids={0, 1, 2, 3, 4, 5}
    )

    assert result[0]["marker_ids"] == [0, 5]
    assert result[0]["markers_detected"] == 2
    assert result[0]["corners"] == corner_valid_0 + corner_valid_5


def test_filter_frames_valid_ids_all_valid_untouched(tmp_path: Path) -> None:
    (tmp_path / "frame_0000.jpg").write_bytes(b"img")

    corners = [[[0, 0], [1, 0], [1, 1], [0, 1]]] * 2

    detections_data = {
        "detections": [
            {
                "filename": "frame_0000.jpg",
                "frame_index": 0,
                "markers_detected": 2,
                "marker_ids": [3, 7],
                "corners": corners,
            },
        ]
    }

    result = filter_frames(
        detections_data, tmp_path, min_markers=1, valid_ids={3, 7, 15}
    )

    assert result[0]["marker_ids"] == [3, 7]
    assert result[0]["markers_detected"] == 2
    assert result[0]["corners"] == corners


def test_filter_frames_valid_ids_only_spurious_zeros_count_frame_kept(
    tmp_path: Path,
) -> None:
    (tmp_path / "frame_0000.jpg").write_bytes(b"img")

    detections_data = {
        "detections": [
            {
                "filename": "frame_0000.jpg",
                "frame_index": 0,
                "markers_detected": 2,
                "marker_ids": [86, 128],
                "corners": [[[0, 0], [1, 0], [1, 1], [0, 1]]] * 2,
            },
        ]
    }

    result = filter_frames(
        detections_data, tmp_path, min_markers=0, valid_ids={0, 1, 2}
    )

    assert result[0]["marker_ids"] == []
    assert result[0]["markers_detected"] == 0
    assert result[0]["corners"] == []
