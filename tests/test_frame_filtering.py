import json
from pathlib import Path

from frame_filtering import filter_frames, save_manifest, strip_invalid_ids
from schemas import DetectionEntry, DetectionsFile


def make_detections(entries: list[DetectionEntry]) -> DetectionsFile:
    return DetectionsFile(
        dictionary="DICT_4X4_250",
        total_frames=len(entries),
        frames_with_detections=sum(1 for e in entries if e.markers_detected > 0),
        detections=entries,
    )


# --- filter_frames ---


def test_filter_frames_keeps_frames_at_or_above_threshold(tmp_path: Path) -> None:
    (tmp_path / "frame_0000.jpg").write_bytes(b"img")
    (tmp_path / "frame_0001.jpg").write_bytes(b"img")

    detections = make_detections(
        [
            DetectionEntry("frame_0000.jpg", 0, 2, [1, 2], []),
            DetectionEntry("frame_0001.jpg", 1, 0, [], []),
        ]
    )

    result = filter_frames(detections, tmp_path, min_markers=1)

    assert len(result) == 1
    assert result[0].filename == "frame_0000.jpg"


def test_filter_frames_copies_passing_frames_to_filtered_subdir(tmp_path: Path) -> None:
    (tmp_path / "frame_0000.jpg").write_bytes(b"img")

    detections = make_detections([DetectionEntry("frame_0000.jpg", 0, 1, [5], [])])

    filter_frames(detections, tmp_path, min_markers=1)

    assert (tmp_path / "filtered" / "frame_0000.jpg").exists()


def test_filter_frames_does_not_copy_failing_frames(tmp_path: Path) -> None:
    (tmp_path / "frame_0000.jpg").write_bytes(b"img")

    detections = make_detections([DetectionEntry("frame_0000.jpg", 0, 0, [], [])])

    filter_frames(detections, tmp_path, min_markers=1)

    assert not (tmp_path / "filtered" / "frame_0000.jpg").exists()


def test_filter_frames_skips_missing_source_file(tmp_path: Path) -> None:
    detections = make_detections([DetectionEntry("missing.jpg", 0, 3, [1, 2, 3], [])])

    result = filter_frames(detections, tmp_path, min_markers=1)
    assert result == []


# --- save_manifest ---


def test_save_manifest_writes_file(tmp_path: Path) -> None:
    detections = make_detections([])
    save_manifest([], detections, tmp_path, min_markers=1)
    assert (tmp_path / "manifest.json").exists()


def test_save_manifest_schema(tmp_path: Path) -> None:
    passing = DetectionEntry(
        filename="frame_0000.jpg",
        frame_index=0,
        markers_detected=2,
        marker_ids=[1, 2],
        corners=[
            [[0, 0], [1, 0], [1, 1], [0, 1]],
            [[2, 0], [3, 0], [3, 1], [2, 1]],
        ],
    )
    failing = DetectionEntry(
        filename="frame_0001.jpg",
        frame_index=1,
        markers_detected=0,
        marker_ids=[],
        corners=[],
    )
    detections = make_detections([passing, failing])

    save_manifest([passing], detections, tmp_path, min_markers=1)

    output = json.loads((tmp_path / "manifest.json").read_text())
    assert output["dictionary"] == "DICT_4X4_250"
    assert output["min_markers"] == 1
    assert output["total_frames_input"] == 2
    assert output["frames_filtered_out"] == 1
    assert output["frames"] == ["frame_0000.jpg"]
    assert output["marker_detections"] == {
        "frame_0000.jpg": [
            {"id": 1, "corners": [[0, 0], [1, 0], [1, 1], [0, 1]]},
            {"id": 2, "corners": [[2, 0], [3, 0], [3, 1], [2, 1]]},
        ]
    }


# --- valid_ids filtering ---


def test_filter_frames_valid_ids_none_leaves_data_unchanged(tmp_path: Path) -> None:
    (tmp_path / "frame_0000.jpg").write_bytes(b"img")

    entry = DetectionEntry(
        "frame_0000.jpg", 0, 3, [0, 5, 99], [[[0, 0], [1, 0], [1, 1], [0, 1]]] * 3
    )
    detections = make_detections([entry])

    result = filter_frames(detections, tmp_path, min_markers=1, valid_ids=None)

    assert result[0].marker_ids == [0, 5, 99]
    assert result[0].markers_detected == 3
    assert len(result[0].corners) == 3


def test_filter_frames_valid_ids_strips_spurious_and_realigns_corners(
    tmp_path: Path,
) -> None:
    (tmp_path / "frame_0000.jpg").write_bytes(b"img")

    corner_valid_0 = [[[0, 0], [1, 0], [1, 1], [0, 1]]]
    corner_valid_5 = [[[2, 0], [3, 0], [3, 1], [2, 1]]]
    corner_spurious = [[[9, 0], [9, 1], [9, 2], [9, 3]]]

    entry = DetectionEntry(
        "frame_0000.jpg",
        0,
        3,
        [0, 99, 5],
        corner_valid_0 + corner_spurious + corner_valid_5,
    )
    detections = make_detections([entry])

    result = filter_frames(
        detections, tmp_path, min_markers=1, valid_ids={0, 1, 2, 3, 4, 5}
    )

    assert result[0].marker_ids == [0, 5]
    assert result[0].markers_detected == 2
    assert result[0].corners == corner_valid_0 + corner_valid_5


def test_filter_frames_valid_ids_all_valid_untouched(tmp_path: Path) -> None:
    (tmp_path / "frame_0000.jpg").write_bytes(b"img")

    corners = [[[0, 0], [1, 0], [1, 1], [0, 1]]] * 2
    entry = DetectionEntry("frame_0000.jpg", 0, 2, [3, 7], corners)
    detections = make_detections([entry])

    result = filter_frames(detections, tmp_path, min_markers=1, valid_ids={3, 7, 15})

    assert result[0].marker_ids == [3, 7]
    assert result[0].markers_detected == 2
    assert result[0].corners == corners


def test_filter_frames_valid_ids_only_spurious_zeros_count_frame_kept(
    tmp_path: Path,
) -> None:
    (tmp_path / "frame_0000.jpg").write_bytes(b"img")

    entry = DetectionEntry(
        "frame_0000.jpg", 0, 2, [86, 128], [[[0, 0], [1, 0], [1, 1], [0, 1]]] * 2
    )
    detections = make_detections([entry])

    result = filter_frames(detections, tmp_path, min_markers=0, valid_ids={0, 1, 2})

    assert result[0].marker_ids == []
    assert result[0].markers_detected == 0
    assert result[0].corners == []


# --- strip_invalid_ids ---


def test_strip_invalid_ids_returns_new_entry_with_only_valid_ids() -> None:
    entry = DetectionEntry(
        "a.jpg", 0, 2, [1, 99], [[[0, 0], [1, 0], [1, 1], [0, 1]]] * 2
    )
    result = strip_invalid_ids(entry, valid_ids={1})
    assert result.marker_ids == [1]
    assert result.markers_detected == 1
    assert entry.marker_ids == [1, 99]  # original entry is untouched
