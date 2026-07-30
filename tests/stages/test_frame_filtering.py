import json
from pathlib import Path

import cv2
import numpy as np

from aruco_pipeline.core.schemas import DetectionEntry, DetectionsFile, FilterManifest
from aruco_pipeline.stages.frame_filtering import (
    compute_blur_rejects,
    filter_frames,
    save_manifest,
    sharpness_score,
    strip_invalid_ids,
)


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


# --- sharpness / blur filtering ---


def write_sharp_frame(path: Path, rng: np.random.Generator) -> None:
    img = rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def write_blurred_frame(path: Path, rng: np.random.Generator) -> None:
    img = rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
    cv2.imwrite(str(path), cv2.GaussianBlur(img, (31, 31), 12.0))


def make_session(tmp_path: Path, n_sharp: int, n_blurred: int) -> DetectionsFile:
    rng = np.random.default_rng(42)
    entries = []
    for i in range(n_sharp + n_blurred):
        name = f"frame_{i:04d}.jpg"
        if i < n_sharp:
            write_sharp_frame(tmp_path / name, rng)
        else:
            write_blurred_frame(tmp_path / name, rng)
        entries.append(
            DetectionEntry(name, i, 1, [3], [[[0, 0], [1, 0], [1, 1], [0, 1]]])
        )
    return make_detections(entries)


def test_sharpness_score_higher_for_sharp_frame(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    write_sharp_frame(tmp_path / "sharp.jpg", rng)
    write_blurred_frame(tmp_path / "blurred.jpg", rng)

    sharp = sharpness_score(tmp_path / "sharp.jpg")
    blurred = sharpness_score(tmp_path / "blurred.jpg")

    assert sharp is not None and blurred is not None
    assert sharp > blurred


def test_sharpness_score_none_for_unreadable_file(tmp_path: Path) -> None:
    (tmp_path / "bogus.jpg").write_bytes(b"img")
    assert sharpness_score(tmp_path / "bogus.jpg") is None


def test_compute_blur_rejects_flags_blurred_outliers(tmp_path: Path) -> None:
    detections = make_session(tmp_path, n_sharp=8, n_blurred=2)

    rejects, stats = compute_blur_rejects(detections, tmp_path)

    assert rejects == {"frame_0008.jpg", "frame_0009.jpg"}
    assert stats["metric"] == "variance_of_laplacian"
    assert stats["threshold"] is not None
    assert stats["frames_rejected"] == 2


def test_compute_blur_rejects_uniform_session_rejects_nothing(tmp_path: Path) -> None:
    detections = make_session(tmp_path, n_sharp=6, n_blurred=0)

    rejects, stats = compute_blur_rejects(detections, tmp_path)

    assert rejects == set()
    assert stats["frames_rejected"] == 0


def test_compute_blur_rejects_skips_unscoreable_frames(tmp_path: Path) -> None:
    (tmp_path / "frame_0000.jpg").write_bytes(b"img")
    (tmp_path / "frame_0001.jpg").write_bytes(b"img")
    detections = make_detections(
        [
            DetectionEntry("frame_0000.jpg", 0, 1, [3], []),
            DetectionEntry("frame_0001.jpg", 1, 1, [3], []),
        ]
    )

    rejects, stats = compute_blur_rejects(detections, tmp_path)

    assert rejects == set()
    assert stats["frames_scored"] == 0


def test_filter_frames_excludes_blur_rejects(tmp_path: Path) -> None:
    detections = make_session(tmp_path, n_sharp=8, n_blurred=2)
    rejects, _ = compute_blur_rejects(detections, tmp_path)

    result = filter_frames(detections, tmp_path, min_markers=1, blur_rejects=rejects)

    kept = {e.filename for e in result}
    assert "frame_0008.jpg" not in kept
    assert "frame_0009.jpg" not in kept
    assert len(kept) == 8
    assert not (tmp_path / "filtered" / "frame_0008.jpg").exists()


def test_save_manifest_records_sharpness_stats(tmp_path: Path) -> None:
    detections = make_session(tmp_path, n_sharp=8, n_blurred=2)
    rejects, stats = compute_blur_rejects(detections, tmp_path)
    filtered = filter_frames(detections, tmp_path, min_markers=1, blur_rejects=rejects)

    save_manifest(filtered, detections, tmp_path, min_markers=1, sharpness=stats)

    output = json.loads((tmp_path / "manifest.json").read_text())
    assert output["sharpness"]["metric"] == "variance_of_laplacian"
    assert output["sharpness"]["frames_rejected"] == 2
    assert output["sharpness"]["threshold"] is not None


# --- A2: tool versions in manifest ---


def test_save_manifest_records_tool_versions(tmp_path: Path) -> None:
    detections = make_detections([])

    save_manifest([], detections, tmp_path, min_markers=1)

    output = json.loads((tmp_path / "manifest.json").read_text())
    versions = output["tool_versions"]
    assert versions["python"]
    assert versions["opencv"] == cv2.__version__
    assert versions["numpy"] == np.__version__


def test_filter_manifest_parses_old_schema_without_new_fields() -> None:
    old = {
        "dictionary": "DICT_4X4_250",
        "min_markers": 1,
        "total_frames_input": 1,
        "frames_filtered_out": 0,
        "frames": ["frame_0000.jpg"],
        "marker_detections": {"frame_0000.jpg": []},
    }
    manifest = FilterManifest.from_dict(old)
    assert manifest.sharpness is None
    assert manifest.tool_versions is None


# --- strip_invalid_ids ---


def test_strip_invalid_ids_returns_new_entry_with_only_valid_ids() -> None:
    entry = DetectionEntry(
        "a.jpg", 0, 2, [1, 99], [[[0, 0], [1, 0], [1, 1], [0, 1]]] * 2
    )
    result = strip_invalid_ids(entry, valid_ids={1})
    assert result.marker_ids == [1]
    assert result.markers_detected == 1
    assert entry.marker_ids == [1, 99]  # original entry is untouched
