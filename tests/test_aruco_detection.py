import json
from pathlib import Path

import cv2
import numpy as np

from aruco_detection import detect_markers, save_detections
from schemas import DetectionEntry, FrameEntry, VideoMetadata


def make_metadata(tmp_path: Path, frames: list[FrameEntry]) -> VideoMetadata:
    return VideoMetadata(
        source_video=str(tmp_path / "video.mp4"),
        fps=30.0,
        total_frames=len(frames),
        resolution={"width": 640, "height": 480},
        stride=1,
        frames_extracted=len(frames),
        extracted_at="2026-07-01T00:00:00",
        opencv_version=cv2.__version__,
        frames=frames,
    )


def write_blank_jpeg(path: Path) -> None:
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.imwrite(str(path), blank)


def write_marker_jpeg(
    path: Path,
    marker_id: int = 0,
    size: int = 200,
    border: int = 40,
    dictionary: int = cv2.aruco.DICT_4X4_250,
) -> None:
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size)
    padded = np.full((size + 2 * border, size + 2 * border), 255, dtype=np.uint8)
    padded[border : border + size, border : border + size] = marker
    cv2.imwrite(str(path), padded)


# --- detect_markers ---


def test_detect_markers_blank_image_returns_zero_markers(tmp_path: Path) -> None:
    write_blank_jpeg(tmp_path / "frame_0000.jpg")
    metadata = make_metadata(
        tmp_path,
        [FrameEntry(frame_index=0, timestamp_s=0.0, filename="frame_0000.jpg")],
    )

    result = detect_markers(metadata, tmp_path)

    assert len(result) == 1
    assert result[0].markers_detected == 0
    assert result[0].marker_ids == []


def test_detect_markers_finds_aruco_marker(tmp_path: Path) -> None:
    write_marker_jpeg(tmp_path / "frame_0000.jpg", marker_id=0)
    metadata = make_metadata(
        tmp_path,
        [FrameEntry(frame_index=0, timestamp_s=0.0, filename="frame_0000.jpg")],
    )

    result = detect_markers(metadata, tmp_path)

    assert len(result) == 1
    assert result[0].markers_detected == 1
    assert result[0].marker_ids == [0]


def test_detect_markers_uses_requested_dictionary(tmp_path: Path) -> None:
    write_marker_jpeg(
        tmp_path / "frame_0000.jpg", marker_id=0, dictionary=cv2.aruco.DICT_5X5_50
    )
    metadata = make_metadata(
        tmp_path,
        [FrameEntry(frame_index=0, timestamp_s=0.0, filename="frame_0000.jpg")],
    )

    result = detect_markers(metadata, tmp_path, dictionary_name="DICT_5X5_50")

    assert result[0].markers_detected == 1
    assert result[0].marker_ids == [0]


def test_detect_markers_default_dictionary_misses_other_dictionary_marker(
    tmp_path: Path,
) -> None:
    write_marker_jpeg(
        tmp_path / "frame_0000.jpg", marker_id=0, dictionary=cv2.aruco.DICT_5X5_50
    )
    metadata = make_metadata(
        tmp_path,
        [FrameEntry(frame_index=0, timestamp_s=0.0, filename="frame_0000.jpg")],
    )

    result = detect_markers(metadata, tmp_path)  # default DICT_4X4_250

    assert result[0].markers_detected == 0


def test_detect_markers_skips_missing_frame(tmp_path: Path) -> None:
    metadata = make_metadata(
        tmp_path, [FrameEntry(frame_index=0, timestamp_s=0.0, filename="missing.jpg")]
    )

    result = detect_markers(metadata, tmp_path)

    assert len(result) == 1
    assert result[0].markers_detected == 0
    assert result[0].marker_ids == []
    assert result[0].corners == []
    assert result[0].filename == "missing.jpg"
    assert result[0].frame_index == 0


# --- save_detections ---


def test_save_detections_writes_file(tmp_path: Path) -> None:
    metadata = make_metadata(tmp_path, [])
    save_detections([], metadata, tmp_path, "DICT_4X4_250")
    assert (tmp_path / "detections.json").exists()


def test_save_detections_schema(tmp_path: Path) -> None:
    metadata = make_metadata(tmp_path, [])
    save_detections([], metadata, tmp_path, "DICT_4X4_250")

    output = json.loads((tmp_path / "detections.json").read_text())
    assert "dictionary" in output
    assert "total_frames" in output
    assert "frames_with_detections" in output
    assert "detections" in output


def test_save_detections_records_requested_dictionary(tmp_path: Path) -> None:
    metadata = make_metadata(tmp_path, [])
    save_detections([], metadata, tmp_path, "DICT_5X5_50")

    output = json.loads((tmp_path / "detections.json").read_text())
    assert output["dictionary"] == "DICT_5X5_50"


def test_save_detections_counts_correctly(tmp_path: Path) -> None:
    detections = [
        DetectionEntry(
            filename="a.jpg",
            frame_index=0,
            markers_detected=2,
            marker_ids=[0, 1],
            corners=[],
        ),
        DetectionEntry(
            filename="b.jpg",
            frame_index=1,
            markers_detected=0,
            marker_ids=[],
            corners=[],
        ),
        DetectionEntry(
            filename="c.jpg",
            frame_index=2,
            markers_detected=1,
            marker_ids=[3],
            corners=[],
        ),
    ]
    metadata = make_metadata(
        tmp_path,
        [
            FrameEntry(frame_index=i, timestamp_s=0.0, filename=f"f{i}.jpg")
            for i in range(3)
        ],
    )
    save_detections(detections, metadata, tmp_path, "DICT_4X4_250")

    output = json.loads((tmp_path / "detections.json").read_text())
    assert output["frames_with_detections"] == 2
    assert output["total_frames"] == 3
