import pytest

from schemas import (
    ComparisonFile,
    DetectionEntry,
    DetectionsFile,
    FilterManifest,
    FrameEntry,
    MarkerDetection,
    MarkerSheetManifest,
    VideoMetadata,
)

# --- FrameEntry ---


def test_frame_entry_round_trip() -> None:
    entry = FrameEntry(frame_index=3, timestamp_s=1.5, filename="frame_0003.jpg")
    assert FrameEntry.from_dict(entry.to_dict(), source="test") == entry


def test_frame_entry_from_dict_missing_key_raises() -> None:
    with pytest.raises(ValueError, match="filename"):
        FrameEntry.from_dict({"frame_index": 0, "timestamp_s": 0.0}, source="test.json")


# --- VideoMetadata ---


def _video_metadata_dict() -> dict:
    return {
        "source_video": "video.mp4",
        "fps": 30.0,
        "total_frames": 10,
        "resolution": {"width": 640, "height": 480},
        "stride": 1,
        "frames_extracted": 1,
        "extracted_at": "2026-07-01T00:00:00",
        "opencv_version": "4.13.0",
        "frames": [
            {"frame_index": 0, "timestamp_s": 0.0, "filename": "frame_0000.jpg"}
        ],
    }


def test_video_metadata_round_trip() -> None:
    data = _video_metadata_dict()
    metadata = VideoMetadata.from_dict(data, source="metadata.json")
    assert metadata.to_dict() == data


def test_video_metadata_from_dict_missing_key_raises() -> None:
    data = _video_metadata_dict()
    del data["fps"]
    with pytest.raises(ValueError, match="fps"):
        VideoMetadata.from_dict(data, source="metadata.json")


# --- DetectionEntry / DetectionsFile ---


def _detections_file_dict() -> dict:
    return {
        "dictionary": "DICT_4X4_250",
        "total_frames": 1,
        "frames_with_detections": 1,
        "detections": [
            {
                "filename": "frame_0000.jpg",
                "frame_index": 0,
                "markers_detected": 1,
                "marker_ids": [3],
                "corners": [[[0, 0], [1, 0], [1, 1], [0, 1]]],
            }
        ],
    }


def test_detections_file_round_trip() -> None:
    data = _detections_file_dict()
    detections = DetectionsFile.from_dict(data, source="detections.json")
    assert detections.to_dict() == data


def test_detections_file_from_dict_missing_key_raises() -> None:
    data = _detections_file_dict()
    del data["dictionary"]
    with pytest.raises(ValueError, match="dictionary"):
        DetectionsFile.from_dict(data, source="detections.json")


def test_detection_entry_from_dict_missing_key_raises() -> None:
    with pytest.raises(ValueError, match="marker_ids"):
        DetectionEntry.from_dict(
            {
                "filename": "a.jpg",
                "frame_index": 0,
                "markers_detected": 0,
                "corners": [],
            },
            source="test.json",
        )


# --- MarkerDetection / FilterManifest ---


def _filter_manifest_dict() -> dict:
    return {
        "dictionary": "DICT_4X4_250",
        "min_markers": 1,
        "total_frames_input": 2,
        "frames_filtered_out": 1,
        "frames": ["frame_0000.jpg"],
        "marker_detections": {
            "frame_0000.jpg": [{"id": 3, "corners": [[0, 0], [1, 0], [1, 1], [0, 1]]}]
        },
    }


def test_filter_manifest_round_trip() -> None:
    data = _filter_manifest_dict()
    manifest = FilterManifest.from_dict(data, source="manifest.json")
    assert manifest.to_dict() == data


def test_filter_manifest_from_dict_missing_key_raises() -> None:
    data = _filter_manifest_dict()
    del data["marker_detections"]
    with pytest.raises(ValueError, match="marker_detections"):
        FilterManifest.from_dict(data, source="manifest.json")


def test_marker_detection_from_dict_missing_key_raises() -> None:
    with pytest.raises(ValueError, match="corners"):
        MarkerDetection.from_dict({"id": 3}, source="test.json")


# --- MarkerSheetManifest ---


def _marker_sheet_manifest_dict() -> dict:
    return {
        "dictionary": "DICT_4X4_50",
        "num_markers": 2,
        "ids": [0, 1],
        "side_pixels": 236,
        "margin_pixels": 59,
        "total_image_side_pixels": 354,
        "dpi": 300,
        "total_image_side_mm": 30.0,
        "generated_at": "2026-07-01T00:00:00+00:00",
    }


def test_marker_sheet_manifest_round_trip() -> None:
    data = _marker_sheet_manifest_dict()
    manifest = MarkerSheetManifest.from_dict(data, source="manifest.json")
    assert manifest.to_dict() == data


def test_marker_sheet_manifest_from_dict_missing_key_raises() -> None:
    data = _marker_sheet_manifest_dict()
    del data["dpi"]
    with pytest.raises(ValueError, match="dpi"):
        MarkerSheetManifest.from_dict(data, source="manifest.json")


# --- ComparisonFile ---


def _comparison_file_dict() -> dict:
    return {
        "dictionary": "DICT_4X4_250",
        "model": "deeparuco",
        "weights_path": "/weights",
        "total_frames": 0,
        "summary": {},
        "frames": [],
    }


def test_comparison_file_round_trip() -> None:
    data = _comparison_file_dict()
    comparison = ComparisonFile.from_dict(data, source="comparison.json")
    assert comparison.to_dict() == data


def test_comparison_file_from_dict_missing_key_raises() -> None:
    data = _comparison_file_dict()
    del data["model"]
    with pytest.raises(ValueError, match="model"):
        ComparisonFile.from_dict(data, source="comparison.json")
