import json
from pathlib import Path

from aruco_pipeline.core.schemas import DetectionsFile, VideoMetadata
from aruco_pipeline.stages.frame_extraction import probe_focal_35mm
from aruco_pipeline.stages.frame_filtering import save_manifest

# --- probe_focal_35mm ---


def _mp4_with(payload: bytes, tmp_path: Path) -> Path:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00" * 64 + payload + b"\x00" * 64)
    return video


def test_probe_finds_xmp_focal_rational(tmp_path: Path) -> None:
    payload = b"<exif:FocalLengthIn35mmFilm>27/1</exif:FocalLengthIn35mmFilm>"
    assert probe_focal_35mm(_mp4_with(payload, tmp_path)) == 27.0


def test_probe_finds_xmp_focal_plain(tmp_path: Path) -> None:
    payload = b"<exif:FocalLengthIn35mmFilm>24</exif:FocalLengthIn35mmFilm>"
    assert probe_focal_35mm(_mp4_with(payload, tmp_path)) == 24.0


def test_probe_finds_apple_quicktime_key(tmp_path: Path) -> None:
    payload = (
        b"com.apple.quicktime.camera.focal_length.35mm_equivalent"
        + b"\x00" * 20
        + b"26.5"
    )
    assert probe_focal_35mm(_mp4_with(payload, tmp_path)) == 26.5


def test_probe_metadata_free_video_returns_none(tmp_path: Path) -> None:
    video = tmp_path / "bare.mp4"
    video.write_bytes(bytes(range(256)) * 64)
    assert probe_focal_35mm(video) is None


def test_probe_rejects_implausible_values(tmp_path: Path) -> None:
    payload = b"<exif:FocalLengthIn35mmFilm>9000</exif:FocalLengthIn35mmFilm>"
    assert probe_focal_35mm(_mp4_with(payload, tmp_path)) is None


# --- camera block passthrough into the filtering manifest ---


def _metadata(tmp_path: Path, focal: float | None) -> None:
    meta = VideoMetadata(
        source_video="clip.mp4",
        fps=30.0,
        total_frames=10,
        resolution={"width": 480, "height": 852},
        stride=5,
        frames_extracted=2,
        extracted_at="2026-07-14T00:00:00",
        opencv_version="4.13.0",
        frames=[],
        focal_length_35mm=focal,
    )
    (tmp_path / "metadata.json").write_text(json.dumps(meta.to_dict()))


def _detections() -> DetectionsFile:
    return DetectionsFile(
        dictionary="DICT_4X4_250",
        total_frames=0,
        frames_with_detections=0,
        detections=[],
    )


def test_manifest_camera_block_with_focal(tmp_path: Path) -> None:
    _metadata(tmp_path, focal=27.0)

    save_manifest([], _detections(), tmp_path, min_markers=1)

    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data["camera"] == {
        "focal_length_35mm": 27.0,
        "width_px": 480,
        "height_px": 852,
    }


def test_manifest_camera_block_with_null_focal(tmp_path: Path) -> None:
    _metadata(tmp_path, focal=None)

    save_manifest([], _detections(), tmp_path, min_markers=1)

    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data["camera"]["focal_length_35mm"] is None
    assert data["camera"]["width_px"] == 480


def test_manifest_no_metadata_file_omits_camera_block(tmp_path: Path) -> None:
    save_manifest([], _detections(), tmp_path, min_markers=1)

    data = json.loads((tmp_path / "manifest.json").read_text())
    assert "camera" not in data


def test_video_metadata_old_schema_parses_without_focal() -> None:
    old = {
        "source_video": "v.mp4",
        "fps": 30.0,
        "total_frames": 1,
        "resolution": {"width": 100, "height": 100},
        "stride": 1,
        "frames_extracted": 1,
        "extracted_at": "t",
        "opencv_version": "4",
        "frames": [],
    }
    assert VideoMetadata.from_dict(old).focal_length_35mm is None
