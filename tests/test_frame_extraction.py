import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from frame_extraction import extract_frames, save_metadata, validate_input
from schemas import FrameEntry


def make_test_video(
    path: Path,
    num_frames: int = 10,
    fps: float = 30.0,
    width: int = 64,
    height: int = 64,
) -> None:
    out = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
    )
    for _ in range(num_frames):
        out.write(np.zeros((height, width, 3), dtype=np.uint8))
    out.release()


# --- validate_input ---


def test_validate_input_raises_when_file_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_input(tmp_path / "nonexistent.mp4")


def test_validate_input_raises_on_unsupported_extension(tmp_path: Path) -> None:
    bad = tmp_path / "video.txt"
    bad.write_bytes(b"")
    with pytest.raises(ValueError):
        validate_input(bad)


@pytest.mark.parametrize("extension", [".mp4", ".avi", ".mov", ".mkv"])
def test_validate_input_passes_for_valid_extensions(
    tmp_path: Path, extension: str
) -> None:
    video = tmp_path / f"video{extension}"
    video.write_bytes(b"")
    validate_input(video)  # must not raise


# --- extract_frames ---


def test_extract_frames_creates_output_dir(tmp_path: Path) -> None:
    video_path = tmp_path / "test.avi"
    make_test_video(video_path)
    output_dir = tmp_path / "frames" / "session"

    extract_frames(video_path, stride=1, output_dir=output_dir)

    assert output_dir.is_dir()


def test_extract_frames_writes_jpeg_files(tmp_path: Path) -> None:
    video_path = tmp_path / "test.avi"
    make_test_video(video_path, num_frames=5)
    output_dir = tmp_path / "out"

    extract_frames(video_path, stride=1, output_dir=output_dir)

    jpegs = list(output_dir.glob("*.jpg"))
    assert len(jpegs) > 0


def test_extract_frames_respects_stride(tmp_path: Path) -> None:
    video_path = tmp_path / "test.avi"
    make_test_video(video_path, num_frames=10)
    output_dir = tmp_path / "out"

    frames, *_ = extract_frames(video_path, stride=5, output_dir=output_dir)

    assert len(frames) == 2  # frames 0 and 5


def test_extract_frames_returns_frame_entries(tmp_path: Path) -> None:
    video_path = tmp_path / "test.avi"
    make_test_video(video_path, num_frames=5)
    output_dir = tmp_path / "out"

    frames, *_ = extract_frames(video_path, stride=1, output_dir=output_dir)

    for entry in frames:
        assert isinstance(entry, FrameEntry)
        assert entry.filename.endswith(".jpg")


# --- save_metadata ---


def test_save_metadata_writes_file(tmp_path: Path) -> None:
    save_metadata(
        video_path=tmp_path / "video.mp4",
        fps=30.0,
        total_frames=300,
        frame_width=1920,
        frame_height=1080,
        stride=1,
        frames_metadata=[],
        output_dir=tmp_path,
    )
    assert (tmp_path / "metadata.json").exists()


def test_save_metadata_schema(tmp_path: Path) -> None:
    save_metadata(
        video_path=tmp_path / "video.mp4",
        fps=30.0,
        total_frames=300,
        frame_width=1920,
        frame_height=1080,
        stride=5,
        frames_metadata=[
            FrameEntry(frame_index=0, timestamp_s=0.0, filename="frame_0.jpg")
        ],
        output_dir=tmp_path,
    )

    output = json.loads((tmp_path / "metadata.json").read_text())
    assert "source_video" in output
    assert "fps" in output
    assert "total_frames" in output
    assert "resolution" in output
    assert "stride" in output
    assert "frames_extracted" in output
    assert "extracted_at" in output
    assert "opencv_version" in output
    assert "frames" in output
    assert output["stride"] == 5
    assert output["frames_extracted"] == 1
