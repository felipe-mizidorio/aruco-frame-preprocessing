import argparse
import logging
import re
from datetime import datetime
from pathlib import Path

import cv2

from ..config import load_config
from ..core import pipeline_io
from ..core.schemas import FrameEntry, VideoMetadata

VALID_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

logger = logging.getLogger(__name__)

# Container metadata lives in the moov/udta boxes near the start or end of the
# file; scanning this many bytes from each end covers both layouts without
# reading multi-GB videos whole.
_PROBE_WINDOW_BYTES = 8 * 1024 * 1024
# 35mm-equivalent focal lengths outside this range are treated as parse noise.
_F35_PLAUSIBLE_MM = (10.0, 200.0)

_XMP_F35_RE = re.compile(
    rb"FocalLengthIn35mmFilm[^0-9]{0,4}(\d+)(?:/(\d+))?", re.IGNORECASE
)
_APPLE_F35_KEY = b"com.apple.quicktime.camera.focal_length.35mm_equivalent"
_APPLE_F35_VALUE_RE = re.compile(rb"(\d+(?:\.\d+)?)")


def probe_focal_35mm(video_path: Path) -> float | None:
    """Best-effort 35mm-equivalent focal length from the video container.

    Looks for XMP/EXIF `FocalLengthIn35mmFilm` and the Apple QuickTime
    per-capture key. Returns None when metadata is absent — the common case
    for domestic videos transferred through messaging apps, which strip it.
    """
    try:
        size = video_path.stat().st_size
        with video_path.open("rb") as f:
            head = f.read(_PROBE_WINDOW_BYTES)
            if size > 2 * _PROBE_WINDOW_BYTES:
                f.seek(-_PROBE_WINDOW_BYTES, 2)
                data = head + f.read(_PROBE_WINDOW_BYTES)
            else:
                data = head + f.read()
    except OSError as exc:
        logger.warning("Could not probe video metadata: %s", exc)
        return None

    candidates: list[float] = []
    for m in _XMP_F35_RE.finditer(data):
        numerator = float(m.group(1))
        denominator = float(m.group(2)) if m.group(2) else 1.0
        if denominator:
            candidates.append(numerator / denominator)

    key_pos = data.find(_APPLE_F35_KEY)
    if key_pos != -1:
        start = key_pos + len(_APPLE_F35_KEY)
        value = _APPLE_F35_VALUE_RE.search(data[start : start + 64])
        if value:
            candidates.append(float(value.group(1)))

    for f35 in candidates:
        if _F35_PLAUSIBLE_MM[0] <= f35 <= _F35_PLAUSIBLE_MM[1]:
            logger.info("Focal length (35mm equivalent) from container: %.1f mm", f35)
            return f35
    logger.info("No usable focal-length metadata in the video container.")
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames from a video file at a given stride interval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input video file. Supported formats: .mp4, .avi, .mov, .mkv",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Extract one frame every N frames.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/frames"),
        help="Directory where extracted frames will be saved. \
        Created if it does not exist.",
    )
    return parser.parse_args()


def validate_input(video_path: Path) -> None:
    if not video_path.is_file():
        logger.error("File not found: %s", video_path)
        raise FileNotFoundError(f"File not found: {video_path}")

    if video_path.suffix.lower() not in VALID_EXTENSIONS:
        logger.error(
            "Unsupported file extension '%s': %s", video_path.suffix, video_path
        )
        raise ValueError(
            f"Unsupported file extension '{video_path.suffix}'. "
            f"Valid extensions: {VALID_EXTENSIONS}"
        )


def extract_frames(
    video_path: Path, stride: int, output_dir: Path
) -> tuple[list[FrameEntry], float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        logger.error("Could not open video file: %s", video_path)
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        cap.release()
        raise ValueError(f"Video reports invalid FPS ({fps}): {video_path}")

    logger.info(
        "Opened '%s' — %.2f fps, %d frames, %dx%d",
        video_path.name,
        fps,
        total_frames,
        frame_width,
        frame_height,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    pad_width = len(str(total_frames))
    frames_metadata: list[FrameEntry] = []
    frames_expected = len(range(0, total_frames, stride))

    try:
        for i, frame_index in enumerate(range(0, total_frames, stride)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, image = cap.read()

            if not ret:
                logger.warning("Could not read frame %d — skipping.", frame_index)
                continue

            file_name = f"frame_{str(frame_index).zfill(pad_width)}.jpg"
            cv2.imwrite(str(output_dir / file_name), image)

            frames_metadata.append(
                FrameEntry(
                    frame_index=frame_index,
                    timestamp_s=round(frame_index / fps, 4),
                    filename=file_name,
                )
            )

            pipeline_io.log_progress(i, frames_expected)

    finally:
        cap.release()

    logger.info(
        "Extraction complete. %d frames saved to '%s'.",
        len(frames_metadata),
        output_dir,
    )
    return frames_metadata, fps, total_frames, frame_width, frame_height


def save_metadata(
    video_path: Path,
    fps: float,
    total_frames: int,
    frame_width: int,
    frame_height: int,
    stride: int,
    frames_metadata: list[FrameEntry],
    output_dir: Path,
    focal_length_35mm: float | None = None,
) -> None:
    metadata = VideoMetadata(
        source_video=str(video_path.resolve()),
        fps=fps,
        total_frames=total_frames,
        resolution={"width": frame_width, "height": frame_height},
        stride=stride,
        frames_extracted=len(frames_metadata),
        extracted_at=datetime.now().isoformat(timespec="seconds"),
        opencv_version=cv2.__version__,
        frames=frames_metadata,
        focal_length_35mm=focal_length_35mm,
    )

    metadata_path = output_dir / "metadata.json"
    pipeline_io.save_json(metadata.to_dict(), metadata_path)

    logger.info("Metadata saved to '%s'.", metadata_path)


def main() -> None:
    pipeline_io.configure_logging()
    args = parse_args()

    validate_input(args.input)

    output_session_dir = (
        args.output_dir
        / f"{args.input.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    cfg = load_config()
    stride = args.stride if args.stride is not None else cfg.frame_extraction.stride

    frames_metadata, fps, total_frames, frame_width, frame_height = extract_frames(
        args.input, stride, output_session_dir
    )

    save_metadata(
        video_path=args.input,
        fps=fps,
        total_frames=total_frames,
        frame_width=frame_width,
        frame_height=frame_height,
        stride=stride,
        frames_metadata=frames_metadata,
        output_dir=output_session_dir,
        focal_length_35mm=probe_focal_35mm(args.input),
    )


if __name__ == "__main__":
    main()
