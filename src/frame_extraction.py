import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import cv2

VALID_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

logger = logging.getLogger(__name__)


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
        default=1,
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
) -> tuple[list[dict], float, int, int, int]:
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
    frames_metadata: list[dict] = []
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
                {
                    "frame_index": frame_index,
                    "timestamp_s": round(frame_index / fps, 4),
                    "filename": file_name,
                }
            )

            if i % 50 == 0:
                logger.info("Progress: frame %d / %d extracted", i, frames_expected)

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
    frames_metadata: list[dict],
    output_dir: Path,
) -> None:
    metadata = {
        "source_video": str(video_path.resolve()),
        "fps": fps,
        "total_frames": total_frames,
        "resolution": {"width": frame_width, "height": frame_height},
        "stride": stride,
        "frames_extracted": len(frames_metadata),
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
        "opencv_version": cv2.__version__,
        "frames": frames_metadata,
    }

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Metadata saved to '%s'.", metadata_path)


def main() -> None:
    args = parse_args()

    validate_input(args.input)

    session_dir = (
        args.output_dir
        / f"{args.input.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    frames_metadata, fps, total_frames, frame_width, frame_height = extract_frames(
        args.input, args.stride, session_dir
    )

    save_metadata(
        video_path=args.input,
        fps=fps,
        total_frames=total_frames,
        frame_width=frame_width,
        frame_height=frame_height,
        stride=args.stride,
        frames_metadata=frames_metadata,
        output_dir=session_dir,
    )


if __name__ == "__main__":
    main()
