# aruco-frame-preprocessing

Pipeline for extracting, filtering, and preprocessing video frames using ArUco marker detection. Designed to produce clean, marker-validated frames and foreground masks as input for downstream 3D reconstruction (e.g., COLMAP).

## Description

Given a video source, the pipeline:

1. **Extracts** frames from video files
2. **Detects** ArUco markers in each frame using OpenCV
3. **Filters** frames based on detection quality and marker presence
4. **Masks** the subject in each filtered frame with Grounded-SAM-2, tracked through the video by SAM 2 (foreground masks for COLMAP)
5. **Generates** ArUco marker images for testing and calibration

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `opencv-contrib-python` | >=4.13 | ArUco detection and image processing |
| `opencv-python` | >=4.13 | Core OpenCV image I/O and processing |
| `numpy` | >=2.4 | Numerical operations |
| `torch` | >=2.11 | Grounding DINO + SAM 2 inference |
| `torchvision` | >=0.26 | Image transforms for PyTorch |
| `transformers` | >=4.56 | Grounding DINO + SAM 2 (Grounded-SAM-2 masks) |

**Dev dependencies:** `ruff` (lint + format), `pyright` (type checking), `pre-commit`

## Installation

Requires Python >=3.12, <3.15 (excluding 3.14.1) and [uv](https://docs.astral.sh/uv/).

```bash
# Clone the repo
git clone https://github.com/felipe-mizidorio/aruco-frame-preprocessing.git
cd aruco-frame-preprocessing

# Install dependencies (including dev)
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install
```

## Usage

`uv sync` installs the package in editable mode and registers the console
scripts below, so every stage runs as `uv run aruco-<stage>` — no more
`python src/<file>.py`. Each stage chains off the previous one's JSON
artifact, written into a shared session directory.

```bash
# 1. Extract frames from a video
uv run aruco-extract --input path/to/video.mp4

# 2. Detect ArUco markers in the extracted frames
uv run aruco-detect --metadata <session-dir>/metadata.json

# 3. Filter frames by detection quality / marker presence
uv run aruco-filter --detections <session-dir>/detections.json

# 4. Generate foreground masks (for COLMAP) from the filtered frames
uv run aruco-mask --manifest <session-dir>/manifest.json

# Generate ArUco marker images for testing/calibration (standalone utility)
uv run aruco-generate-markers
```

`aruco-mask` anchors the subject with Grounded-SAM-2 and tracks it through the
filtered frames with the SAM 2 video model: Grounding DINO finds the
`--text-prompt` (default `"head."`) once, and SAM 2 follows that object from
frame to frame, so the mask keeps the same extent across the session. The
ArUco markers pick the right DINO box (if DINO misses, the marker bbox is the
SAM 2 prompt instead) and check every tracked mask: if a mask loses the
markers or suddenly grows or shrinks, the frame is re-prompted, and if that
fails it gets a keep-all mask and tracking restarts on the next frame. Per-frame
counts (`frames_tracked`, `reanchors`, `track_resets`, ...) are written to
`manifest.json` under `mask_generation`. Model weights
(`grounding-dino-base`, `sam2.1-hiera-large`) download to the Hugging Face
cache on first run. It uses CUDA when available (`--device auto|cuda|cpu`).
CPU works, but expect several seconds per frame.

Outputs to `data/markers` by default. Marker shape/count come from `configs/pipeline.yaml`'s `markers:` block (currently 20 markers, `DICT_4X4_50`, 236px coded side + 59px white margin per side, 300 DPI) unless overridden via CLI flags (`--num-markers`, `--side-pixels`, `--margin-pixels`, `--dictionary`, `--dpi`, `--output-dir`).

### Configuration

Session defaults live in `configs/pipeline.yaml` — the default ArUco
dictionary, `frame_extraction.stride`, `frame_filtering.min_markers` /
`valid_ids`, marker-sheet generation settings, the Grounded-SAM-2 mask settings
(`mask_generation.text_prompt`, `detector_model`, `segmenter_model`,
`device`).

Precedence for every configurable value is:

**CLI flag > `configs/pipeline.yaml` > hardcoded fallback in `aruco_pipeline/config.py`.**

Algorithm constants that tune detection/filtering behavior (e.g.
`BLUR_MAD_K`, `BOX_THRESHOLD`, `BOX_MARGIN_MARKER_SIDES`) are **not** in the yaml — they stay
as constants in code since they are tuning knobs for the algorithms
themselves, not per-session settings.

## Running with Docker (GPU)

Target: a Linux machine with an NVIDIA GPU (built for an RTX 5090). The torch
wheels ship CUDA 13, so the host needs:

- NVIDIA driver **>= 580**
- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

```bash
docker compose build
```

Put videos under `./data`, which is mounted at `/data` in the container.
Hugging Face weights persist in the `hf-cache` volume.

```bash
docker compose run --rm pipeline python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability())"
```

```bash
docker compose run --rm pipeline aruco-extract --input /data/video.mp4 --output-dir /data
```

```bash
docker compose run --rm pipeline aruco-detect --metadata /data/<session>/metadata.json
```

```bash
docker compose run --rm pipeline aruco-filter --detections /data/<session>/detections.json
```

```bash
docker compose run --rm pipeline aruco-mask --manifest /data/<session>/manifest.json
```

Export `UID`/`GID` (e.g. `export UID GID=$(id -g)`) so session files are
owned by your host user.

## Project Structure

```
aruco-frame-preprocessing/
├── src/
│   └── aruco_pipeline/
│       ├── __init__.py           # Logging configuration
│       ├── config.py             # PipelineConfig + load_config() (yaml + fallbacks)
│       ├── core/
│       │   ├── pipeline_io.py    # Shared I/O, ARUCO_DICTIONARIES map, logging
│       │   └── schemas.py        # Dataclass wire formats for on-disk JSON artifacts
│       ├── stages/
│       │   ├── frame_extraction.py     # Video frame extraction
│       │   ├── aruco_detection.py      # ArUco marker detection
│       │   ├── frame_filtering.py      # Frame quality filtering
│       │   └── mask_generation.py      # Grounded-SAM-2 tracked foreground masks
│       └── markers/
│           └── generate_markers.py     # ArUco marker image generation
├── configs/
│   └── pipeline.yaml            # Session defaults (see Configuration above)
├── data/                        # Input data (videos, raw frames)
├── outputs/                     # Processed frames and results
├── notebooks/                   # Jupyter notebooks for exploration
├── pyproject.toml
└── .pre-commit-config.yaml
```

## Development

```bash
# Lint and format
uv run ruff check --fix src/aruco_pipeline/
uv run ruff format src/aruco_pipeline/

# Type check
uv run pyright
```

Pre-commit hooks run `ruff-check` and `ruff-format` automatically on each commit.
