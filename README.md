# aruco-frame-preprocessing

Pipeline for extracting, filtering, and preprocessing video frames using ArUco marker detection. Designed to produce clean, marker-validated frames as input for deep learning models (e.g., DeepArUco).

## Description

Given a video source, the pipeline:

1. **Extracts** frames from video files
2. **Detects** ArUco markers in each frame using OpenCV
3. **Filters** frames based on detection quality and marker presence
4. **Generates** ArUco marker images for testing and calibration
5. **Compares** classical OpenCV detection results against DeepArUco model outputs

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `opencv-contrib-python` | >=4.13 | ArUco detection and image processing |
| `opencv-python` | >=4.13 | Core OpenCV image I/O and processing |
| `numpy` | >=2.4 | Numerical operations |
| `torch` | >=2.11 | DeepArUco inference |
| `torchvision` | >=0.26 | Image transforms for PyTorch |

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

`deeparuco_comparison.py` needs the deep-learning stack (`tensorflow`, `ultralytics`), which is an optional extra rather than a core dependency:

```bash
uv sync --dev --extra deeparuco
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

# 5. Compare OpenCV detections against DeepArUco++ model outputs
uv run aruco-compare --detections <session-dir>/detections.json

# Generate ArUco marker images for testing/calibration (standalone utility)
uv run aruco-generate-markers
```

Outputs to `data/markers` by default. Marker shape/count come from `configs/pipeline.yaml`'s `markers:` block (currently 20 markers, `DICT_4X4_50`, 236px coded side + 59px white margin per side, 300 DPI) unless overridden via CLI flags (`--num-markers`, `--side-pixels`, `--margin-pixels`, `--dictionary`, `--dpi`, `--output-dir`).

### Configuration

Session defaults live in `configs/pipeline.yaml` — the default ArUco
dictionary, `frame_extraction.stride`, `frame_filtering.min_markers` /
`valid_ids`, marker-sheet generation settings, and the DeepArUco weight
download settings (`base_url`, `weights_dir`, `weights`).

Precedence for every configurable value is:

**CLI flag > `configs/pipeline.yaml` > hardcoded fallback in `aruco_pipeline/config.py`.**

Algorithm constants that tune detection/filtering behavior (e.g.
`BLUR_MAD_K`, `HULL_MARGIN_MARKER_SIDES`) are **not** in the yaml — they stay
as constants in code since they are tuning knobs for the algorithms
themselves, not per-session settings.

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
│       │   ├── mask_generation.py      # Foreground mask generation
│       │   └── deeparuco_comparison.py # OpenCV vs DeepArUco comparison
│       ├── markers/
│       │   └── generate_markers.py     # ArUco marker image generation
│       └── deeparuco_vendor/     # Vendored third-party DeepArUco++ code
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
