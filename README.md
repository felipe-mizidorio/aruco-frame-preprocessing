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

## Usage

### Generate ArUco markers

```bash
uv run python src/generate_markers.py
```

Outputs 10 markers (200x200 px, `DICT_4X4_250`) to `data/raw/`.

## Project Structure

```
aruco-frame-preprocessing/
├── src/
│   ├── __init__.py              # Logging configuration
│   ├── generate_markers.py      # ArUco marker image generation
│   ├── frame_extraction.py      # Video frame extraction
│   ├── aruco_detection.py       # ArUco marker detection
│   ├── frame_filtering.py       # Frame quality filtering
│   └── deeparuco_comparison.py  # OpenCV vs DeepArUco comparison
├── data/                        # Input data (videos, raw frames)
├── outputs/                     # Processed frames and results
├── notebooks/                   # Jupyter notebooks for exploration
├── pyproject.toml
└── .pre-commit-config.yaml
```

## Development

```bash
# Lint and format
uv run ruff check --fix src/
uv run ruff format src/

# Type check
uv run pyright
```

Pre-commit hooks run `ruff-check` and `ruff-format` automatically on each commit.
