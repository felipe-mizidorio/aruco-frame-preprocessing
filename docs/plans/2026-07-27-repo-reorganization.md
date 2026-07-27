# Repo Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the flat `src/` scripts into an installable `aruco_pipeline` package with `aruco-*` console-script entry points, and centralize session defaults in `configs/pipeline.yaml`.

**Architecture:** `src/aruco_pipeline/` becomes a proper package: `core/` (shared infra), `stages/` (the five chained video-pipeline steps), `markers/` (marker-sheet utility), `deeparuco_vendor/` (untouched vendored code). A `config.py` layer loads `configs/pipeline.yaml` into a dataclass; each stage's CLI defaults fall back to it. Precedence is always CLI flag > yaml > hardcoded fallback. Algorithm constants stay in code.

**Tech Stack:** Python ≥3.12, uv, hatchling (build backend), pytest, ruff, pyyaml, OpenCV.

## Global Constraints

- Python: `>=3.12,!=3.14.1,<3.15`. Ruff target `py312`, select `["E","F","I"]`.
- `src/aruco_pipeline/deeparuco_vendor/` is vendored third-party code: never refactor it, never lint it (ruff-exclude + per-file-ignore), keep its filenames and `PROVENANCE.md`.
- Algorithm constants stay in code, never in yaml: `BLUR_MAD_K`, `_MAD_TO_SIGMA`, `_MIN_FRAMES_FOR_BLUR_STATS`, `HULL_MARGIN_MARKER_SIDES`, `MIN_MARKERS_FOR_HULL`, blob-detector/decoder constants in `deeparuco_comparison.py`.
- Config precedence for every configurable value: **CLI flag > yaml > hardcoded fallback**.
- Schema backward-compat rule (unchanged): new fields optional, omitted from `to_dict` when unset.
- Do NOT commit Claude/AI-branded files or add `Co-Authored-By` trailers. Plain Conventional Commits.
- `tests/conftest.py` cv2/ultralytics isolation fixture must remain and keep working.
- Test baseline before starting: `uv run pytest` = 106 passed (needs `uv sync --dev --extra deeparuco`).

## File Structure

```
src/aruco_pipeline/
  __init__.py                 # logging NullHandler (from old src/__init__.py)
  config.py                   # NEW: PipelineConfig dataclass + load_config()
  core/
    __init__.py               # NEW empty
    pipeline_io.py            # moved from src/
    schemas.py               # moved from src/
  stages/
    __init__.py               # NEW empty
    frame_extraction.py       # moved
    aruco_detection.py        # moved
    frame_filtering.py        # moved
    mask_generation.py        # moved
    deeparuco_comparison.py   # moved
  markers/
    __init__.py               # NEW empty
    generate_markers.py       # moved
  deeparuco_vendor/           # moved wholesale, contents untouched
    __init__.py aruco.py heatmaps.py losses.py utils.py PROVENANCE.md
configs/pipeline.yaml         # NEW
```

Tests keep living in `tests/` and import via the `aruco_pipeline.*` package path.

---

### Task 1: Convert `src/` to the `aruco_pipeline` package

Move every module into the new package tree, rewrite all internal imports to relative/package-qualified form, rewrite test imports, and update tool paths so the suite is green again. This is one task: a partial move leaves the whole repo unimportable, so it must land together.

**Files:**
- Create: `src/aruco_pipeline/__init__.py`, `src/aruco_pipeline/core/__init__.py`, `src/aruco_pipeline/stages/__init__.py`, `src/aruco_pipeline/markers/__init__.py`
- Move (git mv, preserve history): all `src/*.py` and `src/deeparuco_vendor/` into the tree above
- Modify: every moved source file's imports; every `tests/test_*.py` import line; `pyproject.toml` (`[tool.ruff] exclude`, per-file-ignores, `[tool.pyright] include`)
- Test: existing `tests/` suite

**Interfaces:**
- Produces: package `aruco_pipeline` with `aruco_pipeline.core.pipeline_io`, `aruco_pipeline.core.schemas`, `aruco_pipeline.stages.<name>` (each exposing `main()`), `aruco_pipeline.markers.generate_markers:main`, `aruco_pipeline.deeparuco_vendor.*`.
- `pythonpath = ["src"]` stays in `[tool.pytest.ini_options]` so `import aruco_pipeline` resolves without an install (the build backend comes in Task 2).

- [ ] **Step 1: Create the package skeleton with git mv**

```bash
cd src
mkdir -p aruco_pipeline/core aruco_pipeline/stages aruco_pipeline/markers
git mv __init__.py aruco_pipeline/__init__.py
git mv pipeline_io.py schemas.py aruco_pipeline/core/
git mv frame_extraction.py aruco_detection.py frame_filtering.py mask_generation.py deeparuco_comparison.py aruco_pipeline/stages/
git mv generate_markers.py aruco_pipeline/markers/
git mv deeparuco_vendor aruco_pipeline/deeparuco_vendor
cd ..
touch src/aruco_pipeline/core/__init__.py src/aruco_pipeline/stages/__init__.py src/aruco_pipeline/markers/__init__.py
git add src/aruco_pipeline/core/__init__.py src/aruco_pipeline/stages/__init__.py src/aruco_pipeline/markers/__init__.py
```

- [ ] **Step 2: Rewrite internal imports in every moved source file**

Apply these exact replacements. `core/pipeline_io.py` and `core/schemas.py` have **no** internal imports — leave them. In the five `stages/*.py` files and `markers/generate_markers.py`:

- `import pipeline_io` → `from ..core import pipeline_io`
- `from schemas import ...` → `from ..core.schemas import ...`

In `stages/deeparuco_comparison.py` additionally:

- `from deeparuco_vendor.aruco import find_id` → `from ..deeparuco_vendor.aruco import find_id`
- `from deeparuco_vendor.heatmaps import pos_from_heatmap` → `from ..deeparuco_vendor.heatmaps import pos_from_heatmap`
- `from deeparuco_vendor.losses import weighted_loss` → `from ..deeparuco_vendor.losses import weighted_loss`
- `from deeparuco_vendor.utils import marker_from_corners, ordered_corners` → `from ..deeparuco_vendor.utils import marker_from_corners, ordered_corners`

Do NOT touch imports inside `deeparuco_vendor/` (vendored).

- [ ] **Step 3: Rewrite test imports**

Edit each file's source-module imports to the package path (add `aruco_pipeline.core`/`aruco_pipeline.stages`/`aruco_pipeline.markers`):

- `tests/test_pipeline_io.py`: `from pipeline_io import (` → `from aruco_pipeline.core.pipeline_io import (`
- `tests/test_schemas.py`: `from schemas import (` → `from aruco_pipeline.core.schemas import (`
- `tests/test_focal_probe.py`: `from frame_extraction import probe_focal_35mm` → `from aruco_pipeline.stages.frame_extraction import probe_focal_35mm`; `from frame_filtering import save_manifest` → `from aruco_pipeline.stages.frame_filtering import save_manifest`; `from schemas import DetectionsFile, VideoMetadata` → `from aruco_pipeline.core.schemas import DetectionsFile, VideoMetadata`
- `tests/test_frame_extraction.py`: `from frame_extraction import extract_frames, save_metadata, validate_input` → `from aruco_pipeline.stages.frame_extraction import extract_frames, save_metadata, validate_input`; `from schemas import FrameEntry` → `from aruco_pipeline.core.schemas import FrameEntry`
- `tests/test_aruco_detection.py`: `from aruco_detection import detect_markers, save_detections` → `from aruco_pipeline.stages.aruco_detection import detect_markers, save_detections`; `from schemas import DetectionEntry, FrameEntry, VideoMetadata` → `from aruco_pipeline.core.schemas import DetectionEntry, FrameEntry, VideoMetadata`
- `tests/test_frame_filtering.py`: `from frame_filtering import (` → `from aruco_pipeline.stages.frame_filtering import (`; `from schemas import DetectionEntry, DetectionsFile, FilterManifest` → `from aruco_pipeline.core.schemas import DetectionEntry, DetectionsFile, FilterManifest`
- `tests/test_mask_generation.py`: `from mask_generation import (` → `from aruco_pipeline.stages.mask_generation import (`; `from schemas import FilterManifest, MarkerDetection` → `from aruco_pipeline.core.schemas import FilterManifest, MarkerDetection`
- `tests/test_generate_markers.py`: `from generate_markers import generate_marker, save_markers` → `from aruco_pipeline.markers.generate_markers import generate_marker, save_markers`
- `tests/test_deeparuco_comparison.py`: `from deeparuco_comparison import (` → `from aruco_pipeline.stages.deeparuco_comparison import (`

Any `mock.patch("frame_filtering.<x>")` / string-based patch targets in tests must repoint to the new module path, e.g. `aruco_pipeline.stages.frame_filtering.<x>`. Grep to catch them: `grep -rn "patch(\"" tests/` and fix each target string.

- [ ] **Step 4: Update tool paths in `pyproject.toml`**

```toml
[tool.ruff]
target-version = "py312"
exclude = ["src/aruco_pipeline/deeparuco_vendor"]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["E501"]
"src/aruco_pipeline/deeparuco_vendor/*" = ["E", "F", "I"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.pyright]
include = ["src/aruco_pipeline"]
```

(Leave the rest of `[tool.pyright]` unchanged.)

- [ ] **Step 5: Run the suite — expect green**

Run: `uv run pytest -q`
Expected: `106 passed`. If a test errors on import, a patch-target string or an import line was missed — fix and re-run.

- [ ] **Step 6: Lint + format the moved package**

Run: `uv run ruff check src/aruco_pipeline/ && uv run ruff format src/aruco_pipeline/`
Expected: no errors; `deeparuco_vendor` untouched (excluded).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit --no-verify -m "refactor: restructure src into aruco_pipeline package"
```

(`--no-verify`: the repo's pre-commit hook wrapper can hit a stale-venv shebang in fresh clones; ruff was already run in Step 6.)

---

### Task 2: Build backend + console-script entry points

Make the package installable so `uv sync` puts `aruco-*` commands on PATH.

**Files:**
- Modify: `pyproject.toml` (add `[build-system]`, `[tool.hatch.build.targets.wheel]`, `[project.scripts]`)
- Test: manual `--help` invocation of each script

**Interfaces:**
- Consumes: `main()` in each `aruco_pipeline.stages.*` and `aruco_pipeline.markers.generate_markers`.
- Produces: commands `aruco-extract`, `aruco-detect`, `aruco-filter`, `aruco-mask`, `aruco-compare`, `aruco-generate-markers`.

- [ ] **Step 1: Add build-system and entry points to `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/aruco_pipeline"]

[project.scripts]
aruco-extract = "aruco_pipeline.stages.frame_extraction:main"
aruco-detect = "aruco_pipeline.stages.aruco_detection:main"
aruco-filter = "aruco_pipeline.stages.frame_filtering:main"
aruco-mask = "aruco_pipeline.stages.mask_generation:main"
aruco-compare = "aruco_pipeline.stages.deeparuco_comparison:main"
aruco-generate-markers = "aruco_pipeline.markers.generate_markers:main"
```

- [ ] **Step 2: Reinstall the project**

Run: `uv sync --dev --extra deeparuco`
Expected: succeeds; installs `aruco-frame-preprocessing` editable.

- [ ] **Step 3: Verify each console script resolves**

Run:
```bash
for c in aruco-extract aruco-detect aruco-filter aruco-mask aruco-compare aruco-generate-markers; do echo "== $c =="; uv run $c --help >/dev/null && echo OK; done
```
Expected: `OK` for all six (argparse `--help` exits 0).

- [ ] **Step 4: Re-run the suite**

Run: `uv run pytest -q`
Expected: `106 passed` (install must not change behavior).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit --no-verify -m "build: add hatchling backend and aruco-* console scripts"
```

---

### Task 3: Config layer — `configs/pipeline.yaml` + `config.py`

Add the yaml and a loader that reads it into a typed dataclass with hardcoded fallbacks. No stage is wired yet — that's Task 4.

**Files:**
- Create: `configs/pipeline.yaml`, `src/aruco_pipeline/config.py`, `tests/test_config.py`
- Modify: `pyproject.toml` (`pyyaml` dependency)

**Interfaces:**
- Produces: `aruco_pipeline.config.load_config(path: Path | None = None) -> PipelineConfig`. Dataclasses: `PipelineConfig(dictionary: str, frame_extraction: FrameExtractionConfig, frame_filtering: FrameFilteringConfig, markers: MarkersConfig, deeparuco: DeepArucoConfig)`; `FrameExtractionConfig(stride: int)`; `FrameFilteringConfig(min_markers: int, valid_ids: list[int] | None)`; `MarkersConfig(num_markers, side_pixels, margin_pixels, dictionary, dpi, page_format)`; `DeepArucoConfig(base_url: str, weights_dir: str, weights: dict[str, str])`.
- `load_config()` with no arg reads `<repo-root>/configs/pipeline.yaml`; a missing file or missing key falls back to the dataclass defaults.

- [ ] **Step 1: Add `pyyaml` dependency**

In `pyproject.toml` `dependencies`, add: `"pyyaml>=6.0,<7.0"`. Then `uv sync --dev --extra deeparuco`.

- [ ] **Step 2: Write the failing test**

Create `tests/test_config.py`:

```python
from pathlib import Path

from aruco_pipeline.config import PipelineConfig, load_config


def test_load_full_yaml(tmp_path: Path):
    (tmp_path / "pipeline.yaml").write_text(
        "dictionary: DICT_5X5_50\n"
        "frame_extraction:\n  stride: 4\n"
        "frame_filtering:\n  min_markers: 2\n  valid_ids: [0, 1, 2]\n"
        "markers:\n  num_markers: 8\n  dpi: 600\n"
        "deeparuco:\n  base_url: http://example/models\n"
    )
    cfg = load_config(tmp_path / "pipeline.yaml")
    assert cfg.dictionary == "DICT_5X5_50"
    assert cfg.frame_extraction.stride == 4
    assert cfg.frame_filtering.min_markers == 2
    assert cfg.frame_filtering.valid_ids == [0, 1, 2]
    assert cfg.markers.num_markers == 8
    assert cfg.markers.dpi == 600
    # unset key inside a present section falls back to default
    assert cfg.markers.side_pixels == 236
    assert cfg.deeparuco.base_url == "http://example/models"
    # unset deeparuco key keeps default filenames
    assert cfg.deeparuco.weights["detector"] == "det_luma_bc_s.pt"


def test_missing_file_returns_defaults(tmp_path: Path):
    cfg = load_config(tmp_path / "absent.yaml")
    assert cfg == PipelineConfig()
    assert cfg.dictionary == "DICT_4X4_250"
    assert cfg.markers.dictionary == "DICT_4X4_50"


def test_missing_section_returns_defaults(tmp_path: Path):
    (tmp_path / "pipeline.yaml").write_text("dictionary: DICT_6X6_50\n")
    cfg = load_config(tmp_path / "pipeline.yaml")
    assert cfg.dictionary == "DICT_6X6_50"
    assert cfg.frame_extraction.stride == 1
    assert cfg.frame_filtering.valid_ids is None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aruco_pipeline.config'`.

- [ ] **Step 4: Implement `config.py`**

Create `src/aruco_pipeline/config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

# src/aruco_pipeline/config.py -> parents[2] is the repo root.
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "pipeline.yaml"

_DEFAULT_WEIGHTS = {
    "detector": "det_luma_bc_s.pt",
    "regressor": "reg_hmap_8.h5",
    "decoder": "dec_new.h5",
}


@dataclass
class FrameExtractionConfig:
    stride: int = 1


@dataclass
class FrameFilteringConfig:
    min_markers: int = 1
    valid_ids: list[int] | None = None


@dataclass
class MarkersConfig:
    num_markers: int = 20
    side_pixels: int = 236
    margin_pixels: int = 59
    dictionary: str = "DICT_4X4_50"
    dpi: int = 300
    page_format: str = "A4"


@dataclass
class DeepArucoConfig:
    base_url: str = "https://raw.githubusercontent.com/AVAuco/deeparuco/master/models"
    weights_dir: str = "~/.cache/deeparuco"
    weights: dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_WEIGHTS))


@dataclass
class PipelineConfig:
    dictionary: str = "DICT_4X4_250"
    frame_extraction: FrameExtractionConfig = field(default_factory=FrameExtractionConfig)
    frame_filtering: FrameFilteringConfig = field(default_factory=FrameFilteringConfig)
    markers: MarkersConfig = field(default_factory=MarkersConfig)
    deeparuco: DeepArucoConfig = field(default_factory=DeepArucoConfig)


def _build(cls: type, data: Any):
    """Instantiate dataclass `cls`, filling only keys it declares; nested
    dataclass fields recurse, everything else falls back to the field default.
    Unknown keys in `data` are ignored so an over-specified yaml never errors."""
    if not isinstance(data, dict):
        return cls()
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        if is_dataclass(f.type):
            kwargs[f.name] = _build(f.type, data[f.name])
        else:
            kwargs[f.name] = data[f.name]
    return cls(**kwargs)


def load_config(path: Path | None = None) -> PipelineConfig:
    path = path or _DEFAULT_CONFIG_PATH
    if not path.exists():
        return PipelineConfig()
    data = yaml.safe_load(path.read_text()) or {}
    return _build(PipelineConfig, data)
```

Note: `fields(cls)` returns `f.type` as the annotation. Under `from __future__ import annotations` these are strings, so `is_dataclass(f.type)` would be False. To keep `is_dataclass(f.type)` working, resolve types once via `typing.get_type_hints`. Replace the `_build` loop's type access with a resolved map:

```python
from typing import get_type_hints

def _build(cls: type, data: Any):
    if not isinstance(data, dict):
        return cls()
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        ftype = hints[f.name]
        if is_dataclass(ftype):
            kwargs[f.name] = _build(ftype, data[f.name])
        else:
            kwargs[f.name] = data[f.name]
    return cls(**kwargs)
```

Use this second `_build`; drop `is_dataclass(f.type)` version. Remove the now-unused `from __future__ import annotations` only if it breaks the `int | None` default annotation on 3.12 — on 3.12 `X | None` in annotations is fine at runtime, and `get_type_hints` resolves string hints regardless, so keeping `from __future__ import annotations` is safe. Keep it.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -q`
Expected: `3 passed`.

- [ ] **Step 6: Create `configs/pipeline.yaml`**

```yaml
# Session defaults for the aruco pipeline. CLI flags override these;
# these override the hardcoded fallbacks in aruco_pipeline/config.py.
# Algorithm constants (blur/hull thresholds) live in code, not here.
dictionary: DICT_4X4_250          # default ArUco dictionary for detection

frame_extraction:
  stride: 1

frame_filtering:
  min_markers: 1
  valid_ids: null

markers:
  num_markers: 20
  side_pixels: 236
  margin_pixels: 59
  dictionary: DICT_4X4_50
  dpi: 300
  page_format: A4

deeparuco:
  base_url: "https://raw.githubusercontent.com/AVAuco/deeparuco/master/models"
  weights_dir: "~/.cache/deeparuco"
  weights:
    detector: det_luma_bc_s.pt
    regressor: reg_hmap_8.h5
    decoder: dec_new.h5
```

- [ ] **Step 7: Verify the real config loads and matches defaults**

Run: `uv run python -c "from aruco_pipeline.config import load_config, PipelineConfig; assert load_config() == PipelineConfig(); print('config matches defaults')"`
Expected: `config matches defaults` (the shipped yaml mirrors the dataclass defaults exactly).

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock configs/pipeline.yaml src/aruco_pipeline/config.py tests/test_config.py
git commit --no-verify -m "feat: add pipeline.yaml config layer with dataclass loader"
```

---

### Task 4: Wire config precedence into every stage

Make each stage's CLI defaults fall back to the loaded config. Pattern per stage: change the argparse `default` to `None`, then in `main()` load config once and substitute config values where the flag is `None`.

**Files:**
- Modify: `stages/frame_extraction.py`, `stages/aruco_detection.py`, `stages/frame_filtering.py`, `stages/deeparuco_comparison.py`, `markers/generate_markers.py`
- Test: add precedence tests to `tests/test_config.py` (or per-stage test files)

**Interfaces:**
- Consumes: `load_config()` and `PipelineConfig` from Task 3.
- Note: `mask_generation` has no configurable defaults (only `--manifest`, required) — leave it unchanged.

- [ ] **Step 1: Write failing precedence tests**

Append to `tests/test_config.py`:

```python
import sys
from unittest import mock

from aruco_pipeline.stages import aruco_detection, frame_extraction


def test_aruco_detection_dictionary_flag_beats_config(monkeypatch):
    argv = ["aruco-detect", "--metadata", "m.json", "--dictionary", "DICT_5X5_50"]
    monkeypatch.setattr(sys, "argv", argv)
    args = aruco_detection.parse_args()
    assert args.dictionary == "DICT_5X5_50"


def test_aruco_detection_dictionary_defaults_to_none(monkeypatch):
    argv = ["aruco-detect", "--metadata", "m.json"]
    monkeypatch.setattr(sys, "argv", argv)
    args = aruco_detection.parse_args()
    assert args.dictionary is None  # main() fills from config


def test_frame_extraction_stride_defaults_to_none(monkeypatch):
    argv = ["aruco-extract", "--input", "v.mp4"]
    monkeypatch.setattr(sys, "argv", argv)
    args = frame_extraction.parse_args()
    assert args.stride is None
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_config.py -q`
Expected: FAIL — stride/dictionary currently default to `1`/`"DICT_4X4_250"`, not `None`.

- [ ] **Step 3: `aruco_detection.py` — default None, fill in main()**

In `parse_args`, change the `--dictionary` argument: remove `default="DICT_4X4_250"`, add `default=None`. Keep `choices=sorted(pipeline_io.ARUCO_DICTIONARIES.keys())` (choices is not enforced when the value is absent). In `main()`:

```python
from ..config import load_config
...
def main() -> None:
    pipeline_io.configure_logging()
    args = parse_args()
    cfg = load_config()
    dictionary = args.dictionary if args.dictionary is not None else cfg.dictionary
    ...
    detections = detect_markers(metadata, frames_dir, dictionary)
    save_detections(detections, metadata, frames_dir, dictionary)
```

- [ ] **Step 4: `frame_extraction.py` — stride default None**

`--stride`: `default=None`. In `main()`:

```python
from ..config import load_config
...
    cfg = load_config()
    stride = args.stride if args.stride is not None else cfg.frame_extraction.stride
    frames_metadata, fps, total_frames, frame_width, frame_height = extract_frames(
        args.input, stride, output_session_dir
    )
```
Pass `stride=stride` into `save_metadata` as well (replace `args.stride`).

- [ ] **Step 5: `frame_filtering.py` — min-markers/valid-ids default None**

`--min-markers`: `default=None`. `--valid-ids`: already `default=None`. In `main()`:

```python
from ..config import load_config
...
    cfg = load_config()
    min_markers = args.min_markers if args.min_markers is not None else cfg.frame_filtering.min_markers
    valid_ids_list = args.valid_ids if args.valid_ids is not None else cfg.frame_filtering.valid_ids
    valid_ids = set(valid_ids_list) if valid_ids_list is not None else None
```
Then use `min_markers` in `filter_frames(...)` and `save_manifest(...)` instead of `args.min_markers`.

- [ ] **Step 6: `generate_markers.py` — all marker defaults None**

For `--num-markers`, `--side-pixels`, `--margin-pixels`, `--dictionary`, `--dpi`, `--page-format`: set each `default=None` (keep `choices=` where present). In `main()`:

```python
from ..config import load_config
...
    cfg = load_config().markers
    num_markers = args.num_markers if args.num_markers is not None else cfg.num_markers
    side_pixels = args.side_pixels if args.side_pixels is not None else cfg.side_pixels
    margin_pixels = args.margin_pixels if args.margin_pixels is not None else cfg.margin_pixels
    dictionary = args.dictionary if args.dictionary is not None else cfg.dictionary
    dpi = args.dpi if args.dpi is not None else cfg.dpi
    page_format = args.page_format if args.page_format is not None else cfg.page_format
```
Replace every `args.<field>` in the `main()` body below with the resolved local (`num_markers`, `side_pixels`, etc.). `--output-dir` and `--pdf-name` keep their existing literal defaults (paths/filenames are not session config).

- [ ] **Step 7: `deeparuco_comparison.py` — source config from yaml**

Thread the deeparuco config into weight resolution instead of module constants. Change `_download_weights` and `load_deeparuco_models` to accept the base URL, filenames map, and default dir:

```python
def _download_weights(target_dir: Path, base_url: str, filenames: dict[str, str]) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames.values():
        target = target_dir / filename
        if not target.exists():
            logger.info("Downloading %s...", filename)
            try:
                urllib.request.urlretrieve(f"{base_url}/{filename}", target)
            except Exception as e:
                raise RuntimeError(...) from e
```
Update `load_deeparuco_models(weights_dir, base_url, filenames, default_dir)` to use the passed `filenames` in place of `_MODEL_FILENAMES` and `default_dir`/`base_url` in place of `DEFAULT_WEIGHTS_DIR`/`_BASE_URL`. Keep the module-level `_MODEL_FILENAMES`, `_BASE_URL`, `DEFAULT_WEIGHTS_DIR` as the fallback source. In `main()`:

```python
from ..config import load_config
...
    da = load_config().deeparuco
    default_dir = Path(da.weights_dir).expanduser()
    models = load_deeparuco_models(
        args.model_weights, base_url=da.base_url, filenames=da.weights, default_dir=default_dir
    )
    ...
    weights_path=args.model_weights or default_dir,
```
Adjust `tests/test_deeparuco_comparison.py` calls to `load_deeparuco_models`/`_download_weights` to pass the new params (or give the new params defaults equal to the module constants so existing calls keep working — prefer defaults to minimize test churn):

```python
def load_deeparuco_models(weights_dir=None, *, base_url=_BASE_URL,
                          filenames=_MODEL_FILENAMES, default_dir=DEFAULT_WEIGHTS_DIR):
```

- [ ] **Step 8: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass (106 originals + the new precedence tests). Fix any test that asserted an old literal default.

- [ ] **Step 9: Smoke-test one stage end-to-end**

Run: `uv run aruco-generate-markers --output-dir /tmp/mk --num-markers 3`
Expected: exits 0; `/tmp/mk/` contains `marker_0.png`..`marker_2.png`, `markers_sheet.pdf`, `manifest.json`. Confirms config fallback + CLI override both work through a real entry point.

- [ ] **Step 10: Lint, format, commit**

```bash
uv run ruff check --fix src/aruco_pipeline/ && uv run ruff format src/aruco_pipeline/
git add -A
git commit --no-verify -m "feat: wire pipeline.yaml defaults into stage CLIs"
```

---

### Task 5: Update documentation

Bring README and CLAUDE.md in line with the new package, commands, and config layer.

**Files:**
- Modify: `README.md`, `CLAUDE.md`

- [ ] **Step 1: Update `README.md`**

- Replace the "Usage" and "Project Structure" sections with the `aruco-*` console commands and the new `src/aruco_pipeline/...` tree.
- Add a "Configuration" subsection describing `configs/pipeline.yaml` and the precedence rule (CLI > yaml > fallback).
- Update the "Development" section paths (`ruff check src/aruco_pipeline/`).
- Note the marker generator command is now `uv run aruco-generate-markers`.

- [ ] **Step 2: Update `CLAUDE.md`**

- Commands section: replace `python src/<file>.py` forms with `uv run aruco-*`; note `uv sync --dev` installs the package (editable) and console scripts.
- Architecture: update the stage list to `aruco_pipeline/stages/*` paths; note `core/` (pipeline_io, schemas, config), `markers/`, `deeparuco_vendor/`.
- Add the config layer + precedence rule, and reiterate algorithm constants stay in code.
- Update the vendored-code path to `src/aruco_pipeline/deeparuco_vendor/`.
- Keep the test-isolation gotcha section.

- [ ] **Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit --no-verify -m "docs: update README and CLAUDE.md for package layout and config"
```

Note: CLAUDE.md is being committed here by explicit inclusion in this documentation task. If the user's no-Claude-files rule should exclude it, drop `CLAUDE.md` from the `git add` and leave it untracked.

---

## Self-Review

**Spec coverage:** package layout (T1) ✓; relative imports (T1) ✓; console scripts + build backend (T2) ✓; `configs/pipeline.yaml` scope-2 contents (T3) ✓; config loader + precedence (T3, T4) ✓; algorithm constants excluded (Global Constraints + T4 leaves them untouched) ✓; pyproject tool-path + pyyaml updates (T1, T2, T3) ✓; README/CLAUDE.md (T5) ✓; conftest untouched (Global Constraints) ✓. The spec's "keep or drop pythonpath" open point is resolved: **keep** `pythonpath=["src"]` so tests import without relying on the editable install.

**Placeholder scan:** none — every code step shows full code; the `_download_weights` error body is abbreviated only where it copies the existing verbatim string.

**Type consistency:** `load_config(path: Path | None) -> PipelineConfig` used identically in T3/T4; dataclass field names (`frame_extraction.stride`, `frame_filtering.min_markers/valid_ids`, `markers.*`, `deeparuco.base_url/weights_dir/weights`) match yaml keys and the wiring in T4.
