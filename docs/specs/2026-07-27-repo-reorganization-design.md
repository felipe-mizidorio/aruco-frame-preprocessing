# Repo Reorganization — Design

**Date:** 2026-07-27
**Status:** Approved, pending implementation plan

## Goal

Reorganize the repository for a cleaner, standard structure:

1. Convert the flat `src/` scripts into a proper installable Python package with
   grouped subpackages.
2. Ship each pipeline stage as a named **console script** (`aruco-*`) instead of
   running files by path.
3. Introduce `configs/pipeline.yaml` to centralize **session defaults** (not
   algorithm constants), loaded through a small config layer.

Already done by the user, out of scope here: removed the empty `notebooks/`
folder; created the (currently empty) `configs/` folder this design fills.

## Non-goals

- Externalizing algorithm constants (`BLUR_MAD_K`, `_MAD_TO_SIGMA`,
  `HULL_MARGIN_MARKER_SIDES`, `MIN_MARKERS_FOR_HULL`, blob-detector params, etc.).
  These are tuned/derived values, not settings. They stay in code to preserve the
  knob-free, domestic-use design and its "degrades toward under-rejection, never
  over-rejection" guarantees.
- Any change to `deeparuco_vendor/` internals or its provenance record.
- Any change to the `conftest.py` cv2/ultralytics isolation fixture.

## Target package layout

```
src/aruco_pipeline/
  __init__.py                 # logging NullHandler (moved from src/__init__.py)
  config.py                   # loads configs/pipeline.yaml into a dataclass
  core/
    __init__.py
    pipeline_io.py
    schemas.py
  stages/
    __init__.py
    frame_extraction.py
    aruco_detection.py
    frame_filtering.py
    mask_generation.py
    deeparuco_comparison.py
  markers/
    __init__.py
    generate_markers.py       # standalone utility, not a video-pipeline stage
  deeparuco_vendor/
    __init__.py
    aruco.py
    heatmaps.py
    losses.py
    utils.py
    PROVENANCE.md
```

Rationale: `core/` = shared infrastructure imported by everything; `stages/` = the
five chained video-pipeline steps; `markers/` = the separate marker-generation
utility; `deeparuco_vendor/` = untouched vendored third-party code (keeps its
name so PROVENANCE references stay valid).

## Imports

All internal imports move from flat to relative, package-qualified:

| Before | After |
|---|---|
| `import pipeline_io` | `from ..core import pipeline_io` |
| `from schemas import X` | `from ..core.schemas import X` |
| `from deeparuco_vendor.aruco import find_id` | `from ..deeparuco_vendor.aruco import find_id` |

Tests move from bare imports to package imports, e.g.
`from aruco_pipeline.core import pipeline_io`,
`from aruco_pipeline.stages import mask_generation`.

## Invocation — console scripts

Add a build backend so `uv sync` installs the package editable, and declare one
entry point per stage.

`pyproject.toml` additions:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/aruco_pipeline"]

[project.scripts]
aruco-extract           = "aruco_pipeline.stages.frame_extraction:main"
aruco-detect            = "aruco_pipeline.stages.aruco_detection:main"
aruco-filter            = "aruco_pipeline.stages.frame_filtering:main"
aruco-mask              = "aruco_pipeline.stages.mask_generation:main"
aruco-compare           = "aruco_pipeline.stages.deeparuco_comparison:main"
aruco-generate-markers  = "aruco_pipeline.markers.generate_markers:main"
```

Command mapping:

| Old | New |
|---|---|
| `python src/frame_extraction.py --input v.mp4` | `uv run aruco-extract --input v.mp4` |
| `python src/aruco_detection.py --metadata metadata.json` | `uv run aruco-detect --metadata metadata.json` |
| `python src/frame_filtering.py --detections detections.json` | `uv run aruco-filter --detections detections.json` |
| `python src/mask_generation.py --manifest manifest.json` | `uv run aruco-mask --manifest manifest.json` |
| `python src/deeparuco_comparison.py --detections detections.json` | `uv run aruco-compare --detections detections.json` |
| `python src/generate_markers.py` | `uv run aruco-generate-markers` |

## Configuration — `configs/pipeline.yaml`

Scope: **session defaults only.** Contents:

```yaml
dictionary: DICT_4X4_250          # detection default (aruco-detect)
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

These values mirror today's argparse defaults and module constants in
`deeparuco_comparison.py` (`_MODEL_FILENAMES`, `_BASE_URL`, `DEFAULT_WEIGHTS_DIR`).

### Config layer (`aruco_pipeline/config.py`)

- Loads `configs/pipeline.yaml` once into a typed dataclass (mirrors the
  yaml shape). Locates the file relative to the repo root; path overridable via an
  env var or `--config` flag (decide during planning — default location is
  `configs/pipeline.yaml`).
- Missing file or missing key falls back to a hardcoded default baked into the
  dataclass, so the pipeline still runs with no yaml present.

### Precedence

For every configurable value: **CLI flag > yaml > hardcoded fallback.**

Implementation pattern: each stage's argparse `default` becomes `None`. After
parsing, `None` values are filled from the loaded config. This keeps an explicit
flag authoritative while yaml supplies the baseline.

Adds a `pyyaml` runtime dependency.

## Other file/config updates

- `pyproject.toml`: add `pyyaml` to `dependencies`; update
  `[tool.ruff] exclude` and per-file-ignores to
  `src/aruco_pipeline/deeparuco_vendor/*`; `[tool.pyright] include` to
  `src/aruco_pipeline`. Keep or drop `pythonpath = ["src"]` — with an editable
  install pytest resolves `aruco_pipeline` without it; keep only if it simplifies
  test collection (decide during planning).
- `README.md`: rewrite install/usage/structure sections for the new commands and
  layout; document `configs/pipeline.yaml`.
- `CLAUDE.md`: update commands, architecture paths, the session-dir/schemas
  conventions (unchanged in behavior, new file paths), and add the config layer +
  precedence rule.
- `conftest.py`: cv2/ultralytics isolation fixture unchanged.

## Testing

- All existing tests keep passing after the import rewrite (behavior unchanged).
- Add tests for `config.py`: yaml load into dataclass, missing-file fallback,
  missing-key fallback, and CLI-flag-over-yaml precedence for at least one stage.
- Verify each console script is invocable after `uv sync` (entry point resolves
  and `--help` works).

## Migration order (high level; detailed plan follows separately)

1. Create package skeleton + move files with `git mv` (preserve history).
2. Rewrite internal imports and test imports.
3. Add build-system + `[project.scripts]`; `uv sync`; confirm scripts resolve.
4. Add `configs/pipeline.yaml` + `config.py` + wire precedence into each stage.
5. Update `pyproject.toml` tool paths, README, CLAUDE.md.
6. Full test run + manual smoke of one stage end-to-end.
