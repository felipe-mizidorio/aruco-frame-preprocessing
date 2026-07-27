from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

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
    frame_extraction: FrameExtractionConfig = field(
        default_factory=FrameExtractionConfig
    )
    frame_filtering: FrameFilteringConfig = field(default_factory=FrameFilteringConfig)
    markers: MarkersConfig = field(default_factory=MarkersConfig)
    deeparuco: DeepArucoConfig = field(default_factory=DeepArucoConfig)


def _build(cls: type, data: Any):
    """Instantiate dataclass `cls`, filling only keys it declares; nested
    dataclass fields recurse, everything else falls back to the field default.
    Unknown keys in `data` are ignored so an over-specified yaml never errors."""
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


def load_config(path: Path | None = None) -> PipelineConfig:
    path = path or _DEFAULT_CONFIG_PATH
    if not path.exists():
        return PipelineConfig()
    data = yaml.safe_load(path.read_text()) or {}
    return _build(PipelineConfig, data)
