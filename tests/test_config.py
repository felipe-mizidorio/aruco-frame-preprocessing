import sys
from pathlib import Path

from aruco_pipeline.config import PipelineConfig, load_config
from aruco_pipeline.stages import aruco_detection, frame_extraction


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
