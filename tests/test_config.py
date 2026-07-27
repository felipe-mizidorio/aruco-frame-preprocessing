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
