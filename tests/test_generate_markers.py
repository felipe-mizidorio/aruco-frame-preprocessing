import cv2
import numpy as np

from generate_markers import generate_marker, save_markers

# --- generate_marker ---


def test_generate_marker_returns_correct_shape() -> None:
    image = generate_marker(cv2.aruco.DICT_4X4_250, marker_id=0, side_pixels=100)
    assert image.shape == (100, 100)


def test_generate_marker_returns_uint8() -> None:
    image = generate_marker(cv2.aruco.DICT_4X4_250, marker_id=0, side_pixels=100)
    assert image.dtype == np.uint8


# --- save_markers ---


def test_save_markers_creates_correct_number_of_files(tmp_path) -> None:
    save_markers(
        num_markers=3,
        side_pixels=50,
        margin_pixels=10,
        output_dir=tmp_path,
        dictionary_name="DICT_4X4_50",
        dpi=72,
    )
    png_files = list(tmp_path.glob("*.png"))
    assert len(png_files) == 3


def test_save_markers_files_named_correctly(tmp_path) -> None:
    save_markers(
        num_markers=3,
        side_pixels=50,
        margin_pixels=10,
        output_dir=tmp_path,
        dictionary_name="DICT_4X4_50",
        dpi=72,
    )
    for i in range(3):
        assert (tmp_path / f"marker_{i}.png").exists()
