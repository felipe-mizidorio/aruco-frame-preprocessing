"""Regression tests for cross-test contamination of the global `cv2` module.

See tests/conftest.py for the mechanism these guard against.
"""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def test_imread_grayscale_returns_two_dimensional_array(tmp_path) -> None:
    """`cv2.imread` with IMREAD_GRAYSCALE must yield a 2-D array.

    `src/mask_generation.py` unpacks `height, width = img.shape`, so a third
    dimension breaks it. Importing `ultralytics` anywhere in the process
    replaces `cv2.imread` with a version that forces 3 dimensions, which is
    what this test detects.
    """
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), np.zeros((8, 6, 3), dtype=np.uint8))

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    assert img.ndim == 2, f"expected 2-D grayscale array, got shape {img.shape}"
    assert img.shape == (8, 6)


def test_cv2_imread_is_not_monkeypatched_by_a_third_party() -> None:
    """`cv2.imread` must still be OpenCV's own builtin, not a replacement.

    Checks the cause rather than the symptom, so the failure message points at
    whichever library did the patching.
    """
    module = getattr(cv2.imread, "__module__", None)

    assert module is None, (
        f"cv2.imread has been replaced by {module}.{cv2.imread.__name__}; "
        "a third-party import leaked into the global cv2 module"
    )


def test_mask_generation_passes_after_deeparuco_comparison() -> None:
    """The two modules must pass in the order that used to fail.

    This is the original bug: `test_mask_generation.py` passed alone and failed
    when it ran after `test_deeparuco_comparison.py`, because the latter
    imports ultralytics for real.

    Runs in a subprocess so the ordering is exercised in a fresh interpreter.
    The cheap in-process guards above sort before `test_deeparuco_comparison`
    alphabetically, so on their own they would not catch a regression here.
    """
    tests_dir = Path(__file__).parent
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:randomly",
            str(tests_dir / "stages" / "test_deeparuco_comparison.py"),
            str(tests_dir / "stages" / "test_mask_generation.py"),
        ],
        capture_output=True,
        text=True,
        cwd=tests_dir.parent,
    )

    assert result.returncode == 0, (
        "cv2 isolation regression: test_mask_generation fails when it runs "
        f"after test_deeparuco_comparison.\n{result.stdout[-2000:]}"
    )
