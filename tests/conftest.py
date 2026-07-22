"""Shared test fixtures.

Contains the guard against cross-test contamination of the global `cv2` module.
"""

import cv2
import pytest

# Functions ultralytics replaces on the global cv2 module. See _restore_cv2_patches.
_PATCHED_CV2_FUNCTIONS = ("imread", "imwrite", "imshow")


@pytest.fixture(autouse=True)
def _restore_cv2_patches():
    """Undo third-party monkey-patching of `cv2` between tests.

    Mechanism this guards against
    -----------------------------
    `ultralytics/utils/__init__.py` runs, at import time and only on Windows::

        cv2.imread, cv2.imwrite, cv2.imshow = imread, imwrite, imshow

    to add non-ASCII path support. Its replacement `imread` ends with::

        return im[..., None] if im is not None and im.ndim == 2 else im

    so a grayscale read returns shape ``(H, W, 1)`` rather than ``(H, W)``.
    `src/mask_generation.py` does ``height, width = img.shape`` and raises
    ``ValueError: too many values to unpack (expected 2)``.

    The leak reaches the test suite through
    `tests/test_deeparuco_comparison.py`, which calls `load_deeparuco_models`.
    That function imports `ultralytics` *before* it validates the model paths,
    so even the test asserting a missing-file RuntimeError imports it for real.

    Why the existing `mock.patch.dict("sys.modules", ...)` does not contain it:
    ultralytics mutates attributes on the already-imported `cv2` module object.
    Restoring `sys.modules` afterwards leaves those attributes replaced, so the
    patch persists for the remainder of the process and every later test sees
    it. That made the suite order-dependent: `test_mask_generation.py` passed
    alone and failed when it ran after `test_deeparuco_comparison.py`.

    This fixture snapshots the affected attributes before each test and puts
    them back afterwards, so the pollution cannot cross a test boundary
    whichever test triggers the import.

    This is a test-isolation fix only. It does not change ultralytics'
    behaviour, and production is unaffected: `mask_generation.py` and
    `deeparuco_comparison.py` are separate CLI entry points that never share a
    process.
    """
    original = {name: getattr(cv2, name) for name in _PATCHED_CV2_FUNCTIONS}
    try:
        yield
    finally:
        for name, func in original.items():
            setattr(cv2, name, func)
