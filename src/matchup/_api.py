from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np


if sys.platform != "win32":
    raise ImportError("matchup currently supports Windows only")

_PACKAGE_DIR = Path(__file__).resolve().parent
_DLL_DIRECTORY_HANDLES: list[object] = []
if hasattr(os, "add_dll_directory"):
    _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(_PACKAGE_DIR))

from . import _native  # noqa: E402


WORKING_IMAGE_SIZE = 64


def _as_uint8_image(image: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError(f"{name} must contain numeric pixel values")
    if image.ndim not in (2, 3):
        raise ValueError(f"{name} must be a grayscale, RGB, or RGBA image")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(f"{name} must be a grayscale, RGB, or RGBA image")

    if np.issubdtype(image.dtype, np.floating):
        pixels = (np.clip(image, 0.0, 1.0).astype(np.float32) * 255).astype(
            np.uint8
        )
    elif image.dtype == np.uint8:
        pixels = image
    else:
        maximum = np.iinfo(image.dtype).max
        pixels = np.clip(
            image.astype(np.float32) * (255.0 / maximum), 0, 255
        ).astype(np.uint8)
    return np.ascontiguousarray(pixels)


class PreparedImage:
    """Cache one fixed-size preprocessing result for repeated matching."""

    def __init__(self, image: np.ndarray) -> None:
        self._pixels = _as_uint8_image(image, "image").copy()
        self._shape = self._pixels.shape
        self._prepared = _native.PreparedImage(self._pixels)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def match(self, other: PreparedImage | np.ndarray) -> float:
        """Use cached features for equal sizes and cold-resize unequal sizes."""
        if isinstance(other, PreparedImage):
            if self._shape != other._shape:
                return picture_match(self._pixels, other._pixels)
            return float(self._prepared.match(other._prepared))

        pixels = _as_uint8_image(other, "other")
        if self._shape != pixels.shape:
            return picture_match(self._pixels, pixels)
        return float(self._prepared.match(pixels))


def picture_match(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """Return a symmetric similarity score in the inclusive range [0, 1]."""
    pixels_a = _as_uint8_image(image_a, "image_a")
    pixels_b = _as_uint8_image(image_b, "image_b")
    if pixels_a.shape[2:] != pixels_b.shape[2:]:
        raise ValueError(
            "Images must use the same channel layout, got "
            f"{pixels_a.shape[2:]} and {pixels_b.shape[2:]}"
        )
    pixels_a, pixels_b = _resize_to_common_shape(pixels_a, pixels_b)
    return float(_native.picture_match(pixels_a, pixels_b))


def picture_match_64(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """Compare two images whose spatial shape is exactly 64 x 64."""
    pixels_a = _as_uint8_image(image_a, "image_a")
    pixels_b = _as_uint8_image(image_b, "image_b")
    if pixels_a.shape != pixels_b.shape:
        raise ValueError(
            f"Images must have identical shapes, got {pixels_a.shape} and "
            f"{pixels_b.shape}"
        )
    return float(_native.picture_match_64(pixels_a, pixels_b))


def resize_image(
    image: np.ndarray, target_height: int, target_width: int
) -> np.ndarray:
    """Resize an image with native OpenCV bicubic interpolation."""
    pixels = _as_uint8_image(image, "image")
    if target_height <= 0 or target_width <= 0:
        raise ValueError("target dimensions must be positive")
    return np.asarray(_native.resize_image(pixels, target_height, target_width))


def _resize_to_common_shape(
    pixels_a: np.ndarray, pixels_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if pixels_a.shape[:2] == pixels_b.shape[:2]:
        return pixels_a, pixels_b
    if _spatial_size_key(pixels_b) < _spatial_size_key(pixels_a):
        pixels_a = resize_image(pixels_a, *pixels_b.shape[:2])
    else:
        pixels_b = resize_image(pixels_b, *pixels_a.shape[:2])
    return pixels_a, pixels_b


def _spatial_size_key(image: np.ndarray) -> tuple[int, int, int, int, int]:
    height, width = image.shape[:2]
    return (
        min(height, width),
        height * width,
        max(height, width),
        height,
        width,
    )
