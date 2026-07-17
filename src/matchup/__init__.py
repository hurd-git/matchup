"""Traditional image matching backed by a single-threaded C++ core."""

from ._api import PreparedImage, picture_match, picture_match_64, resize_image


__version__ = "0.1.0"
__all__ = [
    "PreparedImage",
    "picture_match",
    "picture_match_64",
    "resize_image",
]
