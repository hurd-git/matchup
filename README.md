# matchup

`matchup` is a traditional computer-vision image matcher implemented in C++
with a small Python API. It uses OpenCV SIFT, geometric verification, edge
agreement and appearance checks. It does not use neural-network models.

The current algorithm uses a fixed 64 x 64 working unit, bidirectional SIFT
geometry, edge and appearance evidence, and aligned central-content
confirmation. See [MATCHING_IMPLEMENTATION.md](MATCHING_IMPLEMENTATION.md) for
the current design and [MATCHING_FLOWCHART.md](MATCHING_FLOWCHART.md) for the
complete Mermaid flowcharts.

The first release supports 64-bit Windows and CPython 3.10 through 3.14.

## Installation

```powershell
python -m pip install matchup
```

The Windows wheel contains the native extension and required OpenCV DLLs.
Users do not need to install OpenCV, CMake or Visual Studio.

## Usage

```python
import matchup

score = matchup.picture_match(image_a, image_b)
```

Inputs are numeric NumPy arrays in grayscale, RGB or RGBA layout. The returned
score is a symmetric `float` in the inclusive range `[0.0, 1.0]`.

When spatial sizes differ, the image with the larger short edge is resized to
the other image's exact width and height with bicubic interpolation. The C++
matching core always receives equal-size arrays.

For repeated matching, prepare an image once:

```python
reference = matchup.PreparedImage(image_a)

score_hot_cold = reference.match(image_b)
other = matchup.PreparedImage(image_b)
score_hot_hot = reference.match(other)
```

Equal-size prepared images reuse all preprocessing. A cross-size prepared
match intentionally falls back to a cold resize and does not create cached
pair-specific variants.

Internally, images whose short edge is above 64 are reduced proportionally to
a 64-pixel short edge. Smaller images are not enlarged. Long images are split
into corresponding 64 x 64 windows and their evidence is aggregated without
allowing one accidental high tile to dominate the result.

The fixed working-unit API is also available:

```python
score = matchup.picture_match_64(image_64_a, image_64_b)
resized = matchup.resize_image(image, target_height=80, target_width=80)
```

## Development

Prerequisites:

- Windows x64
- Visual Studio 2022 with the C++ desktop toolchain
- Python 3.10 or newer
- uv

Create the environment and build a wheel:

```powershell
uv sync --no-install-project --group dev
uv build --wheel
```

CMake downloads the pinned OpenCV 4.12.0 source archive. To reuse an existing
OpenCV source tree during local development:

```powershell
$env:CMAKE_ARGS = "-DMATCHUP_OPENCV_SOURCE_DIR=C:\path\to\opencv\sources"
uv build --wheel
```

Install and test the wheel in a clean environment:

```powershell
uv venv .wheel-test
uv pip install --python .wheel-test\Scripts\python.exe dist\matchup-*.whl
.wheel-test\Scripts\python.exe -m pytest tests
```

OpenCV is configured without an internal parallel framework, and the native
module verifies single-thread mode when imported.

## License

`matchup` is licensed under the MIT License. See `NOTICE` for third-party
components and their licenses.
