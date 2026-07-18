# Matchup image-matching implementation

This document describes the current `matchup` 1.0.7 traditional computer-
vision pipeline. It does not use neural-network models, file names, or learned
sample labels. The public score is an engineering similarity in `[0.0, 1.0]`,
not a probability and not a built-in classification threshold.

The complete Mermaid diagrams are in
[MATCHING_FLOWCHART.md](MATCHING_FLOWCHART.md).

## Public API

```python
matchup.picture_match(image_a, image_b)
matchup.picture_match_64(image_64_a, image_64_b)
matchup.resize_image(image, target_height, target_width)

prepared = matchup.PreparedImage(image_a)
prepared.match(image_b)
prepared.match(matchup.PreparedImage(image_b))
```

Inputs are numeric NumPy arrays in grayscale, RGB, or RGBA layout. Both inputs
must use the same channel layout. Python converts them to contiguous `uint8`
arrays before entering the C++ core.

`picture_match` is cold-cold. `PreparedImage.match(ndarray)` is hot-cold for an
equal-size candidate, and two prepared objects use the hot-hot path. Cross-size
prepared matching deliberately falls back to the public cold resize path and
does not maintain pair-specific cached variants.

## Size normalization and tiling

When spatial sizes differ, Python chooses the smaller image with this stable
key:

```text
(short edge, pixel area, long edge, height, width)
```

The image with the larger key is resized, without preserving its aspect ratio,
to the other image's exact width and height. All resizing uses bicubic
interpolation. The native core therefore always receives equal spatial sizes.

The C++ core then computes:

```text
scale = min(1, 64 / short edge)
```

- A short edge above 64 is reduced to 64 while preserving aspect ratio.
- A short edge at or below 64 is never enlarged.
- If both edges fit within 64, content is centered on a `64 x 64` canvas.
- Padding uses each image's per-channel border median instead of a common black
  or white border.
- If the long edge exceeds 64, `ceil(long edge / 64)` windows are distributed
  uniformly from the first to the last valid start position. Corresponding
  windows are compared in pairs.

## Prepared 64 x 64 unit

Each working tile owns or caches:

- Original pixels and a 16-byte BLAKE2b content digest.
- A grayscale image.
- Gaussian-blurred Canny edges.
- Edge rows encoded as 64-bit masks for fast translation search.
- Sobel-gradient SIFT features for three spatial scopes.
- Raw-image SIFT features, prepared lazily only when ambiguous single-tile
  matching needs them.

The content digest only establishes deterministic A/B ordering. It is not an
image feature and does not contribute to the score.

## Outline branch

The outline branch searches translations from `-5` through `+5` pixels in both
axes. It calculates the best edge F1 in both directions and averages them.

The central 60% of the image also supplies a per-channel mean absolute
appearance error:

```text
appearance = exp(-normalized_error / 0.08)
outline_score = bidirectional_edge_f1 * appearance
```

This keeps a shared silhouette from scoring highly by itself.

## Gradient fingerprint branch

The primary SIFT input is normalized Sobel gradient magnitude rather than raw
color. This suppresses absolute brightness and color while retaining local
shape, corners, and texture direction.

SIFT is configured for the 64-pixel scale:

```text
nfeatures = 500
contrastThreshold = 0.006
edgeThreshold = 8
sigma = 0.8
```

Three feature scopes are evaluated:

1. Strict: exclude the outer 20%.
2. Expanded: exclude the outer 15%.
3. Anchored full image: extract everywhere, but require at least one endpoint
   of each accepted match to lie in the center.

## Bidirectional geometric verification

Each scope matches A to B and B to A:

1. L2 nearest-neighbor descriptor search with two neighbors.
2. Lowe ratio `0.8`, represented as a squared-distance ratio of `0.64`.
3. Duplicate keypoint removal within `1.5px` in either source or destination.
4. Partial-affine RANSAC with a `3px` reprojection threshold.
5. Accepted scale range from `0.5` through `2.0`.
6. At least four inliers and evidence distributed over a `4 x 4` grid.

The two directional scores use a harmonic mean. Additional confidence checks
cover inverse-transform consistency, reciprocal scale, rotation, inlier
support, aligned dense error, aligned edge containment, and central inliers.

The partial-affine model supports translation, rotation, and uniform scaling.
It does not promise arbitrary perspective, nonlinear deformation, or mirror
reflection handling.

## Three-scope fusion

```text
strict = strict-scope score
secondary = max(expanded score, anchored score)

if strict == 0:
    score = secondary * 0.5
else:
    score = strict * 0.7 + max(strict, secondary) * 0.3
```

The strict center remains the primary evidence, but a single center miss does
not force the fingerprint to zero.

## Aligned central-content confirmation

For single-tile images, complete bidirectional RANSAC geometry is reused to
compare dense content in the central `32 x 32` region. No extra SIFT pass is
needed. Both alignment directions measure:

- Exact central edge F1.
- Per-channel mean absolute pixel error.
- The number of valid central edge pixels.

Current thresholds:

- Fewer than 32 central edge pixels: do not penalize.
- Edge F1 below `0.30`: begin penalizing; at `0.18`, reach the maximum.
- Normalized color error above `0.095`: begin penalizing; at `0.13`, reach the
  maximum.
- Maximum penalty: multiply the fingerprint score by `0.20`.
- Edge and color mismatches take the stronger result and are not added.

When central edge F1 is at least `0.50`, the foreground structure is considered
complete enough that color error is ignored. This preserves the same icon over
colorful or textured backgrounds. It does not make partial crops high-scoring,
because their complete structural evidence is missing.

The local-content factor is applied after the three feature scopes are fused.
A wider SIFT scope therefore cannot use a shared border or UI template to
bypass negative evidence already established by verified central geometry.

## Conditional raw confirmation

For a single tile, a gradient fingerprint between zero and `0.8` that still
beats the outline score receives a second, raw-image fingerprint confirmation:

```text
fingerprint *= sqrt(raw_fingerprint)
```

This confirmation can only lower the result. Obvious negatives and
self-sufficient high-confidence positives avoid the extra work. Multi-tile
images skip this step because independent scene backgrounds would repeatedly
penalize otherwise corresponding long-image structure.

## Tile aggregation

One to three tile scores use their arithmetic mean. For four or more tiles:

```text
effective_count = sum(score)^2 / sum(score^2)
```

When evidence is isolated to one or two tiles, aggregation remains close to the
arithmetic mean. When useful evidence is distributed over most tiles, the
result blends toward `sum(score^2) / sum(score)`. This discounts isolated
failed tiles without letting one accidental local match dominate a long image.

## Determinism and threading

- BLAKE2b fixes internal input order.
- OpenCV's RANSAC seed is reset before every affine estimate.
- Matching is symmetric under input exchange.
- The native module calls `cv::setNumThreads(1)` and verifies the result.
- OpenCV is built without an internal parallel framework.
- pybind11 releases the Python GIL while native matching runs.

## OpenCV optimization profile

### Intel IPP decision

The published 1.0.7 build configuration requested Intel IPP, but the actual
OpenCV feature set depended on whether OpenCV's IPPICV archive was downloaded
during the build. This produced two useful comparison builds on the same
machine:

- A local wheel without IPP, approximately 6.1 MB.
- A GitHub Release wheel with IPP, approximately 19.0 MB.

Both builds used CPython 3.14, OpenCV single-threaded execution, the same
matcher source, and the same 291-image sample set. Each measurement was run
three times in multiple fresh processes, including reversed test order. The
benchmark covered 1,569 equal-size pairs and 337 unequal-size pairs.

| Operation | Without IPP | With IPP | IPP slowdown |
| --- | ---: | ---: | ---: |
| Prepare 291 images | 1.20 s | 2.15 s | about 79% |
| Cold-cold, 1,569 pairs | 10.42 s | 18.66 s | about 79% |
| Hot-cold, 1,569 pairs | 6.32 s | 11.12 s | about 76% |
| Hot-hot, 1,569 pairs | 1.97 s | 3.36 s | about 70% |
| Unequal-size cold-cold, 337 pairs | 1.68 s | 2.90 s | about 72% |

IPP also changed some scores. Among the equal-size pairs, 83 of 1,569 scores
changed, with mean absolute difference `0.00026486` and maximum difference
`0.0889639`. Among the unequal-size pairs, 163 of 337 changed, with mean
absolute difference `0.00011855` and maximum difference `0.0159530`. Each
individual build remained exactly repeatable across repeated runs.

For this matcher, IPP is therefore a negative optimization: its setup and
dispatch costs dominate the small `64 x 64` working units, it increases the
wheel by about 13 MB, and it makes the score-producing implementation depend
on build-time download success. The source build now explicitly sets
`WITH_IPP=OFF` and `BUILD_IPP_IW=OFF`. This is a speed, package-size, and
reproducibility decision, not a threading decision. The already-published
1.0.7 artifacts remain unchanged historical artifacts.

### Other OpenCV optimizations

An `ON` value in OpenCV's CMake cache does not by itself mean that a facility
is linked into the matcher or runs during matching. The relevant settings are:

| Facility | Current state | Assessment |
| --- | --- | --- |
| OpenCV CPU intrinsics | Enabled | Used in compiled OpenCV code. The CPU-dispatch benchmark confirms that OpenCV's own SIMD is beneficial; unlike IPP, this has no large external runtime library. |
| CPU dispatch | SSE4.1, SSE4.2, AVX, FP16, and AVX2 over an SSE3 baseline | Keep the full list. Both SSE3-only and an SSE4.1-plus-AVX2 subset were slower. FP16 currently compiles no dispatched source files, but removing the low-coverage entries together saved almost no space and reduced speed. |
| OpenCV tracing (`CV_TRACE`) | Explicitly disabled | Disabling it saved about 21 KB and produced mixed changes within benchmark noise. It is disabled for configuration hygiene, not as a claimed speed optimization. |
| OpenCL | Explicitly disabled | No current runtime cost. The matcher uses `cv::Mat`, not `cv::UMat`. |
| TBB, OpenMP, pthreads, and ITT | Explicitly disabled | No hidden OpenCV worker pool. Runtime is additionally fixed to one OpenCV thread. |
| DirectML, DirectX, ADE, FlatBuffers, OBSENSOR, and Win32 UI | Explicitly disabled | Their consuming modules or interoperability APIs are outside the matcher. Disabling them removes configuration probing and unused DirectX interoperability code without changing any measured score. |
| LAPACK | Explicitly disabled | The previous build did not find a LAPACK implementation, and the matcher does not use the affected solver paths. |
| zlib | Built and linked by `opencv_core` | The matcher does not intentionally use OpenCV FileStorage. Removing it needs a separate build and ABI/functionality check before it can be considered safe. |
| Precompiled headers | Enabled | Affects compilation time only, not matching runtime. |

### CPU dispatch and tracing results

The remaining runtime candidates were tested without IPP on the same 291
images. To control for thermal and run-order variation, the full and reduced
dispatch builds were run in alternating order across three fresh processes.
Each process measured all five benchmark flows once; the table uses the median
of those independent processes.

| Operation | Full dispatch | SSE4.1 + AVX2 | Reduced build slowdown |
| --- | ---: | ---: | ---: |
| Prepare 291 images | 1.20 s | 1.26 s | about 4.7% |
| Cold-cold, 1,569 pairs | 11.74 s | 15.26 s | about 30.0% |
| Hot-cold, 1,569 pairs | 6.85 s | 7.82 s | about 14.1% |
| Hot-hot, 1,569 pairs | 2.55 s | 2.84 s | about 11.6% |
| Unequal-size cold-cold, 337 pairs | 1.90 s | 2.28 s | about 20.0% |

The reduced wheel was only 11 KB smaller than the 6.1 MB full-dispatch wheel,
and all 1,906 measured scores were exactly equal. An SSE3-only wheel was about
1.14 MB smaller, but was approximately 34% to 94% slower across the five
flows. It also changed eight equal-size scores (maximum absolute change
`0.0286485`) and three unequal-size scores (maximum absolute change
`0.0000201`). The full OpenCV CPU dispatch should therefore remain enabled.

Disabling `CV_TRACE` was also tested in three alternating fresh processes. It
made preparation about 1.3% faster and hot-hot matching about 4.5% faster, but
made cold-cold, hot-cold, and unequal-size matching about 1.2% to 1.9% slower.
These mixed differences are within the observed machine variation. The wheel
was 21 KB smaller and every score was exactly equal. Disabling tracing is a
reasonable build-cleanup choice, but the benchmark does not support claiming
a runtime performance gain.

The source configuration also explicitly disables unused codec builders,
binding generators, DirectML, DirectX interoperability, ADE, FlatBuffers,
LAPACK, OBSENSOR, Win32 UI, and OpenCL D3D11 interoperability. A final wheel
with these settings passed the complete test suite and produced exactly the
same 1,906 benchmark scores as the no-IPP, full-dispatch reference. Only
`opencv_core` changed code size, shrinking its executable section by about
400 bytes as unused DirectX interoperability code was removed. zlib remains
enabled because it is an actual `opencv_core` link dependency rather than an
unused cache default.

## Validation

```powershell
uv run pytest
uv run python benchmarks/benchmark_matchup.py
uv build --wheel
```

Tests cover type and channel validation, unequal-size fallback, the fixed
64-pixel API, symmetry, input ownership, and cold/hot equivalence. Central
content and multi-tile regression cases are also included in the 1.0.7 test
suite.

## Known boundaries

- A significantly cropped foreground is intentionally conservative rather than
  forced to a high score.
- Extremely narrow inputs contain little recoverable structural evidence.
- Scores are not probabilities and do not obey transitivity.
- Matchup does not embed a 35% or any other classification threshold.
- Applications should set thresholds from their own labeled score
  distributions.
