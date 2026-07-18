# Changelog

All notable changes to this project will be documented in this file.

## 1.0.7 - 2026-07-18

- Add aligned central-content confirmation to reject icons that share an outer
  template but differ internally.
- Apply verified central negative evidence after the three SIFT scopes are
  merged so wider feature scopes cannot bypass it.
- Preserve high scores for a complete matching foreground icon over colorful
  or textured backgrounds when aligned central edges agree.
- Keep partial or cropped foreground icons conservative instead of forcing a
  high match score.
- Aggregate long-image tiles according to distributed evidence and skip the
  single-tile raw confirmation path for multi-tile images.
- Avoid eager raw-feature preparation for multi-tile images.
- Add current implementation and Mermaid flowchart documentation.

## 0.1.0

- Add the C++17 and pybind11 matching core.
- Add cold-cold, hot-cold and hot-hot matching modes.
- Add bicubic normalization for unequal image sizes.
- Add fixed 64 x 64 working-unit matching and arbitrary aspect-ratio tiling.
- Add Windows wheel build and release automation.
