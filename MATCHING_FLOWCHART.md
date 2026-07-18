# Matchup matching flowcharts

These diagrams describe the current `matchup` 1.0.8 Python and C++ execution
paths. See [MATCHING_IMPLEMENTATION.md](MATCHING_IMPLEMENTATION.md) for detailed
parameters and design notes.

## End-to-end flow

```mermaid
flowchart TD
    A["Two ndarray inputs A and B"] --> B["Validate dtype, shape, and channels<br/>Convert to contiguous uint8"]
    B --> C{"Same spatial size?"}
    C -- "No" --> D["Choose the smaller image by short edge and area"]
    D --> E["Bicubic-resize the larger image<br/>to the smaller width and height"]
    C -- "Yes" --> F["Enter the C++ core"]
    E --> F
    F --> G{"Short edge above 64?"}
    G -- "Yes" --> H["Proportionally reduce<br/>the short edge to 64"]
    G -- "No" --> I["Keep original pixels<br/>Do not enlarge"]
    H --> J["Equal normalized shapes"]
    I --> J
    J --> K{"Both edges at most 64?"}
    K -- "Yes" --> L["Center on a 64 x 64 canvas<br/>Pad with border-median color"]
    K -- "No" --> M["Distribute ceil(long edge / 64)<br/>windows along the long edge"]
    M --> N["Pad every window to 64 x 64"]
    L --> O["One corresponding tile pair"]
    N --> P["Multiple corresponding tile pairs"]
    O --> Q["Run the 64 x 64 matching core"]
    P --> Q
    Q --> R["Collect tile scores"]
    R --> S{"Fewer than four tiles?"}
    S -- "Yes" --> T["Arithmetic mean"]
    S -- "No" --> U["Measure distributed effective evidence"]
    U --> V["Blend arithmetic and evidence-weighted means"]
    T --> W["Return a score from 0.0 to 1.0"]
    V --> W
```

## One 64 x 64 working unit

```mermaid
flowchart TD
    A["One 64 x 64 image pair"] --> B["Prepare grayscale, Canny edges,<br/>and Sobel gradient magnitude"]
    B --> C["Outline branch"]
    B --> D["Gradient fingerprint branch"]
    subgraph OUTLINE["Outline branch"]
        C --> C1["Search edge translations within 5 pixels"]
        C1 --> C2["Best A-to-B and B-to-A edge F1"]
        C2 --> C3["Average the directional F1 scores"]
        C3 --> C4["Measure central per-channel appearance error"]
        C4 --> C5["Outline = edge agreement x appearance factor"]
    end
    subgraph FINGERPRINT["Gradient fingerprint branch"]
        D --> D1["Extract SIFT features on gradients"]
        D1 --> D2["Strict scope<br/>Exclude outer 20%"]
        D1 --> D3["Expanded scope<br/>Exclude outer 15%"]
        D1 --> D4["Anchored full scope<br/>One endpoint must be central"]
        D2 --> E["Bidirectional descriptor matching"]
        D3 --> E
        D4 --> E
        E --> E1["Ratio test and keypoint deduplication"]
        E1 --> E2["Bidirectional partial-affine RANSAC"]
        E2 --> E3{"Scale, inliers, and spatial coverage valid?"}
        E3 -- "No" --> E4["Scope score is zero"]
        E3 -- "Yes" --> E5["Geometric consistency score"]
        E5 --> E6["Inverse transform, scale, rotation,<br/>and inlier confidence"]
        E6 --> E7["Aligned dense error and edge containment"]
        E7 --> E8["Central inliers can strengthen confidence"]
    end
    E8 --> F["Aligned central-content confirmation"]
    F --> G["Reuse verified affine transforms<br/>to align pixels and edges"]
    G --> H["Inspect the central 32 x 32 region"]
    H --> I["Central edge F1"]
    H --> J["Central per-channel color error"]
    I --> K{"Central edge F1 at least 0.50?"}
    K -- "Yes" --> L["Foreground is structurally complete<br/>Ignore background color error"]
    K -- "No" --> M["Keep color difference as negative evidence"]
    I --> N["Edge F1 below 0.30 begins a penalty<br/>0.18 reaches the maximum"]
    J --> M
    L --> O["Create local-content factor"]
    M --> O
    N --> O
    O --> P["Minimum factor is 0.20"]
    P --> Q["Fuse three SIFT scopes"]
    Q --> Q1["Strict evidence contributes 70%"]
    Q --> Q2["Best secondary evidence contributes 30%"]
    Q1 --> R["Apply verified local negative evidence<br/>after scope fusion"]
    Q2 --> R
    R --> S{"Single tile, fingerprint below 0.8,<br/>and above outline?"}
    S -- "Yes" --> T["Raw-image fingerprint confirmation<br/>can only lower the score"]
    S -- "No" --> U["Keep gradient fingerprint"]
    T --> V["Final fingerprint score"]
    U --> V
    C5 --> W["Take max of outline and fingerprint"]
    V --> W
    W --> X["Return the working-unit score"]
```

## Central-content decision

```mermaid
flowchart LR
    A["Reliable bidirectional geometry"] --> B["Inspect central 32 x 32"]
    B --> C{"Foreground edge F1 at least 0.50?"}
    C -- "Yes" --> D["Complete matching foreground<br/>Tolerate colorful or textured background"]
    C -- "No" --> E{"Strong edge or color mismatch?"}
    E -- "Yes" --> F["Lower the score<br/>Reject shared outer templates"]
    E -- "No" --> G["Keep the score"]
```

## Cold and prepared paths

```mermaid
flowchart TD
    A{"Public call"}
    A -- "picture_match(A, B)" --> B["Cold-prepare A"]
    B --> C["Cold-prepare B"]
    A -- "PreparedImage(A).match(B)" --> D["Reuse A preparation<br/>Cold-prepare B"]
    A -- "PreparedImage(A).match(PreparedImage(B))" --> E["Reuse A and B preparations"]
    C --> F["Run the shared scoring pipeline"]
    D --> F
    E --> F
```
