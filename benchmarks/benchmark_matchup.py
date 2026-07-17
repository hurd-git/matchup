from time import perf_counter

import numpy as np

import matchup


def main() -> None:
    random = np.random.default_rng(7)
    images = [
        random.integers(0, 256, (80, 80, 3), dtype=np.uint8)
        for _ in range(64)
    ]
    pairs = [
        (row, column)
        for row in range(len(images))
        for column in range(row + 1, len(images))
    ]

    start = perf_counter()
    cold = [matchup.picture_match(images[a], images[b]) for a, b in pairs]
    cold_seconds = perf_counter() - start

    start = perf_counter()
    prepared = [matchup.PreparedImage(image) for image in images]
    preparation_seconds = perf_counter() - start

    start = perf_counter()
    hot = [prepared[a].match(prepared[b]) for a, b in pairs]
    hot_seconds = perf_counter() - start

    if cold != hot:
        raise AssertionError("Cold and hot scores differ")
    print(f"images: {len(images)}")
    print(f"pairs: {len(pairs)}")
    print(f"cold-cold: {cold_seconds:.4f}s")
    print(f"prepare: {preparation_seconds:.4f}s")
    print(f"hot-hot: {hot_seconds:.4f}s")


if __name__ == "__main__":
    main()
