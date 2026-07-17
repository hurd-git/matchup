import unittest

import numpy as np

import matchup


def structured_image(height: int, width: int, channels: int = 3) -> np.ndarray:
    y, x = np.indices((height, width))
    red = ((x * 7 + y * 3) % 256).astype(np.uint8)
    green = (((x // 5) * 31 + (y // 7) * 17) % 256).astype(np.uint8)
    blue = np.where((x - width // 2) ** 2 + (y - height // 2) ** 2 < 400, 230, 20).astype(np.uint8)
    if channels == 1:
        return red
    result = np.stack((red, green, blue), axis=2)
    if channels == 4:
        alpha = np.full((height, width, 1), 255, dtype=np.uint8)
        result = np.concatenate((result, alpha), axis=2)
    return result


class MatchupApiTests(unittest.TestCase):
    def test_identical_image_scores_one(self) -> None:
        image = structured_image(80, 80)
        self.assertEqual(matchup.picture_match(image, image), 1.0)

    def test_score_is_symmetric(self) -> None:
        image_a = structured_image(80, 80)
        image_b = np.roll(image_a, shift=(2, -3), axis=(0, 1))
        self.assertEqual(
            matchup.picture_match(image_a, image_b),
            matchup.picture_match(image_b, image_a),
        )

    def test_cold_hot_modes_are_identical(self) -> None:
        image_a = structured_image(63, 255)
        image_b = np.roll(image_a, shift=(1, -4), axis=(0, 1))
        cold = matchup.picture_match(image_a, image_b)
        prepared_a = matchup.PreparedImage(image_a)
        prepared_b = matchup.PreparedImage(image_b)
        self.assertEqual(cold, prepared_a.match(image_b))
        self.assertEqual(cold, prepared_a.match(prepared_b))

    def test_cross_size_prepared_match_falls_back_to_cold(self) -> None:
        image_a = structured_image(73, 121, 4)
        image_b = matchup.resize_image(image_a, 88, 197)
        cold = matchup.picture_match(image_a, image_b)
        prepared_a = matchup.PreparedImage(image_a)
        prepared_b = matchup.PreparedImage(image_b)
        self.assertEqual(cold, prepared_a.match(image_b))
        self.assertEqual(cold, prepared_a.match(prepared_b))
        self.assertEqual(cold, matchup.picture_match(image_b, image_a))

    def test_float_grayscale_and_rgba_inputs(self) -> None:
        gray = structured_image(31, 57, 1).astype(np.float32) / 255.0
        rgba = structured_image(64, 64, 4)
        self.assertEqual(matchup.picture_match(gray, gray), 1.0)
        self.assertEqual(matchup.picture_match(rgba, rgba), 1.0)

    def test_resize_preserves_channel_axis(self) -> None:
        gray = structured_image(4, 6, 1)
        self.assertEqual(matchup.resize_image(gray, 9, 11).shape, (9, 11))
        gray_with_axis = gray[:, :, np.newaxis]
        self.assertEqual(
            matchup.resize_image(gray_with_axis, 9, 11).shape,
            (9, 11, 1),
        )

    def test_prepared_image_owns_input_pixels(self) -> None:
        source = structured_image(64, 64)
        original = source.copy()
        candidate = np.roll(original, shift=(1, -2), axis=(0, 1))
        prepared = matchup.PreparedImage(source)
        source.fill(0)
        self.assertEqual(
            prepared.match(candidate),
            matchup.picture_match(original, candidate),
        )

    def test_picture_match_64_rejects_other_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "64 x 64"):
            matchup.picture_match_64(
                np.zeros((63, 64), dtype=np.uint8),
                np.zeros((63, 64), dtype=np.uint8),
            )

    def test_channel_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "same channel layout"):
            matchup.picture_match(
                structured_image(64, 64, 3),
                structured_image(80, 80, 4),
            )


if __name__ == "__main__":
    unittest.main()
