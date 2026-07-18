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


def draw_shared_template_icon(use_star: bool) -> np.ndarray:
    image = np.full((64, 64, 3), 35, dtype=np.uint8)
    y, x = np.indices((64, 64))
    ellipse = ((x - 32) / 24) ** 2 + ((y - 33) / 28) ** 2 <= 1
    image[ellipse] = (225, 225, 210)
    border = np.logical_and(
        ((x - 32) / 24) ** 2 + ((y - 33) / 28) ** 2 <= 1.0,
        ((x - 32) / 21) ** 2 + ((y - 33) / 25) ** 2 >= 1.0,
    )
    image[border] = (25, 25, 25)
    image[39:42, 12:53] = (80, 100, 180)
    if use_star:
        center = np.logical_or(
            np.logical_and(np.abs(x - 32) <= 5, np.abs(y - 32) <= 15),
            np.logical_and(np.abs(y - 32) <= 5, np.abs(x - 32) <= 15),
        )
    else:
        center = np.logical_or(
            np.abs((x - 32) - (y - 32)) <= 3,
            np.abs((x - 32) + (y - 32)) <= 3,
        )
        center &= np.logical_and(np.abs(x - 32) <= 15, np.abs(y - 32) <= 15)
    image[center] = (40, 180, 240)
    return image


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

    def test_shared_outer_template_with_different_center_is_penalized(self) -> None:
        image_a = draw_shared_template_icon(True)
        image_b = draw_shared_template_icon(False)
        self.assertLess(matchup.picture_match(image_a, image_b), 0.60)

    def test_matching_foreground_tolerates_colorful_background(self) -> None:
        icon = draw_shared_template_icon(True)
        image_a = icon.copy()
        image_b = icon.copy()
        y, x = np.indices((64, 64))
        foreground = np.any(icon != (35, 35, 35), axis=2)
        background_a = np.stack(
            ((x * 7) % 180, (y * 9) % 180, ((x + y) * 5) % 180), axis=2
        ).astype(np.uint8)
        background_b = np.stack(
            (((x + y) * 11) % 190, (x * 3) % 190, (y * 13) % 190), axis=2
        ).astype(np.uint8)
        image_a[~foreground] = background_a[~foreground]
        image_b[~foreground] = background_b[~foreground]

        self.assertGreater(matchup.picture_match(image_a, image_b), 0.80)

    def test_multi_tile_cold_and_hot_paths_are_identical(self) -> None:
        image_a = structured_image(64, 320)
        image_b = np.roll(image_a, shift=(1, -3), axis=(0, 1))
        cold = matchup.picture_match(image_a, image_b)
        prepared_a = matchup.PreparedImage(image_a)
        prepared_b = matchup.PreparedImage(image_b)
        self.assertEqual(cold, prepared_a.match(image_b))
        self.assertEqual(cold, prepared_a.match(prepared_b))


if __name__ == "__main__":
    unittest.main()
