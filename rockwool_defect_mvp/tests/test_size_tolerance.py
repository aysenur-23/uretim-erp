from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from src.config import load_config
from src.vision.product_detection import ProductROI
from src.vision.size_check import detect_size_tolerance


def _fake_product(corners: list[tuple[float, float]]) -> ProductROI:
    dummy = np.zeros((1, 1), dtype=np.uint8)
    return ProductROI(
        roi=np.zeros((1, 1, 3), dtype=np.uint8),
        bbox=(0, 0, 1, 1),
        rotated_box=tuple((int(x), int(y)) for x, y in corners),
        contour=np.zeros((1, 1, 2), dtype=np.int32),
        mask=dummy,
        shape_mask=dummy,
        shape_bbox=(0, 0, 1, 1),
        shape_roi=np.zeros((1, 1, 3), dtype=np.uint8),
        shape_mask_roi=dummy,
        area_ratio=0.5,
        corners=tuple(corners),
    )


class SizeToleranceTests(unittest.TestCase):
    def setUp(self) -> None:
        # px_per_mm = 1 → piksel = mm; beklenen 400x200 mm, tolerans 5 mm, gönye 1.5°.
        self.config = dataclasses.replace(
            load_config(),
            size_check_enabled=True,
            px_per_mm=1.0,
            expected_width_mm=400.0,
            expected_height_mm=200.0,
            size_tolerance_mm=5.0,
            squareness_tolerance_deg=1.5,
        )

    def test_in_tolerance_scores_zero(self) -> None:
        product = _fake_product([(0, 0), (400, 0), (400, 200), (0, 200)])
        result = detect_size_tolerance(product, self.config)
        self.assertTrue(result["enabled"])
        self.assertFalse(result["is_suspicious"])
        self.assertEqual(result["score"], 0.0)
        self.assertAlmostEqual(result["measured_width_mm"], 400.0, delta=0.5)

    def test_oversize_beyond_tolerance_is_suspicious(self) -> None:
        # 412 mm genişlik → 12 mm sapma, 5 mm toleransın çok üstünde.
        product = _fake_product([(0, 0), (412, 0), (412, 200), (0, 200)])
        result = detect_size_tolerance(product, self.config)
        self.assertTrue(result["is_suspicious"])
        self.assertGreater(result["score"], 0.0)

    def test_out_of_square_corner_is_suspicious(self) -> None:
        # Belirgin gönye hatası: bir köşe 90°'den ~5° sapmış.
        product = _fake_product([(0, 15), (400, 0), (400, 200), (0, 200)])
        result = detect_size_tolerance(product, self.config)
        self.assertGreater(result["squareness_deg"], 1.5)
        self.assertTrue(result["is_suspicious"])

    def test_rotation_allowed_matches_swapped_dims(self) -> None:
        # 200x400 (90° dönmüş) beklenen 400x200 ile eşleşmeli, toleransta.
        product = _fake_product([(0, 0), (200, 0), (200, 400), (0, 400)])
        result = detect_size_tolerance(product, self.config)
        self.assertFalse(result["is_suspicious"])

    def test_disabled_when_uncalibrated(self) -> None:
        config = dataclasses.replace(self.config, px_per_mm=0.0)
        product = _fake_product([(0, 0), (412, 0), (412, 200), (0, 200)])
        result = detect_size_tolerance(product, config)
        self.assertFalse(result["enabled"])
        self.assertFalse(result["is_suspicious"])
        self.assertEqual(result["score"], 0.0)

    def test_disabled_when_no_corners(self) -> None:
        result = detect_size_tolerance(None, self.config)
        self.assertFalse(result["enabled"])
        self.assertEqual(result["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
