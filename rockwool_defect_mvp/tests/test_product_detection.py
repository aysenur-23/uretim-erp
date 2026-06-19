from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.config import load_config
from src.vision.product_detection import find_product_roi


class ProductDetectionTests(unittest.TestCase):
    def test_finds_rectangle_product_roi(self) -> None:
        image = np.full((360, 640, 3), 35, dtype=np.uint8)
        cv2.rectangle(image, (110, 90), (540, 260), (150, 170, 185), -1)
        cv2.line(image, (110, 90), (540, 260), (130, 145, 160), 2)

        product = find_product_roi(image, load_config())

        self.assertIsNotNone(product)
        assert product is not None
        x, y, width, height = product.shape_bbox
        self.assertGreater(width, height)
        self.assertGreater(product.area_ratio, 0.15)
        self.assertLess(abs(x - 110), 90)
        self.assertLess(abs(y - 90), 90)

    def test_product_color_prior_beats_large_gray_rectangle(self) -> None:
        image = np.full((420, 720, 3), 25, dtype=np.uint8)
        cv2.rectangle(image, (20, 30), (700, 390), (95, 95, 95), -1)
        cv2.rectangle(image, (150, 115), (610, 300), (150, 170, 185), -1)

        product = find_product_roi(image, load_config())

        self.assertIsNotNone(product)
        assert product is not None
        x, y, width, height = product.shape_bbox
        self.assertLess(x, 220)
        self.assertGreater(x, 90)
        self.assertLess(width, 560)
        self.assertGreater(width, height)


if __name__ == "__main__":
    unittest.main()
