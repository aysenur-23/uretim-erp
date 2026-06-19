from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from src.config import load_config
from src.vision.preprocessing import resize_image
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

    def test_bbox_snaps_to_outer_panel_edges(self) -> None:
        image = np.full((420, 720, 3), 25, dtype=np.uint8)
        cv2.rectangle(image, (100, 80), (620, 330), (88, 96, 74), -1)
        cv2.rectangle(image, (128, 108), (592, 302), (120, 150, 105), -1)
        cv2.rectangle(image, (100, 80), (620, 330), (180, 190, 160), 3)

        product = find_product_roi(image, load_config())

        self.assertIsNotNone(product)
        assert product is not None
        x, y, width, height = product.shape_bbox
        self.assertLessEqual(abs(x - 100), 18)
        self.assertLessEqual(abs(y - 80), 18)
        self.assertGreaterEqual(width, 500)
        self.assertGreaterEqual(height, 235)

    def test_bbox_keeps_colored_panel_when_internal_edges_are_strong(self) -> None:
        image = np.full((460, 360, 3), 30, dtype=np.uint8)
        cv2.rectangle(image, (55, 35), (310, 425), (138, 165, 128), -1)
        cv2.rectangle(image, (55, 35), (310, 425), (210, 220, 190), 4)
        for x in (120, 180, 245):
            cv2.line(image, (x, 45), (x, 415), (70, 85, 65), 6)
        for y in range(60, 405, 18):
            cv2.line(image, (70, y), (295, y), (165, 185, 145), 2)

        product = find_product_roi(image, load_config())

        self.assertIsNotNone(product)
        assert product is not None
        x, y, width, height = product.shape_bbox
        self.assertLessEqual(abs(x - 55), 20)
        self.assertLessEqual(abs(y - 35), 20)
        self.assertGreaterEqual(width, 235)
        self.assertGreaterEqual(height, 365)

    def test_tall_textured_panel_uses_full_height_frame(self) -> None:
        image_path = Path("data/raw/20260619_003206_116660_upload_defects_image8.jpeg")
        if not image_path.exists():
            self.skipTest("Local tall textured panel sample is not available.")

        data = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        self.assertIsNotNone(image)
        assert image is not None

        image = resize_image(image)
        product = find_product_roi(image, load_config())

        self.assertIsNotNone(product)
        assert product is not None
        _x, y, _width, height = product.shape_bbox
        self.assertLess(y, image.shape[0] * 0.08)
        self.assertGreater(height, image.shape[0] * 0.88)


if __name__ == "__main__":
    unittest.main()
