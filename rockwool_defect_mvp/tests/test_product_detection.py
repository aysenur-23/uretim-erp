from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.config import load_config
from src.vision.product_detection import find_product_roi


def _corner_angles(corners: np.ndarray) -> list[float]:
    angles = []
    for i in range(4):
        a = corners[(i - 1) % 4] - corners[i]
        b = corners[(i + 1) % 4] - corners[i]
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        angles.append(float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))))
    return angles


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


    def test_fits_real_corners_on_skewed_panel(self) -> None:
        base = np.full((300, 520, 3), 30, dtype=np.uint8)
        cv2.rectangle(base, (60, 50), (460, 250), (150, 170, 185), -1)
        src = np.float32([[60, 50], [460, 50], [460, 250], [60, 250]])
        dst = np.float32([[80, 60], [440, 45], [470, 240], [70, 255]])
        warp = cv2.getPerspectiveTransform(src, dst)
        skewed = cv2.warpPerspective(base, warp, (520, 300), borderValue=(30, 30, 30))

        product = find_product_roi(skewed, load_config())
        assert product is not None
        self.assertEqual(len(product.corners), 4)
        # Eğik panelde en az bir köşe belirgin şekilde 90°'den sapmalı (gönye).
        deviation = max(abs(a - 90.0) for a in _corner_angles(np.array(product.corners)))
        self.assertGreater(deviation, 3.0)

    def test_axis_aligned_panel_has_square_corners(self) -> None:
        image = np.full((300, 520, 3), 30, dtype=np.uint8)
        cv2.rectangle(image, (60, 50), (460, 250), (150, 170, 185), -1)

        product = find_product_roi(image, load_config())
        assert product is not None
        deviation = max(abs(a - 90.0) for a in _corner_angles(np.array(product.corners)))
        self.assertLess(deviation, 2.0)
        self.assertGreater(product.roi_confidence, 0.5)

    def test_background_diff_recovers_low_contrast_panel(self) -> None:
        # Panel arka planla neredeyse aynı gri; Otsu ayıramaz ama boş-bant
        # referansı fark maskesiyle paneli bulur.
        background = np.full((300, 520, 3), 120, dtype=np.uint8)
        cv2.randn(background, 120, 4)
        scene = background.copy()
        cv2.rectangle(scene, (90, 70), (430, 230), (128, 128, 128), -1)

        with tempfile.TemporaryDirectory() as tmp:
            ref_path = Path(tmp) / "bg.png"
            cv2.imwrite(str(ref_path), background)
            config = dataclasses.replace(
                load_config(), background_reference_path=ref_path
            )
            product = find_product_roi(scene, config)

        assert product is not None
        self.assertEqual(product.detection_method, "background_diff")
        x, y, width, height = product.shape_bbox
        self.assertLess(abs(x - 90), 40)
        self.assertLess(abs(width - 340), 80)


if __name__ == "__main__":
    unittest.main()
