from __future__ import annotations

import unittest

import cv2

from src.config import load_config
from src.vision.defect_rules import (
    detect_glass_burn,
    detect_raw_fiber,
    detect_shape_deformation,
)
from src.vision.product_detection import find_product_roi
from src.vision.inspection_pipeline import product_display_bbox, product_display_roi
from tests.synthetic_panels import (
    make_bowed_panel,
    make_panel,
    paint_burn,
    paint_crack,
    paint_raw_fiber,
)


class SurfaceDefectTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()

    def _roi(self, img):
        product = find_product_roi(img, self.config)
        assert product is not None
        return product, product_display_roi(product), product_display_bbox(product)

    def test_glass_burn_flags_compact_dark_blob(self) -> None:
        img = paint_burn(make_panel(), (260, 150), radius=34)
        _, roi, _ = self._roi(img)
        result = detect_glass_burn(img, roi, self.config)
        self.assertTrue(result["is_suspicious"])

    def test_glass_burn_ignores_thin_crack(self) -> None:
        img = paint_crack(make_panel(), (120, 90), (400, 220), thickness=3)
        _, roi, _ = self._roi(img)
        result = detect_glass_burn(img, roi, self.config)
        self.assertFalse(result["is_suspicious"])

    def test_glass_burn_clean_panel_low_score(self) -> None:
        _, roi, _ = self._roi(make_panel())
        result = detect_glass_burn(make_panel(), roi, self.config)
        self.assertLess(result["score"], 0.15)

    def test_raw_fiber_flags_desaturated_patch(self) -> None:
        img = paint_raw_fiber(make_panel(), (180, 110, 340, 210))
        _, roi, _ = self._roi(img)
        result = detect_raw_fiber(img, roi, self.config)
        self.assertTrue(result["is_suspicious"])

    def test_raw_fiber_ignores_burn(self) -> None:
        img = paint_burn(make_panel(), (260, 150), radius=34)
        _, roi, _ = self._roi(img)
        result = detect_raw_fiber(img, roi, self.config)
        self.assertFalse(result["is_suspicious"])

    def test_raw_fiber_clean_panel_low_score(self) -> None:
        _, roi, _ = self._roi(make_panel())
        result = detect_raw_fiber(make_panel(), roi, self.config)
        self.assertLess(result["score"], self.config.raw_fiber_threshold)

    def test_deformation_flags_bowed_panel(self) -> None:
        # İçbükey (eğilmiş/ezilmiş) uzun kenarlı deforme panel.
        img = make_bowed_panel(bow=95)
        product, roi, bbox = self._roi(img)
        result = detect_shape_deformation(img, roi, bbox, self.config, product)
        self.assertTrue(result["is_suspicious"])

    def test_deformation_clean_panel_not_suspicious(self) -> None:
        img = make_panel()
        product, roi, bbox = self._roi(img)
        result = detect_shape_deformation(img, roi, bbox, self.config, product)
        self.assertFalse(result["is_suspicious"])


if __name__ == "__main__":
    unittest.main()
