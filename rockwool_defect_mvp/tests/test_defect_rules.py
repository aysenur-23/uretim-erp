from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.config import load_config
from src.vision.defect_rules import detect_color_anomaly, detect_dark_crack_like_regions


class DefectRulesTests(unittest.TestCase):
    def test_dark_crack_rule_flags_long_dark_line(self) -> None:
        roi = np.full((220, 420, 3), (155, 175, 190), dtype=np.uint8)
        cv2.line(roi, (60, 110), (360, 115), (15, 15, 15), 4)

        result = detect_dark_crack_like_regions(roi, roi, load_config())

        self.assertGreater(result["score"], 0.0)
        self.assertIsNotNone(result["mask"])

    def test_flags_multiple_vertical_cracks_as_suspicious(self) -> None:
        roi = np.full((360, 260, 3), (100, 135, 92), dtype=np.uint8)
        for x in (55, 105, 155, 205):
            cv2.line(roi, (x, 35), (x + 12, 320), (38, 55, 35), 3)
            cv2.line(roi, (x + 6, 120), (x + 10, 220), (20, 32, 25), 2)

        result = detect_dark_crack_like_regions(roi, roi, load_config())

        self.assertTrue(result["is_suspicious"])
        self.assertGreaterEqual(result["score"], 0.45)
        self.assertGreaterEqual(result["component_count"], 3)
        self.assertIsNotNone(result["mask"])

    def test_flags_subtle_vertical_crack_on_green_panel(self) -> None:
        roi = np.full((360, 260, 3), (96, 132, 86), dtype=np.uint8)
        for y in range(20, 340, 14):
            cv2.line(roi, (126, y), (130, min(350, y + 9)), (48, 68, 42), 2)
        cv2.line(roi, (178, 60), (185, 300), (42, 61, 38), 2)

        result = detect_dark_crack_like_regions(roi, roi, load_config())

        self.assertTrue(result["is_suspicious"])
        self.assertGreaterEqual(result["score"], 0.30)
        self.assertGreaterEqual(result["vertical_coverage"], 0.25)

    def test_regular_dense_fiber_texture_is_not_crack(self) -> None:
        roi = np.full((360, 520, 3), (150, 168, 135), dtype=np.uint8)
        for x in range(18, 500, 9):
            cv2.line(roi, (x, 20), (x + 3, 340), (118, 136, 112), 1)
        for y in range(35, 340, 18):
            cv2.line(roi, (20, y), (500, y + 2), (170, 184, 150), 1)

        result = detect_dark_crack_like_regions(roi, roi, load_config())

        self.assertFalse(result["is_suspicious"])
        self.assertTrue(result["dense_texture"])

    def test_color_strategy_flags_compact_stain(self) -> None:
        roi = np.full((260, 420, 3), (120, 150, 105), dtype=np.uint8)
        cv2.rectangle(roi, (180, 90), (260, 165), (42, 58, 135), -1)

        result = detect_color_anomaly(roi, roi, load_config())

        self.assertTrue(result["is_suspicious"])
        self.assertIsNotNone(result["mask"])
        self.assertGreater(result["largest_component_ratio"], 0.01)
        self.assertIn("Lab", result["strategy"])

    def test_color_strategy_keeps_uniform_panel_normal(self) -> None:
        roi = np.full((260, 420, 3), (120, 150, 105), dtype=np.uint8)

        result = detect_color_anomaly(roi, roi, load_config())

        self.assertFalse(result["is_suspicious"])
        self.assertIsNone(result["mask"])


if __name__ == "__main__":
    unittest.main()
