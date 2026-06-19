from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.config import load_config
from src.vision.defect_rules import (
    detect_color_anomaly,
    detect_dark_crack_like_regions,
    detect_glass_burn,
    detect_raw_fiber,
)
from src.vision.product_detection import _refine_shape_mask_by_panel_color


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

    def test_crack_overlay_adds_nearby_faint_line_components(self) -> None:
        roi = np.full((360, 260, 3), (96, 132, 86), dtype=np.uint8)
        cv2.line(roi, (82, 35), (92, 330), (32, 48, 30), 3)
        for y in range(58, 305, 20):
            cv2.line(roi, (158, y), (164, y + 12), (64, 82, 54), 2)

        result = detect_dark_crack_like_regions(roi, roi, load_config())

        self.assertTrue(result["is_suspicious"])
        self.assertGreater(result["display_crack_area_ratio"], result["crack_area_ratio"])
        self.assertLess(result["display_crack_area_ratio"] - result["crack_area_ratio"], 0.01)

    def test_crack_overlay_adds_independent_faint_cracks_on_cracked_panel(self) -> None:
        roi = np.full((380, 270, 3), (110, 140, 92), dtype=np.uint8)
        for x in (52, 116, 196):
            cv2.line(roi, (x, 35), (x + 9, 330), (34, 48, 30), 3)
        for x in (82, 154, 226):
            for y in range(70, 300, 34):
                cv2.line(roi, (x, y), (x + 5, y + 18), (58, 76, 48), 2)

        result = detect_dark_crack_like_regions(roi, roi, load_config())

        self.assertTrue(result["is_suspicious"])
        self.assertGreater(result["display_crack_area_ratio"], result["crack_area_ratio"])
        self.assertLess(result["display_crack_area_ratio"] - result["crack_area_ratio"], 0.02)

    def test_product_color_refinement_does_not_crop_mixed_color_panel(self) -> None:
        config = load_config()
        image = np.full((240, 140, 3), (92, 92, 78), dtype=np.uint8)
        image[18:222, 22:118] = (84, 110, 82)
        image[118:222, 22:118] = (92, 150, 104)
        mask = np.zeros((240, 140), dtype=np.uint8)
        cv2.rectangle(mask, (22, 18), (117, 221), 255, -1)

        refined = _refine_shape_mask_by_panel_color(image, mask, config)

        original_area = cv2.countNonZero(mask)
        refined_area = cv2.countNonZero(refined)
        self.assertGreaterEqual(refined_area, original_area * 0.78)

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

    def test_glass_burn_flags_broad_dark_stain_without_crack(self) -> None:
        roi = np.full((320, 240, 3), (96, 132, 86), dtype=np.uint8)
        cv2.rectangle(roi, (70, 70), (160, 230), (22, 36, 45), -1)

        burn = detect_glass_burn(roi, roi, load_config())
        crack = detect_dark_crack_like_regions(roi, roi, load_config())

        self.assertTrue(burn["is_suspicious"])
        self.assertGreaterEqual(burn["score"], 0.7)
        self.assertIsNotNone(burn["mask"])
        self.assertFalse(crack["is_suspicious"])

    def test_glass_burn_ignores_bottom_lighting_shadow(self) -> None:
        roi = np.full((320, 240, 3), (96, 132, 86), dtype=np.uint8)
        gradient = np.linspace(1.12, 0.62, roi.shape[0], dtype=np.float32)[:, None, None]
        shaded = np.clip(roi.astype(np.float32) * gradient, 0, 255).astype(np.uint8)

        result = detect_glass_burn(shaded, shaded, load_config())

        self.assertFalse(result["is_suspicious"])
        self.assertIsNone(result["mask"])

    def test_glass_burn_finds_local_stain_under_lighting_gradient(self) -> None:
        roi = np.full((320, 240, 3), (96, 132, 86), dtype=np.uint8)
        gradient = np.linspace(1.12, 0.68, roi.shape[0], dtype=np.float32)[:, None, None]
        shaded = np.clip(roi.astype(np.float32) * gradient, 0, 255).astype(np.uint8)
        cv2.circle(shaded, (128, 210), 34, (24, 34, 42), -1)

        result = detect_glass_burn(shaded, shaded, load_config())

        self.assertTrue(result["is_suspicious"])
        self.assertIsNotNone(result["mask"])

    def test_raw_fiber_flags_light_desaturated_patch(self) -> None:
        roi = np.full((260, 420, 3), (120, 150, 105), dtype=np.uint8)
        cv2.rectangle(roi, (150, 80), (260, 170), (210, 220, 205), -1)

        result = detect_raw_fiber(roi, roi, load_config())

        self.assertTrue(result["is_suspicious"])
        self.assertIsNotNone(result["mask"])

    def test_raw_fiber_flags_raised_fibrous_texture(self) -> None:
        roi = np.full((340, 240, 3), (95, 125, 82), dtype=np.uint8)
        for y in range(18, 330, 14):
            cv2.line(roi, (8, y), (232, y + 2), (112, 145, 98), 2)
        for center in ((82, 108), (142, 205), (92, 268)):
            cv2.ellipse(roi, center, (22, 54), -8, 0, 360, (170, 185, 155), -1)
            for offset in range(-16, 18, 8):
                cv2.line(
                    roi,
                    (center[0] + offset, center[1] - 42),
                    (center[0] + offset + 8, center[1] + 42),
                    (214, 220, 198),
                    3,
                )

        result = detect_raw_fiber(roi, roi, load_config())

        self.assertTrue(result["is_suspicious"])
        self.assertGreater(result["raw_fiber_relief_ratio"], 0.01)
        self.assertIsNotNone(result["mask"])

    def test_raw_fiber_flags_bright_glass_fiber_strands(self) -> None:
        roi = np.full((300, 260, 3), (96, 132, 86), dtype=np.uint8)
        for y in range(45, 260, 22):
            cv2.line(roi, (42, y), (215, y + 7), (215, 225, 205), 2)
        for x in (80, 135, 190):
            cv2.line(roi, (x, 55), (x + 18, 250), (225, 232, 214), 2)

        result = detect_raw_fiber(roi, roi, load_config())

        self.assertTrue(result["is_suspicious"])
        self.assertGreater(result["glass_fiber_ratio"], 0.02)
        self.assertIsNotNone(result["mask"])

    def test_raw_fiber_keeps_uniform_panel_normal(self) -> None:
        roi = np.full((260, 420, 3), (120, 150, 105), dtype=np.uint8)

        result = detect_raw_fiber(roi, roi, load_config())

        self.assertFalse(result["is_suspicious"])
        self.assertIsNone(result["mask"])

    def test_color_strategy_keeps_uniform_panel_normal(self) -> None:
        roi = np.full((260, 420, 3), (120, 150, 105), dtype=np.uint8)

        result = detect_color_anomaly(roi, roi, load_config())

        self.assertFalse(result["is_suspicious"])
        self.assertIsNone(result["mask"])


if __name__ == "__main__":
    unittest.main()
