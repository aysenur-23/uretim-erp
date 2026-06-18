from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.config import load_config
from src.vision.color_calibration import calibrate_product_hsv


class ColorCalibrationTests(unittest.TestCase):
    def test_calibrates_hsv_from_detected_product(self) -> None:
        image = np.full((360, 640, 3), 35, dtype=np.uint8)
        cv2.rectangle(image, (120, 100), (540, 260), (150, 170, 185), -1)

        lower, upper = calibrate_product_hsv(image, load_config())

        self.assertEqual(len(lower), 3)
        self.assertEqual(len(upper), 3)
        self.assertLessEqual(lower[0], upper[0])
        self.assertLessEqual(lower[1], upper[1])
        self.assertLessEqual(lower[2], upper[2])


if __name__ == "__main__":
    unittest.main()
