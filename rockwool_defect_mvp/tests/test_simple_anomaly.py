from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.anomaly.simple_anomaly import detect_local_anomaly
from src.config import load_config


class SimpleAnomalyTests(unittest.TestCase):
    def test_detects_local_dark_patch(self) -> None:
        roi = np.full((220, 420, 3), (155, 175, 190), dtype=np.uint8)
        cv2.rectangle(roi, (160, 80), (240, 145), (35, 45, 55), -1)

        result = detect_local_anomaly(roi, load_config())

        self.assertGreater(result["score"], 0.0)
        self.assertIsNotNone(result["heatmap"])


if __name__ == "__main__":
    unittest.main()
