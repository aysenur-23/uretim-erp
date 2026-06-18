from __future__ import annotations

import unittest
from pathlib import Path

from src.config import load_config
from tools.evaluate_gallery import _evaluate_item, _load_items


class GalleryCalibrationTests(unittest.TestCase):
    def test_current_gallery_matches_operator_feedback(self) -> None:
        ground_truth_path = Path("calibration/gallery_ground_truth.json")
        config = load_config()
        if not config.database_path.exists():
            self.skipTest("Local gallery database is not available.")

        try:
            items = _load_items(ground_truth_path, config.database_path)
        except RuntimeError as exc:
            self.skipTest(str(exc))

        mismatches = []
        for item in items:
            result = _evaluate_item(item, config)
            if not result["exact"]:
                mismatches.append(result)

        self.assertEqual([], mismatches)


if __name__ == "__main__":
    unittest.main()
