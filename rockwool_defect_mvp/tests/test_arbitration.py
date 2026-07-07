from __future__ import annotations

import unittest

import numpy as np

from src.decision.arbitration import arbitrate_overlaps


def _mask(coords: list[tuple[slice, slice]]) -> np.ndarray:
    mask = np.zeros((100, 100), dtype=np.uint8)
    for rows, cols in coords:
        mask[rows, cols] = 255
    return mask


class ArbitrationTests(unittest.TestCase):
    def test_long_crack_wins_over_burn_when_overlapping(self) -> None:
        shared = _mask([(slice(10, 90), slice(40, 55))])
        results = {
            "dark_crack": {"score": 0.8, "is_suspicious": True, "mask": shared, "max_length_ratio": 0.5},
            "glass_burn": {"score": 0.6, "is_suspicious": True, "mask": shared.copy()},
        }
        out = arbitrate_overlaps(results)
        self.assertTrue(out["dark_crack"]["is_suspicious"])
        self.assertFalse(out["glass_burn"]["is_suspicious"])
        self.assertEqual(out["glass_burn"]["arbitrated_by"], "dark_crack")

    def test_burn_wins_when_no_long_crack(self) -> None:
        shared = _mask([(slice(40, 60), slice(40, 60))])
        results = {
            "dark_crack": {"score": 0.5, "is_suspicious": True, "mask": shared, "max_length_ratio": 0.1},
            "glass_burn": {"score": 0.7, "is_suspicious": True, "mask": shared.copy()},
        }
        out = arbitrate_overlaps(results)
        self.assertFalse(out["dark_crack"]["is_suspicious"])
        self.assertTrue(out["glass_burn"]["is_suspicious"])
        self.assertEqual(out["dark_crack"]["arbitrated_by"], "glass_burn")

    def test_no_arbitration_when_masks_disjoint(self) -> None:
        results = {
            "dark_crack": {
                "score": 0.8,
                "is_suspicious": True,
                "mask": _mask([(slice(0, 20), slice(0, 20))]),
                "max_length_ratio": 0.5,
            },
            "glass_burn": {
                "score": 0.7,
                "is_suspicious": True,
                "mask": _mask([(slice(80, 100), slice(80, 100))]),
            },
        }
        out = arbitrate_overlaps(results)
        self.assertTrue(out["dark_crack"]["is_suspicious"])
        self.assertTrue(out["glass_burn"]["is_suspicious"])

    def test_no_arbitration_when_only_one_suspicious(self) -> None:
        shared = _mask([(slice(40, 60), slice(40, 60))])
        results = {
            "dark_crack": {"score": 0.8, "is_suspicious": True, "mask": shared, "max_length_ratio": 0.5},
            "glass_burn": {"score": 0.2, "is_suspicious": False, "mask": shared.copy()},
        }
        out = arbitrate_overlaps(results)
        self.assertTrue(out["dark_crack"]["is_suspicious"])
        self.assertFalse(out["glass_burn"]["is_suspicious"])


if __name__ == "__main__":
    unittest.main()
