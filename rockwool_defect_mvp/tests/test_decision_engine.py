from __future__ import annotations

import unittest

from src.config import load_config
from src.decision.decision_engine import decide_quality


class DecisionEngineTests(unittest.TestCase):
    def test_combines_multiple_suspicious_rules_as_defect(self) -> None:
        config = load_config()
        decision = decide_quality(
            {
                "edge_damage": {"score": 0.4, "is_suspicious": True, "message": "edge"},
                "color_anomaly": {"score": 0.35, "is_suspicious": True, "message": "color"},
                "dark_crack": {"score": 0.0, "is_suspicious": False, "message": "crack"},
            },
            config,
        )

        self.assertEqual(decision.label, "HATALI")
        self.assertGreaterEqual(len(decision.reasons), 2)

    def test_strong_crack_signal_is_defect(self) -> None:
        config = load_config()
        decision = decide_quality(
            {
                "edge_damage": {"score": 0.0, "is_suspicious": False, "message": "edge"},
                "color_anomaly": {"score": 0.0, "is_suspicious": False, "message": "color"},
                "dark_crack": {"score": 0.44, "is_suspicious": True, "message": "crack"},
            },
            config,
        )

        self.assertEqual(decision.label, "HATALI")
        self.assertIn("Catlak", decision.reasons[0])


if __name__ == "__main__":
    unittest.main()
