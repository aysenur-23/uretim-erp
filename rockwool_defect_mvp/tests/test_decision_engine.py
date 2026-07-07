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
                "dark_crack": {"score": 0.5, "is_suspicious": True, "message": "crack"},
            },
            config,
        )

        self.assertEqual(decision.label, "HATALI")
        self.assertIn("Catlak", decision.reasons[0])

    def test_local_anomaly_alone_is_only_suspect(self) -> None:
        # Destekleyici sinyal tek başına RED veremez → en fazla SUPHELI.
        config = load_config()
        decision = decide_quality(
            {
                "local_anomaly": {"score": 0.9, "is_suspicious": True, "message": "anomaly"},
                "dark_crack": {"score": 0.1, "is_suspicious": False, "message": "crack"},
            },
            config,
        )

        self.assertEqual(decision.label, "SUPHELI")

    def test_size_tolerance_reject_is_defect(self) -> None:
        config = load_config()
        decision = decide_quality(
            {
                "size_tolerance": {"score": 0.65, "is_suspicious": True, "message": "olcu"},
            },
            config,
        )

        self.assertEqual(decision.label, "HATALI")

    def test_two_warns_escalate_to_defect(self) -> None:
        config = load_config()
        decision = decide_quality(
            {
                "raw_fiber": {"score": 0.32, "is_suspicious": True, "message": "fiber"},
                "deformation": {"score": 0.31, "is_suspicious": True, "message": "deform"},
            },
            config,
        )

        self.assertEqual(decision.label, "HATALI")

    def test_clean_scores_are_healthy(self) -> None:
        config = load_config()
        decision = decide_quality(
            {
                "dark_crack": {"score": 0.1, "is_suspicious": False, "message": "crack"},
                "local_anomaly": {"score": 0.2, "is_suspicious": False, "message": "anomaly"},
            },
            config,
        )

        self.assertEqual(decision.label, "SAGLAM")


if __name__ == "__main__":
    unittest.main()
