from __future__ import annotations

import dataclasses
import unittest

from src.config import load_config
from src.decision.decision_engine import decide_quality


class PhoneModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixed = dataclasses.replace(load_config(), inspection_mode="fixed_camera")
        self.phone = dataclasses.replace(load_config(), inspection_mode="phone")

    def test_deformation_alone_rejects_in_fixed_mode(self) -> None:
        rules = {"deformation": {"score": 0.9, "is_suspicious": True, "message": "deform"}}
        self.assertEqual(decide_quality(rules, self.fixed).label, "HATALI")

    def test_deformation_alone_only_suspect_in_phone_mode(self) -> None:
        # Telefonda perspektif kaynaklı deformasyon tek başına RED üretmemeli.
        rules = {"deformation": {"score": 0.9, "is_suspicious": True, "message": "deform"}}
        self.assertEqual(decide_quality(rules, self.phone).label, "SUPHELI")

    def test_deformation_plus_surface_warn_not_auto_reject_in_phone_mode(self) -> None:
        # Deformasyon destekleyici olduğu için "2 uyarı -> RED" kuralı tetiklenmez.
        rules = {
            "deformation": {"score": 0.9, "is_suspicious": True, "message": "deform"},
            "raw_fiber": {"score": 0.32, "is_suspicious": True, "message": "fiber"},
        }
        self.assertEqual(decide_quality(rules, self.phone).label, "SUPHELI")

    def test_real_surface_defect_still_rejects_in_phone_mode(self) -> None:
        # Gerçek yüzey/çizgi hatası telefon modunda da RED üretir.
        rules = {"dark_crack": {"score": 0.5, "is_suspicious": True, "message": "crack"}}
        self.assertEqual(decide_quality(rules, self.phone).label, "HATALI")

    def test_size_tolerance_cannot_reject_in_phone_mode(self) -> None:
        rules = {"size_tolerance": {"score": 0.9, "is_suspicious": True, "message": "olcu"}}
        self.assertEqual(decide_quality(rules, self.phone).label, "SUPHELI")


if __name__ == "__main__":
    unittest.main()
