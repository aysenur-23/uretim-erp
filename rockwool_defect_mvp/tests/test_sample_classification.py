from __future__ import annotations

import unittest
from pathlib import Path

import cv2

from src.config import load_config
from src.vision.inspection_pipeline import process_frame


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SampleClassificationTests(unittest.TestCase):
    """Gerçek örnek görüntülerle uçtan uca regresyon koruması.

    Her hata türünün kendi sınıfında ateşlemesini ve sağlam ürünün geçmesini
    sabitler; ileride bir ayar sınıf ayrımını bozarsa test kırılır.
    """

    def setUp(self) -> None:
        self.config = load_config()

    def _analyze(self, relative_path: str):
        image = cv2.imread(str(PROJECT_ROOT / relative_path))
        self.assertIsNotNone(image, f"Örnek görüntü yüklenemedi: {relative_path}")
        analysis = process_frame(image, self.config)
        self.assertIsNotNone(analysis.decision)
        suspicious = {name for name, res in analysis.rule_results.items() if res.get("is_suspicious")}
        return analysis.decision.label, suspicious

    def test_crack_sample_flags_dark_crack(self) -> None:
        label, suspicious = self._analyze("frontend/public/samples/catlak.png")
        self.assertEqual(label, "HATALI")
        self.assertIn("dark_crack", suspicious)

    def test_raw_fiber_sample_flags_raw_fiber(self) -> None:
        label, suspicious = self._analyze("frontend/public/samples/cigelyaf.png")
        self.assertIn(label, {"HATALI", "SUPHELI"})
        self.assertIn("raw_fiber", suspicious)

    def test_healthy_sample_passes(self) -> None:
        label, suspicious = self._analyze("frontend/public/samples/saglikli.jpg")
        self.assertEqual(label, "SAGLAM")
        self.assertEqual(suspicious, set())


if __name__ == "__main__":
    unittest.main()
