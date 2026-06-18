from __future__ import annotations

import unittest

import numpy as np

from src.anomaly.patchcore_adapter import PatchCoreAdapter


class DummyPatchCoreBackend:
    def predict(self, image: np.ndarray) -> tuple[float, np.ndarray | None]:
        return 0.8, np.zeros(image.shape[:2], dtype=np.uint8)


class PatchCoreAdapterTests(unittest.TestCase):
    def test_reports_unavailable_without_backend(self) -> None:
        adapter = PatchCoreAdapter(model_path="models/patchcore.onnx")

        self.assertFalse(adapter.is_available)
        with self.assertRaisesRegex(RuntimeError, "not configured"):
            adapter.predict(np.zeros((32, 32, 3), dtype=np.uint8))

    def test_wraps_backend_prediction(self) -> None:
        adapter = PatchCoreAdapter(backend=DummyPatchCoreBackend(), threshold=0.5)

        prediction = adapter.predict(np.zeros((32, 32, 3), dtype=np.uint8))

        self.assertTrue(adapter.is_available)
        self.assertTrue(prediction.is_suspicious)
        self.assertEqual(prediction.score, 0.8)
        self.assertIsNotNone(prediction.heatmap)


if __name__ == "__main__":
    unittest.main()
