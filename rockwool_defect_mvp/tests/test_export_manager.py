from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import cv2

from src.config import load_config
from src.dataset.export_manager import export_dataset
from src.storage.sqlite_store import SQLiteStore


class ExportManagerTests(unittest.TestCase):
    def test_exports_records_to_dataset_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw.jpg"
            overlay = root / "overlay.jpg"
            cv2.imwrite(str(raw), np.full((20, 20, 3), 120, dtype=np.uint8))
            cv2.imwrite(str(overlay), np.full((20, 20, 3), 180, dtype=np.uint8))

            config = load_config()
            config = config.__class__(
                **{
                    **config.__dict__,
                    "database_path": root / "database" / "quality.sqlite",
                    "output_overlay_path": root / "overlays",
                }
            )
            store = SQLiteStore(config.database_path)
            store.insert_inspection_record(
                {
                    "timestamp": "20260608_010203_000001",
                    "image_path": str(raw),
                    "overlay_path": str(overlay),
                    "product_type": "tas_yunu_panel",
                    "model_result": "SAGLAM",
                    "anomaly_score": 0.2,
                    "edge_damage_score": 0.1,
                    "color_anomaly_score": 0.2,
                    "crack_score": 0.0,
                    "local_anomaly_score": 0.3,
                    "operator_label": "saglam",
                    "operator_note": "ok",
                    "is_model_wrong": 0,
                    "is_confirmed": 1,
                }
            )

            export_root = export_dataset(config)

            self.assertTrue((export_root / "metadata.csv").exists())
            self.assertTrue(any((export_root / "images").rglob("*.jpg")))
            self.assertTrue(any((export_root / "overlays").rglob("*.jpg")))


if __name__ == "__main__":
    unittest.main()
