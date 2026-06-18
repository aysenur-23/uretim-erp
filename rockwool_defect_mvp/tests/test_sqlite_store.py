from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.storage.sqlite_store import SQLiteStore


class SQLiteStoreTests(unittest.TestCase):
    def test_inserts_and_reads_recent_records_with_local_anomaly_score(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteStore(Path(temp_dir) / "quality.sqlite")

            record_id = store.insert_inspection_record(
                {
                    "timestamp": "20260608_010203_000001",
                    "image_path": "raw.jpg",
                    "overlay_path": "overlay.jpg",
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

            rows = store.fetch_recent_inspection_records(limit=10)
            filtered_rows = store.fetch_recent_inspection_records(limit=10, model_result="SAGLAM")
            missing_rows = store.fetch_recent_inspection_records(limit=10, model_result="HATALI")
            updated = store.update_operator_feedback(record_id, "catlak", "changed", True)
            model_updated = store.update_model_result(
                record_id,
                "HATALI",
                0.8,
                0.1,
                0.2,
                0.7,
                0.3,
                previous_overlay_path="old_overlay.jpg",
                previous_model_result="SAGLAM",
                previous_anomaly_score=0.2,
                last_reprocessed_at="20260608_020304_000002",
            )
            updated_record = store.fetch_inspection_record(record_id)
            summary = store.fetch_quality_summary()
            deleted = store.delete_inspection_record(record_id)
            rows_after_delete = store.fetch_recent_inspection_records(limit=10)

        self.assertEqual(record_id, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(filtered_rows), 1)
        self.assertEqual(missing_rows, [])
        self.assertEqual(rows[0]["local_anomaly_score"], 0.3)
        self.assertTrue(updated)
        self.assertTrue(model_updated)
        assert updated_record is not None
        self.assertEqual(updated_record["operator_label"], "catlak")
        self.assertEqual(updated_record["operator_note"], "changed")
        self.assertEqual(updated_record["is_model_wrong"], 1)
        self.assertEqual(updated_record["model_result"], "HATALI")
        self.assertEqual(updated_record["previous_overlay_path"], "old_overlay.jpg")
        self.assertEqual(updated_record["previous_model_result"], "SAGLAM")
        self.assertEqual(updated_record["previous_anomaly_score"], 0.2)
        self.assertEqual(updated_record["last_reprocessed_at"], "20260608_020304_000002")
        self.assertEqual(summary["totals"]["total_count"], 1)
        self.assertEqual(summary["model_result_counts"][0]["label"], "HATALI")
        self.assertIsNotNone(deleted)
        assert deleted is not None
        self.assertEqual(deleted["id"], record_id)
        self.assertEqual(rows_after_delete, [])

    def test_deletes_all_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteStore(Path(temp_dir) / "quality.sqlite")
            for index in range(2):
                store.insert_inspection_record(
                    {
                        "timestamp": f"20260608_010203_00000{index}",
                        "image_path": f"raw_{index}.jpg",
                        "overlay_path": f"overlay_{index}.jpg",
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

            deleted = store.delete_all_inspection_records()
            rows_after_delete = store.fetch_recent_inspection_records(limit=10)

        self.assertEqual(len(deleted), 2)
        self.assertEqual(rows_after_delete, [])


if __name__ == "__main__":
    unittest.main()
