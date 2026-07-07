from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.main import app


class FastApiBackendTests(unittest.TestCase):
    def test_upload_list_image_reprocess_and_delete_flow(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "camera_source: both",
                        f"image_folder_path: {(root / 'samples').as_posix()}",
                        "webcam_index: 0",
                        f"output_overlay_path: {(root / 'outputs' / 'overlays').as_posix()}",
                        f"database_path: {(root / 'data' / 'database' / 'quality.sqlite').as_posix()}",
                        "min_product_area_ratio: 0.15",
                        "product_hsv_lower: [5, 25, 70]",
                        "product_hsv_upper: [55, 190, 255]",
                        "product_color_profile_threshold: 0.45",
                        "edge_damage_threshold: 0.35",
                        "color_anomaly_threshold: 0.30",
                        "crack_darkness_threshold: 0.45",
                        "local_anomaly_threshold: 0.42",
                        "anomaly_score_suspicious: 0.40",
                        "anomaly_score_defect: 0.70",
                    ]
                ),
                encoding="utf-8",
            )
            previous_config = os.environ.get("MEGA_CONFIG_PATH")
            os.environ["MEGA_CONFIG_PATH"] = str(config_path)
            try:
                with TestClient(app) as client:
                    defect_types_response = client.get("/api/defect-types")
                    self.assertEqual(defect_types_response.status_code, 200)
                    defect_types = defect_types_response.json()
                    self.assertIn("pipeline", defect_types)
                    self.assertIn("items", defect_types)
                    self.assertIn("glass_burn", {item["type"] for item in defect_types["items"]})

                    image_bytes = _synthetic_panel_jpeg()

                    response = client.post(
                        "/api/analyze?source=upload",
                        files={"file": ("Ekran görüntüsü panel.jpg", image_bytes, "image/jpeg")},
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    item = response.json()["item"]
                    record_id = item["id"]
                    self.assertEqual(item["source"], "upload")
                    self.assertIn(item["verdict"], {"KABUL", "RED", "UYARI"})
                    self.assertGreaterEqual(len(item["pipeline"]), 5)
                    self.assertIn("roiConfidence", item)
                    self.assertGreaterEqual(item["roiConfidence"], 0.0)

                    list_response = client.get("/api/analyses")
                    self.assertEqual(list_response.status_code, 200)
                    self.assertEqual(len(list_response.json()["items"]), 1)

                    feedback_response = client.post(
                        f"/api/analyses/{record_id}/feedback",
                        json={
                            "expectedVerdict": "KABUL",
                            "expectedDefects": [],
                            "roiOk": True,
                            "note": "operator check",
                        },
                    )
                    self.assertEqual(feedback_response.status_code, 200, feedback_response.text)
                    metrics_response = client.get("/api/calibration/metrics")
                    self.assertEqual(metrics_response.status_code, 200)
                    metrics = metrics_response.json()
                    self.assertEqual(metrics["feedbackCount"], 1)
                    self.assertEqual(metrics["roiFeedback"]["ok"], 1)
                    self.assertIn("perDefect", metrics)

                    image_response = client.get(f"/api/images/{record_id}?v=overlay")
                    self.assertEqual(image_response.status_code, 200)
                    self.assertGreater(len(image_response.content), 100)

                    with sqlite3.connect(root / "data" / "database" / "quality.sqlite") as connection:
                        connection.row_factory = sqlite3.Row
                        record = connection.execute(
                            "select image_path, overlay_path from inspection_records where id = ?",
                            (record_id,),
                        ).fetchone()
                    self.assertIsNotNone(record)
                    self.assertTrue(Path(record["image_path"]).exists())
                    self.assertTrue(Path(record["overlay_path"]).exists())
                    self.assertNotIn("ü", Path(record["image_path"]).name)
                    self.assertNotIn("ü", Path(record["overlay_path"]).name)

                    reprocess_response = client.post(f"/api/analyses/{record_id}/reprocess")
                    self.assertEqual(reprocess_response.status_code, 200, reprocess_response.text)
                    self.assertTrue(reprocess_response.json()["item"]["previousOverlaySrc"])

                    delete_response = client.delete(f"/api/analyses/{record_id}")
                    self.assertEqual(delete_response.status_code, 200)
                    self.assertEqual(client.get("/api/analyses").json()["items"], [])
            finally:
                if previous_config is None:
                    os.environ.pop("MEGA_CONFIG_PATH", None)
                else:
                    os.environ["MEGA_CONFIG_PATH"] = previous_config


    def test_size_calibration_flow(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "camera_source: both",
                        f"image_folder_path: {(root / 'samples').as_posix()}",
                        "webcam_index: 0",
                        f"output_overlay_path: {(root / 'outputs' / 'overlays').as_posix()}",
                        f"database_path: {(root / 'data' / 'database' / 'quality.sqlite').as_posix()}",
                        "min_product_area_ratio: 0.15",
                        "product_hsv_lower: [5, 25, 70]",
                        "product_hsv_upper: [55, 190, 255]",
                        "product_color_profile_threshold: 0.45",
                        "edge_damage_threshold: 0.35",
                        "color_anomaly_threshold: 0.30",
                        "crack_darkness_threshold: 0.45",
                        "local_anomaly_threshold: 0.42",
                        "anomaly_score_suspicious: 0.40",
                        "anomaly_score_defect: 0.70",
                        f"size_calibration_path: {(root / 'calibration' / 'size.json').as_posix()}",
                    ]
                ),
                encoding="utf-8",
            )
            previous_config = os.environ.get("MEGA_CONFIG_PATH")
            os.environ["MEGA_CONFIG_PATH"] = str(config_path)
            try:
                with TestClient(app) as client:
                    before = client.get("/api/calibration/size").json()
                    self.assertFalse(before["calibrated"])

                    upload = client.post(
                        "/api/analyze?source=upload",
                        files={"file": ("panel.jpg", _synthetic_panel_jpeg(), "image/jpeg")},
                    )
                    record_id = upload.json()["item"]["id"]

                    # Panel ~430x170 px; bilinen 430x170 mm dersek px/mm ~= 1.
                    calibrate = client.post(
                        "/api/calibration/size",
                        json={"recordId": int(record_id), "knownWidthMm": 430.0, "knownHeightMm": 170.0},
                    )
                    self.assertEqual(calibrate.status_code, 200, calibrate.text)
                    px_per_mm = calibrate.json()["pxPerMm"]
                    self.assertGreater(px_per_mm, 0.7)
                    self.assertLess(px_per_mm, 1.4)

                    after = client.get("/api/calibration/size").json()
                    self.assertTrue(after["calibrated"])
                    self.assertTrue(after["enabled"])
            finally:
                if previous_config is None:
                    os.environ.pop("MEGA_CONFIG_PATH", None)
                else:
                    os.environ["MEGA_CONFIG_PATH"] = previous_config


def _synthetic_panel_jpeg() -> bytes:
    image = np.full((360, 640, 3), 35, dtype=np.uint8)
    cv2.rectangle(image, (110, 90), (540, 260), (150, 170, 185), -1)
    cv2.line(image, (110, 90), (540, 260), (130, 145, 160), 2)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Could not encode test image")
    return encoded.tobytes()


if __name__ == "__main__":
    unittest.main()
