from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.config import AppConfig
from src.decision.decision_engine import DecisionResult
from src.storage.sqlite_store import SQLiteStore


class DatasetManager:
    """Saves inspection images and creates operator-confirmed records."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.raw_dir = config.database_path.parents[1] / "raw"
        self.overlay_dir = config.output_overlay_path
        self.store = SQLiteStore(config.database_path)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.overlay_dir.mkdir(parents=True, exist_ok=True)

    def save_inspection(
        self,
        raw_frame: np.ndarray,
        overlay_frame: np.ndarray,
        product_type: str,
        decision: DecisionResult,
        rule_results: dict[str, dict],
        operator_label: str,
        operator_note: str,
    ) -> int:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = self.raw_dir / f"{timestamp}_raw.jpg"
        overlay_path = self.overlay_dir / f"{timestamp}_overlay.jpg"

        self._write_image(image_path, raw_frame)
        self._write_image(overlay_path, overlay_frame)

        record = {
            "timestamp": timestamp,
            "image_path": str(image_path),
            "overlay_path": str(overlay_path),
            "product_type": product_type,
            "model_result": decision.label,
            "anomaly_score": decision.anomaly_score,
            "edge_damage_score": _rule_score(rule_results, "edge_damage"),
            "color_anomaly_score": _rule_score(rule_results, "color_anomaly"),
            "crack_score": _rule_score(rule_results, "dark_crack"),
            "local_anomaly_score": _rule_score(rule_results, "local_anomaly"),
            "operator_label": operator_label,
            "operator_note": operator_note,
            "is_model_wrong": 0,
            "is_confirmed": 1,
        }
        return self.store.insert_inspection_record(record)

    def _write_image(self, path: Path, image: np.ndarray) -> None:
        success = cv2.imwrite(str(path), image)
        if not success:
            raise RuntimeError(f"Image could not be written: {path}")


def _rule_score(rule_results: dict[str, dict], name: str) -> float:
    return float(rule_results.get(name, {}).get("score", 0.0))
