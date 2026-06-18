from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.config import AppConfig


AnomalyResult = dict[str, Any]


def detect_local_anomaly(roi: np.ndarray, config: AppConfig) -> AnomalyResult:
    """Create a lightweight local anomaly score and heatmap for the product ROI."""
    valid_mask = _valid_product_mask(roi)
    analysis_mask = _inner_product_mask(valid_mask)
    valid_area = cv2.countNonZero(analysis_mask)
    if roi.size == 0 or valid_area < 64:
        return {
            "score": 0.0,
            "is_suspicious": False,
            "message": "Anomali analizi icin yeterli urun pikseli yok.",
            "strategy": "Ic ROI'de lokal Lab residual heatmap",
            "heatmap": None,
        }

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    baseline = cv2.GaussianBlur(lab, (41, 41), 0)
    residual = np.linalg.norm(lab - baseline, axis=2)
    residual[analysis_mask == 0] = 0.0
    residual = cv2.GaussianBlur(residual, (17, 17), 0)

    valid_residual = residual[analysis_mask > 0]
    median = float(np.median(valid_residual))
    mad = float(np.median(np.abs(valid_residual - median))) + 1e-6
    high_threshold = max(float(np.percentile(valid_residual, 96)), median + mad * 4.0)
    hot_mask = ((residual >= high_threshold) & (analysis_mask > 0)).astype(np.uint8) * 255
    hot_mask = _filter_local_anomaly_components(hot_mask, valid_area)

    hot_area = float(cv2.countNonZero(hot_mask))
    hot_ratio = hot_area / float(valid_area)
    largest_ratio = _largest_component_area_ratio(hot_mask, valid_area)
    score = _clip01(max(hot_ratio * 10.0, largest_ratio * 24.0))

    return {
        "score": round(score, 4),
        "is_suspicious": score >= config.local_anomaly_threshold,
        "message": (
            "Yerel anomali/lekelenme bolgesi supheli."
            if score >= config.local_anomaly_threshold
            else "Yerel anomali sinyali normal."
        ),
        "strategy": "Ic ROI'de lokal Lab residual heatmap ve kompakt sicak bolge filtresi",
        "hot_ratio": round(hot_ratio, 4),
        "largest_component_ratio": round(largest_ratio, 4),
        "heatmap": _create_heatmap(residual, hot_mask),
    }


def _create_heatmap(residual: np.ndarray, hot_mask: np.ndarray) -> np.ndarray:
    valid_values = residual[hot_mask > 0]
    if valid_values.size == 0:
        return cv2.cvtColor(hot_mask, cv2.COLOR_GRAY2BGR)

    upper = max(float(np.percentile(valid_values, 98)), 1.0)
    normalized = np.clip((residual / upper) * 255.0, 0, 255).astype(np.uint8)
    normalized[hot_mask == 0] = 0
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    heatmap[hot_mask == 0] = 0
    return heatmap


def _filter_local_anomaly_components(mask: np.ndarray, valid_area: int) -> np.ndarray:
    output = np.zeros_like(mask)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(48, int(valid_area * 0.0015))

    for label in range(1, component_count):
        _x, _y, width, height, area = stats[label]
        if area < min_area:
            continue

        long_side = max(width, height)
        short_side = max(1, min(width, height))
        if long_side / float(short_side) > 18.0:
            continue

        output[labels == label] = 255

    return output


def _largest_component_area_ratio(mask: np.ndarray, valid_area: int) -> float:
    component_count, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if component_count <= 1 or valid_area <= 0:
        return 0.0

    largest_area = max(float(stats[label, cv2.CC_STAT_AREA]) for label in range(1, component_count))
    return largest_area / float(valid_area)


def _valid_product_mask(roi: np.ndarray) -> np.ndarray:
    if roi.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def _inner_product_mask(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape[:2]
    margin = max(5, min(height, width) // 30)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin, margin))
    return cv2.erode(mask, kernel, iterations=1)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
