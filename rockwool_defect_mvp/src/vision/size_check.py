from __future__ import annotations

from typing import Any

import numpy as np

from src.config import AppConfig
from src.vision.product_detection import ProductROI


RuleResult = dict[str, Any]


def detect_size_tolerance(product: ProductROI | None, config: AppConfig) -> RuleResult:
    """Boyut/gönye hatası: sabit kamera px/mm kalibrasyonuyla ölçü kontrolü.

    Saf geometri dedektörüdür; görüntüye bakmaz, ürünün fit edilmiş köşelerini
    kullanır. Kalibrasyon (px_per_mm) veya köşe yoksa devre dışı sonuç döner.
    """
    if (
        not config.size_check_enabled
        or config.px_per_mm <= 0.0
        or product is None
        or len(product.corners) != 4
    ):
        return {
            "score": 0.0,
            "is_suspicious": False,
            "message": "Boyut kontrolu kapali veya px/mm kalibrasyonu yok.",
            "enabled": False,
            "strategy": "Sabit kamera px/mm kalibrasyonu ile olcu ve gonye kontrolu",
        }

    corners = np.array(product.corners, dtype=np.float64)
    tl, tr, br, bl = corners
    width_px = (float(np.linalg.norm(tr - tl)) + float(np.linalg.norm(br - bl))) / 2.0
    height_px = (float(np.linalg.norm(bl - tl)) + float(np.linalg.norm(br - tr))) / 2.0

    measured_w = width_px / config.px_per_mm
    measured_h = height_px / config.px_per_mm

    expected_w = config.expected_width_mm
    expected_h = config.expected_height_mm
    # 90° dönüşe izin ver: (w,h) veya (h,w) eşleşmesinden hatası küçük olanı seç.
    direct_error = abs(measured_w - expected_w) + abs(measured_h - expected_h)
    swapped_error = abs(measured_w - expected_h) + abs(measured_h - expected_w)
    if swapped_error < direct_error:
        expected_w, expected_h = expected_h, expected_w

    tolerance = max(1e-6, config.size_tolerance_mm)
    width_err = max(0.0, abs(measured_w - expected_w) - tolerance)
    height_err = max(0.0, abs(measured_h - expected_h) - tolerance)

    squareness_dev = _max_corner_angle_deviation(corners)
    sq_tolerance = max(1e-6, config.squareness_tolerance_deg)
    sq_err = max(0.0, squareness_dev - sq_tolerance)

    size_component = max(width_err, height_err) / (2.0 * tolerance) * 0.7
    squareness_component = sq_err / (2.0 * sq_tolerance) * 0.3
    score = _clip01(size_component + squareness_component)
    is_suspicious = score > 0.0

    return {
        "score": float(round(score, 4)),
        "is_suspicious": bool(is_suspicious),
        "message": (
            "Olcu/gonye toleransi disinda." if is_suspicious else "Olcu ve gonye toleransta."
        ),
        "enabled": True,
        "strategy": "Sabit kamera px/mm kalibrasyonu ile olcu ve gonye kontrolu",
        "measured_width_mm": round(measured_w, 2),
        "measured_height_mm": round(measured_h, 2),
        "expected_width_mm": round(expected_w, 2),
        "expected_height_mm": round(expected_h, 2),
        "squareness_deg": round(squareness_dev, 3),
        "px_per_mm": round(config.px_per_mm, 5),
    }


def _max_corner_angle_deviation(corners: np.ndarray) -> float:
    max_dev = 0.0
    for i in range(4):
        a = corners[(i - 1) % 4] - corners[i]
        b = corners[(i + 1) % 4] - corners[i]
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        if norm < 1e-6:
            continue
        cos_angle = float(np.dot(a, b) / norm)
        angle = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
        max_dev = max(max_dev, abs(angle - 90.0))
    return max_dev


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
