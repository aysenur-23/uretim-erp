from __future__ import annotations

import cv2
import numpy as np

from src.config import AppConfig
from src.vision.inspection_pipeline import product_display_mask, product_display_roi
from src.vision.product_detection import find_product_roi


def calibrate_product_hsv(frame: np.ndarray, config: AppConfig) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Estimate a stable HSV band from the detected product ROI."""
    product = find_product_roi(frame, config)
    if product is None:
        raise ValueError("Ürün bulunamadığı için renk kalibrasyonu yapılamadı.")

    roi = product_display_roi(product)
    mask = product_display_mask(product)
    if mask.shape[:2] != roi.shape[:2]:
        mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask > 0]
    if len(pixels) < 128:
        raise ValueError("Renk kalibrasyonu için yeterli ürün pikseli yok.")

    lower = np.percentile(pixels, 8, axis=0)
    upper = np.percentile(pixels, 92, axis=0)
    padding = np.array([8, 18, 22])
    lower = np.maximum(lower - padding, [0, 0, 0])
    upper = np.minimum(upper + padding, [179, 255, 255])
    return tuple(int(value) for value in lower), tuple(int(value) for value in upper)
