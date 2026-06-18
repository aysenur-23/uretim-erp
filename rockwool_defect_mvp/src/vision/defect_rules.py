from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.config import AppConfig


RuleResult = dict[str, Any]


def detect_edge_damage(
    image: np.ndarray,
    roi: np.ndarray,
    bbox: tuple[int, int, int, int],
    config: AppConfig,
) -> RuleResult:
    """Estimate edge irregularity with contour solidity and border gaps."""
    del image, bbox
    valid_mask = _valid_product_mask(roi)
    contour = _largest_contour(valid_mask)
    if contour is None:
        return _result(0.0, False, "Kenar analizi icin urun maskesi bulunamadi.")

    area = float(cv2.contourArea(contour))
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity_loss = 1.0 - (area / hull_area) if hull_area else 0.0
    border_gap_ratio = _border_gap_ratio(valid_mask)

    score = _clip01(solidity_loss * 10.0 + border_gap_ratio * 0.03)
    return {
        **_result(
        score,
        score >= config.edge_damage_threshold,
        "Kenar duzensizligi/kirik supheli." if score >= config.edge_damage_threshold else "Kenar sinyali normal.",
        ),
        "strategy": "Kenar konturu, konveks govde kaybi ve sinir bosluk orani",
        "solidity_loss": round(solidity_loss, 4),
        "border_gap_ratio": round(border_gap_ratio, 4),
    }


def detect_color_anomaly(image: np.ndarray, roi: np.ndarray, config: AppConfig) -> RuleResult:
    """Detect broad color deviations in Lab color space."""
    del image
    valid_mask = _valid_product_mask(roi)
    valid_pixels = roi[valid_mask > 0]
    if len(valid_pixels) < 64:
        return _result(0.0, False, "Renk analizi icin yeterli urun pikseli yok.")

    lab_pixels = cv2.cvtColor(valid_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    median = np.median(lab_pixels, axis=0)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    distance = np.linalg.norm(lab.astype(np.float32) - median.astype(np.float32), axis=2)
    distance[valid_mask == 0] = 0.0

    valid_distance = distance[valid_mask > 0]
    robust_scale = np.percentile(valid_distance, 75) + 1e-6
    distance_threshold = max(float(np.percentile(valid_distance, 94)), robust_scale * 2.6)
    anomaly_mask = np.where((distance >= distance_threshold) & (valid_mask > 0), 255, 0).astype(np.uint8)
    anomaly_mask = _filter_color_components(anomaly_mask, cv2.countNonZero(valid_mask))

    anomalous_ratio = float(cv2.countNonZero(anomaly_mask)) / float(max(1, cv2.countNonZero(valid_mask)))
    largest_ratio = _largest_component_area_ratio(anomaly_mask, cv2.countNonZero(valid_mask))
    spread_score = float(np.percentile(valid_distance, 95) / 80.0)
    score = _clip01(max(anomalous_ratio * 3.0, largest_ratio * 9.0, spread_score * 0.30))

    return {
        **_result(
            score,
            score >= config.color_anomaly_threshold,
            "Belirgin renk/leke sapmasi supheli." if score >= config.color_anomaly_threshold else "Renk sinyali normal.",
        ),
        "strategy": "Lab renk uzayinda panel medyanindan sapan bolgesel leke maskesi",
        "mask": anomaly_mask if cv2.countNonZero(anomaly_mask) > 0 else None,
        "anomalous_ratio": round(anomalous_ratio, 4),
        "largest_component_ratio": round(largest_ratio, 4),
        "distance_threshold": round(distance_threshold, 4),
    }


def detect_dark_crack_like_regions(image: np.ndarray, roi: np.ndarray, config: AppConfig) -> RuleResult:
    """Detect dark thin connected components that may indicate cracks."""
    del image
    valid_mask = _valid_product_mask(roi)
    if cv2.countNonZero(valid_mask) < 64:
        return {
            **_result(0.0, False, "Catlak analizi icin yeterli urun pikseli yok."),
            "mask": None,
        }

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    valid_pixels = gray[valid_mask > 0]
    percentile_threshold = np.percentile(valid_pixels, max(1.0, config.crack_darkness_threshold * 45.0))
    median_threshold = float(np.median(valid_pixels)) - 25.0
    darkness_threshold = min(percentile_threshold, median_threshold)
    absolute_dark_mask = np.where((gray <= darkness_threshold) & (valid_mask > 0), 255, 0).astype(np.uint8)

    baseline = cv2.GaussianBlur(gray, (31, 31), 0)
    local_dark = cv2.subtract(baseline, gray)
    local_values = local_dark[valid_mask > 0]
    local_threshold = max(12.0, float(np.percentile(local_values, 92)))
    local_dark_mask = np.where((local_dark >= local_threshold) & (valid_mask > 0), 255, 0).astype(np.uint8)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 17))
    tall_vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 31))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, vertical_kernel)
    tall_blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, tall_vertical_kernel)
    blackhat = cv2.max(blackhat, tall_blackhat)
    blackhat_values = blackhat[valid_mask > 0]
    blackhat_threshold = max(10.0, float(np.percentile(blackhat_values, 94)))
    blackhat_mask = np.where((blackhat >= blackhat_threshold) & (valid_mask > 0), 255, 0).astype(np.uint8)

    column_shadow_mask = _vertical_shadow_mask(gray, valid_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dark_mask = cv2.bitwise_or(absolute_dark_mask, local_dark_mask)
    dark_mask = cv2.bitwise_or(dark_mask, blackhat_mask)
    dark_mask = cv2.bitwise_or(dark_mask, column_shadow_mask)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, vertical_kernel, iterations=1)
    crack_mask, crack_stats = _filter_crack_components(dark_mask)

    valid_area = float(cv2.countNonZero(valid_mask))
    crack_area = float(cv2.countNonZero(crack_mask))
    crack_area_ratio = (crack_area / valid_area) if valid_area else 0.0
    dense_texture = crack_stats["component_count"] >= 45
    texture_penalty = 0.22 if dense_texture else 1.0
    score = _clip01(
        max(
            crack_area_ratio * 42.0,
            crack_stats["max_length_ratio"] * 1.08,
            crack_stats["vertical_coverage"] * 0.92,
            min(crack_stats["component_count"], 8.0) * 0.10,
        )
        * texture_penalty
    )
    if dense_texture:
        score = min(score, 0.30)
    is_suspicious = (
        (score >= config.crack_darkness_threshold and not dense_texture)
        or (crack_area_ratio >= 0.10 and crack_stats["component_count"] >= 3 and not dense_texture)
        or (crack_stats["max_length_ratio"] >= 0.24 and crack_stats["component_count"] <= 40)
        or (crack_stats["vertical_coverage"] >= 0.30 and score >= 0.22 and crack_stats["component_count"] <= 40)
        or (crack_stats["component_count"] >= 3 and score >= 0.22 and not dense_texture)
    )

    return {
        **_result(
            score,
            is_suspicious,
            "Koyu ince cizgisel bolgeler supheli." if is_suspicious else "Catlak sinyali normal.",
        ),
        "strategy": "Dikey black-hat, lokal karanlik vadi ve ince uzun bilesen filtresi",
        "mask": crack_mask if crack_area > 0 else None,
        "component_count": crack_stats["component_count"],
        "crack_area_ratio": round(crack_area_ratio, 4),
        "dense_texture": dense_texture,
        "max_length_ratio": round(crack_stats["max_length_ratio"], 4),
        "vertical_coverage": round(crack_stats["vertical_coverage"], 4),
    }


def _vertical_shadow_mask(gray: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Highlight long local dark valleys without turning texture into a full mask."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    vertical_close = cv2.morphologyEx(
        blurred,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 35)),
    )
    valleys = cv2.subtract(vertical_close, blurred)
    values = valleys[valid_mask > 0]
    if values.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)

    threshold = max(9.0, float(np.percentile(values, 93)))
    mask = np.where((valleys >= threshold) & (valid_mask > 0), 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 19)),
        iterations=1,
    )
    return mask


def _filter_color_components(mask: np.ndarray, valid_area: int) -> np.ndarray:
    output = np.zeros_like(mask)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(36, int(valid_area * 0.0012))

    for label in range(1, component_count):
        _x, _y, width, height, area = stats[label]
        if area < min_area:
            continue

        long_side = max(width, height)
        short_side = max(1, min(width, height))
        if long_side / float(short_side) > 14.0:
            continue

        output[labels == label] = 255

    return output


def _result(score: float, is_suspicious: bool, message: str) -> RuleResult:
    return {
        "score": float(round(_clip01(score), 4)),
        "is_suspicious": bool(is_suspicious),
        "message": message,
    }


def _valid_product_mask(roi: np.ndarray) -> np.ndarray:
    if roi.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _border_gap_ratio(mask: np.ndarray) -> float:
    height, width = mask.shape[:2]
    band = max(2, min(height, width) // 25)
    border_pixels = np.concatenate(
        [
            mask[:band, :].ravel(),
            mask[-band:, :].ravel(),
            mask[:, :band].ravel(),
            mask[:, -band:].ravel(),
        ]
    )
    return float(np.mean(border_pixels == 0))


def _largest_component_area_ratio(mask: np.ndarray, valid_area: int) -> float:
    component_count, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if component_count <= 1 or valid_area <= 0:
        return 0.0

    largest_area = max(float(stats[label, cv2.CC_STAT_AREA]) for label in range(1, component_count))
    return largest_area / float(valid_area)


def _filter_crack_components(mask: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    output = np.zeros_like(mask)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    image_height = max(1, mask.shape[0])
    accepted_count = 0
    max_length_ratio = 0.0
    covered_rows = np.zeros(image_height, dtype=bool)

    for label in range(1, component_count):
        x, y, width, height, area = stats[label]
        if area < 10:
            continue

        long_side = max(width, height)
        short_side = max(1, min(width, height))
        aspect_ratio = long_side / float(short_side)
        fill_ratio = area / float(max(1, width * height))

        is_crack_like = (
            (aspect_ratio >= 2.4 and fill_ratio <= 0.88)
            or (height >= image_height * 0.14 and width <= max(24, image_height * 0.16))
        )
        if is_crack_like:
            output[labels == label] = 255
            accepted_count += 1
            max_length_ratio = max(max_length_ratio, long_side / float(image_height))
            covered_rows[y : y + height] = True

    return output, {
        "component_count": float(accepted_count),
        "max_length_ratio": float(max_length_ratio),
        "vertical_coverage": float(np.mean(covered_rows)),
    }


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
