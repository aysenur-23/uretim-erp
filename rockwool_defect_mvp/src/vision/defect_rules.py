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


def detect_glass_burn(image: np.ndarray, roi: np.ndarray, config: AppConfig) -> RuleResult:
    """Detect local dark brown/black burn-like stain regions after illumination correction."""
    del image
    valid_mask = _valid_product_mask(roi)
    valid_area = cv2.countNonZero(valid_mask)
    if valid_area < 64:
        return {**_result(0.0, False, "Cam yanigi analizi icin yeterli urun pikseli yok."), "mask": None}

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    valid_lab = lab[valid_mask > 0]

    median_l = float(np.median(valid_lab[:, 0]))
    local_baseline = _illumination_baseline(gray, valid_mask)
    local_drop = np.clip(local_baseline.astype(np.float32) - gray.astype(np.float32), 0, 255).astype(np.uint8)
    local_values = local_drop[valid_mask > 0]
    local_threshold = max(10.0, float(np.percentile(local_values, 88)))
    corrected_v = np.clip(
        val.astype(np.float32) + (np.median(local_baseline[valid_mask > 0]) - local_baseline.astype(np.float32)),
        0,
        255,
    ).astype(np.uint8)
    corrected_values = corrected_v[valid_mask > 0]
    corrected_dark_threshold = min(
        float(np.percentile(corrected_values, 28)),
        float(np.median(corrected_values) - 8.0),
    )
    brown_or_black = (
        (((hue <= 34) | (hue >= 165)) & (sat >= 35))
        | ((lab[:, :, 0].astype(np.float32) <= median_l - 18.0) & (sat >= 20))
    )
    burn_mask = np.where(
        (valid_mask > 0)
        & brown_or_black
        & (
            ((local_drop >= local_threshold) & (corrected_v <= corrected_dark_threshold))
            | ((lab[:, :, 0].astype(np.float32) <= median_l - 22.0) & (local_drop >= 8))
        ),
        255,
        0,
    ).astype(np.uint8)
    burn_mask = _filter_blob_components(burn_mask, valid_area, min_area_ratio=0.0015, max_aspect_ratio=10.0)

    burn_ratio = float(cv2.countNonZero(burn_mask)) / float(max(1, valid_area))
    largest_ratio = _largest_component_area_ratio(burn_mask, valid_area)
    mask_pixels = burn_mask > 0
    if np.any(mask_pixels):
        mean_mask_sat = float(np.mean(sat[mask_pixels]))
        mean_mask_corrected_v = float(np.mean(corrected_v[mask_pixels]))
    else:
        mean_mask_sat = 0.0
        mean_mask_corrected_v = 255.0
    score = _clip01(max(burn_ratio * 5.0, largest_ratio * 12.0))
    color_is_burn_like = mean_mask_sat >= 70.0 or mean_mask_corrected_v <= 90.0
    compact_burn_like = burn_ratio >= 0.020 and largest_ratio >= 0.010
    is_suspicious = (
        score >= max(0.14, config.color_anomaly_threshold * 0.45)
        and compact_burn_like
        and color_is_burn_like
    )
    if is_suspicious:
        score = max(score, 0.72)
    else:
        score = min(score, 0.24)

    return {
        **_result(
            score,
            is_suspicious,
            "Koyu bolgesel cam yanigi/leke supheli." if is_suspicious else "Cam yanigi sinyali normal.",
        ),
        "strategy": "HSV/Lab koyu kahverengi-siyah bolge ve lokal parlaklik dususu",
        "mask": burn_mask if cv2.countNonZero(burn_mask) > 0 else None,
        "burn_ratio": round(burn_ratio, 4),
        "largest_component_ratio": round(largest_ratio, 4),
        "mean_mask_sat": round(mean_mask_sat, 4),
        "mean_mask_corrected_v": round(mean_mask_corrected_v, 4),
        "local_threshold": round(local_threshold, 4),
        "corrected_dark_threshold": round(corrected_dark_threshold, 4),
    }


def detect_raw_fiber(image: np.ndarray, roi: np.ndarray, config: AppConfig) -> RuleResult:
    """Detect light, desaturated or raised raw-fiber like exposed patches."""
    del image
    valid_mask = _valid_product_mask(roi)
    valid_area = cv2.countNonZero(valid_mask)
    if valid_area < 64:
        return {**_result(0.0, False, "Cig elyaf analizi icin yeterli urun pikseli yok."), "mask": None}

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    valid_gray = gray[valid_mask > 0]
    valid_sat = hsv[:, :, 1][valid_mask > 0]

    bright_threshold = max(float(np.percentile(valid_gray, 90)), float(np.median(valid_gray) + 28.0))
    low_sat_threshold = min(float(np.percentile(valid_sat, 42)), float(np.median(valid_sat) - 12.0))
    fiber_mask = np.where(
        (valid_mask > 0)
        & (gray >= bright_threshold)
        & (hsv[:, :, 1] <= max(12.0, low_sat_threshold)),
        255,
        0,
    ).astype(np.uint8)
    fiber_mask = _filter_blob_components(fiber_mask, valid_area, min_area_ratio=0.001, max_aspect_ratio=12.0)
    relief_mask = _raw_fiber_relief_mask(gray, hsv[:, :, 1], valid_mask)
    combined_mask = cv2.bitwise_or(fiber_mask, relief_mask)

    fiber_ratio = float(cv2.countNonZero(fiber_mask)) / float(max(1, valid_area))
    relief_ratio = float(cv2.countNonZero(relief_mask)) / float(max(1, valid_area))
    combined_ratio = float(cv2.countNonZero(combined_mask)) / float(max(1, valid_area))
    largest_ratio = _largest_component_area_ratio(combined_mask, valid_area)
    score = _clip01(max(fiber_ratio * 3.2, relief_ratio * 4.0, largest_ratio * 10.0))
    is_suspicious = score >= 0.26
    return {
        **_result(
            score,
            is_suspicious,
            "Acik/kabarik cig elyaf bolgesi supheli." if is_suspicious else "Cig elyaf sinyali normal.",
        ),
        "strategy": "Parlak dusuk doygunluklu bolge + lokal kabarik/dokusal lif maskesi",
        "mask": combined_mask if cv2.countNonZero(combined_mask) > 0 else None,
        "raw_fiber_ratio": round(fiber_ratio, 4),
        "raw_fiber_relief_ratio": round(relief_ratio, 4),
        "raw_fiber_combined_ratio": round(combined_ratio, 4),
        "largest_component_ratio": round(largest_ratio, 4),
    }


def detect_shape_deformation(
    image: np.ndarray,
    roi: np.ndarray,
    bbox: tuple[int, int, int, int],
    config: AppConfig,
) -> RuleResult:
    """Detect product shape deformation beyond ordinary edge noise."""
    del image, bbox, config
    valid_mask = _valid_product_mask(roi)
    contour = _largest_contour(valid_mask)
    if contour is None:
        return _result(0.0, False, "Deformasyon analizi icin urun konturu bulunamadi.")

    area = float(cv2.contourArea(contour))
    x, y, width, height = cv2.boundingRect(contour)
    rect_area = float(max(1, width * height))
    extent = area / rect_area
    rect = cv2.minAreaRect(contour)
    box_area = float(max(1.0, rect[1][0] * rect[1][1]))
    rectangularity = area / box_area
    score = _clip01(max((0.86 - extent) * 1.8, (0.88 - rectangularity) * 2.2))
    return {
        **_result(
            score,
            score >= 0.24,
            "Plaka formunda deformasyon supheli." if score >= 0.24 else "Deformasyon sinyali normal.",
        ),
        "strategy": "Kontur extent ve minAreaRect dikdortgensellik kaybi",
        "extent": round(extent, 4),
        "rectangularity": round(rectangularity, 4),
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
    valid_median_gray = float(np.median(valid_pixels))
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
    weak_crack_mask = _weak_crack_candidate_mask(
        gray,
        valid_mask,
        local_dark,
        local_values,
        blackhat,
        blackhat_values,
        valid_pixels,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dark_mask = cv2.bitwise_or(absolute_dark_mask, local_dark_mask)
    dark_mask = cv2.bitwise_or(dark_mask, blackhat_mask)
    dark_mask = cv2.bitwise_or(dark_mask, column_shadow_mask)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, vertical_kernel, iterations=1)
    seed_crack_mask, crack_stats = _filter_crack_components(dark_mask)
    display_crack_mask = seed_crack_mask
    if cv2.countNonZero(seed_crack_mask) > 0:
        fine_crack_mask = _secondary_crack_component_mask(weak_crack_mask, seed_crack_mask, valid_mask)
        display_crack_mask = cv2.bitwise_or(seed_crack_mask, fine_crack_mask)

    valid_area = float(cv2.countNonZero(valid_mask))
    crack_area = float(cv2.countNonZero(seed_crack_mask))
    display_crack_area = float(cv2.countNonZero(display_crack_mask))
    crack_area_ratio = (crack_area / valid_area) if valid_area else 0.0
    display_crack_area_ratio = (display_crack_area / valid_area) if valid_area else 0.0
    dense_texture = crack_stats["component_count"] >= 45
    crack_pixels = seed_crack_mask > 0
    if np.any(crack_pixels):
        crack_mean_gray = float(np.mean(gray[crack_pixels]))
        crack_mean_sat = float(np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 1][crack_pixels]))
    else:
        crack_mean_gray = 255.0
        crack_mean_sat = 0.0
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
    line_evidence = (
        crack_stats["max_length_ratio"] >= 0.26
        or (crack_stats["vertical_coverage"] >= 0.45 and crack_stats["max_length_ratio"] >= 0.22)
        or (
            crack_stats["vertical_coverage"] >= 0.42
            and crack_stats["component_count"] >= 3
            and crack_area_ratio <= 0.018
        )
    )
    broad_stain_like = crack_area_ratio >= 0.08 and crack_stats["max_length_ratio"] < 0.22
    strong_crack_geometry = (
        (crack_stats["vertical_coverage"] >= 0.55 and crack_stats["max_length_ratio"] >= 0.32)
        or (crack_stats["vertical_coverage"] >= 0.50 and crack_stats["component_count"] >= 4)
    )
    dark_valley = (valid_median_gray - crack_mean_gray) >= 8.0 or crack_mean_gray <= 100.0
    light_low_sat_relief = (
        crack_mean_gray >= 135.0
        and crack_mean_sat <= 70.0
        and crack_stats["component_count"] >= 3
        and crack_area_ratio >= 0.007
        and not strong_crack_geometry
    )
    rough_relief = (
        0.018 <= crack_area_ratio <= 0.040
        and crack_stats["max_length_ratio"] >= 0.28
        and crack_stats["component_count"] >= 4
        and not dark_valley
    )
    raw_fiber_relief_like = light_low_sat_relief or rough_relief
    is_suspicious = (
        not dense_texture
        and not broad_stain_like
        and not raw_fiber_relief_like
        and line_evidence
        and (
            score >= config.crack_darkness_threshold
            or crack_stats["max_length_ratio"] >= 0.30
            or (crack_stats["vertical_coverage"] >= 0.34 and score >= 0.24)
            or (
                crack_stats["vertical_coverage"] >= 0.42
                and crack_stats["component_count"] >= 3
                and crack_area_ratio <= 0.018
            )
        )
    )

    return {
        **_result(
            score,
            is_suspicious,
            "Koyu ince cizgisel bolgeler supheli." if is_suspicious else "Catlak sinyali normal.",
        ),
        "strategy": "Dikey black-hat, lokal karanlik vadi ve ince uzun bilesen filtresi",
        "mask": display_crack_mask if display_crack_area > 0 else None,
        "component_count": crack_stats["component_count"],
        "crack_area_ratio": round(crack_area_ratio, 4),
        "display_crack_area_ratio": round(display_crack_area_ratio, 4),
        "dense_texture": dense_texture,
        "broad_stain_like": broad_stain_like,
        "raw_fiber_relief_like": raw_fiber_relief_like,
        "strong_crack_geometry": strong_crack_geometry,
        "dark_valley": dark_valley,
        "light_low_sat_relief": light_low_sat_relief,
        "crack_mean_gray": round(crack_mean_gray, 4),
        "crack_mean_sat": round(crack_mean_sat, 4),
        "valid_median_gray": round(valid_median_gray, 4),
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


def _weak_crack_candidate_mask(
    gray: np.ndarray,
    valid_mask: np.ndarray,
    local_dark: np.ndarray,
    local_values: np.ndarray,
    blackhat: np.ndarray,
    blackhat_values: np.ndarray,
    valid_pixels: np.ndarray,
) -> np.ndarray:
    if local_values.size == 0 or blackhat_values.size == 0 or valid_pixels.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)

    weak_local_threshold = max(7.0, float(np.percentile(local_values, 84)))
    weak_blackhat_threshold = max(6.0, float(np.percentile(blackhat_values, 88)))
    weak_absolute_threshold = min(float(np.percentile(valid_pixels, 24)), float(np.median(valid_pixels) - 12.0))
    weak_mask = np.where(
        (valid_mask > 0)
        & (
            (local_dark >= weak_local_threshold)
            | (blackhat >= weak_blackhat_threshold)
            | (gray <= weak_absolute_threshold)
        ),
        255,
        0,
    ).astype(np.uint8)
    weak_mask = cv2.morphologyEx(
        weak_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 17)),
        iterations=1,
    )
    return weak_mask


def _secondary_crack_component_mask(
    candidate_mask: np.ndarray,
    seed_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    candidates = cv2.bitwise_and(candidate_mask, valid_mask)
    if cv2.countNonZero(seed_mask) == 0 or cv2.countNonZero(candidates) == 0:
        return np.zeros_like(candidate_mask, dtype=np.uint8)

    candidates = cv2.morphologyEx(
        candidates,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )
    candidates = cv2.morphologyEx(
        candidates,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 9)),
        iterations=1,
    )

    output = np.zeros_like(candidate_mask)
    seed_reach = cv2.dilate(seed_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41)), iterations=1)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    image_height = max(1, candidate_mask.shape[0])
    image_width = max(1, candidate_mask.shape[1])
    min_dimension = max(1, min(image_height, image_width))

    for label in range(1, component_count):
        x, y, width, height, area = stats[label]
        if area < 8:
            continue
        long_side = max(width, height)
        short_side = max(1, min(width, height))
        aspect_ratio = long_side / float(short_side)
        fill_ratio = area / float(max(1, width * height))
        thickness_ratio = short_side / float(min_dimension)
        length_ratio = long_side / float(image_height)
        near_seed = np.any(seed_reach[labels == label] > 0)
        verticalish = height >= width * 1.7
        diagonal_or_vertical = height >= 18 and width <= max(18, int(image_width * 0.09))
        if not near_seed:
            continue
        if aspect_ratio < 2.4 and not diagonal_or_vertical:
            continue
        if not verticalish and aspect_ratio < 3.6:
            continue
        if fill_ratio > 0.55 and thickness_ratio > 0.045:
            continue
        if length_ratio < 0.035:
            continue
        output[labels == label] = 255
    return output


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


def _filter_blob_components(
    mask: np.ndarray,
    valid_area: int,
    *,
    min_area_ratio: float,
    max_aspect_ratio: float,
) -> np.ndarray:
    output = np.zeros_like(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(36, int(valid_area * min_area_ratio))

    for label in range(1, component_count):
        _x, _y, width, height, area = stats[label]
        if area < min_area:
            continue
        long_side = max(width, height)
        short_side = max(1, min(width, height))
        if long_side / float(short_side) > max_aspect_ratio:
            continue
        output[labels == label] = 255
    return output


def _raw_fiber_relief_mask(gray: np.ndarray, sat: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    valid_area = cv2.countNonZero(valid_mask)
    if valid_area < 64:
        return np.zeros_like(gray, dtype=np.uint8)

    baseline = _illumination_baseline(gray, valid_mask)
    local_lift = np.clip(gray.astype(np.float32) - baseline.astype(np.float32), 0, 255)
    texture = cv2.absdiff(gray, cv2.GaussianBlur(gray, (0, 0), 3))

    valid_lift = local_lift[valid_mask > 0]
    valid_texture = texture[valid_mask > 0]
    valid_sat = sat[valid_mask > 0]
    valid_gray = gray[valid_mask > 0]
    lift_threshold = max(float(np.percentile(valid_lift, 88)), float(np.median(valid_lift) + 22.0), 35.0)
    texture_threshold = max(float(np.percentile(valid_texture, 88)), 28.0)
    sat_limit = min(float(np.percentile(valid_sat, 90)), float(np.median(valid_sat) + 65.0), 170.0)
    gray_floor = float(np.median(valid_gray) - 60.0)

    mask = np.where(
        (valid_mask > 0)
        & (local_lift >= lift_threshold)
        & (texture >= texture_threshold)
        & (sat <= sat_limit)
        & (gray >= gray_floor),
        255,
        0,
    ).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    output = np.zeros_like(mask)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(24, int(valid_area * 0.00035))
    for label in range(1, component_count):
        x, _y, width, height, area = stats[label]
        if area < min_area:
            continue
        long_side = max(width, height)
        short_side = max(1, min(width, height))
        aspect_ratio = long_side / float(short_side)
        fill_ratio = area / float(max(1, width * height))
        touches_vertical_border = x <= 1 or x + width >= mask.shape[1] - 1
        if aspect_ratio > 24.0 and fill_ratio < 0.22 and touches_vertical_border:
            continue
        output[labels == label] = 255
    return output


def _illumination_baseline(gray: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Estimate slow lighting/shadow variation without treating it as a defect."""
    height, width = gray.shape[:2]
    kernel_size = max(41, int(max(height, width) * 0.28) | 1)
    gray_float = gray.astype(np.float32)
    mask_float = (valid_mask > 0).astype(np.float32)
    weighted = cv2.GaussianBlur(gray_float * mask_float, (kernel_size, kernel_size), 0)
    weights = cv2.GaussianBlur(mask_float, (kernel_size, kernel_size), 0)
    baseline = weighted / np.maximum(weights, 1e-3)

    valid_values = gray[valid_mask > 0]
    fill_value = float(np.median(valid_values)) if valid_values.size else float(np.median(gray))
    baseline[weights < 1e-3] = fill_value
    return np.clip(baseline, 0, 255).astype(np.uint8)


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
    image_width = max(1, mask.shape[1])
    min_dimension = max(1, min(image_height, image_width))
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
        thickness_ratio = short_side / float(min_dimension)
        length_ratio = long_side / float(image_height)
        relative_area = area / float(max(1, mask.size))
        touches_border = (
            x <= 1
            or y <= 1
            or x + width >= image_width - 1
            or y + height >= image_height - 1
        )

        is_crack_like = (
            not touches_border
            and aspect_ratio >= 5.0
            and fill_ratio <= 0.72
            and thickness_ratio <= 0.085
            and relative_area <= 0.035
            and long_side >= min_dimension * 0.12
        )
        if is_crack_like:
            output[labels == label] = 255
            accepted_count += 1
            max_length_ratio = max(max_length_ratio, length_ratio)
            covered_rows[y : y + height] = True

    return output, {
        "component_count": float(accepted_count),
        "max_length_ratio": float(max_length_ratio),
        "vertical_coverage": float(np.mean(covered_rows)),
    }


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
