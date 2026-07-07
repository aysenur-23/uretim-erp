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
    product: "ProductROI | None" = None,
) -> RuleResult:
    """Estimate LOCAL edge irregularity: solidity loss, border gaps, notches.

    Kenar bozukluğu = sınırdaki lokal kusurlar (çentik, kopma, boşluk).
    Global form bozulması deformasyon dedektörünün işidir.
    """
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

    # Köşe çentiği: konveksite kusurlarının en derini / kısa kenar.
    notch_depth_ratio = _max_convexity_defect_ratio(contour)

    # Kenar düzlüğü: fit edilen köşeler varsa her kenarın gerçek köşe-köşe
    # doğrusundan maksimum sapması (lokal kopma/girinti).
    max_side_deviation_ratio = 0.0
    if product is not None and len(getattr(product, "corners", ())) == 4:
        offset = _roi_offset(product)
        max_side_deviation_ratio = _max_side_deviation_ratio(contour, product.corners, offset)

    # Ölü bant: normal perspektif ve maske pürüzü sıfır katkı versin; yalnız
    # belirgin çentik/kopma sinyal üretsin (sağlam üründe false pozitif olmasın).
    notch_signal = max(0.0, notch_depth_ratio - 0.10)
    side_signal = max(0.0, max_side_deviation_ratio - 0.10)
    score = _clip01(
        solidity_loss * 10.0
        + border_gap_ratio * 0.03
        + side_signal * 6.0
        + notch_signal * 8.0
    )
    return {
        **_result(
            score,
            score >= config.edge_damage_threshold,
            "Kenar duzensizligi/kirik supheli." if score >= config.edge_damage_threshold else "Kenar sinyali normal.",
        ),
        "strategy": "Kenar konturu, konveks govde kaybi, sinir bosluk orani ve kose centigi",
        "solidity_loss": round(solidity_loss, 4),
        "border_gap_ratio": round(border_gap_ratio, 4),
        "notch_depth_ratio": round(notch_depth_ratio, 4),
        "max_side_deviation_ratio": round(max_side_deviation_ratio, 4),
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
    # L kanalı yarı ağırlıkla: saf koyuluk cam yanığının işi, kromatik sapma
    # renk anomalisinin. Böylece iki sınıf birbirine karışmaz.
    delta = lab.astype(np.float32) - median.astype(np.float32)
    delta[:, :, 0] *= 0.5
    distance = np.linalg.norm(delta, axis=2)
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
    """Detect local dark brown/black burn-like stain regions after illumination correction.

    Cam yanığı = KOMPAKT koyu kahverengi/siyah bölge. Şekil kapısıyla (kompaktlık)
    ince çizgisel çatlaklardan yapısal olarak ayrılır — bastırma kuralı gerekmez.
    """
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
    # Kompakt blob kapısı: çatlaktan ayrım için düşük en-boy oranı ve dolgu şartı.
    burn_mask = _filter_blob_components(
        burn_mask,
        valid_area,
        min_area_ratio=0.0015,
        max_aspect_ratio=4.0,
        min_fill_ratio=0.35,
        min_short_side=6,
    )

    burn_ratio = float(cv2.countNonZero(burn_mask)) / float(max(1, valid_area))
    largest_ratio = _largest_component_area_ratio(burn_mask, valid_area)
    mask_pixels = burn_mask > 0
    if np.any(mask_pixels):
        mean_mask_sat = float(np.mean(sat[mask_pixels]))
        mean_mask_corrected_v = float(np.mean(corrected_v[mask_pixels]))
    else:
        mean_mask_sat = 0.0
        mean_mask_corrected_v = 255.0

    # Skor sabitleme YOK — sürekli skor kalibrasyonu için.
    score = _clip01(max(burn_ratio * 5.0, largest_ratio * 12.0))
    is_suspicious = score >= config.glass_burn_threshold and largest_ratio >= 0.008

    return {
        **_result(
            score,
            is_suspicious,
            "Koyu bolgesel cam yanigi/leke supheli." if is_suspicious else "Cam yanigi sinyali normal.",
        ),
        "strategy": "HSV/Lab koyu kahverengi-siyah kompakt bolge ve lokal parlaklik dususu",
        "mask": burn_mask if cv2.countNonZero(burn_mask) > 0 else None,
        "burn_ratio": round(burn_ratio, 4),
        "largest_component_ratio": round(largest_ratio, 4),
        "mean_mask_sat": round(mean_mask_sat, 4),
        "mean_mask_corrected_v": round(mean_mask_corrected_v, 4),
    }


def detect_raw_fiber(image: np.ndarray, roi: np.ndarray, config: AppConfig) -> RuleResult:
    """Detect light, desaturated or raised raw-fiber like exposed patches.

    Çiğ elyaf = açık renkli, düşük doygunluklu, lifsi/kabarık homojen olmayan yüzey.
    Üç işaret birleştirilir: soluk yama + doku kabartması + parlak cam iplikleri.
    """
    del image
    valid_mask = _valid_product_mask(roi)
    valid_area = cv2.countNonZero(valid_mask)
    if valid_area < 64:
        return {**_result(0.0, False, "Cig elyaf analizi icin yeterli urun pikseli yok."), "mask": None}

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    sat = hsv[:, :, 1]
    valid_gray = gray[valid_mask > 0]
    valid_sat = sat[valid_mask > 0]
    median_sat = float(np.median(valid_sat))
    median_gray = float(np.median(valid_gray))

    # 1) Soluk yama: çiğ elyaf lekeleri panelden BELİRGİN doygunluk düşüşü gösterir
    # (beyazımsı-gri bulut), parlaklıkları panel medyanına yakın olsa da. Karanlık
    # gölgeleri dışlamak için gray tabanı; gürültüyü elemek için blob filtresi.
    sat_floor = min(float(np.percentile(valid_sat, 22)), median_sat - 10.0)
    pale_mask = np.where(
        (valid_mask > 0)
        & (sat <= sat_floor)
        & (gray >= median_gray - 28.0),
        255,
        0,
    ).astype(np.uint8)
    pale_mask = _filter_blob_components(pale_mask, valid_area, min_area_ratio=0.004, max_aspect_ratio=8.0)

    # 2) Lifli doku kabartması, 3) parlak cam iplikleri.
    relief_mask = _raw_fiber_relief_mask(gray, sat, valid_mask)
    strand_mask = _glass_fiber_mask(gray, sat, valid_mask)

    combined_mask = cv2.bitwise_or(cv2.bitwise_or(pale_mask, relief_mask), strand_mask)
    combined_mask = _filter_blob_components(combined_mask, valid_area, min_area_ratio=0.0008, max_aspect_ratio=40.0)

    pale_ratio = float(cv2.countNonZero(pale_mask)) / float(max(1, valid_area))
    relief_ratio = float(cv2.countNonZero(relief_mask)) / float(max(1, valid_area))
    strand_ratio = float(cv2.countNonZero(strand_mask)) / float(max(1, valid_area))
    largest_ratio = _largest_component_area_ratio(combined_mask, valid_area)

    score = _clip01(
        max(
            pale_ratio * 2.5,
            relief_ratio * 3.4,
            strand_ratio * 2.1,
            largest_ratio * 9.0,
        )
    )
    is_suspicious = score >= config.raw_fiber_threshold

    return {
        **_result(
            score,
            is_suspicious,
            "Cam/cig elyaf bolgesi supheli." if is_suspicious else "Cam/cig elyaf sinyali normal.",
        ),
        "strategy": "Parlak camsi lif + dusuk doygunluklu bolge + lokal kabarik/dokusal lif maskesi",
        "mask": combined_mask if cv2.countNonZero(combined_mask) > 0 else None,
        "raw_fiber_ratio": round(pale_ratio, 4),
        "raw_fiber_relief_ratio": round(relief_ratio, 4),
        "glass_fiber_ratio": round(strand_ratio, 4),
        "largest_component_ratio": round(largest_ratio, 4),
    }


def detect_shape_deformation(
    image: np.ndarray,
    roi: np.ndarray,
    bbox: tuple[int, int, int, int],
    config: AppConfig,
    product: "ProductROI | None" = None,
) -> RuleResult:
    """Detect GLOBAL product shape deformation (bending, warping, skew).

    Deformasyon = ürünün genel formundaki bozulma (eğilme, ezilme, yamulma).
    Global kenar yayı ve köşe çarpıklığına bakar; lokal çentikler kenar
    bozukluğu dedektörünün işidir.
    """
    del image, bbox
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

    # Kenar yayı = eğilme/ezilme. Köşe açısı (gönye) burada KULLANILMAZ: düz bir
    # panelin açılı çekimi köşe açısını değiştirir ama kenarları düz kalır; gönye
    # yalnız kalibrasyonlu size_tolerance dedektöründe ölçülür (perspektif karışmasın).
    bow_ratio = 0.0
    corner_angle_dev = 0.0
    if product is not None and len(getattr(product, "corners", ())) == 4:
        corners = np.array(product.corners, dtype=np.float32)
        corner_angle_dev = _max_corner_angle_deviation(corners)  # sadece raporlama
        offset = _roi_offset(product)
        bow_ratio = _max_edge_bow_ratio(contour, product.corners, offset)

    # Ölü bant: maske pürüzünden kaynaklı küçük yay sıfırlanır.
    bow_signal = max(0.0, bow_ratio - 0.06)
    score = _clip01(
        max(
            (0.84 - extent) * 2.0,
            (0.86 - rectangularity) * 2.4,
            bow_signal * 9.0,
        )
    )
    is_suspicious = score >= config.deformation_threshold
    return {
        **_result(
            score,
            is_suspicious,
            "Plaka formunda deformasyon supheli." if is_suspicious else "Deformasyon sinyali normal.",
        ),
        "strategy": "Kontur extent, dikdortgensellik kaybi, kose carpikligi ve kenar yayi",
        "extent": round(extent, 4),
        "rectangularity": round(rectangularity, 4),
        "corner_angle_dev": round(corner_angle_dev, 4),
        "bow_ratio": round(bow_ratio, 4),
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
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    wide_horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 5))
    # Çatlaklar her yönde olabilir: dikey + yatay + iki diyagonal black-hat'in maksimumu.
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, vertical_kernel)
    blackhat = cv2.max(blackhat, cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, tall_vertical_kernel))
    blackhat = cv2.max(blackhat, cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, horizontal_kernel))
    blackhat = cv2.max(blackhat, cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, wide_horizontal_kernel))
    for angle in (45.0, -45.0):
        diagonal_kernel = _rotated_line_kernel(21, angle)
        blackhat = cv2.max(blackhat, cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, diagonal_kernel))
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
    max_length_ratio = crack_stats["max_length_ratio"]
    vertical_coverage = crack_stats["vertical_coverage"]
    component_count = crack_stats["component_count"]
    dense_texture = component_count >= 45
    texture_penalty = 0.22 if dense_texture else 1.0

    # Çatlak = UZUN sürekli çizgisel yapı. Skoru uzunluk-odaklı hesapla; alan
    # tek başına doygunluk yaratmasın (çiğ elyaf dokusu geniş alan ama kısa
    # parçalardan oluşur). Uzun bileşen yoksa skor sınırlanır (texture ayrımı).
    score = _clip01(
        max(
            max_length_ratio * 1.6,
            crack_area_ratio * 16.0,
            vertical_coverage * 0.55,
        )
        * texture_penalty
    )
    if max_length_ratio < 0.20:
        score = min(score, 0.34)
    if dense_texture:
        score = min(score, 0.30)

    # Şüphe için ya gerçek uzun bir çatlak (self-contained, çatlağı tanımlayan
    # uzunluk özelliği) ya da uzunca-kompakt bir koyu ayrışma gerekir.
    has_long_crack = max_length_ratio >= 0.24 and component_count <= 40
    strong_compact = crack_area_ratio >= 0.06 and max_length_ratio >= 0.16
    is_suspicious = (not dense_texture) and score >= 0.30 and (has_long_crack or strong_compact)

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

        # Uzama oranını eksene hizalı bbox yerine minAreaRect'ten hesapla ki
        # diyagonal çatlaklar (bbox'u neredeyse kare) elenmesin.
        component_points = np.column_stack(np.where(labels == label)[::-1]).astype(np.int32)
        rect = cv2.minAreaRect(component_points)
        rect_long = max(rect[1])
        rect_short = max(1.0, min(rect[1]))
        aspect_ratio = rect_long / rect_short
        fill_ratio = area / float(max(1, width * height))

        is_crack_like = (
            (aspect_ratio >= 2.4 and fill_ratio <= 0.88)
            or (max(width, height) >= image_height * 0.14 and rect_short <= max(24.0, image_height * 0.16))
        )
        if is_crack_like:
            output[labels == label] = 255
            accepted_count += 1
            max_length_ratio = max(max_length_ratio, rect_long / float(image_height))
            covered_rows[y : y + height] = True

    return output, {
        "component_count": float(accepted_count),
        "max_length_ratio": float(max_length_ratio),
        "vertical_coverage": float(np.mean(covered_rows)),
    }


def _filter_blob_components(
    mask: np.ndarray,
    valid_area: int,
    *,
    min_area_ratio: float,
    max_aspect_ratio: float,
    min_fill_ratio: float = 0.0,
    min_short_side: int = 0,
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
        if short_side < min_short_side:
            continue
        if min_fill_ratio > 0.0:
            fill_ratio = area / float(max(1, width * height))
            if fill_ratio < min_fill_ratio:
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


def _glass_fiber_mask(gray: np.ndarray, sat: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    valid_area = cv2.countNonZero(valid_mask)
    if valid_area < 64:
        return np.zeros_like(gray, dtype=np.uint8)

    baseline = _illumination_baseline(gray, valid_mask)
    local_lift = np.clip(gray.astype(np.float32) - baseline.astype(np.float32), 0, 255).astype(np.uint8)
    vertical_tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 21)))
    horizontal_tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5)))
    bright_fiber = cv2.max(local_lift, cv2.max(vertical_tophat, horizontal_tophat))

    valid_bright = bright_fiber[valid_mask > 0]
    valid_sat = sat[valid_mask > 0]
    valid_gray = gray[valid_mask > 0]
    bright_threshold = max(float(np.percentile(valid_bright, 95)), float(np.median(valid_bright) + 30.0), 38.0)
    sat_limit = min(float(np.percentile(valid_sat, 66)), float(np.median(valid_sat) + 34.0), 128.0)
    gray_floor = max(float(np.percentile(valid_gray, 62)), float(np.median(valid_gray) + 8.0))

    mask = np.where(
        (valid_mask > 0)
        & (bright_fiber >= bright_threshold)
        & (sat <= sat_limit)
        & (gray >= gray_floor),
        255,
        0,
    ).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9)), iterations=1)

    output = np.zeros_like(mask)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(16, int(valid_area * 0.00028))
    for label in range(1, component_count):
        _x, _y, width, height, area = stats[label]
        if area < min_area:
            continue
        long_side = max(width, height)
        short_side = max(1, min(width, height))
        aspect_ratio = long_side / float(short_side)
        fill_ratio = area / float(max(1, width * height))
        if aspect_ratio < 1.8 and fill_ratio > 0.68:
            continue
        if area < max(24, int(valid_area * 0.00045)) and aspect_ratio < 2.8:
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


def _roi_offset(product: "ProductROI") -> tuple[int, int]:
    """Görüntü koordinatlarını ROI (shape_roi) koordinatlarına taşımak için ofset."""
    shape_bbox = getattr(product, "shape_bbox", getattr(product, "bbox", (0, 0, 0, 0)))
    return int(shape_bbox[0]), int(shape_bbox[1])


def _max_convexity_defect_ratio(contour: np.ndarray) -> float:
    """En derin konveksite kusuru / kısa kenar (köşe çentiği göstergesi)."""
    if len(contour) < 4:
        return 0.0
    try:
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        if hull_indices is None or len(hull_indices) < 3:
            return 0.0
        defects = cv2.convexityDefects(contour, hull_indices)
    except cv2.error:
        return 0.0
    if defects is None or defects.size == 0:
        return 0.0
    _x, _y, width, height = cv2.boundingRect(contour)
    short_side = float(max(1, min(width, height)))
    depths = defects.reshape(-1, 4)[:, 3].astype(np.float32)
    max_depth = float(np.max(depths)) / 256.0  # OpenCV derinliği 8-bit sabitli
    return _clip01(max_depth / short_side)


def _assign_points_to_sides(points: np.ndarray, corners: np.ndarray) -> np.ndarray:
    segments = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    distances = np.stack(
        [_point_to_segment_distance(points, seg[0], seg[1]) for seg in segments],
        axis=1,
    )
    return np.argmin(distances, axis=1)


def _point_to_segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    length_sq = float(ab[0] ** 2 + ab[1] ** 2)
    if length_sq <= 1e-6:
        return np.hypot(points[:, 0] - a[0], points[:, 1] - a[1])
    t = ((points[:, 0] - a[0]) * ab[0] + (points[:, 1] - a[1]) * ab[1]) / length_sq
    t = np.clip(t, 0.0, 1.0)
    proj_x = a[0] + t * ab[0]
    proj_y = a[1] + t * ab[1]
    return np.hypot(points[:, 0] - proj_x, points[:, 1] - proj_y)


def _max_side_deviation_ratio(
    contour: np.ndarray,
    corners: tuple[tuple[float, float], ...],
    offset: tuple[int, int],
) -> float:
    """Her kenarın gerçek köşe-köşe doğrusundan maksimum sapması / kısa kenar."""
    points = contour.reshape(-1, 2).astype(np.float32)
    corners_roi = np.array(corners, dtype=np.float32) - np.array(offset, dtype=np.float32)
    _x, _y, width, height = cv2.boundingRect(contour)
    short_side = float(max(1, min(width, height)))
    assignments = _assign_points_to_sides(points, corners_roi)

    max_dev = 0.0
    for side_index in range(4):
        side_points = points[assignments == side_index]
        if len(side_points) < 8:
            continue
        a = corners_roi[side_index]
        b = corners_roi[(side_index + 1) % 4]
        dist = _point_to_segment_distance(side_points, a, b)
        max_dev = max(max_dev, float(np.percentile(dist, 96)))
    return _clip01(max_dev / short_side)


def _max_edge_bow_ratio(
    contour: np.ndarray,
    corners: tuple[tuple[float, float], ...],
    offset: tuple[int, int],
) -> float:
    """Kenar orta bandının düz köşe-köşe doğrusundan ortalama sapması / kenar boyu."""
    points = contour.reshape(-1, 2).astype(np.float32)
    corners_roi = np.array(corners, dtype=np.float32) - np.array(offset, dtype=np.float32)
    assignments = _assign_points_to_sides(points, corners_roi)

    max_bow = 0.0
    for side_index in range(4):
        side_points = points[assignments == side_index]
        if len(side_points) < 8:
            continue
        a = corners_roi[side_index]
        b = corners_roi[(side_index + 1) % 4]
        side_length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        if side_length < 1.0:
            continue
        dist = _point_to_segment_distance(side_points, a, b)
        # Yay = ortalama sapma (tek çentik değil, tüm kenarın eğilmesi).
        max_bow = max(max_bow, float(np.mean(dist)) / side_length)
    return _clip01(max_bow)


def _max_corner_angle_deviation(corners: np.ndarray) -> float:
    """4 köşenin 90°'den maksimum sapması (derece) — global çarpıklık/gönye."""
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


def _rotated_line_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Verilen açıda ince çizgi biçimli morfoloji çekirdeği (diyagonal çatlaklar)."""
    base = np.zeros((length, length), dtype=np.uint8)
    base[length // 2, :] = 1
    center = (length / 2.0 - 0.5, length / 2.0 - 0.5)
    rotation = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(base, rotation, (length, length), flags=cv2.INTER_NEAREST)
    if not rotated.any():
        rotated[length // 2, length // 2] = 1
    return rotated


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
