from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.config import AppConfig
from src.vision.preprocessing import blur_image, normalize_lighting


@dataclass(frozen=True)
class ProductROI:
    roi: np.ndarray
    bbox: tuple[int, int, int, int]
    rotated_box: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]
    contour: np.ndarray
    mask: np.ndarray
    shape_mask: np.ndarray
    shape_bbox: tuple[int, int, int, int]
    shape_roi: np.ndarray
    shape_mask_roi: np.ndarray
    area_ratio: float
    # Geriye uyumlu ek alanlar (default'lu).
    corners: tuple[tuple[float, float], ...] = ()  # TL, TR, BR, BL (görüntü koordinatı)
    roi_confidence: float = 0.0
    detection_method: str = "color_otsu"  # "background_diff" | "color_otsu" | "grabcut"


def find_product_roi(image: np.ndarray, config: AppConfig) -> ProductROI | None:
    """Find the main product region using the largest foreground contour.

    Tespit kaskadı: (1) sabit kamera boş-bant referansı varsa arka plan farkı,
    (2) Otsu/HSV maskesi, (3) çerçeveyi dolduran kontur için GrabCut geri düşüşü.
    """
    if image.size == 0:
        raise ValueError("Cannot detect product in an empty image.")

    detection_method = "color_otsu"
    mask = _build_background_diff_mask(image, config)
    contour = _largest_contour(mask) if mask is not None else None
    if contour is not None and _min_area_ratio(contour, image.shape) >= config.min_product_area_ratio:
        detection_method = "background_diff"
    else:
        mask = _build_product_mask(image, config)
        contour = _largest_contour(mask)

    if contour is not None and _is_overfilled_bbox(contour, image.shape):
        fallback_mask = _build_grabcut_mask(image)
        fallback_contour = _largest_contour(fallback_mask)
        if fallback_contour is not None:
            mask = fallback_mask
            contour = fallback_contour
            detection_method = "grabcut"

    if contour is None:
        return None

    x, y, width, height = cv2.boundingRect(contour)
    rotated_box = _rotated_box_points(contour)
    shape_mask = _mask_from_rotated_box(image.shape, rotated_box)
    shape_mask = _refine_shape_mask_by_panel_color(image, shape_mask, config)
    rotated_box = _box_from_mask(shape_mask) or rotated_box
    shape_bbox, shape_roi, shape_mask_roi = _extract_shape_roi(image, shape_mask)
    contour_area = float(cv2.contourArea(contour))
    image_area = float(image.shape[0] * image.shape[1])
    area_ratio = contour_area / image_area if image_area else 0.0

    if area_ratio < config.min_product_area_ratio:
        return None

    corners = _fit_quadrilateral_corners(contour, image.shape)
    if corners is not None:
        # minAreaRect 90°'ye zorlar; fit edilen köşeler gerçek gönye bilgisi taşır.
        rotated_box = tuple((int(round(px)), int(round(py))) for px, py in corners)
    corners_tuple = tuple((float(px), float(py)) for px, py in corners) if corners is not None else ()

    roi_confidence = estimate_roi_confidence(
        image, shape_mask, contour, corners, detection_method
    )

    roi = image[y : y + height, x : x + width].copy()
    return ProductROI(
        roi=roi,
        bbox=(x, y, width, height),
        rotated_box=rotated_box,
        contour=contour,
        mask=mask,
        shape_mask=shape_mask,
        shape_bbox=shape_bbox,
        shape_roi=shape_roi,
        shape_mask_roi=shape_mask_roi,
        area_ratio=area_ratio,
        corners=corners_tuple,
        roi_confidence=roi_confidence,
        detection_method=detection_method,
    )


def _min_area_ratio(contour: np.ndarray, image_shape: tuple[int, ...]) -> float:
    image_area = float(image_shape[0] * image_shape[1])
    return float(cv2.contourArea(contour)) / image_area if image_area else 0.0


def _build_background_diff_mask(image: np.ndarray, config: AppConfig) -> np.ndarray | None:
    """Sabit kamera boş-bant referansı ile Lab farkı maskesi (yoksa None).

    En güvenilir ayrım: kamera sabit olduğundan boş bandın referans görüntüsüyle
    fark alınır. Lab uzayında, aydınlatma kaymasına dayanıklı olması için L kanalı
    daha düşük ağırlıkla değerlendirilir.
    """
    reference_path = config.background_reference_path
    if reference_path is None:
        return None
    reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if reference is None:
        return None

    height, width = image.shape[:2]
    if reference.shape[:2] != (height, width):
        reference = cv2.resize(reference, (width, height), interpolation=cv2.INTER_AREA)

    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = np.abs(image_lab - reference_lab)
    weighted = diff[:, :, 0] * 0.5 + diff[:, :, 1] * 1.0 + diff[:, :, 2] * 1.0
    weighted = cv2.GaussianBlur(weighted, (7, 7), 0)
    weighted_u8 = np.clip(weighted, 0, 255).astype(np.uint8)

    otsu_value, _ = cv2.threshold(weighted_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(18.0, float(otsu_value))
    mask = (weighted_u8 >= threshold).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def _build_product_mask(image: np.ndarray, config: AppConfig) -> np.ndarray:
    normalized = normalize_lighting(image)
    blurred = blur_image(normalized, kernel_size=7)
    color_mask = _build_product_color_mask(image, config)

    _, bright_mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    _, dark_mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    mask = _choose_likely_product_mask(bright_mask, dark_mask, color_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def _build_product_color_mask(image: np.ndarray, config: AppConfig) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array(config.product_hsv_lower, dtype=np.uint8)
    upper = np.array(config.product_hsv_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)


def _choose_likely_product_mask(
    bright_mask: np.ndarray,
    dark_mask: np.ndarray,
    color_mask: np.ndarray,
) -> np.ndarray:
    color_score = _largest_area_ratio(color_mask)
    if color_score >= 0.03:
        return color_mask

    bright_score = _largest_area_ratio(bright_mask)
    dark_score = _largest_area_ratio(dark_mask)
    base_mask = bright_mask if bright_score >= dark_score else dark_mask

    if color_score > 0.0:
        color_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        expanded_color = cv2.dilate(color_mask, color_kernel, iterations=1)
        refined = cv2.bitwise_and(base_mask, expanded_color)
        if _largest_area_ratio(refined) > 0.0:
            return refined

    return base_mask


def _largest_area_ratio(mask: np.ndarray) -> float:
    contour = _largest_contour(mask)
    if contour is None:
        return 0.0

    image_area = float(mask.shape[0] * mask.shape[1])
    return float(cv2.contourArea(contour)) / image_area if image_area else 0.0


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _rotated_box_points(
    contour: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    rectangle_box = _rectangle_candidate_box(contour)
    if rectangle_box is not None:
        return rectangle_box

    points = contour.reshape(-1, 2).astype(np.float32)
    if len(points) < 8:
        rotated_rect = cv2.minAreaRect(contour)
        box_points = cv2.boxPoints(rotated_rect)
        return tuple((int(x), int(y)) for x, y in box_points)

    mean = points.mean(axis=0)
    centered = points - mean
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axes = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    projected = centered @ axes

    # Trim contour extremes so fibers, shadows, and supports do not dominate the box.
    main_min = np.percentile(projected[:, 0], 2)
    main_max = np.percentile(projected[:, 0], 98)
    side_min = np.percentile(projected[:, 1], 15)
    side_max = np.percentile(projected[:, 1], 92)

    corners = np.array(
        [
            [main_min, side_min],
            [main_max, side_min],
            [main_max, side_max],
            [main_min, side_max],
        ],
        dtype=np.float32,
    )
    box_points = corners @ axes.T + mean
    return tuple((int(x), int(y)) for x, y in box_points)


def _rectangle_candidate_box(
    contour: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None

    best_score = -1.0
    best_box: np.ndarray | None = None
    for epsilon_ratio in (0.01, 0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
        if len(approx) < 4:
            continue

        rect = cv2.minAreaRect(approx)
        rect_width, rect_height = rect[1]
        rect_area = float(rect_width * rect_height)
        if rect_area <= 0:
            continue

        long_side = max(rect_width, rect_height)
        short_side = max(1.0, min(rect_width, rect_height))
        aspect_ratio = long_side / short_side
        if aspect_ratio < 1.2 or aspect_ratio > 8.0:
            continue

        approx_area = float(cv2.contourArea(approx))
        rectangularity = approx_area / rect_area
        simplicity_bonus = 0.18 if len(approx) == 4 else max(0.0, 0.10 - (len(approx) - 4) * 0.015)
        score = rectangularity + simplicity_bonus

        if score > best_score:
            best_score = score
            best_box = cv2.boxPoints(rect)

    if best_box is None:
        return None

    return tuple((int(x), int(y)) for x, y in best_box)


def _mask_from_rotated_box(
    image_shape: tuple[int, ...],
    rotated_box: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    height, width = image_shape[:2]
    points = np.array(rotated_box, dtype=np.int32)
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    return mask


def _box_from_mask(
    mask: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0:
        return None

    box_points = cv2.boxPoints(cv2.minAreaRect(contour))
    return tuple((int(x), int(y)) for x, y in box_points)


def _order_corners_clockwise(points: np.ndarray) -> np.ndarray:
    """4 noktayı TL, TR, BR, BL sırasına diz."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    ordered = pts[order]
    # arctan2 saat yönünün tersini verir; TL'den başlatmak için en sol-üstü bul.
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    # Saat yönü (TL, TR, BR, BL) olacak şekilde yönü sabitle.
    if _polygon_signed_area(ordered) < 0:
        ordered = ordered[::-1]
        start = int(np.argmin(ordered.sum(axis=1)))
        ordered = np.roll(ordered, -start, axis=0)
    return ordered.astype(np.float32)


def _polygon_signed_area(pts: np.ndarray) -> float:
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _line_intersection(
    line_a: tuple[float, float, float, float],
    line_b: tuple[float, float, float, float],
) -> tuple[float, float] | None:
    """fitLine formatındaki (vx, vy, x0, y0) iki doğrunun kesişimi."""
    vx_a, vy_a, x0_a, y0_a = line_a
    vx_b, vy_b, x0_b, y0_b = line_b
    denominator = vx_a * vy_b - vy_a * vx_b
    if abs(denominator) < 1e-6:
        return None
    t = ((x0_b - x0_a) * vy_b - (y0_b - y0_a) * vx_b) / denominator
    return (x0_a + t * vx_a, y0_a + t * vy_a)


def _fit_quadrilateral_corners(
    contour: np.ndarray | None,
    image_shape: tuple[int, ...],
) -> np.ndarray | None:
    """Panel köşelerini (4,2) olarak fit et; gönye ölçümü için gerçek köşeler.

    minAreaRect köşeleri 90°'ye zorlar; burada ham ürün konturunun her kenarına
    doğru fit edilerek (kırık köşelere dayanıklı, %10 uç nokta atılarak) gerçek
    köşe açıları korunur. Eksene hizalanmış shape_mask yerine ham kontur kullanılır.
    """
    if contour is None or cv2.contourArea(contour) <= 0:
        return None

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)  # 4 köşe, sıralı
    box = _order_corners_clockwise(box)

    points = contour.reshape(-1, 2).astype(np.float32)
    diagonal = float(np.hypot(*(box[0] - box[2])))
    if diagonal <= 0:
        return None

    # Her kenar segmenti (i -> i+1); her kontur noktasını en yakın kenara ata.
    side_segments = [(box[i], box[(i + 1) % 4]) for i in range(4)]
    distances = np.stack(
        [_point_segment_distance(points, seg[0], seg[1]) for seg in side_segments],
        axis=1,
    )
    assignments = np.argmin(distances, axis=1)

    fitted_lines: list[tuple[float, float, float, float] | None] = []
    for side_index in range(4):
        side_points = points[assignments == side_index]
        if len(side_points) < 20:
            fitted_lines.append(None)
            continue
        seg = side_segments[side_index]
        side_dist = _point_segment_distance(side_points, seg[0], seg[1])
        keep = side_dist <= np.percentile(side_dist, 90)
        kept_points = side_points[keep]
        if len(kept_points) < 10:
            kept_points = side_points
        line = cv2.fitLine(kept_points, cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
        fitted_lines.append((float(line[0]), float(line[1]), float(line[2]), float(line[3])))

    height, width = image_shape[:2]
    margin = diagonal * 0.12
    corners = []
    for corner_index in range(4):
        prev_side = (corner_index - 1) % 4
        line_prev = fitted_lines[prev_side]
        line_curr = fitted_lines[corner_index]
        point: tuple[float, float] | None = None
        if line_prev is not None and line_curr is not None:
            point = _line_intersection(line_prev, line_curr)
        box_corner = box[corner_index]
        # Fit edilen kesişim, minAreaRect köşesine makul yakınlıkta ve görüntü
        # sınırları içinde (küçük taşma payıyla) değilse boxPoints'e geri düş.
        valid = (
            point is not None
            and np.hypot(point[0] - box_corner[0], point[1] - box_corner[1]) <= diagonal * 0.20
            and -margin <= point[0] <= width + margin
            and -margin <= point[1] <= height + margin
        )
        if not valid:
            point = (float(box_corner[0]), float(box_corner[1]))
        corners.append(point)

    return _order_corners_clockwise(np.array(corners, dtype=np.float32))


def _point_segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Nokta kümesinin a-b segmentine dik uzaklıkları (vektörize)."""
    ab = b - a
    length_sq = float(ab[0] ** 2 + ab[1] ** 2)
    if length_sq <= 1e-6:
        return np.hypot(points[:, 0] - a[0], points[:, 1] - a[1])
    t = ((points[:, 0] - a[0]) * ab[0] + (points[:, 1] - a[1]) * ab[1]) / length_sq
    t = np.clip(t, 0.0, 1.0)
    proj_x = a[0] + t * ab[0]
    proj_y = a[1] + t * ab[1]
    return np.hypot(points[:, 0] - proj_x, points[:, 1] - proj_y)


def estimate_roi_confidence(
    image: np.ndarray,
    shape_mask: np.ndarray,
    contour: np.ndarray,
    corners: np.ndarray | None,
    detection_method: str,
) -> float:
    """ROI tespitinin güvenilirliğini [0,1] aralığında tahmin et."""
    image_height, image_width = image.shape[:2]
    image_area = float(image_height * image_width)
    if image_area <= 0:
        return 0.0

    contour_area = float(cv2.contourArea(contour))
    area_ratio = contour_area / image_area
    # Beklenen alan bandı: %15–%85 arası tam güven.
    area_score = float(np.clip((area_ratio - 0.05) / 0.10, 0.0, 1.0))
    area_score *= float(np.clip((0.95 - area_ratio) / 0.10, 0.0, 1.0))

    rectangularity = 1.0
    if corners is not None:
        quad_area = abs(_polygon_signed_area(np.asarray(corners, dtype=np.float32)))
        if quad_area > 0:
            rectangularity = float(np.clip(contour_area / quad_area, 0.0, 1.0))

    # Kenar desteği: quad çevresi boyunca Canny kenarlarıyla örtüşme.
    edge_support = 0.5
    if corners is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 45, 130)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        edge_support = _perimeter_edge_support(edges, np.asarray(corners, dtype=np.float32))

    x, y, width, height = cv2.boundingRect(contour)
    touches = (
        int(x <= 1)
        + int(y <= 1)
        + int(x + width >= image_width - 1)
        + int(y + height >= image_height - 1)
    )
    border_penalty = 0.12 * touches

    confidence = 0.4 * area_score + 0.3 * rectangularity + 0.3 * edge_support
    if detection_method == "background_diff":
        confidence += 0.1
    confidence -= border_penalty
    return float(np.clip(confidence, 0.0, 1.0))


def _perimeter_edge_support(edges: np.ndarray, corners: np.ndarray) -> float:
    """Quad çevresi üzerinde kenar pikseliyle desteklenen örnek oranı."""
    height, width = edges.shape[:2]
    supported = 0
    total = 0
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        samples = max(2, int(np.hypot(p2[0] - p1[0], p2[1] - p1[1]) / 4))
        for s in range(samples):
            t = s / float(samples - 1) if samples > 1 else 0.0
            px = int(round(p1[0] + t * (p2[0] - p1[0])))
            py = int(round(p1[1] + t * (p2[1] - p1[1])))
            if 0 <= px < width and 0 <= py < height:
                total += 1
                if edges[py, px] > 0:
                    supported += 1
    if total == 0:
        return 0.0
    return supported / float(total)


def _refine_shape_mask_by_panel_color(
    image: np.ndarray,
    shape_mask: np.ndarray,
    config: AppConfig,
) -> np.ndarray:
    shape_bbox, _, _ = _extract_shape_roi(image, shape_mask)
    x, y, width, height = shape_bbox
    if width < 20 or height < 20:
        return shape_mask

    crop = image[y : y + height, x : x + width]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower = np.array(config.product_hsv_lower, dtype=np.uint8)
    upper = np.array(config.product_hsv_upper, dtype=np.uint8)
    panel_like = cv2.inRange(hsv, lower, upper) > 0
    panel_ratio = float(panel_like.mean())

    threshold = config.product_color_profile_threshold
    row_range = _dominant_profile_range(panel_like.mean(axis=1), threshold=threshold)
    col_range = _dominant_profile_range(panel_like.mean(axis=0), threshold=threshold)
    if row_range is None and col_range is None:
        return shape_mask

    refined = shape_mask.copy()
    if row_range is not None:
        start, end = row_range
        if (end - start + 1) >= height * 0.55:
            refined[: y + start, :] = 0
            refined[y + end + 1 :, :] = 0

    if col_range is not None:
        start, end = col_range
        if (end - start + 1) >= width * 0.55:
            refined[:, : x + start] = 0
            refined[:, x + end + 1 :] = 0

    if panel_ratio >= 0.08:
        color_mask = np.zeros_like(shape_mask)
        crop_color_mask = (panel_like.astype(np.uint8) * 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        crop_color_mask = cv2.morphologyEx(crop_color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        crop_color_mask = cv2.morphologyEx(crop_color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        color_mask[y : y + height, x : x + width] = crop_color_mask
        expanded = cv2.dilate(color_mask, kernel, iterations=1)
        candidate = cv2.bitwise_and(refined, expanded)
        if _largest_area_ratio(candidate) >= _largest_area_ratio(refined) * 0.55:
            refined = candidate

    return refined


def _dominant_profile_range(profile: np.ndarray, threshold: float) -> tuple[int, int] | None:
    if profile.size == 0:
        return None

    window = min(25, max(3, profile.size // 20))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.convolve(profile.astype(np.float32), kernel, mode="same")
    indices = np.where(smoothed >= threshold)[0]
    if len(indices) == 0:
        return None

    return int(indices[0]), int(indices[-1])


def _extract_shape_roi(
    image: np.ndarray,
    shape_mask: np.ndarray,
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray]:
    contours, _ = cv2.findContours(shape_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        empty = np.zeros((1, 1, 3), dtype=image.dtype)
        empty_mask = np.zeros((1, 1), dtype=np.uint8)
        return (0, 0, 1, 1), empty, empty_mask

    contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(contour)
    image_crop = image[y : y + height, x : x + width].copy()
    mask_crop = shape_mask[y : y + height, x : x + width].copy()
    roi = cv2.bitwise_and(image_crop, image_crop, mask=mask_crop)
    return (x, y, width, height), roi, mask_crop


def _is_overfilled_bbox(contour: np.ndarray, image_shape: tuple[int, ...]) -> bool:
    image_height, image_width = image_shape[:2]
    x, y, width, height = cv2.boundingRect(contour)
    width_ratio = width / float(image_width)
    height_ratio = height / float(image_height)
    touches_multiple_edges = (
        int(x <= 1)
        + int(y <= 1)
        + int(x + width >= image_width - 1)
        + int(y + height >= image_height - 1)
    ) >= 3
    return touches_multiple_edges and (width_ratio > 0.92 or height_ratio > 0.92)


def _build_grabcut_mask(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    margin_x = max(1, int(width * 0.05))
    margin_y = max(1, int(height * 0.05))
    rect = (
        margin_x,
        margin_y,
        max(1, width - 2 * margin_x),
        max(1, height - 2 * margin_y),
    )

    grabcut_mask = np.zeros((height, width), dtype=np.uint8)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.setRNGSeed(12345)  # tekrarlanabilir sonuç için
    cv2.grabCut(
        image,
        grabcut_mask,
        rect,
        background_model,
        foreground_model,
        3,
        cv2.GC_INIT_WITH_RECT,
    )

    mask = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
