from __future__ import annotations

import cv2
import numpy as np


Color = tuple[int, int, int]


def draw_bbox(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: Color = (0, 180, 0),
    thickness: int = 3,
) -> np.ndarray:
    """Draw a bounding box on a copy of the image."""
    result = image.copy()
    x, y, width, height = bbox
    cv2.rectangle(result, (x, y), (x + width, y + height), color, thickness)
    return result


def draw_rotated_box(
    image: np.ndarray,
    rotated_box: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
    color: Color = (0, 220, 255),
    thickness: int = 3,
) -> np.ndarray:
    """Draw a rotated bounding box on a copy of the image."""
    result = image.copy()
    points = np.array(rotated_box, dtype=np.int32)
    cv2.polylines(result, [points], isClosed=True, color=color, thickness=thickness)
    return result


def draw_contour_overlay(
    image: np.ndarray,
    contour: np.ndarray,
    color: Color = (0, 180, 0),
    alpha: float = 0.22,
) -> np.ndarray:
    """Draw a translucent product contour fill on a copy of the image."""
    result = image.copy()
    overlay = result.copy()
    cv2.drawContours(overlay, [contour], contourIdx=-1, color=color, thickness=-1)
    result = cv2.addWeighted(overlay, alpha, result, 1.0 - alpha, 0)
    cv2.drawContours(result, [contour], contourIdx=-1, color=color, thickness=2)
    return result


def draw_shape_analysis(
    image: np.ndarray,
    contour: np.ndarray,
    rotated_box: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    """Draw product contour and shape-aware rotated bounding box."""
    del contour
    result = image.copy()
    overlay = result.copy()
    points = np.array(rotated_box, dtype=np.int32)
    cv2.fillPoly(overlay, [points], (0, 180, 0))
    result = cv2.addWeighted(overlay, 0.16, result, 0.84, 0)
    return draw_rotated_box(result, rotated_box)


def draw_text_panel(
    image: np.ndarray,
    lines: list[str],
    origin: tuple[int, int] = (16, 28),
    color: Color = (255, 255, 255),
    background: Color = (30, 30, 30),
) -> np.ndarray:
    """Draw a compact text panel over a copy of the image."""
    result = image.copy()
    if not lines:
        return result

    x, y = origin
    line_height = 26
    panel_width = max(260, max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0] for line in lines) + 24)
    panel_height = line_height * len(lines) + 14

    overlay = result.copy()
    cv2.rectangle(overlay, (x - 8, y - 22), (x + panel_width, y - 22 + panel_height), background, -1)
    result = cv2.addWeighted(overlay, 0.72, result, 0.28, 0)

    for index, line in enumerate(lines):
        cv2.putText(
            result,
            line,
            (x, y + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    return result


def stack_original_and_result(original: np.ndarray, result: np.ndarray) -> np.ndarray:
    """Stack original and processed images side by side with matched height."""
    original_copy = _ensure_bgr(original)
    result_copy = _ensure_bgr(result)

    height = min(original_copy.shape[0], result_copy.shape[0])
    original_resized = _resize_to_height(original_copy, height)
    result_resized = _resize_to_height(result_copy, height)
    return np.hstack([original_resized, result_resized])


def create_mask_preview(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create a BGR mask preview sized like the input image."""
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_bgr[:, :, 0] = 0
    mask_bgr[:, :, 1] = mask
    mask_bgr[:, :, 2] = 0
    return cv2.addWeighted(_ensure_bgr(image), 0.72, mask_bgr, 0.28, 0)


def _resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    current_height, current_width = image.shape[:2]
    if current_height == height:
        return image

    scale = height / float(current_height)
    width = int(current_width * scale)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()
