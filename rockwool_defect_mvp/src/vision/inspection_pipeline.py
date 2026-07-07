from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.config import AppConfig
from src.anomaly.simple_anomaly import detect_local_anomaly
from src.decision.arbitration import arbitrate_overlaps
from src.decision.decision_engine import DecisionResult, decide_quality
from src.vision.defect_rules import (
    detect_color_anomaly,
    detect_dark_crack_like_regions,
    detect_edge_damage,
    detect_glass_burn,
    detect_raw_fiber,
    detect_shape_deformation,
)
from src.vision.defect_taxonomy import DEFECT_TAXONOMY
from src.vision.preprocessing import resize_image
from src.vision.product_detection import ProductROI, find_product_roi
from src.vision.size_check import detect_size_tolerance
from src.vision.visualization import draw_shape_analysis, draw_size_annotations


# Kural-özel maske çizilecek sınıflar; çatlak en sonda çizilir (kırmızı üstte kalsın).
_MASK_OVERLAY_ORDER = ("glass_burn", "raw_fiber", "color_anomaly", "dark_crack")


@dataclass(frozen=True)
class AnalysisView:
    original: np.ndarray
    overlay: np.ndarray
    roi: np.ndarray | None
    mask_preview: np.ndarray | None
    crack_preview: np.ndarray | None
    heatmap_preview: np.ndarray | None
    product: ProductROI | None
    rule_results: dict[str, dict]
    decision: DecisionResult | None


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def product_display_roi(product: ProductROI) -> np.ndarray:
    return getattr(product, "shape_roi", product.roi)


def product_display_mask(product: ProductROI) -> np.ndarray:
    return getattr(product, "shape_mask_roi", product.mask)


def product_display_bbox(product: ProductROI) -> tuple[int, int, int, int]:
    return getattr(product, "shape_bbox", product.bbox)


def overlay_crack_mask(roi: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return roi.copy()

    result = roi.copy()
    red_overlay = np.zeros_like(result)
    red_overlay[:, :, 2] = mask
    return cv2.addWeighted(result, 0.76, red_overlay, 0.24, 0)


def draw_crack_mask_on_image(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    crack_mask: np.ndarray | None,
) -> np.ndarray:
    if crack_mask is None:
        return image

    x, y, width, height = bbox
    result = image.copy()
    crop = result[y : y + height, x : x + width]
    if crop.shape[:2] != crack_mask.shape[:2]:
        crack_mask = cv2.resize(crack_mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)

    red_overlay = np.zeros_like(crop)
    red_overlay[:, :, 2] = crack_mask
    crop_with_overlay = cv2.addWeighted(crop, 0.78, red_overlay, 0.22, 0)
    crop[crack_mask > 0] = crop_with_overlay[crack_mask > 0]
    return result


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """'#rrggbb' -> OpenCV BGR üçlüsü."""
    value = hex_color.lstrip("#")
    if len(value) != 6:
        return (0, 0, 255)
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return (b, g, r)


def draw_rule_mask_on_image(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    mask: np.ndarray | None,
    color: tuple[int, int, int],
    alpha: float = 0.35,
) -> np.ndarray:
    """Bir kural maskesini verilen renkte ROI üzerine yarı saydam çizer."""
    if mask is None:
        return image

    x, y, width, height = bbox
    result = image.copy()
    crop = result[y : y + height, x : x + width]
    if crop.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)

    color_overlay = np.zeros_like(crop)
    color_overlay[:] = color
    blended = cv2.addWeighted(crop, 1.0 - alpha, color_overlay, alpha, 0)
    crop[mask > 0] = blended[mask > 0]
    return result


def draw_heatmap_on_image(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    heatmap: np.ndarray | None,
) -> np.ndarray:
    if heatmap is None:
        return image

    x, y, width, height = bbox
    result = image.copy()
    crop = result[y : y + height, x : x + width]
    if crop.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_AREA)

    active_mask = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY) > 0
    blended = cv2.addWeighted(crop, 0.82, heatmap, 0.18, 0)
    crop[active_mask] = blended[active_mask]
    return result


def run_defect_rules(
    frame: np.ndarray,
    product: ProductROI,
    config: AppConfig,
) -> dict[str, dict]:
    """8 bağımsız dedektörü çalıştırır; sadece çatlak/yanık çakışması arbitre edilir."""
    roi = product_display_roi(product)
    bbox = product_display_bbox(product)
    results = {
        "edge_damage": detect_edge_damage(frame, roi, bbox, config, product),
        "deformation": detect_shape_deformation(frame, roi, bbox, config, product),
        "size_tolerance": detect_size_tolerance(product, config),
        "glass_burn": detect_glass_burn(frame, roi, config),
        "raw_fiber": detect_raw_fiber(frame, roi, config),
        "color_anomaly": detect_color_anomaly(frame, roi, config),
        "dark_crack": detect_dark_crack_like_regions(frame, roi, config),
        "local_anomaly": detect_local_anomaly(roi, config),
    }
    return arbitrate_overlaps(results)


def process_frame(frame: np.ndarray, config: AppConfig) -> AnalysisView:
    """Run ROI detection, OpenCV rules, decision logic, and overlay rendering."""
    display_frame = resize_image(frame)
    product = find_product_roi(display_frame, config)

    if product is None:
        return AnalysisView(
            original=display_frame,
            overlay=display_frame,
            roi=None,
            mask_preview=None,
            crack_preview=None,
            heatmap_preview=None,
            product=None,
            rule_results={},
            decision=None,
        )

    roi = product_display_roi(product)
    bbox = product_display_bbox(product)
    rule_results = run_defect_rules(display_frame, product, config)
    decision = decide_quality(rule_results, config)
    crack_mask = rule_results["dark_crack"].get("mask")
    heatmap = rule_results["local_anomaly"].get("heatmap")

    overlay = draw_shape_analysis(display_frame, product.contour, product.rotated_box)

    # Her şüpheli dedektörün maskesi taksonomi renginde çizilir (çatlak en üstte).
    has_specific_overlay = False
    for name in _MASK_OVERLAY_ORDER:
        result = rule_results.get(name, {})
        if result.get("is_suspicious") and result.get("mask") is not None:
            color = _hex_to_bgr(DEFECT_TAXONOMY[name].overlay_color)
            overlay = draw_rule_mask_on_image(overlay, bbox, result["mask"], color)
            has_specific_overlay = True

    # Kural-özel maske yoksa genel anomali heatmap'i gösterilir (gürültü bastırma).
    if not has_specific_overlay:
        overlay = draw_heatmap_on_image(overlay, bbox, heatmap)

    # Boyut/gönye kontrolü açıksa ölçü etiketleri çizilir.
    size_result = rule_results.get("size_tolerance", {})
    if size_result.get("enabled") and len(product.corners) == 4:
        overlay = draw_size_annotations(overlay, product.corners, size_result)
    return AnalysisView(
        original=display_frame,
        overlay=overlay,
        roi=roi,
        mask_preview=mask_to_bgr(product_display_mask(product)),
        crack_preview=overlay_crack_mask(roi, crack_mask),
        heatmap_preview=heatmap,
        product=product,
        rule_results=rule_results,
        decision=decision,
    )
