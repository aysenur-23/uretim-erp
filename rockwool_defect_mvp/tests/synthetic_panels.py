"""Dedektör testleri için sentetik taş yünü panel üreticileri."""
from __future__ import annotations

import cv2
import numpy as np


def make_panel(
    width: int = 520,
    height: int = 300,
    bgr: tuple[int, int, int] = (150, 170, 185),
    background: int = 30,
    margin: int = 55,
) -> np.ndarray:
    """Koyu arka plan üzerinde tan renkli dikdörtgen panel."""
    img = np.full((height, width, 3), background, dtype=np.uint8)
    cv2.rectangle(img, (margin, margin), (width - margin, height - margin), bgr, -1)
    return img


def panel_rect(img: np.ndarray, margin: int = 55) -> tuple[int, int, int, int]:
    height, width = img.shape[:2]
    return margin, margin, width - margin, height - margin


def paint_crack(
    img: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    thickness: int = 3,
    darkness: int = 35,
) -> np.ndarray:
    cv2.line(img, p1, p2, (darkness, darkness, darkness), thickness)
    return img


def paint_burn(
    img: np.ndarray,
    center: tuple[int, int],
    radius: int = 34,
    bgr: tuple[int, int, int] = (22, 40, 70),
) -> np.ndarray:
    """Yumuşak kenarlı kompakt kahve-siyah yanık lekesi."""
    overlay = img.copy()
    cv2.ellipse(overlay, center, (radius, int(radius * 0.8)), 0, 0, 360, bgr, -1)
    blurred = cv2.GaussianBlur(overlay, (0, 0), radius / 4.0)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, center, (radius + 6, int(radius * 0.8) + 6), 0, 0, 360, 255, -1)
    img[mask > 0] = blurred[mask > 0]
    return img


def paint_raw_fiber(
    img: np.ndarray,
    rect: tuple[int, int, int, int],
    seed: int = 0,
) -> np.ndarray:
    """Açık renkli, düşük doygunluklu, homojen olmayan lifsi/bulutsu bölge."""
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = rect
    region = img[y0:y1, x0:x1].astype(np.int16)
    # Belirgin doygunluk düşüşü için parlak nötr griye güçlü kaydırma
    # (çiğ elyafın beyazımsı-gri bulut görünümü).
    gray_target = np.full_like(region, 215)
    blend = 0.78
    region = (region * (1 - blend) + gray_target * blend).astype(np.int16)
    # Lifsi doku için benek gürültüsü — TÜM kanallara eşit (luminans) eklenir ki
    # düşük doygunluk korunsun (kanal-bağımsız gürültü sat'ı yapay yükseltir).
    h, w = region.shape[:2]
    noise = rng.normal(0, 16, (h, w, 1)).astype(np.int16)
    region = np.clip(region + noise, 0, 255).astype(np.uint8)
    img[y0:y1, x0:x1] = region
    return img


def make_bowed_panel(
    width: int = 520,
    height: int = 300,
    bgr: tuple[int, int, int] = (150, 170, 185),
    background: int = 30,
    margin: int = 55,
    bow: int = 40,
) -> np.ndarray:
    """Uzun kenarı içbükey (eğilmiş/ezilmiş) deforme panel."""
    img = np.full((height, width, 3), background, dtype=np.uint8)
    top_mid = (width // 2, margin + bow)  # üst kenar orta noktası içeri çekilmiş
    polygon = np.array(
        [
            [margin, margin],
            top_mid,
            [width - margin, margin],
            [width - margin, height - margin],
            [margin, height - margin],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(img, [polygon], bgr)
    return img


def warp_panel(img: np.ndarray, corner_offsets: list[tuple[int, int]]) -> np.ndarray:
    """Paneli köşe ofsetleriyle perspektif dönüşümle eğ (gönye/deformasyon)."""
    height, width = img.shape[:2]
    margin = 55
    src = np.float32(
        [[margin, margin], [width - margin, margin], [width - margin, height - margin], [margin, height - margin]]
    )
    dst = np.float32([[src[i][0] + corner_offsets[i][0], src[i][1] + corner_offsets[i][1]] for i in range(4)])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (width, height), borderValue=(30, 30, 30))
