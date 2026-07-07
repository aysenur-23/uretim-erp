from __future__ import annotations

from typing import Any

import cv2
import numpy as np


RuleResult = dict[str, Any]


def arbitrate_overlaps(results: dict[str, RuleResult]) -> dict[str, RuleResult]:
    """Sınıflar arası TEK açık hakemlik: çatlak vs. cam yanığı çakışması.

    İki dedektör de şüpheli VE maskeleri IoU >= 0.5 ile örtüşüyorsa, şekil kanıtı
    güçlü olan kazanır (uzun-ince ise çatlak, aksi halde yanık). Kaybeden skorunu
    korur ama is_suspicious=False alır ve 'arbitrated_by' ile işaretlenir.

    Bu, revert'e yol açan dedektörler-arası bastırma ağının YERİNE geçen tek,
    açık ve test edilebilir kuraldır. Başka hiçbir çift arbitre edilmez.
    """
    crack = results.get("dark_crack")
    burn = results.get("glass_burn")
    if crack is None or burn is None:
        return results
    if not crack.get("is_suspicious") or not burn.get("is_suspicious"):
        return results

    crack_mask = crack.get("mask")
    burn_mask = burn.get("mask")
    if crack_mask is None or burn_mask is None:
        return results

    if _mask_iou(crack_mask, burn_mask) < 0.5:
        return results

    crack_wins = float(crack.get("max_length_ratio", 0.0)) >= 0.24
    winner = "dark_crack" if crack_wins else "glass_burn"
    loser = "glass_burn" if crack_wins else "dark_crack"

    updated = dict(results)
    loser_result = dict(updated[loser])
    loser_result["is_suspicious"] = False
    loser_result["arbitrated_by"] = winner
    updated[loser] = loser_result
    return updated


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if mask_a.shape != mask_b.shape:
        mask_b = cv2.resize(mask_b, (mask_a.shape[1], mask_a.shape[0]), interpolation=cv2.INTER_NEAREST)
    a = mask_a > 0
    b = mask_b > 0
    intersection = float(np.count_nonzero(a & b))
    union = float(np.count_nonzero(a | b))
    if union <= 0:
        return 0.0
    return intersection / union
