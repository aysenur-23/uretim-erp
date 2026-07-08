from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import AppConfig


@dataclass(frozen=True)
class DecisionResult:
    label: str
    anomaly_score: float
    reasons: list[str]


# Sınıf-bazlı karar profili: her hata sınıfı için uyarı/red eşiği ve ağırlık.
# reject=None → o sınıf tek başına RED veremez (destekleyici sinyal).
DEFAULT_PROFILE: dict[str, dict[str, float | None]] = {
    "dark_crack": {"warn": 0.30, "reject": 0.45, "weight": 1.0},
    "glass_burn": {"warn": 0.30, "reject": 0.55, "weight": 1.0},
    "edge_damage": {"warn": 0.35, "reject": 0.60, "weight": 0.8},
    "size_tolerance": {"warn": 0.25, "reject": 0.60, "weight": 1.0},
    "deformation": {"warn": 0.30, "reject": 0.65, "weight": 0.7},
    "raw_fiber": {"warn": 0.30, "reject": 0.60, "weight": 0.8},
    "color_anomaly": {"warn": 0.30, "reject": 0.70, "weight": 0.6},
    "local_anomaly": {"warn": 0.42, "reject": None, "weight": 0.4},  # destekleyici
}

# Telefon modu ön ayarı: telefonla yüklenen görüntülerde mesafe/açı değişken
# olduğundan boyut/gönye mm ölçümü ve deformasyon güvenilir değildir. Bu sınıflar
# reject=None yapılır (tek başına RED veremez, "destekleyici" olur) ve ağırlıkları
# düşürülür; karar yüzey/çizgi/kenar hatalarına dayanır, perspektif yanlış RED üretmez.
PHONE_PROFILE_OVERRIDES: dict[str, dict[str, float | None]] = {
    "deformation": {"warn": 0.45, "reject": None, "weight": 0.3},
    "size_tolerance": {"warn": 0.60, "reject": None, "weight": 0.0},
}


def _resolve_profile(config: AppConfig) -> dict[str, dict[str, float | None]]:
    """Varsayılan profili mod ön ayarı ve config.decision_profile ile birleştir.

    Öncelik (düşükten yükseğe): DEFAULT_PROFILE < telefon ön ayarı < kullanıcı override.
    """
    profile: dict[str, dict[str, float | None]] = {
        name: dict(entry) for name, entry in DEFAULT_PROFILE.items()
    }

    if str(getattr(config, "inspection_mode", "fixed_camera")).lower() == "phone":
        for name, entry in PHONE_PROFILE_OVERRIDES.items():
            base = profile.setdefault(name, {"warn": 0.30, "reject": 0.60, "weight": 1.0})
            base.update(entry)

    overrides = getattr(config, "decision_profile", None) or {}
    for name, entry in overrides.items():
        base = profile.setdefault(name, {"warn": 0.30, "reject": 0.60, "weight": 1.0})
        for key, value in entry.items():
            base[key] = value
    return profile


def _is_supporting(entry: dict[str, float | None]) -> bool:
    """reject=None olan sınıf tek başına RED veremez → destekleyici sinyaldir."""
    return entry.get("reject", 0.60) is None


def decide_quality(rule_results: dict[str, dict[str, Any]], config: AppConfig) -> DecisionResult:
    """Sınıf-bazlı eşik/ağırlıklarla OpenCV kural skorlarını karara dönüştürür."""
    profile = _resolve_profile(config)

    weighted_scores = []
    reject_hits: list[str] = []
    suspicious_classes: list[str] = []
    non_supporting_suspicious: list[str] = []
    for name, result in rule_results.items():
        score = float(result.get("score", 0.0))
        is_suspicious = bool(result.get("is_suspicious", False))
        entry = profile.get(name, {"warn": 0.30, "reject": 0.60, "weight": 1.0})
        weight = float(entry.get("weight", 1.0) or 0.0)
        reject = entry.get("reject", 0.60)

        weighted_scores.append(weight * score)

        # Her dedektörün KENDİ is_suspicious kararı temel alınır (ham skor değil);
        # böylece dedektörün alan mantığı (ör. çatlak uzunluk kapısı) korunur.
        if not is_suspicious:
            continue
        suspicious_classes.append(name)
        if reject is not None and score >= float(reject):
            reject_hits.append(name)
        if not _is_supporting(entry):
            non_supporting_suspicious.append(name)

    anomaly_score = max(weighted_scores) if weighted_scores else 0.0

    # Çoklu-şüphe: destekleyici olmayan (reject=None olmayan) sınıflardan >=2'si RED.

    if reject_hits or len(non_supporting_suspicious) >= 2:
        label = "HATALI"
    elif suspicious_classes or anomaly_score >= config.anomaly_score_suspicious:
        label = "SUPHELI"
    else:
        label = "SAGLAM"

    reasons = [
        f"{_rule_display_name(name)}: {result.get('message', '')}"
        for name, result in rule_results.items()
        if bool(result.get("is_suspicious", False))
    ]
    if not reasons:
        reasons = ["Belirgin supheli sinyal bulunmadi."]

    return DecisionResult(
        label=label,
        anomaly_score=round(float(anomaly_score), 4),
        reasons=reasons,
    )


def _rule_display_name(name: str) -> str:
    return {
        "edge_damage": "Kenar",
        "deformation": "Deformasyon",
        "size_tolerance": "Boyut/Gonye",
        "glass_burn": "Cam yanigi",
        "raw_fiber": "Cam/cig elyaf",
        "color_anomaly": "Renk",
        "dark_crack": "Catlak",
        "local_anomaly": "Yerel anomali",
    }.get(name, name)
