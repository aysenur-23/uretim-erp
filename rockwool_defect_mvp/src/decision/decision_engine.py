from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import AppConfig


@dataclass(frozen=True)
class DecisionResult:
    label: str
    anomaly_score: float
    reasons: list[str]


def decide_quality(rule_results: dict[str, dict[str, Any]], config: AppConfig) -> DecisionResult:
    """Combine OpenCV rule scores into an MVP quality decision."""
    scores = [float(result.get("score", 0.0)) for result in rule_results.values()]
    anomaly_score = max(scores) if scores else 0.0
    suspicious_rules = [
        name for name, result in rule_results.items() if bool(result.get("is_suspicious", False))
    ]

    crack_result = rule_results.get("dark_crack", {})
    strong_crack_signal = bool(crack_result.get("is_suspicious", False)) and float(crack_result.get("score", 0.0)) >= 0.42

    if anomaly_score >= config.anomaly_score_defect or len(suspicious_rules) >= 2 or strong_crack_signal:
        label = "HATALI"
    elif anomaly_score >= config.anomaly_score_suspicious or suspicious_rules:
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
        "color_anomaly": "Renk",
        "glass_burn": "Cam yanigi",
        "raw_fiber": "Cam/cig elyaf",
        "dark_crack": "Catlak",
        "local_anomaly": "Yerel anomali",
    }.get(name, name)
