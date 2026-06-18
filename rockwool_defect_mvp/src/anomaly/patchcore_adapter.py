from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class PatchCorePrediction:
    score: float
    heatmap: np.ndarray | None
    is_suspicious: bool
    message: str


class PatchCoreBackend(Protocol):
    def predict(self, image: np.ndarray) -> tuple[float, np.ndarray | None]:
        """Return anomaly score and optional heatmap for a BGR ROI."""


class PatchCoreAdapter:
    """Thin adapter boundary for plugging a trained PatchCore-like backend into the MVP."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        backend: PatchCoreBackend | None = None,
        threshold: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.backend = backend
        self.threshold = float(threshold)

    @property
    def is_available(self) -> bool:
        return self.backend is not None

    def predict(self, roi: np.ndarray) -> PatchCorePrediction:
        if roi.size == 0:
            raise ValueError("PatchCoreAdapter cannot process an empty ROI.")
        if self.backend is None:
            model_hint = f" Model path: {self.model_path}" if self.model_path else ""
            raise RuntimeError(f"PatchCore backend is not configured yet.{model_hint}")

        score, heatmap = self.backend.predict(roi)
        score = _clip01(score)
        return PatchCorePrediction(
            score=score,
            heatmap=heatmap,
            is_suspicious=score >= self.threshold,
            message=(
                "PatchCore anomali skoru supheli."
                if score >= self.threshold
                else "PatchCore anomali skoru normal."
            ),
        )


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
