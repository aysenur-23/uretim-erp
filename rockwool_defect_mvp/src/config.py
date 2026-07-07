from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

REQUIRED_KEYS = {
    "camera_source",
    "image_folder_path",
    "webcam_index",
    "output_overlay_path",
    "database_path",
    "min_product_area_ratio",
    "product_hsv_lower",
    "product_hsv_upper",
    "product_color_profile_threshold",
    "edge_damage_threshold",
    "color_anomaly_threshold",
    "crack_darkness_threshold",
    "local_anomaly_threshold",
    "anomaly_score_suspicious",
    "anomaly_score_defect",
}


# config.yaml'da bulunmayabilecek yeni ayarlar; REQUIRED_KEYS büyütülmez ki
# mevcut config dosyaları ve testlerdeki inline config'ler kırılmasın.
OPTIONAL_KEYS_WITH_DEFAULTS: dict[str, Any] = {
    # ürün tespiti
    "background_reference_path": "",
    "roi_snap_enabled": True,
    # yeni dedektör eşikleri
    "glass_burn_threshold": 0.35,
    "raw_fiber_threshold": 0.30,
    "deformation_threshold": 0.30,
    # boyut / gönye kontrolü
    "size_check_enabled": False,
    "px_per_mm": 0.0,
    "expected_width_mm": 600.0,
    "expected_height_mm": 1200.0,
    "size_tolerance_mm": 5.0,
    "squareness_tolerance_deg": 1.5,
    "size_calibration_path": "data/calibration/size_calibration.json",
    # karar motoru sınıf-bazlı override
    "decision_profile": {},
}

# Kalibrasyon sidecar JSON'undan config.yaml'ı ezebilecek anahtarlar.
_CALIBRATION_OVERRIDE_KEYS = (
    "px_per_mm",
    "size_check_enabled",
    "background_reference_path",
)


@dataclass(frozen=True)
class AppConfig:
    camera_source: str
    image_folder_path: Path
    webcam_index: int
    output_overlay_path: Path
    database_path: Path
    min_product_area_ratio: float
    product_hsv_lower: tuple[int, int, int]
    product_hsv_upper: tuple[int, int, int]
    product_color_profile_threshold: float
    edge_damage_threshold: float
    color_anomaly_threshold: float
    crack_darkness_threshold: float
    local_anomaly_threshold: float
    anomaly_score_suspicious: float
    anomaly_score_defect: float
    # opsiyonel / yeni ayarlar
    background_reference_path: Path | None
    roi_snap_enabled: bool
    glass_burn_threshold: float
    raw_fiber_threshold: float
    deformation_threshold: float
    size_check_enabled: bool
    px_per_mm: float
    expected_width_mm: float
    expected_height_mm: float
    size_tolerance_mm: float
    squareness_tolerance_deg: float
    size_calibration_path: Path
    decision_profile: Mapping[str, Mapping[str, float]] = field(default_factory=dict)


def load_yaml_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the raw YAML configuration."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    missing_keys = REQUIRED_KEYS.difference(config)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise KeyError(f"Missing required config keys: {missing}")

    for key, default in OPTIONAL_KEYS_WITH_DEFAULTS.items():
        config.setdefault(key, default)

    _apply_size_calibration_sidecar(config)

    return config


def _apply_size_calibration_sidecar(config: dict[str, Any]) -> None:
    """Operatör kalibrasyonunu (varsa) config.yaml'ı ezmeden uygula.

    Kalibrasyon API'si px/mm ve ilgili ayarları bir JSON sidecar dosyasına
    yazar; config.yaml yorumlarını bozmadan bu değerler yüklemede uygulanır.
    """
    sidecar_value = config.get("size_calibration_path") or ""
    if not sidecar_value:
        return
    sidecar_path = resolve_project_path(sidecar_value)
    if not sidecar_path.exists():
        return
    try:
        with sidecar_path.open("r", encoding="utf-8") as sidecar_file:
            data = json.load(sidecar_file)
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(data, dict):
        return
    for key in _CALIBRATION_OVERRIDE_KEYS:
        if key in data:
            config[key] = data[key]


def resolve_project_path(value: str | Path) -> Path:
    """Resolve project-relative paths from config.yaml."""
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load config.yaml into a typed AppConfig object."""
    raw = load_yaml_config(config_path)

    return AppConfig(
        camera_source=str(raw["camera_source"]),
        image_folder_path=resolve_project_path(raw["image_folder_path"]),
        webcam_index=int(raw["webcam_index"]),
        output_overlay_path=resolve_project_path(raw["output_overlay_path"]),
        database_path=resolve_project_path(raw["database_path"]),
        min_product_area_ratio=float(raw["min_product_area_ratio"]),
        product_hsv_lower=_parse_hsv_triplet(raw["product_hsv_lower"], "product_hsv_lower"),
        product_hsv_upper=_parse_hsv_triplet(raw["product_hsv_upper"], "product_hsv_upper"),
        product_color_profile_threshold=float(raw["product_color_profile_threshold"]),
        edge_damage_threshold=float(raw["edge_damage_threshold"]),
        color_anomaly_threshold=float(raw["color_anomaly_threshold"]),
        crack_darkness_threshold=float(raw["crack_darkness_threshold"]),
        local_anomaly_threshold=float(raw["local_anomaly_threshold"]),
        anomaly_score_suspicious=float(raw["anomaly_score_suspicious"]),
        anomaly_score_defect=float(raw["anomaly_score_defect"]),
        background_reference_path=(
            resolve_project_path(raw["background_reference_path"])
            if raw.get("background_reference_path")
            else None
        ),
        roi_snap_enabled=bool(raw["roi_snap_enabled"]),
        glass_burn_threshold=float(raw["glass_burn_threshold"]),
        raw_fiber_threshold=float(raw["raw_fiber_threshold"]),
        deformation_threshold=float(raw["deformation_threshold"]),
        size_check_enabled=bool(raw["size_check_enabled"]),
        px_per_mm=float(raw["px_per_mm"]),
        expected_width_mm=float(raw["expected_width_mm"]),
        expected_height_mm=float(raw["expected_height_mm"]),
        size_tolerance_mm=float(raw["size_tolerance_mm"]),
        squareness_tolerance_deg=float(raw["squareness_tolerance_deg"]),
        size_calibration_path=resolve_project_path(raw["size_calibration_path"]),
        decision_profile=_parse_decision_profile(raw["decision_profile"]),
    )


def ensure_runtime_directories(config: AppConfig) -> None:
    """Create directories needed by the current config."""
    config.image_folder_path.mkdir(parents=True, exist_ok=True)
    config.output_overlay_path.mkdir(parents=True, exist_ok=True)
    config.database_path.parent.mkdir(parents=True, exist_ok=True)


def _parse_decision_profile(value: Any) -> dict[str, dict[str, float]]:
    """Sınıf-bazlı karar override'ını doğrula ve normalize et."""
    if not value:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("decision_profile must be a mapping of class -> settings.")

    parsed: dict[str, dict[str, float]] = {}
    for defect_type, overrides in value.items():
        if not isinstance(overrides, Mapping):
            raise ValueError(f"decision_profile[{defect_type}] must be a mapping.")
        entry: dict[str, float] = {}
        for field_name, field_value in overrides.items():
            if field_value is None:
                entry[str(field_name)] = None  # type: ignore[assignment]
            else:
                entry[str(field_name)] = float(field_value)
        parsed[str(defect_type)] = entry
    return parsed


def _parse_hsv_triplet(value: Any, key: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{key} must be a list with 3 HSV values.")

    parsed = tuple(int(item) for item in value)
    if not all(0 <= item <= 255 for item in parsed):
        raise ValueError(f"{key} values must be between 0 and 255.")

    return parsed


if __name__ == "__main__":
    app_config = load_config()
    ensure_runtime_directories(app_config)
    print("Config loaded successfully.")
    print(f"camera_source={app_config.camera_source}")
    print(f"image_folder_path={app_config.image_folder_path}")
