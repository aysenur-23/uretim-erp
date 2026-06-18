from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

    return config


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
    )


def ensure_runtime_directories(config: AppConfig) -> None:
    """Create directories needed by the current config."""
    config.image_folder_path.mkdir(parents=True, exist_ok=True)
    config.output_overlay_path.mkdir(parents=True, exist_ok=True)
    config.database_path.parent.mkdir(parents=True, exist_ok=True)


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
