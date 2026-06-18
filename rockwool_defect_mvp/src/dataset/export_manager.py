from __future__ import annotations

import csv
import shutil
from pathlib import Path

from src.config import AppConfig
from src.storage.sqlite_store import SQLiteStore


def export_dataset(config: AppConfig, split_mode: bool = True) -> Path:
    """Export gallery records into an image dataset folder with metadata CSV."""
    store = SQLiteStore(config.database_path)
    records = store.fetch_recent_inspection_records(limit=500)
    export_root = config.database_path.parents[1] / "exports" / "latest_dataset"
    if export_root.exists():
        shutil.rmtree(export_root)
    export_root.mkdir(parents=True, exist_ok=True)

    image_root = export_root / "images"
    overlay_root = export_root / "overlays"
    metadata_path = export_root / "metadata.csv"
    image_root.mkdir(parents=True, exist_ok=True)
    overlay_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, record in enumerate(records):
        split = _split_name(index, len(records)) if split_mode else "all"
        label = str(record.get("operator_label") or "etiketsiz")
        target_image_dir = image_root / split / label
        target_overlay_dir = overlay_root / split / label
        target_image_dir.mkdir(parents=True, exist_ok=True)
        target_overlay_dir.mkdir(parents=True, exist_ok=True)

        image_target = _copy_if_exists(record.get("image_path"), target_image_dir, f"{record['id']}_raw.jpg")
        overlay_target = _copy_if_exists(record.get("overlay_path"), target_overlay_dir, f"{record['id']}_overlay.jpg")
        rows.append(
            {
                **record,
                "split": split,
                "export_image_path": str(image_target) if image_target else "",
                "export_overlay_path": str(overlay_target) if overlay_target else "",
            }
        )

    if rows:
        with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        metadata_path.write_text("", encoding="utf-8")

    return export_root


def _copy_if_exists(path_value: object, target_dir: Path, filename: str) -> Path | None:
    if not path_value:
        return None
    source = Path(str(path_value))
    if not source.exists():
        return None
    target = target_dir / filename
    shutil.copy2(source, target)
    return target


def _split_name(index: int, total: int) -> str:
    if total < 5:
        return "train"
    ratio = index / float(total)
    if ratio < 0.7:
        return "train"
    if ratio < 0.85:
        return "valid"
    return "test"
