from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.vision.inspection_pipeline import process_frame


DISPLAY_THRESHOLDS = {
    "edge_damage": 0.45,
    "deformation": 0.24,
    "glass_burn": 0.25,
    "raw_fiber": 0.25,
    "color_anomaly": 0.30,
    "dark_crack": 0.30,
    "local_anomaly": 0.60,
}

MODEL_TO_UI = {
    "SAGLAM": "KABUL",
    "SUPHELI": "UYARI",
    "HATALI": "RED",
}


@dataclass(frozen=True)
class EvalItem:
    record_id: int
    image_path: Path
    expected_verdict: str
    expected_defects: set[str]
    note: str


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the local gallery calibration set.")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("calibration/gallery_ground_truth.json"),
        help="Ground-truth JSON path.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    config = load_config()
    items = _load_items(args.ground_truth, config.database_path)
    rows = [_evaluate_item(item, config) for item in items]
    summary = _summarize(rows)

    if args.json:
        print(json.dumps({"summary": summary, "items": rows}, ensure_ascii=False, indent=2))
    else:
        _print_report(rows, summary)

    return 0 if summary["all_exact"] else 1


def _load_items(ground_truth_path: Path, database_path: Path) -> list[EvalItem]:
    payload = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    records = _records_by_id(database_path)
    items: list[EvalItem] = []
    for raw in payload["items"]:
        record_id = int(raw["id"])
        record = records.get(record_id)
        if record is None:
            raise RuntimeError(f"Record #{record_id} is not in the local gallery database.")
        image_path = Path(str(record["image_path"]))
        if not image_path.exists():
            raise RuntimeError(f"Image for record #{record_id} does not exist: {image_path}")
        items.append(
            EvalItem(
                record_id=record_id,
                image_path=image_path,
                expected_verdict=str(raw["expected_verdict"]),
                expected_defects=set(raw.get("expected_defects", [])),
                note=str(raw.get("note", "")),
            )
        )
    return items


def _records_by_id(database_path: Path) -> dict[int, sqlite3.Row]:
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT id, image_path FROM inspection_records").fetchall()
    return {int(row["id"]): row for row in rows}


def _evaluate_item(item: EvalItem, config: Any) -> dict[str, Any]:
    frame = _read_image(item.image_path)
    analysis = process_frame(frame, config)
    model_label = analysis.decision.label if analysis.decision else "SUPHELI"
    predicted_verdict = MODEL_TO_UI.get(model_label, "UYARI")
    predicted_defects = _visible_defects(analysis.rule_results)

    expected_defects = item.expected_defects
    false_positive = sorted(predicted_defects - expected_defects)
    false_negative = sorted(expected_defects - predicted_defects)
    verdict_ok = predicted_verdict == item.expected_verdict
    defects_ok = not false_positive and not false_negative

    return {
        "id": item.record_id,
        "expected_verdict": item.expected_verdict,
        "predicted_verdict": predicted_verdict,
        "expected_defects": sorted(expected_defects),
        "predicted_defects": sorted(predicted_defects),
        "false_positive": false_positive,
        "false_negative": false_negative,
        "verdict_ok": verdict_ok,
        "defects_ok": defects_ok,
        "exact": verdict_ok and defects_ok,
        "note": item.note,
        "scores": {
            name: round(float(result.get("score", 0.0)), 4)
            for name, result in analysis.rule_results.items()
        },
    }


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def _visible_defects(rule_results: dict[str, dict[str, Any]]) -> set[str]:
    defects = set()
    for name, result in rule_results.items():
        score = float(result.get("score", 0.0))
        if score >= DISPLAY_THRESHOLDS.get(name, 0.25):
            defects.add(name)
    return defects


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    defect_types = sorted(
        {
            defect
            for row in rows
            for defect in set(row["expected_defects"]) | set(row["predicted_defects"])
        }
    )
    by_defect = {}
    for defect in defect_types:
        tp = sum(defect in row["expected_defects"] and defect in row["predicted_defects"] for row in rows)
        fp = sum(defect not in row["expected_defects"] and defect in row["predicted_defects"] for row in rows)
        fn = sum(defect in row["expected_defects"] and defect not in row["predicted_defects"] for row in rows)
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        by_defect[defect] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
        }

    return {
        "total": len(rows),
        "exact": sum(row["exact"] for row in rows),
        "verdict_ok": sum(row["verdict_ok"] for row in rows),
        "defects_ok": sum(row["defects_ok"] for row in rows),
        "all_exact": all(row["exact"] for row in rows),
        "by_defect": by_defect,
    }


def _print_report(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    print(f"Gallery calibration: {summary['exact']}/{summary['total']} exact")
    print(f"Verdict accuracy: {summary['verdict_ok']}/{summary['total']}")
    print(f"Defect-set accuracy: {summary['defects_ok']}/{summary['total']}")
    print()
    print("Per defect:")
    for defect, stats in summary["by_defect"].items():
        print(
            f"  {defect:14} TP={stats['tp']} FP={stats['fp']} FN={stats['fn']} "
            f"P={stats['precision']:.3f} R={stats['recall']:.3f}"
        )
    print()
    print("Mismatches:")
    mismatches = [row for row in rows if not row["exact"]]
    if not mismatches:
        print("  none")
        return
    for row in mismatches:
        print(
            f"  #{row['id']}: expected {row['expected_verdict']} {row['expected_defects']} "
            f"got {row['predicted_verdict']} {row['predicted_defects']} "
            f"FP={row['false_positive']} FN={row['false_negative']}"
        )
        print(f"      note: {row['note']}")
        print(f"      scores: {row['scores']}")


if __name__ == "__main__":
    raise SystemExit(main())
