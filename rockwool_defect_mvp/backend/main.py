from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
from shutil import copy2
from typing import Any
import unicodedata

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.config import AppConfig, ensure_runtime_directories, load_config
from src.storage.sqlite_store import SQLiteStore
from src.vision.defect_taxonomy import PIPELINE_STEPS, ordered_defects
from src.vision.inspection_pipeline import AnalysisView, process_frame, product_display_bbox
from src.vision.preprocessing import resize_image


app = FastAPI(title="MEGA QC API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FeedbackPayload(BaseModel):
    expectedVerdict: str
    expectedDefects: list[str] = []
    note: str = ""


def get_config() -> AppConfig:
    config_path = os.environ.get("MEGA_CONFIG_PATH")
    config = load_config(config_path) if config_path else load_config()
    ensure_runtime_directories(config)
    return config


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"ok": "true"}


@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    source: str = Query("upload", pattern="^(upload|camera)$"),
) -> dict[str, Any]:
    config = get_config()
    frame = await _read_upload_image(file)
    analysis = process_frame(frame, config)
    record_id = _save_analysis_record(config, frame, analysis, source=source, filename=file.filename or "image.jpg")
    record = SQLiteStore(config.database_path).fetch_inspection_record(record_id)
    if record is None:
        raise HTTPException(status_code=500, detail="Record could not be created")
    return {"ok": True, "item": _record_to_item(record)}


@app.get("/api/analyses")
def list_analyses(limit: int = 120, verdict: str | None = None) -> dict[str, Any]:
    config = get_config()
    model_filter = _ui_verdict_to_model(verdict) if verdict else None
    records = SQLiteStore(config.database_path).fetch_recent_inspection_records(
        limit=limit,
        model_result=model_filter,
    )
    return {"items": [_record_to_item(record) for record in records]}


@app.get("/api/stats")
def stats() -> dict[str, int]:
    config = get_config()
    rows = SQLiteStore(config.database_path).fetch_recent_inspection_records(limit=500)
    counts = {"total": len(rows), "kabul": 0, "red": 0, "uyari": 0}
    for row in rows:
        verdict = _model_to_ui_verdict(row.get("model_result"))
        if verdict == "KABUL":
            counts["kabul"] += 1
        elif verdict == "RED":
            counts["red"] += 1
        else:
            counts["uyari"] += 1
    return counts


@app.get("/api/reference")
def reference() -> dict[str, Any]:
    config = get_config()
    rows = SQLiteStore(config.database_path).fetch_recent_inspection_records(limit=500, model_result="SAGLAM")
    if not rows:
        return {"reference": None}

    samples = []
    for row in rows[:20]:
        image = _read_optional_image(row.get("image_path"))
        if image is None:
            continue
        analysis = process_frame(image, config)
        if analysis.roi is None:
            continue
        hsv = cv2.cvtColor(analysis.roi, cv2.COLOR_BGR2HSV)
        samples.append(np.median(hsv.reshape(-1, 3), axis=0))

    if not samples:
        return {"reference": None}
    mean_hsv = np.mean(np.vstack(samples), axis=0)
    return {
        "reference": {
            "n": len(samples),
            "meanH": float(mean_hsv[0]),
            "meanS": float(mean_hsv[1] / 255 * 100),
            "meanV": float(mean_hsv[2] / 255 * 100),
        }
    }


@app.get("/api/defect-types")
def defect_types() -> dict[str, Any]:
    return {
        "pipeline": PIPELINE_STEPS,
        "items": [
            {
                "type": meta.defect_type,
                "label": meta.label,
                "category": meta.category,
                "overlayColor": meta.overlay_color,
                "strategy": meta.strategy,
                "description": meta.description,
                "decisionImpact": meta.decision_impact,
            }
            for meta in ordered_defects()
        ],
    }


@app.get("/api/calibration/metrics")
def calibration_metrics() -> dict[str, Any]:
    config = get_config()
    rows = SQLiteStore(config.database_path).fetch_operator_feedback(limit=1000)
    return _calibration_metrics(rows)


@app.post("/api/analyses/{record_id}/feedback")
def save_feedback(record_id: int, payload: FeedbackPayload) -> dict[str, Any]:
    config = get_config()
    store = SQLiteStore(config.database_path)
    record = store.fetch_inspection_record(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Record not found")

    expected_verdict = payload.expectedVerdict.upper()
    if expected_verdict not in {"KABUL", "RED", "UYARI"}:
        raise HTTPException(status_code=400, detail="Unsupported expected verdict")

    known_defects = {meta.defect_type for meta in ordered_defects()}
    expected_defects = sorted({item for item in payload.expectedDefects if item in known_defects})
    feedback_id = store.insert_operator_feedback(
        record_id=record_id,
        expected_verdict=expected_verdict,
        expected_defects=json.dumps(expected_defects, ensure_ascii=False),
        note=payload.note.strip()[:500],
        created_at=datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
    )
    store.update_operator_feedback(
        record_id,
        operator_label=_operator_label_from_feedback(expected_verdict, expected_defects),
        operator_note=payload.note.strip()[:500],
        is_model_wrong=_model_to_ui_verdict(record.get("model_result")) != expected_verdict
        or set(_defect_types_from_record(record)) != set(expected_defects),
    )
    metrics = _calibration_metrics(store.fetch_operator_feedback(limit=1000))
    return {"ok": True, "feedbackId": feedback_id, "metrics": metrics}


@app.get("/api/images/{record_id}")
def get_image(record_id: int, v: str = Query("overlay", pattern="^(raw|overlay|previous)$")) -> FileResponse:
    config = get_config()
    record = SQLiteStore(config.database_path).fetch_inspection_record(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Record not found")
    path_key = {"raw": "image_path", "overlay": "overlay_path", "previous": "previous_overlay_path"}[v]
    path_value = record.get(path_key)
    if not path_value:
        raise HTTPException(status_code=404, detail="Image not available")
    path = Path(str(path_value))
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    return FileResponse(path, media_type="image/jpeg", filename=path.name)


@app.post("/api/analyses/{record_id}/reprocess")
def reprocess(record_id: int) -> dict[str, Any]:
    config = get_config()
    store = SQLiteStore(config.database_path)
    record = store.fetch_inspection_record(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Record not found")

    raw_image = _read_optional_image(record.get("image_path"))
    if raw_image is None:
        raise HTTPException(status_code=404, detail="Raw image not found")

    analysis = process_frame(raw_image, config)
    previous_overlay_path = _archive_current_overlay(record, config)
    overlay_path = Path(str(record.get("overlay_path") or ""))
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    _write_image(overlay_path, analysis.overlay)

    decision_label = analysis.decision.label if analysis.decision else "SUPHELI"
    anomaly_score = analysis.decision.anomaly_score if analysis.decision else 1.0
    updated = store.update_model_result(
        record_id,
        decision_label,
        anomaly_score,
        _roi_confidence_from_analysis(analysis),
        _rule_score(analysis, "edge_damage"),
        _rule_score(analysis, "deformation"),
        _rule_score(analysis, "color_anomaly"),
        _rule_score(analysis, "glass_burn"),
        _rule_score(analysis, "raw_fiber"),
        _rule_score(analysis, "dark_crack"),
        _rule_score(analysis, "local_anomaly"),
        previous_overlay_path=previous_overlay_path,
        previous_model_result=str(record.get("model_result") or ""),
        previous_anomaly_score=float(record.get("anomaly_score") or 0.0),
        last_reprocessed_at=datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
    )
    if not updated:
        raise HTTPException(status_code=500, detail="Record could not be updated")
    refreshed = store.fetch_inspection_record(record_id)
    return {"ok": True, "item": _record_to_item(refreshed)}


@app.delete("/api/analyses/{record_id}")
def delete_analysis(record_id: int) -> dict[str, Any]:
    config = get_config()
    deleted = SQLiteStore(config.database_path).delete_inspection_record(record_id)
    if deleted is None:
        raise HTTPException(status_code=404, detail="Record not found")

    for path_key in ("image_path", "overlay_path", "previous_overlay_path"):
        path_value = deleted.get(path_key)
        if not path_value:
            continue
        path = Path(str(path_value))
        if path.exists():
            path.unlink()
    return {"ok": True}


async def _read_upload_image(file: UploadFile) -> np.ndarray:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    data = np.frombuffer(content, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Unsupported image")
    return frame


def _save_analysis_record(
    config: AppConfig,
    frame: np.ndarray,
    analysis: AnalysisView,
    *,
    source: str,
    filename: str,
) -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    raw_dir = config.database_path.parents[1] / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    config.output_overlay_path.mkdir(parents=True, exist_ok=True)

    suffix = Path(filename).suffix.lower() or ".jpg"
    safe_stem = _safe_ascii_stem(filename)
    raw_path = raw_dir / f"{timestamp}_{source}_{safe_stem}{suffix}"
    overlay_path = config.output_overlay_path / f"{timestamp}_{source}_{safe_stem}_overlay.jpg"

    resized = resize_image(frame)
    _write_image(raw_path, resized)
    _write_image(overlay_path, analysis.overlay)

    decision_label = analysis.decision.label if analysis.decision else "SUPHELI"
    anomaly_score = analysis.decision.anomaly_score if analysis.decision else 1.0
    record = {
        "timestamp": timestamp,
        "image_path": str(raw_path),
        "overlay_path": str(overlay_path),
        "product_type": source,
        "model_result": decision_label,
        "anomaly_score": anomaly_score,
        "roi_confidence": _roi_confidence_from_analysis(analysis),
        "edge_damage_score": _rule_score(analysis, "edge_damage"),
        "deformation_score": _rule_score(analysis, "deformation"),
        "color_anomaly_score": _rule_score(analysis, "color_anomaly"),
        "glass_burn_score": _rule_score(analysis, "glass_burn"),
        "raw_fiber_score": _rule_score(analysis, "raw_fiber"),
        "crack_score": _rule_score(analysis, "dark_crack"),
        "local_anomaly_score": _rule_score(analysis, "local_anomaly"),
        "operator_label": _default_operator_label(decision_label),
        "operator_note": "",
        "is_model_wrong": 0,
        "is_confirmed": 1,
    }
    return SQLiteStore(config.database_path).insert_inspection_record(record)


def _record_to_item(record: dict[str, Any] | None) -> dict[str, Any]:
    if record is None:
        raise HTTPException(status_code=404, detail="Record not found")

    record_id = str(record["id"])
    source = str(record.get("product_type") or "upload")
    if source not in {"upload", "camera"}:
        source = "upload"
    verdict = _model_to_ui_verdict(record.get("model_result"))
    confidence = max(0.0, min(100.0, float(record.get("anomaly_score") or 0.0) * 100))
    return {
        "id": record_id,
        "created_at": _timestamp_to_epoch_ms(record.get("timestamp")),
        "filename": f"#{record_id} - {_display_timestamp(record.get('timestamp'))} - {verdict}",
        "source": source,
        "verdict": verdict,
        "confidence": confidence,
        "roiConfidence": round(float(record.get("roi_confidence") or 0.0), 3),
        "defects": _defects_from_record(record),
        "pipeline": PIPELINE_STEPS,
        "metrics": _metrics_from_record(record),
        "originalSrc": f"/api/images/{record_id}?v=raw",
        "overlaySrc": f"/api/images/{record_id}?v=overlay",
        "previousOverlaySrc": f"/api/images/{record_id}?v=previous" if record.get("previous_overlay_path") else None,
        "meta": f"{_display_timestamp(record.get('timestamp'))} · {'Kamera' if source == 'camera' else 'Yükleme'}",
    }


def _defects_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    if record.get("model_result") == "SAGLAM":
        return []

    taxonomy_defects = []
    display_thresholds = {
        "edge_damage": 0.45,
        "deformation": 0.24,
        "glass_burn": 0.25,
        "raw_fiber": 0.25,
        "color_anomaly": 0.30,
        "dark_crack": 0.32,
        "local_anomaly": 0.60,
    }
    for meta in ordered_defects():
        score = float(record.get(meta.score_key) or 0.0)
        if score < display_thresholds.get(meta.defect_type, 0.25):
            continue
        taxonomy_defects.append(
            {
                "type": meta.defect_type,
                "label": meta.label,
                "score": round(score * 100),
                "severity": _severity(score),
                "confidence": round(score * 100),
                "reason": _defect_reason(meta.defect_type, score),
                "category": meta.category,
                "overlayColor": meta.overlay_color,
                "strategy": meta.strategy,
                "description": meta.description,
                "decisionImpact": meta.decision_impact,
            }
        )
    return taxonomy_defects


def _metrics_from_record(record: dict[str, Any]) -> dict[str, float]:
    return {
        "meanH": 0.0,
        "meanS": 0.0,
        "meanV": 0.0,
        "brightSpotRatio": float(record.get("color_anomaly_score") or 0.0),
        "darkSpotRatio": float(record.get("glass_burn_score") or record.get("local_anomaly_score") or 0.0),
        "longLineScore": float(record.get("crack_score") or 0.0),
        "rectangularity": max(0.0, 1.0 - float(record.get("deformation_score") or record.get("edge_damage_score") or 0.0)),
        "squarenessDeg": float(record.get("edge_damage_score") or 0.0) * 10,
    }


def _roi_confidence_from_analysis(analysis: AnalysisView) -> float:
    product = analysis.product
    if product is None:
        return 0.0
    x, y, width, height = product.shape_bbox
    image_height, image_width = analysis.original.shape[:2]
    bbox_area_ratio = (width * height) / float(max(1, image_width * image_height))
    aspect_ratio = max(width, height) / float(max(1, min(width, height)))
    area_score = min(1.0, max(0.0, (bbox_area_ratio - 0.12) / 0.58))
    aspect_score = 1.0 - min(1.0, abs(aspect_ratio - 1.85) / 2.3)
    border_score = 1.0
    if x <= 0 or y <= 0 or x + width >= image_width or y + height >= image_height:
        border_score = 0.82
    return round(max(0.0, min(1.0, area_score * 0.55 + aspect_score * 0.35 + border_score * 0.10)), 3)


def _defect_reason(defect_type: str, score: float) -> str:
    if defect_type == "dark_crack":
        return "İnce, uzun ve koyu çizgisel bileşenler eşik üstünde."
    if defect_type == "glass_burn":
        return "Işık dengesi sonrası geniş koyu/kahverengi leke sinyali var."
    if defect_type == "raw_fiber":
        return "Parlak camsı lif, açık düşük doygunluk veya kabarık lif dokusu bulundu."
    if defect_type == "edge_damage":
        return "Ürün kenarında kontur/boşluk düzensizliği bulundu."
    if defect_type == "deformation":
        return "Dikdörtgen forma göre şekil sapması ölçüldü."
    if defect_type == "color_anomaly":
        return "Ürün renginden ayrışan kompakt renk/leke alanı bulundu."
    if defect_type == "local_anomaly":
        return "Genel doku/parlaklık heatmap sinyali eşik üstünde."
    return f"Skor eşik üstünde: {score:.3f}"


def _calibration_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    defect_keys = [meta.defect_type for meta in ordered_defects()]
    per_defect = {key: {"tp": 0, "fp": 0, "fn": 0} for key in defect_keys}
    verdict = {"total": 0, "correct": 0}
    mismatches = []

    for row in rows:
        if not row.get("model_result"):
            continue
        verdict["total"] += 1
        expected_verdict = str(row.get("expected_verdict") or "")
        predicted_verdict = _model_to_ui_verdict(row.get("model_result"))
        if expected_verdict == predicted_verdict:
            verdict["correct"] += 1

        expected_defects = set(_load_expected_defects(row.get("expected_defects")))
        predicted_defects = set(_defect_types_from_record(row))
        false_positive = sorted(predicted_defects - expected_defects)
        false_negative = sorted(expected_defects - predicted_defects)
        for defect in defect_keys:
            if defect in expected_defects and defect in predicted_defects:
                per_defect[defect]["tp"] += 1
            elif defect not in expected_defects and defect in predicted_defects:
                per_defect[defect]["fp"] += 1
            elif defect in expected_defects and defect not in predicted_defects:
                per_defect[defect]["fn"] += 1

        if expected_verdict != predicted_verdict or false_positive or false_negative:
            mismatches.append(
                {
                    "recordId": row.get("record_id"),
                    "expectedVerdict": expected_verdict,
                    "predictedVerdict": predicted_verdict,
                    "falsePositive": false_positive,
                    "falseNegative": false_negative,
                    "note": row.get("note") or "",
                }
            )

    per_defect_metrics = []
    for defect, values in per_defect.items():
        tp = values["tp"]
        fp = values["fp"]
        fn = values["fn"]
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        per_defect_metrics.append(
            {
                "type": defect,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
            }
        )

    accuracy = verdict["correct"] / verdict["total"] if verdict["total"] else 1.0
    return {
        "feedbackCount": verdict["total"],
        "verdictAccuracy": round(accuracy, 3),
        "perDefect": per_defect_metrics,
        "mismatches": mismatches[:50],
        "nextPhases": [
            "Çatlak, cam yanığı ve kenar deformasyon kuralları FP/FN metriklerine göre ayrı ayrı güçlendirilecek.",
            "Yeterli etiketli veri birikince segmentation model eğitim seti çıkarılacak.",
            "VLM sadece açıklama, rapor ve operatör özetlerinde kullanılacak; karar motorunun yerine geçmeyecek.",
        ],
    }


def _load_expected_defects(value: object) -> list[str]:
    if not value:
        return []
    try:
        loaded = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in loaded if isinstance(item, str)]


def _defect_types_from_record(record: dict[str, Any]) -> list[str]:
    if record.get("model_result") == "SAGLAM":
        return []
    return [item["type"] for item in _defects_from_record(record)]


def _operator_label_from_feedback(expected_verdict: str, expected_defects: list[str]) -> str:
    if expected_verdict == "KABUL":
        return "saglam"
    if expected_defects:
        return ",".join(expected_defects)
    return "diger"


def _model_to_ui_verdict(label: object) -> str:
    if label == "SAGLAM":
        return "KABUL"
    if label == "HATALI":
        return "RED"
    return "UYARI"


def _ui_verdict_to_model(verdict: str | None) -> str | None:
    if verdict == "KABUL":
        return "SAGLAM"
    if verdict == "RED":
        return "HATALI"
    if verdict == "UYARI":
        return "SUPHELI"
    return None


def _default_operator_label(model_label: str) -> str:
    if model_label == "SAGLAM":
        return "saglam"
    if model_label == "HATALI":
        return "diger"
    return "diger"


def _severity(score: float) -> str:
    if score >= 0.6:
        return "high"
    if score >= 0.3:
        return "medium"
    return "low"


def _rule_score(analysis: AnalysisView, name: str) -> float:
    return float(analysis.rule_results.get(name, {}).get("score", 0.0))


def _archive_current_overlay(record: dict[str, Any], config: AppConfig) -> str | None:
    overlay_path = Path(str(record.get("overlay_path") or ""))
    if not overlay_path.exists():
        return None
    history_dir = config.output_overlay_path / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archive_path = history_dir / f"record_{record['id']}_{timestamp}_previous.jpg"
    copy2(overlay_path, archive_path)
    return str(archive_path)


def _read_optional_image(path_value: object) -> np.ndarray | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if not path.exists():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _write_image(path: Path, image: np.ndarray) -> None:
    extension = path.suffix.lower() or ".jpg"
    if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
        extension = ".jpg"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise HTTPException(status_code=500, detail=f"Image could not be written: {path}")
    path.write_bytes(encoded.tobytes())


def _safe_ascii_stem(filename: str) -> str:
    stem = Path(filename).stem or "image"
    normalized = unicodedata.normalize("NFKD", stem)
    ascii_stem = normalized.encode("ascii", "ignore").decode("ascii")
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in ascii_stem)
    safe = "_".join(part for part in safe.split("_") if part)
    return (safe or "image")[:80]


def _timestamp_to_epoch_ms(value: object) -> int:
    if not value:
        return int(datetime.now().timestamp() * 1000)
    raw = str(value)
    try:
        parsed = datetime.strptime(raw, "%Y%m%d_%H%M%S_%f")
    except ValueError:
        return int(datetime.now().timestamp() * 1000)
    return int(parsed.timestamp() * 1000)


def _display_timestamp(value: object) -> str:
    if not value:
        return "-"
    raw = str(value)
    try:
        parsed = datetime.strptime(raw, "%Y%m%d_%H%M%S_%f")
    except ValueError:
        return raw
    return parsed.strftime("%d.%m.%Y %H:%M")
