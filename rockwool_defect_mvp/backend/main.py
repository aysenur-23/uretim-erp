from __future__ import annotations

from datetime import datetime
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

from src.config import AppConfig, ensure_runtime_directories, load_config
from src.storage.sqlite_store import SQLiteStore
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
        _rule_score(analysis, "edge_damage"),
        _rule_score(analysis, "color_anomaly"),
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
        "edge_damage_score": _rule_score(analysis, "edge_damage"),
        "color_anomaly_score": _rule_score(analysis, "color_anomaly"),
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
        "defects": _defects_from_record(record),
        "metrics": _metrics_from_record(record),
        "originalSrc": f"/api/images/{record_id}?v=raw",
        "overlaySrc": f"/api/images/{record_id}?v=overlay",
        "previousOverlaySrc": f"/api/images/{record_id}?v=previous" if record.get("previous_overlay_path") else None,
        "meta": f"{_display_timestamp(record.get('timestamp'))} · {'Kamera' if source == 'camera' else 'Yükleme'}",
    }


def _defects_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    if record.get("model_result") == "SAGLAM":
        return []

    specs = [
        ("edge_damage_score", "edge_damage", "Kenar"),
        ("color_anomaly_score", "color_anomaly", "Renk/Leke"),
        ("crack_score", "dark_crack", "Çatlak"),
        ("local_anomaly_score", "local_anomaly", "Yerel anomali"),
    ]
    defects = []
    for key, defect_type, label in specs:
        score = float(record.get(key) or 0.0)
        if score < 0.25:
            continue
        defects.append(
            {
                "type": defect_type,
                "label": label,
                "score": round(score * 100),
                "severity": _severity(score),
            }
        )
    return defects


def _metrics_from_record(record: dict[str, Any]) -> dict[str, float]:
    return {
        "meanH": 0.0,
        "meanS": 0.0,
        "meanV": 0.0,
        "brightSpotRatio": float(record.get("color_anomaly_score") or 0.0),
        "darkSpotRatio": float(record.get("local_anomaly_score") or 0.0),
        "longLineScore": float(record.get("crack_score") or 0.0),
        "rectangularity": max(0.0, 1.0 - float(record.get("edge_damage_score") or 0.0)),
        "squarenessDeg": float(record.get("edge_damage_score") or 0.0) * 10,
    }


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
