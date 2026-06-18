from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from shutil import copy2
from time import sleep
from urllib.parse import quote

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.camera.uploaded_image_camera import UploadedImageCamera
from src.camera.webcam_camera import WebcamCamera
from src.config import AppConfig, ensure_runtime_directories, load_config
from src.dataset.dataset_manager import DatasetManager
from src.dataset.export_manager import export_dataset
from src.dataset.label_schema import OPERATOR_LABELS
from src.storage.sqlite_store import SQLiteStore
from src.vision.color_calibration import calibrate_product_hsv
from src.vision.inspection_pipeline import AnalysisView, process_frame, product_display_bbox
from src.vision.preprocessing import resize_image


CAMERA_RESOLUTIONS = {
    "640 x 480": (640, 480),
    "1280 x 720": (1280, 720),
    "1920 x 1080": (1920, 1080),
}

APP_TITLE = "Taş yünü kalite kontrol"
APP_SUBTITLE = "Önce plaka tespit edilir, sonra her hata kendine özel algoritmayla aranır."
GALLERY_CANVAS_SIZE = (640, 480)

DEFECT_SCAN_LABELS = {
    "edge_damage": "Kenar",
    "color_anomaly": "Renk / leke",
    "dark_crack": "Çatlak",
    "local_anomaly": "Yerel anomali",
}

LABEL_DISPLAY_NAMES = {
    "saglam": "Sağlam",
    "catlak": "Çatlak",
    "kenar_kirigi": "Kenar kırığı",
    "kose_kirigi": "Köşe kırığı",
    "renk_bozuklugu": "Renk bozukluğu",
    "yanik": "Yanık",
    "ezilme": "Ezilme",
    "nem_suphesi": "Nem şüphesi",
    "recine_problemi": "Reçine problemi",
    "kalinlik_boyut_suphesi": "Kalınlık/boyut şüphesi",
    "diger": "Diğer",
}


def mega_logo_svg(class_name: str = "mega-logo") -> str:
    svg = """<svg viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg">
<defs>
<linearGradient id="mega-logo-gradient" x1="0" x2="1" y1="0" y2="1">
<stop offset="0" stop-color="#E11D2A"/>
<stop offset="1" stop-color="#B81522"/>
</linearGradient>
</defs>
<rect x="6" y="6" width="48" height="48" rx="10" transform="rotate(45 30 30)" fill="url(#mega-logo-gradient)"/>
<rect x="13" y="13" width="34" height="34" rx="7" transform="rotate(45 30 30)" fill="#1E2A6B"/>
<text x="30" y="38" text-anchor="middle" font-family="Arial, sans-serif" font-weight="900" font-size="22" fill="#fff">M</text>
</svg>"""
    return f'<img class="{class_name}" src="data:image/svg+xml;utf8,{quote(svg)}" alt="MEGA" />'


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def fit_image_to_canvas(
    frame: np.ndarray,
    canvas_size: tuple[int, int] = GALLERY_CANVAS_SIZE,
    fill_color: tuple[int, int, int] = (18, 24, 32),
) -> np.ndarray:
    canvas_width, canvas_height = canvas_size
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return np.full((canvas_height, canvas_width, 3), fill_color, dtype=np.uint8)

    scale = min(canvas_width / width, canvas_height / height)
    target_width = max(1, int(width * scale))
    target_height = max(1, int(height * scale))
    resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((canvas_height, canvas_width, 3), fill_color, dtype=np.uint8)
    x = (canvas_width - target_width) // 2
    y = (canvas_height - target_height) // 2
    canvas[y : y + target_height, x : x + target_width] = resized
    return canvas


def gallery_image(frame: np.ndarray) -> np.ndarray:
    return bgr_to_rgb(fit_image_to_canvas(frame))


def decision_badge(label: str) -> None:
    classes = {
        "HATALI": "status-badge status-defect",
        "SUPHELI": "status-badge status-warning",
        "SAGLAM": "status-badge status-ok",
    }
    st.markdown(
        f'<span class="{classes.get(label, "status-badge")}">{display_decision(label)}</span>',
        unsafe_allow_html=True,
    )


def display_label(label: str) -> str:
    if label == "Tum":
        return "Tüm"
    return LABEL_DISPLAY_NAMES.get(label, label)


def display_decision(label: str) -> str:
    return {
        "Tum": "Tüm",
        "SAGLAM": "SAĞLAM",
        "SUPHELI": "ŞÜPHELİ",
        "HATALI": "HATALI",
    }.get(label, label)


def upload_widget_key() -> str:
    return f"uploaded_image_{st.session_state.get('upload_widget_version', 0)}"


def update_upload_frame(widget_key: str) -> bool:
    uploaded_file = st.session_state.get(widget_key)
    if uploaded_file is None:
        return False

    image_bytes = uploaded_file.getvalue()
    signature = sha256(image_bytes).hexdigest()
    if st.session_state.get("upload_signature") == signature:
        return False

    camera = UploadedImageCamera(image_bytes)
    st.session_state.upload_frame = camera.get_frame()
    st.session_state.upload_signature = signature
    st.session_state.upload_name = uploaded_file.name
    st.session_state.upload_widget_version = st.session_state.get("upload_widget_version", 0) + 1
    return True


def capture_camera_frame(config: AppConfig, resolution: tuple[int, int]) -> None:
    camera = None
    try:
        camera = WebcamCamera(config.webcam_index, width=resolution[0], height=resolution[1])
        st.session_state.camera_frame = camera.get_frame()
    finally:
        if camera is not None:
            camera.release()


def capture_live_camera_tick(config: AppConfig, resolution: tuple[int, int]) -> AnalysisView:
    camera = None
    try:
        camera = WebcamCamera(config.webcam_index, width=resolution[0], height=resolution[1])
        frame = camera.get_frame()
        st.session_state.camera_frame = frame
        return process_frame(frame, config)
    finally:
        if camera is not None:
            camera.release()


def render_summary(analysis: AnalysisView) -> None:
    if analysis.product is None or analysis.decision is None:
        st.warning("Ürün bulunamadı")
        return

    decision_col, score_col, area_col = st.columns(3)
    with decision_col:
        decision_badge(analysis.decision.label)
    score_col.metric("Skor", f"{analysis.decision.anomaly_score:.3f}")
    area_col.metric("Alan", f"{analysis.product.area_ratio:.3f}")


def render_header(config: AppConfig) -> None:
    st.markdown(
        f"""<div class="mega-hero">
<div class="mega-hero-bg"></div>
<div class="mega-hero-glow"></div>
<div class="mega-hero-content">
<div class="mega-hero-topline">
<div class="mega-hero-heading">
<div class="mega-pill">Mega Insulation Solutions</div>
<h1>{APP_TITLE}</h1>
<p>{APP_SUBTITLE}</p>
</div>
<a class="mega-settings-link" href="#settings">Ayarlar →</a>
</div>
</div>
</div>""",
        unsafe_allow_html=True,
    )


def render_quality_overview(config: AppConfig) -> None:
    summary = SQLiteStore(config.database_path).fetch_quality_summary()
    totals = summary.get("totals", {})
    total_count = int(totals.get("total_count") or 0)
    wrong_count = int(totals.get("wrong_count") or 0)
    avg_score = _format_metric(totals.get("avg_anomaly_score"))
    avg_crack = _format_metric(totals.get("avg_crack_score"))

    defect_count = 0
    for row in summary.get("model_result_counts", []):
        if row.get("label") == "HATALI":
            defect_count = int(row.get("count") or 0)
            break
    defect_rate = (defect_count / total_count) if total_count else 0.0

    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <span>Denetim kaydı</span>
                <strong>{total_count}</strong>
            </div>
            <div class="kpi-card">
                <span>Hatalı oranı</span>
                <strong>{defect_rate:.1%}</strong>
                <small>{defect_count} kayıt</small>
            </div>
            <div class="kpi-card">
                <span>Ortalama skor</span>
                <strong>{avg_score}</strong>
            </div>
            <div class="kpi-card">
                <span>Çatlak ortalaması</span>
                <strong>{avg_crack}</strong>
                <small>model yanlış {wrong_count}</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _analysis_status_text(analysis: AnalysisView) -> str:
    if analysis.product is None or analysis.decision is None:
        return "ürün bulunamadı"
    return f"{display_decision(analysis.decision.label)} / skor {analysis.decision.anomaly_score:.3f}"


def render_rule_summary(analysis: AnalysisView) -> None:
    if analysis.decision is None:
        return

    st.caption("Karar nedenleri")
    for reason in analysis.decision.reasons:
        st.caption(f"- {reason}")

    rows = [
        {
            "kural": name,
            "skor": result.get("score", 0.0),
            "supheli": result.get("is_suspicious", False),
            "mesaj": result.get("message", ""),
        }
        for name, result in analysis.rule_results.items()
    ]
    with st.expander("Kural skorları", expanded=False):
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_analysis_scan_summary(analysis: AnalysisView) -> None:
    if not analysis.rule_results:
        return

    st.caption("Ayrı hata taramaları")
    rows = []
    for rule_name, display_name in DEFECT_SCAN_LABELS.items():
        result = analysis.rule_results.get(rule_name, {})
        rows.append(
            {
                "tarama": display_name,
                "sonuç": "Riskli" if bool(result.get("is_suspicious", False)) else "Normal",
                "skor": _format_metric(result.get("score")),
                "strateji": result.get("strategy", "-"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_record_scan_summary(record: dict, config: AppConfig) -> None:
    rows = [
        {
            "tarama": "Kenar",
            "sonuç": _score_status(record.get("edge_damage_score"), config.edge_damage_threshold),
            "skor": _format_metric(record.get("edge_damage_score")),
            "strateji": "Kontur / kenar bütünlüğü",
        },
        {
            "tarama": "Renk / leke",
            "sonuç": _score_status(record.get("color_anomaly_score"), config.color_anomaly_threshold),
            "skor": _format_metric(record.get("color_anomaly_score")),
            "strateji": "Lab renk sapması / leke bölgesi",
        },
        {
            "tarama": "Çatlak",
            "sonuç": _score_status(record.get("crack_score"), 0.42),
            "skor": _format_metric(record.get("crack_score")),
            "strateji": "Dikey koyu çizgi / yarık bileşeni",
        },
        {
            "tarama": "Yerel anomali",
            "sonuç": _score_status(record.get("local_anomaly_score"), config.local_anomaly_threshold),
            "skor": _format_metric(record.get("local_anomaly_score")),
            "strateji": "Lokal residual heatmap",
        },
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_record_scan_line(record: dict, config: AppConfig) -> None:
    items = [
        ("Kenar", record.get("edge_damage_score"), config.edge_damage_threshold),
        ("Renk", record.get("color_anomaly_score"), config.color_anomaly_threshold),
        ("Çatlak", record.get("crack_score"), 0.42),
        ("Yerel", record.get("local_anomaly_score"), config.local_anomaly_threshold),
    ]
    parts = [
        f"{label}: {_score_status(score, threshold)}"
        for label, score, threshold in items
    ]
    st.caption(" | ".join(parts))


def render_analysis_scan_line(analysis: AnalysisView) -> None:
    if not analysis.rule_results:
        return

    parts = []
    for rule_name, label in DEFECT_SCAN_LABELS.items():
        result = analysis.rule_results.get(rule_name, {})
        status = "Riskli" if bool(result.get("is_suspicious", False)) else "Normal"
        parts.append(f"{label}: {status}")
    st.caption(" | ".join(parts))


def _score_status(value: object, threshold: float) -> str:
    if value is None:
        return "-"
    return "Riskli" if float(value) >= threshold else "Normal"


def render_technical_details(analysis: AnalysisView) -> None:
    with st.expander("Teknik detaylar", expanded=False):
        if analysis.product is not None:
            st.caption(f"Şekle göre bbox: {product_display_bbox(analysis.product)}")
        render_analysis_scan_summary(analysis)
        render_rule_summary(analysis)
        if analysis.roi is not None:
            st.image(bgr_to_rgb(analysis.roi), caption="ROI", use_container_width=True)
        if analysis.mask_preview is not None:
            st.image(bgr_to_rgb(analysis.mask_preview), caption="Maske", use_container_width=True)
        if analysis.heatmap_preview is not None:
            st.image(bgr_to_rgb(analysis.heatmap_preview), caption="Yerel anomali heatmap", use_container_width=True)
        if analysis.crack_preview is not None:
            st.image(bgr_to_rgb(analysis.crack_preview), caption="Koyu çizgi adayları", use_container_width=True)


def render_save_form(
    source_name: str,
    frame: np.ndarray,
    analysis: AnalysisView,
    config: AppConfig,
) -> None:
    if analysis.product is None or analysis.decision is None:
        return

    with st.form(f"{source_name}_save_form", clear_on_submit=False):
        label_col, button_col = st.columns([2, 1])
        with label_col:
            operator_label = st.selectbox(
                "Etiket",
                OPERATOR_LABELS,
                format_func=display_label,
                key=f"{source_name}_operator_label",
                label_visibility="collapsed",
            )
        with button_col:
            submitted = st.form_submit_button("Kayıt oluştur", type="primary", use_container_width=True)
        operator_note = st.text_input(
            "Not",
            placeholder="Opsiyonel not",
            key=f"{source_name}_operator_note",
            label_visibility="collapsed",
        )

    if submitted:
        try:
            manager = DatasetManager(config)
            record_id = manager.save_inspection(
                raw_frame=resize_image(frame),
                overlay_frame=analysis.overlay,
                product_type="tas_yunu_panel",
                decision=analysis.decision,
                rule_results=analysis.rule_results,
                operator_label=operator_label,
                operator_note=operator_note,
            )
            st.success(f"Denetim kaydı oluşturuldu. ID: {record_id}")
        except Exception as exc:
            st.error(f"Kayıt sırasında hata oluştu: {exc}")


def render_scan_result(source_name: str, frame_key: str, config: AppConfig) -> None:
    frame = st.session_state.get(frame_key)
    if frame is None:
        st.info("Henüz görüntü yok.")
        return

    analysis = process_frame(frame, config)
    image_col, action_col = st.columns([2, 1])
    with image_col:
        st.image(bgr_to_rgb(analysis.overlay), use_container_width=True)
    with action_col:
        render_summary(analysis)
        if st.button("Bu renkle kalibre et", key=f"calibrate_{source_name}", use_container_width=True):
            calibrate_from_frame(frame, config)
            st.rerun()
        render_save_form(source_name, frame, analysis, config)
        render_technical_details(analysis)


def calibrate_from_frame(frame: np.ndarray, config: AppConfig) -> None:
    try:
        lower, upper = calibrate_product_hsv(resize_image(frame), config)
        st.session_state.product_hsv_lower = lower
        st.session_state.product_hsv_upper = upper
        st.success(f"Renk kalibre edildi: {lower} - {upper}")
    except Exception as exc:
        st.error(f"Renk kalibrasyonu yapılamadı: {exc}")


def render_upload_workflow(config: AppConfig) -> None:
    st.markdown('<div class="mega-drop-title">Görseli buraya sürükle ya da seç</div>', unsafe_allow_html=True)
    st.markdown('<div class="mega-drop-subtitle">JPG, PNG - tekil seçim destekli</div>', unsafe_allow_html=True)
    widget_key = upload_widget_key()
    uploaded_file = st.file_uploader(
        "Dosya Seç",
        type=["jpg", "jpeg", "png"],
        key=widget_key,
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        try:
            if update_upload_frame(widget_key):
                st.rerun()
            st.caption(f"Son yüklenen: {st.session_state.get('upload_name', uploaded_file.name)}")
            st.caption("Analiz sonucu denetim kayıtlarında kaydedilmemiş kart olarak görünecek.")
        except Exception as exc:
            st.error(f"Yüklenen görsel okunamadı: {exc}")
    elif "upload_name" in st.session_state:
        st.caption(f"Son yüklenen: {st.session_state.upload_name}")
        st.caption("Yeni görüntü seçene kadar son analiz denetim kayıtlarında tutulur.")


def render_camera_workflow(config: AppConfig) -> None:
    st.markdown('<div class="mega-drop-title">Canlı kamera</div>', unsafe_allow_html=True)
    st.markdown('<div class="mega-drop-subtitle">Kamera açıkken ürün canlı görüntü üzerinden taranır.</div>', unsafe_allow_html=True)
    resolution_label = st.selectbox("Kamera çözünürlüğü", list(CAMERA_RESOLUTIONS), index=0)
    resolution = CAMERA_RESOLUTIONS[resolution_label]
    live_col, stop_col, capture_col = st.columns(3)
    with live_col:
        if st.button("Kamerayı Aç", type="primary", use_container_width=True):
            st.session_state.camera_live_active = True
            st.rerun()
    with stop_col:
        if st.button("Kapat", use_container_width=True):
            st.session_state.camera_live_active = False
            st.rerun()
    with capture_col:
        if st.button("Çek ve Analiz Et", use_container_width=True):
            try:
                capture_camera_frame(config, resolution)
                st.rerun()
            except Exception as exc:
                st.error(f"Kamera görüntüsü alınamadı: {exc}")

    if st.session_state.get("camera_live_active", False):
        try:
            analysis = capture_live_camera_tick(config, resolution)
            st.caption(f"Canlı: {_analysis_status_text(analysis)}")
            st.image(bgr_to_rgb(analysis.overlay), use_container_width=True)
            if analysis.product is None:
                st.warning("Kadrajda ürün bulunamadı. Ürünü kameraya hizala veya ışığı artır.")
            sleep(0.35)
            st.rerun()
        except Exception as exc:
            st.session_state.camera_live_active = False
            st.error(f"Canlı kamera başlatılamadı: {exc}")

    if "camera_frame" in st.session_state:
        st.caption("Son kamera karesi denetim kayıtlarında kaydedilmemiş kart olarak görünüyor.")


def render_gallery(config: AppConfig) -> None:
    store = SQLiteStore(config.database_path)
    summary = store.fetch_quality_summary()
    totals = summary.get("totals", {})
    total_count = int(totals.get("total_count") or 0)
    wrong_count = int(totals.get("wrong_count") or 0)
    avg_score = _format_metric(totals.get("avg_anomaly_score"))

    title_col, stats_col = st.columns([1.35, 1], vertical_alignment="center")
    with title_col:
        st.markdown('<h2 class="mega-section-title">Galeri</h2>', unsafe_allow_html=True)
        st.markdown(
            '<p class="mega-section-subtitle">Tüm analizler - işaretler görsel üzerinde</p>',
            unsafe_allow_html=True,
        )
    with stats_col:
        st.markdown(
            f"""<div class="mega-stat-row">
<div class="mega-stat"><strong>{total_count}</strong><span>Toplam</span></div>
<div class="mega-stat ok"><strong>{max(total_count - wrong_count, 0)}</strong><span>Sağlam</span></div>
<div class="mega-stat defect"><strong>{wrong_count}</strong><span>Risk</span></div>
<div class="mega-stat"><strong>{avg_score}</strong><span>Skor</span></div>
</div>""",
            unsafe_allow_html=True,
        )

    with st.expander("Filtreler ve veri işlemleri", expanded=False):
        filter_col, label_col, export_col = st.columns([1, 1, 1])
        with filter_col:
            selected_result = st.selectbox("Karar", ["Tum", "SAGLAM", "SUPHELI", "HATALI"], format_func=display_decision)
        with label_col:
            selected_label = st.selectbox("Etiket", ["Tum", *OPERATOR_LABELS], format_func=display_label)
        with export_col:
            records_for_csv = pd.DataFrame(
                store.fetch_recent_inspection_records(limit=24)
            )
            st.download_button(
                "Kayıt CSV indir",
                data=records_for_csv.to_csv(index=False).encode("utf-8"),
                file_name="quality_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
            if st.button("Veri seti dışa aktar", use_container_width=True):
                try:
                    export_path = export_dataset(config)
                    st.success(f"Veri seti hazır: {export_path}")
                except Exception as exc:
                    st.error(f"Export başarısız: {exc}")
            if st.button("Kayıtları yeniden tara", use_container_width=True):
                try:
                    updated_count = reprocess_gallery_records(config)
                    st.success(f"{updated_count} kayıt yeniden tarandı.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Yeniden tarama başarısız: {exc}")
        with st.expander("Toplu sil", expanded=False):
            confirm_clear = st.checkbox("Tüm denetim kayıtlarını sil")
            if st.button("Temizle", disabled=not confirm_clear, use_container_width=True):
                delete_all_gallery_records(config)
                st.rerun()
        with st.expander("Kalite özeti", expanded=False):
            chart_col, label_col = st.columns(2)
            with chart_col:
                st.write("Model karar dağılımı")
                render_count_distribution(
                    summary.get("model_result_counts", []),
                    label_key="label",
                    label_formatter=display_decision,
                )
            with label_col:
                st.write("Operatör etiket dağılımı")
                render_count_distribution(
                    summary.get("operator_label_counts", []),
                    label_key="label",
                    label_formatter=display_label,
                )
            render_threshold_suggestions(summary)

    model_filter = None if selected_result == "Tum" else selected_result
    label_filter = None if selected_label == "Tum" else selected_label
    recent_records = store.fetch_recent_inspection_records(
        limit=24,
        model_result=model_filter,
        operator_label=label_filter,
    )

    st.caption(
        f"{total_count} kayıt | {len(recent_records)} gösteriliyor | "
        f"ortalama skor {_format_metric(totals.get('avg_anomaly_score'))} | "
        f"model yanlış {wrong_count}"
    )

    pending_cards = [
        ("upload", "Yüklenen görüntü", st.session_state.get("upload_frame")),
        ("camera", "Kamera karesi", st.session_state.get("camera_frame")),
    ]
    visible_pending_cards = [(key, title, frame) for key, title, frame in pending_cards if frame is not None]
    cards_to_render = len(visible_pending_cards) + len(recent_records)
    columns = st.columns(3)

    if cards_to_render == 0:
        st.info("Henüz kayıt yok. Görüntü yükle veya kameradan kare al, sonra denetim kaydı oluştur.")
        return

    card_index = 0
    for source_name, title, frame in visible_pending_cards:
        with columns[card_index % len(columns)]:
            render_pending_gallery_item(source_name, title, frame, config)
        card_index += 1

    if not recent_records and not visible_pending_cards:
        st.info("Bu filtrede kayıt yok.")
    else:
        for index, record in enumerate(recent_records):
            with columns[(card_index + index) % len(columns)]:
                render_gallery_item(record, config)


def render_capture_workspace(config: AppConfig) -> None:
    with st.container(border=True):
        st.markdown(
            '<div class="mega-workspace-head"><span>Görsel Yükle</span><span>Canlı Kamera</span></div>',
            unsafe_allow_html=True,
        )
        mode = st.radio(
            "Denetim modu",
            ["📤 Görsel Yükle", "📷 Canlı Kamera"],
            horizontal=True,
            label_visibility="collapsed",
            key="mega_input_mode",
        )
        st.markdown(
            '<div class="mega-reference">Referans: 1 KABUL örneği · H36° / S23% / V69%</div>',
            unsafe_allow_html=True,
        )
        if "Görsel" in mode:
            render_upload_workflow(config)
        else:
            render_camera_workflow(config)


def render_count_distribution(
    rows: object,
    *,
    label_key: str,
    label_formatter,
) -> None:
    items = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        count = int(row.get("count") or 0)
        if count <= 0:
            continue
        label = label_formatter(row.get(label_key) or "-")
        items.append((label, count))

    if not items:
        st.caption("Henüz veri yok.")
        return

    max_count = max(count for _, count in items)
    html_rows = []
    for label, count in items:
        width = max(8, round((count / max_count) * 100))
        html_rows.append(
            f"""<div class="mega-dist-row">
<div class="mega-dist-label"><span>{label}</span><strong>{count}</strong></div>
<div class="mega-dist-track"><div style="width:{width}%"></div></div>
</div>"""
        )
    st.markdown("".join(html_rows), unsafe_allow_html=True)


def render_performance_panel(summary: dict) -> None:
    totals = summary.get("totals", {})
    total_count = int(totals.get("total_count") or 0)
    wrong_count = int(totals.get("wrong_count") or 0)
    wrong_rate = (wrong_count / total_count) if total_count else 0.0
    col_total, col_score, col_wrong, col_local = st.columns(4)
    col_total.metric("Toplam", total_count)
    col_score.metric("Ortalama skor", _format_metric(totals.get("avg_anomaly_score")))
    col_wrong.metric("Model yanlış", wrong_count, f"{wrong_rate:.1%}")
    col_local.metric("Yerel anomali", _format_metric(totals.get("avg_local_anomaly_score")))


def render_threshold_suggestions(summary: dict) -> None:
    totals = summary.get("totals", {})
    total_count = int(totals.get("total_count") or 0)
    wrong_count = int(totals.get("wrong_count") or 0)
    avg_crack = float(totals.get("avg_crack_score") or 0.0)
    avg_local = float(totals.get("avg_local_anomaly_score") or 0.0)
    avg_color = float(totals.get("avg_color_anomaly_score") or 0.0)

    suggestions: list[str] = []
    if total_count < 5:
        suggestions.append("Daha güvenilir ayar için en az 5-10 farklı örnek kaydı topla.")
    if wrong_count:
        suggestions.append("Model yanlış işaretlenen kayıtları Detay ekranında karşılaştır; eşik ayarı buradan netleşir.")
    if avg_crack >= 0.30:
        suggestions.append("Çatlak skoru yükseliyor; çatlak eşiğini düşürmek yerine önce doğru/yanlış örnekleri etiketle.")
    if avg_local >= 0.45:
        suggestions.append("Yerel anomali skoru yüksek; ışık/gölge sabit değilse kalibrasyon ve kamera ışığı kontrol edilmeli.")
    if avg_color >= 0.25:
        suggestions.append("Renk sapması artıyor; ürün renginden kalibrasyon yapmak gürültüyü azaltır.")
    if not suggestions:
        suggestions.append("Mevcut kayıtlar dengeli görünüyor; yeni hatalı örneklerle doğruluk kontrolüne devam et.")

    st.caption("Operasyon önerileri")
    for item in suggestions[:4]:
        st.caption(f"- {item}")


def reprocess_gallery_records(config: AppConfig) -> int:
    store = SQLiteStore(config.database_path)
    records = store.fetch_recent_inspection_records(limit=500)
    updated_count = 0
    for record in records:
        if reprocess_gallery_record(config, int(record["id"])):
            updated_count += 1
    return updated_count


def reprocess_gallery_record(config: AppConfig, record_id: int) -> bool:
    store = SQLiteStore(config.database_path)
    record = store.fetch_inspection_record(record_id)
    if record is None:
        return False

    raw_image = _read_optional_image(record.get("image_path"))
    if raw_image is None:
        return False

    analysis = process_frame(raw_image, config)
    if analysis.decision is None:
        return False

    previous_overlay_path = _archive_current_overlay(record, config)
    overlay_path = Path(str(record.get("overlay_path") or ""))
    if overlay_path:
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(overlay_path), analysis.overlay)

    return store.update_model_result(
        record_id,
        analysis.decision.label,
        analysis.decision.anomaly_score,
        0.0,
        float(analysis.rule_results.get("edge_damage", {}).get("score", 0.0)),
        float(analysis.rule_results.get("deformation", {}).get("score", 0.0)),
        float(analysis.rule_results.get("color_anomaly", {}).get("score", 0.0)),
        float(analysis.rule_results.get("glass_burn", {}).get("score", 0.0)),
        float(analysis.rule_results.get("raw_fiber", {}).get("score", 0.0)),
        float(analysis.rule_results.get("dark_crack", {}).get("score", 0.0)),
        float(analysis.rule_results.get("local_anomaly", {}).get("score", 0.0)),
        previous_overlay_path=previous_overlay_path,
        previous_model_result=str(record.get("model_result") or ""),
        previous_anomaly_score=float(record.get("anomaly_score") or 0.0),
        last_reprocessed_at=datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
    )


def _archive_current_overlay(record: dict, config: AppConfig) -> str | None:
    overlay_path = Path(str(record.get("overlay_path") or ""))
    if not overlay_path.exists():
        return None

    history_dir = config.output_overlay_path / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archive_path = history_dir / f"record_{record['id']}_{timestamp}_previous.jpg"
    copy2(overlay_path, archive_path)
    return str(archive_path)


def render_pending_gallery_item(
    source_name: str,
    title: str,
    frame: np.ndarray,
    config: AppConfig,
) -> None:
    analysis = process_frame(frame, config)
    st.image(gallery_image(analysis.overlay), use_container_width=True)

    if analysis.decision is None:
        st.caption(f"Yeni - {title} - ürün bulunamadı")
    else:
        st.caption(f"Yeni - {title} - {display_decision(analysis.decision.label)}")
        st.caption(f"skor {_format_metric(analysis.decision.anomaly_score)} | kaydedilmemiş")
        render_analysis_scan_line(analysis)

    if st.button("Bu renkle kalibre et", key=f"pending_calibrate_{source_name}", use_container_width=True):
        calibrate_from_frame(frame, config)
        st.rerun()

    if analysis.product is None or analysis.decision is None:
        return

    render_analysis_scan_summary(analysis)

    with st.expander("Kayıt oluştur", expanded=False):
        with st.form(f"pending_save_{source_name}", clear_on_submit=False):
            operator_label = st.selectbox(
                "Etiket",
                OPERATOR_LABELS,
                format_func=display_label,
                key=f"pending_{source_name}_label",
            )
            operator_note = st.text_input(
                "Not",
                placeholder="Opsiyonel not",
                key=f"pending_{source_name}_note",
            )
            submitted = st.form_submit_button("Denetim kaydı oluştur", type="primary", use_container_width=True)

        if submitted:
            try:
                manager = DatasetManager(config)
                record_id = manager.save_inspection(
                    raw_frame=resize_image(frame),
                    overlay_frame=analysis.overlay,
                    product_type="tas_yunu_panel",
                    decision=analysis.decision,
                    rule_results=analysis.rule_results,
                    operator_label=operator_label,
                    operator_note=operator_note,
                )
                st.session_state.pop(f"{source_name}_frame", None)
                st.success(f"Denetim kaydı oluşturuldu. ID: {record_id}")
                st.rerun()
            except Exception as exc:
                st.error(f"Kayıt sırasında hata oluştu: {exc}")


def render_gallery_item(record: dict, config: AppConfig) -> None:
    record_id = int(record["id"])
    overlay = _read_optional_image(record.get("overlay_path"))
    if overlay is not None:
        st.image(gallery_image(overlay), use_container_width=True)
    else:
        st.warning("Görsel bulunamadı.")

    st.caption(_record_label([record], record_id))
    st.caption(
        f"skor {_format_metric(record.get('anomaly_score'))} | "
        f"etiket {display_label(record.get('operator_label') or '-')}"
    )
    render_record_scan_line(record, config)
    with st.expander("İşlemler", expanded=False):
        raw_image = _read_optional_image(record.get("image_path"))
        overlay_image = _read_optional_image(record.get("overlay_path"))
        detail_tab, edit_tab = st.tabs(["Detay", "Düzenle"])
        with detail_tab:
            raw_col, overlay_col = st.columns(2)
            with raw_col:
                if raw_image is not None:
                    st.image(gallery_image(raw_image), caption="Ham görüntü", use_container_width=True)
            with overlay_col:
                if overlay_image is not None:
                    st.image(gallery_image(overlay_image), caption="Overlay", use_container_width=True)

            st.caption("Ayrı hata taramaları")
            render_record_scan_summary(record, config)
            score_cols = st.columns(4)
            score_cols[0].metric("Kenar", _format_metric(record.get("edge_damage_score")))
            score_cols[1].metric("Renk", _format_metric(record.get("color_anomaly_score")))
            score_cols[2].metric("Çatlak", _format_metric(record.get("crack_score")))
            score_cols[3].metric("Yerel", _format_metric(record.get("local_anomaly_score")))
            action_col, calibrate_col = st.columns(2)
            with action_col:
                if st.button("Tekrar tara", key=f"reprocess_record_{record_id}", use_container_width=True):
                    if reprocess_gallery_record(config, record_id):
                        st.success("Kayıt yeniden tarandı.")
                        st.rerun()
                    else:
                        st.warning("Kayıt yeniden taranamadı.")
            with calibrate_col:
                if raw_image is not None and st.button(
                    "Renk kalibre et",
                    key=f"calib_record_{record_id}",
                    use_container_width=True,
                ):
                    calibrate_from_frame(raw_image, config)
                    st.rerun()

            render_previous_scan(record)

        with edit_tab:
            render_gallery_edit_form(record, config)


def render_previous_scan(record: dict) -> None:
    previous_overlay = _read_optional_image(record.get("previous_overlay_path"))
    if previous_overlay is None:
        return

    previous_label = display_decision(str(record.get("previous_model_result") or "-"))
    previous_score = _format_metric(record.get("previous_anomaly_score"))
    with st.expander(f"Önceki sonucu göster → {previous_label} / skor {previous_score}", expanded=False):
        st.image(gallery_image(previous_overlay), caption="Önceki overlay", use_container_width=True)
        if record.get("last_reprocessed_at"):
            st.caption(f"Son yeniden tarama: {_format_timestamp(record.get('last_reprocessed_at'))}")


def render_gallery_edit_form(record: dict, config: AppConfig) -> None:
    record_id = int(record["id"])
    current_label = record.get("operator_label") or OPERATOR_LABELS[0]
    label_index = OPERATOR_LABELS.index(current_label) if current_label in OPERATOR_LABELS else 0
    new_label = st.selectbox(
        "Etiket",
        OPERATOR_LABELS,
        index=label_index,
        key=f"label_{record_id}",
        format_func=display_label,
    )
    new_note = st.text_input(
        "Not",
        value=str(record.get("operator_note") or ""),
        key=f"note_{record_id}",
    )
    is_model_wrong = st.checkbox(
        "Model yanlış",
        value=bool(record.get("is_model_wrong")),
        key=f"wrong_{record_id}",
    )
    save_col, delete_col = st.columns(2)
    with save_col:
        if st.button("Kaydet", key=f"update_gallery_{record_id}", use_container_width=True):
            update_gallery_record(config, record_id, new_label, new_note, is_model_wrong)
            st.rerun()
    with delete_col:
        if st.button("Sil", key=f"delete_gallery_{record_id}", use_container_width=True):
            delete_gallery_record(config, record_id)
            st.rerun()


def delete_gallery_record(config: AppConfig, record_id: int) -> None:
    store = SQLiteStore(config.database_path)
    deleted = store.delete_inspection_record(record_id)
    if deleted is None:
        st.warning("Kayıt zaten silinmiş.")
        return

    for path_key in ("image_path", "overlay_path", "previous_overlay_path"):
        path_value = deleted.get(path_key)
        if not path_value:
            continue
        path = Path(str(path_value))
        if path.exists():
            path.unlink()
    st.success(f"Denetim kaydı silindi. ID: {record_id}")


def delete_all_gallery_records(config: AppConfig) -> None:
    store = SQLiteStore(config.database_path)
    deleted_records = store.delete_all_inspection_records()
    for record in deleted_records:
        for path_key in ("image_path", "overlay_path", "previous_overlay_path"):
            path_value = record.get(path_key)
            if not path_value:
                continue
            path = Path(str(path_value))
            if path.exists():
                path.unlink()
    st.success(f"{len(deleted_records)} denetim kaydı silindi.")


def update_gallery_record(
    config: AppConfig,
    record_id: int,
    label: str,
    note: str,
    is_model_wrong: bool,
) -> None:
    store = SQLiteStore(config.database_path)
    if store.update_operator_feedback(record_id, label, note, is_model_wrong):
        st.success("Kayıt güncellendi.")
    else:
        st.warning("Kayıt bulunamadı.")


def _format_metric(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def _record_label(records: list[dict], record_id: int) -> str:
    record = next((item for item in records if int(item["id"]) == record_id), None)
    if record is None:
        return str(record_id)
    return f"#{record_id} - {_format_timestamp(record.get('timestamp'))} - {display_decision(record.get('model_result'))}"


def _format_timestamp(value: object) -> str:
    if not value:
        return "-"
    raw = str(value)
    try:
        parsed = datetime.strptime(raw, "%Y%m%d_%H%M%S_%f")
        return parsed.strftime("%d.%m.%Y %H:%M")
    except ValueError:
        return raw


def _read_optional_image(path_value: object) -> np.ndarray | None:
    if not path_value:
        return None

    path = Path(str(path_value))
    if not path.exists():
        return None
    return cv2.imread(str(path))


def render_runtime_tuning(config: AppConfig) -> AppConfig:
    lower_default = st.session_state.get("product_hsv_lower", config.product_hsv_lower)
    upper_default = st.session_state.get("product_hsv_upper", config.product_hsv_upper)
    tuned_config = replace(
        config,
        product_hsv_lower=lower_default,
        product_hsv_upper=upper_default,
    )
    st.session_state.product_hsv_lower = tuned_config.product_hsv_lower
    st.session_state.product_hsv_upper = tuned_config.product_hsv_upper
    return tuned_config


def render_footer() -> None:
    logo = mega_logo_svg("mega-logo footer-logo")
    st.markdown(
        f"""<footer class="mega-footer">
<div class="mega-footer-grid">
<div>
<div class="mega-footer-brand">
{logo}
<span>MEGA Insulation</span>
</div>
<p>Taş yünü üretiminde görsel tabanlı kalite kontrol çözümleri.</p>
</div>
<div>
<h4>Sistem</h4>
<ul>
<li>Plaka tespit → hata arama akışı</li>
<li>Çatlak · Cam yanığı · Çiğ elyaf · Deformasyon</li>
<li>Paylaşılan galeri & online kalibrasyon</li>
</ul>
</div>
<div id="iletisim">
<h4>İletişim</h4>
<ul>
<li>info@mega-insulation.com</li>
<li>+90 ___ ___ __ __</li>
</ul>
</div>
</div>
<div class="mega-footer-bottom">© {datetime.now().year} Mega Insulation Solutions</div>
</footer>""",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.markdown(
        """
        <style>
        :root {
            --mega-red: #E11D2A;
            --mega-red-dark: #B81522;
            --mega-navy: #1E2A6B;
            --mega-navy-dark: #141C4D;
            --mega-bg: #F7F8FB;
            --mega-surface: #FFFFFF;
            --mega-border: #E5E7EB;
            --mega-text: #0F172A;
            --mega-muted: #64748B;
        }
        html, body, [data-testid="stAppViewContainer"], .stApp {
            background:
                linear-gradient(135deg, #4b2a6f 0%, #17235f 51%, #3a1f55 100%) top left / 100% 505px no-repeat,
                var(--mega-bg) !important;
            color: var(--mega-text);
        }
        .block-container {
            padding-top: 0 !important;
            max-width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        header[data-testid="stHeader"],
        div[data-testid="stToolbar"],
        div[data-testid="stToolbarActions"],
        div[data-testid="stAppDeployButton"],
        div[data-testid="stDecoration"],
        div[data-testid="stStatusWidget"],
        span[data-testid="stMainMenu"],
        button[data-testid="stBaseButton-header"],
        button[data-testid="stMainMenuButton"],
        .stDeployButton {
            display: none !important;
        }
        .mega-hero {
            position: relative;
            overflow: hidden;
            margin: 0 -1rem 0 -1rem;
            min-height: 158px;
        }
        .mega-hero-bg {
            position: absolute;
            inset: 0;
            background: transparent;
        }
        .mega-hero-glow {
            position: absolute;
            inset: 0;
            opacity: 0;
        }
        .mega-hero-content {
            position: relative;
            max-width: 1104px;
            margin: 0 auto;
            padding: 30px 0 0 0;
            color: white;
        }
        .mega-hero-topline {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 2rem;
        }
        .mega-hero-heading {
            margin-bottom: 0;
        }
        .mega-pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            background: var(--mega-red);
            padding: 0.28rem 0.78rem;
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.01em;
        }
        .mega-hero h1 {
            margin: 0.7rem 0 0.25rem 0;
            font-size: 2.45rem;
            line-height: 1.08;
            color: white;
            font-weight: 800;
            letter-spacing: 0;
        }
        .mega-hero p {
            margin: 0;
            max-width: 700px;
            color: rgba(255,255,255,0.82);
            font-size: 0.98rem;
        }
        .mega-settings-link {
            color: #fff;
            font-size: 0.78rem;
            text-decoration: underline;
            margin-bottom: 0.15rem;
            font-weight: 500;
        }
        div[data-testid="stLayoutWrapper"]:has(.mega-workspace-head) {
            max-width: 1104px;
            margin: 0 auto 2.5rem auto;
            border: 1px solid var(--mega-border) !important;
            border-radius: 12px !important;
            background: #fff !important;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05) !important;
            padding: 0 !important;
            position: relative;
            z-index: 3;
        }
        div[data-testid="stLayoutWrapper"]:has(.mega-workspace-head) > div {
            padding: 1rem 1.25rem 1.25rem 1.25rem !important;
        }
        .mega-workspace-head {
            display: none;
        }
        .mega-gallery-start {
            display: none;
        }
        div[data-testid="stLayoutWrapper"]:has(.mega-gallery-start) {
            max-width: 1152px;
            margin: 0 auto;
            padding: 2.5rem 1.5rem 0 1.5rem;
        }
        div[role="radiogroup"] {
            display: flex;
            gap: 0.5rem;
            padding: 0;
            border-bottom: 0;
            flex-wrap: wrap;
        }
        div[role="radiogroup"] label {
            border: 1px solid var(--mega-border);
            border-radius: 8px;
            background: #fff;
            padding: 0.52rem 1.1rem;
            min-height: 42px;
            transition: all 0.15s ease;
        }
        div[role="radiogroup"] label:has(input:checked) {
            background: var(--mega-red);
            border-color: var(--mega-red);
            color: #fff;
        }
        div[role="radiogroup"] label p {
            font-size: 0.88rem;
            font-weight: 700;
        }
        .mega-reference {
            color: #64748b;
            font-size: 0.72rem;
            text-align: right;
            margin-top: -2.1rem;
            margin-bottom: 2.1rem;
        }
        .mega-drop-title {
            display: none;
        }
        .mega-drop-subtitle {
            color: var(--mega-text);
            text-align: center;
            color: #4d668a;
            font-size: 0.86rem;
            margin-top: 0;
            margin-bottom: -4.7rem;
            position: relative;
            z-index: 2;
        }
        .status-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 82px;
            padding: 0.32rem 0.55rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 750;
            letter-spacing: 0;
            border: 1px solid transparent;
        }
        .status-ok { color: #166534; background: #dcfce7; border-color: #bbf7d0; }
        .status-warning { color: #92400e; background: #fef3c7; border-color: #fde68a; }
        .status-defect { color: #fff; background: var(--mega-red); border-color: var(--mega-red-dark); }
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin-bottom: 0.75rem;
        }
        .kpi-card {
            border: 1px solid #dbe3ef;
            border-radius: 8px;
            background: #ffffff;
            padding: 0.8rem 0.9rem;
        }
        .kpi-card span {
            display: block;
            color: #64748b;
            font-size: 0.78rem;
            font-weight: 650;
            margin-bottom: 0.25rem;
        }
        .kpi-card strong {
            display: block;
            color: #0f172a;
            font-size: 1.35rem;
            line-height: 1.15;
        }
        .kpi-card small {
            color: #64748b;
            font-size: 0.78rem;
        }
        h1 { margin-bottom: 0.1rem; }
        h2, h3 { color: var(--mega-navy); }
        h3 { margin-top: 0.35rem; font-size: 1.15rem; }
        .mega-section-title {
            margin: 0;
            font-size: 1.55rem;
            line-height: 1.2;
            color: var(--mega-navy);
            font-weight: 800;
        }
        .mega-section-subtitle {
            color: var(--mega-muted);
            margin: 0.2rem 0 1rem 0;
            font-size: 0.88rem;
        }
        .mega-stat-row {
            display: flex;
            justify-content: flex-end;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-bottom: 1rem;
        }
        .mega-stat {
            min-width: 70px;
            text-align: center;
            background: #fff;
            border: 1px solid var(--mega-border);
            border-radius: 12px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            padding: 0.38rem 0.62rem;
        }
        .mega-stat strong {
            display: block;
            color: var(--mega-navy);
            font-size: 1.05rem;
            line-height: 1.05;
        }
        .mega-stat span {
            display: block;
            color: var(--mega-muted);
            font-size: 0.56rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 0.18rem;
        }
        .mega-stat.ok strong { color: #047857; }
        .mega-stat.defect strong { color: var(--mega-red); }
        .mega-dist-row {
            display: grid;
            gap: 0.35rem;
            margin: 0.55rem 0;
        }
        .mega-dist-label {
            display: flex;
            justify-content: space-between;
            gap: 0.8rem;
            color: var(--mega-muted);
            font-size: 0.82rem;
        }
        .mega-dist-label strong {
            color: var(--mega-navy);
        }
        .mega-dist-track {
            height: 8px;
            border-radius: 999px;
            overflow: hidden;
            background: #eef2f7;
        }
        .mega-dist-track div {
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, var(--mega-navy), var(--mega-red));
        }
        div[data-testid="stMetric"] {
            background: var(--mega-surface);
            border: 1px solid var(--mega-border);
            padding: 0.75rem;
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        div[data-testid="stMetric"] label {
            color: #64748b !important;
            font-size: 0.8rem !important;
        }
        div[data-testid="stVerticalBlock"] { gap: 0.65rem; }
        button {
            border-radius: 8px !important;
            font-weight: 650 !important;
        }
        div[data-testid="stButton"] button[kind="primary"],
        button[kind="primary"] {
            background: var(--mega-red) !important;
            border-color: var(--mega-red) !important;
            color: #fff !important;
        }
        div[data-testid="stButton"] button:hover {
            border-color: var(--mega-navy) !important;
        }
        div[data-testid="stAlert"] { border-radius: 8px; }
        div[data-testid="stFileUploader"] section {
            border-radius: 12px;
            border: 2px dashed #dde3ed;
            background: #fff;
            min-height: 196px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0;
            transition: all 0.15s ease;
        }
        div[data-testid="stFileUploader"] section:hover {
            border-color: var(--mega-red);
            background: #fff5f5;
        }
        div[data-testid="stImage"] img {
            border-radius: 0;
        }
        div[data-testid="stExpander"] {
            border: 1px solid var(--mega-border);
            border-radius: 10px;
            background: #fff;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        div[data-testid="stDataFrame"] {
            border-radius: 8px;
            overflow: hidden;
        }
        .mega-footer {
            margin: 5rem -1rem 0 -1rem;
            border-top: 1px solid var(--mega-border);
            background: #fff;
        }
        .mega-footer-grid {
            max-width: 1280px;
            margin: 0 auto;
            padding: 2.5rem 1.5rem;
            display: grid;
            grid-template-columns: 1.2fr 1fr 1fr;
            gap: 2rem;
            color: var(--mega-muted);
            font-size: 0.86rem;
        }
        .mega-footer-brand {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            color: var(--mega-navy);
            font-weight: 800;
            margin-bottom: 0.75rem;
        }
        .footer-logo { width: 34px; height: 34px; }
        .mega-footer h4 {
            color: var(--mega-text);
            font-size: 0.95rem;
            margin: 0 0 0.7rem 0;
        }
        .mega-footer ul {
            list-style: none;
            padding: 0;
            margin: 0;
            display: grid;
            gap: 0.45rem;
        }
        .mega-footer-bottom {
            border-top: 1px solid var(--mega-border);
            padding: 0.85rem;
            text-align: center;
            color: var(--mega-muted);
            font-size: 0.75rem;
        }
        @media (max-width: 760px) {
            .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .mega-topbar-inner { align-items: center; padding: 0 1rem; }
            .mega-topbar-status { justify-content: flex-start; }
            .mega-footer-grid { grid-template-columns: 1fr; }
            div[data-testid="stLayoutWrapper"]:has(.mega-workspace-head) { margin: -3rem 0 2rem 0; }
            div[data-testid="stLayoutWrapper"]:has(.mega-gallery-start) { padding: 2rem 0 0 0; }
            .mega-stat-row { justify-content: flex-start; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    try:
        config = load_config()
        ensure_runtime_directories(config)

        render_header(config)
        config = render_runtime_tuning(config)

        render_capture_workspace(config)
        with st.container():
            st.markdown('<div class="mega-gallery-start"></div>', unsafe_allow_html=True)
            render_gallery(config)

        render_footer()

    except Exception as exc:
        st.error(f"Uygulama başlatılırken hata oluştu: {exc}")


if __name__ == "__main__":
    main()
