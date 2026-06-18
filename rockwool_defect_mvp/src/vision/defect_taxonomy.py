from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DefectMeta:
    defect_type: str
    score_key: str
    label: str
    category: str
    overlay_color: str
    strategy: str
    description: str
    decision_impact: str


DEFECT_TAXONOMY: dict[str, DefectMeta] = {
    "edge_damage": DefectMeta(
        defect_type="edge_damage",
        score_key="edge_damage_score",
        label="Kenar",
        category="Geometri",
        overlay_color="#ffd400",
        strategy="Önce plaka çerçevesi bulunur; ROI kenarlarında eksik, kırık veya taşan bölgeler aranır.",
        description="Ürün sınırında kopma, ezilme, tırnaklanma veya kenar kaybı.",
        decision_impact="Yüksek kenar hasarı RED; düşük sinyal UYARI olarak ele alınır.",
    ),
    "deformation": DefectMeta(
        defect_type="deformation",
        score_key="deformation_score",
        label="Deformasyon",
        category="Geometri",
        overlay_color="#f59e0b",
        strategy="Dikdörtgen ürün beklentisiyle açı, oran ve kontur uyumu kontrol edilir.",
        description="Plakanın dikdörtgen formdan sapması, eğilmesi veya şekil bozulması.",
        decision_impact="Şekil bozulması diğer hatalarla birleşirse karar sertleştirilir.",
    ),
    "glass_burn": DefectMeta(
        defect_type="glass_burn",
        score_key="glass_burn_score",
        label="Cam yanığı",
        category="Renk / yuzey",
        overlay_color="#ff5a1f",
        strategy="ROI içinde geniş koyu kahverengi/siyah lekeler, lokal ışık tabanı çıkarıldıktan sonra aranır.",
        description="Dairesel veya geniş koyu renk farkı; çatlak gibi ince çizgi sayılmaz.",
        decision_impact="Belirgin cam yanığı tek başına RED sebebidir.",
    ),
    "raw_fiber": DefectMeta(
        defect_type="raw_fiber",
        score_key="raw_fiber_score",
        label="Çiğ elyaf",
        category="Malzeme",
        overlay_color="#38bdf8",
        strategy="Açık renkli, düşük doygunluklu ve lifsi görünen bölgeler renk/eşik maskesiyle ayrılır.",
        description="Bağlayıcı veya yüzey kaplaması oturmamış açık, lifsi alanlar.",
        decision_impact="Alan büyüdükçe UYARI'dan RED'e yükseltilir.",
    ),
    "color_anomaly": DefectMeta(
        defect_type="color_anomaly",
        score_key="color_anomaly_score",
        label="Renk/Leke",
        category="Renk / yuzey",
        overlay_color="#a855f7",
        strategy="Lab/HSV renk uzaklığıyla referans ürün renginden sapan kompakt lekeler aranır.",
        description="Cam yanığı kadar koyu olmayan ama ürün renginden ayrışan lekeler.",
        decision_impact="Tek başına orta risk; cam yanığı veya yerel anomaliyle birleşirse karar sertleşir.",
    ),
    "dark_crack": DefectMeta(
        defect_type="dark_crack",
        score_key="crack_score",
        label="Çatlak",
        category="Çizgisel hata",
        overlay_color="#dc2626",
        strategy="Sadece ince, uzun, çizgisel ve sınırdan bağımsız koyu hatlar çatlak kabul edilir.",
        description="Geniş/dairesel leke değil; uzun-ince koyu ayrışma veya yarılma.",
        decision_impact="Güçlü çatlak RED; zayıf/tekil sinyal UYARI.",
    ),
    "local_anomaly": DefectMeta(
        defect_type="local_anomaly",
        score_key="local_anomaly_score",
        label="Yerel anomali",
        category="Genel tarama",
        overlay_color="#ef4444",
        strategy="Doku ve parlaklık farkları için bölgesel anomali heatmap'i üretilir.",
        description="Kurala tam oturmayan ama normal dokudan ayrışan lokal bölgeler.",
        decision_impact="Destekleyici sinyaldir; diğer hata türleriyle birlikte karar güçlenir.",
    ),
}


DEFECT_ORDER = (
    "edge_damage",
    "deformation",
    "glass_burn",
    "raw_fiber",
    "color_anomaly",
    "dark_crack",
    "local_anomaly",
)


PIPELINE_STEPS = [
    {
        "key": "roi",
        "label": "1. Plaka çerçevesi",
        "description": "Önce ürünün dikdörtgen ROI/bbox alanı bulunur; hata aramaları sadece bu alanda çalışır.",
    },
    {
        "key": "illumination",
        "label": "2. Işık dengesi",
        "description": "Alt/üst bölgedeki gölge ve parlaklık farkları lokal ışık tabanıyla dengelenir.",
    },
    {
        "key": "surface",
        "label": "3. Yüzey hataları",
        "description": "Cam yanığı, renk/leke ve çiğ elyaf ayrı renk stratejileriyle aranır.",
    },
    {
        "key": "line",
        "label": "4. Çizgisel hata",
        "description": "Çatlak sadece uzun-ince form sağlıyorsa işaretlenir; geniş lekeler çatlak sayılmaz.",
    },
    {
        "key": "decision",
        "label": "5. Karar motoru",
        "description": "Her hata skoru birleşir; belirgin cam yanığı veya güçlü hata RED kararına gider.",
    },
]


def ordered_defects() -> list[DefectMeta]:
    return [DEFECT_TAXONOMY[key] for key in DEFECT_ORDER]


def defect_meta(defect_type: str) -> DefectMeta:
    return DEFECT_TAXONOMY[defect_type]
