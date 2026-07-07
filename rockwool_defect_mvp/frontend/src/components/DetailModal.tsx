"use client";

import { useEffect, useState } from "react";
import { VerdictBadge, type Verdict } from "./VerdictBadge";

export type DefectDetail = {
  type: string;
  label: string;
  score: number;
  confidence?: number;
  severity: string;
  category?: string;
  overlayColor?: string;
  reason?: string;
  strategy?: string;
  description?: string;
  decisionImpact?: string;
};

export type PipelineStep = {
  key: string;
  label: string;
  description: string;
};

export type DetailItem = {
  id: string;
  filename: string;
  source: "upload" | "camera";
  originalSrc: string;
  overlaySrc: string;
  verdict: Verdict;
  confidence: number;
  roiConfidence?: number;
  defects: DefectDetail[];
  pipeline?: PipelineStep[];
  metrics?: {
    meanH: number;
    meanS: number;
    meanV: number;
    brightSpotRatio: number;
    darkSpotRatio: number;
    longLineScore: number;
    rectangularity: number;
    squarenessDeg: number;
  };
};

export function DetailModal({
  item,
  onClose,
  onReprocess,
  onDelete,
  onFeedback,
  actionBusy,
}: {
  item: DetailItem | null;
  onClose: () => void;
  onReprocess?: (id: string) => void;
  onDelete?: (id: string) => void;
  onFeedback?: (id: string, payload: { expectedVerdict: string; expectedDefects: string[]; roiOk: boolean; note: string }) => Promise<void>;
  actionBusy?: boolean;
}) {
  const [showOverlay, setShowOverlay] = useState(true);
  const [expectedVerdict, setExpectedVerdict] = useState<Verdict>("KABUL");
  const [expectedDefects, setExpectedDefects] = useState<string[]>([]);
  const [roiOk, setRoiOk] = useState(true);
  const [feedbackNote, setFeedbackNote] = useState("");

  useEffect(() => {
    if (!item) return;
    setShowOverlay(true);
    setExpectedVerdict(item.verdict);
    setExpectedDefects(item.defects.map((defect) => defect.type));
    setRoiOk(true);
    setFeedbackNote("");
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [item, onClose]);

  if (!item) return null;

  const severityClass = (severity: string) =>
    severity === "high"
      ? "bg-red-100 text-red-700"
      : severity === "medium"
        ? "bg-amber-100 text-amber-700"
        : "bg-slate-100 text-slate-600";

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur flex items-stretch" onClick={onClose}>
      <div className="m-auto w-full max-w-6xl bg-white rounded-xl overflow-hidden shadow-2xl flex flex-col md:flex-row max-h-[95vh]" onClick={(event) => event.stopPropagation()}>
        <div className="relative bg-slate-900 flex-1 min-h-[300px] md:min-h-[500px]">
          <img
            src={showOverlay ? item.overlaySrc : item.originalSrc}
            alt={item.filename}
            className="absolute inset-0 w-full h-full object-contain"
          />
          <button
            onClick={() => setShowOverlay((value) => !value)}
            className="absolute bottom-3 left-3 text-xs bg-white/95 px-3 py-1.5 rounded shadow hover:bg-white"
          >
            {showOverlay ? "Orijinali Gör" : "Analizi Gör"}
          </button>
          <div className="absolute top-3 right-3">
            <VerdictBadge v={item.verdict} c={item.confidence} />
          </div>
        </div>

        <div className="md:w-96 p-5 overflow-y-auto bg-white">
          <div className="flex items-start justify-between gap-2 mb-3">
            <div className="min-w-0">
              <div className="font-semibold truncate" title={item.filename}>{item.filename}</div>
              <div className="text-[11px] text-[var(--text-muted)]">{item.source === "camera" ? "Kamera" : "Yükleme"}</div>
            </div>
            <button onClick={onClose} className="text-[var(--text-muted)] hover:text-[var(--text)] text-xl leading-none" aria-label="Kapat">x</button>
          </div>

          <div className="mt-3">
            <h3 className="text-xs uppercase tracking-wider text-[var(--text-muted)] mb-2">Bulgular</h3>
            {item.defects.length === 0 ? (
              <p className="text-sm text-emerald-700">Belirgin hata tespit edilmedi.</p>
            ) : (
              <ul className="space-y-2">
                {item.defects.map((defect) => (
                  <li key={defect.type} className="rounded-lg border border-[var(--border)] p-3">
                    <div className="flex items-center justify-between gap-2">
                      <span className="flex items-center gap-2 text-sm font-semibold">
                        <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: defect.overlayColor ?? "#64748b" }} />
                        {defect.label}
                      </span>
                      <span className={`px-2 py-0.5 rounded text-[11px] font-semibold ${severityClass(defect.severity)}`}>%{defect.confidence ?? defect.score}</span>
                    </div>
                    {defect.category && <div className="mt-1 text-[11px] uppercase tracking-wider text-[var(--text-muted)]">{defect.category}</div>}
                  </li>
                ))}
              </ul>
            )}
          </div>

          {onFeedback && (
            <div className="mt-5 rounded-lg border border-[var(--border)] p-3">
              <h3 className="text-xs uppercase tracking-wider text-[var(--text-muted)] mb-2">Doğrulama</h3>
              <label className="text-xs text-[var(--text-muted)]">Beklenen karar</label>
              <select
                value={expectedVerdict}
                onChange={(event) => setExpectedVerdict(event.target.value as Verdict)}
                className="mt-1 w-full rounded-lg border border-[var(--border)] px-3 py-2 text-sm"
              >
                <option value="KABUL">KABUL</option>
                <option value="UYARI">UYARI</option>
                <option value="RED">RED</option>
              </select>
              <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
                <label className="col-span-2 flex items-center gap-2 rounded-lg bg-slate-50 px-2 py-2">
                  <input
                    type="checkbox"
                    checked={roiOk}
                    onChange={(event) => setRoiOk(event.target.checked)}
                  />
                  Ürün çerçevesi doğru
                </label>
                {DEFECT_OPTIONS.map((option) => (
                  <label key={option.type} className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={expectedDefects.includes(option.type)}
                      onChange={(event) => {
                        setExpectedDefects((items) => event.target.checked
                          ? Array.from(new Set([...items, option.type]))
                          : items.filter((value) => value !== option.type));
                      }}
                    />
                    {option.label}
                  </label>
                ))}
              </div>
              <textarea
                value={feedbackNote}
                onChange={(event) => setFeedbackNote(event.target.value)}
                placeholder="Kısa not"
                className="mt-3 w-full rounded-lg border border-[var(--border)] px-3 py-2 text-xs"
                rows={2}
              />
              <button
                disabled={actionBusy}
                onClick={() => onFeedback(item.id, { expectedVerdict, expectedDefects, roiOk, note: feedbackNote })}
                className="btn btn-primary mt-2 w-full py-2 text-xs disabled:opacity-50"
              >
                Doğrulamayı kaydet
              </button>
            </div>
          )}

          <div className="mt-5 pt-4 border-t border-[var(--border)] flex gap-2">
            <a href={item.overlaySrc} download={`${item.filename}_analiz.jpg`} className="btn btn-outline text-xs flex-1 py-2">Analizi indir</a>
            <a href={item.originalSrc} download={item.filename} className="btn btn-outline text-xs flex-1 py-2">Orijinali indir</a>
          </div>
          {(onReprocess || onDelete) && (
            <div className="mt-2 flex gap-2">
              {onReprocess && (
                <button disabled={actionBusy} onClick={() => onReprocess(item.id)} className="btn btn-outline text-xs flex-1 py-2 disabled:opacity-50">
                  {actionBusy ? "İşleniyor..." : "Tekrar tara"}
                </button>
              )}
              {onDelete && (
                <button disabled={actionBusy} onClick={() => onDelete(item.id)} className="btn btn-outline text-xs flex-1 py-2 text-[var(--mega-red)] disabled:opacity-50">
                  Sil
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const DEFECT_OPTIONS = [
  { type: "edge_damage", label: "Kenar" },
  { type: "deformation", label: "Deformasyon" },
  { type: "size_tolerance", label: "Boyut/Gönye" },
  { type: "glass_burn", label: "Cam yanığı" },
  { type: "raw_fiber", label: "Cam/çiğ elyaf" },
  { type: "color_anomaly", label: "Renk/Leke" },
  { type: "dark_crack", label: "Çatlak" },
  { type: "local_anomaly", label: "Yerel anomali" },
];
