"use client";
import { useEffect, useState } from "react";
import { VerdictBadge, type Verdict } from "./VerdictBadge";

export type DefectDetail = {
  type: string;
  label: string;
  score: number;
  severity: string;
  category?: string;
  overlayColor?: string;
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
  defects: DefectDetail[];
  pipeline?: PipelineStep[];
  createdAt?: number;
  metrics?: {
    meanH: number; meanS: number; meanV: number;
    brightSpotRatio: number; darkSpotRatio: number;
    longLineScore: number;
    rectangularity: number; squarenessDeg: number;
  };
};

export function DetailModal({
  item,
  onClose,
  onReprocess,
  onDelete,
}: {
  item: DetailItem | null;
  onClose: () => void;
  onReprocess?: (id: string) => void;
  onDelete?: (id: string) => void;
}) {
  const [showOverlay, setShowOverlay] = useState(true);
  useEffect(() => {
    if (!item) return;
    setShowOverlay(true);
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => { window.removeEventListener("keydown", onKey); document.body.style.overflow = ""; };
  }, [item, onClose]);
  if (!item) return null;

  const m = item.metrics;
  const sev = (s: string) =>
    s === "high" ? "bg-red-100 text-red-700"
    : s === "medium" ? "bg-amber-100 text-amber-700"
    : "bg-slate-100 text-slate-600";

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur flex items-stretch" onClick={onClose}>
      <div className="m-auto w-full max-w-6xl bg-white rounded-xl overflow-hidden shadow-2xl flex flex-col md:flex-row max-h-[95vh]" onClick={(e) => e.stopPropagation()}>
        {/* Image */}
        <div className="relative bg-slate-900 flex-1 min-h-[300px] md:min-h-[500px]">
          <img
            src={showOverlay ? item.overlaySrc : item.originalSrc}
            alt={item.filename}
            className="absolute inset-0 w-full h-full object-contain"
          />
          <button
            onClick={() => setShowOverlay((s) => !s)}
            className="absolute bottom-3 left-3 text-xs bg-white/95 px-3 py-1.5 rounded shadow hover:bg-white"
          >
            {showOverlay ? "Orijinali Gör" : "Analizi Gör"}
          </button>
          <div className="absolute top-3 right-3">
            <VerdictBadge v={item.verdict} c={item.confidence} />
          </div>
        </div>

        {/* Sidebar */}
        <div className="md:w-80 p-5 overflow-y-auto bg-white">
          <div className="flex items-start justify-between gap-2 mb-3">
            <div className="min-w-0">
              <div className="font-semibold truncate" title={item.filename}>{item.filename}</div>
              {item.createdAt && (
                <div className="text-[11px] text-[var(--text-muted)]">
                  {new Date(item.createdAt).toLocaleString("tr-TR")} · {item.source === "camera" ? "Kamera" : "Yükleme"}
                </div>
              )}
            </div>
            <button onClick={onClose} className="text-[var(--text-muted)] hover:text-[var(--text)] text-xl leading-none" aria-label="Kapat">✕</button>
          </div>

          <div className="mt-3">
            <h3 className="text-xs uppercase tracking-wider text-[var(--text-muted)] mb-2">Tespitler</h3>
            {item.defects.length === 0 ? (
              <p className="text-sm text-emerald-700">Tespit edilen hata yok.</p>
            ) : (
              <ul className="space-y-2">
                {item.defects.map((d) => (
                  <li key={d.type} className="rounded-lg border border-[var(--border)] p-3">
                    <div className="flex items-center justify-between gap-2">
                      <span className="flex items-center gap-2 text-sm font-semibold">
                        <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: d.overlayColor ?? "#64748b" }} />
                        {d.label}
                      </span>
                      <span className={`px-2 py-0.5 rounded text-[11px] font-semibold ${sev(d.severity)}`}>%{d.score}</span>
                    </div>
                    {d.category && <div className="mt-1 text-[11px] uppercase tracking-wider text-[var(--text-muted)]">{d.category}</div>}
                    {d.description && <p className="mt-2 text-xs text-[var(--text-muted)]">{d.description}</p>}
                    {d.strategy && (
                      <p className="mt-2 text-xs text-[var(--text)]">
                        <span className="font-semibold">Strateji:</span> {d.strategy}
                      </p>
                    )}
                    {d.decisionImpact && <p className="mt-1 text-[11px] text-[var(--text-muted)]">{d.decisionImpact}</p>}
                  </li>
                ))}
              </ul>
            )}
          </div>

          {item.pipeline && item.pipeline.length > 0 && (
            <div className="mt-5">
              <h3 className="text-xs uppercase tracking-wider text-[var(--text-muted)] mb-2">Analiz hattı</h3>
              <ol className="space-y-2">
                {item.pipeline.map((step) => (
                  <li key={step.key} className="rounded-lg bg-slate-50 px-3 py-2">
                    <div className="text-xs font-semibold text-[var(--text)]">{step.label}</div>
                    <div className="mt-1 text-[11px] text-[var(--text-muted)]">{step.description}</div>
                  </li>
                ))}
              </ol>
            </div>
          )}

          {m && (
            <div className="mt-5">
              <h3 className="text-xs uppercase tracking-wider text-[var(--text-muted)] mb-2">Ham metrikler</h3>
              <dl className="text-xs grid grid-cols-2 gap-x-3 gap-y-1.5 text-[var(--text-muted)]">
                <dt>Plaka medyan H</dt><dd className="text-right text-[var(--text)] tabular-nums">{m.meanH.toFixed(1)}°</dd>
                <dt>Plaka medyan S</dt><dd className="text-right text-[var(--text)] tabular-nums">{m.meanS.toFixed(1)}%</dd>
                <dt>Plaka medyan V</dt><dd className="text-right text-[var(--text)] tabular-nums">{m.meanV.toFixed(1)}%</dd>
                <dt>Açık leke oranı</dt><dd className="text-right text-[var(--text)] tabular-nums">{(m.brightSpotRatio * 100).toFixed(2)}%</dd>
                <dt>Koyu leke oranı</dt><dd className="text-right text-[var(--text)] tabular-nums">{(m.darkSpotRatio * 100).toFixed(2)}%</dd>
                <dt>Çatlak uzunluk</dt><dd className="text-right text-[var(--text)] tabular-nums">{(m.longLineScore * 100).toFixed(1)}%</dd>
                <dt>Dikdörtgensellik</dt><dd className="text-right text-[var(--text)] tabular-nums">{(m.rectangularity * 100).toFixed(1)}%</dd>
                <dt>Gönye sapma</dt><dd className="text-right text-[var(--text)] tabular-nums">{m.squarenessDeg.toFixed(2)}°</dd>
              </dl>
            </div>
          )}

          <div className="mt-5 pt-4 border-t border-[var(--border)] flex gap-2">
            <a href={item.overlaySrc} download={`${item.filename}_analiz.jpg`} className="btn btn-outline text-xs flex-1 py-2">Analizi İndir</a>
            <a href={item.originalSrc} download={item.filename} className="btn btn-outline text-xs flex-1 py-2">Orijinali İndir</a>
          </div>
          {(onReprocess || onDelete) && (
            <div className="mt-2 flex gap-2">
              {onReprocess && (
                <button onClick={() => onReprocess(item.id)} className="btn btn-outline text-xs flex-1 py-2">
                  Tekrar tara
                </button>
              )}
              {onDelete && (
                <button onClick={() => onDelete(item.id)} className="btn btn-outline text-xs flex-1 py-2 text-[var(--mega-red)]">
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
