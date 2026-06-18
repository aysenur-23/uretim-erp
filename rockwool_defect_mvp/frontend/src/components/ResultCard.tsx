"use client";
import { useState } from "react";
import { VerdictBadge, DefectChips, type Verdict } from "./VerdictBadge";

export type CardItem = {
  id: string;
  filename: string;
  source: "upload" | "camera";
  originalSrc: string;
  overlaySrc: string;
  verdict: Verdict | null;
  confidence: number;
  defects: { type: string; label: string; score: number; severity: string }[];
  meta?: string;
  pending?: boolean;
};

export function ResultCard({
  item,
  onOpen,
  onReprocess,
  onDelete,
  actionBusy,
}: {
  item: CardItem;
  onOpen?: (id: string) => void;
  onReprocess?: (id: string) => void;
  onDelete?: (id: string) => void;
  actionBusy?: boolean;
}) {
  const [showOverlay, setShowOverlay] = useState(true);
  const canAct = !item.pending && !actionBusy;
  return (
    <div className="card overflow-hidden group">
      <div
        className="relative aspect-[4/3] bg-slate-100 cursor-zoom-in"
        onClick={() => onOpen?.(item.id)}
        title="Detay görmek için tıkla"
      >
        <img
          src={showOverlay ? item.overlaySrc : item.originalSrc}
          alt={item.filename}
          className="w-full h-full object-contain"
          loading="lazy"
        />
        <button
          onClick={(e) => { e.stopPropagation(); setShowOverlay((s) => !s); }}
          className="absolute bottom-2 left-2 text-[11px] bg-white/90 px-2 py-1 rounded shadow"
        >
          {showOverlay ? "Orijinali Gör" : "Analizi Gör"}
        </button>
        <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition text-[10px] bg-black/50 text-white px-2 py-1 rounded">Detay →</div>
      </div>
      <div className="p-3 space-y-2">
        <div className="flex items-center justify-between gap-2">
          <div className="text-sm font-medium truncate" title={item.filename}>{item.filename}</div>
          {item.pending
            ? <span className="text-[11px] font-semibold px-2 py-1 rounded bg-amber-100 text-amber-700 animate-pulse">Analiz ediliyor…</span>
            : <VerdictBadge v={item.verdict!} c={item.confidence} />}
        </div>
        {!item.pending && <DefectChips defects={item.defects} />}
        {item.meta && <div className="text-[10px] text-[var(--text-muted)]">{item.meta}</div>}
        {!item.pending && (onReprocess || onDelete) && (
          <div className="grid grid-cols-2 gap-2 pt-1">
            {onReprocess && (
              <button
                type="button"
                disabled={!canAct}
                onClick={(e) => { e.stopPropagation(); onReprocess(item.id); }}
                className="btn btn-outline text-xs py-2 disabled:opacity-50"
              >
                {actionBusy ? "İşleniyor..." : "Tekrar tara"}
              </button>
            )}
            {onDelete && (
              <button
                type="button"
                disabled={!canAct}
                onClick={(e) => { e.stopPropagation(); onDelete(item.id); }}
                className="btn btn-outline text-xs py-2 text-[var(--mega-red)] disabled:opacity-50"
              >
                Sil
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
