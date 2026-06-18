"use client";
import { useEffect, useRef, useState } from "react";

export function UploadZone({ onFiles, busy }: { onFiles: (files: FileList | null) => void; busy: boolean }) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [drag, setDrag] = useState(false);
  useEffect(() => {
    const el = ref.current; if (!el) return;
    const over = (e: DragEvent) => { e.preventDefault(); setDrag(true); };
    const leave = (e: DragEvent) => { e.preventDefault(); setDrag(false); };
    const drop = (e: DragEvent) => { e.preventDefault(); setDrag(false); onFiles(e.dataTransfer?.files ?? null); };
    el.addEventListener("dragover", over);
    el.addEventListener("dragleave", leave);
    el.addEventListener("drop", drop);
    return () => { el.removeEventListener("dragover", over); el.removeEventListener("dragleave", leave); el.removeEventListener("drop", drop); };
  }, [onFiles]);
  return (
    <div ref={ref} className={`m-3 p-10 border-2 border-dashed rounded-xl text-center transition ${drag ? "border-[var(--mega-red)] bg-red-50" : "border-[var(--border)]"}`}>
      <p className="text-lg font-medium">Görseli buraya sürükle ya da seç</p>
      <p className="text-sm text-[var(--text-muted)] mt-1">JPG, PNG, WebP — çoklu seçim destekli</p>
      <input id="upl" type="file" accept="image/*" multiple onChange={(e) => onFiles(e.target.files)} className="hidden" />
      <label htmlFor="upl" className="btn btn-primary mt-4 cursor-pointer">Dosya Seç</label>
      {busy && <p className="text-sm text-[var(--text-muted)] mt-3">Analiz ediliyor…</p>}
    </div>
  );
}
