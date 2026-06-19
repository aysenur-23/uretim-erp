"use client";
import { useEffect, useRef, useState } from "react";

export function UploadZone({ onFiles, busy }: { onFiles: (files: FileList | null) => void; busy: boolean }) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [drag, setDrag] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const over = (event: DragEvent) => {
      event.preventDefault();
      setDrag(true);
    };
    const leave = (event: DragEvent) => {
      event.preventDefault();
      setDrag(false);
    };
    const drop = (event: DragEvent) => {
      event.preventDefault();
      setDrag(false);
      onFiles(event.dataTransfer?.files ?? null);
    };
    el.addEventListener("dragover", over);
    el.addEventListener("dragleave", leave);
    el.addEventListener("drop", drop);
    return () => {
      el.removeEventListener("dragover", over);
      el.removeEventListener("dragleave", leave);
      el.removeEventListener("drop", drop);
    };
  }, [onFiles]);

  return (
    <div ref={ref} className={`m-3 p-10 border-2 border-dashed rounded-xl text-center transition ${drag ? "border-[var(--mega-red)] bg-red-50" : "border-[var(--border)]"}`}>
      <p className="text-lg font-medium">Görsel yükleyin</p>
      <p className="text-sm text-[var(--text-muted)] mt-1">JPG, PNG ve WebP formatları desteklenir.</p>
      <input id="upl" type="file" accept="image/*" multiple onChange={(event) => onFiles(event.target.files)} className="hidden" />
      <label htmlFor="upl" className="btn btn-primary mt-4 cursor-pointer">Dosya Seç</label>
      {busy && <p className="text-sm text-[var(--text-muted)] mt-3">Analiz ediliyor...</p>}
    </div>
  );
}
