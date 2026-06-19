"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { UploadZone } from "@/components/UploadZone";
import { CameraPanel } from "@/components/CameraPanel";
import { ResultCard, type CardItem } from "@/components/ResultCard";
import { SettingsModal } from "@/components/SettingsModal";
import { DetailModal, type DefectDetail, type DetailItem, type PipelineStep } from "@/components/DetailModal";

export type StoredAnalysis = {
  id: string;
  created_at: number;
  filename: string;
  source: "upload" | "camera";
  originalSrc: string;
  overlaySrc: string;
  previousOverlaySrc?: string | null;
  verdict: "KABUL" | "RED" | "UYARI";
  confidence: number;
  roiConfidence?: number;
  defects: DefectDetail[];
  pipeline?: PipelineStep[];
  metrics?: DetailItem["metrics"];
  meta?: string;
};

export type Stats = { total: number; kabul: number; red: number; uyari: number };
type ReferenceProfile = null | { n: number; meanH: number; meanS: number; meanV: number };
type PendingItem = {
  id: string;
  filename: string;
  source: "upload" | "camera";
  originalDataUrl: string;
};
type Filter = "" | "KABUL" | "RED" | "UYARI";

export default function AppShell({
  initialItems,
  initialStats,
  initialReference,
}: {
  initialItems: StoredAnalysis[];
  initialStats: Stats;
  initialReference?: ReferenceProfile;
}) {
  const [mode, setMode] = useState<"upload" | "camera">("upload");
  const [pending, setPending] = useState<PendingItem[]>([]);
  const [stored, setStored] = useState<StoredAnalysis[]>(initialItems);
  const [stats, setStats] = useState<Stats>(initialStats);
  const [reference, setReference] = useState<ReferenceProfile>(initialReference ?? null);
  const [busy, setBusy] = useState(false);
  const [filter, setFilter] = useState<Filter>("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [detail, setDetail] = useState<DetailItem | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [actionId, setActionId] = useState<string | null>(null);

  const reload = useCallback(async () => {
    const [analysesRes, statsRes, referenceRes] = await Promise.all([
      fetch("/api/analyses?limit=120", { cache: "no-store" }),
      fetch("/api/stats", { cache: "no-store" }),
      fetch("/api/reference", { cache: "no-store" }),
    ]);
    if (analysesRes.ok) {
      const data = await analysesRes.json();
      setStored(data.items ?? []);
    }
    if (statsRes.ok) setStats(await statsRes.json());
    if (referenceRes.ok) {
      const data = await referenceRes.json();
      setReference(data.reference ?? null);
    }
  }, []);

  useEffect(() => {
    reload().catch(() => setError("Kayıtlar yüklenemedi."));
  }, [reload]);

  const refLabel = useMemo(() => {
    if (!reference) return "Ön çalışma";
    return `${reference.n} kabul örneği ile ön kalibrasyon`;
  }, [reference]);

  function openDetail(id: string) {
    const item = stored.find((x) => x.id === id);
    if (!item) return;
    setDetail(itemToDetail(item));
  }

  const runAnalysis = useCallback(async (file: File | Blob, filename: string, source: "upload" | "camera") => {
    setError(null);
    const pendingId = crypto.randomUUID();
    const originalDataUrl = await blobToDataUrl(file);
    setPending((items) => [{ id: pendingId, filename, source, originalDataUrl }, ...items]);

    const form = new FormData();
    form.append("file", file, filename);
    const response = await fetch(`/api/analyze?source=${source}`, { method: "POST", body: form });
    setPending((items) => items.filter((item) => item.id !== pendingId));
    if (!response.ok) {
      const message = await response.text();
      throw new Error(message || "Analiz başarısız oldu.");
    }
    await reload();
  }, [reload]);

  const handleFiles = useCallback(async (files: FileList | null) => {
    if (!files) return;
    setBusy(true);
    try {
      for (const file of Array.from(files)) {
        if (file.type.startsWith("image/")) await runAnalysis(file, file.name, "upload");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Görsel analiz edilemedi.");
    } finally {
      setBusy(false);
    }
  }, [runAnalysis]);

  const handleSnap = useCallback(async (blob: Blob, filename: string) => {
    setBusy(true);
    try {
      await runAnalysis(blob, filename, "camera");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Kamera görüntüsü analiz edilemedi.");
    } finally {
      setBusy(false);
    }
  }, [runAnalysis]);

  async function reprocessItem(id: string) {
    setError(null);
    setActionId(id);
    try {
      const response = await fetch(`/api/analyses/${id}/reprocess`, { method: "POST" });
      if (!response.ok) {
        setError("Kayıt yeniden taranamadı.");
        return;
      }
      const data = await response.json();
      await reload();
      if (data.item) setDetail(itemToDetail(data.item));
    } catch {
      setError("Kayıt yeniden taranamadı.");
    } finally {
      setActionId(null);
    }
  }

  async function deleteItem(id: string) {
    setError(null);
    setActionId(id);
    try {
      const response = await fetch(`/api/analyses/${id}`, { method: "DELETE" });
      if (!response.ok) {
        setError("Kayıt silinemedi.");
        return;
      }
      setDetail(null);
      await reload();
    } catch {
      setError("Kayıt silinemedi.");
    } finally {
      setActionId(null);
    }
  }

  async function saveFeedback(id: string, payload: { expectedVerdict: string; expectedDefects: string[]; roiOk: boolean; note: string }) {
    setError(null);
    setActionId(id);
    try {
      const response = await fetch(`/api/analyses/${id}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        setError("Geri bildirim kaydedilemedi.");
        return;
      }
      await reload();
    } catch {
      setError("Geri bildirim kaydedilemedi.");
    } finally {
      setActionId(null);
    }
  }

  const visibleStored = filter ? stored.filter((item) => item.verdict === filter) : stored;
  const pendingCards: CardItem[] = pending.map((item) => ({
    id: item.id,
    filename: item.filename,
    source: item.source,
    originalSrc: item.originalDataUrl,
    overlaySrc: item.originalDataUrl,
    verdict: null,
    confidence: 0,
    defects: [],
    pending: true,
    meta: "Analiz ediliyor...",
  }));
  const storedCards: CardItem[] = visibleStored.map((item) => ({
    id: item.id,
    filename: item.filename,
    source: item.source,
    originalSrc: item.originalSrc,
    overlaySrc: item.overlaySrc,
    verdict: item.verdict,
    confidence: item.confidence,
    roiConfidence: item.roiConfidence,
    defects: item.defects,
    meta: item.meta ?? (item.source === "camera" ? "Kamera" : "Yükleme"),
  }));

  return (
    <div>
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-[var(--mega-navy)] to-[var(--mega-navy-dark)]" />
        <div
          className="absolute inset-0 opacity-20"
          style={{ backgroundImage: "radial-gradient(circle at 20% 20%, #E11D2A 0, transparent 40%), radial-gradient(circle at 80% 80%, #E11D2A 0, transparent 40%)" }}
        />
        <div className="relative max-w-6xl mx-auto px-6 pt-10 pb-12 text-white">
          <div className="flex items-end justify-between flex-wrap gap-4 mb-6">
            <div>
              <div className="inline-block text-xs font-semibold tracking-wider bg-[var(--mega-red)] px-3 py-1 rounded-full">MEGA INSULATION SOLUTIONS</div>
              <h1 className="text-3xl md:text-4xl font-bold mt-3">Taş yünü kalite kontrol</h1>
              <p className="text-white/80 mt-1 text-sm md:text-base">Görüntü tabanlı kalite kontrol için hazırlanmış ön çalışma ekranı.</p>
            </div>
            <button onClick={() => setSettingsOpen(true)} className="text-xs text-white/80 hover:text-white underline">Ön çalışma notu</button>
          </div>

          <div className="card p-2">
            <div className="flex items-center justify-between px-3 py-2 border-b border-[var(--border)] flex-wrap gap-2">
              <div className="flex gap-2">
                <button onClick={() => setMode("upload")} className={`btn text-sm py-2 px-4 ${mode === "upload" ? "btn-primary" : "btn-outline"}`}>Görsel Yükle</button>
                <button onClick={() => setMode("camera")} className={`btn text-sm py-2 px-4 ${mode === "camera" ? "btn-primary" : "btn-outline"}`}>Canlı Kamera</button>
              </div>
              <div className="text-[11px] text-[var(--text-muted)]">{refLabel}</div>
            </div>
            {mode === "upload"
              ? <UploadZone onFiles={handleFiles} busy={busy} />
              : <CameraPanel onCapture={handleSnap} busy={busy} />}
          </div>
          {error && <div className="mt-3 rounded-lg bg-red-50 px-4 py-2 text-sm text-red-700">{error}</div>}
        </div>
      </section>

      <section className="max-w-6xl mx-auto px-6 py-10">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
          <div>
            <h2 className="text-2xl font-bold text-[var(--mega-navy)]">Galeri</h2>
            <p className="text-sm text-[var(--text-muted)]">Pilot analiz sonuçları</p>
          </div>
          <div className="flex gap-2">
            <Stat label="Toplam" value={stats.total} color="text-[var(--mega-navy)]" />
            <Stat label="KABUL" value={stats.kabul} color="text-emerald-700" />
            <Stat label="RED" value={stats.red} color="text-[var(--mega-red)]" />
            <Stat label="UYARI" value={stats.uyari} color="text-amber-600" />
          </div>
        </div>
        <div className="flex gap-2 mb-5 flex-wrap">
          {(["", "KABUL", "RED", "UYARI"] as const).map((value) => (
            <button
              key={value}
              onClick={() => setFilter(value)}
              className={`px-4 py-1.5 rounded-full text-sm border transition ${filter === value ? "bg-[var(--mega-navy)] text-white border-transparent" : "bg-white border-[var(--border)] hover:border-[var(--mega-navy)]"}`}
            >
              {value || "Tümü"}
            </button>
          ))}
        </div>
        {pendingCards.length + storedCards.length === 0 ? (
          <div className="card p-10 text-center text-[var(--text-muted)]">Henüz analiz yok.</div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {pendingCards.map((card) => <ResultCard key={card.id} item={card} />)}
            {storedCards.map((card) => (
              <ResultCard
                key={card.id}
                item={card}
                onOpen={openDetail}
                onReprocess={reprocessItem}
                onDelete={deleteItem}
                actionBusy={actionId === card.id}
              />
            ))}
          </div>
        )}
      </section>

      <SettingsModal open={settingsOpen} onClose={() => setSettingsOpen(false)} />
      <DetailModal
        item={detail}
        onClose={() => setDetail(null)}
        onReprocess={reprocessItem}
        onDelete={deleteItem}
        onFeedback={saveFeedback}
        actionBusy={!!detail && actionId === detail.id}
      />
    </div>
  );
}

function itemToDetail(item: StoredAnalysis): DetailItem {
  return {
    id: item.id,
    filename: item.filename,
    source: item.source,
    originalSrc: item.originalSrc,
    overlaySrc: item.overlaySrc,
    verdict: item.verdict,
    confidence: item.confidence,
    roiConfidence: item.roiConfidence,
    defects: item.defects,
    pipeline: item.pipeline,
    metrics: item.metrics,
  };
}

function Stat({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="card px-3 py-1.5 text-center min-w-[70px]">
      <div className={`text-lg font-bold ${color}`}>{value}</div>
      <div className="text-[9px] text-[var(--text-muted)] uppercase tracking-wider">{label}</div>
    </div>
  );
}

function blobToDataUrl(blob: Blob) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}
