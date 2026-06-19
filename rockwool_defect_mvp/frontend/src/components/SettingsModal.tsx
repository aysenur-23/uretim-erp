"use client";

import { useEffect, useState } from "react";

type DefectMetric = {
  type: string;
  tp: number;
  fp: number;
  fn: number;
  precision: number;
  recall: number;
};

type CalibrationMetrics = {
  feedbackCount: number;
  verdictAccuracy: number;
  roiFeedback?: {
    total: number;
    ok: number;
    bad: number;
    accuracy: number;
    avgConfidence: number;
  };
  perDefect: DefectMetric[];
  mismatches: unknown[];
  nextPhases: string[];
};

export function SettingsModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [metrics, setMetrics] = useState<CalibrationMetrics | null>(null);

  useEffect(() => {
    if (!open) return;
    fetch("/api/calibration/metrics", { cache: "no-store" })
      .then((response) => response.ok ? response.json() : null)
      .then((data) => setMetrics(data))
      .catch(() => setMetrics(null));
  }, [open]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50" onClick={onClose}>
      <div className="card w-full max-w-2xl p-6 max-h-[92vh] overflow-y-auto" onClick={(event) => event.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-[var(--mega-navy)]">Ön çalışma notu</h2>
          <button onClick={onClose} className="text-[var(--text-muted)] hover:text-[var(--text)] text-xl leading-none" aria-label="Kapat">x</button>
        </div>

        <div className="grid gap-4 text-sm text-[var(--text-muted)]">
          <div className="rounded-lg border border-[var(--border)] bg-slate-50 p-4">
            <div className="font-semibold text-[var(--mega-navy)] mb-2">Amaç</div>
            <p className="leading-relaxed">
              Bu ekran, taş yünü üretiminde görüntü tabanlı kalite kontrol yaklaşımını göstermek için hazırlanmış bir ön çalışmadır.
              Nihai kabul kriterleri, saha görüntüleri ve operatör doğrulamalarıyla birlikte netleştirilmelidir.
            </p>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <MetricCard label="Analiz" value={metrics ? String(metrics.feedbackCount || "-") : "-"} />
            <MetricCard label="Karar uyumu" value={metrics?.feedbackCount ? `%${(metrics.verdictAccuracy * 100).toFixed(1)}` : "-"} />
            <MetricCard label="Çerçeve onayı" value={metrics?.roiFeedback?.total ? `%${((metrics.roiFeedback?.accuracy ?? 1) * 100).toFixed(1)}` : "-"} />
            <MetricCard label="Kontrol" value="Pilot" />
          </div>

          <div className="rounded-lg border border-[var(--border)] p-4">
            <div className="font-semibold text-[var(--mega-navy)] mb-2">Mevcut kapsam</div>
            <ul className="grid gap-1 list-disc list-inside">
              <li>Görsel yükleme ve canlı kamera üzerinden ön kontrol yapılabilir.</li>
              <li>Ürün çerçevesi belirlenir ve tespit edilen bulgular görsel üzerinde işaretlenir.</li>
              <li>Operatör doğrulaması alınarak kararların iyileştirilmesi için veri biriktirilir.</li>
            </ul>
          </div>

          <div className="rounded-lg border border-[var(--border)] p-4">
            <div className="font-semibold text-[var(--mega-navy)] mb-2">Sonraki adım</div>
            <p className="leading-relaxed">
              Farklı ışık, açı, ürün tipi ve hata örnekleriyle daha geniş bir veri seti toplanmalıdır.
              Bu doğrulama sonrasında karar eşikleri ve hata sınıfları üretim standardına göre kalibre edilebilir.
            </p>
          </div>
        </div>

        <div className="flex justify-end pt-5 mt-5 border-t border-[var(--border)]">
          <button onClick={onClose} className="btn btn-primary">Kapat</button>
        </div>
      </div>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-[var(--border)] bg-white p-4">
      <div className="text-[11px] uppercase tracking-wider text-[var(--text-muted)]">{label}</div>
      <div className="mt-1 text-2xl font-bold text-[var(--mega-navy)]">{value}</div>
    </div>
  );
}
