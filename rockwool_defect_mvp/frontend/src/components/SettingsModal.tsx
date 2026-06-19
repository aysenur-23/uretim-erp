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
  mismatches: {
    recordId: number;
    roiOk?: boolean;
    expectedVerdict: string;
    predictedVerdict: string;
    falsePositive: string[];
    falseNegative: string[];
    note: string;
  }[];
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
      <div className="card w-full max-w-3xl p-6 max-h-[92vh] overflow-y-auto" onClick={(event) => event.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-[var(--mega-navy)]">Kalibrasyon ve analiz ayarları</h2>
          <button onClick={onClose} className="text-[var(--text-muted)] hover:text-[var(--text)] text-xl leading-none" aria-label="Kapat">x</button>
        </div>

        <div className="grid gap-4 text-sm text-[var(--text-muted)]">
          <div className="rounded-lg border border-[var(--border)] bg-slate-50 p-4">
            <div className="font-semibold text-[var(--mega-navy)] mb-2">Aktif akış</div>
            <ul className="grid gap-1">
              <li>Görsel veya kamera karesi FastAPI backend'e gönderilir.</li>
              <li>Önce ROI/bbox confidence hesaplanır, sonra hata kuralları ayrı ayrı çalışır.</li>
              <li>Operatör geri bildirimi ROI doğrulaması ve FP/FN metriklerine dönüştürülür.</li>
              <li>VLM sadece açıklama ve rapor tarafında konumlandırılır; karar motorunun yerine geçmez.</li>
            </ul>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <MetricCard label="Feedback" value={metrics ? String(metrics.feedbackCount) : "-"} />
            <MetricCard label="Karar doğruluğu" value={metrics ? `%${(metrics.verdictAccuracy * 100).toFixed(1)}` : "-"} />
            <MetricCard label="ROI onayı" value={metrics ? `%${((metrics.roiFeedback?.accuracy ?? 1) * 100).toFixed(1)}` : "-"} />
            <MetricCard label="ROI red" value={metrics ? String(metrics.roiFeedback?.bad ?? 0) : "-"} />
          </div>

          <div className="rounded-lg border border-[var(--border)] bg-white p-4">
            <div className="font-semibold text-[var(--mega-navy)] mb-2">Genel bbox kalibrasyonu</div>
            <p className="text-xs leading-relaxed">
              Ürün çerçevesi her yeni görselde renk profili, kenar yoğunluğu ve dikdörtgenlik sinyaliyle hesaplanır.
              Operatörün "Ürün bbox/ROI doğru" doğrulaması bu alanın gerçek sahada ne kadar güvenilir olduğunu ölçer.
              ROI red sayısı arttığında eşik ve snap kuralları görüntüye özel değil, genel kalibrasyon olarak güncellenir.
            </p>
            <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
              <div className="rounded bg-slate-50 p-2">Toplam: <b>{metrics?.roiFeedback?.total ?? 0}</b></div>
              <div className="rounded bg-emerald-50 p-2 text-emerald-700">Doğru: <b>{metrics?.roiFeedback?.ok ?? 0}</b></div>
              <div className="rounded bg-red-50 p-2 text-red-700">Yanlış: <b>{metrics?.roiFeedback?.bad ?? 0}</b></div>
            </div>
          </div>

          <div className="rounded-lg border border-[var(--border)] overflow-hidden">
            <div className="bg-slate-50 px-4 py-2 font-semibold text-[var(--mega-navy)]">False positive / false negative</div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead className="text-left text-[var(--text-muted)]">
                  <tr>
                    <th className="px-4 py-2">Hata</th>
                    <th className="px-4 py-2">TP</th>
                    <th className="px-4 py-2">FP</th>
                    <th className="px-4 py-2">FN</th>
                    <th className="px-4 py-2">Precision</th>
                    <th className="px-4 py-2">Recall</th>
                  </tr>
                </thead>
                <tbody>
                  {(metrics?.perDefect ?? []).map((row) => (
                    <tr key={row.type} className="border-t border-[var(--border)]">
                      <td className="px-4 py-2 font-medium text-[var(--text)]">{DEFECT_LABELS[row.type] ?? row.type}</td>
                      <td className="px-4 py-2">{row.tp}</td>
                      <td className="px-4 py-2 text-red-700">{row.fp}</td>
                      <td className="px-4 py-2 text-amber-700">{row.fn}</td>
                      <td className="px-4 py-2">%{(row.precision * 100).toFixed(1)}</td>
                      <td className="px-4 py-2">%{(row.recall * 100).toFixed(1)}</td>
                    </tr>
                  ))}
                  {metrics && metrics.perDefect.length === 0 && (
                    <tr><td className="px-4 py-4 text-center" colSpan={6}>Henüz feedback yok.</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          <div className="rounded-lg border border-[var(--border)] p-4">
            <div className="font-semibold text-[var(--mega-navy)] mb-2">Sonraki fazlar</div>
            <ol className="grid gap-1 list-decimal list-inside">
              {(metrics?.nextPhases ?? [
                "Feedback biriktikçe çatlak, cam yanığı ve kenar deformasyon kuralları ayrı ayrı güçlendirilecek.",
                "Yeterli etiketli veri birikince segmentation modeline geçilecek.",
                "VLM sadece açıklama/rapor tarafında kullanılacak.",
              ]).map((item) => <li key={item}>{item}</li>)}
            </ol>
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

const DEFECT_LABELS: Record<string, string> = {
  edge_damage: "Kenar",
  deformation: "Deformasyon",
  glass_burn: "Cam yanığı",
  raw_fiber: "Cam/çiğ elyaf",
  color_anomaly: "Renk/Leke",
  dark_crack: "Çatlak",
  local_anomaly: "Yerel anomali",
};
