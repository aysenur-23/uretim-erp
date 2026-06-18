"use client";

export function SettingsModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50" onClick={onClose}>
      <div className="card w-full max-w-xl p-6" onClick={(event) => event.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-[var(--mega-navy)]">Analiz Ayarları</h2>
          <button onClick={onClose} className="text-[var(--text-muted)] hover:text-[var(--text)] text-xl leading-none" aria-label="Kapat">×</button>
        </div>

        <div className="space-y-4 text-sm text-[var(--text-muted)]">
          <p>
            Analiz motoru Python backend üzerinde çalışır. Ürün tespiti, bbox, çatlak, renk, kenar ve yerel anomali
            kuralları mevcut OpenCV hattından gelir.
          </p>
          <div className="rounded-lg border border-[var(--border)] bg-slate-50 p-4">
            <div className="font-semibold text-[var(--mega-navy)] mb-2">Aktif akış</div>
            <ul className="grid gap-1">
              <li>Görsel / kamera karesi FastAPI backend'e gönderilir.</li>
              <li>Backend `process_frame` ile ürünü ve hata adaylarını işler.</li>
              <li>Sonuç galeriye otomatik kaydedilir.</li>
            </ul>
          </div>
          <p className="text-xs">
            Eşikler ve kamera ayarları `config.yaml` üzerinden yönetilir.
          </p>
        </div>

        <div className="flex justify-end pt-5 mt-5 border-t border-[var(--border)]">
          <button onClick={onClose} className="btn btn-primary">Kapat</button>
        </div>
      </div>
    </div>
  );
}
