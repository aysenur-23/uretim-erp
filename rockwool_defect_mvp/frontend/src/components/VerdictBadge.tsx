export type Verdict = "KABUL" | "RED" | "UYARI";

export function VerdictBadge({ v, c }: { v: Verdict; c: number }) {
  const color = v === "KABUL" ? "bg-emerald-600" : v === "RED" ? "bg-[var(--mega-red)]" : "bg-amber-500";
  return <span className={`${color} text-white text-[11px] font-bold px-2 py-1 rounded`}>{v} · %{Math.round(c)}</span>;
}

export function DefectChips({ defects }: { defects: { label: string; score: number; severity: string }[] }) {
  if (defects.length === 0) return <p className="text-xs text-emerald-700">Tespit edilen hata yok.</p>;
  return (
    <ul className="space-y-1">
      {defects.map((d, i) => (
        <li key={`${d.label}-${i}-${d.score}`} className="flex items-center justify-between text-xs">
          <span>{d.label}</span>
          <span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${
            d.severity === "high" ? "bg-red-100 text-red-700"
              : d.severity === "medium" ? "bg-amber-100 text-amber-700"
              : "bg-slate-100 text-slate-600"
          }`}>%{d.score}</span>
        </li>
      ))}
    </ul>
  );
}
