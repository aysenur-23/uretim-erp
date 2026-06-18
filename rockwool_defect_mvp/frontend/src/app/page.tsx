import AppShell, { type Stats, type StoredAnalysis } from "./_AppShell";

export const dynamic = "force-dynamic";

const emptyStats: Stats = { total: 0, kabul: 0, red: 0, uyari: 0 };
const apiBase = process.env.MEGA_API_BASE ?? "http://127.0.0.1:8000";

export default async function Home() {
  const [items, stats, reference] = await Promise.all([loadItems(), loadStats(), loadReference()]);
  return <AppShell initialItems={items} initialStats={stats} initialReference={reference} />;
}

async function loadItems(): Promise<StoredAnalysis[]> {
  try {
    const response = await fetch(`${apiBase}/api/analyses?limit=120`, { cache: "no-store" });
    if (!response.ok) return [];
    const data = await response.json();
    return data.items ?? [];
  } catch {
    return [];
  }
}

async function loadStats(): Promise<Stats> {
  try {
    const response = await fetch(`${apiBase}/api/stats`, { cache: "no-store" });
    if (!response.ok) return emptyStats;
    return response.json();
  } catch {
    return emptyStats;
  }
}

async function loadReference() {
  try {
    const response = await fetch(`${apiBase}/api/reference`, { cache: "no-store" });
    if (!response.ok) return null;
    const data = await response.json();
    return data.reference ?? null;
  } catch {
    return null;
  }
}
