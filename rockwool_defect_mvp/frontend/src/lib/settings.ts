"use client";
import { DEFAULT_THRESHOLDS, type Thresholds } from "./analysis";

const KEY = "mqc_thresholds_v1";
const ENGINE_KEY = "mqc_engine_v2";

export type Engine = "classical" | "claude" | "gemini" | "hybrid_claude" | "hybrid_gemini";

export function loadThresholds(): Thresholds {
  if (typeof window === "undefined") return DEFAULT_THRESHOLDS;
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return DEFAULT_THRESHOLDS;
    return { ...DEFAULT_THRESHOLDS, ...JSON.parse(raw) };
  } catch {
    return DEFAULT_THRESHOLDS;
  }
}

export function saveThresholds(t: Thresholds) {
  if (typeof window === "undefined") return;
  localStorage.setItem(KEY, JSON.stringify(t));
}

export function resetThresholds() {
  if (typeof window === "undefined") return;
  localStorage.removeItem(KEY);
}

export function loadEngine(): Engine {
  if (typeof window === "undefined") return "classical";
  // Migrate old toggle
  const legacy = localStorage.getItem("mqc_use_vlm_v1");
  if (legacy === "1") {
    localStorage.removeItem("mqc_use_vlm_v1");
    localStorage.setItem(ENGINE_KEY, "claude");
    return "claude";
  }
  const v = localStorage.getItem(ENGINE_KEY);
  if (v === "claude" || v === "gemini" || v === "classical" || v === "hybrid_claude" || v === "hybrid_gemini") return v;
  return "classical";
}

export function saveEngine(e: Engine) {
  if (typeof window === "undefined") return;
  localStorage.setItem(ENGINE_KEY, e);
}
