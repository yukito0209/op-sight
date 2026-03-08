import type { BatchResult, SentimentResult } from "../types";

const BASE = "/";   // 通过 vite proxy 转发到 :8000

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}

export const analyzeText = (text: string) =>
  post<SentimentResult>("analyze", { text });

export const batchAnalyze = (texts: string[]) =>
  post<BatchResult>("analyze/batch", { texts });
