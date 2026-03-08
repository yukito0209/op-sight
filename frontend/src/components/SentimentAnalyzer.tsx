import { useState } from "react";
import { analyzeText, batchAnalyze } from "../api/sentiment";
import type { BatchResult, Sentiment, SentimentResult } from "../types";

// ──────────────────────────────────────────────
// 常量
// ──────────────────────────────────────────────
const COLOR: Record<Sentiment, string> = {
  negative: "#ff4d4f",
  neutral: "#faad14",
  positive: "#52c41a",
};

const DEMO_TEXTS = [
  "太棒了，80抽又歪了😅 chfm！",
  "雷神yyds！伤害爆炸，这角色绝了！",
  "求问这期深渊12层带什么配队比较好？",
  "圣遗物坐牢三个月，一个能用的都没有(╯‵□′)╯︵┻━┻",
  "剧情太刀了，哭死我了😭",
  "可太会设计了呢，真是优秀的策划啊",
];

// ──────────────────────────────────────────────
// 子组件
// ──────────────────────────────────────────────
function IntensityBar({ intensity }: { intensity: number }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <span style={{ fontSize: 13, color: "#555" }}>强度</span>
      {Array(5)
        .fill(0)
        .map((_, i) => (
          <div
            key={i}
            style={{
              width: 8,
              height: 20,
              borderRadius: 2,
              backgroundColor: i < intensity ? "#ff4d4f" : "#f0f0f0",
            }}
          />
        ))}
      <span style={{ fontSize: 13, color: "#888" }}>{intensity}/5</span>
    </div>
  );
}

function ConfidenceBar({ confidence }: { confidence: number }) {
  const pct = Math.round(confidence * 100);
  const color = confidence > 0.8 ? "#52c41a" : confidence > 0.6 ? "#faad14" : "#ff4d4f";
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, color: "#555", marginBottom: 4 }}>
        <span>置信度</span>
        <span>{pct}%</span>
      </div>
      <div style={{ height: 8, background: "#f0f0f0", borderRadius: 4, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, transition: "width .3s" }} />
      </div>
    </div>
  );
}

function ResultCard({ result, text }: { result: SentimentResult; text: string }) {
  const borderColor = COLOR[result.sentiment];
  return (
    <div style={{ border: `2px solid ${borderColor}`, borderRadius: 8, padding: 16, background: "#fafafa", marginTop: 12 }}>
      {/* 原文 */}
      <div style={{ marginBottom: 10, fontSize: 14, color: "#333" }}>
        <strong>原文：</strong>{text}
      </div>

      {/* 标签行 */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12, flexWrap: "wrap" }}>
        <span style={{ padding: "3px 12px", borderRadius: 4, background: borderColor, color: "#fff", fontWeight: "bold", fontSize: 14 }}>
          {result.sentiment_zh}
        </span>
        {result.is_sarcasm && (
          <span style={{ padding: "3px 8px", borderRadius: 4, background: "#722ed1", color: "#fff", fontSize: 12 }}>
            反讽
          </span>
        )}
        <span style={{ fontSize: 12, color: "#888" }}>耗时 {result.processing_time_ms} ms</span>
      </div>

      <IntensityBar intensity={result.intensity} />
      <div style={{ marginTop: 10 }}>
        <ConfidenceBar confidence={result.confidence} />
      </div>

      {/* 归因 */}
      <div style={{ marginTop: 10, fontSize: 13 }}>
        <strong>归因：</strong>{result.category}
      </div>

      {/* 关键词 */}
      {result.keywords.length > 0 && (
        <div style={{ marginTop: 8, fontSize: 13 }}>
          <strong>关键词：</strong>
          {result.keywords.map((kw) => (
            <span key={kw} style={{ display: "inline-block", padding: "1px 8px", background: "#e6f7ff", borderRadius: 4, marginRight: 6, marginBottom: 4, fontSize: 12 }}>
              {kw}
            </span>
          ))}
        </div>
      )}

      {/* 建议 */}
      <div style={{ marginTop: 10, padding: 8, background: "#fff7e6", borderRadius: 4, fontSize: 13 }}>
        <strong>建议：</strong>{result.suggested_action}
      </div>

      {/* 反讽说明 */}
      {result.sarcasm_reason && (
        <div style={{ marginTop: 6, padding: 8, background: "#f9f0ff", borderRadius: 4, fontSize: 12, color: "#722ed1" }}>
          反讽原因：{result.sarcasm_reason}
        </div>
      )}

      {/* 概率分布 */}
      <div style={{ marginTop: 10, fontSize: 12, color: "#888", display: "flex", gap: 12 }}>
        {Object.entries(result.probabilities).map(([k, v]) => (
          <span key={k}>
            {k === "negative" ? "负面" : k === "neutral" ? "中性" : "正面"}：{(v * 100).toFixed(1)}%
          </span>
        ))}
      </div>
    </div>
  );
}

// ──────────────────────────────────────────────
// 主组件
// ──────────────────────────────────────────────
export default function SentimentAnalyzer() {
  const [tab, setTab] = useState<"single" | "batch">("single");
  const [text, setText] = useState("");
  const [result, setResult] = useState<SentimentResult | null>(null);
  const [batchInput, setBatchInput] = useState("");
  const [batchResult, setBatchResult] = useState<BatchResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      setResult(await analyzeText(text.trim()));
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const handleBatch = async () => {
    const texts = batchInput.split("\n").map((t) => t.trim()).filter(Boolean);
    if (!texts.length) return;
    setLoading(true);
    setError(null);
    try {
      setBatchResult(await batchAnalyze(texts));
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const btnStyle = (active: boolean) => ({
    padding: "8px 20px",
    border: "none",
    borderRadius: 4,
    cursor: "pointer",
    background: active ? "#1890ff" : "#f0f0f0",
    color: active ? "#fff" : "#333",
    fontWeight: active ? "bold" : "normal",
  } as React.CSSProperties);

  const primaryBtn = {
    padding: "10px 24px",
    background: loading ? "#aaa" : "#1890ff",
    color: "#fff",
    border: "none",
    borderRadius: 4,
    cursor: loading ? "not-allowed" : "pointer",
    fontSize: 15,
  } as React.CSSProperties;

  return (
    <div style={{ maxWidth: 800, margin: "0 auto", padding: "24px 16px", fontFamily: "system-ui, sans-serif" }}>
      <h1 style={{ marginBottom: 4 }}>玩家评论情绪分析工具</h1>
      <p style={{ color: "#888", marginBottom: 20, fontSize: 14 }}>
        针对二次元游戏社区（TapTap / NGA / B站）的精准情感识别
      </p>

      {/* Tab */}
      <div style={{ display: "flex", gap: 8, marginBottom: 20 }}>
        <button style={btnStyle(tab === "single")} onClick={() => setTab("single")}>单条分析</button>
        <button style={btnStyle(tab === "batch")} onClick={() => setTab("batch")}>批量分析</button>
      </div>

      {error && (
        <div style={{ padding: 12, background: "#fff2f0", border: "1px solid #ffccc7", borderRadius: 4, color: "#cf1322", marginBottom: 16 }}>
          {error}
        </div>
      )}

      {/* 单条 */}
      {tab === "single" && (
        <>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="输入玩家评论..."
            rows={4}
            style={{ width: "100%", padding: 12, borderRadius: 4, border: "1px solid #d9d9d9", fontSize: 14, resize: "vertical", boxSizing: "border-box" }}
          />
          {/* 快速示例 */}
          <div style={{ marginTop: 8, fontSize: 12, color: "#888" }}>
            快速示例：
            {DEMO_TEXTS.map((t) => (
              <button
                key={t}
                onClick={() => setText(t)}
                style={{ marginLeft: 8, marginBottom: 4, padding: "2px 8px", fontSize: 12, cursor: "pointer", background: "#f5f5f5", border: "1px solid #d9d9d9", borderRadius: 4 }}
              >
                {t.slice(0, 12)}…
              </button>
            ))}
          </div>
          <div style={{ marginTop: 12 }}>
            <button onClick={handleAnalyze} disabled={loading || !text.trim()} style={primaryBtn}>
              {loading ? "分析中..." : "分析情感"}
            </button>
          </div>
          {result && <ResultCard result={result} text={text} />}
        </>
      )}

      {/* 批量 */}
      {tab === "batch" && (
        <>
          <textarea
            value={batchInput}
            onChange={(e) => setBatchInput(e.target.value)}
            placeholder={"每行一条评论，例如：\n太棒了，80抽又歪了\n雷神yyds！\n求问圣遗物怎么搭？"}
            rows={8}
            style={{ width: "100%", padding: 12, borderRadius: 4, border: "1px solid #d9d9d9", fontSize: 14, resize: "vertical", boxSizing: "border-box" }}
          />
          <div style={{ marginTop: 12 }}>
            <button onClick={handleBatch} disabled={loading || !batchInput.trim()} style={primaryBtn}>
              {loading ? "批量分析中..." : "批量分析"}
            </button>
          </div>

          {batchResult && (
            <div style={{ marginTop: 20 }}>
              {/* 汇总 */}
              <div style={{ display: "flex", gap: 24, marginBottom: 16, padding: 16, background: "#f5f5f5", borderRadius: 8 }}>
                {(["negative", "neutral", "positive"] as const).map((s) => (
                  <div key={s} style={{ textAlign: "center" }}>
                    <div style={{ fontSize: 28, fontWeight: "bold", color: COLOR[s] }}>
                      {batchResult.sentiment_distribution[s]}
                    </div>
                    <div style={{ fontSize: 13, color: "#555" }}>
                      {s === "negative" ? "负面" : s === "neutral" ? "中性" : "正面"}
                    </div>
                  </div>
                ))}
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 28, fontWeight: "bold", color: "#1890ff" }}>
                    {(batchResult.avg_confidence * 100).toFixed(1)}%
                  </div>
                  <div style={{ fontSize: 13, color: "#555" }}>平均置信度</div>
                </div>
              </div>

              {batchResult.results.map((r, i) => (
                <ResultCard key={i} result={r} text={batchInput.split("\n").filter(Boolean)[i] ?? ""} />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
