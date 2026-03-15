import { useState } from "react";
import { analyzeText, batchAnalyze } from "../api/sentiment";
import type { BatchResult, Sentiment, SentimentResult } from "../types";

// ──────────────────────────────────────────────
// 设计 token（与 index.css 保持一致）
// ──────────────────────────────────────────────
const SENTIMENT_STYLE: Record<Sentiment, { bg: string; color: string; border: string; label: string }> = {
  negative: { bg: "#FDECEA", color: "#C0392B", border: "#F1A9A0", label: "负面" },
  neutral:  { bg: "#F0F0EE", color: "#666",    border: "#D5D3CF", label: "中性" },
  positive: { bg: "#EAF4ED", color: "#27AE60", border: "#A8D5B5", label: "正面" },
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
function IntensityDots({ intensity }: { intensity: number }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <span style={{ fontSize: 12, color: "var(--text-muted)", fontFamily: "var(--font-ui)" }}>强度</span>
      {Array(5).fill(0).map((_, i) => (
        <div
          key={i}
          style={{
            width: 10,
            height: 10,
            borderRadius: "50%",
            background: i < intensity ? "var(--accent)" : "var(--border)",
            transition: "background 0.2s",
          }}
        />
      ))}
      <span style={{ fontSize: 12, color: "var(--text-muted)" }}>{intensity}/5</span>
    </div>
  );
}

function ConfidenceBar({ confidence }: { confidence: number }) {
  const pct = Math.round(confidence * 100);
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "var(--text-muted)", marginBottom: 5 }}>
        <span>置信度</span>
        <span style={{ fontWeight: 500, color: "var(--text-secondary)" }}>{pct}%</span>
      </div>
      <div style={{ height: 4, background: "var(--border)", borderRadius: 2, overflow: "hidden" }}>
        <div
          style={{
            width: `${pct}%`,
            height: "100%",
            background: "linear-gradient(90deg, #DA7756, #E8956D)",
            borderRadius: 2,
            transition: "width 0.4s ease",
          }}
        />
      </div>
    </div>
  );
}

function ResultCard({ result, text }: { result: SentimentResult; text: string }) {
  const style = SENTIMENT_STYLE[result.sentiment];
  return (
    <div
      className="result-enter"
      style={{
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius)",
        padding: "20px 22px",
        marginTop: 16,
        boxShadow: "var(--card-shadow)",
      }}
    >
      {/* 原文 */}
      <div style={{ fontSize: 14, color: "var(--text-secondary)", marginBottom: 14, lineHeight: 1.7 }}>
        <span style={{ color: "var(--text-muted)", fontSize: 12, marginRight: 6 }}>原文</span>
        {text}
      </div>

      {/* 标签行 */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16, flexWrap: "wrap" }}>
        <span style={{
          padding: "3px 12px",
          borderRadius: 20,
          background: style.bg,
          color: style.color,
          border: `1px solid ${style.border}`,
          fontWeight: 500,
          fontSize: 13,
          fontFamily: "var(--font-ui)",
        }}>
          {style.label}
        </span>
        {result.is_sarcasm && (
          <span style={{
            padding: "3px 10px",
            borderRadius: 20,
            background: "#FEF3E2",
            color: "#E67E22",
            border: "1px solid #F5CBA7",
            fontSize: 12,
          }}>
            反讽
          </span>
        )}
        <span style={{ fontSize: 12, color: "var(--text-muted)", marginLeft: "auto" }}>
          {result.processing_time_ms} ms
        </span>
      </div>

      <IntensityDots intensity={result.intensity} />
      <div style={{ marginTop: 14 }}>
        <ConfidenceBar confidence={result.confidence} />
      </div>

      {/* 归因 */}
      <div style={{ marginTop: 14, fontSize: 13, color: "var(--text-secondary)" }}>
        <span style={{ color: "var(--text-muted)", fontSize: 12, marginRight: 6 }}>归因</span>
        {result.category}
      </div>

      {/* 关键词 */}
      {result.keywords.length > 0 && (
        <div style={{ marginTop: 10, display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center" }}>
          <span style={{ fontSize: 12, color: "var(--text-muted)" }}>关键词</span>
          {result.keywords.map((kw) => (
            <span key={kw} style={{
              padding: "2px 9px",
              background: "#F0EDE8",
              borderRadius: 20,
              fontSize: 12,
              color: "var(--text-secondary)",
            }}>
              {kw}
            </span>
          ))}
        </div>
      )}

      {/* 建议 */}
      <div style={{
        marginTop: 14,
        padding: "10px 14px",
        background: "#F9F6F0",
        borderLeft: "3px solid #D4A574",
        borderRadius: "0 8px 8px 0",
        fontSize: 13,
        fontFamily: "var(--font-display)",
        fontStyle: "italic",
        color: "var(--text-secondary)",
      }}>
        {result.suggested_action}
      </div>

      {/* 反讽说明 */}
      {result.sarcasm_reason && (
        <div style={{ marginTop: 8, fontSize: 12, color: "var(--text-muted)", fontStyle: "italic" }}>
          反讽原因：{result.sarcasm_reason}
        </div>
      )}

      {/* 概率分布 */}
      <div style={{ marginTop: 12, fontSize: 12, color: "var(--text-muted)", display: "flex", gap: 14 }}>
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

  const tabStyle = (active: boolean): React.CSSProperties => ({
    padding: "8px 4px",
    border: "none",
    borderBottom: active ? "2px solid var(--accent)" : "2px solid transparent",
    background: "transparent",
    cursor: "pointer",
    fontSize: 14,
    fontFamily: "var(--font-ui)",
    fontWeight: active ? 600 : 400,
    color: active ? "var(--text-primary)" : "var(--text-muted)",
    transition: "color 0.2s, border-color 0.2s",
    marginRight: 20,
  });

  const primaryBtnStyle: React.CSSProperties = {
    padding: "10px 26px",
    background: loading ? "#aaa" : "var(--btn-bg)",
    color: "#fff",
    border: "none",
    borderRadius: 8,
    cursor: loading ? "not-allowed" : "pointer",
    fontSize: 14,
    fontFamily: "var(--font-ui)",
    fontWeight: 500,
    transition: "background 0.2s, transform 0.1s",
    letterSpacing: "0.01em",
  };

  const textareaStyle: React.CSSProperties = {
    width: "100%",
    padding: "12px 14px",
    borderRadius: 10,
    border: "1.5px solid var(--border)",
    background: "var(--surface)",
    fontSize: 14,
    fontFamily: "var(--font-ui)",
    color: "var(--text-primary)",
    resize: "vertical",
    boxSizing: "border-box",
    boxShadow: "inset 0 1px 3px rgba(0,0,0,0.04)",
    outline: "none",
    transition: "border-color 0.2s",
    lineHeight: 1.7,
  };

  return (
    <div style={{ maxWidth: 720, margin: "0 auto", padding: "48px 24px 80px" }}>
      {/* 页头 */}
      <div style={{ marginBottom: 36 }}>
        <div style={{
          fontFamily: "var(--font-ui)",
          fontSize: 12,
          fontWeight: 600,
          letterSpacing: "0.12em",
          textTransform: "uppercase",
          color: "var(--accent)",
          marginBottom: 8,
        }}>
          op-sight
        </div>
        <h1 style={{
          fontFamily: "var(--font-display)",
          fontSize: "1.75rem",
          fontWeight: 600,
          color: "var(--text-primary)",
          lineHeight: 1.3,
          marginBottom: 8,
        }}>
          玩家评论情绪分析
        </h1>
        <p style={{ color: "var(--text-muted)", fontSize: 14 }}>
          针对二次元游戏社区（TapTap / NGA / B站）的精准情感识别
        </p>
      </div>

      {/* Tab */}
      <div style={{ display: "flex", borderBottom: "1px solid var(--border)", marginBottom: 24 }}>
        <button style={tabStyle(tab === "single")} onClick={() => setTab("single")}>单条分析</button>
        <button style={tabStyle(tab === "batch")} onClick={() => setTab("batch")}>批量分析</button>
      </div>

      {/* 错误提示 */}
      {error && (
        <div style={{
          padding: "10px 14px",
          background: "#FDECEA",
          border: "1px solid #F1A9A0",
          borderRadius: 8,
          color: "#C0392B",
          fontSize: 13,
          marginBottom: 16,
        }}>
          {error}
        </div>
      )}

      {/* 单条分析 */}
      {tab === "single" && (
        <>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            onFocus={(e) => { e.currentTarget.style.borderColor = "var(--border-focus)"; }}
            onBlur={(e) => { e.currentTarget.style.borderColor = "var(--border)"; }}
            placeholder="输入玩家评论..."
            rows={4}
            style={textareaStyle}
          />
          {/* 快速示例 */}
          <div style={{ marginTop: 10, display: "flex", flexWrap: "wrap", alignItems: "center", gap: 6 }}>
            <span style={{ fontSize: 12, color: "var(--text-muted)" }}>示例</span>
            {DEMO_TEXTS.map((t) => (
              <button
                key={t}
                onClick={() => setText(t)}
                style={{
                  padding: "3px 10px",
                  fontSize: 12,
                  cursor: "pointer",
                  background: "#EEEAE4",
                  border: "none",
                  borderRadius: 20,
                  color: "var(--text-secondary)",
                  fontFamily: "var(--font-ui)",
                  transition: "background 0.15s, transform 0.1s",
                }}
                onMouseEnter={(e) => { e.currentTarget.style.background = "#E0DBD4"; e.currentTarget.style.transform = "scale(1.02)"; }}
                onMouseLeave={(e) => { e.currentTarget.style.background = "#EEEAE4"; e.currentTarget.style.transform = "scale(1)"; }}
              >
                {t.slice(0, 12)}…
              </button>
            ))}
          </div>
          <div style={{ marginTop: 16 }}>
            <button
              onClick={handleAnalyze}
              disabled={loading || !text.trim()}
              style={primaryBtnStyle}
              onMouseEnter={(e) => { if (!loading) e.currentTarget.style.background = "var(--btn-hover)"; }}
              onMouseLeave={(e) => { if (!loading) e.currentTarget.style.background = "var(--btn-bg)"; }}
              onMouseDown={(e) => { e.currentTarget.style.transform = "scale(0.97)"; }}
              onMouseUp={(e) => { e.currentTarget.style.transform = "scale(1)"; }}
            >
              {loading ? "分析中…" : "分析情感"}
            </button>
          </div>
          {result && <ResultCard result={result} text={text} />}
        </>
      )}

      {/* 批量分析 */}
      {tab === "batch" && (
        <>
          <textarea
            value={batchInput}
            onChange={(e) => setBatchInput(e.target.value)}
            onFocus={(e) => { e.currentTarget.style.borderColor = "var(--border-focus)"; }}
            onBlur={(e) => { e.currentTarget.style.borderColor = "var(--border)"; }}
            placeholder={"每行一条评论，例如：\n太棒了，80抽又歪了\n雷神yyds！\n求问圣遗物怎么搭？"}
            rows={8}
            style={textareaStyle}
          />
          <div style={{ marginTop: 16 }}>
            <button
              onClick={handleBatch}
              disabled={loading || !batchInput.trim()}
              style={primaryBtnStyle}
              onMouseEnter={(e) => { if (!loading) e.currentTarget.style.background = "var(--btn-hover)"; }}
              onMouseLeave={(e) => { if (!loading) e.currentTarget.style.background = "var(--btn-bg)"; }}
              onMouseDown={(e) => { e.currentTarget.style.transform = "scale(0.97)"; }}
              onMouseUp={(e) => { e.currentTarget.style.transform = "scale(1)"; }}
            >
              {loading ? "批量分析中…" : "批量分析"}
            </button>
          </div>

          {batchResult && (
            <div style={{ marginTop: 24 }}>
              {/* 汇总卡片 */}
              <div style={{
                display: "flex",
                gap: 0,
                marginBottom: 20,
                background: "var(--surface)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius)",
                overflow: "hidden",
                boxShadow: "var(--card-shadow)",
              }}>
                {(["negative", "neutral", "positive"] as const).map((s, idx) => (
                  <div key={s} style={{
                    flex: 1,
                    textAlign: "center",
                    padding: "20px 12px",
                    borderRight: idx < 2 ? "1px solid var(--border)" : "none",
                  }}>
                    <div style={{
                      fontSize: 30,
                      fontWeight: 600,
                      fontFamily: "var(--font-display)",
                      color: SENTIMENT_STYLE[s].color,
                    }}>
                      {batchResult.sentiment_distribution[s]}
                    </div>
                    <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 4 }}>
                      {SENTIMENT_STYLE[s].label}
                    </div>
                  </div>
                ))}
                <div style={{ flex: 1, textAlign: "center", padding: "20px 12px" }}>
                  <div style={{ fontSize: 30, fontWeight: 600, fontFamily: "var(--font-display)", color: "var(--accent)" }}>
                    {(batchResult.avg_confidence * 100).toFixed(1)}%
                  </div>
                  <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 4 }}>平均置信度</div>
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
