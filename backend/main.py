"""
玩家评论情绪分析 API

启动：
  uvicorn backend.main:app --reload --port 8000
"""

import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference.sentiment_analyzer import GameSentimentAnalyzer
from preprocessor.text_preprocessor import GameTextPreprocessor

# ──────────────────────────────────────────────
# 全局组件（在 lifespan 里初始化，避免 import 时崩溃）
# ──────────────────────────────────────────────
analyzer: GameSentimentAnalyzer | None = None
preprocessor: GameTextPreprocessor | None = None

MODEL_PATH = "./models/sentiment/best_model"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer, preprocessor
    print("正在加载模型...")
    preprocessor = GameTextPreprocessor()
    analyzer = GameSentimentAnalyzer(MODEL_PATH)
    print("模型加载完成")
    yield
    print("服务关闭")


app = FastAPI(
    title="玩家评论情绪分析 API",
    description="针对二次元游戏社区评论的情感分析服务（米哈游校招作品）",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# 数据模型
# ──────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: Annotated[str, Field(min_length=1, max_length=1000, description="评论内容")]


class BatchAnalyzeRequest(BaseModel):
    texts: Annotated[
        list[str],
        Field(description="评论列表（1-100 条）"),
    ]

    model_config = {"json_schema_extra": {"example": {"texts": ["这游戏太好玩了", "氪金骗局"]}}}

    def model_post_init(self, __context):
        if not 1 <= len(self.texts) <= 100:
            raise ValueError("texts 长度必须在 1-100 之间")


class AnalyzeResponse(BaseModel):
    sentiment: str
    sentiment_zh: str
    confidence: float
    intensity: int
    category: str
    keywords: list[str]
    is_sarcasm: bool
    sarcasm_reason: str | None
    probabilities: dict[str, float]
    suggested_action: str
    processing_time_ms: float


class BatchAnalyzeResponse(BaseModel):
    results: list[AnalyzeResponse]
    total_count: int
    avg_confidence: float
    sentiment_distribution: dict[str, int]


# ──────────────────────────────────────────────
# 端点
# ──────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "running",
        "model_loaded": analyzer is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """单条评论情感分析"""
    if analyzer is None:
        raise HTTPException(503, "模型未加载")

    t0 = time.perf_counter()
    processed, _ = preprocessor.preprocess(request.text)
    result = analyzer.analyze(request.text, processed)
    result["processing_time_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return AnalyzeResponse(**result)


@app.post("/analyze/batch", response_model=BatchAnalyzeResponse)
async def analyze_batch(request: BatchAnalyzeRequest):
    """批量评论情感分析（最多 100 条）"""
    if analyzer is None:
        raise HTTPException(503, "模型未加载")

    t0 = time.perf_counter()
    results = []
    for text in request.texts:
        processed, _ = preprocessor.preprocess(text)
        r = analyzer.analyze(text, processed)
        r["processing_time_ms"] = 0.0
        results.append(r)

    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    for r in results:
        r["processing_time_ms"] = elapsed / len(results)

    sentiments = [r["sentiment"] for r in results]
    return BatchAnalyzeResponse(
        results=[AnalyzeResponse(**r) for r in results],
        total_count=len(results),
        avg_confidence=round(sum(r["confidence"] for r in results) / len(results), 4),
        sentiment_distribution={
            "negative": sentiments.count("negative"),
            "neutral": sentiments.count("neutral"),
            "positive": sentiments.count("positive"),
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
