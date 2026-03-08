import re
from typing import Any

import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer

from config.model_config import ID2LABEL, ID2LABEL_ZH
from data.slang_dict import (
    CATEGORY_KEYWORDS,
    SARCASM_EMOJIS,
    SARCASM_NEGATIVE_CONTEXTS,
    SARCASM_PATTERNS,
    SARCASM_POSITIVE_WORDS,
    STRONG_NEGATIVE_WORDS,
    STRONG_POSITIVE_WORDS,
)


class GameSentimentAnalyzer:
    """游戏评论情感分析器（推理专用）"""

    def __init__(self, model_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备：{self.device}")

        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    # ──────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────

    def analyze(self, original_text: str, processed_text: str | None = None) -> dict[str, Any]:
        """
        分析单条评论。

        Args:
            original_text:   原始文本（用于反讽检测，保留原始 Emoji）
            processed_text:  预处理后文本（用于模型推理）；None 则直接用原始文本
        """
        input_text = processed_text or original_text

        # 1. 模型推理
        sentiment, confidence, probs = self._model_inference(input_text)

        # 2. 反讽检测（必须用原始文本，保留 Emoji）
        is_sarcasm, sarcasm_reason = self._detect_sarcasm(original_text, sentiment, confidence)

        # 3. 反讽修正
        if is_sarcasm and sentiment == "positive":
            sentiment = "negative"
            confidence = min(confidence * 0.8, 0.9)

        # 4. 情绪强度（原始文本，含原始标点）
        intensity = self._calc_intensity(original_text, sentiment, confidence)

        # 5. 归因分类
        category, keywords = self._categorize(original_text)

        # 6. 建议动作
        suggested_action = self._suggest_action(sentiment, intensity, category, is_sarcasm)

        return {
            "sentiment": sentiment,
            "sentiment_zh": ID2LABEL_ZH[{"negative": 0, "neutral": 1, "positive": 2}[sentiment]],
            "confidence": round(confidence, 4),
            "intensity": intensity,
            "category": category,
            "keywords": keywords,
            "is_sarcasm": is_sarcasm,
            "sarcasm_reason": sarcasm_reason if is_sarcasm else None,
            "probabilities": {ID2LABEL[i]: round(p, 4) for i, p in enumerate(probs)},
            "suggested_action": suggested_action,
        }

    def batch_analyze(
        self,
        texts: list[str],
        processed_texts: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if processed_texts is None:
            processed_texts = [None] * len(texts)
        return [
            self.analyze(orig, proc)
            for orig, proc in zip(texts, processed_texts)
        ]

    # ──────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────

    def _model_inference(self, text: str) -> tuple[str, float, list[float]]:
        inputs = self.tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=1)[0]
            pred_id = torch.argmax(probs).item()

        return ID2LABEL[pred_id], probs[pred_id].item(), probs.cpu().tolist()

    def _detect_sarcasm(
        self, original_text: str, sentiment: str, confidence: float
    ) -> tuple[bool, str]:
        """反讽检测（在原始文本上运行，保留 Emoji）"""
        if sentiment != "positive":
            return False, ""

        text = original_text

        # 规则 1：正负面词汇共现
        has_positive = any(w in text for w in SARCASM_POSITIVE_WORDS)
        has_negative = any(w in text for w in SARCASM_NEGATIVE_CONTEXTS)
        if has_positive and has_negative and confidence > 0.7:
            return True, "正负面词汇冲突，疑似反讽"

        # 规则 2：反讽句式
        for pattern in SARCASM_PATTERNS:
            if re.search(pattern, text):
                return True, f"匹配反讽句式：{pattern}"

        # 规则 3：反讽 Emoji（原始文本保留 Emoji）
        if any(e in text for e in SARCASM_EMOJIS):
            return True, "包含反讽暗示 Emoji"

        return False, ""

    def _calc_intensity(self, text: str, sentiment: str, confidence: float) -> int:
        """情绪强度 1-5"""
        intensity = 1

        if confidence > 0.9:
            intensity += 1
        if confidence > 0.95:
            intensity += 1

        if sentiment == "negative":
            if any(w in text for w in STRONG_NEGATIVE_WORDS):
                intensity = min(5, intensity + 2)
            if re.search(r"[!！]{2,}|[?？]{2,}", text):
                intensity = min(5, intensity + 1)
        elif sentiment == "positive":
            if any(w in text for w in STRONG_POSITIVE_WORDS):
                intensity = min(5, intensity + 2)

        return intensity

    def _categorize(self, text: str) -> tuple[str, list[str]]:
        scores: dict[str, int] = {}
        matched: list[str] = []

        for category, keywords in CATEGORY_KEYWORDS.items():
            hits = [kw for kw in keywords if kw in text]
            if hits:
                scores[category] = len(hits)
                matched.extend(hits)

        best = max(scores, key=scores.get) if scores else "其他"
        return best, list(set(matched))

    def _suggest_action(
        self, sentiment: str, intensity: int, category: str, is_sarcasm: bool
    ) -> str:
        if sentiment == "negative":
            if intensity >= 4:
                prefix = "高危：疑似核心玩家反讽，建议人工复核" if is_sarcasm else "高危：强烈负面情绪，建议标记客服跟进"
                return prefix
            if intensity >= 3:
                return "中危：明显负面情绪，建议关注趋势"
            return "低危：轻微负面情绪，常规监控"

        if sentiment == "positive":
            if intensity >= 4:
                return "优质：高度正面反馈，可作为 UGC 素材"
            return "正面：正常正面反馈"

        # neutral
        if category in ("战斗系统", "养成系统"):
            return "中性：攻略/求助类内容，无需干预"
        return "中性：无明确情绪倾向"
