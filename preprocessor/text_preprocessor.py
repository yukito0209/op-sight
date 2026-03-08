import re
from typing import Tuple

import emoji

from data.slang_dict import SLANG_DICT


class GameTextPreprocessor:
    """游戏评论文本预处理器"""

    # Emoji → 语义标签（处理高频情感 Emoji）
    EMOJI_MAP = {
        "❤️": "[爱心-正面]", "💕": "[爱心-正面]", "😍": "[喜欢-正面]",
        "🥰": "[喜欢-正面]", "😘": "[喜欢-正面]", "👍": "[赞-正面]",
        "🎉": "[庆祝-正面]", "✨": "[闪耀-正面]", "🔥": "[火热-正面]",
        "💯": "[满分-正面]", "😭": "[大哭-负面]", "😢": "[哭泣-负面]",
        "😤": "[生气-负面]", "😡": "[愤怒-负面]", "🤬": "[暴怒-负面]",
        "💔": "[心碎-负面]", "😅": "[汗颜-反讽]", "😂": "[笑哭-复杂]",
        "🤣": "[大笑-复杂]", "🙄": "[白眼-负面]", "😒": "[不屑-负面]",
        "😑": "[无语-负面]", "😐": "[无表情-中性]", "🤔": "[思考-中性]",
        "👎": "[踩-负面]", "💀": "[死亡-负面]", "☠️": "[死亡-负面]",
        "💥": "[爆炸-正面]",
    }

    # 正面颜文字
    POSITIVE_KAOMOJIS = [
        "(๑•̀ㅂ•́)و✧", "(｡♥‿♥｡)", "(◕‿◕✿)", "(ﾉ◕ヮ◕)ﾉ*:･ﾟ✧",
        "(✿ ♥‿♥)", "(◍•ᴗ•◍)❤", "(✧ω✧)", "(◠‿◠✿)",
    ]

    # 负面颜文字
    NEGATIVE_KAOMOJIS = [
        "(╯‵□′)╯︵┻━┻", "(┳Д┳)", "(╥﹏╥)", "(；′⌒`)",
        "(╯°□°）╯︵ ┻━┻", "(；一_一)", "(¬_¬)", "(；￣Д￣）",
        "(눈_눈)", "(¬▂¬)", "(╬ Ò﹏Ó)",
    ]

    def __init__(self):
        # 按 key 长度降序排列，防止短词提前替换长词
        self._sorted_slangs = sorted(SLANG_DICT.keys(), key=len, reverse=True)

    def preprocess(self, text: str) -> Tuple[str, dict]:
        """
        预处理流水线。

        Returns:
            processed_text: 处理后文本
            metadata: 预处理元数据（供反讽检测和日志使用）
        """
        metadata: dict = {"original": text}

        # 注意：反讽检测需要原始文本（含原始 Emoji），
        # 所以 metadata 里保留原始文本，供 analyzer 在推理后使用。

        text = self._convert_emoji(text, metadata)
        text = self._convert_kaomoji(text, metadata)
        text = self._normalize_slang(text, metadata)
        text = self._compress_punctuation(text)
        text = self._clean_special_chars(text)

        metadata["processed"] = text
        return text, metadata

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _convert_emoji(self, text: str, metadata: dict) -> str:
        tags = []
        for emoji_char, tag in self.EMOJI_MAP.items():
            if emoji_char in text:
                text = text.replace(emoji_char, tag)
                tags.append(tag)
        # 其余未映射的 Emoji 用 demojize 转换
        text = emoji.demojize(text, delimiters=("[", "]"))
        metadata["emoji_tags"] = tags
        return text

    def _convert_kaomoji(self, text: str, metadata: dict) -> str:
        pos, neg = 0, 0
        for km in self.POSITIVE_KAOMOJIS:
            if km in text:
                text = text.replace(km, "[颜文字-正面]")
                pos += 1
        for km in self.NEGATIVE_KAOMOJIS:
            if km in text:
                text = text.replace(km, "[颜文字-负面]")
                neg += 1
        metadata["kaomoji"] = {"positive": pos, "negative": neg}
        return text

    def _normalize_slang(self, text: str, metadata: dict) -> str:
        replaced = []
        for slang in self._sorted_slangs:
            if slang in text:
                text = text.replace(slang, SLANG_DICT[slang])
                replaced.append(slang)
        metadata["slang_replaced"] = replaced
        return text

    def _compress_punctuation(self, text: str) -> str:
        text = re.sub(r"！{2,}", "[!强调!]", text)
        text = re.sub(r"!{2,}", "[!强调!]", text)
        text = re.sub(r"？{2,}", "[?疑问?]", text)
        text = re.sub(r"\?{2,}", "[?疑问?]", text)
        text = re.sub(r"。{3,}", "[...省略]", text)
        text = re.sub(r"\.{3,}", "[...省略]", text)
        text = re.sub(r"~{2,}", "[~拖音~]", text)
        text = re.sub(r"～{2,}", "[~拖音~]", text)
        return text

    def _clean_special_chars(self, text: str) -> str:
        # 保留中文、英文、数字、常用标点及方括号（语义标签用）
        text = re.sub(
            r"[^\u4e00-\u9fa5a-zA-Z0-9\s\[\]【】！？。，、；：""''（）()]",
            "",
            text,
        )
        text = re.sub(r"\s+", " ", text)
        return text.strip()
