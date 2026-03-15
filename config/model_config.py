from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    # 模型基础
    model_name: str = "hfl/chinese-roberta-wwm-ext"
    num_labels: int = 3          # 0=负面 1=中性 2=正面
    max_length: int = 256        # TapTap 评论普遍较短，256 足够覆盖长评论

    # 训练（RTX 4070 Super 12GB）
    batch_size: int = 32
    gradient_accumulation_steps: int = 2   # 有效 batch = 64
    learning_rate: float = 2e-5
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # 显存优化
    fp16: bool = True
    gradient_checkpointing: bool = False   # 110M 参数不需要

    # 路径
    data_dir: str = "./data/processed"
    output_dir: str = "./models/sentiment"

    # 日志
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 200

    # 类别权重：补偿 neutral 样本少（负面/中性/正面 ≈ 1.4:0.4:2.2 万）
    class_weights: List[float] = field(default_factory=lambda: [1.0, 3.5, 0.7])


# 标签映射
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
ID2LABEL_ZH = {0: "负面", 1: "中性", 2: "正面"}


def rating_to_label(rating: int) -> str:
    """TapTap 星级 → 三分类标签"""
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"
