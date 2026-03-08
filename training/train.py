"""
模型训练脚本

用法：
  python training/train.py
  python training/train.py --epochs 3 --batch_size 16
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import ID2LABEL, ID2LABEL_ZH, LABEL2ID, ModelConfig
from training.dataset import SentimentDataset


# ──────────────────────────────────────────────
# 带类别权重的 Trainer
# ──────────────────────────────────────────────
class WeightedTrainer(Trainer):
    def __init__(self, class_weights: list[float] | None = None, **kwargs):
        super().__init__(**kwargs)
        if class_weights:
            w = torch.tensor(class_weights, dtype=torch.float32)
            if torch.cuda.is_available():
                w = w.cuda()
            self.loss_fn = nn.CrossEntropyLoss(weight=w)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


# ──────────────────────────────────────────────
# 评估指标
# ──────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    report = classification_report(
        labels,
        preds,
        target_names=[ID2LABEL_ZH[i] for i in range(3)],
        output_dict=True,
        zero_division=0,
    )

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "negative_f1": report[ID2LABEL_ZH[0]]["f1-score"],
        "neutral_f1": report[ID2LABEL_ZH[1]]["f1-score"],
        "positive_f1": report[ID2LABEL_ZH[2]]["f1-score"],
    }


# ──────────────────────────────────────────────
# 主训练函数
# ──────────────────────────────────────────────
def train(cfg: ModelConfig):
    data_dir = Path(cfg.data_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"加载 tokenizer：{cfg.model_name}")
    tokenizer = BertTokenizer.from_pretrained(cfg.model_name)

    # 添加游戏领域词汇
    new_tokens = [
        "原神", "星铁", "绝区零", "崩坏", "小保底", "大保底",
        "圣遗物", "命座", "强度党", "策划", "深渊", "满星", "坐牢",
        "歪了", "双黄", "氪金", "月卡", "白嫖", "yyds", "yygq",
    ]
    added = tokenizer.add_tokens(new_tokens)
    print(f"新增词表 token：{added} 个")

    print(f"加载模型：{cfg.model_name}")
    model = BertForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.resize_token_embeddings(len(tokenizer))

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_ds = SentimentDataset(data_dir / "train.json", tokenizer, cfg.max_length)
    val_ds = SentimentDataset(data_dir / "val.json", tokenizer, cfg.max_length)
    print(f"训练集：{len(train_ds)} 条 | 验证集：{len(val_ds)} 条")

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=int(cfg.warmup_ratio * (len(train_ds) // (cfg.batch_size * cfg.gradient_accumulation_steps)) * cfg.num_epochs),
        fp16=cfg.fp16 and torch.cuda.is_available(),
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to=["none"],
        dataloader_num_workers=0,   # Windows 下设为 0 避免多进程问题
        remove_unused_columns=False,
    )

    trainer = WeightedTrainer(
        class_weights=cfg.class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\n开始训练...")
    trainer.train()

    best_dir = out_dir / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"\n最佳模型已保存至：{best_dir}")

    print("\n最终验证集评估：")
    results = trainer.evaluate()
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = ModelConfig()
    if args.epochs:
        cfg.num_epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.learning_rate = args.lr
    train(cfg)
