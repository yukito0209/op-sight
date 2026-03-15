# 玩家评论情绪分析工具

> 针对二次元游戏社区（TapTap / NGA / B 站）的中文玩家评论情感分析系统

## 项目简介

本项目实现了一套面向二次元游戏社区的玩家评论情感分析工具，支持**负面 / 中性 / 正面**三分类，并集成了反讽检测、情绪强度评估、问题归因等功能。

**技术栈**：Chinese RoBERTa + FastAPI + React/TypeScript

---

## 实现原理

### 1. 数据与弱监督标注

直接使用 [TapTap 游戏评论 Kaggle 数据集](https://www.kaggle.com/datasets/karwinwang/taptap-mobile-game-reviews-chinese)，通过**星级评分作为代理标签**（Weak Supervision），避免人工标注：

| 星级                  | 情感标签         |
| --------------------- | ---------------- |
| ⭐⭐（1–2 星）       | negative（负面） |
| ⭐⭐⭐（3 星）        | neutral（中性）  |
| ⭐⭐⭐⭐⭐（4–5 星） | positive（正面） |

数据集按 **80 / 10 / 10** 划分训练 / 验证 / 测试集，最终规模：

```
train: 31,989 条  │  val: 3,998 条  │  test: 3,998 条
类别分布（训练集）：positive 17,695 / negative 10,945 / neutral 3,349
```

### 2. 文本预处理

`preprocessor/text_preprocessor.py` 实现了面向游戏社区的预处理流程：

1. **Emoji 语义化**：将常见 Emoji 替换为语义标签（😭 → `[大哭]`），保留情绪信号
2. **颜文字识别**：匹配 `(╯‵□′)╯` 等日系颜文字并转为标签
3. **游戏黑话归一化**：基于 `data/slang_dict.py` 中的 `SLANG_DICT` 字典，将 `yygq` → `阴阳怪气`、`chfm` → `擦，非酋`等，关键字按**长度降序排序**以避免短词覆盖长词
4. **标点压缩**：三个以上连续标点压缩为两个
5. **特殊字符清洗**：去除零宽字符、控制符等

> **设计要点**：`preprocess()` 返回 `(processed_text, metadata)`，其中 `metadata["original"]` 保留原始文本。推理时反讽检测**始终使用原始文本**（保留 Emoji），模型推理使用预处理后文本。

### 3. 模型架构

**基础模型**：[hfl/chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext)（110M 参数）

在预训练 RoBERTa 基础上接分类头（`BertForSequenceClassification`），扩展了 20 个游戏领域词汇（`原神`、`圣遗物`、`yyds`、`坐牢`等），使这些高频词不再被拆分为子词。

推理层（`inference/sentiment_analyzer.py`）使用 `AutoTokenizer` / `AutoModelForSequenceClassification` 加载，与具体模型实现解耦，便于后续替换 backbone。

### 4. 训练策略

| 超参数         | 值                                  |
| -------------- | ----------------------------------- |
| 最大序列长度   | 256                                 |
| Batch size     | 32（梯度累积 ×2 = 有效 64）        |
| 学习率         | 2e-5                                |
| 训练轮数       | 5 epoch（EarlyStopping patience=3） |
| 优化器         | AdamW + Weight Decay 0.01           |
| 学习率调度     | Linear Warmup（10%）→ Linear Decay |
| 混合精度       | FP16（CUDA）                        |
| Label Smoothing | 0.1（缓解弱监督标签噪声）          |

**类别权重损失**（应对中性类严重不足）：

```python
CrossEntropyLoss(weight=[1.0, 3.5, 0.7], label_smoothing=0.1)
#               negative  neutral  positive
```

中性样本（3349 条）仅为正面（17695 条）的 19%，赋予 3.5× 权重显著改善中性召回率。Label Smoothing 则缓解弱监督标注（3 星评论）带来的标签噪声。

### 5. 反讽检测

基于规则的三层检测，**必须在原始文本（含 Emoji）上运行**：

1. **正负词共现**：文本同时包含 `SARCASM_POSITIVE_WORDS`（"优秀""感谢策划"等）和 `SARCASM_NEGATIVE_CONTEXTS`（"又歪""坐牢"等），且模型置信度 > 0.7
2. **句式匹配**：正则匹配反讽句式（"真是.*啊""感谢.*让我"等）
3. **反讽 Emoji**：文本包含 😅🙃🫠 等暗示反讽的 Emoji

检测到反讽时，正面情感自动修正为负面，置信度衰减至 ×0.8。

---

## 训练结果

在验证集上（3,998 条）的最终指标：

| 指标               | 值                        |
| ------------------ | ------------------------- |
| **Macro F1** | **0.642**           |
| Weighted F1        | 0.764                     |
| 负面 F1            | 0.753                     |
| 中性 F1            | 0.324                     |
| 正面 F1            | 0.849                     |
| 训练总时长         | ~7 分钟（RTX 4070 SUPER） |

中性 F1 偏低（0.32）是预期内的，主要原因：

- 中性样本数量少（约 19% 占比）
- 3 星弱监督标签噪声较大（3 星评论情绪复杂，可能含正面/负面混合表达）

---

## 项目结构

```
op-sight/
├── config/
│   └── model_config.py        # 超参数、标签映射、星级→标签转换
├── data/
│   ├── slang_dict.py          # 游戏黑话词典、反讽词库、归因关键词
│   └── processed/             # 预处理后的 train/val/test.json（gitignore）
├── preprocessor/
│   └── text_preprocessor.py  # Emoji/颜文字/黑话/标点预处理
├── training/
│   ├── dataset.py             # PyTorch Dataset（tokenize + pad）
│   └── train.py               # WeightedTrainer + 训练入口
├── inference/
│   └── sentiment_analyzer.py # 推理 + 反讽检测 + 强度/归因/建议
├── backend/
│   └── main.py                # FastAPI：/analyze、/analyze/batch、/health
├── frontend/                  # Vite + React + TypeScript 前端
│   ├── src/
│   │   ├── components/SentimentAnalyzer.tsx
│   │   ├── api/sentiment.ts
│   │   ├── index.css          # 全局样式与动画
│   │   └── types.ts
│   └── vite.config.ts         # Proxy /analyze → :8000
├── scripts/
│   └── prepare_data.py        # CSV → train/val/test.json
└── requirements.txt
```

---

## 使用方法

### 环境要求

- Python 3.12+
- CUDA 12.x（可选，CPU 也可运行但较慢）
- Node.js 18+

### 1. 安装依赖

```bash
# Python 依赖
pip install -r requirements.txt

# PyTorch（CUDA 版，推荐）
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 2. 数据准备

下载 [TapTap 数据集](https://www.kaggle.com/datasets/karwinwang/taptap-mobile-game-reviews-chinese)，放至项目根目录，然后：

```bash
python scripts/prepare_data.py \
    --input taptap_game_reviews.csv \
    --output_dir data/processed
```

### 3. 模型训练

```bash
python training/train.py
# 可选参数：--epochs 3 --batch_size 16 --lr 3e-5
```

训练完成后，最佳模型保存于 `models/sentiment/best_model/`。

### 4. 启动后端

```bash
uvicorn backend.main:app --reload --port 8000
```

API 文档访问：http://localhost:8000/docs

### 5. 启动前端

```bash
cd frontend
npm install
npm run dev
```

访问：http://localhost:5173

### API 示例

```bash
# 单条分析
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "雷神yyds！伤害爆炸，这角色绝了！"}'

# 批量分析
curl -X POST http://localhost:8000/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["太棒了，80抽又歪了😅", "圣遗物坐牢三个月"]}'
```

---

## 改进方向

### 数据层

- **人工标注**：对 3 星评论进行二次标注，解决弱监督噪声（3 星评论实际情绪分布复杂）
- **数据增强**：对中性样本进行回译（中→英→中）扩充，缓解类别不平衡
- **多平台数据**：引入 NGA、B 站弹幕等平台的评论，提升领域泛化性

### 模型层

- **换用情感预训练模型**：替换为 `IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment`（已在中文情感数据集上预训练），无需修改训练代码即可提升起点
- **更大规模模型**：`hfl/chinese-macbert-large`（330M）或 Qwen2.5-1.5B（decoder-only fine-tune），预期 macro F1 可提升 5–10%
- **多任务学习**：情感分类 + 问题归因联合训练，共享表示以提升归因准确率
- **对比学习**：引入 SimCSE / SupCon Loss，增强类间距离

### 工程层

- **模型量化**：INT8 量化（`torch.quantization`），推理显存减半，延迟降低约 30%
- **批量推理优化**：当前批量接口逐条推理，改为真正的 batch forward
- **异步推理**：后端接入 `asyncio` + 线程池，支持高并发请求

---

## 已知局限

1. **反讽检测依赖规则**：基于词典和正则，对新型网络用语覆盖不足
2. **中性类表现较弱**：受限于弱监督标签质量和样本数量，中性 F1 仅 0.32
3. **领域迁移**：模型主要在 TapTap 数据上训练，对 NGA 长文攻略贴等场景泛化性未验证
