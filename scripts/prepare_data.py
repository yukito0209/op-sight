"""
数据准备脚本：TapTap CSV → 三分类 JSON

标签映射：
  rating 1-2 → 0 (negative)   共 ~13,632 条
  rating 3   → 1 (neutral)    共  ~4,181 条
  rating 4-5 → 2 (positive)   共 ~22,172 条

划分比例：train 80% / val 10% / test 10%

用法：
  python scripts/prepare_data.py
  python scripts/prepare_data.py --input data/raw/taptap_game_reviews.csv --max_per_class 5000
"""

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# 把项目根目录加入 PATH，以便导入 preprocessor
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import rating_to_label, LABEL2ID
from preprocessor.text_preprocessor import GameTextPreprocessor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="taptap_game_reviews.csv")
    p.add_argument("--output_dir", default="data/processed")
    p.add_argument(
        "--max_per_class",
        type=int,
        default=None,
        help="每个类别最多使用多少条（None=全量）",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no_preprocess",
        action="store_true",
        help="跳过预处理，直接存原始文本（调试用）",
    )
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    root = Path(__file__).parent.parent
    csv_path = root / args.input
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"读取数据：{csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df.dropna(subset=["review_content", "rating"])
    df["rating"] = df["rating"].astype(int)
    df["label_str"] = df["rating"].apply(rating_to_label)
    df["label"] = df["label_str"].map(LABEL2ID)

    print(f"总计 {len(df)} 条，标签分布：")
    print(df["label_str"].value_counts().to_string())

    # 按类别分组，可选截断
    groups: dict[str, list] = {}
    for label_str in ["negative", "neutral", "positive"]:
        rows = df[df["label_str"] == label_str].to_dict("records")
        random.shuffle(rows)
        if args.max_per_class:
            rows = rows[: args.max_per_class]
        groups[label_str] = rows

    preprocessor = GameTextPreprocessor() if not args.no_preprocess else None

    def build_item(row: dict) -> dict:
        text = str(row["review_content"]).strip()
        if preprocessor:
            processed, _ = preprocessor.preprocess(text)
        else:
            processed = text
        return {
            "text": processed,
            "original_text": text,
            "label": int(row["label"]),
            "label_str": row["label_str"],
            "rating": int(row["rating"]),
            "game": row.get("game_name", ""),
        }

    # 合并并打乱
    all_rows = [r for rows in groups.values() for r in rows]
    random.shuffle(all_rows)

    print(f"\n预处理并构建数据集（共 {len(all_rows)} 条）...")
    items = [build_item(r) for r in tqdm(all_rows)]

    # 划分
    n = len(items)
    n_val = int(n * 0.1)
    n_test = int(n * 0.1)
    test = items[:n_test]
    val = items[n_test: n_test + n_val]
    train = items[n_test + n_val:]

    splits = {"train": train, "val": val, "test": test}
    for split, data in splits.items():
        path = out_dir / f"{split}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # 打印标签分布
        from collections import Counter
        dist = Counter(d["label_str"] for d in data)
        print(f"{split}: {len(data)} 条  {dict(dist)}")

    print(f"\n数据已保存至 {out_dir}/")


if __name__ == "__main__":
    main()
