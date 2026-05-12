#!/usr/bin/env python
"""从标准化数据集中构建固定 few-shot 示例文件。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import load_processed_split
from src.io.files import write_jsonl
from src.utils import load_yaml, project_root


def build_one(dataset_key: str) -> None:
    """为单个数据集生成 few-shot 示例。"""
    config = load_yaml(project_root() / "configs" / "datasets" / f"{dataset_key}.yaml")
    source_split = config["fewshot_source_split"]
    samples = load_processed_split(dataset_key, source_split, limit=3)
    rows = []
    for sample in samples:
        rows.append(
            {
                "id": sample.id,
                "dataset": sample.dataset,
                "question": sample.question,
                "choices": sample.choices,
                "answer": sample.gold_answer,
                "metadata": sample.metadata,
            }
        )
    output_path = Path("data/processed/fewshot") / f"{dataset_key}_fewshot.jsonl"
    write_jsonl(project_root() / output_path, rows)


def main() -> None:
    """脚本入口。"""
    for dataset_key in ["gsm8k", "csqa", "mmlu"]:
        build_one(dataset_key)
        print(f"few-shot 示例已生成：{dataset_key}")


if __name__ == "__main__":
    main()
