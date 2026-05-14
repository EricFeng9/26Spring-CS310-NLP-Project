#!/usr/bin/env python
"""下载并标准化任务A所需数据集。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from datasets import DatasetDict, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_yaml, project_root


DATASET_KEYS = ["gsm8k", "csqa", "mmlu"]


def normalize_gsm8k_item(item: dict, split: str, index: int) -> dict:
    """标准化 GSM8K 单条样本。"""
    answer_text = str(item["answer"])
    final_answer = answer_text.split("####")[-1].strip().replace(",", "")
    return {
        "id": f"gsm8k_{split}_{index}",
        "dataset": "gsm8k",
        "question": str(item["question"]).strip(),
        "choices": [],
        "answer": final_answer,
        "metadata": {
            "split": split,
            "raw_answer": answer_text,
            "fewshot_reasoning": answer_text.split("####")[0].strip(),
        },
    }


def normalize_csqa_item(item: dict, split: str, index: int) -> dict:
    """标准化 CommonsenseQA 单条样本。"""
    choice_labels = item["choices"]["label"]
    choice_texts = item["choices"]["text"]
    choices = [f"{label}. {text}" for label, text in zip(choice_labels, choice_texts)]
    return {
        "id": str(item["id"]) if item.get("id") else f"csqa_{split}_{index}",
        "dataset": "csqa",
        "question": str(item["question"]).strip(),
        "choices": choices,
        "answer": str(item["answerKey"]).strip().upper(),
        "metadata": {
            "split": split,
            "fewshot_reasoning": (
                "Compare the options using commonsense knowledge, eliminate the implausible choices, "
                f"and conclude that the best answer is {item['answerKey'].strip().upper()}."
            ),
        },
    }


def normalize_mmlu_item(item: dict, split: str, index: int) -> dict:
    """标准化 MMLU 单条样本。"""
    answer_index = int(item["answer"])
    labels = ["A", "B", "C", "D"]
    choices = [f"{label}. {text}" for label, text in zip(labels, item["choices"])]
    return {
        "id": f"mmlu_{split}_{index}",
        "dataset": "mmlu",
        "question": str(item["question"]).strip(),
        "choices": choices,
        "answer": labels[answer_index],
        "metadata": {
            "split": split,
            "subject": str(item["subject"]),
            "fewshot_reasoning": (
                "Use the subject knowledge step by step, rule out the incorrect options, "
                f"and conclude that the final answer is {labels[answer_index]}."
            ),
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """写出 JSONL 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_one(dataset_key: str) -> None:
    """下载并处理单个数据集。"""
    config = load_yaml(project_root() / "configs" / "datasets" / f"{dataset_key}.yaml")
    raw_dir = project_root() / "data" / "raw" / dataset_key
    processed_dir = project_root() / config["processed_dir"]

    dataset = load_dataset(config["huggingface_path"], config["subset"])
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"{dataset_key} 下载结果不是 DatasetDict。")
    dataset.save_to_disk(str(raw_dir))

    for split_alias, split_name in config["splits"].items():
        rows: list[dict] = []
        for index, item in enumerate(dataset[split_name]):
            if dataset_key == "gsm8k":
                rows.append(normalize_gsm8k_item(item, split_alias, index))
            elif dataset_key == "csqa":
                rows.append(normalize_csqa_item(item, split_alias, index))
            elif dataset_key == "mmlu":
                rows.append(normalize_mmlu_item(item, split_alias, index))
            else:
                raise ValueError(f"未知数据集：{dataset_key}")
        write_jsonl(processed_dir / f"{split_alias}.jsonl", rows)


def main() -> None:
    """脚本入口。"""
    for dataset_key in DATASET_KEYS:
        print(f"开始处理数据集：{dataset_key}")
        prepare_one(dataset_key)
        print(f"数据集处理完成：{dataset_key}")


if __name__ == "__main__":
    main()
