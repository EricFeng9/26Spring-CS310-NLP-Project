#!/usr/bin/env python
"""生成任务B的复杂度启发实验表。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io.files import write_csv
from src.sc.improved_voting import build_path_views, stable_majority_vote
from src.utils import resolve_project_path


DEFAULT_MODELS = ["qwen2_5_0_5b_instruct", "qwen2_5_1_5b_instruct", "tinyllama_1_1b_chat"]
DEFAULT_DATASETS = ["gsm8k", "csqa", "mmlu"]


def load_jsonl(path: Path) -> list[dict]:
    """读取任务A Self-Consistency JSONL。"""
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在：{path}")

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))

    if not rows:
        raise ValueError(f"输入文件为空：{path}")
    return rows


def split_model_dataset(jsonl_path: Path) -> tuple[str, str]:
    """从文件名解析模型和数据集。"""
    model_key, dataset_key = jsonl_path.stem.rsplit("_", 1)
    return model_key, dataset_key


def complexity_vote(dataset_key: str, row: dict) -> tuple[str, dict[str, int]]:
    """按复杂度启发做投票。

    规则：
    1. 先挑复杂度最高的路径作为 `most_complex`
    2. 再做复杂度加权投票，权重等于复杂度分数
    """
    path_views = build_path_views(dataset_key, row)
    valid_views = [view for view in path_views if view["normalized_valid"]]
    if not valid_views:
        return "", {}

    most_complex_view = max(valid_views, key=lambda view: (view["complexity_score"], -view["path_index"]))
    most_complex_prediction = most_complex_view["normalized_answer"]

    weighted_votes: list[str] = []
    for view in valid_views:
        weight = max(int(view["complexity_score"]), 1)
        weighted_votes.extend([view["normalized_answer"]] * weight)
    weighted_prediction, weighted_counts = stable_majority_vote(weighted_votes)
    return most_complex_prediction, weighted_counts


def evaluate_row(dataset_key: str, row: dict) -> list[dict]:
    """返回单样本上的复杂度实验结果。"""
    path_views = build_path_views(dataset_key, row)
    valid_views = [view for view in path_views if view["normalized_valid"]]
    if valid_views:
        most_complex_view = max(valid_views, key=lambda view: (view["complexity_score"], -view["path_index"]))
        most_complex_prediction = most_complex_view["normalized_answer"]
        weighted_votes: list[str] = []
        for view in valid_views:
            weight = max(int(view["complexity_score"]), 1)
            weighted_votes.extend([view["normalized_answer"]] * weight)
        weighted_prediction, weighted_counts = stable_majority_vote(weighted_votes)
    else:
        most_complex_prediction = ""
        weighted_prediction, weighted_counts = "", {}

    return [
        {
            "strategy": "most_complex",
            "prediction": most_complex_prediction,
            "correct": most_complex_prediction == str(row["gold_answer"]).strip(),
            "vote_count": 1 if most_complex_prediction else 0,
            "avg_complexity": round(
                sum(view["complexity_score"] for view in valid_views) / len(valid_views), 6
            ) if valid_views else 0.0,
            "max_complexity": max((view["complexity_score"] for view in valid_views), default=0),
        },
        {
            "strategy": "complexity_weighted_vote",
            "prediction": weighted_prediction,
            "correct": weighted_prediction == str(row["gold_answer"]).strip(),
            "vote_count": sum(weighted_counts.values()) if weighted_counts else 0,
            "avg_complexity": round(
                sum(view["complexity_score"] for view in valid_views) / len(valid_views), 6
            ) if valid_views else 0.0,
            "max_complexity": max((view["complexity_score"] for view in valid_views), default=0),
        },
    ]


def main() -> None:
    """脚本入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="output/raw/A/lightweight_100/self_consistency")
    parser.add_argument("--output-csv", default="output/log/B/complexity_ablation.csv")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    args = parser.parse_args()

    input_dir = resolve_project_path(args.input_dir)
    selected_models = set(args.models)
    selected_datasets = set(args.datasets)
    jsonl_paths: list[Path] = []
    for path in sorted(input_dir.glob("*.jsonl")):
        model_key, dataset_key = split_model_dataset(path)
        if model_key in selected_models and dataset_key in selected_datasets:
            jsonl_paths.append(path)

    if not jsonl_paths:
        raise ValueError(f"没有匹配的输入文件：{input_dir}")

    result_rows: list[dict] = []
    for jsonl_path in jsonl_paths:
        model_key, dataset_key = split_model_dataset(jsonl_path)
        rows = load_jsonl(jsonl_path)
        for strategy in ["most_complex", "complexity_weighted_vote"]:
            correct_count = 0
            vote_count_sum = 0
            avg_complexity_sum = 0.0
            max_complexity_sum = 0
            sample_count = len(rows)
            for row in rows:
                sample_rows = evaluate_row(dataset_key, row)
                sample_row = next(item for item in sample_rows if item["strategy"] == strategy)
                correct_count += int(sample_row["correct"])
                vote_count_sum += int(sample_row["vote_count"])
                avg_complexity_sum += float(sample_row["avg_complexity"])
                max_complexity_sum += int(sample_row["max_complexity"])

            result_rows.append(
                {
                    "model_key": model_key,
                    "dataset_key": dataset_key,
                    "strategy": strategy,
                    "sample_count": sample_count,
                    "correct_count": correct_count,
                    "accuracy": round(correct_count / sample_count, 6),
                    "mean_vote_count": round(vote_count_sum / sample_count, 6),
                    "mean_avg_complexity": round(avg_complexity_sum / sample_count, 6),
                    "mean_max_complexity": round(max_complexity_sum / sample_count, 6),
                }
            )

    write_csv(resolve_project_path(args.output_csv), result_rows)
    print(f"任务B复杂度实验完成：{resolve_project_path(args.output_csv)}")


if __name__ == "__main__":
    main()
