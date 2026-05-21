#!/usr/bin/env python
"""生成任务B的后处理消融实验表。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io.files import write_csv
from src.sc.improved_voting import build_path_views, is_valid_answer, stable_majority_vote
from src.utils import resolve_project_path


DEFAULT_MODELS = ["qwen2_5_0_5b_instruct", "qwen2_5_1_5b_instruct", "tinyllama_1_1b_chat"]
DEFAULT_DATASETS = ["gsm8k", "csqa", "mmlu"]
METHOD_ORDER = ["original_sc", "normalized_vote", "filtered_vote", "normalized_filtered_vote"]


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


def vote_answers(method: str, dataset_key: str, row: dict, max_paths: int) -> tuple[str, dict[str, int]]:
    """在前 `max_paths` 条路径上执行指定任务B方法。"""
    limited_row = dict(row)
    limited_row["paths"] = row["paths"][:max_paths]
    path_views = build_path_views(dataset_key, limited_row)

    if method == "original_sc":
        answers = [str(path.get("answer", "")).strip() for path in limited_row["paths"]]
    elif method == "normalized_vote":
        answers = [view["normalized_answer"] for view in path_views]
    elif method == "filtered_vote":
        answers = [
            view["raw_answer"]
            for view in path_views
            if is_valid_answer(dataset_key, view["raw_answer"])
        ]
    elif method == "normalized_filtered_vote":
        answers = [
            view["normalized_answer"]
            for view in path_views
            if view["normalized_valid"]
        ]
    else:
        raise ValueError(f"未知方法：{method}")

    return stable_majority_vote(answers)


def mean_majority_share(rows: list[dict], method: str, dataset_key: str, max_paths: int) -> float:
    """计算指定方法在指定路径数下的平均多数票占比。"""
    shares: list[float] = []
    for row in rows:
        _, vote_counts = vote_answers(method, dataset_key, row, max_paths)
        if vote_counts:
            shares.append(max(vote_counts.values()) / max_paths)
        else:
            shares.append(0.0)
    return sum(shares) / len(shares)


def evaluate_method(rows: list[dict], method: str, dataset_key: str, max_paths: int) -> dict:
    """评估一个消融配置。"""
    correct_count = 0
    valid_vote_count = 0
    changed_count = 0
    original_correct_count = sum(1 for row in rows if bool(row["correct"]))
    for row in rows:
        prediction, vote_counts = vote_answers(method, dataset_key, row, max_paths)
        correct_count += int(prediction == row["gold_answer"])
        valid_vote_count += int(bool(vote_counts))
        changed_count += int(prediction != str(row["prediction"]).strip())

    sample_count = len(rows)
    accuracy = correct_count / sample_count
    original_accuracy = original_correct_count / sample_count
    return {
        "method": method,
        "num_paths_used": max_paths,
        "sample_count": sample_count,
        "correct_count": correct_count,
        "accuracy": round(accuracy, 6),
        "absolute_gain_vs_original_sc": round(accuracy - original_accuracy, 6),
        "valid_vote_rate": round(valid_vote_count / sample_count, 6),
        "vote_changed_rate": round(changed_count / sample_count, 6),
        "mean_majority_share": round(mean_majority_share(rows, method, dataset_key, max_paths), 6),
    }


def build_num_paths_ablation(jsonl_paths: list[Path]) -> list[dict]:
    """生成路径数消融表。"""
    result_rows: list[dict] = []
    for jsonl_path in jsonl_paths:
        model_key, dataset_key = split_model_dataset(jsonl_path)
        rows = load_jsonl(jsonl_path)
        max_available_paths = min(len(row["paths"]) for row in rows)
        for method in ["original_sc", "normalized_filtered_vote"]:
            for num_paths in range(1, max_available_paths + 1):
                metrics = evaluate_method(rows, method, dataset_key, num_paths)
                result_rows.append(
                    {
                        "model_key": model_key,
                        "dataset_key": dataset_key,
                        **metrics,
                    }
                )
    return result_rows


def build_component_ablation(jsonl_paths: list[Path]) -> list[dict]:
    """生成任务B组件消融表。"""
    result_rows: list[dict] = []
    for jsonl_path in jsonl_paths:
        model_key, dataset_key = split_model_dataset(jsonl_path)
        rows = load_jsonl(jsonl_path)
        max_available_paths = min(len(row["paths"]) for row in rows)
        for method in METHOD_ORDER:
            metrics = evaluate_method(rows, method, dataset_key, max_available_paths)
            result_rows.append(
                {
                    "model_key": model_key,
                    "dataset_key": dataset_key,
                    "component_setting": method,
                    **metrics,
                }
            )
    return result_rows


def build_dataset_average_rows(component_rows: list[dict]) -> list[dict]:
    """生成按数据集和方法聚合的平均消融表。"""
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in component_rows:
        key = (row["dataset_key"], row["component_setting"])
        grouped.setdefault(key, []).append(row)

    average_rows: list[dict] = []
    for (dataset_key, component_setting), rows in sorted(grouped.items()):
        average_rows.append(
            {
                "dataset_key": dataset_key,
                "component_setting": component_setting,
                "model_count": len(rows),
                "mean_accuracy": round(sum(float(row["accuracy"]) for row in rows) / len(rows), 6),
                "mean_absolute_gain_vs_original_sc": round(
                    sum(float(row["absolute_gain_vs_original_sc"]) for row in rows) / len(rows),
                    6,
                ),
                "mean_vote_changed_rate": round(
                    sum(float(row["vote_changed_rate"]) for row in rows) / len(rows),
                    6,
                ),
                "mean_majority_share": round(
                    sum(float(row["mean_majority_share"]) for row in rows) / len(rows),
                    6,
                ),
            }
        )
    return average_rows


def main() -> None:
    """脚本入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="output/raw/A/lightweight_100/self_consistency")
    parser.add_argument("--num-paths-csv", default="output/log/B/num_paths_ablation.csv")
    parser.add_argument("--component-csv", default="output/log/B/component_ablation.csv")
    parser.add_argument("--dataset-average-csv", default="output/log/B/component_dataset_average.csv")
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

    num_paths_rows = build_num_paths_ablation(jsonl_paths)
    component_rows = build_component_ablation(jsonl_paths)
    dataset_average_rows = build_dataset_average_rows(component_rows)

    write_csv(resolve_project_path(args.num_paths_csv), num_paths_rows)
    write_csv(resolve_project_path(args.component_csv), component_rows)
    write_csv(resolve_project_path(args.dataset_average_csv), dataset_average_rows)

    print("任务B消融实验完成。")
    print(f"路径数消融表：{resolve_project_path(args.num_paths_csv)}")
    print(f"组件消融表：{resolve_project_path(args.component_csv)}")
    print(f"数据集平均表：{resolve_project_path(args.dataset_average_csv)}")


if __name__ == "__main__":
    main()
