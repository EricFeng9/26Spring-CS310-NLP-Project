#!/usr/bin/env python
"""运行任务B的 Self-Consistency 轻量后处理改进实验。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io.files import write_csv, write_jsonl
from src.sc.improved_voting import (
    build_path_views,
    duplicate_reasoning_ratio,
    is_valid_answer,
    stable_majority_vote,
)
from src.utils import resolve_project_path


DEFAULT_MODELS = ["qwen2_5_0_5b_instruct", "qwen2_5_1_5b_instruct", "tinyllama_1_1b_chat"]
DEFAULT_DATASETS = ["gsm8k", "csqa", "mmlu"]
METHOD_ORDER = ["original_sc", "normalized_vote", "filtered_vote", "normalized_filtered_vote"]


def load_jsonl(path: Path) -> list[dict]:
    """读取 JSONL 文件，文件缺失或为空直接报错。"""
    if not path.exists():
        raise FileNotFoundError(f"任务B输入文件不存在：{path}")

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))

    if not rows:
        raise ValueError(f"任务B输入文件为空：{path}")
    return rows


def split_model_dataset(jsonl_path: Path) -> tuple[str, str]:
    """从任务A结果文件名解析模型和数据集。"""
    model_key, dataset_key = jsonl_path.stem.rsplit("_", 1)
    return model_key, dataset_key


def vote_for_method(method: str, dataset_key: str, row: dict, path_views: list[dict]) -> tuple[str, dict[str, int], list[str]]:
    """根据任务B方法配置生成预测、票数和参与投票答案。"""
    if method == "original_sc":
        answers = [str(path.get("answer", "")).strip() for path in row["paths"]]
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
        raise ValueError(f"未知任务B方法：{method}")

    prediction, vote_counts = stable_majority_vote(answers)
    return prediction, vote_counts, answers


def evaluate_file(jsonl_path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """评估单个模型-数据集的任务B后处理结果。"""
    model_key, dataset_key = split_model_dataset(jsonl_path)
    rows = load_jsonl(jsonl_path)
    detail_rows: list[dict] = []
    change_rows: list[dict] = []
    summary_rows: list[dict] = []

    method_stats = {
        method: {
            "correct_count": 0,
            "vote_changed_count": 0,
            "wrong_to_correct_count": 0,
            "correct_to_wrong_count": 0,
            "valid_path_count": 0,
            "total_path_count": 0,
            "unique_answers_before_sum": 0,
            "unique_answers_after_sum": 0,
            "majority_share_before_sum": 0.0,
            "majority_share_after_sum": 0.0,
            "duplicate_reasoning_ratio_sum": 0.0,
        }
        for method in METHOD_ORDER
    }

    for row in rows:
        path_views = build_path_views(dataset_key, row)
        original_prediction = str(row["prediction"]).strip()
        original_correct = bool(row["correct"])
        original_vote_counts = row["vote_counts"]
        original_majority = max(int(count) for count in original_vote_counts.values())
        original_majority_share = original_majority / len(row["paths"])
        duplicate_ratio = duplicate_reasoning_ratio(row["paths"])

        for method in METHOD_ORDER:
            prediction, vote_counts, voted_answers = vote_for_method(method, dataset_key, row, path_views)
            correct = prediction == row["gold_answer"]
            valid_path_count = len(voted_answers)
            total_path_count = len(row["paths"])
            majority_share_after = 0.0
            if vote_counts:
                majority_share_after = max(int(count) for count in vote_counts.values()) / total_path_count

            stats = method_stats[method]
            stats["correct_count"] += int(correct)
            stats["vote_changed_count"] += int(prediction != original_prediction)
            stats["wrong_to_correct_count"] += int((not original_correct) and correct)
            stats["correct_to_wrong_count"] += int(original_correct and (not correct))
            stats["valid_path_count"] += valid_path_count
            stats["total_path_count"] += total_path_count
            stats["unique_answers_before_sum"] += len(original_vote_counts)
            stats["unique_answers_after_sum"] += len(vote_counts)
            stats["majority_share_before_sum"] += original_majority_share
            stats["majority_share_after_sum"] += majority_share_after
            stats["duplicate_reasoning_ratio_sum"] += duplicate_ratio

            detail = {
                "model_key": model_key,
                "dataset_key": dataset_key,
                "id": row["id"],
                "method": method,
                "gold_answer": row["gold_answer"],
                "original_prediction": original_prediction,
                "prediction": prediction,
                "original_correct": original_correct,
                "correct": correct,
                "vote_changed": prediction != original_prediction,
                "voted_answers": voted_answers,
                "vote_counts": vote_counts,
                "path_views": path_views,
            }
            detail_rows.append(detail)

            if method != "original_sc" and prediction != original_prediction:
                change_rows.append(
                    {
                        "model_key": model_key,
                        "dataset_key": dataset_key,
                        "id": row["id"],
                        "method": method,
                        "gold_answer": row["gold_answer"],
                        "original_prediction": original_prediction,
                        "prediction": prediction,
                        "original_correct": original_correct,
                        "correct": correct,
                        "original_vote_counts": original_vote_counts,
                        "new_vote_counts": vote_counts,
                    }
                )

    sample_count = len(rows)
    for method in METHOD_ORDER:
        stats = method_stats[method]
        total_paths = stats["total_path_count"]
        valid_paths = stats["valid_path_count"]
        summary_rows.append(
            {
                "model_key": model_key,
                "dataset_key": dataset_key,
                "method": method,
                "sample_count": sample_count,
                "correct_count": stats["correct_count"],
                "accuracy": round(stats["correct_count"] / sample_count, 6),
                "valid_path_rate": round(valid_paths / total_paths, 6),
                "answer_extraction_failure_rate": round(1 - valid_paths / total_paths, 6),
                "vote_changed_rate": round(stats["vote_changed_count"] / sample_count, 6),
                "mean_unique_answers_before": round(stats["unique_answers_before_sum"] / sample_count, 6),
                "mean_unique_answers_after": round(stats["unique_answers_after_sum"] / sample_count, 6),
                "mean_majority_share_before": round(stats["majority_share_before_sum"] / sample_count, 6),
                "mean_majority_share_after": round(stats["majority_share_after_sum"] / sample_count, 6),
                "mean_duplicate_reasoning_ratio": round(stats["duplicate_reasoning_ratio_sum"] / sample_count, 6),
                "wrong_to_correct_count": stats["wrong_to_correct_count"],
                "correct_to_wrong_count": stats["correct_to_wrong_count"],
            }
        )

    return detail_rows, summary_rows, change_rows


def build_normalization_rows(summary_rows: list[dict]) -> list[dict]:
    """生成答案归一化投票专用结果表。"""
    return [
        row
        for row in summary_rows
        if row["method"] in {"original_sc", "normalized_vote", "normalized_filtered_vote"}
    ]


def build_filtering_rows(summary_rows: list[dict]) -> list[dict]:
    """生成路径过滤统计表。"""
    return [
        row
        for row in summary_rows
        if row["method"] in {"original_sc", "filtered_vote", "normalized_filtered_vote"}
    ]


def main() -> None:
    """脚本入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="output/raw/A/lightweight_100/self_consistency")
    parser.add_argument("--raw-output", default="output/raw/B/sc_improvement_results.jsonl")
    parser.add_argument("--summary-csv", default="output/log/B/sc_improvement_results.csv")
    parser.add_argument("--normalization-csv", default="output/log/B/normalization_vote_results.csv")
    parser.add_argument("--filtering-csv", default="output/log/B/path_filtering_statistics.csv")
    parser.add_argument("--change-csv", default="output/log/B/vote_change_cases.csv")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    args = parser.parse_args()

    input_dir = resolve_project_path(args.input_dir)
    selected_models = set(args.models)
    selected_datasets = set(args.datasets)
    jsonl_paths = []
    for path in sorted(input_dir.glob("*.jsonl")):
        model_key, dataset_key = split_model_dataset(path)
        if model_key in selected_models and dataset_key in selected_datasets:
            jsonl_paths.append(path)

    if not jsonl_paths:
        raise ValueError(f"没有匹配的任务A SC 输入文件：{input_dir}")

    all_detail_rows: list[dict] = []
    all_summary_rows: list[dict] = []
    all_change_rows: list[dict] = []
    for jsonl_path in jsonl_paths:
        detail_rows, summary_rows, change_rows = evaluate_file(jsonl_path)
        all_detail_rows.extend(detail_rows)
        all_summary_rows.extend(summary_rows)
        all_change_rows.extend(change_rows)

    write_jsonl(resolve_project_path(args.raw_output), all_detail_rows)
    write_csv(resolve_project_path(args.summary_csv), all_summary_rows)
    write_csv(resolve_project_path(args.normalization_csv), build_normalization_rows(all_summary_rows))
    write_csv(resolve_project_path(args.filtering_csv), build_filtering_rows(all_summary_rows))
    if all_change_rows:
        write_csv(resolve_project_path(args.change_csv), all_change_rows)

    print("任务B改进实验完成。")
    print(f"逐题结果：{resolve_project_path(args.raw_output)}")
    print(f"汇总表：{resolve_project_path(args.summary_csv)}")
    print(f"归一化表：{resolve_project_path(args.normalization_csv)}")
    print(f"过滤统计表：{resolve_project_path(args.filtering_csv)}")
    if all_change_rows:
        print(f"投票变化样本：{resolve_project_path(args.change_csv)}")
    else:
        print("投票变化样本：无")


if __name__ == "__main__":
    main()
