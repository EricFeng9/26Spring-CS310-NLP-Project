#!/usr/bin/env python
"""生成任务B的诊断分析表。"""

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
METHOD_ORDER = ["original_sc", "normalized_filtered_vote"]


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


def normalized_filtered_prediction(dataset_key: str, row: dict) -> tuple[str, dict[str, int]]:
    """计算任务B推荐方法的预测和票数。"""
    path_views = build_path_views(dataset_key, row)
    answers = [view["normalized_answer"] for view in path_views if view["normalized_valid"]]
    return stable_majority_vote(answers)


def majority_bucket(vote_counts: dict[str, int], num_paths: int) -> str:
    """将多数票占比分桶，便于分析投票置信度。"""
    if not vote_counts:
        return "empty"
    share = max(vote_counts.values()) / num_paths
    if share <= 1 / 3:
        return "one_third"
    if share <= 2 / 3:
        return "two_thirds"
    return "unanimous"


def build_oracle_rows(jsonl_paths: list[Path]) -> list[dict]:
    """统计如果至少一条路径包含正确答案时的 oracle 上界。"""
    result_rows: list[dict] = []
    for jsonl_path in jsonl_paths:
        model_key, dataset_key = split_model_dataset(jsonl_path)
        rows = load_jsonl(jsonl_path)
        sample_count = len(rows)
        original_correct_count = 0
        improved_correct_count = 0
        raw_oracle_count = 0
        normalized_oracle_count = 0

        for row in rows:
            gold_answer = str(row["gold_answer"]).strip()
            original_correct_count += int(bool(row["correct"]))
            improved_prediction, _ = normalized_filtered_prediction(dataset_key, row)
            improved_correct_count += int(improved_prediction == gold_answer)

            raw_answers = [str(path.get("answer", "")).strip() for path in row["paths"]]
            normalized_answers = [
                view["normalized_answer"]
                for view in build_path_views(dataset_key, row)
                if view["normalized_valid"]
            ]
            raw_oracle_count += int(gold_answer in raw_answers)
            normalized_oracle_count += int(gold_answer in normalized_answers)

        original_accuracy = original_correct_count / sample_count
        improved_accuracy = improved_correct_count / sample_count
        raw_oracle_rate = raw_oracle_count / sample_count
        normalized_oracle_rate = normalized_oracle_count / sample_count
        result_rows.append(
            {
                "model_key": model_key,
                "dataset_key": dataset_key,
                "sample_count": sample_count,
                "original_accuracy": round(original_accuracy, 6),
                "normalized_filtered_accuracy": round(improved_accuracy, 6),
                "raw_oracle_rate": round(raw_oracle_rate, 6),
                "normalized_oracle_rate": round(normalized_oracle_rate, 6),
                "original_gap_to_normalized_oracle": round(normalized_oracle_rate - original_accuracy, 6),
                "improved_gap_to_normalized_oracle": round(normalized_oracle_rate - improved_accuracy, 6),
            }
        )
    return result_rows


def build_confidence_rows(jsonl_paths: list[Path]) -> list[dict]:
    """按多数票占比分桶统计准确率。"""
    grouped: dict[tuple[str, str, str, str], dict[str, int]] = {}
    for jsonl_path in jsonl_paths:
        model_key, dataset_key = split_model_dataset(jsonl_path)
        rows = load_jsonl(jsonl_path)
        for row in rows:
            gold_answer = str(row["gold_answer"]).strip()
            num_paths = len(row["paths"])
            method_votes = {
                "original_sc": (str(row["prediction"]).strip(), row["vote_counts"]),
                "normalized_filtered_vote": normalized_filtered_prediction(dataset_key, row),
            }
            for method, (prediction, vote_counts) in method_votes.items():
                bucket = majority_bucket(vote_counts, num_paths)
                key = (model_key, dataset_key, method, bucket)
                grouped.setdefault(key, {"sample_count": 0, "correct_count": 0})
                grouped[key]["sample_count"] += 1
                grouped[key]["correct_count"] += int(prediction == gold_answer)

    result_rows: list[dict] = []
    for (model_key, dataset_key, method, bucket), stats in sorted(grouped.items()):
        sample_count = stats["sample_count"]
        correct_count = stats["correct_count"]
        result_rows.append(
            {
                "model_key": model_key,
                "dataset_key": dataset_key,
                "method": method,
                "majority_bucket": bucket,
                "sample_count": sample_count,
                "correct_count": correct_count,
                "accuracy": round(correct_count / sample_count, 6),
            }
        )
    return result_rows


def build_transition_rows(jsonl_paths: list[Path]) -> list[dict]:
    """统计原始SC到推荐方法的正确性转移矩阵。"""
    result_rows: list[dict] = []
    for jsonl_path in jsonl_paths:
        model_key, dataset_key = split_model_dataset(jsonl_path)
        rows = load_jsonl(jsonl_path)
        counts = {
            "same_correct": 0,
            "same_wrong": 0,
            "wrong_to_correct": 0,
            "correct_to_wrong": 0,
            "prediction_changed": 0,
        }

        for row in rows:
            gold_answer = str(row["gold_answer"]).strip()
            original_prediction = str(row["prediction"]).strip()
            original_correct = original_prediction == gold_answer
            improved_prediction, _ = normalized_filtered_prediction(dataset_key, row)
            improved_correct = improved_prediction == gold_answer
            counts["prediction_changed"] += int(original_prediction != improved_prediction)

            if original_correct and improved_correct:
                counts["same_correct"] += 1
            elif (not original_correct) and (not improved_correct):
                counts["same_wrong"] += 1
            elif (not original_correct) and improved_correct:
                counts["wrong_to_correct"] += 1
            elif original_correct and (not improved_correct):
                counts["correct_to_wrong"] += 1

        sample_count = len(rows)
        result_rows.append(
            {
                "model_key": model_key,
                "dataset_key": dataset_key,
                "sample_count": sample_count,
                **counts,
                "net_correct_gain": counts["wrong_to_correct"] - counts["correct_to_wrong"],
                "prediction_changed_rate": round(counts["prediction_changed"] / sample_count, 6),
            }
        )
    return result_rows


def build_model_average_rows(oracle_rows: list[dict]) -> list[dict]:
    """按模型汇总原始SC、推荐方法和oracle上界。"""
    grouped: dict[str, list[dict]] = {}
    for row in oracle_rows:
        grouped.setdefault(row["model_key"], []).append(row)

    result_rows: list[dict] = []
    for model_key, rows in sorted(grouped.items()):
        result_rows.append(
            {
                "model_key": model_key,
                "dataset_count": len(rows),
                "mean_original_accuracy": round(
                    sum(float(row["original_accuracy"]) for row in rows) / len(rows),
                    6,
                ),
                "mean_normalized_filtered_accuracy": round(
                    sum(float(row["normalized_filtered_accuracy"]) for row in rows) / len(rows),
                    6,
                ),
                "mean_normalized_oracle_rate": round(
                    sum(float(row["normalized_oracle_rate"]) for row in rows) / len(rows),
                    6,
                ),
            }
        )
    return result_rows


def main() -> None:
    """脚本入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="output/raw/A/lightweight_100/self_consistency")
    parser.add_argument("--oracle-csv", default="output/log/B/oracle_path_analysis.csv")
    parser.add_argument("--confidence-csv", default="output/log/B/majority_confidence_buckets.csv")
    parser.add_argument("--transition-csv", default="output/log/B/original_to_improved_transition.csv")
    parser.add_argument("--model-average-csv", default="output/log/B/model_average_diagnostics.csv")
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

    oracle_rows = build_oracle_rows(jsonl_paths)
    confidence_rows = build_confidence_rows(jsonl_paths)
    transition_rows = build_transition_rows(jsonl_paths)
    model_average_rows = build_model_average_rows(oracle_rows)

    write_csv(resolve_project_path(args.oracle_csv), oracle_rows)
    write_csv(resolve_project_path(args.confidence_csv), confidence_rows)
    write_csv(resolve_project_path(args.transition_csv), transition_rows)
    write_csv(resolve_project_path(args.model_average_csv), model_average_rows)

    print("任务B诊断分析完成。")
    print(f"oracle 上界表：{resolve_project_path(args.oracle_csv)}")
    print(f"多数票置信度表：{resolve_project_path(args.confidence_csv)}")
    print(f"转移矩阵表：{resolve_project_path(args.transition_csv)}")
    print(f"模型平均诊断表：{resolve_project_path(args.model_average_csv)}")


if __name__ == "__main__":
    main()
