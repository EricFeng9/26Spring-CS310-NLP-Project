#!/usr/bin/env python
"""Summarize Task C Self-Consistency parameter ablations."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io.files import write_csv, write_jsonl
from src.utils import resolve_project_path


BASELINE_NUM_PATHS = 3
BASELINE_TEMPERATURE = 0.7


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file."""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Empty JSONL file: {path}")
    return rows


def parse_c_filename(path: Path) -> tuple[str, str, int, float]:
    """Parse Task C result file name."""
    match = re.match(r"(.+)_(gsm8k|csqa|mmlu)_n(\d+)_t([0-9p]+)$", path.stem)
    if match is None:
        raise ValueError(f"Cannot parse Task C file name: {path.name}")
    model_key, dataset_key, num_paths, temperature = match.groups()
    return model_key, dataset_key, int(num_paths), float(temperature.replace("p", "."))


def parse_a_filename(path: Path) -> tuple[str, str]:
    """Parse Task A baseline file name."""
    model_key, dataset_key = path.stem.rsplit("_", 1)
    return model_key, dataset_key


def entropy(counter: Counter[str], total: int) -> float:
    """Compute answer entropy."""
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counter.values():
        probability = count / total
        if probability > 0:
            value -= probability * math.log2(probability)
    return value


def row_metrics(rows: list[dict], model_key: str, dataset_key: str, num_paths: int, temperature: float, source: str) -> dict:
    """Compute aggregate metrics for one result file."""
    sample_count = len(rows)
    correct_count = sum(1 for row in rows if bool(row.get("correct")))
    majority_shares: list[float] = []
    unique_answer_counts: list[int] = []
    entropies: list[float] = []
    tie_count = 0
    invalid_answer_count = 0
    total_paths = 0

    for row in rows:
        answers = [str(path.get("answer", "")).strip() for path in row.get("paths", [])]
        if not answers and "prediction" in row:
            answers = [str(row.get("prediction", "")).strip()]
        answer_count = len(answers)
        total_paths += answer_count
        counter = Counter(answers)
        if counter:
            max_count = max(counter.values())
            majority_shares.append(max_count / max(answer_count, 1))
            unique_answer_counts.append(len(counter))
            entropies.append(entropy(counter, answer_count))
            tie_count += int(sum(1 for value in counter.values() if value == max_count) > 1)
        else:
            majority_shares.append(0.0)
            unique_answer_counts.append(0)
            entropies.append(0.0)
        invalid_answer_count += sum(1 for answer in answers if answer == "")

    accuracy = correct_count / sample_count
    relative_path_cost = num_paths / BASELINE_NUM_PATHS
    return {
        "model_key": model_key,
        "dataset_key": dataset_key,
        "num_paths": num_paths,
        "temperature": temperature,
        "source": source,
        "sample_count": sample_count,
        "correct_count": correct_count,
        "accuracy": round(accuracy, 6),
        "total_paths": total_paths,
        "relative_path_cost": round(relative_path_cost, 6),
        "mean_majority_share": round(sum(majority_shares) / sample_count, 6),
        "mean_unique_answers": round(sum(unique_answer_counts) / sample_count, 6),
        "mean_answer_entropy": round(sum(entropies) / sample_count, 6),
        "tie_rate": round(tie_count / sample_count, 6),
        "invalid_answer_rate": round(invalid_answer_count / max(total_paths, 1), 6),
        "answer_extraction_success_rate": round(1 - invalid_answer_count / max(total_paths, 1), 6),
    }


def attach_baseline_gains(rows: list[dict]) -> list[dict]:
    """Attach accuracy gains relative to N=3, T=0.7 for each model/dataset."""
    baselines: dict[tuple[str, str], float] = {}
    for row in rows:
        if row["num_paths"] == BASELINE_NUM_PATHS and float(row["temperature"]) == BASELINE_TEMPERATURE:
            baselines[(row["model_key"], row["dataset_key"])] = float(row["accuracy"])

    enriched: list[dict] = []
    for row in rows:
        baseline_accuracy = baselines.get((row["model_key"], row["dataset_key"]))
        new_row = dict(row)
        if baseline_accuracy is None:
            new_row["baseline_accuracy"] = ""
            new_row["accuracy_gain_vs_A_sc"] = ""
            new_row["gain_per_extra_path"] = ""
        else:
            gain = float(row["accuracy"]) - baseline_accuracy
            new_row["baseline_accuracy"] = round(baseline_accuracy, 6)
            new_row["accuracy_gain_vs_A_sc"] = round(gain, 6)
            if int(row["num_paths"]) > BASELINE_NUM_PATHS:
                new_row["gain_per_extra_path"] = round(gain / (int(row["num_paths"]) - BASELINE_NUM_PATHS), 6)
            else:
                new_row["gain_per_extra_path"] = ""
        enriched.append(new_row)
    return enriched


def collect_rows(args: argparse.Namespace) -> list[dict]:
    """Collect Task A baseline rows and Task C rows."""
    rows: list[dict] = []
    baseline_dir = resolve_project_path(args.baseline_dir)
    c_dir = resolve_project_path(args.c_dir)
    selected_models = set(args.models)
    selected_datasets = set(args.datasets)

    for path in sorted(baseline_dir.glob("*.jsonl")):
        model_key, dataset_key = parse_a_filename(path)
        if model_key in selected_models and dataset_key in selected_datasets:
            rows.append(
                row_metrics(
                    load_jsonl(path),
                    model_key=model_key,
                    dataset_key=dataset_key,
                    num_paths=BASELINE_NUM_PATHS,
                    temperature=BASELINE_TEMPERATURE,
                    source="taskA_baseline",
                )
            )

    for path in sorted(c_dir.glob("*.jsonl")):
        model_key, dataset_key, num_paths, temperature = parse_c_filename(path)
        if model_key in selected_models and dataset_key in selected_datasets:
            rows.append(
                row_metrics(
                    load_jsonl(path),
                    model_key=model_key,
                    dataset_key=dataset_key,
                    num_paths=num_paths,
                    temperature=temperature,
                    source="taskC",
                )
            )
    return attach_baseline_gains(rows)


def filter_num_paths(rows: list[dict]) -> list[dict]:
    """Rows for num-path ablation at baseline temperature."""
    return [row for row in rows if float(row["temperature"]) == BASELINE_TEMPERATURE]


def filter_temperature(rows: list[dict]) -> list[dict]:
    """Rows for temperature ablation at baseline path count."""
    return [row for row in rows if int(row["num_paths"]) == BASELINE_NUM_PATHS]


def build_case_studies(args: argparse.Namespace, limit: int) -> list[dict]:
    """Build a small set of changed-prediction examples."""
    baseline_dir = resolve_project_path(args.baseline_dir)
    c_dir = resolve_project_path(args.c_dir)
    selected_models = set(args.models)
    selected_datasets = set(args.datasets)
    baseline_by_key: dict[tuple[str, str], dict[str, dict]] = {}

    for path in sorted(baseline_dir.glob("*.jsonl")):
        model_key, dataset_key = parse_a_filename(path)
        if model_key in selected_models and dataset_key in selected_datasets:
            baseline_by_key[(model_key, dataset_key)] = {row["id"]: row for row in load_jsonl(path)}

    examples: list[dict] = []
    for path in sorted(c_dir.glob("*.jsonl")):
        model_key, dataset_key, num_paths, temperature = parse_c_filename(path)
        if model_key not in selected_models or dataset_key not in selected_datasets:
            continue
        baseline_rows = baseline_by_key.get((model_key, dataset_key), {})
        for row in load_jsonl(path):
            base_row = baseline_rows.get(row["id"])
            if base_row is None or base_row.get("prediction") == row.get("prediction"):
                continue
            examples.append(
                {
                    "model_key": model_key,
                    "dataset_key": dataset_key,
                    "num_paths": num_paths,
                    "temperature": temperature,
                    "id": row["id"],
                    "question": row["question"],
                    "gold_answer": row["gold_answer"],
                    "baseline_prediction": base_row.get("prediction"),
                    "baseline_correct": base_row.get("correct"),
                    "new_prediction": row.get("prediction"),
                    "new_correct": row.get("correct"),
                    "baseline_vote_counts": base_row.get("vote_counts"),
                    "new_vote_counts": row.get("vote_counts"),
                }
            )
            if len(examples) >= limit:
                return examples
    return examples


def main() -> None:
    """Script entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", default="output/raw/A/lightweight_100/self_consistency")
    parser.add_argument("--c-dir", default="output/raw/C/self_consistency")
    parser.add_argument("--output-dir", default="output/log/C")
    parser.add_argument("--models", nargs="+", default=["qwen2_5_1_5b_instruct"])
    parser.add_argument("--datasets", nargs="+", default=["gsm8k", "csqa"])
    parser.add_argument("--case-limit", type=int, default=20)
    args = parser.parse_args()

    rows = collect_rows(args)
    if not rows:
        raise ValueError("No Task C or baseline rows matched the requested filters.")

    output_dir = resolve_project_path(args.output_dir)
    write_csv(output_dir / "sc_parameter_summary.csv", rows)
    write_csv(output_dir / "num_paths_ablation.csv", filter_num_paths(rows))
    write_csv(output_dir / "temperature_ablation.csv", filter_temperature(rows))
    write_csv(output_dir / "path_diversity_statistics.csv", rows)
    write_jsonl(output_dir / "case_study_examples.jsonl", build_case_studies(args, args.case_limit))

    print(f"Task C summary written to {output_dir}")


if __name__ == "__main__":
    main()
