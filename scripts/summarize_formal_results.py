#!/usr/bin/env python
"""汇总任务A正式实验结果并生成最终表格与论文对照说明。"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io.files import write_csv
from src.io.runtime_logging import tee_console_to_file
from src.utils import load_yaml, resolve_project_path


@dataclass
class ExperimentRecord:
    """保存单个正式实验配置与结果路径。"""

    config_path: Path
    mode: str
    model_key: str
    dataset_key: str
    split: str
    output_dir: Path
    jsonl_path: Path
    csv_path: Path
    num_paths: int | None


def load_formal_experiment_records(config_dir: Path) -> list[ExperimentRecord]:
    """从正式实验配置目录加载全部预期实验记录。

    这里直接把配置文件作为事实来源，原因很直接：
    1. 正式实验配置的命名和数量已经固定。
    2. 统计脚本必须对缺失结果直接报错，不能靠扫描输出目录“猜”有哪些实验。
    """
    records: list[ExperimentRecord] = []
    for config_path in sorted(config_dir.glob("*.yaml")):
        payload = load_yaml(config_path)
        output_dir = resolve_project_path(payload["output_dir"])
        result_name = f"{payload['model_key']}_{payload['dataset_key']}"
        records.append(
            ExperimentRecord(
                config_path=config_path,
                mode=str(payload["mode"]),
                model_key=str(payload["model_key"]),
                dataset_key=str(payload["dataset_key"]),
                split=str(payload["split"]),
                output_dir=output_dir,
                jsonl_path=output_dir / f"{result_name}.jsonl",
                csv_path=output_dir / f"{result_name}.csv",
                num_paths=int(payload["num_paths"]) if "num_paths" in payload else None,
            )
        )
    if not records:
        raise ValueError(f"正式实验配置目录为空：{config_dir}")
    return records


def load_jsonl_rows(path: Path) -> list[dict]:
    """读取 JSONL 结果文件。"""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))
    return rows


def ensure_result_files_exist(records: list[ExperimentRecord]) -> None:
    """检查所有正式实验结果文件是否存在。

    这里不做兜底跳过，原因是缺任何一个文件都会让最终表格口径不完整。
    """
    missing_paths: list[str] = []
    for record in records:
        if not record.jsonl_path.exists():
            missing_paths.append(str(record.jsonl_path))
        if not record.csv_path.exists():
            missing_paths.append(str(record.csv_path))
    if missing_paths:
        missing_message = "\n".join(missing_paths)
        raise FileNotFoundError(f"正式实验结果缺失，请先补跑这些文件：\n{missing_message}")


def build_baseline_main_rows(records: list[ExperimentRecord]) -> list[dict]:
    """生成正式 baseline 主结果表。"""
    rows: list[dict] = []
    for record in records:
        result_rows = load_jsonl_rows(record.jsonl_path)
        sample_count = len(result_rows)
        if sample_count == 0:
            raise ValueError(f"结果文件为空：{record.jsonl_path}")
        correct_count = sum(1 for row in result_rows if bool(row["correct"]))
        rows.append(
            {
                "model_key": record.model_key,
                "mode": record.mode,
                "dataset_key": record.dataset_key,
                "split": record.split,
                "sample_count": sample_count,
                "correct_count": correct_count,
                "accuracy": round(correct_count / sample_count, 6),
                "jsonl_path": str(record.jsonl_path),
                "csv_path": str(record.csv_path),
            }
        )
    return rows


def build_sc_path_stat_rows(records: list[ExperimentRecord]) -> list[dict]:
    """生成 Self-Consistency 路径统计表。"""
    rows: list[dict] = []
    sc_records = [record for record in records if record.mode == "self_consistency"]
    for record in sc_records:
        result_rows = load_jsonl_rows(record.jsonl_path)
        unique_answer_counts: list[int] = []
        majority_shares: list[float] = []
        total_paths = 0
        extracted_paths = 0

        for row in result_rows:
            paths = row["paths"]
            vote_counts = row["vote_counts"]
            total_paths += len(paths)
            extracted_paths += sum(1 for path in paths if str(path["answer"]).strip() != "")
            unique_answer_counts.append(len(vote_counts))
            majority_vote = max(int(count) for count in vote_counts.values())
            majority_shares.append(majority_vote / len(paths))

        if total_paths == 0:
            raise ValueError(f"Self-Consistency 结果没有路径：{record.jsonl_path}")

        rows.append(
            {
                "model_key": record.model_key,
                "dataset_key": record.dataset_key,
                "split": record.split,
                "num_paths": record.num_paths,
                "sample_count": len(result_rows),
                "answer_extraction_success_rate": round(extracted_paths / total_paths, 6),
                "mean_unique_answers": round(sum(unique_answer_counts) / len(unique_answer_counts), 6),
                "mean_majority_share": round(sum(majority_shares) / len(majority_shares), 6),
            }
        )
    return rows


def build_cot_vs_sc_rows(baseline_main_rows: list[dict]) -> list[dict]:
    """生成同模型同数据集下的单条 CoT 与 SC 对照表。

    这里把 `zero_shot` 和 `few_shot` 都视为单条 CoT。
    每个单条 CoT 配置都与同模型同数据集的 `self_consistency` 做一条配对记录。
    """
    index: dict[tuple[str, str, str], dict] = {}
    for row in baseline_main_rows:
        index[(row["model_key"], row["dataset_key"], row["mode"])] = row

    rows: list[dict] = []
    model_keys = sorted({row["model_key"] for row in baseline_main_rows})
    dataset_keys = sorted({row["dataset_key"] for row in baseline_main_rows})
    for model_key in model_keys:
        for dataset_key in dataset_keys:
            sc_row = index[(model_key, dataset_key, "self_consistency")]
            for cot_mode in ["zero_shot", "few_shot"]:
                cot_row = index[(model_key, dataset_key, cot_mode)]
                rows.append(
                    {
                        "model_key": model_key,
                        "dataset_key": dataset_key,
                        "cot_mode": cot_mode,
                        "cot_accuracy": cot_row["accuracy"],
                        "sc_accuracy": sc_row["accuracy"],
                        "absolute_gain": round(sc_row["accuracy"] - cot_row["accuracy"], 6),
                    }
                )
    return rows


def build_paper_comparison_markdown(
    baseline_main_rows: list[dict],
    cot_vs_sc_rows: list[dict],
    output_path: Path,
) -> None:
    """生成论文对照说明。

    输出内容固定包括：
    1. 方法趋势对照
    2. 接近论文风格的平均结果表
    """
    comparison_outcomes = {"improved": 0, "equal": 0, "worse": 0}
    for row in cot_vs_sc_rows:
        gain = float(row["absolute_gain"])
        if gain > 0:
            comparison_outcomes["improved"] += 1
        elif gain < 0:
            comparison_outcomes["worse"] += 1
        else:
            comparison_outcomes["equal"] += 1

    averages: dict[tuple[str, str], list[float]] = {}
    for row in baseline_main_rows:
        averages.setdefault((row["mode"], row["dataset_key"]), []).append(float(row["accuracy"]))

    mode_order = ["zero_shot", "few_shot", "self_consistency"]
    dataset_order = ["gsm8k", "csqa", "mmlu"]
    average_table_rows: list[str] = []
    for mode in mode_order:
        values = []
        for dataset in dataset_order:
            dataset_values = averages[(mode, dataset)]
            values.append(f"{sum(dataset_values) / len(dataset_values):.4f}")
        average_table_rows.append(f"| {mode} | {' | '.join(values)} |")

    lines = [
        "# 任务A论文对照说明",
        "",
        "## 1. 核心参考论文",
        "- Wang et al. *Self-Consistency Improves Chain of Thought Reasoning*",
        "",
        "## 2. 对照边界说明",
        "- 本项目复现的是论文方法：单条 CoT 与 Self-Consistency 的对照。",
        "- 本项目没有复刻原论文的同模型、同数据集、同超参数数值结果。",
        "- 本项目当前正式实验模型为 `Mistral / Olmo / Llama2`，与原论文大模型设置不同。",
        "",
        "## 3. 方法趋势对照",
        "- 论文核心结论：Self-Consistency 通常优于单条 CoT。",
        f"- 本项目在 {len(cot_vs_sc_rows)} 组同模型同数据集对照中：",
        f"  - 提升：{comparison_outcomes['improved']}",
        f"  - 持平：{comparison_outcomes['equal']}",
        f"  - 下降：{comparison_outcomes['worse']}",
        "",
        "结论判定规则：",
        "- 如果提升组数占多数，则记为“趋势一致”。",
        "- 如果提升和下降混合明显，则记为“部分一致”。",
        "- 如果下降占多数，则记为“不一致”。",
        "",
        "## 4. 接近论文风格的平均结果表",
        "",
        "| method | gsm8k | csqa | mmlu |",
        "| --- | --- | --- | --- |",
        *average_table_rows,
        "",
        "## 5. 可能原因",
        "- 当前模型规模明显小于原论文主实验模型。",
        "- 当前数据集口径包含 `MMLU`，与原论文主 benchmark 组合不同。",
        "- 第三个模型使用的是 `Llama-2 base`，推理能力弱于 instruct 模型。",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """脚本入口。"""
    formal_config_dir = resolve_project_path("configs/runs/formal")
    final_tables_dir = resolve_project_path("output/log/A/final_tables")
    final_reports_dir = resolve_project_path("output/log/A/final_reports")
    summary_log_path = resolve_project_path("output/log/A/final_reports/summarize_formal_results.log")

    with tee_console_to_file(summary_log_path):
        print(f"[formal_summary] config_dir={formal_config_dir}")
        print(f"[formal_summary] summary_log_path={summary_log_path}")

        records = load_formal_experiment_records(formal_config_dir)
        ensure_result_files_exist(records)

        baseline_main_rows = build_baseline_main_rows(records)
        cot_vs_sc_rows = build_cot_vs_sc_rows(baseline_main_rows)
        sc_path_stat_rows = build_sc_path_stat_rows(records)

        write_csv(final_tables_dir / "baseline_main_results.csv", baseline_main_rows)
        write_csv(final_tables_dir / "cot_vs_sc_comparison.csv", cot_vs_sc_rows)
        write_csv(final_tables_dir / "sc_path_stats.csv", sc_path_stat_rows)
        build_paper_comparison_markdown(
            baseline_main_rows=baseline_main_rows,
            cot_vs_sc_rows=cot_vs_sc_rows,
            output_path=final_reports_dir / "paper_comparison.md",
        )

        print("正式实验结果汇总完成。")
        print(f"主结果表：{final_tables_dir / 'baseline_main_results.csv'}")
        print(f"CoT vs SC 对照表：{final_tables_dir / 'cot_vs_sc_comparison.csv'}")
        print(f"SC 路径统计表：{final_tables_dir / 'sc_path_stats.csv'}")
        print(f"论文对照说明：{final_reports_dir / 'paper_comparison.md'}")


if __name__ == "__main__":
    main()
