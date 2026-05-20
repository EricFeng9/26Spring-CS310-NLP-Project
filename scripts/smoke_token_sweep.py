#!/usr/bin/env python
"""轻量模型 token 上限冒烟测试。

该脚本覆盖任务A的模型、数据集和推理模式，用少量样本评估不同
max_new_tokens 的耗时、答案格式遵循和疑似截断情况。
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_formal_model_group import DATASET_SPLITS, LIGHTWEIGHT_MODELS, _build_lightweight_run_config
from src.data.loaders import load_jsonl_samples, load_processed_split
from src.models.runner import LocalModelRunner
from src.pipeline import load_core_configs
from src.prompts.builders import build_few_shot_prompt, build_self_consistency_prompt, build_zero_shot_prompt
from src.sc.answering import extract_answer
from src.utils import load_yaml, resolve_project_path


DEFAULT_MODE_TOKEN_GRID = {
    "zero_shot": [128, 192, 256],
    "few_shot": [128, 192, 256],
    "self_consistency": [192, 256, 384],
}

MODE_ORDER = ["zero_shot", "few_shot", "self_consistency"]
DATASET_ORDER = ["gsm8k", "csqa", "mmlu"]
MODEL_ORDER = ["qwen2_5_0_5b_instruct", "qwen2_5_1_5b_instruct", "tinyllama_1_1b_chat"]


def _looks_truncated(text: str, max_new_tokens: int) -> bool:
    """用文本形态粗略判断输出是否疑似因为 token 上限被截断。"""
    stripped = text.strip()
    if "The answer is" in stripped:
        return False
    if not stripped:
        return True
    if stripped.endswith((".", "!", "?", ")")):
        return max_new_tokens <= 128
    return True


def _answer_quality(dataset_key: str, text: str, choices: list[str]) -> tuple[bool, bool, str]:
    """返回是否包含最终答案行、是否抽取为短答案、抽取结果。"""
    has_final_answer = bool(re.search(r"The answer is", text, flags=re.IGNORECASE))
    prediction = extract_answer(dataset_key, text, strict=False, choices=choices)
    short_answer = False
    if dataset_key == "gsm8k":
        short_answer = bool(re.fullmatch(r"-?\d+(?:\.\d+)?", prediction.strip()))
    elif dataset_key in {"csqa", "mmlu"}:
        short_answer = bool(re.fullmatch(r"[A-E]", prediction.strip()))
    return has_final_answer, short_answer, prediction


def _build_prompt(mode: str, sample, prompt_config: dict, fewshot_examples: list) -> str:
    if mode == "zero_shot":
        return build_zero_shot_prompt(sample, prompt_config)
    if mode == "few_shot":
        return build_few_shot_prompt(sample, prompt_config, fewshot_examples)
    if mode == "self_consistency":
        return build_self_consistency_prompt(sample, prompt_config)
    raise ValueError(f"unknown mode: {mode}")


def _parse_token_grid(args: argparse.Namespace) -> dict[str, list[int]]:
    if args.quick:
        return {
            "zero_shot": [192, 256],
            "few_shot": [256],
            "self_consistency": [256],
        }
    return DEFAULT_MODE_TOKEN_GRID


def run_sweep(args: argparse.Namespace) -> list[dict]:
    """执行 token sweep 并返回逐组合汇总。"""
    _, _, prompt_config = load_core_configs("configs/runs/formal/zero_shot_mistral_7b_instruct_v0_3_gsm8k.yaml")
    rows: list[dict] = []
    token_grid = _parse_token_grid(args)
    for model_key in MODEL_ORDER:
        if model_key not in LIGHTWEIGHT_MODELS:
            continue
        model_config = load_yaml(resolve_project_path(f"configs/models/{model_key}.yaml"))
        runner = LocalModelRunner(model_config, prompt_config["system_prompt"])
        for mode in MODE_ORDER:
            for dataset_key in DATASET_ORDER:
                samples = load_processed_split(
                    dataset_key,
                    DATASET_SPLITS[dataset_key],
                    sample_size=args.samples_per_dataset,
                    sample_seed=args.sample_seed,
                )
                fewshot_examples = []
                if mode == "few_shot":
                    fewshot_examples = load_jsonl_samples(f"data/processed/fewshot/{dataset_key}_fewshot.jsonl")
                for max_new_tokens in token_grid[mode]:
                    run_config = _build_lightweight_run_config(
                        model_key,
                        mode,
                        dataset_key,
                        output_root=str(args.output_root),
                        sample_size=args.samples_per_dataset,
                        sample_seed=args.sample_seed,
                    )
                    run_config["max_new_tokens"] = max_new_tokens
                    if mode == "self_consistency":
                        run_config["num_paths"] = args.num_paths

                    total_generations = 0
                    total_seconds = 0.0
                    final_answer_hits = 0
                    short_answer_hits = 0
                    truncated_hits = 0
                    correct_hits = 0
                    for sample in samples:
                        prompt = _build_prompt(mode, sample, prompt_config, fewshot_examples)
                        repeats = args.num_paths if mode == "self_consistency" else 1
                        for _ in range(repeats):
                            started = time.perf_counter()
                            output = runner.generate(
                                prompt=prompt,
                                max_new_tokens=max_new_tokens,
                                temperature=run_config["temperature"],
                                top_p=run_config["top_p"],
                                do_sample=run_config["do_sample"],
                            )
                            elapsed = time.perf_counter() - started
                            has_final_answer, short_answer, prediction = _answer_quality(
                                dataset_key,
                                output.generated_text,
                                sample.choices,
                            )
                            total_generations += 1
                            total_seconds += elapsed
                            final_answer_hits += int(has_final_answer)
                            short_answer_hits += int(short_answer)
                            truncated_hits += int(_looks_truncated(output.generated_text, max_new_tokens))
                            correct_hits += int(prediction == sample.gold_answer)
                    row = {
                        "model": model_key,
                        "mode": mode,
                        "dataset": dataset_key,
                        "max_new_tokens": max_new_tokens,
                        "samples": len(samples),
                        "generations": total_generations,
                        "num_paths": args.num_paths if mode == "self_consistency" else 1,
                        "seconds_total": round(total_seconds, 3),
                        "seconds_per_generation": round(total_seconds / max(total_generations, 1), 3),
                        "final_answer_rate": round(final_answer_hits / max(total_generations, 1), 4),
                        "short_answer_rate": round(short_answer_hits / max(total_generations, 1), 4),
                        "truncated_rate": round(truncated_hits / max(total_generations, 1), 4),
                        "sample_accuracy": round(correct_hits / max(total_generations, 1), 4),
                    }
                    rows.append(row)
                    print(row, flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-dataset", type=int, default=3)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--num-paths", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("output/raw/A/token_sweep_smoke"))
    parser.add_argument("--summary-csv", type=Path, default=Path("output/log/A/token_sweep_smoke/summary.csv"))
    args = parser.parse_args()

    rows = run_sweep(args)
    summary_path = resolve_project_path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[token_sweep] wrote summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
