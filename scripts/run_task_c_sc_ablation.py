#!/usr/bin/env python
"""Run Task C Self-Consistency parameter ablations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io.runtime_logging import tee_console_to_file
from src.models.runner import LocalModelRunner
from src.pipeline import run_self_consistency, save_results
from src.utils import load_yaml, resolve_project_path


DATASET_SPLITS = {
    "gsm8k": "test",
    "csqa": "validation",
    "mmlu": "validation",
}


def format_float_for_name(value: float) -> str:
    """Format a float for stable file names."""
    return str(value).replace(".", "p")


def build_run_config(
    model_key: str,
    dataset_key: str,
    num_paths: int,
    temperature: float,
    output_dir: str,
    sample_size: int,
    sample_seed: int,
    max_new_tokens: int,
    top_p: float,
) -> dict:
    """Build one SC run config for Task C."""
    run_config = {
        "mode": "self_consistency",
        "model_key": model_key,
        "dataset_key": dataset_key,
        "split": DATASET_SPLITS[dataset_key],
        "limit": None,
        "output_dir": output_dir,
        "strict_answer_parsing": False,
        "sample_size": sample_size,
        "sample_seed": sample_seed,
        "num_paths": num_paths,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True,
        "few_shot_examples_path": f"data/processed/fewshot/{dataset_key}_fewshot.jsonl",
    }
    return run_config


def result_name(model_key: str, dataset_key: str, num_paths: int, temperature: float) -> str:
    """Build a parameter-aware result name."""
    return f"{model_key}_{dataset_key}_n{num_paths}_t{format_float_for_name(temperature)}"


def result_paths(output_dir: str, name: str) -> tuple[Path, Path]:
    """Return JSONL and CSV result paths."""
    output_path = resolve_project_path(output_dir)
    return output_path / f"{name}.jsonl", output_path / f"{name}.csv"


def main() -> None:
    """Script entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2_5_1_5b_instruct")
    parser.add_argument("--datasets", nargs="+", default=["gsm8k", "csqa"], choices=sorted(DATASET_SPLITS))
    parser.add_argument("--num-paths", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--output-dir", default="output/raw/C/self_consistency")
    parser.add_argument("--log-dir", default="output/log/C/runs")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    model_config = load_yaml(resolve_project_path(f"configs/models/{args.model}.yaml"))
    prompt_config = load_yaml(resolve_project_path("configs/prompts/base_prompt.yaml"))
    runner = LocalModelRunner(model_config, prompt_config["system_prompt"])
    print(
        f"[taskC] loaded model={args.model} datasets={args.datasets} "
        f"num_paths={args.num_paths} temperature={args.temperature}",
        flush=True,
    )

    for dataset_key in args.datasets:
        name = result_name(args.model, dataset_key, args.num_paths, args.temperature)
        jsonl_path, csv_path = result_paths(args.output_dir, name)
        if not args.no_skip_existing and jsonl_path.exists() and csv_path.exists():
            print(f"[taskC] skip existing {jsonl_path}", flush=True)
            continue

        log_path = resolve_project_path(args.log_dir) / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        run_config = build_run_config(
            model_key=args.model,
            dataset_key=dataset_key,
            num_paths=args.num_paths,
            temperature=args.temperature,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            sample_seed=args.sample_seed,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
        )
        with tee_console_to_file(log_path):
            print(f"[taskC] start name={name} log_path={log_path}", flush=True)
            rows = run_self_consistency(run_config, model_config, prompt_config, runner=runner)
            save_results(args.output_dir, name, rows)
            print(f"[taskC] completed name={name} jsonl={jsonl_path}", flush=True)


if __name__ == "__main__":
    main()
