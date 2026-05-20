#!/usr/bin/env python
"""按模型分组运行正式 baseline，单次加载模型后串行跑完该模型的 formal 配置。"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io.runtime_logging import derive_log_dir_from_output_dir, tee_console_to_file
from src.models.runner import LocalModelRunner
from src.pipeline import (
    load_core_configs,
    run_few_shot,
    run_self_consistency,
    run_zero_shot,
    save_results,
)
from src.utils import load_yaml, resolve_project_path


FORMAL_CONFIGS: dict[str, list[str]] = {
    "mistral_7b_instruct_v0_3": [
        "configs/runs/formal/zero_shot_mistral_7b_instruct_v0_3_gsm8k.yaml",
        "configs/runs/formal/zero_shot_mistral_7b_instruct_v0_3_csqa.yaml",
        "configs/runs/formal/zero_shot_mistral_7b_instruct_v0_3_mmlu.yaml",
        "configs/runs/formal/few_shot_mistral_7b_instruct_v0_3_gsm8k.yaml",
        "configs/runs/formal/few_shot_mistral_7b_instruct_v0_3_csqa.yaml",
        "configs/runs/formal/few_shot_mistral_7b_instruct_v0_3_mmlu.yaml",
        "configs/runs/formal/self_consistency_mistral_7b_instruct_v0_3_gsm8k.yaml",
        "configs/runs/formal/self_consistency_mistral_7b_instruct_v0_3_csqa.yaml",
        "configs/runs/formal/self_consistency_mistral_7b_instruct_v0_3_mmlu.yaml",
    ],
    "olmo3_7b_instruct": [
        "configs/runs/formal/zero_shot_olmo3_7b_instruct_gsm8k.yaml",
        "configs/runs/formal/zero_shot_olmo3_7b_instruct_csqa.yaml",
        "configs/runs/formal/zero_shot_olmo3_7b_instruct_mmlu.yaml",
        "configs/runs/formal/few_shot_olmo3_7b_instruct_gsm8k.yaml",
        "configs/runs/formal/few_shot_olmo3_7b_instruct_csqa.yaml",
        "configs/runs/formal/few_shot_olmo3_7b_instruct_mmlu.yaml",
        "configs/runs/formal/self_consistency_olmo3_7b_instruct_gsm8k.yaml",
        "configs/runs/formal/self_consistency_olmo3_7b_instruct_csqa.yaml",
        "configs/runs/formal/self_consistency_olmo3_7b_instruct_mmlu.yaml",
    ],
    "llama2_7b_hf": [
        "configs/runs/formal/zero_shot_llama2_7b_hf_gsm8k.yaml",
        "configs/runs/formal/zero_shot_llama2_7b_hf_csqa.yaml",
        "configs/runs/formal/zero_shot_llama2_7b_hf_mmlu.yaml",
        "configs/runs/formal/few_shot_llama2_7b_hf_gsm8k.yaml",
        "configs/runs/formal/few_shot_llama2_7b_hf_csqa.yaml",
        "configs/runs/formal/few_shot_llama2_7b_hf_mmlu.yaml",
        "configs/runs/formal/self_consistency_llama2_7b_hf_gsm8k.yaml",
        "configs/runs/formal/self_consistency_llama2_7b_hf_csqa.yaml",
        "configs/runs/formal/self_consistency_llama2_7b_hf_mmlu.yaml",
    ],
}

LIGHTWEIGHT_MODELS = {
    "qwen2_5_0_5b_instruct",
    "qwen2_5_1_5b_instruct",
    "tinyllama_1_1b_chat",
}

DATASET_SPLITS = {
    "gsm8k": "test",
    "csqa": "validation",
    "mmlu": "test",
}

MODE_DEFAULTS = {
    "zero_shot": {
        "max_new_tokens": 256,
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
    },
    "few_shot": {
        "max_new_tokens": 256,
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
    },
    "self_consistency": {
        "max_new_tokens": 384,
        "temperature": 0.7,
        "top_p": 0.95,
        "do_sample": True,
        "num_paths": 3,
    },
}


def _build_lightweight_run_config(
    model_key: str,
    mode: str,
    dataset_key: str,
    output_root: str | None,
    sample_size: int | None,
    sample_seed: int | None,
) -> dict:
    """动态生成轻量模型 formal 配置，避免复制 27 个 YAML。"""
    base_output_root = output_root or "output/raw/A/lightweight_100"
    run_config = {
        "mode": mode,
        "model_key": model_key,
        "dataset_key": dataset_key,
        "split": DATASET_SPLITS[dataset_key],
        "limit": None,
        "output_dir": str(Path(base_output_root) / mode),
        "strict_answer_parsing": False,
        "sample_size": sample_size if sample_size is not None else 100,
        "sample_seed": sample_seed if sample_seed is not None else 42,
        **MODE_DEFAULTS[mode],
    }
    if mode == "few_shot":
        run_config["few_shot_examples_path"] = f"data/processed/fewshot/{dataset_key}_fewshot.jsonl"
    return run_config


def _result_paths(run_config: dict) -> tuple[Path, Path]:
    """返回单个配置的 JSONL/CSV 结果路径。"""
    output_dir = PROJECT_ROOT / run_config["output_dir"]
    result_name = f"{run_config['model_key']}_{run_config['dataset_key']}"
    return output_dir / f"{result_name}.jsonl", output_dir / f"{result_name}.csv"


@contextmanager
def _config_lock(lock_path: Path):
    """用原子目录锁避免多个 Slurm 作业同时写同一份结果。"""
    try:
        lock_path.mkdir(parents=True)
    except FileExistsError:
        yield False
        return
    try:
        yield True
    finally:
        lock_path.rmdir()


def run_one_config(
    config_path: str,
    runner: LocalModelRunner,
    output_root: str | None,
    sample_size: int | None,
    sample_seed: int | None,
    skip_existing: bool,
) -> None:
    """运行单个 formal 配置。"""
    run_config, model_config, prompt_config = load_core_configs(config_path)
    if output_root is not None:
        run_config["output_dir"] = str(Path(output_root) / run_config["mode"])
    if sample_size is not None:
        run_config["sample_size"] = sample_size
        run_config["limit"] = None
    if sample_seed is not None:
        run_config["sample_seed"] = sample_seed
    jsonl_path, csv_path = _result_paths(run_config)
    if skip_existing and jsonl_path.exists() and csv_path.exists():
        print(
            f"[formal_group] skip existing config={config_path} jsonl={jsonl_path} csv={csv_path}",
            flush=True,
        )
        return
    lock_path = jsonl_path.with_suffix(jsonl_path.suffix + ".lock")
    with _config_lock(lock_path) as acquired:
        if not acquired:
            print(f"[formal_group] skip locked config={config_path} lock={lock_path}", flush=True)
            return
        _run_loaded_config(config_path, run_config, model_config, prompt_config, runner)


def _run_loaded_config(
    config_name: str,
    run_config: dict,
    model_config: dict,
    prompt_config: dict,
    runner: LocalModelRunner,
) -> None:
    """运行已经完成加载和并发保护的配置。"""
    result_name = f"{run_config['model_key']}_{run_config['dataset_key']}"
    log_dir = derive_log_dir_from_output_dir(run_config["output_dir"])
    log_path = log_dir / f"{result_name}.log"
    with tee_console_to_file(log_path):
        print(f"[formal_group] config={config_name}", flush=True)
        print(f"[formal_group] log_path={log_path}", flush=True)
        if run_config["mode"] in {"zero_shot", "zero_shot_cot"}:
            rows = run_zero_shot(run_config, model_config, prompt_config, runner=runner)
        elif run_config["mode"] in {"few_shot", "few_shot_cot"}:
            rows = run_few_shot(run_config, model_config, prompt_config, runner=runner)
        elif run_config["mode"] == "self_consistency":
            rows = run_self_consistency(run_config, model_config, prompt_config, runner=runner)
        else:
            raise ValueError(f"未知 mode: {run_config['mode']}")
        save_results(run_config["output_dir"], result_name, rows)
        print(f"[formal_group] completed config={config_name}", flush=True)


def run_one_lightweight_config(
    run_config: dict,
    runner: LocalModelRunner,
    model_config: dict,
    prompt_config: dict,
    skip_existing: bool,
) -> None:
    """运行一个动态生成的轻量模型 formal 配置。"""
    jsonl_path, csv_path = _result_paths(run_config)
    config_name = f"dynamic:{run_config['mode']}:{run_config['model_key']}:{run_config['dataset_key']}"
    if skip_existing and jsonl_path.exists() and csv_path.exists():
        print(
            f"[formal_group] skip existing config={config_name} jsonl={jsonl_path} csv={csv_path}",
            flush=True,
        )
        return
    lock_path = jsonl_path.with_suffix(jsonl_path.suffix + ".lock")
    with _config_lock(lock_path) as acquired:
        if not acquired:
            print(f"[formal_group] skip locked config={config_name} lock={lock_path}", flush=True)
            return
        _run_loaded_config(config_name, run_config, model_config, prompt_config, runner)


def main() -> None:
    """脚本入口。"""
    parser = argparse.ArgumentParser()
    model_choices = sorted(set(FORMAL_CONFIGS) | LIGHTWEIGHT_MODELS)
    parser.add_argument("--model", required=True, choices=model_choices)
    parser.add_argument("--output-root")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument("--mode", choices=["zero_shot", "few_shot", "self_consistency"])
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    if args.model in LIGHTWEIGHT_MODELS:
        if args.mode is None:
            raise ValueError("轻量模型运行必须显式传入 --mode")
        run_configs = [
            _build_lightweight_run_config(
                args.model,
                args.mode,
                dataset_key,
                args.output_root,
                args.sample_size,
                args.sample_seed,
            )
            for dataset_key in ("gsm8k", "csqa", "mmlu")
        ]
        model_config = load_yaml(resolve_project_path(f"configs/models/{args.model}.yaml"))
        prompt_config = load_core_configs("configs/runs/formal/zero_shot_mistral_7b_instruct_v0_3_gsm8k.yaml")[2]
        runner = LocalModelRunner(model_config, prompt_config["system_prompt"])
        print(
            f"[formal_group] loaded shared runner for model={args.model} "
            f"configs={len(run_configs)} mode={args.mode}",
            flush=True,
        )
        for run_config in run_configs:
            run_one_lightweight_config(
                run_config,
                runner,
                model_config,
                prompt_config,
                skip_existing=not args.no_skip_existing,
            )
        print(f"[formal_group] all configs completed for model={args.model}", flush=True)
        return

    config_paths = FORMAL_CONFIGS[args.model]
    if args.mode is not None:
        config_paths = [path for path in config_paths if load_core_configs(path)[0]["mode"] == args.mode]
        if not config_paths:
            raise ValueError(f"模型 {args.model} 没有 mode={args.mode} 的 formal 配置")
    first_run_config, model_config, prompt_config = load_core_configs(config_paths[0])
    runner = LocalModelRunner(model_config, prompt_config["system_prompt"])
    print(
        f"[formal_group] loaded shared runner for model={args.model} "
        f"configs={len(config_paths)} split={first_run_config['split']}"
        ,
        flush=True,
    )
    for config_path in config_paths:
        run_one_config(
            config_path,
            runner,
            args.output_root,
            args.sample_size,
            args.sample_seed,
            skip_existing=not args.no_skip_existing,
        )
    print(f"[formal_group] all configs completed for model={args.model}", flush=True)


if __name__ == "__main__":
    main()
