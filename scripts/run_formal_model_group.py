#!/usr/bin/env python
"""按模型分组运行正式 baseline，单次加载模型后串行跑完该模型的 formal 配置。"""

from __future__ import annotations

import argparse
import sys
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


def run_one_config(
    config_path: str,
    runner: LocalModelRunner,
    output_root: str | None,
    sample_size: int | None,
    sample_seed: int | None,
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
    result_name = f"{run_config['model_key']}_{run_config['dataset_key']}"
    log_dir = derive_log_dir_from_output_dir(run_config["output_dir"])
    log_path = log_dir / f"{result_name}.log"
    with tee_console_to_file(log_path):
        print(f"[formal_group] config={config_path}", flush=True)
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
        print(f"[formal_group] completed config={config_path}", flush=True)


def main() -> None:
    """脚本入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(FORMAL_CONFIGS))
    parser.add_argument("--output-root")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--sample-seed", type=int)
    args = parser.parse_args()

    config_paths = FORMAL_CONFIGS[args.model]
    first_run_config, model_config, prompt_config = load_core_configs(config_paths[0])
    runner = LocalModelRunner(model_config, prompt_config["system_prompt"])
    print(
        f"[formal_group] loaded shared runner for model={args.model} "
        f"configs={len(config_paths)} split={first_run_config['split']}"
        ,
        flush=True,
    )
    for config_path in config_paths:
        run_one_config(config_path, runner, args.output_root, args.sample_size, args.sample_seed)
    print(f"[formal_group] all configs completed for model={args.model}", flush=True)


if __name__ == "__main__":
    main()
