#!/usr/bin/env python
"""运行 Zero-shot CoT。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io.runtime_logging import derive_log_dir_from_output_dir, tee_console_to_file
from src.pipeline import load_core_configs, run_zero_shot, save_results


def main() -> None:
    """脚本入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    run_config, model_config, prompt_config = load_core_configs(args.config)
    result_name = f"{run_config['model_key']}_{run_config['dataset_key']}"
    log_dir = derive_log_dir_from_output_dir(run_config["output_dir"])
    log_path = log_dir / f"{result_name}.log"
    with tee_console_to_file(log_path):
        print(f"[zero_shot] config={args.config}")
        print(f"[zero_shot] log_path={log_path}")
        results = run_zero_shot(run_config, model_config, prompt_config)
        save_results(run_config["output_dir"], result_name, results)
        print(f"Zero-shot 运行完成，结果已写入：{Path(run_config['output_dir'])}")


if __name__ == "__main__":
    main()
