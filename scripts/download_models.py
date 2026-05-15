#!/usr/bin/env python
"""下载任务A所需模型。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_yaml, project_root


MODEL_CONFIG_DIR = project_root() / "configs" / "models"


def list_model_keys() -> list[str]:
    """返回全部模型键名。"""
    preferred_order = [
        "mistral_7b_instruct_v0_3",
        "olmo3_7b_instruct",
        "llama2_7b_hf",
        "phi3_5_mini_instruct",
        "qwen2_5_7b_instruct",
    ]
    return [model_key for model_key in preferred_order if (MODEL_CONFIG_DIR / f"{model_key}.yaml").exists()]


def has_complete_local_model(local_dir: Path) -> bool:
    """检查本地模型目录是否已经具备最基本的可加载文件。

    这里直接以 `config.json` 和至少一个 `*.safetensors` 文件作为完成标记。
    任务A当前目标是稳定复用已经下载好的本地模型，避免 smoke test 每次都重新访问远端。
    """
    if not local_dir.exists():
        return False
    if not (local_dir / "config.json").exists():
        return False
    if not list(local_dir.glob("*.safetensors")):
        return False
    return True


def download_one(model_key: str) -> None:
    """下载单个模型。"""
    config = load_yaml(MODEL_CONFIG_DIR / f"{model_key}.yaml")
    local_dir = project_root() / config["local_dir"]
    local_dir.mkdir(parents=True, exist_ok=True)
    if has_complete_local_model(local_dir):
        print(f"检测到本地模型已存在，跳过下载：{model_key}")
        return
    try:
        snapshot_download(
            repo_id=config["repo_id"],
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except HfHubHTTPError as error:
        if error.response is not None and error.response.status_code in {401, 403}:
            raise RuntimeError(
                "模型仓库需要 Hugging Face 授权访问。"
                f" 请先确认账号已获得 `{config['repo_id']}` 的访问权限，并配置可用的 HF token。"
            ) from error
        raise


def main() -> None:
    """脚本入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list_model_keys() + ["all"], default="all")
    args = parser.parse_args()

    targets = list_model_keys() if args.model == "all" else [args.model]
    for model_key in targets:
        print(f"开始下载模型：{model_key}")
        download_one(model_key)
        print(f"模型下载完成：{model_key}")


if __name__ == "__main__":
    main()
