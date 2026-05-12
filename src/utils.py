"""项目级通用工具函数。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """返回项目根目录。"""
    return Path(__file__).resolve().parent.parent


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置并返回字典。"""
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_project_path(path: str | Path) -> Path:
    """将相对路径解析为相对于项目根目录的绝对路径。"""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate
