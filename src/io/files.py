"""统一写出 JSONL 与 CSV。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def ensure_parent(path: Path) -> None:
    """保证目标文件的父目录存在。"""
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出 JSONL 文件。"""
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出 CSV 文件。"""
    ensure_parent(path)
    if not rows:
        raise ValueError("CSV 写出失败，因为输入结果为空。")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
