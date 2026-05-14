"""任务A运行期日志工具。"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, TextIO

from src.io.files import ensure_parent
from src.utils import project_root, resolve_project_path


class TeeTextStream:
    """把控制台输出同时写到终端和日志文件。"""

    def __init__(self, terminal_stream: TextIO, log_stream: TextIO) -> None:
        self.terminal_stream = terminal_stream
        self.log_stream = log_stream

    def write(self, text: str) -> int:
        """同步写入终端与日志。"""
        self.terminal_stream.write(text)
        self.log_stream.write(text)
        return len(text)

    def flush(self) -> None:
        """同步刷新终端与日志。"""
        self.terminal_stream.flush()
        self.log_stream.flush()

    def isatty(self) -> bool:
        """保持终端语义一致。"""
        return bool(getattr(self.terminal_stream, "isatty", lambda: False)())


def derive_log_dir_from_output_dir(output_dir: str | Path) -> Path:
    """把原始结果目录映射到日志目录。

    目录约定：
    - 原始结果：`output/raw/...`
    - 日志记录：`output/log/...`
    """
    resolved_output_dir = resolve_project_path(output_dir)
    raw_root = project_root() / "output" / "raw"
    try:
        relative_path = resolved_output_dir.relative_to(raw_root)
    except ValueError as error:
        raise ValueError(f"output_dir 不在 output/raw/ 下，无法推导日志目录：{resolved_output_dir}") from error
    return project_root() / "output" / "log" / relative_path


@contextmanager
def tee_console_to_file(log_path: str | Path) -> Iterator[None]:
    """把 stdout/stderr 同时写到日志文件。"""
    resolved_log_path = resolve_project_path(log_path)
    ensure_parent(resolved_log_path)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with resolved_log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        log_file.write(f"\n===== run started at {datetime.now().isoformat(timespec='seconds')} =====\n")
        sys.stdout = TeeTextStream(original_stdout, log_file)
        sys.stderr = TeeTextStream(original_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
