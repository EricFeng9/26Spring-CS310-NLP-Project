"""答案抽取和投票逻辑。"""

from __future__ import annotations

import re
from collections import Counter


def extract_answer(dataset_key: str, generated_text: str, strict: bool = True) -> str:
    """按数据集类型抽取最终答案。

    `strict=True` 时，抽取失败直接抛错。
    `strict=False` 时，抽取失败返回原始生成文本，方便 smoke test 先跑通链路。
    """
    if dataset_key == "gsm8k":
        match = re.search(r"The answer is\s*([^\n\.]+)", generated_text, flags=re.IGNORECASE)
        if match is None:
            if strict:
                raise ValueError(f"GSM8K 答案抽取失败，原始文本为：{generated_text}")
            return generated_text.strip()
        answer = match.group(1).strip().replace(",", "")
        answer = answer.replace("$", "")
        return answer

    if dataset_key in {"csqa", "mmlu"}:
        match = re.search(r"The answer is\s*([A-E])", generated_text, flags=re.IGNORECASE)
        if match is None:
            if strict:
                raise ValueError(f"{dataset_key} 答案抽取失败，原始文本为：{generated_text}")
            return generated_text.strip()
        return match.group(1).upper()

    raise ValueError(f"未知数据集：{dataset_key}")


def majority_vote(answers: list[str]) -> tuple[str, dict[str, int]]:
    """执行未加权多数投票。"""
    counter = Counter(answers)
    most_common = counter.most_common()
    if not most_common:
        raise ValueError("多数投票失败，因为答案列表为空。")
    winner = most_common[0][0]
    return winner, dict(counter)
