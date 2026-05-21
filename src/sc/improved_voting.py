"""任务B使用的 Self-Consistency 后处理改进逻辑。"""

from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal, InvalidOperation


VALID_CHOICE_LETTERS = {"A", "B", "C", "D", "E"}


def normalize_choice_text(choice: str) -> str:
    """归一化选项文本，用于把模型输出的选项内容映射回字母。"""
    normalized = choice.strip().lower()
    normalized = re.sub(r"^[A-E]\.\s*", "", normalized, flags=re.IGNORECASE)
    normalized = normalized.replace(",", "")
    normalized = normalized.replace("$", "")
    normalized = re.sub(r"[\*\.\!\?\"'\)\]\}>]+$", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def normalize_gsm8k_answer(answer: str) -> str:
    """将 GSM8K 路径答案归一化为统一数字字符串。

    任务B只处理任务A已经抽取出的 `answer` 字段，因此这里不重新解析完整推理文本。
    如果答案中没有可识别数字，直接返回空字符串，把问题暴露给过滤统计。
    """
    cleaned = str(answer).strip()
    cleaned = cleaned.replace(",", "").replace("$", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if not matches:
        return ""

    number_text = matches[-1]
    try:
        number = Decimal(number_text)
    except InvalidOperation:
        return ""

    if number == number.to_integral_value():
        return str(number.quantize(Decimal("1")))
    return format(number.normalize(), "f")


def map_choice_content_to_letter(answer: str, choices: list[str]) -> str:
    """将选择题答案文本映射回 `A-E` 字母。"""
    normalized_answer = normalize_choice_text(str(answer))
    for choice in choices:
        letter_match = re.match(r"^\s*([A-E])\.", choice, flags=re.IGNORECASE)
        if letter_match is None:
            continue
        letter = letter_match.group(1).upper()
        choice_text = normalize_choice_text(choice)
        if normalized_answer == choice_text:
            return letter
        if normalized_answer.endswith(choice_text):
            return letter
    return ""


def normalize_choice_answer(answer: str, choices: list[str]) -> str:
    """将 CSQA/MMLU 路径答案归一化为 `A-E`。"""
    answer_text = str(answer).strip()
    if not answer_text:
        return ""

    labelled_match = re.search(
        r"(?:option|choice|answer|best answer|correct answer|final answer)\s*(?:is|:)?\s*([A-E])\b",
        answer_text,
        flags=re.IGNORECASE,
    )
    if labelled_match:
        return labelled_match.group(1).upper()

    leading_match = re.match(
        r"^[\"'`\s\*\(\[]*([A-E])(?:\s*[\.\):：、-]|\s*$)",
        answer_text,
        flags=re.IGNORECASE,
    )
    if leading_match:
        return leading_match.group(1).upper()

    standalone_match = re.fullmatch(r"[A-E]", answer_text.strip(), flags=re.IGNORECASE)
    if standalone_match:
        return standalone_match.group(0).upper()

    mapped_letter = map_choice_content_to_letter(answer_text, choices)
    if mapped_letter:
        return mapped_letter

    return ""


def normalize_answer(dataset_key: str, answer: str, choices: list[str]) -> str:
    """按数据集类型归一化单条路径答案。"""
    if dataset_key == "gsm8k":
        return normalize_gsm8k_answer(answer)
    if dataset_key in {"csqa", "mmlu"}:
        return normalize_choice_answer(answer, choices)
    raise ValueError(f"未知数据集：{dataset_key}")


def is_valid_answer(dataset_key: str, answer: str) -> bool:
    """判断归一化后的答案是否可参与投票。"""
    normalized = str(answer).strip()
    if not normalized:
        return False
    if dataset_key == "gsm8k":
        return re.fullmatch(r"-?\d+(?:\.\d+)?", normalized) is not None
    if dataset_key in {"csqa", "mmlu"}:
        return normalized in VALID_CHOICE_LETTERS
    raise ValueError(f"未知数据集：{dataset_key}")


def stable_majority_vote(answers: list[str]) -> tuple[str, dict[str, int]]:
    """执行稳定多数投票，平票时保留最早出现的最高票答案。"""
    counter = Counter(answers)
    if not counter:
        return "", {}

    max_count = max(counter.values())
    for answer in answers:
        if counter[answer] == max_count:
            return answer, dict(counter)

    raise RuntimeError("多数投票内部错误：没有找到最高票答案。")


def duplicate_reasoning_ratio(paths: list[dict]) -> float:
    """统计完全重复推理路径比例，用于任务B分析，不参与过滤。"""
    if not paths:
        return 0.0
    reasonings = [str(path.get("reasoning", "")).strip() for path in paths]
    duplicate_count = len(reasonings) - len(set(reasonings))
    return duplicate_count / len(reasonings)


def build_path_views(dataset_key: str, row: dict) -> list[dict]:
    """将原始路径转换为带归一化结果和有效性标记的结构。"""
    views: list[dict] = []
    choices = row.get("choices", [])
    for index, path in enumerate(row["paths"], start=1):
        raw_answer = str(path.get("answer", "")).strip()
        normalized_answer = normalize_answer(dataset_key, raw_answer, choices)
        views.append(
            {
                "path_index": index,
                "raw_answer": raw_answer,
                "normalized_answer": normalized_answer,
                "raw_valid": bool(raw_answer),
                "normalized_valid": is_valid_answer(dataset_key, normalized_answer),
            }
        )
    return views
