"""答案抽取和投票逻辑。"""

from __future__ import annotations

import re
from collections import Counter


def _normalize_choice_text(choice: str) -> str:
    """将选项文本归一化，方便把模型输出映射回选项字母。"""
    normalized = choice.strip()
    normalized = re.sub(r"^[A-E]\.\s*", "", normalized, flags=re.IGNORECASE)
    normalized = normalized.strip().lower()
    normalized = normalized.replace(",", "")
    normalized = normalized.replace("$", "")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _map_choice_content_to_letter(answer_text: str, choices: list[str]) -> str | None:
    """把模型输出的选项内容反推为选项字母。

    设计原因：
    1. 一些基础模型会输出 `The answer is 6.`，而不是 `The answer is D.`。
    2. baseline 统计需要统一到选项字母层面，因此这里直接做规范化映射。
    """
    normalized_answer = answer_text.strip().lower()
    normalized_answer = normalized_answer.replace(",", "")
    normalized_answer = normalized_answer.replace("$", "")
    normalized_answer = re.sub(r"[\*\.\!\?\"'\)\]\}>]+$", "", normalized_answer)
    normalized_answer = re.sub(r"\s+", " ", normalized_answer)

    for choice in choices:
        letter_match = re.match(r"^\s*([A-E])\.", choice, flags=re.IGNORECASE)
        if letter_match is None:
            continue
        letter = letter_match.group(1).upper()
        choice_text = _normalize_choice_text(choice)
        if normalized_answer == choice_text:
            return letter
        if normalized_answer.endswith(choice_text):
            return letter
    return None


def _extract_choice_letter(answer_text: str, allow_standalone: bool = True) -> str | None:
    """从最终答案片段中抽取选项字母，避免把 because 等普通单词误当答案。"""
    normalized = answer_text.strip()
    normalized = re.sub(r"^[\"'`\s\*\(\[]+", "", normalized)

    labelled_match = re.search(
        r"(?:option|choice|answer|best answer|correct answer|final answer)\s*(?:is|:)?\s*([A-E])\b",
        normalized,
        flags=re.IGNORECASE,
    )
    if labelled_match:
        return labelled_match.group(1).upper()

    leading_match = re.match(r"^([A-E])(?:\s*[\.\):：、-]|\s*$)", normalized, flags=re.IGNORECASE)
    if leading_match:
        return leading_match.group(1).upper()

    standalone_match = re.search(r"\b([A-E])\b", normalized, flags=re.IGNORECASE)
    if allow_standalone and standalone_match:
        return standalone_match.group(1).upper()

    return None


def extract_answer(dataset_key: str, generated_text: str, strict: bool = True, choices: list[str] | None = None) -> str:
    """按数据集类型抽取最终答案。

    `strict=True` 时，抽取失败直接抛错。
    `strict=False` 时，抽取失败返回原始生成文本，方便 smoke test 先跑通链路。
    """
    if dataset_key == "gsm8k":
        matches = re.findall(r"The answer is\s*([^\n]+)", generated_text, flags=re.IGNORECASE)
        if not matches:
            if strict:
                raise ValueError(f"GSM8K 答案抽取失败，原始文本为：{generated_text}")
            return generated_text.strip()
        answer_text = matches[-1].strip()
        answer_text = re.sub(r"[\*\.\!\?\"'\)\]]+$", "", answer_text)
        answer_text = answer_text.replace(",", "").replace("$", "")
        number_match = re.search(r"-?\d+(?:\.\d+)?", answer_text)
        if number_match is None:
            if strict:
                raise ValueError(f"GSM8K 数值答案抽取失败，原始文本为：{generated_text}")
            return answer_text
        return number_match.group(0)

    if dataset_key in {"csqa", "mmlu"}:
        answer_lines = re.findall(r"The answer is\s*([^\n]+)", generated_text, flags=re.IGNORECASE)
        if not answer_lines:
            tail_text = "\n".join(generated_text.strip().splitlines()[-3:])
            fallback_letter = _extract_choice_letter(tail_text, allow_standalone=False)
            if fallback_letter is not None:
                return fallback_letter
            if choices:
                mapped_letter = _map_choice_content_to_letter(tail_text, choices)
                if mapped_letter is not None:
                    return mapped_letter
            if strict:
                raise ValueError(f"{dataset_key} 答案抽取失败，原始文本为：{generated_text}")
            return generated_text.strip()
        final_answer_line = answer_lines[-1]
        choice_letter = _extract_choice_letter(final_answer_line)
        if choice_letter is not None:
            return choice_letter

        if choices:
            mapped_letter = _map_choice_content_to_letter(final_answer_line, choices)
            if mapped_letter is not None:
                return mapped_letter

        if strict:
            raise ValueError(f"{dataset_key} 选项答案抽取失败，原始文本为：{generated_text}")
        return final_answer_line.strip()

    raise ValueError(f"未知数据集：{dataset_key}")


def majority_vote(answers: list[str]) -> tuple[str, dict[str, int]]:
    """执行未加权多数投票。"""
    counter = Counter(answers)
    most_common = counter.most_common()
    if not most_common:
        raise ValueError("多数投票失败，因为答案列表为空。")
    winner = most_common[0][0]
    return winner, dict(counter)
