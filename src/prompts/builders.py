"""统一构造 Zero-shot、Few-shot、Self-Consistency prompt。"""

from __future__ import annotations

from src.data.types import TaskSample


def format_choices(choices: list[str]) -> str:
    """将选项列表格式化成多行文本。"""
    if not choices:
        return ""
    return "\n".join(choices)


def sample_to_block(sample: TaskSample, include_answer: bool) -> str:
    """将样本格式化成 prompt 内的一个块。"""
    lines = [f"Question: {sample.question}"]
    if sample.choices:
        lines.append("Choices:")
        lines.append(format_choices(sample.choices))
    if include_answer:
        reasoning = sample.metadata["fewshot_reasoning"]
        lines.append(f"Reasoning: {reasoning}")
        lines.append(f"The answer is {sample.gold_answer}.")
    return "\n".join(lines)


def build_zero_shot_prompt(sample: TaskSample, prompt_config: dict) -> str:
    """构造 Zero-shot CoT prompt。"""
    lines = [
        prompt_config["zero_shot_instruction"].strip(),
        sample_to_block(sample, include_answer=False),
        "Reasoning:",
    ]
    return "\n\n".join(lines)


def build_few_shot_prompt(sample: TaskSample, prompt_config: dict, fewshot_samples: list[TaskSample]) -> str:
    """构造 Few-shot CoT prompt。"""
    example_blocks = [sample_to_block(example, include_answer=True) for example in fewshot_samples]
    lines = [
        prompt_config["few_shot_instruction"].strip(),
        "\n\n".join(example_blocks),
        sample_to_block(sample, include_answer=False),
        "Reasoning:",
    ]
    return "\n\n".join(lines)


def build_self_consistency_prompt(sample: TaskSample, prompt_config: dict) -> str:
    """构造 Self-Consistency 使用的单次推理 prompt。"""
    lines = [
        prompt_config["self_consistency_instruction"].strip(),
        sample_to_block(sample, include_answer=False),
        "Reasoning:",
    ]
    return "\n\n".join(lines)
