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


def final_answer_instruction(sample: TaskSample) -> str:
    """根据任务类型生成最终答案格式约束。

    设计原则：
    1. 多选题必须强制输出选项字母，避免模型把推理出的数值直接当最终答案写出。
    2. 数值题必须强制输出最终数值，方便后续统一抽取与统计。
    """
    if sample.choices:
        return "Your final line must be exactly: The answer is <A/B/C/D/E>."
    return "Your final line must be exactly: The answer is <number>."


def build_zero_shot_prompt(sample: TaskSample, prompt_config: dict) -> str:
    """构造 Zero-shot CoT prompt。"""
    lines = [
        prompt_config["zero_shot_instruction"].strip(),
        final_answer_instruction(sample),
        sample_to_block(sample, include_answer=False),
        "Reasoning:",
    ]
    return "\n\n".join(lines)


def build_few_shot_prompt(sample: TaskSample, prompt_config: dict, fewshot_samples: list[TaskSample]) -> str:
    """构造 Few-shot CoT prompt。"""
    example_blocks = [sample_to_block(example, include_answer=True) for example in fewshot_samples]
    lines = [
        prompt_config["few_shot_instruction"].strip(),
        final_answer_instruction(sample),
        "\n\n".join(example_blocks),
        sample_to_block(sample, include_answer=False),
        "Reasoning:",
    ]
    return "\n\n".join(lines)


def build_self_consistency_prompt(
    sample: TaskSample,
    prompt_config: dict,
    fewshot_samples: list[TaskSample] | None = None,
) -> str:
    """构造 Self-Consistency 使用的单次推理 prompt。"""
    if fewshot_samples:
        example_blocks = [sample_to_block(example, include_answer=True) for example in fewshot_samples]
        lines = [
            prompt_config["few_shot_instruction"].strip(),
            final_answer_instruction(sample),
            "\n\n".join(example_blocks),
            sample_to_block(sample, include_answer=False),
            "Reasoning:",
        ]
        return "\n\n".join(lines)

    lines = [
        prompt_config["self_consistency_instruction"].strip(),
        final_answer_instruction(sample),
        sample_to_block(sample, include_answer=False),
        "Reasoning:",
    ]
    return "\n\n".join(lines)
