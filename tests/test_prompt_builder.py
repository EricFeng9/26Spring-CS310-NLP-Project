"""基础 prompt 构造测试。"""

from src.data.types import TaskSample
from src.prompts.builders import build_zero_shot_prompt


def test_zero_shot_prompt_contains_reasoning_tag() -> None:
    """确认 Zero-shot prompt 会生成推理起始标记。"""
    prompt = build_zero_shot_prompt(
        TaskSample(
            id="1",
            dataset="gsm8k",
            question="1+1=?",
            choices=[],
            gold_answer="2",
            metadata={},
        ),
        {
            "zero_shot_instruction": "请逐步思考。",
        },
    )
    assert "Reasoning:" in prompt
