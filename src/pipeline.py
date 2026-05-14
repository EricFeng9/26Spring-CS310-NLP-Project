"""任务A三类推理模式的统一执行入口。"""

from __future__ import annotations

from pathlib import Path

from src.data.loaders import load_jsonl_samples, load_processed_split
from src.io.files import write_csv, write_jsonl
from src.models.runner import LocalModelRunner
from src.prompts.builders import build_few_shot_prompt, build_self_consistency_prompt, build_zero_shot_prompt
from src.sc.answering import extract_answer, majority_vote
from src.utils import load_yaml, resolve_project_path


def load_fewshot_examples(path: str | Path) -> list:
    """读取 few-shot 示例。"""
    return load_jsonl_samples(path)


def run_zero_shot(run_config: dict, model_config: dict, prompt_config: dict) -> list[dict]:
    """执行 Zero-shot CoT。"""
    samples = load_processed_split(run_config["dataset_key"], run_config["split"], run_config["limit"])
    runner = LocalModelRunner(model_config, prompt_config["system_prompt"])
    strict_answer_parsing = bool(run_config.get("strict_answer_parsing", True))
    results: list[dict] = []
    for sample in samples:
        prompt = build_zero_shot_prompt(sample, prompt_config)
        output = runner.generate(
            prompt=prompt,
            max_new_tokens=run_config["max_new_tokens"],
            temperature=run_config["temperature"],
            top_p=run_config["top_p"],
            do_sample=run_config["do_sample"],
        )
        prediction = extract_answer(
            run_config["dataset_key"],
            output.generated_text,
            strict=strict_answer_parsing,
            choices=sample.choices,
        )
        results.append(
            {
                "id": sample.id,
                "dataset": sample.dataset,
                "question": sample.question,
                "choices": sample.choices,
                "gold_answer": sample.gold_answer,
                "prediction": prediction,
                "reasoning": output.generated_text,
                "correct": prediction == sample.gold_answer,
                "prompt": output.prompt,
            }
        )
    return results


def run_few_shot(run_config: dict, model_config: dict, prompt_config: dict) -> list[dict]:
    """执行 Few-shot CoT。"""
    samples = load_processed_split(run_config["dataset_key"], run_config["split"], run_config["limit"])
    fewshot_examples = load_fewshot_examples(run_config["few_shot_examples_path"])
    runner = LocalModelRunner(model_config, prompt_config["system_prompt"])
    strict_answer_parsing = bool(run_config.get("strict_answer_parsing", True))
    results: list[dict] = []
    for sample in samples:
        prompt = build_few_shot_prompt(sample, prompt_config, fewshot_examples)
        output = runner.generate(
            prompt=prompt,
            max_new_tokens=run_config["max_new_tokens"],
            temperature=run_config["temperature"],
            top_p=run_config["top_p"],
            do_sample=run_config["do_sample"],
        )
        prediction = extract_answer(
            run_config["dataset_key"],
            output.generated_text,
            strict=strict_answer_parsing,
            choices=sample.choices,
        )
        results.append(
            {
                "id": sample.id,
                "dataset": sample.dataset,
                "question": sample.question,
                "choices": sample.choices,
                "gold_answer": sample.gold_answer,
                "prediction": prediction,
                "reasoning": output.generated_text,
                "correct": prediction == sample.gold_answer,
                "prompt": output.prompt,
            }
        )
    return results


def run_self_consistency(run_config: dict, model_config: dict, prompt_config: dict) -> list[dict]:
    """执行 Self-Consistency。"""
    samples = load_processed_split(run_config["dataset_key"], run_config["split"], run_config["limit"])
    runner = LocalModelRunner(model_config, prompt_config["system_prompt"])
    strict_answer_parsing = bool(run_config.get("strict_answer_parsing", True))
    results: list[dict] = []
    for sample in samples:
        prompt = build_self_consistency_prompt(sample, prompt_config)
        paths: list[dict] = []
        answers: list[str] = []
        for _ in range(run_config["num_paths"]):
            output = runner.generate(
                prompt=prompt,
                max_new_tokens=run_config["max_new_tokens"],
                temperature=run_config["temperature"],
                top_p=run_config["top_p"],
                do_sample=run_config["do_sample"],
            )
            answer = extract_answer(
                run_config["dataset_key"],
                output.generated_text,
                strict=strict_answer_parsing,
                choices=sample.choices,
            )
            answers.append(answer)
            paths.append({"reasoning": output.generated_text, "answer": answer})
        prediction, vote_counts = majority_vote(answers)
        results.append(
            {
                "id": sample.id,
                "dataset": sample.dataset,
                "question": sample.question,
                "choices": sample.choices,
                "gold_answer": sample.gold_answer,
                "prediction": prediction,
                "reasoning": paths[0]["reasoning"],
                "correct": prediction == sample.gold_answer,
                "prompt": prompt,
                "paths": paths,
                "vote_counts": vote_counts,
            }
        )
    return results


def save_results(output_dir: str | Path, result_name: str, rows: list[dict]) -> None:
    """按统一格式写出 JSONL 和 CSV。"""
    output_path = resolve_project_path(output_dir)
    jsonl_path = output_path / f"{result_name}.jsonl"
    csv_rows = []
    for row in rows:
        csv_rows.append(
            {
                "id": row["id"],
                "dataset": row["dataset"],
                "gold_answer": row["gold_answer"],
                "prediction": row["prediction"],
                "correct": row["correct"],
            }
        )
    write_jsonl(jsonl_path, rows)
    write_csv(output_path / f"{result_name}.csv", csv_rows)


def load_core_configs(run_config_path: str | Path) -> tuple[dict, dict, dict]:
    """按运行配置自动加载模型与 prompt 配置。"""
    run_config = load_yaml(run_config_path)
    model_config = load_yaml(resolve_project_path(f"configs/models/{run_config['model_key']}.yaml"))
    prompt_config = load_yaml(resolve_project_path("configs/prompts/base_prompt.yaml"))
    return run_config, model_config, prompt_config
