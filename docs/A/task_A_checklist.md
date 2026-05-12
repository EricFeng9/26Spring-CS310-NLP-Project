# 任务A执行清单

## 1. 目标
- 搭建任务A完整 baseline 工程。
- 支持 `GSM8K`、`CommonsenseQA`、`MMLU` 三个数据集。
- 支持 `Qwen2.5-7B-Instruct`、`Mistral-7B-Instruct-v0.3`、`Olmo-3-7B-Instruct` 三个模型。
- 支持 `Zero-shot CoT`、`Few-shot CoT`、`Self-Consistency` 三类推理流程。
- 完成环境、模型、数据、脚本、配置、冒烟测试的全量准备。

## 2. 目录检查
- `configs/`：模型、数据集、prompt、运行配置。
- `scripts/`：环境安装、资源下载、构建 few-shot、运行推理、冒烟测试。
- `src/`：数据加载、prompt 构造、模型推理、答案抽取、投票、结果导出。
- `data/`：原始与标准化数据。
- `models/`：模型本地目录。
- `outputs/`：运行日志与预测结果。

## 3. 一次性准备命令
```bash
bash scripts/setup_env.sh
conda activate fjm_CoT
bash scripts/verify_env.sh
python scripts/download_datasets.py
python scripts/build_fewshot_prompts.py
python scripts/download_models.py --model qwen2_5_7b_instruct
python scripts/download_models.py --model mistral_7b_instruct_v0_3
python scripts/download_models.py --model olmo3_7b_instruct
```

## 4. 冒烟测试命令
```bash
bash scripts/run_smoke_tests.sh
```

说明：
- 脚本默认使用 `CUDA_VISIBLE_DEVICES=6` 在单卡上执行轻量探活。
- 三个模型均使用公开 Hugging Face 仓库，正常网络环境下无需额外申请 gated 权限。

## 5. 正式运行入口
```bash
python scripts/run_zero_shot.py --config configs/runs/zero_shot_smoke.yaml
python scripts/run_few_shot.py --config configs/runs/few_shot_smoke.yaml
python scripts/run_self_consistency.py --config configs/runs/self_consistency_smoke.yaml
```

## 6. 输出检查
- `outputs/smoke_tests/zero_shot/`
- `outputs/smoke_tests/few_shot/`
- `outputs/smoke_tests/self_consistency/`
- 每条记录必须包含：
  - `id`
  - `dataset`
  - `question`
  - `choices`
  - `gold_answer`
  - `prediction`
  - `reasoning`
  - `correct`

## 7. 完成标准
- `fjm_CoT` 环境可创建并可激活。
- 三个数据集都能成功下载并转成统一 JSONL。
- 三个模型都能下载到本地目录。
- 三类脚本都能完成最小推理。
- Self-Consistency 能保存多条路径和投票结果。
- 冒烟测试全部返回码为 0。
