# AutoDL + GSM8K 快速开始

这条链路是 BadcaseFlow 的第一条推荐真实任务线：用大家熟悉的 GSM8K 数学题，先跑通数据准备、训练 recipe、训练任务追踪、指标收集、模型登记和回评计划。它不追求一次把模型训到最好，目标是让平台的飞轮闭环先稳定运转起来。

## 适合谁

- 已经有 Agent 或模型训练需求，想把 badcase、评测、训练和上线门禁串起来的人。
- 准备在 AutoDL 或自己的 GPU 服务器上跑 LLaMA-Factory、verl、vLLM、SGLang 的人。
- 想先用一个标准数学任务验证训练链路，再迁移到自己 Agent 项目的人。

## 当前能做什么

- 把 GSM8K 风格 JSONL 转成 BadcaseFlow task、eval suite、SFT messages、RL prompt 和 verl prompt。
- 在装了 `pyarrow` 的环境里输出 verl 可读的 parquet。
- 用 `bcf train dry-run/run` 生成并执行 LLaMA-Factory 或 verl 训练命令。
- 用 `bcf launcher`、`bcf job` 追踪训练任务、日志和轻量指标。
- 用 `bcf registry` 记录数据版本、训练 run、候选模型、评测报告和上线门禁。
- 用 `bcf workbench build` 导出本地工作台，查看 round、job、模型和 action。

## 还需要继续增强

- 交互式 Web 工作台还没有做成完整产品，目前先用静态 HTML 查看状态。
- GPU 队列、并发、重试和资源调度还是轻量 job runner。
- 线上 trace 自动回流、灰度对比和自动回滚还处在部署计划阶段。
- case 聚类、judge 复核、数据去重和数据质量评分还需要继续做深。

## 本地先跑通 tiny 数据

```bash
python -m pip install -e .
python -m badcaseflow doctor --strict

python -m badcaseflow data prepare-gsm8k \
  --input examples/datasets/gsm8k_tiny.jsonl \
  --output-dir runs/gsm8k
```

输出会包含：

```text
runs/gsm8k/task_samples.jsonl
runs/gsm8k/eval_suite.jsonl
runs/gsm8k/sft_train.jsonl
runs/gsm8k/sft_val.jsonl
runs/gsm8k/rl_train.jsonl
runs/gsm8k/rl_val.jsonl
runs/gsm8k/verl_train.jsonl
runs/gsm8k/verl_val.jsonl
runs/gsm8k/dataset_info.json
runs/gsm8k/manifest.json
```

如果本机装了 parquet 依赖，可以直接生成 verl 训练数据：

```bash
python -m pip install -e ".[parquet]"
python -m badcaseflow data prepare-gsm8k \
  --input examples/datasets/gsm8k_tiny.jsonl \
  --output-dir runs/gsm8k \
  --parquet \
  --register
```

检查数据版本：

```bash
bcf dataset list
bcf dataset inspect --path runs/gsm8k/sft_train.jsonl --kind sft --dataset-id gsm8k-sft-train
bcf dataset inspect --path runs/gsm8k/rl_train.jsonl --kind rl_prompt --dataset-id gsm8k-rl-train
```

## 三条训练入口

先做 SFT：

```bash
bcf train dry-run \
  --workspace gsm8k-agent \
  --recipe examples/recipes/gsm8k_sft_llamafactory.example.yaml
```

用本地生成的 parquet 走 verl GRPO：

```bash
python -m badcaseflow train dry-run \
  --workspace gsm8k-agent \
  --recipe examples/recipes/gsm8k_grpo_verl.example.yaml
```

上面的 recipe 需要 `runs/gsm8k/verl_train.parquet` 和 `runs/gsm8k/verl_val.parquet`。如果只生成了 JSONL，请先安装 parquet 可选依赖并重新执行带 `--parquet` 的数据准备命令。

在 AutoDL 上用 verl 官方 GSM8K 数据路径跑 PPO：

```bash
bcf train dry-run \
  --workspace gsm8k-agent \
  --recipe examples/recipes/gsm8k_ppo_verl_autodl.example.yaml
```

## AutoDL 建议流程

在 AutoDL 实例里：

```bash
git clone https://github.com/ZackLiang/BadcaseFlow.git /root/BadcaseFlow
cd /root/BadcaseFlow
python -m pip install -e ".[parquet]"
bcf doctor --strict
```

安装并准备 verl 的 GSM8K 数据：

```bash
git clone https://github.com/verl-project/verl.git /root/verl
cd /root/verl
python -m pip install -e .
python examples/data_preprocess/gsm8k.py --local_save_dir /root/data/gsm8k
```

回到 BadcaseFlow 生成训练计划：

```bash
cd /root/BadcaseFlow
bcf train dry-run \
  --workspace gsm8k-agent \
  --recipe examples/recipes/gsm8k_ppo_verl_autodl.example.yaml
```

确认命令没问题后，用 job runner 跟踪执行：

```bash
bcf launcher plan \
  --kind train \
  --launcher local \
  --mode run \
  --workspace gsm8k-agent \
  --recipe examples/recipes/gsm8k_ppo_verl_autodl.example.yaml \
  --run-id gsm8k-ppo-autodl \
  --output runs/gsm8k-ppo-plan.json

bcf job submit --plan runs/gsm8k-ppo-plan.json --job-id gsm8k-ppo-autodl
bcf job run --job runs/jobs/gsm8k-ppo-autodl
bcf job status --job runs/jobs/gsm8k-ppo-autodl
bcf job logs --job runs/jobs/gsm8k-ppo-autodl --stream stdout --tail 80
```

训练完成后登记产物：

```bash
bcf registry index-train-run --run runs/train_runs/gsm8k-ppo-autodl
bcf registry register-model \
  --run runs/train_runs/gsm8k-ppo-autodl \
  --model-id gsm8k-agent-ppo-v0.1 \
  --model-path artifacts/gsm8k-agent/ppo-v0.1
bcf registry model-lineage --model-id gsm8k-agent-ppo-v0.1
bcf workbench build --registry-root .
```

## 回评怎么接

GSM8K 的第一层回评可以先用训练框架自带的验证集 reward 和 pass rate。之后可以再加一条独立评测 recipe，例如 LightEval、OpenCompass 或自己的命令行评测脚本。BadcaseFlow 会负责记录评测 run、报告、promotion gate 和下一轮 action。

如果你已经有 Agent 输出 trace，可以把 `runs/gsm8k/eval_suite.jsonl` 作为回归集，trace 里 `sample_id` 对齐 `gsm8k-val-000001` 这类 ID，最终答案写成 `#### 23` 这种格式，然后运行：

```bash
bcf import-traces --workspace gsm8k-agent --input your_traces.jsonl --run runs/gsm8k-eval
bcf eval --workspace gsm8k-agent --run runs/gsm8k-eval --suite runs/gsm8k/eval_suite.jsonl
bcf analyze-cases --workspace gsm8k-agent --run runs/gsm8k-eval
```

## 推荐第一步

先在本地跑 `gsm8k_tiny.jsonl`，确认数据、recipe、registry、workbench 都能通。然后在 AutoDL 上用 `Qwen/Qwen2.5-0.5B-Instruct` 跑一轮 GSM8K PPO 或 GRPO。等这条链路稳定，再接你的 Agent 项目，把真实 badcase 和工具调用 trace 接进来。
