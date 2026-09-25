# 示例说明

## GSM8K 训练链路示例

推荐先用 `datasets/gsm8k_tiny.jsonl` 验证数据准备和训练 recipe：

```bash
bcf data prepare-gsm8k --input examples/datasets/gsm8k_tiny.jsonl --output-dir runs/gsm8k
bcf train dry-run --workspace gsm8k-agent --recipe examples/recipes/gsm8k_sft_llamafactory.example.yaml
bcf train dry-run --workspace gsm8k-agent --recipe examples/recipes/gsm8k_grpo_verl.example.yaml
```

AutoDL 上可以参考 `docs/autodl_gsm8k_quickstart_zh.md`，`recipes/gsm8k_ppo_verl_autodl.example.yaml` 对齐 verl 官方 GSM8K 数据路径。

这个目录提供一组可直接运行的 synthetic 示例，帮助用户理解数据协议、工具协议和训练 recipe。

## 目录

- `datasets/seed_tasks.jsonl`：合成种子任务，用来模拟线上 trace 或人工构造任务。
- `datasets/eval_tasks.jsonl`：合成评测任务，定义每条样本需要通过哪些检查。
- `datasets/gsm8k_tiny.jsonl`：离线 GSM8K 风格数学题，适合验证数据准备命令。
- `tools/tool_manifest.example.json`：mock 工具清单，演示工具 schema、只读/有副作用边界。
- `evals/builtin_rule.example.yaml`：BadcaseFlow 内置规则评测 recipe 示例。
- `evals/command_eval_smoke.example.yaml`：本地评测 harness smoke 示例。
- `evals/promptfoo.example.yaml`：promptfoo dry-run recipe 示例。
- `evals/opencompass.example.yaml`：OpenCompass dry-run recipe 示例。
- `evals/lighteval.example.yaml`：LightEval dry-run recipe 示例。
- `recipes/sft_llamafactory.example.yaml`：LLaMA-Factory SFT recipe 示例。
- `recipes/dpo_llamafactory.example.yaml`：LLaMA-Factory DPO recipe 示例。
- `recipes/kto_llamafactory.example.yaml`：LLaMA-Factory KTO recipe 示例。
- `recipes/sft_verl.example.yaml`：verl SFT recipe 示例。
- `recipes/grpo_verl.example.yaml`：verl GRPO recipe 示例。
- `recipes/gsm8k_sft_llamafactory.example.yaml`：GSM8K + LLaMA-Factory SFT 示例。
- `recipes/gsm8k_grpo_verl.example.yaml`：GSM8K + verl GRPO 示例。
- `recipes/gsm8k_ppo_verl_autodl.example.yaml`：AutoDL + verl PPO 示例。
- `recipes/opd_verl.example.yaml`：verl OPD recipe 示例。
- `recipes/command_smoke.example.yaml`：本地训练 harness smoke 示例，不依赖 GPU。

可以用这些示例跑一轮完整本地飞轮：

```bash
bcf round run-local \
  --workspace demo-agent \
  --round runs/rounds/round-smoke \
  --train-mode run \
  --recipe examples/recipes/command_smoke.example.yaml \
  --model-id demo-agent-candidate \
  --attach-round-eval
bcf round inspect --round runs/rounds/round-smoke
```

查看和更新下一轮行动项：

```bash
bcf action list --plan runs/rounds/round-smoke/iteration_plan.json
bcf action update \
  --plan runs/rounds/round-smoke/iteration_plan.json \
  --action-id action-001 \
  --status accepted \
  --owner engineer-a
```

生成本地静态工作台：

```bash
bcf workbench build --registry-root .
```

如果要在训练后接候选模型回评测，可以给 round 传入 eval recipe：

```bash
bcf round run-local \
  --workspace demo-agent \
  --round runs/rounds/round-candidate \
  --train-mode run \
  --recipe examples/recipes/command_smoke.example.yaml \
  --model-id demo-agent-candidate \
  --candidate-eval-recipe examples/evals/builtin_rule.example.yaml \
  --candidate-eval-run-id candidate-eval
```

也可以单独验证评测 adapter：

```bash
bcf evals adapters
bcf evals run --recipe examples/evals/command_eval_smoke.example.yaml --run-id command-eval-smoke
bcf evals dry-run --recipe examples/evals/promptfoo.example.yaml
```

也可以单独检查和登记示例数据：

```bash
bcf dataset inspect --path examples/datasets/seed_tasks.jsonl --kind task --dataset-id seed-tasks
bcf dataset inspect --path examples/datasets/eval_tasks.jsonl --kind eval_suite --dataset-id eval-suite
bcf dataset register --path examples/datasets/seed_tasks.jsonl --kind task --dataset-id seed-tasks
bcf dataset list
```

如果 recipe 里使用 `data.dataset_version_id`、`data.trace_dataset_version_id` 或 `data.suite_dataset_version_id`，可以在 dry-run 时加 `--registry-root .`，让 BadcaseFlow 从 registry 解析成实际数据路径。

为训练或评测生成启动计划：

```bash
bcf launcher plan \
  --kind train \
  --launcher local \
  --mode run \
  --recipe examples/recipes/command_smoke.example.yaml \
  --run-id launcher-train
```

把启动计划提交成可追踪 job：

```bash
bcf launcher plan \
  --kind train \
  --launcher local \
  --mode run \
  --recipe examples/recipes/command_smoke.example.yaml \
  --run-id job-smoke \
  --output runs/job-smoke-plan.json
bcf job submit --plan runs/job-smoke-plan.json --job-id job-smoke
bcf job run --job runs/jobs/job-smoke
bcf job status --job runs/jobs/job-smoke
bcf job logs --job runs/jobs/job-smoke --stream stdout --tail 20
```

候选模型通过门禁后，可以生成部署计划：

```bash
bcf deploy plan \
  --model-id demo-agent-candidate \
  --backend vllm \
  --deployment-id demo-agent-v1 \
  --canary-percent 10 \
  --register
```

## 为什么示例内容使用通用办公 Agent

第一版 demo 建议使用办公工具 Agent，而不是垂直行业场景。原因是：

- 工具调用轨迹清晰，适合展示 Agent 训练飞轮；
- 成败标准容易规则化；
- 示例可以完全合成；
- 可以覆盖 SFT、偏好数据、RL prompt、OPD 数据等多种训练输入。

后续用户可以通过 adapter 接入自己的工具、评测集、reward 和训练平台。
