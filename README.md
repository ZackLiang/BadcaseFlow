# BadcaseFlow

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-Apache--2.0-green)
[![CI](https://github.com/ZackLiang/BadcaseFlow/actions/workflows/ci.yml/badge.svg)](https://github.com/ZackLiang/BadcaseFlow/actions/workflows/ci.yml)

> 面向 Agent 的训练飞轮控制层：把 trace、评测、badcase、SFT/RL/OPD 训练、模型门禁和下一轮迭代连接起来。

BadcaseFlow 适合已经有 Agent 原型、评测集或训练需求的团队。它不替代 LLaMA-Factory、verl、OpenCompass、vLLM 等成熟项目，而是负责统一数据协议、执行计划、运行记录、模型血缘和迭代动作。

## GSM8K / AutoDL 快速入口

如果你想先用一个大家熟悉的数学任务跑通真实训练链路，可以从 [AutoDL + GSM8K 快速开始](docs/autodl_gsm8k_quickstart_zh.md) 开始。核心命令：

```bash
python -m pip install -e .
python -m badcaseflow data prepare-gsm8k --input examples/datasets/gsm8k_tiny.jsonl --output-dir runs/gsm8k
bcf train dry-run --workspace gsm8k-agent --recipe examples/recipes/gsm8k_sft_llamafactory.example.yaml
bcf train dry-run --workspace gsm8k-agent --recipe examples/recipes/gsm8k_grpo_verl.example.yaml
```

BadcaseFlow 是一个面向 Agent 的受控自进化平台：把失败 case、线上 trace、评测、数据生成、SFT/RL/OPD 后训练和上线门禁串成一个可追踪闭环。

名字里的 `Badcase` 是核心入口，`Flow` 是产品方法：失败样本不应该停在日志里，而应该流向归因、数据、训练、回评测和下一轮上线决策。

## 当前版本

当前首版是 `v0.1.0`，定位为“本地可验证的 Agent 训练飞轮控制层”。它已经覆盖：

- 任务和 trace 导入、mock rollout、规则评测、case 分析和训练数据导出；
- LLaMA-Factory、verl、command backend 的训练 recipe、dry-run、run、inspect 和指标记录；
- builtin、command、promptfoo、OpenCompass、LightEval 的评测 adapter；
- dataset version、model registry、promotion gate、deployment plan 和一轮飞轮编排；
- 静态工作台导出，把 round、case action、registry、训练和评测摘要写成 HTML；
- action lifecycle，把下一轮行动从 `proposed` 推进到 `accepted`、`in_progress`、`done` 等状态；
- job runner，把 launcher plan 登记成可追踪任务，支持本地执行、SSH 计划、状态、日志和 collect；
- `bcf version`、`bcf doctor`、单元测试和示例 recipe，方便做首版验收。

## 三种使用方式

### 1. 不接 GPU，验证平台闭环

```bash
python -m badcaseflow demo --run runs/demo
python -m badcaseflow round run-local \
  --workspace demo-agent \
  --round runs/rounds/round-smoke \
  --train-mode run \
  --recipe examples/recipes/command_smoke.example.yaml \
  --model-id demo-agent-candidate \
  --attach-round-eval
```

### 2. 接本地 GPU 或 AutoDL，运行真实训练

先用 `train dry-run` 检查 recipe，再用 `launcher` 和 `job` 执行。推荐从 [GSM8K 快速开始](docs/autodl_gsm8k_quickstart_zh.md) 入手。

```bash
python -m badcaseflow train dry-run \
  --workspace gsm8k-agent \
  --recipe examples/recipes/gsm8k_ppo_verl_autodl.example.yaml
```

### 3. 接自己的 Agent trace 和评测

将 trace 转成统一 JSONL 后使用 `import-traces` 导入，再用自己的 eval suite、评测命令或外部 adapter 完成回评测。

```bash
python -m badcaseflow import-traces \
  --workspace my-agent \
  --input path/to/traces.jsonl \
  --run runs/my-agent-eval
python -m badcaseflow eval \
  --workspace my-agent \
  --run runs/my-agent-eval \
  --suite path/to/eval_suite.jsonl
```

推荐先跑这几条命令确认环境：

```bash
python -m pip install -e .
bcf version
bcf doctor --strict
bcf demo --run runs/demo
```

然后跑一轮包含训练 harness、模型登记和门禁决策的本地飞轮：

```bash
bcf round run-local \
  --workspace demo-agent \
  --round runs/rounds/round-smoke \
  --train-mode run \
  --recipe examples/recipes/command_smoke.example.yaml \
  --model-id demo-agent-candidate \
  --attach-round-eval
bcf round inspect --round runs/rounds/round-smoke
bcf round plan-next --round runs/rounds/round-smoke
bcf action list --plan runs/rounds/round-smoke/iteration_plan.json
bcf launcher plan --kind train --launcher local --mode run --recipe examples/recipes/command_smoke.example.yaml --run-id job-smoke --output runs/job-smoke-plan.json
bcf job submit --plan runs/job-smoke-plan.json --job-id job-smoke
bcf job run --job runs/jobs/job-smoke
bcf workbench build --registry-root .
```

## 解决什么痛点

很多团队已经能做出 Agent demo，也能零散地跑 SFT、DPO、GRPO、OPD 或评测脚本，但真正持续迭代时会卡在这些地方：

| 痛点 | 现在常见情况 | 这个平台要解决的问题 |
|---|---|---|
| badcase 难复用 | 失败样本散落在日志、表格、人工反馈里 | 把失败 case 结构化成可评测、可训练、可追踪的数据资产 |
| 训练和评测割裂 | 训练看 loss，评测另跑脚本，上线靠人工判断 | 每个模型版本绑定数据版本、训练 recipe、评测报告和上线决策 |
| Agent trace 不好进训练 | 工具调用、观察结果、最终回答混在一起 | 用统一 trace schema 保存工具轨迹、环境观察、reward 和人工反馈 |
| RL/OPD 工程门槛高 | 数据格式、资源池、reward、指标、checkpoint 分散在脚本里 | 提供 recipe、preflight、dry-run、指标收集和训练后自动回评测 |
| 迭代靠经验 | 不知道该补数据、改 prompt、改 reward，还是改工具 | 对 badcase 做聚类和归因，生成下一轮数据/评测/训练建议 |

一句话目标：

```text
把 Agent 的失败案例，稳定转化为下一轮可评测、可训练、可上线验证的能力改进。
```

## 用户是谁

第一批用户建议定位为：

```text
已经有 Agent 原型或线上 Agent 的团队，
他们有 trace、评测集、badcase 和训练需求，
但缺少一个把数据、评测、训练、case 分析、上线回流连起来的平台。
```

典型角色：

- 后训练工程师：负责 SFT、DPO、GRPO、PPO、OPD、模型评测和实验管理。
- Agent 工程师：负责工具调用、workflow、RAG、线上 trace 和失败样本分析。
- AI 应用负责人：关心新模型是否真的变好、能否灰度、能否回滚、问题能否追溯。

更完整的用户分析见 [产品定位](docs/product_positioning_zh.md)。

## 怎么开发

这个项目不重复造训练框架、观测平台或模型服务，而是做一个“飞轮控制层”，把成熟开源项目组装起来。

```text
Trace / Feedback
  Langfuse / Phoenix / OpenTelemetry
        |
        v
BadcaseFlow
  dataset versioning
  eval suite
  train recipe
  case analyzer
  model registry
  promotion gate
        |
        +--> Training: LLaMA-Factory / verl / Axolotl / OpenRLHF / TRL
        +--> Evaluation: OpenCompass / LightEval / promptfoo / custom eval
        +--> Serving: vLLM / SGLang / OpenAI-compatible API
        +--> Metrics: TensorBoard / MLflow / W&B / SwanLab
```

平台自己重点做五件事：

1. 定义 Agent 训练飞轮的数据协议：任务、trace、reward、评测、训练 run、模型版本。
2. 把 badcase 转成下一轮 action：补 SFT 数据、构造偏好对、加入 RL prompt、调整评测集或修改工具 schema。
3. 管理训练 recipe：常规后训练走 LLaMA-Factory，RL/OPD/SFT 等阶段可以走 verl，其他框架通过 adapter 接入。
4. 做训练前后门禁：preflight、dry-run、训练指标采集、训练后自动评测、回归集对比。
5. 记录模型上线血缘：哪个数据集、哪个 recipe、哪个 checkpoint、哪个评测报告支撑这次发布。

## 第一版范围

第一版先做本地可运行、可验收、可继续扩展的闭环，不追求马上训练出最强模型：

```text
合成种子任务
  -> mock Agent rollout
  -> 规则 / judge 评测
  -> accepted / rejected 切分
  -> SFT 数据导出
  -> 训练 dry-run manifest
  -> badcase 报告
  -> 下一轮任务建议
```

希望命令形态长这样：

```bash
bcf ingest --workspace demo-agent --input examples/datasets/seed_tasks.jsonl --run runs/demo
bcf rollout --workspace demo-agent --run runs/demo --agent mock
bcf eval --workspace demo-agent --run runs/demo --suite examples/datasets/eval_tasks.jsonl
bcf export sft --workspace demo-agent --run runs/demo --output runs/demo/sft.jsonl
bcf train dry-run --workspace demo-agent --recipe examples/recipes/sft_llamafactory.example.yaml
bcf analyze-cases --workspace demo-agent --run runs/demo
```

## 本地运行

当前仓库已经包含一个不依赖 GPU 和外部服务的本地 CLI 原型。可以先用源码方式运行：

```bash
cd badcaseflow
python -m pip install -e .
bcf doctor --strict
bcf demo --run runs/demo
```

`bcf demo` 会完整执行导入、mock rollout、评测、case 分析、SFT/preference/RL prompt 导出和 SFT recipe dry-run。

也可以拆开执行：

```bash
bcf ingest --workspace demo-agent --input examples/datasets/seed_tasks.jsonl --run runs/demo
bcf rollout --workspace demo-agent --run runs/demo --agent mock
bcf eval --workspace demo-agent --run runs/demo --suite examples/datasets/eval_tasks.jsonl
bcf analyze-cases --workspace demo-agent --run runs/demo
bcf export sft --workspace demo-agent --run runs/demo --output runs/demo/sft.jsonl
bcf export preference --workspace demo-agent --run runs/demo --output runs/demo/preference.jsonl
bcf train dry-run --workspace demo-agent --recipe examples/recipes/sft_llamafactory.example.yaml
```

也可以不安装包，直接用 `PYTHONPATH` 运行：

```bash
PYTHONPATH=src python -m badcaseflow.cli inspect --run runs/demo
```

示例里的 mock Agent 会故意制造一个未确认就预订的 calendar badcase，所以 `bcf eval` 会显示 1 条失败样本。随后 `bcf analyze-cases` 会把它归因为 `PATCH_WORKFLOW`，表达这个产品的核心判断：不是所有 badcase 都应该直接进入训练，有些应该先改 workflow、工具协议或评测门禁。

如果已经有自己的 Agent trace，可以直接导入 trace 再评测：

```bash
bcf import-traces --workspace demo-agent --input examples/datasets/external_traces.jsonl --run runs/external-demo
bcf eval --workspace demo-agent --run runs/external-demo --suite examples/datasets/eval_tasks.jsonl
bcf analyze-cases --workspace demo-agent --run runs/external-demo
```

导入的 trace 需要包含最小字段：`trace_id`、`sample_id`、`workspace_id`、`steps`、`final_answer`。

## 数据集版本

BadcaseFlow 把每个可复用数据文件都登记成 `dataset_version`，包含数据类型、记录数、字节数、`sha256`、schema 检查结果和来源信息。这样后续训练、评测、模型注册和 case 分析都能追溯到具体数据版本，而不是只记一个文件路径。

当前支持的数据类型包括：

| 类型 | 用途 |
|---|---|
| `task` | 种子任务、回归任务、线上回流任务 |
| `trace` | Agent 执行轨迹，包括工具调用和最终回答 |
| `eval_suite` | 样本级评测项和检查规则 |
| `sft` | SFT messages 数据 |
| `preference` | DPO/KTO 等偏好数据候选 |
| `rl_prompt` | RL/GRPO/PPO/OPD 可使用的 prompt 数据 |
| `eval_result` / `eval_report` | 样本级评测结果和汇总报告 |

检查一个数据文件：

```bash
bcf dataset inspect \
  --path examples/datasets/seed_tasks.jsonl \
  --kind task \
  --dataset-id seed-tasks \
  --workspace demo-agent
```

登记到本地 registry：

```bash
bcf dataset register \
  --path examples/datasets/seed_tasks.jsonl \
  --kind task \
  --dataset-id seed-tasks \
  --workspace demo-agent
bcf dataset list
```

训练或评测 recipe 也可以引用数据版本，再由 dry-run 自动解析成真实路径：

```yaml
data:
  dataset_version_id: dataset-seed-tasks-xxxxxx
```

```bash
bcf train dry-run \
  --recipe examples/recipes/sft_verl.example.yaml \
  --registry-root .
```

对内置 Agent trace 评测，可以分别引用 trace 数据和 suite 数据：

```yaml
data:
  trace_dataset_version_id: dataset-external-traces-xxxxxx
  suite_dataset_version_id: dataset-eval-suite-xxxxxx
```

`bcf round run-local` 会自动为本轮产出的 `tasks`、`traces`、`eval_results`、`eval_report`、`accepted`、`rejected`、`sft`、`preference` 和 `rl_prompts` 写入 `data_run/dataset_versions.json`。如果这一轮同时注册候选模型，这些数据版本也会被写入 registry，方便之后查看模型血缘。

## 评测适配器

BadcaseFlow 现在有两种评测入口：

- `bcf eval`：直接对当前 run 里的 `traces.jsonl` 跑内置规则评测；
- `bcf evals`：通过 adapter 管理不同评测后端的 dry-run、执行、日志、指标和 gate 状态。

查看支持矩阵：

```bash
bcf evals adapters
```

当前已覆盖：

| 后端 | 阶段 | 典型用途 |
|---|---|---|
| builtin | `agent_trace` | 使用 BadcaseFlow 内置规则评测 Agent trace |
| command | `custom` | 包装任意评测命令，统一记录日志和指标 |
| promptfoo | `suite` | 接 promptfoo eval 配置 |
| OpenCompass | `benchmark` | 接 OpenCompass benchmark 配置 |
| LightEval | `benchmark` | 接 LightEval benchmark 任务 |

本地 smoke 评测：

```bash
bcf evals run \
  --recipe examples/evals/command_eval_smoke.example.yaml \
  --run-id command-eval-smoke
bcf evals inspect --run runs/eval_runs/command-eval-smoke
```

对 demo trace 跑内置规则评测：

```bash
bcf demo --run runs/demo
bcf evals run \
  --workspace demo-agent \
  --recipe examples/evals/builtin_rule.example.yaml \
  --run-id builtin-rule-demo
```

外部评测框架可以先 dry-run 看命令：

```bash
bcf evals dry-run --recipe examples/evals/promptfoo.example.yaml
bcf evals dry-run --recipe examples/evals/opencompass.example.yaml
bcf evals dry-run --recipe examples/evals/lighteval.example.yaml
```

一次 eval run 会写入 `runs/eval_runs/<eval_run_id>/`，包括 `manifest.json`、`status.json`、`events.jsonl`、`metrics.jsonl`、`stdout.log`、`stderr.log` 和可选的 `eval_report.json`。其中 `quality_status` 表示本次评测是否通过 gate，后续可以接到 model registry 的 `attach-eval`。

## 训练适配器

BadcaseFlow 不直接实现训练算法，而是把训练框架作为 adapter 接进飞轮。当前 dry-run 已经支持按 `backend + stage` 生成训练命令：

| 后端 | 已覆盖阶段 | 典型用途 |
|---|---|---|
| LLaMA-Factory | `pt`、`sft`、`rm`、`ppo`、`dpo`、`kto`、`export` | 常规后训练、偏好优化、奖励模型、LoRA 合并导出 |
| verl | `sft`、`ppo`、`grpo`、`dapo`、`rloo`、`remax`、`reinforce_plus_plus`、`gspo`、`opd`、`eval` | SFT、RL、GRPO、在线策略蒸馏和离线评测 |

查看当前支持矩阵：

```bash
bcf train adapters
```

生成 dry-run manifest：

```bash
bcf train dry-run --workspace demo-agent --recipe examples/recipes/grpo_verl.example.yaml
```

执行本地训练 harness：

```bash
bcf train run --workspace demo-agent --recipe examples/recipes/command_smoke.example.yaml
```

一次 train run 会写入 `runs/train_runs/<train_run_id>/`，里面包含：

- `manifest.json`：训练 run 的完整元数据；
- `status.json`：最终状态、退出码、耗时和 artifact 索引；
- `events.jsonl`：开始、日志行、结束等事件流；
- `metrics.jsonl`：从日志中抽取的轻量指标，例如 `loss=...`、`reward=...`、`kl=...`；
- `diagnosis.json`：失败类别、证据行和建议动作；
- `artifacts.json`：本次训练 run 的产物清单；
- `stdout.log` / `stderr.log`：训练命令输出。

查看训练 run：

```bash
bcf train list --root runs/train_runs
bcf train inspect --run runs/train_runs/<train_run_id>
```

把训练 run 纳入本地 registry，并注册候选模型：

```bash
bcf registry index-train-run --run runs/train_runs/<train_run_id>
bcf registry register-model \
  --run runs/train_runs/<train_run_id> \
  --model-id demo-agent-candidate \
  --model-path artifacts/demo-agent/candidate
bcf registry models
bcf registry show-model --model-id demo-agent-candidate
```

把候选模型和评测报告绑定，生成门禁决策和可追踪谱系：

```bash
bcf registry attach-eval \
  --model-id demo-agent-candidate \
  --eval-report runs/demo/eval_report.json \
  --min-pass-rate 0.8
bcf registry evaluations
bcf registry promotions
bcf registry model-lineage --model-id demo-agent-candidate
```

`attach-eval` 会把 `eval_report.json` 摘要、失败分桶、通过率阈值和 decision 写入 registry。默认策略是通过率达标且没有失败样本才标记为 `approved`；否则标记为 `blocked`，并保留原因码，方便下一轮继续补 case、改 workflow 或重新训练。

如果评测是通过 `bcf evals run` 产生的，也可以直接绑定 eval run：

```bash
bcf registry index-eval-run --run runs/eval_runs/<eval_run_id>
bcf registry attach-eval-run \
  --model-id demo-agent-candidate \
  --eval-run runs/eval_runs/<eval_run_id>
```

recipe 里建议把 `stage` 放到顶层，框架原生字段可以继续放在 `training`、`data`、`rollout`、`teacher`、`overrides` 等区域。BadcaseFlow 会先做 preflight，再把 recipe 翻译成目标框架命令；执行时由统一 harness 负责日志、状态、退出码和产物登记。真正训练可以在自己的 GPU 服务器或 AutoDL 环境里运行。

## 飞轮轮次

当单步命令都能跑通后，可以用 `round` 命令把一次本地迭代固化成 `FlywheelRound`。它会记录每个阶段的输入、输出、指标、事件流和最终质量状态：

```bash
bcf round run-local \
  --workspace demo-agent \
  --round runs/rounds/round-smoke \
  --train-mode run \
  --recipe examples/recipes/command_smoke.example.yaml \
  --model-id demo-agent-candidate \
  --attach-round-eval
bcf round inspect --round runs/rounds/round-smoke
bcf round plan-next --round runs/rounds/round-smoke
bcf round list --root runs/rounds
```

训练后也可以接一条候选模型回评测链路。`--candidate-eval-recipe` 会通过 eval harness 产生 eval run；如果该 eval run 写出了 `eval_report.json`，registry 会把它绑定到候选模型并生成 promotion decision：

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

一次 round 会写入：

- `round.json`：本轮状态、阶段列表、关键指标、模型和门禁摘要；
- `events.jsonl`：round/stage 事件流；
- `iteration_plan.json` / `iteration_plan.md`：下一轮行动计划；
- `data_run/`：任务、trace、评测报告、case report、训练数据导出和 `dataset_versions.json`；
- `train_runs/`：可选训练 run；
- `eval_runs/`：可选候选模型回评测 run；
- registry 记录：可选数据版本、模型候选、评测绑定和 promotion decision。

`iteration_plan` 会优先使用候选模型回评测的 report 和 case report；如果没有候选回评测，则使用本轮数据生成阶段的评测结果。它会把 case report 和 promotion decision 转成行动项，例如暂缓候选模型、修复 workflow、补 SFT 数据、构造 preference pair、加入回归评测样本或准备候选模型进入下一阶段。所有行动默认是 `proposed` 状态，方便后续接 UI 的人工确认和任务分派。

行动项可以继续更新状态、负责人和备注：

```bash
bcf action list --plan runs/rounds/round-smoke/iteration_plan.json
bcf action update \
  --plan runs/rounds/round-smoke/iteration_plan.json \
  --action-id action-001 \
  --status accepted \
  --owner engineer-a \
  --note "先处理阻塞门禁的失败样本"
```

默认 recipe 是本地 smoke harness，不依赖 GPU。真实训练时可以把 `--recipe` 换成 LLaMA-Factory 或 verl recipe，把 `--train-mode` 设为 `plan` 或 `run`，并配合自己的服务器环境执行。

## 静态工作台

跑完 demo、训练 run、评测 run 或 round 后，可以生成一个不依赖前端框架的本地 HTML 工作台：

```bash
bcf workbench build \
  --root . \
  --registry-root . \
  --rounds-root runs/rounds \
  --output runs/workbench/index.html
```

它会同时写出：

```text
runs/workbench/index.html
runs/workbench/workbench.json
```

HTML 里能查看飞轮轮次、下一轮行动、执行 job、模型注册表、promotion decision、训练 run、评测 run、数据版本和部署计划。`workbench.json` 是同一份结构化数据，后续可以直接复用到真正的 Web 工作台。

## 接自己的服务器或 AutoDL

BadcaseFlow 本身先做控制层，训练可以发生在自己的 GPU 服务器、AutoDL 或其他远程环境。当前版本先提供 remote dry-run 计划，不会直接登录执行：

```bash
bcf init-workspace --workspace demo-agent
bcf remote add --name autodl --host root@your-server --workdir /root/BadcaseFlow --python python
bcf remote plan \
  --target autodl \
  --workspace demo-agent \
  --run runs/demo \
  --recipe examples/recipes/sft_llamafactory.example.yaml
```

这会生成：

```text
runs/demo/remote_plan_autodl.json
runs/demo/remote_plan_autodl.ps1
```

plan 里包含远程目录创建、run 数据同步、recipe 同步、训练 dry-run 和产物回收命令。等 SSH key、AutoDL 地址和环境准备好以后，可以把这些 dry-run 命令逐步变成真实执行。

更通用的方式是使用 launcher plan。它可以为 train/eval recipe 生成 local 或 SSH 启动计划，并写出可审查的 `.ps1` 脚本：

```bash
bcf launcher plan \
  --kind train \
  --launcher local \
  --mode run \
  --workspace demo-agent \
  --recipe examples/recipes/command_smoke.example.yaml \
  --run-id launcher-train

bcf launcher plan \
  --kind eval \
  --launcher ssh \
  --target autodl \
  --config .badcaseflow/remotes.json \
  --mode plan \
  --recipe examples/evals/command_eval_smoke.example.yaml \
  --run-id launcher-eval
```

`mode=dry-run` 会只生成 dry-run 命令，`mode=plan` 会生成 `--plan-only` 命令，`mode=run` 会生成真实执行命令。SSH launcher 当前仍是“生成计划和脚本”，用户确认后再执行脚本；这样可以先检查路径、recipe、run id 和产物回收命令。

如果希望把启动计划纳入平台状态管理，可以把 launch plan 提交成 job：

```bash
bcf launcher plan \
  --kind train \
  --launcher local \
  --mode run \
  --workspace demo-agent \
  --recipe examples/recipes/command_smoke.example.yaml \
  --run-id job-smoke \
  --output runs/job-smoke-plan.json

bcf job submit --plan runs/job-smoke-plan.json --job-id job-smoke
bcf job run --job runs/jobs/job-smoke
bcf job status --job runs/jobs/job-smoke
bcf job logs --job runs/jobs/job-smoke --stream stdout --tail 40
bcf job collect --job runs/jobs/job-smoke
```

SSH job 也可以先只登记计划，确认后再显式执行：

```bash
bcf launcher plan \
  --kind train \
  --launcher ssh \
  --target autodl \
  --config .badcaseflow/remotes.json \
  --mode run \
  --recipe examples/recipes/sft_llamafactory.example.yaml \
  --run-id remote-train \
  --output runs/remote-train-plan.json

bcf job submit --plan runs/remote-train-plan.json --job-id remote-train
bcf job run --job runs/jobs/remote-train
```

每个 job 会写入 `status.json`、`events.jsonl`、`stdout.log`、`stderr.log` 和 `launch_plan.json`，后续静态工作台也会读取 `runs/jobs` 展示任务状态。

## 上线部署计划

候选模型通过 promotion gate 后，可以生成服务部署计划。当前支持 `vllm`、`sglang` 和已有 OpenAI-compatible endpoint 的 runbook。部署计划会记录模型路径、服务端口、canary 比例、验证命令和回滚动作，并可以登记到 registry：

```bash
bcf deploy plan \
  --model-id demo-agent-candidate \
  --backend vllm \
  --deployment-id demo-agent-v1 \
  --canary-percent 10 \
  --register
bcf deploy list
bcf deploy show --deployment-id demo-agent-v1
```

默认只有 `approved` 模型能生成部署计划；如果只是做演练，可以显式传 `--allow-blocked`。部署计划会写出 JSON 和 `.ps1` 脚本，用户检查后再执行服务启动、健康检查和流量切换。

## 后续路线

| 阶段 | 目标 | 重点能力 |
|---|---|---|
| v0.1.0 | 首个可用版本 | 本地飞轮、训练/评测 adapter、dataset version、model registry、promotion gate、job runner、deployment plan |
| v0.2 | Web 工作台 | 数据集浏览、评测结果、训练 run、case board、模型注册表 |
| v0.3 | 执行编排 | 任务队列、远程 launcher、重试、资源调度、运行状态恢复 |
| v0.4 | 上线回流 | vLLM/SGLang serving、canary、线上 trace 采样、部署后回评测 |

详细路线见 [路线图](docs/mvp_roadmap_zh.md) 和 [首版范围](docs/first_version_scope_zh.md)。

## 仓库结构

```text
badcaseflow/
  src/badcaseflow/       # CLI、数据协议、harness、adapter、registry
  tests/                 # 离线单元测试
  examples/
    datasets/            # synthetic 数据和 GSM8K tiny 数据
    recipes/             # LLaMA-Factory、verl、command 示例
    evals/               # 内置和外部评测 adapter 示例
  docs/                  # 产品定位、技术方案、路线图和上手文档
  .github/workflows/     # GitHub Actions CI
```

## 开源发布

准备上传或更新 GitHub 仓库时，按照 [GitHub 发布检查清单](docs/github_release_checklist_zh.md) 执行。贡献代码前请阅读 [贡献指南](CONTRIBUTING_zh.md)。

## 可参考和组装的成熟项目

| 方向 | 项目 | 我们怎么用 |
|---|---|---|
| Trace / Eval / Feedback | [Langfuse](https://github.com/langfuse/langfuse)、[Phoenix](https://github.com/Arize-ai/phoenix) | 借鉴 trace、dataset、experiment、eval 和人工反馈流，也可做 adapter |
| SFT / DPO / 常规后训练 | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)、[Axolotl](https://github.com/axolotl-ai-cloud/axolotl)、[TRL](https://github.com/huggingface/trl) | 生成配置、做数据 preflight、收集训练产物和指标 |
| RL / GRPO / OPD | [verl](https://github.com/verl-project/verl)、[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | 作为 RL/OPD 训练后端，平台负责编排、数据、指标和回评测 |
| 标准评测 | [OpenCompass](https://github.com/open-compass/opencompass)、[LightEval](https://github.com/huggingface/lighteval)、[promptfoo](https://github.com/promptfoo/promptfoo) | 作为 eval backend，平台统一保存样本级结果和回归门禁 |
| 模型服务 | [vLLM](https://github.com/vllm-project/vllm)、[SGLang](https://github.com/sgl-project/sglang) | 提供 OpenAI-compatible serving，支持灰度和 trace 回流 |
| 实验指标 | [TensorBoard](https://github.com/tensorflow/tensorboard)、[MLflow](https://github.com/mlflow/mlflow)、[W&B](https://wandb.ai/)、[SwanLab](https://github.com/SwanHubX/SwanLab) | 记录训练曲线、reward、KL、throughput、artifact |

## 仓库内容

- [产品定位](docs/product_positioning_zh.md)
- [执行计划](docs/execution_plan_zh.md)
- [首版范围](docs/first_version_scope_zh.md)
- [更新日志](CHANGELOG.md)
- [命名说明](docs/naming_zh.md)
- [技术方案](docs/technical_design_zh.md)
- [开源选型](docs/reference_stack_zh.md)
- [路线图](docs/mvp_roadmap_zh.md)
- [AutoDL + GSM8K 快速开始](docs/autodl_gsm8k_quickstart_zh.md)
- [GitHub 发布检查清单](docs/github_release_checklist_zh.md)
- [贡献指南](CONTRIBUTING_zh.md)
- [示例数据](examples/datasets/seed_tasks.jsonl)
- [GSM8K tiny 示例数据](examples/datasets/gsm8k_tiny.jsonl)
- [工具清单示例](examples/tools/tool_manifest.example.json)
- [评测 recipe 示例](examples/evals/builtin_rule.example.yaml)
- [SFT recipe 示例](examples/recipes/sft_llamafactory.example.yaml)
- [DPO recipe 示例](examples/recipes/dpo_llamafactory.example.yaml)
- [GRPO recipe 示例](examples/recipes/grpo_verl.example.yaml)
- [GSM8K SFT recipe 示例](examples/recipes/gsm8k_sft_llamafactory.example.yaml)
- [GSM8K GRPO recipe 示例](examples/recipes/gsm8k_grpo_verl.example.yaml)
- [AutoDL PPO recipe 示例](examples/recipes/gsm8k_ppo_verl_autodl.example.yaml)
- [OPD recipe 示例](examples/recipes/opd_verl.example.yaml)
- [训练 harness smoke 示例](examples/recipes/command_smoke.example.yaml)
