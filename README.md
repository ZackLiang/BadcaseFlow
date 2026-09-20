# BadcaseFlow

BadcaseFlow 是一个面向 Agent 的受控自进化平台：把失败 case、线上 trace、评测、数据生成、SFT/RL/OPD 后训练和上线门禁串成一个可追踪闭环。

名字里的 `Badcase` 是核心入口，`Flow` 是产品方法：失败样本不应该停在日志里，而应该流向归因、数据、训练、回评测和下一轮上线决策。

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
3. 管理训练 recipe：SFT/DPO 走 LLaMA-Factory，RL/OPD 优先走 verl，其他框架通过 adapter 接入。
4. 做训练前后门禁：preflight、dry-run、训练指标采集、训练后自动评测、回归集对比。
5. 记录模型上线血缘：哪个数据集、哪个 recipe、哪个 checkpoint、哪个评测报告支撑这次发布。

## 第一版 MVP

第一版先做本地可运行的最小闭环，不追求马上训练出最强模型：

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

## 后续路线

| 阶段 | 目标 | 重点能力 |
|---|---|---|
| v0.1 | 本地飞轮 | CLI、数据协议、mock rollout、评测、SFT export、case report |
| v0.2 | 训练适配 | LLaMA-Factory SFT/DPO adapter、verl OPD/GRPO adapter、preflight、dry-run |
| v0.3 | Web 工作台 | 数据集、评测结果、训练 run、case board、模型注册表 |
| v0.4 | 上线回流 | vLLM/SGLang serving、canary、线上 trace 采样、训练后回评测 |

详细路线见 [MVP 路线图](docs/mvp_roadmap_zh.md)。

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
- [命名说明](docs/naming_zh.md)
- [技术方案](docs/technical_design_zh.md)
- [开源选型](docs/reference_stack_zh.md)
- [MVP 路线图](docs/mvp_roadmap_zh.md)
- [示例数据](examples/datasets/seed_tasks.jsonl)
- [工具清单示例](examples/tools/tool_manifest.example.json)
- [SFT recipe 示例](examples/recipes/sft_llamafactory.example.yaml)
- [OPD recipe 示例](examples/recipes/opd_verl.example.yaml)
