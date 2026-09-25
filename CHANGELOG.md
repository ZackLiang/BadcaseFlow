# 更新日志

## v0.1.0 - 首个可用版本

BadcaseFlow 的首版目标是跑通 Agent 训练飞轮的可验证闭环：从任务或 trace 进入平台，完成评测、case 分析、训练数据导出、训练/评测 adapter 编排、候选模型登记、门禁决策和部署计划。

### 新增

- GSM8K 真实任务入口：新增 `bcf data prepare-gsm8k`，可生成 task、eval suite、SFT、RL prompt、verl JSONL/parquet，并补充 AutoDL 快速开始文档和 SFT/GRPO/PPO recipes。
- 本地 CLI：`ingest`、`rollout`、`eval`、`analyze-cases`、`export`、`demo`。
- 飞轮轮次：`bcf round run-local` 串联数据、评测、训练、候选回评测、registry 和下一轮计划。
- 训练 adapter：支持 LLaMA-Factory、verl 和 command smoke backend 的 dry-run、run、inspect、list。
- 评测 adapter：支持 builtin、command、promptfoo、OpenCompass 和 LightEval 的 dry-run、run、inspect、list。
- 数据版本：为 task、trace、eval suite、SFT、preference、RL prompt、eval result 和 eval report 生成 `dataset_version`。
- 本地 registry：登记 artifact、dataset、model、eval、promotion 和 deployment。
- 上线计划：支持 vLLM、SGLang 和 OpenAI-compatible endpoint 的部署 runbook。
- 迭代计划：把 case report 和 promotion decision 转成下一轮 proposed actions。
- 行动状态：`bcf action list/update` 支持更新行动项状态、负责人、备注和历史记录。
- 执行任务：`bcf job submit/run/status/logs/collect` 支持把 launcher plan 登记成可追踪 job，本地执行并记录日志，SSH 任务可先登记计划再显式执行。
- 静态工作台：`bcf workbench build` 导出 HTML 和 `workbench.json`，集中查看 round、case action、registry、训练、评测和部署摘要。
- 验收命令：新增 `bcf version` 和 `bcf doctor`。

### 示例

- `examples/datasets/`：种子任务、评测集和外部 trace 示例。
- `examples/recipes/`：LLaMA-Factory、verl 和 command smoke 训练 recipe。
- `examples/evals/`：builtin、command、promptfoo、OpenCompass 和 LightEval 评测 recipe。

### 验收方式

```bash
bcf version
bcf doctor --strict
bcf demo --run runs/demo
bcf round run-local --workspace demo-agent --round runs/rounds/round-smoke --train-mode run --recipe examples/recipes/command_smoke.example.yaml --model-id demo-agent-candidate --attach-round-eval
bcf action list --plan runs/rounds/round-smoke/iteration_plan.json
bcf launcher plan --kind train --launcher local --mode run --recipe examples/recipes/command_smoke.example.yaml --run-id job-smoke --output runs/job-smoke-plan.json
bcf job submit --plan runs/job-smoke-plan.json --job-id job-smoke
bcf job run --job runs/jobs/job-smoke
bcf workbench build --registry-root .
```

### 已知不足

- 首版工作台是静态只读导出，还没有交互式筛选、审批、编辑和多人协作。
- 分析器以规则和可解释归因为主，后续需要接入更强的聚类、judge 和行动优先级排序。
- 远程启动已经能进入 job 记录，但后续还需要补队列、失败重试和资源调度 adapter。
- 上线链路先生成 runbook，后续需要接 trace 回流、灰度指标和自动回滚建议。
